import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ServerError
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from cilpbro.progress import Bar, track, info, ok, warn

load_dotenv()
client = genai.Client()

DEFAULT_MODEL = "gemini-3.8-flash"   # change to whichever Gemini model your key can use


class HighlightClip(BaseModel):
    clip_title: str
    start_timestamp: float
    end_timestamp: float
    reasoning: str


class HighlightResponse(BaseModel):
    clips: list[HighlightClip]


@retry(
    retry=retry_if_exception_type(ServerError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=10),
    reraise=True,
)
def _call_gemini_with_retry(model, contents, config):
    return client.models.generate_content(model=model, contents=contents, config=config)


def _slim_payload(payload, include_words=False):
    """Copy of the payload without fields the model doesn't need (saves tokens)."""
    transcript = []
    for seg in payload["transcript"]:
        s = {k: v for k, v in seg.items() if k != "words" or include_words}
        transcript.append(s)

    images = []
    for img in payload["visual_context"]["images"]:
        images.append({
            "image_id": img["image_id"],
            "start_timestamp": img["start_timestamp"],
            "end_timestamp": img["end_timestamp"],
            "grid": img["grid"],
            "frames": [
                {"segment_id": f["segment_id"], "timestamp": f["timestamp"],
                 "tile_position": f["tile_position"]}
                for f in img["frames"]
            ],
        })

    return {
        "metadata": payload["metadata"],
        "transcript": transcript,
        "visual_context": {"format": "contact_sheet", "images": images},
    }


def _load_sheet_bytes(sheet):
    """Prefer the local file; fall back to the cloud URL only if the file is missing."""
    path = sheet.get("file_path")
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()

    url = sheet.get("url")
    if url:
        import requests
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return r.content

    raise FileNotFoundError(f"No local file or URL for {sheet['image_id']}")


def analyze_video_for_highlights(
    payload_input,
    num_clips=5,
    min_len=15,
    max_len=60,
    model=DEFAULT_MODEL,
    include_words=False,
):
    if isinstance(payload_input, str):
        with open(payload_input, "r", encoding="utf-8") as f:
            payload = json.load(f)
    elif isinstance(payload_input, dict):
        payload = payload_input
    else:
        raise ValueError("Input must be a file path string or a payload dictionary.")

    meta = payload["metadata"]
    video_id = meta["video_id"]
    duration = meta["duration_seconds"]
    sheets = payload["visual_context"]["images"]
    category = meta.get("category", "general")
    clip_hint = meta.get("clip_hint", "")

    prompt = f"""
You are an expert short-form video editor.
Video ID: {video_id}. Duration: {duration}s. Category: {category}.
{clip_hint}

Find the top {num_clips} most engaging highlight clips ({min_len} to {max_len} seconds each).

Rules:
- Clips must cover distinct moments and must not overlap.
- Rank from most to least engaging.
- Start and end on natural boundaries: a clip should begin at the start of a sentence or beat and end after the payoff, never mid-sentence.
- Use the signals per transcript segment: energy (0-1, relative loudness), wps (words per second),
  pause_before (silence before the line), face (emotion), scene_cut/cuts (camera changes), replay (viewer replay score 0-1, null if unknown).
- Each contact sheet is a grid read left-to-right, top-to-bottom. The "frames" list maps each tile_position
  to a segment_id and timestamp. Cross-reference dialogue with what is visible.
- Timestamps are in seconds and must be inside 0 to {duration}.
"""

    contents = [prompt, json.dumps(_slim_payload(payload, include_words), ensure_ascii=False)]

    for sheet in track(sheets, "analyze: loading sheets"):
        try:
            contents.append(types.Part.from_bytes(data=_load_sheet_bytes(sheet), mime_type="image/jpeg"))
            contents.append(
                f"Above image is {sheet['image_id']} covering "
                f"{sheet['start_timestamp']}s to {sheet['end_timestamp']}s."
            )
        except Exception as e:
            warn(f"Skipping {sheet['image_id']}: {e}")

    with Bar("gemini: analysing", None):          # pulsing bar, finishes itself
        response = _call_gemini_with_retry(
            model="gemini-3.8-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=HighlightResponse,
            ),
        )

    clips = json.loads(response.text)["clips"]

    # Sanity-check what the model returned
    cleaned = []
    for c in clips:
        start = max(0.0, float(c["start_timestamp"]))
        end = min(float(duration), float(c["end_timestamp"]))
        if end - start < 1:
            warn(f"Dropping invalid clip: {c['clip_title']}")
            continue
        cleaned.append({**c, "start_timestamp": round(start, 2), "end_timestamp": round(end, 2)})

    ok(f"Gemini returned {len(cleaned)} usable clips")
    return cleaned