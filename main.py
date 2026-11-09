import static_ffmpeg

static_ffmpeg.add_paths()

from cilpbro.downloader import download_video
from cilpbro.transcribe import transcribe_audio
from cilpbro.metadata import fetch_video_context
from cilpbro.diarize import diarize_audio, assign_speakers
from cilpbro.signals import add_audio_signals
from cilpbro.faces import add_face_signals

from cilpbro.scenes import add_scene_signals
from cilpbro.replay import add_replay_signals
from cilpbro.category import detect_category, CLIP_HINTS

from cilpbro.frames import extract_frames_and_build_json
from cilpbro.storage import ObjectStorage
from cilpbro.analyze import analyze_video_for_highlights
from cilpbro.cutter import cut_clips_for_platforms
from cilpbro import progress
import json
import os
import re
from urllib.parse import urlparse, parse_qs


def extract_youtube_id(url: str) -> str:
    """Extracts the 11-character YouTube video ID from various URL formats."""
    parsed_url = urlparse(url)
    if parsed_url.netloc in ('youtu.be', 'www.youtu.be'):
        return parsed_url.path.lstrip('/')
    if parsed_url.netloc in ('www.youtube.com', 'youtube.com', 'm.youtube.com'):
        if parsed_url.path == '/watch':
            return parse_qs(parsed_url.query).get('v', [None])[0]
        elif parsed_url.path.startswith(('/embed/', '/v/', '/shorts/')):
            return parsed_url.path.split('/')[2]

    # Fallback regex
    match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract YouTube ID from URL: {url}")


def main():
    video_url = "https://www.youtube.com/watch?v=Fj5gIuBko-0"

    # 1. Extract Video ID dynamically
    video_id = extract_youtube_id(video_url)
    progress.info(f"Target Video ID: {video_id}")

    # 2. Define dynamic file paths
    download_dir = r"D:\Projects\clipBro\video"
    os.makedirs(download_dir, exist_ok=True)
    video_path = os.path.join(download_dir, f"{video_id}.mp4")

    meta = fetch_video_context(video_url)
    progress.info(f"Context:    {meta['context']}")
    progress.info(f"Vocabulary: {meta['vocabulary']}")

    # 3. Download video if it doesn't exist locally
    if not os.path.exists(video_path):
        progress.info("Video file not found locally. Downloading from YouTube...")
        downloaded_file = download_video(video_url.strip())
        progress.ok(f"Video downloaded to: {downloaded_file}")
    else:
        progress.ok(f"Found existing local video file: {video_path}")

        progress.info("No payload cache found. Transcribing and extracting frames...")

        raw_segments = transcribe_audio(
            video_path,
            model_size="small",
            context=meta["context"],
            vocabulary=meta["vocabulary"],
            corrections={},  # leave empty for now
        )
        turns = diarize_audio(video_path)  # add num_speakers=2 if you know it
        raw_segments = assign_speakers(raw_segments, turns)
        raw_segments = add_audio_signals(raw_segments, video_path)
        raw_segments = add_face_signals(raw_segments, video_path)
        raw_segments = add_scene_signals(raw_segments, video_path)
        raw_segments = add_replay_signals(raw_segments, meta["heatmap"])

        category = detect_category(meta, raw_segments)  # or detect_category(..., override="podcast_debate")
        progress.info(f"Category: {category}")

        progress.log(f"raw segments: {raw_segments}")
        UPLOAD_TO_CLOUD = False  # flip to True when you want sheets in Cloudinary

        storage = None
        if UPLOAD_TO_CLOUD:
            storage = ObjectStorage()

        data = extract_frames_and_build_json(
            video_path,
            raw_segments,
            video_id="Fj5gIuBko-0",
            upload_to_cloud=UPLOAD_TO_CLOUD,
            storage=storage,
            extra_metadata={
                "title": meta["title"],
                "category": category,
                "clip_hint": CLIP_HINTS.get(category, CLIP_HINTS["general"]),
            },
        )
        progress.log(f"json {json.dumps(data, indent=2)}")

        clips = analyze_video_for_highlights(data, num_clips=5)  # data = the payload dict (or cached JSON)

        progress.log(json.dumps(clips, indent=2))

        cut_clips_for_platforms(
            video_path,
            clips,
            platforms=["instagram_reels", "instagram_feed", "facebook_feed", "youtube"],
            output_dir=r"D:\Projects\clipBro\clips",
            video_id=video_id,
            mode="crop",  # or "blur"
        )




if __name__ == "__main__":
    main()
