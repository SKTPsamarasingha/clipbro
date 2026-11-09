import re
from dotenv import load_dotenv
load_dotenv()
from faster_whisper import WhisperModel

from cilpbro import progress


def transcribe_audio(
    video_path,
    model_size="small",
    context=None,
    vocabulary=None,
    corrections=None,
    # --- accuracy tuning ---
    beam_size=8,
    language="en",
    vad_filter=True,
    condition_on_previous_text=True,
    temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    repetition_penalty=1.1,
):
    vocabulary = vocabulary or []
    corrections = corrections or {}

    with progress.Bar("Loading Whisper model", total=None):
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
    progress.ok(f"Whisper model ({model_size}) loaded")

    prompt_parts = []
    if context:
        prompt_parts.append(context.strip())
    if vocabulary:
        prompt_parts.append("Names and terms: " + ", ".join(vocabulary) + ".")
    initial_prompt = " ".join(prompt_parts) or None

    segments, info = model.transcribe(
        video_path,
        beam_size=beam_size,
        language=language,
        initial_prompt=initial_prompt,
        hotwords=", ".join(vocabulary) or None,
        vad_filter=vad_filter,
        condition_on_previous_text=condition_on_previous_text,
        temperature=list(temperature),
        repetition_penalty=repetition_penalty,
        word_timestamps=True,
    )
    total_duration = info.duration or 1.0

    bar = progress.Bar("Transcribing", total=total_duration)
    timestamped_segments = []
    for seg in segments:
        text = apply_corrections(seg.text.strip(), corrections)
        words = [
            {"w": w.word.strip(), "s": round(w.start, 2), "e": round(w.end, 2)}
            for w in (seg.words or [])
        ]
        timestamped_segments.append({
            "start": seg.start,
            "end": seg.end,
            "text": text,
            "words": words,
        })

        bar.update(min(seg.end, total_duration))

    bar.finish()
    progress.ok("Transcription complete")
    return timestamped_segments


def apply_corrections(text, corrections):
    for wrong, right in corrections.items():
        text = re.sub(re.escape(wrong), right, text, flags=re.IGNORECASE)
    return text