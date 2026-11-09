import os
import wave
import subprocess

import numpy as np
import torch
import imageio_ffmpeg
from dotenv import load_dotenv

load_dotenv()

from pyannote.audio import Pipeline

from cilpbro import progress


def _extract_wav(video_path, wav_path="diarize_audio.wav"):
    """16 kHz mono WAV, using the ffmpeg binary we already have."""
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", video_path,
         "-vn", "-ac", "1", "-ar", "16000", wav_path],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return wav_path


def _load_waveform(wav_path):
    """Read the WAV ourselves so pyannote never has to decode audio (avoids torchcodec issues on Windows)."""
    with wave.open(wav_path, "rb") as f:
        sr = f.getframerate()
        audio = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
    waveform = torch.from_numpy(audio.astype(np.float32) / 32768.0).unsqueeze(0)  # (1, samples)
    return {"waveform": waveform, "sample_rate": sr}


def diarize_audio(video_path, num_speakers=None, min_speakers=None, max_speakers=None):
    """Returns a list of (start, end, speaker) turns."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not found in .env")

    with progress.Bar("Loading diarization model", total=None):
        try:
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
        except TypeError:  # older pyannote versions use a different argument name
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
    if pipeline is None:
        raise RuntimeError("Model load failed. Did you accept the terms on the Hugging Face model pages?")
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    progress.ok("Diarization model loaded")

    wav_path = _extract_wav(video_path)
    audio = _load_waveform(wav_path)

    with progress.DiarizationHook() as hook:
        output = pipeline(
            audio,
            hook=hook,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

    # pyannote 3.x returns an Annotation; 4.x wraps it in an output object
    annotation = getattr(output, "speaker_diarization", output)

    turns = [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]

    os.remove(wav_path)
    speakers = sorted({s for _, _, s in turns})
    progress.ok(f"Diarization complete: {len(speakers)} speaker(s) found, {len(turns)} turns")
    return turns


def assign_speakers(segments, turns):
    """Give each transcript segment the speaker who talks the most during it."""
    for seg in segments:
        overlap = {}
        for t_start, t_end, speaker in turns:
            o = min(seg["end"], t_end) - max(seg["start"], t_start)
            if o > 0:
                overlap[speaker] = overlap.get(speaker, 0.0) + o

        if overlap:
            seg["speaker"] = max(overlap, key=overlap.get)
        else:
            # No overlap (e.g. music): use the nearest turn instead
            mid = (seg["start"] + seg["end"]) / 2
            nearest = min(turns, key=lambda t: min(abs(mid - t[0]), abs(mid - t[1])), default=None)
            seg["speaker"] = nearest[2] if nearest else "UNKNOWN"
    return segments