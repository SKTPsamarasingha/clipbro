import os
import subprocess
import wave

import numpy as np
import imageio_ffmpeg

from cilpbro import progress


def _load_audio(video_path, wav_path="energy_audio.wav"):
    """Decode to 16 kHz mono and return (float32 samples, sample_rate)."""
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", video_path,
         "-vn", "-ac", "1", "-ar", "16000", wav_path],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    with wave.open(wav_path, "rb") as f:
        sr = f.getframerate()
        audio = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
    os.remove(wav_path)
    return audio.astype(np.float32) / 32768.0, sr


def add_audio_signals(segments, video_path):
    """Adds 'energy' (0-1), 'wps' (words/sec) and 'pause_before' (sec) to each segment."""
    audio, sr = _load_audio(video_path)

    # Loudness per segment in dB
    dbs = []
    for seg in segments:
        chunk = audio[int(seg["start"] * sr):int(seg["end"] * sr)]
        rms = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) else 0.0
        dbs.append(20 * np.log10(rms + 1e-9))

    # Scale relative to THIS video (5th-95th percentile) so one huge
    # spike doesn't squash every other segment toward 0
    if dbs:
        lo, hi = np.percentile(dbs, [5, 95])
    for i, seg in enumerate(segments):
        if hi - lo < 1e-6:
            energy = 0.5
        else:
            energy = float(np.clip((dbs[i] - lo) / (hi - lo), 0.0, 1.0))
        seg["energy"] = round(energy, 2)

        # Words per second: first word start -> last word end
        words = seg.get("words") or []
        if words:
            span = words[-1]["e"] - words[0]["s"]
            n_words = len(words)
        else:
            span = seg["end"] - seg["start"]
            n_words = len(seg["text"].split())
        seg["wps"] = round(n_words / span, 1) if span > 0.05 else 0.0

        # Silence before this segment (sudden silence / dramatic pause)
        prev_end = segments[i - 1]["end"] if i > 0 else 0.0
        seg["pause_before"] = round(max(seg["start"] - prev_end, 0.0), 2)

    progress.ok(f"Audio signals added to {len(segments)} segments")
    return segments