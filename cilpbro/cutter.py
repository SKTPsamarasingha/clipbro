import json
import os
import subprocess

import static_ffmpeg
static_ffmpeg.add_paths()

from cilpbro import progress

# width, height, max duration in seconds (None = no limit I'm enforcing)
PLATFORMS = {
    "instagram_reels":   (1080, 1920, 90),
    "tiktok":            (1080, 1920, 180),
    "youtube_shorts":    (1080, 1920, 60),
    "facebook_reels":    (1080, 1920, 90),
    "instagram_feed":    (1080, 1350, 60),   # 4:5
    "facebook_feed":     (1080, 1350, 120),  # 4:5
    "square":            (1080, 1080, 60),   # 1:1
    "youtube":           (1920, 1080, None), # 16:9
    "landscape":         (1920, 1080, None),
}


def _crop_filter(w, h, focus_x, focus_y):
    # Crop the largest w:h region, positioned by focus (0 = left/top, 0.5 = center, 1 = right/bottom)
    return (
        f"crop=w='min(iw,ih*{w}/{h})':h='min(ih,iw*{h}/{w})':"
        f"x='(iw-ow)*{focus_x}':y='(ih-oh)*{focus_y}',"
        f"scale={w}:{h}:flags=lanczos,setsar=1"
    )


def _blur_filter(w, h):
    # Whole frame fitted inside, with a blurred zoomed copy filling the background
    return (
        f"split=2[bg][fg];"
        f"[bg]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},boxblur=25:5[bgb];"
        f"[fg]scale={w}:{h}:force_original_aspect_ratio=decrease[fgs];"
        f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2,setsar=1"
    )


def cut_video_clip(
    video_path,
    start_time,
    end_time,
    output_dir,
    output_filename,
    platform=None,        # key of PLATFORMS, or None to keep the original shape
    mode="crop",          # "crop" = fill the frame, "blur" = keep full frame on a blurred background
    focus_x=0.5,          # crop mode only: 0 left ... 1 right
    focus_y=0.5,
):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    duration = end_time - start_time
    if duration <= 0:
        raise ValueError(f"Invalid clip range: {start_time} to {end_time}")

    cmd = ["ffmpeg", "-y", "-ss", str(start_time), "-i", video_path, "-t", str(duration)]

    if platform:
        if platform not in PLATFORMS:
            raise ValueError(f"Unknown platform '{platform}'. Options: {list(PLATFORMS)}")
        w, h, max_dur = PLATFORMS[platform]
        if max_dur and duration > max_dur:
            progress.warn(f"{platform} allows ~{max_dur}s but this clip is {duration:.0f}s")

        if mode == "blur":
            cmd += ["-filter_complex", f"[0:v]{_blur_filter(w, h)}[v]", "-map", "[v]", "-map", "0:a?"]
        else:
            cmd += ["-vf", _crop_filter(w, h, focus_x, focus_y)]

    cmd += [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        output_path,
    ]

    progress.info(f"Cutting {start_time}s -> {end_time}s" + (f" for {platform} ({mode})" if platform else ""))
    with progress.Bar("Encoding clip", total=None):
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        progress.error(result.stderr[-1500:])
        raise RuntimeError("FFmpeg failed to cut the clip.")

    progress.ok(f"Saved: {output_path}")
    return output_path


def cut_clips_for_platforms(video_path, clips, platforms, output_dir, video_id, mode="crop", focus_x=0.5):
    """Cut every clip once per platform. Returns a list of output paths."""
    outputs = []
    for rank, clip in enumerate(clips, start=1):
        for platform in platforms:
            outputs.append(cut_video_clip(
                video_path,
                clip["start_timestamp"],
                clip["end_timestamp"],
                output_dir,
                f"{video_id}_clip{rank}_{platform}.mp4",
                platform=platform,
                mode=mode,
                focus_x=focus_x,
            ))
    return outputs