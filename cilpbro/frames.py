import math
import os
import cv2
import numpy as np
from datetime import timedelta

from cilpbro import progress

GRID_COLS = 2
MAX_TILES_PER_SHEET = 4   # 2x2
TILE_WIDTH = 512          # 2 tiles -> sheet is ~1024px wide
JPEG_QUALITY = 80


def fmt_time(seconds):
    return str(timedelta(seconds=int(seconds)))


def split_evenly(n_items, max_per_sheet=MAX_TILES_PER_SHEET):
    if n_items == 0:
        return []
    n_sheets = math.ceil(n_items / max_per_sheet)
    base, extra = divmod(n_items, n_sheets)
    groups, start = [], 0
    for i in range(n_sheets):
        size = base + (1 if i < extra else 0)
        groups.append((start, size))
        start += size
    return groups


def make_tile(frame, label, tile_w, tile_h):
    tile = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.65, 2
    (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, thickness)
    cv2.rectangle(tile, (0, 0), (text_w + 14, text_h + baseline + 12), (0, 0, 0), -1)
    cv2.putText(tile, label, (7, text_h + 6), font, scale, (255, 255, 255),
                thickness, cv2.LINE_AA)
    return tile


def build_contact_sheet(tiles, tile_w, tile_h):
    n_rows = math.ceil(len(tiles) / GRID_COLS)
    blank = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    padded = tiles + [blank] * (n_rows * GRID_COLS - len(tiles))
    rows = [np.hstack(padded[r * GRID_COLS:(r + 1) * GRID_COLS]) for r in range(n_rows)]
    return np.vstack(rows), n_rows


def encode_jpeg_bytes(image):
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("Failed to encode contact sheet as JPEG")
    return buffer.tobytes()


def extract_frames_and_build_json(
    video_path,
    segments,
    video_id=None,
    frame_extraction_method="segment_midpoint",
    output_dir="sheets",
    upload_to_cloud=False,     # True -> upload each sheet and add "url" to the JSON
    storage=None,              # ObjectStorage instance, required when upload_to_cloud=True
    keep_local=True,           # False -> don't keep local JPEGs (only valid with upload_to_cloud=True)
    extra_metadata=None,       # e.g. {"title": ..., "category": ..., "clip_hint": ...}
):
    if upload_to_cloud and storage is None:
        raise ValueError("upload_to_cloud=True requires a storage object")
    if not upload_to_cloud:
        keep_local = True      # otherwise the sheets would be lost

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    if keep_local:
        os.makedirs(output_dir, exist_ok=True)

    vid_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    vid_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps and frame_count else (segments[-1]["end"] if segments else 0)

    tile_w = TILE_WIDTH
    tile_h = int(TILE_WIDTH * vid_h / vid_w)

    if video_id is None:
        video_id = os.path.splitext(os.path.basename(video_path))[0]

    transcript = [
        {
            "id": f"seg_{i}",
            "speaker": seg.get("speaker"),
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"],
            "signals": {
                "energy": seg.get("energy"),
                "wps": seg.get("wps"),
                "pause_before": seg.get("pause_before"),
                "face": seg.get("face"),
                "scene_cut": seg.get("scene_cut"),
                "cuts": seg.get("cut_count"),
                "replay": seg.get("replay"),
            },
            "words": seg.get("words", []),
        }
        for i, seg in enumerate(segments)
    ]

    images = []
    total = len(segments) or 1
    done = 0
    bar = progress.Bar("Building contact sheets", total=total)

    for sheet_idx, (group_start, size) in enumerate(split_evenly(len(segments))):
        group = segments[group_start:group_start + size]
        tiles, frames = [], []

        for tile_idx, seg in enumerate(group):
            seg_idx = group_start + tile_idx
            mid_sec = (seg["start"] + seg["end"]) / 2.0

            cap.set(cv2.CAP_PROP_POS_MSEC, mid_sec * 1000.0)
            success, frame = cap.read()

            if success:
                tiles.append(make_tile(frame, f"seg_{seg_idx} | {fmt_time(mid_sec)}", tile_w, tile_h))
            else:
                tiles.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))

            frames.append({
                "segment_id": f"seg_{seg_idx}",
                "timestamp": round(mid_sec, 2),
                "tile_position": tile_idx,
                "row": tile_idx // GRID_COLS,
                "col": tile_idx % GRID_COLS,
                "frame_available": bool(success),
            })

            done += 1
            bar.update(done)

        sheet, n_rows = build_contact_sheet(tiles, tile_w, tile_h)
        image_id = f"sheet_{sheet_idx + 1:03d}"
        jpeg_bytes = encode_jpeg_bytes(sheet)

        entry = {
            "image_id": image_id,
            "start_timestamp": frames[0]["timestamp"],
            "end_timestamp": frames[-1]["timestamp"],
            "grid": f"{GRID_COLS}x{n_rows}",
            "frames": frames,
            "mime_type": "image/jpeg",
        }

        # Local copy
        if keep_local:
            file_path = os.path.join(output_dir, f"{video_id}_{image_id}.jpg")
            with open(file_path, "wb") as f:
                f.write(jpeg_bytes)
            entry["file_path"] = file_path

        # Cloud upload
        if upload_to_cloud:
            key = f"{video_id}/{image_id}.jpg"
            progress.info(f"Uploading {image_id}")
            entry["storage_key"] = key
            entry["url"] = storage.upload_jpeg(jpeg_bytes, key)

        images.append(entry)

    cap.release()
    bar.finish()
    msg = "Contact sheets built and uploaded" if upload_to_cloud else "Contact sheets built"
    progress.ok(msg)

    return {
        "metadata": {
            "source": "video",
            "duration_seconds": round(duration, 2),
            "video_id": video_id,
            "frame_extraction_method": frame_extraction_method,
            **(extra_metadata or {}),
        },
        "transcript": transcript,
        "visual_context": {
            "format": "contact_sheet",
            "max_grid": f"{GRID_COLS}x{MAX_TILES_PER_SHEET // GRID_COLS}",
            "images": images,
        },
    }