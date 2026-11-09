import cv2

from cilpbro import progress

CUT_THRESHOLD = 0.45   # Bhattacharyya distance, 0 = identical, 1 = completely different
SAMPLE_FPS = 10        # frames analysed per second of video
MIN_GAP = 0.3          # ignore cuts closer together than this (flashes, explosions)
LEAD = 0.3             # a cut this close BEFORE a segment still counts for it


def _hist(frame):
    small = cv2.resize(frame, (160, int(160 * frame.shape[0] / frame.shape[1])))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    return cv2.normalize(h, h).flatten()


def detect_cuts(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    step = max(int(round(fps / SAMPLE_FPS)), 1)

    bar = progress.Bar("Detecting scene cuts", total=total_frames)
    cuts, prev, idx = [], None, 0
    while True:
        if idx % step == 0:
            ok, frame = cap.read()
        else:
            ok = cap.grab()          # skip decoding work for frames we don't analyse
            frame = None
        if not ok:
            break

        if frame is not None:
            h = _hist(frame)
            if prev is not None:
                dist = cv2.compareHist(prev, h, cv2.HISTCMP_BHATTACHARYYA)
                t = idx / fps
                if dist > CUT_THRESHOLD and (not cuts or t - cuts[-1] > MIN_GAP):
                    cuts.append(round(t, 2))
            prev = h

        idx += 1
        if idx % 300 == 0:
            bar.update(min(idx, total_frames))

    cap.release()
    bar.finish()
    progress.ok(f"Scene cuts found: {len(cuts)}")
    return cuts


def add_scene_signals(segments, video_path):
    cuts = detect_cuts(video_path)
    for seg in segments:
        inside = [c for c in cuts if seg["start"] - LEAD <= c <= seg["end"]]
        seg["scene_cut"] = bool(inside)
        seg["cut_count"] = len(inside)
    return segments