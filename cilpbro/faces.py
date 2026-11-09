import os
import urllib.request
from collections import Counter

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from cilpbro import progress

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = os.path.join("models", "face_landmarker.task")

MIN_FACE_AREA = 0.01      # ignore faces smaller than 1% of the frame (blendshapes are noise)
EMOTION_THRESHOLD = 0.35  # below this, the face counts as neutral
MAX_SIDE = 960            # downscale big frames for speed


def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        progress.info("Downloading face landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def score_emotions(bs):
    """bs: {blendshape_name: score}. Returns {emotion: 0-1 score}. Tune the weights as needed."""
    g = lambda n: bs.get(n, 0.0)
    avg = lambda a, b: (g(a) + g(b)) / 2

    smile_l, smile_r = g("mouthSmileLeft"), g("mouthSmileRight")
    return {
        "happy": avg("mouthSmileLeft", "mouthSmileRight"),
        # one-sided smile
        "smirk": max(smile_l, smile_r) if abs(smile_l - smile_r) > 0.25 else 0.0,
        "surprise": 0.4 * avg("eyeWideLeft", "eyeWideRight")
                    + 0.3 * g("jawOpen") + 0.3 * g("browInnerUp"),
        "anger": 0.6 * avg("browDownLeft", "browDownRight")
                 + 0.2 * avg("noseSneerLeft", "noseSneerRight")
                 + 0.2 * avg("mouthPressLeft", "mouthPressRight"),
        "sad": 0.6 * avg("mouthFrownLeft", "mouthFrownRight") + 0.4 * g("browInnerUp"),
    }


def classify(bs):
    scores = score_emotions(bs)
    label, value = max(scores.items(), key=lambda kv: kv[1])
    if value < EMOTION_THRESHOLD:
        return "neutral", round(value, 2)
    return label, round(value, 2)


def _largest_face(result):
    """Return (blendshape_dict, area) for the biggest face, or None."""
    best = None
    for landmarks, shapes in zip(result.face_landmarks, result.face_blendshapes):
        xs = [p.x for p in landmarks]
        ys = [p.y for p in landmarks]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        if area >= MIN_FACE_AREA and (best is None or area > best[1]):
            best = ({c.category_name: c.score for c in shapes}, area)
    return best


def add_face_signals(segments, video_path):
    """Adds seg['face'] = {count, emotion, intensity, shift} (or None if no usable face)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=_ensure_model()),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=4,
        output_face_blendshapes=True,
    )

    total = len(segments) or 1
    bar = progress.Bar("Analysing faces", total=total)
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        for i, seg in enumerate(segments):
            dur = seg["end"] - seg["start"]
            if dur < 0.6:
                times = [(seg["start"] + seg["end"]) / 2]
            else:  # start, middle, end of the line
                times = [seg["start"] + 0.15 * dur,
                         (seg["start"] + seg["end"]) / 2,
                         seg["end"] - 0.15 * dur]

            samples, max_faces = [], 0
            for t in times:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
                ok, frame = cap.read()
                if not ok:
                    samples.append(None)
                    continue

                h, w = frame.shape[:2]
                if max(h, w) > MAX_SIDE:
                    s = MAX_SIDE / max(h, w)
                    frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                max_faces = max(max_faces, len(result.face_landmarks))

                face = _largest_face(result)
                samples.append(classify(face[0]) if face else None)

            found = [s for s in samples if s]
            if not found:
                seg["face"] = None
            else:
                non_neutral = [s for s in found if s[0] != "neutral"]
                pool = non_neutral or found
                emotion = Counter(e for e, _ in pool).most_common(1)[0][0]
                intensity = max(v for e, v in pool if e == emotion)
                shift = bool(samples[0] and samples[-1] and samples[0][0] != samples[-1][0])
                seg["face"] = {
                    "count": max_faces,
                    "emotion": emotion,
                    "intensity": intensity,
                    "shift": shift,
                }

            bar.update(i + 1)

    cap.release()
    bar.finish()
    progress.ok("Face analysis complete")
    return segments