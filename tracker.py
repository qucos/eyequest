"""Runs MediaPipe face landmark detection on JPEG frames sent from the browser.

Frames are analysed in memory and never written to disk.
"""
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

import gaze

MODEL_PATH = Path(__file__).parent / "models" / "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


BOTH_EYES_PREFERENCE = 0.8


class MediaPipeDetector:
    """Loads the model on first use, because loading takes about a second."""

    def __init__(self):
        self._landmarker = None

    def _load(self):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        if not MODEL_PATH.exists():
            MODEL_PATH.parent.mkdir(exist_ok=True)
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_facial_transformation_matrixes=True,
        )
        self._mp = mp
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect(self, rgb, timestamp_ms):
        if self._landmarker is None:
            self._load()
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.face_landmarks:
            return None
        matrices = result.facial_transformation_matrixes
        return [(p.x, p.y) for p in result.face_landmarks[0]], matrices[0] if matrices else None


def _combined_gaze(models, found, state):
    """Use the model that calibrated best, leaning towards using both eyes together.

    If one eye drifts, its own model and the two-eye model score worse, so the
    steadier eye takes over.
    """
    options = []
    for eye in ("left", "right"):
        if state[eye]["gaze"] is not None:
            options.append((models[eye]["error"], state[eye]["gaze"]))
    if "both" in models and state["left"]["open"] and state["right"]["open"]:
        guess = gaze.predict(models["both"], gaze.both_signal(found["left"], found["right"]))
        options.append((models["both"]["error"] * BOTH_EYES_PREFERENCE, guess))
    return min(options, key=lambda option: option[0])[1] if options else None


class Tracker:
    def __init__(self, detector=None):
        self._detector = detector or MediaPipeDetector()
        self._lock = threading.Lock()
        self._last_ts = 0

    def process(self, jpeg_bytes, saved=None):
        """Return the tracking state for one frame, or None if it can't be decoded."""
        saved = saved or {}
        models = saved.get("models", {})
        head_model = saved.get("head_model")
        tuning = saved.get("tuning")
        buffer = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR) if buffer.size else None
        if frame is None:
            return None
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with self._lock:  # the video detector needs strictly increasing timestamps
            self._last_ts = max(self._last_ts + 1, int(time.monotonic() * 1000))
            detected = self._detector.detect(rgb, self._last_ts)

        if detected is None:
            return {"face": False, "quality": "no_face", "left": None, "right": None}

        landmarks, matrix = detected
        found = gaze.extract(landmarks, width, height, matrix)
        pose = found["pose"]
        state = {
            "face": True,
            "quality": gaze.framing_quality(
                found["face_width"], found["nose_offset"], float(frame.mean())
            ),
        }
        for eye in ("left", "right"):
            features = found[eye]
            model = models.get(eye)
            eye_signal = gaze.signal(features, eye) if features else None
            opened = gaze.is_open(features) and not (model and gaze.is_blinking(model, eye_signal))
            state[eye] = {
                "features": features,
                "open": opened,
                "gaze": gaze.predict(model, eye_signal) if opened and model else None,
                "points": {
                    name: [float(landmarks[index][0]), float(landmarks[index][1])]
                    for name, index in gaze.EYES[eye].items()
                },
            }
        finish = lambda point: gaze.tuned(tuning, gaze.adjust_for_head(head_model, pose, point))
        combined = _combined_gaze(models, found, state)
        state["both"] = {"gaze": finish(combined) if combined else None}
        for eye in ("left", "right"):
            if state[eye]["gaze"]:
                state[eye]["gaze"] = finish(state[eye]["gaze"])
        state["head_moved"] = gaze.head_moved(head_model, pose)
        state["head_pose"] = {
            "yaw": pose[gaze.POSE_YAW],
            "pitch": pose[gaze.POSE_PITCH],
            "roll": pose[gaze.POSE_ROLL],
            "distance": pose[gaze.POSE_DISTANCE],
        }
        return state
