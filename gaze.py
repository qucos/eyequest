"""Turns face landmarks into per-eye features, and features into screen positions.

Eyes are named anatomically: "right" is the person's own right eye, which sits
on the left side of an unmirrored camera image.

Gaze positions are fractions of the whole screen (not the browser window), so a
calibration done in full screen stays correct in a smaller window.
"""
import numpy as np

# Bump this when the features or model change, so older calibrations get redone.
VERSION = 5

# MediaPipe Face Landmarker indices (478-point model with irises).
# "iris" is the centre point, and the next four indices are the iris outline.
EYES = {
    "right": {"outer": 33, "inner": 133, "top": 159, "bottom": 145, "iris": 468},
    "left": {"outer": 263, "inner": 362, "top": 386, "bottom": 374, "iris": 473},
}
NOSE_TIP = 1
FACE_LEFT = 234
FACE_RIGHT = 454
FOREHEAD = 10
CHIN = 152

# Feature order: iris_x, iris_y, openness, iris_to_lids, head_x, head_y, yaw, pitch, scale,
# then the real 3D head pose (see head_pose below), then upper_lid.
POSE_START = 9
POSE_END = 15
UPPER_LID = 15
FEATURE_COUNT = 16
POSE_YAW, POSE_PITCH, POSE_ROLL, POSE_X, POSE_Y, POSE_DISTANCE = range(6)
OPENNESS, HEAD_X, HEAD_Y, SCALE = 2, 4, 5, 8

BLINK_OPENNESS = 0.14
BLINK_SHARE = 0.7
MIN_TARGETS = 8
# Chosen by testing on real calibration data, scored on dots left out of the fit.
CURVE_WIDTH = 3.0
CURVE_NOISE = 0.05
SHAKY_RATIO = 3.0
SHAKY_FLOOR = 0.012
STRETCH_LIMITS = (0.7, 1.6)
MIN_STRETCH_SPAN = 0.05
TUNE_REACH = 0.25
TUNE_GAIN = 0.8
TUNE_MAX_MISS = 0.45
TUNE_MAX_CLICKS = 40
# How far the head can move before the tracker warns: yaw and pitch degrees, distance share.
MOVED_LIMITS = (15.0, 12.0, 0.25)
# Gaze signal: iris_x, iris_y, openness, upper_lid, then yaw, pitch, head_x, head_y.
EYE_SIGNAL_SIZE = 4
# The smallest spread each signal is scaled by. Without head turns to learn from, tiny head
# wobbles must not look like big movements (degrees, then centimeters, for the head terms).
SIGNAL_FLOORS = (1e-6, 1e-6, 1e-6, 1e-6, 3.0, 3.0, 1.0, 1.0)
MIN_DOT_LOOKS = 3
# How far the head may drift from where it was during calibration before the guide speaks up:
# degrees for turns, centimeters for moves, and a share for distance. On real data, a head tilted
# 8 degrees up made the top of the screen miss by up to 38% of its height.
GUIDE_LIMITS = {"pitch": 4.0, "yaw": 6.0, "x": 3.0, "y": 3.0, "distance": 0.12}


def head_pose(matrix):
    """Real head pose from MediaPipe's 4x4 face matrix.

    Returns [yaw, pitch, roll, x, y, distance]. Angles are degrees and lengths are
    centimeters. Yaw is positive when the nose turns toward the right of the camera image,
    which is the person's own left. Pitch is positive when the nose points down. x is
    positive toward the right of the image and y is positive upward. Exact centimeters
    depend on the camera's lens, so distance is approximate.
    """
    if matrix is None:
        return [0.0] * 6
    m = np.asarray(matrix, dtype=float).reshape(4, 4)
    forward, sideways = m[:3, 2], m[:3, 0]
    return [
        float(np.degrees(np.arctan2(forward[0], forward[2]))),
        float(np.degrees(np.arctan2(-forward[1], forward[2]))),
        float(np.degrees(np.arctan2(sideways[1], sideways[0]))),
        float(m[0, 3]),
        float(m[1, 3]),
        float(-m[2, 3]),
    ]


def eye_features(points, eye, head):
    ids = EYES[eye]
    outer, inner = points[ids["outer"]], points[ids["inner"]]
    centre = (outer + inner) / 2
    axis = inner - outer
    width = float(np.linalg.norm(axis))
    if width < 1e-6:
        return None
    unit = axis / width
    normal = np.array([-unit[1], unit[0]])
    # Averaging the centre and outline points is much steadier than the centre alone.
    iris = points[ids["iris"]: ids["iris"] + 5].mean(axis=0)
    top, bottom = points[ids["top"]], points[ids["bottom"]]
    offset = iris - centre
    in_eye = lambda point: [
        float((np.asarray(point) - centre) @ unit) / width,
        float((np.asarray(point) - centre) @ normal) / width,
    ]
    landmark = in_eye(iris)
    return [
        landmark[0],
        landmark[1],
        float(np.linalg.norm(top - bottom)) / width,
        float((iris - (top + bottom) / 2) @ normal) / width,
        *head,
        in_eye(top)[1],
    ]


def _head_pose(points, frame_w, frame_h):
    """Return [head_x, head_y, yaw, pitch, scale] from the face's geometry."""
    nose = points[NOSE_TIP]
    side_axis = points[FACE_RIGHT] - points[FACE_LEFT]
    tall_axis = points[CHIN] - points[FOREHEAD]
    face_width = float(np.linalg.norm(side_axis))
    face_height = float(np.linalg.norm(tall_axis))
    side_mid = (points[FACE_RIGHT] + points[FACE_LEFT]) / 2
    tall_mid = (points[CHIN] + points[FOREHEAD]) / 2
    yaw = float((nose - side_mid) @ side_axis) / face_width**2 if face_width > 1e-6 else 0.0
    pitch = float((nose - tall_mid) @ tall_axis) / face_height**2 if face_height > 1e-6 else 0.0
    return [
        float(nose[0] - frame_w / 2) / frame_w,
        float(nose[1] - frame_h / 2) / frame_w,
        yaw,
        pitch,
        face_width / frame_w,
    ]


def extract(normalized, frame_w, frame_h, matrix=None):
    """Convert normalized landmarks to features for both eyes plus framing info."""
    points = np.asarray(normalized, dtype=float)[:, :2] * [frame_w, frame_h]
    pose = head_pose(matrix)
    head = _head_pose(points, frame_w, frame_h) + pose
    return {
        "pose": pose,
        "left": eye_features(points, "left", head),
        "right": eye_features(points, "right", head),
        "face_width": head[SCALE - HEAD_X],
        "nose_offset": head[0],
    }


def is_open(features):
    return features is not None and features[OPENNESS] >= BLINK_OPENNESS


def framing_quality(face_width, nose_offset, brightness):
    if brightness < 45:
        return "too_dark"
    if face_width < 0.17:
        return "too_far"
    if face_width > 0.7:
        return "too_close"
    if abs(nose_offset) > 0.22:
        return "off_center"
    return "good"


def signal(features, eye):
    """The part of one eye's features used for gaze (see EYE_SIGNAL_SIZE).

    The upper lid follows up-down looks much more clearly than the iris does, and small head
    tilts move both, so the fit needs to see the lid and the head's pose.

    Each eye measures along its own outer-to-inner axis, so the eyes are mirrored.
    Flipping the left eye makes looking the same way read the same in both eyes.
    """
    sign = -1.0 if eye == "left" else 1.0
    return [
        sign * features[0],
        sign * features[1],
        features[OPENNESS],
        sign * features[UPPER_LID],
        features[POSE_START + POSE_YAW],
        features[POSE_START + POSE_PITCH],
        features[POSE_START + POSE_X],
        features[POSE_START + POSE_Y],
    ]


def both_signal(left, right):
    """Averages the two eyes, which is steadier than either one alone."""
    a, b = signal(left, "left"), signal(right, "right")
    return [(x + y) / 2 for x, y in zip(a, b)]


def _fit_curve(points, targets):
    """A straight-line fit plus a smooth curve that absorbs what the line misses."""
    mean = points.mean(axis=0)
    std = np.maximum(points.std(axis=0), SIGNAL_FLOORS)
    z = (points - mean) / std
    line_terms = np.column_stack([np.ones(len(z)), z])
    line = np.linalg.solve(
        line_terms.T @ line_terms + 1e-3 * np.eye(line_terms.shape[1]), line_terms.T @ targets
    )
    leftover = targets - line_terms @ line
    kernel = _kernel(z, z) + CURVE_NOISE * np.eye(len(z))
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "line": line.tolist(),
        "curve": np.linalg.solve(kernel, leftover).tolist(),
        "anchors": z.tolist(),
    }


def _kernel(a, b):
    squared = ((a[:, None] - b[None]) ** 2).sum(axis=2)
    return np.exp(-squared / (2 * CURVE_WIDTH**2))


def predict(model, eye_signal):
    z = (np.atleast_2d(eye_signal) - np.asarray(model["mean"])) / np.asarray(model["std"])
    line_terms = np.column_stack([np.ones(len(z)), z])
    xy = (
        line_terms @ np.asarray(model["line"])
        + _kernel(z, np.asarray(model["anchors"])) @ np.asarray(model["curve"])
    )[0]
    correction = model.get("correction")
    if correction:
        xy = _apply_correction(correction, xy)
    return [float(np.clip(xy[0], -0.1, 1.1)), float(np.clip(xy[1], -0.1, 1.1))]


def head_moved(head_model, pose):
    """True when the head has moved further than calibration taught the tracker about."""
    if not head_model or not pose or pose[POSE_DISTANCE] <= 0:
        return False
    reference = head_model["reference"]
    if reference[POSE_DISTANCE] <= 0:
        return False
    yaw_limit, pitch_limit, distance_limit = MOVED_LIMITS
    return (
        abs(pose[POSE_YAW] - reference[POSE_YAW]) > yaw_limit
        or abs(pose[POSE_PITCH] - reference[POSE_PITCH]) > pitch_limit
        or abs(pose[POSE_DISTANCE] / reference[POSE_DISTANCE] - 1) > distance_limit
    )


def head_guide(head_model, pose):
    """Which way to move back to the calibrated head position, or None when close enough.

    Directions are the person's own: the camera image isn't mirrored, so a nose turned toward
    the right of the image is turned to the person's left.
    """
    if not head_model or not pose or pose[POSE_DISTANCE] <= 0:
        return None
    reference = head_model["reference"]
    if reference[POSE_DISTANCE] <= 0:
        return None
    drift = {
        "pitch": pose[POSE_PITCH] - reference[POSE_PITCH],
        "yaw": pose[POSE_YAW] - reference[POSE_YAW],
        "x": pose[POSE_X] - reference[POSE_X],
        "y": pose[POSE_Y] - reference[POSE_Y],
        "distance": pose[POSE_DISTANCE] / reference[POSE_DISTANCE] - 1,
    }
    advice = {
        "pitch": ("raise_chin", "lower_chin"),
        "yaw": ("turn_right", "turn_left"),
        "x": ("move_right", "move_left"),
        "y": ("sit_lower", "sit_up"),
        "distance": ("move_closer", "move_back"),
    }
    worst = max(drift, key=lambda name: abs(drift[name]) / GUIDE_LIMITS[name])
    if abs(drift[worst]) <= GUIDE_LIMITS[worst]:
        return None
    return advice[worst][0] if drift[worst] > 0 else advice[worst][1]


def _settled_dots(groups):
    """One steady reading per dot, leaving out dots where the eye or head was moving."""
    wobble = {}
    for target, values in groups.items():
        eye_part = values[:, :EYE_SIGNAL_SIZE]
        offsets = eye_part - np.median(eye_part, axis=0)
        wobble[target] = float(np.percentile(np.linalg.norm(offsets, axis=1), 85))
    limit = max(float(np.median(list(wobble.values()))) * SHAKY_RATIO, SHAKY_FLOOR)
    return {t: np.median(v, axis=0) for t, v in groups.items() if wobble[t] <= limit}


def fit_eye(pairs, aspect):
    """Fit from (signal, target) pairs.

    Returns (model, error), where error is the median miss in screen heights on dots
    left out of the fit, or (None, None) when too few steady dots were seen.
    """
    groups = {}
    for eye_signal, target in pairs:
        groups.setdefault(tuple(target), []).append(eye_signal)
    enough = {t: np.asarray(v, dtype=float) for t, v in groups.items() if len(v) >= MIN_DOT_LOOKS}
    settled = _settled_dots(enough) if enough else {}
    if len(settled) < MIN_TARGETS:
        return None, None

    targets = np.asarray(list(settled), dtype=float)
    points = np.asarray(list(settled.values()), dtype=float)
    misses = []
    for index in range(len(targets)):
        keep = np.arange(len(targets)) != index
        guess = predict(_fit_curve(points[keep], targets[keep]), points[index])
        delta = (np.asarray(guess) - targets[index]) * [aspect, 1.0]
        misses.append(float(np.linalg.norm(delta)))
    error = float(np.median(misses))

    model = _fit_curve(points, targets)
    model["error"] = error
    # A blink closes the lids more than looking down does, whichever way this person looks.
    model["blink_below"] = float(points[:, 2].min()) * BLINK_SHARE
    return model, error


def is_blinking(model, eye_signal):
    return eye_signal[2] < model.get("blink_below", BLINK_OPENNESS)


def fit_calibration(samples, aspect):
    """Fit each eye and both together. Samples look like {"target", "left", "right"}.

    Dots seen with the head turned ("phase") train the same fit, which is how it learns
    what head movement does to the eye readings.
    """
    pairs = {"left": [], "right": [], "both": []}
    for sample in samples:
        left, right = sample.get("left"), sample.get("right")
        left_open, right_open = is_open(left), is_open(right)
        if left_open:
            pairs["left"].append((signal(left, "left"), sample["target"]))
        if right_open:
            pairs["right"].append((signal(right, "right"), sample["target"]))
        if left_open and right_open:
            pairs["both"].append((both_signal(left, right), sample["target"]))

    result = {"version": VERSION, "models": {}, "errors": {}}
    for name, eye_pairs in pairs.items():
        model, error = fit_eye(eye_pairs, aspect)
        if model is not None:
            result["models"][name] = model
            result["errors"][name] = error
    if not result["models"]:
        return None
    worst = max(result["errors"].values())
    # Webcams are imprecise, so the on-screen hit zone grows with the measured error.
    result["radius"] = float(np.clip(worst * 1.15, 0.10, 0.24))
    result["error"] = worst
    poses = [
        sample[eye][POSE_START:POSE_END]
        for sample in samples
        if not sample.get("phase")
        for eye in ("left", "right")
        if is_open(sample.get(eye))
    ]
    result["head_model"] = {"reference": np.median(np.asarray(poses, dtype=float), axis=0).tolist()} if poses else None
    return result


def _median_miss(guesses, targets, aspect):
    return float(np.median(np.linalg.norm((guesses - targets) * [aspect, 1.0], axis=1)))


def _axis_correction(guesses, targets):
    """Fit [stretch, shift] for one axis, using a plain shift if the dots can't support more."""
    if np.ptp(guesses) >= MIN_STRETCH_SPAN:
        stretch, shift = np.polyfit(guesses, targets, 1)
        if STRETCH_LIMITS[0] <= stretch <= STRETCH_LIMITS[1]:
            return [float(stretch), float(shift)]
    return [1.0, float(np.median(targets - guesses))]


def _apply_correction(correction, point):
    return np.array([
        correction["x"][0] * point[0] + correction["x"][1],
        correction["y"][0] * point[1] + correction["y"][1],
    ])


def _correction_for(guesses, targets):
    return {
        "x": _axis_correction(guesses[:, 0], targets[:, 0]),
        "y": _axis_correction(guesses[:, 1], targets[:, 1]),
    }


def fit_correction(guesses, targets, aspect):
    """Measure accuracy on fresh dots, and remove any steady shift or squeeze.

    "after" scores each dot with a correction worked out from the other dots only, so
    it predicts real accuracy instead of flattering the fit. Returns before, after and
    correction (None if correcting doesn't help).
    """
    guesses = np.asarray(guesses, dtype=float)
    targets = np.asarray(targets, dtype=float)
    before = _median_miss(guesses, targets, aspect)
    corrected = np.zeros_like(guesses)
    for index in range(len(guesses)):
        keep = np.arange(len(guesses)) != index
        corrected[index] = _apply_correction(
            _correction_for(guesses[keep], targets[keep]), guesses[index]
        )
    after = _median_miss(corrected, targets, aspect)
    if after >= before:
        return {"before": before, "after": before, "correction": None}
    return {"before": before, "after": after, "correction": _correction_for(guesses, targets)}


def tuned(entries, point):
    """Apply click fine-tuning. Each click nudges gaze near where it was made."""
    if not entries:
        return point
    spots = np.asarray([entry["at"] for entry in entries], dtype=float)
    nudges = np.asarray([entry["nudge"] for entry in entries], dtype=float)
    squared = ((spots - np.asarray(point)) ** 2).sum(axis=1)
    reach = np.exp(-squared / (2 * TUNE_REACH**2))
    moved = np.asarray(point) + (reach[:, None] * nudges).sum(axis=0)
    return [float(np.clip(moved[0], -0.1, 1.1)), float(np.clip(moved[1], -0.1, 1.1))]


def add_tune(entries, guess, target, aspect):
    """Learn from one click. Returns (entries, miss, used).

    A click far from the gaze probably wasn't looked at, so it isn't learned from.
    """
    gap = np.asarray(target, dtype=float) - np.asarray(guess, dtype=float)
    miss = float(np.linalg.norm(gap * [aspect, 1.0]))
    if miss > TUNE_MAX_MISS:
        return entries, miss, False
    entry = {"at": [float(guess[0]), float(guess[1])], "nudge": (gap * TUNE_GAIN).tolist()}
    return ([*entries, entry])[-TUNE_MAX_CLICKS:], miss, True
