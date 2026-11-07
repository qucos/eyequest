"""Turns face landmarks into per-eye features, and features into screen positions.

Eyes are named anatomically: "right" is the person's own right eye, which sits
on the left side of an unmirrored camera image.

Gaze positions are fractions of the whole screen (not the browser window), so a
calibration done in full screen stays correct in a smaller window.
"""
import numpy as np

# Bump this when the features or model change, so older calibrations get redone.
VERSION = 4

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
# then the real 3D head pose (see head_pose below).
POSE_START = 9
FEATURE_COUNT = 15
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
# How far the head can move before the tracker warns, with and without head correction.
MOVED_LIMITS = {True: (25.0, 20.0, 0.35), False: (8.0, 8.0, 0.2)}
# Screen fractions per centimeter: a screen from 10 cm to 100 cm across.
HEAD_GAIN_LIMITS = (0.01, 0.1)
MIN_HEAD_DOTS = 4
MIN_HEAD_LOOKS = 5
HEAD_MUST_IMPROVE = 0.95


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
    return [
        float(offset @ unit) / width,
        float(offset @ normal) / width,
        float(np.linalg.norm(top - bottom)) / width,
        float((iris - (top + bottom) / 2) @ normal) / width,
        *head,
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
    """The part of one eye's features used for gaze: [iris_x, iris_y, openness].

    Each eye measures along its own outer-to-inner axis, so the eyes are mirrored.
    Flipping the left eye makes looking the same way read the same in both eyes.
    """
    sign = -1.0 if eye == "left" else 1.0
    return [sign * features[0], sign * features[1], features[OPENNESS]]


def both_signal(left, right):
    """Averages the two eyes, which is steadier than either one alone."""
    a, b = signal(left, "left"), signal(right, "right")
    return [(a[i] + b[i]) / 2 for i in range(3)]


def _fit_curve(points, targets):
    """A straight-line fit plus a smooth curve that absorbs what the line misses."""
    mean = points.mean(axis=0)
    std = points.std(axis=0)
    std[std < 1e-6] = 1.0
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
    """True when the head has moved further than the tracker can account for."""
    if not head_model or not pose or pose[POSE_DISTANCE] <= 0:
        return False
    reference = head_model["reference"]
    if reference[POSE_DISTANCE] <= 0:
        return False
    yaw_limit, pitch_limit, distance_limit = MOVED_LIMITS[bool(head_model["used"])]
    return (
        abs(pose[POSE_YAW] - reference[POSE_YAW]) > yaw_limit
        or abs(pose[POSE_PITCH] - reference[POSE_PITCH]) > pitch_limit
        or abs(pose[POSE_DISTANCE] / reference[POSE_DISTANCE] - 1) > distance_limit
    )


def _pose_shift(pose, reference):
    """How far (cm) the head's aim on the screen plane moved: [toward screen left, down]."""
    aim = lambda state, angle: state[POSE_DISTANCE] * np.tan(np.radians(state[angle]))
    return (
        aim(pose, POSE_YAW) - aim(reference, POSE_YAW) + pose[POSE_X] - reference[POSE_X],
        aim(pose, POSE_PITCH) - aim(reference, POSE_PITCH) - (pose[POSE_Y] - reference[POSE_Y]),
    )


def _head_adjusted(point, pose, reference, gains):
    left, down = _pose_shift(pose, reference)
    # Same eye angles from further away land further from the camera, at top center.
    stretch = pose[POSE_DISTANCE] / reference[POSE_DISTANCE] - 1
    return [
        float(point[0] - gains[0] * left + (point[0] - 0.5) * stretch),
        float(point[1] + gains[1] * down + point[1] * stretch),
    ]


def adjust_for_head(head_model, pose, point):
    """Apply the head correction, if it proved itself during calibration."""
    if not head_model or not head_model["used"] or not pose or pose[POSE_DISTANCE] <= 0:
        return point
    moved = _head_adjusted(point, pose, head_model["reference"], head_model["gains"])
    return [float(np.clip(moved[0], -0.1, 1.1)), float(np.clip(moved[1], -0.1, 1.1))]


def _fit_gains(dots, reference):
    """Fit how far one centimeter of head aim moves gaze on screen, for each axis."""
    top_x = bottom_x = top_y = bottom_y = 0.0
    for dot in dots:
        left, down = _pose_shift(dot["pose"], reference)
        stretch = dot["pose"][POSE_DISTANCE] / reference[POSE_DISTANCE] - 1
        need_x = dot["target"][0] - dot["guess"][0] - (dot["guess"][0] - 0.5) * stretch
        need_y = dot["target"][1] - dot["guess"][1] - dot["guess"][1] * stretch
        top_x += -left * need_x
        bottom_x += left * left
        top_y += down * need_y
        bottom_y += down * down
    return [
        float(np.clip(top_x / max(bottom_x, 1e-6), *HEAD_GAIN_LIMITS)),
        float(np.clip(top_y / max(bottom_y, 1e-6), *HEAD_GAIN_LIMITS)),
    ]


def _head_dots(samples, model, eye):
    """One steady reading per dot per head direction."""
    groups = {}
    for sample in samples:
        left, right = sample.get("left"), sample.get("right")
        chosen = None
        if eye == "both" and is_open(left) and is_open(right):
            chosen = (both_signal(left, right), left)
        elif eye != "both" and is_open(sample.get(eye)):
            chosen = (signal(sample[eye], eye), sample[eye])
        if chosen:
            key = (tuple(sample["target"]), sample["phase"])
            groups.setdefault(key, []).append([*chosen[0], *chosen[1][POSE_START:]])
    dots = []
    for (target, phase), values in groups.items():
        if len(values) >= MIN_HEAD_LOOKS:
            middle = np.median(np.asarray(values, dtype=float), axis=0)
            dots.append({
                "target": target,
                "phase": phase,
                "guess": predict(model, middle[:3]),
                "pose": middle[3:].tolist(),
            })
    return dots


def _dot_miss(dot, point, aspect):
    return float(np.hypot((point[0] - dot["target"][0]) * aspect, point[1] - dot["target"][1]))


def fit_head(still_samples, head_samples, models, aspect):
    """Learn the head correction, and only use it if it helps on unseen head positions.

    Each head direction is scored with a correction fitted to the other directions.
    """
    poses = [
        s[eye][POSE_START:] for s in still_samples for eye in ("left", "right") if is_open(s.get(eye))
    ]
    if not poses:
        return None
    reference = np.median(np.asarray(poses, dtype=float), axis=0).tolist()
    result = {"reference": reference, "gains": [0.0, 0.0], "used": False, "before": None, "after": None}
    usable = reference[POSE_DISTANCE] > 0 and models
    eye = ("both" if "both" in models else min(models, key=lambda name: models[name]["error"])) if usable else None
    dots = _head_dots(head_samples, models[eye], eye) if eye else []
    if len(dots) < MIN_HEAD_DOTS:
        return result

    before, after = [], []
    for dot in dots:
        others = [other for other in dots if other["phase"] != dot["phase"]]
        before.append(_dot_miss(dot, dot["guess"], aspect))
        if others:
            gains = _fit_gains(others, reference)
            after.append(_dot_miss(dot, _head_adjusted(dot["guess"], dot["pose"], reference, gains), aspect))
    result["before"] = float(np.median(before))
    result["after"] = float(np.median(after)) if after else result["before"]
    result["gains"] = _fit_gains(dots, reference)
    result["used"] = result["after"] < result["before"] * HEAD_MUST_IMPROVE
    return result


def _settled_dots(groups):
    """One steady reading per dot, leaving out dots where the eye or head was moving."""
    wobble = {}
    for target, values in groups.items():
        offsets = values - np.median(values, axis=0)
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
    settled = _settled_dots({t: np.asarray(v, dtype=float) for t, v in groups.items()})
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
    """Fit each eye and both together. Samples look like {"target", "left", "right"}."""
    still = [sample for sample in samples if not sample.get("phase")]
    turned = [sample for sample in samples if sample.get("phase")]
    pairs = {"left": [], "right": [], "both": []}
    for sample in still:
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
    result["head_model"] = fit_head(still, turned, result["models"], aspect)
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
