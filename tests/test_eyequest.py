import io
import json
import socket
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import app as eyequest
import curriculum
import gaze
import progress
import research
import start


# ---------- streak rules ----------

@pytest.mark.parametrize(
    "streak,last,freezes,today,expected",
    [
        (0, None, 0, date(2026, 1, 10), (1, 0, "started")),
        (4, "2026-01-10", 0, date(2026, 1, 10), (4, 0, "same_day")),
        (4, "2026-01-09", 0, date(2026, 1, 10), (5, 0, "extended")),
        (4, "2026-01-08", 1, date(2026, 1, 10), (5, 0, "frozen")),
        (4, "2026-01-08", 0, date(2026, 1, 10), (1, 0, "reset")),
        (6, "2026-01-09", 0, date(2026, 1, 10), (7, 1, "extended")),
        (13, "2026-01-09", 2, date(2026, 1, 10), (14, 2, "extended")),
    ],
)
def test_update_streak(streak, last, freezes, today, expected):
    assert progress.update_streak(streak, last, freezes, today) == expected


def test_displayed_streak_drops_to_zero_when_unsavable():
    today = date(2026, 1, 10)
    assert progress.current_streak(5, "2026-01-09", 0, today) == 5
    assert progress.current_streak(5, "2026-01-08", 1, today) == 5
    assert progress.current_streak(5, "2026-01-08", 0, today) == 0


def test_xp_stops_after_daily_limit():
    assert progress.xp_for(3, 0) == 15
    assert progress.xp_for(3, progress.DAILY_LESSON_LIMIT) == 0


# ---------- gaze maths ----------

def fake_landmarks(ix=0.0, iy=0.0, head=(0.5, 0.5), yaw=0.0):
    """A face whose irises are shifted by (ix, iy) eye-widths within each eye."""
    points = np.tile(np.array(head, dtype=float), (478, 1))
    points[gaze.FACE_LEFT] = [head[0] - 0.2, head[1]]
    points[gaze.FACE_RIGHT] = [head[0] + 0.2, head[1]]
    points[gaze.FOREHEAD] = [head[0], head[1] - 0.25]
    points[gaze.CHIN] = [head[0], head[1] + 0.25]
    points[gaze.NOSE_TIP] = [head[0] + yaw * 0.4, head[1]]
    width = 0.05
    for eye, side in (("right", -0.08), ("left", 0.08)):
        ids = gaze.EYES[eye]
        centre = np.array([head[0] + side, head[1] - 0.05])
        direction = np.array([1.0 if eye == "right" else -1.0, 0.0])
        points[ids["outer"]] = centre - direction * width / 2
        points[ids["inner"]] = centre + direction * width / 2
        points[ids["top"]] = centre + [0, -0.012]
        points[ids["bottom"]] = centre + [0, 0.012]
        # Both eyes look the same way, so the iris shifts the same way in the image.
        iris = centre + np.array([ix * width, iy * width])
        # The outline surrounds the centre, so all five points average to the centre.
        points[ids["iris"]: ids["iris"] + 5] = iris + np.array(
            [[0, 0], [0.004, 0], [-0.004, 0], [0, 0.004], [0, -0.004]]
        )
    return points


def fake_matrix(yaw=0.0, pitch=0.0, x=0.0, y=0.0, distance=60.0):
    """A face matrix with the nose turned yaw degrees and tipped pitch degrees."""
    a, b = np.radians(yaw), np.radians(pitch)
    forward = np.array([np.sin(a) * np.cos(b), -np.sin(b), np.cos(a) * np.cos(b)])
    sideways = np.array([np.cos(a), 0.0, -np.sin(a)])
    upward = np.cross(forward, sideways)
    matrix = np.eye(4)
    matrix[:3, 0], matrix[:3, 1], matrix[:3, 2] = sideways, upward, forward
    matrix[:3, 3] = [x, y, -distance]
    return matrix


def test_right_eye_is_on_image_left():
    assert fake_landmarks()[gaze.EYES["right"]["iris"]][0] < 0.5


def test_features_track_iris_shift():
    features = gaze.extract(fake_landmarks(ix=0.1, iy=0.05), 1, 1)
    # The eyes are measured in mirrored directions, and signal() lines them back up.
    assert features["left"][0] == pytest.approx(-0.1)
    assert features["right"][0] == pytest.approx(0.1)
    assert gaze.signal(features["left"], "left")[:2] == pytest.approx([0.1, 0.05])
    assert gaze.signal(features["right"], "right")[:2] == pytest.approx([0.1, 0.05])
    assert gaze.both_signal(features["left"], features["right"])[:2] == pytest.approx([0.1, 0.05])
    assert features["left"][2] == pytest.approx(0.48)
    assert len(features["left"]) == gaze.FEATURE_COUNT


def test_closed_eye_counts_as_blink():
    points = fake_landmarks()
    ids = gaze.EYES["left"]
    points[ids["top"]] = points[ids["bottom"]]
    found = gaze.extract(points, 1, 1)
    assert not gaze.is_open(found["left"])
    assert gaze.is_open(found["right"])


def calibration_samples(seed=1, noise=0.003):
    rng = np.random.default_rng(seed)
    samples = []
    targets = [(x, y) for x in (0.1, 0.5, 0.9) for y in (0.1, 0.5, 0.9)]
    targets += [(0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7)]
    for tx, ty in targets:
        for _ in range(15):
            ix = (tx - 0.5) * 0.3 + rng.normal(0, noise)
            iy = (ty - 0.5) * 0.15 + rng.normal(0, noise)
            found = gaze.extract(fake_landmarks(ix, iy), 640, 480)
            samples.append({"target": [tx, ty], "left": found["left"], "right": found["right"]})
    return samples


def test_calibration_recovers_gaze_position():
    fitted = gaze.fit_calibration(calibration_samples(), 1.6)
    assert set(fitted["models"]) == {"left", "right", "both"}
    assert fitted["error"] < 0.1
    found = gaze.extract(fake_landmarks(0.06, 0.03), 640, 480)  # true gaze: (0.7, 0.7)
    for name, eye_signal in (
        ("left", gaze.signal(found["left"], "left")),
        ("right", gaze.signal(found["right"], "right")),
        ("both", gaze.both_signal(found["left"], found["right"])),
    ):
        x, y = gaze.predict(fitted["models"][name], eye_signal)
        assert x == pytest.approx(0.7, abs=0.05)
        assert y == pytest.approx(0.7, abs=0.05)


def test_shaky_dots_are_left_out_of_the_fit():
    samples = calibration_samples()
    rng = np.random.default_rng(5)
    shaky_target = [0.5, 0.5]
    for sample in samples:
        if sample["target"] == shaky_target:
            found = gaze.extract(fake_landmarks(rng.uniform(-0.2, 0.2), rng.uniform(-0.1, 0.1)), 640, 480)
            sample["left"], sample["right"] = found["left"], found["right"]
    fitted = gaze.fit_calibration(samples, 1.6)
    assert len(fitted["models"]["left"]["anchors"]) == 12


def test_head_pose_reads_real_angles_and_distance():
    pose = gaze.head_pose(fake_matrix(yaw=12, pitch=-7, x=3, y=-2, distance=55))
    assert pose[gaze.POSE_YAW] == pytest.approx(12, abs=0.5)
    assert pose[gaze.POSE_PITCH] == pytest.approx(-7, abs=0.5)
    assert pose[gaze.POSE_X] == pytest.approx(3)
    assert pose[gaze.POSE_Y] == pytest.approx(-2)
    assert pose[gaze.POSE_DISTANCE] == pytest.approx(55)
    assert gaze.head_pose(None) == [0.0] * 6


SCREEN_WIDTH_CM, SCREEN_HEIGHT_CM = 30.0, 19.0


def turned_samples(rng=None, directions=None):
    """Head turns where the eye keeps the same in-head angle, so only the head aims gaze."""
    rng = rng or np.random.default_rng(9)
    turns = directions or {"left": (10, 0), "right": (-10, 0), "up": (0, -8), "down": (0, 8)}
    samples = []
    for phase, (yaw, pitch) in turns.items():
        for eye_tx, eye_ty in [(0.35, 0.35), (0.65, 0.65)]:
            # A head turned toward screen left needs a look further right to hold a spot.
            shift_x = 60 * np.tan(np.radians(yaw)) / SCREEN_WIDTH_CM
            shift_y = 60 * np.tan(np.radians(pitch)) / SCREEN_HEIGHT_CM
            target = [float(eye_tx - shift_x), float(eye_ty + shift_y)]
            for _ in range(15):
                ix = (eye_tx - 0.5) * 0.3 + rng.normal(0, 0.002)
                iy = (eye_ty - 0.5) * 0.15 + rng.normal(0, 0.002)
                found = gaze.extract(fake_landmarks(ix, iy), 640, 480, fake_matrix(yaw, pitch))
                samples.append({"target": target, "phase": phase, "left": found["left"], "right": found["right"]})
    return samples


def still_samples():
    samples = calibration_samples()
    for sample in samples:
        for eye in ("left", "right"):
            sample[eye][gaze.POSE_START + gaze.POSE_DISTANCE] = 60.0
    return samples


def turned_miss(model):
    """Median miss on head-turned dots, with each eye signal built as the tracker builds it."""
    misses = []
    for sample in turned_samples(np.random.default_rng(4)):
        guess = gaze.predict(model, gaze.both_signal(sample["left"], sample["right"]))
        misses.append(np.hypot(guess[0] - sample["target"][0], guess[1] - sample["target"][1]))
    return float(np.median(misses))


def test_fit_learns_what_head_turns_do_to_gaze():
    still_only = gaze.fit_calibration(still_samples(), 1.6)["models"]["both"]
    with_turns = gaze.fit_calibration(still_samples() + turned_samples(), 1.6)["models"]["both"]
    assert turned_miss(with_turns) < turned_miss(still_only) * 0.5


def test_small_head_wobbles_do_not_shift_gaze_without_head_training():
    model = gaze.fit_calibration(still_samples(), 1.6)["models"]["both"]
    found = gaze.extract(fake_landmarks(0.02, 0.01), 640, 480, fake_matrix())
    wobbled = gaze.extract(fake_landmarks(0.02, 0.01), 640, 480, fake_matrix(yaw=0.5, pitch=0.5))
    steady = gaze.predict(model, gaze.both_signal(found["left"], found["right"]))
    moved = gaze.predict(model, gaze.both_signal(wobbled["left"], wobbled["right"]))
    assert np.hypot(steady[0] - moved[0], steady[1] - moved[1]) < 0.05


def test_gaze_signal_uses_the_lid_and_head_pose():
    found = gaze.extract(fake_landmarks(0.02, 0.01), 640, 480, fake_matrix(yaw=5, pitch=-3))
    left = gaze.signal(found["left"], "left")
    assert len(left) == len(gaze.SIGNAL_FLOORS)
    assert left[3] == -found["left"][gaze.UPPER_LID]
    assert left[4] == pytest.approx(found["pose"][gaze.POSE_YAW])


def test_head_moved_warns_past_what_calibration_covered():
    head = {"reference": gaze.head_pose(fake_matrix())}
    assert not gaze.head_moved(head, gaze.head_pose(fake_matrix(yaw=10)))
    assert gaze.head_moved(head, gaze.head_pose(fake_matrix(yaw=30)))


def test_correction_removes_a_steady_shift():
    targets = [[0.2, 0.2], [0.8, 0.2], [0.5, 0.5], [0.2, 0.8], [0.8, 0.8]]
    guesses = [[x + 0.1, y - 0.05] for x, y in targets]
    result = gaze.fit_correction(guesses, targets, 1.6)
    assert result["after"] < 0.01 < result["before"]
    assert result["correction"]["x"] == pytest.approx([1.0, -0.1])
    assert result["correction"]["y"] == pytest.approx([1.0, 0.05])


def test_correction_undoes_a_squeezed_axis():
    targets = [[0.2, 0.22], [0.8, 0.22], [0.5, 0.5], [0.2, 0.78], [0.8, 0.78]]
    guesses = [[x, 0.5 + (y - 0.5) * 0.7 + 0.1] for x, y in targets]
    result = gaze.fit_correction(guesses, targets, 1.6)
    assert result["after"] < 0.01 < result["before"]
    assert result["correction"]["y"][0] == pytest.approx(1 / 0.7)


def test_a_wild_stretch_falls_back_to_a_shift():
    targets = np.array([0.2, 0.5, 0.8, 0.3, 0.7])
    assert gaze._axis_correction(targets * 0.2, targets)[0] == 1.0


def test_correction_score_is_not_flattered_by_its_own_dots():
    targets = np.array([[0.2, 0.2], [0.8, 0.2], [0.5, 0.5], [0.2, 0.8], [0.8, 0.8]])
    noise = np.array([[0.1, -0.1], [-0.1, 0.1], [0.08, 0.09], [-0.09, -0.08], [0.1, 0.1]])
    result = gaze.fit_correction((targets + noise).tolist(), targets.tolist(), 1.0)
    assert result["after"] >= 0.1


def test_correction_is_skipped_when_it_cannot_help():
    targets = [[0.2, 0.2], [0.8, 0.2], [0.5, 0.5]]
    result = gaze.fit_correction(targets, targets, 1.6)
    assert result["correction"] is None and result["after"] == result["before"]


def test_calibration_needs_enough_targets():
    assert gaze.fit_calibration(calibration_samples()[:30], 1.6) is None


# ---------- starting the app ----------

def test_install_counts_as_done_only_after_it_finishes(tmp_path):
    venv, requirements = tmp_path / ".venv", tmp_path / "requirements.txt"
    venv.mkdir()
    requirements.write_text("flask\n")
    assert start.needs_install(venv, requirements)
    (venv / ".installed").write_text(start.requirements_id(requirements))
    assert not start.needs_install(venv, requirements)
    requirements.write_text("flask\nnumpy\n")
    assert start.needs_install(venv, requirements)


def test_a_failed_install_stays_unfinished_and_says_what_to_do(tmp_path, monkeypatch):
    venv, requirements = tmp_path / ".venv", tmp_path / "requirements.txt"
    requirements.write_text("flask\n")
    monkeypatch.setattr(start, "_runs", lambda _command: True)
    monkeypatch.setattr(start.subprocess, "run", lambda *_a, **_k: SimpleNamespace(returncode=1))
    with pytest.raises(start.SetupError, match="run this command again"):
        start.install(venv, requirements)
    assert start.needs_install(venv, requirements)


def test_a_busy_port_is_skipped():
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = taken.getsockname()[1]
        assert not start.is_free(port)
        assert start.free_port(port, 50) != port
        with pytest.raises(start.SetupError, match="in use"):
            start.free_port(port, 1)


def test_waiting_for_the_app_stops_if_it_crashes():
    crashed = SimpleNamespace(poll=lambda: 1)
    assert start.wait_until_ready("http://localhost:9", crashed, seconds=30) is False


# ---------- full screen ----------

def test_opening_a_page_directly_loads_it_in_the_full_screen_frame(world):
    client, _, _ = world
    page = client.get("/settings?tab=1", headers={"Sec-Fetch-Dest": "document"}).get_data(as_text=True)
    assert '<iframe id="app" src="/settings?tab=1"' in page
    assert 'allow="camera; fullscreen"' in page


def test_pages_inside_the_frame_load_normally(world):
    client, _, _ = world
    page = client.get("/login", headers={"Sec-Fetch-Dest": "iframe"}).get_data(as_text=True)
    assert '<iframe id="app"' not in page
    assert 'name="password"' in page


def test_frame_cannot_be_pointed_at_another_website(world):
    client, _, _ = world
    for path in ("//evil.example/page", "/\\evil.example/page"):
        page = client.get(path, headers={"Sec-Fetch-Dest": "document"}).get_data(as_text=True)
        assert 'src="//' not in page and 'src="/\\' not in page


# ---------- lesson path ----------

def test_lessons_unlock_in_order():
    first, second = curriculum.ORDER[:2]
    assert curriculum.is_unlocked(first, {})
    assert not curriculum.is_unlocked(second, {})
    assert curriculum.is_unlocked(second, {first: 1})
    assert not curriculum.is_unlocked("mix", {first: 1})
    assert curriculum.next_lesson_id({k: 1 for k in curriculum.ORDER}) == "mix"


def test_unsure_users_alternate_eyes():
    assert curriculum.solo_eye("left", 1) == "left"
    assert {curriculum.solo_eye("unknown", 1), curriculum.solo_eye("unknown", 2)} == {"left", "right"}


# ---------- the web app ----------

class FakeDetector:
    def __init__(self):
        self.face = True

    def detect(self, _rgb, _timestamp):
        return (fake_landmarks().tolist(), fake_matrix()) if self.face else None


class Clock:
    def __init__(self):
        self.value = date(2026, 3, 2)

    def __call__(self):
        return self.value


@pytest.fixture
def world(tmp_path):
    clock, detector = Clock(), FakeDetector()
    client = eyequest.create_app(tmp_path, detector, clock).test_client()
    return client, clock, detector


def signup(client, name="bryan", password="correct horse"):
    return client.post("/signup", data={"username": name, "password": password})


def finish(client, lesson_id, score=90):
    step = {"game": "firefly", "eye": "both", "score": score, "metrics": {"left_hit": 80, "right_hit": 70}}
    return client.post("/api/lesson/complete", json={"lesson_id": lesson_id, "steps": [step]})


def bright_jpeg():
    return cv2.imencode(".jpg", np.full((480, 640, 3), 128, dtype=np.uint8))[1].tobytes()


def test_pages_require_login(world):
    client, _, _ = world
    assert client.get("/learn").headers["Location"].endswith("/login")
    assert client.post("/api/frame", data=b"x").status_code == 401


def test_signup_login_logout(world):
    client, _, _ = world
    assert signup(client).headers["Location"].endswith("/onboarding")
    client.post("/logout")
    assert client.post("/login", data={"username": "bryan", "password": "wrong-password"}).status_code == 401
    assert client.get("/learn").status_code == 302
    good = client.post("/login", data={"username": "BRYAN", "password": "correct horse"})
    assert good.status_code == 302


def test_duplicate_and_weak_signups_rejected(tmp_path):
    client = eyequest.create_app(tmp_path, FakeDetector(), Clock()).test_client()
    signup(client)
    other = eyequest.create_app(tmp_path, FakeDetector(), Clock()).test_client()
    assert b"taken" in signup(other).data
    assert signup(other, "newname", "short").status_code == 400


def test_password_is_hashed(tmp_path):
    client = eyequest.create_app(tmp_path, FakeDetector(), Clock()).test_client()
    signup(client)
    assert b"correct horse" not in (tmp_path / "eyequest.db").read_bytes()


def test_other_hosts_are_refused(world):
    client, _, _ = world
    assert client.get("/", headers={"Host": "evil.example"}).status_code == 403


def test_frame_endpoint(world):
    client, _, detector = world
    signup(client)
    assert client.post("/api/frame", data=b"not a jpeg").status_code == 400
    state = client.post("/api/frame", data=bright_jpeg()).get_json()
    assert state["face"] and state["left"]["open"] and state["left"]["gaze"] is None
    assert len(state["right"]["points"]["iris"]) == 2
    assert state["head_pose"]["distance"] == pytest.approx(60)
    assert client.get("/eyes").status_code == 200
    detector.face = False
    assert client.post("/api/frame", data=bright_jpeg()).get_json()["quality"] == "no_face"


def test_calibration_then_frames_report_gaze(world):
    client, _, _ = world
    signup(client)
    client.post("/onboarding", data={"weak_eye": "left", "goal_xp": "20"})
    assert client.get("/lesson/u1l1").headers["Location"].endswith("/calibrate?lesson=u1l1")
    result = client.post("/api/calibrate", json={"samples": calibration_samples(), "aspect": 1.6}).get_json()
    assert result["ok"]
    state = client.post("/api/frame", data=bright_jpeg()).get_json()
    assert state["left"]["gaze"] is not None and state["both"]["gaze"] is not None
    assert client.get("/lesson/u1l1").status_code == 200
    assert client.post("/api/calibrate", json={"samples": [{"target": "x"}], "aspect": 1.6}).status_code == 400

    gaze_now = state["left"]["gaze"]
    points = [
        {"target": [x, y], "left": [x + 0.05, y], "right": [x + 0.05, y], "both": [x + 0.05, y]}
        for x, y in [(0.2, 0.22), (0.8, 0.22), (0.5, 0.5), (0.2, 0.78), (0.8, 0.78)]
    ]
    assert client.post("/api/calibrate/check", json={"points": ["bad"], "aspect": 1.6}).status_code == 400
    checked = client.post("/api/calibrate/check", json={"points": points, "aspect": 1.6}).get_json()
    assert checked["ok"] and checked["accuracy"] < 0.01
    assert checked["radius"] == 0.08
    # The steady shift is now removed from live frames too.
    corrected = client.post("/api/frame", data=bright_jpeg()).get_json()["left"]["gaze"]
    assert corrected[0] == pytest.approx(gaze_now[0] - 0.05, abs=0.01)
    assert client.post("/api/calibrate/check", json={"points": points, "aspect": 1.6}).status_code == 409


def mixed_up_samples():
    """Eye readings paired with the wrong dots, as when the person wasn't looking at them."""
    samples = calibration_samples()
    targets = sorted({tuple(s["target"]) for s in samples})
    swap = dict(zip(targets, targets[::-1][3:] + targets[::-1][:3]))
    return [{**s, "target": list(swap[tuple(s["target"])])} for s in samples]


def check_points(shift):
    return [
        {"target": [x, y], "left": [x + shift, y], "right": [x + shift, y], "both": [x + shift, y]}
        for x, y in [(0.2, 0.22), (0.8, 0.22), (0.5, 0.5), (0.2, 0.78), (0.8, 0.78)]
    ]


def test_a_calibration_that_cannot_follow_the_eyes_is_not_saved(world):
    client, _, _ = world
    signup(client)
    client.post("/onboarding", data={"weak_eye": "left", "goal_xp": "20"})
    result = client.post("/api/calibrate", json={"samples": mixed_up_samples(), "aspect": 1.6}).get_json()
    assert not result["ok"] and "didn't save" in result["error"]
    assert client.post("/api/frame", data=bright_jpeg()).get_json()["left"]["gaze"] is None
    assert "/calibrate" in client.get("/lesson/u1l1").headers["Location"]


def test_failing_the_accuracy_test_puts_back_the_last_good_calibration(world):
    client, clock, _ = world
    signup(client)
    client.post("/onboarding", data={"weak_eye": "left", "goal_xp": "20"})
    client.post("/api/calibrate", json={"samples": calibration_samples(), "aspect": 1.6})
    assert client.post("/api/calibrate/check", json={"points": check_points(0.05), "aspect": 1.6}).get_json()["ok"]
    good = client.post("/api/frame", data=bright_jpeg()).get_json()["left"]["gaze"]

    clock.value = date.fromordinal(clock.value.toordinal() + 1)
    client.post("/api/calibrate", json={"samples": calibration_samples(seed=2), "aspect": 1.6})
    # Every test dot lands far away, at random-looking spots no shift or stretch can fix.
    wild = [{**p, "both": [1 - p["target"][0], 1 - p["target"][1]]} for p in check_points(0)]
    wild = [{**p, "left": p["both"], "right": p["both"]} for p in wild]
    result = client.post("/api/calibrate/check", json={"points": wild, "aspect": 1.6}).get_json()
    assert result["rejected"] and not result["ok"]
    assert client.post("/api/frame", data=bright_jpeg()).get_json()["left"]["gaze"] == pytest.approx(good)
    # The kept calibration is from yesterday, so today's lessons still ask for a new one.
    assert "/calibrate" in client.get("/lesson/u1l1").headers["Location"]


def test_a_failed_calibration_saved_earlier_is_not_used(tmp_path):
    client = eyequest.create_app(tmp_path, FakeDetector(), Clock()).test_client()
    signup(client)
    client.post("/onboarding", data={"weak_eye": "left", "goal_xp": "20"})
    client.post("/api/calibrate", json={"samples": calibration_samples(), "aspect": 1.6})
    # Mark it as one that failed its accuracy test, which an older version still saved.
    connection = eyequest.db.connect(tmp_path / "eyequest.db")
    stored = json.loads(connection.execute("SELECT calibration FROM users").fetchone()["calibration"])
    connection.execute("UPDATE users SET calibration = ?", (json.dumps({**stored, "checked": True, "accuracy": 0.53}),))
    connection.commit()
    connection.close()
    # A fresh app, so nothing is remembered in memory from before the change.
    client = eyequest.create_app(tmp_path, FakeDetector(), Clock()).test_client()
    client.post("/login", data={"username": "bryan", "password": "correct horse"})
    assert client.post("/api/frame", data=bright_jpeg()).get_json()["left"]["gaze"] is None
    assert "/calibrate" in client.get("/lesson/u1l1").headers["Location"]


def test_older_calibrations_must_be_redone(tmp_path):
    clock = Clock()
    application = eyequest.create_app(tmp_path, FakeDetector(), clock)
    client = application.test_client()
    signup(client)
    client.post("/onboarding", data={"weak_eye": "left", "goal_xp": "20"})
    connection = eyequest.db.connect(tmp_path / "eyequest.db")
    connection.execute(
        "UPDATE users SET calibration = ?, calib_date = ?",
        ('{"models": {"left": {"mean": [0,0,0,0,0], "std": [1,1,1,1,1], "weights": []}}}', clock().isoformat()),
    )
    connection.commit()
    connection.close()
    assert "/calibrate" in client.get("/lesson/u1l1").headers["Location"]
    assert client.post("/api/frame", data=bright_jpeg()).get_json()["left"]["gaze"] is None


def test_lesson_flow_awards_xp_and_streak(world):
    client, clock, _ = world
    signup(client)
    client.post("/onboarding", data={"weak_eye": "left", "goal_xp": "10", "mouse_mode": "1"})
    assert finish(client, "u1l2").status_code == 403  # still locked
    first = finish(client, "u1l1").get_json()
    assert first["xp"] == 15 and first["stars"] == 3 and first["streak"] == 1
    assert first["goal_just_met"] and "First look" in first["new_badges"]
    assert finish(client, "u1l2", 50).get_json()["goal_just_met"] is False
    clock.value = date(2026, 3, 3)
    assert finish(client, "u1l3").get_json()["streak"] == 2
    assert client.get("/learn").status_code == 200
    assert client.get("/progress").status_code == 200


def test_replay_keeps_best_stars_and_rest_limit_stops_xp(world):
    client, _, _ = world
    signup(client)
    client.post("/onboarding", data={"weak_eye": "right", "goal_xp": "20", "mouse_mode": "1"})
    results = [finish(client, "u1l1").get_json() for _ in range(progress.DAILY_LESSON_LIMIT + 1)]
    assert results[progress.DAILY_LESSON_LIMIT - 1]["rest"] is True
    assert results[-1]["xp"] == 0
    assert finish(client, "u1l1", 10).get_json()["stars"] == 1


def test_bad_completion_payload_rejected(world):
    client, _, _ = world
    signup(client)
    assert client.post("/api/lesson/complete", json={"lesson_id": ["u1l1"], "steps": [1]}).status_code == 400
    assert client.post("/api/lesson/complete", json={"lesson_id": "nope"}).status_code == 400


def test_solo_step_covers_the_stronger_eye(world):
    client, _, _ = world
    signup(client)
    client.post("/onboarding", data={"weak_eye": "left", "goal_xp": "20", "mouse_mode": "1"})
    for lesson_id in curriculum.ORDER[:curriculum.ORDER.index("u2l1")]:
        finish(client, lesson_id)
    page = client.get("/lesson/u2l1").get_data(as_text=True)
    assert '"cover":"right"' in page.replace(" ", "")


def test_jellyfish_lesson_opens_and_saves_its_score(world):
    client, _, _ = world
    signup(client)
    client.post("/onboarding", data={"weak_eye": "left", "goal_xp": "20", "mouse_mode": "1"})
    for lesson_id in curriculum.ORDER[:curriculum.ORDER.index("u1l5")]:
        finish(client, lesson_id)
    page = client.get("/lesson/u1l5").get_data(as_text=True)
    assert "Jellyfish Rescue" in page
    step = {"game": "jellyfish", "eye": "both", "score": 80, "metrics": {"rescued": 10, "level": 3}}
    result = client.post("/api/lesson/complete", json={"lesson_id": "u1l5", "steps": [step]})
    assert result.status_code == 200
    assert result.get_json()["score"] == 80


# ---------- tracer and fine-tuning ----------

def test_tune_converges_on_the_clicked_spot():
    entries, target = [], [0.3, 0.2]
    point = [0.36, 0.42]
    guess = point
    for _ in range(6):
        entries, miss, used = gaze.add_tune(entries, guess, target, 1.6)
        assert used
        guess = gaze.tuned(entries, point)
    assert guess == pytest.approx(target, abs=0.01)


def test_tune_stays_local_to_where_it_clicked():
    entries, _, _ = gaze.add_tune([], [0.2, 0.2], [0.2, 0.05], 1.6)
    assert gaze.tuned(entries, [0.2, 0.2])[1] == pytest.approx(0.08, abs=0.01)
    assert gaze.tuned(entries, [0.9, 0.9]) == pytest.approx([0.9, 0.9], abs=0.001)


def test_tune_ignores_clicks_far_from_the_gaze():
    entries, miss, used = gaze.add_tune([], [0.1, 0.1], [0.9, 0.9], 1.6)
    assert not used and entries == [] and miss > gaze.TUNE_MAX_MISS


def test_tune_api_changes_live_gaze_and_can_be_cleared(world):
    client, _, _ = world
    signup(client)
    client.post("/onboarding", data={"weak_eye": "left", "goal_xp": "20"})
    click = {"target": [0.5, 0.4], "guess": [0.5, 0.5], "aspect": 1.6}
    assert client.post("/api/calibrate/tune", json=click).status_code == 409
    client.post("/api/calibrate", json={"samples": calibration_samples(), "aspect": 1.6})
    before = client.post("/api/frame", data=bright_jpeg()).get_json()["both"]["gaze"]
    assert client.post("/api/calibrate/tune", json={**click, "target": "x"}).status_code == 400
    result = client.post(
        "/api/calibrate/tune", json={**click, "guess": before, "target": [before[0], before[1] - 0.1]}
    ).get_json()
    assert result["used"] and result["count"] == 1 and result["typical"] == pytest.approx(0.1)
    after = client.post("/api/frame", data=bright_jpeg()).get_json()["both"]["gaze"]
    assert after[1] == pytest.approx(before[1] - 0.08, abs=0.005)
    client.post("/api/calibrate/tune/reset")
    cleared = client.post("/api/frame", data=bright_jpeg()).get_json()["both"]["gaze"]
    assert cleared == pytest.approx(before)


def test_tracer_settings_save_and_reject_bad_values(world):
    client, _, _ = world
    signup(client)
    client.post("/onboarding", data={"weak_eye": "left", "goal_xp": "20"})
    assert '"autostart":true' in client.get("/eyes").get_data(as_text=True).replace(" ", "")
    client.post("/settings", data={
        "weak_eye": "left", "goal_xp": "20", "tracer_shape": "cross", "tracer_color": "#4aa8ff",
        "tracer_size": "999", "tracer_opacity": "60", "tracer_trail": "long",
        "tracer_smoothing": "steady", "tracer_in_games": "1",
    })
    page = client.get("/eyes").get_data(as_text=True).replace(" ", "")
    for piece in ('"shape":"cross"', '"color":"#4aa8ff"', '"size":80', '"opacity":60',
                  '"trail":"long"', '"smoothing":"steady"', '"in_games":true', '"autostart":false'):
        assert piece in page
    client.post("/settings", data={"tracer_shape": "<script>", "tracer_color": "javascript:x"})
    page = client.get("/eyes").get_data(as_text=True).replace(" ", "")
    assert '"shape":"ring"' in page and '"color":"#ff6b5e"' in page


def lesson_body_class(client):
    page = client.get("/lesson/u1l1").get_data(as_text=True)
    return page.split('<body class="', 1)[1].split('"', 1)[0]


def test_games_use_the_background_calibration_was_made_on(world):
    client, _, _ = world
    signup(client)
    client.post("/onboarding", data={"weak_eye": "left", "goal_xp": "20"})
    client.post("/settings", data={"weak_eye": "left", "goal_xp": "20", "tracer_stage": "off"})
    client.post("/api/calibrate", json={"samples": calibration_samples(), "aspect": 1.6, "stage": "bright"})
    # Calibrated on the bright stage, so games stay bright even though the setting now says off.
    assert lesson_body_class(client) == "stage-page bright"
    client.post("/api/calibrate", json={"samples": calibration_samples(), "aspect": 1.6, "stage": "<b>"})
    assert lesson_body_class(client) == "stage-page"


def test_bright_stage_setting_is_saved_and_used_with_the_mouse(world):
    client, _, _ = world
    signup(client)
    client.post("/onboarding", data={"weak_eye": "left", "goal_xp": "20", "mouse_mode": "1"})
    assert '"stage":"auto"' in client.get("/settings").get_data(as_text=True).replace(" ", "")
    client.post("/settings", data={"weak_eye": "left", "goal_xp": "20", "mouse_mode": "1", "tracer_stage": "on"})
    assert lesson_body_class(client) == "stage-page bright"
    client.post("/settings", data={"weak_eye": "left", "goal_xp": "20", "mouse_mode": "1", "tracer_stage": "sideways"})
    assert '"stage":"auto"' in client.get("/settings").get_data(as_text=True).replace(" ", "")
    assert lesson_body_class(client) == "stage-page"


def test_existing_databases_gain_the_tracer_column(tmp_path):
    import sqlite3
    old = sqlite3.connect(tmp_path / "eyequest.db")
    old.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT, pw_hash TEXT)")
    old.commit()
    old.close()
    eyequest.create_app(tmp_path, FakeDetector(), Clock())
    check = sqlite3.connect(tmp_path / "eyequest.db")
    assert "tracer" in [row[1] for row in check.execute("PRAGMA table_info(users)")]


# ---------- head position guide ----------

def test_head_guide_is_quiet_near_the_calibrated_position():
    head = {"reference": gaze.head_pose(fake_matrix())}
    assert gaze.head_guide(head, gaze.head_pose(fake_matrix(yaw=2, pitch=1.5))) is None
    assert gaze.head_guide(None, gaze.head_pose(fake_matrix(pitch=20))) is None


@pytest.mark.parametrize("change, advice", [
    ({"pitch": 8}, "raise_chin"),
    ({"pitch": -8}, "lower_chin"),
    ({"yaw": 12}, "turn_right"),
    ({"yaw": -12}, "turn_left"),
])
def test_head_guide_says_which_way_to_turn_back(change, advice):
    head = {"reference": gaze.head_pose(fake_matrix())}
    assert gaze.head_guide(head, gaze.head_pose(fake_matrix(**change))) == advice


def test_head_guide_handles_moving_closer_further_and_sideways():
    reference = gaze.head_pose(fake_matrix())
    head = {"reference": reference}
    moved = lambda index, amount: [v + amount if i == index else v for i, v in enumerate(reference)]
    assert gaze.head_guide(head, moved(gaze.POSE_DISTANCE, reference[gaze.POSE_DISTANCE] * 0.3)) == "move_closer"
    assert gaze.head_guide(head, moved(gaze.POSE_DISTANCE, -reference[gaze.POSE_DISTANCE] * 0.3)) == "move_back"
    assert gaze.head_guide(head, moved(gaze.POSE_X, 6)) == "move_right"
    assert gaze.head_guide(head, moved(gaze.POSE_Y, -6)) == "sit_up"
    # The biggest drift, compared with its own limit, decides the advice.
    both = moved(gaze.POSE_X, 3.5)
    both[gaze.POSE_PITCH] += 12
    assert gaze.head_guide(head, both) == "raise_chin"


# ---------- error pages and icon ----------

def test_missing_pages_get_a_friendly_page_and_apis_get_json(world):
    client, _, _ = world
    page = client.get("/no-such-page")
    assert page.status_code == 404 and "Page not found" in page.get_data(as_text=True)
    api = client.post("/api/no-such-thing")
    assert api.status_code == 404 and "error" in api.get_json()


def test_pages_have_a_tab_icon(world):
    client, _, _ = world
    assert "favicon.svg" in client.get("/login").get_data(as_text=True)
    assert client.get("/static/favicon.svg").status_code == 200


# ---------- research page ----------

def test_research_page_is_open_to_everyone_and_honest(world):
    client, _, _ = world
    page = client.get("/research").get_data(as_text=True)
    assert "hasn't been tested in any study" in page
    assert page.count("https://doi.org/") == len(research.STUDIES)
    assert "research" in client.get("/").get_data(as_text=True)


def test_research_page_matches_the_evidence_document():
    evidence = (Path(__file__).parent.parent / "EVIDENCE.md").read_text()
    for study in research.STUDIES:
        assert study["doi"].lower() in evidence.lower()
