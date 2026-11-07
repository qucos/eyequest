"""EyeQuest: daily eye practice with webcam tracking, streaks and a lesson path."""
import json
import logging
import os
import re
import secrets
from datetime import date, timedelta
from functools import wraps
from pathlib import Path

from flask import (
    Flask, abort, flash, g, jsonify, redirect, render_template, request, session, url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash

import curriculum
import db
import gaze
import progress
import tracer_settings
from tracker import Tracker

USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,24}$")
MIN_PASSWORD = 8
ALLOWED_HOSTS = {"localhost", "127.0.0.1"}
MAX_SAMPLES = 2000
MIN_CHECK_POINTS = 3
TYPICAL_CLICKS = 12
# A typical miss above this share of the screen's height is too rough to play with. Good
# calibrations on real data scored 0.09-0.16; a failed one scored 0.44-0.53.
MAX_ACCEPTED_MISS = 0.30
POOR_CALIBRATION = (
    "The tracker couldn't follow your eyes well enough, so I didn't save this calibration. "
    "Sit about an arm's length from the screen with light on your face, close other heavy apps, "
    "keep your head still, and look right at the center of each dot."
)


def _secret_key(data_dir):
    path = data_dir / "secret.key"
    if not path.exists():
        path.touch(mode=0o600)
        path.write_text(secrets.token_hex(32))
    return path.read_text()


def _clean_steps(raw_steps):
    """Keep only well-formed numbers from the browser's lesson results."""
    steps = []
    for raw in raw_steps if isinstance(raw_steps, list) else []:
        is_valid = (
            isinstance(raw, dict)
            and raw.get("game") in curriculum.GAMES
            and isinstance(raw.get("score"), (int, float))
        )
        if is_valid:
            metrics = raw.get("metrics") if isinstance(raw.get("metrics"), dict) else {}
            steps.append({
                "game": raw["game"],
                "eye": raw.get("eye") if raw.get("eye") in ("left", "right", "both") else "both",
                "score": max(0.0, min(100.0, float(raw["score"]))),
                "metrics": {
                    str(k)[:32]: round(float(v), 3)
                    for k, v in metrics.items()
                    if isinstance(v, (int, float))
                },
            })
    return steps


def _is_valid_sample(sample):
    if not isinstance(sample, dict):
        return False
    target = sample.get("target")
    if not (isinstance(target, list) and len(target) == 2):
        return False
    if sample.get("phase") not in (None, "left", "right", "up", "down"):
        return False
    values = list(target)
    for eye in ("left", "right"):
        features = sample.get(eye)
        if features is not None:
            if not (isinstance(features, list) and len(features) == gaze.FEATURE_COUNT):
                return False
            values.extend(features)
    return all(isinstance(v, (int, float)) for v in values)


def _is_point(value):
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(v, (int, float)) and -1 <= v <= 2 for v in value)
    )


def _is_valid_check_point(point):
    if not isinstance(point, dict):
        return False
    pairs = [point.get("target"), point.get("left"), point.get("right"), point.get("both")]
    if pairs[0] is None:
        return False
    return all(
        pair is None
        or (
            isinstance(pair, list)
            and len(pair) == 2
            and all(isinstance(v, (int, float)) for v in pair)
        )
        for pair in pairs
    )


def create_app(data_dir=None, detector=None, today=date.today):
    data_dir = Path(data_dir or Path(__file__).parent / "data")
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "eyequest.db"
    db.init(db_path)

    app = Flask(__name__)
    app.secret_key = _secret_key(data_dir)
    app.config.update(
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_HTTPONLY=True,
        PERMANENT_SESSION_LIFETIME=timedelta(days=60),
        MAX_CONTENT_LENGTH=4 * 1024 * 1024,
    )
    tracker = Tracker(detector)
    model_cache = {}

    def conn():
        if "db" not in g:
            g.db = db.connect(db_path)
        return g.db

    def saved_calibration(user):
        """The user's calibration, ignoring any made before the tracker last changed.

        Also ignores one that failed its accuracy test, such as one saved before failures
        were turned away.
        """
        saved = json.loads(user["calibration"]) if user["calibration"] else {}
        too_rough = saved.get("checked") and saved.get("accuracy", 0) > MAX_ACCEPTED_MISS
        return saved if saved.get("version") == gaze.VERSION and not too_rough else {}

    def live_calibration(user):
        """The calibration used for every frame, kept in memory so frames stay fast."""
        if user["id"] not in model_cache:
            model_cache[user["id"]] = saved_calibration(user)
        return model_cache[user["id"]]

    def lesson_stage(user, saved):
        """Light or dark background for games: the one calibration used, if there is one."""
        if saved.get("stage") in ("bright", "dark") and not user["mouse_mode"]:
            return saved["stage"]
        return "bright" if tracer_settings.load(user["tracer"])["stage"] == "on" else "dark"

    def store_calibration(user, saved):
        conn().execute(
            "UPDATE users SET calibration = ? WHERE id = ?", (json.dumps(saved), user["id"])
        )
        conn().commit()
        model_cache[user["id"]] = saved

    @app.teardown_appcontext
    def close_db(_error):
        connection = g.pop("db", None)
        if connection is not None:
            connection.close()

    @app.before_request
    def guard_and_load_user():
        # Blocks other websites from reaching this app through DNS tricks.
        if request.host.split(":")[0] not in ALLOWED_HOSTS:
            abort(403)
        # Opening a page directly loads the full-screen frame, which loads that page inside it.
        opened_directly = request.method == "GET" and request.headers.get("Sec-Fetch-Dest") == "document"
        if opened_directly and request.endpoint != "static" and not request.path.startswith("/api/"):
            # Stripping leading slashes keeps the frame from loading another website.
            return render_template("shell.html", page="/" + request.full_path.lstrip("/\\").rstrip("?"))
        g.user = None
        if "uid" in session:
            g.user = conn().execute(
                "SELECT * FROM users WHERE id = ?", (session["uid"],)
            ).fetchone()

    def login_required(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            if g.user is None:
                if request.path.startswith("/api/"):
                    return jsonify(error="Please log in again."), 401
                return redirect(url_for("login"))
            return view(*args, **kwargs)
        return wrapped

    def completed_stars(user_id):
        rows = conn().execute(
            "SELECT lesson_id, stars FROM lesson_progress WHERE user_id = ?", (user_id,)
        ).fetchall()
        return {row["lesson_id"]: row["stars"] for row in rows}

    def today_stats(user):
        row = conn().execute(
            "SELECT COUNT(*) AS lessons, COALESCE(SUM(xp), 0) AS xp FROM sessions "
            "WHERE user_id = ? AND day = ?",
            (user["id"], today().isoformat()),
        ).fetchone()
        return {"lessons": row["lessons"], "xp": row["xp"]}

    @app.context_processor
    def inject_header():
        user = g.get("user")
        if user is None:
            return {}
        return {
            "streak": progress.current_streak(
                user["streak"], user["last_active"], user["freezes"], today()
            ),
            "level": progress.level_for(user["xp"]),
            "tracer": tracer_settings.load(user["tracer"]),
        }

    def start_session(user_id):
        session.clear()
        session["uid"] = user_id
        session.permanent = True

    # ---------- accounts ----------

    @app.route("/")
    def home():
        if g.user is None:
            return render_template("welcome.html")
        return redirect(url_for("learn" if g.user["onboarded"] else "onboarding"))

    @app.route("/signup", methods=["GET", "POST"])
    def signup():
        if request.method == "POST":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")
            problem = None
            if not USERNAME_RE.match(username):
                problem = "Usernames are 3-24 letters, numbers, dots, dashes or underscores."
            elif len(password) < MIN_PASSWORD:
                problem = f"Passwords need at least {MIN_PASSWORD} characters."
            elif conn().execute(
                "SELECT 1 FROM users WHERE username = ?", (username,)
            ).fetchone():
                problem = "That username is taken."
            if problem:
                flash(problem)
                return render_template("auth.html", mode="signup", username=username), 400
            cursor = conn().execute(
                "INSERT INTO users (username, pw_hash) VALUES (?, ?)",
                (username, generate_password_hash(password)),
            )
            conn().commit()
            start_session(cursor.lastrowid)
            return redirect(url_for("onboarding"))
        return render_template("auth.html", mode="signup", username="")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            username = request.form.get("username", "").strip()
            user = conn().execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            ).fetchone()
            if user and check_password_hash(user["pw_hash"], request.form.get("password", "")):
                start_session(user["id"])
                return redirect(url_for("home"))
            flash("That username and password don't match.")
            return render_template("auth.html", mode="login", username=username), 401
        return render_template("auth.html", mode="login", username="")

    @app.post("/logout")
    def logout():
        session.clear()
        return redirect(url_for("home"))

    def save_preferences():
        weak_eye = request.form.get("weak_eye")
        goal = request.form.get("goal_xp", type=int)
        conn().execute(
            "UPDATE users SET weak_eye = ?, goal_xp = ?, mouse_mode = ?, onboarded = 1 WHERE id = ?",
            (
                weak_eye if weak_eye in ("left", "right") else "unknown",
                goal if goal in progress.GOAL_OPTIONS else 20,
                1 if request.form.get("mouse_mode") else 0,
                g.user["id"],
            ),
        )
        conn().commit()

    @app.route("/onboarding", methods=["GET", "POST"])
    @login_required
    def onboarding():
        if request.method == "POST":
            save_preferences()
            return redirect(url_for("learn"))
        return render_template("onboarding.html", goals=progress.GOAL_OPTIONS)

    @app.route("/settings", methods=["GET", "POST"])
    @login_required
    def settings():
        if request.method == "POST":
            save_preferences()
            conn().execute(
                "UPDATE users SET tracer = ? WHERE id = ?",
                (json.dumps(tracer_settings.from_form(request.form)), g.user["id"]),
            )
            conn().commit()
            flash("Saved.")
            return redirect(url_for("settings"))
        return render_template(
            "settings.html",
            goals=progress.GOAL_OPTIONS,
            tracer_options={
                "shapes": tracer_settings.SHAPES,
                "colors": tracer_settings.COLORS,
                "trails": tracer_settings.TRAILS,
                "smoothing": tracer_settings.SMOOTHING,
                "stages": tracer_settings.STAGES,
                "size": tracer_settings.SIZE_RANGE,
                "opacity": tracer_settings.OPACITY_RANGE,
            },
        )

    # ---------- main screens ----------

    @app.route("/learn")
    @login_required
    def learn():
        if not g.user["onboarded"]:
            return redirect(url_for("onboarding"))
        done = completed_stars(g.user["id"])
        current = curriculum.next_lesson_id(done)
        units = []
        for unit in curriculum.UNITS:
            nodes = []
            for index, (title, _steps) in enumerate(unit["lessons"], start=1):
                lesson_id = f"{unit['id']}l{index}"
                nodes.append({
                    "id": lesson_id,
                    "title": title,
                    "stars": done.get(lesson_id, 0),
                    "state": "done" if lesson_id in done
                    else "current" if lesson_id == current else "locked",
                })
            units.append({**unit, "nodes": nodes})
        stats = today_stats(g.user)
        return render_template(
            "learn.html",
            units=units,
            current=current,
            today=stats,
            goal=g.user["goal_xp"],
            path_done=current == "mix",
            rest=stats["lessons"] >= progress.DAILY_LESSON_LIMIT,
        )

    @app.route("/eyes")
    @login_required
    def eyes():
        return render_template("eyes.html", calibrated=bool(saved_calibration(g.user)))

    @app.route("/calibrate")
    @login_required
    def calibrate():
        lesson_id = request.args.get("lesson", "")
        return render_template(
            "calibrate.html", lesson_id=lesson_id if lesson_id in curriculum.BY_ID else ""
        )

    @app.route("/lesson/<lesson_id>")
    @login_required
    def lesson(lesson_id):
        chosen = curriculum.BY_ID.get(lesson_id)
        if chosen is None:
            abort(404)
        done = completed_stars(g.user["id"])
        if not curriculum.is_unlocked(lesson_id, done):
            flash("Finish the earlier lessons first.")
            return redirect(url_for("learn"))
        needs_camera = any(curriculum.GAMES[s["game"]]["gaze"] for s in chosen["steps"])
        calibrated_today = (
            g.user["calib_date"] == today().isoformat() and bool(saved_calibration(g.user))
        )
        if needs_camera and not g.user["mouse_mode"] and not calibrated_today:
            return redirect(url_for("calibrate", lesson=lesson_id))

        working = curriculum.solo_eye(g.user["weak_eye"], today().toordinal())
        steps = []
        for raw in chosen["steps"]:
            solo = raw["eye"] == "solo"
            steps.append({
                **raw,
                **curriculum.GAMES[raw["game"]],
                "eye": working if solo else "both",
                "cover": curriculum.other_eye(working) if solo else None,
            })
        saved = saved_calibration(g.user)
        config = {
            "id": lesson_id,
            "title": chosen["title"],
            "steps": steps,
            "mouse": bool(g.user["mouse_mode"]),
            "radius": saved.get("radius", 0.14),
            "stage": lesson_stage(g.user, saved),
        }
        return render_template("lesson.html", config=config)

    @app.route("/progress")
    @login_required
    def progress_page():
        rows = conn().execute(
            "SELECT day, input, steps FROM sessions WHERE user_id = ? ORDER BY id", (g.user["id"],)
        ).fetchall()
        series = {"follow_left": {}, "follow_right": {}, "gap": {}}
        for row in rows:
            # Mouse practice isn't eye data, so it stays off the charts.
            for step in json.loads(row["steps"]) if row["input"] == "camera" else []:
                metrics = step["metrics"]
                if step["game"] == "firefly" and step["eye"] == "both":
                    for eye in ("left", "right"):
                        if f"{eye}_hit" in metrics:
                            series[f"follow_{eye}"].setdefault(row["day"], []).append(metrics[f"{eye}_hit"])
                if step["game"] == "gap" and "smallest" in metrics and step["eye"] != "both":
                    series["gap"].setdefault(row["day"], []).append(metrics["smallest"])

        def by_day(values, pick):
            return [{"day": day, "value": round(pick(v), 1)} for day, v in sorted(values.items())]

        charts = {
            "left": by_day(series["follow_left"], lambda v: sum(v) / len(v)),
            "right": by_day(series["follow_right"], lambda v: sum(v) / len(v)),
            "gap": by_day(series["gap"], min),
        }
        day_rows = conn().execute(
            "SELECT day, SUM(xp) AS xp FROM sessions WHERE user_id = ? GROUP BY day", (g.user["id"],)
        ).fetchall()
        active = {row["day"]: row["xp"] for row in day_rows}
        start = today() - timedelta(days=today().weekday() + 7 * 11)
        calendar = [
            {
                "day": (start + timedelta(days=i)).isoformat(),
                "active": (start + timedelta(days=i)).isoformat() in active,
                "future": start + timedelta(days=i) > today(),
            }
            for i in range(84)
        ]
        have = {
            row["badge"]
            for row in conn().execute(
                "SELECT badge FROM badges WHERE user_id = ?", (g.user["id"],)
            ).fetchall()
        }
        lesson_total = conn().execute(
            "SELECT COUNT(*) AS n FROM sessions WHERE user_id = ?", (g.user["id"],)
        ).fetchone()["n"]
        return render_template(
            "progress.html",
            charts=charts,
            calendar=calendar,
            badges=progress.BADGES,
            have=have,
            lesson_total=lesson_total,
            days_active=len(active),
        )

    # ---------- API ----------

    @app.post("/api/frame")
    @login_required
    def api_frame():
        state = tracker.process(request.get_data(), live_calibration(g.user))
        if state is None:
            return jsonify(error="Couldn't read that camera frame."), 400
        return jsonify(state)

    @app.post("/api/calibrate")
    @login_required
    def api_calibrate():
        body = request.get_json(silent=True) or {}
        samples = body.get("samples")
        aspect = body.get("aspect")
        if (
            not isinstance(samples, list)
            or len(samples) > MAX_SAMPLES
            or not isinstance(aspect, (int, float))
            or not 0.5 <= aspect <= 4
            or not all(_is_valid_sample(s) for s in samples)
        ):
            return jsonify(error="Calibration data wasn't valid."), 400
        fitted = gaze.fit_calibration(samples, float(aspect))
        if fitted is None:
            return jsonify(ok=False, error="Not enough clear looks at the dots. Let's try again."), 200
        if fitted["error"] > MAX_ACCEPTED_MISS:
            return jsonify(ok=False, error=POOR_CALIBRATION), 200
        # Kept so a calibration that fails its accuracy test can be undone.
        older = json.loads(g.user["calibration"]) if g.user["calibration"] else None
        if older:
            older.pop("previous", None)
        fitted["previous"] = {"calibration": older, "date": g.user["calib_date"]}
        # Eyes look different under a bright screen, so games use the background this was made on.
        fitted["stage"] = "bright" if body.get("stage") == "bright" else "dark"
        conn().execute(
            "UPDATE users SET calibration = ?, calib_date = ? WHERE id = ?",
            (json.dumps(fitted), today().isoformat(), g.user["id"]),
        )
        conn().commit()
        model_cache[g.user["id"]] = fitted
        return jsonify(
            ok=True,
            error_size=fitted["error"],
            radius=fitted["radius"],
            eyes=sorted(fitted["models"]),
        )

    @app.post("/api/calibrate/check")
    @login_required
    def api_calibrate_check():
        """Scores a fresh calibration on new dots and removes any steady shift."""
        body = request.get_json(silent=True) or {}
        points = body.get("points")
        aspect = body.get("aspect")
        saved = saved_calibration(g.user)
        if (
            not isinstance(points, list)
            or len(points) > 50
            or not isinstance(aspect, (int, float))
            or not 0.5 <= aspect <= 4
            or not all(_is_valid_check_point(p) for p in points)
        ):
            return jsonify(error="Accuracy test data wasn't valid."), 400
        if not saved or saved.get("checked"):
            return jsonify(error="Please calibrate first."), 409

        results = {}
        for eye, model in saved["models"].items():
            seen = [p for p in points if p.get(eye)]
            if len(seen) >= MIN_CHECK_POINTS:
                results[eye] = gaze.fit_correction(
                    [p[eye] for p in seen], [p["target"] for p in seen], float(aspect)
                )
                model["correction"] = results[eye]["correction"]
                model["error"] = results[eye]["after"]
        if not results:
            return jsonify(ok=False)

        worst = max(result["after"] for result in results.values())
        accuracy = results["both"]["after"] if "both" in results else worst
        if accuracy > MAX_ACCEPTED_MISS:
            # Too rough to play with, so put back whatever was there before.
            previous = saved.get("previous") or {}
            conn().execute(
                "UPDATE users SET calibration = ?, calib_date = ? WHERE id = ?",
                (
                    json.dumps(previous["calibration"]) if previous.get("calibration") else None,
                    previous.get("date"),
                    g.user["id"],
                ),
            )
            conn().commit()
            model_cache.pop(g.user["id"], None)
            return jsonify(ok=False, rejected=True, accuracy=accuracy, error=POOR_CALIBRATION)
        saved["checked"] = True
        saved["accuracy"] = accuracy
        saved["accuracy_by_eye"] = results
        saved["radius"] = float(min(0.24, max(0.08, worst * 1.25)))
        store_calibration(g.user, saved)
        return jsonify(ok=True, accuracy=accuracy, radius=saved["radius"], eyes=sorted(results))

    @app.post("/api/calibrate/tune")
    @login_required
    def api_tune():
        """Learns from one click: the person looked at a spot and clicked it."""
        body = request.get_json(silent=True) or {}
        target, guess, aspect = body.get("target"), body.get("guess"), body.get("aspect")
        saved = saved_calibration(g.user)
        if (
            not _is_point(target)
            or not _is_point(guess)
            or not isinstance(aspect, (int, float))
            or not 0.5 <= aspect <= 4
        ):
            return jsonify(error="That click wasn't valid."), 400
        if not saved:
            return jsonify(error="Please calibrate first."), 409
        entries, miss, used = gaze.add_tune(saved.get("tuning", []), guess, target, float(aspect))
        saved["tuning"] = entries
        if used:
            saved["click_misses"] = (saved.get("click_misses", []) + [miss])[-TYPICAL_CLICKS:]
        store_calibration(g.user, saved)
        recent = sorted(saved.get("click_misses", []))
        return jsonify(
            used=used,
            miss=miss,
            count=len(recent),
            typical=recent[len(recent) // 2] if recent else None,
        )

    @app.post("/api/calibrate/tune/reset")
    @login_required
    def api_tune_reset():
        saved = saved_calibration(g.user)
        saved.pop("tuning", None)
        saved.pop("click_misses", None)
        if saved:
            store_calibration(g.user, saved)
        return jsonify(ok=True)

    @app.post("/api/lesson/complete")
    @login_required
    def api_complete():
        body = request.get_json(silent=True) or {}
        lesson_id = body.get("lesson_id")
        steps = _clean_steps(body.get("steps"))
        chosen = curriculum.BY_ID.get(lesson_id) if isinstance(lesson_id, str) else None
        if chosen is None or not steps:
            return jsonify(error="That lesson result wasn't valid."), 400
        user = g.user
        before = completed_stars(user["id"])
        if not curriculum.is_unlocked(lesson_id, before):
            return jsonify(error="That lesson is still locked."), 403

        now = today()
        score = sum(s["score"] for s in steps) / len(steps)
        stars = progress.stars_for(score)
        done_today = today_stats(user)["lessons"]
        xp = progress.xp_for(stars, done_today)
        streak, freezes, outcome = progress.update_streak(
            user["streak"], user["last_active"], user["freezes"], now
        )
        total_xp = user["xp"] + xp
        best_streak = max(user["best_streak"], streak)
        previous_stars = before.get(lesson_id, 0)

        connection = conn()
        connection.execute(
            "INSERT INTO sessions (user_id, day, lesson_id, score, stars, xp, input, steps) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (user["id"], now.isoformat(), lesson_id, score, stars, xp,
             "mouse" if user["mouse_mode"] else "camera", json.dumps(steps)),
        )
        connection.execute(
            "INSERT INTO lesson_progress (user_id, lesson_id, stars) VALUES (?, ?, ?) "
            "ON CONFLICT(user_id, lesson_id) DO UPDATE SET stars = MAX(stars, excluded.stars)",
            (user["id"], lesson_id, stars),
        )
        connection.execute(
            "UPDATE users SET xp = ?, streak = ?, best_streak = ?, freezes = ?, last_active = ? WHERE id = ?",
            (total_xp, streak, best_streak, freezes, now.isoformat(), user["id"]),
        )
        lesson_total = connection.execute(
            "SELECT COUNT(*) AS n FROM sessions WHERE user_id = ?", (user["id"],)
        ).fetchone()["n"]
        have = {
            row["badge"]
            for row in connection.execute(
                "SELECT badge FROM badges WHERE user_id = ?", (user["id"],)
            ).fetchall()
        }
        new_badges = sorted(
            progress.earned_badges(best_streak, total_xp, lesson_total, max(stars, previous_stars)) - have
        )
        connection.executemany(
            "INSERT INTO badges (user_id, badge, day) VALUES (?, ?, ?)",
            [(user["id"], badge, now.isoformat()) for badge in new_badges],
        )
        connection.commit()

        xp_today = today_stats(user)["xp"]
        return jsonify(
            score=round(score),
            stars=stars,
            xp=xp,
            streak=streak,
            streak_outcome=outcome,
            freezes=freezes,
            xp_today=xp_today,
            goal=user["goal_xp"],
            goal_just_met=xp_today >= user["goal_xp"] > xp_today - xp,
            rest=done_today + 1 >= progress.DAILY_LESSON_LIMIT,
            new_badges=[progress.BADGES[badge][0] for badge in new_badges],
        )

    return app


if __name__ == "__main__":
    logging.getLogger("werkzeug").setLevel(logging.WARNING)  # frames would flood the log
    port = int(os.environ.get("EYEQUEST_PORT", "5050"))
    print(f"\n  EyeQuest is running at http://localhost:{port}\n  Press Ctrl+C to stop.\n")
    create_app().run(host="127.0.0.1", port=port, threaded=True)
