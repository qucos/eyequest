"""The lesson path. Each lesson takes two to four minutes.

eye="both" keeps both eyes open. eye="solo" means the stronger eye is covered, so
the lazy eye does the work.
"""
GAMES = {
    "firefly": {
        "name": "Firefly Follow",
        "how": "Follow the firefly with your eyes. Keep your head still.",
        "gaze": True,
    },
    "stars": {
        "name": "Star Hop",
        "how": "Look at each star to collect it. Only your eyes move.",
        "gaze": True,
    },
    "steady": {
        "name": "Steady Glow",
        "how": "Keep looking at the glowing dot. Blinking is fine.",
        "gaze": True,
    },
    "gap": {
        "name": "Gap Spotter",
        "how": "Which way does the gap in the ring point? Press an arrow key. It gets smaller as you improve.",
        "gaze": False,
    },
}


def step(game, eye, **params):
    return {"game": game, "eye": eye, "params": params}


def firefly(eye, seconds=30, speed=0.6):
    return step("firefly", eye, seconds=seconds, speed=speed)


def stars(eye, seconds=30, dwell=0.4):
    return step("stars", eye, seconds=seconds, dwell=dwell)


def steady(eye, seconds=20):
    return step("steady", eye, seconds=seconds)


def gap(eye, trials=16, contrast=1.0, start=70):
    return step("gap", eye, trials=trials, contrast=contrast, start=start)


UNITS = [
    {
        "id": "u1",
        "title": "Warm up",
        "blurb": "Learn the games with both eyes open.",
        "lessons": [
            ("Meet the firefly", [firefly("both", 25, 0.45), steady("both", 15)]),
            ("Star hopping", [stars("both", 25), firefly("both", 25, 0.5)]),
            ("Hold it steady", [steady("both", 20), stars("both", 25)]),
            ("Spot the gap", [gap("both", 12), firefly("both", 25, 0.55)]),
        ],
    },
    {
        "id": "u2",
        "title": "Solo eye",
        "blurb": "Cover your stronger eye and let the lazy eye lead.",
        "lessons": [
            ("First solo", [firefly("solo", 20, 0.4), gap("solo", 12)]),
            ("Solo stars", [stars("solo", 25, 0.45), steady("solo", 15)]),
            ("Solo sharpness", [gap("solo", 16), firefly("solo", 25, 0.45)]),
            ("Solo mix", [firefly("solo", 25, 0.5), stars("solo", 25), gap("solo", 12)]),
        ],
    },
    {
        "id": "u3",
        "title": "Sharpen up",
        "blurb": "Smaller, fainter gaps and quicker targets.",
        "lessons": [
            ("Fine detail", [gap("solo", 20, 1.0, 50), steady("solo", 20)]),
            ("Fainter gaps", [gap("solo", 20, 0.6, 60), firefly("solo", 25, 0.6)]),
            ("Quick hops", [stars("solo", 30, 0.3), gap("solo", 14, 0.6, 50)]),
            ("Low light", [gap("solo", 20, 0.35, 70), stars("solo", 30, 0.3)]),
        ],
    },
    {
        "id": "u4",
        "title": "Team up",
        "blurb": "Bring both eyes back together.",
        "lessons": [
            ("Back together", [firefly("both", 30, 0.7), steady("both", 25)]),
            ("Long glow", [steady("solo", 30), steady("both", 30)]),
            ("Fast firefly", [firefly("solo", 25, 0.75), firefly("both", 30, 0.85)]),
            ("Full workout", [firefly("solo", 25, 0.7), gap("solo", 16, 0.5, 50), stars("both", 30, 0.3)]),
        ],
    },
]

DAILY_MIX = {
    "id": "mix",
    "title": "Daily mix",
    "unit": "Daily mix",
    "steps": [firefly("solo", 30, 0.7), gap("solo", 18, 0.6, 50), stars("solo", 30, 0.3), steady("both", 20)],
}


def _flatten():
    lessons = []
    for unit in UNITS:
        for index, (title, steps) in enumerate(unit["lessons"], start=1):
            lessons.append(
                {"id": f"{unit['id']}l{index}", "title": title, "unit": unit["title"], "steps": steps}
            )
    return lessons


LESSONS = _flatten()
ORDER = [lesson["id"] for lesson in LESSONS]
BY_ID = {lesson["id"]: lesson for lesson in LESSONS}
BY_ID["mix"] = DAILY_MIX


def next_lesson_id(completed):
    """The first unfinished lesson, or the daily mix once the path is done."""
    remaining = [lesson_id for lesson_id in ORDER if lesson_id not in completed]
    return remaining[0] if remaining else "mix"


def is_unlocked(lesson_id, completed):
    if lesson_id == "mix":
        return all(known in completed for known in ORDER)
    position = ORDER.index(lesson_id)
    return position == 0 or ORDER[position - 1] in completed


def solo_eye(weak_eye, day_ordinal):
    """Which eye does the work. If unsure, the eyes take turns day by day."""
    if weak_eye in ("left", "right"):
        return weak_eye
    return "left" if day_ordinal % 2 == 0 else "right"


def other_eye(eye):
    return "right" if eye == "left" else "left"
