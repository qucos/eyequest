"""The game rules: streaks, freezes, XP, stars, levels and badges."""
from datetime import date

MAX_FREEZES = 2
FREEZE_EVERY = 7
DAILY_LESSON_LIMIT = 4  # past this we tell people to rest, and stop awarding XP
GOAL_OPTIONS = {10: "Light", 20: "Regular", 30: "Focused"}
XP_PER_LEVEL = 100

BADGES = {
    "first_lesson": ("First look", "Finish your first lesson"),
    "streak_3": ("Three in a row", "Reach a 3-day streak"),
    "streak_7": ("Full week", "Reach a 7-day streak"),
    "streak_30": ("Month of focus", "Reach a 30-day streak"),
    "xp_100": ("Warmed up", "Earn 100 XP"),
    "xp_500": ("Going strong", "Earn 500 XP"),
    "three_stars": ("Bullseye", "Get 3 stars on a lesson"),
    "ten_lessons": ("Regular", "Finish 10 lessons"),
}


def update_streak(streak, last_active, freezes, today):
    """Apply one day's activity. Returns (streak, freezes, what happened).

    The outcome is "same_day", "started", "extended", "frozen" or "reset".
    """
    if last_active is None:
        streak, outcome = 1, "started"
    else:
        gap = (today - date.fromisoformat(last_active)).days
        missed = gap - 1
        if gap <= 0:
            return streak, freezes, "same_day"
        if missed == 0:
            streak, outcome = streak + 1, "extended"
        elif missed <= freezes:
            freezes -= missed
            streak, outcome = streak + 1, "frozen"
        else:
            streak, outcome = 1, "reset"
    if streak % FREEZE_EVERY == 0:
        freezes = min(MAX_FREEZES, freezes + 1)
    return streak, freezes, outcome


def current_streak(streak, last_active, freezes, today):
    """The streak to display, which is 0 once it can no longer be saved."""
    if last_active is None:
        return 0
    missed = (today - date.fromisoformat(last_active)).days - 1
    return streak if missed <= freezes else 0


def stars_for(score):
    if score >= 85:
        return 3
    if score >= 60:
        return 2
    return 1


def xp_for(stars, lessons_done_today):
    if lessons_done_today >= DAILY_LESSON_LIMIT:
        return 0
    return {1: 10, 2: 12, 3: 15}[stars]


def level_for(xp):
    return xp // XP_PER_LEVEL + 1


def earned_badges(streak, xp, lessons, best_stars):
    checks = {
        "first_lesson": lessons >= 1,
        "streak_3": streak >= 3,
        "streak_7": streak >= 7,
        "streak_30": streak >= 30,
        "xp_100": xp >= 100,
        "xp_500": xp >= 500,
        "three_stars": best_stars >= 3,
        "ten_lessons": lessons >= 10,
    }
    return {badge for badge, met in checks.items() if met}
