"""SQLite storage. Everything stays in data/eyequest.db on this computer."""
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
    pw_hash TEXT NOT NULL,
    onboarded INTEGER NOT NULL DEFAULT 0,
    weak_eye TEXT NOT NULL DEFAULT 'unknown',
    goal_xp INTEGER NOT NULL DEFAULT 20,
    mouse_mode INTEGER NOT NULL DEFAULT 0,
    xp INTEGER NOT NULL DEFAULT 0,
    streak INTEGER NOT NULL DEFAULT 0,
    best_streak INTEGER NOT NULL DEFAULT 0,
    freezes INTEGER NOT NULL DEFAULT 0,
    last_active TEXT,
    calibration TEXT,
    calib_date TEXT,
    tracer TEXT
);
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    day TEXT NOT NULL,
    lesson_id TEXT NOT NULL,
    score REAL NOT NULL,
    stars INTEGER NOT NULL,
    xp INTEGER NOT NULL,
    input TEXT NOT NULL,
    steps TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS sessions_user_day ON sessions(user_id, day);
CREATE TABLE IF NOT EXISTS lesson_progress (
    user_id INTEGER NOT NULL REFERENCES users(id),
    lesson_id TEXT NOT NULL,
    stars INTEGER NOT NULL,
    PRIMARY KEY (user_id, lesson_id)
);
CREATE TABLE IF NOT EXISTS badges (
    user_id INTEGER NOT NULL REFERENCES users(id),
    badge TEXT NOT NULL,
    day TEXT NOT NULL,
    PRIMARY KEY (user_id, badge)
);
"""


def connect(path):
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init(path):
    connection = connect(path)
    connection.executescript(SCHEMA)
    columns = {row["name"] for row in connection.execute("PRAGMA table_info(users)")}
    if "tracer" not in columns:
        connection.execute("ALTER TABLE users ADD COLUMN tracer TEXT")
        connection.commit()
    connection.close()
