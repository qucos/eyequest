# EyeQuest

Daily eye practice for a lazy eye, using the webcam already in your laptop. Short games follow where you look, and logins, streaks and XP make practice a daily habit. Everything runs and stays on your computer.

**EyeQuest isn't a medical device and can't diagnose or treat amblyopia.** Use it alongside an eye doctor's care, not in place of it. Stop if you get a headache, eye pain or double vision. If double vision or a change in your vision comes on suddenly, get medical care right away.

## Why

Lazy-eye practice only helps if people keep doing it, and daily exercises get boring fast, especially for kids. EyeQuest turns practice into a few minutes of games a day and uses eye tracking to check that the eyes are actually doing the work. It needs no special hardware.

## Run it

You need [Python](https://www.python.org/downloads/) 3.11 or newer. In a terminal, go to this folder and run:

```bash
python3 start.py
```

The first run takes a few minutes to install everything, then the app opens in your browser. Allow the camera when your browser asks. Press Ctrl+C in the terminal to stop.

If the install gets interrupted, run the same command again and it picks up where it left off. If port 5050 is busy, it finds another one and tells you which.

## A two-minute tour

1. **Sign up** and choose which eye is your lazy eye.
2. **Calibrate** (about a minute): look at each dot, then at the green test dots. You get an accuracy score.
3. Open **Eye Tracker** to see your irises marked and a tracer following your gaze.
4. Play **Meet the firefly**, the first lesson, and watch your streak and XP start.
5. Check **Progress** for your practice calendar, badges, and how closely each eye followed the target.

No camera, or a demo going wrong? Turn on **Practice with the mouse** in Settings to play every game with the mouse.

## What's inside

- **Accounts:** usernames and hashed passwords, stored only on this computer. There's no email reset.
- **Daily habit:** a lesson path, XP, a daily goal, stars, badges and a streak. You earn streak protection every 7 days, up to 2. After 4 lessons in a day, lessons stop earning XP and you're told to rest.
- **Games:**
  - Firefly Follow: follow a moving target.
  - Star Hop: jump your eyes between targets.
  - Steady Glow: hold your gaze on one spot.
  - Gap Spotter: find the gap in a ring that gets smaller and fainter. Press an arrow key to answer.
  - Jellyfish Rescue: hold your gaze on each jellyfish until its ring fills.
- **Solo-eye lessons:** ask you to cover your stronger eye so the lazy eye does the work.
- **Progress:** a practice calendar, how closely each eye follows the firefly, and the smallest ring you spotted.
- **Head-position guide:** a short message such as "Lower your chin a little" if your head drifts from where you calibrated.
- **Bright stage:** a light background for calibration and games, which lights up your face in a dim room. It can switch on automatically.
- **Full screen:** your first click or key press turns it on, and it stays on as you move between pages. Press Esc to see your tabs.
- **Custom tracer:** choose its shape, color, size, opacity, trail and movement in Settings, with a live preview.

## How the tracking works

1. The browser captures the camera, 15 frames a second, and sends each frame to the local Python app. Frames are analyzed in memory and never saved.
2. MediaPipe Face Landmarker finds each iris, eyelid and eye corner, plus the head's 3D position and angle (`tracker.py`).
3. Calibration shows 17 dots and learns how your irises and upper eyelids move as you look around (`gaze.py`). The upper eyelid turned out to follow up-and-down looks much more clearly than the iris. Shaky dots are shown again. It builds a fit for each eye and one for both together: a straight line plus a smooth curve.
4. An accuracy test on 7 new dots gives your score. A calibration that misses by more than 30% of the screen's height isn't saved, and your previous one is kept.
5. Small head movements shift where your eyes sit, so the tracker remembers your head's position and the guide helps you hold it. Before each game, a one-dot check corrects any small leftover shift.

On the Eye Tracker screen, **Fine-tune** lets you look at any spot and click it, which corrects the tracker around that area and reports your real miss.

### How accurate it is

On the developer's own eyes with an M1 MacBook Pro's built-in camera, the best live calibration so far missed by a typical **6.6% of the screen's height**: about 4 to 8% across the top of the screen and better in the middle. That's one person and one camera, so expect different results. Good light, the camera at eye level, sitting about an arm's length away, and a still head all help.

Webcam tracking is much rougher than a clinical eye tracker, so hit zones in games grow to match the error measured during calibration. Progress numbers are useful as long-term trends, but they aren't vision measurements.

## Privacy

- Video never leaves your computer and is never saved. Each frame is analyzed in memory, then dropped.
- Accounts, progress and calibration live in `data/eyequest.db` on your computer, which is never uploaded to GitHub.
- The app only accepts connections from your own computer.

## Project layout

| Path | What it does |
|---|---|
| `start.py` | One-command setup and start |
| `app.py` | Web routes and API |
| `tracker.py`, `gaze.py` | Eye detection, gaze mapping and the head-position guide |
| `curriculum.py` | Lessons and game settings |
| `progress.py` | Streak, XP, star and badge rules |
| `db.py` | SQLite tables (data is saved in `data/`) |
| `static/`, `templates/` | Games and screens |

## Tests

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```

The tests also run automatically on GitHub, on Python 3.11 and 3.13, every time code is pushed.
