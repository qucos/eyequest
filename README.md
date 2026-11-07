# EyeQuest

Daily eye practice for a lazy eye: webcam eye tracking, short games, logins, streaks and XP. Everything runs and stays on your computer.

**EyeQuest isn't a medical device and can't diagnose or treat amblyopia.** Use it alongside an eye doctor's care, not in place of it. Stop if you get a headache, eye pain or double vision. If double vision or a change in your vision comes on suddenly, get medical care right away.

## Run it

```bash
./run.sh
```

The first run takes a minute or two to install everything. The app then opens at <http://localhost:5050>. Chrome is the safest choice. Allow the camera when your browser asks. Press Ctrl+C in the terminal to stop.

## What's inside

- **Accounts:** usernames and hashed passwords, stored only on this computer. There's no email reset.
- **Daily habit:** a lesson path, XP, a daily goal, stars, badges and a streak. You earn streak protection every 7 days, up to 2. After 4 lessons in a day, lessons stop earning XP and you're told to rest.
- **Eye Tracker screen:** asks for the camera when it opens, then shows live markers on your irises and a tracer where you're looking (once calibrated).
- **Custom tracer:** choose its shape, color, size, opacity, trail and movement in Settings, with a live preview.
- **Games:**
  - Firefly Follow: follow a moving target.
  - Star Hop: jump your eyes between targets.
  - Steady Glow: hold your gaze on one spot.
  - Gap Spotter: find the gap in a ring that gets smaller and fainter. Press an arrow key to answer.
- **Solo-eye lessons:** ask you to cover your stronger eye so the lazy eye does the work.
- **Progress:** a practice calendar, how closely each eye follows the firefly, and the smallest ring you spotted.

## How the tracking works

1. The browser captures the camera and sends each frame to the local Python app.
2. MediaPipe Face Landmarker finds each iris and eye corner (`tracker.py`). Frames are never saved.
3. A daily calibration of 17 dots learns how your eyes and eyelids move as you look around (`gaze.py`). Dots where your eye or head was still moving are shown again. It builds a fit for each eye and one for both together, which uses a straight line plus a smooth curve.
4. An accuracy test on 7 new dots gives your score and removes any steady shift or squeeze.
5. Before each game, a one-dot check corrects small head shifts.

On the Eye Tracker screen, **Fine-tune** lets you look at any spot and click it, which corrects the tracker around that area and reports your real miss. Head tracking uses MediaPipe's 3D head pose: your real turn and tilt angles in degrees, plus your position and approximate distance from the camera. A guided calibration stage has you turn your head while watching dots, which teaches the tracker how far a head movement shifts your gaze on your screen. Each head direction is scored using a correction fitted to the other directions, and head correction switches off if it doesn't help. The Eye Tracker screen shows your live head angles, and warns you if you've moved further than the tracker can account for.

Webcam tracking is much rougher than a clinical eye tracker. Hit zones grow to match the error measured during calibration. Progress numbers are useful as long-term trends, but they aren't vision measurements. Good light, a still head and full screen all help.

## Project layout

| Path | What it does |
|---|---|
| `app.py` | Web routes and API |
| `tracker.py`, `gaze.py` | Eye detection and gaze mapping |
| `curriculum.py` | Lessons and game settings |
| `progress.py` | Streak, XP, star and badge rules |
| `db.py` | SQLite tables (data is saved in `data/`) |
| `static/`, `templates/` | Games and screens |

Run the tests with `.venv/bin/python -m pytest`.
