#!/usr/bin/env python3
"""Sets up EyeQuest if it needs it, starts it, and opens it in your browser.

Run it with: python3 start.py
It uses only Python's standard library, because nothing else is installed yet.
"""
import argparse
import hashlib
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"
MIN_PYTHON = (3, 10)
FIRST_PORT = 5050
PORT_TRIES = 20
READY_SECONDS = 60


class SetupError(Exception):
    """A problem the person can act on. The message says what to do."""


def venv_python(venv=VENV):
    folder = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    return venv / folder


def requirements_id(requirements=REQUIREMENTS):
    return hashlib.sha256(requirements.read_bytes()).hexdigest()


def needs_install(venv=VENV, requirements=REQUIREMENTS):
    """True until an install has finished for these exact requirements.

    The finished mark is written last, so an interrupted install is simply run again.
    """
    marker = venv / ".installed"
    return not marker.exists() or marker.read_text().strip() != requirements_id(requirements)


def _runs(command):
    try:
        return subprocess.run(command, capture_output=True).returncode == 0
    except OSError:
        return False


def install(venv=VENV, requirements=REQUIREMENTS):
    python = venv_python(venv)
    if not _runs([str(python), "-m", "pip", "--version"]):
        print("Setting up EyeQuest for the first time...", flush=True)
        if not _runs([sys.executable, "-m", "venv", "--clear", str(venv)]):
            raise SetupError(
                "I couldn't create EyeQuest's private Python folder. "
                "Reinstalling Python from https://www.python.org/downloads/ usually fixes this."
            )
    print("Installing what EyeQuest needs. The first time can take a few minutes...", flush=True)
    finished = subprocess.run([str(python), "-m", "pip", "install", "--disable-pip-version-check", "-r", str(requirements)])
    if finished.returncode != 0:
        raise SetupError(
            "The install didn't finish. Check your internet connection, then run this command "
            "again. It picks up where it left off."
        )
    (venv / ".installed").write_text(requirements_id(requirements))


def is_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        return probe.connect_ex(("127.0.0.1", port)) != 0


def free_port(first=FIRST_PORT, tries=PORT_TRIES):
    port = next((p for p in range(first, first + tries) if is_free(p)), None)
    if port is None:
        raise SetupError(f"Ports {first} to {first + tries - 1} are all in use. Close some apps and try again.")
    return port


def wait_until_ready(url, process, seconds=READY_SECONDS):
    """True once the app answers. False if it stops or takes too long."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and process.poll() is None:
        try:
            with urllib.request.urlopen(url, timeout=2):
                return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.3)
    return False


def run(open_browser):
    if sys.version_info < MIN_PYTHON:
        wanted = ".".join(map(str, MIN_PYTHON))
        raise SetupError(f"EyeQuest needs Python {wanted} or newer. Get it from https://www.python.org/downloads/")
    if needs_install():
        install()
    port = free_port()
    if port != FIRST_PORT:
        print(f"Port {FIRST_PORT} is busy, so EyeQuest will use port {port}.", flush=True)
    url = f"http://localhost:{port}"
    app = subprocess.Popen([str(venv_python()), "app.py"], cwd=ROOT, env={**os.environ, "EYEQUEST_PORT": str(port)})
    try:
        if not wait_until_ready(url, app):
            app.terminate()
            raise SetupError("EyeQuest didn't start. The messages above should say why.")
        if open_browser:
            webbrowser.open(url)
        return app.wait()
    except KeyboardInterrupt:
        app.terminate()
        app.wait()
        return 0


def main():
    parser = argparse.ArgumentParser(description="Set up and start EyeQuest.")
    parser.add_argument("--no-browser", action="store_true", help="start without opening a browser")
    arguments = parser.parse_args()
    try:
        sys.exit(run(not arguments.no_browser))
    except SetupError as problem:
        print(f"\n{problem}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
