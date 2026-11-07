#!/bin/zsh
# First run sets everything up; after that it just starts the app.
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv || exit 1
  .venv/bin/pip install -q -r requirements.txt || exit 1
fi
( sleep 2; open "http://localhost:5050" ) &
exec .venv/bin/python app.py
