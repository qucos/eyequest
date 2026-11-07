#!/bin/zsh
# Kept for anyone used to this command. start.py does all the work.
cd "$(dirname "$0")" || exit 1
if ! command -v python3 >/dev/null; then
  echo "EyeQuest needs Python 3. Get it from https://www.python.org/downloads/"
  exit 1
fi
exec python3 start.py "$@"
