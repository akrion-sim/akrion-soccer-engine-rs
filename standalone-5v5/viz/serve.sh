#!/usr/bin/env bash
# Assemble the self-contained page and serve it locally.
set -e
PORT="${PORT:-8080}"
cd "$(dirname "$0")/.."
python3 viz/assemble.py
cd viz
echo "serving http://localhost:${PORT}/  (Ctrl-C to stop)"
exec python3 -m http.server "${PORT}"
