#!/usr/bin/env bash
# Assemble the self-contained page and serve it on localhost:5060.
set -e
cd "$(dirname "$0")/.."
python3 viz/assemble.py
cd viz
echo "serving http://localhost:5060/  (Ctrl-C to stop)"
exec python3 -m http.server 5060
