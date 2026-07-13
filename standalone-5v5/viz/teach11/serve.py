#!/usr/bin/env python3
"""Serve the 11-a-side teaching page on localhost:6013.

    python3 viz/teach11/serve.py            # -> http://localhost:6013
    PORT=6013 python3 viz/teach11/serve.py

Static, no-cache (so a re-run of gen_data.py shows up on refresh). Stdlib only.
Regenerates data.json first if it is missing.
"""
import os, sys, subprocess
from http.server import HTTPServer, SimpleHTTPRequestHandler

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("PORT", os.environ.get("SOCCER_TEACH11_PORT", "6013")))


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=HERE, **k)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def log_message(self, fmt, *args):   # quieter
        sys.stderr.write("  " + (fmt % args) + "\n")


def main():
    data = os.path.join(HERE, "data.json")
    if not os.path.exists(data):
        print("data.json missing — generating…")
        subprocess.run([sys.executable, os.path.join(HERE, "gen_data.py")], check=True)
    httpd = HTTPServer(("127.0.0.1", PORT), Handler)
    print(f"5-a-side teaching page  →  http://localhost:{PORT}")
    print("Ctrl-C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
