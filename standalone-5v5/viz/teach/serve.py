#!/usr/bin/env python3
"""Serve the 5-a-side teaching page on localhost:6012.

    python3 viz/teach/serve.py            # -> http://localhost:6012
    PORT=6012 python3 viz/teach/serve.py

Static, no-cache (so a re-run of gen_data.py shows up on refresh). Stdlib only.
Regenerates data.json first if it is missing.
"""
import os, sys, subprocess
from http.server import HTTPServer, SimpleHTTPRequestHandler

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("PORT", os.environ.get("SOCCER_TEACH_PORT", "6012")))


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=HERE, **k)


    def guess_type(self, path):
        # UTF-8 charset on every text response: the pages embed em-dashes and
        # arrows; served without a charset, browsers may fall back to Latin-1
        # and render mojibake ('\u00e2\u20ac\u201d' where '\u2014' belongs).
        base = super().guess_type(path)
        if isinstance(base, str) and base.startswith(("text/", "application/javascript")) and "charset" not in base:
            return base + "; charset=utf-8"
        return base
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
