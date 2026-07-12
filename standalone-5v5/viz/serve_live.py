#!/usr/bin/env python3
"""Live server for the 5-a-side viz.

Serves the static viz (index.html, etc.) AND a `/newgame?seed=N` endpoint that
records a FRESH match with the trained model on demand — this is what the
"New Game" button calls. Each click generates a genuinely new game (new seed)
that the trained policy plays out, then streams it back for playback.

    ./target/release/fiveaside  must exist (build first) and out/actor.txt etc.
    must exist (train first). Then:  PORT=8080 python3 viz/serve_live.py
"""
import json
import os
import subprocess
import tempfile
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BIN = os.path.join(ROOT, "target", "release", "fiveaside")
OUT_DIR = os.path.join(ROOT, "out")
PORT = int(os.environ.get("PORT", "8080"))
TIMEOUT = int(os.environ.get("NEWGAME_TIMEOUT", "120"))


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=HERE, **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/newgame":
            return self._new_game(parse_qs(parsed.query))
        return super().do_GET()

    def _new_game(self, query):
        try:
            seed = int(query.get("seed", ["7"])[0]) % (2 ** 63)
        except (ValueError, IndexError):
            seed = 7
        # unique output path so concurrent clicks don't collide
        fd, out_path = tempfile.mkstemp(suffix=".json", prefix="match_live_")
        os.close(fd)
        try:
            proc = subprocess.run(
                [BIN, "play", str(seed), "--out-dir", OUT_DIR, "--out", out_path],
                cwd=ROOT, capture_output=True, text=True, timeout=TIMEOUT,
            )
            if proc.returncode != 0:
                return self._json(500, {"error": "play failed",
                                        "detail": (proc.stderr or proc.stdout)[-400:]})
            with open(out_path) as fh:
                body = fh.read().encode()
        except subprocess.TimeoutExpired:
            return self._json(504, {"error": "match generation timed out"})
        except FileNotFoundError:
            return self._json(500, {"error": f"binary not found at {BIN}; build first"})
        except Exception as exc:  # noqa: BLE001 - surface any failure to the client
            return self._json(500, {"error": str(exc)})
        finally:
            try:
                os.unlink(out_path)
            except OSError:
                pass
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    print(f"live viz on http://localhost:{PORT}  (New Game -> {BIN} play <seed>)")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
