#!/usr/bin/env python3
"""Serve the self-play champion viewer on :8081.

`/newgame` records a fresh champion-vs-champion match with the current best
policy. This mirrors `serve_live.py`'s safety contract: localhost-only, clamped
env parsing, bounded concurrent match generation, and explicit subprocess
errors instead of silent `{}` responses.
"""
import json
import os
import subprocess
import tempfile
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "out")
BIN = os.path.join(ROOT, "target", "release", "fiveaside")
MAX_SEED = 2 ** 63 - 1


def env_int(name, default, lo=None, hi=None):
    try:
        value = int(os.environ.get(name, str(default)), 0)
    except ValueError:
        value = default
    if lo is not None:
        value = max(lo, value)
    if hi is not None:
        value = min(hi, value)
    return value


PORT = env_int("PORT", 8081, lo=1, hi=65535)
TIMEOUT = env_int("NEWGAME_TIMEOUT", 120, lo=1, hi=3600)
MAX_NEWGAME_CONCURRENCY = env_int("NEWGAME_MAX_CONCURRENCY", 2, lo=1, hi=16)
NEWGAME_SEMAPHORE = threading.BoundedSemaphore(MAX_NEWGAME_CONCURRENCY)


def policy_complete(path):
    return all(os.path.exists(os.path.join(path, name)) for name in ("actor.txt", "critic.txt", "speedor.txt"))


def champion_dir():
    if policy_complete(OUT):
        return OUT
    cdir = os.path.join(OUT, "champions")
    gens = []
    if os.path.isdir(cdir):
        for name in os.listdir(cdir):
            if name.startswith("gen") and name[3:].isdigit():
                path = os.path.join(cdir, name)
                if policy_complete(path):
                    gens.append((int(name[3:]), path))
    gens.sort()
    if gens:
        return gens[-1][1]
    raise FileNotFoundError(f"no complete champion policy found under {OUT}")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=OUT, **kwargs)

    def log_message(self, *args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/newgame":
            return self._new_game(parse_qs(parsed.query))
        return super().do_GET()

    def _new_game(self, query):
        try:
            seed = max(0, min(MAX_SEED, int(query.get("seed", ["7"])[0], 0)))
        except (ValueError, IndexError):
            seed = 7
        if not NEWGAME_SEMAPHORE.acquire(blocking=False):
            return self._json(429, {"error": "too many match generations in flight"})

        fd, tmp = tempfile.mkstemp(suffix=".json", prefix="selfplay_match_")
        os.close(fd)
        try:
            champ = champion_dir()
            proc = subprocess.run(
                [BIN, "play", str(seed), "--out-dir", champ, "--opponent", champ, "--out", tmp],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=TIMEOUT,
            )
            if proc.returncode != 0:
                return self._json(500, {"error": "play failed", "detail": (proc.stderr or proc.stdout)[-400:]})
            with open(tmp, "rb") as fh:
                body = fh.read()
        except subprocess.TimeoutExpired:
            return self._json(504, {"error": "match generation timed out"})
        except FileNotFoundError as exc:
            return self._json(500, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - this is an operator-facing local tool
            return self._json(500, {"error": str(exc)})
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            NEWGAME_SEMAPHORE.release()

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
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    print(
        f"self-play champion viewer: http://127.0.0.1:{PORT}/ "
        f"(max {MAX_NEWGAME_CONCURRENCY} new games in flight)"
    )
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    server.daemon_threads = True
    server.serve_forever()
