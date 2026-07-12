#!/usr/bin/env python3
"""Sidecar that turns the 11v11 league training logs into a climb time-series JSON,
so the :5055 live viewer can draw the same kind of learning-climb charts the 5v5
viewers (:8080/:8081) show — goals for/against, pass completion, turnovers, shots.

It re-parses the logs on every request (cheap: a few hundred `league_kpi` lines),
so the charts stay live as training advances. CORS is open so the :5055 page
(a different origin/port) can fetch it.

    CLIMB_PORT=5056 python3 climb_server.py
"""
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Which training arms to surface, in draw order. label -> log path.
def _arms_from_env():
    """CLIMB_ARMS="label:path,label:path" overrides the arms; else the default 11v11 A/B logs."""
    raw = os.environ.get("CLIMB_ARMS", "").strip()
    out = []
    for part in raw.split(","):
        if ":" in part:
            label, path = part.split(":", 1)
            if label.strip() and path.strip():
                out.append((label.strip(), path.strip()))
    return out or [
        ("control", "/tmp/conv_climb_train.log"),
        ("grounded", "/tmp/conv_anchor_train.log"),
    ]


ARMS = _arms_from_env()

# Fields pulled from each `league_kpi round=N ...` line. name -> (kind).
INT_FIELDS = ["round", "games", "goals_for", "goals_against", "gd_total",
              "shots_for", "shots_against", "sot_for", "sot_against",
              "completed_passes", "forward_passes", "chain_net_losses", "assists"]
FLOAT_FIELDS = ["gd_per_game", "pass_completion", "forward_pass_margin_per_game",
                "net_forward_pass_margin_per_game"]

KPI_RE = re.compile(r"\bleague_kpi\b")
KV_RE = re.compile(r"([a-z_]+)=(-?[0-9]+(?:\.[0-9]+)?)")


def env_int(name, default, lo=None, hi=None):
    try:
        v = int(os.environ.get(name, str(default)), 0)
    except ValueError:
        v = default
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v


PORT = env_int("CLIMB_PORT", 5056, lo=1, hi=65535)


def parse_log(path):
    """Return a list of per-round dicts parsed from a training log, in round order."""
    try:
        with open(path, "r", errors="replace") as fh:
            text = fh.read()
    except FileNotFoundError:
        return []
    except OSError:
        return []
    rows = {}
    for line in text.splitlines():
        if not KPI_RE.search(line):
            continue
        kv = {k: v for k, v in KV_RE.findall(line)}
        if "round" not in kv:
            continue
        row = {}
        for f in INT_FIELDS:
            if f in kv:
                try:
                    row[f] = int(float(kv[f]))
                except ValueError:
                    pass
        for f in FLOAT_FIELDS:
            if f in kv:
                try:
                    row[f] = float(kv[f])
                except ValueError:
                    pass
        rnd = row.get("round")
        if rnd is None:
            continue
        # a round can emit several league_kpi lines; keep the richest (most keys)
        if rnd not in rows or len(row) > len(rows[rnd]):
            rows[rnd] = row
    return [rows[r] for r in sorted(rows)]


def build_payload():
    arms = {}
    for label, path in ARMS:
        arms[label] = parse_log(path)
    return {
        "arms": arms,
        "arm_order": [label for label, _ in ARMS],
        # what each series means, for the client legend/labels
        "series": {
            "goals_for": "Goals for", "goals_against": "Goals against",
            "gd_per_game": "Goal diff / game", "pass_completion": "Pass completion",
            "chain_net_losses": "Turnovers (chain losses)",
            "sot_for": "Shots on target for", "sot_against": "Shots on target against",
        },
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/climb.json", "/climb", "/"):
            try:
                body = json.dumps(build_payload()).encode()
            except Exception as exc:  # noqa: BLE001 - always answer the poller
                body = json.dumps({"error": str(exc), "arms": {}}).encode()
            return self._send(200, body)
        return self._send(404, b'{"error":"not found"}')

    def log_message(self, *args):  # quiet
        pass


if __name__ == "__main__":
    counts = {label: len(parse_log(path)) for label, path in ARMS}
    print(f"climb sidecar on http://127.0.0.1:{PORT}/climb.json  rounds={counts}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
