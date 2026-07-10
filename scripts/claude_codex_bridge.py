#!/usr/bin/env python3
"""Tiny Claude/Codex LAN bridge backed by JSONL files in /tmp.

The Codex machine normally runs this on:
  - 0.0.0.0:8765 for the Codex inbox
  - 0.0.0.0:8767 for the Codex outbox pull endpoint

Routes intentionally include both the documented names and the short names that
show up in operator notes:
  - /messages and /codex read/write the inbox file
  - /outbox and /claude read/write the outbox file
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import secrets
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


DEFAULT_INBOX = Path("/tmp/codex_claude_inbox.jsonl")
DEFAULT_OUTBOX = Path("/tmp/codex_claude_outbox.jsonl")
DEFAULT_TOKEN_FILE = Path("/tmp/codex_claude_bridge_token")


@dataclass(frozen=True)
class BridgeConfig:
    host: str
    port: int
    role: str
    inbox: Path
    outbox: Path
    token: str
    token_source: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_or_create_token(token_file: Path) -> tuple[str, str]:
    env_token = os.environ.get("CODEX_CLAUDE_BRIDGE_TOKEN", "").strip()
    if env_token:
        return env_token, "env:CODEX_CLAUDE_BRIDGE_TOKEN"

    if token_file.exists():
        token = token_file.read_text(encoding="utf-8").strip()
        if token:
            return token, str(token_file)

    token = secrets.token_urlsafe(32)
    token_file.parent.mkdir(parents=True, exist_ok=True)
    token_file.write_text(f"{token}\n", encoding="utf-8")
    os.chmod(token_file, 0o600)
    return token, str(token_file)


def json_response(handler: BaseHTTPRequestHandler, status: HTTPStatus, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def read_jsonl(path: Path, since: int) -> tuple[list[dict[str, Any]], int]:
    messages: list[dict[str, Any]] = []
    max_id = 0
    if not path.exists():
        return messages, max_id

    with path.open("r", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        try:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                record_id = int(record.get("id", 0) or 0)
                max_id = max(max_id, record_id)
                if record_id > since:
                    messages.append(record)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return messages, max_id


def append_jsonl(path: Path, payload: dict[str, Any], route: str, role: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            max_id = 0
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                max_id = max(max_id, int(record.get("id", 0) or 0))

            record = dict(payload)
            record["id"] = max_id + 1
            record.setdefault("received_at", utc_now())
            record.setdefault("bridge_role", role)
            record.setdefault("route", route)
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return record


class BridgeHandler(BaseHTTPRequestHandler):
    server_version = "ClaudeCodexBridge/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write("%s - - [%s] %s\n" % (self.client_address[0], self.log_date_time_string(), format % args))

    @property
    def config(self) -> BridgeConfig:
        return self.server.bridge_config  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path.rstrip("/") or "/"

        if route in {"/", "/health"}:
            self.handle_health()
            return

        if route not in {"/messages", "/outbox", "/codex", "/claude"}:
            json_response(self, HTTPStatus.NOT_FOUND, {"error": "unknown route", "route": route})
            return

        if not self.authorized():
            return

        query = parse_qs(parsed.query)
        since_raw = query.get("since", ["0"])[0]
        try:
            since = max(0, int(since_raw))
        except ValueError:
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "since must be an integer"})
            return

        path = self.path_for_route(route)
        messages, next_id = read_jsonl(path, since)
        json_response(
            self,
            HTTPStatus.OK,
            {
                "messages": messages,
                "next_id": next_id,
                "path": str(path),
                "route": route,
            },
        )

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path.rstrip("/") or "/"

        if route not in {"/messages", "/outbox", "/codex", "/claude"}:
            json_response(self, HTTPStatus.NOT_FOUND, {"error": "unknown route", "route": route})
            return

        if not self.authorized():
            return

        length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(length)
        try:
            payload = json.loads(body.decode("utf-8") if body else "{}")
        except json.JSONDecodeError as exc:
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": f"invalid json: {exc}"})
            return

        if not isinstance(payload, dict):
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "payload must be a JSON object"})
            return

        record = append_jsonl(self.path_for_route(route), payload, route, self.config.role)
        json_response(self, HTTPStatus.CREATED, {"message": record})

    def handle_health(self) -> None:
        json_response(
            self,
            HTTPStatus.OK,
            {
                "service": "claude-codex-bridge",
                "role": self.config.role,
                "host": self.config.host,
                "port": self.config.port,
                "auth": "bearer",
                "token_source": self.config.token_source,
                "routes": ["/health", "/messages", "/outbox", "/codex", "/claude"],
                "inbox": str(self.config.inbox),
                "outbox": str(self.config.outbox),
            },
        )

    def authorized(self) -> bool:
        header = self.headers.get("Authorization", "")
        expected = f"Bearer {self.config.token}"
        if secrets.compare_digest(header, expected):
            return True
        self.send_response(HTTPStatus.UNAUTHORIZED)
        self.send_header("WWW-Authenticate", "Bearer")
        self.send_header("Content-Type", "application/json")
        body = b'{"error":"missing or invalid bearer token"}'
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        return False

    def path_for_route(self, route: str) -> Path:
        if route in {"/outbox", "/claude"}:
            return self.config.outbox
        return self.config.inbox


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Claude/Codex JSONL LAN bridge.")
    parser.add_argument("--host", default=os.environ.get("CODEX_CLAUDE_BRIDGE_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("CODEX_CLAUDE_BRIDGE_PORT", "8765")))
    parser.add_argument("--role", default=os.environ.get("CODEX_CLAUDE_BRIDGE_ROLE", "codex"))
    parser.add_argument("--inbox", type=Path, default=Path(os.environ.get("CODEX_CLAUDE_INBOX", DEFAULT_INBOX)))
    parser.add_argument("--outbox", type=Path, default=Path(os.environ.get("CODEX_CLAUDE_OUTBOX", DEFAULT_OUTBOX)))
    parser.add_argument(
        "--token-file",
        type=Path,
        default=Path(os.environ.get("CODEX_CLAUDE_BRIDGE_TOKEN_FILE", DEFAULT_TOKEN_FILE)),
    )
    parser.add_argument("--check", action="store_true", help="Validate config and print non-secret status without binding a socket.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token, token_source = load_or_create_token(args.token_file)
    config = BridgeConfig(
        host=args.host,
        port=args.port,
        role=args.role,
        inbox=args.inbox,
        outbox=args.outbox,
        token=token,
        token_source=token_source,
    )

    if args.check:
        print(
            json.dumps(
                {
                    "service": "claude-codex-bridge",
                    "role": config.role,
                    "host": config.host,
                    "port": config.port,
                    "inbox": str(config.inbox),
                    "outbox": str(config.outbox),
                    "token_source": config.token_source,
                    "routes": ["/health", "/messages", "/outbox", "/codex", "/claude"],
                },
                sort_keys=True,
            )
        )
        return 0

    server = ThreadingHTTPServer((config.host, config.port), BridgeHandler)
    server.bridge_config = config  # type: ignore[attr-defined]
    print(
        json.dumps(
            {
                "service": "claude-codex-bridge",
                "role": config.role,
                "listening": f"{config.host}:{config.port}",
                "token_source": config.token_source,
                "routes": ["/health", "/messages", "/outbox", "/codex", "/claude"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
