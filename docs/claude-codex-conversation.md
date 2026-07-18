# Claude-Codex LAN Conversation Runbook

This runbook describes how two coding agents on two MacBooks on the same WiFi
network should coordinate during a long debugging or learning run.

The goal is not to make two agents chat endlessly. The goal is to make one agent
able to send concise findings, questions, and patch status to the other, while
both keep a shared source of truth through Git and local runtime evidence.

## Recommended Roles

- Orchestrator: owns the active task state, decides the next experiment, and
  avoids duplicate edits.
- Reviewer or second operator: challenges the diagnosis, proposes falsification
  tests, reviews diffs, and can run independent experiments on the other
  machine.

Only one agent should edit the same file at a time. If both machines need to
produce code, use Git to sync through `main` and communicate which files are
being touched before editing.

## Network Shape

Each machine can expose a tiny HTTP bridge on its LAN IP:

- Codex machine inbox: `http://<codex-lan-ip>:8765`
- Codex machine outbox: `http://<codex-lan-ip>:8767`
- Claude machine inbox: `http://<claude-lan-ip>:8766`

Use concrete LAN IPs from `ipconfig getifaddr en0` or a Bonjour host such as
`MacBook-Pro.local`. Do not hardcode old IPs in scripts unless the DHCP lease is
stable.

The bridge should bind to `0.0.0.0` or the real LAN IP when it must be reachable
from the other Mac. Binding only to `127.0.0.1` is local-only.

## Endpoints

Each bridge should expose:

```text
GET  /health
GET  /messages?since=<id>
POST /messages
```

If using an outbox pull model:

```text
GET  /outbox?since=<id>
POST /outbox
```

For the local Codex bridge implementation in
[`scripts/claude_codex_bridge.py`](../scripts/claude_codex_bridge.py), the short
operator endpoints are also supported:

```text
GET  /codex?since=<id>   # alias for /messages, backed by /tmp/codex_claude_inbox.jsonl
POST /codex              # alias for /messages
GET  /claude?since=<id>  # alias for /outbox, backed by /tmp/codex_claude_outbox.jsonl
POST /claude             # alias for /outbox
```

On the Codex machine, Claude should normally post to `/codex` or `/messages` and
poll `/claude` or `/outbox`.

The outbox model is useful when one agent's sandbox can receive local writes but
cannot make outbound LAN requests. In that case:

1. The restricted agent writes messages to its local outbox.
2. The unrestricted machine polls `GET /outbox?since=<id>`.
3. The polling machine posts replies to the restricted agent's inbox, or stores
   replies for the restricted agent to poll locally.

## Authentication

Use a bearer token for every non-health endpoint:

```text
Authorization: Bearer <shared-random-token>
```

Do not commit tokens. Put them in environment variables or temp files outside
the repo, for example:

```text
CODEX_CLAUDE_BRIDGE_TOKEN=<shared-random-token>
```

The repo bridge script also accepts `CODEX_CLAUDE_BRIDGE_TOKEN_FILE`. If neither
is set, it creates `/tmp/codex_claude_bridge_token` with mode `0600`; share that
token out of band with the other machine. `GET /health` must not reveal it.

`GET /health` should not require auth. It should return only service identity,
host, port, and endpoint names. It must not return the token.

## Message Format

Use small JSON payloads:

```json
{
  "from": "codex",
  "topic": "plateau",
  "prompt": "Current evidence, question, or request."
}
```

Preferred fields:

- `from`: `codex` or `claude`
- `topic`: short stable topic, such as `plateau`, `diff-review`, or
  `runtime-health`
- `prompt`: concise human-readable message
- `run_id`: optional active run id
- `commit`: optional Git SHA
- `files`: optional list of touched files
- `metrics`: optional structured scoreboard

Keep messages short. Send evidence and a question, not a transcript dump.

## Health Checks

From machine A to machine B:

```bash
curl -sS --max-time 5 http://<machine-b-lan-ip>:<port>/health
```

Authenticated message read:

```bash
curl -sS --max-time 5 \
  -H "Authorization: Bearer $CODEX_CLAUDE_BRIDGE_TOKEN" \
  "http://<machine-b-lan-ip>:<port>/messages?since=0"
```

Authenticated message post:

```bash
curl -sS --max-time 5 \
  -X POST "http://<machine-b-lan-ip>:<port>/messages" \
  -H "Authorization: Bearer $CODEX_CLAUDE_BRIDGE_TOKEN" \
  -H "Content-Type: application/json" \
  --data '{"from":"codex","topic":"plateau","prompt":"Please review this diagnosis."}'
```

## Launching the Codex Bridge

Run bridge servers from a normal host Terminal, not from a sandbox that cannot
bind listening sockets. From the repo root:

```bash
python3 scripts/claude_codex_bridge.py --check
nohup python3 scripts/claude_codex_bridge.py --host 0.0.0.0 --port 8765 --role codex-inbox > /tmp/codex_claude_bridge_8765.log 2>&1 &
nohup python3 scripts/claude_codex_bridge.py --host 0.0.0.0 --port 8767 --role codex-outbox > /tmp/codex_claude_bridge_8767.log 2>&1 &
```

Expected local checks:

```bash
curl -sS --max-time 5 http://127.0.0.1:8765/health
curl -sS --max-time 5 http://127.0.0.1:8767/health
```

Expected WiFi/LAN checks from the Codex machine itself and from Claude's Mac:

```bash
curl -sS --max-time 5 http://<codex-lan-ip>:8765/health
curl -sS --max-time 5 http://<codex-lan-ip>:8767/health
```

Authenticated alias checks:

```bash
TOKEN="$(cat /tmp/codex_claude_bridge_token)"
curl -sS --max-time 5 -H "Authorization: Bearer $TOKEN" "http://<codex-lan-ip>:8765/codex?since=0"
curl -sS --max-time 5 -X POST "http://<codex-lan-ip>:8767/claude" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  --data '{"from":"codex","topic":"bridge-health","prompt":"Codex bridge POST /claude test."}'
```

If `lsof -nP -iTCP:8765 -iTCP:8767 -sTCP:LISTEN` shows a listener but
`/health` refuses connections, treat that process as stale only after confirming
its command/cwd with `lsof -nP -p <pid>`. Stop the stale process from the
terminal that owns it, or use `kill <pid>` only after confirming it is the broken
bridge and not another useful service.

## NordVPN and LAN Routing

A NordVPN dedicated public IP, including a Houston endpoint, is not the bridge
address for two Macs on the same WiFi. Use the local WiFi address from
`ipconfig getifaddr en0`, such as `192.168.100.19`, for Claude-to-Codex traffic.

The VPN should leave the local subnet route on `en0`:

```bash
netstat -rn -f inet
```

Look for a route like `192.168.100 link#... en0`. If Claude cannot reach
`http://<codex-lan-ip>:8765/health` while localhost works, check NordVPN's LAN
visibility/local-network setting, macOS Firewall, and whether the bridge is bound
to `0.0.0.0` or the LAN IP rather than `127.0.0.1`.

## One-Way Network Failure

If `A -> B` works but `B -> A` fails:

1. Confirm both machines are on the same WiFi.
2. Confirm both LAN IPs with `ipconfig getifaddr en0`.
3. Confirm the bridge binds to `0.0.0.0` or the LAN IP, not just `127.0.0.1`.
4. Check macOS Firewall and app permissions.
5. Check whether the agent sandbox blocks outbound LAN traffic.
6. If outbound is blocked, use the outbox pull model.

The outbox pull model is the preferred fallback because it avoids changing agent
sandbox permissions during a long training run.

## Git Sync Discipline

For shared code work:

1. Stay on `main`.
2. Before editing, say which files are being touched.
3. Pull before starting a patch.
4. Commit coherent changes with tests.
5. Push to `main`.
6. Tell the other agent the commit SHA and the relevant test output.

Use this shape for sync messages:

```json
{
  "from": "codex",
  "topic": "git-sync",
  "prompt": "Pushed abc1234. Touched src/bin/main_soccer_learning_run.rs. Tests: cargo test --bin main_soccer_learning_run local_trial_anchor_rejects_bad_fitness_even_when_checkpoint_is_held --features postgres-persistence."
}
```

Do not let both agents edit the same file concurrently unless they are using
separate commits and are ready to resolve conflicts semantically.

## Long-Run Learning Heartbeats

For training or plateau work, send a heartbeat every few minutes with:

- active run id
- completed windows
- current metric table
- publish or hold decision
- whether the live viewer uses protected or candidate weights
- current hypothesis
- next falsification experiment

Example:

```json
{
  "from": "codex",
  "topic": "plateau-heartbeat",
  "run_id": "codex-soccer-learning-local-neural-authoritative-...",
  "metrics": {
    "windows": [
      {"window": 1, "mean_match_fitness": -1.1121, "decision": "held"},
      {"window": 2, "mean_match_fitness": -0.9025, "decision": "held"}
    ],
    "publish_gate": -0.25,
    "live_policy": "protected"
  },
  "prompt": "Trend is improving but still below gate. Challenge whether the next experiment should target action expression or defensive reward balance."
}
```

## What Good Collaboration Looks Like

Good messages are falsifiable:

- "The candidate is not promoted because mean fitness is below the gate."
- "The viewer is showing protected weights, not candidate weights."
- "The reward is not sparse; nonzero rewards are present on every transition."
- "The likely blocker is action expression because learned candidates exist but
  are selected rarely."

Weak messages are vague:

- "It seems better."
- "Maybe the model is learning."
- "Try more exploration."

For plateau work, always include the scoreboard and the next experiment that
would prove or disprove the diagnosis.

## Minimal Bridge Implementation

A bridge can be a tiny HTTP server that appends JSON lines to a temp file. The
server needs only:

- `GET /health`
- `GET /messages?since=<id>`
- `POST /messages`
- monotonically increasing `id`
- bearer-token auth for non-health endpoints

Store bridge state outside the repo, for example:

```text
/tmp/codex_claude_inbox.jsonl
/tmp/codex_claude_outbox.jsonl
```

That keeps coordination durable enough for overnight runs without polluting the
working tree.
