# Live viewer ports

What each local port serves. All are localhost-only.

| Port | What | Game | Source / server | Net it plays |
|------|------|------|-----------------|--------------|
| **:5055** | Live 11v11 match viewer (dense telemetry + KPI tiles + pass-mix bar, theme-aware) | **11v11** (the full engine) | Rust binary `main_soccer_live` (`src/bin/main_soccer_live.rs`, HTML in `src/des/general/soccer_live_ui.html`), deployed as `/tmp/dd-soccer-rs-live` | The current **conversion-climb frontier** (auto-promoted; keeper backup at `~/akrion_proof/overnight/serve_5055.keeper-gen76.json`) |
| **:8080** | 5-a-side "before / after learning" showcase (before/after pitches, learning curve, trend charts) | **5v5** standalone | `standalone-5v5/viz/serve_live.py` **run from the `tmp/worktrees/five-a-side` worktree** | A **`train`-mode** policy (learner **vs scripted baseline**) — a completed/static demo |
| **:8081** | 5-a-side **self-play champion** viewer (champion-vs-champion match + ladder-climb chart) | **5v5** standalone | `standalone-5v5/viz/serve_selfplay.py` (from the primary checkout) | The current **self-play champion** (both teams learned; `selfplay` command) |

## Key distinctions

- **:8080 vs :8081 are two *different* 5v5 efforts.** `:8080` = `train` mode (one learner vs the fixed scripted baseline). `:8081` = `selfplay` mode (a champion-ladder where *both* teams are learned policies and a new winner must beat the old champion to advance). They live in different worktrees / `out/` dirs and do not share nets.
- **:5055 shows the 11v11 net that is actively training**, refreshed on each frontier promotion by the auto-promote loop (`scripts`/scratchpad `conv_5055_autopromote.sh`). `:8080` is a *finished* demo; `:8081` reflects the *running* self-play champion.

## How each is (re)built / served

- **:5055** — `cargo build --release --bin main_soccer_live --features live-server`, then run with `SOCCER_LIVE_PORT=5055 SOCCER_LIVE_POLICY_PATH=<serve.json> SOCCER_LIVE_AUTOLOAD_POLICY=1 DD_SOCCER_ENABLE_RELATIONAL_ATTENTION=1 …`. **Port env var is `SOCCER_LIVE_PORT` (or `SOCCER_PORT`), NOT `PORT`.** The live match is client-driven (a browser must click **Run**; the server does not auto-advance).
- **:8080** — `cargo run --release -- train …` in the 5v5 crate, then `python3 viz/assemble.py` (builds `index.html` from `out/run_manifest.json`), then `PORT=8080 python3 viz/serve_live.py`.
- **:8081** — `cargo run --release -- selfplay …` (writes `out/champions/`, `out/selfplay_ladder.csv`), then `python3 viz/selfplay_viz.py` (records a champion-vs-champion match + builds `out/index.html` from the ladder), then `PORT=8081 python3 viz/serve_selfplay.py`. The **New Game** button records a fresh champion-vs-champion match via `fiveaside play <seed> --out-dir <champ> --opponent <champ>`.

## Blocked ports

`5060` / `5061` are browser-blocked (SIP) — do not serve viewers there. Use `5055`, `8080`, `8081`, or another open port.
