<!-- BEGIN k8s-cluster-submodule-notice -->
> [!NOTE]
> **Canonical source.** This repository is the source of truth for its code. It
> is also vendored as a **secondary** git submodule of
> [ORESoftware/k8s-cluster](https://github.com/ORESoftware/k8s-cluster) at
> `remote/submodules/soccer-sim-game-engine.rs` — make changes here, not in that submodule checkout.
>
> On disk: source clone `~/codes/ores/soccer-sim-game-engine.rs` · submodule checkout `~/codes/ores/k8s-cluster/remote/submodules/soccer-sim-game-engine.rs`.
<!-- END k8s-cluster-submodule-notice -->

# soccer_engine (soccer-sim-game-engine.rs)

Agnostic 2D soccer simulation + reinforcement-learning game engine. The soccer domain
(match engine, rules, agents, planner, rotation, learning) extracted out of
[`discrete-event-system.rs`](../discrete-event-system.rs) (`des_engine`), which supplies
the generic optimization (formation LP solved by Clarabel/IPM, with deterministic fallback)
and learning (neural MLP, policy-gradient, MDP/POMDP, PRNG, animation) primitives it builds on.

Transport-agnostic by design: construct a `SoccerMatch` / `SoccerRealtimeSession` and
drive it directly (a desktop game does this) — no HTTP is required. The web servers
(`dd-soccer-rs`, `dd-des-rs`) wrap it.

## Live demo server

```sh
cargo run --bin main_soccer_live      # serves http://127.0.0.1:5055/?game=<uuid>
```

Open `/?game=<uuid>` — the live 2D match UI reads the `?game=` id (minting + writing one
to the address bar on first visit) and scopes its `/api/*` calls to that game.

## Runtime flags (environment variables)

All boolean flags accept `1` / `true` / `yes` / `on` (case-insensitive); anything else,
or unset, is **off**.

### Performance switches — both default OFF

| Flag | Default | Effect |
|------|---------|--------|
| `SOCCER_LIVE_STEP_TELEMETRY` | off | Per-tick step-timing log lines (`# soccer-sim/live step telemetry …`) to stderr. Real I/O cost on a busy engine — leave off unless profiling. |
| `SOCCER_DECISION_TRACE_CAPTURE` | off | Capture every player's scored decision each tick for the live UI's **Pause & Analyze** tool / `GET /api/decision-trace`. Recording all 22 players per tick is real per-tick work — enable only when you want to analyze (with it off, Pause & Analyze is empty). |

> A/B at a fixed kickoff state: logging on ≈ 50 ms/tick vs off ≈ 43 ms/tick. Both off is
> the fast default; the remaining per-tick cost is the per-player decision (scales with
> how congested play is).

### Telemetry and structured logs

| Flag | Default | Effect |
|------|---------|--------|
| `SOCCER_TELEMETRY_ENABLED` | auto | Enables the shared tracing subscriber; cluster jobs set this explicitly. |
| `SOCCER_LOG_JSON` | off | Emit tracing events as newline-delimited JSON for OTel filelog collection. |
| `SOCCER_RUST_LOG` | `info,soccer_engine=info` | Soccer-specific tracing filter; falls back to `RUST_LOG`. |
| `SOCCER_CLUSTER_NAME` | local/unset | Cluster resource attribute for logs/traces. |
| `SOCCER_OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | `http://127.0.0.1:4318/v1/traces` | OTLP/HTTP trace endpoint when trace export is enabled. |

See `docs/observability.md` for the collector manifest and CockroachDB TTL schema.

### Live server

| Flag | Default | Effect |
|------|---------|--------|
| `HOST` / `PORT` | `0.0.0.0` / `8113` | Bind address for the `dd-soccer-rs` server (`main_soccer_live` binds 5055). |
| `SOCCER_LIVE_MAX_GAMES` | engine default | Cap on concurrent live games. |
| `SOCCER_LIVE_COACHING_FEEDBACK_PATH` | — | File path for persisted Pause-&-Analyze coaching notes. |

### Tactical / debug toggles (disable a discipline to A/B its effect)

| Flag | Effect when set |
|------|-----------------|
| `DD_SOCCER_DISABLE_ANTI_BUNCH` | Disable the anti-bunchball intent discipline. |
| `DD_SOCCER_DISABLE_SPACING_NUDGE` | Disable the territorial-spacing nudge. |
| `DD_SOCCER_DISABLE_DEFENSIVE_PUSHUP` | Disable the defensive line push-up. |
| `DD_SOCCER_DISABLE_FORMATION_STAGGER` | Disable formation staggering. |
| `SOCCER_FORMATION_LP_IPM` / `SOCCER_FORMATION_LP_INTERNAL_SIMPLEX` | Run the exact per-tick formation solve through fast Clarabel/IPM. **Default ON in optimized builds, OFF in debug**; if disabled, the tick uses a heuristic-anchor fallback. `SOCCER_FORMATION_LP_INTERNAL_SIMPLEX` is a legacy alias, not the solver choice. Use `SOCCER_FORMATION_LP_DETERMINISTIC=1` only for headless reproducibility; that routes to the slower deterministic internal simplex. |
| `LP_SOLVER` | Override the soccer-rotation LP-relaxation backend. When unset, the rotation policy tries local HiGHS dual simplex first and falls back to the internal simplex if the fast path is unavailable. |
| `MIP_LP_ALGO` / `MIP_ALLOW_EXTERNAL_SOLVERS` | Rotation-demo compatibility knobs from the generic DES layer. The soccer formation path does not use MILP/branch-and-bound; it uses continuous LP/IPM allocation plus per-player MPC execution. |
| `SOCCER_LP_DEBUG` / `SOCCER_SHOW_LP_BOUND` / `SOCCER_GOAL_DEBUG` | Extra LP / goal diagnostics. |

### Learning / Postgres / artifacts

| Flag | Effect |
|------|--------|
| `SOCCER_PARALLEL_GAMES` | Parallel games per learning run (1–100). |
| `SOCCER_PG_TLS_INSECURE` | Encrypt the Postgres wire but skip cert verification (RDS escape hatch). |
| `SOCCER_PG_CONNECT_MAX_ATTEMPTS` | Postgres connect retry budget. |
| `SOCCER_ARTIFACTS_PLAYBACK_ONLY` / `SOCCER_ARTIFACTS_WITH_LEARNING` | Artifact-generation modes. |
| `SOCCER_SITE_SEED` | Deterministic seed for site/artifact generation. |

## Tests

```sh
./shell cargo test --lib
```
## Build lifecycle

The repository shell routes build-producing Cargo commands through
`scripts/cargo-fresh`. When the soccer/DES source, toolchain, or Cargo build arguments change, it
uses package- and profile-aware `cargo clean` for only `soccer_engine` and its local `des_engine`
dependency before compiling the replacement. An identical build reuses the current cache, a debug
check cannot remove release executables, and a target profile backing a running executable is never
cleaned or marked current; it is reclaimed by the first matching build after that process exits.
Incremental compilation is disabled, preventing superseded `dep-graph.bin`,
`libsoccer_engine-*`, `des_engine*`, and related compiler generations from accumulating
indefinitely. Direct `cargo ...` remains available when bypassing this lifecycle is intentional.
