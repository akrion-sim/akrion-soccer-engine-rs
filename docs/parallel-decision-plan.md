# Plan: parallel per-tick decisions (simultaneous-move) + serial resolve + soft-deadline fallback

## Why this is correct, not just fast
At `dt = 1/15 s` (66.7 ms) no agent action *completes* within a tick (a player advances a
fraction of a yard; the ball barely moves). The current loop applies each player's intent
*immediately*, so player N+1 decides against N's already-applied effect — modelling a sub-tick
reaction that doesn't physically exist. A **simultaneous-move** model (everyone decides from the
*tick-start* world, then all intents are applied together) is parallelizable *and* a more faithful
model at this granularity. "State must be agreed upon" is preserved by keeping the **resolve**
phase serial.

## PRIMARY lever: cadence-gated (staggered) decisions — decide less, not just faster
At dt = 1/30 (33 ms) a player does not need a fresh deliberative decision every tick — human
re-planning is ~100–250 ms. So **gate the expensive decide by a per-player cadence**: a player
recomputes at most once every `DECISION_INTERVAL_TICKS` (~3 ticks ≈ 100 ms), **staggered** across
players (phase by `player_id % interval`); on the in-between ticks it just **replays its held intent
and keeps moving/animating** (cheap). Only ~`22/3 ≈ 7` players decide per tick instead of 22 → the
dominant cost (~74% of the step) drops ~3×.

This reuses existing machinery — **no new replay logic**: `continuation_intent()` already returns the
sustainable held intent (`HoldShape`/`MoveTo`/`Dribble`/`ControlTouch`) or `None` for a spent
one-shot (pass/shot/tackle). Gate:
```
let due = now.saturating_sub(player.last_decision_tick) >= interval;       // + stagger phase
if !due { if let Some(held) = player.continuation_intent() { /* skip decide, apply held */ } }
// else: decide_player_on_snapshot(...); player.last_decision_tick = now;
```
A player whose held intent is a spent one-shot (`continuation_intent() == None`) decides regardless
(can't replay a played ball) — so carriers/finishers never get stale.

**MEASURED (implemented on `feature/parallel-decisions`, dt=1/30, headless live-http, 1500 ticks,
neural on):**

| cadence | tps | possession-decide ms | support-decide ms |
|---|---|---|---|
| OFF (every tick) | 43.5 | 13268 | 4129 |
| interval=3 (~100ms) | 76.2 (+75%) | 6320 (−52%) | 1922 (−53%) |
| interval=6 (~200ms) | 104.5 (+140%) | 3628 (−73%) | 1078 (−74%) |

**interval=6 is the sweet spot**: at dt=1/30 it gives a ~200ms real-time *decision* rate — the SAME
as the original 1/15 (refractory ~3–4 ticks × 67ms) — so behaviour should stay close while movement
runs at 30 fps (smoother). Headless 104 tps; applying the observed ~2.9× headless→live factor
projects **~36 tps live > the 30 tps real-time target → clears real-time SINGLE-THREADED.** So
**parallelism is now optional** (not needed to hit real-time). It still composes if wanted: cadence
picks *who* decides each tick (~7 players), the pool decides them *in parallel*, both via
`decide_player_on_snapshot`.

Implementation (landed): stateless staggered gate in the step loop —
`!cadence_due && continuation_intent().is_some()` ⇒ replay held intent + apply (keeps moving),
`continue`; else decide as before. Gate `SOCCER_DECISION_CADENCE_INTERVAL_TICKS` (default 3; `1`
disables). Human-controlled players never gated; spent one-shots decide regardless.

### Status (worktree `feature/parallel-decisions`, all compiling)
- **dt = 1/30** ✓; **cadence gate** ✓ (default interval 6); **`decide_player_on_snapshot` primitive** ✓.
- **#2 constant rescale ✓ (inference-critical, ×2):** `LIVE_MPC_DEFAULT/MAX_PLAYER_HORIZON` (12→24,
  45→90), `MPC_PASS_HORIZON_STEPS` (24→48), `DECISION_REFRACTORY_MIN_GAP/WINDOW_TICKS` (3→6, 7→14),
  `PLAYER_DECISION_COMMITMENT_SHORT/LONG_WINDOW_TICKS` (3→6, 7→14), `PERCEPTION_REFRESH_TICKS` (4→8),
  `SOCCER_FORMATION_LP_REFRESH_TICKS` (2→4), `ADVERSARIAL_EMBEDDING_SIGNAL_REFRESH_TICKS` (5→10),
  `LOOSE_BALL_TEAM_STALL_WINDOW_TICKS` (3→6). `secs_to_ticks`-derived constants auto-correct.
  **DEFERRED (learning-time, only matter for the 1/30 retrain, not frozen-policy play):** the
  per-skill `*_REWARD_WINDOW_TICKS` / `*_SAMPLE_INTERVAL_TICKS` (aerial/loose-ball/receive/shot/
  long-pass/spacing/support/line-depth), `TURNOVER_PENALTY_WINDOW_TICKS`, autosave intervals. The
  `*_MIN_TRAINING_STEPS` (=200) are sample counts, NOT time — leave.
- **#1 play-quality A/B (PRELIMINARY, fresh net via `measure_offense`, 2×5000 ticks):**
  structurally safe (full action repertoire, no freezing) but cadence-on **shifts play texture** —
  at interval=6 off-ball runs over-commit (`overlap-run` 8.1%→14.0%, +72% absolute player-ticks;
  `support-shape`/`run-in-behind` ↑) and on-ball re-decisions −20%. Expected from holding intents 6
  ticks. **NOT a green light:** (a) fresh net validates the *mechanism*, not trained quality — the
  real gate needs the gen-N policy (live run or `soccer_eval_gate_run` with a loaded artifact);
  (b) tune interval (3–4 likely shifts less while keeping most of the win).

### Remaining before merge
1. **Trained-policy eval** (the real gate): gen-N goals/xG/possession/pass-completion, original 1/15
   vs new 1/30+cadence. Needs the trained net.
2. **Interval tuning** (3–6) against that eval.
3. **Reward-window rescale + retrain at 1/30** for full adoption.
4. Optional: parallel pool over the ~7 due-deciders if more headroom wanted (not needed for real-time).

Cost of dt = 1/30 (do this to keep behaviour faithful): every TICK-based constant must scale ×2 —
the cadence/refractory itself (`DECISION_REFRACTORY_MIN_GAP_TICKS` 3→6, `WINDOW_TICKS` 7→14), MPC
horizons in steps (`LIVE_MPC_DEFAULT_PLAYER_HORIZON` 12→24, `MPC_PASS_HORIZON_STEPS` 24→48),
`SOCCER_FORMATION_LP_REFRESH_TICKS`, `SOCCER_*_REFRESH_TICKS`, etc. — seconds-based constants are
auto-correct. And the gen-N policy was trained at 1/15 → **revalidate / retrain at 1/30**.

## Do we need dt = 1/30 to parallelize? — No, but you want it for cadence/fidelity
Concern: the ball moves in 1/15 s, so simultaneous decisions could read a stale ball. But the
**free ball already does not move during the player loop today**: `run_ball_time_step` is scheduled
*last* (Ball schedule score = 0.0, below every player) and the ball-kinematics integration runs
*after* the loop. So nearly all players ALREADY decide on the tick-start ball — the simultaneous
model doesn't add free-ball staleness. The only sequential-vs-parallel difference is mid-loop
**possession-change** effects (`sync_held_ball_to_holder` / keep-out / shield) seen by the few
*late-scheduled* players — niche, and quantified by the A/B.

Conclusion: **parallelize at dt = 1/15 first.** dt = 1/30 is *perf-feasible* with parallelism
(serial floor ~20 ms/tick fits the 33 ms budget; decisions parallelize into the rest) but it's a
**separate, larger change**: every step-based constant (MPC horizons in *steps*, refractory/cadence
/refresh ticks) must be rescaled ×2 to preserve real-time behavior, and the trained policy needs
revalidation at the new dt. Treat 1/30 as an independent fidelity upgrade, pursued only if the A/B
shows possession-contest regressions or finer physics is wanted for its own sake.

## On Fisher-Yates while parallelizing
Today the live order is a deterministic ball-relevance *sort* (`agent_schedule_for_field_entities`);
`field_entities_use_fisher_yates` is inert (default off, only asserted in tests, no runtime shuffle).
Regardless: **simultaneity makes ordering irrelevant for *decide*** — every agent reads the identical
tick-start state, so no one has an ordering advantage (that's exactly the fairness FY approximated,
achieved for free). Ordering only matters for *resolve* (who wins a contest); keep that serial in
canonical order, and if randomized fairness is wanted there, run the FY shuffle in serial Phase A on
the main RNG to define the apply order. So: shuffle (optional) for resolve, parallel + order-free for
decide.

## Where the time goes (why the per-player layer is the right target)
| Layer | Share of step | |
|---|---|---|
| **Player decisions** | **~74%** | the 22 independent per-player calcs (`fieldPlayerDecisionMs` 39163/53081 ms) |
| Team (formation LP / central brain) | negligible | LP refreshes every 2 ticks (`SOCCER_FORMATION_LP_REFRESH_TICKS=2`); per-tick `CentralBrain` loop arm is empty `{}` |
| World (ball, officials) | negligible | `fieldBallMs` ≈ 11ms/2000 ticks |

So the bottleneck is also the part that scales with the 22 agents → Amdahl-favorable (~74%
parallelizable). Team/world stay in the cheap serial phases, untouched.

## Target tps
Live tick ≈ 77 ms ≈ **~57 ms decision (parallelizable) + ~20 ms serial** (snapshot, intent apply,
ball, learning, serialize). Across 8 worker threads: ~57/8 + overhead ≈ **~9 ms** → ~30 ms/tick →
**~33 tps**: ~2.2× margin under 1/15 (67 ms); meets 1/30 (33 ms) on the mean (tail covered by the
fallback). Hard ceiling if we did *nothing* but offload peripherals is the 57 ms decision itself
(~17 tps) — so parallelizing decide is the real lever; offloading peripherals is complementary.

## Current shape (what changes / what doesn't)
`world.rs` step loop (~7928–8091): serial `for scheduled in field_schedule` →
per player: build snapshot from *current* `self`; drain human input; neural infer;
`players[actor].run_time_step_with_context(&snapshot, …, &mut self.rng)` (mutates player
*metadata* + draws the **shared** RNG); `apply_post_decision_movement_discipline`; then
**`apply_player_intent` immediately** + `sync_held_ball_to_holder` / `enforce_restart_keepout_for`
/ `enforce_shield_body_barrier_for`.

Order-dependence sources: (1) shared `&mut self.rng`; (2) snapshot rebuilt from intra-loop-mutated
`self`; (3) immediate per-player apply. The plan removes (1) and (2) from *decide* and keeps the
*apply* phase byte-identical (same order, same effects).

## Three-phase tick

**Phase A — serial, tick start.** Build the ≤2 shared context snapshots ONCE from tick-start `self`
(`from_match_for_agent_decision`, `from_match_for_defensive_or_loose_agent_decision`); wrap in
`Arc` for sharing. Compute the schedule. Drain human inputs for controller slots (I/O on
`self.human_inputs` / `latched_human_inputs`) — unchanged, serial. Partition AI vs
human-controlled players (controller_slot.is_some()).

**Phase B — parallel, pure (AI players only).** On a dedicated pool:
`pool.install(|| ai.par_iter().map(compute_decision).collect())`.
`compute_decision` reads only immutable shared state (`&self` for the frozen net/config + the `Arc`
snapshot) and a **per-player RNG**, returning a `DecisionOutput`. No `&mut self`.
- Per-player RNG: `SeededRandom::new(mix(base_seed, tick, player_id))` — independent stream per
  player, deterministic, scheduling-independent. Replaces the shared serial `self.rng` draw.
- `learned_action_for_player_with_context(&snapshot, …)` — read-only net inference (must be `Sync`).
- **Crux is simpler than feared — compute on a player CLONE, no `_inner` surgery.** Confirmed:
  `PlayerAgent: Clone`; `run_time_step_with_context` mutates only the *player's own* metadata
  (`last_decision`/`last_intent`/`decision_commit_ticks`/`behavior_probability`) and needs only
  `(snapshot, rng)` — not `&SoccerMatch`; the two match-reads (`learned_action_for_player_with_context`,
  `apply_post_decision_movement_discipline`) are both `&self`/`&SoccerMatch` (read-only). So Phase B
  per player = clone the player, run the EXISTING `run_time_step_with_context` on the clone (mutating
  the clone) against the shared snapshot + per-player RNG, run the read-only discipline/learned-action
  against shared `&SoccerMatch`, and return the mutated clone + intent. Nothing on the real `self` is
  written in Phase B. Phase C swaps the clones back in (serial, canonical order). This means **zero
  refactor of the 130-line `_inner`** — only the loop is restructured.

```
struct DecisionOutput {
    player_id: usize,
    player_after: PlayerAgent,   // the decided clone (carries all metadata mutations)
    intent: PlayerIntent,
}
```

- Requires `SoccerMatch: Sync` for the shared `&self` read across threads — **audit the decide
  read-path for interior mutability** (`Cell`/`RefCell`/non-atomic caches), especially in
  `learned_action_for_player_with_context` (net inference). `self.rng` is *not* used in Phase B
  (per-player RNG instead), and gate `OnceLock`s are read-only post-init.

### Canonical-order invariant (the Fisher-Yates point)
Apply/resolve MUST follow the **scheduler's canonical order**, never the order parallel threads
finish in. Today that order is the deterministic ball-relevance sort in
`agent_schedule_for_field_entities` (score → kind → id). The `field_entities_use_fisher_yates`
flag is currently defined-but-inert (default off, only asserted in tests, no runtime shuffle); if
it's ever wired on, the FY shuffle runs in **serial Phase A on the main `self.rng`** and *defines*
the canonical order. Either way: Phase A fixes the order (and consumes the main RNG for any
shuffle); Phase B is order-free (each player decides from the same tick-start state with its own
RNG, so completion order is irrelevant); Phase C applies strictly in the Phase-A order. Per-player
decision RNG is independent of the schedule/shuffle RNG.

**Phase C — serial, canonical schedule order (resolve + agree on state).**
- Human-controlled players: decide serially via the existing path (input_frame + pending_human_control).
  Few/zero in the live demo.
- For each AI `DecisionOutput` in **schedule order**: write metadata back to `players[id]`; then run
  the EXISTING apply exactly as today — `apply_player_intent`, `sync_held_ball_to_holder`,
  `enforce_restart_keepout_for`, `enforce_shield_body_barrier_for`. Then the unchanged
  `resolve_dribble_hold_up_contests` / `resolve_player_collisions` / ball kinematics / spacing / etc.

Net: **apply/resolve is unchanged**; only the *inputs to decide* changed (tick-start snapshot +
per-player RNG). The behavior delta is localized and revalidatable.

## Soft-deadline fallback (the ≥99.99% mechanism, not the pool)
A thread pool gives throughput/margin, never a hard deadline (the join waits for the *slowest*
player; per-player work is data-dependent; the OS isn't real-time). Guarantee comes from a budget:
- `SOCCER_LIVE_DECISION_DEADLINE_MS` (default ≈ half the frame, ~30 ms). Cooperative check: each
  `compute_decision` checks remaining budget before its expensive scoring; if exceeded it returns
  the player's **last-tick decision** (the engine already has decision refractory/cadence, so a
  1-tick-stale decision is in-distribution and imperceptible).
- Converts "tail spike → missed frame" into "≤k players' decisions are 1 tick stale." That is the
  four-nines frame adherence. Cap per-player compute so one player can't blow the budget.
- Determinism: the fallback is timing-dependent → **disable it** (infinite deadline) in
  headless/determinism/test runs; only the live server uses a finite deadline.

## Thread pool (configurable)
Dedicated `rayon::ThreadPool` built once at server start (NOT the global pool), threads named
`soccer-decide-N`, sized by `SOCCER_LIVE_DECISION_THREADS` (**default 8, clamp [1, 24]**). On the
16-core box, 8 leaves headroom for the http workers, the neural backend thread, and the OS; 24 is
for larger hosts (beyond physical cores → diminishing returns + worse join tail). `threads = 1`
must reproduce the serial path exactly (same results as the refactor run serially).
`rayon` is not yet a dependency — add it.

## Complementary: offload peripherals off the step lock
Independent of the above, shrink the critical section so serialization/IO never blocks stepping:
- `/api/state` and the `/api/step` response: take the session lock only long enough to **clone a
  cheap frame snapshot**, release, then JSON-encode off-lock (encoding ~19 KB/frame currently
  competes with stepping on the same mutex).
- Decision-trace capture, policy autosave, moment/DB writes: ensure none run synchronously on the
  step path (push to a background thread / channel). The PG policy refresher is already background.
- The UI HTML is `include_str!` (static) — no server-side HTML rendering to move.

## Validation
1. **Refactor-equivalence (serial):** Phase-B pure `compute_decision` + Phase-C apply, run with
   `threads=1` and shared-RNG emulation, must be **byte-identical** to today for a fixed seed
   (proves the decide→apply split is faithful before any parallelism).
2. **Simultaneous-vs-sequential A/B (play quality):** headless self-play, old serial vs new
   simultaneous (threads=1, per-player RNG, no deadline), N matches — goals / pass completion /
   possession / xG must be statistically indistinguishable. This validates the *model* change.
3. **Perf + tail:** live p50/p99/p99.99 step-ms and tps, threads ∈ {1,8,16,24}.
4. **Deadline:** inject artificial per-player stalls; assert frame adherence holds via stale fallback.

## Risks / gotchas
- Simultaneous model regressing a *contested micro-interaction* (50/50 ball, shielding, keep-out):
  if A/B shows it, run just the small set of contesting players in a serial pass after the parallel
  one (hybrid) — keeps 90%+ of players parallel.
- Hidden interior mutability in the decide read-path (thread_local/`Cell`/per-call caches): audit.
  Gate `OnceLock`s are read-only post-init (fine). Perception noise must use the per-player RNG, not
  a shared cell.
- `Send + Sync` on the shared snapshot + net handle (snapshot is plain `Clone`/serde data — fine;
  confirm the net handle).
- Large, concurrently-edited `world.rs` — land on a dedicated branch/worktree, not the live tree.
- Gate everything behind `SOCCER_LIVE_DECISION_THREADS` (and `=1` == today) for safe rollout.

## Phasing
1. Decide→pure (`DecisionOutput`) + per-player RNG, **keep serial** (threads=1). Validate (1)+(2).
2. Add the dedicated pool + `par_iter` (threads=8). Validate (3).
3. Add soft-deadline + stale fallback. Validate (4).
4. (Parallel track) peripheral-offload / lock-shrink for `/api/state`.
5. Tune defaults, ship env-gated, default-on for live.
```
