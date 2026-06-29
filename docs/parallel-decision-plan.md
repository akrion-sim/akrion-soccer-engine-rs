# Plan: parallel per-tick decisions (simultaneous-move) + serial resolve + soft-deadline fallback

## Why this is correct, not just fast
At `dt = 1/15 s` (66.7 ms) no agent action *completes* within a tick (a player advances a
fraction of a yard; the ball barely moves). The current loop applies each player's intent
*immediately*, so player N+1 decides against N's already-applied effect — modelling a sub-tick
reaction that doesn't physically exist. A **simultaneous-move** model (everyone decides from the
*tick-start* world, then all intents are applied together) is parallelizable *and* a more faithful
model at this granularity. "State must be agreed upon" is preserved by keeping the **resolve**
phase serial.

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
- The crux: split `run_time_step_with_context` / `_inner` so the player-metadata writes
  (`last_intent`, `last_decision`, `decision_commit_ticks`, learned `behavior_probability`, refractory
  state) become **fields of `DecisionOutput`** instead of `self.* =`. Audit `_inner` to prove it
  touches *no* positional/ball/other-player state (those stay in apply).
- `apply_post_decision_movement_discipline(&snapshot, &self, intent)` → confirm it reads only
  immutable tick-start state; if so it runs inside Phase B, else move it to Phase C.

```
struct DecisionOutput {
    player_id: usize,
    intent: PlayerIntent,
    last_intent: PlayerIntent,
    last_decision: AgentDecisionTrace,
    decision_commit_tick: u64,
    behavior_probability: Option<f64>,
    // human players: computed in Phase C instead (see below)
}
```

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
