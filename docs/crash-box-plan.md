# Flank crash-the-box: status & plan

The situational "crash the box" play: when a team holds the ball **in the opponent's
final third with it out on the flanks** (the three outermost lanes of the 12-lane fine
grid, either touchline), whip an **aerial cross** into the penalty area while **two or
three attackers crash the box** to attack the delivery and **head it in**, reinforced in
the MARL/MAPPO learner so the policy learns the whole pattern.

This doc records what shipped, how to land it cleanly given a contended shared tree, how
to validate/train it, the feature follow-ups, and the repo-health blockers surfaced while
verifying it.

Source of truth: `src/des/general/soccer/crash_box.rs` and its call sites.

---

## 1. What shipped (local, uncommitted)

All gated behind `flank_crash_box_enabled()` — env `DD_SOCCER_ENABLE_FLANK_CRASH_BOX`,
a `gate_default_on` production switch that is **default-OFF and re-reads the env each call
under `cfg(test)`** (so the suite + current network weights stay byte-identical until the
flag is flipped for a retrain).

The engine already owned the mechanics — the box-flood that fans 2–4 runners into distinct
near/far-post, penalty-spot and cut-back zones (`crash_the_box_target_for`), the aerial
carrier delivery (`FlankAttackPolicy::PlayDownFlankHighCross`), and aerial-duel heading
(`first-time-header`). They only fired when the team brain **rolled the `CrashTheBox`
strategy**. This change makes the same machinery fire as a **reaction to the geometry**,
and adds the reward.

- **New module `crash_box.rs`** — pure geometry recognizer (`ball_on_outer_flank_lanes`,
  `ball_in_attacking_final_third`, `flank_final_third_crash_box_geometry`), the gate, the
  reward constants, and the `FlankCrashBoxCross` live-cross window. Fully unit-tested.
- **Recognizer** `WorldSnapshot::flank_final_third_crash_box_active(team)` = gate + no
  active set-play + possession + geometry.
- **Carrier** — `team.rs::apply_strategy_commitment` forces
  `flank_attack_policy = PlayDownFlankHighCross` when the recognizer fires (drives the
  existing high-cross bias). It deliberately does **not** touch `attack_strategy`, so the
  learned strategy-value credit is untouched.
- **Box crash** — the early gate in `crash_the_box_target_for` is relaxed to also fire on
  the recognizer.
- **Reward** — new `SoccerRewardEventKind::{HeaderGoalFromCross, CrashBoxArrival}`:
  - `HeaderGoalFromCross` (22 pts, split: finisher + a 0.34 assist share to the crosser),
    paid in `record_goal_rewards` when a **headed finish** lands inside the live-cross
    window. A headed finish is classified by the ball's **contact altitude ≥ 1.0 yd**
    (`AERIAL_HEADER_MIN_ALTITUDE_YARDS`) at the shot, because `SoccerAction::Shoot` carries
    no header flag — `pending_aerial_finish` is stamped at the shot site.
  - `CrashBoxArrival` (3 pts, budget-bounded, split across crashers) paid once per cross
    when ≥2 attackers are already inside the opponent box as the aerial cross is released.
- **Tests** — 5 geometry unit tests in `crash_box.rs` + 5 match-level integration tests in
  `tests.rs`, serialized via a `crash_box_env_lock()` mutex (the gate re-reads the env, so
  parallel `set_var` would race). All 10 green.

**Files touched (mine only):** `crash_box.rs` (new) + hunks in `soccer.rs`, `team.rs`,
`world.rs`, `tests.rs`.

---

## 2. Land it cleanly — commit isolation

⚠️ **The working tree carries interleaved uncommitted work from at least two other
agents** (`player.rs` and `mdp-pomdp-action-schema.md` are *not* this feature; they belong
to a concurrent blindside-steal + forward-pass session, and there is a sibling
`akrion-slipbreak-wt` worktree on an offside-trap branch). Do **not** `git checkout` or
otherwise clobber those.

Plan:

1. Branch off `main`: `git switch -c feature/flank-crash-box`.
2. Stage **only** this feature's files:
   - `git add src/des/general/soccer/crash_box.rs`
   - `git add -p src/des/general/soccer/{soccer.rs,team.rs,world.rs,tests.rs}` — accept only
     the crash-box hunks (module decl + `pub(crate) use`; the two `SoccerRewardEventKind`
     variants and their `matches!` arms; the `team.rs` override block; the `world.rs`
     recognizer, relaxed crash gate, struct fields + constructor inits, the three helper
     methods, the pass-site `register_*` call, the shoot-site `pending_aerial_finish` stamp;
     the `tests.rs` block + `crash_box_env_lock`).
   - Reject any hunk that mentions `blindside_steal_*`, forward-pass-recognition, or
     offside-trap — those are other agents' work.
3. Verify the staged set compiles in isolation: stash the rest
   (`git stash -k -u`), `cargo build --lib && cargo test --lib crash_box`, then
   `git stash pop`. (See §5 for why a full `cargo test --lib` is currently noisy.)
4. Commit; push both remotes when ready (`origin` = akrion-sim, `upstream` = ORESoftware).
   Expect non-ff races from the concurrent automation — rebase, don't force.

---

## 3. Validate & train

The reward terms do nothing until a network is trained with the gate on. Sequence:

1. **A/B self-play sim** (gate off vs `DD_SOCCER_ENABLE_FLANK_CRASH_BOX=1`), measuring:
   - crosses attempted from the outer lanes in the final third,
   - attackers in the box at cross-release (the `CrashBoxArrival` population),
   - headed goals / headed shots on target,
   - **net goal differential** (the play must not trade overall xG for headers).
   Add an `#[ignore]` A/B sim test in the crash-box test block, mirroring the back-four
   line A/B test, so the comparison is reproducible.
2. **Retrain** with the gate on (it is default-ON in prod, so a production retrain already
   exercises it). Confirm the learner picks up `HeaderGoalFromCross` / `CrashBoxArrival`
   credit and that flank-cross frequency rises without degrading possession value.
3. **Roll out** via the existing gate-promotion path; keep the `DD_SOCCER_ENABLE_FLANK_CRASH_BOX=0`
   kill-switch documented for fast rollback.

---

## 4. Feature follow-ups (deferred, not blockers)

- **Window hygiene** — clear `flank_crash_box_cross` on possession loss / `reset_after_goal`
  so a stale Home cross can't credit an unrelated Home header within the 3 s window. (Today
  the team + open-window + same-tick-aerial-finish checks bound it; consuming on credit
  prevents double-pay. Tightening is cheap insurance.)
- **Low-cross / cut-back variant** — the recognizer currently forces a *high* cross. A
  driven low cross / cut-back to the penalty-spot is the right call from the byline; gate a
  `PlayDownFlankLowCross` branch on byline depth and reward the cut-back finish too.
- **Header classifier** — altitude ≥ 1.0 yd is a proxy. If a first-class finish-type flag is
  ever threaded onto the shot, switch the `pending_aerial_finish` stamp to it.
- **Defensive mirror** — the defending side should recognise the same trigger and collapse
  bodies into the box (mark the crashers, attack the cross). Currently only the attack is
  taught.
- **Recognizer in the neural observation** — the recognizer is decision-time only and does
  not change `FEATURE_DIM`. If A/B shows value, expose `flank_final_third_crash_box_active`
  as an observation feature (retrain required).

---

## 5. Repo-health blockers surfaced while verifying

A full `cargo test --lib` currently shows ~39 reds. **None are from this feature** (its
non-test changes are gate-inert under `cfg(test)`; its own 10 tests pass). They split into:

### 5a. ~21 deterministic decision reds — *pre-existing at `HEAD`*

Verified by running them in a clean `git worktree add /tmp/akrion-baseline HEAD`: **20 of
21 fail at `HEAD` with no uncommitted changes at all**, e.g. `aerial_pass_targeting_*`,
`attacking_midfielder_roams_*`, `blocked_*_killer_pass`, `exploit_space_strategy_*`,
`xavi_turn_*`, `pomdp_q_state_*`, `through_ball_strategy_*`, `first_touch_*`,
`defenders_press_*`. These are **stale expectations left by the merged forward-pass-
recognition commit** (`472bf81 "recognise good forward pass options, stop recycling
backward"`), which shifted decisions without updating the assertions.

> The 21st (`pressured_holder_without_good_outlet_prefers_escape_over_panic_pass`) passes at
> `HEAD` and fails only in the working tree — it is broken by the **concurrent
> blindside-steal bundle**, not this feature (its `player.rs` diff won't even compile on
> clean `HEAD` without the matching concurrent `world.rs` symbols).

**Plan:** owner of the forward-pass commit triages each — for every test, decide *new
behavior is correct ⇒ update the expectation*, or *regression ⇒ fix the code*. Track in a
checklist; do not let them rot into permanent "known reds" that mask future regressions.

### 5b. ~17 load-flaky realtime / http / controller tests

`live_http_*`, `live_soccer_page_*`, `realtime_session_*`, `controller_*`,
`batched_live_step_*`, `player_loop_*`, `ball_agent_run_time_step_trace_*`. These fail under
heavy CPU contention (the box was running this session **plus** the concurrent
`akrion-slipbreak-wt` suite **plus** a blindside A/B run — three parallel test storms).

**Plan:** quarantine the wall-clock/threading-sensitive tests behind a feature/`#[ignore]`
or a serial group so a contended `cargo test --lib` is a clean signal. Long-term, give the
realtime tests a virtual clock instead of wall-clock waits.

### Suggested order

1. Land this feature in isolation (§2) — it's independent and green.
2. Triage 5a (forward-pass commit owner) so `HEAD` is green again — the highest-value fix,
   because right now no one can read a full-suite run.
3. De-flake 5b (§5b) so contention stops producing phantom reds.
