# `fork_for_rollout()` — decision-time rollout spec for the soccer POMDP

Status: design spec (not yet implemented). Author: climb effort, 2026-07-09.
Companion to `akrion-climb-state` memory. Work on `main` per AGENTS.md.

---

## 0. Framing: we do NOT need MCTS (answering "is MCTS important for NNs to solve POMDPs?")

**No. MCTS is neither necessary for neural nets, nor for POMDPs, nor for this climb.**

- **Neural nets solve POMDPs without any search.** The canonical recipe is a *recurrent/belief
  policy* (RNN hidden state = belief over history) trained by policy gradient / actor-critic. That
  is exactly what our own `extlib-pomdp-solver` and `burn-pomdp-solver` already are (GRU belief cell
  → actor + critic). They contain **zero** MCTS. DQN, PPO, SAC, R2D2, recurrent-PPO — none use search.
- **MCTS is a planning method for a specific regime:** you have a (known or learned) model, the
  problem is (near) fully observable, and the branching factor is huge — chess/Go (AlphaZero). In a
  *POMDP* MCTS is actively awkward: search has to run in belief space (POMCP / particle-filter trees),
  which is finicky and rarely worth it.
- **The policy-improvement guarantee we care about belongs to ROLLOUT, not to MCTS's tree.** Rollout
  is the *minimal* form of decision-time planning: for each candidate action `a`, simulate the base
  policy forward from the current state, average the returns, pick `argmax_a Q̂(s,a)`. That flat,
  one-ply, embarrassingly-parallel procedure already gives **V^π' ≥ V^π** (Bertsekas rollout /
  Tesauro). MCTS is just a way to spend more compute building an asymmetric tree on top of the same
  idea — it is *not* what confers the guarantee.
- **The engine's `neural_mcts` is a misnomer** — the trace showed it is a PUCT-weighted *bandit
  re-ranker* over a statically pre-scored, pre-gated shortlist (`value_estimate` is fixed at node
  construction; no `run_time_step`, no dynamics in the loop; `world.rs:14983-15010`). It is not tree
  search and not on this critical path.

**Conclusion:** drop "MCTS" from the vocabulary. Build **flat rollout** — the simplest search that
carries the improvement guarantee — using the DES engine itself as the model. This spec is the
enabling primitive for that: a cheap, faithful *fork* of the match state you can step forward and
throw away.

(Caveat kept honest: search of *any* kind is leverage, not necessity. You could also beat parity by
fixing candidate *exposure* — the diagnosed wall — and letting model-free RL learn. Rollout is
attractive because (a) it has the guarantee and (b) it uniquely exploits the perfect simulator we
already own. But rollout inherits the same exposure requirement: it must roll out over the
**un-gated** action set, or it is starved exactly like the net.)

---

## 1. The three primitives rollout needs

1. **`fork_for_rollout(&self, seed) -> SoccerMatch`** — clone the physics/decision state, null the
   training/telemetry baggage, reseed the RNG. (§2, §3)
2. **Action injection** — force the ball-carrier's *tick-0* action to the candidate, then let the
   base policy drive the remaining horizon. (§4)
3. **A leaf value** — evaluate the ball state at horizon end (EPV grid lookup, or terminal
   goal/turnover). The EPV grid is scoped separately (`pitch_value.rs:144`, `fit_epv_grid.py`); this
   spec assumes it as a black-box `leaf_value(&SoccerMatch) -> f64`. (§6)

Rollout loop (pseudocode):

```
fn rollout_q(base: &SoccerMatch, carrier: usize, cand: Candidate,
             horizon_ticks: u32, seeds: &[u64]) -> f64 {
    let mut acc = 0.0;
    for &s in seeds {                                  // Common Random Numbers (§5)
        let mut w = base.fork_for_rollout(s);
        w.rollout_forced_action = Some((carrier, cand.label, cand.target));  // §4
        for _ in 0..horizon_ticks { w.run_time_step(); if w.ball_dead_or_scored() { break } }
        acc += leaf_value(&w);
    }
    acc / seeds.len() as f64
}
// pick: argmax over candidates of rollout_q(...). Guarantee: ≥ base policy, in expectation.
```

---

## 2. `fork_for_rollout()` — field-level KEEP / NULL / RESEED

`SoccerMatch` is `world.rs:1109-1611`, 78 owned fields. **Every heavy field is training/telemetry;
none is physics.** That is what makes this cheap.

### KEEP — deep-clone (the physics + within-play decision state)

Core: `config`, `tick`, `clock_seconds`, `players`, `officials`, `ball`, `shared_positions`,
`score_home`, `score_away`.

In-flight ball physics (⚠️ **do not confuse with the `pending_<head>Decision` training buffers below** —
these carry an airborne pass/shot that MUST continue): `pending_pass`, `pending_shot`,
`pending_rebound`, `pending_aerial_finish`, `flank_crash_box_cross`, `suppress_generic_oob_turnover`.

Team shape / play bias: `central_brain`, `active_set_play`, `home_genome`, `away_genome`,
`home_mpc_latent_objective`, `away_mpc_latent_objective`, `neural_blend` (⚠️ see §6 — set `Off` for a
pure-analytic base rollout).

Within-play continuity guards (all small maps/scalars; several are empty under default config):
`gk_handling_since_clock`, `goal_celebration_remaining_ticks`, `goal_celebration_kickoff_team`,
`last_touch_player`, `ball_stationary_ticks`, `ball_stuck_anchor`, `restart_double_touch_guard`,
`goal_kick_must_clear_box`, `recent_dispossession`, `last_controlled_possession_team`,
`possession_chain`, `possession_chain_previous_touch_team`, `back_four_line_latch`,
`back_four_anchor_pivots`, `defensive_delay_clocks`, `defensive_beat_clocks`, `offside_clocks`,
`teammate_proximity_seconds`, `teammate_spacing_notices`, `teammate_spacing_near_clocks`,
`teammate_spacing_far_clocks`, `teammate_spacing_yield`, `defensive_clear_hold_trackers`,
`loose_ball_uncontested_since_tick`, `player_tick_carryover`, `opponent_press_belief`,
`coach_set_play_hints`, `coach_position_hints`, `possession_progress_tracker`,
`forward_carry_tracker`.

Learned predictor heads — **KEEP as `Arc::clone` (refcount bump, ~free)**: all ~22
`Option<Arc<…Head>>` (`pass_completion_head`, `mpc_objective_head`, `line_depth_head`,
`defender_line_head`, `loose_ball_commit_head`, `attack_spacing_head`, `support_scorer_head`,
`receive_approach_head`, `lane_affinity_head`, `goal_side_recovery_head`, `winger_pinch_head`,
`separation_floor_head`, `pass_lane_yield_head`, `head_scan_head`, `crash_box_head`,
`run_prediction_head`, `slip_break_head`, `onside_support_head`, `long_pass_run_head`,
`give_and_go_head`, `aerial_reception_head`, `shot_trigger_head`). They must ride along so the
rollout's dynamics match whatever policy is live; the `Arc` makes it free.

Human control (empty in headless self-play — clone is trivial): `human_inputs`,
`latched_human_inputs`, `pending_human_control`.

`mpc_player_controllers` (the warm-start MPC cache) — **KEEP-clone if `PlanarPointMassMpc: Clone`;
else NULL.** It is small (lazily created for the *active subset* — ball carrier + near-ball
contesters, not all 22). Keeping it gives tick-0 trajectory fidelity + fast convergence; nulling it
costs only a few extra solver iterations on tick 0, not correctness. (Verify Clone; the grep didn't
locate the derive — it's `crate::des::general::mpc_point_mass::PlanarPointMassMpc`.)

### RESEED

`rng: SeededRandom::new(rollout_seed)` — forks the stochastic stream so the main sim is untouched and
candidates can share Common Random Numbers (§5). `SeededRandom: Clone` (mulberry32,
`discrete-event-system.rs/.../capabilities.rs:49`), but we **reseed rather than clone** on purpose.

### NULL / empty (the training + telemetry baggage — the entire clone-cost win)

Learners & actors (the biggest fields; not needed to *play* the analytic base — see §6 if rolling out
the neural actor): `neural_learner`, `away_neural_learner`, `policy_head`, `skill_policy_heads`,
`keeper_policy_head`, `world_model`.

All ~22 RL sample corpora → `Vec::new()`: `learning_transitions`, `episode_learning_transitions`,
`pass_outcome_samples`, `mpc_objective_samples`, `line_depth_samples`, `defender_line_samples`,
`loose_ball_commit_samples`, `receive_approach_samples`, `lane_affinity_samples`,
`goal_side_recovery_samples`, `winger_pinch_samples`, `separation_floor_samples`,
`pass_lane_yield_samples`, `head_scan_samples`, `crash_box_samples`, `run_prediction_samples`,
`slip_break_samples`, `onside_support_samples`, `long_pass_run_samples`, `give_and_go_samples`,
`aerial_reception_samples`, `shot_trigger_samples`, `attack_spacing_samples`, `support_move_samples`.

All `pending_<head>Decision` open-decision buffers (⚠️ NOT `pending_pass/shot/rebound`) → empty:
`pending_line_depth`, `pending_defender_line`, `pending_loose_ball_commit`,
`pending_receive_approach`, `pending_lane_affinity`, `pending_goal_side_recovery`,
`pending_winger_pinch`, `pending_separation_floor`, `pending_pass_lane_yield`, `pending_head_scan`,
`pending_crash_box`, `pending_run_prediction`, `pending_slip_break`, `pending_onside_support`,
`pending_long_pass_run`, `pending_give_and_go`, `pending_aerial_reception`, `pending_shot_trigger`,
`pending_attack_spacing`, `pending_support_decisions`.

Reward/replay bookkeeping → default/empty: `deferred_reward_credits`,
`deferred_reward_transitions`, `recent_learning_history`, `full_game_learning_applied`,
`full_game_learning_replay_transitions`, `turnover_penalty_history`, `last_turnover_penalty_tick`,
`pending_turnover_outcome`, `wasted_energy_history`, `player_last_ball_interaction_tick`,
`team_last_positive_outcome_tick`, `completed_pass_chain`, `pass_chain_meter`,
`dribble_mpc_objective`, `deferred_reward_credits`.

Retrieval + telemetry → default/empty: `episode_config_captures`, `retrieval_action_prior`,
`retrieved_action_prior_cache`, `adversarial_moment_memory`, `adversarial_moment_index_cache`,
`adversarial_embedding_signal_cache`, `events`, `stats` (`MatchStats::default()`),
`tactical_summary`, `step_timing_stats`, `mpc_reconcile_stats`, `controller_yield_stats`,
`last_agent_schedule`, `decision_trace_by_player`, `decision_trace_capture_enabled`,
`possession_swaps`, `specialist_curriculum_round`, `coach_set_play_hints` (if unused).

Learned tabular policy: `learned_policy`, `team_policies` — KEEP if the base policy is the tabular
brain; NULL for pure analytic. (Cheap either way.)

> Net clone cost after nulling: **22 `PlayerAgent` + ball + officials + `CentralBrain` + `config` +
> `mpc_player_controllers` (≈3–6 entries) + a handful of small maps.** No neural weights, no sample
> corpora, no replay deques. This is the "moderate → cheap" transformation.

---

## 3. Implementation route: hand-written constructor, NOT `#[derive(Clone)]`

Do **not** add `#[derive(Clone)]` to `SoccerMatch`:

- It requires *all* 78 fields be `Clone` — including `SoccerNeuralLearner` / `SoccerWorldModel`,
  which may not be, and would fail to compile.
- Even if it compiled, it would deep-clone the heavy learners we intend to throw away — the exact
  waste we're avoiding.

Instead, a hand-written `fork_for_rollout` that constructs a fresh `SoccerMatch` explicitly, cloning
the KEEP fields and using `None` / `Vec::new()` / `Default::default()` for the NULL fields. This
sidesteps the "is the learner `Clone`" question entirely (we never clone it) and is maximally cheap.
`SoccerMatch` has no `Default` (78 fields), so every field must be listed — tedious but mechanical
and one-time. A `debug_assert!` that the fork's ball/player positions equal the source's guards
against a mis-wired field.

```rust
impl SoccerMatch {
    /// Physics/decision-faithful shallow copy for decision-time rollout.
    /// Clones only what `run_time_step` reads to advance play; nulls all learners,
    /// sample corpora, replay buffers, and telemetry. RNG is reseeded (main sim untouched).
    pub fn fork_for_rollout(&self, rollout_seed: u64) -> SoccerMatch {
        SoccerMatch {
            config: self.config.clone(),
            tick: self.tick, clock_seconds: self.clock_seconds,
            players: self.players.clone(),
            officials: self.officials.clone(),
            ball: self.ball.clone(),
            central_brain: self.central_brain.clone(),
            shared_positions: self.shared_positions.clone(),
            score_home: self.score_home, score_away: self.score_away,
            pending_pass: self.pending_pass.clone(),
            pending_shot: self.pending_shot.clone(),
            pending_rebound: self.pending_rebound.clone(),
            // … all remaining KEEP fields (deep clone) …
            // … all Arc<…Head> fields: self.x_head.clone()  (refcount bump) …
            rng: SeededRandom::new(rollout_seed),                 // RESEED

            neural_learner: None, away_neural_learner: None, world_model: None,
            policy_head: None, skill_policy_heads: None, keeper_policy_head: None,
            pass_outcome_samples: Vec::new(),
            // … all *_samples: Vec::new(), all pending_<head>: Vec::new() …
            stats: MatchStats::default(),
            step_timing_stats: Default::default(),
            rollout_forced_action: None,                          // new field, §4
            // … all remaining NULL fields …
        }
    }
}
```

Prototype shortcut (for the §7 cost number only): if `#[derive(Clone)]` happens to compile,
`let mut w = self.clone(); w.clear_training_state(); w.rng = SeededRandom::new(seed);` gets a number
fast, then swap in the hand-written constructor for the real thing.

---

## 4. Action injection hook

`run_time_step(&mut self)` takes no action argument (`world.rs:20258`). Add one field consumed on the
first tick:

```rust
/// Rollout-only: forces this player's action on the NEXT run_time_step, then self-clears.
/// `None` in all normal play ⇒ byte-identical. Consumed in the player-decision path.
pub(crate) rollout_forced_action: Option<(usize /*player*/, SoccerActionLabel, Option<Vec2> /*target*/)>,
```

Consume it where the per-player learned plan is applied. The player layer *already* accepts an
injected plan — `PlayerAgent::run_time_step(&snapshot, human_input, learned_plan, rng)`
(`player.rs:8455`) — so the injection point exists; only the match-level plumbing is missing. On tick 0,
translate `rollout_forced_action` into that player's `learned_plan` (or directly stage
`pending_pass`/`pending_shot` for a pass/shot candidate), then set it to `None` so ticks 1..H run
under the base policy. This is what makes it *rollout* (evaluate one forced action, then follow the
base policy) rather than exhaustive search.

Candidates must come from the **un-gated** legal set (bypass the pass-like margin gate — invert the
guard at `world.rs:14883`, and/or raise `SOCCER_NEURAL_MCTS_PASS_TARGET_CANDIDATES` to 11), or the
rollout is starved of forward passes exactly like the net.

---

## 5. Variance reduction: Common Random Numbers (CRN)

Rollout value is stochastic. To compare candidates fairly, evaluate **every candidate against the
same set of M seeds** `{s_1..s_M}`. Then `Q̂(cand) = mean_m value(fork(s_m)+cand)`, and the *difference*
`Q̂(a) − Q̂(b)` has far lower variance because both saw identical noise draws (same coin flips for
tackles, deflections, etc.). This is the single most important cheap trick — it lets small M (4–8)
discriminate candidates that would otherwise need M in the hundreds. `fork_for_rollout(seed)` reseeds
precisely to enable this.

---

## 6. Fidelity caveats (where the guarantee frays)

The improvement guarantee **V^π' ≥ V^π** assumes the rollout simulates (a) the *same base policy* π
you're comparing against and (b) the *same dynamics* as live play. Every approximation in the fork
weakens it toward "improvement w.r.t. the rollout's approximate model":

- **Base-policy identity.** To claim improvement *over the analytic engine*, roll out the analytic
  base: set `neural_blend = Off` and NULL `policy_head`/learners so the forked ticks run pure
  analytic. To roll out the *current neural* policy instead, you must keep `neural_blend` + the actor
  heads (heads are `Arc`, cheap; `policy_head` is not `Arc` — clone it). Pick one deliberately; don't
  mix (rolling out policy A to improve policy B has no guarantee).
- **MPC warm-start.** Nulling `mpc_player_controllers` perturbs tick-0 trajectories slightly. Keep it
  cloned for the fidelity run.
- **LP re-solve cost vs. fidelity (see §7).** Freezing/subsampling the `central_brain` formation LP
  in the rollout is the biggest speed lever but the biggest fidelity risk. Start faithful; degrade
  only if the cost number forces it, and re-measure the guarantee empirically (rollout policy should
  beat its own base in head-to-head — verify, don't assume).

---

## 7. Cost model + the GATING measurement (do this first)

Per on-ball decision: `C` candidates × `M` seeds × `H` ticks `run_time_step` calls.
At 15 Hz, `H = horizon_seconds × 15` (2 s → 30 ticks; 3 s → 45 ticks). Example `C=8, M=4, H=30` →
**960 `run_time_step` calls per decision.**

The unknown is the per-tick wall-clock, dominated by the `central_brain` LP (Clarabel) + per-player
MPC solves. **Measure it before building the full rollout** — if 960× per decision is too slow for the
target regime, we pivot to offline-only (training/eval) or degrade fidelity.

Measurement plan (cheap, no rollout needed):
1. There is already a per-step timer: `step_timing_stats` / `SoccerStepTimingStats` (accessor
   `world.rs:19219`). Run an existing headless bin (`measure_offense.rs`, `main_soccer.rs`) and read
   mean per-tick µs, split by phase (LP vs MPC vs decision) if the struct breaks it down.
2. Back-of-envelope: time a full headless match (≈ 90 min × 15 Hz ≈ 81k ticks) wall-clock; per-tick ≈
   total / 81k. If per-tick ≈ 100 µs → 960× ≈ **~100 ms/decision** — fine offline and for live
   *on-ball-only* (one agent, a few decisions/sec), too slow for all-22 live.

Speed levers if needed (each trades fidelity, §6): freeze the formation LP for the horizon (re-solve
once, not per tick); cap MPC iterations in the fork; shrink `H` and lean on the EPV leaf (shallow
search + strong leaf = the whole point); reduce `M` (CRN already minimizes it); parallelize candidates
(rollouts are independent — `rayon`).

**Decision gate:** if per-decision cost ≤ ~a few ms → viable live on-ball. If ~10–100 ms → offline
eval/training-distillation only (still valuable: rollout as an *expert* to distill into the cheap net,
AlphaZero-style). If ≫ 100 ms even after LP-freeze → reconsider.

---

## 8. Build sequence & effort

1. **[gate, ~0 code] Measure per-tick cost** via `step_timing_stats` on an existing bin. Decides
   live-vs-offline before any build. (§7)
2. **[MODERATE] `fork_for_rollout` + `clear_training_state`** hand-written constructor; `debug_assert`
   position-equality vs source. One-time tedious field listing. (§2, §3)
3. **[SMALL] `rollout_forced_action` field + tick-0 consumption** in the player-decision path;
   default `None` ⇒ byte-identical. (§4)
4. **[SMALL] Un-gate the candidate set** for rollout (invert `world.rs:14883` guard for the rollout
   path; env cap to 11). (§4)
5. **[depends on EPV] `leaf_value`** = EPV grid lookup on ball state (separate workstream:
   `pitch_value.rs:144`, `fit_epv_grid.py`). Until then, a crude analytic leaf (shot-prob / territory)
   unblocks a first end-to-end test.
6. **[SMALL] Rollout driver + CRN** loop (§1, §5), on-ball carrier only.
7. **[eval] Head-to-head** rollout-policy vs analytic on the locked FP-completion discriminator
   (`akrion-climb-state`). The guarantee predicts ≥ parity — verify empirically; if it underperforms,
   the fork fidelity (§6) or the leaf value is the suspect, not the method.

Prereq facts (verified 2026-07-09): sim = 15 Hz (`DEFAULT_DT_SECONDS = 1/15`, `soccer.rs:131`);
`PlayerAgent`/`BallAgent`/`OfficialAgent`/`CentralBrain`/`MatchConfig`/`SharedPlayerPositions`/
`SeededRandom` all `#[derive(Clone)]`; `run_time_step` self-contained (no external RNG/queue/sink);
RNG seeded + in-struct (fork does not perturb main stream); analytic per-player decision already
runs on an isolated `&WorldSnapshot` in tests (`tests.rs:13470`).
