# Learnability conversion roadmap

A grounded audit of hardcoded decision surfaces that can be converted to learning,
with the **exact live call sites** and a scoped conversion for each, modeled on
the back-four line model (`back_four_line.rs` + `docs/back-four-line-model.md`):
a feature struct → normalized vector, a `FeedForwardNetwork` head, a gated seam
that is byte-identical when off, a deterministic analytic seed, and a named
training signal.

Call sites should be reverified against `main` before implementation. This file avoids frozen
line numbers because the soccer module moves quickly.

Legend — **State**: `DORMANT` (net/seam built, not consumed live) · `HARDCODED`
(closed-form, no seam) · `PARTIAL` (gated seam landed, some live use, learning still incomplete).

---

## Priority 1 — On-ball action policy (the actual decision)

**State: PARTIAL (highest ROI).** This is *what to do* — pass/shoot/dribble/carry
and their parameters.

- **Live call site:** `WorldSnapshot::learned_action_for_player_with_context`
  feeds a `learned_plan` into `players[actor].run_time_step_with_context(...)`; the plan is produced
  by `exploration_action_for_player` -> `neural_blended_action` ->
  `best_action_for_state_observation_with_prior` -> `mpc_reconciled_learned_plan_for_policy`.
- **The gate:** `learned_policy_inference_enabled()` =
  `config.learning_enabled || team_policies.is_some() || learned_policy.is_some()`. **Off in a vanilla
  sim** — the heuristic action ordering runs unless a policy is loaded or learning
  is on.
- **What's implemented:** `SOCCER_POLICY_ACTIONS` has `113` append-only labels, including
  `40` kick-power bucket labels. `learned_discretized_kick_candidate_label` and
  MCTS expansion can offer pass/aerial/shot bucket variants behind
  `DD_SOCCER_ENABLE_DISCRETIZED_KICK`, and execution/scoring can preserve the selected bucket
  power.
- **What's still hardcoded/incomplete:** full direction, curve, elevation, and bounded aim heads
  are not live learned heads. Bucket selection also lacks explicit entropy/softmax exploration
  telemetry, and ConfidenceGated still depends on visit counts for the fine candidate key.

**Conversion (mostly wiring, low build):**
1. Finish the factored kick path: keep the landed speed-bucket labels, then add gated learned
   direction/curve/elevation/aim heads only after bucket selection has entropy and held-out proof.
2. Ensure `learned_policy_inference_enabled()` is true in the deployed
   learner/live-server (load the active gen — the live server already can, see
   [[soccer-live-server-pg-bridge-and-policy-bloat]]).
3. Strengthen `neural_blend` readiness so a trained head actually steers (today it
   blends toward heuristic at low readiness).
- **Training signal:** already present — actor-critic + GAE on `SoccerPolicyHead`,
  PPO-clip ([[soccer-real-ppo-multi-epoch]]); reward = existing local events +
  pitch-value×xT (Priority 2).
- **Parity:** gated by the existing inference flag; off = today's heuristic.

---

## Priority 2 — Learned xT / pitch-value surface (the reward everything trains on)

**State: HARDCODED seed.** Converting this lifts *every* positioning/▶action net
at once, because it is (or should be) their reward and critic basis.

- **Live call sites:** `expected_threat` / `team_expected_threat` /
  `territorial_advantage` ([pitch_value.rs](../src/des/general/soccer/pitch_value.rs))
  have **no live consumers** today — they feed only `pitch_value_reward_delta`,
  which is **gated off** (`DD_SOCCER_ENABLE_PITCH_VALUE_REWARD`). `pitch_control_home`
  is a closed-form time-to-arrive logistic in `pitch_value.rs`.
- **What's hardcoded:** `expected_threat` is a closed-form forward^3 × shooting-zone
  ramp; the module doc itself calls it *"a seed the learners may later replace with
  a learned Markov xT grid."*

**Conversion:**
1. Learn a **Markov xT grid** from self-play possession chains (value of having the
   ball in each cell = P(score before turnover) via the empirical transition
   matrix) — replace `expected_threat` with a grid lookup, keeping the closed-form
   as the cold-start prior.
2. Optionally learn `pitch_control_home` from *who actually won the next loose
   ball* given the arrival-race features.
- **Training signal:** possession-chain outcomes (goal / shot / turnover) already
  flowing through the learner; no new instrumentation needed for the grid.
- **Parity:** swap is behind the existing pitch-value gate; the grid defaults to
  the analytic prior until enough chains accumulate.

---

## Priority 3 — Midfield line + press line depth (replicate the back four)

**State: HARDCODED, but the seam already exists.** Cheap, parity-safe wins —
literally the back-four pattern applied to the next two lines.

- **Live call sites:** the midfield line **already has a band function**,
  `midfield_line_band_adjusted_target` /
  `midfield_line_band_adjusted_intent` — the exact analogue
  of `defender_line_band_average_adjusted_y`. The press line is
  `defensive_midfielder_press_focus`, consumed in `defensive_shape_for`.
- **What's hardcoded:** `defensive_shape_for` is a wall of constants —
  `role_line_bias` (0.70/0.48/0.28/0.10), `width_factor` clamps (0.56–0.92), per-role
  press distances (6.0/3.8/8.5), and lateral-pull coefficients.

**Conversion:** generalize `back_four_line.rs` into a `LineDepthModel` reused for
the midfield band (and the press trigger line), with the same
`BackFourLineInputs`-style feature vector + head + analytic seed + blend +
band-clamp. The midfield band fn is the drop-in seam; press focus is a second
small head.
- **Training signal:** same pitch-value×xT territorial reward as the back four.
- **Parity:** one gate per line, off = identical (the back-four template guarantees
  this by construction).

---

## Tier 2 — Hardcoded probability / value models (clean swaps, sharpen the sim)

| # | Surface | Live call site | Hardcoded as | Conversion / training signal |
|---|---------|----------------|--------------|------------------------------|
| 4 | **Pass completion** (~~DORMANT~~ **DONE**) | consumed live in `pass_target_quality_for_snapshot` behind `DD_SOCCER_ENABLE_LEARNED_PASS_COMPLETION`, blended with the analytic estimate once the head clears `PASS_COMPLETION_HEAD_MIN_TRAINING_STEPS` | ~~head trained but not consumed~~ — **wired**; off ⇒ analytic estimate stands alone (parity) | **DONE** — `SoccerPassCompletionHead::predict` is in the ranker; gate it on in the trained learner |
| 5 | **Pass velocity/power** (PARTIAL) | `dd_soccer_disable_learnable_pass_velocity()` plus kick-power bucket labels/candidates | gated seam **landed on main**, but the chosen speed still needs explicit bucket-entropy proof or a learned head | feed the power buckets to a small head trained on completion×xT-gain at speed if label-level bucket learning is not enough |
| 6 | **GK save probability** (HARDCODED) | `goalkeeper_save_probability_from_traits` | closed-form trait/physics formula | learned save model on (shot geometry, pace, keeper traits) → saved?; label is the shot outcome already logged |
| 7 | **Tackle / duel success** (HARDCODED) | `tackle_success_probability` / `slide_tackle_success_probability` / `slide_tackle_foul_probability` | skill-profile formula × `SLIDE_TACKLE_SUCCESS_BOOST` clamp | learned duel model on (closing speed, angle, from-behind, skills) → won/lost/foul; label is the duel outcome |

---

## Tier 3 — Other positioning surfaces (back-four pattern again)

| # | Surface | Live call site | Note |
|---|---------|----------------|------|
| 8 | **Whole `defensive_shape_for`** | `defensive_shape_for`, consumed by player defensive-shape paths | the fore-aft axis is Priority 3; the **lateral** width/compactness constants are a second learnable axis |
| 9 | **GK positioning** | `goalkeeper_ball_goal_tracking_target`, `goalkeeper_buildup_lane_target_for` | sweep heuristic; learn the start position from shots-faced / chances-conceded |
| 10 | **Off-ball support runs** | `open_space_for` | heuristic-scored run targets; learn from pitch-value×xT gain of the run |

---

## Learning-signal & evaluation infrastructure (landed)

These are not action-surface conversions — they raise the *rate* and *trustworthiness*
of learning across every surface above.

### Terminal won-game reward — the "long" rung of the quasi-win ladder (gated, default-OFF)

The short-horizon quasi-wins were already priced — a 2+ forward-pass combo
(`PASS_CHAIN_TWO_FORWARD_EVENT_REWARD_POINTS` 7.5 / three-net-forward 10.0), a
defender beaten on the dribble (`DRIBBLE_BEAT_REWARD_POINTS` 6.0 / `NUTMEG_BEAT` 7.0,
×1.5 if the defender committed), a shot off/on target (10/40), a goal (100). The one
rung missing from the *learning* signal was the final result: `soccer_full_game_replay_transitions`
built its return purely from per-tick SHAPED rewards, so a side that farmed shaping
but **lost** still trained on a high return. At dt = 1/15s a match is thousands of
ticks, so a terminal reward funnelled through the existing 0.995/tick discount decays
to ~0 within ~40s — it could only credit the dying minutes.

So the result is broadcast as a flat **Monte-Carlo outcome label** `z` (AlphaZero-style)
added to *every* transition of a team: `+win` / `draw` / `−win` with a capped per-goal
margin bonus (`MATCH_OUTCOME_*` consts). The critic learns `E[outcome | state]`, so the
advantage `return − V(s)` credits whether THIS game beat the pre-result expectation —
the constant is not absorbed because it differs by the realised result, which the state
alone cannot predict.
- **Seam:** `soccer_full_game_replay_transitions(transitions, match_outcome: Option<MatchOutcomeReward>)`
  ([soccer.rs](../src/des/general/soccer.rs)); the caller `apply_full_game_learning_if_ready`
  passes `Some(..)` from the final score only when the gate is on.
- **Gate:** `DD_SOCCER_ENABLE_MATCH_OUTCOME_REWARD` (`match_outcome_reward_enabled()`),
  default-OFF ⇒ `None` ⇒ byte-identical replay.
- **Magnitudes** (`WIN=200.0`, `PER_GOAL_MARGIN=15.0`, `MARGIN_CAP=5`) are a starting point
  and **must be A/B'd through the promotion eval gate below**, never tuned on raw reward.

### Promotion eval gate — held-out Elo / cross-play / exploitability (pure, always-on tool)

Raw training reward is a training instrument, not the truth: a candidate must beat a
diverse **frozen** field on matches it never trained on before it replaces the incumbent.
- **Module:** `soccer_eval_gate` ([soccer_eval_gate.rs](../src/des/general/soccer_eval_gate.rs)) —
  `evaluate_promotion(reports, candidate, baseline, thresholds) -> PromotionVerdict`. Pure,
  deterministic, folds held-out `MatchReport`s through the existing `EloRatings` +
  `CrossPlayMatrix`. Promotes iff all three clear: **strength** (mean cross-play payoff vs
  field + held-out Elo Δ), **confidence** (Wilson score lower bound on the field mean, so a
  lucky few-game edge cannot promote), and **robustness** (worst-case single-opponent payoff
  ≥ floor — a hard-countered brain is *different, not better*).
- **Runner:** `soccer_eval_gate_run` bin plays a candidate vs a frozen pool over a held-out
  seed range (disjoint from training) with the frozen `EngineMatchRunner`, then prints the
  verdict + every metric. This is the gate every reward/architecture change is judged by.

## Tier 4 — Tactical "manager" (the meta-policy above the players)

- **Live call site:** `tactical_directive_for_team` sets `home_directive` / `away_directive`.
  It is a **rule selector** (the flaky-on-main
  `tactical_directive_rule_selector_can_invoke_every_attack_strategy` test exercises it).
- **What's hardcoded:** formation, press intensity, line height, width, and attack
  strategy are chosen by rules.
- **Conversion:** a learned higher-level policy choosing the directive from
  (score, time, fatigue, territorial state) — a hierarchical RL agent sitting
  *above* the per-player policies. Larger build; do after Priorities 1–3 so the
  lower policies it commands are already learned.

---

## Suggested order

1. **Priority 1** — finish the discretized/factored action path: bucket entropy and proof first,
   then direction/curve/elevation/aim heads.
2. **Priority 2** — learned xT grid (lifts every net's reward, including the back
   four).
3. **Priority 3** — midfield/press line via the back-four template (cheap,
   parity-safe).
4. Tier 2 promotions (#4 pass-completion is nearly free — just consume the
   already-trained head).
5. Tier 3 / Tier 4 as larger follow-ups.

Every item keeps the back-four contract: **gated, off = byte-identical, analytic
seed as cold-start, the net earns its blend.**
