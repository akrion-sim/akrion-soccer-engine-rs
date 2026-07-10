# Plan: wire the discretized kick action (the actor owns kick parameters)

Roadmap **Priority 1** ([learnability-conversion-roadmap.md](learnability-conversion-roadmap.md)) —
the highest-ROI dormant surface. The learned actor already picks the *family*
(`pass` / `killer-pass` / `shoot` / `flank-high-cross` / …), but **how the ball is
struck — power, direction, curve, elevation — is still chosen by heuristics**. This
plan makes those parameters a *learned discrete choice*, so the actor owns the whole
on-ball decision, not just its label.

Every change keeps the back-four contract: **gated, off ⇒ byte-identical, analytic
seed as cold-start, the net earns its blend**, and each phase is promoted only on a
held-out win-rate verdict from the eval gate ([learning-rate-audit.md](learning-rate-audit.md)
companion infra: `soccer_eval_gate_run` / `scripts/run_outcome_ab.sh`).

## Current state (verified against `main`, 2026-07-09)

The execution substrate is **already in place and shared** — passes and shots both
lower a `KickReleaseSpec` through `kick_release_clamped_to_pitch` into a
`LoweredKickRelease` (velocity + curl + altitude) that sets `ball.velocity`:
- Passes still build from a target, speed envelope, curve/bend, and flight.
- Shots still build from shot power and the same release path.

The **kick-power bucket slice is now live**, not only a scaffold:

- `SOCCER_POLICY_ACTIONS` has `113` append-only labels. `40` of them are kick-power bucket
  labels: `pass-kp0..9`, `aerial-pass-kp0..9`, `shoot-kp0..9`, and
  `first-time-shot-kp0..9`.
- `learned_discretized_kick_candidate_label` and
  `learned_discretized_kick_speed_bucket_for_action_label` preserve those buckets through
  ranked pass/shot labels such as `pass1-kp7`.
- `neural_mcts_expanded_candidate_plans` offers pass, aerial-pass, shot, and first-time-shot
  kick-power variants when `DD_SOCCER_ENABLE_DISCRETIZED_KICK` is on and the MCTS kick-power
  candidate knobs allow them.
- Scoring and execution can read the selected bucket's power through
  `learned_discretized_kick_power_for_action_label`, so telemetry and reward labels can see
  the selected bucket.

The **full factored kick action** is still not a learned head:

- `DiscretizedKickAction { speed_bucket, direction_bucket, curve, elevation }` and the
  lowering helpers exist, but live candidate expansion currently uses speed-bucket labels
  around plausible pass/shot targets, not a full learned direction/curve/elevation/aim head.
- `DiscretizedKickDither::sample` returns bounded inside-bucket offsets, and production
  pass release uses it only when `DD_SOCCER_ENABLE_DISCRETIZED_KICK_DITHER` is enabled.
  This provides local release exploration/robustness, not a learned bucket-choice head
  or extra credit assignment.

The decision path the wiring plugs into remains:
`learned_action_for_player_with_context` -> `neural_blended_action` ->
`mpc_reconciled_learned_plan_for_policy` -> `apply_player_intent`.

The learnable-head pattern to mirror: `back_four_line.rs` —
`Inputs → to_features() → FeedForwardNetwork → analytic seed → MIN_TRAINING_STEPS
gate → predict() → train_reward_weighted()`.

## The gap

The actor's output space is no longer just broad families: it includes `113` labels,
including the `40` kick-power bucket labels above. That means learned/MCTS selection can express
"same family, different power bucket."

The remaining gap is narrower but still important: direction, curve, elevation, and bounded aim
are still mostly analytic after the selected family/target. Bucket selection is also not yet a
stochastic learned exploration policy with entropy pressure, so early low-scored buckets can still
starve unless MCTS expansion, rank draw, or future bucket sampling keeps them alive.
The ceiling-break proof profile also enables
`DD_SOCCER_ENABLE_FORWARD_RELEASE_ROOT_CANDIDATE` plus
`SOCCER_NEURAL_MCTS_MIN_PASS_LIKE_ROOT_CANDIDATES=1`, so forward pass targets that the analytic
pass-target cap would hide can reach the neural scorer with the same pass-space and
kick-power variants as ordinary pass targets, then be measured in `DD_SOCCER_FWD_TRACE`
before any default-on promotion.

## Design

### Why factored parameter heads (not an expanded action set)

Folding parameters into the flat action set is combinatorial: 10 speed × 36 direction
× 3 curve × 4 elevation = **4 320** actions — unlearnable and serde-breaking. Instead,
**keep the family actor as-is** and add a small set of **factored kick-parameter heads**
that, conditioned on `(state, chosen family)`, choose each bucket. This mirrors the
specialist-head pattern and keeps each head's output space tiny.

Heads (each a `FeedForwardNetwork` over a shared kick-parameter feature vector):
| Head | Output | Phase |
|------|--------|-------|
| `KickSpeedHead` | softmax over 10 speed buckets | 1 |
| `KickElevationHead` | softmax over {Floor, Aerial, Chip, Scoop} | 2 |
| `KickCurveHead` | softmax over {None, Left, Right} (+ bend regressor) | 3 |
| `KickAimHead` | **bounded** lead/aim offset around the heuristic target (±N°), not a free 36-way absolute bucket | 4 |

Receiver/target *selection* (who to pass to) stays with the existing
`best_pass_target_quality_for_snapshot` ranker initially; the heads own *how* the ball
is struck toward the chosen target. Phase 4 only *refines aim around* that target
(lead the runner, aim into the open corner) — a bounded offset is far more learnable
and parity-safe than an absolute direction bucket.

### Analytic seed = today's heuristic (the parity anchor)

Each head ships with an `analytic_*` seed that reproduces the current heuristic choice
**exactly**: `discretized_kick_speed_bucket_for_power(heuristic_power)` for speed,
`DiscretizedKickElevation::from_pass_flight(flight)` for elevation, the current
curve-decision for curve, zero offset for aim. With the gate off, the head is bypassed
and the heuristic `KickReleaseSpec` is built as today ⇒ **byte-identical**. With the
gate on but `training_steps < MIN`, the head returns the analytic seed ⇒ still
byte-identical until it has learned something.

### Wiring point

In `apply_player_intent`'s pass/shot branches, immediately before building the
`KickReleaseSpec`: if `learned_kick_<param>_enabled()` and the head is ready, replace
the heuristic `speed`/`flight`/`curve`/`aimed_target` with the head's choice lowered
through `lower_discretized_kick_release` (reusing `min/max_speed` from the existing
pass-speed envelope and `reference_distance` = distance to target). Off ⇒ the existing
code runs untouched. The lowering already clamps to pitch and masks illegal curve
(`masked_discretized_kick_curve`), so no new physics.

### Feature vector

Reuse the policy state features plus a short kick-specific tail: distance to target,
receiver openness, passer pressure, angle to goal, defender-in-lane proximity, flight
legality flags. Append-only with zero-pad migration (the established `FEATURE_DIM`
convention).

## Training signal

The heads train by the **same actor advantage** already computed in
`neural_policy_training_samples` (GAE + outcome credit), reward-weighted on the
realised kick outcome: completion × xT-gain for passes, on-target × xG for shots
(both already logged). The chosen bucket is the action index; the advantage is the
sample weight — identical in spirit to `BackFourLineHead::train_reward_weighted`. The
behavior probability (for the PPO ratio) is the head's softmax at decision time,
threaded the same way as the existing `behavior_policy_probability`.

Note the interaction the audit flagged: a new per-decision signal must not silently
inflate the effective LR — these heads are advantage-driven, so they ride the existing
`dd_soccer_standardize_policy_advantages()` standardization (already coupled to the
outcome reward / advantage-norm gate).

## Phased rollout (each gated, each A/B'd on held-out Elo)

- **Phase 0 — plumbing (byte-identical).** Build the shared feature vector + a
  `lower_discretized_kick_release` call site behind a master gate that, when on,
  reconstructs the *heuristic* `DiscretizedKickAction` and lowers it. Assert the
  lowered velocity/altitude matches the current heuristic path to fp tolerance — this
  proves the discrete round-trip is loss-free before any net is involved.
- **Phase 1 — bucket ownership and entropy.** The speed-bucket labels and MCTS candidate
  expansion are landed. The missing production step is to make selected-bucket exploration and
  entropy explicit, then prove with held-out eval that the bucket choice improves outcomes rather
  than merely moving labels.
- **Phase 1b — `KickSpeedHead`** (`DD_SOCCER_ENABLE_LEARNED_KICK_SPEED`). Highest ROI,
  clearest reward if a separate specialist head is still needed after bucket-label training.
  Learn "how hard" (weight the through-ball, drive the clearance).
- **Phase 2 — `KickElevationHead`** (`…_KICK_ELEVATION`). Floor vs chip vs aerial.
- **Phase 3 — `KickCurveHead`** (`…_KICK_CURVE`). Bend crosses/shots around defenders.
- **Phase 4 — `KickAimHead`** (`…_KICK_AIM`). Bounded lead/aim refinement.

Each phase: implement → unit tests (analytic seed == heuristic; lowering round-trip;
gate-off byte-identical) → `scripts/run_outcome_ab.sh`-style isolated A/B (head on vs
off, advantage-norm on in both) → promote only on a positive held-out verdict from
`evaluate_promotion`. A null/negative verdict ⇒ leave the phase gated-off (as we did
for the won-game reward), do not ship a neutral default.

## Test plan

1. `lower_discretized_kick_release` round-trip: heuristic params → bucket → lower →
   velocity/altitude within fp tolerance of the direct `kick_release` path.
2. Analytic seed equals the heuristic choice for representative states (speed,
   elevation, curve).
3. Gate-off byte-identical: a fixed-seed match produces an identical event stream with
   each gate off (the established parity test shape).
4. Head learns its corpus (loss decreases; accuracy/value improves) on a synthetic
   reward-weighted set.
5. Legality preserved: illegal curve masked, pitch-clamped target, NaN-guarded.

## Risks / landmines

- **Parity drift in Phase 0.** The discrete round-trip must be loss-free or every
  phase inherits a silent diff. Gate Phase 0 and assert equality before trusting it.
- **Direction as absolute bucket** would be a large, hard-to-learn, parity-hostile
  output — explicitly deferred to a *bounded offset* in Phase 4.
- **Reward sparsity for curve/aim** (rare events) ⇒ slow learning; expect Phase 3–4
  to need more samples, and accept gated-off if the A/B is null.
- **`DiscretizedKickDither` is inside-bucket only** — do not describe it as stochastic
  bucket selection or a learned parameter head. It must stay within +/- half a bucket,
  and the learner must still credit the selected bucket explicitly.
- **Concurrent automation on `main`** (parallel-feature collisions have happened) —
  keep each phase a small, self-contained commit; verify combined-green after any merge.
- **MPC pass execution** (`mpc_predicted_receiver_path`, off by default) is a separate
  path; ensure the head wiring sits before/around it consistently, not double-applied.

## Definition of done (per phase)

Gate exists and defaults off; off ⇒ byte-identical (parity test green); analytic seed
== heuristic (unit test green); head trains; and an isolated held-out A/B has been run
and its verdict recorded — shipping the default-on **only** if the eval gate says
PROMOTE, otherwise the code lands gated-off with the verdict documented.
