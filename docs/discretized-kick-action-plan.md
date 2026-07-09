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
- Pass: [world.rs:12603](../src/des/general/soccer/world.rs#L12603) — builds the spec
  from a heuristic `aimed_target`, `speed` (`modulated_pass_speed_yps`), `curve`,
  `curve_bend_yards`, and `flight`.
- Shot: [world.rs ~12799](../src/des/general/soccer/world.rs) — `speed` from
  `shot_speed_yps_from_power`, same `kick_release_clamped_to_pitch` path.

The **discrete representation already exists but is constructed only in tests**:
- `DiscretizedKickAction { speed_bucket: u8 (10), direction_bucket: u8 (36), curve, technique, elevation }`
  ([soccer.rs:63788](../src/des/general/soccer.rs#L63788)) with bucket↔value mappings.
- `lower_discretized_kick_release(origin, action, min_speed, max_speed, ref_distance, bend, dither)`
  → `LoweredKickRelease` ([soccer.rs:64092](../src/des/general/soccer.rs#L64092)) — the
  exact lowering the wiring needs. **Its only callers are in `#[cfg(test)]`**
  ([soccer.rs:69143-69177](../src/des/general/soccer.rs#L69143)).
- `DiscretizedKickDither::sample` is currently a no-op (returns zero offsets) — a
  stub awaiting a learned/stochastic source.

The decision path the wiring plugs into:
`learned_action_for_player_with_context` ([world.rs:13924](../src/des/general/soccer/world.rs#L13924))
→ `neural_blended_action` ([world.rs:4689](../src/des/general/soccer/world.rs#L4689))
→ `mpc_reconciled_learned_plan_for_policy` ([world.rs:5225](../src/des/general/soccer/world.rs#L5225))
→ executed in `apply_player_intent` ([world.rs:11663](../src/des/general/soccer/world.rs#L11663)).

The learnable-head pattern to mirror: `back_four_line.rs` —
`Inputs → to_features() → FeedForwardNetwork → analytic seed → MIN_TRAINING_STEPS
gate → predict() → train_reward_weighted()`.

## The gap

The actor's output space is the ~45 family labels in `SOCCER_POLICY_ACTIONS`. Once a
family is chosen, the strike parameters are a closed-form heuristic (power curve ×
aim-at-target × pressure-tuned curve/flight). So the policy cannot *learn* to hit a
through-ball weighted into stride vs. driven, to bend a cross around the last
defender, to clip a chip vs. drive it, or to pull a shot to the far corner — those
are hard-coded. The discrete buckets that would make them learnable are built and
unit-tested but never reach live play.

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
- **Phase 1 — `KickSpeedHead`** (`DD_SOCCER_ENABLE_LEARNED_KICK_SPEED`). Highest ROL,
  clearest reward. Learn "how hard" (weight the through-ball, drive the clearance).
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
- **`DiscretizedKickDither` is a no-op stub** — leave it zero unless a phase explicitly
  needs sub-bucket exploration; if used, it must stay within ±½ bucket (already
  sanitized) so it can't change the lowered legality.
- **Concurrent automation on `main`** (parallel-feature collisions have happened) —
  keep each phase a small, self-contained commit; verify combined-green after any merge.
- **MPC pass execution** (`mpc_predicted_receiver_path`, off by default) is a separate
  path; ensure the head wiring sits before/around it consistently, not double-applied.

## Definition of done (per phase)

Gate exists and defaults off; off ⇒ byte-identical (parity test green); analytic seed
== heuristic (unit test green); head trains; and an isolated held-out A/B has been run
and its verdict recorded — shipping the default-on **only** if the eval gate says
PROMOTE, otherwise the code lands gated-off with the verdict documented.
