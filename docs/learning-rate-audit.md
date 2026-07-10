# Learning-rate audit & hardening

An audit of every step-size / stability knob in the soccer learner, the failure
modes they leave open, and the hardening applied. Companion to
[reward-shaping-audit.md](reward-shaping-audit.md) (what the reward *is*) and
[learnability-conversion-roadmap.md](learnability-conversion-roadmap.md) (which
surfaces learn). This doc is about *how fast and how safely* the nets move.

## The surface (all in `soccer.rs` unless noted)

| Knob | Value | Role |
|------|-------|------|
| `DEFAULT_SOCCER_NEURAL_LEARNING_RATE` | 0.015 | value/critic SGD step |
| `SOCCER_POLICY_LEARNING_RATE` | 0.05 | actor (policy-gradient) step |
| `SOCCER_WORLD_MODEL_LEARNING_RATE` | 0.02 | world-model regression step |
| `MAX_SOCCER_NEURAL_LEARNING_RATE` | 0.25 | hard clamp via `sanitized_learning_rate` |
| `SOCCER_NEURAL_GRAD_CLIP_NORM` / `SOCCER_POLICY_GRAD_CLIP_NORM` / world-model | 8.0 / 5.0 / 8.0 | per-step L2 gradient-norm ceilings |
| `DEFAULT_SOCCER_NEURAL_TARGET_SCALE` / `_TARGET_CLIP` | 30.0 / 3.0 | value target is `reward/scale` clipped to ±clip |
| `SOCCER_POLICY_GAE_LAMBDA` | 0.95 | advantage trace decay |
| `SOCCER_POLICY_ENTROPY_COEFF` | 0.01 | anti-collapse entropy bonus |
| `DEFAULT_SOCCER_MAPPO_CLIP_EPSILON` | 0.20 | PPO/MAPPO ratio clip |
| `SOCCER_FULL_GAME_RETURN_{DISCOUNT,BLEND,CLIP}` | 0.995/tick, 0.35, 250.0 | full-game return recursion |
| advantage standardization | gated, default OFF | zero-mean/unit-variance over the policy batch ([world.rs](../src/des/general/soccer/world.rs)) |

**Already-solid guards (no change needed):** LR clamped ≤ 0.25; three grad-norm
ceilings; the non-finite-gradient guard in `FeedForwardNetwork::train_sample_clipped`
drops poisoned steps so a NaN can't reach live play; value targets bounded to ±3.0;
actor samples filter `advantage.is_finite()`; full-game return clamped to ±250.

## Findings

**F1 — Effective learning rate drifts with reward magnitude (the core issue).**
The policy-gradient step is `LR · advantage`, so the *effective* LR scales with the
advantage magnitude, which scales with reward magnitude. Advantage standardization —
the textbook fix that decouples the step from reward scale — is **OFF by default**.
Any change that shifts reward magnitude (a new shaping term, and especially the flat
terminal won-game label) silently changes the effective LR. With GAE, a *constant*
per-step label is worst early in training: the trace `A_i = Σ (γλ)^k δ_{i+k}` sums the
label over the trajectory before the critic learns to absorb it, so early advantages
can be large ∝ trajectory length.

**F2 — The keeper head is baseline-free and unnormalized.** `neural_keeper_policy_training_samples`
sets `advantage = transition.reward` directly — no critic baseline, no standardization
([world.rs](../src/des/general/soccer/world.rs)). The flat won-game label (±8–12.5)
then *dominates* the keeper's own shaping rewards (a save is ~4–8), so the keeper would
learn "we won ⇒ every action I took was good," independent of quality.

**F3 — Value-target saturation (pre-existing, flagged not fixed).** Targets are
`reward/30` clipped to ±3.0, i.e. the critic cannot represent a return beyond ±90 raw,
while the full-game return clips at ±250. High-magnitude games saturate the critic and
bias the advantage baseline. The won-game label (±12.5 ⇒ ±0.42 in target space) stays
well inside this, so it does not worsen F3, but the tension is worth tracking.

**F4 — Other reward-fed trainers are safe.** The tabular Q trainer takes a flat
per-team label as a constant that cancels in per-team argmax; the value net clips (F3);
the actor specialists reuse the (now standardized) actor advantage; the pass-completion
/ long-pass / beat-defender heads train on their own targets, not the replay reward.

## Hardening applied

A flat Monte-Carlo outcome label is only safe *with* advantage standardization, so the
won-game reward now forces it on (default path unchanged):

- New `dd_soccer_standardize_policy_advantages() = dd_soccer_enable_advantage_normalization() || match_outcome_reward_enabled()`
  and a shared `policy_advantage_standardization(&[f64]) -> Option<(mean, std)>` helper
  ([world.rs](../src/des/general/soccer/world.rs)).
- **Actor** (`neural_policy_training_samples`): refactored onto the shared helper +
  the coupled gate. Byte-identical for every pre-existing gate state (the actor already
  standardized under the advantage-normalization gate); standardization is now also
  forced when the won-game reward is on.
- **Keeper** (`neural_keeper_policy_training_samples`): adds standardization gated on
  `match_outcome_reward_enabled()` alone — byte-identical for every pre-existing path
  (the keeper never standardized), and removes the common-mode label + bounds the step
  scale when the label is on. Standardization keeps the realised win-vs-loss *contrast*
  (winner's advantages stay above the loser's); it only removes the common mode.

Off ⇒ byte-identical. Covered by `policy_advantage_standardization_*` (degenerate /
non-finite / zero-mean+unit-variance + contrast-preserved) and the existing
advantage/keeper/actor/mappo suites.

## Recommendations (gated; validate via the promotion eval gate, not raw reward)

1. **Enable advantage normalization in trained runs** (`DD_SOCCER_ENABLE_ADVANTAGE_NORMALIZATION=1`)
   so the effective LR is reward-scale-invariant even without the won-game reward; A/B
   held-out Elo via `soccer_eval_gate_run` before making it the default.
2. **A/B the won-game magnitudes** (`MATCH_OUTCOME_WIN`=8.0, margin 1.5, cap 4) on
   held-out win-rate — the eval gate is the instrument.
3. **Track F3**: if high-scoring self-play games become common, lift `target_clip` or
   the return clip together so the critic baseline isn't saturated.
