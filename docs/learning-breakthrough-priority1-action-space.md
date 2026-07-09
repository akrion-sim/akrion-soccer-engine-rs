# The real breakthrough: wire the learnable action space + close the policy-improvement loop

> **Context.** After fixing the value-head collapse (see
> [`value-collapse-and-target-standardization.md`](value-collapse-and-target-standardization.md)),
> the un-collapsed value net still lands at **parity** (0.478 over 200 games vs the analytic
> engine, confirmed on two machines). This note explains *why* — the plateau is **structural**, not
> a training bug — and lays out the concrete, code-grounded fix. Grounded in
> [`learnability-conversion-roadmap.md`](learnability-conversion-roadmap.md) Priority 1 (which the
> repo itself flags as DORMANT / highest-ROI).

## Why un-collapsing the value can't break parity

Three compounding structural ceilings, all verified in code:

1. **The net now has a partial learned parameter surface, but not the whole one.**
   Kick-power bucket labels are present in `SOCCER_POLICY_ACTIONS` and MCTS can expand pass,
   aerial-pass, shot, and first-time-shot bucket variants behind `DD_SOCCER_ENABLE_DISCRETIZED_KICK`.
   That fixes the older "never reaches candidates" diagnosis for speed buckets. Direction, curve,
   elevation, and bounded aim are still mostly analytic, and bucket selection still needs entropy
   telemetry/proof.
2. **The value regresses onto the *tabular* Q.** Target = `r + γ·max_next` with `max_next =`
   tabular `best_value_hierarchical` ([world.rs](../src/des/general/soccer/world.rs)). A value
   trained to match the tabular Q **cannot exceed the analytic ceiling** — it's imitation, not
   policy improvement.
3. **The action encoding is still not fully structured.** Labels such as `pass-kp7` let the actor
   distinguish bucket choices, and the transition context can expose selected kick power, but the
   broader parameter space is not yet a clean structured block for direction/curve/elevation/aim.
   Do not treat the landed power labels as a complete factored action representation.

## The drift that produced this

~95% of the last 100 commits are analytic/tactical heuristic tweaks (back-four, pass-risk, flank,
blindside, magnus, dribble, press) + gated tactical features. The core learnable action-policy (P1)
and learned reward (P2) stayed dormant. **The engine got better at analytic play — raising the bar
the net must beat — while the net was never given the tools to beat it.**

## The fix — three parts (all real, needs a retrain to evaluate)

### Part A — structured action-param feature block (append-only)
Add a `SOCCER_NEURAL_ACTION_PARAM_FEATURE_DIM` block (append-only, so old nets zero-pad — the
established migration pattern) encoding the candidate's **aim**, from `AgentActionTargetTrace.point`
relative to the ball/actor: target dx/dy (normalized), distance, forward-ness (toward attacking
goal), direction sin/cos, has-target. Gate `DD_SOCCER_ENABLE_ACTION_PARAM_FEATURES` (off ⇒ zero
block ⇒ byte-identical). **This replaces the opaque hash with structure the net can generalize
over** — even for existing candidates.

### Part B — discretized-kick variant candidates
The first slice of this is already implemented for speed: the actor vocabulary has `40` kick-power
bucket labels and MCTS can add bucket variants when `DD_SOCCER_ENABLE_DISCRETIZED_KICK` is on.
The remaining work is to add the missing structured dimensions:

- direction/aim as a bounded offset around the current target, not a free absolute 36-way choice;
- curve and elevation as small masked heads;
- selected-bucket entropy and net-changed-action telemetry, so the code proves the bucket path
  changes behavior instead of only changing labels.

### Part C — true policy improvement (break tabular imitation)
Either (i) bootstrap the value off *its own* successor value (real neural n-step / Q-learning) so it
can rate a policy *better* than tabular, or (ii) let the **actor** (`SoccerPolicyHead`, PPO+GAE —
already built, but currently subordinate to the authoritative λ=8 value) drive, since policy
gradient pushes toward higher *return*, not toward matching the analytic Q. This is the only piece
here that can structurally *exceed* analytic.

### (P2, parallel) learn the reward basis
Replace closed-form `expected_threat` with a **Markov xT grid** learned from self-play possession
chains (P(score before turnover) per cell); enable `DD_SOCCER_ENABLE_PITCH_VALUE_REWARD`. Lifts
every net's reward/critic basis at once.

## Evaluation
Each part is gated (off ⇒ byte-identical). The action-space change (A+B) needs a **full retrain**
from a fresh net to be meaningful (existing nets have no param features / never saw the variants).
Success bar unchanged: sustained > 0.55 vs the analytic field with Wilson lower bound > 0.5.

**Bottom line:** the breakthrough is not a new optimizer or bonus. The speed-bucket slice of the
learnable action space is now wired; the remaining breakthrough is to make bucket exploration and
causal influence measurable, then finish the structured direction/curve/elevation/aim surfaces and
close the policy-improvement loop (C).
