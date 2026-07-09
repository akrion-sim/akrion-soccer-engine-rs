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

1. **The net ranks heuristic actions with heuristic *parameters*.** The learnable factored action
   space (`DiscretizedKickAction` — actor owns kick power/direction/curve) exists and is unit-tested
   but is only used at the **execution/lowering** layer ([world.rs:14650](../src/des/general/soccer/world.rs#L14650),
   gated `DD_SOCCER_ENABLE_DISCRETIZED_KICK`) — *never as candidates the net ranks*. Pass/shoot
   params are a continuous power × ~72-site heuristic scan.
2. **The value regresses onto the *tabular* Q.** Target = `r + γ·max_next` with `max_next =`
   tabular `best_value_hierarchical` ([world.rs](../src/des/general/soccer/world.rs)). A value
   trained to match the tabular Q **cannot exceed the analytic ceiling** — it's imitation, not
   policy improvement.
3. **The action encoding has no structured params.** In `soccer_neural_transition_features_with_action`
   the action is encoded as **3 family bits** (`soccer_neural_action_family_features`) **+ one opaque
   FNV hash** ([soccer.rs:40402](../src/des/general/soccer.rs#L40402)). The hash lets the net
   *distinguish* labels but has no structure — `pass|spd:6` and `pass|spd:7` map to unrelated
   scalars, so the net **cannot learn "this kick speed/direction is better here."**

## The drift that produced this

~95% of the last 100 commits are analytic/tactical heuristic tweaks (back-four, pass-risk, flank,
blindside, magnus, dribble, press) + gated tactical features. The core learnable action-policy (P1)
and learned reward (P2) stayed dormant. **The engine got better at analytic play — raising the bar
the net must beat — while the net was never given the tools to beat it.**

## The fix — three parts (all real, needs a retrain to evaluate)

### Part A — structured action-param feature block (append-only)
Add a `SOCCER_NEURAL_ACTION_PARAM_FEATURE_DIM` block (append-only, so old nets zero-pad — the
established migration pattern, see [soccer.rs:4505](../src/des/general/soccer.rs#L4505)) encoding the
candidate's **aim**, from `AgentActionTargetTrace.point` ([soccer.rs:7805](../src/des/general/soccer.rs#L7805))
relative to the ball/actor: target dx/dy (normalized), distance, forward-ness (toward attacking
goal), direction sin/cos, has-target. Gate `DD_SOCCER_ENABLE_ACTION_PARAM_FEATURES` (off ⇒ zero
block ⇒ byte-identical). **This replaces the opaque hash with structure the net can generalize
over** — even for existing candidates.

### Part B — discretized-kick variant candidates
In the candidate assembly ([world.rs:5623](../src/des/general/soccer/world.rs#L5623), where legal
`SOCCER_POLICY_ACTIONS` are injected), for kick families (pass/shoot/cross) generate a *spread* of
`DiscretizedKickAction` variants over the speed/direction buckets toward plausible targets, each
carrying its Part-A param features. The net ranks them; the chosen variant's params are lowered via
the existing `DiscretizedKickAction::from_power_direction` path. **Now the policy's output space
matches the trainable action set** — it can express kicks the heuristic scan can't.

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

**Bottom line:** the breakthrough is not a new optimizer or bonus — it's **wiring the learnable
action space (A+B) and closing the policy-improvement loop (C)** that the team built but left
dormant. That is what lets learning *exceed* the analytic engine instead of converging to it.
