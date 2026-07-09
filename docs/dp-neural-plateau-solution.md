# DP, Neural Nets, And The Plateau Fix

## Short Answer

Approximate dynamic programming is part of the solution, but it is not a
replacement for neural nets.

The better framing is:

1. **DP/Bellman updates clean up credit assignment.**
   They propagate delayed soccer outcomes backward through the decisions that
   caused them: completed passes, loose passes, interceptions, shots, goals,
   conceded danger, and turnovers.

2. **Neural nets generalize that cleaned-up signal.**
   They see the full actor-relative field vector: 22 players plus ball position,
   velocity, acceleration, and jerk. They are the right tool for continuous
   soccer state, because no two field states are identical.

3. **MPC executes and repairs the selected intent.**
   MPC should keep decisions physically feasible and field-aware. It should not
   become the primary team-level learner.

4. **Population perturbation is only an escape hatch.**
   It can propose new neural weights when gradient/on-policy learning stalls,
   but promotion still has to be held-out HOME-vs-analytic and local-best gated.

So the solution is not "DP instead of neural nets." It is:

```text
cross-tick measurement
  -> Bellman / approximate DP propagation
  -> neural value + actor learning over full field context
  -> short learned-model lookahead, with neural MCTS optional
  -> MPC feasibility and execution
  -> strict held-out promotion
```

## Why DP Helps Right Now

The current plateau was not just an architecture problem. It was also a signal
problem.

If the learner does not correctly know that a pass was completed, a dribble
created advantage, a shot was high quality, or a bad pass was intercepted two
ticks later, then the neural net can train forever on weak or misplaced labels.
That creates the appearance of "learning" without changing field behavior.

Approximate DP helps because it can replay a whole-game trajectory and push
later consequences backward:

- a completed pass credits the original pass decision;
- a loose pass recovered by the opponent penalizes the original pass decision;
- an intercepted backward pass carries a larger penalty than a lateral/forward
  failed pass;
- a shot on target and goal-creating action propagate value to the buildup;
- conceded danger and turnovers propagate negative value to the defensive or
  possession decision that caused the transition.

This is especially useful in soccer because many good actions are not rewarded
immediately. The correct pass may matter because it creates the next pass, and
the next pass creates the shot.

## Why Neural Nets Still Matter

DP alone is not enough for full soccer.

Exact DP requires a finite state table. Soccer does not have a small finite
state table. The live state includes 22 players, ball motion, team shape, partial
observability, possession context, tactical roles, pressure, fatigue, and
continuous geometry. A tabular state bucket will always alias very different
field states together.

That is why the neural net remains central:

- it consumes the high-dimensional field vector;
- it generalizes between similar but not identical situations;
- it estimates value for states never seen exactly before;
- it lets learned MPC and optional neural MCTS score candidate futures;
- it supports MAPPO/MARL style team credit beyond one player's immediate reward.

The neural net should learn from DP-improved targets. It should not be forced to
discover all delayed credit from scratch.

## The Correct Division Of Labor

| Layer | Job | Should It Be Primary? |
| --- | --- | --- |
| Cross-tick measurement | Detect pass, dribble, shot, turnover, and conceded-danger outcomes across multiple `run_time_step` calls | Yes, for reward truth |
| Approximate DP / Bellman replay | Propagate delayed outcomes backward through the trajectory | Yes, for target quality |
| Neural actor/value | Generalize value and action preference over the full field vector | Yes, for learning and decisions |
| Learned world model / optional neural MCTS | Score short candidate futures and teach better action targets | Yes for the world model; MCTS is optional/non-primary |
| MPC | Make the selected intent physically feasible and field-aware | Yes, as executor/guard |
| Population perturbation | Escape local optima when the neural policy stalls | No, only as plateau escape |
| League/tournament evolution | Broad ranking and exploration | No, not while debugging this plateau |

## What This Means For The Current Local Run

The local training setup should keep these active:

- neural-authoritative decisions;
- hidden units at 128;
- MAPPO/MARL team reward sharing;
- full 22-player plus ball motion vector;
- analytic opponent for held-out pressure;
- approximate DP replay passes (forward-pass climb profiles now default to the
  full 8-pass replay budget unless `SOCCER_APPROX_DP_REPLAY_PASSES` overrides);
- DP-bootstrapped replay returns for the climb stack:
  `DD_SOCCER_ENABLE_DP_BOOTSTRAP=1`, `DD_SOCCER_DP_BOOTSTRAP_HORIZON=64`, and
  `DD_SOCCER_DP_BOOTSTRAP_SWEEPS=200`;
- DP-bootstrapped critic targets, so the neural critic can bootstrap from
  `V_DP(b')` instead of the tabular Q ceiling:
  `DD_SOCCER_ENABLE_DP_CRITIC_TARGET=1`;
- bounded DP/Q policy prior in authoritative neural scoring:
  `SOCCER_APPROX_DP_POLICY_PRIOR_WEIGHT=0.5`;
- neural world model, with neural MCTS optional/non-primary;
- local MPC with field-aware, reconcile, and latent-objective gates;
- guarded snapshot propagation, so newest weights are used only when they are
  not much worse than the batch's best HOME objective;
- neural population search as a held-out plateau escape layer;
- old league/tournament evolution disabled as the primary learner.

## What Not To Do

Do not treat "latest weights every tick" as automatically better. On-policy
training usually collects a rollout under a stable behavior policy, then updates
at an episode or batch boundary. Refreshing every tick can make the target move
under the sample and destabilize credit assignment.

Do not let a flat training loss convince us the policy is good. A tiny loss can
mean the value model is accurately predicting mediocre behavior.

Do not reward safe possession so much that successive passes beat shots, shots
on target, and goal creation.

Do not use raw tournament rank as the live promotion rule while debugging this.
Promotion should compare the HOME neural side against a frozen analytic
baseline and protected local best.

## How To Know It Is Working

The plateau is not fixed just because one candidate wins a held-out mini-eval.
It is fixed only when the live trend climbs repeatedly.

Track each 6-game window:

| Metric | Why It Matters |
| --- | --- |
| HOME mean match fitness | Main corrected objective |
| Best match fitness | Confirms occasional high-quality candidates exist |
| Mean play quality | Prevents gaming the score with ugly behavior |
| Score, shots, SOT | Checks attacking and defensive reality |
| Pass completion and interceptions | Checks possession quality |
| Route-one count | Catches regression into long-ball local optima |
| Neural batch selected training steps and selected fitness | Verifies weight propagation is fresh but not reckless |
| Population accepted/held events | Shows plateau escape is useful but not primary |
| Policy entropy | Detects collapse into one safe action mode |
| World-model validation loss | Checks whether neural planning is trustworthy |
| MPC replan rate | Shows whether planner proposals are executable |

The bar remains:

```text
5+ consecutive corrected HOME-vs-analytic windows improving
and no behavior regression in shots, SOT, pass quality, or route-one rate
and promotion only when protected local-best gates pass
```

## Practical Conclusion

DP is the bridge out of the plateau because it improves the learning signal.
Neural nets are still the destination because they generalize that signal across
the real field vector and make decisions in continuous soccer space.

The immediate solution is therefore hybrid:

```text
measure outcomes correctly across ticks
use approximate DP to propagate delayed credit
train neural actor/value on the corrected trajectory
let the learned model propose short-horizon improvements, with MCTS optional
let MPC execute safely
promote only through strict held-out gates
```

That is the simplest path that does not throw away the neural work and does not
pretend a tabular DP solver can own the whole POMDP.
