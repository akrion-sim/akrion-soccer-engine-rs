# Fixing Learning Plateaus

How to get the soccer learner out of local optima, avoid flat/noisy training loops, and
prove that newer weights are actually better than older weights.

This is the practical companion to [climbing.md](climbing.md) and
[reframe-july-2026.md](reframe-july-2026.md). Those docs explain the July 2026 reframe:
make the neural net authoritative, persist its training warmth, verify dense rewards reach
the value net, and train on-policy. This doc explains what to do when that working setup
still plateaus.

## The Short Version

A plateau does not mean feedback stopped. It usually means the policy has learned all the
reward signal currently distinguishes. The net can have hundreds of field features and dense
per-action feedback, but if the reward function mostly grades "competent soccer," the model
can race to competence and then sit there. At that point, smaller training loss just means it
predicts its own current outcomes well; it does not prove the policy is becoming stronger.

The escape strategy is:

1. Prove the plateau with a low-variance held-out metric.
2. Keep gradient RL as the primary learner: neural-authoritative, on-policy, local JSON,
   protected local-best publishing.
3. Raise the reward ceiling so goals, shots on target, high-xG goal entry, and safe
   progressive play beat sterile pass chains.
4. Add learned model-based lookahead in the trainer: world-model + shallow neural MCTS
   scoring candidate futures, with per-player MPC as the feasibility/execution guard.
5. Add controlled exploration: entropy/top-k action sampling, small weight/gradient noise,
   and replay diversity.
6. Use population perturbation only as a plateau-escape layer, evaluated against a frozen
   analytic/anchor opponent, then train accepted variants on-policy before publishing.
7. Publish only when a candidate beats the protected local-best on held-out games.

Do not use tournament/league rank as the primary learning signal while diagnosing a plateau.
It is too easy for self-play fitness to become a weighted random walk.

## What A Plateau Means Here

There are three different "flat" states that can look similar in logs:

| Symptom | Meaning | Response |
|---|---|---|
| Training loss is tiny but game behavior is not better | The value net predicts its own current outcomes, not necessarily better play | Improve reward ceiling, opponent, and planning signal |
| Candidate windows bounce up and down | Exploration/eval variance, not confirmed climb | Increase held-out samples; require multi-window trend |
| New generations publish but live behavior regresses | Promotion gate is not comparative enough | Compare against protected incumbent on frozen seeds/anchors |

For this engine, "local minimum" usually means a local optimum in policy/reward space, not a
literal neural loss minimum. The policy has found a basin where small action/value changes do
not improve expected reward. Escaping requires either a better gradient signal, broader
exploration, or a planner that exposes better futures.

## First Prove It Is Really Plateaued

Do this before changing knobs:

1. Evaluate HOME neural vs a frozen analytic/anchor opponent, not only self-play.
2. Use a fixed seed pool plus a rotating held-out pool. Fixed seeds catch regressions;
   rotating seeds catch overfitting.
3. Track at least six-game windows, but do not claim "climbing" until five consecutive
   checkpoint deltas are positive or a rolling confidence interval is clearly up.
4. Separate candidate quality from protected live quality. `localhost:5055` should normally
   serve protected promoted weights, not every exploratory candidate.
5. Keep behavior metrics next to fitness: score, shots, shots on target, goals conceded,
   backward-pass turnovers, pass-chain net loss, progressive carries, and goal-entry actions.
6. Separate current telemetry from missing diagnostics. Today the trainer reports MCTS
   selection/candidate shares, MPC replans, priority samples, distillation, and policy entropy.
   It does not yet prove whether the net changed the executed action, whether ConfidenceGated
   opened on that selected action, or whether selected kick-power buckets retained entropy.

Minimum local report:

| Metric | Why it matters | Good signal |
|---|---|---|
| Held-out HOME mean fitness | Corrected objective for neural-vs-anchor | Positive rolling trend |
| Best match fitness | Detects occasional breakthrough policies | Higher peaks without worse mean |
| Mean play quality | Catches "winning ugly" or reward hacks | Stable or rising |
| Shots / shots on target | Verifies attacking intent | Rising SOT without conceding more |
| Backward-pass interceptions | Verifies bad-pass discipline | Falling, especially in own half |
| Net-changed action rate | Proves learned scoring changes executed behavior | Nonzero and rising in learning arms |
| ConfidenceGate selected-open rate | Proves low-visit actions can use the net | Nonzero when ConfidenceGated is the blend |
| Selected kick-bucket entropy | Detects early collapse into one power bucket | Stable/diverse by family before annealing |
| Old-prob missing / PPO clip rate | Checks PPO/MAPPO denominator health | Low missing rate; sane ratio and clip stats |
| World-model validation loss | Whether learned lookahead is trustworthy | Stable/down before increasing MCTS budget |
| Neural-MCTS selection rate | Whether planning affects choices | Nonzero but not overriding everything blindly |
| MPC replan rate | Whether feasibility guard is repairing plans | Low/moderate; spikes mean planner is proposing infeasible futures |
| Publish count | Whether localhost should visibly improve | Protected local-best publish after held-out win |

The net-changed, gate-open, bucket-entropy, and PPO-ratio rows are required diagnostics, not
current `world_model_training` fields. Add them before declaring that an MCTS or bucket change
caused the policy to climb.

## Root Causes And Fixes

### 1. Neural Is Not Authoritative

If the net is only a small additive nudge on top of tabular Q or analytic heuristics, it can
learn without changing play. The fix is already the preferred local mode:

- `SOCCER_NEURAL_BLEND_MODE=authoritative`
- `SOCCER_NEURAL_BLEND_LAMBDA=1.0`
- `SOCCER_NEURAL_BLEND_WARMUP_STEPS=0`
- persisted `trainingSteps` and `averageLoss`

Keep the tabular/analytic layer as legality, priors, and tie-breaks, not the owner of action
selection.

### 2. Off-Policy Distribution Shift

If the analytic engine drives training but the neural net drives inference, the net learns
states created by analytic play and then fails in states created by its own choices. The fix
is on-policy authoritative training: the net must learn from the states it creates.

This is the main learner. Population search, eval gates, and planner changes should support
this loop rather than replace it.

### 3. Reward Ceiling Is Too Low

Dense rewards speed learning, but reward design sets the ceiling. A model can max out a
"competent soccer" rubric and stop improving. Raise the ceiling:

- Goals dominate everything.
- Shots on target and high-quality shots beat sterile pass completion.
- Goal-entry passes, killer passes, in-behind runs, and progressive carries beat safe
  recycling.
- Completed pass chains are useful only when they move the ball toward danger.
- Backward pass intercepted by the opponent should be penalized more than a forward/lateral
  interception, especially in the defensive half.
- Conceding shots on target should hurt even if no goal is scored.

The key is not "more reward channels." It is reward channels that distinguish great play from
merely safe play.

### 4. Opponent Equilibrium

Self-play can converge to a stalemate where both sides are equally mediocre or equally
competent. The win signal becomes noise. Use opponent ladders:

- Frozen analytic opponent for the main objective.
- Protected local-best as the incumbent.
- Diverse frozen anchors from prior checkpoints.
- Fresh/self-play opponents only as a secondary robustness test.

Do not let a league table decide promotion while the plateau is being debugged.

### 5. Planning Is Not Yet Learned Enough

MPC is an execution/reconciliation layer. It should keep one actor's motion dynamically
feasible; it should not become team-level MPC or replace the policy. To make neural nets help
solve MPC/planning:

1. Train a learned world model locally.
2. In the trainer, use shallow neural MCTS/lookahead over a bounded candidate set.
3. Score futures with learned value + dense reward estimates.
4. Let analytic/QP MPC reject or repair infeasible execution.
5. Log world-model validation loss, MCTS selection rate, MPC replan rate, policy entropy,
   and held-out HOME-vs-analytic fitness before trusting it.

Keep this trainer-first. The viewer can stay conservative until the planned policy publishes.

### 6. Exploration Is Too Narrow Or Too Destructive

Pure gradient descent can stay in a basin. Pure random mutation can destroy learned structure.
Use controlled exploration in layers:

- Top-k/Boltzmann sampling during training, with small epsilon.
- Entropy regularization where MAPPO supports it.
- Learning-rate schedules or temporary step-size bumps when validation is stable but fitness
  is flat.
- Small weight perturbations around the incumbent.
- Crossover only between strong anchors, not arbitrary noisy candidates.
- Replay diversity so the net keeps seeing rare high-value states.

Exploration should create useful on-policy data. It should not publish directly.

### 7. Representation Bottleneck

The 22 players plus ball field vector is the right signal, but it must be used with the right
inductive bias:

- Keep actor-relative canonical coordinates.
- Keep position, velocity, acceleration, and jerk.
- Normalize channels.
- Use relational attention/set encoding so player order does not become an artificial
  feature.
- Increase capacity when the policy truly plateaus: 128 hidden units is a reasonable local
  baseline; 256/512 may be justified only with validation and timing checks.
- Use embedding retrieval with cosine similarity for rare tactical moments. Do not use raw
  Euclidean kNN on the full feature vector.

## Population Search: Correct Role

Population search is useful, but only as a plateau escape hatch.

Good role:

1. Start from protected incumbent/local-best.
2. Create bounded variants: micro/base/pinpoint Gaussian mutation, structured layer-wise
   mutation, or crossover between two strong anchors.
3. Evaluate variants against the frozen analytic/anchor opponent.
4. Accept only variants with held-out gain.
5. Feed the accepted variant into on-policy gradient training.
6. Publish only after the normal protected local-best gate passes.

Bad role:

- Replacing gradient learning.
- Publishing the best one-game mutation.
- Comparing candidates only to other noisy candidates.
- Averaging several workers so a rare breakthrough is smoothed away before it can train.
- Treating league rank as proof of policy improvement.

Population search should widen the basin; gradient RL should climb inside the new basin.

## Promotion And Rollback Rules

Use two separate decisions:

1. **Not good enough to publish.** Candidate did not beat the protected local-best; keep
   training or hold it as exploratory.
2. **Bad enough to restore.** Candidate is clearly worse; restore protected local-best or
   reduce exploration.

Do not collapse these into one gate. A candidate can be not publishable but still worth
training for another window.

Promotion should require:

- Minimum held-out sample size.
- HOME neural objective in analytic-opponent mode.
- Comparison against protected incumbent/local-best.
- No regression in play quality.
- No surge in conceded shots/SOT or backward-pass turnovers.
- Optional Wilson/ELO lower-bound check for win-rate style metrics.

## Order Of Operations

When the learner is flat or regressing, follow this order:

1. Confirm the live viewer is serving the intended protected weights.
2. Confirm trainer is local-only unless RDS is explicitly needed.
3. Confirm neural is authoritative, warm, and on-policy.
4. Confirm dense rewards reach the value net.
5. Confirm the reward ceiling values goals/SOT/high-xG actions above pass chains.
6. Train against frozen analytic/anchor opponents.
7. Add or inspect the missing causality counters: net-changed action rate, selected
   ConfidenceGate-open rate, selected kick-power bucket entropy, old-prob missing rate,
   PPO/MAPPO ratio and clip stats, and target/advantage histograms.
8. Turn on learned world-model training.
9. Turn on trainer-side shallow neural MCTS/lookahead.
10. Run paired MCTS-on/off ablations over the same fixed seeds before increasing planning budget.
11. Add controlled action exploration, preferably bucket sampling/temperature/entropy rather than
   relying on `DiscretizedKickDither` (currently zero-offset).
12. Use population perturbation only when gradient/on-policy windows are flat.
13. Promote only if held-out local-best gates pass.
14. If still flat, increase representation/capacity and add embedding retrieval.

## Current Local Interpretation

For local `localhost:5055` work, a candidate window can improve while the viewer still looks
unchanged. That is expected when:

- `SOCCER_LIVE_POLICY_INCLUDE_UNPROMOTED=0`
- no `checkpoint_local_best_published` row exists for the active run
- only `.candidate` artifacts have newer exploratory weights

In that case, report honestly: "candidate/exploration is moving, protected live weights have
not improved yet." Do not claim the viewer reflects learning until protected weights publish
or the viewer is intentionally relaunched with candidate weights for inspection.

## What Success Looks Like

The plateau is broken only when all of these are true:

1. Five or more consecutive held-out checkpoint windows climb, or a larger held-out eval has
   a clear confidence-bounded improvement.
2. The candidate beats the protected local-best, not just a noisy self-play opponent.
3. Behavior improves in football terms: more dangerous shots, fewer bad turnovers, better
   goal-entry choices, and fewer sterile pass chains.
4. World-model/planning metrics are stable if neural MCTS is active.
5. The improved policy publishes to protected live weights, and `localhost:5055` loads those
   weights.

Until then, call it what it is: exploration, recovery, or candidate movement. Climb requires
protected, held-out, behavior-visible improvement.
