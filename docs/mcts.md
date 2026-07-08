# MCTS in the Soccer Engine

This engine uses MCTS as a small, bounded action reranker. It is not a full
match-tree simulator, not the primary learner, and not a replacement for the
MDP/POMDP policy, neural actor/critic, approximate DP, or MPC execution layer.

The useful mental model is:

```text
fresh field vector + belief
  -> MDP/POMDP legal action options
  -> tabular Q / actor / critic / retrieval / MPC priors score candidates
  -> neural MCTS reranks a tiny candidate set
  -> MPC checks and reconciles physical execution
  -> transition/reward data trains the neural and DP layers
```

## Where It Lives

Main implementation:

- `src/des/general/soccer/world.rs`
  - `neural_mcts_action_from_candidates`
  - `neural_mcts_expanded_candidate_plans`
  - `neural_mcts_root_candidates`
  - `ensure_neural_mcts_root_family_candidates`
  - `dribble_selection_floor_rank`
  - `technical_selection_floor_rank`
  - `discretized_kick_selection_floor_rank`
  - `maybe_log_neural_mcts_dribble_diagnostic`
- `src/des/general/soccer.rs`
  - `SoccerDecisionContext`
  - `AgentDecisionTrace`
  - `MatchStats`
  - `SoccerPlanningValidationStats`
- `src/bin/main_soccer_learning_run.rs`
  - trainer env wiring for `SOCCER_NEURAL_MCTS_*`
  - `world_model_training` telemetry
  - learning-action outcome summaries

The local strict launcher currently sets the active trainer-side MCTS knobs in:

- `out/local-learning/soccer-learning-resume-anchor-strict-launch.sh`

## What MCTS Searches

MCTS searches over already-scored candidate actions. The candidate source is the
normal learned-policy/action-ranking path, not a cloned full world simulation.

The search can expand some action families:

- Passes: top visible/threaded pass targets, with optional discretized kick-power
  buckets.
- Aerial passes: visible aerial targets, with aerial kick-power buckets.
- Shots and first-time shots: shot target plus optional shot-power buckets.
- Dribbles: deterministic technical variants such as `carry-forward`,
  `carry-out-left`, `carry-out-right`, `left-cut`, `right-cut`, `side-step`,
  `protect-ball`, `open-passing-lane`, `nutmeg`, `xavi-turn`,
  `fake-left-cut-right`, and `fake-right-cut-left`.

Candidate plans are filtered through feasibility checks. Pass and dribble
families use MPC execution probability. Shot buckets require finite shot-quality
priors. This is deliberate: MCTS may propose bolder choices, but MPC is still the
physical guardrail.

## How Selection Works

`neural_mcts_action_from_candidates` does a shallow PUCT-style rerank:

1. Cap the root set by `mcts_candidates`.
2. Compute candidate priors from existing policy scores and safe finite priors.
3. Convert score span into a normalized value estimate.
4. Run `mcts_simulations` lightweight visits over the candidate nodes.
5. Sort by visits, mean value, raw score, Q visits, then label.
6. Apply optional family floors for discretized kicks, dribbles, or technical
   actions.
7. Return a `SoccerPolicyActionChoice` with telemetry counts.

This means the search is local and cheap. It can change which candidate wins, but
it cannot invent arbitrary match states or freely optimize continuous controls.

## Relationship to MPC

MCTS chooses among tactical ideas. MPC executes or vetoes/reconciles the idea.

The MCTS layer uses MPC in three ways:

- Candidate scoring: pass, shot, dribble, and pitch-value bonuses include MPC/QP
  feasibility signals.
- Safety filtering: plans that need a hard MPC replan or fall below minimum safe
  execution probability are not treated as clean candidates.
- Distillation/replan traces: when MCTS improves a policy choice, the trace can
  become learning data, but only if the replacement is close enough in score and
  physically executable.

So MCTS is not "neural nets solving MPC." It is neural/world-model/value planning
over candidate futures, with MPC as the feasibility and execution guard.

## Relationship to MDP/POMDP and Neural Learning

The MDP/POMDP stack still supplies the legal action vocabulary and belief-state
context. The neural actor/critic still learns from realized transitions. MCTS
only reranks candidates after those systems have produced the state-conditioned
options.

Important consequences:

- If the field vector, belief state, or latest weights are stale, MCTS will only
  rerank stale options.
- If a useful action family is never generated, MCTS cannot select it.
- If a family is generated but consistently low-scored, MCTS diagnostics should
  show a candidate present with a poor score gap.
- If a family is selected but later reconciled away by MPC/safety, the candidate
  counters still need to remain visible.

That last point is why candidate telemetry is separate from
`neural_mcts_selected`. A strict selected flag answers "did this exact MCTS
choice execute?" Candidate counters answer "did MCTS even see this family?"

## Key Gates and Knobs

Core enablement:

- `SOCCER_NEURAL_MCTS_ENABLED`
- `SOCCER_NEURAL_MCTS_SIMULATIONS`
- `SOCCER_NEURAL_MCTS_CANDIDATES`
- `SOCCER_NEURAL_MCTS_DEPTH`
- `SOCCER_NEURAL_MCTS_MODEL_WEIGHT`
- `SOCCER_NEURAL_WORLD_MODEL`

Candidate expansion:

- `SOCCER_NEURAL_MCTS_PASS_TARGET_CANDIDATES`
- `SOCCER_NEURAL_MCTS_KICK_POWER_CANDIDATES`
- `SOCCER_NEURAL_MCTS_MIN_DISCRETIZED_KICK_CANDIDATES`
- `DD_SOCCER_ENABLE_DISCRETIZED_KICK`
- `SOCCER_NEURAL_MCTS_AERIAL_PASS_ENABLED`

Family protection and exploration:

- `SOCCER_NEURAL_MCTS_PASS_LIKE_NON_PASS_MARGIN`
- `SOCCER_NEURAL_MCTS_MIN_PASS_LIKE_ROOT_CANDIDATES`
- `SOCCER_NEURAL_MCTS_MIN_DRIBBLE_ROOT_CANDIDATES`
- `SOCCER_NEURAL_MCTS_FAMILY_FLOOR_MAX_SCORE_REGRESSION`
- `SOCCER_NEURAL_MCTS_DISCRETIZED_KICK_SELECTION_FLOOR`
- `SOCCER_NEURAL_MCTS_DISCRETIZED_KICK_SELECTION_MAX_SCORE_REGRESSION`
- `SOCCER_NEURAL_MCTS_DRIBBLE_SELECTION_FLOOR`
- `SOCCER_NEURAL_MCTS_DRIBBLE_SELECTION_MAX_SCORE_REGRESSION`
- `SOCCER_NEURAL_MCTS_TECHNICAL_SELECTION_FLOOR`
- `SOCCER_NEURAL_MCTS_TECHNICAL_SELECTION_MAX_SCORE_REGRESSION`

MPC and scoring priors:

- `SOCCER_NEURAL_MCTS_PASS_MPC_PRIOR_WEIGHT`
- `SOCCER_NEURAL_MCTS_TECHNICAL_MPC_EXECUTION_WEIGHT`
- `SOCCER_NEURAL_MCTS_MIN_REPLAN_SAFE_EXECUTION_PROBABILITY`
- `SOCCER_NEURAL_MCTS_DISCRETIZED_KICK_BASE_VALUE_WEIGHT`
- `SOCCER_NEURAL_MCTS_PITCH_VALUE_CANDIDATE_WEIGHT`
- `SOCCER_NEURAL_MCTS_PITCH_VALUE_COUNTER_WEIGHT`

Diagnostics:

- `SOCCER_NEURAL_MCTS_DRIBBLE_DIAGNOSTIC_INTERVAL`

## Telemetry to Watch

The trainer's `world_model_training` line reports:

- `neural_mcts_selection_rate`
- `neural_mcts_discretized_kick_rate`
- `neural_mcts_technical_action_rate`
- `neural_mcts_discretized_kick_candidate_share`
- `neural_mcts_root_discretized_kick_candidate_share`
- `neural_mcts_dribble_candidate_share`
- `neural_mcts_root_dribble_candidate_share`
- `neural_mcts_discretized_kick_candidate_set_rate`
- `neural_mcts_root_discretized_kick_candidate_set_rate`
- `neural_mcts_dribble_candidate_set_rate`
- `neural_mcts_root_dribble_candidate_set_rate`
- `mpc_replan_neural_mcts_rate`
- `neural_mcts_distillation_rate`

The sampled `neural_mcts_dribble_diagnostic` line reports:

- total candidates
- dribble candidates
- root candidates
- root dribble candidates
- eligible root dribbles
- best score
- best dribble score
- best root dribble score
- dribble score gap
- floor-selected rank, if any
- selected rank and label
- whether the selected label is dribble-like

Use these rows to classify failures:

- `dribble_candidates=0`: candidate generation did not offer dribble actions.
- `dribble_candidates>0` and `root_dribble_candidates=0`: root pruning removed
  the family.
- `root_dribble_candidates>0` and `eligible_root_dribbles=0`: safety/regression
  gates made them ineligible.
- `eligible_root_dribbles>0` but `selected_is_dribble=0`: scoring/PUCT still
  preferred another family.
- `selected_is_dribble=1` but action outcomes show no dribble execution:
  reconciliation, MPC, or final action labeling swallowed the selected plan.

## Current Local Plateau Use

For the local plateau-resolution runs, MCTS is intentionally a secondary
planning/diagnostic layer, not the primary learning loop.

The primary learning direction remains:

- neural-authoritative policy/value learning
- approximate DP/Bellman replay
- analytic-difference reward
- pitch-value reward
- skill heads and discretized action ownership
- held-out HOME-vs-analytic promotion gates

MCTS is useful when it helps answer:

- Did the policy have access to better candidates?
- Did the learned value prefer them?
- Did MPC reject them?
- Did the actor receive distillation or outcome credit for the improved choice?

MCTS is harmful if it hides those answers, over-selects safe analytic-looking
actions, or turns into a noisy second policy that the actor never learns to
imitate. If climb stalls with MCTS enabled, run a paired ablation with
`SOCCER_NEURAL_MCTS_ENABLED=0` over the same fixed seeds and compare:

- HOME-vs-analytic fitness
- score/shots/SOT
- policy entropy
- action-family distribution
- dribble/pass/shot delayed outcome rewards
- MPC replan rates
- world-model validation loss

## When to Gate It Off

Gate MCTS off when:

- the world model is cold or non-finite;
- MCTS selection rate rises but held-out HOME-vs-analytic fitness falls;
- candidate-family telemetry shows the search is mostly reranking one dominant
  family;
- MPC replan rates spike after MCTS-selected actions;
- the actor does not receive enough distillation/outcome samples to learn the
  searched behavior.

Keep it on when:

- it exposes candidate families the actor should learn;
- held-out confirmation improves, not just training-window score;
- selected actions remain MPC-executable;
- distillation samples are nonzero and behavior improves over multiple windows.

## Practical Debug Checklist

1. Confirm the current launcher logs `soccer_local_training_mcts_gate`.
2. Confirm `SOCCER_NEURAL_MCTS_ENABLED=1` only for MCTS-on experiments.
3. Confirm `world_model_training` includes dribble/root-dribble candidate shares.
4. Inspect `neural_mcts_dribble_diagnostic` for generation vs pruning vs scoring
   failures.
5. Compare `learning_action_outcomes` against MCTS telemetry. Candidate presence
   without action outcomes means execution or labeling is still blocking the
   learning signal.
6. Do not promote based on an MCTS training spike alone. Promotion still needs
   held-out HOME-vs-analytic confirmation.

