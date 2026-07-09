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

Pass-target expansion is capped by `SOCCER_NEURAL_MCTS_PASS_TARGET_CANDIDATES`.
The current forward-pass climb work uses `8` in the strict launcher/continuous
manifest. This is not an unbounded receiver search; if a forward target ranks
below the cap, MCTS never sees it unless the candidate generator or cap changes.

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
- `DD_SOCCER_DUMP_MCTS_PASS_TARGET_DIAG`
- `DD_SOCCER_DUMP_PASS_CAND_DIAG`

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

It does **not** yet report the causality counters needed to prove that the learned/planning layer
is changing play rather than only observing candidates. Add these before using MCTS telemetry as a
plateau diagnosis:

- `net_changed_action_rate` - selected/executed action differed from the tabular/heuristic
  baseline.
- `confidence_gate_open_rate` - ConfidenceGated opened for the candidate that survived to
  execution.
- `selected_kick_speed_bucket_entropy` - selected kick-power buckets stayed diverse instead of
  collapsing into one bucket/family.

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

The `mcts_pass_target_diag` line reports the target-cap path for pass receivers:

- `reached`: number of diagnostic call sites reached.
- `forward_inside_cap`: a forward target existed inside the active target cap.
- `forward_pruned_by_cap`: a forward target existed, but only after the cap.
- `no_forward_candidate`: no forward target was found in the ranked target list.
- `avg_best_forward_rank_1based`: average rank of the first forward target when present.
- `cap`: the active `SOCCER_NEURAL_MCTS_PASS_TARGET_CANDIDATES` limit.

Use this row before blaming value learning. If `forward_pruned_by_cap` is high,
MCTS is not refusing forward passing; it is never seeing the forward receiver.
If `forward_inside_cap` is high but forward passes still do not execute, inspect
policy score gaps, safety/MPC reconciliation, and action-outcome credit next.

The `pass_cand_diag` line is a broader candidate-availability read. It reports:

- `any_fwd_precap`: decisions where any ranked target is forward before the cap.
- `any_fwd_postcap`: decisions where a forward target survives into expansion.
- `any_fwd_OPEN`: decisions where at least one forward target has no opponent within
  the nearest-defender openness proxy.
- target mix percentages for forward, lateral, and backward targets.
- open-vs-marked percentages among forward targets.

Use `pass_cand_diag` to decide whether the bottleneck is off-ball availability,
candidate exposure, or target marking. Use `mcts_pass_target_diag` when the question
is specifically whether the first forward receiver rank falls inside the configured
`SOCCER_NEURAL_MCTS_PASS_TARGET_CANDIDATES` cap.

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
`SOCCER_NEURAL_MCTS_ENABLED=0` or `SOCCER_DISABLE_NEURAL_MCTS=1` over the same fixed seeds
and compare:

- HOME-vs-analytic fitness
- score/shots/SOT
- policy entropy
- action-family distribution
- selected kick-power bucket distribution
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
4. Inspect `neural_mcts_dribble_diagnostic`, `pass_cand_diag`, and
   `mcts_pass_target_diag` for generation vs pruning vs scoring failures.
5. Compare `learning_action_outcomes` against MCTS telemetry. Candidate presence
   without action outcomes means execution or labeling is still blocking the
   learning signal.
6. If diagnosing a plateau, add or inspect the missing causality counters:
   net-changed action rate, ConfidenceGated selected-candidate open rate, and selected
   kick-power bucket entropy. Without them, MCTS can look busy while the executed policy
   remains analytic.
7. Do not promote based on an MCTS training spike alone. Promotion still needs
   held-out HOME-vs-analytic confirmation.
