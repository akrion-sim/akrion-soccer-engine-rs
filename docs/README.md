# Akrion Soccer Engine Docs

These docs describe the current intent of the engine: football behaviour is chosen from
POMDP/MDP observations, enriched by whole-field vectors for the ball and all 22 players,
learned through self-play/MARL-style reward signals, and executed through per-player MPC.
The code is the source of truth; generation numbers, cluster state, and learned snapshot
metrics should be verified live before quoting them.

## Maintenance contract

Before changing docs, verify claims against the code or live system when they involve:

- feature dimensions, neural input layouts, or append-only tail blocks;
- gate names/defaults and whether a head is only trained, only persisted, or consumed live;
- cluster generation numbers, pod health, rollout state, or tournament cadence;
- whether planning is analytic, bounded MCTS/PUCT, full-state search, or MPC execution.

Prefer current constants, env names, inspect endpoints, and DB queries over dated examples.

## Start here

- [learning-architecture.md](learning-architecture.md) - end-to-end self-play, learner,
  auxiliary heads, persistence, and serving flow.
- [architecture-audit.md](architecture-audit.md) - what is live, what is bounded/gated,
  and what is still not a full search/planning stack.
- [flowchart.md](flowchart.md) - high-level runtime and learning flow.
- [runbook/gameplay-integrity-audit.md](runbook/gameplay-integrity-audit.md) - gameplay
  integrity checks and audit notes.

## Tactical decision models

- [position-mdp-pomdp-decisions.md](position-mdp-pomdp-decisions.md) - each assigned
  position's MDP/POMDP decision contract.
- [position-decision-space.md](position-decision-space.md) - how the 11 positions map onto
  role/home-position predicates and legal action families.
- [back-four-line-model.md](back-four-line-model.md) - back-four and midfield line-depth
  model, 169-feature field vector, and gated live head consumption.
- [reward-shaping-audit.md](reward-shaping-audit.md) - reward channels and shaping risks.
- [learning-design.md](learning-design.md) - broader learning-design notes and constraints.

## Ops and live learning

- [live-runner-weights.md](live-runner-weights.md) - loading neural/tactical weights into a
  live runner and verifying them through inspect endpoints.
- [observability.md](observability.md) - metrics and runtime visibility.
- [thread-model.md](thread-model.md) - threaded learner/runtime model.
- [../k8s/LEARNING_RUNBOOK.md](../k8s/LEARNING_RUNBOOK.md) - cluster learning and tournament
  operations.

## Plans and backlog

- [learnability-conversion-roadmap.md](learnability-conversion-roadmap.md) - conversion path
  from heuristics to trainable heads.
- [remaining-next-steps.md](remaining-next-steps.md) - active follow-up work.
- [future.md](future.md) - longer-range ideas.
- Focused plans: [crash-box-plan.md](crash-box-plan.md),
  [blindside-steal-plan.md](blindside-steal-plan.md),
  [discretized-kick-action-plan.md](discretized-kick-action-plan.md),
  [parallel-decision-plan.md](parallel-decision-plan.md),
  [pass-risk-appetite-ab-plan.md](pass-risk-appetite-ab-plan.md), and
  [learning-rate-audit.md](learning-rate-audit.md).
