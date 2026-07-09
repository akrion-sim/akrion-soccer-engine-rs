# Current Learning State

This is the source-backed status snapshot for the soccer learning and planning stack.
It separates what the code does now from diagnostics and roadmap items that are not
implemented yet.

## Implemented Now

- The runtime decision stack is hybrid: MDP/POMDP observations and analytic beliefs
  generate legal options, neural policy/value and tabular/retrieval priors score them,
  bounded MCTS/PUCT may rerank a capped candidate set, and per-player MPC executes or
  reconciles the chosen intent.
- `SOCCER_POLICY_ACTIONS` is an append-only actor vocabulary with `113` labels. `40`
  of those labels are kick-power bucket variants: `pass-kp0..9`, `aerial-pass-kp0..9`,
  `shoot-kp0..9`, and `first-time-shot-kp0..9`.
- Discretized kick-power candidates are live in the MCTS expansion path behind
  `DD_SOCCER_ENABLE_DISCRETIZED_KICK` and the `SOCCER_NEURAL_MCTS_*KICK*` knobs. The
  selected bucket is preserved in labels such as `pass1-kp7`, and scoring/execution can
  use the bucket's power.
- `DiscretizedKickDither::sample` currently returns zero offsets. It is a bounded
  placeholder, not exploration and not a source of bucket-crossing credit.
- `SoccerNeuralBlendMode::ConfidenceGated` blends toward the neural estimate only when
  the candidate's visit count is below `min_confidence_visits`; otherwise it keeps the
  tabular/analytic value for that candidate.
- `behavior_policy_probability` is carried on decisions and exported into trajectory
  rows, with deterministic fallback behavior when the probability is absent.
- `SOCCER_TRAJECTORY_MAX_DECISION_GAP_TICKS` is `secs_to_ticks(0.3)`. It links near-tick
  delayed outcomes; it is not a long build-up credit window.
- The current `world_model_training` line reports planning decisions, MCTS selection
  and candidate-family shares, MPC replan rates, priority-sample rates, distillation
  rates, and policy entropy.

## Missing Diagnostics

These are useful next counters, but they are not current telemetry:

- `net_changed_action_rate`: how often the learned/planner score changed the executed
  action compared with the tabular/heuristic baseline.
- `confidence_gate_open_rate`: how often ConfidenceGated actually opened for the
  candidate that survived to execution.
- `selected_kick_speed_bucket_entropy`: whether selected kick-power buckets collapse
  early or remain diverse by family.
- behavior-probability diagnostics such as missing-rate, mean old probability,
  PPO/MAPPO ratio mean, ratio clip fraction, and entropy by role/action family.
- target and advantage diagnostics such as raw/scaled target histograms, clip fraction,
  advantage mean/std by role, and positive-advantage rate.
- a frozen eval ladder that compares candidates against the analytic baseline, protected
  local best, past checkpoints, and weaker/randomized opponents on held-out seeds.

## Future Work

- Coarsen confidence for action keys where the current factored labels fragment visits,
  while keeping fine parameters in neural features and labels.
- Add stochastic bucket sampling or temperature/epsilon exploration over candidate buckets,
  plus entropy regularization and bucket-entropy telemetry. Do not describe dither as this
  mechanism; it is not.
- Use paired MCTS-on/off ablations (`SOCCER_NEURAL_MCTS_ENABLED=0` or
  `SOCCER_DISABLE_NEURAL_MCTS=1`, depending on the runner path) to test whether MCTS is
  exposing useful actions or hiding actor causality behind analytic-looking choices.
- Train and evaluate against a pool of frozen checkpoints so mirror self-play parity does
  not hide absolute improvement.
- Add small-sided or high-event curriculum lanes if full 11v11 reward remains too sparse
  or noisy to prove policy improvement quickly.
