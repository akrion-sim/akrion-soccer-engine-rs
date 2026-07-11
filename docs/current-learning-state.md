# Current Learning State

This is the source-backed status snapshot for the soccer learning and planning stack.
It separates what the code does now from diagnostics and roadmap items that are not
implemented yet.

## Implemented Now

- `DD_SOCCER_ENABLE_HERMETIC_NEURAL_CRITIC` removes the aliased tabular Q from
  neural critic targets. It implies a full-weight neural successor bootstrap and
  uses zero only for cold start or terminal states; the tabular policy remains
  available for separately gated behavior/fallback paths, but cannot leak into
  this critic target.
- Reward events now cross one typed calibration seam before entering learning.
  `DD_SOCCER_REWARD_KIND_SCALES` accepts bounded, sign-safe per-kind utility
  multipliers such as `CompletedForwardPass=0.8,BadPassChainPenalty=1.3`.
  Event facts and attribution stay unchanged; only learner utility changes.
  This is the deployment seam for an outer held-out optimizer to publish tuned
  weights. Do not update these weights from the same trajectories they score:
  tune on disjoint match batches, freeze for a training window, and accept a
  calibration only through the existing held-out WDL/Wilson/goal-difference
  promotion gate.
- `scripts/optimize_reward_calibration.py` is the persistent outer loop for that
  seam. It uses cross-entropy-method proposals in bounded log-scale space,
  freezes each proposal for a complete analytic-opponent training window, and
  evaluates on a disjoint analytic seed range. Its authoritative objective is
  the held-out Wilson lower bound, with small goal-difference tie-breaking and
  a turnover-regression tax. Every proposal, seed, snapshot, log, and verdict is
  stored under the requested output directory; `--resume` continues the audit
  trail instead of restarting or silently selecting on training reward.

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
- `DiscretizedKickDither::sample` now produces bounded inside-bucket release offsets,
  and production pass release uses it only when
  `DD_SOCCER_ENABLE_DISCRETIZED_KICK_DITHER` is enabled. This is release robustness
  and local exploration inside the chosen bucket, not a separate bucket-choice policy.
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
- Forward-pass climb is not judged by raw volume alone. The active eval/league path
  tracks completed forward passes, pass turnovers, and net forward passes
  (`completed_forward_passes - pass_turnovers`). Promotion still requires scoreline
  protection: the eval gate requires the normal WDL/Wilson verdict plus a positive
  held-out goal-difference floor by default, and the league checkpoint validation
  requires non-negative validation goal difference by default.
- `DD_SOCCER_FORWARD_PASS_CLIMB_CURRICULUM` keeps executable learned pass choices
  from being replaced by the option-score safety override during the forward-pass
  curriculum. Receiver selection also has a net-forward quality floor through
  `SOCCER_LEARNED_PASS_RECEIVER_MIN_NET_FORWARD_QUALITY`, so a high receipt-only
  turnover trap does not win just because the ball can arrive.
- `DD_SOCCER_ENABLE_DEFERRED_PASS_CREDIT` is implemented and enabled by the
  all-gates/continuous training path. When on, delayed pass completion and
  turnover credit is back-dated to the decision transition that launched the pass
  instead of being assigned only to the later reception/interception tick.
- Pass-target MCTS pruning can be inspected with `DD_SOCCER_DUMP_MCTS_PASS_TARGET_DIAG`.
  The diagnostic reports whether the first forward target was inside the configured
  target cap, pruned by the cap, or absent.
- `DD_SOCCER_DUMP_PASS_CAND_DIAG` separately reports pass candidate availability,
  cap survival, and a nearest-defender openness proxy before value/MPC selection.
- `DD_SOCCER_FWD_TRACE` emits per-decision forward-pass action-causality rows:
  candidate exposure, root survival, net-changed family/label, selected family,
  selected kick bucket, forward-root injection survival/selection, and tabular
  argmax comparison. `scripts/fwd_trace_report.py` summarizes these into forward
  exposure, injection, selection, action-change, and selected kick-bucket entropy
  rates for proof artifacts.
- Dimensionality experiments are now explicit and mutually separable. The hard
  projection gate `SOCCER_NEURAL_COMPACT_INPUTS=1` / `DD_SOCCER_NEURAL_COMPACT_INPUTS=1`
  trains a 100-input value critic from the first 90 core features plus the 10
  action-parameter features, and is only a destructive diagnostic. The learned
  bottleneck gate `SOCCER_NEURAL_BOTTLENECK_DIM=100` keeps the full current critic
  input and adds a learned 100-unit latent before the normal hidden/value layer,
  matching the cheap `620 -> 100 -> hidden -> value` compression experiment Claude
  recommended after finding no near-dead critic inputs.

## Missing Diagnostics

These are useful next counters, but they are not current telemetry:

- `confidence_gate_open_rate`: how often ConfidenceGated actually opened for the
  candidate that survived to execution.
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
  plus entropy regularization and bucket-entropy telemetry. The current dither jitters
  only the release inside a selected bucket; it does not replace bucket-choice exploration.
- Use paired MCTS-on/off ablations (`SOCCER_NEURAL_MCTS_ENABLED=0` or
  `SOCCER_DISABLE_NEURAL_MCTS=1`, depending on the runner path) to test whether MCTS is
  exposing useful actions or hiding actor causality behind analytic-looking choices.
- Run the ceiling-break proof profile with
  `DD_SOCCER_ENABLE_FORWARD_RELEASE_ROOT_CANDIDATE=1` and
  `SOCCER_NEURAL_MCTS_MIN_PASS_LIKE_ROOT_CANDIDATES=1` so forward options hidden
  behind the analytic pass-target cap reach the same neural-scored pass-space and
  kick-power candidate surface as ordinary pass targets, then are measured by the
  forward trace.
- Keep `RUN_ANALYTIC_EVAL=1` in the overnight proof harness. The ceiling-break
  verdict must include `full-vs-analytic.aggregate.txt` and its forward trace, not
  only `full-vs-fresh`; beating a fresh net is not the same as escaping analytic
  parity.
- Compare `BOTTLENECK_INPUT_PROFILE=1` against the full-input and
  `COMPACT_INPUT_PROFILE=1` proof arms on the same held-out seeds before claiming
  anything about input reduction. These are optional diagnostics, not the main
  ceiling-break path. The expected useful signal is not raw parameter count; it is
  whether the learned 100-d latent improves or preserves forward-pass, turnover,
  shots-on-frame, dribble, and goal guardrails better than the destructive compact
  projection.
- Train and evaluate against a pool of frozen checkpoints so mirror self-play parity does
  not hide absolute improvement.
- Add small-sided or high-event curriculum lanes if full 11v11 reward remains too sparse
  or noisy to prove policy improvement quickly.
