# Soccer AI architecture audit (policy / neural / planning layer)

Grounded in the actual `soccer-sim-game-engine.rs` code (not a generic guess). The
trigger was an external design audit that assumed a "vanilla feedforward NN mapping
snapshot → action scores." That premise is **out of date for this engine**: the NN is
not the solver. The engine is a **hybrid heuristic + tabular-RL + feedforward-neural-head**
system in which the learned layer **re-ranks a small heuristic candidate set**
(`SOCCER_RETRIEVAL_PRIOR_RERANK_LIMIT = 12`, stochastic top-k), executed by MPC.

## What the engine already has (vs the 8 audit checkpoints)

| # | Audit concern | Reality in this engine | Where |
|---|---|---|---|
| 1 | Stateless NN (no POMDP belief)? | **Belief exists** (analytic, not recurrent): opponent press beliefs, FOV vision-gating + Kalman confidence, perception-noise / occlusion (gated), intended-pass-target belief | `update_opponent_press_beliefs`, `opponent_belief_enabled`, `INTENDED_PASS_TARGET_BELIEF_CONFIDENCE` (world.rs) |
| 2 | Probs vs Q-values? | Both + tabular value: `FeedForwardNetwork` policy/value heads over a tabular `value_micros` retrieval prior | soccer.rs neural heads + `des_soccer_learning_policy_entries` |
| 3 | Behavior prob stored for PPO? | **Yes** — `old_action_probability`, comment: *"PPO/MAPPO importance weighting must use THIS as the behaviour [prob]"* | soccer.rs:6744, 34136 |
| 4 | Reward too sparse? | **No — densely shaped**: `PITCH_VALUE` (xT-like), `REWARD_PER_YARD`, half-field weighting, `REWARD_MIN_PRESSURE`, possession progression, `EFFORT_REWARD`, shot/pass ladders | soccer.rs reward consts |
| 5 | 11 independent NNs? | **No — MARL/MAPPO**: team reward weight 0.35 / intermediate 1.0, centralized lane-discipline, balanced-team component, MAPPO feature channels | `DEFAULT_SOCCER_MARL_TEAM_REWARD_WEIGHT`, `field_numbers` |
| 6 | Player relationships modeled? | **Engineered, not learned**: 22-body `field_numbers` vector, relational shape (neighbors), lane affinity, nearest-opponent/marking | `mod field_numbers`, `RELATIONAL_SHAPE_NEIGHBORS` |
| 7 | Sequence search? | **Bounded and opt-in**: analytic lookahead (pass-lane 2s, reactive 1.4s, energy 10s) + MPC are always available; neural MCTS/PUCT can rerank a capped set of legal candidates when `mcts_enabled` is on. There is still no full match-state tree search. | `SoccerNeuralBlendConfig::{mcts_*}`, `neural_mcts_action_from_candidates`, `*_LOOKAHEAD_SECONDS` consts |
| 8 | MPC = execution only? | **Yes (correct)** — MPC/QP gates ball control + pass execution; tactics decided upstream | `bounded_accel_qp_control_fit`, "MPC pass execution" |

## What is genuinely missing (the audit's valid points)

1. **Learned representation.** The heads are `FeedForwardNetwork` MLPs over **hand-crafted**
   features (`SOCCER_NEURAL_BASE_FEATURE_DIM = 192` + appended channels). No learned
   GNN/attention over the player graph; no learned RNN/Transformer temporal belief. The
   relational + temporal signal is *feature-engineered in*.
2. **Richer learned-model search.** The engine now has a shallow, capped neural MCTS/PUCT
   reranker over already-legal candidates. What is still missing is MuZero/POMCP-style full-state
   belief search or a deep learned transition rollout over the whole match state.

## The reality check the external audit missed (decisive here)

This engine runs **22 agents at a 66.7 ms / tick (15 Hz) soft-real-time budget** — the whole
recent perf effort was keeping ticks under that. A per-tick **GNN + RNN + MCTS** for 22
players would blow the budget by 10–100×. The current design — *engineered relational/belief
features + lightweight MLP heads + analytic lookahead + MPC* — is largely a **deliberate
response to that constraint**. The cited references (MuZero, MADDPG, GRF-GNN) are
offline / turn-based / batch-trained; none carries a 15 Hz × 22-agent inference budget.
So "upgrade the neural net" must be read as "upgrade *without* adding per-tick learned
graph/recurrent/tree-search cost."

## Pragmatic roadmap (ranked by value / effort for THIS engine)

1. **Keep MDP/POMDP + MPC + heuristic re-rank.** Right backbone for real-time. No change.
2. **Small learned relational encoder, offline-distilled (captures audit #1 at ~zero runtime cost).**
   - Train a tiny GNN/attention encoder *offline* on the existing engineered features +
     logged transitions (`des_soccer_learning_run_deltas`).
   - **Distill** its output into the *same MLP head input shape* the engine already consumes,
     so runtime inference cost is unchanged (still one dense forward pass).
   - First step: export a feature+outcome dataset from PG; train encoder→head offline;
     ship only the distilled dense weights through the existing neural-snapshot path
     (append at tail, zero-pad old snapshots — already supported, see "new variables").
3. **Keep bounded MCTS/PUCT on the top-k only (captures audit #7 within budget).**
   - Already implemented as an opt-in reranker over legal candidates, with hard caps on
     candidates, simulations, and depth.
   - Do not widen it into team MPC or full match-tree search; deepen only through cheap
     one-step/world-model value estimates that preserve the tick budget.
4. **Lean into the CTDE you already have.** Strengthen the centralized critic *offline*
   (MAPPO team reward already wired); runtime stays decentralized per-player.
5. **New-variable discipline (already-correct pattern).** Append features at the tail and
   zero-pad old neural snapshots (engine already migrates this way — soccer.rs:3675/3685).
   Keep new variables OUT of the tabular `state_key` (`soccer_policy_postgres_state_key`)
   unless you are prepared to re-accumulate that experiment, because changing the key
   encoding orphans existing tabular entries.

## Notes / landmines

- `visits` in `des_soccer_learning_policy_entries` is **saturated at i32-max** for most rows
  (overflow bug), so `SOCCER_LIVE_POLICY_MIN_VISITS` can't trim the policy load — gen-308 is
  1.32M entries with no usable visit filter. Worth fixing (i64 / capped counter) to enable
  fast partial loads.
- Older live-generation examples in this file may drift quickly. For current generation health,
  query `des_soccer_learning_policy_versions` / `des_soccer_learning_runs` for the active
  experiment instead of trusting a dated snapshot.

## Offline distilled-encoder harness — status

**Step 1 (dataset export): DONE.** `scripts/export_offline_dataset.sh` streams a
supervised `(state → action value)` dataset from `des_soccer_learning_run_deltas`
to JSONL (filtered by the experiment's recent `run_id`s — the indexed FK path; the
created_at/experiment 3-way join over the 10M-row table times out). Verified on
overnight gen-308: **47 distinct actions, 205 binned engineered features
(int/bool/str), value target ≈ [-7, 11], visit-weighted.**

**Step 2 (offline train + distill): planned.**
1. Flatten `state_key` → fixed feature vector: one-hot the categorical/bool bins,
   scale the int bins; freeze the column order as the canonical feature layout.
2. Train a small relational/attention encoder → value (and per-action policy) head
   offline, sample-weighted by `weight_micros` (effective visits). Compare held-out
   value MSE vs the tabular baseline.
3. **Distill** the encoder→head into the engine's existing `FeedForwardNetwork`
   input shape (dense forward pass only), so runtime per-tick cost is unchanged.
   Ship via the neural-snapshot path; append new channels at the tail (zero-pad old
   snapshots — already supported) so it composes with the live policy.
4. A/B the distilled head behind a gate (default-OFF) using the existing
   `soccer_eval_gate` (held-out Elo / cross-play) before promotion.
