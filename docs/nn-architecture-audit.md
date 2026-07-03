# Neural-architecture audit (soccer-sim-game-engine)

Audit prompted by an external design review of "vanilla NN + MDP/POMDP". That review
was written **without access to this codebase** and assumed a single stateless MLP mapping
state → action scores. This document records what is actually implemented, maps the review's
eight checkpoints to real code, and tracks the one genuine architectural gap that was closed
plus the one deferred.

## Verdict

The engine is **not** a single feedforward action scorer. It is an actor–critic stack with a
learned world model, neural-guided MCTS, MAPPO multi-agent training, a Bayesian
perception/belief layer, and an MPC executor — assembled around a shared `FeedForwardNetwork`
primitive (`discrete-event-system.rs/src/des/general/neural_network.rs`). Partial observability
and player relationships are handled with an **append-only feature-block schema** (each block
documented at its `SOCCER_NEURAL_*_FEATURE_DIM` const in `soccer.rs`, migrated for old snapshots
via `SOCCER_NEURAL_LEGACY_FEATURE_DIMS` zero-padding).

| # | Review checkpoint | Reality | Status |
|---|---|---|---|
| 1 | NN is stateless (POMDP)? | No learned recurrence, but belief handled: `perception.rs` (occlusion + noise + `perceived_position`), position/ball history buffers, multimodal opponent-intent head (`run_prediction.rs`), belief feature blocks (`SOCCER_NEURAL_BELIEF_FEATURE_DIM`, `SOCCER_NEURAL_OPP_BELIEF_DIM`), decision-commitment history | mostly covered; recurrence deferred |
| 2 | Action probs vs Q-values? | Both. Actor-critic: `SoccerPolicyHead` (actor) + value/critic heads + per-role/specialist sub-heads; tabular Q is a coarse fallback | done |
| 3 | Stores behavior probability? | Yes — top-k 70/20/10 + Boltzmann sampling, MAPPO clip-ε, `SoccerLearnedPolicyTrace` carries sampled probs | done |
| 4 | Reward too sparse? | No — PBRS (`pitch_value.rs`), learned pass-completion model, `reward_shaping.rs`, shaping-budget discipline, dozens of gated terms | done |
| 5 | Players trained independently? | No — MAPPO with team-reward-share, centralized critic (`central_brain_team_advantage`), MARL team component + clamp, GAE/PPO-clip | done |
| 6 | NN understands player relationships? | Relational info fed (22-player+ball motion block, 12×24 grid, team-center centroids) but historically only as a flat list + fixed aggregates — **no permutation-invariant attention/graph readout** | **gap → closed (see below)** |
| 7 | Planner searches sequences? | Yes — neural-guided **MCTS** re-ranker with learned world-model rollouts (`mcts_enabled/simulations/candidates/depth`) | done |
| 8 | MPC only executes intent? | Yes — MPC is the executor; the tactical brain is separate | done |

## Gap closed: relational attention readout

Checkpoint #6 was the one real conceptual gap. The whole-field motion block gave the value/actor
heads every player as a **flat, fixed-order** list of actor-relative pos/vel/acc/jerk, and the
team-center block gave fixed mean centroids — but nothing exposed a **permutation-invariant,
content-addressed** relational summary, so the MLP had to re-derive "who is near me / closing on
me" from raw coordinates on every forward pass (the exact weakness a GNN/attention encoder
addresses).

Implemented as an append-only feature block (`soccer.rs`):

- `SOCCER_NEURAL_RELATIONAL_ATTENTION_FEATURE_DIM = 8` — teammate {depth, width, closing,
  entropy} then opponent {depth, width, closing, entropy}.
- `soccer_relational_attention_group` — softmax-over-proximity (graph-attention-style)
  neighbourhood pooling: closer members get higher attention; outputs attention-weighted forward
  offset (support/threat depth), lateral spread, approach speed (support arriving / pressure
  closing), and normalised attention entropy (how many members are in play). Permutation-invariant
  by construction.
- `soccer_relational_attention_readout` — splits the motion block into teammates/opponents,
  excludes the actor's own slot (nearest the origin), returns the eight channels.

Gated by **`DD_SOCCER_ENABLE_RELATIONAL_ATTENTION`** (default **OFF**). When off the block is
all-zeros, so content is byte-identical to the prior schema and an A/B is a clean on/off within
identical network dimensions; older nets migrate by zero-padding (the pre-block total is registered
in `SOCCER_NEURAL_LEGACY_FEATURE_DIMS`). Turning it on requires a retrain to learn the new
channels — same lifecycle as every prior feature block. Tests: `relational_attention_*` in
`tests.rs` (zeros-on-bad-length, contiguous in-range indices, permutation invariance, closing
sign, entropy bounds, self-exclusion + team separation), plus the feature-schema chain/legacy
assertions updated for the new block.

## Deferred: learned temporal recurrence

A true recurrent/transformer belief layer (hidden state carried across ticks, BPTT) is **not**
implemented and is intentionally deferred. It is not a feature-block change: the net is a stateless
MLP trained on independent transitions, so recurrence would require restructuring the transition
store and training loop. Temporal information is currently supplied as engineered derivatives
(velocity/acceleration/jerk per entity) and history-derived blocks rather than a learned hidden
state. Revisit if the relational-attention A/B shows the policy is still memory-limited.
