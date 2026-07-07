# extlib-pomdp-solver — a candle-backed POMDP solver for the soccer engine

Branch `feature/external-nn-pomdp-solver`. This is the **Lock-2 (architecture)** track from
[`docs/plateau-root-cause-and-nn-solver-analysis.md`](../docs/plateau-root-cause-and-nn-solver-analysis.md):
a neural net that can actually *represent* a POMDP solution, replacing the 1-hidden-layer hand-rolled
MLP that structurally cannot.

## Why (the analysis, in one paragraph)

The current custom net is a **single-hidden-layer tanh MLP** with **no learned memory** and **no learned
attention** (its "relational attention" is hand-computed summary scalars). A feedforward net on one
engineered-belief observation is a *belief-MDP reduction* — it cannot integrate history, so it cannot
solve a true POMDP. Soccer is a 22-agent POMDP: the optimal action depends on belief over history
(an unseen run, a developing overload). This crate builds the standard architecture for that.

## The architecture (`src/lib.rs`, working skeleton)

```
entities (23 = 22 players + ball), each entity_dim
  └─ per-entity embed → SELF-ATTENTION (softmax(QKᵀ/√d)·V) → mean-pool   ← permutation-INVARIANT
      └─ GRU belief cell, hidden carried across decisions                ← POMDP MEMORY over history
          ├─ ACTOR  head → π(action-family | belief)                     ← policy-gradient toward RETURN
          └─ CRITIC head → V(belief)                                     ← centralized critic (CTDE)
```

`cargo run` runs a 3-decision rollout and shows the belief norm growing across decisions (the
recurrent memory accumulating) — the property the flat MLP lacks.

## Why candle (validated)

Pure-Rust, builds in ~16s (no 2 GB libtorch), real autodiff + attention, and the **Metal (GPU)** feature
compiles on this Mac (`--features metal`). Keeps the codebase pure-Rust and uses the GPU. (Alternatives:
`burn` for maximum modularity; `tch` for the full PyTorch arsenal at the cost of the libtorch dependency.)

## Integration plan (when Lock-1 / COMA confirms formulation isn't enough alone)

1. **Feature adapter** — map the engine's per-decision `SoccerDecisionContext` + whole-field motion
   block into the `(n_entities, entity_dim)` entity tensor (one entity per player + ball).
2. **Parallel gated backend** — add a `SOCCER_NEURAL_BACKEND=candle` path alongside the custom
   `FeedForwardNetwork`, so the two can be A/B'd on the same replay. Do **not** rip out the custom net.
3. **CTDE training loop** — per-agent actor (decentralized obs) + a centralized critic that sees the
   global state at train time; train the actor by policy-gradient toward **return / advantage-over-
   analytic** (the COMA lever), NOT tabular imitation. Belief hidden state carried per agent across a
   possession/trajectory.
4. **Snapshot bridge** — serialize candle `VarMap` weights into a snapshot the engine can persist +
   the eval harness can load.
5. **A/B** — candle-POMDP-solver vs the custom MLP, fresh nets, same eval ladder vs the analytic field.

## Status

Skeleton compiles + forwards. Not yet wired to the engine. Held until the COMA / actor-critic read
tells us whether the formulation fix alone breaks parity on the current net (if it does, this is
optional; if it plateaus, this is the next big build).
