# Plan: offline distilled encoder — Step 2 (train + distill)

Follows Step 1 (dataset export, DONE — `scripts/export_offline_dataset.sh`). Goal: learn
a better value/policy head **offline** from the logged transitions and ship it through the
engine's existing `FeedForwardNetwork` path so **runtime per-tick cost is unchanged**.

## Dataset (verified, Step 1)
`(state_key, action, value_micros, weight_micros)` from `des_soccer_learning_run_deltas`:
- 47 distinct actions; **205 binned engineered features** (int / bool / str bins);
- value target ≈ [−7, 11] units; `weight_micros` = effective visits (sample weight).

## Staged approach (ship the cheap win first; GNN later)

### Step 2a — offline value/policy head on the engineered features (format-compatible) — DO NOW
The 205 binned features already encode the relational/belief signal (field_numbers,
pressure, lanes, opponent belief…). So the first, shippable win is a **better-trained
value head** distilled into the engine's `FeedForwardNetwork`, with **identical runtime**.

1. **Feature flattener (canonical layout).** `state_key` → fixed f32 vector:
   one-hot each categorical/bool bin; min-max scale each int bin. Freeze the column order
   (write it to `data/soccer/offline/feature_layout.json`) so train-time and engine-time
   encodings match exactly. Append-only (new bins go at the tail → zero-pad old snapshots).
2. **Train.** Weighted regression to `value_micros` (and optional per-action policy head),
   `weight = weight_micros`. Hold out 15% by `run_id` (no leakage). Metric: weighted value
   MSE + ranking agreement vs the tabular baseline on held-out states.
3. **Distill into `FeedForwardNetwork`.** Target the engine's dense layer config
   (`DenseLayerConfig`, activations) so the trained weights serialize into the existing
   neural-snapshot format and load unchanged at runtime (one dense forward pass).
4. **Integrate behind a default-OFF gate** (`DD_SOCCER_ENABLE_OFFLINE_DISTILLED_HEAD`).
   Append channels at the tail; zero-pad old snapshots (already supported, soccer.rs:3675).
5. **A/B before promotion** with the existing `soccer_eval_gate` (held-out Elo / cross-play /
   Wilson / exploitability). Promote only on a positive verdict.

**Tooling:** prototype training in Python/numpy for fast iteration, then a small Rust
`soccer_offline_distill` bin that loads the trained dense weights into a
`FeedForwardNetwork` and writes the engine snapshot — OR train directly in the Rust FFN
(`train_sample_clipped`) to guarantee format match. Recommend the Rust-native trainer bin
to avoid a Python↔Rust weight-format bridge.

### Step 2b — learned relational encoder over the RAW player graph (later)
Only after 2a proves value. Replace hand-crafted relational bins with a tiny GNN/attention
encoder over the 22-body graph, trained offline, then **distilled into the same dense head
shape** (still one forward pass at runtime). This is the audit's #1 done within the 66 ms
budget. Larger effort; gated and eval-gated identically.

## Build order
1. `soccer_offline_distill` bin scaffold: read JSONL → flatten (emit `feature_layout.json`)
   → train weighted FFN value head → write engine snapshot + held-out metrics.
2. Wire a default-OFF consume path for the distilled head in the decision layer.
3. `soccer_eval_gate` A/B; promote on positive verdict.

## Non-goals (explicit)
- No full-state per-tick GNN/RNN/deep MCTS (blows the 22-agent / 66 ms budget — see
  architecture-audit.md). Bounded MCTS/PUCT remains an opt-in reranker over capped legal
  candidates, not part of this offline encoder step.
- No change to the tabular `state_key` encoding (would orphan existing tabular entries).

## Findings during implementation (2026-06-28) — must resolve before the Rust bin

1. **Feature-space mismatch (design decision).** The exported dataset's `state_key` is the
   **tabular discretization (~205 bins)**, NOT the engine's existing **192-dim continuous
   neural feature vector** (`SOCCER_NEURAL_BASE_FEATURE_DIM`). So the distilled head is a
   *new* learned value head over the tabular bins (a smooth generalization of the Q-table),
   **not** a drop-in replacement for an existing 192-dim head. Two options:
   - (a) Train over the bin features and add a decision-time **bin-feature extractor** +
     a dedicated consume gate (cleanest first cut — directly generalizes the Q-table).
   - (b) Re-export the dataset using the engine's 192-dim `neural_extended` features
     instead of `state_key`, so it distills into an existing-head shape. Requires logging
     those features in the deltas (they aren't today) or recomputing them from snapshots.
   Recommend (a) for 2a; revisit (b) with 2b's learned encoder.
2. **Access.** `neural_network`/`prng` are `pub(crate)` in the engine, so a standalone bin
   can't reach `FeedForwardNetwork`. Add a single `pub fn soccer_offline_distill_value_head(
   dataset, out_snapshot, hidden, epochs, lr) -> Report` in the soccer module (uses the
   internal FFN + `soccer_neural_network_snapshot` + the pub `SoccerNeuralNetworkSnapshot`),
   and a thin `src/bin/soccer_offline_distill.rs` that calls it. Avoids widening NN's API.
