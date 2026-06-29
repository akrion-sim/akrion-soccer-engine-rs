# Remaining next steps (roadmap)

Master checklist of outstanding work, prioritized. Detailed designs live in the
referenced docs; this is the actionable index. Status as of 2026-06-28.

## Done & deployed (for context — no action)
- Live `/soccer/` jumpiness fixed: WebSocket endpoint + **server-authoritative push**
  (autonomous stepper) in `dd-soccer-rs`, plus a client **jitter buffer**. Engine pinned
  to the `learning` branch on both clusters. (See `architecture-audit.md` for the perf
  reality this established.)
- Pass-eval dedup (byte-identical, ~1.3%) merged into `learning`.
- Offline-encoder **Step 1** (dataset export) and **Step 2a prototype** (numpy trainer,
  ~47% held-out gain) — `scripts/export_offline_dataset.sh`, `scripts/train_offline_head.py`.

## P1 — Offline distilled value head, productionized (`offline-encoder-step2-plan.md`)
The prototype proved the data is learnable; turn it into a shippable, gated head.
1. **Rust `soccer_offline_distill` bin.** Read the JSONL dataset → flatten features with the
   canonical layout (emit `data/soccer/offline/feature_layout.json`) → train a
   `FeedForwardNetwork` value head via `train_sample_clipped` (weighted by `weight_micros`).
   - **Weight-export bridge:** `DenseLayerConfig`/`FeedForwardNetwork` aren't `Serialize`;
     add a thin serializable mirror (weights `Vec<Vec<f64>>` + biases + activation names)
     via `to_layer_configs()`, written as the engine's neural-snapshot JSON.
2. **Gated consume path** in the decision layer: `DD_SOCCER_ENABLE_OFFLINE_DISTILLED_HEAD`
   (default-OFF = byte-identical). Append channels at the tail; zero-pad old snapshots.
3. **A/B via `soccer_eval_gate`** (held-out Elo / cross-play / Wilson / exploitability);
   promote only on a positive verdict.
4. Add `run_id` to the export so the held-out split is leak-free (currently random split).

## P2 — Fix the `visits` i32 overflow (`visits-overflow-fix-plan.md`)
Unblocks fast partial policy loads (load top-visited N instead of the full 1.3M rows — the
slow WAN load on `:5099`).
1. **Rust u32→u64** on `PolicyEntry.visits` + merge/read/write paths (harmless ahead of the
   migration — values still clamp at the i32 column until it's widened).
2. **Non-locking `bigint` migration** (add `visits_64 bigint` + backfill + swap), run in a
   **coordinated low-activity window** — NOT a naive `ALTER COLUMN TYPE` during active
   learning (shared prod RDS, live AWS learner writes this table).
3. Re-enable a high `SOCCER_LIVE_POLICY_MIN_VISITS` / top-N load; re-measure `:5099` load time.

## P3 — Cluster learning ops
- **Hetzner `dd-soccer-learning-queue` is failing** (Error pods seen this session; node SSH
  was also refusing connections). Diagnose: `kubectl logs` the failed jobs, check the
  cronjob, confirm the node/SSH. The AWS `dd-soccer-learning-rds-continuous` is healthy;
  the Hetzner queue is the gap.
- Confirm the latest `learning`-branch roll (`dd-soccer-rs` @ jitter-buffer) reached **1/1
  on both clusters** (Hetzner monitor; AWS via SSM) — last verified state was mid-roll.

## P4 — Step 2b (later): learned relational encoder
Only after P1 proves value. Tiny GNN/attention encoder over the raw 22-body graph, trained
offline and **distilled into the same dense head shape** (still one forward pass at runtime).
Gated + eval-gated identically. The audit's #1 done within the 66 ms / 22-agent budget.

## Cross-cutting constraints (do not violate)
- **66 ms / 15 Hz / 22-agent real-time budget** — no per-tick GNN/RNN/MCTS (`architecture-audit.md`).
- **Determinism** — new learnable paths land **default-OFF / byte-identical**; A/B behind the eval gate.
- **New variables** — append at the tail (zero-pad old neural snapshots); keep them OUT of the
  tabular `state_key` unless re-accumulating that experiment.
- **Shared prod RDS** — schema changes only via non-locking add/backfill/swap in a window.
