# Policy-version bloat + promotion stall (experiment `soccer-self-play-k8s-overnight`)

**Status:** diagnosed 2026-07-08. Read-optimizing index already applied to RDS (see §6.3). Code
fixes for the promotion stall (§3a → §4) and the neural-blob bloat at source (§6.1) are
**implemented on this branch** (`promotion-bloat-fix`) — not merged, not deployed. The one-time
reclaim of existing rows (§5) remains a **draft pending operator sign-off**; nothing in §5 has been
run and no rows have been deleted.

> **Implementation note / correction to §3a.** Tracing the code showed `gate.sampleGames` is the
> **occupancy of the evolution sample window** (`evolution_search_samples`, capped by
> `SOCCER_EVOLUTION_WINDOW_GAMES`) — one `sample_games` per `MatchSummary` — **not** `eliteGames`
> directly (`eliteGames` is the breeding-elite count). So the true fix floors the effective evolution
> *window* at the gate's `min_sample_games` when the gate is enabled, leaving breeding dynamics
> (`SOCCER_EVOLUTION_ELITE_GAMES`) untouched. Implemented as
> `evolution_window_games_floored_for_promotion_gate` in `src/bin/main_soccer_learning_run.rs`, plus a
> guard log in `soccer_policy_version_status_after_promotion_sample_floor` when a would-be-active
> candidate is force-archived purely by the floor. Change B (bloat) strips `neuralNetwork` from
> persisted `metrics` only for versions that are **non-active AND `promotion.status == "archived"`** —
> exactly the rows every live loader already excludes, so the `:5055`/resume/neural load paths never
> lose the net they actually read. Gated by `SOCCER_PG_RETAIN_ARCHIVED_POLICY_VERSION_NEURAL`
> (defaults to safe stripping). Commit on branch `promotion-bloat-fix`.

## 1. Summary

Two independent problems, one shared symptom (the local `:5055` viewer could not load learned
weights because a probe query had to detoast ~2 GB of JSONB per call):

- **Promotion stall (behavioural).** The evolutionary tactical search evaluates each candidate over a
  sample window that is smaller than the promotion gate's `sampleGames >= 8` requirement, so every
  would-be promotion is force-downgraded to `archived`. The active policy has been frozen at
  **gen 456** since 2026-07-03, while the generation counter cycles within **gens 457–460**.
- **Unbounded version-row retention (operational).** Every archived candidate row keeps its full
  `neuralNetwork` blob in `metrics` (~400 KB each). Nothing ever deletes a *version* row or strips
  its neural blob — the only existing pruning targets the separate `policy_entries` (tabular) table.
  The `des_soccer_learning_policy_versions` table is now **~3.2 GB** and grows **~0.5 GB/day**.

The plateau itself (best self-play fitness flat at ~2.88M for 6 days) is expected RL behaviour and is
tracked separately (see `docs/fix-plateau.md`, `docs/climbing.md`). This doc is about the two
*mechanical* bugs above.

## 2. Evidence (read-only, RDS)

Rows per day for the experiment — note the regime change on ~Jul 2–4 (rows explode while distinct
generations collapse to ~4/day, i.e. generation advancement stalls):

| day | rows | distinct gens |
|---|---|---|
| …Jun 21–Jul 1 | 1–98/day | ≈ rows (1 version/gen — healthy) |
| Jul 2 | 171 | 87 |
| Jul 3 | 410 | 17 |
| Jul 4 | 1206 | **4** |
| Jul 5 | 1350 | **4** |
| Jul 6 | 1148 | **4** |
| Jul 7 | 1402 | **4** |
| Jul 8 | 604 | **4** |
| Jul 9 | 291 | **4** |

Per-generation snapshot (gens 457–460): 6256 rows, **all `status='archived'`**, `source_kind` ∈
{`merge` 4319, `mutation` 1919, `replay` 18}, **one** `branch_key`, 4257 distinct parents, every row
retains a `neuralNetwork` blob (avg 422 KB), `full_entries_pruned_at` NULL for all. The single
`active` row is gen 456 (`…-neural-checkpoint`, fitness 1.26M). Best archived fitness ~2.88M
(measured on ~3 games — noisy) has not improved since Jul 3.

Table footprint: `pg_total_relation_size = 3194 MB`; 6987 of 6988 experiment rows are archived rows
carrying a neural blob. Only **1** archived row is referenced as a `parent_policy_version_id` by a
surviving (non-archived) row.

## 3. Root causes (code)

### 3a. Promotion sample-floor vs. evolution sample window

`soccer_policy_version_status_after_promotion_sample_floor` (`src/des/soccer_learning_pg.rs`)
downgrades any version requested as `active` to `archived` when `gate.sampleGames < min_sample_games`:

```rust
let min_sample_games = gate.get("minSampleGames")…    // default from …
    .max(configured_min_sample_games);                 // soccer_neural_snapshot_min_sample_games()
if sample_games.is_some_and(|s| s >= min_sample_games) { requested_status }
else { SOCCER_POLICY_STATUS_ARCHIVED }
```

- `min_sample_games` default = **8** — `soccer_neural_snapshot_min_sample_games()` (env
  `SOCCER_POLICY_PROMOTION_MIN_SAMPLE_GAMES`).
- `gate.sampleGames` = the number of `MatchSummary` samples in the evolution window
  (`evolution_search_samples`, capped by `SOCCER_EVOLUTION_WINDOW_GAMES`), populated via
  `run_evolution_search_metadata` → `policy_promotion_search_metadata` →
  `evaluate_soccer_policy_promotion_gate`. Observed **3** live. (`eliteGames`,
  `DEFAULT_SOCCER_EVOLUTION_ELITE_GAMES = 4`, is the *breeding-elite* count — a red herring for the
  gate, though it is what surfaced first in the metrics.)

**window occupancy (3) < gate (8), structurally, so no candidate can satisfy the gate.** Promotion is
impossible by construction; the active policy is frozen.

### 3b. No retention on version rows / neural blobs

- The write+archive path (`insert_policy_version_with_id_inner` and neighbours,
  `src/des/soccer_learning_pg.rs`) inserts `metrics` (including `neuralNetwork`) for **every** version,
  archived or not.
- `full_entries_retained` (via `soccer_policy_version_retains_full_entries`) only governs whether the
  **tabular `policy_entries`** are written — it does **not** touch the neural blob in
  `policy_versions.metrics`.
- The only pruning that exists (`prune_superseded_branch_entries_batched`; gated by
  `SOCCER_PG_INLINE_POLICY_PRUNE`) deletes from `des_soccer_learning_policy_entries`, never from
  `des_soccer_learning_policy_versions`.
- There is **no** `DELETE FROM des_soccer_learning_policy_versions` anywhere in the tree, and no code
  path strips `neuralNetwork` from an archived version's `metrics`.

Net: archived candidates accumulate forever with full neural weights.

## 4. Fix — Part 1: promotion reachable (IMPLEMENTED, config-safe)

Make candidates *able* to promote, which unblocks the plateau's promotion path and stops the per-day
churn from being pointless.

**Implemented:** `evolution_window_games_floored_for_promotion_gate(window, gate_enabled, min_sample_games)`
floors the effective evolution window at `min_sample_games` **only when the gate is enabled**, never
lowering a larger operator-set window. Breeding (`SOCCER_EVOLUTION_ELITE_GAMES`) and the gate itself
are untouched (the gate exists to avoid promoting on 3-game noise). A guard log fires when a
would-be-active candidate is force-archived purely by the sample floor, so the failure is visible
instead of silent.

Env levers still honoured: `SOCCER_EVOLUTION_WINDOW_GAMES`, `SOCCER_EVOLUTION_ELITE_GAMES`,
`SOCCER_POLICY_PROMOTION_MIN_SAMPLE_GAMES`. No data is modified. Reversible by reverting the branch.

> This alone does **not** shrink the table — it stops future waste and lets the active advance.

## 5. Fix — Part 2: reclaim the ~3.2 GB (DRAFT — guarded, staged, reversible-with-backup)

**Constraints that make naive deletion unsafe:**

- Self-referencing FK `des_soccer_learning_policy_versio_parent_policy_version_id_fkey`
  (`parent_policy_version_id → id`, no `ON DELETE`), so a row that is any other row's parent cannot be
  deleted without first handling the child (delete leaves-first, or `NULL` the child's parent).
- The active row and the current branch tip(s) must be preserved, plus the 1 archived row referenced
  by a survivor (§2).

**Measured reclaim (read-only dry-run, 2026-07-08):** stripping `neuralNetwork` from archived,
non-tip versions (keeping the newest tip per branch and the 1 archived row referenced by a survivor)
covers **6986 rows / ~4.2 GB of uncompressed JSON text** — essentially the whole table minus a handful
of live rows. On-disk reclaim after `VACUUM` is a large fraction of the current 3.2 GB (TOAST is
compressed, so the freed disk is less than the raw JSON figure but still dominant).

**Recommended reclaim = strip the blob, keep the row** (preserves lineage, fitness, provenance;
sidesteps the FK entirely):

```sql
-- DRY RUN: rows / bytes reclaimable
select count(*), pg_size_pretty(sum(length(metrics::text))::bigint)
from des_soccer_learning_policy_versions
where experiment_id = '7c7e2f5a-a2d5-49f7-b9c0-339d8853b23b'
  and status = 'archived' and metrics ? 'neuralNetwork'
  and id not in ( /* safety-keep: newest tip per branch_key + parents-of-survivors */ );

-- APPLY (only after sign-off + backup), batched; repeat until 0 rows; each batch its own txn:
with victims as (
  select id from des_soccer_learning_policy_versions
  where experiment_id = '7c7e2f5a-…' and status='archived' and metrics ? 'neuralNetwork'
    and id not in ( /* safety-keep set */ )
  limit 500
)
update des_soccer_learning_policy_versions v
set metrics = v.metrics - 'neuralNetwork'
from victims where v.id = victims.id;
```

Safety rails: **back up first** (`pg_dump -t … --where="experiment_id=…"` or a snapshot table); run
~500-row batches with a short `statement_timeout`; **`VACUUM` (or `pg_repack`) afterward** to return
space to the OS; do it while the learner is briefly paused / off-peak. **Alternative (row delete)**
only if the row count itself matters: delete archived non-tip rows leaves-first, or `NULL` survivors'
`parent_policy_version_id` into the victim set first — more moving parts than the strip.

Note: once §6.1 (strip-on-archive) is deployed, this one-time catch-up never needs repeating.

## 6. Fix — Part 3: make it durable

1. **Strip on archive (IMPLEMENTED).** In the write path, when a version is non-active AND its
   `promotion.status == "archived"`, persist `metrics` **without** `neuralNetwork` (lightweight
   provenance/fitness/tactical fields preserved; `postgresRetention.neuralSnapshotStripped` recorded).
   The active head and any still-promotable version keep their blob, so the `:5055`/resume/neural
   loaders never lose the net they read. Gated by `SOCCER_PG_RETAIN_ARCHIVED_POLICY_VERSION_NEURAL`
   (defaults to safe stripping).
2. **Version retention window (NOT implemented — out of scope).** A follow-up could cap archived tips
   per branch or last M days in the same best-effort post-commit maintenance that prunes entries
   (`SOCCER_PG_INLINE_POLICY_PRUNE`). Superseded *old actives* still keep their blob (one per
   promotion — not the per-generation population bloat, which §6.1 fixes).
3. **Index (already applied to RDS).** `des_soccer_learning_policy_versions_neural_gen_idx` on
   `(experiment_id, generation DESC)` lets the live `:5055` neural probe do a bounded index scan
   instead of a full-table detoast (query 2 min → 2 ms). The live viewer also runs with
   `SOCCER_POLICY_PROMOTION_BASELINE_LOOKBACK_GENERATIONS=0` so its candidate window stays on the
   latest generation (see `~/bin/soccer-live-5055.sh`, `docs/live-runner-weights.md`).

## 7. Suggested order of operations

1. Deploy §4 + §6.1 (this branch) to the learner (stops waste, unblocks promotion, self-limits the
   table).
2. Confirm the active policy starts advancing past gen 456 (watch for a new `status='active'`).
3. §5 dry-run → operator sign-off + backup → batched strip → `VACUUM` for the one-time backlog.

## 8. Open questions for the operator

- Is the ~3-game evaluation window intended, with the 8-game gate a leftover default? (§4 floors the
  window up to the gate; if instead the gate should drop, set `SOCCER_POLICY_PROMOTION_MIN_SAMPLE_GAMES`.)
- Are archived candidate **weights** needed for any offline analysis, or is fitness+provenance enough?
  (If disposable, §5 strip is clearly correct.)
- Should generation numbering advance per accepted candidate rather than cycling 457–460? (Surface
  signal that promotion is stuck; separate from bloat.)
