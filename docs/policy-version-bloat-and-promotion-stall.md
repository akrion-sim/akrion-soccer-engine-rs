# Policy-version bloat + promotion stall (experiment `soccer-self-play-k8s-overnight`)

**Status:** diagnosed 2026-07-08. Read-optimizing index already applied (see §6). The
config fix (§4) and reclaim (§5) are **drafts pending operator sign-off** — nothing in §5 has
been run. No rows have been deleted.

## 1. Summary

Two independent problems, one shared symptom (the local `:5055` viewer could not load learned
weights because a probe query had to detoast ~2 GB of JSONB per call):

- **Promotion stall (behavioural).** The evolutionary tactical search evaluates each candidate on
  only ~3 games, but the promotion gate requires `sampleGames >= 8`. Every would-be promotion is
  therefore force-downgraded to `archived`. The active policy has been frozen at **gen 456** since
  2026-07-03, while the generation counter cycles within **gens 457–460**.
- **Unbounded version-row retention (operational).** Every archived candidate row keeps its full
  `neuralNetwork` blob in `metrics` (~400 KB each). Nothing ever deletes a *version* row or strips
  its neural blob — the only existing pruning targets the separate `policy_entries` (tabular) table.
  The `des_soccer_learning_policy_versions` table is now **~3.2 GB** and grows **~0.5 GB/day**.

The plateau itself (best self-play fitness flat at ~2.88M for 6 days) is expected RL behaviour and
is tracked separately (see `docs/fix-plateau.md`, `docs/climbing.md`). This doc is about the two
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

### 3a. Promotion sample-floor vs. evolution eliteGames

`soccer_policy_version_status_after_promotion_sample_floor`
([`src/des/soccer_learning_pg.rs:435`](../src/des/soccer_learning_pg.rs#L435)) downgrades any version
requested as `active` to `archived` when `gate.sampleGames < min_sample_games`:

```rust
let min_sample_games = gate.get("minSampleGames")…    // default from …
    .max(configured_min_sample_games);                 // soccer_neural_snapshot_min_sample_games()
if sample_games.is_some_and(|s| s >= min_sample_games) { requested_status }
else { SOCCER_POLICY_STATUS_ARCHIVED }
```

- `min_sample_games` default = **8** — `soccer_neural_snapshot_min_sample_games()`
  ([`soccer_learning_pg.rs:5497`](../src/des/soccer_learning_pg.rs#L5497), env
  `SOCCER_POLICY_PROMOTION_MIN_SAMPLE_GAMES`).
- Evolution runs each candidate on `eliteGames` games — `DEFAULT_SOCCER_EVOLUTION_ELITE_GAMES = 4`
  ([`main_soccer_learning_run.rs:95`](../src/bin/main_soccer_learning_run.rs#L95), env
  `SOCCER_EVOLUTION_ELITE_GAMES`), and `elite_count = evolution_elite_games.min(ranked_samples.len())`
  ([`main_soccer_learning_run.rs:10647`](../src/bin/main_soccer_learning_run.rs#L10647)) — observed **3**
  in the live metrics.

**3 (or 4) < 8, structurally, so no candidate can ever satisfy the gate.** Promotion is impossible
by construction; the active policy is frozen.

### 3b. No retention on version rows / neural blobs

- The write+archive path
  ([`soccer_learning_pg.rs:1972`–`2166`](../src/des/soccer_learning_pg.rs#L1972)) inserts `metrics`
  (including `neuralNetwork`) for **every** version, archived or not.
- `full_entries_retained` (via `soccer_policy_version_retains_full_entries`,
  [`soccer_learning_pg.rs:428`](../src/des/soccer_learning_pg.rs#L428)) only governs whether the
  **tabular `policy_entries`** are written — it does **not** touch the neural blob in
  `policy_versions.metrics`.
- The only pruning that exists (`prune_superseded_branch_entries_batched`,
  [`soccer_learning_pg.rs:2176`](../src/des/soccer_learning_pg.rs#L2176); gated by
  `SOCCER_PG_INLINE_POLICY_PRUNE`) deletes from `des_soccer_learning_policy_entries`, never from
  `des_soccer_learning_policy_versions`.
- There is **no** `DELETE FROM des_soccer_learning_policy_versions` anywhere in the tree, and no code
  path strips `neuralNetwork` from an archived version's `metrics`.

Net: archived candidates accumulate forever with full neural weights.

## 4. Fix — Part 1: stop the bleeding (config, zero deletion) ✅ recommended first

Make candidates *able* to promote, which both unblocks the plateau's promotion path and stops the
per-day churn from being pointless. Two levers (pick per learning intent, not both blindly):

- **Raise evaluation samples to meet the gate:** set `SOCCER_EVOLUTION_ELITE_GAMES` (and the
  evolution window, `default_evolution_window_games`) so a promoted candidate's `gate.sampleGames >= 8`.
  This is the principled fix — the gate exists to avoid promoting on 3-game noise.
- **Or lower the gate** `SOCCER_POLICY_PROMOTION_MIN_SAMPLE_GAMES` to match the search's real sample
  count — only if 3–4-game promotion is genuinely acceptable (it invites overfit promotions; not
  recommended while chasing the plateau).

These are env changes on the learner Deployment (`main_soccer_learning_run`,
`soccer-self-play-k8s-overnight`). No data is modified. Reversible by reverting the env.

> This alone does **not** shrink the table — it stops future waste and lets the active advance.

## 5. Fix — Part 2: reclaim the ~3.2 GB (guarded, staged, reversible-with-backup)

**Constraints that make naive deletion unsafe:**

- Self-referencing FK `des_soccer_learning_policy_versio_parent_policy_version_id_fkey`
  (`parent_policy_version_id → id`, no `ON DELETE`), so a row that is any other row's parent cannot be
  deleted without first handling the child (delete leaves-first, or `NULL` the child's parent).
- The active row and the current branch tip(s) must be preserved, plus the 1 archived row referenced
  by a survivor (§2).

**Recommended reclaim = strip the blob, keep the row** (preserves lineage, fitness, provenance;
sidesteps the FK entirely; reclaims ~95% of the bytes):

```sql
-- DRY RUN: how many rows / how much space would be reclaimed
select count(*), pg_size_pretty(sum(length(metrics::text))::bigint)
from des_soccer_learning_policy_versions
where experiment_id = '7c7e2f5a-a2d5-49f7-b9c0-339d8853b23b'
  and status = 'archived'
  and metrics ? 'neuralNetwork'
  and id not in (                         -- keep the newest tip per (branch_key) as a safety margin
    select distinct on (branch_key) id
    from des_soccer_learning_policy_versions
    where experiment_id = '7c7e2f5a-…' order by branch_key, generation desc, updated_at desc
  );

-- APPLY (only after sign-off + backup), batched to avoid long locks / WAL spikes:
--   repeat until 0 rows affected; each batch is its own transaction.
with victims as (
  select id from des_soccer_learning_policy_versions
  where experiment_id = '7c7e2f5a-…' and status='archived' and metrics ? 'neuralNetwork'
    and id not in ( /* safety-keep set as above */ )
  limit 500
)
update des_soccer_learning_policy_versions v
set metrics = v.metrics - 'neuralNetwork'
from victims where v.id = victims.id;
```

Notes / safety rails:
- **Back up first** (`pg_dump -t des_soccer_learning_policy_versions --where="experiment_id=…"` or a
  `create table … as select` snapshot) so the strip is recoverable. Stripping the blob is otherwise
  irreversible for those weights (acceptable: they are superseded archived candidates).
- Run batches of ~500 with a short `statement_timeout`; the update rewrites rows → **`VACUUM
  (or pg_repack)` afterward** to actually return space to the OS and shrink the TOAST table.
- Do it while the learner is briefly paused, or off-peak, to avoid dead-tuple churn racing autovacuum.
- **Alternative (row delete)** if you want the rows gone: delete archived, non-tip rows **leaves-first**
  (`where not exists (select 1 … where child.parent_policy_version_id = v.id)`), looping until stable,
  or first `update … set parent_policy_version_id = null` on survivors pointing into the victim set.
  More moving parts than the strip; only worth it if row count itself is a problem.

## 6. Fix — Part 3: make it durable (code)

1. **Strip on archive.** In the write path (§3b), when the effective status is `archived` (or when a
   tip is superseded), write `metrics` **without** `neuralNetwork` for non-retained versions — or set
   `full_entries_pruned_at` and add a version-level retention analogous to the entries pruning. This
   is the real fix; §5 is a one-time catch-up.
2. **Version retention window.** Add an experiment-scoped cap (keep newest N archived tips per branch,
   or last M days) enforced in the same best-effort post-commit maintenance that already prunes
   entries (`SOCCER_PG_INLINE_POLICY_PRUNE`).
3. **Index (already applied).** `des_soccer_learning_policy_versions_neural_gen_idx` on
   `(experiment_id, generation DESC)` lets the live `:5055` neural probe do a bounded index scan
   instead of a full-table detoast (query 2 min → 2 ms). Keep it; it also de-risks the table while
   §5/§6.1 land. The live viewer additionally runs with
   `SOCCER_POLICY_PROMOTION_BASELINE_LOOKBACK_GENERATIONS=0` so its candidate window stays on the
   latest generation (see `~/bin/soccer-live-5055.sh`, `docs/live-runner-weights.md`).

## 7. Suggested order of operations

1. §4 config fix on the learner (stops waste, unblocks promotion) — **safe now**.
2. Confirm the active policy starts advancing past gen 456 (watch for a new `status='active'`).
3. §5 dry-run → operator sign-off + backup → batched strip → `VACUUM`.
4. §6.1/§6.2 code change so the table self-limits; then the one-time §5 never needs repeating.

## 8. Open questions for the operator

- Is 3–4-game evaluation intended, with the 8-game gate a leftover default? (Determines §4 lever.)
- Are archived candidate **weights** needed for any offline analysis, or is fitness+provenance
  enough? (If weights are disposable, §5 strip is clearly correct.)
- Should generation numbering advance per accepted candidate rather than cycling 457–460? (Separate
  from bloat, but it's the surface signal that promotion is stuck.)
