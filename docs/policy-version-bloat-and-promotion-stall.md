# Policy-version bloat + promotion "stall" (experiment `soccer-self-play-k8s-overnight`)

**Status:** diagnosed 2026-07-08, **corrected 2026-07-09 after an audit against RDS**. One code fix
is implemented on branch `promotion-bloat-fix`: the **neural-blob bloat** strip (§3b → §6.1). The
originally-drafted promotion fix (flooring the evolution sample window at the gate) was **reverted** —
the audit showed the "promotion stall" is the genuine **plateau**, not a mechanical sample-count bug
(see §3a). The one-time RDS reclaim (§5) is a **draft pending operator sign-off**; nothing in §5 has
been run and no rows deleted.

> **Correction (why the promotion fix was reverted).** My first pass claimed the sample floor
> (`min_sample_games`=8) archived every candidate because the evolution window was ~3. The RDS
> provenance disproves that: candidates that go through the FULL promotion gate carry
> `gate.minSampleGames=8`, `sampleGames=8`, and **67 of them are `eligible:true`** (59 at gen ≥456).
> So candidates ARE evaluated over 8 games and CAN pass the gate. The active is still frozen at gen
> 456 because, on the honest 8-game evaluation, later candidates **regress** vs the incumbent —
> rejection reasons are dominated by `incumbent_mean_match_fitness … below incumbent …` and
> `incumbent_mean_play_quality … regression 0.0500`. The ~2.88M "fitness" that looked promotable was
> the **3-game tactical-search self-play score** (noisy, overfit); the same candidates score ~0.31–0.45
> mean_match_fitness over 8 games vs the incumbent's ~0.78–1.18. **This is the plateau, and the
> anti-regression gate is working as designed.** No promotion *code* change is warranted; the lever is
> the climb work (search/objective/capacity — see `docs/fix-plateau.md`, `docs/climbing.md`).

## 1. Summary

- **Version-row bloat (operational bug — FIXED on this branch).** Every archived candidate keeps its
  full `neuralNetwork` blob in `metrics` (~400 KB). Nothing ever deletes a version row or strips its
  neural blob (the only pruning targets the separate `policy_entries` table). The
  `des_soccer_learning_policy_versions` table is ~3.2 GB and grows ~0.5 GB/day. This is a real bug and
  is what forced the `:5055` neural probe to detoast ~2 GB per call.
- **Promotion "stall" (NOT a bug — the plateau).** The active policy has been frozen at gen 456 since
  2026-07-03 while the evolution search churns ~1000+ candidates/day that never beat the incumbent on
  the real 8-game evaluation. The gate correctly rejects regressions. Addressed by learning work, not
  a patch.

## 2. Evidence (read-only, RDS)

Rows/day show a regime change ~Jul 2–4 (rows explode, distinct generations collapse to ~4/day):
Jun 21–Jul 1 ≈ 1 version/gen (healthy); Jul 4–9 ≈ 1000–1400 versions/day across only 4 generations
(457–460). Gens 457–460 = 6256 rows, **all `status='archived'`**, `source_kind` merge 4319 / mutation
1919 / replay 18, one `branch_key`, every row keeps a `neuralNetwork` blob (avg 422 KB),
`full_entries_pruned_at` NULL for all. Table `pg_total_relation_size = 3194 MB`.

Promotion provenance (the corrected part):
- Active row = gen 456, `…-neural-checkpoint`, `source_kind=replay`, force-activated as a checkpoint;
  its own gate shows `eligible:false` with `rejectionReasons:["incumbent_mean_play_quality 0.3823
  below incumbent 0.4380 - regression 0.0500"]`.
- Full-gate rows (`minSampleGames=8`): **67 `eligible:true`** (gens 372–459, 59 at gen ≥456); the
  `eligible:false` majority is rejected for match-fitness and/or play-quality regression vs incumbent.
- Evolution branch-tips (gens 457–460) carry a *looser* internal gate (`minSampleGames=1`,
  `promotion.status='active'`) but are written as branch tips and immediately superseded → row
  `status='archived'`. They never reach the real active-promotion path. (All 7310 rows whose evolution
  gate says `promotion.status='active'` are row-`archived`.)

## 3. Root causes

### 3a. Promotion — NOT a mechanical bug (the plateau)

The persistence sample floor `soccer_policy_version_status_after_promotion_sample_floor`
(`src/des/soccer_learning_pg.rs`) does downgrade a would-be-`active` version to `archived` when
`gate.sampleGames < max(gate.minSampleGames, SOCCER_POLICY_PROMOTION_MIN_SAMPLE_GAMES)`. **But the
data shows this is not the current blocker:** full-gate candidates have `sampleGames=8` and pass the
floor, and 59 are `eligible:true` at gen ≥456 yet the active never advanced. The binding constraint is
the **anti-regression ratchet** (`active_max_fitness_regression` / play-quality regression): later
candidates genuinely lose vs the strong incumbent (gen 456) on the 8-game evaluation. That is the
plateau, tracked in `docs/fix-plateau.md` / `docs/climbing.md` and [[akrion-climb-state]] — a
learning-quality problem, not a code defect. **No promotion code change is shipped.**

### 3b. Version-row / neural-blob retention (real bug — fixed)

- The write+archive path inserts `metrics` (including `neuralNetwork`) for **every** version.
- `full_entries_retained` only governs the tabular `policy_entries`, never the neural blob in
  `policy_versions.metrics`.
- The only pruning (`prune_superseded_branch_entries_batched`, env `SOCCER_PG_INLINE_POLICY_PRUNE`)
  deletes from `policy_entries`, never from `policy_versions`. There is no `DELETE FROM
  des_soccer_learning_policy_versions` anywhere, and nothing strips the archived blob.

Net: archived candidates accumulate forever with full neural weights.

## 4. Fix — Part 1: promotion (NO code change)

Reverted the drafted window-floor. The promotion freeze is the plateau; the levers are learning-side
(evolution objective is optimising match fitness while play quality regresses; incumbent gen 456 is a
strong checkpoint nothing has beaten on 8-game eval). Operator knobs that exist if a *policy* decision
is wanted (not shipped, judgment calls): `SOCCER_BATCH_POLICY_ACTIVE_MAX_FITNESS_REGRESSION` /
`SOCCER_POLICY_ACTIVE_MAX_FITNESS_REGRESSION` (loosen the ratchet — invites quality regressions),
`SOCCER_POLICY_PROMOTION_MIN_SAMPLE_GAMES`. The real work is in `docs/fix-plateau.md`.

## 5. Fix — Part 2: reclaim the ~3.2 GB (DRAFT — guarded, staged, reversible-with-backup)

**Constraints.** Self-referencing FK `parent_policy_version_id → id` (no `ON DELETE`) — a row that is
another row's parent can't be deleted without handling the child. Preserve the active row, current
branch tip(s), and the 1 archived row referenced by a survivor.

**Measured reclaim (read-only dry-run, 2026-07-08):** stripping `neuralNetwork` from archived, non-tip
versions covers **6986 rows / ~4.2 GB of uncompressed JSON text**. On-disk reclaim after `VACUUM` is a
large fraction of the 3.2 GB (TOAST compressed).

**Recommended = strip the blob, keep the row** (preserves lineage/fitness/provenance; sidesteps FK):

```sql
-- DRY RUN
select count(*), pg_size_pretty(sum(length(metrics::text))::bigint)
from des_soccer_learning_policy_versions
where experiment_id = '7c7e2f5a-a2d5-49f7-b9c0-339d8853b23b'
  and status='archived' and metrics ? 'neuralNetwork'
  and id not in ( /* safety-keep: newest tip per branch_key + parents-of-survivors */ );

-- APPLY (after sign-off + backup), batched; repeat until 0 rows; each batch its own txn:
with victims as (
  select id from des_soccer_learning_policy_versions
  where experiment_id='7c7e2f5a-…' and status='archived' and metrics ? 'neuralNetwork'
    and id not in ( /* safety-keep set */ ) limit 500
)
update des_soccer_learning_policy_versions v set metrics = v.metrics - 'neuralNetwork'
from victims where v.id = victims.id;
```

Rails: back up first (`pg_dump -t … --where`); ~500-row batches, short `statement_timeout`; `VACUUM` /
`pg_repack` after; run off-peak / learner paused. Once §6.1 is deployed, this catch-up never repeats.

## 6. Fix — Part 3: durable

1. **Strip on archive (IMPLEMENTED, this branch).** When a version is non-active AND its
   `promotion.status == "archived"`, persist `metrics` **without** `neuralNetwork`
   (`postgresRetention.neuralSnapshotStripped` recorded; lightweight fields kept). Gated by
   `SOCCER_PG_RETAIN_ARCHIVED_POLICY_VERSION_NEURAL` (defaults to safe stripping). **Audited safe
   across every loader** (see below); the active head and any non-gate-archived version always keep
   their blob.
2. **Version retention window (NOT implemented).** Follow-up: cap archived tips per branch / last M
   days in the post-commit maintenance. Superseded old *actives* still keep their blob (one per
   promotion — negligible vs the population bloat §6.1 fixes).
3. **Index (already applied to RDS).** `des_soccer_learning_policy_versions_neural_gen_idx` on
   `(experiment_id, generation DESC)` → `:5055` neural probe 2 min → 2 ms. Viewer also runs with
   `SOCCER_POLICY_PROMOTION_BASELINE_LOOKBACK_GENERATIONS=0`.

### §6.1 loader-safety audit (2026-07-09)

Strip removes the blob ONLY for non-active rows with `promotion.status=="archived"`. Every path that
reads a version's neural net excludes exactly those rows:
- `load_latest_policy_metadata` (incl. `_with_min_visits`) — WHERE excludes `promotion.status='archived'`.
- `load_latest_neural_policy_metadata` — WHERE requires `status='active'` OR (`promotion.status!='archived'`
  AND `anchorPromotion.status!='archived'` AND no checkpoint reason).
- `load_policy_version_with_min_visits` (by-id, used by resume at `main_soccer_learning_run.rs:4002,8513`)
  — the id is always `pg_baseline.policy_version_id` from `load_strongest_recent_policy_promotion_baseline`,
  whose WHERE requires `promotion.status='active'`.
- `load_strongest_recent_policy_promotion_baseline` — reads only gate scalars, no neural.
- `load_latest_tournament_elite_neural_pool` — reads a different table (`des_soccer_tournament_team_brains`).
- `load_latest_active_policy*` — active only.

No loader can select a stripped version. Direction is conservative (keeps more than strictly needed —
e.g. anchor-archived-but-promotion-active rows keep their blob). Verified by
`policy_version_strips_neural_snapshot_only_for_gate_archived_non_active` and
`policy_version_retains_neural_snapshot_mirrors_loader_reachability`.

## 7. Suggested order of operations

1. Deploy §6.1 (this branch) — stops the table self-bloating.
2. §5 dry-run → sign-off + backup → batched strip → `VACUUM` for the backlog.
3. Promotion/plateau: separate track (`docs/fix-plateau.md`), not this branch.

## 8. Open questions for the operator

- Is the incumbent gen-456 checkpoint (itself gate-`ineligible`, force-activated) the right baseline,
  or should the ratchet/anchor be re-based? (Plateau track.)
- Are archived candidate **weights** needed for offline analysis, or is fitness+provenance enough? (If
  disposable, §5 strip is clearly correct.)
