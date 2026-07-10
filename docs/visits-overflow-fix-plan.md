# Plan: fix the policy `visits` i32 overflow

## Problem (grounded)
- In-memory `visits` is `u32` (`soccer_learning.rs:1676`, `PolicyEntry`).
- The PG column `des_soccer_learning_policy_entries.visits` is `integer` (**i32**,
  max 2,147,483,647). The merge accumulates visits across runs/generations.
- Hot states (e.g. `Midfielder / BuildUp`, visited ~every game over 300+ generations ×
  thousands of games) exceed i32::MAX and **clamp to 2147483647** on write; read back as
  `i32` (`soccer_learning_pg.rs:2293`, `:2964`: `row.get::<_, i32>(…).max(0) as u32`).
- Consequence: **median `visits` = 2147483647** across the gen-308 tip (1.32M entries),
  so `SOCCER_LIVE_POLICY_MIN_VISITS` cannot rank/trim — every local load pulls the full
  1.3M rows (the slow WAN load), and "load the most-trained N states" is impossible.

## Goal
Make `visits` a faithful count again so it can (a) rank states for **fast partial
loads** (`visits >= threshold`, or top-N by visits) and (b) weight learning correctly.

## Options

### A. Widen to 64-bit (correct, but a prod schema change) — PREFERRED, staged
1. **Rust:** `visits: u32 → u64` on `PolicyEntry` and the merge/accumulate paths
   (`saturating_add` stays, now u64); PG read `get::<_, i32>` → `get::<_, i64>`,
   write binds `i64`. Keep `min_visits` API as `i64`.
2. **DB migration:** `ALTER TABLE des_soccer_learning_policy_entries ALTER COLUMN visits
   TYPE bigint;` (and the same on `run_deltas.visit_delta` if it can exceed i32, plus
   `des_soccer_learning_run_deltas`/policy tables that carry visit columns).
   - **Risk:** full-table rewrite + lock on a 4.4M-row table in **shared production RDS**
     used by the live AWS learner. Do it in a low-activity window, or use the
     add-new-column + backfill + swap pattern to avoid a long lock.
   - Pre-existing clamped values (2147483647) are **not recoverable** — they stay until
     the state is re-visited, then grow correctly. Acceptable (they're "very hot" either way).
3. **Backward-compat:** code must read both old (i32-capped) and new (i64) rows. A bigint
   column returns i64 for both, so once migrated this is automatic.

### B. Cap in Rust at a sane max (no prod migration) — FALLBACK
- Cap accumulation at e.g. `VISITS_SOFT_MAX = 50_000_000` (well under i32::MAX) so writes
  never clamp at the column limit, and the long tail (< cap) stays rankable.
- Cheaper/safer (no schema change) but hot states above the cap remain indistinguishable
  among themselves; ranking only works below the cap.

## Recommendation
Ship **A** in two safe steps: (1) the Rust u32→u64 change first (harmless against the
current i32 column — values just still clamp on write until the column is widened), then
(2) the `bigint` migration in a maintenance window using add-column+backfill+rename to
avoid a long exclusive lock. Until (2) lands, keep using local-file / full-load on `:5099`.

## Verification
- After (1): unit test that a > i32::MAX visit count round-trips in-memory.
- After (2): `SELECT max(visits) > 2147483647` becomes possible on newly-merged rows;
  `MIN_VISITS=<P99>` yields a small fast load; re-measure `:5099` load time with a high
  `SOCCER_LIVE_POLICY_MIN_VISITS`.

## Blast radius / coordination
Shared prod RDS + active AWS learner write to this table. The migration must be
coordinated (or done via the non-locking add/backfill/swap). **Do not** run a naive
`ALTER COLUMN TYPE` during active learning.
