#!/usr/bin/env bash
# Step 1 of the offline distilled-encoder harness (see docs/architecture-audit.md).
# Exports a supervised (state -> action value) dataset from the cluster learner's
# logged transitions to local JSONL, for offline encoder/head training & distillation.
#
#   SOCCER_DATABASE_URL=postgres://... \
#   scripts/export_offline_dataset.sh <experiment-slug> <num-recent-runs> <out.jsonl>
#
# Each JSONL line: {state_key:{...~205 binned engineered features...}, action,
#                   value_micros, weight_micros, target_fine, target_tactical}
# Verified shape (overnight gen-308): 47 actions, 205 features (int/bool/str bins),
# value target ~[-7, 11], visit-weighted.
#
# NOTE: filter by run_id (indexed FK), NOT the created_at/experiment 3-way join —
# the deltas table is ~10M rows and the join + DESC sort times out. Resolve the
# experiment's most-recent run ids first (small), then stream their deltas.
set -euo pipefail
SLUG="${1:-soccer-self-play-k8s-overnight}"
RUNS="${2:-40}"
OUT="${3:-/tmp/soccer_offline_${SLUG}.jsonl}"
MAXROWS="${4:-100000}"
: "${SOCCER_DATABASE_URL:?set SOCCER_DATABASE_URL}"
RUNIDS=$(psql "$SOCCER_DATABASE_URL" -X -t -A -v ON_ERROR_STOP=1 -c "
  SELECT string_agg(quote_literal(id::text), ',')
  FROM (SELECT id FROM des_soccer_learning_runs
        WHERE experiment_id=(SELECT id FROM des_soccer_learning_experiments WHERE slug='${SLUG}')
        ORDER BY updated_at DESC LIMIT ${RUNS}) r;")
[ -z "${RUNIDS}" ] && { echo "no runs for experiment ${SLUG}"; exit 1; }
psql "$SOCCER_DATABASE_URL" -X -q -v ON_ERROR_STOP=1 -c "\copy (
  SELECT json_build_object(
           'state_key', state_key, 'action', action,
           'value_micros', after_value_micros, 'weight_micros', effective_visit_micros,
           'target_fine', target_fine_cell_id, 'target_tactical', target_tactical_cell_id)
  FROM des_soccer_learning_run_deltas
  WHERE run_id IN (${RUNIDS}) LIMIT ${MAXROWS}
) TO STDOUT" > "$OUT"
echo "exported $(wc -l < "$OUT") rows from ${SLUG} (last ${RUNS} runs) -> $OUT ($(du -h "$OUT" | cut -f1))"
