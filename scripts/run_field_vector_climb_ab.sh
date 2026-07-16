#!/usr/bin/env bash
# Matched, bounded 11v11 climb A/B: legacy entity/Q-baseline/24-unit actor
# versus field-vector-v2/global-V-baseline/128-unit actor. Both arms start fresh,
# use the same seeds, fixtures, training budget, and held-out evaluation settings.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="${FIELD_VECTOR_AB_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${FIELD_VECTOR_AB_RUN_DIR:-$ROOT/tmp/field-vector-climb-ab-$STAMP}"
ROUNDS="${FIELD_VECTOR_AB_ROUNDS:-2}"
GAMES_PER_OPP="${FIELD_VECTOR_AB_GAMES_PER_OPP:-2}"
MINUTES="${FIELD_VECTOR_AB_MINUTES:-0.5}"
EVAL_GAMES="${FIELD_VECTOR_AB_EVAL_GAMES:-5}"
EVAL_MINUTES="${FIELD_VECTOR_AB_EVAL_MINUTES:-0.5}"
EVAL_POOL="${FIELD_VECTOR_AB_EVAL_POOL:-3}"

mkdir -p "$RUN_DIR/legacy/champions" "$RUN_DIR/v2/champions" "$RUN_DIR/logs"
cd "$ROOT"

bash "$ROOT/scripts/prune_local_cargo_artifacts.sh" "${CARGO_TARGET_DIR:-$ROOT/target}"
CARGO_INCREMENTAL="${CARGO_INCREMENTAL:-0}" cargo build --release --bin soccer_league_train --bin soccer_eval_gate_run \
  >"$RUN_DIR/logs/build.log" 2>&1

run_arm() {
  arm="$1"
  field_v2="$2"
  central_v2="$3"
  actor_hidden="$4"
  frontier="$RUN_DIR/$arm/learned-params.json"
  candidate="$RUN_DIR/$arm/learned-params.candidate.json"
  train_log="$RUN_DIR/logs/$arm-train.log"
  eval_log="$RUN_DIR/logs/$arm-eval.log"

  eval_status=0
  env \
    DD_SOCCER_ENABLE_FIELD_VECTOR_V2="$field_v2" \
    DD_SOCCER_ENABLE_CENTRAL_CRITIC_V2="$central_v2" \
    SOCCER_POLICY_HIDDEN_UNITS="$actor_hidden" \
    SOCCER_LEAGUE_FRONTIER="$frontier" \
    SOCCER_LEAGUE_CANDIDATE_FRONTIER="$candidate" \
    SOCCER_LEAGUE_ARCHIVE="$RUN_DIR/$arm/champions" \
    SOCCER_LEAGUE_SEED=20260714 \
    SOCCER_LEAGUE_MAX_ROUNDS="$ROUNDS" \
    SOCCER_LEAGUE_GAMES_PER_OPP="$GAMES_PER_OPP" \
    SOCCER_LEAGUE_MINUTES="$MINUTES" \
    SOCCER_LEAGUE_FRESH_OPPONENTS=1 \
    SOCCER_LEAGUE_PARALLELISM=1 \
    SOCCER_LEAGUE_SELF_PLAY_LADDER=0 \
    SOCCER_LEAGUE_FIXED_ANALYTIC_TRAINING_ANCHOR=1 \
    SOCCER_LEAGUE_ANALYTIC_OPPONENTS=1 \
    SOCCER_LEAGUE_CHECKPOINT_EVERY_ROUNDS=1 \
    SOCCER_LEAGUE_CHECKPOINT_VALIDATE_GAMES=0 \
    SOCCER_LEAGUE_ANALYTIC_ANCHOR_GATE=0 \
    SOCCER_LEAGUE_CHECKPOINT_REQUIRE_FORWARD_PASS_CLIMB=0 \
    SOCCER_LEAGUE_CHECKPOINT_MIN_FORWARD_PASS_MARGIN=-999 \
    SOCCER_LEAGUE_CHECKPOINT_VALIDATE_MIN_NET_FORWARD_PASS_MARGIN=-999 \
    SOCCER_LEAGUE_CHECKPOINT_VALIDATE_MIN_GOAL_DIFF_MARGIN=-999 \
    "$ROOT/target/release/soccer_league_train" >"$train_log" 2>&1

  eval_frontier="$candidate"
  if [ -s "$frontier" ]; then
    eval_frontier="$frontier"
  fi
  env \
    DD_SOCCER_ENABLE_FIELD_VECTOR_V2="$field_v2" \
    DD_SOCCER_ENABLE_CENTRAL_CRITIC_V2="$central_v2" \
    SOCCER_POLICY_HIDDEN_UNITS="$actor_hidden" \
    SOCCER_EVAL_ANALYTIC_FIELD=1 \
    SOCCER_EVAL_CANDIDATE_PATH="$eval_frontier" \
    SOCCER_EVAL_PARALLELISM=1 \
    "$ROOT/target/release/soccer_eval_gate_run" \
      "$EVAL_GAMES" "$EVAL_MINUTES" "$EVAL_POOL" 0 >"$eval_log" 2>&1 \
      || eval_status="$?"
  printf 'eval_exit_status=%s\n' "$eval_status" >>"$eval_log"
}

run_arm legacy 0 0 24
run_arm v2 1 1 128

summary="$RUN_DIR/summary.tsv"
printf 'arm\tfield_vector_v2\tcentral_critic_v2\tactor_hidden\tlast_train_kpi\teval_payoff\twilson_lower\teval_exit_status\n' >"$summary"
for arm in legacy v2; do
  if [ "$arm" = legacy ]; then
    fv=0; cv=0; hidden=24
  else
    fv=1; cv=1; hidden=128
  fi
  train_kpi="$(grep 'league_kpi ' "$RUN_DIR/logs/$arm-train.log" | tail -1 | tr '\t' ' ')"
  payoff="$(grep 'cross-play: mean payoff vs field' "$RUN_DIR/logs/$arm-eval.log" | tail -1 | sed -E 's/.*mean payoff vs field ([^ ]+).*/\1/' || true)"
  wilson="$(grep 'Wilson lower bound:' "$RUN_DIR/logs/$arm-eval.log" | tail -1 | awk '{print $4}' || true)"
  eval_status="$(grep 'eval_exit_status=' "$RUN_DIR/logs/$arm-eval.log" | tail -1 | cut -d= -f2 || true)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$arm" "$fv" "$cv" "$hidden" "$train_kpi" "${payoff:-unknown}" "${wilson:-unknown}" "${eval_status:-unknown}" >>"$summary"
done

printf 'field-vector climb A/B complete: %s\n' "$RUN_DIR"
printf 'summary: %s\n' "$summary"
