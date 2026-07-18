#!/usr/bin/env bash
set -euo pipefail

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
BIN="$ROOT/target/release/soccer_outcome_ab_run"
OUT_DIR="${1:-$ROOT/out/forward-select-learning-ab-$(date -u +%Y%m%dT%H%M%SZ)}"
TRAIN_GAMES="${TRAIN_GAMES:-60}"
TRAIN_MINUTES="${TRAIN_MINUTES:-0.35}"
EVAL_GAMES_PER_OPPONENT="${EVAL_GAMES_PER_OPPONENT:-8}"
EVAL_MINUTES="${EVAL_MINUTES:-0.35}"
ANALYTIC_POOL_SIZE="${ANALYTIC_POOL_SIZE:-5}"
TRAIN_ANALYTIC_POOL_SIZE="${TRAIN_ANALYTIC_POOL_SIZE:-4}"
TRAIN_SEED="${TRAIN_SEED:-E1200000}"
EVAL_SEED="${EVAL_SEED:-F5200000}"

bash "$ROOT/scripts/prune_local_cargo_artifacts.sh" "${CARGO_TARGET_DIR:-$ROOT/target}"
CARGO_INCREMENTAL="${CARGO_INCREMENTAL:-0}" cargo build --release --manifest-path "$ROOT/Cargo.toml" --bin soccer_outcome_ab_run
mkdir -p "$OUT_DIR"

common_train_env=(
  env
  DD_SOCCER_OUTCOME_CREDIT=0
  DD_SOCCER_ENABLE_MATCH_OUTCOME_REWARD=1
  DD_SOCCER_ENABLE_CHANCE_QUALITY_REWARD=1
  DD_SOCCER_ENABLE_CHANCE_QUALITY_COMPOSITE=1
  DD_SOCCER_ENABLE_DP_BOOTSTRAP=1
  DD_SOCCER_ENABLE_DP_TERMINAL_OUTCOME_CREDIT=0
  DD_SOCCER_DP_BOOTSTRAP_HORIZON=64
  DD_SOCCER_DP_BOOTSTRAP_SWEEPS=200
  SOCCER_DYNAMIC_REWARD_WEIGHTS=1
  SOCCER_NEURAL_TARGET_SCALE=30
  SOCCER_NEURAL_TARGET_CLIP=15
  SOCCER_NEURAL_TARGET_POPART=1
  SOCCER_TRAIN_ANALYTIC_POOL_SIZE="$TRAIN_ANALYTIC_POOL_SIZE"
  SOCCER_TRAIN_ANALYTIC_ALTERNATE_SIDES=1
)

# Arm 1: incumbent actor. Arm 2: scalar learns only from executed forward-pass samples.
# Arm 3: planner-teacher adds qualified missed forward opportunities to break the cold-start loop.
"${common_train_env[@]}" DD_SOCCER_ENABLE_FORWARD_SELECT_LOGIT=0 DD_SOCCER_ENABLE_PLANNER_TEACHER_MISSED_OPPORTUNITY=0 \
  "$BIN" train-analytic "$OUT_DIR/incumbent.json" "$TRAIN_GAMES" "$TRAIN_MINUTES" "$TRAIN_SEED" \
  >"$OUT_DIR/incumbent-train.log" 2>&1 &
pid_incumbent=$!
"${common_train_env[@]}" DD_SOCCER_ENABLE_FORWARD_SELECT_LOGIT=1 DD_SOCCER_ENABLE_PLANNER_TEACHER_MISSED_OPPORTUNITY=0 \
  "$BIN" train-analytic "$OUT_DIR/forward-learned.json" "$TRAIN_GAMES" "$TRAIN_MINUTES" "$TRAIN_SEED" \
  >"$OUT_DIR/forward-learned-train.log" 2>&1 &
pid_learned=$!
"${common_train_env[@]}" DD_SOCCER_ENABLE_FORWARD_SELECT_LOGIT=1 DD_SOCCER_ENABLE_PLANNER_TEACHER_MISSED_OPPORTUNITY=1 \
  "$BIN" train-analytic "$OUT_DIR/forward-teacher.json" "$TRAIN_GAMES" "$TRAIN_MINUTES" "$TRAIN_SEED" \
  >"$OUT_DIR/forward-teacher-train.log" 2>&1 &
pid_teacher=$!

wait "$pid_incumbent"
wait "$pid_learned"
wait "$pid_teacher"

printf 'learned forward-select weights: incumbent=%s learned=%s teacher=%s\n' \
  "$(jq -r '.policyHead.forwardSelectLogitWeight // 0' "$OUT_DIR/incumbent.json")" \
  "$(jq -r '.policyHead.forwardSelectLogitWeight // 0' "$OUT_DIR/forward-learned.json")" \
  "$(jq -r '.policyHead.forwardSelectLogitWeight // 0' "$OUT_DIR/forward-teacher.json")"

env DD_SOCCER_ENABLE_FORWARD_SELECT_LOGIT=0 DD_SOCCER_ENABLE_PLANNER_TEACHER_MISSED_OPPORTUNITY=0 \
  "$BIN" eval-analytic "$OUT_DIR/incumbent.json" "$EVAL_GAMES_PER_OPPONENT" "$EVAL_MINUTES" \
  "$ANALYTIC_POOL_SIZE" "$EVAL_SEED" >"$OUT_DIR/incumbent-eval.log" 2>&1 &
pid_eval_incumbent=$!
env DD_SOCCER_ENABLE_FORWARD_SELECT_LOGIT=1 DD_SOCCER_ENABLE_PLANNER_TEACHER_MISSED_OPPORTUNITY=0 \
  "$BIN" eval-analytic "$OUT_DIR/forward-learned.json" "$EVAL_GAMES_PER_OPPONENT" "$EVAL_MINUTES" \
  "$ANALYTIC_POOL_SIZE" "$EVAL_SEED" >"$OUT_DIR/forward-learned-eval.log" 2>&1 &
pid_eval_learned=$!
env DD_SOCCER_ENABLE_FORWARD_SELECT_LOGIT=1 DD_SOCCER_ENABLE_PLANNER_TEACHER_MISSED_OPPORTUNITY=1 \
  "$BIN" eval-analytic "$OUT_DIR/forward-teacher.json" "$EVAL_GAMES_PER_OPPONENT" "$EVAL_MINUTES" \
  "$ANALYTIC_POOL_SIZE" "$EVAL_SEED" >"$OUT_DIR/forward-teacher-eval.log" 2>&1 &
pid_eval_teacher=$!

wait "$pid_eval_incumbent"
wait "$pid_eval_learned"
wait "$pid_eval_teacher"

for arm in incumbent forward-learned forward-teacher; do
  printf '\n[%s]\n' "$arm"
  grep -E 'held-out Elo:|payoff vs analytic field|completed forward passes/game:|candidate:|analytic:' \
    "$OUT_DIR/$arm-eval.log" || true
done

printf '\nforward-select learning A/B artifacts: %s\n' "$OUT_DIR"
