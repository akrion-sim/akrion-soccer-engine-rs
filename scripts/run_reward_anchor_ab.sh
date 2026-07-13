#!/usr/bin/env bash
set -euo pipefail

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
BIN="$ROOT/target/release/soccer_outcome_ab_run"
OUT_DIR="${1:-$ROOT/out/reward-anchor-ab-$(date -u +%Y%m%dT%H%M%SZ)}"
TRAIN_GAMES="${TRAIN_GAMES:-60}"
TRAIN_MINUTES="${TRAIN_MINUTES:-0.35}"
EVAL_GAMES_PER_OPPONENT="${EVAL_GAMES_PER_OPPONENT:-8}"
EVAL_MINUTES="${EVAL_MINUTES:-0.35}"
ANALYTIC_POOL_SIZE="${ANALYTIC_POOL_SIZE:-5}"
TRAIN_ANALYTIC_POOL_SIZE="${TRAIN_ANALYTIC_POOL_SIZE:-4}"
TRAIN_SEED="${TRAIN_SEED:-A1200000}"
EVAL_SEED="${EVAL_SEED:-E1200000}"

if [[ ! -x "$BIN" ]]; then
  cargo build --release --manifest-path "$ROOT/Cargo.toml" --bin soccer_outcome_ab_run
fi

mkdir -p "$OUT_DIR"

common_train_env=(
  env
  DD_SOCCER_ENABLE_CHANCE_QUALITY_REWARD=1
  DD_SOCCER_ENABLE_CHANCE_QUALITY_COMPOSITE=1
  SOCCER_DYNAMIC_REWARD_WEIGHTS=1
  SOCCER_NEURAL_TARGET_SCALE=120
  SOCCER_NEURAL_TARGET_CLIP=15
  SOCCER_NEURAL_TARGET_POPART=1
  SOCCER_TRAIN_ANALYTIC_POOL_SIZE="$TRAIN_ANALYTIC_POOL_SIZE"
  SOCCER_TRAIN_ANALYTIC_ALTERNATE_SIDES=1
)

"${common_train_env[@]}" DD_SOCCER_ENABLE_MATCH_OUTCOME_REWARD=0 \
  "$BIN" train-analytic "$OUT_DIR/outcome-off.json" "$TRAIN_GAMES" "$TRAIN_MINUTES" "$TRAIN_SEED" \
  >"$OUT_DIR/outcome-off-train.log" 2>&1 &
pid_off=$!
"${common_train_env[@]}" DD_SOCCER_ENABLE_MATCH_OUTCOME_REWARD=1 DD_SOCCER_MATCH_WIN_REWARD_POINTS=200 \
  "$BIN" train-analytic "$OUT_DIR/win200.json" "$TRAIN_GAMES" "$TRAIN_MINUTES" "$TRAIN_SEED" \
  >"$OUT_DIR/win200-train.log" 2>&1 &
pid_200=$!
"${common_train_env[@]}" DD_SOCCER_ENABLE_MATCH_OUTCOME_REWARD=1 DD_SOCCER_MATCH_WIN_REWARD_POINTS=400 \
  "$BIN" train-analytic "$OUT_DIR/win400.json" "$TRAIN_GAMES" "$TRAIN_MINUTES" "$TRAIN_SEED" \
  >"$OUT_DIR/win400-train.log" 2>&1 &
pid_400=$!

wait "$pid_off"
wait "$pid_200"
wait "$pid_400"

for arm in outcome-off win200 win400; do
  "$BIN" eval-analytic "$OUT_DIR/$arm.json" "$EVAL_GAMES_PER_OPPONENT" "$EVAL_MINUTES" \
    "$ANALYTIC_POOL_SIZE" "$EVAL_SEED" >"$OUT_DIR/$arm-eval.log" 2>&1 &
done
wait

for arm in outcome-off win200 win400; do
  printf '\n[%s]\n' "$arm"
  grep -E 'held-out Elo:|payoff vs analytic field|completed forward passes/game:|candidate:|analytic:' \
    "$OUT_DIR/$arm-eval.log" || true
done

printf '\nreward-anchor A/B artifacts: %s\n' "$OUT_DIR"
