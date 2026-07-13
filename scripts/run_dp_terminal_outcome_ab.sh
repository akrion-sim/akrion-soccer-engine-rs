#!/usr/bin/env bash
set -euo pipefail

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
BIN="$ROOT/target/release/soccer_outcome_ab_run"
OUT_DIR="${1:-$ROOT/out/dp-terminal-outcome-ab-$(date -u +%Y%m%dT%H%M%SZ)}"
TRAIN_GAMES="${TRAIN_GAMES:-60}"
TRAIN_MINUTES="${TRAIN_MINUTES:-0.35}"
EVAL_GAMES_PER_OPPONENT="${EVAL_GAMES_PER_OPPONENT:-8}"
EVAL_MINUTES="${EVAL_MINUTES:-0.35}"
ANALYTIC_POOL_SIZE="${ANALYTIC_POOL_SIZE:-5}"
TRAIN_ANALYTIC_POOL_SIZE="${TRAIN_ANALYTIC_POOL_SIZE:-4}"
TRAIN_SEED="${TRAIN_SEED:-D1200000}"
EVAL_SEED="${EVAL_SEED:-F3200000}"
WIN_REWARD="${WIN_REWARD:-200}"
TARGET_SCALE="${TARGET_SCALE:-30}"

if [[ ! "$WIN_REWARD" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  printf 'WIN_REWARD must be a non-negative number\n' >&2
  exit 2
fi

cargo build --release --manifest-path "$ROOT/Cargo.toml" --bin soccer_outcome_ab_run
effective_win_reward="$(env SOCCER_DYNAMIC_REWARD_WEIGHTS=1 DD_SOCCER_MATCH_WIN_REWARD_POINTS="$WIN_REWARD" \
  "$BIN" effective-win-reward)"
printf 'requested/effective win reward: %s/%s\n' "$WIN_REWARD" "$effective_win_reward"

mkdir -p "$OUT_DIR"

common_train_env=(
  env
  DD_SOCCER_OUTCOME_CREDIT=0
  DD_SOCCER_ENABLE_CHANCE_QUALITY_REWARD=1
  DD_SOCCER_ENABLE_CHANCE_QUALITY_COMPOSITE=1
  DD_SOCCER_ENABLE_DP_BOOTSTRAP=1
  DD_SOCCER_DP_BOOTSTRAP_HORIZON=64
  DD_SOCCER_DP_BOOTSTRAP_SWEEPS=200
  SOCCER_DYNAMIC_REWARD_WEIGHTS=1
  SOCCER_NEURAL_TARGET_SCALE="$TARGET_SCALE"
  SOCCER_NEURAL_TARGET_CLIP=15
  SOCCER_NEURAL_TARGET_POPART=1
  SOCCER_TRAIN_ANALYTIC_POOL_SIZE="$TRAIN_ANALYTIC_POOL_SIZE"
  SOCCER_TRAIN_ANALYTIC_ALTERNATE_SIDES=1
)

# Same DP learner and fixtures; only result-credit structure changes.
"${common_train_env[@]}" DD_SOCCER_ENABLE_DP_TERMINAL_OUTCOME_CREDIT=0 DD_SOCCER_ENABLE_MATCH_OUTCOME_REWARD=0 \
  "$BIN" train-analytic "$OUT_DIR/dp-no-outcome.json" "$TRAIN_GAMES" "$TRAIN_MINUTES" "$TRAIN_SEED" \
  >"$OUT_DIR/dp-no-outcome-train.log" 2>&1 &
pid_none=$!
"${common_train_env[@]}" DD_SOCCER_ENABLE_DP_TERMINAL_OUTCOME_CREDIT=0 DD_SOCCER_ENABLE_MATCH_OUTCOME_REWARD=1 DD_SOCCER_MATCH_WIN_REWARD_POINTS="$WIN_REWARD" \
  "$BIN" train-analytic "$OUT_DIR/dp-flat-outcome.json" "$TRAIN_GAMES" "$TRAIN_MINUTES" "$TRAIN_SEED" \
  >"$OUT_DIR/dp-flat-outcome-train.log" 2>&1 &
pid_flat=$!
"${common_train_env[@]}" DD_SOCCER_ENABLE_DP_TERMINAL_OUTCOME_CREDIT=1 DD_SOCCER_ENABLE_MATCH_OUTCOME_REWARD=1 DD_SOCCER_MATCH_WIN_REWARD_POINTS="$WIN_REWARD" \
  "$BIN" train-analytic "$OUT_DIR/dp-terminal-outcome.json" "$TRAIN_GAMES" "$TRAIN_MINUTES" "$TRAIN_SEED" \
  >"$OUT_DIR/dp-terminal-outcome-train.log" 2>&1 &
pid_terminal=$!

wait "$pid_none"
wait "$pid_flat"
wait "$pid_terminal"

for arm in dp-no-outcome dp-flat-outcome dp-terminal-outcome; do
  "$BIN" eval-analytic "$OUT_DIR/$arm.json" "$EVAL_GAMES_PER_OPPONENT" "$EVAL_MINUTES" \
    "$ANALYTIC_POOL_SIZE" "$EVAL_SEED" >"$OUT_DIR/$arm-eval.log" 2>&1 &
done
wait

for arm in dp-no-outcome dp-flat-outcome dp-terminal-outcome; do
  printf '\n[%s]\n' "$arm"
  grep -E 'held-out Elo:|payoff vs analytic field|completed forward passes/game:|candidate:|analytic:' \
    "$OUT_DIR/$arm-eval.log" || true
done

printf '\nDP terminal-outcome A/B artifacts: %s\n' "$OUT_DIR"
