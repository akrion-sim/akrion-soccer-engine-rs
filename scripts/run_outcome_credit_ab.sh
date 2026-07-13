#!/usr/bin/env bash
set -euo pipefail

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
BIN="$ROOT/target/release/soccer_outcome_ab_run"
OUT_DIR="${1:-$ROOT/out/outcome-credit-ab-$(date -u +%Y%m%dT%H%M%SZ)}"
TRAIN_GAMES="${TRAIN_GAMES:-60}"
TRAIN_MINUTES="${TRAIN_MINUTES:-0.35}"
EVAL_GAMES_PER_OPPONENT="${EVAL_GAMES_PER_OPPONENT:-8}"
EVAL_MINUTES="${EVAL_MINUTES:-0.35}"
ANALYTIC_POOL_SIZE="${ANALYTIC_POOL_SIZE:-5}"
TRAIN_ANALYTIC_POOL_SIZE="${TRAIN_ANALYTIC_POOL_SIZE:-4}"
TRAIN_SEED="${TRAIN_SEED:-C1200000}"
EVAL_SEED="${EVAL_SEED:-F2200000}"
WIN_REWARD="${WIN_REWARD:-200}"

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
  DD_SOCCER_ENABLE_CHANCE_QUALITY_REWARD=1
  DD_SOCCER_ENABLE_CHANCE_QUALITY_COMPOSITE=1
  DD_SOCCER_ENABLE_DP_BOOTSTRAP=0
  SOCCER_DYNAMIC_REWARD_WEIGHTS=1
  SOCCER_NEURAL_TARGET_SCALE=120
  SOCCER_NEURAL_TARGET_CLIP=15
  SOCCER_NEURAL_TARGET_POPART=1
  SOCCER_TRAIN_ANALYTIC_POOL_SIZE="$TRAIN_ANALYTIC_POOL_SIZE"
  SOCCER_TRAIN_ANALYTIC_ALTERNATE_SIDES=1
)

# Keep every non-credit variable identical. This isolates three replay semantics:
# no match result, the flat result broadcast, and production outcome-credit replay.
"${common_train_env[@]}" DD_SOCCER_OUTCOME_CREDIT=0 DD_SOCCER_ENABLE_MATCH_OUTCOME_REWARD=0 \
  "$BIN" train-analytic "$OUT_DIR/outcome-off.json" "$TRAIN_GAMES" "$TRAIN_MINUTES" "$TRAIN_SEED" \
  >"$OUT_DIR/outcome-off-train.log" 2>&1 &
pid_off=$!
"${common_train_env[@]}" DD_SOCCER_OUTCOME_CREDIT=0 DD_SOCCER_ENABLE_MATCH_OUTCOME_REWARD=1 DD_SOCCER_MATCH_WIN_REWARD_POINTS="$WIN_REWARD" \
  "$BIN" train-analytic "$OUT_DIR/flat-outcome.json" "$TRAIN_GAMES" "$TRAIN_MINUTES" "$TRAIN_SEED" \
  >"$OUT_DIR/flat-outcome-train.log" 2>&1 &
pid_flat=$!
"${common_train_env[@]}" DD_SOCCER_OUTCOME_CREDIT=1 DD_SOCCER_ENABLE_MATCH_OUTCOME_REWARD=1 DD_SOCCER_MATCH_WIN_REWARD_POINTS="$WIN_REWARD" \
  "$BIN" train-analytic "$OUT_DIR/outcome-credit.json" "$TRAIN_GAMES" "$TRAIN_MINUTES" "$TRAIN_SEED" \
  >"$OUT_DIR/outcome-credit-train.log" 2>&1 &
pid_credit=$!

wait "$pid_off"
wait "$pid_flat"
wait "$pid_credit"

for arm in outcome-off flat-outcome outcome-credit; do
  "$BIN" eval-analytic "$OUT_DIR/$arm.json" "$EVAL_GAMES_PER_OPPONENT" "$EVAL_MINUTES" \
    "$ANALYTIC_POOL_SIZE" "$EVAL_SEED" >"$OUT_DIR/$arm-eval.log" 2>&1 &
done
wait

for arm in outcome-off flat-outcome outcome-credit; do
  printf '\n[%s]\n' "$arm"
  grep -E 'held-out Elo:|payoff vs analytic field|completed forward passes/game:|candidate:|analytic:' \
    "$OUT_DIR/$arm-eval.log" || true
done

printf '\noutcome-credit A/B artifacts: %s\n' "$OUT_DIR"
