#!/usr/bin/env bash
set -euo pipefail

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
BIN="$ROOT/target/release/soccer_outcome_ab_run"
SOURCE_CHECKPOINT="${1:-$ROOT/out/forward-select-learning-ab-E120-F520/incumbent.json}"
OUT_DIR="${2:-$ROOT/out/forward-select-dose-ab-$(date -u +%Y%m%dT%H%M%SZ)}"
EVAL_GAMES_PER_OPPONENT="${EVAL_GAMES_PER_OPPONENT:-8}"
EVAL_MINUTES="${EVAL_MINUTES:-0.35}"
ANALYTIC_POOL_SIZE="${ANALYTIC_POOL_SIZE:-5}"
EVAL_SEED="${EVAL_SEED:-F5300000}"

if [[ ! -f "$SOURCE_CHECKPOINT" ]]; then
  printf 'source checkpoint does not exist: %s\n' "$SOURCE_CHECKPOINT" >&2
  exit 2
fi

cargo build --release --manifest-path "$ROOT/Cargo.toml" --bin soccer_outcome_ab_run
mkdir -p "$OUT_DIR"

# Isolate the runtime dose on one frozen actor. The learned checkpoint is not retrained;
# restore-time overrides install the requested scalar in each independent evaluation process.
env DD_SOCCER_ENABLE_FORWARD_SELECT_LOGIT=0 \
  "$BIN" eval-analytic "$SOURCE_CHECKPOINT" "$EVAL_GAMES_PER_OPPONENT" "$EVAL_MINUTES" \
  "$ANALYTIC_POOL_SIZE" "$EVAL_SEED" >"$OUT_DIR/weight-0-eval.log" 2>&1 &
pid_0=$!
env DD_SOCCER_ENABLE_FORWARD_SELECT_LOGIT=1 DD_SOCCER_FORWARD_SELECT_LOGIT_WEIGHT=0.05 \
  "$BIN" eval-analytic "$SOURCE_CHECKPOINT" "$EVAL_GAMES_PER_OPPONENT" "$EVAL_MINUTES" \
  "$ANALYTIC_POOL_SIZE" "$EVAL_SEED" >"$OUT_DIR/weight-005-eval.log" 2>&1 &
pid_005=$!
env DD_SOCCER_ENABLE_FORWARD_SELECT_LOGIT=1 DD_SOCCER_FORWARD_SELECT_LOGIT_WEIGHT=0.20 \
  "$BIN" eval-analytic "$SOURCE_CHECKPOINT" "$EVAL_GAMES_PER_OPPONENT" "$EVAL_MINUTES" \
  "$ANALYTIC_POOL_SIZE" "$EVAL_SEED" >"$OUT_DIR/weight-020-eval.log" 2>&1 &
pid_020=$!

wait "$pid_0"
wait "$pid_005"
wait "$pid_020"

for arm in weight-0 weight-005 weight-020; do
  printf '\n[%s]\n' "$arm"
  grep -E 'held-out Elo:|payoff vs analytic field|completed forward passes/game:|candidate:|analytic:' \
    "$OUT_DIR/$arm-eval.log" || true
done

printf '\nforward-select fixed-dose A/B artifacts: %s\n' "$OUT_DIR"
