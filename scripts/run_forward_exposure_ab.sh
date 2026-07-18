#!/usr/bin/env bash
set -euo pipefail

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
BIN="$ROOT/target/release/soccer_outcome_ab_run"
SOURCE_CHECKPOINT="${1:-$ROOT/out/forward-select-learning-ab-E120-F520/incumbent.json}"
OUT_DIR="${2:-$ROOT/out/forward-exposure-ab-$(date -u +%Y%m%dT%H%M%SZ)}"
EVAL_GAMES_PER_OPPONENT="${EVAL_GAMES_PER_OPPONENT:-8}"
EVAL_MINUTES="${EVAL_MINUTES:-0.35}"
ANALYTIC_POOL_SIZE="${ANALYTIC_POOL_SIZE:-5}"
EVAL_SEED="${EVAL_SEED:-F5400000}"
FORWARD_SELECT_WEIGHT="${FORWARD_SELECT_WEIGHT:-8.0}"

if [[ ! -f "$SOURCE_CHECKPOINT" ]]; then
  printf 'source checkpoint does not exist: %s\n' "$SOURCE_CHECKPOINT" >&2
  exit 2
fi

bash "$ROOT/scripts/prune_local_cargo_artifacts.sh" "${CARGO_TARGET_DIR:-$ROOT/target}"
CARGO_INCREMENTAL="${CARGO_INCREMENTAL:-0}" cargo build --release --manifest-path "$ROOT/Cargo.toml" --bin soccer_outcome_ab_run
mkdir -p "$OUT_DIR"

# Frozen-checkpoint mechanism screen:
#   control       = current no-floor path;
#   exposure      = preserve one safe pass-family root candidate;
#   exposure-bias = same exposure plus the checkpoint scalar at a fixed diagnostic dose.
env DD_SOCCER_ENABLE_FORWARD_RELEASE_ROOT_CANDIDATE=0 \
  SOCCER_NEURAL_MCTS_MIN_PASS_LIKE_ROOT_CANDIDATES=0 \
  DD_SOCCER_ENABLE_FORWARD_SELECT_LOGIT=0 \
  "$BIN" eval-analytic "$SOURCE_CHECKPOINT" "$EVAL_GAMES_PER_OPPONENT" "$EVAL_MINUTES" \
  "$ANALYTIC_POOL_SIZE" "$EVAL_SEED" >"$OUT_DIR/control-eval.log" 2>&1 &
pid_control=$!
env DD_SOCCER_ENABLE_FORWARD_RELEASE_ROOT_CANDIDATE=1 \
  SOCCER_NEURAL_MCTS_MIN_PASS_LIKE_ROOT_CANDIDATES=1 \
  DD_SOCCER_ENABLE_FORWARD_SELECT_LOGIT=0 \
  "$BIN" eval-analytic "$SOURCE_CHECKPOINT" "$EVAL_GAMES_PER_OPPONENT" "$EVAL_MINUTES" \
  "$ANALYTIC_POOL_SIZE" "$EVAL_SEED" >"$OUT_DIR/exposure-eval.log" 2>&1 &
pid_exposure=$!
env DD_SOCCER_ENABLE_FORWARD_RELEASE_ROOT_CANDIDATE=1 \
  SOCCER_NEURAL_MCTS_MIN_PASS_LIKE_ROOT_CANDIDATES=1 \
  DD_SOCCER_ENABLE_FORWARD_SELECT_LOGIT=1 \
  DD_SOCCER_FORWARD_SELECT_LOGIT_WEIGHT="$FORWARD_SELECT_WEIGHT" \
  "$BIN" eval-analytic "$SOURCE_CHECKPOINT" "$EVAL_GAMES_PER_OPPONENT" "$EVAL_MINUTES" \
  "$ANALYTIC_POOL_SIZE" "$EVAL_SEED" >"$OUT_DIR/exposure-bias-eval.log" 2>&1 &
pid_exposure_bias=$!

wait "$pid_control"
wait "$pid_exposure"
wait "$pid_exposure_bias"

printf '\nfixed exposure treatment: pass-family floor=1; bias weight=%s\n' \
  "$FORWARD_SELECT_WEIGHT"
for arm in control exposure exposure-bias; do
  printf '\n[%s]\n' "$arm"
  grep -E 'held-out Elo:|payoff vs analytic field|completed forward passes/game:|candidate:|analytic:' \
    "$OUT_DIR/$arm-eval.log" || true
done

printf '\nforward-exposure A/B artifacts: %s\n' "$OUT_DIR"
