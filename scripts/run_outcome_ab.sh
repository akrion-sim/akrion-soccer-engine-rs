#!/usr/bin/env bash
# Isolated A/B for the terminal won-game reward, judged on held-out Elo.
#
# Both arms train with advantage normalization ON, so the ONLY difference is the
# won-game label (DD_SOCCER_ENABLE_MATCH_OUTCOME_REWARD). The two training arms run
# in parallel (each its own process — the gate is OnceLock-cached per process), then
# a frozen head-to-head eval over a HELD-OUT seed range prints the promotion verdict.
#
# Usage: scripts/run_outcome_ab.sh [train_games=40] [eval_games=30] [minutes=3]
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd -P)"

TRAIN_GAMES="${1:-40}"
EVAL_GAMES="${2:-30}"
MINUTES="${3:-3}"
TARGET="${CARGO_TARGET_DIR:-.cargo-target-evalgate}"
OUT="$(mktemp -d -t soccer-outcome-ab.XXXXXX)"
BIN="$TARGET/release/soccer_outcome_ab_run"

echo "=== building soccer_outcome_ab_run (release) ==="
bash "$ROOT/scripts/prune_local_cargo_artifacts.sh" "$TARGET"
CARGO_INCREMENTAL="${CARGO_INCREMENTAL:-0}" CARGO_TARGET_DIR="$TARGET" cargo build --release --bin soccer_outcome_ab_run

echo "=== training both arms in parallel (train_games=$TRAIN_GAMES minutes=$MINUTES) ==="
echo "out dir: $OUT"
# Baseline: advantage norm ON, won-game reward OFF.
DD_SOCCER_ENABLE_ADVANTAGE_NORMALIZATION=1 \
  "$BIN" train "$OUT/baseline.json" "$TRAIN_GAMES" "$MINUTES" 71A10000 \
  > "$OUT/train_baseline.log" 2>&1 &
PID_BASE=$!
# Treatment: won-game reward ON (forces advantage norm on via the hardening coupling;
# set it explicitly too so the arms differ ONLY by the label).
DD_SOCCER_ENABLE_MATCH_OUTCOME_REWARD=1 DD_SOCCER_ENABLE_ADVANTAGE_NORMALIZATION=1 \
  "$BIN" train "$OUT/treatment.json" "$TRAIN_GAMES" "$MINUTES" 71A10000 \
  > "$OUT/train_treatment.log" 2>&1 &
PID_TREAT=$!

wait "$PID_BASE" || { echo "baseline train failed; see $OUT/train_baseline.log"; exit 1; }
wait "$PID_TREAT" || { echo "treatment train failed; see $OUT/train_treatment.log"; exit 1; }
echo "baseline:  $(tail -1 "$OUT/train_baseline.log")"
echo "treatment: $(tail -1 "$OUT/train_treatment.log")"

echo "=== held-out eval: treatment vs baseline (eval_games=$EVAL_GAMES) ==="
# Eval process: NO gates set (brains are frozen — no learning at eval).
"$BIN" eval "$OUT/treatment.json" "$OUT/baseline.json" "$EVAL_GAMES" "$MINUTES" E7A10000 \
  | tee "$OUT/eval.log"

echo ""
echo "artifacts in: $OUT"
