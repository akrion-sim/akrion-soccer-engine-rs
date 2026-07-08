#!/usr/bin/env bash
# STEP-0 A-FALSIFIER (see docs/how-to-climb-codex-conversation.md).
#
# Question: under a FIXED 73-action interface, does the pitch_value potential-based
# reward (DD_SOCCER_ENABLE_PITCH_VALUE_REWARD) actually climb, or does it just re-rank
# without adding separation to the draw stalemate?
#
# Two arms differ ONLY by the pitch_value gate. Everything else identical:
#   - discretized-kick OFF  -> interface masked to the base 73 actions (buckets 73-112 inert)
#   - same seeds, same training budget, same net config
# Gate is a per-process OnceLock, so each arm is its own process (train), then a third
# process plays the two frozen brains head-to-head over a HELD-OUT (disjoint) seed range.
#
# PRE-REGISTERED VERDICT (Codex): Wilson lower bound of the pitch_value-ON arm vs the
# pitch_value-OFF baseline over held-out games.  > 0.500 = real life;  <= 0.500 = no.
# Secondary reads: draw-rate DOWN, goals/game UP, GD UP for the ON arm.
#
# Usage: scripts/run_pitch_value_ab.sh [train_games=80] [eval_games=160] [minutes=3]
set -uo pipefail
cd "$(dirname "$0")/.."

TRAIN_GAMES="${1:-80}"
EVAL_GAMES="${2:-160}"
MINUTES="${3:-3}"
TARGET="${CARGO_TARGET_DIR:-/tmp/pitchvalue-ab-target}"
BIN="$TARGET/release/soccer_outcome_ab_run"
OUT="${OUT_DIR:-/tmp/pitchvalue-ab}"
mkdir -p "$OUT"

# Interface freeze for BOTH arms — the only thing that makes this a fixed-73 test.
export DD_SOCCER_ENABLE_DISCRETIZED_KICK=0

echo "=== [$(date -u +%FT%TZ)] building soccer_outcome_ab_run (release) into $TARGET ==="
CARGO_TARGET_DIR="$TARGET" nice -n 10 cargo build --release --bin soccer_outcome_ab_run \
  > "$OUT/build.log" 2>&1 || { echo "BUILD FAILED — see $OUT/build.log"; tail -20 "$OUT/build.log"; exit 1; }
echo "=== build ok ==="

echo "=== [$(date -u +%FT%TZ)] training both arms (train_games=$TRAIN_GAMES minutes=$MINUTES) ==="
# Baseline arm: pitch_value OFF (current default behavior under fixed interface).
DD_SOCCER_ENABLE_PITCH_VALUE_REWARD=0 \
  nice -n 12 "$BIN" train "$OUT/pv_off.json" "$TRAIN_GAMES" "$MINUTES" 71A10000 \
  > "$OUT/train_pv_off.log" 2>&1 &
PID_OFF=$!
# Treatment arm: pitch_value ON.
DD_SOCCER_ENABLE_PITCH_VALUE_REWARD=1 \
  nice -n 12 "$BIN" train "$OUT/pv_on.json" "$TRAIN_GAMES" "$MINUTES" 71A10000 \
  > "$OUT/train_pv_on.log" 2>&1 &
PID_ON=$!

wait "$PID_OFF" || { echo "baseline (pv_off) train failed; see $OUT/train_pv_off.log"; exit 1; }
wait "$PID_ON"  || { echo "treatment (pv_on) train failed; see $OUT/train_pv_on.log"; exit 1; }
echo "pv_off: $(tail -1 "$OUT/train_pv_off.log")"
echo "pv_on:  $(tail -1 "$OUT/train_pv_on.log")"

echo "=== [$(date -u +%FT%TZ)] held-out eval: pv_on (candidate) vs pv_off (baseline), $EVAL_GAMES games ==="
# Eval process: NO reward gates (brains frozen — no learning at eval); interface stays 73.
nice -n 12 "$BIN" eval "$OUT/pv_on.json" "$OUT/pv_off.json" "$EVAL_GAMES" "$MINUTES" E7A10000 \
  | tee "$OUT/verdict.log"

echo ""
echo "=== [$(date -u +%FT%TZ)] DONE. artifacts in $OUT ; verdict in $OUT/verdict.log ==="
