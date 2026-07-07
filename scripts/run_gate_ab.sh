#!/usr/bin/env bash
# Generic single-variable gate A/B for the climb effort (docs/how-to-climb-codex-conversation.md).
#
# baseline arm  = clean production defaults (fixed-73 interface only)
# treatment arm = production defaults + the TREATMENT_ENV gates turned on
# eval          = plays the two frozen brains head-to-head over a HELD-OUT disjoint seed
#                 range IN THE TREATMENT (lever-active) WORLD, printing the Wilson lower bound.
#
# Why baseline sets nothing: some gates (e.g. MULTIMODAL_RUN_PREDICTION) are `.is_ok()` —
# ANY value including "0" turns them ON — so "off" means UNSET. Baseline = incumbent defaults.
#
# The gate is a per-process OnceLock, so each arm is its own train process; the eval process
# reads the treatment gates once (lever-active world — the world where the treatment's training
# is relevant). Per-team head isolation does not exist, so eval applies the lever to BOTH brains;
# the question this answers is "did training WITH the lever produce a stronger policy for the
# lever-active world" — a detection test, not a diverse-field promotion proof.
#
# PRE-REGISTERED VERDICT: Wilson lower bound of treatment vs baseline > 0.500 = the lever climbs.
# Secondary chance-creation reads: GF/GA (goals/game), draw-rate.
#
# Usage: LABEL=offball TREATMENT_ENV="DD_SOCCER_ENABLE_MULTIMODAL_RUN_PREDICTION=1 DD_SOCCER_ENABLE_LEARNED_LONG_PASS_RUN=1" \
#        scripts/run_gate_ab.sh [train_games=80] [eval_games=140] [minutes=3]
set -uo pipefail
cd "$(dirname "$0")/.."

LABEL="${LABEL:?set LABEL}"
TREATMENT_ENV="${TREATMENT_ENV:?set TREATMENT_ENV (space-separated KEY=VAL gate enables)}"
TRAIN_GAMES="${1:-80}"
EVAL_GAMES="${2:-140}"
MINUTES="${3:-3}"
TARGET="${CARGO_TARGET_DIR:-/tmp/gate-ab-target}"
BIN="$TARGET/release/soccer_outcome_ab_run"
OUT="${OUT_DIR:-/tmp/gate-ab-$LABEL}"
mkdir -p "$OUT"

echo "=== [$(date -u +%FT%TZ)] gate A/B '$LABEL'  treatment='$TREATMENT_ENV' ==="
echo "=== building soccer_outcome_ab_run (release) into $TARGET ==="
CARGO_TARGET_DIR="$TARGET" nice -n 10 cargo build --release --bin soccer_outcome_ab_run \
  > "$OUT/build.log" 2>&1 || { echo "BUILD FAILED — see $OUT/build.log"; tail -20 "$OUT/build.log"; exit 1; }
echo "=== build ok ==="

echo "=== [$(date -u +%FT%TZ)] training both arms (train_games=$TRAIN_GAMES minutes=$MINUTES) ==="
# Baseline: clean incumbent defaults; only the interface freeze is explicit.
env -i PATH="$PATH" HOME="$HOME" DD_SOCCER_ENABLE_DISCRETIZED_KICK=0 \
  nice -n 12 "$BIN" train "$OUT/baseline.json" "$TRAIN_GAMES" "$MINUTES" 71A10000 \
  > "$OUT/train_baseline.log" 2>&1 &
PID_BASE=$!
# Treatment: incumbent defaults + the unspent lever gates on.
env -i PATH="$PATH" HOME="$HOME" DD_SOCCER_ENABLE_DISCRETIZED_KICK=0 $TREATMENT_ENV \
  nice -n 12 "$BIN" train "$OUT/treatment.json" "$TRAIN_GAMES" "$MINUTES" 71A10000 \
  > "$OUT/train_treatment.log" 2>&1 &
PID_TREAT=$!

wait "$PID_BASE"  || { echo "baseline train failed; see $OUT/train_baseline.log"; exit 1; }
wait "$PID_TREAT" || { echo "treatment train failed; see $OUT/train_treatment.log"; exit 1; }
echo "baseline:  $(tail -1 "$OUT/train_baseline.log")"
echo "treatment: $(tail -1 "$OUT/train_treatment.log")"

echo "=== [$(date -u +%FT%TZ)] held-out eval (lever-active world): treatment vs baseline, $EVAL_GAMES games ==="
env -i PATH="$PATH" HOME="$HOME" DD_SOCCER_ENABLE_DISCRETIZED_KICK=0 $TREATMENT_ENV \
  nice -n 12 "$BIN" eval "$OUT/treatment.json" "$OUT/baseline.json" "$EVAL_GAMES" "$MINUTES" E7A10000 \
  | tee "$OUT/verdict.log"

echo ""
echo "=== [$(date -u +%FT%TZ)] DONE '$LABEL'. artifacts in $OUT ; verdict in $OUT/verdict.log ==="
