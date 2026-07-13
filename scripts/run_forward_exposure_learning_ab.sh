#!/usr/bin/env bash
set -euo pipefail

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
BIN="$ROOT/target/release/soccer_outcome_ab_run"
OUT_DIR="${1:-$ROOT/out/forward-exposure-learning-ab-$(date -u +%Y%m%dT%H%M%SZ)}"
TRAIN_GAMES="${TRAIN_GAMES:-60}"
TRAIN_MINUTES="${TRAIN_MINUTES:-0.35}"
EVAL_GAMES_PER_OPPONENT="${EVAL_GAMES_PER_OPPONENT:-8}"
EVAL_MINUTES="${EVAL_MINUTES:-0.35}"
ANALYTIC_POOL_SIZE="${ANALYTIC_POOL_SIZE:-5}"
TRAIN_ANALYTIC_POOL_SIZE="${TRAIN_ANALYTIC_POOL_SIZE:-4}"
TRAIN_SEED="${TRAIN_SEED:-E1300000}"
EVAL_SEED="${EVAL_SEED:-F5500000}"

cargo build --release --manifest-path "$ROOT/Cargo.toml" --bin soccer_outcome_ab_run
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
  DD_SOCCER_ENABLE_FORWARD_SELECT_LOGIT=0
  SOCCER_DYNAMIC_REWARD_WEIGHTS=1
  SOCCER_NEURAL_TARGET_SCALE=30
  SOCCER_NEURAL_TARGET_CLIP=15
  SOCCER_NEURAL_TARGET_POPART=1
  SOCCER_TRAIN_ANALYTIC_POOL_SIZE="$TRAIN_ANALYTIC_POOL_SIZE"
  SOCCER_TRAIN_ANALYTIC_ALTERNATE_SIDES=1
)

# Control keeps the current root. Exposure preserves one safe pass-family root candidate.
# Exposure-teacher adds counterfactual planner samples only after that candidate is expressible.
"${common_train_env[@]}" DD_SOCCER_ENABLE_FORWARD_RELEASE_ROOT_CANDIDATE=0 \
  SOCCER_NEURAL_MCTS_MIN_PASS_LIKE_ROOT_CANDIDATES=0 \
  DD_SOCCER_ENABLE_PLANNER_TEACHER_MISSED_OPPORTUNITY=0 \
  "$BIN" train-analytic "$OUT_DIR/control.json" "$TRAIN_GAMES" "$TRAIN_MINUTES" "$TRAIN_SEED" \
  >"$OUT_DIR/control-train.log" 2>&1 &
pid_control=$!
"${common_train_env[@]}" DD_SOCCER_ENABLE_FORWARD_RELEASE_ROOT_CANDIDATE=1 \
  SOCCER_NEURAL_MCTS_MIN_PASS_LIKE_ROOT_CANDIDATES=1 \
  DD_SOCCER_ENABLE_PLANNER_TEACHER_MISSED_OPPORTUNITY=0 \
  "$BIN" train-analytic "$OUT_DIR/exposure.json" "$TRAIN_GAMES" "$TRAIN_MINUTES" "$TRAIN_SEED" \
  >"$OUT_DIR/exposure-train.log" 2>&1 &
pid_exposure=$!
"${common_train_env[@]}" DD_SOCCER_ENABLE_FORWARD_RELEASE_ROOT_CANDIDATE=1 \
  SOCCER_NEURAL_MCTS_MIN_PASS_LIKE_ROOT_CANDIDATES=1 \
  DD_SOCCER_ENABLE_PLANNER_TEACHER_MISSED_OPPORTUNITY=1 \
  "$BIN" train-analytic "$OUT_DIR/exposure-teacher.json" "$TRAIN_GAMES" "$TRAIN_MINUTES" "$TRAIN_SEED" \
  >"$OUT_DIR/exposure-teacher-train.log" 2>&1 &
pid_exposure_teacher=$!

wait "$pid_control"
wait "$pid_exposure"
wait "$pid_exposure_teacher"

env DD_SOCCER_ENABLE_FORWARD_RELEASE_ROOT_CANDIDATE=0 \
  SOCCER_NEURAL_MCTS_MIN_PASS_LIKE_ROOT_CANDIDATES=0 \
  DD_SOCCER_ENABLE_FORWARD_SELECT_LOGIT=0 \
  DD_SOCCER_ENABLE_PLANNER_TEACHER_MISSED_OPPORTUNITY=0 \
  "$BIN" eval-analytic "$OUT_DIR/control.json" "$EVAL_GAMES_PER_OPPONENT" "$EVAL_MINUTES" \
  "$ANALYTIC_POOL_SIZE" "$EVAL_SEED" >"$OUT_DIR/control-eval.log" 2>&1 &
pid_eval_control=$!
env DD_SOCCER_ENABLE_FORWARD_RELEASE_ROOT_CANDIDATE=1 \
  SOCCER_NEURAL_MCTS_MIN_PASS_LIKE_ROOT_CANDIDATES=1 \
  DD_SOCCER_ENABLE_FORWARD_SELECT_LOGIT=0 \
  DD_SOCCER_ENABLE_PLANNER_TEACHER_MISSED_OPPORTUNITY=0 \
  "$BIN" eval-analytic "$OUT_DIR/exposure.json" "$EVAL_GAMES_PER_OPPONENT" "$EVAL_MINUTES" \
  "$ANALYTIC_POOL_SIZE" "$EVAL_SEED" >"$OUT_DIR/exposure-eval.log" 2>&1 &
pid_eval_exposure=$!
env DD_SOCCER_ENABLE_FORWARD_RELEASE_ROOT_CANDIDATE=1 \
  SOCCER_NEURAL_MCTS_MIN_PASS_LIKE_ROOT_CANDIDATES=1 \
  DD_SOCCER_ENABLE_FORWARD_SELECT_LOGIT=0 \
  DD_SOCCER_ENABLE_PLANNER_TEACHER_MISSED_OPPORTUNITY=1 \
  "$BIN" eval-analytic "$OUT_DIR/exposure-teacher.json" "$EVAL_GAMES_PER_OPPONENT" "$EVAL_MINUTES" \
  "$ANALYTIC_POOL_SIZE" "$EVAL_SEED" >"$OUT_DIR/exposure-teacher-eval.log" 2>&1 &
pid_eval_exposure_teacher=$!

wait "$pid_eval_control"
wait "$pid_eval_exposure"
wait "$pid_eval_exposure_teacher"

for arm in control exposure exposure-teacher; do
  printf '\n[%s]\n' "$arm"
  grep -E 'held-out Elo:|payoff vs analytic field|completed forward passes/game:|candidate:|analytic:' \
    "$OUT_DIR/$arm-eval.log" || true
done

printf '\nforward-exposure learning A/B artifacts: %s\n' "$OUT_DIR"
