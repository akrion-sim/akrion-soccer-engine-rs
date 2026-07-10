#!/usr/bin/env bash
set -u

RUN_ID="${RUN_ID:-overnight-fullstack-proof-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_DIR="${OUT_DIR:-out/local-learning/${RUN_ID}}"
CLAUDE_OUTBOX="${CLAUDE_OUTBOX:-/tmp/codex_claude_outbox.jsonl}"
TRAIN_GAMES="${TRAIN_GAMES:-24}"
TRAIN_MINUTES="${TRAIN_MINUTES:-0.35}"
CKPT_EVERY="${CKPT_EVERY:-6}"
EVAL_GAMES="${EVAL_GAMES:-48}"
EVAL_MINUTES="${EVAL_MINUTES:-0.20}"
TRAIN_SEED="${TRAIN_SEED:-C0D40000}"
FRESH_SEED="${FRESH_SEED:-C0D30000}"
HOLDOUT_SEED="${HOLDOUT_SEED:-C0D50000}"
SKIP_BUILD="${SKIP_BUILD:-0}"

export RUN_ID OUT_DIR

mkdir -p "${OUT_DIR}"
LOG_PATH="${OUT_DIR}/overnight.log"

post_claude() {
  local message="$1"
  MESSAGE="${message}" RUN_ID="${RUN_ID}" OUT_DIR="${OUT_DIR}" CLAUDE_OUTBOX="${CLAUDE_OUTBOX}" python3 - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

path = Path(os.environ["CLAUDE_OUTBOX"])
payload = {
    "from": "codex",
    "topic": "soccer-learning-proof",
    "run_id": os.environ["RUN_ID"],
    "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "prompt": os.environ["MESSAGE"],
    "raw": {
        "from": "codex",
        "topic": "soccer-learning-proof",
        "message": os.environ["MESSAGE"],
        "out_dir": os.environ["OUT_DIR"],
    },
}
with path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
PY
}

run_step() {
  local label="$1"
  shift
  echo
  echo "===== ${label} ====="
  post_claude "overnight ${label} started for ${RUN_ID}; out=${OUT_DIR}"
  "$@"
  local status=$?
  post_claude "overnight ${label} finished for ${RUN_ID} with status=${status}; out=${OUT_DIR}"
  return "${status}"
}

summarize_artifacts() {
  python3 - <<'PY'
import json
from pathlib import Path

out = Path(__import__("os").environ["OUT_DIR"])
aggregate = out / "full-vs-fresh.aggregate.txt"
jsonl = out / "full-vs-fresh.jsonl"
summary = {
    "aggregate_exists": aggregate.exists(),
    "jsonl_rows": sum(1 for _ in jsonl.open("r", encoding="utf-8")) if jsonl.exists() else 0,
}
if aggregate.exists():
    lines = aggregate.read_text(encoding="utf-8", errors="replace").splitlines()
    summary["verdict"] = next((line for line in lines if line.startswith("VERDICT:")), "")
    summary["pass_turnover"] = next((line for line in lines if line.startswith("pass turnover rate:")), "")
    summary["forward"] = next((line for line in lines if line.startswith("completed FORWARD passes/game:")), "")
print(json.dumps(summary, sort_keys=True))
PY
}

{
  echo "run_id=${RUN_ID}"
  echo "out_dir=${OUT_DIR}"
  echo "train_games=${TRAIN_GAMES} train_minutes=${TRAIN_MINUTES} eval_games=${EVAL_GAMES} eval_minutes=${EVAL_MINUTES}"
  post_claude "overnight full-stack proof launched: ${RUN_ID}. Will train NN+DP+MPC/pass-completion/pass-receiver/support/spacing/carry/shooting/press/shape candidate, eval vs fresh gen0, aggregate with passing, turnover, dribble-success, shooting-quality, recovery, and teamwork guardrails, and report artifacts from ${OUT_DIR}."

  export SOCCER_FORMATION_LP_DETERMINISTIC=1
  export DD_SOCCER_ENABLE_LEARNED_PASS_COMPLETION=1
  export DD_SOCCER_ENABLE_LEARNED_PASS_RECEIVER=1
  export DD_SOCCER_ENABLE_LEARNED_MPC_OBJECTIVE=1
  export DD_SOCCER_ENABLE_LEARNED_MPC_SHOT_OBJECTIVE=1
  export DD_SOCCER_ENABLE_LEARNED_MPC_DRIBBLE_OBJECTIVE=1
  export DD_SOCCER_ENABLE_LEARNED_SPACING_TARGET=1
  export DD_SOCCER_ENABLE_SUPPORT_OUTCOME_REWARD=1
  export DD_SOCCER_ENABLE_DP_CRITIC_TARGET=1
  export DD_SOCCER_ENABLE_DP_BOOTSTRAP=1
  export DD_SOCCER_ENABLE_DEFERRED_PASS_CREDIT=1
  export DD_SOCCER_ENABLE_PROGRESSIVE_CARRY_REWARD=1
  export DD_SOCCER_ENABLE_CARRIER_FORWARD_DRIVE=1
  export DD_SOCCER_ENABLE_OVERDRIBBLE_PENALTY=1
  export DD_SOCCER_ENABLE_LEARNED_EPV="${DD_SOCCER_ENABLE_LEARNED_EPV:-1}"
  export DD_SOCCER_ENABLE_LEARNED_EPV_POTENTIAL="${DD_SOCCER_ENABLE_LEARNED_EPV_POTENTIAL:-1}"
  export DD_SOCCER_ENABLE_QUICK_FORWARD_PASS=1
  export DD_SOCCER_ENABLE_FORWARD_RUN_WHEN_UNMARKED=1
  export DD_SOCCER_ENABLE_CONTINUE_RUN_AFTER_PASS=1
  export DD_SOCCER_ENABLE_IN_STRIDE_PASS_MARGIN=1
  export DD_SOCCER_ENABLE_OFF_BALL_SPACE_DISCIPLINE=1
  export DD_SOCCER_ENABLE_BACKWARD_PASS_DISCIPLINE=1
  export DD_SOCCER_ENABLE_HOPELESS_PASS_VETO=1
  export DD_SOCCER_ENABLE_TERRIBLE_PASS_VETO=1
  export DD_SOCCER_ENABLE_PASS_OOB_PENALTY=1
  export DD_SOCCER_ENABLE_ROUTE_ONE_INTERCEPTION_CREDIT=1
  export DD_SOCCER_ENABLE_NUMBERS_UP_PRESS=1
  export DD_SOCCER_ENABLE_PRESS_COVER=1
  export DD_SOCCER_ENABLE_DEFENSIVE_SHEPHERD=1
  export DD_SOCCER_ENABLE_OFFSIDE_INFRACTION_PENALTY=1
  export DD_SOCCER_PASS_TURNOVER_PENALTY_SCALE="${DD_SOCCER_PASS_TURNOVER_PENALTY_SCALE:-7}"
  export DD_SOCCER_FORWARD_PASS_REWARD_SCALE="${DD_SOCCER_FORWARD_PASS_REWARD_SCALE:-3}"
  export DD_SOCCER_PASS_CHAIN_REWARD_SCALE="${DD_SOCCER_PASS_CHAIN_REWARD_SCALE:-4}"
  export DD_SOCCER_QUICK_FORWARD_RELEASE_REWARD_SCALE="${DD_SOCCER_QUICK_FORWARD_RELEASE_REWARD_SCALE:-1.5}"
  export DD_SOCCER_SHOT_SHAPING_REWARD_SCALE="${DD_SOCCER_SHOT_SHAPING_REWARD_SCALE:-0.6}"
  export DD_SOCCER_LEARNED_EPV_REWARD_SCALE="${DD_SOCCER_LEARNED_EPV_REWARD_SCALE:-20}"
  export SOCCER_OFFBALL_SUPPORT_REWARD_SCALE="${SOCCER_OFFBALL_SUPPORT_REWARD_SCALE:-3}"
  export SOCCER_PROOF_MAX_PASS_TURNOVER_RATE_DELTA="${SOCCER_PROOF_MAX_PASS_TURNOVER_RATE_DELTA:-0.0}"

  if [[ "${SKIP_BUILD}" == "1" || "${SKIP_BUILD}" == "true" ]]; then
    echo
    echo "===== build proof binaries skipped ====="
    post_claude "overnight build proof binaries skipped for ${RUN_ID}; using existing target/debug binaries; out=${OUT_DIR}"
  else
    run_step "build proof binaries" cargo build --bin soccer_proof --bin soccer_eval_gate_run --bin measure_learning_ab || exit $?
  fi
  run_step "fresh gen0 snapshot" target/debug/soccer_proof fresh "${OUT_DIR}/fresh.gen0.json" "${FRESH_SEED}" || exit $?
  run_step "train full-stack checkpoint" target/debug/soccer_proof train-ckpt "${OUT_DIR}/full" "${TRAIN_GAMES}" "${TRAIN_MINUTES}" "${TRAIN_SEED}" "${CKPT_EVERY}" || exit $?

  FINAL_SNAPSHOT="${OUT_DIR}/full.genFINAL.json"
  if [[ ! -f "${FINAL_SNAPSHOT}" ]]; then
    post_claude "overnight ${RUN_ID} stopped: missing final snapshot ${FINAL_SNAPSHOT}"
    exit 2
  fi

  export PROOF_JSONL="${OUT_DIR}/full-vs-fresh.jsonl"
  run_step "eval full vs fresh" target/debug/soccer_proof eval "${FINAL_SNAPSHOT}" "${OUT_DIR}/fresh.gen0.json" "${EVAL_GAMES}" "${EVAL_MINUTES}" "${HOLDOUT_SEED}" || true

  echo
  echo "===== aggregate proof JSONL ====="
  post_claude "overnight aggregate proof JSONL started for ${RUN_ID}; out=${OUT_DIR}"
  python3 scripts/proof_aggregate.py "${OUT_DIR}/full-vs-fresh.jsonl" > "${OUT_DIR}/full-vs-fresh.aggregate.txt"
  aggregate_status=$?
  post_claude "overnight aggregate proof JSONL finished for ${RUN_ID} with status=${aggregate_status}; out=${OUT_DIR}"

  SUMMARY="$(summarize_artifacts)"
  echo "summary=${SUMMARY}"
  post_claude "overnight full-stack proof completed for ${RUN_ID}. Summary: ${SUMMARY}. Artifacts: ${OUT_DIR}/full.genFINAL.json, ${OUT_DIR}/full-vs-fresh.jsonl, ${OUT_DIR}/full-vs-fresh.aggregate.txt"
} >> "${LOG_PATH}" 2>&1
