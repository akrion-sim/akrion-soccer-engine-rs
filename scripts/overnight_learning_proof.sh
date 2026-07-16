#!/usr/bin/env bash
set -u

RUN_ID="${RUN_ID:-overnight-fullstack-proof-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_DIR="${OUT_DIR:-out/local-learning/${RUN_ID}}"
CLAUDE_OUTBOX="${CLAUDE_OUTBOX:-/tmp/codex_claude_outbox.jsonl}"
TRAIN_GAMES="${TRAIN_GAMES:-24}"
TRAIN_MINUTES="${TRAIN_MINUTES:-0.35}"
CKPT_EVERY="${CKPT_EVERY:-6}"
EVAL_GAMES="${EVAL_GAMES:-96}"
EVAL_MINUTES="${EVAL_MINUTES:-0.20}"
TRAIN_SEED="${TRAIN_SEED:-C0D40000}"
FRESH_SEED="${FRESH_SEED:-C0D30000}"
HOLDOUT_SEED="${HOLDOUT_SEED:-C0D50000}"
SKIP_BUILD="${SKIP_BUILD:-0}"
RUN_HEAD_ABLATIONS="${RUN_HEAD_ABLATIONS:-0}"
ABLATION_EVAL_GAMES="${ABLATION_EVAL_GAMES:-${EVAL_GAMES}}"
RUN_DP_ABLATION="${RUN_DP_ABLATION:-0}"
DP_ABLATION_TRAIN_GAMES="${DP_ABLATION_TRAIN_GAMES:-${TRAIN_GAMES}}"
DP_ABLATION_EVAL_GAMES="${DP_ABLATION_EVAL_GAMES:-${ABLATION_EVAL_GAMES}}"
RUN_ANALYTIC_EVAL="${RUN_ANALYTIC_EVAL:-1}"
ANALYTIC_EVAL_GAMES="${ANALYTIC_EVAL_GAMES:-${EVAL_GAMES}}"
ANALYTIC_EVAL_MINUTES="${ANALYTIC_EVAL_MINUTES:-${EVAL_MINUTES}}"
ANALYTIC_HOLDOUT_SEED="${ANALYTIC_HOLDOUT_SEED:-${HOLDOUT_SEED}}"
RUN_FWD_TRACE="${RUN_FWD_TRACE:-1}"
FORCE_PROOF_WITH_ACTIVE_TRAINERS="${FORCE_PROOF_WITH_ACTIVE_TRAINERS:-0}"
FAST_LEARNING_PROFILE="${FAST_LEARNING_PROFILE:-0}"
CEILING_BREAK_PROFILE="${CEILING_BREAK_PROFILE:-0}"
BOTTLENECK_INPUT_PROFILE="${BOTTLENECK_INPUT_PROFILE:-0}"
COMPACT_INPUT_PROFILE="${COMPACT_INPUT_PROFILE:-0}"

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

active_soccer_league_train_report() {
  if ! command -v lsof >/dev/null 2>&1; then
    return 0
  fi
  local repo_root
  repo_root="$(pwd -P)"
  lsof -c soccer_league_train 2>/dev/null | grep -F "${repo_root}" || true
}

guard_no_active_soccer_league_train() {
  if [[ "${FORCE_PROOF_WITH_ACTIVE_TRAINERS}" == "1" || "${FORCE_PROOF_WITH_ACTIVE_TRAINERS}" == "true" ]]; then
    echo "active trainer preflight overridden"
    post_claude "overnight ${RUN_ID} active-trainer preflight overridden; proceeding despite any soccer_league_train process in this repo."
    return 0
  fi

  local active_trainers
  active_trainers="$(active_soccer_league_train_report)"
  if [[ -n "${active_trainers}" ]]; then
    echo "active soccer_league_train detected; refusing to stack another proof run"
    printf '%s\n' "${active_trainers}"
    post_claude "overnight ${RUN_ID} refused to launch: active soccer_league_train is already running in this repo. Stop it first, or set FORCE_PROOF_WITH_ACTIVE_TRAINERS=1 to override intentionally."
    exit 3
  fi
}

apply_proof_profiles() {
  local bottleneck_active="0"
  local compact_active="0"
  if [[ "${BOTTLENECK_INPUT_PROFILE}" == "1" || "${BOTTLENECK_INPUT_PROFILE}" == "true" ]]; then
    bottleneck_active="1"
  fi
  if [[ "${COMPACT_INPUT_PROFILE}" == "1" || "${COMPACT_INPUT_PROFILE}" == "true" ]]; then
    compact_active="1"
  fi
  if [[ "${bottleneck_active}" == "1" && "${compact_active}" == "1" ]]; then
    post_claude "overnight ${RUN_ID} refused to launch: BOTTLENECK_INPUT_PROFILE and COMPACT_INPUT_PROFILE are mutually exclusive dimensionality experiments."
    echo "BOTTLENECK_INPUT_PROFILE and COMPACT_INPUT_PROFILE are mutually exclusive" >&2
    exit 4
  fi

  if [[ "${FAST_LEARNING_PROFILE}" == "1" || "${FAST_LEARNING_PROFILE}" == "true" ]]; then
    export SOCCER_NEURAL_LEARNING_RATE="${SOCCER_NEURAL_LEARNING_RATE:-0.03}"
    export SOCCER_NEURAL_MAX_BATCHES_PER_TICK="${SOCCER_NEURAL_MAX_BATCHES_PER_TICK:-4}"
    export SOCCER_NEURAL_REPLAY_SAMPLES_PER_TICK="${SOCCER_NEURAL_REPLAY_SAMPLES_PER_TICK:-32}"
    export SOCCER_NEURAL_TARGET_POPART="${SOCCER_NEURAL_TARGET_POPART:-1}"
    export SOCCER_NEURAL_TARGET_CLIP="${SOCCER_NEURAL_TARGET_CLIP:-15}"
    post_claude "overnight ${RUN_ID} FAST_LEARNING_PROFILE active: lr=${SOCCER_NEURAL_LEARNING_RATE} max_batches=${SOCCER_NEURAL_MAX_BATCHES_PER_TICK} replay_samples=${SOCCER_NEURAL_REPLAY_SAMPLES_PER_TICK} target_clip=${SOCCER_NEURAL_TARGET_CLIP} popart=${SOCCER_NEURAL_TARGET_POPART}"
  fi

  if [[ "${CEILING_BREAK_PROFILE}" == "1" || "${CEILING_BREAK_PROFILE}" == "true" ]]; then
    export SOCCER_NEURAL_HIDDEN_UNITS="${SOCCER_NEURAL_HIDDEN_UNITS:-64}"
    export SOCCER_POLICY_HIDDEN_UNITS="${SOCCER_POLICY_HIDDEN_UNITS:-64}"
    export SOCCER_POLICY_ENTROPY_COEFF="${SOCCER_POLICY_ENTROPY_COEFF:-0.02}"
    export DD_SOCCER_PASS_TURNOVER_PENALTY_SCALE="${DD_SOCCER_PASS_TURNOVER_PENALTY_SCALE:-10}"
    export DD_SOCCER_ENABLE_ACTION_PARAM_FEATURES="${DD_SOCCER_ENABLE_ACTION_PARAM_FEATURES:-1}"
    export DD_SOCCER_ENABLE_DISCRETIZED_KICK="${DD_SOCCER_ENABLE_DISCRETIZED_KICK:-1}"
    export DD_SOCCER_ENABLE_DISCRETIZED_KICK_DITHER="${DD_SOCCER_ENABLE_DISCRETIZED_KICK_DITHER:-1}"
    export DD_SOCCER_ENABLE_FORWARD_RELEASE_ROOT_CANDIDATE="${DD_SOCCER_ENABLE_FORWARD_RELEASE_ROOT_CANDIDATE:-1}"
    export SOCCER_NEURAL_MCTS_MIN_PASS_LIKE_ROOT_CANDIDATES="${SOCCER_NEURAL_MCTS_MIN_PASS_LIKE_ROOT_CANDIDATES:-1}"
    export SOCCER_NEURAL_MCTS_MIN_DISCRETIZED_KICK_CANDIDATES="${SOCCER_NEURAL_MCTS_MIN_DISCRETIZED_KICK_CANDIDATES:-1}"
    export SOCCER_NEURAL_MCTS_DISCRETIZED_KICK_FORCE_FLOOR="${SOCCER_NEURAL_MCTS_DISCRETIZED_KICK_FORCE_FLOOR:-1}"
    export SOCCER_NEURAL_TARGET_POPART="${SOCCER_NEURAL_TARGET_POPART:-1}"
    export SOCCER_NEURAL_TARGET_CLIP="${SOCCER_NEURAL_TARGET_CLIP:-15}"
    post_claude "overnight ${RUN_ID} CEILING_BREAK_PROFILE active: hidden=${SOCCER_NEURAL_HIDDEN_UNITS} policy_hidden=${SOCCER_POLICY_HIDDEN_UNITS} entropy=${SOCCER_POLICY_ENTROPY_COEFF} turnover_scale=${DD_SOCCER_PASS_TURNOVER_PENALTY_SCALE} action_param_features=${DD_SOCCER_ENABLE_ACTION_PARAM_FEATURES} discretized_kick=${DD_SOCCER_ENABLE_DISCRETIZED_KICK} kick_dither=${DD_SOCCER_ENABLE_DISCRETIZED_KICK_DITHER} forward_root_inject=${DD_SOCCER_ENABLE_FORWARD_RELEASE_ROOT_CANDIDATE} pass_like_floor=${SOCCER_NEURAL_MCTS_MIN_PASS_LIKE_ROOT_CANDIDATES} kick_floor=${SOCCER_NEURAL_MCTS_MIN_DISCRETIZED_KICK_CANDIDATES} target_clip=${SOCCER_NEURAL_TARGET_CLIP} popart=${SOCCER_NEURAL_TARGET_POPART}"
  fi

  if [[ "${bottleneck_active}" == "1" ]]; then
    export SOCCER_NEURAL_BOTTLENECK_DIM="${SOCCER_NEURAL_BOTTLENECK_DIM:-100}"
    post_claude "overnight ${RUN_ID} BOTTLENECK_INPUT_PROFILE active: full ${SOCCER_NEURAL_FEATURE_DIM:-620}-feature critic input is preserved, with learned bottleneck_dim=${SOCCER_NEURAL_BOTTLENECK_DIM} before the value hidden layer."
  fi

  if [[ "${compact_active}" == "1" ]]; then
    export SOCCER_NEURAL_COMPACT_INPUTS="${SOCCER_NEURAL_COMPACT_INPUTS:-1}"
    post_claude "overnight ${RUN_ID} COMPACT_INPUT_PROFILE active: critic uses hard 100-d projection. Diagnostic only; Claude weight scan says naive pruning should lose signal."
  fi
}

fwd_trace_enabled() {
  [[ "${RUN_FWD_TRACE}" == "1" || "${RUN_FWD_TRACE}" == "true" ]]
}

run_eval_with_fwd_trace() {
  local label="$1"
  local candidate="$2"
  local baseline="$3"
  local games="$4"
  local minutes="$5"
  local seed="$6"
  local jsonl_path="${OUT_DIR}/${label}.jsonl"
  local aggregate_path="${OUT_DIR}/${label}.aggregate.txt"
  local fwd_trace_path="${OUT_DIR}/${label}.fwd_trace.jsonl"
  local fwd_trace_report_path="${OUT_DIR}/${label}.fwd_trace.txt"

  export PROOF_JSONL="${jsonl_path}"
  if fwd_trace_enabled; then
    run_step "eval ${label}" env DD_SOCCER_FWD_TRACE="${fwd_trace_path}" target/debug/soccer_proof eval "${candidate}" "${baseline}" "${games}" "${minutes}" "${seed}" || true
  else
    run_step "eval ${label}" target/debug/soccer_proof eval "${candidate}" "${baseline}" "${games}" "${minutes}" "${seed}" || true
  fi

  echo
  echo "===== aggregate ${label} JSONL ====="
  post_claude "overnight aggregate ${label} started for ${RUN_ID}; out=${OUT_DIR}"
  python3 scripts/proof_aggregate.py "${jsonl_path}" > "${aggregate_path}"
  local aggregate_status=$?
  post_claude "overnight aggregate ${label} finished for ${RUN_ID} with status=${aggregate_status}; out=${OUT_DIR}"

  if fwd_trace_enabled; then
    echo
    echo "===== forward trace ${label} ====="
    post_claude "overnight forward trace ${label} started for ${RUN_ID}; out=${OUT_DIR}"
    if [[ -s "${fwd_trace_path}" ]]; then
      python3 scripts/fwd_trace_report.py "${fwd_trace_path}" > "${fwd_trace_report_path}"
    else
      echo "no forward trace rows found" > "${fwd_trace_report_path}"
    fi
    local trace_status=$?
    post_claude "overnight forward trace ${label} finished for ${RUN_ID} with status=${trace_status}; out=${OUT_DIR}"
  fi
}

run_head_ablation() {
  local label="$1"
  local strip="$2"
  run_eval_with_fwd_trace "${label}" "${FINAL_SNAPSHOT}" "${FINAL_SNAPSHOT}#strip=${strip}" "${ABLATION_EVAL_GAMES}" "${EVAL_MINUTES}" "${HOLDOUT_SEED}"
}

run_dp_ablation() {
  local label="full-vs-nodp"
  local nodp_prefix="${OUT_DIR}/nodp"
  local nodp_snapshot="${nodp_prefix}.genFINAL.json"
  local jsonl_path="${OUT_DIR}/${label}.jsonl"
  local aggregate_path="${OUT_DIR}/${label}.aggregate.txt"

  run_step "train no-DP checkpoint" env \
    DD_SOCCER_ENABLE_DP_CRITIC_TARGET=0 \
    DD_SOCCER_ENABLE_DP_BOOTSTRAP=0 \
    DD_SOCCER_ENABLE_DEFERRED_PASS_CREDIT=0 \
    target/debug/soccer_proof train-ckpt "${nodp_prefix}" "${DP_ABLATION_TRAIN_GAMES}" "${TRAIN_MINUTES}" "${TRAIN_SEED}" "${CKPT_EVERY}" || return $?

  if [[ ! -f "${nodp_snapshot}" ]]; then
    post_claude "overnight ${RUN_ID} no-DP ablation stopped: missing snapshot ${nodp_snapshot}"
    return 2
  fi

  run_eval_with_fwd_trace "${label}" "${FINAL_SNAPSHOT}" "${nodp_snapshot}" "${DP_ABLATION_EVAL_GAMES}" "${EVAL_MINUTES}" "${HOLDOUT_SEED}"
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
    "analytic_eval_exists": (out / "full-vs-analytic.aggregate.txt").exists(),
    "analytic_fwd_trace_exists": (out / "full-vs-analytic.fwd_trace.txt").exists(),
    "dp_ablation_exists": (out / "full-vs-nodp.aggregate.txt").exists(),
    "fwd_trace_report_exists": (out / "full-vs-fresh.fwd_trace.txt").exists(),
    "head_ablation_count": len(list(out.glob("full-vs-no-*.aggregate.txt"))),
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
  echo "run_analytic_eval=${RUN_ANALYTIC_EVAL} analytic_eval_games=${ANALYTIC_EVAL_GAMES} analytic_eval_minutes=${ANALYTIC_EVAL_MINUTES}"
  echo "run_head_ablations=${RUN_HEAD_ABLATIONS} ablation_eval_games=${ABLATION_EVAL_GAMES}"
  echo "run_dp_ablation=${RUN_DP_ABLATION} dp_ablation_train_games=${DP_ABLATION_TRAIN_GAMES} dp_ablation_eval_games=${DP_ABLATION_EVAL_GAMES}"
  echo "run_fwd_trace=${RUN_FWD_TRACE}"
  echo "fast_learning_profile=${FAST_LEARNING_PROFILE} ceiling_break_profile=${CEILING_BREAK_PROFILE}"
  echo "bottleneck_input_profile=${BOTTLENECK_INPUT_PROFILE} compact_input_profile=${COMPACT_INPUT_PROFILE}"
  post_claude "overnight full-stack proof preflight started: ${RUN_ID}; out=${OUT_DIR}"
  guard_no_active_soccer_league_train
  apply_proof_profiles
  post_claude "overnight full-stack proof launched: ${RUN_ID}. Will train NN+DP+MPC/pass-completion/pass-receiver/support/spacing/carry/shooting/press/shape candidate, eval vs fresh gen0 and pure analytic, aggregate with passing, turnover, dribble-success, shooting-quality, recovery, teamwork guardrails, and forward-pass action-causality trace, and optionally run same-net head ablations plus a no-DP/deferred-credit training ablation from ${OUT_DIR}."

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
    run_step "prune local cargo artifacts" bash scripts/prune_local_cargo_artifacts.sh "${CARGO_TARGET_DIR:-target}" || exit $?
    run_step "build proof binaries" env CARGO_INCREMENTAL="${CARGO_INCREMENTAL:-0}" cargo build --bin soccer_proof --bin soccer_eval_gate_run --bin measure_learning_ab || exit $?
  fi
  run_step "fresh gen0 snapshot" target/debug/soccer_proof fresh "${OUT_DIR}/fresh.gen0.json" "${FRESH_SEED}" || exit $?
  run_step "train full-stack checkpoint" target/debug/soccer_proof train-ckpt "${OUT_DIR}/full" "${TRAIN_GAMES}" "${TRAIN_MINUTES}" "${TRAIN_SEED}" "${CKPT_EVERY}" || exit $?

  FINAL_SNAPSHOT="${OUT_DIR}/full.genFINAL.json"
  if [[ ! -f "${FINAL_SNAPSHOT}" ]]; then
    post_claude "overnight ${RUN_ID} stopped: missing final snapshot ${FINAL_SNAPSHOT}"
    exit 2
  fi

  run_eval_with_fwd_trace "full-vs-fresh" "${FINAL_SNAPSHOT}" "${OUT_DIR}/fresh.gen0.json" "${EVAL_GAMES}" "${EVAL_MINUTES}" "${HOLDOUT_SEED}"

  if [[ "${RUN_ANALYTIC_EVAL}" == "1" || "${RUN_ANALYTIC_EVAL}" == "true" ]]; then
    run_eval_with_fwd_trace "full-vs-analytic" "${FINAL_SNAPSHOT}" "analytic" "${ANALYTIC_EVAL_GAMES}" "${ANALYTIC_EVAL_MINUTES}" "${ANALYTIC_HOLDOUT_SEED}"
  fi

  if [[ "${RUN_HEAD_ABLATIONS}" == "1" || "${RUN_HEAD_ABLATIONS}" == "true" ]]; then
    run_head_ablation "full-vs-no-pass-head" "pass"
    run_head_ablation "full-vs-no-mpc-head" "mpc"
    run_head_ablation "full-vs-no-support-spacing-heads" "support,spacing"
    run_head_ablation "full-vs-no-execution-offball-heads" "mpc,support,spacing"
  fi

  if [[ "${RUN_DP_ABLATION}" == "1" || "${RUN_DP_ABLATION}" == "true" ]]; then
    run_dp_ablation
  fi

  SUMMARY="$(summarize_artifacts)"
  echo "summary=${SUMMARY}"
  post_claude "overnight full-stack proof completed for ${RUN_ID}. Summary: ${SUMMARY}. Artifacts: ${OUT_DIR}/full.genFINAL.json, ${OUT_DIR}/full-vs-fresh.jsonl, ${OUT_DIR}/full-vs-fresh.aggregate.txt"
} >> "${LOG_PATH}" 2>&1
