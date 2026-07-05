#!/usr/bin/env bash
# Detached local A/B for Bellman trajectory terminal replay.
#
# This script intentionally starts the release build and all long-running loops in
# background children, records their PIDs, then exits quickly. Trainer wrappers wait
# for the build sentinel before execing the release binaries.

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

STAMP="${BELL_AB_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${BELL_AB_RUN_DIR:-$ROOT/tmp/bellman-terminal-ab-$STAMP}"
VALIDATE_SECONDS="${BELL_AB_VALIDATE_SECONDS:-1200}"
EVAL_GAMES_PER_OPP="${BELL_AB_EVAL_GAMES_PER_OPP:-10}"
EVAL_MINUTES="${BELL_AB_EVAL_MINUTES:-2}"
EVAL_POOL_SIZE="${BELL_AB_EVAL_POOL_SIZE:-5}"

SHARED_DIR="$RUN_DIR/shared"
LOG_DIR="$RUN_DIR/logs"
PID_DIR="$RUN_DIR/pids"
BUILD_OK="$RUN_DIR/build.ok"
BUILD_FAILED="$RUN_DIR/build.failed"

mkdir -p "$SHARED_DIR" "$LOG_DIR" "$PID_DIR" \
    "$RUN_DIR/dp-off/champions" "$RUN_DIR/dp-on/champions"

SETUP_LOG="$LOG_DIR/setup.log"
BUILD_LOG="$LOG_DIR/build.log"
VALIDATION_LOG="$LOG_DIR/validation.log"
DP_OFF_LOG="$LOG_DIR/dp-off-trainer.log"
DP_ON_LOG="$LOG_DIR/dp-on-trainer.log"

log_setup() {
    printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$SETUP_LOG"
}

copy_frontier() {
    src="$1"
    dst="$2"
    cp "$src" "$dst"
    sidecar="${src%.json}.neural.json"
    if [ -f "$sidecar" ]; then
        cp "$sidecar" "${dst%.json}.neural.json"
    fi
}

find_frontier() {
    if [ -n "${BELL_AB_SOURCE_FRONTIER:-}" ]; then
        printf '%s\n' "$BELL_AB_SOURCE_FRONTIER"
        return 0
    fi
    find "$ROOT/out" -name learned-params.json -type f -size +0c -print0 2>/dev/null \
        | xargs -0 ls -t 2>/dev/null \
        | head -1
}

SOURCE_FRONTIER="$(find_frontier)"
if [ -z "$SOURCE_FRONTIER" ] || [ ! -s "$SOURCE_FRONTIER" ]; then
    log_setup "ERROR no non-empty learned-params.json frontier found; set BELL_AB_SOURCE_FRONTIER"
    exit 1
fi
if ! grep -q '"neuralNetwork"' "$SOURCE_FRONTIER"; then
    log_setup "ERROR source frontier lacks neuralNetwork: $SOURCE_FRONTIER"
    exit 1
fi

copy_frontier "$SOURCE_FRONTIER" "$SHARED_DIR/learned-params.json"
copy_frontier "$SHARED_DIR/learned-params.json" "$RUN_DIR/dp-off/learned-params.json"
copy_frontier "$SHARED_DIR/learned-params.json" "$RUN_DIR/dp-on/learned-params.json"
printf '%s\n' "$SOURCE_FRONTIER" > "$RUN_DIR/source-frontier.txt"

log_setup "run_dir=$RUN_DIR"
log_setup "source_frontier=$SOURCE_FRONTIER"
log_setup "shared_frontier=$SHARED_DIR/learned-params.json"

(
    printf '[%s] release build start\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    cargo build --release --bin soccer_league_train --bin soccer_eval_gate_run
    status="$?"
    if [ "$status" -eq 0 ]; then
        printf '[%s] release build ok\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$BUILD_OK"
    else
        printf '[%s] release build failed status=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" > "$BUILD_FAILED"
    fi
    exit "$status"
) > "$BUILD_LOG" 2>&1 &
BUILD_PID="$!"
printf '%s\n' "$BUILD_PID" > "$PID_DIR/build.pid"
log_setup "build_pid=$BUILD_PID"

wait_for_build() {
    while [ ! -f "$BUILD_OK" ]; do
        if [ -f "$BUILD_FAILED" ]; then
            printf '[%s] build failed; see %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$BUILD_LOG"
            exit 1
        fi
        sleep 5
    done
}

read_training_steps() {
    path="$1"
    if command -v node >/dev/null 2>&1; then
        node -e '
const fs = require("fs");
const path = process.argv[1];
const value = JSON.parse(fs.readFileSync(path, "utf8"));
const net = value.neuralNetwork || (value.learning && value.learning.neuralNetwork) || {};
const steps = net.trainingSteps ?? net.training_steps ?? value.trainingSteps ?? value.training_steps ?? 0;
console.log(steps);
' "$path" 2>/dev/null && return 0
    fi
    awk 'match($0,/"trainingSteps":[0-9]+/){print substr($0,RSTART+16,RLENGTH-16); exit}' "$path"
}

run_trainer() {
    arm="$1"
    terminals="$2"
    passes="$3"
    frontier="$RUN_DIR/$arm/learned-params.json"
    archive="$RUN_DIR/$arm/champions"

    wait_for_build
    export SOCCER_LEAGUE_FRONTIER="$frontier"
    export SOCCER_LEAGUE_ARCHIVE="$archive"
    export SOCCER_APPROX_DP_REPLAY_PASSES="$passes"
    export SOCCER_LEAGUE_PARALLELISM="${SOCCER_LEAGUE_PARALLELISM:-1}"
    if [ "$terminals" = "off" ]; then
        export DD_SOCCER_ENABLE_BELLMAN_TERMINALS=0
    else
        unset DD_SOCCER_ENABLE_BELLMAN_TERMINALS
    fi
    printf '[%s] trainer_start arm=%s frontier=%s archive=%s passes=%s terminals=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$arm" "$frontier" "$archive" "$passes" "$terminals"
    exec "$ROOT/target/release/soccer_league_train"
}

run_validation_loop() {
    wait_for_build
    cycle=0
    while true; do
        cycle=$((cycle + 1))
        for arm in dp-off dp-on; do
            frontier="$RUN_DIR/$arm/learned-params.json"
            steps="$(read_training_steps "$frontier")"
            started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            printf '\n[%s] validation_start cycle=%s arm=%s frontier=%s training_steps=%s\n' \
                "$started" "$cycle" "$arm" "$frontier" "$steps" >> "$VALIDATION_LOG"
            output="$(
                SOCCER_EVAL_ANALYTIC_FIELD=1 \
                SOCCER_EVAL_CANDIDATE_PATH="$frontier" \
                SOCCER_EVAL_PARALLELISM="${SOCCER_EVAL_PARALLELISM:-2}" \
                "$ROOT/target/release/soccer_eval_gate_run" \
                    "$EVAL_GAMES_PER_OPP" "$EVAL_MINUTES" "$EVAL_POOL_SIZE" 0 2>&1
            )"
            status="$?"
            printf '%s\n' "$output" >> "$VALIDATION_LOG"
            mean_payoff="$(
                printf '%s\n' "$output" \
                    | grep 'cross-play: mean payoff vs field' \
                    | tail -1 \
                    | awk -F 'mean payoff vs field ' '{print $2}' \
                    | awk -F '  vs baseline' '{print $1}'
            )"
            wilson="$(
                printf '%s\n' "$output" \
                    | grep 'Wilson lower bound:' \
                    | tail -1 \
                    | awk '{print $4}'
            )"
            printf '[%s] validation_summary cycle=%s arm=%s status=%s games=%s mean_payoff=%s wilson_lower=%s training_steps=%s\n' \
                "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$cycle" "$arm" "$status" \
                "$((EVAL_GAMES_PER_OPP * (EVAL_POOL_SIZE - 1)))" "${mean_payoff:-unknown}" \
                "${wilson:-unknown}" "$steps" >> "$VALIDATION_LOG"
        done
        sleep "$VALIDATE_SECONDS"
    done
}

(
    run_trainer dp-off off 1
) >> "$DP_OFF_LOG" 2>&1 &
DP_OFF_PID="$!"
printf '%s\n' "$DP_OFF_PID" > "$PID_DIR/dp-off.pid"

(
    run_trainer dp-on on 3
) >> "$DP_ON_LOG" 2>&1 &
DP_ON_PID="$!"
printf '%s\n' "$DP_ON_PID" > "$PID_DIR/dp-on.pid"

(
    run_validation_loop
) >> "$LOG_DIR/validation-wrapper.log" 2>&1 &
VALIDATION_PID="$!"
printf '%s\n' "$VALIDATION_PID" > "$PID_DIR/validation.pid"

cat > "$RUN_DIR/summary.txt" <<SUMMARY
run_dir=$RUN_DIR
source_frontier=$SOURCE_FRONTIER
shared_frontier=$SHARED_DIR/learned-params.json
dp_off_frontier=$RUN_DIR/dp-off/learned-params.json
dp_on_frontier=$RUN_DIR/dp-on/learned-params.json
setup_log=$SETUP_LOG
build_log=$BUILD_LOG
dp_off_log=$DP_OFF_LOG
dp_on_log=$DP_ON_LOG
validation_log=$VALIDATION_LOG
build_pid=$BUILD_PID
dp_off_pid=$DP_OFF_PID
dp_on_pid=$DP_ON_PID
validation_pid=$VALIDATION_PID
eval_games_per_arm=$((EVAL_GAMES_PER_OPP * (EVAL_POOL_SIZE - 1)))
validation_interval_seconds=$VALIDATE_SECONDS
SUMMARY

log_setup "dp_off_pid=$DP_OFF_PID"
log_setup "dp_on_pid=$DP_ON_PID"
log_setup "validation_pid=$VALIDATION_PID"
log_setup "setup_complete"
printf '%s\n' "$RUN_DIR"

wait "$BUILD_PID"
BUILD_WAIT_STATUS="$?"
log_setup "build_wait_status=$BUILD_WAIT_STATUS"
wait "$DP_OFF_PID" "$DP_ON_PID" "$VALIDATION_PID"
