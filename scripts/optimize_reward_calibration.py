#!/usr/bin/env python3
"""Held-out outer-loop optimizer for typed soccer reward utilities.

The inner learner never edits its own objective. Each candidate calibration is
frozen for an entire analytic-opponent training window, then scored on disjoint
analytic fixtures. A cross-entropy-method (CEM) outer loop updates log-scales and
persists every proposal/result so an interrupted search can be audited or resumed.

This deliberately tunes utility weights, not event facts or attribution. The
Rust seam clamps every multiplier to [0, 4] and preserves the event's sign.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
from typing import Any


KINDS = (
    "CompletedForwardPass",
    "ShotOnTarget",
    "Goal",
    "BadPassChainPenalty",
    "TurnoverChainBlame",
    "ShotOffTargetPenalty",
)

PAYOFF_RE = re.compile(r"mean payoff vs field Some\(([-+0-9.eE]+)\)")
WILSON_RE = re.compile(r"Wilson lower bound:\s*([-+0-9.eE]+)")
RECORD_RE = re.compile(r"record:\s*(\d+)W-(\d+)D-(\d+)L\s+GF/GA\s+(\d+)/(\d+)\s+\(GD\s+([-+0-9]+)\)")
TURNOVER_RE = re.compile(r"pass_turnover_rate\s+([-+0-9.]+)%/([-+0-9.]+)%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bin-dir", type=Path, default=Path("/tmp/gate-ab-target/release"))
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--population", type=int, default=6)
    parser.add_argument("--elite", type=int, default=2)
    parser.add_argument("--train-games", type=int, default=30)
    parser.add_argument("--eval-games-per-opp", type=int, default=12)
    parser.add_argument("--minutes", type=float, default=0.5)
    parser.add_argument("--pool-size", type=int, default=4)
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0xC311B)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def calibration_env(scales: dict[str, float]) -> str:
    return ",".join(f"{kind}={scales[kind]:.6f}" for kind in KINDS)


def bounded_scale(log_scale: float) -> float:
    return min(4.0, max(0.0001, math.exp(log_scale)))


def parse_verdict(text: str) -> dict[str, Any]:
    payoff_match = PAYOFF_RE.search(text)
    wilson_match = WILSON_RE.search(text)
    record_match = RECORD_RE.search(text)
    if not payoff_match or not wilson_match or not record_match:
        raise ValueError("promotion output is missing payoff, Wilson, or record")
    wins, draws, losses, gf, ga, gd = (int(value) for value in record_match.groups())
    turnover_match = TURNOVER_RE.search(text)
    candidate_turnover = float(turnover_match.group(1)) if turnover_match else None
    analytic_turnover = float(turnover_match.group(2)) if turnover_match else None
    payoff = float(payoff_match.group(1))
    wilson = float(wilson_match.group(1))
    # Outcome confidence is authoritative. Goal difference breaks near-ties; an
    # avoidable turnover regression is a small explicit tax, never the objective.
    score = wilson + 0.02 * math.tanh(gd / 8.0)
    if candidate_turnover is not None and analytic_turnover is not None:
        score -= 0.001 * max(0.0, candidate_turnover - analytic_turnover)
    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "goals_for": gf,
        "goals_against": ga,
        "goal_difference": gd,
        "payoff": payoff,
        "wilson_lower": wilson,
        "candidate_turnover_pct": candidate_turnover,
        "analytic_turnover_pct": analytic_turnover,
        "objective": score,
    }


def run_checked(
    command: list[str],
    env: dict[str, str],
    log_path: Path,
    accepted_returncodes: tuple[int, ...] = (0,),
) -> str:
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    text = log_path.read_text(encoding="utf-8")
    if process.returncode not in accepted_returncodes:
        raise RuntimeError(f"command failed ({process.returncode}); see {log_path}\n{text[-2000:]}")
    return text


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_or_initialize(args: argparse.Namespace) -> dict[str, Any]:
    state_path = args.out / "reward-calibration-search.json"
    if args.resume and state_path.exists():
        return json.loads(state_path.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "created_at_unix": time.time(),
        "seed": args.seed,
        "kinds": list(KINDS),
        "mean_log_scales": {kind: 0.0 for kind in KINDS},
        "std_log_scales": {kind: 0.45 for kind in KINDS},
        "history": [],
        "best": None,
    }


def evaluate_candidate(
    args: argparse.Namespace,
    generation: int,
    index: int,
    scales: dict[str, float],
) -> dict[str, Any]:
    candidate_dir = args.out / f"g{generation:02d}-c{index:02d}"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    snapshot = candidate_dir / "candidate.json"
    train_log = candidate_dir / "train.log"
    eval_log = candidate_dir / "eval.log"
    train_seed = 0x71000000 + generation * 0x10000 + index * 0x100
    eval_seed = 0xE7000000 + generation * 0x10000 + index * 0x100
    env = os.environ.copy()
    env.update(
        {
            "DD_SOCCER_ENABLE_HERMETIC_NEURAL_CRITIC": "1",
            "DD_SOCCER_ENABLE_MAXA_BOOTSTRAP": "1",
            "DD_SOCCER_REWARD_KIND_SCALES": calibration_env(scales),
            "SOCCER_TRAIN_ANALYTIC_POOL_SIZE": str(max(1, args.pool_size - 1)),
            "SOCCER_TRAIN_ANALYTIC_ALTERNATE_SIDES": "1",
        }
    )
    train_command = [
        str(args.bin_dir / "soccer_outcome_ab_run"),
        "train-analytic",
        str(snapshot),
        str(args.train_games),
        str(args.minutes),
        f"{train_seed:08X}",
    ]
    eval_env = os.environ.copy()
    eval_env.update(
        {
            "SOCCER_EVAL_CANDIDATE_PATH": str(snapshot),
            "SOCCER_EVAL_ANALYTIC_FIELD": "1",
            "SOCCER_EVAL_PARALLELISM": str(args.parallelism),
            "SOCCER_EVAL_HOLDOUT_SEED": str(eval_seed),
        }
    )
    eval_command = [
        str(args.bin_dir / "soccer_eval_gate_run"),
        str(args.eval_games_per_opp),
        str(args.minutes),
        str(args.pool_size),
        "0",
    ]
    if args.dry_run:
        return {
            "generation": generation,
            "candidate": index,
            "scales": scales,
            "calibration_env": calibration_env(scales),
            "train_command": train_command,
            "eval_command": eval_command,
            "objective": float("-inf"),
        }
    run_checked(train_command, env, train_log)
    # The promotion binary returns 1 for a valid REJECT verdict. That is an
    # optimizer observation, not an infrastructure failure.
    verdict = parse_verdict(run_checked(eval_command, eval_env, eval_log, (0, 1)))
    return {
        "generation": generation,
        "candidate": index,
        "scales": scales,
        "calibration_env": calibration_env(scales),
        "train_seed": train_seed,
        "eval_seed": eval_seed,
        "snapshot": str(snapshot),
        **verdict,
    }


def main() -> int:
    args = parse_args()
    if args.population < 2 or not 1 <= args.elite < args.population:
        raise SystemExit("require population >= 2 and 1 <= elite < population")
    args.out.mkdir(parents=True, exist_ok=True)
    state_path = args.out / "reward-calibration-search.json"
    state = load_or_initialize(args)
    rng = random.Random(args.seed)
    start_generation = 0
    if state["history"]:
        start_generation = max(item["generation"] for item in state["history"]) + 1

    for generation in range(start_generation, args.generations):
        proposals: list[tuple[dict[str, float], dict[str, float]]] = []
        for index in range(args.population):
            logs = {
                kind: rng.gauss(state["mean_log_scales"][kind], state["std_log_scales"][kind])
                for kind in KINDS
            }
            proposals.append((logs, {kind: bounded_scale(logs[kind]) for kind in KINDS}))
        generation_results = []
        for index, (logs, scales) in enumerate(proposals):
            result = evaluate_candidate(args, generation, index, scales)
            result["log_scales"] = logs
            state["history"].append(result)
            generation_results.append(result)
            if state["best"] is None or result["objective"] > state["best"]["objective"]:
                state["best"] = result
            atomic_write_json(state_path, state)
            print(json.dumps(result, sort_keys=True), flush=True)
        if args.dry_run:
            break
        elites = sorted(generation_results, key=lambda item: item["objective"], reverse=True)[: args.elite]
        for kind in KINDS:
            values = [item["log_scales"][kind] for item in elites]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            state["mean_log_scales"][kind] = mean
            state["std_log_scales"][kind] = max(0.08, math.sqrt(variance))
        atomic_write_json(state_path, state)

    best_path = args.out / "reward-calibration-best.json"
    atomic_write_json(
        best_path,
        {
            "schema_version": 1,
            "selection": "held-out-analytic-cem",
            "best": state["best"],
            "required_runtime_env": {
                "DD_SOCCER_ENABLE_HERMETIC_NEURAL_CRITIC": "1",
                "DD_SOCCER_ENABLE_MAXA_BOOTSTRAP": "1",
                "DD_SOCCER_REWARD_KIND_SCALES": state["best"]["calibration_env"] if state["best"] else "",
            },
        },
    )
    print(f"state={state_path} best={best_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
