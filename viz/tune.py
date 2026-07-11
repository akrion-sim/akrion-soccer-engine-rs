#!/usr/bin/env python3
"""Reward-weight optimizer for the 11-a-side neural-authoritative ratchet.

This drives target/release/soccer_climb_ratchet as a subprocess with reward
weights supplied through environment variables. It searches the reward vector
with a small dependency-free (mu, lambda) evolution strategy and ranks
candidates by gated goal-difference: real match result first, with hard
penalties for obvious reward hacks such as shotless, no-build-up, or one-sided
policies.

Usage:
    cargo build --release --bin soccer_climb_ratchet
    ./viz/tune.py
    TUNE_GENS=8 TUNE_POP=6 TUNE_MAXPAR=2 ./viz/tune.py
    ./viz/tune.py --smoke
"""

import csv
import math
import os
import random
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BIN = os.path.join(ROOT, "target", "release", "soccer_climb_ratchet")
LOG = os.path.join(HERE, "tune_log.csv")

# (ENV_VAR, default, low, high, log_scale). Defaults mirror the current
# ratchet/hardening path rather than bare engine defaults.
SPACE = [
    ("DD_SOCCER_GOAL_REWARD_SCALE", 1.0, 0.5, 4.0, False),
    ("DD_SOCCER_FORWARD_PASS_REWARD_SCALE", 6.25, 0.0, 12.0, False),
    ("DD_SOCCER_PASS_CHAIN_REWARD_SCALE", 4.25, 0.0, 10.0, False),
    ("DD_SOCCER_PASS_TURNOVER_PENALTY_SCALE", 5.0, 1.0, 12.0, False),
    ("DD_SOCCER_QUICK_FORWARD_RELEASE_REWARD_SCALE", 1.25, 0.0, 2.0, False),
    ("DD_SOCCER_SHOT_SHAPING_REWARD_SCALE", 0.85, 0.0, 1.0, False),
    ("DD_SOCCER_SHOT_COMMITMENT_REWARD_SCALE", 0.0, 0.0, 2.0, False),
    ("DD_SOCCER_LEARNED_EPV_REWARD_SCALE", 1.0, 0.0, 4.0, False),
    ("SOCCER_OFFBALL_SUPPORT_REWARD_SCALE", 1.0, 0.5, 6.0, False),
    ("DD_SOCCER_CARRY_REWARD_SCALE", 1.0, 0.5, 6.0, False),
    ("DD_SOCCER_RECOVERY_REWARD_SCALE", 1.0, 0.5, 6.0, False),
    ("DD_SOCCER_OVERLOAD_PROGRESSION_REWARD_PER_YARD", 0.0, 0.0, 0.12, False),
    ("SOCCER_ANALYTIC_DIFFERENCE_REWARD_NEGATIVE_DENSE_FLOOR", 0.15, 0.0, 0.6, False),
    ("SOCCER_ANALYTIC_DIFFERENCE_REWARD_NEGATIVE_DENSE_SCALE", 1.0, 0.0, 3.0, False),
    ("SOCCER_MAPPO_TEAM_REWARD_SHARE", 0.50, 0.0, 1.0, False),
    ("SOCCER_MARL_TEAM_REWARD_WEIGHT", 0.35, 0.0, 1.5, False),
    ("SOCCER_MARL_INTERMEDIATE_REWARD_WEIGHT", 1.0, 0.2, 3.0, False),
    ("SOCCER_PLANNER_TEACHER_WEIGHT", 1.8, 0.0, 4.0, False),
    ("SOCCER_NEURAL_FINISHING_SELECT_BONUS_WEIGHT", 0.70, 0.0, 2.5, False),
]

ITERS_GAMES = int(os.environ.get("TUNE_INCREMENT_GAMES", "1"))
MINUTES = float(os.environ.get("TUNE_MINUTES", "0.5"))
EVAL_GAMES = int(os.environ.get("TUNE_EVAL_GAMES", "8"))
INCREMENTS = int(os.environ.get("TUNE_INCREMENTS", "1"))
POP = int(os.environ.get("TUNE_POP", "5"))
ELITE = int(os.environ.get("TUNE_ELITE", "2"))
GENS = int(os.environ.get("TUNE_GENS", "4"))
MAXPAR = int(os.environ.get("TUNE_MAXPAR", "1"))
SEEDS = [s.strip() for s in os.environ.get("TUNE_SEEDS", "0x5EED0000").split(",") if s.strip()]
SIGMA0 = float(os.environ.get("TUNE_SIGMA", "0.18"))
RNG = random.Random(int(os.environ.get("TUNE_RNG", "11")))

MIN_WEAK_SIDE = float(os.environ.get("TUNE_MIN_WEAK_SIDE", "0.25"))
MIN_SHOTS = int(os.environ.get("TUNE_MIN_SHOTS", "1"))
MIN_SOT = int(os.environ.get("TUNE_MIN_SOT", "1"))
MIN_SHOTS_AFTER_PASS = int(os.environ.get("TUNE_MIN_SHOTS_AFTER_PASS", "1"))
GATE_PENALTY = float(os.environ.get("TUNE_GATE_PENALTY", "6.0"))

FIXED_ENV = {
    "SOCCER_DYNAMIC_REWARD_WEIGHTS": "1",
    "SOCCER_ENGINE_UNIFORM_ELITE_PLAYERS": "0",
    "SOCCER_CLIMB_TRAIN_SIDE_MODES": "paired",
    "SOCCER_CLIMB_EVAL_PARALLELISM": "1",
    "SOCCER_CLIMB_LEARNING_RATE_CANDIDATES": os.environ.get(
        "SOCCER_CLIMB_LEARNING_RATE_CANDIDATES", "0.015"
    ),
    "SOCCER_CLIMB_AUTHORITATIVE_LAMBDA_CANDIDATES": os.environ.get(
        "SOCCER_CLIMB_AUTHORITATIVE_LAMBDA_CANDIDATES", "8.0"
    ),
    # Keep this process to one reward vector; the outer tuner supplies the vector.
    "SOCCER_CLIMB_REWARD_PROFILES": "",
    "DD_SOCCER_ENABLE_FIELD_VECTOR_SHOT_REWARD": "1",
}

CAND_RE = re.compile(
    r"eval_mean=([\d.]+).*?w/d/l=(\d+)/(\d+)/(\d+).*?"
    r"home=([\d.]+)\([^)]*\)\s+away=([\d.]+)\([^)]*\).*?"
    r"gd=([+-]?\d+).*?gf=(\d+)\s+ga=(\d+)\s+"
    r"shots=(\d+):(\d+)\s+sot=(\d+):(\d+)\s+sap=(\d+)\s+"
    r"fwd_margin=([+-]?\d+)\s+net_fwd_margin=([+-]?\d+)"
)


def env_bool(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def norm(vec):
    out = []
    for (_, _, lo, hi, log), value in zip(SPACE, vec):
        if log:
            value, lo, hi = math.log(value), math.log(lo), math.log(hi)
        out.append((value - lo) / (hi - lo))
    return out


def denorm(nvec):
    out = []
    for (_, _, lo, hi, log), value in zip(SPACE, nvec):
        value = min(1.0, max(0.0, value))
        if log:
            out.append(math.exp(math.log(lo) + value * (math.log(hi) - math.log(lo))))
        else:
            out.append(lo + value * (hi - lo))
    return out


def default_vec():
    return [default for (_, default, *_rest) in SPACE]


def vec_with(overrides):
    vec = default_vec()
    for index, (name, _, lo, hi, _) in enumerate(SPACE):
        if name in overrides:
            vec[index] = min(hi, max(lo, overrides[name]))
    return vec


def parse(stdout):
    best = None
    for line in stdout.splitlines():
        match = CAND_RE.search(line)
        if not match:
            continue
        (
            eval_mean,
            wins,
            draws,
            losses,
            home,
            away,
            gd,
            gf,
            ga,
            shots_for,
            shots_against,
            sot_for,
            sot_against,
            sap,
            fwd_margin,
            net_fwd_margin,
        ) = match.groups()
        metrics = {
            "eval_mean": float(eval_mean),
            "wins": int(wins),
            "draws": int(draws),
            "losses": int(losses),
            "home": float(home),
            "away": float(away),
            "gd": int(gd),
            "gf": int(gf),
            "ga": int(ga),
            "shots_for": int(shots_for),
            "shots_against": int(shots_against),
            "sot_for": int(sot_for),
            "sot_against": int(sot_against),
            "shots_after_pass": int(sap),
            "fwd_margin": int(fwd_margin),
            "net_fwd_margin": int(net_fwd_margin),
        }
        if best is None or fitness_from_metrics(metrics) > fitness_from_metrics(best):
            best = metrics
    return best


def gates_ok(metrics):
    if metrics is None:
        return False
    weak_side = min(metrics["home"], metrics["away"])
    return (
        weak_side >= MIN_WEAK_SIDE
        and metrics["shots_for"] >= MIN_SHOTS
        and metrics["sot_for"] >= MIN_SOT
        and metrics["shots_after_pass"] >= MIN_SHOTS_AFTER_PASS
    )


def fitness_from_metrics(metrics):
    if metrics is None:
        return -100.0
    score = (
        metrics["gd"]
        + (metrics["eval_mean"] - 0.5) * 4.0
        + (metrics["wins"] - metrics["losses"]) * 0.35
        + (metrics["sot_for"] - metrics["sot_against"]) * 0.08
        + metrics["shots_after_pass"] * 0.10
        + metrics["net_fwd_margin"] * 0.03
    )
    if not gates_ok(metrics):
        score -= GATE_PENALTY
    return score


def aggregate_fitness(metrics_by_seed):
    scores = [fitness_from_metrics(metrics) for metrics in metrics_by_seed]
    mean = sum(scores) / len(scores)
    spread = max(scores) - min(scores) if len(scores) > 1 else 0.0
    return mean - 0.5 * spread


def run_one(vec, seed, cand_index, seed_index):
    env = dict(os.environ)
    env.update(FIXED_ENV)
    for (name, *_), value in zip(SPACE, vec):
        env[name] = f"{value:.8g}"
    outdir = os.path.join(ROOT, "out_tune_11v11", f"cand_{cand_index}_seed_{seed_index}")
    cmd = [
        BIN,
        str(ITERS_GAMES),
        str(MINUTES),
        str(EVAL_GAMES),
        str(INCREMENTS),
        seed,
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            env=env,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        out, _ = proc.communicate(timeout=int(os.environ.get("TUNE_TIMEOUT_SECONDS", "3600")))
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        return None
    # The ratchet does not currently write a checkpoint for the tuner, but the
    # outdir path is reserved in the row identity for future artifact capture.
    _ = outdir
    return parse(out)


def eval_pool(vectors):
    jobs = [(ci, si, vec, seed) for ci, vec in enumerate(vectors) for si, seed in enumerate(SEEDS)]
    running = {}
    results = {}
    next_job = 0
    while next_job < len(jobs) or running:
        while next_job < len(jobs) and len(running) < MAXPAR:
            ci, si, vec, seed = jobs[next_job]
            next_job += 1
            env = dict(os.environ)
            env.update(FIXED_ENV)
            for (name, *_), value in zip(SPACE, vec):
                env[name] = f"{value:.8g}"
            cmd = [
                BIN,
                str(ITERS_GAMES),
                str(MINUTES),
                str(EVAL_GAMES),
                str(INCREMENTS),
                seed,
            ]
            proc = subprocess.Popen(
                cmd,
                env=env,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            running[proc] = (ci, si)
        time.sleep(1.0)
        for proc in list(running):
            if proc.poll() is None:
                continue
            ci, si = running.pop(proc)
            out, _ = proc.communicate()
            results[(ci, si)] = parse(out)
    fits = []
    for ci in range(len(vectors)):
        seed_metrics = [results.get((ci, si)) for si in range(len(SEEDS))]
        fits.append((aggregate_fitness(seed_metrics), seed_metrics))
    return fits


def smoke_bad_mutation():
    return vec_with(
        {
            "DD_SOCCER_GOAL_REWARD_SCALE": 0.5,
            "DD_SOCCER_FORWARD_PASS_REWARD_SCALE": 12.0,
            "DD_SOCCER_PASS_CHAIN_REWARD_SCALE": 10.0,
            "DD_SOCCER_PASS_TURNOVER_PENALTY_SCALE": 1.0,
            "DD_SOCCER_QUICK_FORWARD_RELEASE_REWARD_SCALE": 0.0,
            "DD_SOCCER_SHOT_SHAPING_REWARD_SCALE": 0.0,
            "DD_SOCCER_SHOT_COMMITMENT_REWARD_SCALE": 0.0,
            "DD_SOCCER_LEARNED_EPV_REWARD_SCALE": 0.0,
            "SOCCER_MARL_INTERMEDIATE_REWARD_WEIGHT": 3.0,
            "SOCCER_NEURAL_FINISHING_SELECT_BONUS_WEIGHT": 0.0,
        }
    )


def smoke_test():
    vectors = [default_vec(), smoke_bad_mutation()]
    names = ["default", "bad_mutation"]
    print(
        f"smoke 11v11 tuner | weights={len(SPACE)} seeds={SEEDS} "
        f"games={ITERS_GAMES}x{INCREMENTS} eval={EVAL_GAMES} minutes={MINUTES}"
    )
    fits = eval_pool(vectors)
    for name, (fit, metrics) in zip(names, fits):
        print(f"smoke {name}: fitness={fit:+.3f} metrics={metrics}")
    if not fits[0][0] > fits[1][0]:
        sys.exit("smoke failed: default vector did not outrank the bad mutation")
    print("smoke ok: default vector outranks the bad mutation")


def write_header():
    with open(LOG, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["gen", "cand", "fitness"]
            + [name for (name, *_rest) in SPACE]
            + [
                "eval_mean",
                "wins",
                "draws",
                "losses",
                "home",
                "away",
                "gd",
                "gf",
                "ga",
                "shots_for",
                "shots_against",
                "sot_for",
                "sot_against",
                "shots_after_pass",
                "gates_ok",
            ]
        )


def append_rows(gen, vectors, fits):
    with open(LOG, "a", newline="") as file:
        writer = csv.writer(file)
        for cand, (vec, (fit, seed_metrics)) in enumerate(zip(vectors, fits)):
            metrics = seed_metrics[0] or {}
            writer.writerow(
                [gen, cand, f"{fit:.6f}"]
                + [f"{value:.8g}" for value in vec]
                + [
                    metrics.get("eval_mean"),
                    metrics.get("wins"),
                    metrics.get("draws"),
                    metrics.get("losses"),
                    metrics.get("home"),
                    metrics.get("away"),
                    metrics.get("gd"),
                    metrics.get("gf"),
                    metrics.get("ga"),
                    metrics.get("shots_for"),
                    metrics.get("shots_against"),
                    metrics.get("sot_for"),
                    metrics.get("sot_against"),
                    metrics.get("shots_after_pass"),
                    gates_ok(seed_metrics[0]),
                ]
            )


def main():
    if not os.path.exists(BIN):
        sys.exit(f"build first: cargo build --release --bin soccer_climb_ratchet (missing {BIN})")
    if "--smoke" in sys.argv[1:] or env_bool("TUNE_SMOKE"):
        smoke_test()
        return
    print(
        f"tuning {len(SPACE)} 11v11 weights | gens={GENS} pop={POP} elite={ELITE} "
        f"seeds={SEEDS} maxpar={MAXPAR}"
    )
    mean = norm(default_vec())
    sigma = [SIGMA0] * len(SPACE)
    best = None
    write_header()
    for gen in range(GENS):
        candidates_n = [list(mean)]
        for _ in range(POP - 1):
            candidates_n.append(
                [
                    min(1.0, max(0.0, mean[i] + sigma[i] * RNG.gauss(0.0, 1.0)))
                    for i in range(len(SPACE))
                ]
            )
        candidates = [denorm(candidate) for candidate in candidates_n]
        fits = eval_pool(candidates)
        ranked = sorted(range(len(candidates)), key=lambda index: fits[index][0], reverse=True)
        append_rows(gen, candidates, fits)
        elite = ranked[:ELITE]
        mean = [sum(candidates_n[index][dim] for index in elite) / ELITE for dim in range(len(SPACE))]
        sigma = [
            max(
                0.03,
                min(
                    0.40,
                    math.sqrt(sum((candidates_n[index][dim] - mean[dim]) ** 2 for index in elite) / ELITE)
                    * 1.3,
                ),
            )
            for dim in range(len(SPACE))
        ]
        best_index = ranked[0]
        if best is None or fits[best_index][0] > best[0]:
            best = (fits[best_index][0], candidates[best_index], fits[best_index][1])
        print(
            f"gen {gen}: best_fit={fits[best_index][0]:+.3f} "
            f"metrics={fits[best_index][1][0]} weights="
            + ", ".join(f"{name}={value:.4g}" for (name, *_), value in zip(SPACE, candidates[best_index]))
        )
        sys.stdout.flush()
    print("\n=== BEST ===")
    print(f"fitness={best[0]:+.3f} metrics={best[2]}")
    for (name, *_), value in zip(SPACE, best[1]):
        print(f"  {name}={value:.8g}")


if __name__ == "__main__":
    main()
