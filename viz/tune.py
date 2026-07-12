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
    TUNE_POP=10 TUNE_IMMIGRANTS=2 TUNE_PRESET_CANDIDATES=6 ./viz/tune.py
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
try:
    MIN_REWARD_WEIGHT = float(os.environ.get("TUNE_MIN_REWARD_WEIGHT", "0.0001"))
except ValueError:
    MIN_REWARD_WEIGHT = 0.0001
if not math.isfinite(MIN_REWARD_WEIGHT) or MIN_REWARD_WEIGHT <= 0.0:
    MIN_REWARD_WEIGHT = 0.0001


def env_int(name, default, lo=None, hi=None):
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    if lo is not None:
        value = max(lo, value)
    if hi is not None:
        value = min(hi, value)
    return value


def env_float(name, default, lo=None, hi=None):
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    if not math.isfinite(value):
        value = default
    if lo is not None:
        value = max(lo, value)
    if hi is not None:
        value = min(hi, value)
    return value


# (ENV_VAR, default, low, high, log_scale). Defaults are the current 11-a-side
# tuning anchor, not the bare engine defaults. The anchor is intentionally
# close to the best short-run reward profile evidence seen so far, so the
# evolution strategy starts near a plausible football policy instead of a
# known-weak legacy vector. Lower bounds are deliberately positive: a channel
# may become tiny, but never exactly zero and invisible to the learner.
SPACE = [
    # Merge note: the default (anchor) column follows the dynamic-rewards
    # re-anchoring; the low/high bounds and log-scale flags follow the hardening
    # branch (wider search with robust positive floors). The previous anchor is
    # preserved as legacy_default_vec() / the "legacy_default" preset.
    ("DD_SOCCER_GOAL_REWARD_SCALE", 1.2, 1.0, 16.0, False),
    ("DD_SOCCER_FORWARD_PASS_REWARD_SCALE", 8.2, MIN_REWARD_WEIGHT, 20.0, False),
    ("DD_SOCCER_PASS_CHAIN_REWARD_SCALE", 6.6, MIN_REWARD_WEIGHT, 20.0, False),
    ("DD_SOCCER_PASS_TURNOVER_PENALTY_SCALE", 2.8, 1.0, 20.0, False),
    ("DD_SOCCER_QUICK_FORWARD_RELEASE_REWARD_SCALE", 0.75, MIN_REWARD_WEIGHT, 2.0, False),
    ("DD_SOCCER_SHOT_SHAPING_REWARD_SCALE", 0.42, MIN_REWARD_WEIGHT, 4.0, False),
    ("DD_SOCCER_SHOT_COMMITMENT_REWARD_SCALE", 0.18, MIN_REWARD_WEIGHT, 10.0, False),
    ("DD_SOCCER_DRIBBLE_BEAT_REWARD_SCALE", 0.9, MIN_REWARD_WEIGHT, 8.0, False),
    ("DD_SOCCER_LEARNED_EPV_REWARD_SCALE", 0.30, MIN_REWARD_WEIGHT, 200.0, True),
    ("SOCCER_OFFBALL_SUPPORT_REWARD_SCALE", 3.0, 0.5, 6.0, False),
    ("DD_SOCCER_CARRY_REWARD_SCALE", 1.1, 0.5, 6.0, False),
    ("DD_SOCCER_RECOVERY_REWARD_SCALE", 1.0, 0.5, 6.0, False),
    ("DD_SOCCER_OVERLOAD_PROGRESSION_REWARD_PER_YARD", 0.05, MIN_REWARD_WEIGHT, 0.5, False),
    ("SOCCER_ANALYTIC_DIFFERENCE_REWARD_NEGATIVE_DENSE_FLOOR", 0.10, MIN_REWARD_WEIGHT, 1.0, False),
    ("SOCCER_ANALYTIC_DIFFERENCE_REWARD_NEGATIVE_DENSE_SCALE", 1.0, MIN_REWARD_WEIGHT, 25.0, True),
    ("SOCCER_MAPPO_TEAM_REWARD_SHARE", 0.08, MIN_REWARD_WEIGHT, 1.0, False),
    ("SOCCER_MARL_TEAM_REWARD_WEIGHT", 0.08, MIN_REWARD_WEIGHT, 1.5, False),
    ("SOCCER_MARL_INTERMEDIATE_REWARD_WEIGHT", 2.0, 0.2, 3.0, False),
    ("SOCCER_PLANNER_TEACHER_WEIGHT", 0.55, MIN_REWARD_WEIGHT, 4.0, False),
    ("SOCCER_NEURAL_FINISHING_SELECT_BONUS_WEIGHT", 0.45, MIN_REWARD_WEIGHT, 2.5, False),
]

ITERS_GAMES = env_int("TUNE_INCREMENT_GAMES", 1, lo=1)
MINUTES = env_float("TUNE_MINUTES", 0.5, lo=0.01)
EVAL_GAMES = env_int("TUNE_EVAL_GAMES", 8, lo=1)
INCREMENTS = env_int("TUNE_INCREMENTS", 1, lo=1)
POP = env_int("TUNE_POP", 5, lo=1)
ELITE = env_int("TUNE_ELITE", 2, lo=1)
GENS = env_int("TUNE_GENS", 4, lo=1)
MAXPAR = env_int("TUNE_MAXPAR", 1, lo=1)
SEEDS = [s.strip() for s in os.environ.get("TUNE_SEEDS", "0x5EED0000").split(",") if s.strip()]
SIGMA0 = env_float("TUNE_SIGMA", 0.18, lo=0.001, hi=1.0)
RNG = random.Random(env_int("TUNE_RNG", 11))
HALTON_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]

# Fail fast on misconfiguration instead of dividing by zero deep in the sweep
# (aggregate_fitness divides by len(SEEDS); the elite update divides by ELITE).
if not SEEDS:
    sys.exit("TUNE_SEEDS is empty — provide at least one seed")
if POP < 1:
    sys.exit(f"TUNE_POP={POP} must be >= 1")
if not (1 <= ELITE <= POP):
    sys.exit(f"TUNE_ELITE={ELITE} must be in [1, TUNE_POP={POP}]")

MIN_WEAK_SIDE = env_float("TUNE_MIN_WEAK_SIDE", 0.25, lo=0.0, hi=1.0)
MIN_SHOTS = env_int("TUNE_MIN_SHOTS", 1, lo=0)
MIN_SOT = env_int("TUNE_MIN_SOT", 1, lo=0)
MIN_SHOTS_AFTER_PASS = env_int("TUNE_MIN_SHOTS_AFTER_PASS", 1, lo=0)
GATE_PENALTY = env_float("TUNE_GATE_PENALTY", 6.0, lo=0.0)

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
        lo = max(MIN_REWARD_WEIGHT, lo)
        hi = max(lo, hi)
        value = float(value)
        if not math.isfinite(value):
            value = lo
        value = min(hi, max(lo, value))
        if log:
            value, lo, hi = math.log(value), math.log(lo), math.log(hi)
        out.append((value - lo) / (hi - lo))
    return out


def denorm(nvec):
    out = []
    for (_, _, lo, hi, log), value in zip(SPACE, nvec):
<<<<<<< HEAD
        value = float(value)
        if not math.isfinite(value):
            value = 0.0
=======
        lo = max(MIN_REWARD_WEIGHT, lo)
        hi = max(lo, hi)
        value = value if math.isfinite(value) else 0.0
>>>>>>> dynamic-rewards-wip
        value = min(1.0, max(0.0, value))
        if log:
            out.append(math.exp(math.log(lo) + value * (math.log(hi) - math.log(lo))))
        else:
            out.append(lo + value * (hi - lo))
    return out


def clamp_unit(value):
    return min(1.0, max(0.0, value))


def default_vec():
    return [default for (_, default, *_rest) in SPACE]


def vec_with(overrides):
    vec = default_vec()
    for index, (name, _, lo, hi, _) in enumerate(SPACE):
        if name in overrides:
<<<<<<< HEAD
            value = float(overrides[name])
            if math.isfinite(value):
                vec[index] = min(hi, max(lo, value))
=======
            lo = max(MIN_REWARD_WEIGHT, lo)
            hi = max(lo, hi)
            vec[index] = min(hi, max(lo, overrides[name]))
>>>>>>> dynamic-rewards-wip
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


def gate_deficit(metrics):
    if metrics is None:
        return 10.0
    weak_side = min(metrics["home"], metrics["away"])
    return (
        max(0.0, MIN_WEAK_SIDE - weak_side) / max(MIN_WEAK_SIDE, 1e-9)
        + max(0.0, MIN_SHOTS - metrics["shots_for"]) / max(MIN_SHOTS, 1)
        + max(0.0, MIN_SOT - metrics["sot_for"]) / max(MIN_SOT, 1)
        + max(0.0, MIN_SHOTS_AFTER_PASS - metrics["shots_after_pass"]) / max(MIN_SHOTS_AFTER_PASS, 1)
    )


def gates_ok(metrics):
    return gate_deficit(metrics) <= 1e-12


def gate_deficit(metrics):
    if metrics is None:
<<<<<<< HEAD
        return 10.0
    weak_side = min(metrics["home"], metrics["away"])
    return (
        max(0.0, MIN_WEAK_SIDE - weak_side) / max(MIN_WEAK_SIDE, 1e-9)
        + max(0.0, MIN_SHOTS - metrics["shots_for"]) / max(MIN_SHOTS, 1)
        + max(0.0, MIN_SOT - metrics["sot_for"]) / max(MIN_SOT, 1)
        + max(0.0, MIN_SHOTS_AFTER_PASS - metrics["shots_after_pass"])
        / max(MIN_SHOTS_AFTER_PASS, 1)
    )
=======
        return False
    return gate_deficit(metrics) <= 1e-12
>>>>>>> dynamic-rewards-wip


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
    score -= GATE_PENALTY * gate_deficit(metrics)
    return score


def aggregate_fitness(metrics_by_seed):
    scores = [fitness_from_metrics(metrics) for metrics in metrics_by_seed]
    mean = sum(scores) / len(scores)
    spread = max(scores) - min(scores) if len(scores) > 1 else 0.0
    return mean - 0.5 * spread


def summarize_seed_metrics(seed_metrics):
    valid = [metrics for metrics in seed_metrics if metrics is not None]
    if not valid:
        return {}
    avg_fields = ["eval_mean", "home", "away"]
    sum_fields = [
        "wins",
        "draws",
        "losses",
        "gd",
        "gf",
        "ga",
        "shots_for",
        "shots_against",
        "sot_for",
        "sot_against",
        "shots_after_pass",
        "fwd_margin",
        "net_fwd_margin",
    ]
    summary = {field: sum(metrics[field] for metrics in valid) / len(valid) for field in avg_fields}
    summary.update({field: sum(metrics[field] for metrics in valid) for field in sum_fields})
    summary["gate_deficit"] = sum(gate_deficit(metrics) for metrics in valid) / len(valid)
    return summary


def output_tail(output, max_lines=40):
    lines = (output or "").splitlines()
    return "\n".join(lines[-max_lines:])


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
        out, _ = proc.communicate(timeout=env_int("TUNE_TIMEOUT_SECONDS", 3600, lo=1))
        if proc.returncode != 0:
            print(
                f"candidate failed: cand={cand_index} seed={seed} returncode={proc.returncode}\n"
                + output_tail(out),
                file=sys.stderr,
            )
            return None
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        return None
    except OSError as exc:
        print(f"candidate launch failed: cand={cand_index} seed={seed}: {exc}", file=sys.stderr)
        return None
    # The ratchet does not currently write a checkpoint for the tuner, but the
    # outdir path is reserved in the row identity for future artifact capture.
    _ = outdir
    return parse(out)


<<<<<<< HEAD
def eval_pool(vectors):
    # Each child's stdout+stderr goes to a FILE, not a PIPE. A ratchet run prints
    # far more than the ~64KB OS pipe buffer, and the old code only drained the
    # pipe *after* poll() reported exit — so a child could block on write()
    # forever and wedge the whole (multi-hour) sweep. A file sink can't deadlock.
    # Each child also gets a hard deadline and is killed + scored as a failure if
    # it overruns (the safe run_one() already did this; eval_pool now matches).
    timeout_s = int(os.environ.get("TUNE_TIMEOUT_SECONDS", "3600"))
=======
def candidate_fit(results, candidate_index):
    seed_metrics = [results.get((candidate_index, si)) for si in range(len(SEEDS))]
    return aggregate_fitness(seed_metrics), seed_metrics


def append_candidate_row(gen, cand, label, vec, fit, seed_metrics):
    with open(LOG, "a", newline="") as file:
        writer = csv.writer(file)
        metrics = summarize_seed_metrics(seed_metrics)
        candidate_gates_ok = bool(seed_metrics) and all(gates_ok(metrics) for metrics in seed_metrics)
        writer.writerow(
            [gen, cand, label, f"{fit:.6f}"]
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
                metrics.get("gate_deficit"),
                candidate_gates_ok,
            ]
        )


def eval_pool(vectors, gen=None, labels=None):
    if labels is None:
        labels = [f"cand{index}" for index in range(len(vectors))]
>>>>>>> dynamic-rewards-wip
    jobs = [(ci, si, vec, seed) for ci, vec in enumerate(vectors) for si, seed in enumerate(SEEDS)]
    running = {}  # proc -> (ci, si, logpath, logfile, deadline)
    results = {}
    written = set()
    next_job = 0
    timeout_seconds = env_int("TUNE_TIMEOUT_SECONDS", 3600, lo=1)
    while next_job < len(jobs) or running:
        while next_job < len(jobs) and len(running) < MAXPAR:
            ci, si, vec, seed = jobs[next_job]
            next_job += 1
            env = dict(os.environ)
            env.update(FIXED_ENV)
            for (name, *_), value in zip(SPACE, vec):
                env[name] = f"{value:.8g}"
            outdir = os.path.join(ROOT, "out_tune_11v11", f"cand_{ci}_seed_{si}")
            os.makedirs(outdir, exist_ok=True)
            logpath = os.path.join(outdir, "stdout.log")
            logf = open(logpath, "w")
            cmd = [
                BIN,
                str(ITERS_GAMES),
                str(MINUTES),
                str(EVAL_GAMES),
                str(INCREMENTS),
                seed,
            ]
<<<<<<< HEAD
            proc = subprocess.Popen(
                cmd,
                env=env,
                cwd=ROOT,
                stdout=logf,
                stderr=subprocess.STDOUT,
                text=True,
            )
            running[proc] = (ci, si, logpath, logf, time.time() + timeout_s)
        time.sleep(1.0)
        for proc in list(running):
            ci, si, logpath, logf, deadline = running[proc]
            done = proc.poll() is not None
            if not done and time.time() > deadline:
                proc.kill(); proc.wait(); done = True
            if done:
                running.pop(proc)
                logf.close()
                try:
                    with open(logpath) as fh:
                        out = fh.read()
                except OSError:
                    out = ""
                results[(ci, si)] = parse(out)
    if not any(v is not None for v in results.values()):
        sys.exit("NO candidate produced a parseable summary this generation — the "
                 "ratchet output format likely changed (check CAND_RE) or every run "
                 "crashed/timed out. Aborting rather than ranking pure noise.")
=======
            try:
                proc = subprocess.Popen(
                    cmd,
                    env=env,
                    cwd=ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            except OSError as exc:
                results[(ci, si)] = None
                print(f"candidate launch failed: cand={ci} label={labels[ci]} seed={SEEDS[si]}: {exc}")
                continue
            running[proc] = (ci, si, time.time())
        time.sleep(1.0)
        now = time.time()
        for proc in list(running):
            ci, si, started_at = running[proc]
            timed_out = proc.poll() is None and now - started_at > timeout_seconds
            if proc.poll() is None and not timed_out:
                continue
            running.pop(proc)
            if timed_out:
                proc.kill()
                proc.communicate()
                results[(ci, si)] = None
                print(f"candidate timeout: cand={ci} label={labels[ci]} seed={SEEDS[si]}")
            else:
                out, _ = proc.communicate()
                if proc.returncode != 0:
                    results[(ci, si)] = None
                    print(
                        f"candidate failed: cand={ci} label={labels[ci]} seed={SEEDS[si]} "
                        f"returncode={proc.returncode}\n{output_tail(out)}"
                    )
                else:
                    results[(ci, si)] = parse(out)
            if gen is not None and ci not in written:
                candidate_done = all((ci, seed_index) in results for seed_index in range(len(SEEDS)))
                if candidate_done:
                    fit, seed_metrics = candidate_fit(results, ci)
                    append_candidate_row(gen, ci, labels[ci], vectors[ci], fit, seed_metrics)
                    print(
                        f"gen {gen} cand {ci} {labels[ci]}: "
                        f"fit={fit:+.3f} metrics={summarize_seed_metrics(seed_metrics)}"
                    )
                    sys.stdout.flush()
                    written.add(ci)
>>>>>>> dynamic-rewards-wip
    fits = []
    for ci in range(len(vectors)):
        fits.append(candidate_fit(results, ci))
    return fits


<<<<<<< HEAD
def smoke_bad_mutation():
    return vec_with(
        {
            "DD_SOCCER_GOAL_REWARD_SCALE": 1.0,
            "DD_SOCCER_FORWARD_PASS_REWARD_SCALE": 0.0,
            "DD_SOCCER_PASS_CHAIN_REWARD_SCALE": 0.0,
            "DD_SOCCER_PASS_TURNOVER_PENALTY_SCALE": 20.0,
            "DD_SOCCER_QUICK_FORWARD_RELEASE_REWARD_SCALE": 0.0,
            "DD_SOCCER_SHOT_SHAPING_REWARD_SCALE": 0.0,
            "DD_SOCCER_SHOT_COMMITMENT_REWARD_SCALE": 0.0,
            "DD_SOCCER_DRIBBLE_BEAT_REWARD_SCALE": 0.0,
            "DD_SOCCER_LEARNED_EPV_REWARD_SCALE": 0.0,
            "SOCCER_OFFBALL_SUPPORT_REWARD_SCALE": 0.0,
            "DD_SOCCER_CARRY_REWARD_SCALE": 0.0,
            "DD_SOCCER_RECOVERY_REWARD_SCALE": 0.0,
            "DD_SOCCER_OVERLOAD_PROGRESSION_REWARD_PER_YARD": 0.0,
            "SOCCER_MAPPO_TEAM_REWARD_SHARE": 0.0,
            "SOCCER_MARL_TEAM_REWARD_WEIGHT": 0.0,
            "SOCCER_MARL_INTERMEDIATE_REWARD_WEIGHT": 0.2,
            "SOCCER_PLANNER_TEACHER_WEIGHT": 0.0,
            "SOCCER_NEURAL_FINISHING_SELECT_BONUS_WEIGHT": 0.0,
        }
    )
=======
def legacy_default_vec():
    return vec_with({
        "DD_SOCCER_GOAL_REWARD_SCALE": 1.0,
        "DD_SOCCER_FORWARD_PASS_REWARD_SCALE": 6.25,
        "DD_SOCCER_PASS_CHAIN_REWARD_SCALE": 4.25,
        "DD_SOCCER_PASS_TURNOVER_PENALTY_SCALE": 5.0,
        "DD_SOCCER_QUICK_FORWARD_RELEASE_REWARD_SCALE": 1.25,
        "DD_SOCCER_SHOT_SHAPING_REWARD_SCALE": 0.85,
        "DD_SOCCER_SHOT_COMMITMENT_REWARD_SCALE": 0.0,
        "DD_SOCCER_DRIBBLE_BEAT_REWARD_SCALE": 1.0,
        "DD_SOCCER_LEARNED_EPV_REWARD_SCALE": 1.0,
        "SOCCER_OFFBALL_SUPPORT_REWARD_SCALE": 1.0,
        "DD_SOCCER_CARRY_REWARD_SCALE": 1.0,
        "DD_SOCCER_RECOVERY_REWARD_SCALE": 1.0,
        "DD_SOCCER_OVERLOAD_PROGRESSION_REWARD_PER_YARD": 0.0,
        "SOCCER_MAPPO_TEAM_REWARD_SHARE": 0.50,
        "SOCCER_MARL_TEAM_REWARD_WEIGHT": 0.35,
        "SOCCER_MARL_INTERMEDIATE_REWARD_WEIGHT": 1.0,
        "SOCCER_PLANNER_TEACHER_WEIGHT": 1.8,
        "SOCCER_NEURAL_FINISHING_SELECT_BONUS_WEIGHT": 0.70,
    })


def degraded_reward_hack_vec():
    return vec_with({
        "DD_SOCCER_GOAL_REWARD_SCALE": 0.5,
        "DD_SOCCER_FORWARD_PASS_REWARD_SCALE": 0.0,
        "DD_SOCCER_PASS_CHAIN_REWARD_SCALE": 0.0,
        "DD_SOCCER_PASS_TURNOVER_PENALTY_SCALE": 12.0,
        "DD_SOCCER_QUICK_FORWARD_RELEASE_REWARD_SCALE": 0.0,
        "DD_SOCCER_SHOT_SHAPING_REWARD_SCALE": 0.0,
        "DD_SOCCER_SHOT_COMMITMENT_REWARD_SCALE": 0.0,
        "DD_SOCCER_DRIBBLE_BEAT_REWARD_SCALE": 0.0,
        "DD_SOCCER_LEARNED_EPV_REWARD_SCALE": 0.0,
        "SOCCER_OFFBALL_SUPPORT_REWARD_SCALE": 0.5,
        "DD_SOCCER_CARRY_REWARD_SCALE": 0.5,
        "DD_SOCCER_RECOVERY_REWARD_SCALE": 0.5,
        "DD_SOCCER_OVERLOAD_PROGRESSION_REWARD_PER_YARD": 0.0,
        "SOCCER_ANALYTIC_DIFFERENCE_REWARD_NEGATIVE_DENSE_FLOOR": 0.0,
        "SOCCER_ANALYTIC_DIFFERENCE_REWARD_NEGATIVE_DENSE_SCALE": 0.0,
        "SOCCER_MAPPO_TEAM_REWARD_SHARE": 0.0,
        "SOCCER_MARL_TEAM_REWARD_WEIGHT": 0.0,
        "SOCCER_MARL_INTERMEDIATE_REWARD_WEIGHT": 0.2,
        "SOCCER_PLANNER_TEACHER_WEIGHT": 0.0,
        "SOCCER_NEURAL_FINISHING_SELECT_BONUS_WEIGHT": 0.0,
    })


def candidate_presets():
    return [
        ("anchor", default_vec()),
        (
            "finish_reentry",
            vec_with({
                "DD_SOCCER_GOAL_REWARD_SCALE": 1.2,
                "DD_SOCCER_FORWARD_PASS_REWARD_SCALE": 8.2,
                "DD_SOCCER_PASS_CHAIN_REWARD_SCALE": 6.8,
                "DD_SOCCER_PASS_TURNOVER_PENALTY_SCALE": 2.8,
                "DD_SOCCER_QUICK_FORWARD_RELEASE_REWARD_SCALE": 0.6,
                "DD_SOCCER_SHOT_SHAPING_REWARD_SCALE": 0.35,
                "DD_SOCCER_SHOT_COMMITMENT_REWARD_SCALE": 0.12,
                "DD_SOCCER_DRIBBLE_BEAT_REWARD_SCALE": 0.8,
                "DD_SOCCER_LEARNED_EPV_REWARD_SCALE": 0.25,
                "SOCCER_OFFBALL_SUPPORT_REWARD_SCALE": 3.0,
                "DD_SOCCER_CARRY_REWARD_SCALE": 1.0,
                "DD_SOCCER_RECOVERY_REWARD_SCALE": 1.0,
                "DD_SOCCER_OVERLOAD_PROGRESSION_REWARD_PER_YARD": 0.04,
                "SOCCER_MAPPO_TEAM_REWARD_SHARE": 0.05,
                "SOCCER_MARL_TEAM_REWARD_WEIGHT": 0.05,
                "SOCCER_MARL_INTERMEDIATE_REWARD_WEIGHT": 2.0,
                "SOCCER_PLANNER_TEACHER_WEIGHT": 0.5,
                "SOCCER_NEURAL_FINISHING_SELECT_BONUS_WEIGHT": 0.35,
            }),
        ),
        (
            "team_credit_reentry",
            vec_with({
                "DD_SOCCER_GOAL_REWARD_SCALE": 1.0,
                "DD_SOCCER_FORWARD_PASS_REWARD_SCALE": 8.8,
                "DD_SOCCER_PASS_CHAIN_REWARD_SCALE": 7.5,
                "DD_SOCCER_PASS_TURNOVER_PENALTY_SCALE": 3.0,
                "DD_SOCCER_QUICK_FORWARD_RELEASE_REWARD_SCALE": 0.35,
                "DD_SOCCER_SHOT_SHAPING_REWARD_SCALE": 0.15,
                "DD_SOCCER_DRIBBLE_BEAT_REWARD_SCALE": 0.7,
                "SOCCER_OFFBALL_SUPPORT_REWARD_SCALE": 3.25,
                "DD_SOCCER_CARRY_REWARD_SCALE": 1.0,
                "DD_SOCCER_RECOVERY_REWARD_SCALE": 1.25,
                "DD_SOCCER_OVERLOAD_PROGRESSION_REWARD_PER_YARD": 0.12,
                "SOCCER_ANALYTIC_DIFFERENCE_REWARD_NEGATIVE_DENSE_FLOOR": 0.15,
                "SOCCER_MAPPO_TEAM_REWARD_SHARE": 0.12,
                "SOCCER_MARL_TEAM_REWARD_WEIGHT": 0.10,
                "SOCCER_MARL_INTERMEDIATE_REWARD_WEIGHT": 2.1,
                "SOCCER_NEURAL_FINISHING_SELECT_BONUS_WEIGHT": 0.15,
            }),
        ),
        (
            "balanced_plus",
            vec_with({
                "DD_SOCCER_GOAL_REWARD_SCALE": 1.0,
                "DD_SOCCER_FORWARD_PASS_REWARD_SCALE": 9.6,
                "DD_SOCCER_PASS_CHAIN_REWARD_SCALE": 7.4,
                "DD_SOCCER_PASS_TURNOVER_PENALTY_SCALE": 2.2,
                "DD_SOCCER_QUICK_FORWARD_RELEASE_REWARD_SCALE": 0.55,
                "DD_SOCCER_SHOT_SHAPING_REWARD_SCALE": 0.22,
                "DD_SOCCER_DRIBBLE_BEAT_REWARD_SCALE": 1.1,
                "DD_SOCCER_LEARNED_EPV_REWARD_SCALE": 0.25,
                "SOCCER_OFFBALL_SUPPORT_REWARD_SCALE": 3.7,
                "DD_SOCCER_CARRY_REWARD_SCALE": 1.4,
                "DD_SOCCER_RECOVERY_REWARD_SCALE": 1.0,
                "DD_SOCCER_OVERLOAD_PROGRESSION_REWARD_PER_YARD": 0.07,
                "SOCCER_ANALYTIC_DIFFERENCE_REWARD_NEGATIVE_DENSE_FLOOR": 0.08,
                "SOCCER_MAPPO_TEAM_REWARD_SHARE": 0.12,
                "SOCCER_MARL_TEAM_REWARD_WEIGHT": 0.10,
                "SOCCER_MARL_INTERMEDIATE_REWARD_WEIGHT": 2.3,
                "SOCCER_PLANNER_TEACHER_WEIGHT": 0.35,
                "SOCCER_NEURAL_FINISHING_SELECT_BONUS_WEIGHT": 0.25,
            }),
        ),
        (
            "carry_transition",
            vec_with({
                "DD_SOCCER_GOAL_REWARD_SCALE": 0.8,
                "DD_SOCCER_FORWARD_PASS_REWARD_SCALE": 9.8,
                "DD_SOCCER_PASS_CHAIN_REWARD_SCALE": 6.5,
                "DD_SOCCER_PASS_TURNOVER_PENALTY_SCALE": 2.0,
                "DD_SOCCER_QUICK_FORWARD_RELEASE_REWARD_SCALE": 0.4,
                "DD_SOCCER_SHOT_SHAPING_REWARD_SCALE": 0.12,
                "DD_SOCCER_DRIBBLE_BEAT_REWARD_SCALE": 1.8,
                "SOCCER_OFFBALL_SUPPORT_REWARD_SCALE": 2.75,
                "DD_SOCCER_CARRY_REWARD_SCALE": 2.0,
                "DD_SOCCER_RECOVERY_REWARD_SCALE": 1.2,
                "DD_SOCCER_OVERLOAD_PROGRESSION_REWARD_PER_YARD": 0.08,
                "SOCCER_MARL_INTERMEDIATE_REWARD_WEIGHT": 2.4,
            }),
        ),
        (
            "quick_unlock",
            vec_with({
                "DD_SOCCER_GOAL_REWARD_SCALE": 1.1,
                "DD_SOCCER_FORWARD_PASS_REWARD_SCALE": 7.5,
                "DD_SOCCER_PASS_CHAIN_REWARD_SCALE": 6.0,
                "DD_SOCCER_PASS_TURNOVER_PENALTY_SCALE": 4.0,
                "DD_SOCCER_QUICK_FORWARD_RELEASE_REWARD_SCALE": 1.0,
                "DD_SOCCER_SHOT_SHAPING_REWARD_SCALE": 0.25,
                "DD_SOCCER_DRIBBLE_BEAT_REWARD_SCALE": 0.6,
                "DD_SOCCER_LEARNED_EPV_REWARD_SCALE": 0.2,
                "SOCCER_OFFBALL_SUPPORT_REWARD_SCALE": 2.5,
                "DD_SOCCER_CARRY_REWARD_SCALE": 1.2,
                "SOCCER_MAPPO_TEAM_REWARD_SHARE": 0.10,
                "SOCCER_MARL_INTERMEDIATE_REWARD_WEIGHT": 1.8,
                "SOCCER_PLANNER_TEACHER_WEIGHT": 0.35,
                "SOCCER_NEURAL_FINISHING_SELECT_BONUS_WEIGHT": 0.25,
            }),
        ),
        (
            "goal_pressure",
            vec_with({
                "DD_SOCCER_GOAL_REWARD_SCALE": 2.0,
                "DD_SOCCER_FORWARD_PASS_REWARD_SCALE": 7.0,
                "DD_SOCCER_PASS_CHAIN_REWARD_SCALE": 5.5,
                "DD_SOCCER_PASS_TURNOVER_PENALTY_SCALE": 3.5,
                "DD_SOCCER_QUICK_FORWARD_RELEASE_REWARD_SCALE": 0.65,
                "DD_SOCCER_SHOT_SHAPING_REWARD_SCALE": 0.4,
                "DD_SOCCER_SHOT_COMMITMENT_REWARD_SCALE": 0.2,
                "DD_SOCCER_DRIBBLE_BEAT_REWARD_SCALE": 0.9,
                "DD_SOCCER_LEARNED_EPV_REWARD_SCALE": 0.5,
                "SOCCER_OFFBALL_SUPPORT_REWARD_SCALE": 2.25,
                "DD_SOCCER_CARRY_REWARD_SCALE": 1.0,
                "SOCCER_MARL_INTERMEDIATE_REWARD_WEIGHT": 1.7,
                "SOCCER_PLANNER_TEACHER_WEIGHT": 0.7,
                "SOCCER_NEURAL_FINISHING_SELECT_BONUS_WEIGHT": 0.6,
            }),
        ),
        ("legacy_default", legacy_default_vec()),
    ]


def halton_unit(index, base):
    scale = 1.0
    value = 0.0
    while index > 0:
        scale /= base
        value += scale * (index % base)
        index //= base
    return value


def centered_halton(index, base):
    return halton_unit(index, base) * 2.0 - 1.0


def vec_signature(vec):
    return tuple(round(value, 6) for value in vec)


def add_candidate(candidates, seen, label, nvec):
    nvec = [clamp_unit(value) for value in nvec]
    vec = denorm(nvec)
    signature = vec_signature(vec)
    if signature in seen:
        return False
    candidates.append((label, nvec, vec))
    seen.add(signature)
    return True


def halton_around(mean, sigma, sequence, scale=1.0):
    return [
        mean[index] + sigma[index] * centered_halton(sequence, HALTON_BASES[index]) * scale
        for index in range(len(SPACE))
    ]


def halton_immigrant(mean, sequence):
    anchor_blend = env_float("TUNE_IMMIGRANT_ANCHOR_BLEND", 0.35, lo=0.0, hi=1.0)
    return [
        mean[index] * anchor_blend + halton_unit(sequence, HALTON_BASES[index]) * (1.0 - anchor_blend)
        for index in range(len(SPACE))
    ]


def generation_candidates(mean, sigma, gen, best_vec):
    candidates = []
    seen = set()
    presets = candidate_presets()
    if gen == 0:
        preset_limit = env_int("TUNE_PRESET_CANDIDATES", min(POP, 5), lo=0)
        for label, vec in presets[:preset_limit]:
            add_candidate(candidates, seen, f"preset:{label}", norm(vec))
    else:
        add_candidate(candidates, seen, "mean", mean)
        if best_vec is not None:
            add_candidate(candidates, seen, "global_best", norm(best_vec))
        scheduled = env_int("TUNE_SCHEDULED_PRESETS", 1, lo=0)
        rotating = presets[1:-1] or presets
        for offset in range(scheduled):
            label, vec = rotating[(gen + offset - 1) % len(rotating)]
            add_candidate(candidates, seen, f"preset:{label}", norm(vec))

    immigrants = env_int("TUNE_IMMIGRANTS", 1 if POP >= 4 else 0, lo=0)
    skip = env_int("TUNE_HALTON_SKIP", 17, lo=0)
    for slot in range(immigrants):
        sequence = skip + gen * max(1, POP) + slot + 1
        add_candidate(candidates, seen, f"immigrant:{sequence}", halton_immigrant(mean, sequence))

    if env_bool("TUNE_ANTITHETIC", True):
        pair = 0
        while len(candidates) + 1 < POP:
            offsets = [sigma[index] * RNG.gauss(0.0, 1.0) for index in range(len(SPACE))]
            added_plus = add_candidate(
                candidates,
                seen,
                f"anti:{pair}:plus",
                [mean[index] + offsets[index] for index in range(len(SPACE))],
            )
            added_minus = add_candidate(
                candidates,
                seen,
                f"anti:{pair}:minus",
                [mean[index] - offsets[index] for index in range(len(SPACE))],
            )
            pair += 1
            if not added_plus and not added_minus and pair > POP * 2:
                break

    fill_attempt = 0
    while len(candidates) < POP and fill_attempt < max(100, POP * 100):
        sequence = skip + gen * max(1, POP) + fill_attempt + 101
        add_candidate(candidates, seen, f"halton:{sequence}", halton_around(mean, sigma, sequence))
        fill_attempt += 1

    random_attempt = 0
    while len(candidates) < POP and random_attempt < max(100, POP * 100):
        add_candidate(
            candidates,
            seen,
            f"random:{random_attempt}",
            [RNG.random() for _ in range(len(SPACE))],
        )
        random_attempt += 1

    if len(candidates) < POP:
        raise RuntimeError(f"could not generate {POP} unique candidates")

    labels = [label for label, _, _ in candidates[:POP]]
    candidates_n = [nvec for _, nvec, _ in candidates[:POP]]
    vectors = [vec for _, _, vec in candidates[:POP]]
    return labels, candidates_n, vectors
>>>>>>> dynamic-rewards-wip


def smoke_test():
    vectors = [default_vec(), legacy_default_vec(), degraded_reward_hack_vec()]
    names = ["anchor", "legacy_default", "degraded_reward_hack"]
    print(
        f"smoke 11v11 tuner | weights={len(SPACE)} seeds={SEEDS} "
        f"games={ITERS_GAMES}x{INCREMENTS} eval={EVAL_GAMES} minutes={MINUTES}"
    )
    fits = eval_pool(vectors, labels=names)
    for name, (fit, metrics) in zip(names, fits):
        print(f"smoke {name}: fitness={fit:+.3f} metrics={metrics}")
    anchor_fit = fits[0][0]
    for name, (fit, _metrics) in zip(names[1:], fits[1:]):
        if not anchor_fit > fit:
            sys.exit(f"smoke failed: anchor vector did not outrank {name}")
    print("smoke ok: anchor vector outranks legacy and degraded reward-hack vectors")


def write_header():
    with open(LOG, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["gen", "cand", "label", "fitness"]
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
                "gate_deficit",
                "gates_ok",
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
        f"seeds={SEEDS} maxpar={MAXPAR} antithetic={env_bool('TUNE_ANTITHETIC', True)} "
        f"immigrants={os.environ.get('TUNE_IMMIGRANTS', str(1 if POP >= 4 else 0))}"
    )
    mean = norm(default_vec())
    sigma = [SIGMA0] * len(SPACE)
    best = None
    stale = 0
    restart_cursor = 0
    write_header()
    for gen in range(GENS):
        best_vec = best[2] if best is not None else None
        labels, candidates_n, candidates = generation_candidates(mean, sigma, gen, best_vec)
        fits = eval_pool(candidates, gen, labels)
        ranked = sorted(range(len(candidates)), key=lambda index: fits[index][0], reverse=True)
        elite_count = min(max(1, ELITE), len(ranked))
        elite = ranked[:elite_count]
        mean = [
            sum(candidates_n[index][dim] for index in elite) / elite_count
            for dim in range(len(SPACE))
        ]
        sigma = [
            max(
                0.03,
                min(
                    0.40,
                    math.sqrt(
                        sum((candidates_n[index][dim] - mean[dim]) ** 2 for index in elite)
                        / elite_count
                    )
                    * 1.3,
                ),
            )
            for dim in range(len(SPACE))
        ]
        best_index = ranked[0]
        previous_best_fit = best[0] if best is not None else None
        if previous_best_fit is None or fits[best_index][0] > previous_best_fit:
            best = (fits[best_index][0], labels[best_index], candidates[best_index], fits[best_index][1])
            stale = 0
        else:
            stale += 1
        print(
            f"gen {gen}: best_fit={fits[best_index][0]:+.3f} "
            f"label={labels[best_index]} metrics={summarize_seed_metrics(fits[best_index][1])} weights="
            + ", ".join(f"{name}={value:.4g}" for (name, *_), value in zip(SPACE, candidates[best_index]))
        )
        restart_patience = env_int("TUNE_RESTART_PATIENCE", 2, lo=0)
        if restart_patience > 0 and stale >= restart_patience:
            restart_presets = candidate_presets()[1:-1] or candidate_presets()
            restart_label, restart_vec = restart_presets[restart_cursor % len(restart_presets)]
            restart_cursor += 1
            restart_blend = env_float("TUNE_RESTART_BLEND", 0.70, lo=0.0, hi=1.0)
            restart_n = norm(restart_vec)
            mean = [
                mean[index] * (1.0 - restart_blend) + restart_n[index] * restart_blend
                for index in range(len(SPACE))
            ]
            sigma = [max(value, SIGMA0) for value in sigma]
            stale = 0
            print(f"gen {gen}: restart around preset:{restart_label} sigma_floor={SIGMA0:.3f}")
        sys.stdout.flush()
    print("\n=== BEST ===")
    print(f"fitness={best[0]:+.3f} label={best[1]} metrics={best[3]}")
    for (name, *_), value in zip(SPACE, best[2]):
        print(f"  {name}={value:.8g}")


if __name__ == "__main__":
    main()
