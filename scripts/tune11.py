#!/usr/bin/env python3
"""Bounded reward-weight tuner for the 11v11 soccer engine (soccer_proof harness).

Port of the standalone 5v5 optimizer (standalone-5v5/viz/tune.py) to the full
engine: a simple, robust (mu, lambda) evolution strategy over a SPACE of
bounded LIVE reward knobs, with normalized (log-aware) mutations, an
anti-passivity gate, and a CSV log. Zero third-party deps — stdlib only.

Each candidate is TWO subprocess runs of the `soccer_proof` binary:
  1. train:  BIN train-ckpt <jobdir>/cand <TUNE_GAMES> <TUNE_MINUTES> <seed_hex>
             <ckpt_every=TUNE_GAMES>            (candidate reward env applied)
  2. eval:   BIN eval <jobdir>/cand.genFINAL.json analytic <TUNE_EVAL_GAMES>
             <TUNE_EVAL_MINUTES> <TUNE_HOLDOUT>  (frozen; held-out seeds)
Fitness = eval goal-diff/game (candidate − analytic), GATED on candidate
shots-on-target >= TUNE_MIN_SOT (default 0.5/game): a passive net that never
threatens the goal is penalized TUNE_GATE_PENALTY (default 6.0) exactly like
the 5v5 gate pattern. EVAL stats decide fitness — training reward never does.

FIXED ENV every candidate runs under (plus the caller's inherited env):
  DD_SOCCER_ENABLE_ANCHORED_REWARDS=1
      The anchored reward currency (docs/reward-anchoring.md) is the FIXED
      frame: goal=500 / win=1000 / on-frame-shot floor=50 are anchors inside
      the binary and are NOT searched — the tuner discovers everything else,
      in points, inside these sane bounds.
  SOCCER_DYNAMIC_REWARD_WEIGHTS=1
      The env-reread gate for `dynamic_reward_weights_enabled()` in
      src/des/general/soccer.rs. (NOTE: the gate's real name — there is no
      DD_SOCCER_ENABLE_DYNAMIC_REWARD_WEIGHTS.) Without it every reward knob
      is OnceLock-cached at first use AND several knobs use tighter
      static-mode clamps (e.g. carry/off-ball floor 0.1 instead of 1e-4).
      Dynamic mode is the documented "reward search" mode: the binary clamps
      each knob to the same bounds this SPACE searches, so the search space
      == the legal space.
The caller's env passes through untouched otherwise, so learning-stack gates
(SOCCER_LEARNING_ANALYTIC_OPPONENT, DP gates, ...) are the caller's choice.
PROOF_JSONL is redirected per candidate (jobdir/eval.jsonl) so parallel
candidates cannot interleave rows into one shared file.

Usage:
    ./scripts/tune11.py                                   # small overnight-ish defaults
    TUNE_POP=6 TUNE_GENS=8 TUNE_MAXPAR=2 ./scripts/tune11.py
    # tiny end-to-end harness check (one candidate cycle):
    TUNE_GAMES=2 TUNE_EVAL_GAMES=2 TUNE_POP=1 TUNE_GENS=1 ./scripts/tune11.py
Each candidate is a full training run + a held-out eval, so this is an
OVERNIGHT tool — defaults are deliberately small; everything is
env-overridable.
"""
import os, re, csv, sys, math, time, random, subprocess
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BIN = os.environ.get(
    "TUNE_BIN", os.path.join(ROOT, ".cargo-target-anchor", "release", "soccer_proof")
)
# Matches MIN_SOCCER_REWARD_WEIGHT in soccer.rs: channels may get tiny but never
# collapse to zero and vanish from the learned objective.
MIN_WEIGHT = 0.0001


def env_int(name, default, lo=None, hi=None):
    try:
        value = int(os.environ.get(name, str(default)), 0)
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


# ── search space: (ENV_VAR, default, low, high, log_scale) ───────────────────
# The bounded LIVE reward knobs of the 11v11 dense/self-play reward stack (all
# read via reward_weight_env / optional_reward_weight_env in soccer.rs, which
# clamp to these same legal ranges in dynamic mode). The sparse anchors
# (goal/win/on-frame floor) are FIXED by DD_SOCCER_ENABLE_ANCHORED_REWARDS=1
# and not searched.
SPACE = [
    # Completed FORWARD pass reward multiplier — forward-pass primacy: how much a
    # completed progressive ball is worth vs the historical baseline (1.0).
    ("DD_SOCCER_FORWARD_PASS_REWARD_SCALE", 1.0, 0.25, 8, False),
    # Consecutive-forward-pass (build-up chain) event rewards — pays STRINGS of
    # forward passes (2/3-chain events), i.e. sustained progression rather than
    # single hopeful forward balls.
    ("DD_SOCCER_PASS_CHAIN_REWARD_SCALE", 1.0, 0.25, 8, False),
    # Pass-turnover penalty coupling — losing the ball on a pass must scale up
    # with the forward-pass carrot or the learner farms gross forward count.
    # Binary floors this at 1.0 (giveaways never get cheaper than baseline).
    ("DD_SOCCER_PASS_TURNOVER_PENALTY_SCALE", 1.0, 1.0, 8, False),
    # Progressive ball-CARRY / dribble-into-space reward multiplier (distinct
    # from the take-on/dribble-beat reward).
    ("DD_SOCCER_CARRY_REWARD_SCALE", 1.0, 0.25, 6, False),
    # Dense OFF-BALL support reward — "make a run / get open" shaping that
    # creates safe passing options; too high and it becomes a farmable shard.
    ("SOCCER_OFFBALL_SUPPORT_REWARD_SCALE", 1.0, 0.25, 6, False),
    # Shot-commitment reward — pays actually PULLING THE TRIGGER in good
    # positions (the conversion-climb lever); counters terminal passivity.
    ("DD_SOCCER_SHOT_COMMITMENT_REWARD_SCALE", 1.0, 0.25, 6, False),
    # Carrier hold-too-long penalty: -w1 * time_on_ball * perceived_pressure per
    # tick — pushes a quicker release or a move when pressed. Default-off knob
    # in the binary (0.0); searched log-scale from ~off to strong.
    ("DD_SOCCER_HOLD_UNDER_PRESSURE_PENALTY_SCALE", 0.05, 0.0001, 1.5, True),
    # Dribble-too-slow penalty: -w2 * max(0, jog_speed - forward_speed) while
    # dribbling — the optimum becomes carrying AT a jog, not standing on it.
    ("DD_SOCCER_DRIBBLE_MIN_GAIT_PENALTY_SCALE", 0.025, 0.0001, 1.5, True),
    # Turnover POSITION weight: turnover penalty *= 1 + w3 * opponent_threat(at
    # loss spot), capped at 5x — giveaways deep in your own third are punished
    # hard, ambitious final-third losses stay cheap (anti-risk-aversion fix).
    ("DD_SOCCER_TURNOVER_POSITION_WEIGHT_SCALE", 1.0, 0.0001, 6, True),
]

GAMES = env_int("TUNE_GAMES", 30, lo=1)  # training games per candidate
MINUTES = env_float("TUNE_MINUTES", 2.0, lo=0.25)  # training game length (min)
EVALG = env_int("TUNE_EVAL_GAMES", 24, lo=1)  # held-out eval games
EVALMIN = env_float("TUNE_EVAL_MINUTES", 2.0, lo=0.25)  # eval game length (min)
POP = env_int("TUNE_POP", 4, lo=1)
ELITE = min(env_int("TUNE_ELITE", 2, lo=1), POP)
GENS = env_int("TUNE_GENS", 3, lo=1)
MAXPAR = env_int("TUNE_MAXPAR", 1, lo=1)  # candidates are HEAVY: default serial
SEEDS = []
for raw_seed in os.environ.get("TUNE_SEEDS", "71A17001").split(","):
    raw_seed = raw_seed.strip()
    if not raw_seed:
        continue
    try:
        int(raw_seed, 16)  # train-ckpt parses its seed argument as hex
        SEEDS.append(raw_seed)
    except ValueError:
        print(f"warning: ignoring invalid TUNE_SEEDS hex entry {raw_seed!r}", file=sys.stderr)
HOLDOUT = os.environ.get("TUNE_HOLDOUT", "E7A1C0DE")  # eval holdout seed base (hex)
SIGMA0 = env_float("TUNE_SIGMA", 0.25, lo=0.001, hi=1.0)  # initial step, normalized units
TIMEOUT = env_int("TUNE_TIMEOUT", 10800, lo=1)  # per-subprocess hard deadline (s)
MIN_SOT = env_float("TUNE_MIN_SOT", 0.5)  # anti-passivity gate: cand SOT/game floor
GATE_PENALTY = env_float("TUNE_GATE_PENALTY", 6.0)  # fitness penalty when gated out
RNG = random.Random(env_int("TUNE_RNG", 1))
OUT = os.environ.get("TUNE_OUT", os.path.join(ROOT, "out_tune11"))
LOG = os.environ.get("TUNE_LOG", os.path.join(OUT, "tune11_log.csv"))
# Fixed env every candidate inherits — see the module docstring for WHY.
FIXED = {
    "DD_SOCCER_ENABLE_ANCHORED_REWARDS": "1",
    "SOCCER_DYNAMIC_REWARD_WEIGHTS": "1",
}

if not SEEDS:
    sys.exit("TUNE_SEEDS is empty — provide at least one hex seed")
try:
    int(HOLDOUT, 16)
except ValueError:
    sys.exit(f"TUNE_HOLDOUT={HOLDOUT!r} is not a hex seed")

# Eval-output parsers (soccer_proof `eval` guardrails block; sniffed 2026-07-13):
#   goals:        1.50 vs 0.50   (GD +1.00)
#   shots on tgt: 2.50 vs 1.50
GOALS_RE = re.compile(r"goals:\s+([\d.]+) vs ([\d.]+)\s+\(GD ([+-]?[\d.]+)\)")
SOT_RE = re.compile(r"shots on tgt:\s+([\d.]+) vs ([\d.]+)")


def norm(vec):
    """map raw weights -> normalized [0,1] per dim (log-aware)."""
    out = []
    for (name, d, lo, hi, log), v in zip(SPACE, vec):
        lo, hi = max(MIN_WEIGHT, lo), max(max(MIN_WEIGHT, lo), hi)
        v = min(hi, max(lo, v))
        if log:
            v, lo, hi = math.log(v), math.log(lo), math.log(hi)
        out.append((v - lo) / (hi - lo))
    return out


def denorm(nvec):
    """map normalized [0,1] -> raw weights (clipped, log-aware)."""
    out = []
    for (name, d, lo, hi, log), n in zip(SPACE, nvec):
        lo, hi = max(MIN_WEIGHT, lo), max(max(MIN_WEIGHT, lo), hi)
        n = min(1.0, max(0.0, n))
        if log:
            v = math.exp(math.log(lo) + n * (math.log(hi) - math.log(lo)))
        else:
            v = lo + n * (hi - lo)
        out.append(v)
    return out


def clamp_space_value(index, value):
    _, _, lo, hi, _ = SPACE[index]
    lo, hi = max(MIN_WEIGHT, lo), max(max(MIN_WEIGHT, lo), hi)
    return min(hi, max(lo, value))


def ground_reward_vector(vec):
    """Project a candidate onto the legal space. The anchors (goal/win/on-frame
    floor) are constants behind DD_SOCCER_ENABLE_ANCHORED_REWARDS, so grounding
    reduces to per-dimension bound clamping (the binary clamps identically in
    dynamic-reward mode)."""
    return [clamp_space_value(i, v) for i, v in enumerate(vec)]


def candidate_env(vec):
    env = dict(os.environ)
    env.update(FIXED)
    for (name, *_), v in zip(SPACE, vec):
        env[name] = f"{v:.6g}"
    return env


def parse_eval(stdout):
    """Parse goal diff / goals / shots-on-target per game from `eval` output."""
    gm = GOALS_RE.search(stdout)
    sm = SOT_RE.search(stdout)
    if not gm or not sm:
        return None
    return {
        "goals_cand": float(gm.group(1)),
        "goals_base": float(gm.group(2)),
        "goal_diff": float(gm.group(3)),
        "sot_cand": float(sm.group(1)),
        "sot_base": float(sm.group(2)),
    }


def run_logged(cmd, env, logpath):
    """Run one subprocess with stdout+stderr sunk to a FILE (a training run
    prints far more than the OS pipe buffer — a file sink cannot deadlock) and
    a hard deadline. Returns the log text, or None on failure/timeout."""
    with open(logpath, "w") as logf:
        try:
            proc = subprocess.run(
                cmd, env=env, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT, timeout=TIMEOUT
            )
        except subprocess.TimeoutExpired:
            print(f"warning: {' '.join(cmd[:2])} timed out after {TIMEOUT}s ({logpath})",
                  file=sys.stderr)
            return None
        except OSError as exc:
            print(f"warning: {' '.join(cmd[:2])} failed to start: {exc}", file=sys.stderr)
            return None
    try:
        with open(logpath) as fh:
            out = fh.read()
    except OSError:
        out = ""
    if proc.returncode != 0:
        tail = "\n".join(out.splitlines()[-15:])
        print(f"warning: {' '.join(cmd[:2])} exited {proc.returncode} ({logpath}):\n{tail}",
              file=sys.stderr)
        return None
    return out


def run_candidate(gen, ci, si, vec, seed):
    """One candidate x seed cycle: train-ckpt with the candidate reward env,
    then a frozen held-out eval of the FINAL checkpoint vs the pure-analytic
    baseline. Returns parsed eval metrics or None."""
    vec = ground_reward_vector(vec)
    jobdir = os.path.join(OUT, f"g{gen}_c{ci}_s{si}")
    os.makedirs(jobdir, exist_ok=True)
    env = candidate_env(vec)
    prefix = os.path.join(jobdir, "cand")
    train_cmd = [BIN, "train-ckpt", prefix, str(GAMES), str(MINUTES), seed, str(GAMES)]
    if run_logged(train_cmd, env, os.path.join(jobdir, "train.log")) is None:
        return None
    final = f"{prefix}.genFINAL.json"
    if not os.path.exists(final):
        print(f"warning: candidate g{gen}c{ci}s{si} produced no {final}", file=sys.stderr)
        return None
    # Frozen eval: reward weights do not act on frozen brains, but keep the same
    # env for consistency. Redirect PROOF_JSONL per candidate so parallel evals
    # never interleave into one shared file.
    eval_env = dict(env)
    eval_env["PROOF_JSONL"] = os.path.join(jobdir, "eval.jsonl")
    eval_cmd = [BIN, "eval", final, "analytic", str(EVALG), str(EVALMIN), HOLDOUT]
    out = run_logged(eval_cmd, eval_env, os.path.join(jobdir, "eval.log"))
    if out is None:
        return None
    return parse_eval(out)


def gated_fitness(m):
    """Goal-diff/game, gated on candidate shots-on-target (anti-passivity): a
    net that stops threatening the goal cannot win the search on turtling."""
    if m is None:
        return -10.0
    gd = m["goal_diff"]
    if m["sot_cand"] < MIN_SOT:
        gd -= GATE_PENALTY
    return gd


def eval_pool(gen, vectors):
    """Evaluate one generation: (candidate, seed) jobs on a small thread pool
    (each job is two sequential subprocesses with file-sunk output and a hard
    deadline, so one hung run cannot wedge an overnight sweep). Reduce per
    candidate over seeds as mean - 0.5*spread (5v5 robustness pattern)."""
    vectors = [ground_reward_vector(vec) for vec in vectors]
    jobs = [(ci, si) for ci in range(len(vectors)) for si in range(len(SEEDS))]
    results = {}
    with ThreadPoolExecutor(max_workers=MAXPAR) as pool:
        futs = {
            (ci, si): pool.submit(run_candidate, gen, ci, si, vectors[ci], SEEDS[si])
            for ci, si in jobs
        }
        for (ci, si), fut in futs.items():
            try:
                results[(ci, si)] = fut.result()
            except Exception as exc:  # job crash != sweep crash
                print(f"warning: candidate g{gen}c{ci}s{si} raised: {exc}", file=sys.stderr)
                results[(ci, si)] = None
    if not any(v is not None for v in results.values()):
        sys.exit(
            "NO candidate produced a parseable eval this generation — the eval "
            "output format likely changed (check GOALS_RE/SOT_RE against a "
            "manual `soccer_proof eval` run) or every run crashed/timed out. "
            "Aborting rather than ranking pure noise."
        )
    fits = []
    for ci in range(len(vectors)):
        gds = [gated_fitness(results.get((ci, si))) for si in range(len(SEEDS))]
        mean = sum(gds) / len(gds)
        spread = (max(gds) - min(gds)) if len(gds) > 1 else 0.0
        fits.append((mean - 0.5 * spread, gds, results.get((ci, 0))))
    return fits


def main():
    if not os.path.exists(BIN):
        sys.exit(
            f"build first: CARGO_TARGET_DIR=$PWD/.cargo-target-anchor cargo build "
            f"--release --bin soccer_proof  (missing {BIN})"
        )
    os.makedirs(OUT, exist_ok=True)
    print(
        f"tuning {len(SPACE)} reward knobs | train {GAMES}g x {MINUTES}min | "
        f"eval {EVALG}g x {EVALMIN}min vs analytic @0x{HOLDOUT} | pop={POP} elite={ELITE} "
        f"gens={GENS} seeds={SEEDS} maxpar={MAXPAR} | gate: SOT>={MIN_SOT}/g"
    )
    print(f"fixed env: {' '.join(f'{k}={v}' for k, v in FIXED.items())}")
    mean = norm([d for (_, d, *_) in SPACE])
    sigma = [SIGMA0] * len(SPACE)
    best = None
    with open(LOG, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["gen", "cand", "fitness"] + [s[0] for s in SPACE]
            + ["goal_diff", "goals_cand", "goals_base", "sot_cand", "sot_base", "sot_gate_ok"]
        )
    started = time.time()
    for gen in range(GENS):
        # sample POP candidates ~ N(mean, sigma) in normalized space, clipped to [0,1];
        # candidate 0 is the current mean (elitism).
        cands_n = [list(mean)]
        for _ in range(POP - 1):
            cands_n.append(
                [min(1.0, max(0.0, mean[i] + sigma[i] * RNG.gauss(0, 1))) for i in range(len(SPACE))]
            )
        cands = [ground_reward_vector(denorm(n)) for n in cands_n]
        cands_n = [norm(c) for c in cands]
        fits = eval_pool(gen, cands)
        ranked = sorted(range(len(cands)), key=lambda i: fits[i][0], reverse=True)
        with open(LOG, "a", newline="") as f:
            w = csv.writer(f)
            for i in range(len(cands)):
                fit, gds, m = fits[i]
                row = [gen, i, f"{fit:.3f}"] + [f"{v:.5g}" for v in cands[i]]
                if m:
                    row += [m["goal_diff"], m["goals_cand"], m["goals_base"],
                            m["sot_cand"], m["sot_base"], m["sot_cand"] >= MIN_SOT]
                w.writerow(row)
        # update mean/sigma from the elite
        elite = ranked[:ELITE]
        new_mean = [sum(cands_n[i][d] for i in elite) / len(elite) for d in range(len(SPACE))]
        new_sigma = []
        for d in range(len(SPACE)):
            mu = new_mean[d]
            var = sum((cands_n[i][d] - mu) ** 2 for i in elite) / len(elite)
            new_sigma.append(max(0.03, min(0.4, math.sqrt(var) * 1.3)))
        mean, sigma = new_mean, new_sigma
        bi = ranked[0]
        bfit = fits[bi][0]
        if best is None or bfit > best[0]:
            best = (bfit, cands[bi], fits[bi][2])
        bm = fits[bi][2] or {}
        print(
            f"gen {gen}: best_fit={bfit:+.3f} gd={bm.get('goal_diff')} "
            f"sot={bm.get('sot_cand')} ({time.time() - started:.0f}s) | weights="
            + ", ".join(f"{s[0]}={v:.4g}" for s, v in zip(SPACE, cands[bi]))
        )
        sys.stdout.flush()
    print("\n=== BEST ===")
    print(f"fitness={best[0]:+.3f}")
    for s, v in zip(SPACE, best[1]):
        print(f"  {s[0]}={v:.5g}")
    print("\nConfirm at full length + multi-seed before shipping, e.g.:")
    print(
        "  " + " ".join(f"{k}={v}" for k, v in FIXED.items()) + " "
        + " ".join(f"{s[0]}={v:.5g}" for s, v in zip(SPACE, best[1]))
        + f" {BIN} train-ckpt out/full 200 2 71A17001 20"
    )


if __name__ == "__main__":
    main()
