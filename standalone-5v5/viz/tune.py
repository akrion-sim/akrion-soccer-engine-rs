#!/usr/bin/env python3
"""Reward-weight optimizer for the standalone 5-a-side demo.

Runs the Rust trainer as a subprocess with the reward weights set via env vars
(see `Rw`/`wenv` in src/train.rs), parses the FINAL 300-game holdout line + the
trainer's own hardening-gate verdict, and searches the reward vector that
maximizes GATED goal-difference. Uses a simple, robust (mu, lambda) evolution
strategy — well suited to the rugged, cliffy reward landscape (a single gate can
swing goal-diff from ~0 to +6). Zero third-party deps — stdlib only.

Why gated goal-diff and not "max passes" / "max shots"? Because those get farmed
(the 200-passes / 60-shots pinball). Goals/wins are the honest objective; the
trainer's behavior gates (spacing/possession/bunch/completion floors) are hard
constraints that zero out reward-hacked candidates.

Usage:
    ./viz/tune.py                      # default search (frozen-speed action reward)
    TUNE_GENS=15 TUNE_POP=10 ./viz/tune.py
    TUNE_ITERS=400 TUNE_SEEDS=1,2,3 TUNE_MAXPAR=3 ./viz/tune.py
Each candidate is one full training run, so this is compute-heavy — start small.
"""
import os, re, csv, sys, math, time, random, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BIN = os.path.join(ROOT, "target", "release", "fiveaside")
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
# Tune the reward vector that shapes the field-vector-conditioned POMDP rewards.
# Structural knobs (SHOOT_X, BALL_SPEED_CAP) are held fixed here; the scalar
# weights below multiply dynamic terms derived from the 10-player field vector.
# Lower bounds are intentionally positive: channels may get tiny, but never
# collapse to zero and disappear from the learned objective.
SPACE = [
    ("REW_GOAL",      12.0,  6.0,  20.0, False),
    ("REW_CONCEDE",    8.0,  4.0,  16.0, False),
    ("REW_SHOT_BASE",  1.5,  0.3,   3.5, False),
    ("REW_SHOT_Q",     1.0,  MIN_WEIGHT,   2.5, False),
    ("REW_MILESTONE",  0.3,  MIN_WEIGHT,   1.5, False),
    ("REW_PASS",       0.06, MIN_WEIGHT,   0.4, False),
    ("REW_TURNOVER",   0.55, MIN_WEIGHT,   1.5, False),
    ("REW_BAD_PASS_TURNOVER", 0.35, MIN_WEIGHT, 1.5, False),
    ("REW_DRIBBLE_TURNOVER", 0.75, MIN_WEIGHT, 2.0, False),
    ("REW_RECYCLE",    0.18, MIN_WEIGHT,   0.8, False),
    ("REW_RETURN_PASS", 0.35, MIN_WEIGHT,  2.0, False),
    ("REW_RETURN_STALE", 0.55, MIN_WEIGHT, 2.0, False),
    ("REW_WIN_BALL",   0.3,  MIN_WEIGHT,   1.2, False),
    ("REW_DRIBBLE",    0.015, MIN_WEIGHT,  0.12, False),
    ("W_SHAPE",        2.2,  0.5,   4.0, False),
    ("SPACING_W",      0.003, 0.0005, 0.012, True),
    ("W_ADVANCE",      0.04, MIN_WEIGHT,   0.12, False),
    ("W_OPEN",         0.04, MIN_WEIGHT,   0.12, False),
    ("W_WIDTH",        0.045, MIN_WEIGHT,  0.12, False),
    ("W_FLANK",        0.025, MIN_WEIGHT,  0.10, False),
    ("W_GOALSIDE",     0.08, MIN_WEIGHT,   0.25, False),
    ("W_GOALSIDE_RUN", 0.04, MIN_WEIGHT,   0.16, False),
    ("W_AHEAD",        0.035, MIN_WEIGHT,  0.16, False),
    ("W_MAKE_RUN",     0.06, MIN_WEIGHT,   0.20, False),
    ("W_BURST_GEAR",   0.035, MIN_WEIGHT,  0.16, False),
    ("W_FIELD_PASS",   0.08, MIN_WEIGHT,   0.30, False),
    ("W_FIELD_TURNOVER", 0.16, MIN_WEIGHT, 0.50, False),
    ("W_FIELD_GOALSIDE_DELTA", 0.10, MIN_WEIGHT, 0.35, False),
    ("W_FIELD_BURST_DELTA", 0.08, MIN_WEIGHT, 0.35, False),
    ("W_STAND_PEN",    0.02, MIN_WEIGHT,   0.20, False),
    # MARL chance-creation weight — the binary reads this (train.rs wenv("W_CHANCE",..));
    # it was previously frozen at its default and excluded from the search, silently
    # under-covering the 31-weight reward vector. Now tuned like the rest.
    ("W_CHANCE",       0.12, MIN_WEIGHT,   0.60, False),
    # loose-ball pursuit — how hard the favorite commits to winning a free ball.
    ("W_PURSUIT",      0.05, MIN_WEIGHT,   0.25, False),
]

ITERS   = env_int("TUNE_ITERS", 500, lo=1)
FINALG  = env_int("TUNE_FINAL_GAMES", 200, lo=1)
POP     = env_int("TUNE_POP", 8, lo=1)
ELITE   = env_int("TUNE_ELITE", 3, lo=1)
GENS    = env_int("TUNE_GENS", 10, lo=1)
MAXPAR  = env_int("TUNE_MAXPAR", 2, lo=1)
SEEDS   = []
for raw_seed in os.environ.get("TUNE_SEEDS", "20260710").split(","):
    raw_seed = raw_seed.strip()
    if not raw_seed:
        continue
    try:
        SEEDS.append(int(raw_seed, 0))
    except ValueError:
        print(f"warning: ignoring invalid TUNE_SEEDS entry {raw_seed!r}", file=sys.stderr)
SIGMA0  = env_float("TUNE_SIGMA", 0.25, lo=0.001, hi=1.0)   # initial step, in normalized [0,1] units
TIMEOUT = env_int("TUNE_TIMEOUT", 3600, lo=1)   # per-candidate hard deadline (s)
RNG     = random.Random(env_int("TUNE_RNG", 1))
# Fixed env every candidate inherits. Frozen speed = the config that actually
# attacks; we optimize the action-policy reward first.
FIXED   = {"SPEED_WARMUP": os.environ.get("TUNE_SPEED_WARMUP", "250")}
LOG     = os.path.join(HERE, "tune_log.csv")

# Fail fast on misconfiguration instead of dividing by zero deep in the sweep.
if not SEEDS:
    sys.exit("TUNE_SEEDS is empty — provide at least one integer seed")
if POP < 1:
    sys.exit(f"TUNE_POP={POP} must be >= 1")
if not (1 <= ELITE <= POP):
    sys.exit(f"TUNE_ELITE={ELITE} must be in [1, TUNE_POP={POP}]")

# hardening gates on the FINAL line (in addition to the trainer's own gates=)
def gates_ok(m):
    return (4.0 <= m["spacing"] <= 12.0 and m["bunch"] <= 0.40
            and m["passes"] >= 8.0 and m["trainer_gates"])

FINAL_RE = re.compile(
    r"goal_diff=([+-][\d.]+)\s+winrate=([\d.]+)\s+goals\s+([\d.]+)-([\d.]+)"
    r"\s+passes/game\s+([\d.]+)\s+spacing=([\d.]+)\s+bunch=([\d]+)%")
GATE_RE = re.compile(r"gates=(true|false)")


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


def parse(stdout):
    fm = FINAL_RE.search(stdout)
    gm = None
    for line in stdout.splitlines():
        if "best policy at iter" in line:
            gm = GATE_RE.search(line)
    if not fm:
        return None
    gd, wr, ga, gb, passes, sp, bunch = fm.groups()
    return {
        "goal_diff": float(gd), "winrate": float(wr), "goals_a": float(ga),
        "goals_b": float(gb), "passes": float(passes), "spacing": float(sp),
        "bunch": float(bunch) / 100.0,
        "trainer_gates": (gm.group(1) == "true") if gm else False,
    }


def run_one(vec, seed):
    env = dict(os.environ)
    env.update(FIXED)
    for (name, *_), v in zip(SPACE, vec):
        env[name] = f"{v:.6g}"
    cmd = [BIN, "train", str(ITERS), "--seed", str(seed),
           "--final-games", str(FINALG), "--out-dir", os.path.join(ROOT, "out_tune")]
    try:
        p = subprocess.run(cmd, env=env, cwd=ROOT, capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        print(f"warning: candidate seed={seed} timed out after {TIMEOUT}s", file=sys.stderr)
        return None
    except OSError as exc:
        print(f"warning: candidate seed={seed} failed to start: {exc}", file=sys.stderr)
        return None
    if p.returncode != 0:
        tail = (p.stderr or p.stdout)[-800:]
        print(f"warning: candidate seed={seed} exited {p.returncode}: {tail}", file=sys.stderr)
        return None
    return parse(p.stdout)


def fitness(vec):
    """avg gated goal-diff over seeds (min-ish robustness via mean - 0.5*spread)."""
    gds = []
    for s in SEEDS:
        m = run_one(vec, s)
        if m is None:
            gds.append(-10.0)
            continue
        gd = m["goal_diff"] if gates_ok(m) else m["goal_diff"] - 6.0
        gds.append(gd)
    mean = sum(gds) / len(gds)
    spread = (max(gds) - min(gds)) if len(gds) > 1 else 0.0
    return mean - 0.5 * spread, gds


def eval_pool(vectors):
    """evaluate a generation with a small process pool over (candidate, seed).

    Each child's stdout+stderr is redirected to a FILE, not a PIPE: a training
    run prints far more than the ~64KB OS pipe buffer, and the old code only
    drained the pipe *after* poll() reported exit — so a child could block on
    write() forever and wedge the whole sweep. A file sink can't deadlock. Each
    child also gets a hard deadline (TIMEOUT) and is killed + scored as a failure
    if it overruns, so one hung run can't stall a multi-hour search."""
    jobs = [(ci, si, vectors[ci], s) for ci in range(len(vectors)) for si, s in enumerate(SEEDS)]
    results = {}  # (ci, si) -> metrics
    running = {}  # proc -> (ci, si, logpath, logfile, deadline)
    qi = 0
    tmp = os.path.join(ROOT, "out_tune")
    while qi < len(jobs) or running:
        while qi < len(jobs) and len(running) < MAXPAR:
            ci, si, vec, seed = jobs[qi]; qi += 1
            env = dict(os.environ); env.update(FIXED)
            for (name, *_), v in zip(SPACE, vec):
                env[name] = f"{v:.6g}"
            outdir = f"{tmp}_{ci}_{si}"
            os.makedirs(outdir, exist_ok=True)
            logpath = os.path.join(outdir, "stdout.log")
            logf = open(logpath, "w")
            cmd = [BIN, "train", str(ITERS), "--seed", str(seed),
                   "--final-games", str(FINALG), "--out-dir", outdir]
            try:
                proc = subprocess.Popen(cmd, env=env, cwd=ROOT,
                                        stdout=logf, stderr=subprocess.STDOUT, text=True)
            except OSError as exc:
                logf.close()
                print(f"warning: candidate {ci}/{si} failed to start: {exc}", file=sys.stderr)
                results[(ci, si)] = None
                continue
            running[proc] = (ci, si, logpath, logf, time.time() + TIMEOUT)
        time.sleep(1.0)
        for proc in list(running):
            ci, si, logpath, logf, deadline = running[proc]
            done = proc.poll() is not None
            if not done and time.time() > deadline:
                proc.kill(); proc.wait(); done = True  # overran -> parse() of partial log likely None
                print(f"warning: candidate {ci}/{si} timed out after {TIMEOUT}s", file=sys.stderr)
            if done:
                running.pop(proc)
                logf.close()
                try:
                    with open(logpath) as fh:
                        out = fh.read()
                except OSError:
                    out = ""
                if proc.returncode != 0:
                    tail = "\n".join(out.splitlines()[-20:])
                    print(
                        f"warning: candidate {ci}/{si} exited {proc.returncode}: {tail}",
                        file=sys.stderr,
                    )
                    results[(ci, si)] = None
                else:
                    results[(ci, si)] = parse(out)
    if not any(v is not None for v in results.values()):
        sys.exit("NO candidate produced a parseable FINAL line this generation — "
                 "the trainer output format likely changed (check FINAL_RE) or every "
                 "run crashed/timed out. Aborting rather than ranking pure noise.")
    # reduce per-candidate over seeds
    fits = []
    for ci in range(len(vectors)):
        gds = []
        for si in range(len(SEEDS)):
            m = results.get((ci, si))
            if m is None:
                gds.append(-10.0)
            else:
                gds.append(m["goal_diff"] if gates_ok(m) else m["goal_diff"] - 6.0)
        mean = sum(gds) / len(gds)
        spread = (max(gds) - min(gds)) if len(gds) > 1 else 0.0
        fits.append((mean - 0.5 * spread, gds, results.get((ci, 0))))
    return fits


def main():
    if not os.path.exists(BIN):
        sys.exit(f"build first: cargo build --release  (missing {BIN})")
    print(f"tuning {len(SPACE)} weights | iters={ITERS} pop={POP} elite={ELITE} "
          f"gens={GENS} seeds={SEEDS} maxpar={MAXPAR}")
    mean = norm([d for (_, d, *_ ) in SPACE])
    sigma = [SIGMA0] * len(SPACE)
    best = None
    with open(LOG, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gen", "cand", "fitness"] + [s[0] for s in SPACE]
                   + ["goal_diff", "winrate", "passes", "spacing", "bunch", "trainer_gates"])
    for gen in range(GENS):
        # sample POP candidates ~ N(mean, sigma), clipped to [0,1]
        cands_n = []
        cands_n.append(list(mean))  # keep the current mean (elitism)
        for _ in range(POP - 1):
            cands_n.append([min(1.0, max(0.0, mean[i] + sigma[i] * RNG.gauss(0, 1)))
                            for i in range(len(SPACE))])
        cands = [denorm(n) for n in cands_n]
        fits = eval_pool(cands)
        ranked = sorted(range(len(cands)), key=lambda i: fits[i][0], reverse=True)
        with open(LOG, "a", newline="") as f:
            w = csv.writer(f)
            for i in range(len(cands)):
                fit, gds, m = fits[i]
                row = [gen, i, f"{fit:.3f}"] + [f"{v:.5g}" for v in cands[i]]
                if m:
                    row += [m["goal_diff"], m["winrate"], m["passes"], m["spacing"],
                            m["bunch"], m["trainer_gates"]]
                w.writerow(row)
        # update mean/sigma from the elite
        elite = ranked[:ELITE]
        new_mean = [sum(cands_n[i][d] for i in elite) / ELITE for d in range(len(SPACE))]
        new_sigma = []
        for d in range(len(SPACE)):
            mu = new_mean[d]
            var = sum((cands_n[i][d] - mu) ** 2 for i in elite) / ELITE
            new_sigma.append(max(0.03, min(0.4, math.sqrt(var) * 1.3)))
        mean, sigma = new_mean, new_sigma
        bi = ranked[0]
        bfit = fits[bi][0]
        if best is None or bfit > best[0]:
            best = (bfit, cands[bi], fits[bi][2])
        bm = fits[bi][2] or {}
        print(f"gen {gen}: best_fit={bfit:+.3f} "
              f"gd={bm.get('goal_diff')} wr={bm.get('winrate')} sp={bm.get('spacing')} "
              f"| weights=" + ", ".join(f"{s[0]}={v:.4g}" for s, v in zip(SPACE, cands[bi])))
        sys.stdout.flush()
    print("\n=== BEST ===")
    print(f"fitness={best[0]:+.3f}")
    for s, v in zip(SPACE, best[1]):
        print(f"  {s[0]}={v:.5g}")
    print("\nConfirm at full length + multi-seed before shipping, e.g.:")
    print("  " + " ".join(f"{s[0]}={v:.5g}" for s, v in zip(SPACE, best[1]))
          + f" SPEED_WARMUP={FIXED['SPEED_WARMUP']} ./target/release/fiveaside train 1000")


if __name__ == "__main__":
    main()
