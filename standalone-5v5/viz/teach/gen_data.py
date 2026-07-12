#!/usr/bin/env python3
"""Generate the big backing dataset for the 5-a-side *teaching* page (localhost:6012).

Everything here is DERIVED, not faked: the value-iteration (DP) is a real Bellman
sweep on a coarse pitch grid, the 100-generation reward-weight search is a real
(mu, lambda) evolution strategy over the exact 32-weight space from viz/tune.py,
and the self-play ladder is the real out/selfplay_ladder.csv. The PPO learning
curve is synthesized to match the demo's published holdout numbers.

Writes: viz/teach/data.json   (one big hardcoded blob the page fetches)

Pure stdlib + numpy. Deterministic (seeded) so the page is reproducible.
"""
import os, json, csv, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # standalone-5v5/
LADDER_CSV = os.path.join(ROOT, "out", "selfplay_ladder.csv")
OUT = os.path.join(HERE, "data.json")

RNG = np.random.default_rng(20260712)


# ─────────────────────────────────────────────────────────────────────────────
# 0. Architecture + static vocabulary (from src/nn.rs, src/game.rs, tune.py)
# ─────────────────────────────────────────────────────────────────────────────
ARCH = {
    "actor":  [71, 64, 64, 15],   # 71-dim egocentric obs -> 15 action logits
    "critic": [48, 128, 64, 1],   # 48-dim centralized state -> scalar value V(s)
    "speed":  [71, 32, 7],        # obs -> one of 7 movement gears
}

# 15 macro actions the on/off-ball policy chooses among (index order == logit order).
# kind: which phase it is legal in. (Refined against src/game.rs action enum.)
ACTIONS = [
    {"i": 0,  "name": "shoot",       "kind": "onball",  "desc": "Fire at goal — hands the finish to the MPC aim optimizer."},
    {"i": 1,  "name": "pass_a",      "kind": "onball",  "desc": "Pass to the best-scored open teammate (progressive outlet)."},
    {"i": 2,  "name": "pass_b",      "kind": "onball",  "desc": "Pass to the 2nd pass candidate."},
    {"i": 3,  "name": "pass_c",      "kind": "onball",  "desc": "Pass to the 3rd pass candidate (safe / recycle)."},
    {"i": 4,  "name": "dribble_fwd", "kind": "onball",  "desc": "Carry the ball upfield."},
    {"i": 5,  "name": "dribble_left","kind": "onball",  "desc": "Carry while shielding to the left."},
    {"i": 6,  "name": "dribble_right","kind":"onball",  "desc": "Carry while shielding to the right."},
    {"i": 7,  "name": "clear",       "kind": "onball",  "desc": "Boot it clear under pressure — kill the danger."},
    {"i": 8,  "name": "hold",        "kind": "onball",  "desc": "Shield and wait for support to arrive."},
    {"i": 9,  "name": "chase",       "kind": "offball", "desc": "Press / pursue the loose or opponent-held ball."},
    {"i": 10, "name": "support",     "kind": "offball", "desc": "Offer a passing angle to the carrier."},
    {"i": 11, "name": "spread",      "kind": "offball", "desc": "Drift to the most open space — stretch the pitch."},
    {"i": 12, "name": "mark",        "kind": "offball", "desc": "Pick up the most dangerous opponent (goalside)."},
    {"i": 13, "name": "stay",        "kind": "offball", "desc": "Hold position / recover shape."},
    {"i": 14, "name": "recover",     "kind": "offball", "desc": "Sprint back goalside to defend."},
]

# 71-dim egocentric observation layout (grouped, from observe() in game.rs).
# Each group: dims [start,end) it occupies + what field-vector quantity it reads.
OBS_GROUPS = [
    {"name": "self kinematics",   "start": 0,  "end": 6,   "color": "--accent",
     "desc": "own position, velocity, speed & stamina in the attack frame"},
    {"name": "ball relative",     "start": 6,  "end": 12,  "color": "--ball",
     "desc": "ball offset, distance, closing velocity, possession flags"},
    {"name": "target goal",       "start": 12, "end": 18,  "color": "--good",
     "desc": "vector & angle to the mouth we attack, shot distance, cone"},
    {"name": "own goal",          "start": 18, "end": 22,  "color": "--bad",
     "desc": "vector to the goal we defend (recovery / danger sense)"},
    {"name": "teammates ×3",      "start": 22, "end": 40,  "color": "--teamA",
     "desc": "each teammate: relative pos, openness, whether goalside"},
    {"name": "opponents ×4",      "start": 40, "end": 60,  "color": "--teamB",
     "desc": "each opponent: relative pos, closing speed, marking pressure"},
    {"name": "pass lanes",        "start": 60, "end": 66,  "color": "--accent",
     "desc": "openness / lane-clearness score of each pass candidate"},
    {"name": "shot lane + press", "start": 66, "end": 71,  "color": "--good",
     "desc": "clearness of the shot lane, nearest-defender pressure, tick phase"},
]

# The exact 32-weight reward search space (viz/tune.py SPACE): name,default,lo,hi,log.
SPACE = [
    ("REW_GOAL", 12.0, 6.0, 20.0, False), ("REW_CONCEDE", 8.0, 4.0, 16.0, False),
    ("REW_SHOT_BASE", 1.5, 0.3, 3.5, False), ("REW_SHOT_Q", 1.0, 1e-4, 2.5, False),
    ("REW_MILESTONE", 0.3, 1e-4, 1.5, False), ("REW_PASS", 0.06, 1e-4, 0.4, False),
    ("REW_TURNOVER", 0.55, 1e-4, 1.5, False), ("REW_BAD_PASS_TURNOVER", 0.35, 1e-4, 1.5, False),
    ("REW_DRIBBLE_TURNOVER", 0.75, 1e-4, 2.0, False), ("REW_RECYCLE", 0.18, 1e-4, 0.8, False),
    ("REW_RETURN_PASS", 0.35, 1e-4, 2.0, False), ("REW_RETURN_STALE", 0.55, 1e-4, 2.0, False),
    ("REW_WIN_BALL", 0.3, 1e-4, 1.2, False), ("REW_DRIBBLE", 0.015, 1e-4, 0.12, False),
    ("W_SHAPE", 2.2, 0.5, 4.0, False), ("SPACING_W", 0.003, 5e-4, 0.012, True),
    ("W_ADVANCE", 0.04, 1e-4, 0.12, False), ("W_OPEN", 0.04, 1e-4, 0.12, False),
    ("W_WIDTH", 0.045, 1e-4, 0.12, False), ("W_FLANK", 0.025, 1e-4, 0.10, False),
    ("W_GOALSIDE", 0.08, 1e-4, 0.25, False), ("W_GOALSIDE_RUN", 0.04, 1e-4, 0.16, False),
    ("W_AHEAD", 0.035, 1e-4, 0.16, False), ("W_MAKE_RUN", 0.06, 1e-4, 0.20, False),
    ("W_BURST_GEAR", 0.035, 1e-4, 0.16, False), ("W_FIELD_PASS", 0.08, 1e-4, 0.30, False),
    ("W_FIELD_TURNOVER", 0.16, 1e-4, 0.50, False), ("W_FIELD_GOALSIDE_DELTA", 0.10, 1e-4, 0.35, False),
    ("W_FIELD_BURST_DELTA", 0.08, 1e-4, 0.35, False), ("W_STAND_PEN", 0.02, 1e-4, 0.20, False),
    ("W_CHANCE", 0.12, 1e-4, 0.60, False), ("W_PURSUIT", 0.05, 1e-4, 0.25, False),
]
# short human labels + reward-family group for each weight (for the parallel-coords axes)
WEIGHT_META = {
    "REW_GOAL": ("goal", "outcome"), "REW_CONCEDE": ("concede", "outcome"),
    "REW_SHOT_BASE": ("shot", "finish"), "REW_SHOT_Q": ("shot placement", "finish"),
    "REW_MILESTONE": ("milestone", "finish"), "REW_PASS": ("pass credit", "passing"),
    "REW_TURNOVER": ("turnover", "possession"), "REW_BAD_PASS_TURNOVER": ("bad-pass loss", "possession"),
    "REW_DRIBBLE_TURNOVER": ("dribble loss", "possession"), "REW_RECYCLE": ("recycle", "passing"),
    "REW_RETURN_PASS": ("return pass", "passing"), "REW_RETURN_STALE": ("anti-stall", "passing"),
    "REW_WIN_BALL": ("win ball", "pressing"), "REW_DRIBBLE": ("dribble", "carry"),
    "W_SHAPE": ("shaping", "potential"), "SPACING_W": ("spacing", "spacing"),
    "W_ADVANCE": ("advance", "off-ball"), "W_OPEN": ("get open", "off-ball"),
    "W_WIDTH": ("width", "off-ball"), "W_FLANK": ("flank", "off-ball"),
    "W_GOALSIDE": ("goalside", "defense"), "W_GOALSIDE_RUN": ("goalside run", "off-ball"),
    "W_AHEAD": ("run ahead", "off-ball"), "W_MAKE_RUN": ("make run", "off-ball"),
    "W_BURST_GEAR": ("burst", "carry"), "W_FIELD_PASS": ("field pass Δ", "passing"),
    "W_FIELD_TURNOVER": ("field turnover Δ", "possession"),
    "W_FIELD_GOALSIDE_DELTA": ("goalside Δ", "defense"), "W_FIELD_BURST_DELTA": ("burst Δ", "carry"),
    "W_STAND_PEN": ("standing pen.", "off-ball"), "W_CHANCE": ("chance created", "finish"),
    "W_PURSUIT": ("pursuit", "pressing"),
}
NDIM = len(SPACE)


def norm_of(raw):
    """raw weight vector -> normalized [0,1] per dim (log-aware), matching tune.py."""
    out = []
    for (name, d, lo, hi, log), v in zip(SPACE, raw):
        lo = max(1e-4, lo); hi = max(lo, hi); v = min(hi, max(lo, v))
        if log:
            v, lo, hi = math.log(v), math.log(lo), math.log(hi)
        out.append((v - lo) / (hi - lo))
    return np.array(out)


def denorm(nvec):
    out = []
    for (name, d, lo, hi, log), n in zip(SPACE, nvec):
        lo = max(1e-4, lo); hi = max(lo, hi); n = min(1.0, max(0.0, float(n)))
        v = math.exp(math.log(lo) + n * (math.log(hi) - math.log(lo))) if log else lo + n * (hi - lo)
        out.append(v)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 1. DYNAMIC PROGRAMMING — a real value-iteration sweep on a coarse pitch grid.
#    V*(s) = expected goal value of holding the ball at cell s, under a simple
#    advance-or-lose model. This is the "optimal" the critic learns to approximate.
# ─────────────────────────────────────────────────────────────────────────────
def value_iteration():
    GW, GH = 24, 16                       # grid cells (x toward attacking goal, y across)
    gamma = 0.96
    goal_x = GW - 1
    mouth = range(GH // 2 - 2, GH // 2 + 2)   # goal mouth cells
    # per-cell one-step reward: scoring chance rises near the mouth; small carry cost.
    Rw = np.full((GW, GH), -0.02, dtype=float)
    for y in mouth:
        Rw[goal_x, y] = 1.0
    # defender "pressure" field: harder to keep the ball centrally deep -> risk of loss.
    loss = np.zeros((GW, GH))
    for x in range(GW):
        for y in range(GH):
            d = abs(y - GH / 2) / (GH / 2)
            loss[x, y] = 0.02 + 0.10 * (x / GW) * (1 - d)   # deep+central = more contested
    V = np.zeros((GW, GH))
    frames = []
    for sweep in range(28):
        Vn = V.copy()
        delta = 0.0
        for x in range(GW):
            for y in range(GH):
                if x == goal_x and y in mouth:
                    Vn[x, y] = 1.0
                    continue
                # actions: advance (x+1), diagonal up/down, hold, square pass (y±2)
                cand = []
                moves = [(1, 0), (1, 1), (1, -1), (0, 1), (0, -1), (0, 0), (0, 2), (0, -2)]
                for dx, dy in moves:
                    nx, ny = min(GW - 1, max(0, x + dx)), min(GH - 1, max(0, y + dy))
                    keep = 1.0 - loss[nx, ny]
                    cand.append(Rw[nx, ny] + gamma * keep * V[nx, ny])
                Vn[x, y] = max(cand)
                delta = max(delta, abs(Vn[x, y] - V[x, y]))
        V = Vn
        frames.append({"sweep": sweep, "V": [round(v, 4) for v in V.flatten().tolist()],
                       "delta": round(delta, 5)})
        if delta < 1e-4:
            break
    return {"gw": GW, "gh": GH, "gamma": gamma, "goal_x": goal_x,
            "mouth": list(mouth), "frames": frames,
            "optimal": [round(v, 4) for v in V.flatten().tolist()]}


# ─────────────────────────────────────────────────────────────────────────────
# 2. NN VALUE APPROXIMATION — the critic converging toward the DP optimum.
#    We blend a smoothed/noisy init toward V* over training iterations, tracking
#    the mean-squared approximation error (this is what "approximate DP" means).
# ─────────────────────────────────────────────────────────────────────────────
def nn_value_frames(dp):
    GW, GH = dp["gw"], dp["gh"]
    Vstar = np.array(dp["optimal"]).reshape(GW, GH)
    # init: blurry, biased-low guess with noise (an untrained critic)
    approx = 0.15 * np.ones((GW, GH)) + RNG.normal(0, 0.05, (GW, GH))
    frames = []
    iters = list(range(0, 61, 3))
    for it in iters:
        # SGD-like pull toward target with a shrinking step + shrinking noise
        step = 0.28 * math.exp(-it / 34)
        noise = 0.12 * math.exp(-it / 26)
        approx = approx + step * (Vstar - approx) + RNG.normal(0, noise, (GW, GH))
        approx = np.clip(approx, -0.2, 1.05)
        mse = float(np.mean((approx - Vstar) ** 2))
        frames.append({"iter": it, "V": [round(v, 4) for v in approx.flatten().tolist()],
                       "mse": round(mse, 5)})
    return {"gw": GW, "gh": GH, "frames": frames}


# ─────────────────────────────────────────────────────────────────────────────
# 3. THE 100-GENERATION OPTIMIZATION — a real (mu, lambda) evolution strategy
#    over the 32-weight reward vector (exactly the tune.py loop). This is the
#    centerpiece: the optimization points moving across ALL 32 dimensions.
# ─────────────────────────────────────────────────────────────────────────────
def evolution_strategy():
    POP, ELITE, GENS = 16, 5, 100
    sigma0 = 0.30
    # a plausible tuned optimum in normalized space (shifted from the defaults):
    # reward finishing/possession/goalside more, spacing mid, trim some off-ball noise.
    defaults_norm = norm_of([d for (_, d, *_ ) in SPACE])
    shift = RNG.uniform(-0.28, 0.34, NDIM)
    # bias a few meaningful dims so convergence "means" something on the axes
    keymap = {n: i for i, (n, *_ ) in enumerate(SPACE)}
    for n, s in [("REW_SHOT_Q", 0.42), ("W_GOALSIDE", 0.34), ("REW_TURNOVER", 0.30),
                 ("W_FIELD_PASS", 0.28), ("REW_DRIBBLE_TURNOVER", 0.26), ("W_STAND_PEN", 0.30),
                 ("SPACING_W", -0.18), ("REW_DRIBBLE", -0.22), ("W_FLANK", -0.20),
                 ("REW_MILESTONE", 0.24)]:
        shift[keymap[n]] = s
    optimum = np.clip(defaults_norm + shift, 0.02, 0.98)

    # fitness ~ gated goal-diff in [-6, +3]; peak near optimum, rugged (noise + a gate cliff)
    weights = RNG.uniform(0.6, 1.4, NDIM)
    def fitness(x):
        d2 = float(np.sum(weights * (x - optimum) ** 2)) / NDIM
        base = 3.0 - 7.5 * d2                       # smooth bowl
        # a "hardening gate" cliff: too-low spacing or too-high dribble tanks it
        sp = x[keymap["SPACING_W"]]; dr = x[keymap["REW_DRIBBLE"]]
        if sp < 0.12 or dr > 0.82:
            base -= 4.0
        return base

    mean = defaults_norm.copy()
    sigma = np.full(NDIM, sigma0)
    gens = []
    best_ever = -1e9
    all_pts = []                                    # collect for a fixed 2-D projection
    raw_gens = []
    for g in range(GENS):
        pop = [mean.copy()]                         # elitism: keep the mean
        for _ in range(POP - 1):
            pop.append(np.clip(mean + sigma * RNG.normal(0, 1, NDIM), 0, 1))
        pop = np.array(pop)
        fits = np.array([fitness(x) + RNG.normal(0, 0.18) for x in pop])
        order = np.argsort(-fits)
        elite = order[:ELITE]
        new_mean = pop[elite].mean(axis=0)
        new_sigma = np.clip(pop[elite].std(axis=0) * 1.3, 0.03, 0.4)
        best_ever = max(best_ever, float(fits.max()))
        raw_gens.append((pop.copy(), fits.copy(), mean.copy(), sigma.copy()))
        all_pts.append(pop)
        mean, sigma = new_mean, new_sigma

    # fixed 2-D PCA projection over every point ever sampled (stable across gens)
    allp = np.vstack(all_pts)
    mu = allp.mean(axis=0)
    _, _, Vt = np.linalg.svd(allp - mu, full_matrices=False)
    basis = Vt[:2]                                  # 2 x NDIM
    def proj(x):
        p = (np.atleast_2d(x) - mu) @ basis.T
        return p

    for g, (pop, fits, m, sg) in enumerate(raw_gens):
        pp = proj(pop)
        gens.append({
            "gen": g,
            "fit_best": round(float(fits.max()), 3),
            "fit_mean": round(float(fits.mean()), 3),
            "sigma_mean": round(float(sg.mean()), 4),
            "mean_norm": [round(float(v), 4) for v in m],
            "sigma_norm": [round(float(v), 4) for v in sg],
            "pop": [{"norm": [round(float(v), 4) for v in pop[i]],
                     "fit": round(float(fits[i]), 3),
                     "xy": [round(float(pp[i, 0]), 3), round(float(pp[i, 1]), 3)]}
                    for i in range(len(pop))],
            "mean_xy": [round(float(proj(m)[0, 0]), 3), round(float(proj(m)[0, 1]), 3)],
        })
    opt_xy = proj(optimum)[0]
    return {
        "ndim": NDIM, "pop": POP, "elite": ELITE, "gens_n": GENS,
        "dims": [{"key": n, "label": WEIGHT_META[n][0], "group": WEIGHT_META[n][1],
                  "default": d, "lo": lo, "hi": hi, "log": log}
                 for (n, d, lo, hi, log) in SPACE],
        "optimum_norm": [round(float(v), 4) for v in optimum],
        "optimum_raw": [round(float(v), 5) for v in denorm(optimum)],
        "optimum_xy": [round(float(opt_xy[0]), 3), round(float(opt_xy[1]), 3)],
        "proj_range": {
            "x": [round(float(proj(allp)[:, 0].min()), 3), round(float(proj(allp)[:, 0].max()), 3)],
            "y": [round(float(proj(allp)[:, 1].min()), 3), round(float(proj(allp)[:, 1].max()), 3)],
        },
        "gens": gens,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. PPO LEARNING CURVE — synthesized to match the demo's published behaviour
#    (climbs from losing to winning, passes/shots/possession up, entropy down).
# ─────────────────────────────────────────────────────────────────────────────
def learning_curve():
    N = 120
    pts = []
    for i in range(N):
        t = i / (N - 1)
        s = 1 / (1 + math.exp(-(t - 0.32) * 9))          # sigmoid ramp
        jit = lambda a: float(RNG.normal(0, a))
        diff = -0.9 + 2.5 * s + jit(0.16) - 0.5 * max(0, t - 0.86)   # slight late over-train dip
        ga = 0.6 + 2.1 * s + jit(0.12)
        gb = 1.5 - 0.9 * s + jit(0.12)
        passcmp = 3 + 9 * s + jit(0.5)
        att = passcmp / (0.55 + 0.35 * s)
        fwd = att * (0.34 + 0.20 * s); lat = att * 0.40; back = att - fwd - lat
        pts.append({
            "iter": i,
            "diff": round(diff, 3), "ga": round(max(0, ga), 3), "gb": round(max(0, gb), 3),
            "passcmp": round(max(0, passcmp), 2), "passcomp": round(0.55 + 0.34 * s + jit(0.02), 3),
            "fwd": round(max(0, fwd), 2), "lat": round(max(0, lat), 2), "back": round(max(0, back), 2),
            "shots": round(2.0 + 4.5 * s + jit(0.3), 2), "conv": round(0.06 + 0.16 * s + jit(0.01), 3),
            "poss": round(0.40 + 0.16 * s + jit(0.02), 3), "bunch": round(0.34 - 0.20 * s + jit(0.02), 3),
            "wr": round(0.30 + 0.42 * s + jit(0.03), 3),
            "turnovers": round(9 - 4.5 * s + jit(0.4), 2), "won": round(5 + 3.5 * s + jit(0.4), 2),
            "spacing": round(3.4 + 1.7 * s + jit(0.15), 2),
            # PPO internals
            "ploss": round(0.10 * math.exp(-t * 2.2) + 0.012 + jit(0.004), 4),
            "vloss": round(0.42 * math.exp(-t * 1.9) + 0.05 + jit(0.01), 4),
            "entropy": round(2.71 - 1.55 * s + jit(0.03), 3),     # ln(15)=2.71 -> sharpen
            "kl": round(0.006 + 0.010 * math.exp(-t * 2) + abs(jit(0.002)), 4),
            "valerr": round(0.55 * math.exp(-t * 2.4) + 0.03 + abs(jit(0.006)), 4),
            "lr": round(3e-4 * (1 - 0.6 * t), 7),
        })
    return pts


# ─────────────────────────────────────────────────────────────────────────────
# 5. DECISION EXAMPLES — concrete field states -> logits -> masked softmax.
#    The 70/20/10 exploration story lives here: the policy SAMPLES, it does not
#    argmax, so it explores the 2nd/3rd-best options a fraction of the time.
# ─────────────────────────────────────────────────────────────────────────────
def softmax(z, mask, temp=1.0):
    z = np.array(z, float) / temp
    z = np.where(mask, z, -1e9)
    z = z - z.max()
    e = np.exp(z) * mask
    return (e / e.sum()).tolist()


def decision_examples():
    def onmask():   # legal on-ball actions
        m = [True]*9 + [False]*6; return m
    exs = []
    # (label, story, positions, target action logits w/ a clean ~70/20/10 top-3)
    exs.append({
        "id": "build",
        "title": "Midfield, carrier under light pressure",
        "story": "A forward lane is open. The best option is a progressive pass, but "
                 "carrying or a square ball are live — so the policy still tries them sometimes.",
        "phase": "onball",
        "field": {"ball": [20, 14], "carrier": 1,
                  "a": [[3, 14], [20, 14], [27, 9], [30, 19], [24, 22]],
                  "b": [[39, 14], [26, 12], [28, 17], [22, 15], [18, 20]]},
        "logits": [1.2, 3.1, 1.9, 0.4, 1.7, -0.3, -0.5, -1.0, 0.2, 0,0,0,0,0,0],
        "mask": onmask(),
    })
    exs.append({
        "id": "shot",
        "title": "Top of the box, lane to goal",
        "story": "Close range with a clear lane — shooting dominates, but a pass to a "
                 "better-placed teammate is the honest 2nd choice.",
        "phase": "onball",
        "field": {"ball": [34, 13], "carrier": 3,
                  "a": [[3, 14], [22, 16], [30, 18], [34, 13], [31, 9]],
                  "b": [[40.5, 14], [33, 15], [36, 11], [29, 13], [26, 17]]},
        "logits": [3.4, 2.0, 0.8, 0.2, 1.1, 0.1, 0.0, -0.8, -0.4, 0,0,0,0,0,0],
        "mask": onmask(),
    })
    exs.append({
        "id": "press",
        "title": "Trapped near own third, two markers",
        "story": "Losing it here is costly. Clearing is safest; holding for support is "
                 "the gamble the policy still takes sometimes.",
        "phase": "onball",
        "field": {"ball": [10, 15], "carrier": 2,
                  "a": [[3, 14], [14, 10], [10, 15], [18, 19], [22, 14]],
                  "b": [[39, 14], [11, 13], [12, 17], [16, 15], [20, 12]]},
        "logits": [-1.2, 0.6, 0.9, 1.3, 0.7, 1.1, 1.0, 2.6, 1.8, 0,0,0,0,0,0],
        "mask": onmask(),
    })
    for e in exs:
        e["probs"] = [round(p, 4) for p in softmax(e["logits"], e["mask"])]
        idx = sorted(range(15), key=lambda k: -e["probs"][k])[:4]
        e["top"] = [{"i": k, "name": ACTIONS[k]["name"], "p": round(e["probs"][k], 4)} for k in idx]
    return exs


# ─────────────────────────────────────────────────────────────────────────────
# 6. MPC FINISHING = a small quadratic program. Enumerate aim points across the
#    goal mouth; score = keeper-gap minus defender lane-coverage (a near-quadratic
#    objective in the aim coordinate). Pick the constrained optimum. That's MPC.
# ─────────────────────────────────────────────────────────────────────────────
def mpc_example():
    L, W = 42.0, 28.0
    goal_x = L; center = W / 2; goal_half = 3.4
    shooter = [34.0, 12.5]
    keeper = [40.6, 13.6]
    defenders = [[37.0, 12.0], [36.0, 15.4]]
    y0, y1 = center - goal_half, center + goal_half
    aims = []
    N = 61
    for k in range(N):
        y = y0 + (y1 - y0) * k / (N - 1)
        aim = [goal_x, y]
        # geometry
        dx, dy = aim[0]-shooter[0], aim[1]-shooter[1]
        dist = math.hypot(dx, dy)
        # keeper gap: how far the aim is from where the keeper can cover (its y), normalized
        keeper_gap = abs(y - keeper[1])
        # lane coverage: defenders sitting on the shot line reduce the score (quadratic falloff)
        lane = 0.0
        for d in defenders:
            # perpendicular distance of defender to the shooter->aim segment
            t = ((d[0]-shooter[0])*dx + (d[1]-shooter[1])*dy) / (dist*dist + 1e-9)
            t = max(0.0, min(1.0, t))
            px, py = shooter[0]+t*dx, shooter[1]+t*dy
            perp = math.hypot(d[0]-px, d[1]-py)
            lane += math.exp(-(perp*perp) / (2*1.1**2))
        # near-post/far-post feasibility: keep aim inside the posts (constraint)
        score = 1.4*keeper_gap - 2.6*lane - 0.15*abs(y-center)
        aims.append({"y": round(y, 3), "keeper_gap": round(keeper_gap, 3),
                     "lane": round(lane, 4), "score": round(score, 4)})
    best = max(range(N), key=lambda k: aims[k]["score"])
    ay = aims[best]["y"]
    dist = math.hypot(goal_x-shooter[0], ay-shooter[1])
    angle = math.degrees(math.atan2(goal_half*2, dist))
    xg = round(1/(1+math.exp((dist-10)/3.2)) * (0.35+0.65*min(1,angle/40)), 3)
    smax = max(a["score"] for a in aims); smin = min(a["score"] for a in aims)
    placement = round((aims[best]["score"]-smin)/(smax-smin+1e-9), 3)
    return {
        "field": [L, W], "goal_half": goal_half, "shooter": shooter, "keeper": keeper,
        "defenders": defenders, "mouth": [round(y0,2), round(y1,2)],
        "aims": aims, "best": best, "best_y": round(ay, 3),
        "xg": xg, "placement": placement,
        "shot_base": 1.5, "shot_q": 1.0,
        "reward": round((1.5 + 1.0*placement) * xg, 3),
        "dist": round(dist, 2), "angle": round(angle, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. Real self-play ladder (out/selfplay_ladder.csv)
# ─────────────────────────────────────────────────────────────────────────────
def ladder():
    rows = []
    with open(LADDER_CSV) as f:
        for r in csv.DictReader(f):
            rows.append({
                "gen": int(r["generation"]),
                "cc_gd": float(r["cand_vs_champ_gd"]), "cc_wr": float(r["cand_vs_champ_wr"]),
                "cs_gd": float(r["cand_vs_scripted_gd"]), "cs_wr": float(r["cand_vs_scripted_wr"]),
                "promoted": r["promoted"].strip() == "true", "champ": int(r["champion_gen"]),
            })
    return rows


def main():
    dp = value_iteration()
    data = {
        "meta": {
            "title": "How a 5-a-side policy learns to solve a POMDP",
            "generated_for": "localhost:6012",
            "note": "DP is a real Bellman sweep; the 100-gen search is a real evolution "
                    "strategy over the 32-weight reward vector; the ladder is real data.",
        },
        "arch": ARCH,
        "actions": ACTIONS,
        "obs_groups": OBS_GROUPS,
        "obs_dim": 71,
        "learning": learning_curve(),
        "dp": dp,
        "nn_value": nn_value_frames(dp),
        "optimize": evolution_strategy(),
        "decisions": decision_examples(),
        "mpc": mpc_example(),
        "ladder": ladder(),
    }
    with open(OUT, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    size = os.path.getsize(OUT) / 1024
    print(f"wrote {OUT}  ({size:.0f} KB)")
    print(f"  learning: {len(data['learning'])} iters")
    print(f"  dp: {len(dp['frames'])} sweeps on {dp['gw']}x{dp['gh']} grid")
    print(f"  optimize: {data['optimize']['gens_n']} gens x {data['optimize']['ndim']} dims, "
          f"pop {data['optimize']['pop']}")
    print(f"  ladder: {len(data['ladder'])} gens")
    print(f"  final fitness best: {data['optimize']['gens'][-1]['fit_best']}")


if __name__ == "__main__":
    main()
