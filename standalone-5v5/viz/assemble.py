#!/usr/bin/env python3
"""Inject training traces + learning-curve data into the HTML template.

Produces a fully self-contained viz/demo.html (no external assets), suitable to
publish as an Artifact. Zero third-party deps — stdlib only."""
import json, re, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out")


def rd(name):
    with open(os.path.join(OUT, name)) as f:
        return f.read()


def num(x):
    return float(x) if x not in ("", None) else None


def fmt_diff(v):
    s = f"{v:+.2f}"
    return s.replace("-", "−")  # unicode minus for the big scoreboard


def main():
    before = rd("match_before.json").strip()
    after = rd("match_after.json").strip()

    # ---- learning curve + full analytics (header-driven) ----
    lines = rd("learning_curve.csv").strip().splitlines()
    header = lines[0].split(",")
    def to_dict(r):
        c = r.split(",")
        d = {}
        for k, name in enumerate(header):
            d[name] = int(c[0]) if k == 0 else num(c[k])
        return d
    data = [to_dict(r) for r in lines[1:]]
    log_txt = rd("../out_train.log") if os.path.exists(os.path.join(OUT, "..", "out_train.log")) else ""
    mb = re.search(r"best policy at iter (\d+)", log_txt)
    best_iter = int(mb.group(1)) if mb else data[-1]["iter"]
    # last row is the appended FINAL 300-game stat line; the rest are per-iter
    final = data[-1]
    per_iter = data[:-1] if len(data) > 1 else data
    g = lambda d, k: d.get(k) if d.get(k) is not None else 0.0
    curve = [{
        "iter": d["iter"], "diff": g(d, "avg_goal_diff"), "wr": g(d, "winrate"),
        "ga": g(d, "goals_a"), "gb": g(d, "goals_b"),
        "passcmp": g(d, "pass_cmp"), "passcomp": g(d, "pass_completion"),
        "fwd": g(d, "pass_fwd"), "lat": g(d, "pass_lat"), "back": g(d, "pass_back"),
        "shots": g(d, "shots"), "conv": g(d, "conversion"), "poss": g(d, "possession"),
        "bunch": g(d, "bunch"), "spacing": g(d, "spacing"),
        "turnovers": g(d, "turnovers"), "won": g(d, "balls_won"),
    } for d in per_iter if d["iter"] <= best_iter]

    # ---- headline meta, parsed from the training log ----
    log = rd("../out_train.log") if os.path.exists(os.path.join(OUT, "..", "out_train.log")) else ""
    def grab(pat, default=None):
        m = re.search(pat, log)
        return m if m else default

    m_unt = grab(r"untrained-vs-scripted:\s*goal_diff=([-+\d.]+)\s+winrate=([\d.]+)\s+\(A ([\d.]+) / B ([\d.]+)\)")
    m_fin = grab(r"FINAL \(\d+ games\): goal_diff=([-+\d.]+)\s+winrate=([\d.]+)\s+goals ([\d.]+)-([\d.]+)(?:\s+passes/game ([\d.]+))?(?:\s+spacing=([\d.]+))?(?:\s+bunch=([\d.]+)%)?")
    m_seed = grab(r"display seed \d+: before (\d+)-(\d+)\s+->\s+after (\d+)-(\d+)")

    before_diff = float(m_unt.group(1)) if m_unt else curve[0]["diff"]
    before_wr = float(m_unt.group(2)) if m_unt else curve[0]["wr"]
    before_a = float(m_unt.group(3)) if m_unt else curve[0]["ga"]
    before_b = float(m_unt.group(4)) if m_unt else curve[0]["gb"]
    after_diff = float(m_fin.group(1)) if m_fin else curve[-1]["diff"]
    after_wr = float(m_fin.group(2)) if m_fin else curve[-1]["wr"]
    after_a = float(m_fin.group(3)) if m_fin else curve[-1]["ga"]
    after_b = float(m_fin.group(4)) if m_fin else curve[-1]["gb"]

    # Prefer the accurate FINAL 300-game CSV row for all trained-model stats.
    fg = lambda k, d=0.0: (final.get(k) if final.get(k) is not None else d)
    passes = fg("pass_cmp")
    spacing = fg("spacing")
    bunch = fg("bunch") * 100.0
    pa = fg("pass_att")
    pct = lambda x: (100.0 * x / pa) if pa > 0 else 0.0
    iters = best_iter
    meta = {
        "before_diff": fmt_diff(before_diff),
        "before_wr": before_wr,
        "before_score": f"{before_a:.1f}–{before_b:.1f} goals",
        "after_diff": fmt_diff(fg("avg_goal_diff", after_diff)),
        "after_wr": fg("winrate", after_wr),
        "after_score": f"{fg('goals_a', after_a):.1f}–{fg('goals_b', after_b):.1f} · {passes:.0f} passes/gm",
        "swing": fmt_diff(fg("avg_goal_diff", after_diff) - before_diff),
        "passes": round(passes, 1),
        "spacing": round(spacing, 1),
        "bunch": round(bunch, 0),
        # richer analytics (trained model, 300-game average)
        "pass_att": round(pa, 1),
        "pass_completion": round(fg("pass_completion") * 100.0, 0),
        "pass_fwd_pct": round(pct(fg("pass_fwd")), 0),
        "pass_lat_pct": round(pct(fg("pass_lat")), 0),
        "pass_back_pct": round(pct(fg("pass_back")), 0),
        "shots": round(fg("shots"), 1),
        "shots_scored": round(fg("shots_scored"), 1),
        "conversion": round(fg("conversion") * 100.0, 0),
        "possession": round(fg("possession") * 100.0, 0),
        "turnovers": round(fg("turnovers"), 1),
        "balls_won": round(fg("balls_won"), 1),
        "iters": iters,
        "minutes": "~6 min on a laptop",
    }

    tpl = rd_tpl()
    tpl = tpl.replace("/*__BEFORE__*/ null", before)
    tpl = tpl.replace("/*__AFTER__*/ null", after)
    tpl = tpl.replace("/*__CURVE__*/ null", json.dumps(curve))
    tpl = tpl.replace("/*__META__*/ null", json.dumps(meta, ensure_ascii=False))

    out_path = os.path.join(HERE, "index.html")
    with open(out_path, "w") as f:
        f.write(tpl)
    print("wrote", out_path, f"({len(tpl)//1024} KB)")
    print("meta:", json.dumps(meta, ensure_ascii=False))


def rd_tpl():
    with open(os.path.join(HERE, "template.html")) as f:
        return f.read()


if __name__ == "__main__":
    main()
