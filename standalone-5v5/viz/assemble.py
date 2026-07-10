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

    # ---- learning curve ----
    curve = []
    rows = rd("learning_curve.csv").strip().splitlines()[1:]
    for r in rows:
        c = r.split(",")
        curve.append({
            "iter": int(c[0]),
            "diff": num(c[1]),
            "wr": num(c[2]),
            "ga": num(c[3]),
            "gb": num(c[4]),
        })
    # Early stopping: show the curve to the best checkpoint (PPO over-trains after).
    log_txt = rd("../out_train.log") if os.path.exists(os.path.join(OUT, "..", "out_train.log")) else ""
    mb = re.search(r"best policy at iter (\d+)", log_txt)
    best_iter = int(mb.group(1)) if mb else curve[-1]["iter"]
    curve = [p for p in curve if p["iter"] <= best_iter]

    # ---- headline meta, parsed from the training log ----
    log = rd("../out_train.log") if os.path.exists(os.path.join(OUT, "..", "out_train.log")) else ""
    def grab(pat, default=None):
        m = re.search(pat, log)
        return m if m else default

    m_unt = grab(r"untrained-vs-scripted:\s*goal_diff=([-+\d.]+)\s+winrate=([\d.]+)\s+\(A ([\d.]+) / B ([\d.]+)\)")
    m_fin = grab(r"FINAL \(\d+ games\): goal_diff=([-+\d.]+)\s+winrate=([\d.]+)\s+goals ([\d.]+)-([\d.]+)(?:\s+passes/game ([\d.]+))?(?:\s+spacing=([\d.]+))?")
    m_seed = grab(r"display seed \d+: before (\d+)-(\d+)\s+->\s+after (\d+)-(\d+)")

    before_diff = float(m_unt.group(1)) if m_unt else curve[0]["diff"]
    before_wr = float(m_unt.group(2)) if m_unt else curve[0]["wr"]
    before_a = float(m_unt.group(3)) if m_unt else curve[0]["ga"]
    before_b = float(m_unt.group(4)) if m_unt else curve[0]["gb"]
    after_diff = float(m_fin.group(1)) if m_fin else curve[-1]["diff"]
    after_wr = float(m_fin.group(2)) if m_fin else curve[-1]["wr"]
    after_a = float(m_fin.group(3)) if m_fin else curve[-1]["ga"]
    after_b = float(m_fin.group(4)) if m_fin else curve[-1]["gb"]

    passes = float(m_fin.group(5)) if (m_fin and m_fin.group(5)) else 0.0
    spacing = float(m_fin.group(6)) if (m_fin and m_fin.group(6)) else 0.0
    iters = curve[-1]["iter"]
    meta = {
        "before_diff": fmt_diff(before_diff),
        "before_wr": before_wr,
        "before_score": f"{before_a:.1f}–{before_b:.1f} goals",
        "after_diff": fmt_diff(after_diff),
        "after_wr": after_wr,
        "after_score": f"{after_a:.1f}–{after_b:.1f} · {passes:.0f} passes/gm",
        "swing": fmt_diff(after_diff - before_diff),
        "passes": round(passes, 1),
        "spacing": round(spacing, 1),
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
