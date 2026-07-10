#!/usr/bin/env python3
"""Inject training traces + learning-curve data into the HTML template.

Produces a fully self-contained viz/demo.html (no external assets), suitable to
publish as an Artifact. Zero third-party deps — stdlib only."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out")


def rd(name):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}; run `cargo run --release -- train` first")
    with open(path) as f:
        return f.read()


def num(x):
    return float(x) if x not in ("", None) else None


def fmt_diff(v):
    s = f"{v:+.2f}"
    return s.replace("-", "−")  # unicode minus for the big scoreboard


def main():
    manifest = json.loads(rd("run_manifest.json"))
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
    best_iter = int(manifest["selection"]["best_iter"])
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

    # ---- headline meta from manifest, not from stale console logs ----
    untrained = manifest["untrained"]
    selection = manifest["selection"]
    cfg = manifest["config"]
    before_diff = float(untrained["goal_diff"])
    before_wr = float(untrained["winrate"])
    before_a = float(untrained["goals_a"])
    before_b = float(untrained["goals_b"])
    after_diff = float(manifest["final"]["goal_diff"])
    after_wr = float(manifest["final"]["winrate"])
    after_a = float(manifest["final"]["goals_a"])
    after_b = float(manifest["final"]["goals_b"])

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
        "minutes": f"seed {cfg['seed']}",
        "git": manifest.get("git_commit", "unknown"),
        "gate": "cleared" if selection.get("best_cleared_hardening_gates") else "not cleared",
        "display_seed": selection.get("display_seed"),
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
