#!/usr/bin/env python3
"""Inject training traces + learning-curve data into the HTML template.

Produces a fully self-contained viz/demo.html (no external assets), suitable to
publish as an Artifact. Zero third-party deps — stdlib only."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.environ.get("FIVEASIDE_OUT", os.path.join(HERE, "..", "out"))


def out_file(name):
    return os.path.join(OUT, name)


def rd(name):
    path = out_file(name)
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}; run `cargo run --release -- train` first")
    with open(path, encoding="utf-8") as f:
        return f.read()


def rd_path(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def fnv1a64(data):
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def artifact_candidates(raw):
    path = os.path.expanduser(raw)
    if os.path.isabs(path):
        yield path
        return
    yield os.path.abspath(path)
    yield os.path.join(ROOT, path)
    yield os.path.join(OUT, os.path.basename(path))


def resolve_artifact(raw):
    seen = set()
    for path in artifact_candidates(raw):
        path = os.path.abspath(path)
        if path in seen:
            continue
        seen.add(path)
        if os.path.exists(path):
            return path
    raise SystemExit(f"manifest artifact missing: {raw}")


def verify_manifest_artifacts(manifest):
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise SystemExit("run_manifest.json has no artifact checksum list")
    verified = {}
    for artifact in artifacts:
        raw = artifact.get("path")
        if not raw:
            raise SystemExit("manifest artifact is missing a path")
        path = resolve_artifact(raw)
        with open(path, "rb") as f:
            data = f.read()
        try:
            expected_bytes = int(artifact["bytes"])
        except (KeyError, TypeError, ValueError):
            raise SystemExit(f"manifest artifact has invalid byte count: {raw}")
        if expected_bytes != len(data):
            raise SystemExit(
                f"artifact byte mismatch for {raw}: manifest {expected_bytes}, actual {len(data)}"
            )
        expected_hash = artifact.get("fnv1a64")
        if not isinstance(expected_hash, str):
            raise SystemExit(f"manifest artifact has invalid checksum: {raw}")
        actual_hash = fnv1a64(data)
        if expected_hash != actual_hash:
            raise SystemExit(
                f"artifact checksum mismatch for {raw}: manifest {expected_hash}, actual {actual_hash}"
            )
        verified[os.path.basename(path)] = path
    return verified


def required_artifact(verified, name):
    path = verified.get(name)
    if not path:
        raise SystemExit(f"run_manifest.json does not list required artifact {name}")
    return path


def num(x):
    return float(x) if x not in ("", None) else None


def fmt_diff(v):
    s = f"{v:+.2f}"
    return s.replace("-", "−")  # unicode minus for the big scoreboard


def main():
    manifest = json.loads(rd("run_manifest.json"))
    verified = verify_manifest_artifacts(manifest)
    before = rd_path(required_artifact(verified, "match_before.json")).strip()
    after = rd_path(required_artifact(verified, "match_after.json")).strip()

    # ---- learning curve + full analytics (header-driven) ----
    curve_text = rd_path(required_artifact(verified, "learning_curve.csv")).strip()
    if not curve_text:
        raise SystemExit("learning_curve.csv is empty")
    lines = curve_text.splitlines()
    header = lines[0].split(",")
    required_cols = {"iter", "avg_goal_diff", "winrate", "goals_a", "goals_b"}
    missing = sorted(required_cols.difference(header))
    if missing:
        raise SystemExit(f"learning_curve.csv missing columns: {', '.join(missing)}")

    def to_dict(r, line_no):
        c = r.split(",")
        if len(c) != len(header):
            raise SystemExit(
                f"learning_curve.csv row {line_no} has {len(c)} fields, expected {len(header)}"
            )
        d = {}
        for k, name in enumerate(header):
            d[name] = int(c[0]) if k == 0 else num(c[k])
        return d

    data = [to_dict(r, i + 2) for i, r in enumerate(lines[1:])]
    if not data:
        raise SystemExit("learning_curve.csv has no data rows")
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
    if not curve:
        raise SystemExit(
            f"learning_curve.csv has no rows at or before manifest best_iter={best_iter}"
        )

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
        "display_matched": bool(selection.get("display_seed_matched_filter", True)),
    }

    tpl = rd_tpl()
    tpl = tpl.replace("/*__BEFORE__*/ null", before)
    tpl = tpl.replace("/*__AFTER__*/ null", after)
    tpl = tpl.replace("/*__CURVE__*/ null", json.dumps(curve))
    tpl = tpl.replace("/*__META__*/ null", json.dumps(meta, ensure_ascii=False))

    out_path = os.path.join(HERE, "index.html")
    tmp_path = f"{out_path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(tpl)
    os.replace(tmp_path, out_path)
    print("wrote", out_path, f"({len(tpl)//1024} KB)")
    print("meta:", json.dumps(meta, ensure_ascii=False))


def rd_tpl():
    with open(os.path.join(HERE, "template.html"), encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    main()
