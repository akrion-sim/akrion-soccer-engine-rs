#!/usr/bin/env python3
"""Fit a learned EPV (expected possession value) grid from possession-chain JSONL.

Step-2 of docs/how-to-climb-codex-conversation.md. Consumes the output of
`soccer_export_possession_chains` and fits Phi_epv(cell) = E[discounted terminal outcome value
| possession state], the potential that will replace the hardcoded xT seed in pitch_value.rs
(on-ball PBRS) and the territorial-delta reward in the support/run heads (off-ball).

Per-row target:  y = gamma**ticks_to_terminal * value(terminal_outcome)
Grid:            (forward, lateral) binned to ROWS x COLS (default 16x10, matching the
                 pitch_value grid). Phi_epv[cell] = weighted mean of y over rows in that cell.

This is a calibratable scaffold, NOT the final model: the outcome values and gamma are the
knobs, and a smooth/logistic fit can replace the cell-mean once the data volume justifies it.

Usage: scripts/fit_epv_grid.py <chains.jsonl> [out_grid.json] [--gamma 0.997] [--rows 16] [--cols 10]
"""
import json
import sys
import math
from collections import defaultdict

# Terminal-outcome values. Anchored so a goal = 1.0; the rest are pre-shot / possession values
# relative to it. These are the primary calibration knob (Codex: EPV must reward genuine threat,
# not territory). timeout = neutral (possession neither converted nor lost).
OUTCOME_VALUE = {
    "goal": 1.0,
    "shot_on_target": 0.30,
    "shot": 0.10,
    "turnover": -0.15,
    "timeout": 0.0,
}


def parse_args(argv):
    if len(argv) < 2:
        sys.exit(__doc__)
    src = argv[1]
    out = argv[2] if len(argv) > 2 and not argv[2].startswith("--") else "epv_grid.json"
    gamma, rows, cols = 0.997, 16, 10
    for i, a in enumerate(argv):
        if a == "--gamma":
            gamma = float(argv[i + 1])
        elif a == "--rows":
            rows = int(argv[i + 1])
        elif a == "--cols":
            cols = int(argv[i + 1])
    return src, out, gamma, rows, cols


def main():
    src, out, gamma, rows, cols = parse_args(sys.argv)
    # accumulate sum(y) and count per cell
    sums = defaultdict(float)
    counts = defaultdict(int)
    n = 0
    unknown = 0
    y_by_outcome = defaultdict(list)
    for line in open(src):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        oc = r["terminal_outcome"]
        if oc not in OUTCOME_VALUE:
            unknown += 1
            continue
        y = (gamma ** r["ticks_to_terminal"]) * OUTCOME_VALUE[oc]
        fr = min(max(r["forward"], 0.0), 0.999999)
        la = min(max(r["lateral"], 0.0), 0.999999)
        cell = (int(fr * rows), int(la * cols))
        sums[cell] += y
        counts[cell] += 1
        y_by_outcome[oc].append(y)
        n += 1

    grid = [[0.0 for _ in range(cols)] for _ in range(rows)]
    cnt = [[0 for _ in range(cols)] for _ in range(rows)]
    for (rr, cc), s in sums.items():
        grid[rr][cc] = s / counts[(rr, cc)]
        cnt[rr][cc] = counts[(rr, cc)]

    payload = {
        "kind": "soccer_epv_grid",
        "rows": rows,
        "cols": cols,
        "gamma": gamma,
        "outcome_value": OUTCOME_VALUE,
        "n_rows_fit": n,
        "grid": grid,      # grid[forward_bin][lateral_bin] = Phi_epv
        "counts": cnt,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=1)

    # --- sanity report ---
    print(f"fit {n} rows ({unknown} unknown-outcome skipped) -> {out}  gamma={gamma} grid={rows}x{cols}")
    print("mean discounted target by outcome (should be monotone goal>SOT>shot>timeout>turnover):")
    for oc in ["goal", "shot_on_target", "shot", "timeout", "turnover"]:
        v = y_by_outcome.get(oc, [])
        if v:
            print(f"  {oc:15s}: n={len(v):6d} mean_y={sum(v)/len(v):+.4f}")
    # column-averaged Phi_epv by forward band — the key EPV signal (rises toward opponent goal)
    print("Phi_epv averaged across width, by forward band (own half -> attacking third):")
    for rr in range(rows):
        band = [grid[rr][cc] for cc in range(cols) if cnt[rr][cc] > 0]
        c = sum(cnt[rr][cc] for cc in range(cols))
        if band:
            lo, hi = rr / rows, (rr + 1) / rows
            bar = "#" * max(0, int(round((sum(band) / len(band)) * 40)))
            print(f"  fwd[{lo:.2f}-{hi:.2f}] Phi={sum(band)/len(band):+.4f} n={c:5d} {bar}")


if __name__ == "__main__":
    main()
