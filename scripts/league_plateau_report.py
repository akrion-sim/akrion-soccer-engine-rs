#!/usr/bin/env python3
"""Summarize soccer_league_train climb/plateau signatures from league.log.

The forward-pass proof work needs a cheap way to answer "are we still climbing?"
without launching another simulation. This parser reads existing league_kpi lines and
reports rolling means plus a least-squares slope for the main pass metrics.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable


KPI_RE = re.compile(r"^league_kpi\s+(?P<fields>.*)$")


def parse_fields(raw: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in raw.split():
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        fields[key] = value
    return fields


def as_float(fields: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(fields.get(key, default))
    except ValueError:
        return default


def read_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = KPI_RE.match(line.strip())
            if not match:
                continue
            fields = parse_fields(match.group("fields"))
            rows.append(
                {
                    "round": as_float(fields, "round", float(len(rows) + 1)),
                    "forward": as_float(fields, "forward_pass_margin_per_game"),
                    "net_forward": as_float(fields, "net_forward_pass_margin_per_game"),
                    "pass_completion": as_float(fields, "pass_completion"),
                    "gd": as_float(fields, "gd_per_game"),
                    "sot_for": as_float(fields, "sot_for"),
                    "sot_against": as_float(fields, "sot_against"),
                    "dribble_beats": as_float(fields, "dribble_beats"),
                    "chain_gain_yards": as_float(fields, "chain_gain_yards"),
                    "chain_net_losses": as_float(fields, "chain_net_losses"),
                }
            )
    return rows


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def slope(rows: list[dict[str, float]], key: str) -> float:
    if len(rows) < 2:
        return 0.0
    xs = [row["round"] for row in rows]
    ys = [row[key] for row in rows]
    x_mean = mean(xs)
    y_mean = mean(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom <= 1e-12:
        return 0.0
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom


def fmt(value: float, digits: int = 3) -> str:
    if math.isfinite(value):
        return f"{value:+.{digits}f}"
    return "nan"


def summarize_window(label: str, rows: list[dict[str, float]]) -> None:
    print(f"{label}: rounds {int(rows[0]['round'])}-{int(rows[-1]['round'])} n={len(rows)}")
    print(
        "  forward={forward} net_forward={net_forward} pass_completion={pass_completion:.3f} "
        "gd={gd} sot_margin={sot_margin} dribble_beats={dribble_beats:.2f} "
        "chain_gain={chain_gain:.1f} chain_losses={chain_losses:.2f}".format(
            forward=fmt(mean(row["forward"] for row in rows)),
            net_forward=fmt(mean(row["net_forward"] for row in rows)),
            pass_completion=mean(row["pass_completion"] for row in rows),
            gd=fmt(mean(row["gd"] for row in rows), 2),
            sot_margin=fmt(mean(row["sot_for"] - row["sot_against"] for row in rows), 2),
            dribble_beats=mean(row["dribble_beats"] for row in rows),
            chain_gain=mean(row["chain_gain_yards"] for row in rows),
            chain_losses=mean(row["chain_net_losses"] for row in rows),
        )
    )


def plateau_verdict(rows: list[dict[str, float]], window: int) -> str:
    tail = rows[-window:]
    net_mean = mean(row["net_forward"] for row in tail)
    forward_mean = mean(row["forward"] for row in tail)
    net_slope = slope(tail, "net_forward")
    completion_slope = slope(tail, "pass_completion")
    if net_mean > 0.25 and net_slope > 0.0:
        return "climbing: positive tail net-forward mean with positive slope"
    if net_mean <= 0.0 and forward_mean <= 0.0:
        return "regressing/negative plateau: tail forward and net-forward means are not positive"
    if abs(net_slope) < 0.03 and abs(completion_slope) < 0.002:
        return "plateau: tail slope is near flat"
    return "directional only: mixed tail, not proof of climb"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--window", type=int, default=12)
    args = parser.parse_args()

    rows = read_rows(args.log)
    if not rows:
        raise SystemExit(f"no league_kpi rows found in {args.log}")

    window = max(1, min(args.window, len(rows)))
    first = rows[:window]
    tail = rows[-window:]

    print(f"log={args.log}")
    print(f"rows={len(rows)} first_round={int(rows[0]['round'])} last_round={int(rows[-1]['round'])}")
    summarize_window("first_window", first)
    summarize_window("tail_window", tail)
    print(
        "tail_slopes_per_round: forward={forward} net_forward={net_forward} "
        "pass_completion={completion} gd={gd}".format(
            forward=fmt(slope(tail, "forward"), 4),
            net_forward=fmt(slope(tail, "net_forward"), 4),
            completion=fmt(slope(tail, "pass_completion"), 5),
            gd=fmt(slope(tail, "gd"), 4),
        )
    )
    best = max(rows, key=lambda row: row["net_forward"])
    latest = rows[-1]
    print(
        "best_net_forward_round={round} net_forward={net_forward} forward={forward} "
        "pass_completion={completion:.3f} gd={gd}".format(
            round=int(best["round"]),
            net_forward=fmt(best["net_forward"]),
            forward=fmt(best["forward"]),
            completion=best["pass_completion"],
            gd=fmt(best["gd"], 2),
        )
    )
    print(
        "latest_round={round} net_forward={net_forward} forward={forward} "
        "pass_completion={completion:.3f} gd={gd}".format(
            round=int(latest["round"]),
            net_forward=fmt(latest["net_forward"]),
            forward=fmt(latest["forward"]),
            completion=latest["pass_completion"],
            gd=fmt(latest["gd"], 2),
        )
    )
    print(f"verdict={plateau_verdict(rows, window)}")


if __name__ == "__main__":
    main()
