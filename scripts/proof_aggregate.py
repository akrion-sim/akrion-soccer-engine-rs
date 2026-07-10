#!/usr/bin/env python3
"""Aggregate soccer_proof JSONL shards into one paired statistical verdict.

The proof standard is intentionally conservative:
- normal lower bound
- deterministic bootstrap lower bound
- exact one-sided sign test

A 99.99% claim requires all three to agree. This keeps a lucky high-variance
margin from masquerading as learned passing traction.
"""

import json
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence


Z_95 = 1.959_964
Z_9999 = 3.719_016
ALPHA_95 = 0.05
ALPHA_9999 = 0.0001


def f(row: Dict[str, object], key: str) -> float:
    value = row.get(key, 0.0)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isfinite(result):
        return result
    return 0.0


def safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) > 1e-9:
        return numerator / denominator
    return 0.0


def pct(numerator: float, denominator: float) -> float:
    return 100.0 * safe_ratio(numerator, denominator)


def bootstrap_replicates() -> int:
    raw = os.environ.get("PROOF_BOOTSTRAP_REPLICATES", "20000")
    try:
        return max(1000, int(raw))
    except ValueError:
        return 20000


def bootstrap_seed() -> int:
    raw = os.environ.get("PROOF_BOOTSTRAP_SEED", "12648430")
    try:
        return int(raw, 0)
    except ValueError:
        return 12648430


def quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    q = min(max(q, 0.0), 1.0)
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def bootstrap_lower_bounds(values: Sequence[float], reps: int, seed: int) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    means: List[float] = []
    for _ in range(reps):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    return quantile(means, 0.05), quantile(means, ALPHA_9999)


def one_sided_sign_p(values: Sequence[float]) -> tuple[int, int, int, float]:
    wins = sum(1 for value in values if value > 0.0)
    losses = sum(1 for value in values if value < 0.0)
    ties = len(values) - wins - losses
    n = wins + losses
    if n == 0:
        return wins, losses, ties, 1.0
    tail = 0
    for k in range(wins, n + 1):
        tail += math.comb(n, k)
    return wins, losses, ties, tail / (2**n)


@dataclass(frozen=True)
class Stat:
    n: int
    mean: float
    normal_lb95: float
    normal_lb9999: float
    bootstrap_lb95: float
    bootstrap_lb9999: float
    sign_wins: int
    sign_losses: int
    sign_ties: int
    sign_p: float

    def support_95(self, floor: float = 0.0) -> bool:
        return (
            self.mean > floor
            and self.normal_lb95 > floor
            and self.bootstrap_lb95 > floor
            and self.sign_p <= ALPHA_95
            and self.sign_wins > self.sign_losses
        )

    def support_9999(self, floor: float = 0.0) -> bool:
        return (
            self.mean > floor
            and self.normal_lb9999 > floor
            and self.bootstrap_lb9999 > floor
            and self.sign_p <= ALPHA_9999
            and self.sign_wins > self.sign_losses
        )


def paired_stat(values: Sequence[float], seed_offset: int = 0) -> Stat:
    n = len(values)
    if n == 0:
        return Stat(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 1.0)
    mean = sum(values) / n
    if n > 1:
        var = sum((value - mean) ** 2 for value in values) / (n - 1)
    else:
        var = 0.0
    se = math.sqrt(var / n)
    reps = bootstrap_replicates()
    seed = bootstrap_seed() + seed_offset
    wins, losses, ties, sign_p = one_sided_sign_p(values)
    boot_lb95, boot_lb9999 = bootstrap_lower_bounds(values, reps, seed)
    return Stat(
        n=n,
        mean=mean,
        normal_lb95=mean - Z_95 * se,
        normal_lb9999=mean - Z_9999 * se,
        bootstrap_lb95=boot_lb95,
        bootstrap_lb9999=boot_lb9999,
        sign_wins=wins,
        sign_losses=losses,
        sign_ties=ties,
        sign_p=sign_p,
    )


def iter_rows(paths: List[str]) -> Iterable[Dict[str, object]]:
    if not paths:
        paths = ["-"]
    for path in paths:
        handle = sys.stdin if path == "-" else open(path, "r", encoding="utf-8")
        with handle:
            for line_no, line in enumerate(handle, start=1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError as exc:
                    print(f"skip malformed JSON {path}:{line_no}: {exc}", file=sys.stderr)
                    continue
                if isinstance(row, dict):
                    yield row


def add_totals(total: Dict[str, float], row: Dict[str, object], prefix: str) -> None:
    for key in (
        "goals",
        "shots",
        "sot",
        "att",
        "comp",
        "fwd",
        "back",
        "yards",
        "chains",
        "sap",
        "assist",
        "drib",
        "route1",
        "int",
    ):
        total[key] += f(row, f"{prefix}_{key}")


def fmt_metric(label: str, stat: Stat, scale: float = 1.0, unit: str = "") -> str:
    return (
        f"{label:<31} delta {stat.mean * scale:+.3f}{unit}  "
        f"normalLB95 {stat.normal_lb95 * scale:+.3f}{unit}  "
        f"normalLB99.99 {stat.normal_lb9999 * scale:+.3f}{unit}  "
        f"bootLB95 {stat.bootstrap_lb95 * scale:+.3f}{unit}  "
        f"bootLB99.99 {stat.bootstrap_lb9999 * scale:+.3f}{unit}  "
        f"sign {stat.sign_wins}-{stat.sign_losses}-{stat.sign_ties} "
        f"p={stat.sign_p:.6g}  n={stat.n}"
    )


def verdict_for(
    fwd: Stat,
    completion_rate: Stat,
    yards_per_pass: Stat,
    route_one: Stat,
) -> str:
    yards_not_collapsed = yards_per_pass.bootstrap_lb95 > -0.25
    route_one_not_rising = route_one.mean <= 0.10
    if fwd.support_9999() and completion_rate.support_9999() and yards_not_collapsed and route_one_not_rising:
        return (
            "PROVEN@99.99% - completed forward passes and pass completion rate "
            "both pass normal+bootstrap+sign-test, with yards/pass and route-one guards intact"
        )
    if fwd.support_9999() and completion_rate.mean >= 0.0 and yards_not_collapsed and route_one_not_rising:
        return (
            "FORWARD-PASS-PROVEN@99.99% - forward passing passes triple-stat proof, "
            "but pass-completion-rate proof is not independently positive"
        )
    if fwd.support_95() and completion_rate.normal_lb95 >= 0.0 and yards_not_collapsed:
        return (
            "CLIMB@95% - forward passing is statistically positive and completion "
            "does not regress at 95%; more held-out games or a stronger candidate needed for 99.99%"
        )
    if fwd.mean > 0.0 or completion_rate.mean > 0.0:
        return "directional only - movement exists but proof gates do not clear"
    return "no passing advancement"


def main() -> int:
    paths = sys.argv[1:]
    seen = set()
    duplicate_rows = 0
    d_completed: List[float] = []
    d_completion_rate: List[float] = []
    d_fwd: List[float] = []
    d_forward_share: List[float] = []
    d_yards: List[float] = []
    d_yards_per_pass: List[float] = []
    d_route_one: List[float] = []
    d_goals: List[float] = []
    d_sot: List[float] = []
    d_drib: List[float] = []
    cand = {
        key: 0.0
        for key in (
            "goals",
            "shots",
            "sot",
            "att",
            "comp",
            "fwd",
            "back",
            "yards",
            "chains",
            "sap",
            "assist",
            "drib",
            "route1",
            "int",
        )
    }
    base = cand.copy()

    for row in iter_rows(paths):
        key = (row.get("seed"), row.get("cand_home"))
        if key in seen:
            duplicate_rows += 1
            continue
        seen.add(key)

        c_att = f(row, "c_att")
        b_att = f(row, "b_att")
        c_comp = f(row, "c_comp")
        b_comp = f(row, "b_comp")
        c_fwd = f(row, "c_fwd")
        b_fwd = f(row, "b_fwd")
        c_yards = f(row, "c_yards")
        b_yards = f(row, "b_yards")
        c_route = f(row, "c_route1")
        b_route = f(row, "b_route1")

        d_completed.append(c_comp - b_comp)
        d_completion_rate.append(safe_ratio(c_comp, c_att) - safe_ratio(b_comp, b_att))
        d_fwd.append(c_fwd - b_fwd)
        d_forward_share.append(safe_ratio(c_fwd, c_comp) - safe_ratio(b_fwd, b_comp))
        d_yards.append(c_yards - b_yards)
        d_yards_per_pass.append(safe_ratio(c_yards, c_comp) - safe_ratio(b_yards, b_comp))
        d_route_one.append(c_route - b_route)
        d_goals.append(f(row, "c_goals") - f(row, "b_goals"))
        d_sot.append(f(row, "c_sot") - f(row, "b_sot"))
        d_drib.append(f(row, "c_drib") - f(row, "b_drib"))
        add_totals(cand, row, "c")
        add_totals(base, row, "b")

    rows = len(d_fwd)
    if rows == 0:
        print("no proof rows found", file=sys.stderr)
        return 2

    stats = {
        "completed": paired_stat(d_completed, 1),
        "completion_rate": paired_stat(d_completion_rate, 2),
        "fwd": paired_stat(d_fwd, 3),
        "forward_share": paired_stat(d_forward_share, 4),
        "yards": paired_stat(d_yards, 5),
        "yards_per_pass": paired_stat(d_yards_per_pass, 6),
        "route_one": paired_stat(d_route_one, 7),
        "goals": paired_stat(d_goals, 8),
        "sot": paired_stat(d_sot, 9),
        "drib": paired_stat(d_drib, 10),
    }
    nf = float(rows)

    print(
        "===== POOLED PASSING PROGRESSION PROOF "
        f"(candidate - baseline, paired over {rows} held-out games) ====="
    )
    print(
        f"bootstrap_replicates={bootstrap_replicates()} "
        f"bootstrap_seed={bootstrap_seed()} alpha99.99={ALPHA_9999}"
    )
    if duplicate_rows:
        print(f"duplicate fixtures skipped: {duplicate_rows}")

    print("\n----- headline statistics -----")
    print(
        f"completed passes/game:         cand {cand['comp'] / nf:.2f}  "
        f"base {base['comp'] / nf:.2f}"
    )
    print(fmt_metric("completed passes/game", stats["completed"]))
    print(
        f"pass completion rate:          cand {pct(cand['comp'], cand['att']):.2f}%  "
        f"base {pct(base['comp'], base['att']):.2f}%"
    )
    print(fmt_metric("pass completion rate", stats["completion_rate"], 100.0, "pp"))
    print(
        f"completed FORWARD passes/game: cand {cand['fwd'] / nf:.2f}  "
        f"base {base['fwd'] / nf:.2f}"
    )
    print(fmt_metric("completed forward passes/game", stats["fwd"]))
    print(
        f"forward share of completions:  cand {pct(cand['fwd'], cand['comp']):.2f}%  "
        f"base {pct(base['fwd'], base['comp']):.2f}%"
    )
    print(fmt_metric("forward share", stats["forward_share"], 100.0, "pp"))

    print("\n----- anti-gaming guardrails -----")
    print(
        f"yards/completed-pass:          cand {safe_ratio(cand['yards'], cand['comp']):.2f}  "
        f"base {safe_ratio(base['yards'], base['comp']):.2f}"
    )
    print(fmt_metric("yards/completed-pass", stats["yards_per_pass"]))
    print(
        f"forward YARDS gained/game:     cand {cand['yards'] / nf:.1f}  "
        f"base {base['yards'] / nf:.1f}"
    )
    print(fmt_metric("forward yards/game", stats["yards"]))
    print(
        f"route-one/game:                cand {cand['route1'] / nf:.2f}  "
        f"base {base['route1'] / nf:.2f}  (lower is better)"
    )
    print(fmt_metric("route-one/game", stats["route_one"]))

    print("\n----- secondary guardrails -----")
    print(fmt_metric("goals/game", stats["goals"]))
    print(fmt_metric("shots-on-target/game", stats["sot"]))
    print(fmt_metric("dribble beats/game", stats["drib"]))
    print(
        f"shots:        {cand['shots'] / nf:.2f} vs {base['shots'] / nf:.2f} | "
        f"shots-after-pass: {cand['sap'] / nf:.2f} vs {base['sap'] / nf:.2f} | "
        f"assists: {cand['assist'] / nf:.2f} vs {base['assist'] / nf:.2f} | "
        f"interceptions: {cand['int'] / nf:.2f} vs {base['int'] / nf:.2f}"
    )

    print(
        "\nVERDICT: "
        + verdict_for(
            stats["fwd"],
            stats["completion_rate"],
            stats["yards_per_pass"],
            stats["route_one"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
