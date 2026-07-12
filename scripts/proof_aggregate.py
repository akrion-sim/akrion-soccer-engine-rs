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
MAX_BOOTSTRAP_REPLICATES = 200_000


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
        return min(MAX_BOOTSTRAP_REPLICATES, max(1000, int(raw)))
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
        "ipatt",
        "ipcomp",
        "ipfwd",
        "ipback",
        "ipyards",
        "ipto",
        "chains",
        "sap",
        "assist",
        "drib",
        "drib_to",
        "route1",
        "int",
        "int_won",
        "pto",
        "tackles",
        "lbr",
        "team_up",
        "team_near",
        "chase_adv",
        "def_chase",
    ):
        if key == "int_won" and f"{prefix}_int_won" not in row:
            total[key] += f(row, f"{prefix}_int")
        else:
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
    pass_turnover_rate_improvement: Stat,
    proof_rows: int,
    strict_net_fwd: Stat,
    strict_turnover_rate_improvement: Stat,
) -> str:
    yards_not_collapsed = yards_per_pass.bootstrap_lb95 > -0.25
    route_one_not_rising = route_one.mean <= 0.10
    strict_available = (
        strict_net_fwd.n == proof_rows
        and strict_turnover_rate_improvement.n == proof_rows
    )
    headline_fwd = strict_net_fwd if strict_available else fwd
    headline_label = (
        "strict net-forward intentional passes"
        if strict_available
        else "completed forward passes"
    )
    turnover_guard = (
        strict_turnover_rate_improvement
        if strict_available
        else pass_turnover_rate_improvement
    )
    pass_turnovers_available = turnover_guard.n == proof_rows
    pass_turnovers_not_rising = (
        pass_turnovers_available
        and turnover_guard.mean >= 0.0
    )
    if (
        headline_fwd.support_9999()
        and completion_rate.support_9999()
        and yards_not_collapsed
        and route_one_not_rising
        and pass_turnovers_not_rising
    ):
        return (
            f"PROVEN@99.99% - {headline_label} and pass completion rate "
            "both pass normal+bootstrap+sign-test, with yards/pass, route-one, and pass-turnover guards intact"
        )
    if (
        headline_fwd.support_9999()
        and completion_rate.mean >= 0.0
        and yards_not_collapsed
        and route_one_not_rising
        and pass_turnovers_not_rising
    ):
        return (
            f"FORWARD-PASS-PROVEN@99.99% - {headline_label} passes triple-stat proof, "
            "with pass-turnover guard intact, but pass-completion-rate proof is not independently positive"
        )
    if (
        headline_fwd.support_95()
        and completion_rate.normal_lb95 >= 0.0
        and yards_not_collapsed
        and pass_turnovers_not_rising
    ):
        return (
            f"CLIMB@95% - {headline_label} is statistically positive and completion "
            "does not regress at 95%, with pass-turnover guard intact; more held-out games or a stronger candidate needed for 99.99%"
        )
    if not pass_turnovers_available:
        return "anti-gaming guard unavailable - rerun eval so every row has c_pto/b_pto JSONL fields"
    if not pass_turnovers_not_rising:
        return "anti-gaming guard failed - candidate increased pass-turnover rate"
    if headline_fwd.mean > 0.0 or completion_rate.mean > 0.0:
        return f"directional only - {headline_label} movement exists but proof gates do not clear"
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
    d_pass_turnover_rate_improvement: List[float] = []
    d_strict_fwd: List[float] = []
    d_strict_net_fwd: List[float] = []
    d_strict_turnover_rate_improvement: List[float] = []
    d_goals: List[float] = []
    d_sot: List[float] = []
    d_drib: List[float] = []
    d_sot_rate: List[float] = []
    d_goal_per_shot: List[float] = []
    d_goal_per_sot: List[float] = []
    d_worked_sot_share: List[float] = []
    d_dribble_success_rate: List[float] = []
    d_loose_ball_recoveries: List[float] = []
    d_tackles: List[float] = []
    d_teamwork_upfield: List[float] = []
    d_teamwork_near_ball: List[float] = []
    d_chase_advantage: List[float] = []
    d_defensive_chase_load: List[float] = []
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
            "ipatt",
            "ipcomp",
            "ipfwd",
            "ipback",
            "ipyards",
            "ipto",
            "chains",
            "sap",
            "assist",
            "drib",
            "drib_to",
            "route1",
            "int",
            "int_won",
            "pto",
            "tackles",
            "lbr",
            "team_up",
            "team_near",
            "chase_adv",
            "def_chase",
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
        c_goals = f(row, "c_goals")
        b_goals = f(row, "b_goals")
        c_shots = f(row, "c_shots")
        b_shots = f(row, "b_shots")
        c_sot = f(row, "c_sot")
        b_sot = f(row, "b_sot")
        c_sap = f(row, "c_sap")
        b_sap = f(row, "b_sap")
        has_pto = "c_pto" in row and "b_pto" in row
        c_pto = f(row, "c_pto") if has_pto else 0.0
        b_pto = f(row, "b_pto") if has_pto else 0.0
        has_dribble_turnovers = "c_drib_to" in row and "b_drib_to" in row
        c_drib = f(row, "c_drib")
        b_drib = f(row, "b_drib")
        c_drib_to = f(row, "c_drib_to") if has_dribble_turnovers else 0.0
        b_drib_to = f(row, "b_drib_to") if has_dribble_turnovers else 0.0
        has_strict = all(
            key in row
            for key in (
                "c_ipatt",
                "b_ipatt",
                "c_ipcomp",
                "b_ipcomp",
                "c_ipfwd",
                "b_ipfwd",
                "c_ipto",
                "b_ipto",
            )
        )
        c_ipatt = f(row, "c_ipatt") if has_strict else 0.0
        b_ipatt = f(row, "b_ipatt") if has_strict else 0.0
        c_ipfwd = f(row, "c_ipfwd") if has_strict else 0.0
        b_ipfwd = f(row, "b_ipfwd") if has_strict else 0.0
        c_ipto = f(row, "c_ipto") if has_strict else 0.0
        b_ipto = f(row, "b_ipto") if has_strict else 0.0

        d_completed.append(c_comp - b_comp)
        d_completion_rate.append(safe_ratio(c_comp, c_att) - safe_ratio(b_comp, b_att))
        d_fwd.append(c_fwd - b_fwd)
        d_forward_share.append(safe_ratio(c_fwd, c_comp) - safe_ratio(b_fwd, b_comp))
        d_yards.append(c_yards - b_yards)
        d_yards_per_pass.append(safe_ratio(c_yards, c_comp) - safe_ratio(b_yards, b_comp))
        d_route_one.append(c_route - b_route)
        if has_pto:
            d_pass_turnover_rate_improvement.append(
                safe_ratio(b_pto, b_att) - safe_ratio(c_pto, c_att)
            )
        if has_strict:
            d_strict_fwd.append(c_ipfwd - b_ipfwd)
            d_strict_net_fwd.append((c_ipfwd - c_ipto) - (b_ipfwd - b_ipto))
            d_strict_turnover_rate_improvement.append(
                safe_ratio(b_ipto, b_ipatt) - safe_ratio(c_ipto, c_ipatt)
            )
        d_goals.append(c_goals - b_goals)
        d_sot.append(c_sot - b_sot)
        d_drib.append(c_drib - b_drib)
        d_sot_rate.append(safe_ratio(c_sot, c_shots) - safe_ratio(b_sot, b_shots))
        d_goal_per_shot.append(
            safe_ratio(c_goals, c_shots) - safe_ratio(b_goals, b_shots)
        )
        d_goal_per_sot.append(safe_ratio(c_goals, c_sot) - safe_ratio(b_goals, b_sot))
        d_worked_sot_share.append(safe_ratio(c_sap, c_sot) - safe_ratio(b_sap, b_sot))
        if has_dribble_turnovers:
            d_dribble_success_rate.append(
                safe_ratio(c_drib, c_drib + c_drib_to)
                - safe_ratio(b_drib, b_drib + b_drib_to)
            )
        d_loose_ball_recoveries.append(f(row, "c_lbr") - f(row, "b_lbr"))
        d_tackles.append(f(row, "c_tackles") - f(row, "b_tackles"))
        d_teamwork_upfield.append(f(row, "c_team_up") - f(row, "b_team_up"))
        d_teamwork_near_ball.append(f(row, "c_team_near") - f(row, "b_team_near"))
        d_chase_advantage.append(f(row, "c_chase_adv") - f(row, "b_chase_adv"))
        d_defensive_chase_load.append(f(row, "c_def_chase") - f(row, "b_def_chase"))
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
        "pass_turnover_rate_improvement": paired_stat(d_pass_turnover_rate_improvement, 8),
        "strict_fwd": paired_stat(d_strict_fwd, 9),
        "strict_net_fwd": paired_stat(d_strict_net_fwd, 10),
        "strict_turnover_rate_improvement": paired_stat(
            d_strict_turnover_rate_improvement, 11
        ),
        "goals": paired_stat(d_goals, 12),
        "sot": paired_stat(d_sot, 13),
        "drib": paired_stat(d_drib, 14),
        "sot_rate": paired_stat(d_sot_rate, 15),
        "goal_per_shot": paired_stat(d_goal_per_shot, 16),
        "goal_per_sot": paired_stat(d_goal_per_sot, 17),
        "worked_sot_share": paired_stat(d_worked_sot_share, 18),
        "dribble_success_rate": paired_stat(d_dribble_success_rate, 19),
        "loose_ball_recoveries": paired_stat(d_loose_ball_recoveries, 20),
        "tackles": paired_stat(d_tackles, 21),
        "teamwork_upfield": paired_stat(d_teamwork_upfield, 22),
        "teamwork_near_ball": paired_stat(d_teamwork_near_ball, 23),
        "chase_advantage": paired_stat(d_chase_advantage, 24),
        "defensive_chase_load": paired_stat(d_defensive_chase_load, 25),
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
    if stats["strict_fwd"].n == rows:
        print(
            f"strict intentional passes:     cand {cand['ipcomp'] / nf:.2f}/g  "
            f"base {base['ipcomp'] / nf:.2f}/g"
        )
        print(
            f"strict FORWARD passes/game:    cand {cand['ipfwd'] / nf:.2f}  "
            f"base {base['ipfwd'] / nf:.2f}"
        )
        print(fmt_metric("strict forward passes/game", stats["strict_fwd"]))
        print(
            f"strict NET forward/game:       cand {(cand['ipfwd'] - cand['ipto']) / nf:.2f}  "
            f"base {(base['ipfwd'] - base['ipto']) / nf:.2f}"
        )
        print(fmt_metric("strict net forward/game", stats["strict_net_fwd"]))
        print(
            f"strict pass turnover rate:     cand {pct(cand['ipto'], cand['ipatt']):.2f}%  "
            f"base {pct(base['ipto'], base['ipatt']):.2f}%"
        )
        print(
            fmt_metric(
                "strict turnover improvement",
                stats["strict_turnover_rate_improvement"],
                100.0,
                "pp",
            )
        )
    else:
        print(
            "strict intentional passes:     unavailable "
            f"({stats['strict_fwd'].n}/{rows} rows have c_ip*/b_ip*)"
        )

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
    if stats["pass_turnover_rate_improvement"].n == rows:
        print(
            f"pass turnovers/game:           cand {cand['pto'] / nf:.2f}  "
            f"base {base['pto'] / nf:.2f}  (lower is better)"
        )
        print(
            f"pass turnover rate:            cand {pct(cand['pto'], cand['att']):.2f}%  "
            f"base {pct(base['pto'], base['att']):.2f}%"
        )
        print(
            fmt_metric(
                "pass turnover rate improvement",
                stats["pass_turnover_rate_improvement"],
                100.0,
                "pp",
            )
        )
    else:
        print(
            "pass turnovers/game:           unavailable "
            f"({stats['pass_turnover_rate_improvement'].n}/{rows} rows have c_pto/b_pto)"
        )

    print("\n----- secondary guardrails -----")
    print(fmt_metric("goals/game", stats["goals"]))
    print(fmt_metric("shots-on-target/game", stats["sot"]))
    print(
        f"SOT rate:     {pct(cand['sot'], cand['shots']):.2f}% vs "
        f"{pct(base['sot'], base['shots']):.2f}%"
    )
    print(fmt_metric("SOT rate", stats["sot_rate"], 100.0, "pp"))
    print(
        f"goals/shot:   {pct(cand['goals'], cand['shots']):.2f}% vs "
        f"{pct(base['goals'], base['shots']):.2f}% | "
        f"goals/SOT: {pct(cand['goals'], cand['sot']):.2f}% vs "
        f"{pct(base['goals'], base['sot']):.2f}%"
    )
    print(fmt_metric("goals/shot", stats["goal_per_shot"], 100.0, "pp"))
    print(fmt_metric("goals/SOT", stats["goal_per_sot"], 100.0, "pp"))
    print(
        f"worked SOT share: {pct(cand['sap'], cand['sot']):.2f}% vs "
        f"{pct(base['sap'], base['sot']):.2f}%"
    )
    print(fmt_metric("worked SOT share", stats["worked_sot_share"], 100.0, "pp"))
    print(fmt_metric("dribble beats/game", stats["drib"]))
    if stats["dribble_success_rate"].n == rows:
        print(
            f"dribble contest success:    "
            f"{pct(cand['drib'], cand['drib'] + cand['drib_to']):.2f}% vs "
            f"{pct(base['drib'], base['drib'] + base['drib_to']):.2f}% | "
            f"turnovers/game {cand['drib_to'] / nf:.2f} vs {base['drib_to'] / nf:.2f}"
        )
        print(
            fmt_metric(
                "dribble success rate",
                stats["dribble_success_rate"],
                100.0,
                "pp",
            )
        )
    else:
        print(
            "dribble contest success:    unavailable "
            f"({stats['dribble_success_rate'].n}/{rows} rows have c_drib_to/b_drib_to)"
        )
    print(
        f"shots:        {cand['shots'] / nf:.2f} vs {base['shots'] / nf:.2f} | "
        f"shots-after-pass: {cand['sap'] / nf:.2f} vs {base['sap'] / nf:.2f} | "
        f"assists: {cand['assist'] / nf:.2f} vs {base['assist'] / nf:.2f} | "
        f"interceptions won: {cand['int_won'] / nf:.2f} vs {base['int_won'] / nf:.2f}"
    )
    print(
        f"recoveries/tackles: {cand['lbr'] / nf:.2f}/{cand['tackles'] / nf:.2f} vs "
        f"{base['lbr'] / nf:.2f}/{base['tackles'] / nf:.2f}"
    )
    print(fmt_metric("loose-ball recoveries/game", stats["loose_ball_recoveries"]))
    print(fmt_metric("tackles/game", stats["tackles"]))
    print(
        f"teamwork upfield/near-ball: "
        f"{cand['team_up'] / nf:.2f}/{cand['team_near'] / nf:.2f} vs "
        f"{base['team_up'] / nf:.2f}/{base['team_near'] / nf:.2f}"
    )
    print(fmt_metric("teamwork upfield/game", stats["teamwork_upfield"]))
    print(fmt_metric("teamwork near-ball/game", stats["teamwork_near_ball"]))
    print(
        f"chase advantage/defensive load: "
        f"{cand['chase_adv'] / nf:.2f}/{cand['def_chase'] / nf:.2f} vs "
        f"{base['chase_adv'] / nf:.2f}/{base['def_chase'] / nf:.2f}"
    )
    print(fmt_metric("possession chase advantage/game", stats["chase_advantage"]))
    print(fmt_metric("defensive chase load/game", stats["defensive_chase_load"]))

    print(
        "\nVERDICT: "
        + verdict_for(
            stats["fwd"],
            stats["completion_rate"],
            stats["yards_per_pass"],
            stats["route_one"],
            stats["pass_turnover_rate_improvement"],
            rows,
            stats["strict_net_fwd"],
            stats["strict_turnover_rate_improvement"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
