#!/usr/bin/env python3
"""Summarize DD_SOCCER_FWD_TRACE JSONL into proof-facing action-change metrics."""

import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List


def truthy(row: Dict[str, object], key: str) -> bool:
    return bool(row.get(key, False))


def number(row: Dict[str, object], key: str) -> float:
    try:
        value = float(row.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def rate(count: int, denom: int) -> float:
    return count / denom if denom > 0 else 0.0


def pct(count: int, denom: int) -> str:
    return f"{rate(count, denom) * 100.0:.2f}%"


def iter_rows(paths: List[str]) -> Iterator[Dict[str, object]]:
    if not paths:
        paths = ["-"]
    for path in paths:
        if path == "-":
            handle = sys.stdin
            label = "<stdin>"
        else:
            handle = Path(path).open("r", encoding="utf-8")
            label = path
        with handle:
            for line_no, line in enumerate(handle, start=1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError as exc:
                    print(f"skip malformed JSON {label}:{line_no}: {exc}", file=sys.stderr)
                    continue
                if isinstance(row, dict):
                    yield row


def entropy(counts: Iterable[int]) -> float:
    counts = [count for count in counts if count > 0]
    total = sum(counts)
    if total <= 0:
        return 0.0
    return -sum((count / total) * math.log(count / total) for count in counts)


def main() -> int:
    rows = list(iter_rows(sys.argv[1:]))
    total = len(rows)
    if total == 0:
        print("no forward trace rows found", file=sys.stderr)
        return 2

    qualified = sum(1 for row in rows if truthy(row, "qualifies"))
    denom = qualified if qualified > 0 else total
    scoped = [row for row in rows if truthy(row, "qualifies")] or rows

    scored = sum(1 for row in scoped if truthy(row, "e_scored"))
    root = sum(1 for row in scoped if truthy(row, "e_root"))
    net_changed = sum(1 for row in scoped if truthy(row, "s_net_changed"))
    label_changed = sum(1 for row in scoped if truthy(row, "s_label_changed"))
    pick_fwd = sum(1 for row in scoped if truthy(row, "pick_is_fwd"))
    pick_pass = sum(1 for row in scoped if truthy(row, "pick_is_pass"))
    pick_dribble = sum(1 for row in scoped if truthy(row, "pick_is_dribble"))
    pick_shot = sum(1 for row in scoped if truthy(row, "pick_is_shot"))
    tabular_fwd = sum(1 for row in scoped if truthy(row, "tabular_argmax_is_fwd"))
    forward_injected = sum(1 for row in scoped if truthy(row, "forward_injected"))
    forward_injected_survived = sum(
        1 for row in scoped if truthy(row, "forward_injected_survived")
    )
    forward_injected_selected = sum(
        1 for row in scoped if truthy(row, "forward_injected_selected")
    )
    mcts_enabled = sum(1 for row in scoped if truthy(row, "mcts_enabled"))
    pass_exposed = sum(1 for row in scoped if number(row, "n_pass_scored") > 0)
    dribble_exposed = sum(1 for row in scoped if number(row, "n_dribble_scored") > 0)
    shot_exposed = sum(1 for row in scoped if number(row, "n_shot_scored") > 0)
    mean_scored = sum(number(row, "n_scored") for row in scoped) / max(1, len(scoped))
    mean_fwd_scored = sum(number(row, "n_fwd_scored") for row in scoped) / max(1, len(scoped))

    bucket_counts: Counter[int] = Counter()
    label_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    for row in scoped:
        label = str(row.get("pick_label", ""))
        family = str(row.get("pick_family", ""))
        if label:
            label_counts[label] += 1
        if family:
            family_counts[family] += 1
        try:
            bucket = int(row.get("pick_kick_bucket", -1))
        except (TypeError, ValueError):
            bucket = -1
        if 0 <= bucket <= 9:
            bucket_counts[bucket] += 1

    bucket_entropy = entropy(bucket_counts.values())
    bucket_entropy_norm = bucket_entropy / math.log(10.0) if bucket_counts else 0.0
    bucket_selected = sum(bucket_counts.values())

    print("===== FORWARD-PASS TRACE REPORT =====")
    print(f"rows={total} qualified={qualified} denominator={denom}")
    print(f"mcts_enabled_rate={pct(mcts_enabled, denom)}")
    print(f"mean_scored_candidates={mean_scored:.2f} mean_forward_scored_candidates={mean_fwd_scored:.2f}")
    print("")
    print("----- exposure and selection -----")
    print(f"pass_candidate_scored_rate={pct(pass_exposed, denom)}")
    print(f"forward_candidate_scored_rate={pct(scored, denom)}")
    print(f"forward_candidate_root_rate={pct(root, denom)}")
    print(f"tabular_argmax_forward_rate={pct(tabular_fwd, denom)}")
    print(f"picked_forward_rate={pct(pick_fwd, denom)}")
    print(f"net_changed_family_rate={pct(net_changed, denom)}")
    print(f"net_changed_label_rate={pct(label_changed, denom)}")
    print(f"forward_injected_rate={pct(forward_injected, denom)}")
    print(f"forward_injected_survived_rate={pct(forward_injected_survived, denom)}")
    print(f"forward_injected_selected_rate={pct(forward_injected_selected, denom)}")
    print("")
    print("----- selected families -----")
    print(f"picked_pass_rate={pct(pick_pass, denom)}")
    print(f"picked_dribble_rate={pct(pick_dribble, denom)}")
    print(f"picked_shot_rate={pct(pick_shot, denom)}")
    print(f"dribble_candidate_scored_rate={pct(dribble_exposed, denom)}")
    print(f"shot_candidate_scored_rate={pct(shot_exposed, denom)}")
    print("")
    print("----- kick buckets -----")
    print(f"kick_bucket_selected_rate={pct(bucket_selected, denom)}")
    print(f"selected_kick_speed_bucket_entropy={bucket_entropy:.4f}")
    print(f"selected_kick_speed_bucket_entropy_norm={bucket_entropy_norm:.4f}")
    print(
        "selected_kick_speed_bucket_counts="
        + ",".join(f"{bucket}:{bucket_counts.get(bucket, 0)}" for bucket in range(10))
    )
    print("")
    print("top_pick_families=" + ",".join(f"{k}:{v}" for k, v in family_counts.most_common(8)))
    print("top_pick_labels=" + ",".join(f"{k}:{v}" for k, v in label_counts.most_common(12)))

    if qualified == 0:
        verdict = "NO-QUALIFIED-FORWARD-DECISIONS - trace cannot prove pass-selection climb"
    elif rate(scored, denom) < 0.25:
        verdict = "CANDIDATE-EXPOSURE-BOTTLENECK - visible forward options rarely reach scoring"
    elif rate(root, denom) < 0.25:
        verdict = "ROOT-SURVIVAL-BOTTLENECK - forward options score but rarely reach the MCTS/root choice set"
    elif rate(pick_fwd, denom) < 0.10:
        verdict = "SELECTION-BOTTLENECK - forward options are exposed but rarely selected"
    elif bucket_selected > 0 and bucket_entropy_norm < 0.35:
        verdict = "BUCKET-COLLAPSE-RISK - learned kick bucket selections lack entropy"
    elif rate(label_changed, denom) < 0.05:
        verdict = "LOW-CAUSAL-ACTION-CHANGE - net/planner rarely changes the executed label"
    else:
        verdict = "TRACE-SUPPORTS-CAUSAL-EXPOSURE - forward options and action changes are measurable"
    print(f"VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
