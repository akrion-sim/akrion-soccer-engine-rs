#!/usr/bin/env python3
"""Step 2a prototype (see docs/offline-encoder-step2-plan.md): train a weighted value
head offline on the exported (state -> value) dataset and verify it beats the baseline.
Pure-numpy MLP so it runs without sklearn/torch. The Rust format-compatible distill into
FeedForwardNetwork is the documented follow-up.

  scripts/train_offline_head.py <dataset.jsonl> \
    --layout-out /tmp/soccer_offline_layout.json \
    --report-out /tmp/soccer_offline_report.json \
    --weights-out /tmp/soccer_offline_weights.json
"""
import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the Step 2a offline value head and optionally export Rust-loadable artifacts."
    )
    parser.add_argument("dataset", nargs="?", default="/tmp/soccer_offline_sample.jsonl")
    parser.add_argument("--layout-out", help="Write canonical feature layout JSON.")
    parser.add_argument("--report-out", help="Write training metrics/report JSON.")
    parser.add_argument("--weights-out", help="Write trained one-hidden-layer MLP weights JSON.")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--holdout-frac", type=float, default=0.15)
    return parser.parse_args()


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # tolerate a truncated final line from a streaming export
    if len(rows) < 4:
        raise SystemExit(f"need at least 4 rows to train and hold out; loaded {len(rows)}")
    return rows


def build_feature_layout(rows, source_path):
    cat_vocab, int_keys = {}, set()
    for row in rows:
        for key, value in row["state_key"].items():
            if isinstance(value, bool) or isinstance(value, str):
                cat_vocab.setdefault(key, {}).setdefault(str(value), None)
            elif isinstance(value, (int, float)):
                int_keys.add(key)

    cat_cols = [(key, val) for key in sorted(cat_vocab) for val in sorted(cat_vocab[key])]
    int_cols = sorted(int_keys)
    col_index = {("cat", key, val): i for i, (key, val) in enumerate(cat_cols)}
    base = len(cat_cols)
    for j, key in enumerate(int_cols):
        col_index[("int", key)] = base + j

    int_vals = {key: [] for key in int_cols}
    for row in rows:
        for key in int_cols:
            value = row["state_key"].get(key)
            if isinstance(value, (int, float)):
                int_vals[key].append(value)
    int_mean = {key: float(np.mean(int_vals[key])) if int_vals[key] else 0.0 for key in int_cols}
    int_std = {key: float(np.std(int_vals[key])) or 1.0 for key in int_cols}

    categorical_columns = [
        {"key": key, "value": val, "index": col_index[("cat", key, val)]}
        for key, val in cat_cols
    ]
    numeric_columns = [
        {"key": key, "index": col_index[("int", key)], "mean": int_mean[key], "std": int_std[key]}
        for key in int_cols
    ]
    layout = {
        "format": "akrion-soccer-offline-head-layout-v1",
        "source": str(source_path),
        "target": "value_micros / 1e6",
        "weight": "max(weight_micros / 1e6, 1e-3)",
        "input_dim": len(cat_cols) + len(int_cols),
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
    }
    return col_index, int_mean, int_std, layout


def featurize(state_key, input_dim, col_index, int_mean, int_std):
    x = np.zeros(input_dim, dtype=np.float32)
    for key, value in state_key.items():
        if isinstance(value, bool) or isinstance(value, str):
            idx = col_index.get(("cat", key, str(value)))
            if idx is not None:
                x[idx] = 1.0
        elif isinstance(value, (int, float)):
            idx = col_index.get(("int", key))
            if idx is not None:
                x[idx] = float(np.clip((value - int_mean[key]) / int_std[key], -8.0, 8.0))
    return x


def wmse(pred, yy, ww):
    return float(np.average((pred - yy) ** 2, weights=ww))


def train_mlp(X, y, w, train_idx, hidden, epochs, lr, rng):
    W1 = rng.normal(0, 0.1, (X.shape[1], hidden)).astype(np.float32)
    b1 = np.zeros(hidden, np.float32)
    W2 = rng.normal(0, 0.1, (hidden, 1)).astype(np.float32)
    b2 = np.zeros(1, np.float32)

    Xt, yt, wt = X[train_idx], y[train_idx], w[train_idx]
    wn = wt / wt.mean()
    for _ in range(epochs):
        perm = rng.permutation(len(train_idx))
        for start in range(0, len(train_idx), 256):
            batch_idx = perm[start:start + 256]
            xb, yb, wb = Xt[batch_idx], yt[batch_idx], wn[batch_idx]
            z1 = xb @ W1 + b1
            a1 = np.tanh(z1)
            out = (a1 @ W2 + b2)[:, 0]
            grad = (2 * wb * (out - yb) / len(batch_idx)).astype(np.float32)
            gW2 = a1.T @ grad[:, None]
            gb2 = grad.sum(keepdims=True)
            ga1 = grad[:, None] @ W2.T * (1 - a1 ** 2)
            gW1 = xb.T @ ga1
            gb1 = ga1.sum(0)
            W2 -= lr * gW2
            b2 -= lr * gb2
            W1 -= lr * gW1
            b1 -= lr * gb1
    return W1, b1, W2, b2


def predict(X, W1, b1, W2, b2):
    return (np.tanh(X @ W1 + b1) @ W2 + b2)[:, 0]


def write_json(path, payload):
    if not path:
        return
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    global np
    import numpy as np

    path = Path(args.dataset)
    rows = load_rows(path)
    print(f"loaded {len(rows)} rows from {path}")

    col_index, int_mean, int_std, layout = build_feature_layout(rows, path)
    input_dim = layout["input_dim"]
    print(
        f"feature layout: {len(layout['categorical_columns'])} one-hot + "
        f"{len(layout['numeric_columns'])} numeric = {input_dim} dims"
    )

    X = np.stack([featurize(row["state_key"], input_dim, col_index, int_mean, int_std) for row in rows])
    y = np.array([row["value_micros"] / 1e6 for row in rows], dtype=np.float32)
    w = np.array([max(row["weight_micros"] / 1e6, 1e-3) for row in rows], dtype=np.float32)

    rng = np.random.default_rng(args.seed)
    holdout_frac = min(max(args.holdout_frac, 0.05), 0.5)
    cut = int(round(len(rows) * (1.0 - holdout_frac)))
    cut = min(max(cut, 1), len(rows) - 1)
    idx = rng.permutation(len(rows))
    train_idx, holdout_idx = idx[:cut], idx[cut:]

    base_pred = np.average(y[train_idx], weights=w[train_idx])
    base_mse = wmse(np.full(len(holdout_idx), base_pred), y[holdout_idx], w[holdout_idx])

    W1, b1, W2, b2 = train_mlp(X, y, w, train_idx, args.hidden, args.epochs, args.lr, rng)
    mlp_mse = wmse(predict(X[holdout_idx], W1, b1, W2, b2), y[holdout_idx], w[holdout_idx])
    improvement_pct = None if base_mse <= 1e-12 else float(100 * (1 - mlp_mse / base_mse))
    verdict = "mlp_learns_signal" if improvement_pct is not None and mlp_mse < base_mse * 0.97 else "no_clear_signal"

    report = {
        "format": "akrion-soccer-offline-head-report-v1",
        "source": str(path),
        "rows": len(rows),
        "train_rows": int(len(train_idx)),
        "holdout_rows": int(len(holdout_idx)),
        "input_dim": int(input_dim),
        "categorical_columns": len(layout["categorical_columns"]),
        "numeric_columns": len(layout["numeric_columns"]),
        "hidden_dim": int(args.hidden),
        "epochs": int(args.epochs),
        "learning_rate": float(args.lr),
        "seed": int(args.seed),
        "baseline_weighted_mse": float(base_mse),
        "mlp_weighted_mse": float(mlp_mse),
        "improvement_pct": improvement_pct,
        "verdict": verdict,
    }
    weights = {
        "format": "numpy-tanh-mlp-v1",
        "layout_format": layout["format"],
        "input_dim": int(input_dim),
        "hidden_dim": int(args.hidden),
        "activation": "tanh",
        "output": "value_micros / 1e6",
        "W1_row_major": W1.reshape(-1).tolist(),
        "b1": b1.tolist(),
        "W2_row_major": W2.reshape(-1).tolist(),
        "b2": b2.tolist(),
    }

    write_json(args.layout_out, layout)
    write_json(args.report_out, report)
    write_json(args.weights_out, weights)

    imp = "n/a" if improvement_pct is None else f"{improvement_pct:.1f}%"
    print(
        f"held-out weighted MSE: baseline(mean)={base_mse:.4f} "
        f"MLP={mlp_mse:.4f} improvement={imp}"
    )
    print("VERDICT:", verdict)


if __name__ == "__main__":
    main()
