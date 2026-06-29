#!/usr/bin/env python3
"""Step 2a prototype (see docs/offline-encoder-step2-plan.md): train a weighted value
head offline on the exported (state -> value) dataset and verify it beats the baseline.
Pure-numpy MLP so it runs without sklearn/torch. The Rust format-compatible distill into
FeedForwardNetwork is the documented follow-up.

  scripts/train_offline_head.py <dataset.jsonl>
"""
import sys, json, numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/soccer_offline_sample.jsonl"
rows = []
for ln in open(path):
    ln = ln.strip()
    if not ln: continue
    try: rows.append(json.loads(ln))
    except: pass            # tolerate a truncated final line
print(f"loaded {len(rows)} rows from {path}")

# --- canonical feature layout: categorical/bool -> one-hot, int -> standardized ---
cat_vocab, int_keys = {}, set()
for r in rows:
    for k, v in r["state_key"].items():
        if isinstance(v, bool) or isinstance(v, str):
            cat_vocab.setdefault(k, {}).setdefault(str(v), None)
        elif isinstance(v, (int, float)):
            int_keys.add(k)
cat_cols = [(k, val) for k in sorted(cat_vocab) for val in sorted(cat_vocab[k])]
int_cols = sorted(int_keys)
col_index = {("cat", k, val): i for i, (k, val) in enumerate(cat_cols)}
base = len(cat_cols)
for j, k in enumerate(int_cols): col_index[("int", k)] = base + j
D = len(cat_cols) + len(int_cols)
print(f"feature layout: {len(cat_cols)} one-hot + {len(int_cols)} numeric = {D} dims")

# int standardization stats
int_vals = {k: [] for k in int_cols}
for r in rows:
    for k in int_cols:
        v = r["state_key"].get(k)
        if isinstance(v,(int,float)): int_vals[k].append(v)
int_mean = {k:(np.mean(int_vals[k]) if int_vals[k] else 0.0) for k in int_cols}
int_std  = {k: (float(np.std(int_vals[k])) or 1.0) for k in int_cols}

def featurize(sk):
    x = np.zeros(D, dtype=np.float32)
    for k, v in sk.items():
        if isinstance(v, bool) or isinstance(v, str):
            i = col_index.get(("cat", k, str(v)))
            if i is not None: x[i] = 1.0
        elif isinstance(v, (int, float)):
            i = col_index.get(("int", k))
            if i is not None: x[i] = float(np.clip((v - int_mean[k]) / int_std[k], -8.0, 8.0))
    return x

X = np.stack([featurize(r["state_key"]) for r in rows])
y = np.array([r["value_micros"]/1e6 for r in rows], dtype=np.float32)
w = np.array([max(r["weight_micros"]/1e6, 1e-3) for r in rows], dtype=np.float32)

rng = np.random.default_rng(0)
idx = rng.permutation(len(rows)); cut = int(len(rows)*0.85)
tr, te = idx[:cut], idx[cut:]

# baseline = weighted global mean
base_pred = np.average(y[tr], weights=w[tr])
def wmse(pred, yy, ww): return float(np.average((pred-yy)**2, weights=ww))
base_mse = wmse(np.full(len(te), base_pred), y[te], w[te])

# tiny 1-hidden-layer MLP, weighted SGD
H = 64
W1 = rng.normal(0, 0.1, (D, H)).astype(np.float32); b1 = np.zeros(H, np.float32)
W2 = rng.normal(0, 0.1, (H, 1)).astype(np.float32); b2 = np.zeros(1, np.float32)
lr = 0.02
Xt, yt, wt = X[tr], y[tr], w[tr]
wn = wt/ wt.mean()
for ep in range(25):
    p = rng.permutation(len(tr))
    for s in range(0, len(tr), 256):
        bi = p[s:s+256]
        xb, yb, wb = Xt[bi], yt[bi], wn[bi]
        z1 = xb@W1+b1; a1 = np.tanh(z1); out = (a1@W2+b2)[:,0]
        g = (2*wb*(out-yb)/len(bi)).astype(np.float32)
        gW2 = a1.T@g[:,None]; gb2 = g.sum(keepdims=True)
        ga1 = g[:,None]@W2.T*(1-a1**2)
        gW1 = xb.T@ga1; gb1 = ga1.sum(0)
        W2-=lr*gW2; b2-=lr*gb2; W1-=lr*gW1; b1-=lr*gb1
def predict(XX):
    return (np.tanh(XX@W1+b1)@W2+b2)[:,0]
mlp_mse = wmse(predict(X[te]), y[te], w[te])
print(f"held-out weighted MSE:  baseline(mean)={base_mse:.4f}   MLP={mlp_mse:.4f}   improvement={100*(1-mlp_mse/base_mse):.1f}%")
print("VERDICT:", "MLP learns signal ✓" if mlp_mse < base_mse*0.97 else "no clear signal")
