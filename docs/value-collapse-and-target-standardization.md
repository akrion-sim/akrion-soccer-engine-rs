# The Value-Collapse Plateau — Diagnosis and Fix

> **What this documents.** For a long time the neural-authoritative soccer learner sat at a hard
> **parity plateau**: ~0.53 payoff vs the pure analytic engine, unmoved across hundreds of
> thousands of training steps, even though the training loss looked healthy. This note explains
> what was actually broken (the **value head collapsed to a constant**), the fix that un-broke it
> (**value-target standardization** — a *neural-net training* fix), and the role of approximate
> dynamic programming (a **complement**, not a replacement for the net). It also corrects a common
> misconception: this was **not** "use DP instead of neural nets."

## 1. Symptom: a hard parity plateau with healthy-looking loss

The learner runs **neural-authoritative**: each decision ranks the analytic engine's candidate
actions by the neural value head `V_net(s, a)` (`Q_eff = candidate.value + λ·V_net`; λ=8 in this experiment dominates, so the tabular
candidate value is a negligible tie-break floor; the env var `DD_SOCCER_NEURAL_AUTHORITATIVE_LAMBDA`
defaults to 0.0). Against the analytic engine it scored ~0.53 (parity)
and never climbed, while `averageLoss` sat at ~0.0006 — i.e. training *looked* converged and fine.

That combination (flat metric + tiny loss) is the classic signature of a critic that has stopped
distinguishing states.

## 2. Diagnosis: the value head collapsed to a near-constant, action-blind output

Inspecting a trained net (`inputDim=610` — the pre-action-param subtotal
`SOCCER_NEURAL_PRE_ACTION_PARAM_FEATURE_DIM`; the current `SOCCER_NEURAL_FEATURE_DIM` is 620, adding a
10-dim action-param block — one hidden layer of 128 `tanh` units, linear scalar output) showed:

| Measurement | Value | Meaning |
|---|---|---|
| output-layer `mean\|W_out\|` | **0.0055** (≈0) | the readout is nearly zeroed |
| net output std over 300 varied inputs | **0.049** (range [−0.07, 0.20]) | output ≈ constant, vs targets that span ±13 |
| hidden-activation std | **0.64** | the hidden layer *does* vary with state |
| field-input weight vs base | ratio **1.007**, 0/184 dead | the live 22-player+ball vector **is** read |

So the input features (including the whole-field motion vector) are encoded and weighted normally,
and the hidden layer varies with state — but the **output projection was driven to zero**, crushing
all that variation into a constant ~0.044. The value head is **action-blind**.

**Why this produces exactly parity:** a flat value cannot rank the candidate actions, so the
authoritative decision degenerates to the analytic candidate order. The net contributes ~nothing,
so play equals the analytic baseline — parity, by construction.

## 3. Root cause: low-variance targets make "predict the mean" loss-optimal

The value target is `target = (reward + γ·max_next) / target_scale`, with a fixed
`target_scale = 30`. After whole-game blending, discounting, and that ÷30, the per-sample targets
**cluster near a constant** (~0.04). When targets have near-zero variance, the loss-minimizing
solution is literally a **constant output** — so gradient descent drives the readout weights toward
zero and parks the bias at the mean. The tiny `averageLoss` is achieved by predicting that mean.

The net didn't fail to *see* the state; it was **trained into not bothering to use it**, because
the targets never rewarded discrimination.

## 4. The fix: value-target standardization (this is a *neural-net* fix)

**Recenter + rescale each training batch's value targets to ~zero-mean / unit-variance** before the
loss. Now "predict the mean" yields *high* loss, so the optimizer is **forced to encode
state-dependent value** — the output layer un-collapses. Because ranking is invariant to an affine
transform of the value, decisions are unaffected by the rescale itself; only the *learning signal*
changes.

- Gate: `DD_SOCCER_ENABLE_TARGET_STANDARDIZATION` (default-off ⇒ byte-identical).
- Code: `soccer_standardize_sample_targets()` in `src/des/general/soccer/world.rs`, applied in
  `neural_training_samples_filtered` (no-op for tiny/degenerate batches).

**Evidence it works (mechanistic):** with standardization on, the output layer un-collapses within a
few thousand steps — `mean|W_out|` climbs **0.0055 → 0.10+** (toward the ~0.14 that yields
unit-variance outputs) and `averageLoss` rises **0.0003 → ~0.24** (the net is now genuinely fitting
state-dependent targets). A matched baseline arm with the gate off stays collapsed
(`mean|W_out|`≈0.01, loss≈0.0003).

## 5. Approximate DP is a **complement**, not a replacement for the net

It is tempting to read this as "we replaced neural nets with dynamic programming." That is **not**
what happened.

- **Approximate DP** here means improving the *target signal* the net learns from: restoring Bellman
  bootstrapping (`r + γ·V(s')`) instead of a done-flagged whole-game Monte-Carlo return, plus
  optional n-step / value-iterated abstract-state bootstrapping (`DD_SOCCER_ENABLE_DP_BOOTSTRAP`).
  It makes the targets **more state-dependent and lower-variance-in-noise / higher-variance-in-
  signal** — i.e. better credit assignment.
- **DP alone did not fix the plateau.** In a controlled A/B, the DP-only arm stayed **collapsed**
  (`mean|W_out|`≈0.008). Better targets don't help if the net can still win by predicting their mean.
- **Standardization is what un-collapses the net.** DP then makes the signal it's forced to fit
  *richer and more correct*.

So the relationship is: **DP sharpens the target; standardization forces the net to use it.** They
compose; neither is a substitute for the neural value function.

## 6. The role of neural nets — central now, and central later

The neural value head is the **whole point** of this architecture, and remains so:

- The legacy **tabular** Q-state aliases wildly different 22-player configurations onto the same
  bin — the original "long-standing learning plateau." The neural value head exists specifically to
  condition on the **full canonical 22-player + ball motion vector**, which the tabular key cannot.
- The collapse made the neural net *behave* like a constant, throwing that advantage away. The fix
  **restores** the net's ability to discriminate states — it does not move away from neural learning.
- **What's "later":** once the value discriminates states, the next (milder-than-feared) ceiling is
  the **action interface**. Note the interface is *not* a hard top-4 cap: when a learner exists the
  decision **injects every legal `SOCCER_POLICY_ACTIONS` macro family (113-entry table)** as a candidate
  ([world.rs](../src/des/general/soccer/world.rs) ~L5623), so the net's *macro policy* is
  expressive. Only the **specific factored tabular candidates** (a concrete
  `pass|tgt:7|spd:7|dir:135…`) are capped at 4 (`default_soccer_neural_blend_candidates`, now
  env-tunable via `DD_SOCCER_NEURAL_BLEND_CANDIDATES`), and the *fine execution* of a freshly-chosen
  macro comes from MPC/analytic. So the residual gap is "net chooses the family; execution stays
  analytic," plus exploration to fill the tabular set with better-than-analytic factored actions
  (curiosity/novelty). Beyond that, a real permutation-invariant encoder (GNN / self-attention over
  the 23 entities, replacing the hand-crafted "relational attention readout") is the larger upgrade.
  All *more* neural, not less.

## 7. Status & how to run

- **Un-collapse: confirmed, via two independent routes.** (1) target standardization (this doc);
  (2) **Bellman-terminal restore** (proper `r+γ·V(s')` bootstrapping instead of a done-flagged
  whole-game MC return). Cross-checked on two machines: Bellman-restored nets show `mean|W_out|`
  ≈0.20–0.25 (un-collapsed) while pre-Bellman/unnormalized nets sit at ≈0.005–0.01 (collapsed). The
  DP-bootstrap sweep *alone* did **not** un-collapse — standardization or Bellman terminals is what
  does it.
- **Eval climb above parity: necessary-but-not-yet-confirmed-sufficient.** The now-discriminative
  value must still learn the *correct* ordering. Measured by held-out eval vs the analytic engine
  (`SOCCER_EVAL_ANALYTIC_FIELD=1`), bar = sustained above ~0.55 with Wilson lower bound > 0.5. Early
  evidence is mixed-to-encouraging (an un-collapsed arm trended 0.49→0.60 as it trained past the
  un-collapse) but not yet past the noise floor of 40-game evals — do not declare a climb on a
  single lucky eval.

```
# value-target standardization (the un-collapse fix)
DD_SOCCER_ENABLE_TARGET_STANDARDIZATION=1
# approximate-DP target bootstrapping (complementary target-quality lever)
DD_SOCCER_ENABLE_DP_BOOTSTRAP=1
# strongest combined training arm = both on, authoritative value blend
DD_SOCCER_NEURAL_AUTHORITATIVE_LAMBDA=8
```

**One-line summary:** the plateau was a *collapsed neural value head* (constant output from
low-variance targets); **target standardization** un-collapses it; **approximate DP** complements it
by sharpening the targets; the neural net stays central throughout.
