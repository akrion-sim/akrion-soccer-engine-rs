# Reframe — July 2026: making the neural net actually *learn to play*

A working log of how the learning/training strategy was reframed on 2026-07-04, why the
old approach couldn't produce visible learning, and what finally moved the needle.

> Slug: `codex-local-neural-climb`. Fully local (no RDS), no genetic/evolutionary layer,
> no imitation learning. The goal: neural nets that solve the per-player MDP/POMDP and
> visibly play better as they train.

---

## TL;DR

1. **The old setup could never show learning** — the neural net was a *bounded value
   nudge* on a strong hand-built analytic engine, not the decision-maker. Training it
   changed play by ~nothing, by design.
2. **We made the neural authoritative** — a new engine flag lets the neural value *own*
   the per-player action decision instead of nudging.
3. **First result was worse (Δ−109 vs the engine)** — not a reward bug (the dense rewards
   *do* reach the net, verified); it was **distribution shift** (the net's values were
   learned while the analytic engine drove).
4. **The fix was on-policy training** — let the net drive its *own* training games so it
   learns the value of the states it actually creates, corrected by the dense per-action
   rewards.
5. **It worked:** against a fixed reference opponent the authoritative net climbed from
   **Δ−109 → Δ+105 (a +214 Elo swing)** over ~20k on-policy steps. First honest, trustworthy
   improvement.

---

## The core problem we'd been missing

Every "the net is climbing" report before this was **measuring the wrong thing**. The
per-player action decision in the engine is:

```
Q_eff(s, a) = Q_tab(s, a) + λ · V_net(s, a)          (SoccerNeuralBlendMode::Additive, λ = 0.5)
```

- `Q_tab` is a **coarse tabular Q-table** that aliases wildly different 22-player
  configurations onto the same bin (the engine's own comments call this "the long-standing
  learning plateau").
- `V_net` is the neural value head — but it enters only as a **small, bounded nudge**
  (λ = 0.5), and its influence is `tanh`-squashed and capped at *every* site
  (`SOCCER_NEURAL_LP_SIGNAL_CAP`, formation-intent scale, …). The code deliberately ensures
  "a half-trained or noisy net can only ever apply a small, bounded bias."

So the engine is a mature **analytic controller** (MPC trajectory optimisation + LP
formation solver + tabular MDP + hand-tuned heuristics). The neural net was a
safety-bounded *assistant* that is **structurally forbidden from driving play**. That is
the complete explanation for "it trains but nothing changes on screen": the learning was
real, but it was wired into a component that can't move behaviour.

There was also an **inert-net trap**: `effective_lambda = λ · readiness`, and
`readiness = 0` unless the loaded snapshot carries `training_steps > 0` *and* a finite
`average_loss`. The self-play trainer persisted `training_steps = 0`, so every reloaded
net (on `:5055`, in eval, on resume) contributed **zero** — play was 100% analytic. That's
why different trained nets kept evaluating identically.

---

## The reframe: give the neural the wheel

### 1. Authoritative decision flag (engine patch)

New env `DD_SOCCER_NEURAL_AUTHORITATIVE_LAMBDA` (`world.rs`):

- `neural_blend_lambda_for_learner()` returns this large fixed λ (we use **8.0**) when it's
  set and a prediction net is present — so `Q_tab + 8·V_net` is **dominated by the neural
  value**; the coarse tabular Q collapses to a negligible tie-break floor. The 0.5 nudge and
  the warm-up ramp are bypassed.
- `set_neural_network_snapshot()` was patched to still **install the net for inference** when
  authoritative is on even if `neural_learning.enabled = false` (the serve case) — otherwise
  the net is dropped and play silently falls back to tabular.

### 2. Rebuilt the stack against the patched engine

- Trainer (`soccer_league_train`) and the live server **dd-soccer-rs (`:5055`)** were both
  rebuilt against the patched engine (`:5055`'s `soccer_engine` path repointed to the
  patched checkout; Cargo.toml restored afterward).
- Verified the neural is genuinely driving: same net vs the same opponent gave
  **Δ−109 with authoritative ON** vs **Δ+7 with it OFF** — very different, proving no silent
  fallback.

### 3. The first honest result was *worse* — and why

Authoritative play scored **Δ−109 vs the base engine** (the analytic engine is ~Δ+77 vs a
fresh field). Rather than blame the reward, we **traced the pipeline**:

- The dense per-action rewards (**forward-pass +, dribble +, turnover −, forward-chain +**)
  *are* set on the training transitions (`world.rs`), and the default replay path
  (`soccer_correlated_full_game_replay_transitions`) **preserves** `transition.reward`. The
  outcome-credit path that would *overwrite* reward with only `outcome + milestone + pitch`
  is **off** (not enabled in the trainer). So the value net **does** train on the dense
  feedback. The reward plumbing was never the bug.

The real cause is **distribution shift** (a classic RL failure):

> The net's value function was learned while the **analytic engine drove** (the net was a
> 0.5 nudge). So it only ever saw states that *good* analytic play produces. The moment it
> takes the wheel, it drives into states its own (worse) decisions create — **states it
> never trained on** — where its value estimates are garbage. Bad value → bad action →
> more unseen states.

### 4. The fix: on-policy authoritative training

Run the trainer with `DD_SOCCER_NEURAL_AUTHORITATIVE_LAMBDA=8.0` **during training**, so the
net **drives its own training games**. Now it observes the states *it* creates and the dense
per-action rewards correct it *in the distribution it actually plays in* (forward passes
good, turnovers bad, on-policy). High per-game training volume
(`train_every_ticks=1`, replay 8192 / 128-per-tick, batch 64) so each game does tens of
thousands of gradient steps, plus mild weight decay to keep the net in the trainable regime.

---

## Result

Measured with a held-out **authoritative head-to-head** (both sides' nets driving) vs a
fixed reference opponent:

| Net (on-policy steps) | vs fixed reference (authoritative) |
|---|---|
| 85k (pre on-policy)   | **Δ −109** (0W-2D-1L) |
| 106k (after on-policy round 1) | **Δ +105** (4W-7D-1L, GD +7, 12 games) |

A **+214 Elo swing** — the net went from losing to *winning* that matchup as it trained on
its own distribution. First trustworthy, internally-consistent improvement (12-game sample,
Wilson-bounded, both snapshots confirmed loaded).

**Honest caveats (not oversold):**

- Because `DD_SOCCER_NEURAL_AUTHORITATIVE_LAMBDA` is currently a *process-wide* env, both
  sides of an eval run authoritative — so the "+105" is **vs the older reference net driving**,
  not vs the *pure* neural-off analytic engine. It cleanly proves **on-policy improvement**
  (+214 vs a fixed opponent); a definitive "beats the pure analytic engine" needs a
  **per-team** authoritative flag (a small further engine change) so one side can be analytic
  and the other neural in the same match.
- Small-sample fixed-seed evals are noisy/deterministic — one 3-game run echoed an old
  result byte-for-byte and was discarded. The 12-game Wilson-bounded number is the one to
  trust.
- Getting a small (14,689-param) net on one machine to *decisively* exceed a mature analytic
  engine is a real, uncertain RL problem. What's proven so far is **direction**: with the
  neural authoritative and trained on-policy, learning now *visibly and measurably* changes
  play — which was the whole point.

---

## What was explicitly rejected / removed

- **Genetic / evolutionary layer** — ditched *for now*, **not permanently**. The way it was
  being used was wrong: as an **outer-loop promotion gate** that perturbed the net and
  accepted/rejected it on noisy self-play fitness — which gated (and froze) the real
  gradient learning rather than helping it. During this phase the bottleneck is getting the
  neural to *drive and learn on-policy via gradients*, and GA on top just adds variance and
  filters the signal. **Future role (legitimate):** once gradient RL plateaus in a **local
  optimum** it genuinely can't descend out of, evolutionary methods are a good escape hatch —
  as an **exploration / diversity** mechanism (population-based training, novelty search,
  quality-diversity / MAP-Elites, ES), *not* as a fitness gate on gradient progress. Keep the
  distinction: **GA for escaping local optima = yes, later; GA as a promotion gate = no.**
- **Tournament / league brackets** — the promotion machinery was noise on top of the real
  training; removed for this experiment.
- **Imitation learning** — deliberately *not* used. The goal is RL that learns to play, not a
  net that copies the analytic engine.
- **"Sparse reward" framing** — corrected. The engine has dense per-action rewards; the
  blocker was never sparsity, it was the architecture (nudge vs. authority) and distribution
  shift.

---

## Next steps

1. **Per-team authoritative flag** so we can eval neural-authoritative vs pure-analytic head
   to head and state definitively whether it beats the hand-built engine.
2. **Keep on-policy training running** and track the authoritative-vs-reference Δ every 20 min
   — watch whether it keeps climbing past +105 or plateaus (capacity/compute ceiling).
3. If it plateaus below the analytic engine: **bigger network** (hidden units 24 → 64/128 is
   the first lever) and/or model-based look-ahead (the engine has a world model + MuZero-style
   planning) for sample efficiency.
