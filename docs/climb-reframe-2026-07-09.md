# Climb reframe — 2026-07-09: does the net *causally own* the outcome?

A working log of the round-23 reframe. Extends [`reframe-july-2026.md`](reframe-july-2026.md)
(the 2026-07-04 "give the neural the wheel" reframe) with the next structural question,
grounded against the code at HEAD (`feature/action-param-features-and-capacity`) and
adjudicated with Codex (round 23).

> **One-line thesis:** every reward-shaping arm since 2026-07-05 has been falsified vs the
> pure-analytic engine, so before shaping *what* the net rewards we must first prove *the
> net's score actually changes the executed action*. We have never measured that number.

---

## Where we actually are (grounded scoreboard)

- **North star:** beat the **pure-analytic engine** head-to-head (not the trained self-play
  field). See [`akrion-climb-state`] in memory and [`how-to-climb-codex-conversation.md`](how-to-climb-codex-conversation.md).
- **vs trained field:** best ratchet ≈ **0.53 mean payoff / Elo +63** with chance-quality
  reward + un-crushed critic window (`DD_SOCCER_ENABLE_CHANCE_QUALITY_REWARD=1`,
  `SOCCER_NEURAL_TARGET_SCALE=30`, `CLIP=15`, POP-ART on).
- **vs pure analytic (the real bar):** stuck in the **0.28–0.44** band, REJECT. On the
  forward-pass-completion KPI: climb-base **0.358**, fwdpass-249k **0.283**, de-timid
  arm-A **0.308** / arm-B **0.258**, pass-space **0.392 ON / 0.404 OFF** (mildly negative).
- **Carrot** (`DD_SOCCER_QUICK_FORWARD_RELEASE_REWARD_SCALE`, [soccer.rs:23980](../src/des/general/soccer.rs)):
  head-to-head **+0.23 forward-passes/game (+11%, not significant at 40 games)**; the first
  lever to nudge forward-pass *volume* the right way, but the vs-analytic FP number is **not
  yet validly measured** (an eval-format mismatch produced a byte-identical fallback; that
  result was retracted).
- **Verdict as of r18:** blunt reward-shaping is exhausted; the plateau is **STRUCTURAL** —
  the passing-volume deficit is unmoved by spatial / forward-primacy / de-timid shaping.

## What the 2026-07-04 reframe already did

The July-4 reframe correctly diagnosed *"the net trains but nothing changes on screen"* as
an **architecture** problem (the net was a bounded `Additive` nudge, λ=0.5, structurally
forbidden from driving) and fixed it by making the neural **authoritative**:

- Live server defaults to `authoritative`, λ=1.25 ([main_soccer_live.rs:476](../src/des/main_soccer_live.rs)).
- On-policy authoritative training climbed **Δ−109 → Δ+105 (+214 Elo)** vs a *fixed
  reference net* — but **never vs the pure analytic engine** (explicitly caveated in that doc).
- Its "Next steps" are now largely **done**: the per-team/asymmetric analytic eval path exists
  (`SOCCER_EVAL_ANALYTIC_FIELD=1` + `SOCCER_EVAL_CANDIDATE_PATH`), and the 64→128 hidden-unit
  capacity bump is this very branch (`feature/action-param-features-and-capacity`, action-param
  10-dim features + hidden 24→128).

So the crude version of "the net can't drive" was already addressed. The **residual**
structural question is sharper: *even authoritative, on a rich net, does the net's score
change the action that actually executes — or do the downstream analytic/execution layers
put the ball back where the heuristics wanted it?*

---

## The round-23 reframe (external analysis, distilled + corrected)

An external analysis argued the system is pinned at heuristic parity because the net is
**damped before it can causally own the outcome**. Cross-checked against HEAD, here is what
holds and what is stale:

### Holds up (genuinely actionable)
1. **Exploration is effectively OFF.** The value-weighted softmax over candidates exists
   ([policy_select.rs](../src/des/general/soccer/policy_select.rs)) but is gated off **twice**:
   `DD_SOCCER_ENABLE_STOCHASTIC_POLICY_TOPK` defaults off (policy_select.rs:47) **and**
   `boltzmann_temperature` defaults to `0.0` (tunables.rs:280). Greedy selection + mirror
   self-play is the classic no-exploration collapse. **The fix is to switch on an existing
   knob, not to build one.**
2. **There is no net-influence instrument.** Nothing measures *the fraction of decisions on
   which the net's score changed the selected action vs the tabular/heuristic argmax alone*.
   This is the single most diagnostic number in the system and we are blind to it.
3. **The analytic/heuristic disciplines may be doing the playing.** Correct — but see the
   Codex adjudication for *where* the shield actually lives.

### Stale or wrong (do not re-tread — already handled or misread)
- **"Build softmax bucket exploration"** — already built (see #1); it's just off.
- **"Coarsen the action key like the June state-key fix"** — the premise (state key coarsened
  June, action key still the full `{state, action: String}` tuple, [soccer.rs:12174](../src/des/general/soccer.rs))
  is *true*, but the causal conclusion is **backwards** for this codebase (see Codex below).
- **"Team reward share defaults to 0.0"** — wrong; training uses
  `DEFAULT_SOCCER_MAPPO_TEAM_REWARD_SHARE = 0.25` ([soccer.rs:5438](../src/des/general/soccer.rs),
  wired [main_soccer_learning_run.rs:3069](../src/bin/main_soccer_learning_run.rs)).
- **"Add DISABLE_* A/B toggles / eval vs a never-trained heuristic baseline"** — both already
  exist: dozens of `DD_SOCCER_DISABLE_*` toggles, and vs-pure-analytic **is** the north-star eval.
- **"The ~300-tick trajectory gap truncates build-up credit"** — the cap is
  `secs_to_ticks(0.3)` = **0.3 s** ([soccer.rs:5491](../src/des/general/soccer.rs)), not 300 ticks;
  worth auditing what it gates, but not the emergency implied.
- **Reward-shaping magnitude (spatial / forward-primacy / de-timid)** — all falsified below
  base per r14–r18.

---

## Codex round-23 adjudication (grounded)

1. **Gate direction confirmed backwards.** In `ConfidenceGated`, under-visited rows blend
   *toward the net*; high-visit rows revert to tabular ([world.rs:15869](../src/des/general/soccer/world.rs)).
   So coarsening `SoccerQActionKey` would make rows cross `min_confidence_visits` sooner and
   **reduce** value-head authority, not increase it. **Action-key coarsening is rejected.**
   Caveat: the default blend config is **`Additive`**, not `ConfidenceGated`
   ([soccer.rs:18419](../src/des/general/soccer.rs), :18527), unless
   `SOCCER_NEURAL_BLEND_MODE=confidence` — and the live path is `authoritative` — so the gate
   is largely moot for real runs anyway.
2. **The real shield is downstream of the blend** — executable-plan filters, MCTS pass-margin
   filtering, DP/retrieval/MPC bonuses, `mpc_reconciled_learned_plan`, player-layer viability
   gates, option-score safety replacement, and movement discipline. Anchors:
   scorer filters [world.rs:15833](../src/des/general/soccer/world.rs), MCTS/filter selection
   [world.rs:15961](../src/des/general/soccer/world.rs), plan reconciliation
   [world.rs:16605](../src/des/general/soccer/world.rs), learned-plan execution/veto
   [player.rs:10129](../src/des/general/soccer/player.rs).
3. **Sequencing endorsed:** pause the carrot as the primary experiment → **instrument first**
   → get a baseline → **then** flip exploration → **then** resume reward work on top. A reward
   lever cannot prove much if the final executed action rarely changes.
4. **Exploration nuance / caution:** the generic `policy_select.rs` softmax is off by default,
   but the learned-policy path in `world.rs` has its **own** deterministic rank-weighted top-k
   chooser — so one env flip does **not** exercise every selection path. Instrumentation must
   identify the *source* of each selection (sidecar, neural blend, epsilon exploration, or
   weighted tabular fallback).

---

## The plan (instrument-first)

### Step 1 — build the net-influence instrument (no behavior change)
Two measurement sites, per Codex:

- **Scorer-level diagnostic** in `neural_blended_action`
  ([world.rs:15659](../src/des/general/soccer/world.rs), :15961): capture the **tabular
  baseline label** from the original ranked legal list *before* neural vocabulary expansion,
  then compare it to the selected scored candidate *after* sort/selection. Also record:
  selected-candidate `visits < min_confidence_visits`, neural-value-present, candidate count,
  score entropy, and exact-label vs family-label deltas.
- **Headline causal metric** at final commit in the tick loop, after
  `run_time_step_with_context` and `apply_post_decision_movement_discipline`
  ([world.rs:20126](../src/des/general/soccer/world.rs), :20150): compare the baseline
  tabular label vs the final executed `intent`/decision action. `stamp_neural_mcts_selection`
  ([world.rs:14099](../src/des/general/soccer/world.rs)) is the right pattern but answers only
  the narrower MCTS-selected/executed question.

**Report:** exact-label and normalized-family net-influence separately, plus on-ball/off-ball
splits, plus the chosen-candidate score entropy per family (the exploration-collapse graph).

**Codex thresholds for final-executed family-level net influence:**
`<1–2%` dead · `2–5%` very weak · `5–15%` material-but-probably-insufficient ·
`>15–20%` high (over all decisions) · `>25–30%` high on the neural-active (≥2-candidate) subset.

### Step 2 — falsify the reframe
- If net-influence is **high** yet we still sit at parity → the "net can't own the outcome"
  thesis is **wrong**; go back to reward/credit assignment.
- If it is **~0** → thesis **confirmed**; the binding constraint is exploration + the
  downstream shield, and the next moves are: (a) flip on stochastic top-k + a small annealed
  `boltzmann_temperature`, remeasure influence; (b) selectively relax the downstream veto
  layers (via the existing `DD_SOCCER_DISABLE_*` toggles) and remeasure.

### Step 3 — resume reward work on top
Only once opportunity-conditioned pass-rate approaches analytic does the carrot / learned
receiver / target-point aim work resume (the r19 order still stands).

---

## Cross-cutting constraints (unchanged)
- 66 ms / 15 Hz / 22-agent budget; new learnable paths land **default-OFF / byte-identical**,
  A/B behind the eval gate; new variables append at the tail and stay OUT of the tabular
  `state_key` unless re-accumulating that experiment.
- Instrumentation must be pure measurement (zero behavior change) to be a trustworthy baseline.

## See also
- [`reframe-july-2026.md`](reframe-july-2026.md) — the 2026-07-04 authoritative reframe.
- [`how-to-climb-codex-conversation.md`](how-to-climb-codex-conversation.md) — running record (round 23 appended).
- [`reward-shaping-audit.md`](reward-shaping-audit.md) — the falsified shaping arms.
- [`mcts.md`](mcts.md) — the neural MCTS reranker that is part of the downstream shield.
