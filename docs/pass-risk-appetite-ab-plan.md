# Role-aware pass risk/safety appetite: validation & A/B plan

The feature (local, uncommitted) quantifies the risk/safety of a pass and prices it by **who
is on the ball and where**: defenders price the (already-computed) MPC/POMDP lane-interception
risk *higher* (lean safe), forwards price it *lower for a forward ball* and get a forward-
preference lift, escalating in the opponent's final third; sideways/backward balls keep ~full
risk pricing so braveness never becomes a loose square ball.

Source of truth:
- Gate + appetite model: `src/des/general/soccer.rs`
  - `dd_soccer_enable_role_pass_risk_appetite()` — env `DD_SOCCER_ENABLE_ROLE_PASS_RISK_APPETITE`,
    **default-OFF, byte-identical when off** (env `is_ok()` in both `cfg(test)` and prod).
  - `PassRiskAppetite` + `pass_risk_appetite_for_passer()` (gated) → `pass_risk_appetite_table()` (pure).
- Call sites: `src/des/general/soccer/world.rs` — `ranked_pass_targets_filtered_full`
  (floor) and `ranked_aerial_pass_targets_filtered_full` (aerial): the appetite scales the
  interception-risk penalties (direction-aware) and lifts the forward-progress weight.

Unit-level behaviour is already proven (`role_pass_risk_appetite_table_leans_defenders_safe_and_forwards_brave`).
What is **not** yet proven is the *match-level* effect: a single live game is confounded by
field position (forwards are advanced with few forward outlets; defenders have the whole pitch
ahead), so raw per-role pass-direction counts cannot isolate the appetite. This doc defines the
deterministic A/B that can.

---

## 1. Hypotheses to test (pre-registered)

The treatment (gate ON) vs baseline (gate OFF), same seeds, should move these in the stated
direction. Each is paired with the metric that measures it and a guard metric that must NOT
regress.

| # | Hypothesis | Primary metric | Guard (must not regress) |
|---|-----------|----------------|--------------------------|
| H1 | Forwards take more **forward** balls in the final third | forward-pass share for `role=Forward` in attacking third ↑ | overall pass completion ≈ flat |
| H2 | The forward balls forwards attempt are **riskier** (higher appetite, not just more) | mean lane-interception-risk / lower expected-completion on forward passes by forwards ↑ | forward-pass *completion* not collapsing (≥ ~baseline − 5pp) |
| H3 | Defenders make **safer** passes | defender mean interception-risk ↓; defender backward/sideways share ↑ slightly under pressure | defender progressive-pass yards not collapsing |
| H4 | Net attacking output is **same or better** | goals / shots / xG-proxy, completed-pass gain yards in opp half | turnovers conceded in own half not ↑ |
| H5 | No discipline regression | own-half interceptions-conceded ≈ flat | possession retention ≈ flat |

"Correct" = H1–H3 move the right way **and** H4/H5 guards hold. If forwards get braver but
completion / own-half giveaways blow up, the appetite numbers are too aggressive and need
retuning (the `pass_risk_appetite_table` constants) before promotion.

---

## 2. Why deterministic two-process A/B

The gate is resolved **once per process** (`OnceLock`), so the two arms cannot differ inside
one process — each arm is its own process:

- **Baseline arm:** env unset → gate OFF → byte-identical to pre-feature engine.
- **Treatment arm:** `DD_SOCCER_ENABLE_ROLE_PASS_RISK_APPETITE=1` → gate ON.

Same seed set for both arms. The formation LP is forced through the deterministic simplex and
the per-process Clarabel solver seeding makes a given `(seed, ticks)` match byte-identical, so
the arms differ **only** by the gate. Matches diverge after the first decision the appetite
changes (expected — that *is* the treatment); shared seeds make the comparison fair in
aggregate, not frame-locked.

This is a **decision-side** change (pass ranking), not a reward-shaping change, so the right
harness is `measure_offense` (heuristic policy, sees decision changes), **not**
`measure_learning_ab` (which is for learning-signal changes). The live :5055 server runs the
*learned* policy, but the heuristic ranking still generates the pass candidate set the actor
chooses among, so the decision-side A/B is the correct first gate. A learned-policy retrain +
`soccer_eval_gate_run` is the *second*, later gate (Section 6).

---

## 3. Tooling

### 3a. Quick directional pass — `measure_offense` (NO new code)

`cargo run --release --bin measure_offense -- [ticks] [seeds]` already tallies the chosen-action
histogram (pass / killer-pass / wall-pass / …) and final scores per seed, deterministic by
seed `0x5EED_0000 + s`. First, cheap signal:

- Run baseline and treatment over the same `seeds`, diff the action histogram and goals.
- Expectation: `pass` / `killer-pass` / forward-oriented mechanics tick up slightly under
  treatment; goals flat-or-up; no explosion in turnovers-adjacent labels.

Limitation: it has **no per-role pass-direction or per-role risk breakdown**, so it can confirm
"more forward intent, no collapse" (H4-ish) but **cannot** test H1–H3. Hence 3b.

### 3b. Role/risk KPIs — new `measure_pass_risk_appetite` binary (the actual code to write)

A new `src/bin/measure_pass_risk_appetite.rs` modelled on `measure_offense`: run N deterministic
matches, and for every **on-ball pass decision** record, keyed by **passer role × pitch zone
(own-third / middle / final-third)**:

- count of passes; forward / lateral / backward split (reuse the same `chosenForwardYards` /
  `isBackwardPass` semantics already in the decision trace);
- mean **lane-interception-risk** of the chosen pass (the quantity the feature reprices) and
  mean **expected-completion** (risk taken);
- realized **completion** of the chosen pass and whether it was **intercepted** (turnover);
- progressive yards gained.

Plus team-level: goals, shots, completed-pass gain yards in opp half, own-half interceptions
conceded. Output a compact table per role×zone so baseline vs treatment diff directly.

Implementation notes:
- Reuse the engine's existing per-decision fields (the live decision-trace already exposes
  `role`, `isPass`, `chosenForwardYards`, `isBackwardPass`, `expectedPassCompletion`,
  `bestPassReceiverOpenness`, `yardsToGoal`) — tally them in-process rather than over HTTP.
- Pull lane-interception-risk from the chosen pass's `pass_target_quality_for_snapshot` /
  `pass_lane_interception_risk` so H2/H3 have the actual repriced quantity, not a proxy.
- Resolve realized interception from the match event/stat stream (the `summary.stats`
  `passInterceptions*` counters, attributed to the passer when possible).
- Gate read once per process ⇒ A/B = two runs, env unset vs set, same seeds — identical to
  the `measure_learning_ab` pattern.

### 3c. Wrapper — `scripts/run_pass_risk_appetite_ab.sh`

Mirror `scripts/run_outcome_ab.sh`: build `--release` once, run baseline then treatment over a
fixed seed range and tick budget, write both tables to `/tmp/`, and print a side-by-side diff
with PASS/FAIL against the Section 1 criteria.

---

## 4. Run recipe

```
# 0. build once
cargo build --release --bin measure_offense --bin measure_pass_risk_appetite

# 1. quick directional (existing tool), e.g. 20k ticks x 6 seeds
cargo run --release --bin measure_offense -- 20000 6 > /tmp/mo_baseline.txt
DD_SOCCER_ENABLE_ROLE_PASS_RISK_APPETITE=1 \
  cargo run --release --bin measure_offense -- 20000 6 > /tmp/mo_treatment.txt
diff <(grep -A40 'ACTION HISTOGRAM' /tmp/mo_baseline.txt) \
     <(grep -A40 'ACTION HISTOGRAM' /tmp/mo_treatment.txt)

# 2. role/risk KPIs (new tool) — same seeds both arms
cargo run --release --bin measure_pass_risk_appetite -- 24 3 > /tmp/pra_baseline.txt
DD_SOCCER_ENABLE_ROLE_PASS_RISK_APPETITE=1 \
  cargo run --release --bin measure_pass_risk_appetite -- 24 3 > /tmp/pra_treatment.txt

# 3. or just:
scripts/run_pass_risk_appetite_ab.sh
```

Sample size: start ~24 matches × 3 min (≈ the `measure_learning_ab` default) for a same-day
read; bump to 60+ if H1–H3 are directionally right but inside the noise band.

---

## 5. Decision rules

- **All H1–H3 right + H4/H5 guards hold** → promote: flip the gate to `gate_default_on`
  (default-ON prod, still OFF under `cfg(test)`), then schedule a learned-policy retrain so the
  network learns *with* the new candidate ordering, and re-gate via `soccer_eval_gate_run`.
- **Directionally right but completion / own-half giveaways regress** → retune
  `pass_risk_appetite_table` (most likely soften `Forward` final-third
  `forward_risk_penalty_mult` 0.66 → ~0.75, or raise `sideways_risk_penalty_mult`) and re-run.
- **No movement** → the appetite multipliers are too small relative to the other ranking terms;
  increase the spread (defender > 1.3, forward < 0.66) and/or the `forward_preference_lift`.
- **Worse** → leave default-OFF; keep as an opt-in env for experimentation.

Keep the gate **default-OFF** until Section 5 promote criteria are met. The live :5055 viewer
can run with the env set for eyeballing, but that is not the verdict.

---

## 6. Second gate (later): learned-policy A/B

Because the live policy is the learned actor, a full verdict also wants:
1. Train two brains with `measure_learning_ab` / the learner, arms = gate off vs on, same seeds.
2. Freeze and play head-to-head over a **held-out** seed range via `soccer_eval_gate_run`
   (Wilson-bounded Elo verdict, no hard counter).
This is the same train→freeze→held-out-eval discipline as `run_outcome_ab.sh`, and is what
gates the candidate before it replaces the incumbent.

---

## 7. Status

- [ ] 3a quick directional A/B with `measure_offense` (existing tool)
- [ ] 3b write `measure_pass_risk_appetite` binary
- [ ] 3c write `scripts/run_pass_risk_appetite_ab.sh`
- [ ] Run Section 4, fill Section 5 verdict
- [ ] If promote: flip to `gate_default_on` + retrain + `soccer_eval_gate_run`
