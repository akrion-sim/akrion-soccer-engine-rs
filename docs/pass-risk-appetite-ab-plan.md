# Role-aware pass risk/safety appetite: validation & A/B plan

> **⚠️ STATUS UPDATE (verified against HEAD `be11b4d`, 2026-07-09): PROMOTED.** The gate is now
> **default-ON in production** — `dd_soccer_enable_role_pass_risk_appetite()` uses
> `gate_default_on("DD_SOCCER_ENABLE_ROLE_PASS_RISK_APPETITE")` in the `#[cfg(not(test))]` branch
> (soccer.rs:63420; doc-comment "Default-ON in production now … default-OFF under test",
> soccer.rs:63408-63410). `=0`/`off` is the kill switch; still default-OFF under `cfg(test)` so unit
> tests stay byte-identical. **Section 7's "DO NOT PROMOTE" verdict and the Section 8 "if promote"
> checkbox are superseded** — the promote action was taken. The A/B plan below is preserved as the
> validation record. (Also landed: `SOCCER_SEED_VARIED_SKILLS` addresses next-step #1 match
> perturbation, soccer.rs:63424-63427.)

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

## 7. Results — run 1 (2026-06-28)

Tooling built: `src/bin/measure_pass_risk_appetite.rs` (3b). Ran 9000 ticks × 3 seeds per arm,
gate OFF vs ON, deterministic LP. **~3m47s per full match.**

### Critical methodology finding: the engine is seed-deterministic
`MatchConfig.seed` did **not** perturb the match — all 3 seeds produced **byte-identical**
matches (counts exactly 3×, every mean identical to 3dp, all 1–1). So this run is effectively
**one scenario**, not 3 independent samples. It tests **direction**, NOT significance. A real
verdict on goals/turnovers (H4/H5) needs a perturbation that actually varies the match (jittered
initial positions / kickoff). **This is the #1 follow-up before any promote decision.**

### Baseline (OFF) → Treatment (ON), by passer role × zone

| role \| zone | fwd% | avgFwdYds | avgComp (inverse-risk) | back% |
|---|---|---|---|---|
| Defender \| own | 99→99 | 31.3→27.3 | 0.669→0.675 | 0→1 |
| Defender \| mid | 97→97 | 17.3→23.5 | 0.485→**0.527** | 3→1 |
| Midfielder \| own | 86→**59** | 12.3→7.7 | 0.388→0.401 | 12→**37** |
| Midfielder \| mid | 74→80 | 9.8→**16.1** | 0.392→0.350 | 23→16 |
| Midfielder \| final | 67→73 | 6.2→9.8 | 0.433→**0.278** | 33→19 |
| Forward \| mid | 71→72 | 8.6→10.1 | 0.343→**0.261** | 28→26 |
| Forward \| final | 26→**22** | -3.3→-4.0 | 0.376→0.408 | 67→**78** |

Goals (one scenario): baseline 3–3 (6) → treatment 3–0 (3).

### Verdict per hypothesis
- **H2 (forward balls are riskier) — PASS.** Forward\|mid avgComp 0.343→0.261, Mid\|mid
  0.392→0.350, Mid\|final 0.433→0.278: forward balls in build-up/middle third are notably
  riskier (lower completion) under treatment. The risk-repricing works.
- **H3 (defenders safer) — PASS (weak).** Def\|mid completion 0.485→0.527 (safer), own-third
  forward balls shorter (31→27 yd). Directionally right, small.
- **H1 (forwards more forward in the final third) — FAIL.** Forward\|final went the WRONG way:
  fwd% 26→22, back% 67→78, avgFwd -3.3→-4.0. Second-order shape effect: the forward-preference
  lift pushed the **midfielders** much higher (Mid\|mid avgFwd 9.8→16.1, +64%), isolating
  forwards higher up with *fewer* outlets ahead, so they lay off backward to arriving mids more.
  The appetite can't manufacture a forward outlet that the run patterns don't provide.
- **H4 (goals same/better) — FAIL in this scenario** (6→3 total), but **n=1**, both teams share
  the gate, and the drop is asymmetric (away 3→0). **Not interpretable** without varied scenarios.
- **H5 (own-half discipline) — CONCERN.** Mid\|own back% 12→37 (much more own-third backward
  recycling). Could be safer retention or negative recycling — ambiguous without turnover data.

### Decision: DO NOT PROMOTE. Keep gate default-OFF.
The mechanism does what it says on risk *pricing* (H2/H3), but the **global forward-preference
lift has an unintended team-shape side-effect** (mids surge upfield, forwards isolated → H1
backfires) and goals dropped in the single scenario tested. Net: not a clear win.

### Next steps (in priority order)
1. **Add a match perturbation** (initial-position / kickoff jitter keyed on seed) so the A/B has
   real variance, then re-run with 20+ varied scenarios for an H4/H5 verdict with turnovers.
2. **Retune toward risk-pricing over shape-pushing:** shrink / remove `forward_preference_lift`
   (it's what shifts team shape via the mids) and lean on the direction-aware
   `*_risk_penalty_mult` instead, which gave the clean H2/H3 signal without the shape blowback.
   Re-check that Forward\|final no longer regresses.
3. Extend the harness to attribute **realized interceptions and shots/xG** by passer role (3b
   follow-up) so H4/H5 are measured, not proxied.
4. Only if a varied-scenario run shows H1–H3 right with H4/H5 guards holding → flip to
   `gate_default_on`, retrain, and gate via `soccer_eval_gate_run` (Section 6).

## 8. Status

- [x] 3a quick directional A/B with `measure_offense` — superseded by 3b (richer, role-resolved)
- [x] 3b write `measure_pass_risk_appetite` binary
- [ ] 3c write `scripts/run_pass_risk_appetite_ab.sh` (deferred — single binary sufficed)
- [x] Run Section 4, fill Section 7 verdict → **mixed, do not promote, retune + vary scenarios**
- [ ] Add match perturbation for real variance (next step #1)
- [ ] Retune (drop forward_preference_lift), re-run varied A/B
- [ ] If promote: flip to `gate_default_on` + retrain + `soccer_eval_gate_run`
