# How to actually climb past the 0.53 ceiling — Claude ↔ Codex conversation (2026-07-07)

A worked design conversation between Claude (orchestrator, `maca5`) and Codex (repo-grounded
reviewer, via the LAN bridge at `192.168.100.19:8765`) about why the neural learner is pinned
just under the confidence floor and what actually moves it. Companion to
[climbing.md](climbing.md), [fix-plateau.md](fix-plateau.md), and
[learnability-conversion-roadmap.md](learnability-conversion-roadmap.md).

This is the actionable record. Read the ladder in "Verdict" and run it.

---

## The evidence we started from

Local neural-first climb, authoritative on-policy. Prior work already stacked (neural
authoritative λ=1.0 / warmth persisted / dense per-action reward verified reaching the value
net / on-policy training) — that took us from **losing every game (−109) to ~even**. Now:

- **confirm-eval, 200 held-out games:** 67W-**91D**-42L, GF/GA 117/82 (GD +35), held-out
  Elo Δ **+66.2** (1553 vs 1487), cross-play mean payoff vs field **0.563**, worst-case 0.53.
  **Wilson lower bound 0.493 < 0.500 floor → REJECT.**
- validation loop (40-game windows) bounces 0.46–0.66, mostly REJECT; self-play league
  win-rate oscillates ~.500. Newest 200 local games: 232 goals, **40% draws**, 4.46 shots/game,
  1.895 SOT/game.

**Signature: a draw-heavy, low-EPV-separation stalemate.** Elo is clearly positive (+66) but
cross-play payoff is stuck at ~0.53–0.56 and can't clear Wilson 0.5. Elo-positive +
payoff-flat = the net orders choices a bit better but produces no per-game *separation* to
convert to confident wins. Matches the prior finding (AWR/PPO/offline-PPO all topped out ≤0.53)
that the ceiling is **structural (reward/interface), not algorithmic**.

## The debate

Four candidate levers, ranked by Codex for moving **true cross-play payoff** (not Elo):

1. **B — Expand the action interface** (selective off-ball SUPPORT runner). Highest leverage.
   The learner is a *selector over analytic candidates*; it can re-rank on-ball choices but
   cannot manufacture the off-ball conditions (overloads, timed runs, defensive displacement)
   that make a choice decisive. If the other 21 players are scripted, no on-ball reward tweak
   creates separation.
2. **A — Raise the reward ceiling** (`pitch_value` PBRS, `DD_SOCCER_ENABLE_PITCH_VALUE_REWARD`).
   Useful but won't break the ceiling alone: PBRS is policy-invariant in the idealized limit,
   so its upside is "find good on-ball behavior faster," not "create a new actuator." If it
   improves territory but not conversion it can *deepen* the draw.
3. **D — Opponent ladder.** Diagnoses (self-play mirror cancellation inflates draws); as
   training it can avoid a sterile equilibrium, but evaluating broader doesn't add capability.
4. **C — Lower eval variance** (frozen pool + more games). Necessary for decision quality —
   200 games at worst-case 0.53 is underpowered for Wilson>0.5 — but not a ceiling mover.

**Diagnosis: more interface than reward.** Dense reward + on-policy authority got us to
competent/even, then every method saturates at the same payoff — the signature of a learner
that has exhausted the available on-ball improvements and is waiting on scripted off-ball play
to present chances.

### Repo facts that reshaped the plan (Codex, grounded)

- The **off-ball support runner already exists**, gated off, rewarded by **territorial
  advantage delta** — `support_scorer.rs`, `run_prediction_decision.rs`, samples built in
  `world.rs:9205`. So B is not "build a runner"; it's "reward it correctly + train it
  asymmetrically + flip the gates."
- `pitch_value` is already **control × expected-threat** (xT-ish), *not* raw territory — but a
  **hardcoded xT seed**, not calibrated (`pitch_value.rs:402`). Learned Markov xT is the named
  next step (`learnability-conversion-roadmap.md:57`).
- Shot-quality ingredients exist (`shot_on_frame_probability`, `shot_beat_goalkeeper_probability`,
  `shot_block_probability`, `shot_quality`, `soccer.rs:7589`) but **no calibrated xG estimator**.
- The interface is **113 actor labels, not 73** (`soccer.rs:41785`): first 73 are base families
  through `support-push-up`; **73–112 are kick-power buckets**. A "fixed-interface" falsifier
  **must mask to the first 73**.
- Support/run heads are installed **match-wide, not per-team** (`world.rs:9263`, `world.rs:54905`,
  `long_pass_run.rs:477`); `TeamBrain` carries neural/target-Q/options/genome only
  (`tournament.rs:132`, `:1504`). So "learned runner vs frozen-analytic off-ball opponent in one
  match" is **not turnkey**.

## The unifying idea (the real bet)

The draw stalemate is a low-EPV-separation equilibrium, and *both* the carrier's shaping and
the runner's reward currently optimize **territory**, which deepens draws. Fix: build **one
learned EPV potential** `Φ_epv(s)` — a Markov/logistic expected-possession-value over pitch
cells + pre-shot states, fit offline from possession chains labeled by {shot, SOT, goal,
turnover} — and use it in both places:

- **on-ball / global reward:** full PBRS `γΦ_epv(s') − Φ_epv(s)` (replaces the hardcoded xT seed)
- **off-ball support/run reward:** **NOT** the raw shared delta — the **marginal/residual**
  contribution: observed `ΔΦ_epv` minus *expected* `ΔΦ_epv` for that situation (counterfactual
  over the analytic-baseline target). Keep spacing / receivability / onside as **hard guards**.

**Why not the same raw delta twice:** PBRS telescoping only conserves the *one* stream it's
added to. Reusing the same delta as a second (support-head) objective is not "free
conservation" — it becomes an auxiliary objective that credits non-causal off-ball players for
team EPV movement → biases toward "everyone crowds the ball." Same potential: yes. Same raw
delta in two channels: footgun.

**Can we fit `Φ_epv` tonight? No.** The offline export is value-policy rows
(`state_key`/`action`/`value_micros`/`target_*`, `export_offline_dataset.sh:10`); config moments
carry reward/n-step-return/value, **not terminal possession outcomes** (`config_vector.rs:538`).
A `value_micros` proxy is contaminated by current shaping — rejected. `Φ_epv` needs a new
**possession-chain export** first.

## Verdict — the 3-step ladder (Codex: GO)

**Step 0 — A-falsifier (cheap, single-variable, interpretable). Run first.**
`pitch_value` ON, interface **masked to the first 73 actions** (freeze kick-buckets 73–112),
evaluated vs a **frozen diverse anchor pool** with enough games to power Wilson.
- **Pre-registered verdict number:** *Wilson lower bound on candidate mean match score vs the
  frozen field (W=1, D=0.5, L=0).* **> 0.500 = real life; ≤ 0.500 = did not beat the field.**
  Worst-case cross-play is a guardrail, not the verdict.
- Also pre-register: draw-rate DOWN, goals/game UP, worst-case cross-play UP. If territory
  metrics rise but draws stay ~0.53, the **interface hypothesis strengthens**.

**Step 1 — Cheap off-ball capability A/B (no per-team isolation needed).**
Because the heads are match-wide, run it as a **global liveness A/B on paired seeds**:
`same frozen policies + heads OFF` vs `same frozen policies + heads ON` (support scorer +
run-prediction + multimodal-run gates all on, heads warm). Read **chance creation**: SOT/game
and shots/game primary; goals/game and draw-rate secondary; box-entries only once a counter
exists. **Flat payoff with higher SOT is a valid capability signal**; flat chance-creation =
heads aren't creating danger (or aren't warm/gated right) → ceiling is elsewhere.
- Do **not** phrase it as "heads-ON side out-creates the headless side" — the code can't express
  that asymmetry in one match. That asymmetric verdict requires **per-team head isolation**,
  which must be built before any "learned off-ball beats frozen analytic off-ball" claim.

**Step 2 — Build `Φ_epv` and unify both channels** — only after Step 0 or Step 1 shows signal.

**Parallel track (GO now, independent of the falsifiers):** start the **possession-chain
export** behind a local/off-by-default flag so EPV is unblocked the moment a falsifier shows
life. Deliverable is the **data contract**, not a fitted model tonight:
`match_id, chain_id, team, tick, ball cell / pre-shot state, terminal outcome` attached to chain
state rows.

## Implementation caveats to respect

- **Fixed-interface = mask to first 73**, or the A-falsifier isn't fixed-interface (`soccer.rs:41785`).
- **No per-team head isolation exists** — Step 1 is a symmetric global A/B; asymmetric verdicts
  need isolation built first (`world.rs:9263/54905`, `long_pass_run.rs:477`, `tournament.rs:132/1504`).
- `MatchStats` has shots/SOT/goal counters (`soccer.rs:18808`) but **no first-class box-entry
  stat**, and `tournament::MatchOutcome` carries only goals/training-steps (`tournament.rs:336`).
  Read `MatchSummary.stats` via a local harness, or add the missing counters before making
  box-entries a primary metric.
- Don't fit EPV from `value_micros` — it's policy-value data, contaminated by current shaping.
- Constraints from `remaining-next-steps.md` still bind: default-OFF / byte-identical new paths,
  66 ms / 15 Hz / 22-agent budget, append new channels at the tail, A/B behind the eval gate.

## ACTION LOG — 2026-07-07 (execution)

- **Launched Step 0** as a clean single-variable A/B via `scripts/run_pitch_value_ab.sh` (reuses
  `soccer_outcome_ab_run`: two separate train processes since the gate is a per-process OnceLock,
  then frozen head-to-head over a held-out disjoint seed range printing the Wilson lower bound).
  Both arms `DD_SOCCER_ENABLE_DISCRETIZED_KICK=0` (fixed 73 interface); arms differ only by
  `DD_SOCCER_ENABLE_PITCH_VALUE_REWARD` (0 vs 1). Verified live: no `DD_SOCCER_OUTCOME_CREDIT`
  inherited (Codex's contamination watchpoint — that replay path applies pitch shaping bypassing
  the gate), clean per-process env. Artifacts: `/tmp/pitchvalue-ab/`, verdict in `verdict.log`.

- **DISCOVERY that reframes the ladder: `pitch_value` is DEFAULT-ON in release/production**
  (`pitch_value.rs:108` — "Promoted to default-ON in production"; the module's top "gated off by
  default" comment is **stale**). So the 0.53 ceiling was **already measured with pitch_value ON**.
  Consequences:
  - Lever **A is not a fresh climb lever — it's already spent in the incumbent.** The Step-0 run
    is therefore an **ablation** (does removing pitch_value hurt?), diagnostic not a climb, and an
    80-train-game null is a smoke-test null, not a trusted reject (Codex wants 240–320 games or
    3×80 seeds for a trusted verdict). Read: `pv_on ≫ pv_off` → territory reward is load-bearing,
    supports the EPV upgrade; `pv_on ≈ pv_off` → the territory channel is NOT the active
    constraint → redirect straight to lever **B**.
  - The diagnosis gets *stronger*: territory shaping is already on and we still draw at 0.53 →
    territory-without-conversion is confirmed as the wall → the real climb is **B (off-ball
    support runner) + Step 2 (learned EPV replacing the territorial signal in both channels).**
  - **Revised priority:** Step 1 (cheap off-ball capability A/B: flip run/support gates, read
    SOT/game) is now the highest-value *next* experiment, ahead of any further pitch_value work.

## ACTION LOG — 2026-07-07 (execution, continued)

**Gate-default audit — most "levers" are already spent.** Auditing release (`cfg(not(test))`)
defaults revealed the incumbent stuck at 0.53 already ships nearly everything:
- **Default-ON (spent, in the 0.53 baseline):** `pitch_value` reward, `support_scorer`,
  `run_prediction_model`, `onside_support_model`, `xt_terminal_cost`. NB their comments say they
  run an **analytic seed until a warm head is installed** — plumbed-on ≠ trained-and-contributing.
- **Default-OFF (genuinely unspent):** `DD_SOCCER_ENABLE_LEARNED_LONG_PASS_RUN`,
  `DD_SOCCER_ENABLE_MULTIMODAL_RUN_PREDICTION`, `DD_SOCCER_ENABLE_RELATIONAL_ATTENTION`.

**Consequence:** "flip a gate" is mostly *not* a lever — the off-ball support machinery is
already on. The real climb is (a) making those on-but-analytic-seed heads actually *learn*
better than their seed (needs the EPV reward, Step 2), and (b) the few genuinely-off levers.

**Killed the pitch_value ablation** (diagnostic-only of an already-on feature) and pivoted CPU to
the first real climb attempt on an **unspent** lever.

**Launched the off-ball behavioral A/B** — `scripts/run_gate_ab.sh` (reusable generic gate A/B):
- baseline = clean incumbent defaults (via `env -i`; note MULTIMODAL uses `.is_ok()` so "off"
  means UNSET, not `=0`);
- treatment = `MULTIMODAL_RUN_PREDICTION=1 + LEARNED_LONG_PASS_RUN=1`;
- both fixed-73, 80 train / 140 held-out eval, eval in the lever-active world.
- Pre-registered: Wilson lower bound(treatment vs baseline) > 0.500 = the lever climbs; secondary
  reads = goals/game and draw-rate. Artifacts `/tmp/gate-ab-offball/`, verdict in `verdict.log`.
- Caveat (Codex): no per-team head isolation, so eval applies the lever to both brains — this is a
  detection test ("did training WITH the lever build a stronger policy for the lever-on world"),
  not a diverse-field promotion proof.

**Queued next if this is flat:** relational-attention + capacity (own run, larger train budget,
since representational upgrades need more data), then the EPV possession-chain export (Step 2).

## One-line summary

The ceiling is structural: the net is a *selector over analytic candidates* optimizing
**territory**, and territory without conversion draws. Prove it cheaply (Step 0 A-falsifier,
Wilson>0.5 vs frozen field; Step 1 global heads-ON/OFF SOT A/B), then raise the real ceiling by
replacing the territorial signal — in both the on-ball PBRS and the off-ball residual reward —
with **one calibrated learned EPV potential**, which first requires a possession-chain export.
