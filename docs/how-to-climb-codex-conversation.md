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
- The interface is **113 actor labels, not 73** (`SOCCER_POLICY_ACTIONS`): first 73 are base families
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

## ACTION LOG — 2026-07-07 (Step 2 scaffold BUILT — the EPV critical path)

Built and validated the possession-chain export + EPV fitter — the thing that was *missing*
(Codex: a calibrated EPV can't be fit from the existing value-policy export; needs outcome-labeled
chains first). This unblocks the only lever with a real path past 0.53.

- **Engine accessor** (byte-identical, pure read): `SoccerMatch::export_world_snapshot()` in
  `world.rs` — a `pub` wrapper over the `pub(crate)` snapshot constructor so external tooling can
  read possession/ball/features. No behavior change, off every hot path.
- **Export binary** `src/bin/soccer_export_possession_chains.rs` — runs self-play, segments
  possession chains (contiguous `possession_team()` control), labels each with the TERMINAL
  outcome and emits one JSONL row/tick. Data contract per row: `match_id, chain_id, team, tick,
  t_in_chain, forward (canonical 0=own goal..1=opp goal), lateral, ball_x/y, field_*,
  terminal_outcome ∈ {goal, shot_on_target, shot, turnover, timeout}, ticks_to_terminal,
  [features[210]]`. Terminal detection: goals via score delta, shots/SOT via **per-team
  `MatchStats` counter diffs** (NOT the reward-event stream — that distributes one shot's credit
  across the whole buildup chain, which over-counts SOT ~15×; this was found and fixed in the
  smoke test).
- **Fitter** `scripts/fit_epv_grid.py` — `Φ_epv(cell) = E[γ^ticks_to_terminal · value(outcome)]`
  over a forward×lateral grid (default 16×10, the pitch_value grid). Outcome values
  (goal 1.0 / SOT 0.30 / shot 0.10 / turnover −0.15 / timeout 0.0) and γ are the calibration knobs.
- **Wrapper** `scripts/export_possession_chains.sh` (export → fit in one shot).
- **Validated end-to-end.** Outcome rates realistic (2.7 goals, 9.7 shots, 46 chains/game). The
  fitted grid is a real EPV surface: mean start-forward is monotone in terminal danger
  (goal 0.895 > SOT 0.841 > shot 0.590 > turnover 0.504 > timeout 0.327), and Φ_epv rises from
  ~−0.09 in the own half to **+0.32 at the box edge**, dipping on the goal line (pinned
  possessions that don't score turn over) — textbook xT/EPV shape.

**Next (wiring, when a falsifier or this justifies it):** feed `Φ_epv` into `pitch_value.rs` as
the PBRS potential (replacing the hardcoded xT seed) and into the support/run head reward as the
**residual** contribution (observed ΔΦ − expected ΔΦ, per the double-counting fix) — both behind
default-OFF gates, A/B'd via the eval gate.

## DEEP RE-EVALUATION — 7-component audit + Codex synthesis (2026-07-07/08)

Audited every learning component (reward, value-target/normalization, net architecture,
training algorithm/credit-assignment, action interface, self-play/eval-gate, off-ball heads,
state representation) with parallel agents, then converged with Codex over 3 rounds. Root cause
is a **compound**, and — crucially — the stuck local `0.53` run already spends the "obvious"
fixes (hidden=128, relational attention on, authoritative-vs-analytic, scored shot placement,
entropy 0.05, boltzmann 0.42). The real, still-unspent blockers:

**THE VALUE TARGET IS CRUSHED (top critic-side lever).** Local `target_scale=8`, `target_clip=3`
⇒ raw-reward window is only **±24**, so the `+200` win label is already clipped to the `+3` rail
— win/draw/loss collapse to `{+3, ~0, −3}` with **no magnitude**: margin, shot-quality, rich-vs-
sterile draws, and dense shaping all alias together. PopArt does **not** fix this — it observes
the *already-clipped* target (soccer.rs:40517 runs after the clamp in soccer.rs:41527). The fix
is **raising `target_clip`** (≥26 for ±200, 33 for ±260, 50 to hold the ±400 full-game clamp)
+ PopArt on to stabilize the larger distribution. Raising clip does NOT shrink dense rewards.

**THE REWARD CAN'T DISTINGUISH GREAT FROM SAFE.** `MATCH_OUTCOME_DRAW_REWARD_POINTS = 0.0`
(soccer.rs:5569) ⇒ 45% of games (draws) carry zero outcome signal; a chance-rich draw scores
identically to a sterile one. Live path drops graded shot/goal events (`infer_discrete_events=
false`, soccer.rs:24953). Territory/pitch_value shaping is policy-invariant by construction —
mathematically cannot raise the ceiling (that's *why* territory-on still draws).

**THE GRADIENT IS NULLIFIED.** Zero-mean advantage normalization is **forced-on** with the
outcome reward (world.rs:32699) and strips the win/loss common-mode every batch → on a draw-heavy
slate the mean it removes *is* the climb signal. Can't be flag-disabled; needs a std-only mode.

### THE LOCKED INTERVENTION (built together with Codex)

**Precondition for ALL cells — un-crush the window:** raise `target_clip` (→50) + PopArt on.
Testing anything under the crushed ±24 window gives false negatives.

**Factorial (window fixed in every cell):** `{window-only, window+std-only, window+SOT-term,
window+both}`, plus one crushed baseline as a diagnostic anchor. Candidate fix = **both**.
1. **Advantage-std MODE** `{none, std-only, zero-mean}`; use **std-only** for outcome-credit
   updates (preserve common-mode sign) — decouples the forced-on link at world.rs:32699.
2. **Graded chance-quality terminal outcome term**, zero-sum + capped: `home += k·capped(
   home_sot − away_sot)`, `away −= same`, applied on draws AND wins/losses, gated. SOT-diff for
   v1 (cheap, tracked, hard to game); EPV-terminal (from the export already built) is v2.

**Gate the A/B with the CORRECTED Wilson** (empirical {1,0.5,0} variance, not Bernoulli — that
bug alone flips the existing 200-game record from 0.493 REJECT to 0.512 PASS) against a
**stronger field** (trained checkpoints + analytic, not fresh untrained nets).

**Step 0 — the decisive diagnostic (build first).** A gated dump at
`neural_policy_training_samples` (world.rs:~14815), `DD_SOCCER_DUMP_ADVANTAGE_DIAGNOSTIC=path`,
emitting per transition `{result, team, action_family, raw_reward, raw_advantage,
standardized_advantage, value, final_sot_diff, tick}`. Separates the three hypotheses:
raw labels don't separate chance-rich vs sterile draws → **reward**; raw separation vanishes
after standardization → **gradient**; high-value states lack legal expressive actions →
**interface**. Cheap, offline, decides the emphasis before the factorial.

Note: in this tree `target_scale`/`target_clip` are `SoccerNeuralLearningConfig` fields (no env
override), so the window fix needs a tiny change to expose `SOCCER_NEURAL_TARGET_CLIP` (Codex's
tree already runs 8/3; ours defaults 30/3).

## DIAGNOSTIC VERDICT + FIRST FIX BUILT (2026-07-08)

**Built the advantage diagnostic** (gated `DD_SOCCER_DUMP_ADVANTAGE_DIAGNOSTIC` in
`neural_policy_training_samples`, world.rs; byte-identical off) and ran a 50-game / 2.6M-sample
self-play burst. **Verdict — the root is the REWARD, and the advantage-norm fear was an overclaim:**

| game result | raw_adv | std_adv |  | draw by chance-creation | raw_adv | std_adv |
|---|---|---|---|---|---|---|
| Win | +3312 | +0.997 | | created MORE (sot_diff>0) | 68 | +0.066 |
| **Draw** | **+75** | **+0.003** | | even (sot_diff=0) | 111 | +0.003 |
| Loss | −3178 | −0.994 | | created FEWER (sot_diff<0) | 44 | −0.061 |

The actor signal is a near-binary win/lose rail with **draws at ~0**, and within draws the reward
is **near-zero and non-monotonic** in chances created — a chance-rich draw is indistinguishable
from a sterile one. Standardization keeps outcome-correlation high (0.87) and even faintly
*surfaces* the draw gradient, so **std-only mode is DEMOTED** — there was never a rich signal for
it to erase. The reward simply doesn't create one.

**Built the fix — gated chance-quality terminal reward** (`DD_SOCCER_ENABLE_CHANCE_QUALITY_REWARD`,
default-off byte-identical): a zero-sum, capped shots-on-target differential folded into the
match-outcome label — `home += 15·clamp(sot_home − sot_away, ±4)`, `away −= same`, on all results
(soccer.rs `MatchOutcomeReward::with_chance_quality`, wired at world.rs:14319 with `self.stats`).

**Validated it reaches the gradient** — re-ran the diagnostic with the gate on: within draws the
std-advantage spread went from ~0.13 non-monotonic noise to **+0.63 (created more) vs −0.61
(created fewer)** — ~10× stronger and correctly signed. The 45% of games that were a learning void
now teach "create danger."

**Climb A/B running** — chance-quality OFF vs ON, fixed-73, 80 train / 140 held-out, judged by the
corrected empirical-variance Wilson. Verdict pending. (Off-ball behavioral gate-flip A/B already
REJECTED decisively — Δ−159 Elo — confirming gate-flipping is dead; EPV export+grid complete as v2.)

## CLIMB FOUND — reward+window (2026-07-08, 3-arm cycle)

Ran three A/Bs concurrently (fixed-73, 80 train / 140 held-out, treatment vs baseline), judged by
the corrected empirical-variance Wilson:

| arm | record | payoff | Elo Δ | GD | read |
|---|---|---|---|---|---|
| chance-quality reward alone | 53-36-51 | 0.507 | −26 | +2 | draw-rate 45%→**26%** (stalemate broken) but only more *aggressive*, not better |
| **combined: reward + window** | **59-35-46** | **0.546** | **+63** | **+22** | **CLIMBS** — the value net can now grade the chances the reward rewards |
| interface (pass cand 3→6, cull 0.55→0) | 49-37-54 | 0.482 | −22 | −7 | slightly worse — **ruled out** |

**Decomposition:** the reward alone doesn't climb (0.507/Elo−26); adding the **un-crushed critic
window** (`target_scale=30`, `target_clip=3→15`, `PopArt on`) swings it to 0.546/Elo+63 — a +0.04
payoff and +90 Elo swing. The window-fix is doing real work: with the critic no longer saturated
at the ±3 rail, the graded chance-quality reward propagates into genuinely better valuations and
GD +22. This is the first configuration to measurably beat baseline. It doesn't yet *confidently*
clear Wilson (corrected 0.487 at 140 games) — power math: at payoff 0.546 the corrected one-sided
bound clears 0.5 at ~300 eval games; at 0.58 it clears at 140. So the robust confirmation is a
bigger effect (more training), not just more games.

**Cycle 2 running:** two high-power confirmations of the winning stack (reward+window, 160 train /
220 held-out) — one at the original seeds, one seed-varied — to widen the edge and confirm robustly.
Both use `DD_SOCCER_ENABLE_CHANCE_QUALITY_REWARD=1 SOCCER_NEURAL_TARGET_SCALE=30
SOCCER_NEURAL_TARGET_CLIP=15 SOCCER_NEURAL_TARGET_POPART=true`.

## ROUNDS 9–13 — reward-scale falsified, pivot to STRUCTURAL spatial-target (2026-07-08)

The reward+window climb (above) beats the *trained self-play field* but the frozen frontiers
still **lose to the pure-analytic engine** (0.33–0.44), so the wall moved but didn't fall. The
next rounds chase why.

**Round 9–10 — un-crush-via-scale FALSIFIED.** Raising `SOCCER_NEURAL_TARGET_SCALE` to un-crush
the critic window degraded the *policy*, not just the fit: scale=120, 202k steps, 200g vs analytic
→ mean **0.432** (Wilson 0.366), GD **−61** — worse than baseline. Confound (Codex): the actor GAE
path re-multiplies critic predictions by `target_scale` before subtracting, so scale≠a clean
un-crush. Follow-up `clip=10, no-popart` (decoupled: magnitudes fixed, only the rail widens) also
REJECTED — 200g vs analytic **0.393** (Wilson 0.327), l2 climbed to 5.8 (critic fighting the rail).
**Verdict: reward-scale tuning is spent.**

**Round 10 — Codex "GO STRUCTURAL".** Root blocker identified by code audit: the POMDP actor picks
action **TYPE + POWER but not the SPATIAL TARGET**. Pass receiver = analytic shortlist (no
pass-to-space); shot placement = analytic (`scored_shot_placement_x`; MCTS shot expansion hardcodes
goal-center); only DRIBBLE is rich (critic scores 12 move variants). So on pass+shoot the net is
blocked from the highest-information geometric choice. **Lever: gated, default-off spatial-target
hardening.**

**BUILT (committed, tree clean, byte-identical off):**
- `DD_SOCCER_ENABLE_NEURAL_PASS_SPACE` (`world.rs:14005`) — adds a second same-receiver pass
  candidate whose `target_point` is the anticipated lead/space reception point, so the CRITIC owns
  the feet-vs-run-onto choice; `target_player` stays bound (pass MPC/receipt/concede-veto unchanged).
  Pass MPC quality is scored against that point at `world.rs:16569`.
- `DD_SOCCER_ENABLE_SCORED_SHOT_PLACEMENT` (`world.rs:36676`) — but note this is a **global
  execution** change (`world.rs:26662`), so enabling it in eval also helps the analytic opponent.

**Round 12 — eval-gate fix (committed):** `draw_aware_lower_bound` + `DD_SOCCER_ENABLE_DRAW_AWARE_WILSON`
(`soccer_eval_gate.rs:74`, default-off). Empirical {1,0.5,0} variance instead of Bernoulli; flips
the earlier 200-game 0.493-REJECT toward PASS on the same record.

**Live runs (2026-07-08 ~20:00Z):**
1. `climb-target` — mainline confirm of reward+window: `CHANCE_QUALITY=1`, `TARGET_SCALE=30 CLIP=15
   POPART=true`, `hidden=128`, `RELATIONAL_ATTENTION=1`, entropy .05, authoritative λ=8,
   `DRAW_AWARE_WILSON=1`. Frontier `/tmp/soccer-climb`.
2. `newmain` — NEW forward-pass-primacy reward: `FORWARD_PASS_REWARD_SCALE=6`,
   `SHOT_SHAPING_REWARD_SCALE=0.4`, λ=8. Shift buildup toward forward progression; dampen low-value
   shots. Frontier `/tmp/neural-climb-local/fwdpass-frontier.json`.
3. `passspace_long` — pass-space net resumed from 198710 steps, training longer.

**Round 13 — Codex re-sync + the LOST verdict.** The pass-space A/B (`passspace_ab.sh`) trained 5
rounds with the gate ON then ran a 100g gate-ON eval — but **captured no verdict**: the log grep
missed it because `soccer_eval_gate_run` prints "Wilson lower bound" and **exits nonzero on REJECT**,
so the filtered/pipefail'd capture swallowed the output. The only recorded eval of that net
(`confirm_climb.log`, 0.325) ran it with the gate **OFF** → *not a valid lever test*. So pass-space
is **unproven, not falsified.** Codex round-13 calls:
- **Q1 — recover the pass-space verdict first** (higher-value uncertainty than the reward run):
  gate-ON-vs-analytic and gate-OFF-vs-analytic on identical held-out seeds, **separate processes**
  (gate is a process `OnceLock`), full `tee` capture. → `scripts`/`/tmp/passspace_recover.sh`.
- **Q2 — pure analytic is the north star.** "0.546 vs trained field + 0.33–0.44 vs analytic" =
  the neural field is too self-referential to be a promotion target; **analytic anchoring is
  mandatory** in train/eval. Structural expression is unproven, *not* failed.
- **Q3 — isolate the gates.** Pass-space alone, then scored-shot-placement alone; only stack them
  as a later exploratory run. Don't pair (scored-shot-placement's global execution change confounds).

**Housekeeping flagged:** `codex_heartbeat.sh` posts a **stale hardcoded** status ("~0.62 vs
analytic", `idx8` from 07-05) every 4 min — ignore/repoint its `status_line()` at the live logs.
Codex's own repo checkout reported **dirty** (`world.rs, player.rs, tests.rs, soccer_league_train.rs`
modified) while this machine's tree is clean — possible un-synced work on the Codex side.

## STRUCTURAL LEVER — negative, but on a CONFOUNDED stack (2026-07-08, rounds 13–15)

Built Codex's round-10 mainline (spatial-target hardening: goalmouth placement candidates in shot
MCTS `soccer.rs:64626`, per-receiver + open-lane pass `target_point`s, shot lowering honors
`plan.target_point`, tests green incl `world.rs:7664`, gate `DD_SOCCER_ENABLE_NEURAL_PASS_SPACE`).
Also ran the parallel option-1 clip-uncrush. **Both came back negative:**

| arm | games | mean | Wilson | Elo Δ | GD | decision |
|---|---|---|---|---|---|---|
| spatial-target (matched-gate) | 16 | 0.438 | 0.228 | −37 | −7 | REJECT |
| spatial-target (confirm) | 20 | 0.325 | 0.151 | −86 | −14 (exploitable 0.25) | REJECT |
| clip-uncrush (option 1, at power) | 200 | 0.393 | 0.327 | — | vs baseline 0.42 | REJECT |

Spatial *degraded* behavior (negative GD) — same failure **shape** as scale-120, not merely flat.
Train signature: critic `l2` 4.25→5.48, but `away_targets` only 45→342 (frozen-analytic opp barely
trains → weak, non-diverse eval field).

**Round 14 (Claude→Codex):** reported the scoreboard; hypothesized a **stack confound** — the arm
may have trained WITHOUT the graded chance-quality reward that made reward+window climb to
0.546/Elo+63, so the selector was handed richer geometry with no graded chance signal to value it.
Fork: (1) one clean re-run stacked on the confirmed base, vs (2) lock reward+window as incumbent.

**Round 14 (Codex→Claude): run (1) once as a strict falsification.** The confound is real: the
spatial lever expands/reranks concrete targets and depends on the critic/head valuing them
correctly; with chance-quality off, richer geometry is selected without the confirmed graded signal
— not the same test as "does spatial add on top of reward+window?" Hygiene required:
- compare **confirmed base vs confirmed base + spatial** (NOT old baseline);
- explicit env `DD_SOCCER_ENABLE_CHANCE_QUALITY_REWARD=1 SOCCER_NEURAL_TARGET_SCALE=30
  SOCCER_NEURAL_TARGET_CLIP=15 SOCCER_NEURAL_TARGET_POPART=true DD_SOCCER_ENABLE_NEURAL_PASS_SPACE=1`;
- preflight-prove: `targetPopart` persisted, chance-quality gate logged on, pass-space candidates
  AND shot-placement variants actually emitted;
- 160 train / 220 held-out, draw-aware Wilson; if it fails mean/GD vs the same base, **kill spatial,
  no third rescue.**
- Nuance: the league log prints `popart` before applying the env, so `popart=false` in that line is
  unreliable; trust `targetPopart` in the sampled JSON. Codex found no durable proof chance-quality
  was on for the structural arm — "if someone can prove it was on, I flip to (2) immediately."

**PROVEN — the confound is real (`/tmp/passspace_ab.sh:17-20`):** the structural training env sets
ONLY `DD_SOCCER_ENABLE_NEURAL_PASS_SPACE=1` + `DD_SOCCER_NEURAL_AUTHORITATIVE_LAMBDA=8`. It does
**NOT** set `DD_SOCCER_ENABLE_CHANCE_QUALITY_REWARD` nor the window trio. **Chance-quality was OFF;
the graded signal was absent.** The negative-GD structural result is therefore a confounded test,
not a refutation. Codex's (1) stands; (2) is not triggered. Next action: launch the clean
falsification under Codex's hygiene spec — deferred to the driving operator to avoid core
oversubscription with the live protected learner (the reason `passspace_ab.sh` waited on the
clip-uncrush PID before starting).

**Round 15 (Codex→Claude) — plan tightened + a gap caught. PRE-REGISTERED, ready to launch:**
- (a) Preflight must ALSO directly prove the **reward gate**, not just popart+pass-space: a logged
  `DD_SOCCER_ENABLE_CHANCE_QUALITY_REWARD=1` env/gate echo, or (stronger) a sampled nonzero
  chance-quality contribution / changed terminal label. `targetPopart==true` + `>0` pass-space
  emissions only prove the window + pass-space pieces.
- **GAP CAUGHT — env was missing `DD_SOCCER_ENABLE_SCORED_SHOT_PLACEMENT=1`.** The `shoot-kp*`
  power buckets are NOT placement proof — tests show they still target goal center. Real scored
  placement is the separate `scored_shot_placement_x` path behind that flag. Without it the
  "shot-placement" half of the lever is phantom geometry (the exact round-10 risk). **Corrected
  clean-run env:** `DD_SOCCER_ENABLE_CHANCE_QUALITY_REWARD=1 SOCCER_NEURAL_TARGET_SCALE=30
  SOCCER_NEURAL_TARGET_CLIP=15 SOCCER_NEURAL_TARGET_POPART=true DD_SOCCER_ENABLE_NEURAL_PASS_SPACE=1
  DD_SOCCER_ENABLE_SCORED_SHOT_PLACEMENT=1`.
- (b) **Pre-registered verdict (160 train / 220 held-out):** PASS = mean payoff **> 0.5 vs the same
  confirmed base** AND **GD > 0**. Report draw-aware Wilson but do NOT require Wilson-lower > 0.5 at
  220 (that's the follow-on power run, ~300 games). Miss either mean or GD vs that same base →
  **kill spatial, no rescue arm.**

## Round 15 — pre-registered rule LOCKED (2026-07-08)

Claude→Codex proved the confound durably (`/tmp/passspace_ab.sh:17-20` set only
`DD_SOCCER_ENABLE_NEURAL_PASS_SPACE=1` + `NEURAL_AUTHORITATIVE_LAMBDA=8` — chance-quality OFF), then
asked to pre-register the falsification's decision rule and preflight.

**Codex→Claude (r15):** approved, with two tightenings.
- **(a) Preflight must prove the reward gate DIRECTLY**, not by inference. `targetPopart==true` + `>0`
  pass-space emissions prove the window + pass-space pieces but **not** the chance-quality reward.
  Accept either a logged echo of `DD_SOCCER_ENABLE_CHANCE_QUALITY_REWARD=1` or (stronger) a sampled
  nonzero chance-quality contribution / changed terminal label. **Caveat:** `shoot-kp*` power buckets
  are **not** shot-placement proof — they still target goal center; real placement is the separate
  `scored_shot_placement_x` path behind `DD_SOCCER_ENABLE_SCORED_SHOT_PLACEMENT`. Since we keep that
  gate OFF (global-execution confound, r13), the arm emits **pass-space candidates only** and the
  "shot-placement variants emitted" preflight assertion is **dropped**, not satisfied via that gate.
- **(b) PRE-REGISTERED PASS RULE:** at 160 train / 220 held-out, **PASS iff mean payoff > 0.5 vs the
  SAME confirmed base AND GD > 0.** Report draw-aware Wilson but do **not** require Wilson-lower > 0.5
  at 220 (that is the follow-on power run). Miss **either** mean or GD → **kill spatial, no rescue.**

## Round 16 — fresh operator lever: FORWARD-PASS PRIMACY (2026-07-08)

Operator directive (repeated): *measure/reward advancement by **completed forward passes**, not by
shots taken or goals scored.* This is **reward shaping** (what to reward), distinct from the
target-scaling representation that failed. Built as two isolated, default-identical env knobs — a
**cleaner, cheaper attack on the "territory without conversion draws" plateau** than a learned EPV
potential (no possession-chain export needed):

| env knob | range / default | effect | code |
|---|---|---|---|
| `DD_SOCCER_FORWARD_PASS_REWARD_SCALE` | 0..20, dflt 1.0 | multiplies completed-pass shaping reward | `forward_pass_reward_scale()` soccer.rs:23444, applied :23502 |
| `DD_SOCCER_SHOT_SHAPING_REWARD_SCALE` | 0..1, dflt 1.0 | dampens **only** shot-TAKEN shaping proxy (on/off-target pts) | `shot_shaping_reward_scale()` soccer.rs:23461, applied :36473 |

Goal reward (100) and terminal-outcome reward are **untouched** — the net must still finish.
Completed-pass rewards are structurally capped **below** shot(40)/goal(100)
(`COMPLETED_FORWARD_PASS_*` consts, soccer.rs:1245-1260) so "pass forever" can't become optimal; the
scale only tilts the **dense** gradient toward progressive build-up. Regression guards live in
`soccer_learning.rs:9091` (analytic-parity) and `:9105` (backward-recycle penalty). A/B built
(`FWD_PASS_SCALE=6`, `SHOT_SCALE=0.4`, `/tmp/fwdpass_ab.sh`) but **paused** (SIGSTOP, PID 34413/34417)
to avoid core oversubscription with the live protected learner; not yet evaluated.

**Codex→Claude (r16):**
- **Run order: spatial falsification FIRST** (already locked, isolated, no rescue arm). Then
  forward-pass primacy as its **own** independent falsification — same confirmed base, 160/220,
  PASS iff mean payoff > 0.5 **and** GD > 0, Wilson reported only. If using the existing eval runner,
  ignore its `DECISION` when the *only* failure is Wilson (`evaluate_promotion` still gates on Wilson).
- **Forward-pass primacy is conditionally sound but CAN collapse into sterile territory.** Mandatory
  pre-registered discriminator: **held-out GD ≤ 0 while the completed-forward-pass margin/payoff
  rises** ⇒ more forward passing without conversion (relabelled territory) — that kills it.
- **Correction (source-verified):** the "shot 40 / goal 100" cap comment was **stale**. Real consts
  are `SHOT_ON_TARGET_REWARD_POINTS = 80`, `GOAL_REWARD_POINTS = 160` (soccer.rs:1167,1179). At
  `FWD_PASS_SCALE=6` a max forward/flank pass component reaches **≈83.5**, and `SHOT_SCALE=0.4` damps
  SOT shaping to **≈32** — so the lever is **not** structurally capped below shot shaping; it is a
  real dense-gradient tilt. Goal(160)+terminal still dominate. *(Comment fixed in soccer.rs:1240; audit doc corrected.)*
- **Preflight (chance-quality):** log the env echo **and** a nonzero terminal-label delta —
  `with_chance_quality` only bites when `match_outcome_reward_enabled()` builds the terminal label
  (world.rs:18346); PopArt/target-scale alone is not proof. Validate pass-space by the candidate's
  `target_point` differing from receiver feet (NOT by a new label; `shoot-kp*` are power labels).
- **Stacking:** forward-pass (build-up incentive) and chance-quality (terminal draw/chance
  separation) are **potentially additive, not substitutes**. Only test `CQ+FWD` after each single arm
  has an attributed result; call it additive only if the combo beats the **best single arm** on
  mean/GD, not merely base.

**Scoreboard (2026-07-08 ~15:00):** ratchet vs pure analytic holds at **best=0.500** (eval bounces
0.417–0.500); still not clearing >0.5. Both spatial-target and forward-pass primacy are **unproven,
not falsified.** Locked plan: run spatial falsification first, forward-pass second, combo last.

## COORDINATION + 0.600 PUSH (2026-07-08 ~20:45Z, maca5 / this Claude)

**Running now (mine, `nice`-priority, won't starve un-niced training or the live learner):**
- **Track B — the 0.546→0.600 push:** `run_gate_ab` recipe, confirmed reward+window stack
  (`DD_SOCCER_ENABLE_CHANCE_QUALITY_REWARD=1 SOCCER_NEURAL_TARGET_SCALE=30
  SOCCER_NEURAL_TARGET_CLIP=15 SOCCER_NEURAL_TARGET_POPART=true`), **320 train / 320 eval** (double
  cycle-2's 160). Rationale: payoff rose 0.546→0.559 as training doubled 80→160, so more training
  is the lever. Artifacts `/tmp/gate-ab-rwbig/`, verdict → `verdict.log`.
- **Track A — confirm the existing climb:** 200-game eval (2×100 sharded, disjoint seeds) of the
  confirmed cycle-2 nets to Wilson-lock the 0.559 result. Artifacts `/tmp/gate-ab-rwbig/shards/`.
- Using the **prebuilt** `/tmp/climb-target/release/soccer_outcome_ab_run` (08:56) because the
  source tree didn't compile — see next.

**FIXED the compile break** (blocked ALL source rebuilds): `MpcObjectiveSample` gained a 3rd dim
`applied_bend` everywhere except its consumer at `world.rs:32445` — that site destructured the
2-tuple `(features, residual)` but the payload is now the 3-tuple `(features, residual, bend)`
(`world.rs:25758`/`26432`) and built the sample without `applied_bend`. Captured + stored the bend;
`cargo check` clean. Uncommitted (leaving the commit call to the driving operator).

**Live learner status (FYI, pre-existing, NOT mine):** `soccer-learn-local` is *learning* fine
(heads train, `policy_evolved best_fitness=1.0158`) but every cycle ends
`soccer learning run failed: prune soccer completed-run deltas: db error` → `local_cycle_failed
status=1`. It's a **Postgres teardown/prune error**, not compile/resource. Whoever owns the DB
should look — the learning work completes; only the prune step dies.

**Core budget:** load ~10/16, headroom exists; other operator's `newmain league_train` (95653) is
itself `nice 13` (~1 core, healthy). No experiment duplication — different levers. Ask: please
don't launch a competing reward+window run; this push owns that lever tonight.

**FRESH-SEED CONFIRMATION (Track A done):** the confirm2 `0.559` did NOT replicate on fresh
disjoint seeds — 200 games (2×100, D-series) gave **payoff 0.510, GD +34, empirical-variance lower
bound 0.449**. The reward+window edge is REAL (positive, more W than L: 79-46-75) but SMALL (~0.52
seed-robust), so 200-game single-seed evals bounce ±0.05 and it does not robustly clear Wilson 0.5.
Implication: 0.600 is not reachable by more of the *same* training — needs a NEW lever on top.

**NOW EDITING `world.rs` (heads-up — you'd avoided it): pass-space INERT diagnostic.** Added a
gated counter (`DD_SOCCER_DUMP_PASS_SPACE_DIAG`, byte-identical off) at the pass-space creation
guard to MEASURE (not infer) how often `anticipated_pass_reception_point` returns a >1yd-from-feet
lead. NB: my code read *contradicts* the "inert at source" inference — `team_has_possession` is TRUE
at pass-decision time (passer holds the ball → `possession_team()` = passer team), `forward_bias`
(~10yd for a forward) fires, and `led_pass_target_for_receiver` only ADDS forward lead (never
collapses to feet). So the candidate *should* be created often. Instrumenting to settle it with
data — result pending an 8-game diag self-play on a freshly-built instrumented binary
(`/tmp/passdiag-target`). If emission rate is high, the earlier spatial REJECT was purely the
chance-quality confound (already proven) and the lever was testable all along.

**MEASURED — pass-space is NOT inert at source (refutes the inference).** 8000 pass-candidate
expansions in the TRAIN path (gate ON): estimator returns `Some` **100.0%**, distinct
(>1yd-from-feet) lead pushed **99.5%**. The distinct space candidate is created on essentially every
pass expansion — matching the code read (`team_has_possession` true, `forward_bias` fires,
`led_pass_target_for_receiver` only adds lead). So the spatial lever was **live during training**;
the earlier spatial REJECT (0.325–0.438) was the **chance-quality confound (already proven off in
`passspace_ab.sh`)**, not a dead candidate. The clean spatial A/B (confirmed base + chance-quality +
scored-placement + pass-space) is therefore a legitimate, information-bearing experiment.
**Discrepancy to flag:** the other operator's byte-identical result was on the frozen-net **EVAL**
path via `passspace_recover.sh`; my diag exercised the **TRAIN** path. Either the eval-gate process
didn't actually apply `DD_SOCCER_ENABLE_NEURAL_PASS_SPACE`, or the eval decision route differs from
the training MCTS expansion — worth a targeted eval-path diag before trusting "inert". Instrumentation
is gated (`DD_SOCCER_DUMP_PASS_SPACE_DIAG`), byte-identical off; left in place.

**AUTO-QUEUED: clean spatial falsification** (`/tmp/spatial_ab_launch.sh`, fires when Track B trains
finish so it overlaps Track B's eval without oversubscribing). Codex round-15 hygiene spec exactly:
baseline = CONFIRMED base (`chance_quality=1 scale30 clip15 popart`), treatment = base +
`NEURAL_PASS_SPACE=1 SCORED_SHOT_PLACEMENT=1`, 160 train / 220 held-out, fresh seeds
(train F1A10000 / eval F7A10000, avoiding the E-series optimism). Built binary
`/tmp/passdiag-target` (current source, all gates, byte-identical off). **Pre-registered PASS = mean
payoff > 0.5 vs same base AND GD > 0; miss either → kill spatial, no rescue.** Pass-space emission
already proven (99.5%); scored-placement emission not yet independently verified — caveat noted.
Artifacts `/tmp/gate-ab-spatial/`.

## One-line summary

The ceiling is structural: the net is a *selector over analytic candidates* optimizing
**territory**, and territory without conversion draws. Prove it cheaply (Step 0 A-falsifier,
Wilson>0.5 vs frozen field; Step 1 global heads-ON/OFF SOT A/B), then raise the real ceiling by
replacing the territorial signal — in both the on-ball PBRS and the off-ball residual reward —
with **one calibrated learned EPV potential**, which first requires a possession-chain export.

## Rounds 18–19 — operator KPI reframe + timidity diagnosis (2026-07-09)

**Operator architecture (the spine):** learn passing/dribbling/shooting/formation as two layers —
**POMDP decides** (neural + DP: action-type, receiver / pass-to-space, shot type+placement, dribble
type+lane, formation intent); **MPC executes** (neural + QP: trajectory, power, aim, first-touch,
off-ball). Wherever a *decision* lives in the analytic layer, that skill can't be learned.

**KPI reframe:** parity is now **completed forward passes vs the analytic field**, meter =
`AdvancementRecord` (soccer_eval_gate_run.rs:251 scores W/D/L by completed-forward-passes only).
Re-running it (the line was previously grep-swallowed + the deployed binary was stale) exposed the
real picture:

| net | FP-payoff (parity 0.500) | passes/game |
|---|---|---|
| fwd-pass-primacy (×6, shot×0.4) | **0.283** | ~19 |
| climb-base (CQ+uncrushed, best-0.500) | **0.358** | ~19 |
| analytic | — | **~36** |

Two facts reframe everything: (1) we're far below parity; (2) the primacy net is **worse on its own
target metric**, and the gap is passing **VOLUME, not forward-bias** (forward *share* ~6–7% both
sides). **The net is timid** — hoards/dribbles instead of passing. So r14–r16's GD/mean-payoff
"falsifications" judged the wrong axis; they don't falsify the forward-pass objective.

**Codex r18:** accepts the reframe + meter (caveat: "net of turnovers" = interceptions only; safe
recycling can inflate volume — needs quality guards). Prior = **timidity/action-selection, not
forward bias**. Prescribes an opportunity-conditioned **pass-rate** diagnostic (among on-ball
decisions with visible forward options + decent expected completion, how often chose pass — that IS
the POMDP's action-type decision quality). Prefers a **bounded anti-hold carrot over removing the
turnover stick** (source already admits under-rewarded advancement → timidity, soccer.rs:2122).
`LEARNED_PASS_RECEIVER` is **downstream** (post pass-normalization, world.rs:16431) — won't fix a
policy that rarely chooses pass; pessimistic `LEARNED_PASS_COMPLETION` could make a timid actor more
timid. **Locked kill-discriminator:** an arm survives only if FP-payoff > 0.358 (→0.500) AND
completed-forward margin improves AND net-forward doesn't regress AND forward-share doesn't fall AND
pass-gain-yards margin rises AND no completion-rate collapse AND (intercept+loose-ball)/completed-fwd
doesn't worsen; kill if payoff rises only from safe volume with forward-share falling.

**Experiments:** de-timid A/B — Arm B (FWD=3, turnover-penalty OFF) **train FAILED (no frontier)**;
Arm A (FWD=6, shots normal — is shot-damping the poison?) trained, FP-eval landing. scale-10 MPC-off
A/B: per-game forward counts **cand ≤ base** (×10 reward buys no extra forward passes → not a
magnitude problem). Pass-`target_point` plumbing landed on main (fa47c82).

**Claude r19 (driving live):** locked the discriminator; asked Codex to (A) spec the bounded
anti-hold carrot precisely (existing term vs new; env-knob + range, default byte-identical; cap so it
can't out-earn shot/goal or reward recycling; guardrail tied to the discriminator), and (B) confirm
the order — fix action-selection (raise pass-choice via carrot) until opportunity-conditioned
pass-rate ≈ analytic, THEN learned completion/receiver for target quality, with target_point aim on
top. Bonus: reward problem or interface problem, now that it's volume not bias? *(reply pending)*

## FORWARD-PASSING DIAGNOSIS + the scale-semantics fix (2026-07-09, rounds 17)

Measured the metric that matters — completed FORWARD passes (paired cand-vs-baseline), built into
`soccer_outcome_ab_run` eval as a PASS DIAGNOSIS block (att/g, completion%, forward/lateral/back share).

**Diagnosis (confirmed reward+window net, 24g, MPC off):** 37 passes/g, **92% completion**, but
**forward 7% / lateral 91% / back 3%**. Bottleneck is **SELECTION (WHO/WHEN), not execution** — the
POMDP recycles laterally in a safe-possession local optimum. Reward+window forward passes are FLAT
vs baseline (the climb was defensive/finishing, not progressive passing). MPC on/off: forward +23%
(2.1→2.6) directional but underpowered.

**Codex round-17 (grounded) — the bug that reframed it:** `DD_SOCCER_FORWARD_PASS_REWARD_SCALE` was
**NOT forward-only** — `soccer.rs` scaled the WHOLE sum incl. the lateral `1.2`, so scale=6 made a
lateral recycle worth 7.2 (scale=10 → 12), subsidizing the exact recycling it should kill. The
forward:lateral RATIO was invariant to the scale. Confirmed empirically: **scale=10 moved forward
share the WRONG way (7%→5%, GD −4).** Codex also confirmed **hypothesis B** (forward passes booked
immediately, later turnovers add separate negative credit → value net can learn "forward now, bad
state soon") and flagged MCTS expands only the **top-3** pass targets (`SOCCER_NEURAL_MCTS_PASS_TARGET_CANDIDATES=3`).

**FIX (Codex lever #1, implemented): forward-only scale.** `completed_pass_reward_for_pitch` now
multiplies ONLY the forward branch by `forward_pass_reward_scale()`; lateral/backward/flank are
unscaled. Byte-identical at scale=1. Guard test + outcome-dominance test both pass. **NB this changes
the shipped scale=6 production semantics** (lateral was 7.2, now 1.2) — correct fix, Codex-endorsed.

**Running:** forward-only scale=10 vs scale=1 (MPC off) — does the FIXED lever finally move the 7%?
**Codex lever order next:** (2) trim lateral reward / make it conditional on real switch-escape value,
(3) verify outcome-credit replay isn't keeping only positive reward (`soccer.rs:14977`), (4) check
whether a forward option survives into the top-3 MCTS expansion. **Codex on prod scale=6+MPC-on:
A/B before trusting** (canary: forward share, lateral share, turnovers-within-5s-of-forward, pass
inflation, SOT/goals, GD together).

## Round 19 — the buildable carrot spec (2026-07-09)

De-timid A/B **falsified** (Arm A 0.308, Arm B 0.258, both < 0.358 base) → blunt reward shaping
exhausted; volume deficit is structural to *existing* shaping. Codex r19 answers the carrot request
with a complete spec:

**New knob `DD_SOCCER_QUICK_FORWARD_RELEASE_REWARD_SCALE`** (default 0.0 = byte-identical; clamp
0..2; pre-register 0.75 / 1.0 / 1.5, 2.0 ceiling). `bonus = min(6.0, 4.0·scale·quick_release_timing_fit
·forward_gain_fit·opportunity_fit·completion_fit)`, paid ONLY when: action is a pass (not
hold/dribble/shot), completes to teammate control, forward gain ≥ 4yd, before-state had a real forward
opportunity (`visible_forward_pass_options>0` AND `max(quick_forward_pass_value,
best_forward_pass_option_quality) ≥ 0.50`), expected completion ≥ 0.45; zero on loose-ball /
interception / backward / lateral / incomplete. Reuses quick-release timing (soccer.rs:23908, 1.2s
hold, 12yd ref). Cap 6.0 (+existing 5.0) can't substitute a shot/goal or pay recycling. Guardrail =
the locked discriminator.

**Order: agreed** — carrot fixes action-selection until opportunity-conditioned pass-rate ≈ analytic
→ then learned receiver/completion → then target_point aim on top (receiver is downstream at
world.rs:16431, can't make a timid policy pass).

**Reward vs interface:** currently a **reward/credit-assignment** problem (interface already exposes
forward-option + pass-quality; actor family can represent pass-vs-hold/dribble/shoot). IF the carrot
raises FP-payoff but opportunity-conditioned pass-rate stays flat → THEN interface, add an explicit
action-type "release on visible forward option" logit. Not before this reward test.

**Next (Claude, building):** implement the knob per spec in an isolated worktree off fa47c82,
default-identical, then A/B arms {0.75, 1.0, 1.5} scored on the FP meter + locked discriminator.

## COORDINATION (2026-07-09, maca5/this-Claude → other operator)

Two of us are racing the single-lane LAN bridge; your automation wins every free slot, so I can't
reach Codex. Could you **yield one bridge slot** for my round-18? It carries a decisive result you'll
want: **immediate reward magnitude is FALSIFIED for forward passing** — forward-ONLY scale=10 (I fixed
`DD_SOCCER_FORWARD_PASS_REWARD_SCALE` to stop scaling lateral) left forward share at **7%/91% lateral,
identical to baseline**. You cannot bribe the net into forward passing; the blocker is DOWNSTREAM VALUE
(net avoids forward passes that lead to turnover/counter). Levers now: turnover-credit attribution,
top-3 MCTS pass expansion, lateral-reward trim, neural_self_bootstrap (A/B running). Also: I made the
retention-prune non-fatal (was crashing every prod cycle → blank-policy resume). FYI not a file-edit
conflict — different files.

## COORDINATION (2026-07-09, Codex <-> Claude Code, forward-pass guardrail merge)

Claude's dirty-tree inroads are directionally right for the next forward-pass climb:

- Train/evaluate the climb on **net forward passes**, not raw forward completions alone:
  `completed_forward_passes - pass_turnovers`.
- Keep `DD_SOCCER_FORWARD_PASS_CLIMB_CURRICULUM` so executable learned pass choices are not
  replaced by the option-score safety override before the policy can learn them.
- Add a learned receiver quality floor so a high-receipt but turnover-prone pass target does
  not win just because the ball can arrive.
- Increase/inspect the MCTS pass-target cap path with
  `SOCCER_NEURAL_MCTS_PASS_TARGET_CANDIDATES=8` and `DD_SOCCER_DUMP_MCTS_PASS_TARGET_DIAG`.

The semantic correction: forward-pass metrics are the **advancement/ranking signal**, not a
replacement for scoreline protection. The eval gate now requires both:

- the normal scoreline/WDL/Wilson promotion verdict, plus the existing positive held-out
  goal-difference floor (`SOCCER_EVAL_MIN_GOAL_DIFF_MARGIN`, default `0`);
- forward-pass advancement: raw forward margin, net forward margin, rate margin, and minimum
  held-out sample size.

The league checkpoint path follows the same intent: net forward pass margin is the climb metric,
but validation still requires the configured raw/net forward-pass floors and a non-negative
validation goal-difference floor by default
(`SOCCER_LEAGUE_CHECKPOINT_VALIDATE_MIN_GOAL_DIFF_MARGIN=0.0`).

## FIRST ROBUST CLIMB — value-layer stack at power (2026-07-09, CLIMB-600)

After proving reward-magnitude is dead for forward passing (forward-only 10x flat) and that the fast
30-game A/Bs are too underpowered to detect a climb (all ~0.5), ran ONE powered run:
**reward+window + self-bootstrap + MPC-on, 160 train / dual-seed eval**, vs the confirmed reward+window base.

**Result — robust, NOT the 0.55-then-0.51 fragility:**
- seed A2D9 (120g): payoff **0.579**, Wilson 0.490, GD +45
- seed C7F9 (100g): payoff **0.615**, Wilson **0.517 → PROMOTE**, GD +29, Elo +48.6

Both seeds ≥0.58 (mean ~0.60); seed3 is the FIRST config to clear the Wilson floor. The value-layer
stack robustly beats the confirmed base. **Codex r18 caveat:** self-bootstrap is already-on + MC-critic-
bypassed in prod, so the driver is almost certainly **MPC-at-power** (executor head warm) + 160-game
training depth, NOT self-boot. **Forward passing is STILL flat (7%/91% lateral both arms)** — the climb
is finishing/overall-play, not build-up. Codex's forward-share lever (widen MCTS top-4 pass candidate
exposure + deferred-pass/turnover credit) remains the separate, unsolved passing dimension.

**Next:** vs-analytic-field eval of the treatment net (the true 0.600-vs-field test), + Codex's
candidate-exposure lever for forward passing. NB found a sim bug: certain eval seeds produce a
non-terminating match (100% CPU, never completes) — silently stalled several wall-clock evals; wall
limit not catching it.

## FORWARD PASSING CLIMBS — deferred credit is the lever (2026-07-09, rounds 19-20)

Chased forward passing (stuck at 7%/91% lateral) through a full diagnostic ladder, ALL ruled out:
- reward magnitude (forward-only 10x): FLAT
- value self-bootstrap: FLAT
- candidate AVAILABILITY: ruled out — instrumented MCTS pass expansion, forward targets exist in
  72% of decisions, 51-57% of ranked targets are forward.
- candidate EXPOSURE (top-3 cap): ruled out — forward targets survive the cap ~73%.
- target MARKING/openness: ruled out — extended diag with a nearest-defender openness proxy;
  **87% of forward targets are OPEN** (off-ball runs ON didn't change it, already open).

So the net had abundant, exposed, OPEN forward options and refused them → pure SELECTION driven by
**broken credit attribution**. THE FIX (Codex r19/r20, validated): `DD_SOCCER_ENABLE_DEFERRED_PASS_CREDIT`
backdates delayed completion/interception/turnover credit onto the passer's launch transition (full-game
replay) instead of noisy per-tick deferred credit. Default-off, plain env flag.

**RESULT (30 train / 24 eval, winning base + deferred credit vs base):** FORWARD-PASS ADVANCEMENT =
**CLIMB** (first significant one all session): forward passes/game **1.4 → 2.8 (DOUBLED)**, paired
Δ +1.46, 95% lower bound **+0.49 (>0 significant)**, forward share **5% → 10%**, Elo +51. Overall
payoff even (0.5, GD -7) at this low power — forward passing climbed, not yet converted to a win-climb.
The single root cause of the whole 7%-lateral wall was credit attribution, not reward/value/off-ball.

**Now running (COMBO):** winning stack (reward+window+MPC, which robustly beat base at 0.58-0.62 /
one PROMOTE) + deferred credit at power (80/140) — the both-together test: does forward passing climb
AND does payoff hold toward 0.60.
