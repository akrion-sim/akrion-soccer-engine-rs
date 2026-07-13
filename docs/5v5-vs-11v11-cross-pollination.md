# 5-a-side ⇄ 11-a-side: what each should learn from the other

**Is LP/IPM formation the *only* difference?** No — it's the only *intentional design* difference.
There are three kinds of difference:
1. **Intentional (keep):** the 5-a-side deliberately has **no LP/IPM formation machinery**; it replaces
   it with possession-conditioned, field-vector off-ball positioning (spacing/width/advance/goalside/
   ahead/burst). So "formation" is handled *differently*, not absent.
2. **Liability (converge away):** the 11-a-side additionally carries a **tabular-Q base + analytic-veto
   shield + MCTS/MPC layers** that dampen the net (`neural_blended_action` only re-ranks tabular
   candidates; downstream vetoes at `world.rs:15833/15961/16605`, `player.rs:10129`). This is *not* a
   feature to preserve — it's a leading suspect for the parity plateau. Earlier 5-a-side runs showed
   that a hermetic net-only PPO *can* climb without those layers, but the July 12 validation lineage
   did not promote; every new policy still has to prove itself through the held-out ladder.
3. **Inherent (scale):** 22 vs 10 players + ball. Same architecture, more coordination surface.

So the learning architecture SHOULD converge: **hermetic PPO/actor-critic + field-vector, learnable
(tunable-not-fixed) rewards.** The 5-a-side is the reference prototype for that.

## What 11-a-side should adopt from 5-a-side (the climb levers)
1. **Hermetic > tabular.** Strip/neutralize the analytic shield + tabular fallback so the net owns the
   outcome. Sequence with the r23 exploration flip (instrument net-influence → flip exploration → then
   reward). Positive control: the 5-a-side.
2. **Reachable + dominant end product.** The 5-a-side broke parity by making the goal REACHABLE and the
   shot reward field-vector-scaled: shots legal from the whole opponent half + reward shots more +
   **xG(field vector)** so a pot-shot has low expected value and a real chance pays full. Realized
   on-frame events retain the 50-point floor. 11-a-side now wires distance, angle, pressure, blockers,
   lane geometry, and goalkeeper position into its shot reward.
3. **Field-vector rewards via a pre/post context.** The 5-a-side's `FieldRewardContext::from_world`
   derives `pass_value, safe_outlet_value, turnover_risk, dribble_pressure, goalside_score, burst_score,
   chance_value` from positions, captured **pre- and post-action**, and modulates every reward family by
   the field-vector *delta*. Port the pattern; make the dynamic reward-scale vector field-vector-*modulated*,
   not just scalar multipliers.
4. **MARL chance-creation potential.** Reward the TEAM (potential-based, unfarmable) for collectively
   moving into a configuration where a real shot becomes available — drives synchronized attacking runs.
5. **Learnable rewards with non-zero clamps** (done: `SOCCER_DYNAMIC_REWARD_WEIGHTS` extended to
   dense/shoot/dribble channels; frequency-aware positive floors keep channels alive), searched on **gated goal-diff**, not the
   farmable forward-pass KPI. Reward ladders stay grounded: a non-converting shot candidate must
   remain below the relevant conversion reward by at least 5 points.
6. **New learnable mechanics** proven at 5v5: 7 speed gears (incl. the GK as a field-vector gear),
   scooped/aerial passes over a blocked lane (receiver must be more open).
7. **Explicit controlled-possession phase.** Possession, dispossession, and a genuine free-ball 50:50
   are distinct inputs and action-mask regimes. Last touch is history, not ownership.

## What 5-a-side should adopt from 11-a-side
1. **Formation as a field-vector function.** Even without full LP/IPM, a lightweight *team-shape* target
   `shape = g(field_vector)` (a fixed point of a field-dependent objective) could tighten coordination
   beyond the current per-player positioning — the 11-a-side's back-four/midfield-band features are the
   template. Optional; the current positioning already covers most of it at 5v5 scale.
2. **Opponent curriculum / league play.** 5-a-side trains vs ONE fixed scripted baseline; 11-a-side uses
   **league play vs frozen champions + analytic fixtures** (`soccer_league_train.rs`). Adding frozen
   self-play champions to the 5-a-side opponent pool would harden it beyond beating one analytic bot.
3. **Deferred pass credit (attribution).** 11-a-side's `DD_SOCCER_ENABLE_DEFERRED_PASS_CREDIT` back-dates
   completion/turnover credit onto the pass DECISION tick (doubled forward passes). The 5-a-side's
   pre/post `FieldRewardContext` approximates this, but explicit launch-tick attribution could sharpen it.
4. **Belief/POMDP features.** 11-a-side carries belief-state + option-control features; a small
   belief/history feature could help the memoryless 5-a-side actor.
5. **Keeper catch/parry semantics.** A fast edge-of-reach shot stays live as a parry instead of every
   ball inside one radius becoming an automatic catch; distribution clears under immediate pressure.
6. **Full-game and launch-tick attribution.** Broadcast the final win/loss label to every transition,
   and move delayed pass/goal credit back to the POMDP decision / MPC commitment tick before GAE.

## Bottom line
Converge both onto **hermetic net + field-vector learnable rewards**. Keep the 5-a-side's simplicity as
the fast experimentation loop; keep the 11-a-side's formation-as-field-vector and league curriculum.
Everything else in the 11-a-side stack (tabular/shield/MCTS damping) is the thing to shed, not share.

## Capability parity contract

The engines should have the same *football-learning capabilities*, but do not need identical
solvers. This is the code-backed parity target after the semantic branch merge:

| Capability | 5v5 implementation | 11v11 implementation | Required convergence |
|---|---|---|---|
| Hermetic policy ownership | Shared-weight PPO actor/critic; scripted opponent only | Neural-authoritative climb mode with Q policy retained only as a legal candidate shell | Evaluation and promotion must use net-owned actions; tabular scores cannot be the learned frontier |
| Whole-field observation | Relational vector for all 10 players plus ball, goals, roles, and cues | Canonical vector for all 22 players plus ball in POMDP/neural inputs | Preserve team-relative mirroring and include every player and the ball |
| POMDP decisions | Compact action masks, loose-ball belief, shot/pass/space cues, explicit possession/dispossession/50:50 phase | Named belief state, option control, assigned-position decisions, explicit `loose_ball_fifty_fifty` | Phase must be a learner-visible variable; last touch cannot masquerade as controlled possession |
| MPC execution | Field-vector lane opening, finishing, goalkeeper and path-aware spacing | Player MPC, QP/neural objectives, field-vector kinematics | Every MPC target/objective must remain a function of the current field vector |
| Team shape | Learned possession-conditioned targets and spacing; no LP/IPM | Field-dependent LP/IPM formation nudging plus learned objectives | Same capability, intentionally different solver complexity |
| Reward context | Pre/post `FieldRewardContext`, potential differences, tunable positive clamps | Exact typed events, 256-d contextual heads, counterfactual rollout value fitter | Prefer potential/value differences over hand-assigned per-event multipliers |
| Opponent robustness | Scripted baseline | Analytic fixtures and frozen league champions | Add frozen champion curriculum to 5v5 |
| Credit assignment | Launch-tick pass/goal attribution, GAE through the possession, full-game outcome broadcast | Deferred pass/action credit, discounted possession-chain blame, factual future embeddings, full-game outcome broadcast | Credit the POMDP decision and the MPC action actually committed to physics; preserve signed reward and penalty chains |
| Goalkeeper physics | Field-vector angle positioning, pressure-aware distribution, speed/height-aware catch versus parry | Distance/sightline catch caps, rebound/parry physics, keeper MPC and learned specialist head | Keep saves contextual and physically live; never use an unconditional radius catch |
| Promotion evidence | Goal difference, win rate, spacing and behavior gates | Paired Home/Away payoff, goal difference and Wilson bound | Same-seed paired fixtures; no side/seed confounding |

## Non-negotiable reward anchors

The two sparse football outcomes are semantic constants in both engines:

```text
goal conversion base reward = 500
completed-match win base reward = 1000
completed-match loss base penalty = -1000
realised on-frame shot reward is field-vector contextual in [50, 200]
non-converting shot reward + 5 <= 500
```

This is enforced in both runtimes, not merely documented:

- 5v5 projects legacy goal/win requests to 500/1000, back-dates delayed action outcomes, and adds the
  symmetric full-game label to every transition rather than only the final tick.
- 11v11 uses the same 500 conversion base and 1000 broadcast win/loss base; old goal-scale and
  match-win overrides are compatibility inputs that cannot move the anchors. Its full-game return
  clamp is ±1250, so the 1000 label plus bounded context reaches PPO/MAPPO instead of being truncated.
- Positive lower clamps remain in force, so tuning cannot silently delete a football outcome.

Everything below those outcomes—shot quality, passing, pressure, spacing, possession-chain credit,
duel urgency, keeper context, and MPC execution quality—remains learnable/tunable and field-vector
conditioned with finite non-zero clamps. A learner may discover *where* an action matters, but it may
not redefine scoring or winning. In particular, a saved shot from roughly 80 yards receives the
50-point on-frame floor, while a close/central/clear chance can rise toward 200; the lower bound
recognises that the shot reached frame, and the upper bound prevents shot farming from rivaling goals.

## Frequency-aware learning bounds

The tunable number is a **base weight**, not a context-free payout. The current field vector then
modulates it. For example, a progressive carry is weighted most heavily while escaping/building from
one's own half, neutrally through midfield, and at roughly half strength in the final third, where a
clear shot or dangerous pass should dominate. Shooting uses the converse context: distance, angle,
pressure, blockers, goalkeeper position, and lane geometry determine its expected and realized value.

| Reward family | 5v5 search band | 11v11 search band | Why |
|---|---:|---:|---|
| Realized on-frame shot | 50–200 points | 50–200 points | Fixed floor for hitting frame; field-vector quality selects the rest |
| Completed pass base | 0.02–1.00 points | 0.10–4.00× event scale | Sparse but farmable; progress, lane, outlet, and pressure supply context |
| Build-up milestone / chain | 0.05–3.00 points | 0.10–4.00× event scale | Rewards useful sequences without rivaling finishing |
| Ball recovery | 0.10–4.00 points | 0.10–4.00× event scale | Sparse change of control; possession phase and danger matter |
| Turnover | 0.20–8.00 penalty | 1.00–8.00× penalty scale | Field position and opponent threat amplify dangerous losses |
| Bad pass / dribble loss | 0.20–10.00 penalty | Contextual typed-event penalty | A poor action is worse when safe outlets existed or loss creates threat |
| Forward carry / dribble | 0.001–0.10 per tick | 0.05–4.00× dense scale | Narrow because it fires often; zone, pressure, role, progress, and retention modulate it |
| Dense movement / support | 0.001–0.20 per tick | 0.05–4.00× dense scale | Cannot accumulate faster than sparse outcomes |
| Field-vector potential deltas | 0.005–0.75 | learned EPV scale 1–60 | Potential differences telescope; scale remains bounded |
| 5v5 teammate spacing | 0.0005–0.05 | contextual formation/spacing objective | Raw 5v5 curve reaches -100 per player per tick, so a wide coefficient is unsafe |

These bands preserve the established warm-start defaults except where the old units made a channel
effectively unlearnable, and cut away implausible search regions. The 5v5 shot
parameters are expressed directly in points (rather than tiny weights projected to the same floor),
so the learner can now genuinely explore the full 50–200 contextual band.
