# 5-a-side ⇄ 11-a-side: what each should learn from the other

**Is LP/IPM formation the *only* difference?** No — it's the only *intentional design* difference.
There are three kinds of difference:
1. **Intentional (keep):** the 5-a-side deliberately has **no LP/IPM formation machinery**; it replaces
   it with possession-conditioned, field-vector off-ball positioning (spacing/width/advance/goalside/
   ahead/burst). So "formation" is handled *differently*, not absent.
2. **Liability (converge away):** the 11-a-side additionally carries a **tabular-Q base + analytic-veto
   shield + MCTS/MPC layers** that dampen the net (`neural_blended_action` only re-ranks tabular
   candidates; downstream vetoes at `world.rs:15833/15961/16605`, `player.rs:10129`). This is *not* a
   feature to preserve — it's the leading suspect for the parity plateau. The 5-a-side proves a
   **hermetic net-only PPO climbs to +5.3..+6.2 GD / 95–99% wr** vs the analytic baseline with none of
   that in the loop.
3. **Inherent (scale):** 22 vs 10 players + ball. Same architecture, more coordination surface.

So the learning architecture SHOULD converge: **hermetic PPO/actor-critic + field-vector, learnable
(tunable-not-fixed) rewards.** The 5-a-side is the reference prototype for that.

## What 11-a-side should adopt from 5-a-side (the climb levers)
1. **Hermetic > tabular.** Strip/neutralize the analytic shield + tabular fallback so the net owns the
   outcome. Sequence with the r23 exploration flip (instrument net-influence → flip exploration → then
   reward). Positive control: the 5-a-side.
2. **Reachable + dominant end product.** The 5-a-side broke parity by making the goal REACHABLE and the
   shot reward field-vector-scaled: shots legal from the whole opponent half + reward shots more +
   **xG(field vector)** so a pot-shot pays ~0 and a real chance pays full. 11-a-side has
   `analytic_shot_trigger_value` (a field-vector xG) but it isn't wired into the REWARD — do that.
3. **Field-vector rewards via a pre/post context.** The 5-a-side's `FieldRewardContext::from_world`
   derives `pass_value, safe_outlet_value, turnover_risk, dribble_pressure, goalside_score, burst_score,
   chance_value` from positions, captured **pre- and post-action**, and modulates every reward family by
   the field-vector *delta*. Port the pattern; make the dynamic reward-scale vector field-vector-*modulated*,
   not just scalar multipliers.
4. **MARL chance-creation potential.** Reward the TEAM (potential-based, unfarmable) for collectively
   moving into a configuration where a real shot becomes available — drives synchronized attacking runs.
5. **Learnable rewards with non-zero clamps** (done: `SOCCER_DYNAMIC_REWARD_WEIGHTS` extended to
   goal/dense/shoot/dribble; `1e-4` floors so no channel dies), searched on **gated goal-diff**, not the
   farmable forward-pass KPI.
6. **New learnable mechanics** proven at 5v5: 7 speed gears (incl. the GK as a field-vector gear),
   scooped/aerial passes over a blocked lane (receiver must be more open).

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
| POMDP decisions | Compact action masks, loose-ball belief, shot/pass/space cues | Named belief state, option control, assigned-position decisions | Port useful belief/history cues to 5v5 without importing the tabular Q table |
| MPC execution | Field-vector lane opening, finishing, goalkeeper and path-aware spacing | Player MPC, QP/neural objectives, field-vector kinematics | Every MPC target/objective must remain a function of the current field vector |
| Team shape | Learned possession-conditioned targets and spacing; no LP/IPM | Field-dependent LP/IPM formation nudging plus learned objectives | Same capability, intentionally different solver complexity |
| Reward context | Pre/post `FieldRewardContext`, potential differences, tunable positive clamps | Exact typed events, 256-d contextual heads, counterfactual rollout value fitter | Prefer potential/value differences over hand-assigned per-event multipliers |
| Opponent robustness | Scripted baseline | Analytic fixtures and frozen league champions | Add frozen champion curriculum to 5v5 |
| Credit assignment | Immediate pre/post field deltas | Deferred pass/action credit and factual future embeddings | Add launch-tick deferred credit to 5v5 where action latency matters |
| Promotion evidence | Goal difference, win rate, spacing and behavior gates | Paired Home/Away payoff, goal difference and Wilson bound | Same-seed paired fixtures; no side/seed confounding |

## Non-negotiable reward hierarchy

All configured or learned weights are projected onto this football ordering *after* loading tuner
values:

```text
non-converting shot reward + 5 <= goal conversion reward
goal conversion reward + 20 <= completed-match win reward
```

This is enforced in both runtimes, not merely documented:

- 5v5 projects `REW_GOAL` and `REW_MATCH_WIN` after loading all bounded shot weights and emits the
  terminal win/loss reward on the final rollout tick.
- 11v11 bounds the effective on-target pool beneath the tuned goal reward and projects
  `DD_SOCCER_MATCH_WIN_REWARD_POINTS` above the effective tuned goal reward.
- Positive lower clamps remain in force, so tuning cannot silently delete a football outcome.

The margins are deliberately absolute reward points. A learner may discover *where* a shot, pass,
or movement matters from the field vector, but it may not learn that repeatedly missing is worth
more than scoring or winning.
