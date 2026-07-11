# Field-Vector Reward Architecture — rewards/MPC/formation as functions of the field vector

**Principle (from the 5-a-side hermetic work, 2026-07-11).** Every reward, penalty, MPC
objective, and formation target should be a **dynamic function of the FIELD VECTOR** — the
full multi-agent state — not a fixed scalar constant. The value of an action depends on the
configuration of *all* players and the ball; a shot from 18 yds is worth ~0 with two defenders
on the lane and a set keeper, and ~0.4 xG with an open lane and a scrambling keeper. Fixed
reward constants can't express that; **field-vector functions can, and they should be LEARNABLE**
(tunable weights or a neural head trained on the true win/goal objective — see
`SOCCER_DYNAMIC_REWARD_WEIGHTS`, conversation-doc r24/r25).

## The FIELD VECTOR
The POMDP state representation the net already consumes:
- **Base field features** `SOCCER_NEURAL_BASE_FEATURE_DIM = 192` (`soccer.rs:4459`) — the 22
  players (or 10 for 5-a-side) + ball, positions/velocities/relations, egocentric per-decision.
- Plus field-motion, belief (`..._BELIEF_FEATURE_DIM`), learned-MPC-replan, option-control,
  human-intent, and **formation** bands (back-four line `..._BACK_FOUR_LINE_FEATURE_DIM`,
  midfield band, carrier line-break) — `soccer.rs:4477–4511`.
For the 5-a-side hermetic prototype the analogue is the ~34-dim egocentric obs / 70-dim actor
vector + 48-dim centralized global state (`standalone-5v5/src/game.rs`).

## Concrete instances (what to make field-vector functions)

1. **Shot reward = xG(field vector).** A field-vector shot value ALREADY EXISTS:
   `analytic_shot_trigger_value(&ShotTriggerInputs)` (`shot_decision.rs:297`) scores a shot by
   distance + angle + lane coverage + keeper (tests at `:785–835` confirm close>far, open>covered,
   clear>blocked). **It is used to DECIDE shots but NOT to scale the shot REWARD.** Wire it:
   `SHOT_ON_TARGET_REWARD_POINTS * shot_xg(field_vector)` where `shot_xg` = normalized
   `analytic_shot_trigger_value` (or a learned xG head over the 192-dim vector). This mirrors the
   5-a-side win: reward `= (base + q·placement) · xg`, `xg = dist_f²·(0.4+0.6·angle_f)` — a
   long/wide/covered pot-shot pays ~0, a real chance pays full. It also guards a widened shot
   zone from long-shot farming.

2. **Pass reward = f(field vector).** Receiver openness, lane-interception risk, anticipated
   reception point, forward EPV delta — all functions of the field vector. Partly present
   (forward-pass thresholds `world.rs:351–354`); make the *magnitude* a field-vector xT/EPV
   delta, not a flat `0.45/1.35/2.2`.

3. **MPC objective = f(field vector).** Whether QP- or neural-solved, the MPC cost is a function
   of the (predicted) field vector — opponent-reachability, lane geometry, teammate support —
   evaluated on the rolled-out field state, not fixed weights. The learned-MPC-replan features
   already feed the net; the objective should read the same field vector.

4. **Formation LP/IPM = f(field vector) — "a function of itself."** The formation nudging target
   should adapt to the current field configuration (ball, opponent shape, overloads) rather than
   a static template: `formation_target = g(field_vector)`, so the shape is a fixed point of its
   own field-dependent objective. The back-four/midfield-band features are the inputs.

## Why this is the climb lever
The 5-a-side plateau (and the 22-man parity) came from **fixed dense shaping that out-massed the
sparse, field-vector-dependent end product.** Making the end-product rewards (shot xG, goal,
pass EPV) genuine functions of the field vector — and keeping the shaping small and learnable —
let the 5-a-side climb from parity to **+5.3..+6.2 GD / 95–99% wr** vs the analytic baseline.
Same recipe here: field-vector reward functions + `SOCCER_DYNAMIC_REWARD_WEIGHTS` search on
GATED goal-diff.

## Status / next
- Field-vector shot value exists (`analytic_shot_trigger_value`); **TODO: scale the shot reward by
  it** (small, gated change — mirror the 5-a-side xG).
- Dynamic-reward vector now spans shooting/dribbling/goal/dense (r25). Next: make the *shot* and
  *pass* scales **field-vector-modulated**, not just scalar multipliers.
- 5-a-side reference impl pushed: `origin/wip-player-speeds`, `origin/five-a-side-standalone`.
