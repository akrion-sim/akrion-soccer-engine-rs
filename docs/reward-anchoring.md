# Reward Anchoring v2 — win/goal anchors, bounded learnable everything-else

Applies to BOTH games: the 11v11 engine (`src/des/general/soccer/`) and the standalone 5v5
(`standalone-5v5/`). Goal: a fixed, human-meaningful reward currency with hard sanity bounds so
the learner explores inside a sane region and converges faster, while everything that is not an
anchor stays **learnable / tunable / discoverable**.

## 1. The currency (fixed anchors — not tunable)

| event                | points |
|----------------------|--------|
| win the match        | +1000  |
| lose the match       | −1000  |
| draw                 | 0      |
| score a goal         | +500   |
| concede a goal       | −500   |

These two anchors define the unit. They are the only rewards that are *not* weights to be
searched: they are what the weights are priced in.

## 2. The on-frame shot anchor (floor 50, context cap)

A shot that is genuinely **on frame** (would enter the goal unless the GK saves it) pays at least
**50 points** (10% of a goal) — regardless of where it was taken from. The *upper* bound is
position-dependent:

```
shot_points(x, q) = clamp(quality_term(q), 50, cap(x))
cap(x) = 50 + 150 · shot_context(x)          // ∈ [50, 200]
```

- `shot_context(x) ∈ [0,1]`: field-vector context — distance/angle to goal (xG-like). At ~80 yards
  `shot_context ≈ 0`, so cap = floor = 50: an on-frame 80-yard shot is worth exactly the floor,
  never more. Close and central → up to 200 (= 40% of a goal, the existing non-conversion
  fraction).
- Paid **only** when the shot is actually on frame (5v5: on-target + non-rushed gating; 11v11:
  scaled by / gated on `shot_on_frame_probability`). Off-frame shots earn nothing from this
  anchor. This keeps the expected value of bad shots low without needing the floor to be low:
  an 80-yard attempt rarely ends on frame, so its *expected* payment ≈ p(on-frame)·50 ≈ ~1 pt.
- Anti-spam: monitor shots/game in validation; the on-frame gate is the primary defense
  (the 5v5's earned-shot gating is the reference implementation).

## 3. Bounded, field-vector-scaled learnable terms

Every other reward term (pass, dribble/carry, turnover, chain, off-ball support, chance
creation, movement shaping, …) has the shape:

```
points_i(event, x) = clamp( w_i · c_i(x) · base_i(event),  lo_i,  hi_i )
```

- `w_i` — the **learnable/tunable weight** (env var, searched by the tuner). Bounded:
  `w_i ∈ [w_lo_i, w_hi_i]` enforced at parse time (clamp, not error).
- `c_i(x)` — the **field-vector context** multiplier in [0,1] (or a profile), computed from ball
  position/threat. The context encodes the user-specified ordering:
  - **own half**: dribble/carry progression outvalues shooting (shot context ≈ 0 there);
  - **middle third**: forward passing and carries dominate;
  - **final third**: shooting context peaks; dribble context tapers (box play favors
    pass/shot); turnover penalties scale UP with threat-at-loss (the field-vector turnover
    multiplier that already exists in the 11v11).
- `lo_i, hi_i` — hard per-event point bounds. Sane defaults (single event, before context):
  pass completed ∈ [0, 25], forward-progression bonus ∈ [0, 40], dribble/carry gain ∈ [0, 30],
  turnover ∈ [−80, 0] (scaled up by threat-at-loss), win-ball/tackle ∈ [0, 40], off-ball
  support/shaping per-tick terms each ∈ [−2, +2] (dense terms must stay an order of magnitude
  under event terms or they drown the sparse signal — the diagnosed 11v11 failure mode).

## 4. Ordering invariants (enforced by construction + unit tests)

1. `max_single_non_conversion_event ≤ 0.40 · GOAL` (=200). Existing machinery:
   `grounded_non_conversion_reward_scale` (11v11), `grounded_conversion_ladder` (5v5) — extend to
   the new scale.
2. `on_frame_floor (50) ≤ shot_points ≤ cap(x) ≤ 200 < GOAL (500) < WIN (1000)`.
3. Own-half positions: `dribble points-per-event ≥ expected shot points = p_onframe(x)·cap(x)`.
4. Final third: `max shot points > max dribble points`.
5. Sum of dense shaping over a possession ≪ GOAL (shaping cannot buy a goal).

## 5. Back-propagation requirements (the audit)

Anchors are worthless if credit doesn't reach the acting agents:

- WIN/LOSS must be an actual terminal transition reward seen by the learner (not just a fitness
  metric), discount-correct.
- GOAL lands on the scoring transition; deferred credit (assist/pass chains) propagates backward
  (11v11 `DD_SOCCER_ENABLE_DEFERRED_PASS_CREDIT`; 5v5 GAE over the episode).
- No inert reward paths (the 11v11 `SHOT_ON_TARGET_REWARD_POINTS`-telemetry trap is the canonical
  example — every reward knob must be verified to reach a training target).
- Value/target normalization must absorb the point scale (11v11: popart + target_scale; 5v5:
  advantage normalization — verify or add) so 1000-point terminals don't destabilize PPO.
- MPC (11v11): the learned MPC objective/shot objective must price actions in the SAME currency
  (its targets derive from the same reward stream), and the POMDP actor must see the possession
  state it is acting under.

## 6. Possession tri-state as a learnable input (MARL requirement)

MAPPO/IPPO acting depends on possession state. Both games must expose, in the observation vector:
`possession ∈ {ours, theirs, contested/50-50}` (one-hot or 2 bits) plus `i_have_ball` per agent.
The 5v5 needs the contested/50-50 state ported from the 11v11 duel logic.

## 7. Rollout / gating

- 11v11: all of the above behind `DD_SOCCER_ENABLE_ANCHORED_REWARDS` (default off, byte-identical
  off), individual knobs still overridable.
- 5v5: becomes the default (it is our sandbox), old scale removed; tuner searches `w_i` within
  the declared bounds.
