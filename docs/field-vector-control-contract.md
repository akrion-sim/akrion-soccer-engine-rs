# Field-vector control contract

The field vector is the shared state contract for learning and control. Rewards,
POMDP decisions, per-player MPC execution, and 11v11 formation LP/IPM must be
functions of the same current geometry rather than independent constants.

## Canonical 11v11 vector

`SoccerConfigVector` (`src/des/general/soccer/config_vector.rs`) constructs the
team-canonical, mirror-normalized vector from the current `WorldSnapshot`:

1. own 11 players, deterministically ordered;
2. opponent 11 players, deterministically ordered;
3. the ball;
4. possession, phase, and score context.

Each player contributes normalized position, velocity, acceleration, jerk, and
possession state. The ball contributes position, velocity, acceleration, and
jerk. Missing bodies in reduced tests are zero-padded; non-finite inputs are
sanitized. The raw configuration is projected into the persisted 256-dimensional
whole-field embedding used by learned consumers. Team canonicalization means the
same football pattern has the same representation for Home and Away.

The POMDP observation also carries the explicit actor-relative whole-field motion
block. It is not a local-nearest-player approximation: all 22 bodies and the ball
are represented. `SoccerFieldNumbersVector` derives relational overloads,
centroids, momentum, and offensive/defensive urgency from that complete vector.

## Canonical five-a-side vector

`standalone-5v5/src/game.rs` uses the reduced analogue: own five players,
opponent five players, and the ball. The per-agent observation is actor-relative
and the centralized critic state preserves all 10 players plus the ball. Reduced
team size changes only the entity count, not the contract that utility and
control depend on all active bodies.

## Dynamic reward and penalty utility

A `SoccerRewardEventKind` records the factual event and its attribution. Utility
calibration is a separate layer:

- absent calibration: multiplier `1.0` (sane baseline);
- scalar calibration: `DD_SOCCER_REWARD_KIND_SCALES`;
- contextual calibration: `DD_SOCCER_REWARD_CONTEXT_PATH`.

The contextual artifact has schema version 1 and one optional linear head per
event kind over the 256-dimensional whole-field embedding:

```json
{
  "schemaVersion": 1,
  "byKind": {
    "ShotOnTarget": {
      "bias": 0.0,
      "weights": [0.0]
    }
  }
}
```

The example weight array is abbreviated; a live head must contain exactly 256
finite weights. The head produces a bounded log multiplier. Scalar and contextual
multipliers compose, then clamp to `[0.0001, 4.0]`. Therefore a learned model can
make the value of shooting depend on all 22 players and the ball (lane congestion,
keeper geometry, support, counter exposure, and so on), but cannot flip a reward
into a penalty, erase the event completely, or create unbounded return. Missing or
malformed artifacts fall back to `1.0`.

Accepted artifacts should be inherited from prior promoted runs. New weights are
trained or searched on one seed partition, frozen for the inner learner, and
selected on a disjoint held-out partition. Training reward is never the promotion
metric.

`soccer_reward_context_fit` is the outcome-grounded fitter. When retrieval capture
is enabled, the event recorder retains each exact factual typed occurrence (for
example `ShotOnTarget` or `BadPassChainPenalty`) together with the canonical 256-d
embedding at that event's tick and a second canonical embedding after the
configured outcome horizon. The capture intentionally omits the event's
hand-authored magnitude, so the contextual head cannot merely imitate the assumed
reward it is meant to replace. The fitter first learns a hermetic whole-field
state-value head from final W/D/L plus capped goal margin. It then trains each
event utility from the learned value change between the factual event state and
its later field state; penalty heads use the sign mirror. Per-kind RMS value
change supplies a data-derived target scale rather than an assumed reward value.
Value changes are generated out-of-fold by alternating match partitions, then
shrunk by the value model's held-out improvement over a zero predictor. A weak or
non-predictive value model therefore produces a near-neutral `1.0` utility rather
than amplifying noise. The delta normalizer has a `0.05` lower safety clamp so a
tiny empirical RMS cannot turn numerical noise into a saturated target.
A team-match contributes at most one unit of fitting weight to each event kind:
repeated routine occurrences receive inverse-frequency weight, preventing a match
with hundreds of carries from overwhelming rarer shot, pass, or turnover evidence.
A prior schema-v1 artifact can be passed as the fifth argument, so accepted weights
are the next run's initial condition rather than being reset. The generated
artifact is frozen during inner policy training and must still clear a disjoint
promotion evaluation.

## Per-player MPC

MPC remains strictly single-actor. The controlled state is one player's point
mass; every other player and the ball enter as field-vector context and moving
obstacles, never as jointly optimized states.

- QP execution builds obstacle/reference terms from the current snapshot.
- `soccer_local_mpc_whole_field_context` changes target, desired velocity, and
  objective weights using all-player and ball kinematics.
- `SoccerMpcObjectiveHead` receives the same 256-dimensional whole-field embedding
  plus action-family and execution features.

Thus both the analytic QP and learned MPC objective are functions of the field
vector while preserving the architectural rule that MPC controls only one actor.

## Formation LP/IPM

The 11v11 formation is a function of its own current configuration. Every tick,
`SoccerFormationLpBrain::solve_tick` derives objective weights and slot inputs
from the full `WorldSnapshot`, updates LP bounds/objective/constraints from every
player's current position and kinematics plus the ball and tactical directive,
then solves with Clarabel IPM or the deterministic LP fallback.

The LP/IPM allocates the whole-team shape. Its output is then consumed as bounded
formation guidance; per-player MPC executes each assigned movement. Learned
whole-field value intent may bias the LP within bounded limits, but there is no
collective MPC and no integer program.
