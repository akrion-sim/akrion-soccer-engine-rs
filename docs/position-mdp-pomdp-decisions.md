# Position-Specific MDP/POMDP Decisions

This engine treats each player as an autonomous decision maker. Broad roles still exist
(`Goalkeeper`, `Defender`, `Midfielder`, `Forward`), but learning and traces also carry an exact
11-slot assigned position so decisions can specialize by football job.

## The 11 Assigned Positions

| Index | Slot | Meaning |
| --- | --- | --- |
| 0 | `GK` | Goalkeeper |
| 1 | `LB` | Left back |
| 2 | `LCB` | Left centre back |
| 3 | `RCB` | Right centre back |
| 4 | `RB` | Right back |
| 5 | `LM` | Left outside midfielder / winger |
| 6 | `LCM` | Left centre midfielder |
| 7 | `RCM` | Right centre midfielder |
| 8 | `RM` | Right outside midfielder / winger |
| 9 | `LS` | Left striker |
| 10 | `RS` | Right striker |

`soccer_assigned_position_for` derives the slot from the player's formation anchor
(`home_position`) in team-relative space, so Home and Away both agree on "left" and "right"
from their own perspective.

## Where The Slot Exists

The exact slot is intentionally available at every decision layer:

- `PlayerSnapshot.assigned_position` is the live read-model source.
- `SoccerMdpState.assigned_position` makes symbolic MDP state position-specific.
- `SoccerPomdpObservation.assigned_position` gives observation-only POMDP consumers the same slot.
- `SoccerQStateKey.assigned_position` splits tabular Q learning by all 11 football jobs.
- `SoccerDecisionContext.assigned_position` records the slot in traces and learning transitions.
- `SOCCER_ASSIGNED_POSITION_COUNT` documents the 11-slot contract.
- `DD_SOCCER_ENABLE_ASSIGNED_POSITION_EMBEDDING=1` can also expose the same slot as a neural actor
  one-hot while keeping the network dimension stable when the gate is off.

## Outside Mid Final-Third Wide vs Pinch Decision

When the ball is in an outer lane of the opponent final third, `LM` and `RM` have a live support
decision between two families:

- `wide-outlet`: stay wide or get open near the touchline for a recycle, switch, or wide pass.
- `pinch-cross-arrival`: move central/goalward into the half-space, cutback lane, penalty spot, or
  far-post lane to get on the end of a cross.

The support target helper scores the pinch candidates against the normal wide outlet. If the wide
outlet is clearly better, the cross-arrival helper returns `None` and the existing `wide-outlet`
path remains available. If the central/goalward arrival wins, the movement trace uses
`pinch-cross-arrival`.

## MDP/POMDP To MPC Execution

The MDP/POMDP layer chooses the football intention: which support family, pressure choice, receiving
lane, run type, cover job, or distribution option is best under the current state or belief. MPC does
not replace that choice. MPC turns the chosen intention into an executable path by checking timing,
speed, acceleration, turning radius, offside/line constraints, obstacles, opponent pressure, and
shape-preserving guardrails. If the chosen option is not executable, MPC reconciliation can repair or
fallback the movement while keeping the original decision trace available for learning.

| Slot | MDP/POMDP decision room | How MPC helps execute it |
| --- | --- | --- |
| `GK` | Hold line vs sweep, claim cross vs stay, short vs long distribution. | Computes reachable claim/sweep paths, goal-line alignment, body orientation, safe stop distance, and pass/kick feasibility under pressure. |
| `LB` | Overlap vs underlap, stay wide vs rest-defense, press winger vs cover back post. | Fits the run to the touchline/half-space lane, avoids blocking the winger, preserves recovery distance, and times arrival so the back line is not abandoned. |
| `LCB` | Step to ball vs cover, hold offside line vs drop, attack aerial ball vs screen runner. | Keeps goal-side positioning, line integrity, interception reachability, acceleration limits, and safe clearance angles executable. |
| `RCB` | Step to ball vs cover, hold offside line vs drop, attack aerial ball vs screen runner. | Mirrors the centre-back constraints on the right side: line integrity, goal-side route, reachable contest point, and collision-free recovery path. |
| `RB` | Overlap vs underlap, stay wide vs rest-defense, press winger vs cover back post. | Fits wide/inside runs to available lanes, checks recovery feasibility, avoids teammate congestion, and keeps the far-side defensive balance reachable. |
| `LM` | Stay as `wide-outlet` vs `pinch-cross-arrival`, isolate fullback vs recycle, track runner vs release. | Times the central pinch or wide support route, keeps the run onside, bends around defenders, and compares whether the wide lane or box-arrival lane is physically cleaner. |
| `LCM` | Show for possession vs screen counter, third-man run vs hold shape, press ball vs cover lane. | Finds a reachable pocket that preserves midfield spacing, closes or opens passing lanes, and limits overrun when pressure or transition risk changes. |
| `RCM` | Show for possession vs screen counter, third-man run vs hold shape, press ball vs cover lane. | Mirrors the centre-mid execution: reachable support pocket, lane coverage, pressure approach angle, and acceleration-safe recovery if possession turns over. |
| `RM` | Stay as `wide-outlet` vs `pinch-cross-arrival`, isolate fullback vs recycle, track runner vs release. | Executes the wide-or-central choice with timing, offside safety, defender avoidance, and route shaping so cross-arrival does not collapse into aimless crowding. |
| `LS` | Check to ball vs run in behind, near-post vs far-post arrival, press centre back vs screen pass. | Times curved runs to stay onside, reaches the selected finish lane, respects sprint/turn limits, and adjusts the path around covering defenders. |
| `RS` | Check to ball vs run in behind, near-post vs far-post arrival, press centre back vs screen pass. | Mirrors striker execution on the right: onside timing, finish-lane reachability, pressure approach angle, and safe path correction under contact or congestion. |

## Adding Position-Specific Decisions

New per-position MDP/POMDP behavior should branch or score on `SoccerAssignedPosition`, not only on
the broad `PlayerRole`. That keeps room for all 11 jobs to learn distinct choices: fullbacks can
choose overlap vs rest-defense, centre backs can choose step vs cover, centre mids can choose show
vs screen, outside mids can choose stay wide vs pinch, and strikers can split near-post/far-post or
check-in/run-in-behind responsibilities.
