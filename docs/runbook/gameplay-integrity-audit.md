# Gameplay Integrity Audit Runbook

Use this when auditing the soccer runtime for hidden rule conflicts, duplicated
heuristics, unfair scheduling, or dead agent brains. The purpose is not to remove
tactical tradeoffs. A good audit separates intentional balance from two rules
that cannot both be true in the same match state.

Repo reminders:

- Stay on `main`.
- `x` is field width, sideline to sideline.
- `y` is field length, goal to goal.
- Prefer focused tests and `git diff --check` over noisy full-repo validation
  unless the change touches broad contracts.

## Evidence Pack

For every finding, capture enough context for a follow-up patch to be local:

- Scenario: seed, clock/tick, teams, score, possession, ball state, and field
  phase.
- Actors: player ids, teams, roles, official id/kind, and whether a human is
  controlling an actor.
- Rules: code paths, constants, config flags, and tests that assert the behavior.
- Trace: `operation_order`, selected action, target, expected target, and any
  central/home/away brain directive involved.
- Verdict: contradiction, likely redundancy, intentional balance, or inconclusive.

Start with these seams:

- `src/des/general/soccer/world.rs`: match orchestration, schedule, ball state,
  possession, rewards, live frame snapshots.
- `src/des/general/soccer/player.rs`: per-player action selection and
  `AgentDecisionTrace`.
- `src/des/general/soccer/team.rs`: central brain, home/away directives, LP/MPC
  formation guidance.
- `src/des/general/soccer/referee.rs`: official agents and offside-line tracking.
- `src/des/general/soccer/config_vector.rs`: canonical 22-player plus ball state
  vector for retrieval.
- `src/des/general/soccer/inspect.rs`: frame and liveness views for playback.
- `src/des/general/soccer/tests.rs`: regression contracts and scenario builders.
- `src/bin/main_soccer_tournament_run.rs`, `src/des/soccer_learning*.rs`: learned
  team-brain persistence and tournament wiring.

Useful search starters:

```sh
git grep -n "must\\|never\\|illegal\\|hard\\|penalty\\|bonus\\|clamp" src/des/general/soccer
git grep -n "operation_order\\|weighted_agentic_order\\|scheduled_index" src/des/general/soccer
git grep -n "rng.next_float\\|time_window_probability\\|probability" src/des/general/soccer
git grep -n "team_brain\\|home_brain\\|away_brain\\|central_brain" src/des/general/soccer
git grep -n "OfficialKind\\|referee\\|offside" src/des/general/soccer
```

## Task: Find Contradictory Rules

Goal: identify two or more rules that can fire in the same state while demanding
mutually impossible outcomes. Do not flag ordinary attack/defense tension or a
scored tradeoff where one side wins by design.

Method:

- Pick one concrete soccer moment, then list every rule that applies to the same
  actor and target in that tick.
- Separate hard constraints from soft scoring. Hard examples include "never",
  "must", illegal actions, final clamps, and early returns.
- Build a conflict table with columns: state predicate, rule A, rule B, current
  precedence, observed result, and desired arbiter.
- Check whether the precedence is explicit. A contradiction is more serious when
  the winner depends on file order, loop order, or a later clamp silently undoing
  an earlier guarantee.
- Confirm with a minimal scenario test or a playback/decision trace. The trace
  should show both rules were eligible, not just that the final behavior looked
  wrong.

Common hot spots:

- Pass target scoring versus receiver urgency.
- Defensive shape versus loose-ball contesting.
- Formation lane discipline versus tactical overlap/support runs.
- Shot legality versus keeper-position exceptions.
- Referee/offside positioning versus ball-clearance movement.
- Team-brain strategy directives versus per-player immediate survival choices.

Exit criteria:

- Each contradiction has one authoritative owner named in the report: rules
  layer, central brain, home/away team brain, player brain, ball agent, or
  official agent.
- The proposed fix says whether it should become a hard guard, a scoring weight,
  a trace-visible precedence rule, or a test-only stale-contract correction.

## Task: Find Redundant Rules That May Contradict Later

Goal: find duplicated or near-duplicated behavior that is currently harmless but
likely to diverge when one copy is tuned.

Method:

- Search for repeated constants, repeated distance bands, repeated action labels,
  and duplicate concepts implemented in both `player.rs` and `world.rs`.
- For each duplicate, decide whether the duplication is shared policy,
  actor-specific adaptation, cached projection, or accidental drift.
- Compare edge cases, not just the main happy path. Look at zero pressure, high
  pressure, own box, final third, set plays, loose balls, in-flight passes, and
  human-controlled players.
- If two rules are intentionally parallel, write down the invariant that keeps
  them together.

Redundancy smell list:

- Same threshold with different names.
- Same label with different eligibility conditions.
- Same clamp in two places with different min/max values.
- A "final safety" that reverses an earlier tactical decision.
- A test fixture that manually patches state the runtime normally derives.

Exit criteria:

- Report each duplicate as `merge`, `extract helper`, `keep with invariant`, or
  `leave alone`.
- Any follow-up patch should include a test that would fail if the copies drift
  again.

## Task: Find Artificial Or Arbitrary Operation Order

Goal: locate places where player id, vector order, test fixture order, or sort
tie behavior decides football outcomes that should be fair, weighted, or
explicitly canonical.

Important distinction:

- Canonicalization is good when identity must disappear, such as the
  `config_vector.rs` 22-player plus ball retrieval vector.
- Deterministic weighted scheduling can be good when equally plausible actors or
  actions compete for first chance, but the score must come from match state and
  must not be an act-of-god draw.
- Deterministic tie-breaks are good when replay stability matters, but the
  tie-break must be stated and tested.

Method:

- Search for `.iter().find`, `.max_by`, `.min_by`, `sort_by`, `partial_cmp`,
  `.enumerate()`, `first`, `last`, and direct player-index assumptions.
- Classify each order dependence as canonical, replay-stable, agentic-scored,
  deterministic trace-only, or accidental.
- For accidental order, prefer explicit agentic scoring for player-brain action
  order. Existing central-brain, ball, official, and player traces should expose
  scored `operation_order` without randomizing the chosen football outcome.
- Preserve determinism from match state, tick, actor id, and action context.
  Record the order in traces.
- Add a permutation test: reverse or reorder the players while preserving the
  soccer state. The selected outcome should stay the same unless the changed
  state itself justifies a different agentic order.

Focused test filters to consider:

```sh
cargo test -q --lib operation_order
cargo test -q --lib agent_schedule
cargo test -q --lib config_vector
```

Exit criteria:

- Every order-dependent decision has a reason: canonicalization, stable replay,
  or fair weighted selection.
- Liveness telemetry shows `playerOperationOrderAgenticOk`,
  `centralBrainOperationOrderAgenticOk`, `ballOperationOrderAgenticOk`, and
  `officialOperationOrderAgenticOk`.

## Task: Remove Act-Of-God Outcome Causes

Goal: every match outcome should be traceable to agents, geometry, ball physics,
skills, fatigue, pressure, and current match state. Probability-like scores may
rank options or describe risk, but a fresh random draw must not cause the ball to
be lost, saved, blocked, curled, mis-hit, fouled, or controlled.

Allowed uses:

- Agentic ordering or exploration, when the trace records the order and the
  choice remains a function of state, value, visits, urgency, and role.
- Deterministic state-derived variation seeded by tick/player/action context,
  when it only chooses a stable tactic or setup detail.
- Training/search sampling outside live match outcome resolution.

Disallowed uses:

- `rng.next_float() < probability` directly deciding a tackle, steal, save,
  catch, shot block, heavy touch, foul, dribble beat, pass error, shot error, or
  ball-control winner.
- Random speed, direction, deflection, or target offsets after an agent has
  already chosen the action.
- Random fallback axes that move players or the ball when a deterministic
  geometry tie-break can explain the result.

Method:

- Search for direct RNG gates and random execution noise:

```sh
git grep -n "rng.next_float()" src/des/general/soccer
git grep -n "next_float().*<\\|< .*next_float()" src/des/general/soccer
git grep -n "heavy-touch\\|dispossession\\|tackle\\|shot-blocked\\|save\\|parry\\|foul" src/des/general/soccer
```

- For each hit, name the actor and the real cause. Examples: nearest defender
  gap, carrier body-shield geometry, goalkeeper reach, blocker line distance,
  passer skill, receiver openness, fatigue, or ball speed.
- Convert event rolls into deterministic contests or thresholds over those
  causes. Keep the old probability function only if it has become a score, not
  a coin flip.
- Player-brain action commitment should be deterministic from option scores,
  pressure, urgency, role, and `dt`; do not use fresh RNG to decide whether a
  live player shoots, dribbles, clears, tackles, or passes.
- Add a seed-independence regression when the same state should now resolve the
  same way across RNG seeds.

Exit criteria:

- No direct random draw causes a live football outcome.
- Any remaining RNG hit is classified as ordering, exploration, initialization,
  training/search, or a bug.
- Tests cover at least one representative ball-control, kick/shot execution, and
  pressure contest path.

## Task: Audit Energy Conservation And Movement Smoothness

Goal: players should look like athletes managing effort, not frantic point
masses. Sprinting and quick turns are still possible, but acceleration, angular
acceleration, and jerk must be bounded by skill, fatigue, gait, and urgency.

Method:

- Search for direct velocity/facing writes, acceleration caps, yaw-rate caps, and
  jerk computation:

```sh
git grep -n "velocity =\\|velocity +=\\|acceleration\\|jerk\\|yaw_rate\\|yaw_accel" src/des/general/soccer
```

- Check both linear and angular paths. A player can obey max speed while still
  looking frantic if acceleration or facing changes snap every tick.
- Verify off-ball, ball-holder, human-controlled, collision, and set-play paths
  all update kinematics through bounded transitions or clearly documented
  exceptions.
- Under pressure, the ball-holder should accelerate away from the closest
  defender into space. Prefer forward escape lanes first, then lateral lanes,
  then backward lanes, but veto forward if it closes the defender gap.

Exit criteria:

- Linear acceleration and jerk are capped in the normal movement path.
- Yaw rate and yaw acceleration caps are low enough to avoid instant spinning.
- Ball-holder escape targets prioritize forward/lateral/backward lanes as
  `3/2/1` while still increasing distance from the nearest defender.
- Focused tests assert jerk bounds and pressured escape behavior.

## Task: Audit Back-Four Depth And High-Line Ceiling

Goal: the four defenders should stay connected to the ball without pressing the
whole line recklessly high. The full back-four average `y` should normally sit
2-30 yards goal-side of the ball, measured against the ball's current `y`, but
that average must not press more than 5 yards into the opponent half. That
5-yard opponent-half ceiling is the explicit exception to the 30-yard ball
cushion.

Method:

- Inspect `defensive_line_cushion_adjusted_target` and any downstream line
  clamps. The final defender target must honor the opponent-half ceiling for
  both Home and Away.
- Check the full four-defender average, even when only the non-exempt defenders
  receive the corrective target.
- Verify high-ball scenarios where the ball is far in the opponent half; the
  defender target may be more than 30 yards behind the ball there, by design.
- Verify strikers in possession stay free to continue behind the enemy line with
  the ball, while non-holder strikers are given onside recovery targets within
  the 3-second grace model.

Focused test filters to consider:

```sh
cargo test -q --lib back_four
cargo test -q --lib defensive_line
```

Exit criteria:

- Home defenders never receive a line target above midfield plus 5 yards.
- Away defenders never receive a line target below midfield minus 5 yards.
- The 2-30 yard ball cushion still applies whenever it does not conflict with
  that high-line ceiling.
- Non-holder strikers are target-clamped goal-side of the opponent back-four /
  offside line, while a striker with the ball beyond that line is not pulled
  back by support-shape logic.

## Task: Audit Two-Second Teammate Spacing

Goal: outfield teammates should converge toward a Euclidean 5-10 yard spacing
band in open play, with a 3-6 yard exception inside either 18-yard box. The
correction should be predictive: a player chooses a two-second path using teammates' current
position, velocity, and acceleration, not only their instantaneous positions.

Method:

- Check both territory/proximity clocks and final movement-target guards. They
  should share the 5-10 yard open-play and 3-6 yard box contract.
- Verify predictive spacing samples the path over a 2-second horizon and
  projects teammate position from current position, velocity, and acceleration.
- Ensure spacing relief does not invert team layers: midfielders and forwards
  should prefer lateral relief when a naive away vector would pull them backward
  through their own line.
- Keep committed actors exempt where appropriate: ball holder, intended pass
  receiver, human-controlled player, keeper, and committed loose-ball chaser.

Focused test filters to consider:

```sh
cargo test -q --lib teammate_spacing
cargo test -q --lib teammate_territory_spacing
cargo test -q --lib possession_shape_pushes_back_four_up_and_staggers_midfield_pair
```

Exit criteria:

- Open-play pair targets clear the 5-yard lower edge without treating exactly
  5 yards as the ideal; 5-10 yards should score as the healthy band.
- Box pair targets clear the 3-yard lower edge and treat 3-6 yards as the
  legitimate congested-box band.
- Spacing adjustments preserve role-line ordering and do not create frantic
  accelerations or target oscillation.

## Task: Verify Referee, Team Brains, Ball, And 22 Players

Goal: prove the live match has all expected agents alive, scheduled, aware of the
same field, and contributing traces. This means:

- 1 central brain.
- Home and away team brain snapshots with finite directives.
- 22 players.
- 1 ball agent.
- 3 official agents: center referee, near assistant, far assistant.

Method:

- Use `MatchConfig::live_gameplay()` for live behavior unless the task is about
  playback or training.
- Verify the central brain runs before player decisions and sees all 22 players,
  the ball, and all 3 officials.
- Verify home and away brain snapshots track 11 players each, expose finite
  directives, and differ when phase/possession makes them differ.
- Verify each player has a non-empty decision trace over a short live window,
  except human-controlled players whose trace may be `human-input`.
- Verify official agents have `OfficialDecisionTrace` and non-empty
  `operation_order`, and assistants expose offside-line awareness.
- Verify the ball agent records action and `operation_order`.
- In playback liveness JSON, check `playerOperationOrderAgenticOk`,
  `centralBrainOperationOrderAgenticOk`, `ballOperationOrderAgenticOk`, and
  `officialOperationOrderAgenticOk`.

Focused test filters to consider:

```sh
cargo test -q --lib central_brain
cargo test -q --lib team_brain
cargo test -q --lib referee
cargo test -q --lib live_gameplay
cargo test -q --lib decision_trace
```

Exit criteria:

- The frame contract reports 22 tracked players and 3 tracked officials.
- The home and away brain snapshots each track their own side.
- All finite numeric fields stay finite.
- Player operation-order sampling is agentic/scored. Central brain, ball, and
  official operation-order sampling is not fixed when there are enough samples
  to vary.
- Any missing trace has a named reason, such as human input, disabled trace
  capture, or a deliberately inactive actor.

## Task: Audit Learned Brains Versus Runtime Brains

Goal: make sure tournament or learning code is not training a different brain
than live gameplay uses.

Method:

- Trace from `EngineMatchRunner` through `MatchConfig::live_gameplay()` and
  per-team neural brain installation.
- Confirm home and away policy entries, target entries, neural snapshots, and
  genome/directive values stay attached to the right team.
- Confirm frozen-opponent mode still lets the frozen side play while blocking
  its gradient updates.
- Confirm Postgres checkpoints and warm-start loads preserve both team id and
  brain snapshot.

Focused test filters to consider:

```sh
cargo test -q --lib live_gameplay
cargo test -q --lib per_team
cargo test -q --lib tournament
```

Exit criteria:

- A trained home brain scores home actions; a trained away brain scores away
  actions.
- Training, tournament persistence, and live inference agree on team identity.
- Any fallback from per-team to shared brain is explicit in code and tests.

## Reporting Template

```text
Finding:
Task:
State:
Actors:
Rules involved:
Current precedence:
Why this is contradictory or risky:
Evidence:
Recommended owner:
Recommended fix shape:
Validation:
```

## Closeout

For docs-only changes:

```sh
git diff --check
```

For code changes, run the narrow test filter for the audited behavior, then
`git diff --check`. Broaden to `cargo test -q` only when the patch changes a
shared runtime contract.
