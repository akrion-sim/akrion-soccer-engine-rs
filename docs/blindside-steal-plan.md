# Blindside surprise-steal & carrier glance: status & plan

The defensive play: when an opponent **dribbles forward and cannot see behind them** —
especially at a walk/jog/skip — a defender that has crept into their **blind arc** and
**believes it can catch them** sneaks up to nick the ball from behind. The carrier's
countermeasure is to **glance side-to-side** (a head-scan with a real ball-control cost)
to recognise the threat and break away. The whole loop is wired into the POMDP.

This doc records what shipped, how to validate/train it, the feature follow-ups, and the
repo-health note surfaced while verifying it.

Source of truth: `src/des/general/soccer.rs` (gate, constants, `BlindsideStealAssessment`,
observation fields), `src/des/general/soccer/world.rs` (`blindside_steal_assessment`,
observation population + carrier glance), `src/des/general/soccer/player.rs` (defender
action option, `immediate_defensive_steal_target` branch, carrier escape bias).

---

## 1. What shipped (local, uncommitted)

All gated behind `dd_soccer_enable_blindside_steal()` — env `DD_SOCCER_ENABLE_BLINDSIDE_STEAL`,
**default-OFF and re-reading the env each call under `cfg(test)`** (so the suite and the
current network weights stay byte-identical until the flag is flipped). It is decision-only:
the new observation fields are **not** added to the neural feature encoder, so
`SOCCER_NEURAL_FEATURE_DIM` is unchanged.

- **Assessment** `WorldSnapshot::blindside_steal_assessment(defender_id) -> Option<…>`.
  Returns `None` (drawing no RNG, touching no decision) unless **every** condition holds:
  opponent on the ball (not a keeper holding in his box); defender in the carrier's blind
  arc (`to_def · carrier_facing < BLINDSIDE_BEHIND_DOT`); carrier dribbling **forward**
  (`≥ BLINDSIDE_CARRIER_MIN_FORWARD_SPEED_YPS`) and **slowly** (`carrier_slow` is full at/
  below `BLINDSIDE_CARRIER_JOG_SPEED_YPS`, fading to 0 at `…_GETAWAY_SPEED_YPS`); **and the
  defender believes it can catch them** — its flat-out chase pace (`player_top_speed_yps ×
  Sprint`) beats the carrier's getaway pace by `BLINDSIDE_CHASE_SPEED_MARGIN_YPS` AND closes
  the gap inside `BLINDSIDE_CATCH_HORIZON_SECONDS`. This catch-belief is the hard gate: no
  belief ⇒ no creep.
- **Defender POMDP obs** `blindside_steal_opportunity` (+ `blindside_steal_target`).
- **Defender action** — a `blindside-steal` option appended in `defensive_action_options`
  **only when a live assessment exists** (so the option set is byte-identical when off),
  scored ∝ opportunity × press appetite × defending. Movement target + MPC-label registry
  route it to the holder like `tackle`.
- **Ball win** — `immediate_defensive_steal_target` gains a blindside branch, evaluated
  **before** the goal-side early-out (it is deliberately the wrong-side exception). It wins
  the ball from behind only in the `~1.6–2.1 yd` contact band — beyond the existing
  close-range `wrong_side_emergency_contact` reach — against a slow, unaware, catchable carrier.
- **Carrier glance (physics)** — in `observation_for`, a forward-dribbling carrier sweeps
  its blind side each tick: **cost** = `blindside_scan_drift_risk` (rides the same
  ball-control drift channel as the existing look-behind head-scan, scaled by pace);
  **benefit** = lifts perceived confidence on rear opponents within the glance radius.
- **Recognition + escape** — `blindside_threat_from_behind` is computed from rear opponents
  weighted by that (post-glance) perceived confidence × carrier slowness, so the carrier
  only feels the threat **after it has looked**. In `possession_action_options` a high
  threat floors break-away-forward / early-release options and damps slow shielding dwell.

Tests: `blindside_surprise_steal_and_carrier_glance_recognition` (one consolidated test —
the env var is process-global, so separate parallel tests toggling it would race).

---

## 2. Validation & A/B plan (next)

Unit behaviour is proven. The **match-level** effect is not yet measured. Run a deterministic
A/B (gate ON vs OFF, identical seeds) over a batch of games and pre-register:

| Hypothesis | Metric | Guard (must not regress) |
|---|---|---|
| More balls won from behind off slow, unaware carriers | dispossessions where defender was in the carrier's blind arc at contact | total fouls / mistimed lunges; own-half turnovers conceded |
| Carriers stop walking the ball under unseen pressure | mean carrier speed while an opponent is within the blind-arc approach radius | possession retention %, completed-pass count |
| Glance cost is paid, not free | rate of `blindside_scan_active` ticks; mean `blindside_scan_drift_risk`; mis-control events during a glance | dribble success %, carry distance |
| No panic regression | early/forced releases under blind-arc threat vs a real outlet | pass completion under pressure |

Confounder: a defender behind the carrier is, by construction, often goal-side-wrong, so
raw "steals from behind" counts must be filtered to the blind-arc + catch-belief subset that
the feature actually targets. Log `blindside_steal_opportunity` and the assessment parts
(`carrier_unaware`, `carrier_slow`, `catch_belief`) via the inspector to isolate the effect.

---

## 3. Learner / reward integration (next)

Currently the feature is a **recognition + decision bias** only; nothing in the reward
credits it. To let the MARL/MAPPO policy *learn* the play rather than just be nudged:

- **Defender reward** — a small positive on a completed blindside dispossession (ball won
  while `blindside_steal_opportunity` was high), folded through the existing dense-shaping
  budget (`apply_dense_shaping_budget`) so it cannot dominate; a small negative on a
  *failed* committed blindside lunge that springs the carrier (mirror of the slide-tackle
  risk model) so the catch-belief gate is reinforced, not gamed.
- **Carrier reward** — credit a successful escape (broke away / released before being
  robbed once `blindside_threat_from_behind` was high); a small debit for being robbed from
  behind while slow and never having glanced (the "look up" lesson). Pair with a tiny,
  bounded glance **cost** term so the policy learns *when* the look is worth the drift, not
  to scan every tick.
- Keep all reward terms behind the same gate (or a sibling `…_REWARD` gate) so weights stay
  byte-identical until a deliberate retrain. Retrain is required before the obs/decision
  bias changes net behaviour at scale.

---

## 4. Tuning knobs (constants in `soccer.rs`)

`BLINDSIDE_STEAL_APPROACH_RADIUS_YARDS`, `BLINDSIDE_BEHIND_DOT`,
`BLINDSIDE_CARRIER_JOG/GETAWAY_SPEED_YPS`, `BLINDSIDE_CATCH_HORIZON_SECONDS`,
`BLINDSIDE_CHASE_SPEED_MARGIN_YPS`, `BLINDSIDE_STEAL_CONTACT_RADIUS_YARDS`,
`BLINDSIDE_STEAL_COMMIT_THRESHOLD`, `BLINDSIDE_GLANCE_DRIFT_RISK_BASE/_SPEED_SPAN`.
These are seeded heuristics, not hard rules — fold the most behaviourally sensitive ones
(catch horizon, behind-dot, jog/getaway band, glance drift) into the tunables/config vector
so the A/B and the learner can sweep them instead of recompiling.

## 5. Follow-ups / open edges

- **Catch-belief realism** — the chase model is a 1-D pace comparison (chase speed vs
  getaway component). A reachability model that accounts for the defender's current heading/
  turn cost and the carrier's likely acceleration would tighten false positives.
- **Glance trigger** — the carrier currently glances on every forward-dribble tick. Consider
  driving the glance frequency from a learned policy (scan vs commit-to-ball trade-off) so it
  is a decision, not a constant, once the reward exists.
- **Multi-defender** — the assessment is per-defender and independent; a "one creeper, one
  cover" pairing (like `press_cover_target_for`) would avoid two defenders both creeping the
  same blind side.
- **Lateral blind-side** — the geometry recognises the rear arc; a pure side-on (90°) thief
  on a carrier looking straight ahead is only partially covered. Widen if A/B shows value.

## 6. Repo-health note

Verified against the committed HEAD in an isolated worktree: the blindside feature adds
**zero new test reds** (gate-off paths are inert by construction; the one structural edit —
appending the `blindside-steal` option — is conditional on a live assessment). The full-suite
reds observed during verification are (a) pre-existing at HEAD and load-flaky realtime/
threading tests, and (b) failures in pass-recognition, back-four-line and flank/support
ordering that belong to **another agent's concurrent uncommitted edits in the same tree**
(`team.rs` crash-box directive, world.rs forward-recognition refactor). Coordinate before
committing so that in-flight work is not captured under this change.
