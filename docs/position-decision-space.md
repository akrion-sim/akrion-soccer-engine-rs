# Soccer Sim — Per-Position MDP/POMDP Decision Space

> **Companion to [`mdp-pomdp-action-schema.md`](../mdp-pomdp-action-schema.md).** That
> document defines the *shared, factored action space* **A** every agent draws from (pass /
> shoot / dribble / move / press …, each discretised into scored, maskable buckets). **This**
> document maps that space onto the **eleven playing positions**: which decisions each position
> actually faces, and *where in the code* each position-specific decision lives. The goal is to
> make it explicit that **every one of the 11 positions has its own modelled decision context**,
> and to leave clearly-named room for adding more.

## 1. How 11 positions exist on top of 4 roles

The engine's hard type is a coarse 4-value role enum
([`PlayerRole`](../src/des/general/soccer.rs) — `Goalkeeper | Defender | Midfielder | Forward`).
The **eleven positions** are not a second enum; they are derived at decision time from
`role` **plus the player's `home_position`** (its formation anchor), exactly as a real lineup
distinguishes a left-back from a centre-back though both are "defenders":

| Helper (in `world.rs`) | Distinguishes |
|---|---|
| `is_wide_defender(p)` | full-back / wing-back vs centre-back (home x in outer ~28%) |
| `is_wide_midfielder(p)` / `is_wide_attacker(p)` | winger / wide forward vs central mid (home x in outer ~30%) |
| `home_position.x` band | left vs right vs centre channel (see `role_vertical_lane_range`) |
| `home_position.y` + role | line (back / mid / front) |
| `preferences.{defensive,offensive}_mindedness` | holding vs attacking mid; stopper vs ball-player CB |

So a decision is "position-specific" when it is **gated on these predicates** before it can
fire. The sections below enumerate every position and the decisions gated to it.

A position making a decision sees only its **POMDP observation `O`** plus the legal-action
mask, never the full mutable state `S`. That observation is the rich
`SoccerPomdpObservation`/feature-builder surface: role/phase/context scalars plus the
append-only whole-field motion block for 22 players + the ball (position, velocity,
acceleration, jerk and direction channels). Every decision below is therefore a function of
*observable* features only, even when the neural learner also consumes the same field-vector
tail through `SOCCER_NEURAL_FEATURE_DIM`.

## 2. Shared decision families (all positions)

Before the position-specific decisions, every outfield agent runs the common families from the
action schema. These are *not* repeated per position below:

- **On-ball macro**: `pass | shoot | dribble | clear | hold` (`policy_select.rs`,
  `shot_decision.rs`, `pitch_value.rs` for the value/PBRS shaping).
- **Off-ball macro**: `move | run | press | hold | drop` toward a movement target
  (`spacing_target.rs`, `support_scorer.rs`, `field_numbers.rs` for numbers-up reads).
- **Receiving / first touch** (`receive_approach.rs`, `aerial_reception.rs`).
- **Defending** when out of possession: press intensity, jockey vs commit, intercept-lane,
  blindside-steal (`pressing` paths, `slip_break_offside.rs`, blindside-steal).

The per-position sections add the decisions that are **only legal / only scored for that
position**.

## 3. The eleven positions and their position-specific decisions

Formation reference: a 4-3-3 / 4-2-3-1 family. Field is 80 (W) × 120 (L); Home attacks `y = L`.

### 3.1 Goalkeeper (GK)
- **Set-position on the goal line ↔ sweeper-keeper** depth as the ball / loose ball advances.
- **Come vs stay** for crosses and through-balls; **goal-side guard** drop between goal centre
  and a loose ball.
- **Distribution**: throw / roll / short / long, lowered through the shared kick core.
- *Where*: GK positioning tunables + goal-side guard (`world.rs` GK paths,
  `GoalkeeperTunables`); keeper head is gated and currently off in the cluster.

### 3.2 Right-back / Left-back (full-back / wing-back) — wide defenders ×2
- **Overlap vs hold** the back line as the ball goes wide ahead of him.
- **How wide to open** the flank channel, scaled by the ball's lane, and **hold-width-when-ball-
  is-advanced** (don't bomb on over the top of an already-deep ball unless a ground-pass outlet).
- **Lane stickiness**: keep the wide channel in possession instead of drifting infield.
- **Back-four line depth**: push up / drop / step to the offside line as one unit.
- *Where*: `is_wide_defender` gating; `wingback_*` paths and `wide_possession_outlet_target_for`
  (`world.rs`); `back_four_line.rs`; see also `docs/back-four-line-model.md`.

### 3.3 Right / Left centre-back (CB) ×2
- **Step to the line vs drop** (hold the offside trap vs cover in behind) — back-four unit
  decision specialised to the central pair.
- **Engage vs contain** the ball-carrier; **goal-side recovery** on the ball→goal line.
- **Stepping into midfield** to carry / split a press (ball-playing CB), gated on offensive
  mindedness.
- *Where*: `back_four_line.rs`; `defensive_goal_side_*` (`world.rs`); pressing/cover paths.

### 3.4 Defensive / holding midfielder (DM)
- **Screen vs press**: drop to shield the back four vs step out to the ball.
- **Drop between the centre-backs** to make a back-three in build-up (split the CBs).
- **Stay vs join** the box-crash — the holding mid is the one explicitly *excluded* from the
  flood so the team keeps rest-defence (`crash_the_box_target_for` skips a defensive-minded mid).
- *Where*: pressing-audit paths; `crash_the_box_target_for` mindedness gate (`world.rs`);
  team-upfield MARL (`team.rs`).

### 3.5 Central midfielders (CM) ×2 (one box-to-box, one more advanced)
- **Push up to support the carrier vs hold shape** (team upfield-advance MARL).
- **Third-man runs / give-and-go**: initiate or complete a wall pass.
- **Late arrival into the box** for cut-backs (the second-ball / top-of-box zone).
- *Where*: `give_and_go.rs`; `support_scorer.rs`; `long_pass_run.rs`; team-upfield reward
  (`team.rs`); `crash_the_box_target_for` cut-back zone.

### 3.6 Attacking midfielder (CAM / "10")
- **Find the pocket / half-space** between the lines vs check to the ball.
- **Shoot from the top of the box vs slip the through-ball** (shot-trigger vs release).
- **Slip-and-break the offside trap** as the timed runner or the releaser.
- *Where*: `shot_decision.rs`; `slip_break_offside.rs`; `support_scorer.rs`; `pitch_value.rs`.

### 3.7 Right / Left winger (outside mid / wide forward) ×2 — **wide attackers**
This pair has the richest *wide-attack* decision set:
- **Open wide to the touchline**, scaled by the ball's lane (closer ball ⇒ wider) — width,
  overlap and switch outlet.
- **Drive the byline** vs cut inside / stall — solo byline-drive opt-in to the corner then cross.
- **Beat-the-defender 1v1** duel vs keep-distance.
- **★ Stay wide vs pinch in** as the ball reaches the opponent's final third — the decision
  added for this position: hold width (be the deliverer / overlap) vs tuck infield to **get on
  the end of a cross**, into the **half-space (cut-back)** or all the way to the **back post**.
  Discretised buckets `StayWide | PinchHalfSpace | PinchBackPost`, scored from observable
  features (which flank the ball is on, whether a crossing position is on, ball depth in the
  final third, box congestion) with the offside back-post bucket masked.
- *Where*: `is_wide_attacker` gating; `wide_possession_outlet_target_for`,
  `byline_cross_drive_active_for` (`world.rs`); `beat_defender.rs`;
  **`winger_pinch.rs`** (the new stay-wide-vs-pinch decision) →
  `WorldSnapshot::winger_pinch_target_for`.

### 3.8 Striker / centre-forward (ST / "9")
- **Run in behind vs check to feet vs hold-up** the ball (link play).
- **Crash the box**: attack near-post / penalty-spot / far-post / cut-back zones on a cross,
  finishing with a **header / volley** (altitude-gated technique).
- **Shoot vs square** to a better-placed runner; **shape-to-shoot** feint.
- *Where*: `crash_the_box_target_for`, `flank_cross_arrival_target_for` (`world.rs`);
  `aerial_reception.rs`; `shot_decision.rs`; `crash_box.rs` (situational recogniser + reward).

## 4. The winger stay-wide-vs-pinch-in decision (worked example)

This is the position-specific decision built out most fully, as the concrete instance of the
pattern this document describes. See [`winger_pinch.rs`](../src/des/general/soccer/winger_pinch.rs).

**Recogniser (when the decision is live).** In possession, this player `is_wide_attacker`, is
not the carrier or a human-controlled slot, no set-play is active, and the ball is in the
**attacking final third** (`ball_in_attacking_final_third`).

**Observation (POMDP-observable features).**
- `ball_on_my_flank` — is the ball / carrier on the same side of centre as my home channel?
- `crossing_position_on` — is a genuine crossing position on (wide & high, via
  `flank_final_third_crash_box_geometry`)?
- `ball_final_third_depth_frac` — 0 at the final-third edge → 1 at the byline.
- `box_congestion` — own attackers already inside the opponent box.
- `back_post_offside` — would the back-post arrival be offside? (legality mask)

**Action (discretised buckets, scored, masked, argmax).**

| Bucket | Meaning | Leading condition |
|---|---|---|
| `StayWide` | hold the touchline — deliver / overlap / switch | ball on my flank, **or** no cross on yet |
| `PinchHalfSpace` | tuck to the cut-back / top-of-box zone (stays onside) | weak side + cross on, box crowded **or** back post offside |
| `PinchBackPost` | crash the back post (opposite the cross) for the header | weak side + cross on + ball deep + onside |

**Lowering.** The chosen bucket maps to a target on the winger's own side of the goal
(`winger_pinch_target`), then the call site applies the onside clamp and offside guard. It is
applied as an off-ball-target override **before** the late committed box-flood
(`crash_the_box_target_for`), which still wins once the cross is imminent.

**Reward.** Deliberately **none added here**. A pinched winger becomes an arriving attacker and
is credited through the **existing flank cross-arrival / header-goal channel**
(`crash_box.rs`: `CrashBoxArrival`, `HeaderGoalFromCross`), so "getting on the end of the
cross" is reinforced without double-counting. The bucket-scoring weights are calibrated priors,
named so an A/B can tune them and a later **learned head** can replace the scoring while keeping
the same bucket space (the same upgrade path `give_and_go.rs` took).

**Gating.** `winger_pinch_in_enabled()` — `gate_default_on` in production
(`DD_SOCCER_ENABLE_WINGER_PINCH_IN`), **default-OFF under `cfg(test)`** so the suite and current
network weights stay byte-identical until flipped on for a retrain A/B.

## 5. Coverage checklist & room to grow

Every position has at least one modelled, gated decision context (§3). The table below is the
explicit "room for each position" the engine reserves — `✅ built`, `🟡 partial/heuristic`,
`⬜ reserved` (named slot, not yet built).

| Position | Signature decision | Status |
|---|---|---|
| GK | sweeper depth / come-vs-stay / distribution | ✅ / 🟡 (keeper head gated off) |
| Full-back ×2 | overlap-vs-hold, width-by-lane, line depth | ✅ |
| Centre-back ×2 | step-vs-drop, engage-vs-contain, carry-out | ✅ / 🟡 (carry-out) |
| DM | screen-vs-press, drop-to-back-three | ✅ / ⬜ (split-CB build-up) |
| CM ×2 | support-vs-hold, give-and-go, late box arrival | ✅ |
| CAM | pocket / shoot-vs-slip / break-the-line | ✅ |
| Winger ×2 | width-by-lane, byline-drive, **stay-wide-vs-pinch-in**, 1v1 | ✅ |
| Striker | run-in-behind, crash-box, header/volley, shoot-vs-square | ✅ / ⬜ (shape-to-shoot feint) |

**Adding a new position decision** follows the same recipe used by `winger_pinch.rs` /
`crash_box.rs`:
1. New `src/des/general/soccer/<decision>.rs` with a `gate_default_on` switch
   (default-OFF under `cfg(test)`), pure recogniser + discretised bucket scorer + tests.
2. `mod` + `pub(crate) use` it in `soccer.rs`.
3. A `WorldSnapshot` method that gates on the position predicate (§1), gathers POMDP-observable
   features, and lowers the chosen bucket to a target/intent.
4. Wire it into the relevant chain (off-ball target chain, on-ball macro, etc.), ordered so it
   composes with the existing overrides.
5. Reinforce through an existing reward channel where possible; add a new
   `SoccerRewardEventKind` only when the outcome is genuinely uncredited.
