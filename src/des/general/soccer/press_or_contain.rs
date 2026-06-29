//! **Press-or-contain decision** — for a defender (especially a central defender, and more
//! generally any player on the team WITHOUT possession who is the one closest to the carrier),
//! the MDP/POMDP choice between *stepping in to PRESS the ball-carrier* (close the gap, try to win
//! it, cut a passing lane, force a hurried release) and *backing OFF to CONTAIN* (hold a goal-side
//! gap, give the carrier space in front and jockey/shepherd them away from the danger zone rather
//! than diving in and being beaten).
//!
//! The decision is a continuous **press aggression** in `[0, 1]` (1 = commit to the press, 0 =
//! sit off and contain) that is a function of the state the user described:
//!
//!   * the likelihood of WINNING the ball / slowing the attack — closer, more *set* (balanced),
//!     goal-side, with cover behind, a better tackler, a slower carrier — all push toward pressing;
//!   * the value of cutting a dangerous forward passing lane — an open forward option for the
//!     carrier rewards stepping in to deny it;
//!   * the likelihood of being BEATEN — a carrier with a pace advantage who could knock it past
//!     and run, OR a runner threatening the space in behind that a committed press would release —
//!     both push toward containing; the danger of being beaten is worse with no cover and little
//!     space behind (deep in our own third).
//!
//! The live seam modulates the *standoff distance* of the existing press/engage producers
//! (`fast_carrier_engage_target_for`, `holder_press_target_for`, `advancing_carrier_stepup_target_for`):
//! a high aggression tightens the standoff (press); a low one widens it (contain / give space).
//!
//! ## Parity
//!
//! Gated **off** in tests and behind a prod kill-switch (`DD_SOCCER_ENABLE_PRESS_OR_CONTAIN`);
//! when off the producers use their original fixed standoff, so play is byte-identical. The inputs
//! builder, the analytic decision, and the standoff helper are all pure / RNG-free.

use super::*;

/// Reference engage distance (yd) over which closeness-to-carrier saturates.
const PRESS_OR_CONTAIN_PROXIMITY_REF_YARDS: f64 = 12.0;
/// Reference speed (yd/s) normalizing the defender's own speed (set-ness) and the carrier's
/// goalward speed.
const PRESS_OR_CONTAIN_SPEED_REF_YPS: f64 = 7.0;
/// Reference pace gap (yd/s) over which a carrier's speed advantage fully scares the defender off.
const PRESS_OR_CONTAIN_PACE_REF_YPS: f64 = 4.0;
/// Reference "space behind" (yd to our own goal line) below which being beaten is treated as
/// maximally dangerous.
const PRESS_OR_CONTAIN_SPACE_REF_YARDS: f64 = 30.0;
/// Radius (yd) within which a goal-side teammate counts as cover for a committed press.
const PRESS_OR_CONTAIN_COVER_RADIUS_YARDS: f64 = 14.0;
/// Radius (yd) within which a goal-side opponent runner counts as an in-behind threat.
const PRESS_OR_CONTAIN_RUNNER_RADIUS_YARDS: f64 = 22.0;
/// Minimum goalward speed (yd/s) for an opponent to count as an active in-behind runner.
const PRESS_OR_CONTAIN_RUNNER_MIN_SPEED_YPS: f64 = 1.5;
/// Min distance (yd) from the nearest marker for a forward-outlet attacker to count as "open"
/// enough that the carrier could realistically pick them out (so pressing to deny the lane pays).
const PRESS_OR_CONTAIN_FORWARD_OUTLET_OPEN_YARDS: f64 = 2.5;
/// Baseline aggression before the win/risk terms are applied (a slight default lean to press, so
/// an evenly-balanced duel still steps in rather than retreating forever).
const PRESS_OR_CONTAIN_BASE_AGGRESSION: f64 = 0.45;

/// The per-defender state the press-or-contain choice is a function of. All fields are resolved in
/// the defender's defending frame and normalized to `[0, 1]` (or a signed share) so the analytic
/// mapping is mirror-invariant and unit-testable in isolation.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PressOrContainInputs {
    /// Closeness to the carrier in `[0, 1]` (1 = on top of them, 0 = ≥ the proximity reference).
    pub proximity: f64,
    /// How *set* / balanced the defender is in `[0, 1]` (1 = stationary & planted, 0 = at full
    /// tilt) — a set defender can commit; one already running can be turned.
    pub set: f64,
    /// Whether the defender is goal-side of the carrier in `[0, 1]` (1 = between carrier and own
    /// goal, 0 = ball-side / wrong-side).
    pub goal_side: f64,
    /// Cover behind in `[0, 1]` — a goal-side teammate who can sweep up if the press is beaten.
    pub cover_behind: f64,
    /// Defender's tackling/defending ability in `[0, 1]`.
    pub tackling: f64,
    /// How SLOW the carrier is going toward our goal in `[0, 1]` (1 = static/sideways, 0 = bearing
    /// down at the reference speed) — a slow carrier is safer to engage.
    pub carrier_slowness: f64,
    /// The carrier's PACE advantage over the defender in `[0, 1]` (1 = much faster) — the chance
    /// they knock it past and run.
    pub pace_disadvantage: f64,
    /// Threat of an opponent runner in the space behind the defender in `[0, 1]` — releasing them
    /// with a beaten press is fatal.
    pub runner_in_behind: f64,
    /// Danger of an open forward passing lane for the carrier in `[0, 1]` — pressing denies the
    /// time/angle for the killer ball.
    pub forward_lane_danger: f64,
    /// How exposed the space behind the defender is in `[0, 1]` (1 = right on our own goal, 0 = far
    /// upfield) — being beaten deep is catastrophic.
    pub exposed_behind: f64,
}

/// Deterministic, RNG-free press-aggression in `[0, 1]` (1 = commit to the press, 0 = sit off and
/// contain). Encodes the coach's prior: lean to press when you can realistically win it or must
/// deny a lane, lean to contain when you'd likely be beaten or would release a runner in behind.
pub(crate) fn analytic_press_aggression(inputs: &PressOrContainInputs) -> f64 {
    let win = inputs.set * 0.22
        + inputs.proximity * 0.22
        + inputs.goal_side * 0.12
        + inputs.cover_behind * 0.20
        + inputs.tackling * 0.18
        + inputs.carrier_slowness * 0.20
        + inputs.forward_lane_danger * 0.18;
    let risk = inputs.pace_disadvantage * 0.45
        + inputs.runner_in_behind * 0.38
        + inputs.exposed_behind * (1.0 - inputs.cover_behind) * 0.30;
    (PRESS_OR_CONTAIN_BASE_AGGRESSION + win - risk).clamp(0.0, 1.0)
}

/// Resolve the standoff (yd, goal-side of the carrier) the defender should hold, blending from a
/// tight `press_yards` at full aggression to a loose `contain_yards` at zero aggression. Pure.
pub(crate) fn press_or_contain_standoff_yards(
    press_yards: f64,
    contain_yards: f64,
    aggression: f64,
) -> f64 {
    let a = aggression.clamp(0.0, 1.0);
    press_yards + (contain_yards - press_yards) * (1.0 - a)
}

impl WorldSnapshot {
    /// Build the [`PressOrContainInputs`] for defender `me` against the current ball-carrier.
    /// Returns `None` when there is no opposing carrier (nothing to press/contain).
    fn press_or_contain_inputs_for(&self, me: &PlayerSnapshot) -> Option<PressOrContainInputs> {
        let holder_id = self.ball.holder?;
        let holder = self.players.iter().find(|p| p.id == holder_id)?;
        if holder.team == me.team || me.role == PlayerRole::Goalkeeper {
            return None;
        }
        let me_pos = self.player_snapshot_position(me);
        let carrier = self.player_snapshot_position(holder);
        let attack_dir = me.team.attack_dir();
        let own_goal_y = self.own_goal_y_for(me.team);

        let distance = me_pos.distance(carrier);
        let proximity = (1.0 - distance / PRESS_OR_CONTAIN_PROXIMITY_REF_YARDS).clamp(0.0, 1.0);

        let my_speed = self.player_velocity(me.id).unwrap_or(me.velocity).len();
        let set = (1.0 - my_speed / PRESS_OR_CONTAIN_SPEED_REF_YPS).clamp(0.0, 1.0);

        // Goal-side: the defender is between the carrier and our own goal (closer to our goal).
        let me_goal_distance = (me_pos.y - own_goal_y).abs();
        let carrier_goal_distance = (carrier.y - own_goal_y).abs();
        let goal_side = if me_goal_distance <= carrier_goal_distance {
            1.0
        } else {
            0.0
        };

        let tackling = ability01(me.skills.defending.max(me.skills.defensive_tracking));

        let carrier_velocity = self.player_velocity(holder_id).unwrap_or(holder.velocity);
        let carrier_goalward_speed = (-(carrier_velocity.y * attack_dir)).max(0.0);
        let carrier_slowness =
            (1.0 - carrier_goalward_speed / PRESS_OR_CONTAIN_SPEED_REF_YPS).clamp(0.0, 1.0);

        let my_top = player_top_speed_yps(me.role, &me.skills);
        let carrier_top = player_top_speed_yps(holder.role, &holder.skills);
        let pace_disadvantage =
            ((carrier_top - my_top) / PRESS_OR_CONTAIN_PACE_REF_YPS).clamp(0.0, 1.0);

        // Cover behind: a goal-side teammate (not me, not the keeper) within range who could sweep
        // up if the press is beaten.
        let cover_count = self
            .players
            .iter()
            .filter(|p| {
                p.team == me.team
                    && p.id != me.id
                    && p.role != PlayerRole::Goalkeeper
                    && p.id != holder_id
            })
            .filter(|p| {
                let pos = self.player_snapshot_position(p);
                (pos.y - me_pos.y) * attack_dir < -1.0
                    && pos.distance(me_pos) <= PRESS_OR_CONTAIN_COVER_RADIUS_YARDS
            })
            .count();
        let cover_behind = (cover_count as f64 / 2.0).clamp(0.0, 1.0);

        // In-behind threat: an opponent attacker goal-side of the defender, running goalward,
        // within range — a committed press that loses the ball releases them.
        let runner_in_behind = self
            .players
            .iter()
            .filter(|p| {
                p.team == holder.team && p.id != holder_id && p.role != PlayerRole::Goalkeeper
            })
            .filter_map(|p| {
                let pos = self.player_snapshot_position(p);
                if (pos.y - me_pos.y) * attack_dir >= 0.0 {
                    return None; // not behind the defender
                }
                if pos.distance(me_pos) > PRESS_OR_CONTAIN_RUNNER_RADIUS_YARDS {
                    return None;
                }
                let vel = self.player_velocity(p.id).unwrap_or(p.velocity);
                let goalward = (-(vel.y * attack_dir)).max(0.0);
                if goalward < PRESS_OR_CONTAIN_RUNNER_MIN_SPEED_YPS {
                    return None;
                }
                Some((goalward / PRESS_OR_CONTAIN_SPEED_REF_YPS).clamp(0.0, 1.0))
            })
            .fold(0.0f64, f64::max);

        // Forward-lane danger: the carrier has an attacker goal-side of THEM (a forward outlet
        // toward our goal) — pressing denies the time/angle to play it.
        let forward_lane_danger = if self.players.iter().any(|p| {
            p.team == holder.team && p.id != holder_id && p.role != PlayerRole::Goalkeeper && {
                let pos = self.player_snapshot_position(p);
                (pos.y - carrier.y) * attack_dir < -3.0
                    && self.nearest_opponent_distance_at(holder.team, pos)
                        >= PRESS_OR_CONTAIN_RUNNER_MIN_SPEED_YPS
            }
        }) {
            (0.5 + carrier_slowness * 0.5).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let exposed_behind =
            (1.0 - me_goal_distance / PRESS_OR_CONTAIN_SPACE_REF_YARDS).clamp(0.0, 1.0);

        Some(PressOrContainInputs {
            proximity,
            set,
            goal_side,
            cover_behind,
            tackling,
            carrier_slowness,
            pace_disadvantage,
            runner_in_behind,
            forward_lane_danger,
            exposed_behind,
        })
    }

    /// The press-or-contain aggression in `[0, 1]` for defender `me` against the current carrier,
    /// or `None` when the feature is gated off or there is no opposing carrier. The single entry
    /// point the press/engage producers call to modulate their standoff.
    pub(crate) fn press_or_contain_aggression_for(&self, me: &PlayerSnapshot) -> Option<f64> {
        if !dd_soccer_enable_press_or_contain() {
            return None;
        }
        let inputs = self.press_or_contain_inputs_for(me)?;
        Some(analytic_press_aggression(&inputs))
    }
}
