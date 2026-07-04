//! **Field-numbers vector** — the 11v11 kinematic field state, summarized for every
//! player's MDP/POMDP decision as the *numbers-up / numbers-down* picture around them.
//!
//! Every player re-derives intent each tick from a `SoccerPomdpObservation`, but that
//! observation was largely *self-and-ball* centric: the carrier's pressure, the nearest
//! opponent, the best pass lane. What a real player also reads instantly — and what a
//! coach screams about — is the **count and ratio of bodies in front of and behind the
//! ball relative to them**: a 3-v-2 break the attacker must hit NOW, or a back line
//! caught 2-v-4 that has to sprint to recover. That single picture sparks urgency on
//! BOTH sides of the ball, and it was missing from the per-player decision input.
//!
//! This module is that input. From the full field vector — *all* 22 players' position,
//! velocity, acceleration and jerk, projected onto the deciding player's own attack
//! axis — it computes [`SoccerFieldNumbersVector`]:
//!
//!   * the **counts** of teammates / opponents ahead of (toward the attacking goal) and
//!     behind (toward the own goal) the deciding player, and the all-bodies totals;
//!   * the **ratios** ahead and behind (Laplace-smoothed, so 0-v-0 is a well-defined
//!     1.0). Count and ratio are deliberately *both* kept: a 6-v-4 and a 3-v-2 share a
//!     ratio but not a magnitude, and a coach weighs both;
//!   * the signed **overloads** (teammates − opponents) ahead and behind;
//!   * two derived urgencies in `[0,1]` — [`SoccerFieldNumbersVector::offensive_numbers_urgency`]
//!     (an attacking overload ahead you should exploit before the defence recovers) and
//!     [`SoccerFieldNumbersVector::defensive_numbers_urgency`] (an exposure behind you,
//!     opponents goal-side outnumbering cover, you must recover against). Both fold the
//!     attack-axis **momentum** (velocity dominant, acceleration + jerk as small
//!     building-pressure terms) so a number that is *arriving* counts before it lands.
//!
//! ## Parity
//!
//! The live seam ([`crate::des::general::soccer::WorldSnapshot::observation_for`]) only
//! computes this vector — and only folds its urgencies into the observation's
//! offensive/defensive urgency — when the default-ON gate is enabled. Setting
//! `DD_SOCCER_DISABLE_FIELD_NUMBERS_VECTOR` leaves the vector at [`Default`] (all zero)
//! and skips the fold, so the observation and every decision derived from it are
//! byte-identical to before this module existed. The computation here is pure and
//! RNG-free, so the inspector and tests can read it without the live seam.

use super::*;

/// A player can sit this far ahead/behind the deciding player along the attack axis
/// and still count as "level" (neither ahead nor behind). Keeps a body standing
/// shoulder-to-shoulder from flipping the count on sub-yard jitter.
pub const FIELD_NUMBERS_LEVEL_EPSILON_YARDS: f64 = 1.0;

/// Overload magnitude (bodies) at which the offensive/defensive numbers urgency from the
/// count term saturates. Being +4 ahead is already a full-tilt break; more does not add
/// urgency, it adds comfort.
pub const FIELD_NUMBERS_OVERLOAD_SATURATION: f64 = 4.0;

/// Reference attack-axis speed (yards/second) that maps a body's closing/surging run to a
/// full momentum contribution. ~7 yps is a committed sprinting stride.
pub const FIELD_NUMBERS_MOMENTUM_REFERENCE_YPS: f64 = 7.0;

/// One player's kinematic state, already projected onto the *deciding* player's attack
/// frame (so the whole vector is mirror-invariant — Home and Away read identically).
/// `forward_*` are signed along the attack axis: positive = toward the attacking goal.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FieldNumbersBody {
    /// True for a teammate of the deciding player, false for an opponent.
    pub is_teammate: bool,
    /// Attack-axis position of this body relative to the deciding player (yards):
    /// positive = ahead (toward the attacking goal), negative = behind (toward own goal).
    pub forward_offset_yards: f64,
    /// Attack-axis velocity (yards/second): positive = moving toward the attacking goal.
    pub forward_velocity_yps: f64,
    /// Attack-axis acceleration (yards/second²): positive = building speed up-field.
    pub forward_acceleration_yps2: f64,
    /// Attack-axis jerk (yards/second³): positive = an explosive up-field burst starting.
    pub forward_jerk_yps3: f64,
}

impl FieldNumbersBody {
    /// Ahead of the deciding player (toward the attacking goal) beyond the level band.
    fn is_ahead(&self) -> bool {
        self.forward_offset_yards > FIELD_NUMBERS_LEVEL_EPSILON_YARDS
    }

    /// Behind the deciding player (toward the own goal) beyond the level band.
    fn is_behind(&self) -> bool {
        self.forward_offset_yards < -FIELD_NUMBERS_LEVEL_EPSILON_YARDS
    }

    /// Up-field momentum scalar (toward the attacking goal) in `[0, ~1]`: velocity
    /// dominant, with acceleration and jerk as small "this run is still building" terms.
    /// Negative when the body is moving toward its own goal.
    fn up_field_momentum(&self) -> f64 {
        (self.forward_velocity_yps
            + self.forward_acceleration_yps2 * 0.12
            + self.forward_jerk_yps3 * 0.02)
            / FIELD_NUMBERS_MOMENTUM_REFERENCE_YPS
    }
}

/// The numbers-up / numbers-down picture around one deciding player, derived from the
/// full 11v11 field vector. See the module docs for the meaning of each field.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerFieldNumbersVector {
    /// Teammates ahead of the deciding player (toward the attacking goal).
    pub teammates_ahead: usize,
    /// Teammates behind the deciding player (toward the own goal).
    pub teammates_behind: usize,
    /// Opponents ahead of the deciding player (toward the attacking goal).
    pub opponents_ahead: usize,
    /// Opponents behind the deciding player (toward the own goal).
    pub opponents_behind: usize,
    /// All bodies (both teams) ahead of the deciding player.
    pub players_ahead: usize,
    /// All bodies (both teams) behind the deciding player.
    pub players_behind: usize,
    /// Laplace-smoothed teammate:opponent ratio ahead — `(teammates+1)/(opponents+1)`.
    /// `> 1` = attacking overload ahead, `< 1` = outnumbered ahead, `1` = level (incl. 0-v-0).
    pub ahead_ratio: f64,
    /// Laplace-smoothed teammate:opponent ratio behind — `(teammates+1)/(opponents+1)`.
    /// `< 1` = exposed behind (opponents goal-side outnumber cover).
    pub behind_ratio: f64,
    /// Signed teammate − opponent overload ahead (positive = numbers up going forward).
    pub ahead_overload: i32,
    /// Signed teammate − opponent overload behind (negative = numbers down at the back).
    pub behind_overload: i32,
    /// `[0,1]` urgency to exploit an attacking overload ahead before the defence recovers.
    /// Zero when not numbers-up ahead. Scaled by both the overload count and the ratio,
    /// lifted by teammates' up-field momentum.
    pub offensive_numbers_urgency: f64,
    /// `[0,1]` urgency to recover against an exposure behind (opponents goal-side
    /// outnumbering cover). Zero when not numbers-down behind. Scaled by the deficit
    /// count and ratio, lifted by those opponents' momentum at the own goal.
    pub defensive_numbers_urgency: f64,
    /// Signed `[-1,1]` overall tilt: `offensive_numbers_urgency − defensive_numbers_urgency`.
    /// Positive = a situation to attack, negative = a situation to defend.
    pub numbers_advantage: f64,
    // --- Team centroids, in the deciding player's attack frame (forward = toward the
    // attacking goal). "own" includes the deciding player; "opp" is the opposing team.
    // These read the *gestalt* of each team off the full field vector — a whole-team
    // shove forward or a committed counter — that the local ahead/behind counts miss. ---
    /// Own team's centre of mass: mean attack-axis position offset (yards) relative to the
    /// deciding player. Positive = own team is, on average, ahead of (up-field of) them.
    pub own_center_of_mass_forward_yards: f64,
    /// Opposing team's centre of mass: mean attack-axis position offset (yards) relative to
    /// the deciding player. Negative = opponents sit, on average, goal-side of them.
    pub opp_center_of_mass_forward_yards: f64,
    /// Own team's centre of velocity: mean attack-axis velocity (yps). Positive = the team is
    /// collectively driving toward the attacking goal.
    pub own_center_of_velocity_forward_yps: f64,
    /// Opposing team's centre of velocity: mean attack-axis velocity (yps). Negative = the
    /// opponents are collectively bearing down toward the own goal (a committed attack).
    pub opp_center_of_velocity_forward_yps: f64,
    /// Own team's centre of acceleration: mean attack-axis acceleration (yps²) — the team's
    /// building push up-field.
    pub own_center_of_acceleration_forward_yps2: f64,
    /// Opposing team's centre of acceleration: mean attack-axis acceleration (yps²).
    pub opp_center_of_acceleration_forward_yps2: f64,
    /// Net territorial push toward the attacking goal: own − opp centre of velocity (yps).
    /// Positive = our side is winning the collective shove up-field.
    pub team_push_forward_yps: f64,
}

/// Laplace-smoothed `(teammates+1)/(opponents+1)` ratio — well-defined for 0-v-0 (→ 1.0).
fn smoothed_ratio(teammates: usize, opponents: usize) -> f64 {
    (teammates as f64 + 1.0) / (opponents as f64 + 1.0)
}

/// Build the field-numbers vector from every *other* body on the pitch (`bodies`) plus the
/// deciding player's own attack-axis kinematics (`self_forward_velocity_yps` /
/// `self_forward_acceleration_yps2`; their position offset is the frame origin, 0). The
/// deciding player is excluded from the ahead/behind counts but INCLUDED in their own
/// team's centroids. `team_has_possession` gates the offensive-urgency term: you only
/// "exploit a break" when your side has the ball.
///
/// Pure and deterministic: identical inputs ⇒ identical output, no RNG, no env reads.
pub fn compute_field_numbers(
    bodies: &[FieldNumbersBody],
    self_forward_velocity_yps: f64,
    self_forward_acceleration_yps2: f64,
    team_has_possession: bool,
) -> SoccerFieldNumbersVector {
    let mut teammates_ahead = 0usize;
    let mut teammates_behind = 0usize;
    let mut opponents_ahead = 0usize;
    let mut opponents_behind = 0usize;
    // Aggregate up-field momentum of teammates breaking ahead, and of opponents
    // bearing down toward our goal among the bodies behind us.
    let mut teammate_ahead_momentum = 0.0_f64;
    let mut opponent_behind_threat = 0.0_f64;

    // Team centroid accumulators (attack-axis). The deciding player seeds their own team:
    // position offset 0 (the frame origin) and their own velocity/acceleration.
    let mut own_count = 1.0_f64;
    let mut own_pos_sum = 0.0_f64;
    let mut own_vel_sum = self_forward_velocity_yps;
    let mut own_acc_sum = self_forward_acceleration_yps2;
    let mut opp_count = 0.0_f64;
    let mut opp_pos_sum = 0.0_f64;
    let mut opp_vel_sum = 0.0_f64;
    let mut opp_acc_sum = 0.0_f64;

    for body in bodies {
        let ahead = body.is_ahead();
        let behind = body.is_behind();
        if body.is_teammate {
            own_count += 1.0;
            own_pos_sum += body.forward_offset_yards;
            own_vel_sum += body.forward_velocity_yps;
            own_acc_sum += body.forward_acceleration_yps2;
            if ahead {
                teammates_ahead += 1;
                teammate_ahead_momentum += body.up_field_momentum().max(0.0);
            } else if behind {
                teammates_behind += 1;
            }
        } else {
            opp_count += 1.0;
            opp_pos_sum += body.forward_offset_yards;
            opp_vel_sum += body.forward_velocity_yps;
            opp_acc_sum += body.forward_acceleration_yps2;
            if ahead {
                opponents_ahead += 1;
            } else if behind {
                opponents_behind += 1;
                // An opponent behind us (goal-side) running AT our goal has negative
                // up-field momentum in our frame; flip the sign to read it as threat.
                opponent_behind_threat += (-body.up_field_momentum()).max(0.0);
            }
        }
    }

    let ahead_overload = teammates_ahead as i32 - opponents_ahead as i32;
    let behind_overload = teammates_behind as i32 - opponents_behind as i32;
    let ahead_ratio = smoothed_ratio(teammates_ahead, opponents_ahead);
    let behind_ratio = smoothed_ratio(teammates_behind, opponents_behind);

    let own_center_of_mass_forward_yards = own_pos_sum / own_count;
    let own_center_of_velocity_forward_yps = own_vel_sum / own_count;
    let own_center_of_acceleration_forward_yps2 = own_acc_sum / own_count;
    // Opponent centroids are 0 when there are no opponents in the field vector (test setups).
    let (
        opp_center_of_mass_forward_yards,
        opp_center_of_velocity_forward_yps,
        opp_center_of_acceleration_forward_yps2,
    ) = if opp_count > 0.0 {
        (
            opp_pos_sum / opp_count,
            opp_vel_sum / opp_count,
            opp_acc_sum / opp_count,
        )
    } else {
        (0.0, 0.0, 0.0)
    };
    let team_push_forward_yps =
        own_center_of_velocity_forward_yps - opp_center_of_velocity_forward_yps;

    // Team-shape momentum terms, [0,1], folded as a SMALL lift on top of the local
    // overload urgencies so a whole-team shove counts even when the body count is level.
    // Offensive: our centre of velocity driving forward (possession-gated). Defensive: the
    // opposing centre of velocity bearing toward our goal while their centre of mass already
    // sits goal-side of us (a committed counter we must track back against).
    let own_push_term =
        (own_center_of_velocity_forward_yps / FIELD_NUMBERS_MOMENTUM_REFERENCE_YPS).clamp(0.0, 1.0);
    let opp_push_term = if opp_center_of_mass_forward_yards < 0.0 {
        (-opp_center_of_velocity_forward_yps / FIELD_NUMBERS_MOMENTUM_REFERENCE_YPS).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // Offensive: a positive overload ahead, scaled by count (saturating) AND ratio, lifted
    // by how fast those teammates are arriving and by the team's forward shove. Zero unless
    // we both hold the ball and actually have the extra body forward.
    let offensive_numbers_urgency = if team_has_possession && ahead_overload > 0 {
        let count_term = (ahead_overload as f64 / FIELD_NUMBERS_OVERLOAD_SATURATION).clamp(0.0, 1.0);
        let ratio_term = (ahead_ratio - 1.0).clamp(0.0, 1.0);
        let momentum_term = (teammate_ahead_momentum / FIELD_NUMBERS_OVERLOAD_SATURATION).clamp(0.0, 1.0);
        (count_term * 0.50 + ratio_term * 0.25 + momentum_term * 0.13 + own_push_term * 0.12)
            .clamp(0.0, 1.0)
    } else {
        0.0
    };

    // Defensive: opponents goal-side of us outnumber the cover behind (negative overload
    // behind), scaled by count (saturating) AND inverse ratio, lifted by those opponents'
    // momentum at our goal and by the opposing team's committed counter. Independent of
    // possession — a recovering player feels it even off the ball.
    let behind_deficit = (-behind_overload).max(0);
    let defensive_numbers_urgency = if behind_deficit > 0 {
        let count_term = (behind_deficit as f64 / FIELD_NUMBERS_OVERLOAD_SATURATION).clamp(0.0, 1.0);
        let ratio_term = (1.0 - behind_ratio).clamp(0.0, 1.0);
        let momentum_term = (opponent_behind_threat / FIELD_NUMBERS_OVERLOAD_SATURATION).clamp(0.0, 1.0);
        (count_term * 0.50 + ratio_term * 0.25 + momentum_term * 0.13 + opp_push_term * 0.12)
            .clamp(0.0, 1.0)
    } else {
        0.0
    };

    SoccerFieldNumbersVector {
        teammates_ahead,
        teammates_behind,
        opponents_ahead,
        opponents_behind,
        players_ahead: teammates_ahead + opponents_ahead,
        players_behind: teammates_behind + opponents_behind,
        ahead_ratio,
        behind_ratio,
        ahead_overload,
        behind_overload,
        offensive_numbers_urgency,
        defensive_numbers_urgency,
        numbers_advantage: offensive_numbers_urgency - defensive_numbers_urgency,
        own_center_of_mass_forward_yards,
        opp_center_of_mass_forward_yards,
        own_center_of_velocity_forward_yps,
        opp_center_of_velocity_forward_yps,
        own_center_of_acceleration_forward_yps2,
        opp_center_of_acceleration_forward_yps2,
        team_push_forward_yps,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn teammate_at(forward: f64) -> FieldNumbersBody {
        FieldNumbersBody {
            is_teammate: true,
            forward_offset_yards: forward,
            ..Default::default()
        }
    }

    fn opponent_at(forward: f64) -> FieldNumbersBody {
        FieldNumbersBody {
            is_teammate: false,
            forward_offset_yards: forward,
            ..Default::default()
        }
    }

    #[test]
    fn empty_field_is_neutral() {
        let v = compute_field_numbers(&[], 0.0, 0.0, true);
        assert_eq!(v.players_ahead, 0);
        assert_eq!(v.players_behind, 0);
        assert_eq!(v.ahead_ratio, 1.0);
        assert_eq!(v.behind_ratio, 1.0);
        assert_eq!(v.ahead_overload, 0);
        assert_eq!(v.offensive_numbers_urgency, 0.0);
        assert_eq!(v.defensive_numbers_urgency, 0.0);
    }

    #[test]
    fn level_band_counts_as_neither() {
        // Half a yard either side of the deciding player is "level".
        let bodies = [teammate_at(0.5), opponent_at(-0.5)];
        let v = compute_field_numbers(&bodies, 0.0, 0.0, true);
        assert_eq!(v.players_ahead, 0);
        assert_eq!(v.players_behind, 0);
    }

    #[test]
    fn three_v_two_break_sparks_offensive_urgency() {
        // Three teammates ahead, two opponents ahead: numbers up going forward.
        let bodies = [
            teammate_at(10.0),
            teammate_at(15.0),
            teammate_at(20.0),
            opponent_at(12.0),
            opponent_at(18.0),
        ];
        let v = compute_field_numbers(&bodies, 0.0, 0.0, true);
        assert_eq!(v.teammates_ahead, 3);
        assert_eq!(v.opponents_ahead, 2);
        assert_eq!(v.ahead_overload, 1);
        assert!(v.ahead_ratio > 1.0);
        assert!(v.offensive_numbers_urgency > 0.0);
        assert_eq!(v.defensive_numbers_urgency, 0.0);
        assert!(v.numbers_advantage > 0.0);
    }

    #[test]
    fn possession_gates_offensive_urgency() {
        let bodies = [teammate_at(10.0), teammate_at(15.0), opponent_at(12.0)];
        let with_ball = compute_field_numbers(&bodies, 0.0, 0.0, true);
        let without_ball = compute_field_numbers(&bodies, 0.0, 0.0, false);
        assert!(with_ball.offensive_numbers_urgency > 0.0);
        assert_eq!(without_ball.offensive_numbers_urgency, 0.0);
    }

    #[test]
    fn outnumbered_at_the_back_sparks_defensive_urgency_off_ball() {
        // Two opponents goal-side of us, only one covering teammate: exposed at the back.
        let bodies = [
            opponent_at(-10.0),
            opponent_at(-18.0),
            teammate_at(-14.0),
        ];
        // Possession off — defensive urgency must still fire for a recovering player.
        let v = compute_field_numbers(&bodies, 0.0, 0.0, false);
        assert_eq!(v.opponents_behind, 2);
        assert_eq!(v.teammates_behind, 1);
        assert_eq!(v.behind_overload, -1);
        assert!(v.behind_ratio < 1.0);
        assert!(v.defensive_numbers_urgency > 0.0);
        assert!(v.numbers_advantage < 0.0);
    }

    #[test]
    fn count_and_ratio_differ_same_ratio_more_bodies() {
        // 2-v-1 and 4-v-2 share a ratio band but the bigger break carries more count.
        let small = compute_field_numbers(
            &[teammate_at(10.0), teammate_at(12.0), opponent_at(11.0)],
            0.0,
            0.0,
            true,
        );
        let big = compute_field_numbers(
            &[
                teammate_at(10.0),
                teammate_at(12.0),
                teammate_at(14.0),
                teammate_at(16.0),
                opponent_at(11.0),
                opponent_at(13.0),
            ],
            0.0,
            0.0,
            true,
        );
        assert_eq!(small.ahead_overload, 1);
        assert_eq!(big.ahead_overload, 2);
        assert!(big.offensive_numbers_urgency > small.offensive_numbers_urgency);
    }

    #[test]
    fn arriving_momentum_lifts_urgency() {
        let still = [teammate_at(10.0), teammate_at(15.0), opponent_at(12.0)];
        let mut surging = still;
        // Teammates sprinting onto the break.
        surging[0].forward_velocity_yps = 7.0;
        surging[1].forward_velocity_yps = 7.0;
        let v_still = compute_field_numbers(&still, 0.0, 0.0, true);
        let v_surging = compute_field_numbers(&surging, 0.0, 0.0, true);
        assert!(v_surging.offensive_numbers_urgency > v_still.offensive_numbers_urgency);
    }

    #[test]
    fn team_centroids_average_position_velocity_acceleration() {
        // Two teammates ahead (+10, +20), one opponent behind (-15). The deciding player
        // seeds their own team at offset 0 with the given self kinematics.
        let mut t1 = teammate_at(10.0);
        let mut t2 = teammate_at(20.0);
        t1.forward_velocity_yps = 4.0;
        t2.forward_velocity_yps = 6.0;
        t1.forward_acceleration_yps2 = 2.0;
        t2.forward_acceleration_yps2 = 4.0;
        let mut opp = opponent_at(-15.0);
        opp.forward_velocity_yps = -3.0; // bearing toward our goal
        opp.forward_acceleration_yps2 = -1.5;
        let v = compute_field_numbers(&[t1, t2, opp], 1.0, 0.0, true);
        // Own centre of mass = mean(0 [self], 10, 20) = 10.
        assert!((v.own_center_of_mass_forward_yards - 10.0).abs() < 1e-9);
        // Own centre of velocity = mean(1 [self], 4, 6) = 11/3.
        assert!((v.own_center_of_velocity_forward_yps - 11.0 / 3.0).abs() < 1e-9);
        // Own centre of acceleration = mean(0 [self], 2, 4) = 2.
        assert!((v.own_center_of_acceleration_forward_yps2 - 2.0).abs() < 1e-9);
        // Single opponent ⇒ its centroids are its own values.
        assert!((v.opp_center_of_mass_forward_yards + 15.0).abs() < 1e-9);
        assert!((v.opp_center_of_velocity_forward_yps + 3.0).abs() < 1e-9);
        // Net push = own COV − opp COV = 11/3 − (−3) = 11/3 + 3 > 0 (we are shoving up-field).
        assert!(v.team_push_forward_yps > 0.0);
        assert!((v.team_push_forward_yps - (11.0 / 3.0 + 3.0)).abs() < 1e-9);
    }

    #[test]
    fn committed_opposing_counter_lifts_defensive_urgency() {
        // A one-body deficit behind (2 opponents goal-side, 1 covering teammate) fires the
        // base defensive urgency in both cases; the opposing team's centre of velocity
        // bearing toward our goal (a committed counter) must lift it further.
        let still = [opponent_at(-12.0), opponent_at(-18.0), teammate_at(-12.0)];
        let mut driving = still;
        driving[0].forward_velocity_yps = -6.0; // opponents driving at our goal
        driving[1].forward_velocity_yps = -6.0;
        let v_still = compute_field_numbers(&still, 0.0, 0.0, false);
        let v_driving = compute_field_numbers(&driving, 0.0, 0.0, false);
        assert!(v_driving.defensive_numbers_urgency > v_still.defensive_numbers_urgency);
    }
}
