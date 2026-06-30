//! Dynamic, learnable **back-four line model** (forward / y-axis line depth).
//!
//! The forward position of the back four is not a fixed formation slot: at every
//! tick the line has a *centre* (the average y of the defenders holding it) and
//! that centre should be a dynamic function of the match state. This module is
//! the seam through which that function is computed — analytically today, by a
//! neural net once trained — and fed into the single live line chokepoint,
//! [`WorldSnapshot::defender_line_band_average_adjusted_y`].
//!
//! The controlled quantity is the line **centre**, expressed as a *gap fraction*
//! in `[0, 1]`: `0` = the line sits level with the (predicted) ball (highest line
//! / offside trap), `1` = the full `max_gap` behind it (deepest block). The
//! caller maps that to a forward coordinate and re-clamps it to the legal band,
//! so the model chooses *where inside the legal line* the four sit, never an
//! illegal line.
//!
//! Inputs (the 7 groups the model is a function of), all in the defending team's
//! attacking frame so the model is mirror-invariant:
//!   1. ball position / velocity / acceleration,
//!   2. opponent vector — position / velocity / acceleration,
//!   3. the back four's own position / velocity / acceleration,
//!   4. opponent centre of mass (folded into group 2's centroid),
//!   5. which team controls the ball,
//!   6. our own goalkeeper's position,
//!   7. offside-trap state now and (implicitly, via velocity/accel + the predicted
//!      ball) over the next ~3 s.
//!
//! ## Parity
//!
//! Gated **off** by default (`DD_SOCCER_ENABLE_BACK_FOUR_LINE_MODEL`). When unset,
//! [`WorldSnapshot::back_four_line_model_centre_fwd`] returns `None` before
//! building any features, so the line chokepoint is byte-identical to before this
//! module existed. The feature build and analytic fraction are pure and RNG-free,
//! so tests / the inspector can read them without enabling the live seam.
//!
//! See `docs/back-four-line-model.md` for the full design + the path to a learned
//! head.

use super::*;
use crate::des::general::neural_network::{ActivationName, FeedForwardNetwork, RandomNetworkSpec};
use crate::des::general::prng::mulberry32;

/// Number of features in the back-four line state vector. The exact ordering is
/// defined once in [`BackFourLineInputs::to_features`].
pub const BACK_FOUR_LINE_FEATURE_DIM: usize = 27;
/// Weight on the model's own depth vs. the retained existing heuristic centre in
/// the analytic seed (group 8: *leave in existing determinants*). `0` = pure
/// heuristic (parity-ish), `1` = pure model. Kept modest so the well-tuned
/// existing line dominates until a trained head earns more of the blend; a
/// promoted [`BackFourLineHead`] can drive this toward `1`.
pub const BACK_FOUR_LINE_MODEL_BLEND: f64 = 0.5;
/// Hidden width of the learned regression head.
pub const BACK_FOUR_LINE_HIDDEN_UNITS: usize = 24;

/// Reference top speed (yd/s) used to normalize velocity features.
const REF_TOP_SPEED_YPS: f64 = 9.0;
/// Reference acceleration (yd/s²) used to normalize acceleration features.
const REF_ACCEL_YPS2: f64 = 8.0;

/// Env gate enabling the dynamic line model at the live chokepoint. Off (unset)
/// by default so an unconfigured process stays byte-identical.
const BACK_FOUR_LINE_MODEL_ENABLE_ENV: &str = "DD_SOCCER_ENABLE_BACK_FOUR_LINE_MODEL";
/// Env gate enabling the dynamic MIDFIELD line model (dynamic depth + an actual
/// flat line + the 5s consistency window) at the midfield band chokepoint. Off by
/// default ⇒ byte-identical to the existing fixed-ideal midfield band.
const MIDFIELD_LINE_MODEL_ENABLE_ENV: &str = "DD_SOCCER_ENABLE_MIDFIELD_LINE_MODEL";
/// Env gate enabling the **v2 field-anchored back-four line depth**: the line
/// CENTRE is anchored on the [`BACK_FOUR_LINE_ANCHOR_DEPTH_YARDS`] line while the
/// ball is upfield of it, tracks the ball to parity between the keeper's 6 and the
/// anchor, sticks at the 6 inside it, and trails a dynamic
/// possession-aware gap (5..40 in possession, 20..40 out) behind a deep ball, pressed up to fill
/// the space in front of the foremost attackers, never past the halfway+cap. **Default-ON** in
/// production (this is the geometry-anchored centre that removes the self-referential "sine-wave");
/// `DD_SOCCER_ENABLE_BACK_FOUR_LINE_DEPTH_V2=0` is the kill switch back to the legacy ball-relative
/// 20-40yd average band. Default-OFF under test so the line-depth parity suite stays byte-identical.
const BACK_FOUR_LINE_DEPTH_V2_ENABLE_ENV: &str = "DD_SOCCER_ENABLE_BACK_FOUR_LINE_DEPTH_V2";
/// Env gate enabling **wingback-first forward priority** on the back-four line band.
/// Without this, the 20-40yd ball cushion / flat offside line is enforced on the AVERAGE
/// of all four defenders, which couples them badly: high or overlapping wingbacks raise
/// the average, and the flat-line clamp then drags the two CENTRAL defenders FORWARD onto
/// it (the "centre-backs rush up toward the ball" fault), while conserving that average
/// yanks the advanced wingbacks BACKWARD to rebalance. With this on, the line + cushion are
/// anchored on the CENTRAL defenders alone; the wide defenders (wingbacks) are decoupled —
/// free to follow their own (typically more advanced) target and only kept from dropping
/// BEHIND the central line. So the centre-backs hold the band (and on a turnover DROP onto
/// it rather than step up), and the wingbacks get forward first. Off (unset) ⇒ the
/// symmetric all-four-average line stands (byte-identical).
const DEFENSIVE_LINE_WINGBACK_FORWARD_PRIORITY_ENABLE_ENV: &str =
    "DD_SOCCER_ENABLE_WINGBACK_FORWARD_PRIORITY";
/// Env gate (default-ON) for the press-to-attackers compaction of the v2 line centre.
const BACK_FOUR_ATTACKER_PRESS_ENABLE_ENV: &str = "DD_SOCCER_ENABLE_BACK_FOUR_ATTACKER_PRESS";
/// Env gate (default-ON) for the energy-conservation hold deadband on the v2 line target.
const BACK_FOUR_LINE_HOLD_DEADBAND_ENABLE_ENV: &str = "DD_SOCCER_ENABLE_BACK_FOUR_LINE_HOLD_DEADBAND";

/// Comfortable resting gap (yd behind the ball) the CENTRAL defenders hold while defending
/// under wingback-first priority. Decoupling the wingbacks from the line average removed the
/// (incidental) restoring pull that deep-dropping wingbacks used to give the four-man
/// average, letting the centre-back line ride up onto the 20yd edge and break inside it. So
/// the central line is floored DEEPER than this gap (never closer to the ball), keeping it
/// off the 20yd edge and firmly inside the legal 20-40yd band, while still capped at 40 by
/// the band itself. Only ever pulls the line DEEPER — a line already sitting deeper than this
/// is left alone — and only bites when the ball is upfield (in our own third the shelf / 6yd
/// floor take over). Active ball-challengers are already exempt upstream.
pub const DEFENSIVE_LINE_CENTRAL_RESTING_GAP_YARDS: f64 = 27.0;

/// Own-goal depth (yd) the back-four AVERAGE is anchored to while the ball is
/// upfield of it: the line holds here (offside trap) until the ball penetrates
/// inside it, then tracks the ball back to parity, then sticks at the keeper's 6.
pub const BACK_FOUR_LINE_ANCHOR_DEPTH_YARDS: f64 = 15.0;
/// Trailing-gap band (yd behind the ball) in the rising regime once the ball is
/// deep upfield (`ball_depth > anchor + gap`). The dynamic line head picks within
/// it: the min is a higher line (smaller gap), the max a deeper line.
pub const BACK_FOUR_LINE_DESIRED_GAP_MIN_YARDS: f64 = 20.0;
/// While we control possession, the back four may close the ball-to-line gap
/// much more aggressively because the opponent cannot immediately receive in
/// behind from our own attack phase.
pub const BACK_FOUR_LINE_POSSESSION_GAP_MIN_YARDS: f64 = 5.0;
pub const BACK_FOUR_LINE_DESIRED_GAP_MAX_YARDS: f64 = 40.0;
/// Neutral gap fraction (mid-band) used when the line-depth inputs are degenerate
/// (a roster with no defenders/opponents/keeper) so the desired gap falls back to
/// the middle of the 20-40 band rather than an arbitrary literal.
pub const BACK_FOUR_LINE_NEUTRAL_GAP_FRACTION: f64 = 0.5;
/// Target spacing from the back four to the opponent's foremost four attackers.
/// Smaller values mean the defensive line steps up harder to stay connected.
pub const BACK_FOUR_FOREMOST_FOUR_ATTACKER_IDEAL_GAP_IN_POSSESSION_YARDS: f64 = 24.0;
pub const BACK_FOUR_FOREMOST_FOUR_ATTACKER_IDEAL_GAP_DISPOSSESSION_YARDS: f64 = 26.0;
pub const BACK_FOUR_FOREMOST_FOUR_ATTACKER_IDEAL_GAP_DEFENDING_YARDS: f64 = 30.0;
pub const BACK_FOUR_FOREMOST_FOUR_ATTACKER_MIN_GAP_YARDS: f64 = 14.0;
pub const BACK_FOUR_FOREMOST_FOUR_ATTACKER_MAX_PUSH_YARDS: f64 = 12.0;

/// Minimum trailing gap (yd behind the ball) the back four holds **while WE control the ball**.
/// In possession the line may step right up to support the attack, so the floor drops from the
/// out-of-possession [`BACK_FOUR_LINE_DESIRED_GAP_MIN_YARDS`] (20) to this (5). "Dispossession" —
/// the opponent controlling OR a loose/contested ball with NO controller — keeps the deeper 20yd
/// floor. So the dynamic trailing gap is **5..40 in possession, 20..40 out of possession**. The
/// 15yd anchor non-linearity ([`BACK_FOUR_LINE_ANCHOR_DEPTH_YARDS`]) still governs the deep-ball
/// regime in both cases.
pub const BACK_FOUR_LINE_DESIRED_GAP_IN_POSSESSION_MIN_YARDS: f64 = 5.0;

/// MARL/MAPPO seed for the **optimal distance** (yd) the back four sits goal-side of the opponent's
/// foremost-attacker line while in possession or dispossession: the line presses UP to fill the
/// space between the four and the opponent's most advanced attackers rather than leaving a large
/// hole for a pass into feet. Only ever raises a too-deep centre toward the attackers — never steps
/// the line AHEAD of them (that plays them onside). The learned line-depth head refines the live
/// depth via the attacker-compactness reward term; this is the analytic seed / fallback.
pub const BACK_FOUR_OPTIMAL_GAP_TO_ATTACKERS_YARDS: f64 = 12.0;
/// How many of the opponent's most-advanced outfielders define the "foremost attackers" line the
/// back four compresses the space toward (the user's "foremost 4 opponent attackers").
pub const BACK_FOUR_FOREMOST_ATTACKERS_COUNT: usize = 4;

/// Energy-conservation **hold deadband** (yd) on the back-four line target: a line-bound defender
/// already within this distance of its (legal) line target HOLDS rather than re-chasing a target
/// that jitters a yard or two each tick with the predicted ball — the "sine-wave" walk-stop-walk.
/// Always overridden (the defender is corrected) when it sits illegally AHEAD of the offside cap,
/// so the deadband only ever leaves a defender a touch DEEPER than ideal (never plays a runner
/// onside). Damps the cosmetic oscillation and saves the effort the wasted-energy penalty docks.
pub const BACK_FOUR_LINE_HOLD_DEADBAND_YARDS: f64 = 2.0;
/// Per-yard weight of the attacker-compactness term folded into the line-depth RL reward when the
/// attacker-press is on: each yard the back four sits behind the optimal gap to the foremost
/// attackers docks the reward by this much, so the learned head prefers ball-gap fractions that
/// keep the four compact with the attackers. Small relative to the territorial-advantage delta.
pub const BACK_FOUR_ATTACKER_COMPACTNESS_REWARD_PER_YARD: f64 = 0.02;

fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|raw| {
            let v = raw.trim().to_ascii_lowercase();
            matches!(v.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

/// Whether the dynamic back-four line model is consulted at the live line
/// chokepoint this process. Off ⇒ the analytic heuristic line stands (parity).
pub fn back_four_line_model_enabled() -> bool {
    #[cfg(test)]
    {
        env_flag_enabled(BACK_FOUR_LINE_MODEL_ENABLE_ENV)
    }
    #[cfg(not(test))]
    {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| env_flag_enabled(BACK_FOUR_LINE_MODEL_ENABLE_ENV))
    }
}

/// Whether the dynamic MIDFIELD line model is consulted at the midfield band
/// chokepoint this process. Off ⇒ the existing fixed-ideal band stands (parity).
pub fn midfield_line_model_enabled() -> bool {
    #[cfg(test)]
    {
        env_flag_enabled(MIDFIELD_LINE_MODEL_ENABLE_ENV)
    }
    #[cfg(not(test))]
    {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| env_flag_enabled(MIDFIELD_LINE_MODEL_ENABLE_ENV))
    }
}

/// Whether the v2 field-anchored back-four line depth is used at the live line
/// chokepoint this process. The v2 centre is a function of **field geometry + the
/// ball** (15yd anchor, dynamic possession-aware trailing gap, attacker compaction)
/// rather than the AVERAGE of where the four currently are — so the four no longer
/// chase a self-referential, jittering line average (the cosmetic "sine-wave"). It
/// is the home of the 15yd non-linearity, the 5..40/20..40 possession band, the
/// press-to-attackers compaction, and the hold deadband, so it is promoted to
/// **default-ON** in production (env `DD_SOCCER_ENABLE_BACK_FOUR_LINE_DEPTH_V2=0`
/// is the kill switch, reverting to the legacy ball-relative average band).
/// Default-OFF under test so the line-depth parity suite stays byte-identical.
pub fn back_four_line_depth_v2_enabled() -> bool {
    #[cfg(test)]
    {
        env_flag_enabled(BACK_FOUR_LINE_DEPTH_V2_ENABLE_ENV)
    }
    #[cfg(not(test))]
    {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| gate_default_on(BACK_FOUR_LINE_DEPTH_V2_ENABLE_ENV))
    }
}

/// Whether the **press-to-attackers compaction** is applied to the v2 line centre this process:
/// while in possession or dispossession (NOT while the opponent drives at our goal) the back four
/// presses UP to sit at most [`BACK_FOUR_OPTIMAL_GAP_TO_ATTACKERS_YARDS`] goal-side of the
/// opponent's foremost-attacker line, filling the space rather than leaving a hole. MARL/MAPPO:
/// the optimal distance is refined by the line-depth head via the attacker-compactness reward.
/// Default-ON (kill switch `DD_SOCCER_ENABLE_BACK_FOUR_ATTACKER_PRESS=0`); default-OFF under test.
pub fn back_four_press_to_attackers_enabled() -> bool {
    #[cfg(test)]
    {
        env_flag_enabled(BACK_FOUR_ATTACKER_PRESS_ENABLE_ENV)
    }
    #[cfg(not(test))]
    {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| gate_default_on(BACK_FOUR_ATTACKER_PRESS_ENABLE_ENV))
    }
}

/// Whether the **energy-conservation hold deadband** damps the v2 line target this process: a
/// line-bound defender already within [`BACK_FOUR_LINE_HOLD_DEADBAND_YARDS`] of its legal target
/// holds rather than re-chasing a jittering line (the "sine-wave"). Default-ON (kill switch
/// `DD_SOCCER_ENABLE_BACK_FOUR_LINE_HOLD_DEADBAND=0`); default-OFF under test for parity.
pub fn back_four_line_hold_deadband_enabled() -> bool {
    #[cfg(test)]
    {
        env_flag_enabled(BACK_FOUR_LINE_HOLD_DEADBAND_ENABLE_ENV)
    }
    #[cfg(not(test))]
    {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| gate_default_on(BACK_FOUR_LINE_HOLD_DEADBAND_ENABLE_ENV))
    }
}

/// Whether wingback-first forward priority is applied at the back-four line band this
/// process. Off ⇒ the symmetric (equal-share) correction stands (parity).
pub fn defensive_line_wingback_forward_priority_enabled() -> bool {
    #[cfg(test)]
    {
        env_flag_enabled(DEFENSIVE_LINE_WINGBACK_FORWARD_PRIORITY_ENABLE_ENV)
    }
    #[cfg(not(test))]
    {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        // Promoted to default-ON in production (A/B-validated low-risk behavioral hardening).
        *ENABLED
            .get_or_init(|| gate_default_on(DEFENSIVE_LINE_WINGBACK_FORWARD_PRIORITY_ENABLE_ENV))
    }
}

/// Pure, RNG-free **v2 line-centre depth** (yd from own goal). All arguments are
/// yards-from-own-goal in the defending team's attacking frame.
///
/// - `ball_depth`: the (predicted) ball's distance from our own goal.
/// - `desired_gap`: the trailing gap the line wants behind a deep ball (20..40).
/// - `anchor`: the depth the line holds while the ball is upfield of it (15).
/// - `six`: the keeper's sticky floor (≈6) — the line never drops deeper.
/// - `max_depth`: the high-line cap (halfway + into-opp-half allowance).
///
/// Shape vs. `ball_depth`: parity ramp `six→anchor`, a plateau at `anchor` across
/// `[anchor, anchor+desired_gap]`, then a rising `ball_depth − desired_gap` ramp,
/// then the `max_depth` plateau. Monotonic non-decreasing and continuous.
pub fn back_four_line_target_depth_v2(
    ball_depth: f64,
    desired_gap: f64,
    anchor: f64,
    six: f64,
    max_depth: f64,
) -> f64 {
    let raw = ball_depth - desired_gap;
    // The deepest (closest-to-goal) the CENTRE may sit: hold the anchor while the
    // ball is outside it; track the ball back to parity between the 6 and the
    // anchor; stick at the 6 once the ball is inside it (the keeper owns behind).
    let min_depth = if ball_depth > anchor {
        anchor
    } else if ball_depth >= six {
        ball_depth
    } else {
        six
    };
    // Keep the band valid if a degenerate roster pushes `six`/anchor past the cap.
    let lo = min_depth.min(max_depth);
    raw.clamp(lo, max_depth)
}

/// Pure trailing-gap **floor** (yd behind the ball) for the dynamic band: the in-possession floor
/// (5) when WE control the ball, else the deeper out-of-possession floor (20) — which covers both
/// the opponent controlling AND a loose/contested ball ("dispossession"). The band top is always
/// [`BACK_FOUR_LINE_DESIRED_GAP_MAX_YARDS`] (40). Extracted pure so the possession band is unit-
/// tested without toggling a process-global gate (which would race the parallel suite).
pub fn back_four_desired_gap_min_yards(we_control: bool) -> f64 {
    if we_control {
        BACK_FOUR_LINE_DESIRED_GAP_IN_POSSESSION_MIN_YARDS
    } else {
        BACK_FOUR_LINE_DESIRED_GAP_MIN_YARDS
    }
}

/// Pure **press-to-attackers compaction** of the line-CENTRE depth (yd from own goal). Raises a
/// too-deep `centre_depth` UP toward the foremost-attacker line so the four sit at most
/// `optimal_gap` goal-side of `attacker_depth` — never AHEAD of the attackers (no playing runners
/// onside), never below the incoming `centre_depth` (push up only), never past `max_depth` / below
/// `six`. Monotonic non-decreasing in `attacker_depth`.
pub fn back_four_attacker_pressed_depth(
    centre_depth: f64,
    attacker_depth: f64,
    optimal_gap: f64,
    six: f64,
    max_depth: f64,
) -> f64 {
    let press_floor = (attacker_depth - optimal_gap)
        .min(attacker_depth)
        .clamp(six.min(max_depth), max_depth);
    centre_depth.max(press_floor)
}

/// Pure **hold-deadband** decision (energy conservation). Returns the forward target the defender
/// should actually aim at: `cur_fwd` (HOLD) when it is within `deadband` of `target_fwd` and not
/// illegally ahead of `cap`; otherwise `target_fwd`. `cap = None` ⇒ no offside concern (trap
/// lifted), so a within-deadband target always holds. Holding only ever leaves a defender DEEPER
/// than ideal (it never steps it ahead of the cap), so it cannot play a runner onside.
pub fn back_four_line_hold_target_fwd(
    cur_fwd: f64,
    target_fwd: f64,
    cap: Option<f64>,
    deadband: f64,
) -> f64 {
    if let Some(cap) = cap {
        if cur_fwd > cap + 1e-6 {
            return target_fwd;
        }
    }
    if (target_fwd - cur_fwd).abs() < deadband {
        cur_fwd
    } else {
        target_fwd
    }
}

/// Raw (un-normalized) state the line model is a function of, captured in the
/// defending team's **attacking frame** (`fwd` increases toward the opponent
/// goal; `lat` is signed distance from the pitch spine). Kept as a plain struct
/// so it can be logged as a training row, asserted on in tests, and surfaced by
/// the inspector independently of the network.
#[derive(Clone, Debug, PartialEq)]
pub struct BackFourLineInputs {
    /// Pitch length / width, for normalization (`L`, `W`).
    pub field_length: f64,
    pub field_width: f64,

    // 1. Ball.
    pub ball_fwd_from_own_goal: f64,
    pub ball_lat: f64,
    pub ball_vel_fwd: f64,
    pub ball_vel_lat: f64,
    pub ball_acc_fwd: f64,
    pub ball_acc_lat: f64,

    // 2. Opponents (incl. 4: centre of mass = centroid).
    pub opp_foremost_fwd_from_own_goal: f64,
    pub opp_foremost_four_fwd_from_own_goal: f64,
    pub opp_centroid_fwd_from_own_goal: f64,
    pub opp_centroid_lat: f64,
    pub opp_mean_vel_fwd: f64,
    pub opp_mean_vel_lat: f64,
    pub opp_mean_acc_fwd: f64,
    pub opp_mean_acc_lat: f64,

    // 3. Self (the back four).
    pub line_fwd_from_own_goal: f64,
    pub line_lat: f64,
    pub line_vel_fwd: f64,
    pub line_vel_lat: f64,
    pub line_acc_fwd: f64,
    pub line_acc_lat: f64,

    // 5. Possession.
    pub we_control: bool,
    pub they_control: bool,

    // 6. Own goalkeeper.
    pub gk_fwd_from_own_goal: f64,
    pub gk_lat: f64,

    // 7. Offside-trap state.
    pub offside_in_force: bool,
    pub trap_active_or_imminent: bool,

    // 8. Existing determinants to retain: where the engine's current heuristic
    // (directive line target, ball blend, role bias, press focus, the legal band)
    // already puts the line centre. The model learns *relative to* this so the
    // well-tuned existing line is never discarded — only refined.
    pub heuristic_centre_fwd_from_own_goal: f64,
}

impl BackFourLineInputs {
    /// The normalized network input vector. **This ordering is the single source
    /// of truth** for the feature layout — change it and bump
    /// [`BACK_FOUR_LINE_FEATURE_DIM`] and any persisted head.
    pub fn to_features(&self) -> [f64; BACK_FOUR_LINE_FEATURE_DIM] {
        let l = self.field_length.max(1.0);
        let half_w = (self.field_width * 0.5).max(1.0);
        let fwd = |y: f64| (y / l).clamp(-0.2, 1.2);
        let lat = |x: f64| (x / half_w).clamp(-1.5, 1.5);
        let vel = |v: f64| (v / REF_TOP_SPEED_YPS).clamp(-2.0, 2.0);
        let acc = |a: f64| (a / REF_ACCEL_YPS2).clamp(-2.0, 2.0);
        let b = |flag: bool| if flag { 1.0 } else { 0.0 };
        [
            // 1. Ball.
            fwd(self.ball_fwd_from_own_goal),
            lat(self.ball_lat),
            vel(self.ball_vel_fwd),
            vel(self.ball_vel_lat),
            acc(self.ball_acc_fwd),
            acc(self.ball_acc_lat),
            // 2. Opponents (+ 4: centroid = centre of mass).
            fwd(self.opp_foremost_fwd_from_own_goal),
            fwd(self.opp_foremost_four_fwd_from_own_goal),
            fwd(self.opp_centroid_fwd_from_own_goal),
            lat(self.opp_centroid_lat),
            vel(self.opp_mean_vel_fwd),
            vel(self.opp_mean_vel_lat),
            acc(self.opp_mean_acc_fwd),
            acc(self.opp_mean_acc_lat),
            // 3. Self.
            fwd(self.line_fwd_from_own_goal),
            lat(self.line_lat),
            vel(self.line_vel_fwd),
            vel(self.line_vel_lat),
            acc(self.line_acc_fwd),
            acc(self.line_acc_lat),
            // 5. Possession.
            b(self.we_control),
            b(self.they_control),
            // 6. Own GK.
            fwd(self.gk_fwd_from_own_goal),
            lat(self.gk_lat),
            // 7. Offside-trap state.
            b(self.offside_in_force),
            b(self.trap_active_or_imminent),
            // 8. Existing heuristic determinant (the retained base line).
            fwd(self.heuristic_centre_fwd_from_own_goal),
        ]
    }
}

/// Analytic, deterministic, RNG-free seed policy for the line-centre **gap
/// fraction** in `[0, 1]` (`0` = level with the ball / high line, `1` = the full
/// max-gap deep block). This is the bootstrap target and live fallback the
/// learned [`BackFourLineHead`] is meant to first imitate, then beat.
///
/// The intent it encodes (all in the defending frame): drop **deeper** when the
/// opponent is driving the ball at our goal, when they have it in our half, or
/// when no trap is meaningful; hold a **higher** line when we control, when a
/// trap is on, and the further from our own goal the ball is. Velocity and
/// acceleration enter so the line leads the next ~3 s rather than chasing the
/// current frame.
pub fn analytic_line_centre_gap_fraction(inputs: &BackFourLineInputs) -> f64 {
    let l = inputs.field_length.max(1.0);
    // Base depth: deeper as the ball sits closer to our own goal. `ball_fwd /
    // L` is 0 on our line, 1 on theirs; invert so a deep ball ⇒ a deep line.
    let ball_fwd_frac = (inputs.ball_fwd_from_own_goal / l).clamp(0.0, 1.0);
    let mut gap = 0.30 + 0.45 * (1.0 - ball_fwd_frac);

    // A ball accelerating / moving toward our goal (negative fwd velocity) pulls
    // the line back early; one moving upfield lets it step up.
    let approach = (-inputs.ball_vel_fwd / REF_TOP_SPEED_YPS).clamp(-1.0, 1.0);
    gap += 0.18 * approach;
    let approach_acc = (-inputs.ball_acc_fwd / REF_ACCEL_YPS2).clamp(-1.0, 1.0);
    gap += 0.06 * approach_acc;

    // Possession: holding the ball ⇒ step up (smaller gap); conceding it ⇒ sit.
    if inputs.we_control {
        gap -= 0.20;
    } else if inputs.they_control {
        gap += 0.08;
    }

    // Stay connected to the opponent's foremost attacking four. A large gap
    // here is the classic "sine wave" failure: defenders preserve the flat line
    // while floating too far behind the attackers. When we have it, and also in
    // 50:50 dispossession phases, press up more; when they control, apply a
    // milder squeeze so the line still respects the ball threat.
    let attacker_gap =
        (inputs.opp_foremost_four_fwd_from_own_goal - inputs.line_fwd_from_own_goal).max(0.0);
    let (ideal_attacker_gap, press_weight) = if inputs.we_control {
        (
            BACK_FOUR_FOREMOST_FOUR_ATTACKER_IDEAL_GAP_IN_POSSESSION_YARDS,
            0.22,
        )
    } else if inputs.they_control {
        (
            BACK_FOUR_FOREMOST_FOUR_ATTACKER_IDEAL_GAP_DEFENDING_YARDS,
            0.10,
        )
    } else {
        (
            BACK_FOUR_FOREMOST_FOUR_ATTACKER_IDEAL_GAP_DISPOSSESSION_YARDS,
            0.18,
        )
    };
    let excess_attacker_gap = ((attacker_gap - ideal_attacker_gap) / 24.0).clamp(0.0, 1.0);
    gap -= press_weight * excess_attacker_gap;
    let crowded_attacker_gap =
        ((BACK_FOUR_FOREMOST_FOUR_ATTACKER_MIN_GAP_YARDS - attacker_gap) / 14.0).clamp(0.0, 1.0);
    gap += 0.08 * crowded_attacker_gap;

    // A live, meaningful offside trap is the squeeze tool: hold a high line.
    if inputs.trap_active_or_imminent {
        gap -= 0.16;
    }
    // No offside in force (a restart is being played in behind): there is nothing
    // to trap, so drop off.
    if inputs.offside_in_force {
        gap += 0.12;
    }

    gap.clamp(0.0, 1.0)
}

/// Analytic, deterministic, RNG-free seed for the MIDFIELD line's depth, returned
/// as a fraction in `[0, 1]` of the legal ideal-gap band *ahead of the back four*
/// (`0` = compressed onto the defenders, `1` = pushed up to the maximum stagger).
/// The midfield line is NOT an offside tool — it has no trap term; it pushes up to
/// connect with the press and compresses to protect the block. The caller maps the
/// fraction onto its `MID_AHEAD_OF_DEF_{MIN,MAX}` band.
pub fn analytic_midfield_gap_fraction(inputs: &BackFourLineInputs) -> f64 {
    let l = inputs.field_length.max(1.0);
    // Push the midfield UP as the ball advances upfield; compress it as the ball
    // sits deep in our half (protect the back four).
    let ball_fwd_frac = (inputs.ball_fwd_from_own_goal / l).clamp(0.0, 1.0);
    let mut frac = ball_fwd_frac;

    // Possession: when we have it the midfield steps up to support the attack;
    // when they have it the block tucks in a touch.
    if inputs.we_control {
        frac += 0.20;
    } else if inputs.they_control {
        frac -= 0.10;
    }

    // A ball driving at our goal pulls the whole block — midfield included — back.
    let approach = (-inputs.ball_vel_fwd / REF_TOP_SPEED_YPS).clamp(-1.0, 1.0);
    frac -= 0.15 * approach;

    frac.clamp(0.0, 1.0)
}

/// One reward-weighted RL training row for a line-depth head: the state at a line
/// decision, the gap fraction the line ACTUALLY used (the action), and the reward
/// attributed to that decision over the following window (higher = the depth led to
/// a better territorial / fewer-chances-conceded outcome). Collected during
/// self-play, drained to the learner, and consumed by
/// [`BackFourLineHead::train_reward_weighted`]. Mirrors [`SoccerPassOutcomeSample`]'s
/// role for the pass head, but carries a continuous *action + reward* (not a binary
/// outcome) because the line decision is an action to improve, not an event to
/// predict.
#[derive(Clone, Debug, PartialEq)]
pub struct LineDepthSample {
    /// State features at the line decision.
    pub inputs: BackFourLineInputs,
    /// The gap fraction the line actually used (the action), in `[0, 1]`.
    pub action_gap_fraction: f64,
    /// Reward attributed to the decision over the following window (any scale; only
    /// relative-to-batch matters — the trainer baselines it).
    pub reward: f64,
}

/// An open line-depth decision awaiting its windowed reward. Recorded at the
/// decision tick with the team's territorial advantage then; resolved
/// `LINE_DEPTH_REWARD_WINDOW_TICKS` later by differencing the territorial advantage
/// into a [`LineDepthSample`]'s reward.
#[derive(Clone, Debug, PartialEq)]
pub struct PendingLineDepthDecision {
    pub team: Team,
    pub inputs: BackFourLineInputs,
    pub action_gap_fraction: f64,
    pub decision_territorial: f64,
    pub due_tick: u64,
}

/// How often (ticks) a line-depth RL decision is sampled while a line model is on.
pub const LINE_DEPTH_SAMPLE_INTERVAL_TICKS: u64 = 15;
/// Window (ticks) over which a sampled decision's territorial reward is measured.
pub const LINE_DEPTH_REWARD_WINDOW_TICKS: u64 = 45;
/// Cap on the rolling RL sample buffer (drained by the learner).
pub const LINE_DEPTH_SAMPLE_CAP: usize = 4096;
/// Minimum training steps before a trained head is consumed live; below this the
/// regression net is too raw, so the decision falls back to the analytic seed.
pub const LINE_DEPTH_HEAD_MIN_TRAINING_STEPS: usize = 200;

/// Learned regression head for the line-centre gap fraction: a `FeedForwardNetwork`
/// (`DIM → hidden → 1`, sigmoid) mirroring [`SoccerPassCompletionHead`]. The live
/// net is not serde; like the other learned heads it round-trips through the
/// engine's `SoccerNeuralNetworkSnapshot` Postgres path. Construction, `predict`,
/// and `train` are in place; the live chokepoint reads the analytic seed until a
/// trained head is promoted (see `docs/back-four-line-model.md`).
#[derive(Clone, Debug)]
pub struct BackFourLineHead {
    network: FeedForwardNetwork,
    training_steps: usize,
    last_loss: Option<f64>,
}

impl BackFourLineHead {
    pub fn new(seed: u32) -> Self {
        let mut rng = mulberry32(seed ^ 0x2D9F_4E11);
        let network = FeedForwardNetwork::random(
            &RandomNetworkSpec {
                input_dim: BACK_FOUR_LINE_FEATURE_DIM,
                hidden_layers: vec![BACK_FOUR_LINE_HIDDEN_UNITS],
                output_dim: 1,
                hidden_activation: ActivationName::Tanh,
                // Sigmoid output = a gap fraction in (0, 1).
                output_activation: ActivationName::Sigmoid,
                weight_scale: None,
            },
            &mut rng,
        );
        BackFourLineHead {
            network,
            training_steps: 0,
            last_loss: None,
        }
    }

    /// Predicted gap fraction in `(0, 1)` for a captured state, or `None` on a
    /// malformed (mis-dimensioned / non-finite) input.
    pub fn predict(&self, inputs: &BackFourLineInputs) -> Option<f64> {
        let features = inputs.to_features();
        if features.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let out = self.network.predict(&features[..]);
        out.first().copied().filter(|p| p.is_finite())
    }

    /// One SGD epoch regressing the head toward `(features, target_fraction)`
    /// pairs (supervised bootstrap toward the analytic seed, or an RL target).
    /// Returns the mean step loss.
    pub fn train(&mut self, samples: &[(BackFourLineInputs, f64)], learning_rate: f64) -> f64 {
        let mut total = 0.0;
        let mut applied = 0usize;
        for (inputs, target) in samples {
            let features = inputs.to_features();
            if features.iter().any(|v| !v.is_finite()) || !target.is_finite() {
                continue;
            }
            let target = [target.clamp(0.0, 1.0)];
            let result =
                self.network
                    .train_sample_clipped(&features[..], &target, learning_rate, 4.0);
            if result.applied && result.loss.is_finite() {
                total += result.loss;
                applied += 1;
                self.training_steps += 1;
            }
        }
        let mean = if applied > 0 {
            total / applied as f64
        } else {
            0.0
        };
        self.last_loss = Some(mean);
        mean
    }

    /// One **reward-weighted-regression** epoch over RL samples (offline policy
    /// improvement). Regress the head toward each sample's ACTION (the gap actually
    /// used), weighted by how much its reward beat the batch baseline: above-average
    /// decisions are imitated strongly, below-average ones barely (and never pushed
    /// to the opposite action). So the head concentrates on the depths that produced
    /// good outcomes rather than merely imitating the analytic seed. Returns the mean
    /// step loss.
    pub fn train_reward_weighted(
        &mut self,
        samples: &[LineDepthSample],
        learning_rate: f64,
    ) -> f64 {
        let finite: Vec<&LineDepthSample> = samples
            .iter()
            .filter(|s| s.reward.is_finite() && s.action_gap_fraction.is_finite())
            .collect();
        if finite.is_empty() {
            return 0.0;
        }
        // Baseline = mean reward; advantage = (reward − baseline) / std. Standardizing
        // by the batch spread keeps the weight scale stable across reward magnitudes.
        let n = finite.len() as f64;
        let baseline = finite.iter().map(|s| s.reward).sum::<f64>() / n;
        let std = (finite
            .iter()
            .map(|s| (s.reward - baseline).powi(2))
            .sum::<f64>()
            / n)
            .sqrt()
            .max(1e-3);
        let mut total = 0.0;
        let mut applied = 0usize;
        for s in finite {
            let features = s.inputs.to_features();
            if features.iter().any(|v| !v.is_finite()) {
                continue;
            }
            let advantage = (s.reward - baseline) / std;
            // Exponential weight, clamped: above-baseline ⇒ weight > 1, below ⇒ → 0.
            let weight = advantage.clamp(-4.0, 2.0).exp().min(7.5);
            let target = [s.action_gap_fraction.clamp(0.0, 1.0)];
            let result = self.network.train_sample_clipped(
                &features[..],
                &target,
                learning_rate * weight,
                4.0,
            );
            if result.applied && result.loss.is_finite() {
                total += result.loss;
                applied += 1;
                self.training_steps += 1;
            }
        }
        let mean = if applied > 0 {
            total / applied as f64
        } else {
            0.0
        };
        self.last_loss = Some(mean);
        mean
    }

    pub fn training_steps(&self) -> usize {
        self.training_steps
    }

    pub fn last_loss(&self) -> Option<f64> {
        self.last_loss
    }
}

/// Outcome of training a line-depth head on a drained RL corpus, logged by the
/// learning runner so an operator can see the corpus is loaded and learned from.
/// Mirrors `SoccerPassCompletionTrainingReport`.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LineDepthTrainingReport {
    /// Usable samples trained on.
    pub samples: usize,
    /// SGD steps actually applied across all epochs.
    pub training_steps: usize,
    /// Epochs run over the corpus.
    pub epochs: usize,
    /// Mean (weighted) loss of the final epoch.
    pub final_loss: f64,
}

/// Load-and-train entry for a line-depth head: given a drained RL corpus (in
/// practice the cluster learner's `drain_line_depth_samples`, later resumed from
/// Postgres), build a fresh [`BackFourLineHead`] and run `epochs`
/// reward-weighted-regression passes. `None` if no usable samples (so the runner
/// skips cleanly rather than persist an untrained net). Mirrors
/// `train_soccer_pass_completion_head`.
pub fn train_line_depth_head(
    samples: &[LineDepthSample],
    seed: u32,
    epochs: usize,
    learning_rate: f64,
) -> Option<(BackFourLineHead, LineDepthTrainingReport)> {
    let usable = samples
        .iter()
        .filter(|s| s.reward.is_finite() && s.action_gap_fraction.is_finite())
        .count();
    if usable == 0 || epochs == 0 {
        return None;
    }
    let mut head = BackFourLineHead::new(seed);
    let mut final_loss = 0.0;
    for _ in 0..epochs {
        final_loss = head.train_reward_weighted(samples, learning_rate);
    }
    let report = LineDepthTrainingReport {
        samples: usable,
        training_steps: head.training_steps(),
        epochs,
        final_loss,
    };
    Some((head, report))
}

/// Public wrapper returning ONLY the report (the head type is engine-internal). The
/// learning runner uses this to train the drained line-depth corpus and log that it
/// learns. Mirrors `report_soccer_pass_completion_training`.
pub fn report_line_depth_training(
    samples: &[LineDepthSample],
    seed: u32,
    epochs: usize,
    learning_rate: f64,
) -> Option<LineDepthTrainingReport> {
    train_line_depth_head(samples, seed, epochs, learning_rate).map(|(_, report)| report)
}

impl WorldSnapshot {
    /// Build the [`BackFourLineInputs`] for `team`'s back four. Thin wrapper over
    /// the role-parameterized [`Self::build_line_depth_inputs`] — the same feature
    /// machinery serves any defensive line; the back four is `Defender`.
    pub(crate) fn build_back_four_line_inputs(&self, team: Team) -> Option<BackFourLineInputs> {
        self.build_line_depth_inputs(team, PlayerRole::Defender)
    }

    /// Build the [`BackFourLineInputs`] feature state for `team`'s line of role
    /// `self_role` (the back four = `Defender`, the midfield line = `Midfielder`),
    /// in that team's attacking frame. `None` when the team has no players of that
    /// role or no goalkeeper on the field (a degenerate roster the model should not
    /// opine on). Pure / RNG-free. The struct is line-agnostic: only the `line_*`
    /// (self) and `heuristic_centre_*` fields differ by which line is modeled.
    pub(crate) fn build_line_depth_inputs(
        &self,
        team: Team,
        self_role: PlayerRole,
    ) -> Option<BackFourLineInputs> {
        let attack = team.attack_dir();
        let own_goal_fwd = self.own_goal_y_for(team) * attack;
        let mid_x = self.field_width * 0.5;
        // Sign lateral so it reads the same after a mirror: lateral grows the same
        // way for either team relative to the pitch spine.
        let lat = |x: f64| (x - mid_x) * attack;
        let fwd_from_goal = |y: f64| y * attack - own_goal_fwd;

        // 3. This line (self): centroid kinematics over the team's `self_role`s.
        let mut def_pos = Vec2::zero();
        let mut def_vel = Vec2::zero();
        let mut def_acc = Vec2::zero();
        let mut def_n = 0.0;
        // 2/4. Opponent outfield centroid + foremost (deepest toward our goal).
        let mut opp_pos = Vec2::zero();
        let mut opp_vel = Vec2::zero();
        let mut opp_acc = Vec2::zero();
        let mut opp_n = 0.0;
        // Foremost attacker = the one deepest into our territory = the smallest
        // forward-gap-from-our-goal in the defending frame.
        let mut opp_foremost_fwd_from_goal = f64::INFINITY;
        let mut opp_fwd_from_goal_values = Vec::new();
        for p in &self.players {
            if p.team == team {
                if p.role == self_role {
                    def_pos = def_pos + p.position;
                    def_vel = def_vel + p.velocity;
                    def_acc = def_acc + p.acceleration;
                    def_n += 1.0;
                }
            } else if p.role != PlayerRole::Goalkeeper {
                opp_pos = opp_pos + p.position;
                opp_vel = opp_vel + p.velocity;
                opp_acc = opp_acc + p.acceleration;
                opp_n += 1.0;
                let opp_fwd_from_goal = fwd_from_goal(p.position.y);
                opp_foremost_fwd_from_goal = opp_foremost_fwd_from_goal.min(opp_fwd_from_goal);
                opp_fwd_from_goal_values.push(opp_fwd_from_goal);
            }
        }
        if def_n < 1.0 || opp_n < 1.0 || !opp_foremost_fwd_from_goal.is_finite() {
            return None;
        }
        opp_fwd_from_goal_values.sort_by(|a, b| a.total_cmp(b));
        let foremost_four_count = opp_fwd_from_goal_values.len().min(4);
        let opp_foremost_four_fwd_from_goal = if foremost_four_count > 0 {
            opp_fwd_from_goal_values
                .iter()
                .take(foremost_four_count)
                .sum::<f64>()
                / foremost_four_count as f64
        } else {
            opp_foremost_fwd_from_goal
        };
        let gk = self
            .players
            .iter()
            .find(|p| p.team == team && p.role == PlayerRole::Goalkeeper)?;

        let inv_def = 1.0 / def_n;
        let inv_opp = 1.0 / opp_n;
        let def_pos = def_pos * inv_def;
        let def_vel = def_vel * inv_def;
        let def_acc = def_acc * inv_def;
        let opp_pos = opp_pos * inv_opp;
        let opp_vel = opp_vel * inv_opp;
        let opp_acc = opp_acc * inv_opp;

        let controlled = self.controlled_possession_team();
        let trap_active_or_imminent =
            !self.offside_currently_suspended() && controlled != Some(team) && {
                let ball_from_own_goal = fwd_from_goal(self.ball.position.y).max(0.0);
                ball_from_own_goal > self.field_length / 3.0
            };

        Some(BackFourLineInputs {
            field_length: self.field_length,
            field_width: self.field_width,

            ball_fwd_from_own_goal: fwd_from_goal(self.ball.position.y),
            ball_lat: lat(self.ball.position.x),
            ball_vel_fwd: self.ball.velocity.y * attack,
            ball_vel_lat: self.ball.velocity.x * attack,
            ball_acc_fwd: self.ball.acceleration.y * attack,
            ball_acc_lat: self.ball.acceleration.x * attack,

            opp_foremost_fwd_from_own_goal: opp_foremost_fwd_from_goal,
            opp_foremost_four_fwd_from_own_goal: opp_foremost_four_fwd_from_goal,
            opp_centroid_fwd_from_own_goal: fwd_from_goal(opp_pos.y),
            opp_centroid_lat: lat(opp_pos.x),
            opp_mean_vel_fwd: opp_vel.y * attack,
            opp_mean_vel_lat: opp_vel.x * attack,
            opp_mean_acc_fwd: opp_acc.y * attack,
            opp_mean_acc_lat: opp_acc.x * attack,

            line_fwd_from_own_goal: fwd_from_goal(def_pos.y),
            line_lat: lat(def_pos.x),
            line_vel_fwd: def_vel.y * attack,
            line_vel_lat: def_vel.x * attack,
            line_acc_fwd: def_acc.y * attack,
            line_acc_lat: def_acc.x * attack,

            we_control: controlled == Some(team),
            they_control: controlled == Some(team.other()),

            gk_fwd_from_own_goal: fwd_from_goal(gk.position.y),
            gk_lat: lat(gk.position.x),

            offside_in_force: self.offside_currently_suspended(),
            trap_active_or_imminent,

            // Default the retained-heuristic determinant to the line's current
            // centroid; the live chokepoint overrides it with the precise band
            // centre it has in hand (see `back_four_line_model_centre_fwd`).
            heuristic_centre_fwd_from_own_goal: fwd_from_goal(def_pos.y),
        })
    }

    /// The model's preferred **line-centre forward coordinate** for `me`'s back
    /// four, already clamped to the legal `[deepest_fwd, shallowest_fwd]` band.
    /// `None` when the model is gated off or the inputs are degenerate — the
    /// caller then keeps its heuristic centre (parity).
    ///
    /// `heuristic_centre_fwd` is the engine's existing band centre — the retained
    /// determinant (group 8). The result is a **blend** of that and the model's
    /// own depth, so the well-tuned existing line is refined, never discarded. The
    /// model depth comes from the analytic seed today; flip the seed call to
    /// [`BackFourLineHead::predict`] (and raise [`BACK_FOUR_LINE_MODEL_BLEND`])
    /// once a trained head is promoted.
    pub(crate) fn back_four_line_model_centre_fwd(
        &self,
        me: &PlayerSnapshot,
        predicted_fwd: f64,
        deepest_fwd: f64,
        shallowest_fwd: f64,
        max_gap: f64,
        heuristic_centre_fwd: f64,
    ) -> Option<f64> {
        if !back_four_line_model_enabled() {
            return None;
        }
        let mut inputs = self.build_back_four_line_inputs(me.team)?;
        // Group 8: feed the engine's precise existing band centre to the model so
        // it learns relative to the retained determinant.
        let own_goal_fwd = self.own_goal_y_for(me.team) * me.team.attack_dir();
        inputs.heuristic_centre_fwd_from_own_goal = heuristic_centre_fwd - own_goal_fwd;

        // Consume the trained head once it has learned enough; otherwise the analytic
        // seed. The head is trained on back-four decisions (its action = this same
        // analytic fraction), so it refines the depth the corpus showed worked.
        let gap_fraction = self
            .line_depth_head
            .as_ref()
            .filter(|head| head.training_steps() >= LINE_DEPTH_HEAD_MIN_TRAINING_STEPS)
            .and_then(|head| head.predict(&inputs))
            .unwrap_or_else(|| analytic_line_centre_gap_fraction(&inputs));
        if !gap_fraction.is_finite() || !max_gap.is_finite() || !heuristic_centre_fwd.is_finite() {
            return None;
        }
        let model_centre_fwd = predicted_fwd - gap_fraction * max_gap;
        // Blend retained-heuristic ⟷ model depth (group 8 keeps existing
        // determinants in play); the band clamp keeps the result legal.
        let blend = BACK_FOUR_LINE_MODEL_BLEND.clamp(0.0, 1.0);
        let centre_fwd = heuristic_centre_fwd * (1.0 - blend) + model_centre_fwd * blend;
        Some(centre_fwd.clamp(deepest_fwd.min(shallowest_fwd), shallowest_fwd))
    }

    /// The MIDFIELD line model's preferred depth, as a fraction in `[0, 1]` of the
    /// ideal-gap band ahead of the back four (`0` = compressed onto the defenders,
    /// `1` = maximum stagger). `None` when the midfield model is gated off or the
    /// inputs are degenerate — the caller then keeps its fixed ideal (parity). The
    /// fraction comes from [`analytic_midfield_gap_fraction`] today; swap to a
    /// trained [`BackFourLineHead`] once promoted. The caller maps the fraction to
    /// yards and forms the actual flat line over the 5s consistency window.
    pub(crate) fn midfield_line_model_ideal_gap_fraction(
        &self,
        me: &PlayerSnapshot,
    ) -> Option<f64> {
        if !midfield_line_model_enabled() {
            return None;
        }
        let inputs = self.build_line_depth_inputs(me.team, PlayerRole::Midfielder)?;
        let frac = analytic_midfield_gap_fraction(&inputs);
        frac.is_finite().then(|| frac.clamp(0.0, 1.0))
    }
}

impl SoccerMatch {
    /// Gated per-tick RL sample collection for the line-depth heads (Gap 5), driven
    /// off the already-built per-tick learning `snapshot`. A **no-op** (zero cost,
    /// byte-identical) unless a line-depth model is enabled, so the default learner
    /// is unaffected. When on: resolves decisions whose reward window has elapsed
    /// (reward = the defending team's territorial-advantage change over the window —
    /// higher means the chosen line depth kept the opponent out), and samples a fresh
    /// back-four decision every [`LINE_DEPTH_SAMPLE_INTERVAL_TICKS`]. Drained by the
    /// learner via [`Self::drain_line_depth_samples`].
    pub(crate) fn collect_line_depth_rl_samples(&mut self, snapshot: &WorldSnapshot) {
        if !back_four_line_model_enabled() && !midfield_line_model_enabled() {
            return;
        }
        let tick = self.tick;
        // Resolve decisions whose reward window has elapsed.
        let mut i = 0;
        while i < self.pending_line_depth.len() {
            if self.pending_line_depth[i].due_tick <= tick {
                let decision = self.pending_line_depth.swap_remove(i);
                let now_territorial = territorial_advantage(snapshot, decision.team);
                if now_territorial.is_finite() && decision.decision_territorial.is_finite() {
                    // MARL/MAPPO attacker-compactness shaping: dock the reward by how far the back
                    // four sits behind the optimal gap to the opponent's foremost attackers over the
                    // window, so the learned head prefers depths that keep the four compact with the
                    // attackers (fills the space) rather than leaving a hole. Gated with the press.
                    let mut reward = now_territorial - decision.decision_territorial;
                    if back_four_press_to_attackers_enabled() {
                        if let Some(excess) =
                            snapshot.back_four_attacker_gap_excess_yards(decision.team)
                        {
                            reward -= BACK_FOUR_ATTACKER_COMPACTNESS_REWARD_PER_YARD * excess;
                        }
                    }
                    self.line_depth_samples.push(LineDepthSample {
                        inputs: decision.inputs,
                        action_gap_fraction: decision.action_gap_fraction,
                        reward,
                    });
                    if self.line_depth_samples.len() > LINE_DEPTH_SAMPLE_CAP {
                        let overflow = self.line_depth_samples.len() - LINE_DEPTH_SAMPLE_CAP;
                        self.line_depth_samples.drain(0..overflow);
                    }
                }
            } else {
                i += 1;
            }
        }
        // Sample a fresh decision on cadence. The back four is the offside-setting
        // line and the primary line-depth decision; its features also describe the
        // state the midfield model reads, so one sample per team suffices for now.
        if tick % LINE_DEPTH_SAMPLE_INTERVAL_TICKS == 0 {
            for team in [Team::Home, Team::Away] {
                let Some(inputs) = snapshot.build_line_depth_inputs(team, PlayerRole::Defender)
                else {
                    continue;
                };
                let action = analytic_line_centre_gap_fraction(&inputs);
                let territorial = territorial_advantage(snapshot, team);
                if action.is_finite() && territorial.is_finite() {
                    self.pending_line_depth.push(PendingLineDepthDecision {
                        team,
                        inputs,
                        action_gap_fraction: action,
                        decision_territorial: territorial,
                        due_tick: tick + LINE_DEPTH_REWARD_WINDOW_TICKS,
                    });
                }
            }
        }
    }

    /// Drain the collected line-depth RL samples for the cluster learner (train the
    /// head + persist), mirroring [`Self::drain_pass_outcome_samples`]. `pub` so the
    /// learner binary (a separate crate) can drain it.
    pub fn drain_line_depth_samples(&mut self) -> Vec<LineDepthSample> {
        std::mem::take(&mut self.line_depth_samples)
    }

    /// Install the trained back-four line-depth head for live consumption. The
    /// learner carries + trains it across games and re-installs it each game; wrapped
    /// in `Arc` so every per-tick snapshot shares it cheaply. Consumed in
    /// `back_four_line_model_centre_fwd` once it clears
    /// [`LINE_DEPTH_HEAD_MIN_TRAINING_STEPS`].
    pub fn set_line_depth_head(&mut self, head: BackFourLineHead) {
        self.line_depth_head = Some(std::sync::Arc::new(head));
    }
}

#[cfg(test)]
mod back_four_line_tests {
    use super::*;

    /// A neutral mid-pitch state to perturb from.
    fn baseline_inputs() -> BackFourLineInputs {
        BackFourLineInputs {
            field_length: 120.0,
            field_width: 80.0,
            ball_fwd_from_own_goal: 60.0,
            ball_lat: 0.0,
            ball_vel_fwd: 0.0,
            ball_vel_lat: 0.0,
            ball_acc_fwd: 0.0,
            ball_acc_lat: 0.0,
            opp_foremost_fwd_from_own_goal: 40.0,
            opp_foremost_four_fwd_from_own_goal: 48.0,
            opp_centroid_fwd_from_own_goal: 55.0,
            opp_centroid_lat: 0.0,
            opp_mean_vel_fwd: 0.0,
            opp_mean_vel_lat: 0.0,
            opp_mean_acc_fwd: 0.0,
            opp_mean_acc_lat: 0.0,
            line_fwd_from_own_goal: 35.0,
            line_lat: 0.0,
            line_vel_fwd: 0.0,
            line_vel_lat: 0.0,
            line_acc_fwd: 0.0,
            line_acc_lat: 0.0,
            we_control: false,
            they_control: true,
            gk_fwd_from_own_goal: 6.0,
            gk_lat: 0.0,
            offside_in_force: false,
            trap_active_or_imminent: true,
            heuristic_centre_fwd_from_own_goal: 35.0,
        }
    }

    #[test]
    fn features_are_finite_bounded_and_right_dim() {
        let f = baseline_inputs().to_features();
        assert_eq!(f.len(), BACK_FOUR_LINE_FEATURE_DIM);
        for v in f {
            assert!(v.is_finite());
            assert!(v.abs() <= 2.0 + 1e-9, "feature {v} out of normalized range");
        }
    }

    #[test]
    fn analytic_fraction_is_a_valid_gap() {
        let g = analytic_line_centre_gap_fraction(&baseline_inputs());
        assert!((0.0..=1.0).contains(&g), "gap fraction {g} out of [0,1]");
    }

    #[test]
    fn deep_ball_pushes_line_deeper_than_high_ball() {
        let mut deep = baseline_inputs();
        deep.ball_fwd_from_own_goal = 15.0; // ball near our own goal
        let mut high = baseline_inputs();
        high.ball_fwd_from_own_goal = 105.0; // ball near their goal
        assert!(
            analytic_line_centre_gap_fraction(&deep) > analytic_line_centre_gap_fraction(&high),
            "a deeper ball should yield a deeper (larger-gap) line"
        );
    }

    #[test]
    fn our_possession_lifts_the_line_vs_theirs() {
        let mut ours = baseline_inputs();
        ours.we_control = true;
        ours.they_control = false;
        let mut theirs = baseline_inputs();
        theirs.we_control = false;
        theirs.they_control = true;
        assert!(
            analytic_line_centre_gap_fraction(&ours) < analytic_line_centre_gap_fraction(&theirs),
            "controlling the ball should hold a higher (smaller-gap) line"
        );
    }

    #[test]
    fn ball_driving_at_our_goal_drops_the_line() {
        let mut approaching = baseline_inputs();
        approaching.ball_vel_fwd = -8.0; // moving toward our own goal
        let mut receding = baseline_inputs();
        receding.ball_vel_fwd = 8.0; // moving upfield
        assert!(
            analytic_line_centre_gap_fraction(&approaching)
                > analytic_line_centre_gap_fraction(&receding),
            "a ball accelerating at our goal should drop the line off early"
        );
    }

    #[test]
    fn head_predicts_a_bounded_fraction() {
        let head = BackFourLineHead::new(7);
        let p = head.predict(&baseline_inputs()).expect("finite prediction");
        assert!((0.0..=1.0).contains(&p), "head fraction {p} out of (0,1)");
    }

    #[test]
    fn head_trains_toward_a_target() {
        let mut head = BackFourLineHead::new(11);
        let samples: Vec<(BackFourLineInputs, f64)> =
            std::iter::repeat_with(|| (baseline_inputs(), 0.9))
                .take(64)
                .collect();
        let mut last = f64::INFINITY;
        for _ in 0..40 {
            last = head.train(&samples, 0.05);
        }
        assert!(last.is_finite());
        assert!(head.training_steps() > 0);
    }

    #[test]
    fn midfield_fraction_is_a_valid_gap() {
        let f = analytic_midfield_gap_fraction(&baseline_inputs());
        assert!(
            (0.0..=1.0).contains(&f),
            "midfield fraction {f} out of [0,1]"
        );
    }

    #[test]
    fn midfield_pushes_up_with_a_high_ball_and_compresses_with_a_deep_one() {
        let mut high = baseline_inputs();
        high.ball_fwd_from_own_goal = 100.0; // ball near their goal
        let mut deep = baseline_inputs();
        deep.ball_fwd_from_own_goal = 20.0; // ball near our goal
        assert!(
            analytic_midfield_gap_fraction(&high) > analytic_midfield_gap_fraction(&deep),
            "the midfield should push up with a high ball, compress with a deep one"
        );
    }

    #[test]
    fn midfield_steps_up_when_we_control() {
        let mut ours = baseline_inputs();
        ours.we_control = true;
        ours.they_control = false;
        let mut theirs = baseline_inputs();
        theirs.we_control = false;
        theirs.they_control = true;
        assert!(
            analytic_midfield_gap_fraction(&ours) > analytic_midfield_gap_fraction(&theirs),
            "controlling the ball should push the midfield up (larger gap)"
        );
    }

    #[test]
    fn reward_weighted_training_steers_toward_the_high_reward_action() {
        let mut head = BackFourLineHead::new(23);
        let inputs = baseline_inputs();
        let before = head.predict(&inputs).expect("finite");
        // Same state, two competing actions: a DEEP line (0.85) is rewarded, a HIGH
        // line (0.15) is penalized. RWR should pull the head toward the deep action.
        let samples: Vec<LineDepthSample> = (0..32)
            .flat_map(|_| {
                [
                    LineDepthSample {
                        inputs: inputs.clone(),
                        action_gap_fraction: 0.85,
                        reward: 1.0,
                    },
                    LineDepthSample {
                        inputs: inputs.clone(),
                        action_gap_fraction: 0.15,
                        reward: -1.0,
                    },
                ]
            })
            .collect();
        let mut last = f64::INFINITY;
        for _ in 0..80 {
            last = head.train_reward_weighted(&samples, 0.05);
        }
        let after = head.predict(&inputs).expect("finite");
        assert!(last.is_finite());
        assert!(
            after > before && after > 0.5,
            "RWR should move the head toward the high-reward deep action: {before} -> {after}"
        );
    }

    #[test]
    fn train_line_depth_head_entry_skips_empty_and_trains_valid() {
        assert!(train_line_depth_head(&[], 1, 4, 0.02).is_none());
        let samples: Vec<LineDepthSample> = (0..16)
            .map(|i| LineDepthSample {
                inputs: baseline_inputs(),
                action_gap_fraction: if i % 2 == 0 { 0.8 } else { 0.2 },
                reward: if i % 2 == 0 { 1.0 } else { -1.0 },
            })
            .collect();
        let (_, report) =
            train_line_depth_head(&samples, 1, 4, 0.02).expect("trains a usable corpus");
        assert_eq!(report.samples, 16);
        assert_eq!(report.epochs, 4);
        assert!(report.training_steps > 0 && report.final_loss.is_finite());
    }

    #[test]
    fn v2_depth_holds_the_anchor_while_ball_is_outside_it() {
        // anchor 15, six 6, cap 65 (halfway+5 on a 120 pitch), desired_gap 30.
        let d = |ball: f64| back_four_line_target_depth_v2(ball, 30.0, 15.0, 6.0, 65.0);
        // Ball at 30 / 25 → line ~15 (the plateau), not dragged up to ball-30.
        assert!((d(30.0) - 15.0).abs() < 1e-9, "ball 30 -> {}", d(30.0));
        assert!((d(25.0) - 15.0).abs() < 1e-9, "ball 25 -> {}", d(25.0));
        // Plateau holds across [anchor, anchor+gap] = [15, 45].
        assert!((d(45.0) - 15.0).abs() < 1e-9, "ball 45 -> {}", d(45.0));
    }

    #[test]
    fn v2_depth_tracks_to_parity_between_six_and_anchor_then_sticks_at_six() {
        let d = |ball: f64| back_four_line_target_depth_v2(ball, 30.0, 15.0, 6.0, 65.0);
        // Between the 6 and the 15 the centre tracks the ball to parity (gap 0).
        assert!((d(15.0) - 15.0).abs() < 1e-9, "ball 15 -> {}", d(15.0));
        assert!((d(12.0) - 12.0).abs() < 1e-9, "ball 12 -> {}", d(12.0));
        assert!((d(6.0) - 6.0).abs() < 1e-9, "ball 6 -> {}", d(6.0));
        // Inside the 6 the centre sticks at 6 ⇒ a signed (negative) gap behind ball.
        assert!((d(4.0) - 6.0).abs() < 1e-9, "ball 4 -> {}", d(4.0));
        assert!(
            4.0 - d(4.0) < 0.0,
            "ball 4 should give a negative signed gap"
        );
    }

    #[test]
    fn v2_depth_rises_and_caps_when_the_ball_is_deep_upfield() {
        let d = |ball: f64| back_four_line_target_depth_v2(ball, 30.0, 15.0, 6.0, 65.0);
        // Past anchor+gap (45) the line rises, trailing `desired_gap` behind the ball.
        assert!((d(60.0) - 30.0).abs() < 1e-9, "ball 60 -> {}", d(60.0));
        assert!((d(80.0) - 50.0).abs() < 1e-9, "ball 80 -> {}", d(80.0));
        // ...but never past the high-line cap (halfway+5 = 65).
        assert!((d(110.0) - 65.0).abs() < 1e-9, "ball 110 -> {}", d(110.0));
    }

    #[test]
    fn v2_depth_is_monotonic_non_decreasing_in_ball_depth() {
        let d = |ball: f64| back_four_line_target_depth_v2(ball, 30.0, 15.0, 6.0, 65.0);
        let mut prev = d(0.0);
        let mut ball = 0.0;
        while ball <= 120.0 {
            let cur = d(ball);
            assert!(
                cur + 1e-9 >= prev,
                "non-monotonic at ball={ball}: {prev} -> {cur}"
            );
            prev = cur;
            ball += 0.5;
        }
    }

    #[test]
    fn desired_gap_floor_is_possession_aware() {
        // In possession the line may step right up to 5yd; out of possession (incl. a loose ball)
        // it holds the deeper 20yd floor. Band top is always 40 either way.
        assert!(
            (back_four_desired_gap_min_yards(true)
                - BACK_FOUR_LINE_DESIRED_GAP_IN_POSSESSION_MIN_YARDS)
                .abs()
                < 1e-9
        );
        assert!(
            (back_four_desired_gap_min_yards(false) - BACK_FOUR_LINE_DESIRED_GAP_MIN_YARDS).abs()
                < 1e-9
        );
        assert!(back_four_desired_gap_min_yards(true) < back_four_desired_gap_min_yards(false));
    }

    #[test]
    fn attacker_press_raises_a_deep_centre_but_never_steps_ahead() {
        // anchor/six 6, cap 65, optimal gap 12. Attackers' foremost line at depth 40.
        let pressed = |centre: f64| back_four_attacker_pressed_depth(centre, 40.0, 12.0, 6.0, 65.0);
        // A too-deep centre (15) is raised toward the attackers: 40 - 12 = 28.
        assert!((pressed(15.0) - 28.0).abs() < 1e-9, "deep centre -> {}", pressed(15.0));
        // A centre already higher than the press floor is left alone (push up only).
        assert!((pressed(35.0) - 35.0).abs() < 1e-9, "high centre -> {}", pressed(35.0));
        // Never sits AHEAD of the attacker line even with a tiny optimal gap.
        let ahead = back_four_attacker_pressed_depth(10.0, 40.0, 0.0, 6.0, 65.0);
        assert!(ahead <= 40.0 + 1e-9, "must not step ahead of attackers: {ahead}");
    }

    #[test]
    fn attacker_press_is_monotonic_and_respects_six_and_cap() {
        let pressed = |att: f64| back_four_attacker_pressed_depth(0.0, att, 12.0, 6.0, 65.0);
        // A higher attacker line ⇒ a higher (or equal) pressed centre.
        assert!(pressed(50.0) >= pressed(30.0));
        // Never below the keeper's six even when attackers are very deep / centre is 0.
        assert!(pressed(10.0) >= 6.0 - 1e-9, "floored at six: {}", pressed(10.0));
        // Never past the high-line cap.
        assert!(pressed(200.0) <= 65.0 + 1e-9, "capped: {}", pressed(200.0));
    }

    #[test]
    fn hold_deadband_holds_within_band_and_corrects_outside_or_ahead_of_cap() {
        let db = BACK_FOUR_LINE_HOLD_DEADBAND_YARDS;
        // Within the deadband, no cap ⇒ HOLD current position (ignore the jittering target).
        assert!(
            (back_four_line_hold_target_fwd(30.0, 30.0 + db * 0.5, None, db) - 30.0).abs() < 1e-9
        );
        // Outside the deadband ⇒ move to the target.
        let far = 30.0 + db * 2.0;
        assert!((back_four_line_hold_target_fwd(30.0, far, None, db) - far).abs() < 1e-9);
        // Illegally AHEAD of the cap ⇒ always corrected to the target, deadband notwithstanding.
        let cap = 29.0;
        let target = 28.5;
        assert!(
            (back_four_line_hold_target_fwd(30.0, target, Some(cap), db) - target).abs() < 1e-9,
            "ahead of cap must be corrected"
        );
        // Behind the cap and within the band ⇒ HOLD.
        assert!(
            (back_four_line_hold_target_fwd(27.0, 27.0 + db * 0.5, Some(40.0), db) - 27.0).abs()
                < 1e-9
        );
    }

    #[test]
    fn reward_weighted_training_is_a_noop_on_empty_or_nonfinite() {
        let mut head = BackFourLineHead::new(5);
        assert_eq!(head.train_reward_weighted(&[], 0.05), 0.0);
        let bad = vec![LineDepthSample {
            inputs: baseline_inputs(),
            action_gap_fraction: f64::NAN,
            reward: f64::NAN,
        }];
        assert_eq!(head.train_reward_weighted(&bad, 0.05), 0.0);
        assert_eq!(head.training_steps(), 0);
    }
}
