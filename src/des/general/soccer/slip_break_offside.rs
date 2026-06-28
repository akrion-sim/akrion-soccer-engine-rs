//! Slip-and-break-the-offside-trap: a cooperative (MARL/MAPPO) attacking pattern in which a
//! striker / attacking-midfielder **starts a forward sprint first** from a staging band 5–10 yd
//! onside of the opponent back line, and a teammate (a midfielder or a back-four player) **then**
//! slips a firm 25–35 mph ground ball through the seam between two defenders into the space
//! behind the line — released ideally ~2–3 yd before the runner reaches the line so the runner is
//! still onside at the moment the ball is played, then breaks beyond to run onto it. The slip is
//! usually angled away from the keeper but may go straight upfield toward goal.
//!
//! This module holds the **pure, RNG-free** recognition/timing helpers and the feature gates so
//! they can be unit tested in isolation. The stateful pieces that read the world (finding the
//! seam, the runner's timed-break target, the passer's release bias, and the shared reward) live
//! on [`WorldSnapshot`]/`SoccerWorld` in `world.rs`, and the decision wiring is in `player.rs`.
//!
//! Everything here is gated default-OFF (byte-identical parity suite); a learner enables the
//! decision/movement side with `DD_SOCCER_ENABLE_SLIP_BREAK_OFFSIDE=1` and the shared MAPPO
//! reward with `DD_SOCCER_ENABLE_SLIP_BREAK_OFFSIDE_REWARD=1`, then retrains.

use std::sync::OnceLock;

/// Onside staging band: a slip-break runner sets up 5–10 yd on the onside of the second-last
/// defender line so the sprint into the seam can be timed to cross the line *after* the ball is
/// played. Below the min he is too close (he would cross before the ball is slipped); beyond the
/// max he is too deep to arrive in time.
pub(crate) const SLIP_BREAK_STAGE_MIN_YARDS: f64 = 5.0;
pub(crate) const SLIP_BREAK_STAGE_MAX_YARDS: f64 = 10.0;
/// The runner may have been staged a little deeper and still be a valid candidate while he closes;
/// this is the hard onside-side cutoff beyond which we don't treat him as a slip-break runner.
pub(crate) const SLIP_BREAK_STAGE_MAX_DEPTH_YARDS: f64 = 14.0;

/// Ideal release moment: the ball is slipped when the sprinting runner is ~2–3 yd before the line
/// (still onside). Timing peaks here and tapers to zero at the line and at the staging band.
pub(crate) const SLIP_BREAK_RELEASE_TRIGGER_YARDS: f64 = 2.5;
/// Half-width of the timing window around the trigger (so the window spans 0 → 5 yd before line).
pub(crate) const SLIP_BREAK_RELEASE_WINDOW_YARDS: f64 = 2.5;
/// A forward closing speed (yd/s) at/above which the runner is clearly *sprinting* onto the ball;
/// the release-timing strength scales up to full at this speed.
pub(crate) const SLIP_BREAK_SPRINT_REFERENCE_YPS: f64 = 6.0;

/// Firm ground-pass speed band for the slip: 25–35 mph. Short slips sit near the floor; longer
/// threaded balls are driven harder so they beat the recovering defender to the space.
pub(crate) const SLIP_BREAK_PASS_SPEED_MIN_MPH: f64 = 25.0;
pub(crate) const SLIP_BREAK_PASS_SPEED_MAX_MPH: f64 = 35.0;
/// Distance (yd) at/below which the slip is at the speed floor, and the distance over which it
/// ramps to the ceiling.
const SLIP_BREAK_PASS_SPEED_FLOOR_DISTANCE_YARDS: f64 = 8.0;
const SLIP_BREAK_PASS_SPEED_RAMP_YARDS: f64 = 22.0;

/// Lateral gap (yd) between two adjacent defenders below which the seam is too tight to slip a
/// ground ball through, and the gap at which the seam quality is considered ideal.
pub(crate) const SLIP_BREAK_SEAM_MIN_GAP_YARDS: f64 = 4.0;
pub(crate) const SLIP_BREAK_SEAM_IDEAL_GAP_YARDS: f64 = 9.0;
/// Open space behind the line (yd) below which there is nowhere to run, and the depth at which the
/// behind-space contribution saturates.
pub(crate) const SLIP_BREAK_SEAM_MIN_BEHIND_YARDS: f64 = 4.0;
pub(crate) const SLIP_BREAK_SEAM_IDEAL_BEHIND_YARDS: f64 = 15.0;
/// How far in behind the line the seam target point sits (the ball is slipped into space, ahead of
/// the breaking runner) and how far to nudge it laterally away from the keeper.
pub(crate) const SLIP_BREAK_BEHIND_TARGET_YARDS: f64 = 7.0;
pub(crate) const SLIP_BREAK_KEEPER_AVOID_NUDGE_YARDS: f64 = 4.0;
/// How close (yd, in the line-normal direction) an opponent outfielder must be to the second-last
/// defender line to count as part of the line whose seams we slip through (the back four / whoever
/// is holding the trap).
pub(crate) const SLIP_BREAK_LINE_BAND_YARDS: f64 = 12.0;
/// Shared (passer + runner) reward in points for executing a clean slip-break: a firm forward
/// ground ball slipped through a two-defender seam to an onside runner who timed his break, scaled
/// by [`slip_break_opportunity_quality`]. Sized as a modest dense shaping term (well under a shot's
/// reward), split across the two cooperating agents by the involved-team-reward path.
pub(crate) const SLIP_BREAK_REWARD_POINTS: f64 = 4.0;

fn env_truthy(name: &str) -> bool {
    std::env::var(name)
        .map(|raw| {
            let raw = raw.trim();
            raw == "1" || raw.eq_ignore_ascii_case("true")
        })
        .unwrap_or(false)
}

/// Whether the slip-and-break-the-offside-trap **decision/movement** pattern is active this
/// process: the runner's timed-break staging target into a defender seam, and the carrier's bias
/// to slip a firm forward ground ball to that breaking runner. Default-OFF ⇒ byte-identical; set
/// `DD_SOCCER_ENABLE_SLIP_BREAK_OFFSIDE=1`. Under test the env is re-read each call so individual
/// tests can toggle it; in production it is read once.
pub(crate) fn dd_soccer_enable_slip_break_offside() -> bool {
    #[cfg(test)]
    {
        env_truthy("DD_SOCCER_ENABLE_SLIP_BREAK_OFFSIDE")
    }
    #[cfg(not(test))]
    {
        static V: OnceLock<bool> = OnceLock::new();
        *V.get_or_init(|| env_truthy("DD_SOCCER_ENABLE_SLIP_BREAK_OFFSIDE"))
    }
}

/// Whether the shared (centralized MAPPO) **reward** for executing the slip-break is active: when
/// a firm forward ground ball is slipped through a two-defender seam to an onside runner who timed
/// his break, the passer and the runner share the credit. Default-OFF ⇒ byte-identical; set
/// `DD_SOCCER_ENABLE_SLIP_BREAK_OFFSIDE_REWARD=1`.
pub(crate) fn dd_soccer_enable_slip_break_offside_reward() -> bool {
    #[cfg(test)]
    {
        env_truthy("DD_SOCCER_ENABLE_SLIP_BREAK_OFFSIDE_REWARD")
    }
    #[cfg(not(test))]
    {
        static V: OnceLock<bool> = OnceLock::new();
        *V.get_or_init(|| env_truthy("DD_SOCCER_ENABLE_SLIP_BREAK_OFFSIDE_REWARD"))
    }
}

/// Firm ground-pass speed (mph) for a slip of `distance_yards`, clamped to the 25–35 mph band.
/// Short slips sit at the floor; the speed ramps to the ceiling for longer threaded balls so the
/// pass beats the recovering defender into the space. Pure.
pub(crate) fn slip_break_pass_speed_mph(distance_yards: f64) -> f64 {
    let over = (distance_yards - SLIP_BREAK_PASS_SPEED_FLOOR_DISTANCE_YARDS)
        .clamp(0.0, SLIP_BREAK_PASS_SPEED_RAMP_YARDS);
    let frac = over / SLIP_BREAK_PASS_SPEED_RAMP_YARDS;
    SLIP_BREAK_PASS_SPEED_MIN_MPH
        + frac * (SLIP_BREAK_PASS_SPEED_MAX_MPH - SLIP_BREAK_PASS_SPEED_MIN_MPH)
}

/// Quality `[0,1]` of a defender seam to slip a ground ball through: the lateral gap between the
/// two defenders (must clear [`SLIP_BREAK_SEAM_MIN_GAP_YARDS`], ideal at the ideal gap) blended
/// with how much open space lies behind the line to run into. Pure.
pub(crate) fn defender_seam_quality(gap_yards: f64, behind_space_yards: f64) -> f64 {
    let gap_fit = ((gap_yards - SLIP_BREAK_SEAM_MIN_GAP_YARDS)
        / (SLIP_BREAK_SEAM_IDEAL_GAP_YARDS - SLIP_BREAK_SEAM_MIN_GAP_YARDS))
        .clamp(0.0, 1.0);
    let behind_fit = ((behind_space_yards - SLIP_BREAK_SEAM_MIN_BEHIND_YARDS)
        / (SLIP_BREAK_SEAM_IDEAL_BEHIND_YARDS - SLIP_BREAK_SEAM_MIN_BEHIND_YARDS))
        .clamp(0.0, 1.0);
    gap_fit * behind_fit
}

/// Release-timing strength `[0,1]` for slipping the ball NOW, given how many yards the runner is
/// from the line (positive = still onside / short of the line), whether he is onside, and his
/// forward closing speed (yd/s). Peaks at the [`SLIP_BREAK_RELEASE_TRIGGER_YARDS`] trigger and
/// tapers to zero at the line and at the far edge of the window; zero if he is offside or not
/// genuinely closing. This is what makes the pass come *after* the run has started. Pure.
pub(crate) fn slip_break_release_timing(
    runner_yards_to_line: f64,
    runner_onside: bool,
    forward_closing_yps: f64,
) -> f64 {
    if !runner_onside || runner_yards_to_line <= 0.0 {
        return 0.0;
    }
    let window = 1.0
        - (runner_yards_to_line - SLIP_BREAK_RELEASE_TRIGGER_YARDS).abs()
            / SLIP_BREAK_RELEASE_WINDOW_YARDS;
    let window = window.clamp(0.0, 1.0);
    let closing = (forward_closing_yps / SLIP_BREAK_SPRINT_REFERENCE_YPS).clamp(0.0, 1.0);
    window * closing
}

/// Whether a runner at `runner_yards_to_line` onside of the line is in the staging band to be
/// treated as a slip-break candidate (set up 5–10 yd onside, or a touch deeper while closing, and
/// not yet across the line). Pure.
pub(crate) fn slip_break_runner_in_staging_band(runner_yards_to_line: f64) -> bool {
    runner_yards_to_line > 0.0 && runner_yards_to_line <= SLIP_BREAK_STAGE_MAX_DEPTH_YARDS
}

/// Combined opportunity quality `[0,1]` of a slip-break for ranking/reward: a real seam, a
/// well-timed break, and a receiver with some space. Monotonic and bounded. Pure.
pub(crate) fn slip_break_opportunity_quality(
    seam_quality: f64,
    release_timing: f64,
    runner_openness: f64,
) -> f64 {
    let seam = seam_quality.clamp(0.0, 1.0);
    let timing = release_timing.clamp(0.0, 1.0);
    let openness = runner_openness.clamp(0.0, 1.0);
    seam * (0.35 + 0.65 * timing) * (0.6 + 0.4 * openness)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pass_speed_stays_in_firm_band_and_rises_with_distance() {
        for &d in &[0.0, 5.0, 8.0, 15.0, 30.0, 60.0] {
            let mph = slip_break_pass_speed_mph(d);
            assert!(
                (SLIP_BREAK_PASS_SPEED_MIN_MPH..=SLIP_BREAK_PASS_SPEED_MAX_MPH).contains(&mph),
                "distance {d} produced out-of-band speed {mph}"
            );
        }
        // Short slip at the floor, long thread at the ceiling, monotonic in between.
        assert!((slip_break_pass_speed_mph(8.0) - SLIP_BREAK_PASS_SPEED_MIN_MPH).abs() < 1e-9);
        assert!((slip_break_pass_speed_mph(30.0) - SLIP_BREAK_PASS_SPEED_MAX_MPH).abs() < 1e-9);
        assert!(slip_break_pass_speed_mph(12.0) < slip_break_pass_speed_mph(20.0));
    }

    #[test]
    fn seam_quality_needs_a_real_gap_and_space_behind() {
        // Too tight a gap, or no space behind, is not a seam.
        assert_eq!(defender_seam_quality(2.0, 20.0), 0.0);
        assert_eq!(defender_seam_quality(9.0, 2.0), 0.0);
        // A clear gap with room behind is a strong seam, and widening the gap only helps.
        let tight = defender_seam_quality(5.0, 20.0);
        let ideal = defender_seam_quality(9.0, 20.0);
        assert!(tight > 0.0 && ideal > tight);
        assert!((0.0..=1.0).contains(&ideal));
        assert!((defender_seam_quality(12.0, 20.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn release_timing_peaks_just_before_the_line_and_needs_an_onside_sprint() {
        let sprint = SLIP_BREAK_SPRINT_REFERENCE_YPS;
        // Offside (already across the line) or standing still ⇒ no release.
        assert_eq!(slip_break_release_timing(-1.0, true, sprint), 0.0);
        assert_eq!(slip_break_release_timing(2.5, false, sprint), 0.0);
        assert_eq!(slip_break_release_timing(2.5, true, 0.0), 0.0);
        // Peak at the trigger; weaker when staged deep or right at the line.
        let peak = slip_break_release_timing(SLIP_BREAK_RELEASE_TRIGGER_YARDS, true, sprint);
        assert!((peak - 1.0).abs() < 1e-9);
        assert!(slip_break_release_timing(5.0, true, sprint) < peak);
        assert!(slip_break_release_timing(0.5, true, sprint) < peak);
        // Half a sprint ⇒ roughly half strength at the trigger.
        let half = slip_break_release_timing(
            SLIP_BREAK_RELEASE_TRIGGER_YARDS,
            true,
            SLIP_BREAK_SPRINT_REFERENCE_YPS * 0.5,
        );
        assert!((half - 0.5).abs() < 1e-6);
    }

    #[test]
    fn staging_band_excludes_offside_and_too_deep() {
        assert!(!slip_break_runner_in_staging_band(-0.5));
        assert!(slip_break_runner_in_staging_band(7.0));
        assert!(!slip_break_runner_in_staging_band(20.0));
    }

    #[test]
    fn opportunity_quality_is_bounded_and_monotonic() {
        for &s in &[0.0, 0.5, 1.0] {
            for &t in &[0.0, 0.5, 1.0] {
                for &o in &[0.0, 0.5, 1.0] {
                    let q = slip_break_opportunity_quality(s, t, o);
                    assert!((0.0..=1.0).contains(&q), "s={s} t={t} o={o} q={q}");
                }
            }
        }
        // No seam ⇒ no opportunity, whatever the timing.
        assert_eq!(slip_break_opportunity_quality(0.0, 1.0, 1.0), 0.0);
        // Better timing and more openness only help.
        assert!(
            slip_break_opportunity_quality(0.8, 0.4, 0.5)
                < slip_break_opportunity_quality(0.8, 0.9, 0.5)
        );
        assert!(
            slip_break_opportunity_quality(0.8, 0.6, 0.2)
                < slip_break_opportunity_quality(0.8, 0.6, 0.9)
        );
    }
}
