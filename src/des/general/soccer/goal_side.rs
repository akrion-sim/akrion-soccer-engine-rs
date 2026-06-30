//! **Defensive "true goal-side" recovery** — get *on the line between the ball and our goal*,
//! not merely deeper than the ball on the y-axis.
//!
//! When a team loses the ball (an opponent controls it, or it is loose), every outfielder
//! **except the strikers** should recover into the protective channel that runs from the ball
//! to the centre of our own goal. The pre-existing shape machinery only guaranteed *depth* —
//! [`SoccerMatch::goal_side_defensive_target_for_player`] clamped a player's distance-from-own-goal
//! to sit behind the ball, but left their lateral `x` untouched. A full-back stranded on the
//! far touchline therefore counted as "goal-side" simply by being a yard deeper than the ball,
//! even though they were nowhere near the ball→goal line and screened nothing.
//!
//! This module supplies the corrected, purely geometric notion of goal-side — **how far a
//! defender sits along, and how close they sit to, the ball→own-goal line** — and the three
//! integrations that act on it:
//!
//! * **MARL/MAPPO reward** ([`SoccerMatch::update_defensive_goal_side_rewards`]): a small,
//!   budget-bounded per-tick shaped reward for being genuinely goal-side, and a *mild* penalty
//!   for being caught upfield of the ball (failing to recover). Team-shared downstream by the
//!   cooperative MAPPO reward machinery like every other per-player event.
//! * **POMDP observation**: [`SoccerPomdpObservation::defensive_goal_side_quality`] exposes the
//!   signed quality to every decision, and its recovery urgency folds into the off-ball
//!   `defensive_urgency` the policy already consumes.
//! * **MPC / positioning**: [`SoccerMatch::goal_side_defensive_target_for_player`] additionally
//!   shades the off-ball target laterally onto the ball→goal line (a bounded *bias*, not a snap,
//!   so back-line width is preserved by the other shape forces).
//!
//! ## Gating
//!
//! All behaviour is behind [`defensive_goal_side_enabled`]: a `gate_default_on` production
//! switch (live unless `DD_SOCCER_ENABLE_DEFENSIVE_GOAL_SIDE` is set to a falsey value) that is
//! **default-OFF under `cfg(test)`**, so the parity suite and current network weights stay
//! byte-identical until it is flipped on for a retrain A/B. The geometry below is pure and
//! gate-agnostic; the gate is applied at the call sites.

use super::*;

/// Production kill-switch / test enable env var for the true-goal-side defensive recovery.
pub(crate) const DEFENSIVE_GOAL_SIDE_ENABLE_ENV: &str = "DD_SOCCER_ENABLE_DEFENSIVE_GOAL_SIDE";

/// Master gate. Default-ON in production (set the env to `0`/`false`/`no`/`off` to kill it);
/// default-OFF under `cfg(test)` so the suite stays byte-identical. Tests re-read the env each
/// call (so an A/B can toggle it within the process); production caches the decision once.
pub(crate) fn defensive_goal_side_enabled() -> bool {
    #[cfg(test)]
    {
        std::env::var(DEFENSIVE_GOAL_SIDE_ENABLE_ENV)
            .map(|raw| {
                matches!(
                    raw.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false)
    }
    #[cfg(not(test))]
    {
        use std::sync::OnceLock;
        static V: OnceLock<bool> = OnceLock::new();
        *V.get_or_init(|| gate_default_on(DEFENSIVE_GOAL_SIDE_ENABLE_ENV))
    }
}

/// Half-width (yd) of the protective channel either side of the ball→goal line. A defender on
/// the line scores full lateral credit; credit falls linearly to zero at this distance off it.
/// ~14 yd ⇒ a ~28 yd-wide cone, so genuinely shading toward the line counts, but a player marooned
/// on the far flank does not.
pub(crate) const GOAL_SIDE_CHANNEL_HALF_WIDTH_YARDS: f64 = 14.0;

/// Distance (yd) goal-side of the ball at which the depth component of goal-side quality
/// saturates to 1.0. Being a few yards behind the ball is already meaningfully goal-side;
/// there is no extra credit for sitting much deeper than this.
pub(crate) const GOAL_SIDE_DEPTH_SATURATION_YARDS: f64 = 12.0;

/// Learning reward at full positive goal-side quality (`q == +1`, on the line and well goal-side
/// of the threat). Matches the magnitude of the legacy y-axis goal-side reward's best case
/// (`+0.24`) so flipping the gate only changes the *definition* of goal-side (true line vs
/// y-axis), not the weight it carries in the per-player defensive reward.
pub(crate) const GOAL_SIDE_REWARD_BEST_POINTS: f64 = 0.24;

/// Penalty at the worst failure (`q == -1`, caught fully upfield of the threat). Matches the
/// legacy worst case (`-0.42`) — being ball-side is the same magnitude of mistake under either
/// definition; only what counts as "goal-side" changes.
pub(crate) const GOAL_SIDE_FAIL_WORST_POINTS: f64 = 0.42;

/// Weight with which the goal-side recovery urgency lifts the off-ball `defensive_urgency` the
/// policy consumes. Bounded so it nudges, never dominates, the existing defensive urgency.
pub(crate) const GOAL_SIDE_URGENCY_WEIGHT: f64 = 0.30;

/// Fraction of the lateral gap to the ball→goal line that the MPC/positioning target is shaded
/// across in one tick — a *bias*, not a snap, so the back-line keeps its width under the other
/// shape forces. 0 ⇒ pure depth guard (legacy behaviour); 1 ⇒ collapse onto the line.
pub(crate) const GOAL_SIDE_LATERAL_PULL_FRACTION: f64 = 0.35;

/// **Signed goal-side quality** of `player` against the ball→own-goal line, in `[-1, 1]`.
///
/// * `+1` — directly on the line and at/beyond the depth-saturation distance goal-side of the
///   ball (perfectly screening).
/// * `0`  — level with the ball, or off the protective channel entirely.
/// * `-ve` — caught *upfield* of the ball (ball-side): the failure to recover, worsening with how
///   far upfield the player is (saturating at the same distance).
///
/// `own_goal` is the centre of the team's own goal. The positive branch multiplies a depth term
/// by a lateral "on-the-line" term so credit demands *both* goal-side depth and channel
/// closeness; the negative branch tracks only how far upfield the player is, because being
/// ball-side is a failure regardless of lateral position. Pure and deterministic; non-finite
/// inputs collapse to `0.0`.
pub(crate) fn goal_side_quality(
    player: Vec2,
    ball: Vec2,
    own_goal: Vec2,
    channel_half_width_yards: f64,
    depth_saturation_yards: f64,
) -> f64 {
    let axis = own_goal - ball;
    let len = axis.len();
    if !len.is_finite() || len < 1e-3 {
        return 0.0;
    }
    let unit = axis / len; // unit vector pointing from the ball toward our own goal
    let rel = player - ball;
    let along = rel.dot(unit); // > 0 ⇒ goal-side of the ball, < 0 ⇒ caught upfield
    let perp = (rel - unit * along).len(); // lateral distance from the ball→goal line
    if !along.is_finite() || !perp.is_finite() {
        return 0.0;
    }
    let half = channel_half_width_yards.max(1e-3);
    let sat = depth_saturation_yards.max(1e-3);
    let lane = (1.0 - perp / half).clamp(0.0, 1.0); // 1 on the line → 0 at the channel edge
    if along >= 0.0 {
        let depth = (along / sat).clamp(0.0, 1.0);
        (depth * lane).clamp(0.0, 1.0)
    } else {
        // Ball-side: failed to get goal-side. Penalty grows with how far upfield, not lateral.
        (-((-along) / sat).clamp(0.0, 1.0)).clamp(-1.0, 0.0)
    }
}

/// Per-tick shaped reward (points) for a defending outfielder's goal-side quality `q ∈ [-1, 1]`:
/// the positive reward scaled by `q` when goal-side, the *milder* failure penalty scaled by `q`
/// when caught upfield. Caller applies the dense-shaping budget.
pub(crate) fn goal_side_shaping_points(quality: f64) -> f64 {
    if !quality.is_finite() {
        return 0.0;
    }
    if quality >= 0.0 {
        GOAL_SIDE_REWARD_POINTS * quality
    } else {
        GOAL_SIDE_FAIL_PENALTY_POINTS * quality
    }
}

/// Off-ball defensive urgency in `[0, 1]` implied by a goal-side quality `q ∈ [-1, 1]`: zero when
/// perfectly goal-side, rising as the player drifts off the line / upfield. Maps `q = 1 → 0`,
/// `q = 0 → 0.5`, `q = -1 → 1`.
pub(crate) fn goal_side_recovery_urgency(quality: f64) -> f64 {
    if !quality.is_finite() {
        return 0.0;
    }
    ((1.0 - quality) * 0.5).clamp(0.0, 1.0)
}

/// `x`-coordinate where the ball→own-goal line crosses the horizontal at `target_y`, for the MPC
/// lateral pull. Returns `None` when the ball and goal share a `y` (line parallel to the pull
/// axis — no well-defined crossing) or on non-finite input. The parameter is clamped to the
/// ball→goal segment so the pull never aims beyond the ball or behind the goal.
pub(crate) fn ball_goal_line_x_at_y(ball: Vec2, own_goal: Vec2, target_y: f64) -> Option<f64> {
    let dy = own_goal.y - ball.y;
    if !dy.is_finite() || dy.abs() < 1e-3 || !target_y.is_finite() {
        return None;
    }
    let s = ((target_y - ball.y) / dy).clamp(0.0, 1.0);
    let x = ball.x + s * (own_goal.x - ball.x);
    if x.is_finite() {
        Some(x)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HALF: f64 = GOAL_SIDE_CHANNEL_HALF_WIDTH_YARDS;
    const SAT: f64 = GOAL_SIDE_DEPTH_SATURATION_YARDS;

    // Home defends the y = 0 goal; ball is upfield at y = 50. Own goal centre (40, 0).
    fn goal() -> Vec2 {
        Vec2::new(40.0, 0.0)
    }
    fn ball() -> Vec2 {
        Vec2::new(40.0, 50.0)
    }

    #[test]
    fn on_the_line_and_goal_side_scores_high() {
        // Directly on the central line, well goal-side of the ball.
        let q = goal_side_quality(Vec2::new(40.0, 30.0), ball(), goal(), HALF, SAT);
        assert!(q > 0.95, "on-line deep defender ≈ max goal-side, got {q}");
    }

    #[test]
    fn deeper_in_y_but_off_the_line_scores_far_lower_than_on_the_line() {
        // The exact case the design fixes: a full-back deeper than the ball (y = 20, goal-side
        // in depth) but stranded wide (x = 8) is NOT really goal-side.
        let wide = goal_side_quality(Vec2::new(8.0, 20.0), ball(), goal(), HALF, SAT);
        let on_line = goal_side_quality(Vec2::new(40.0, 20.0), ball(), goal(), HALF, SAT);
        assert!(
            wide < on_line - 0.3,
            "wide-but-deep ({wide}) must score well below on-the-line ({on_line})"
        );
    }

    #[test]
    fn caught_upfield_is_a_mild_negative() {
        // Upfield of the ball (y = 70 > 50) on the line ⇒ failed to recover ⇒ negative.
        let q = goal_side_quality(Vec2::new(40.0, 70.0), ball(), goal(), HALF, SAT);
        assert!(q < 0.0, "ball-side defender failed to get goalside, got {q}");
        assert!(q >= -1.0);
    }

    #[test]
    fn reward_is_positive_goalside_mild_penalty_upfield() {
        assert!(goal_side_shaping_points(1.0) > 0.0);
        assert!(goal_side_shaping_points(-1.0) < 0.0);
        // The failure penalty is the milder of the two extremes.
        assert!(goal_side_shaping_points(-1.0).abs() < goal_side_shaping_points(1.0).abs());
        assert_eq!(goal_side_shaping_points(f64::NAN), 0.0);
    }

    #[test]
    fn urgency_falls_to_zero_when_perfectly_goalside() {
        assert!(goal_side_recovery_urgency(1.0) < 1e-9);
        assert!((goal_side_recovery_urgency(0.0) - 0.5).abs() < 1e-9);
        assert!((goal_side_recovery_urgency(-1.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn line_crossing_x_is_between_ball_and_goal() {
        // Ball (60, 50) → goal (40, 0): at y = 25 (midway) x should be 50.
        let x = ball_goal_line_x_at_y(Vec2::new(60.0, 50.0), Vec2::new(40.0, 0.0), 25.0).unwrap();
        assert!((x - 50.0).abs() < 1e-6, "got {x}");
        // Parallel (ball and goal share y) ⇒ no crossing.
        assert!(ball_goal_line_x_at_y(Vec2::new(60.0, 10.0), Vec2::new(40.0, 10.0), 5.0).is_none());
    }

    #[test]
    fn degenerate_inputs_are_neutral() {
        // Ball sitting in the goal mouth ⇒ no axis ⇒ neutral.
        assert_eq!(goal_side_quality(Vec2::new(40.0, 5.0), goal(), goal(), HALF, SAT), 0.0);
    }
}
