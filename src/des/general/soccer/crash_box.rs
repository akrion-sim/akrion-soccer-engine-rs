//! **Situational "crash the box" from a flank aerial cross.**
//!
//! When a team is in possession in the opponent's attacking final third *and the ball is
//! on the flanks* — the three outermost lanes of the 12-lane fine pitch grid on either
//! side — the right football is almost always the same: whip an **aerial cross** into the
//! penalty area while **two or three attackers crash the box** to attack the delivery and
//! **head it in**.
//!
//! The engine already owns every mechanical piece of that move:
//!
//! * carrier-side aerial delivery via [`FlankAttackPolicy::PlayDownFlankHighCross`]
//!   (`flank-high-cross` / byline release, see `player.rs`),
//! * a deterministic box-flood that fans 2–4 runners into distinct near-post / far-post /
//!   penalty-spot / cut-back zones ([`WorldSnapshot::crash_the_box_target_for`]),
//! * aerial-duel heading finishes (`first-time-header`).
//!
//! What was missing is that those pieces only fired when the **team brain happened to roll
//! the `CrashTheBox` strategy** — a probabilistic, committed choice — rather than as a
//! *reaction to the geometry*. This module supplies the situational recognizer that
//! activates the cross + box-crash purely from where the ball is, and the MARL/MAPPO
//! reward shaping that reinforces the whole pattern (a headed goal off the cross, plus a
//! small bounded credit for actually being in the box when the cross comes in).
//!
//! ## Gating
//!
//! All behaviour here is behind [`flank_crash_box_enabled`]: a `gate_default_on`
//! production switch (live unless `DD_SOCCER_ENABLE_FLANK_CRASH_BOX` is set to a falsey
//! value) that is **default-OFF under `cfg(test)`**, so the existing suite and current
//! network weights stay byte-identical until the flag is flipped on for a retrain A/B.
//! The geometry helpers below are pure and gate-agnostic; the gate is applied by the
//! [`WorldSnapshot`] / [`SoccerMatch`] call sites.

use super::*;

/// Production kill-switch / test enable env var for the situational flank crash-the-box play.
pub(crate) const FLANK_CRASH_BOX_ENABLE_ENV: &str = "DD_SOCCER_ENABLE_FLANK_CRASH_BOX";

/// Master gate. Default-ON in production (set the env to `0`/`false`/`no`/`off` to kill it);
/// default-OFF under `cfg(test)` so the suite stays byte-identical. Tests re-read the env each
/// call (so an A/B can toggle it within the process); production caches the decision once.
pub(crate) fn flank_crash_box_enabled() -> bool {
    #[cfg(test)]
    {
        std::env::var(FLANK_CRASH_BOX_ENABLE_ENV)
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
        *V.get_or_init(|| gate_default_on(FLANK_CRASH_BOX_ENABLE_ENV))
    }
}

/// The attacking final third measured as a fraction of pitch length from the goal a team
/// attacks. `1/3` ⇒ the opponent's third.
const FINAL_THIRD_FRACTION: f64 = 1.0 / 3.0;

/// True when the ball sits in one of the **three outermost lanes on either touchline** of
/// the 12-lane fine pitch grid (lanes 0–2 or 9–11) — i.e. a genuine flank position from
/// which a cross is on. Generalises to `columns / 4` outer lanes per side so it tracks the
/// configured fine-grid resolution.
pub(crate) fn ball_on_outer_flank_lanes(ball_x: f64, field_width: f64) -> bool {
    if !ball_x.is_finite() || field_width <= 0.0 {
        return false;
    }
    let columns = PITCH_FINE_GRID_COLUMNS.max(1);
    let outer = (columns / 4).max(1); // 3 outermost lanes of a 12-lane grid, per side.
    let frac = (ball_x / field_width).clamp(0.0, 1.0 - f64::EPSILON);
    let lane = (frac * columns as f64).floor() as usize;
    lane < outer || lane >= columns - outer
}

/// True when the ball is within the attacking final third of the goal at `attacked_goal_y`.
pub(crate) fn ball_in_attacking_final_third(
    ball_y: f64,
    attacked_goal_y: f64,
    field_length: f64,
) -> bool {
    if !ball_y.is_finite() || !attacked_goal_y.is_finite() || field_length <= 0.0 {
        return false;
    }
    (ball_y - attacked_goal_y).abs() <= field_length * FINAL_THIRD_FRACTION
}

/// Pure geometric recognizer: ball on the outer flank lanes **and** in the attacking final
/// third. Possession, set-play state and the master gate are checked by the callers.
pub(crate) fn flank_final_third_crash_box_geometry(
    ball: Vec2,
    attacked_goal_y: f64,
    field_width: f64,
    field_length: f64,
) -> bool {
    ball_on_outer_flank_lanes(ball.x, field_width)
        && ball_in_attacking_final_third(ball.y, attacked_goal_y, field_length)
}

/// Terminal MARL/MAPPO bonus for a **headed goal finished from a flank crash-the-box aerial
/// cross**, on top of the generic goal reward. The deliverer of the cross takes an assist
/// share so the credit covers the whole move, not just the finish.
pub(crate) const HEADER_GOAL_FROM_CROSS_BONUS_POINTS: f64 = 22.0;
/// Fraction of the bonus routed to the crosser (assist); the remainder goes to the header.
pub(crate) const HEADER_GOAL_FROM_CROSS_ASSIST_SHARE: f64 = 0.34;

/// Small, bounded shaped reward paid **once per cross** when at least
/// [`CRASH_BOX_ARRIVAL_MIN_RUNNERS`] attackers are already inside the box as the aerial
/// cross is released — the "be there for the delivery" signal. Split across the crashers and
/// budget-bounded so it can never rival the terminal goal credit.
pub(crate) const CRASH_BOX_ARRIVAL_SHAPED_POINTS: f64 = 3.0;
/// Minimum attackers in the box for the shaped arrival reward to pay out.
pub(crate) const CRASH_BOX_ARRIVAL_MIN_RUNNERS: usize = 2;

/// Seconds a released aerial cross keeps its "live cross" window open for crediting a headed
/// finish. A real cross→header resolves within ~2–3 s.
pub(crate) const FLANK_CRASH_BOX_CROSS_WINDOW_SECONDS: f64 = 3.0;

/// A live flank crash-the-box aerial cross, latched when the delivery is released so a headed
/// finish (and the box-crash arrivals) can be credited back to the move. Runtime-only.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FlankCrashBoxCross {
    pub team: Team,
    pub crosser: usize,
    pub released_tick: u64,
}

impl FlankCrashBoxCross {
    /// Whether this cross window is still open at `tick` for the given tick duration.
    pub(crate) fn is_open_at(&self, tick: u64, dt_seconds: f64) -> bool {
        let dt = if dt_seconds.is_finite() && dt_seconds > 0.0 {
            dt_seconds
        } else {
            return false;
        };
        let window_ticks = (FLANK_CRASH_BOX_CROSS_WINDOW_SECONDS / dt).ceil() as u64;
        tick >= self.released_tick && tick - self.released_tick <= window_ticks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: f64 = 80.0; // field width yards
    const L: f64 = 120.0; // field length yards

    #[test]
    fn outer_three_lanes_recognised_on_both_touchlines() {
        // 12-lane grid over 80yd ⇒ each lane ~6.667yd. Outer 3 = x<20 or x>=60.
        assert!(ball_on_outer_flank_lanes(2.0, W)); // lane 0
        assert!(ball_on_outer_flank_lanes(13.0, W)); // lane 1
        assert!(ball_on_outer_flank_lanes(19.0, W)); // lane 2
        assert!(ball_on_outer_flank_lanes(W - 2.0, W)); // lane 11
        assert!(ball_on_outer_flank_lanes(W - 13.0, W)); // lane 10
    }

    #[test]
    fn central_lanes_are_not_a_flank() {
        assert!(!ball_on_outer_flank_lanes(W * 0.5, W)); // lane 6, dead centre
        assert!(!ball_on_outer_flank_lanes(24.0, W)); // lane 3
        assert!(!ball_on_outer_flank_lanes(W - 24.0, W)); // lane 8
    }

    #[test]
    fn final_third_uses_the_attacked_goal() {
        // Home attacks y = L. Final third ⇒ y >= L - L/3 = 80.
        assert!(ball_in_attacking_final_third(110.0, L, L));
        assert!(!ball_in_attacking_final_third(70.0, L, L));
        // Away attacks y = 0. Final third ⇒ y <= L/3 = 40.
        assert!(ball_in_attacking_final_third(20.0, 0.0, L));
        assert!(!ball_in_attacking_final_third(60.0, 0.0, L));
    }

    #[test]
    fn geometry_requires_both_flank_and_final_third() {
        // Wide and high ⇒ on.
        assert!(flank_final_third_crash_box_geometry(
            Vec2::new(6.0, 108.0),
            L,
            W,
            L
        ));
        // Wide but in midfield ⇒ off.
        assert!(!flank_final_third_crash_box_geometry(
            Vec2::new(6.0, 60.0),
            L,
            W,
            L
        ));
        // High but central ⇒ off (that's a shooting position, not a crossing one).
        assert!(!flank_final_third_crash_box_geometry(
            Vec2::new(W * 0.5, 110.0),
            L,
            W,
            L
        ));
    }

    #[test]
    fn cross_window_opens_then_closes() {
        let cross = FlankCrashBoxCross {
            team: Team::Home,
            crosser: 7,
            released_tick: 100,
        };
        let dt = 1.0 / 60.0; // 60Hz ⇒ 3s window ≈ 180 ticks.
        assert!(cross.is_open_at(100, dt));
        assert!(cross.is_open_at(250, dt));
        assert!(!cross.is_open_at(400, dt));
        assert!(!cross.is_open_at(99, dt)); // never before release
    }
}
