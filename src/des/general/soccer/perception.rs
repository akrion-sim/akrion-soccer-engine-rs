//! **Perception gap-fills**: true body-occlusion + a perceived-position estimate.
//!
//! The engine already models partial observability well — `player_can_see_point`
//! gates real decisions on vision range + a field-of-view cone, and
//! `position_confidence_for_observer` / `kalman_perception_position_confidence`
//! produce a per-target *confidence* that fuses a distance/facing measurement with
//! a motion prediction (a 2D Kalman step), with explicit caps for occluded /
//! behind-the-observer targets. Two pieces are missing, and this module supplies
//! them as the self-driving-car analogy suggests:
//!
//! 1. **True body-occlusion** ([`sightline_occlusion_fraction`]). The existing
//!    visibility/confidence model uses an `in_front` boolean as its only occlusion
//!    proxy; it never asks whether *another player physically stands on the
//!    line of sight*. A defender screened by two attackers genuinely cannot see
//!    the man behind them even though that man is within range and dead ahead.
//!    This is the geometric line-of-sight test the AV occupancy/raycast stage does.
//!
//! 2. **A perceived position, not just a confidence** ([`perceived_position`]).
//!    All the confidence machinery still hands decisions the *true* coordinate; a
//!    low-confidence target should be perceived at a slightly *wrong* place. This
//!    turns the scalar confidence into a bounded, deterministic position error so
//!    a player acts on what it (imperfectly) sees — giving the reaction-time /
//!    `decision-refractory` line a physical basis (you mis-locate, then correct).
//!
//! ## Parity
//!
//! Both extensions are gated **off** by default
//! (`DD_SOCCER_ENABLE_OCCLUSION`, `DD_SOCCER_ENABLE_PERCEPTION_NOISE`). When unset,
//! the live call sites bypass these functions entirely, so an unconfigured process
//! is byte-identical. The functions are pure and deterministic (the position error
//! is a seeded hash, not RNG state), so tests and the inspector can read them
//! without enabling the seams.

use super::*;
use crate::des::general::prng::mulberry32;

const OCCLUSION_ENABLE_ENV: &str = "DD_SOCCER_ENABLE_OCCLUSION";
const PERCEPTION_NOISE_ENABLE_ENV: &str = "DD_SOCCER_ENABLE_PERCEPTION_NOISE";

/// Effective body radius (yards) of an intervening player for the line-of-sight
/// test. A sightline passing within this of another player's centre is screened.
pub const OCCLUSION_BODY_RADIUS_YARDS: f64 = 1.1;
/// Minimum gap (yards) the target must be *behind* the blocker before the blocker
/// counts: a body level with or behind the target screens nothing.
pub const OCCLUSION_MIN_DEPTH_YARDS: f64 = 0.5;
/// Sightline occlusion fraction at/above which `player_can_see_point` treats the
/// target as not visible (a fully-screened man).
pub const OCCLUSION_BLOCK_THRESHOLD: f64 = 0.6;

/// Largest position error (yards) injected at zero confidence. The error radius is
/// `(1 - confidence) · this · sqrt(u)` for a uniform `u`, i.e. sampled across the
/// uncertainty disk (not pinned to its rim), so a clearly-seen target
/// (confidence → 1) is perceived essentially where it is.
pub const PERCEPTION_MAX_POSITION_ERROR_YARDS: f64 = 4.0;
/// Largest staleness (seconds of the target's own motion) the belief lags by at
/// zero confidence. Scales with `(1 - confidence)`: an uncertain observer tracks a
/// motion-extrapolated *past* position rather than the live one — the AV
/// "track through the gap with the prediction" behaviour.
pub const PERCEPTION_MAX_LAG_SECONDS: f64 = 0.4;
/// Ticks a perceived mis-location is held before it refreshes. Bucketing the seed
/// by this keeps a player committed to one (wrong) estimate for ~a reaction window
/// instead of dithering a fresh error every tick. ~0.27 s at 15 ticks/s.
pub const PERCEPTION_REFRESH_TICKS: u64 = 4;

fn env_on(name: &str) -> bool {
    std::env::var(name)
        .map(|v| {
            let v = v.trim().to_ascii_lowercase();
            matches!(v.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

/// Whether true body-occlusion is folded into visibility this process.
pub fn occlusion_enabled() -> bool {
    #[cfg(test)]
    {
        env_on(OCCLUSION_ENABLE_ENV)
    }
    #[cfg(not(test))]
    {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| env_on(OCCLUSION_ENABLE_ENV))
    }
}

/// Whether perceived-position error is injected at decision sites this process.
pub fn perception_noise_enabled() -> bool {
    #[cfg(test)]
    {
        env_on(PERCEPTION_NOISE_ENABLE_ENV)
    }
    #[cfg(not(test))]
    {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| env_on(PERCEPTION_NOISE_ENABLE_ENV))
    }
}

/// How screened the sightline `observer → target` is by intervening `blockers`
/// (each a body centre), in `[0, 1]`. `0` = clear line of sight; `1` = a blocker
/// sits dead on the line. A blocker only counts when it lies *between* observer
/// and target (projection strictly inside the segment, and at least
/// [`OCCLUSION_MIN_DEPTH_YARDS`] nearer than the target) and within
/// [`OCCLUSION_BODY_RADIUS_YARDS`] of the line; the largest such screen wins.
///
/// `blockers` should already exclude the observer and the target themselves.
pub fn sightline_occlusion_fraction(
    observer: Vec2,
    target: Vec2,
    blockers: impl IntoIterator<Item = Vec2>,
    body_radius: f64,
) -> f64 {
    let seg = target - observer;
    let seg_len = seg.len();
    if !seg_len.is_finite() || seg_len <= 1e-6 || !(body_radius > 0.0) {
        return 0.0;
    }
    let dir = Vec2::new(seg.x / seg_len, seg.y / seg_len);
    let mut worst = 0.0_f64;
    for b in blockers {
        if !b.x.is_finite() || !b.y.is_finite() {
            continue;
        }
        let rel = b - observer;
        // Distance along the sightline (depth from observer toward target).
        let t = rel.dot(dir);
        if t <= 0.0 || t >= seg_len - OCCLUSION_MIN_DEPTH_YARDS {
            continue; // beside/behind the observer, or level with/behind the target
        }
        // Perpendicular distance from the body centre to the sightline.
        let perp = (rel - dir * t).len();
        if perp >= body_radius {
            continue;
        }
        let screen = 1.0 - perp / body_radius;
        if screen > worst {
            worst = screen;
        }
    }
    worst.clamp(0.0, 1.0)
}

/// Deterministic, bounded position error injected for a target seen at
/// `confidence`, sampled across the uncertainty **disk** (not its rim): the radius
/// is `(1 - confidence) · PERCEPTION_MAX_POSITION_ERROR_YARDS · sqrt(u)` and the
/// angle is `2π·u'`, both from a stable hash of `seed` (e.g. `observer_id`,
/// `target_id`, a coarse time bucket). So within a tick the same pair always
/// mis-locates the same way (no per-call jitter), different pairs/moments differ,
/// and the error is distributed in magnitude rather than always at the maximum.
/// Returns `true_pos` unchanged at full confidence or on non-finite input.
pub fn perceived_position(true_pos: Vec2, confidence: f64, seed: u64) -> Vec2 {
    if !true_pos.x.is_finite() || !true_pos.y.is_finite() {
        return true_pos;
    }
    let conf = if confidence.is_finite() {
        confidence.clamp(0.0, 1.0)
    } else {
        1.0
    };
    let max_radius = (1.0 - conf) * PERCEPTION_MAX_POSITION_ERROR_YARDS;
    if max_radius <= 1e-6 {
        return true_pos;
    }
    // Two stable draws from mulberry32 (the engine's PRNG), used as a pure hash
    // here — seeded once, no retained state. `sqrt(u)` for the radius makes the
    // sample uniform over the disk rather than concentrated on its edge.
    let mut rng = mulberry32((seed ^ (seed >> 32)) as u32 ^ 0x9E37_79B9);
    let angle = rng.next_float() * std::f64::consts::TAU;
    let radius = max_radius * rng.next_float().sqrt();
    Vec2::new(
        true_pos.x + angle.cos() * radius,
        true_pos.y + angle.sin() * radius,
    )
}

/// Like [`perceived_position`] but first lags the belief along the target's own
/// motion: an uncertain observer tracks where the target *was* up to
/// `(1 - confidence) · PERCEPTION_MAX_LAG_SECONDS` ago, then the disk error is
/// added on top. This is the faithful AV behaviour — when a measurement is poor
/// you coast on the last good track — and it makes fast movers harder to mark
/// precisely (the staleness is proportional to their speed). Returns
/// [`perceived_position`] (no lag) at full confidence or on non-finite velocity.
pub fn perceived_position_lagged(
    true_pos: Vec2,
    velocity: Vec2,
    confidence: f64,
    seed: u64,
) -> Vec2 {
    let conf = if confidence.is_finite() {
        confidence.clamp(0.0, 1.0)
    } else {
        1.0
    };
    let lag = (1.0 - conf) * PERCEPTION_MAX_LAG_SECONDS;
    let believed = if lag > 1e-6 && velocity.x.is_finite() && velocity.y.is_finite() {
        Vec2::new(true_pos.x - velocity.x * lag, true_pos.y - velocity.y * lag)
    } else {
        true_pos
    };
    perceived_position(believed, conf, seed)
}

#[cfg(test)]
mod tests {
    use super::*;
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn gates_default_off() {
        let _l = ENV_LOCK.lock().expect("env lock");
        std::env::remove_var(OCCLUSION_ENABLE_ENV);
        std::env::remove_var(PERCEPTION_NOISE_ENABLE_ENV);
        assert!(!occlusion_enabled() && !perception_noise_enabled());
    }

    #[test]
    fn clear_sightline_is_unoccluded() {
        let occ = sightline_occlusion_fraction(
            Vec2::new(0.0, 0.0),
            Vec2::new(0.0, 20.0),
            [Vec2::new(5.0, 10.0)], // 5yd off the line ⇒ no screen
            OCCLUSION_BODY_RADIUS_YARDS,
        );
        assert_eq!(occ, 0.0);
    }

    #[test]
    fn a_body_on_the_line_fully_screens() {
        let occ = sightline_occlusion_fraction(
            Vec2::new(0.0, 0.0),
            Vec2::new(0.0, 20.0),
            [Vec2::new(0.0, 10.0)], // dead on the line, halfway
            OCCLUSION_BODY_RADIUS_YARDS,
        );
        assert!(occ > 0.99, "a body on the sightline screens fully: {occ}");
        assert!(occ >= OCCLUSION_BLOCK_THRESHOLD);
    }

    #[test]
    fn blocker_behind_target_or_observer_does_not_count() {
        let line = (Vec2::new(0.0, 0.0), Vec2::new(0.0, 20.0));
        // Beyond the target.
        assert_eq!(
            sightline_occlusion_fraction(line.0, line.1, [Vec2::new(0.0, 25.0)], OCCLUSION_BODY_RADIUS_YARDS),
            0.0
        );
        // Behind the observer.
        assert_eq!(
            sightline_occlusion_fraction(line.0, line.1, [Vec2::new(0.0, -5.0)], OCCLUSION_BODY_RADIUS_YARDS),
            0.0
        );
    }

    #[test]
    fn partial_screen_scales_with_offset() {
        let near = sightline_occlusion_fraction(
            Vec2::new(0.0, 0.0),
            Vec2::new(0.0, 20.0),
            [Vec2::new(0.3, 10.0)],
            OCCLUSION_BODY_RADIUS_YARDS,
        );
        let far = sightline_occlusion_fraction(
            Vec2::new(0.0, 0.0),
            Vec2::new(0.0, 20.0),
            [Vec2::new(0.9, 10.0)],
            OCCLUSION_BODY_RADIUS_YARDS,
        );
        assert!(near > far && far > 0.0, "closer to the line screens more: {near} vs {far}");
    }

    #[test]
    fn perceived_position_is_exact_at_full_confidence() {
        let p = Vec2::new(30.0, 40.0);
        assert_eq!(perceived_position(p, 1.0, 12345), p);
    }

    #[test]
    fn perceived_error_grows_as_confidence_drops_and_is_bounded() {
        let p = Vec2::new(30.0, 40.0);
        let high = perceived_position(p, 0.9, 7).distance(p);
        let low = perceived_position(p, 0.1, 7).distance(p);
        assert!(low > high, "lower confidence ⇒ larger error: {low} vs {high}");
        assert!(
            low <= PERCEPTION_MAX_POSITION_ERROR_YARDS + 1e-9,
            "error stays bounded: {low}"
        );
    }

    #[test]
    fn perceived_position_is_deterministic_per_seed() {
        let p = Vec2::new(10.0, 10.0);
        assert_eq!(perceived_position(p, 0.3, 99), perceived_position(p, 0.3, 99));
        assert_ne!(
            perceived_position(p, 0.3, 99),
            perceived_position(p, 0.3, 100),
            "different seeds mis-locate differently"
        );
    }

    #[test]
    fn perceived_error_is_disk_distributed_not_a_fixed_ring() {
        // A ring model would give the SAME magnitude for every seed at a fixed
        // confidence; the disk sampling must produce DIFFERENT magnitudes.
        let p = Vec2::new(10.0, 10.0);
        let m1 = perceived_position(p, 0.2, 1).distance(p);
        let m2 = perceived_position(p, 0.2, 2).distance(p);
        let m3 = perceived_position(p, 0.2, 3).distance(p);
        assert!(
            (m1 - m2).abs() > 1e-6 || (m1 - m3).abs() > 1e-6,
            "error magnitude must vary with the seed (disk, not ring): {m1} {m2} {m3}"
        );
        for m in [m1, m2, m3] {
            assert!(m <= 0.8 * PERCEPTION_MAX_POSITION_ERROR_YARDS + 1e-9, "bounded: {m}");
        }
    }

    #[test]
    fn lagged_belief_trails_a_fast_mover_and_is_exact_when_sure() {
        let true_pos = Vec2::new(0.0, 50.0);
        let velocity = Vec2::new(0.0, 20.0); // sprinting up-field (+y)
        // Fully uncertain: belief lags ~v·MAX_LAG (8yd back), dominating the ≤3.2yd
        // disk error, so the perceived y is clearly behind the true y.
        let unsure = perceived_position_lagged(true_pos, velocity, 0.0, 5);
        assert!(
            unsure.y < true_pos.y,
            "an uncertain observer tracks a stale (trailing) position: {} < {}",
            unsure.y,
            true_pos.y
        );
        // Fully confident: no lag and no error ⇒ exact.
        assert_eq!(
            perceived_position_lagged(true_pos, velocity, 1.0, 5),
            true_pos
        );
    }
}
