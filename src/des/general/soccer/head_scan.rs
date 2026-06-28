//! **Head-scanning**: a settled ball-carrier turns its *head and shoulders*
//! (independently of its body) to find more pass options over time — the elite
//! "check your shoulder before you receive" behaviour — reaching effectively
//! ~360° of awareness given a second or two on the ball.
//!
//! ## Why this exists
//!
//! The base perception model gates pass candidacy on a fixed forward cone:
//! [`player_can_see_point_with_occlusion`](crate::des::general::soccer) hard-hides
//! any teammate outside the ball-holder's ~180° arc, and
//! `position_confidence_for_observer` collapses everything past the shoulders to a
//! flat low-confidence guess. That is too strict: a player on the ball can *glance*
//! to the side and swivel to the rear — it just takes **time** (the head turns
//! faster than the body, but a full shoulder-check to the blind side is ~1–2 s).
//!
//! This module turns "time on the ball" into a widening **scan arc**. As the
//! carrier's accumulated scan time grows, the bearing it has plausibly swept moves
//! out from the cone edge toward directly behind, so teammates in ever-wider arcs
//! become *recognised* pass options — first at the shoulder, later (after the
//! ~1–2 s it takes to check the blind side) behind. A freshly-received ball sees
//! only the forward cone; a carrier who has had a beat to look around finds more.
//!
//! ## Two effects, both bearing- and time-dependent
//!
//! * [`scan_coverage`] — in `[0, 1]`, how thoroughly a bearing off the body's
//!   forward facing has been swept. `1` inside the instantaneous cone; beyond it,
//!   it rises from `0` to `1` as the swept *reach* (`scan_seconds · sweep_rate`)
//!   passes that bearing. The visibility seam treats a bearing as seen once
//!   coverage clears [`HEAD_SCAN_VISIBLE_COVERAGE`].
//! * [`scanned_confidence`] — a bearing the carrier *has* swept is perceived more
//!   confidently than the flat blind-side floor, but still **below** a clean
//!   shoulder-check (a glance is a stale, peripheral read): the confidence seam
//!   lifts the floor toward [`HEAD_SCAN_MAX_SCANNED_CONFIDENCE`] as coverage rises.
//!
//! ## Parity
//!
//! Gated **off** by default (`DD_SOCCER_ENABLE_HEAD_SCAN`). When unset the live
//! call sites bypass this module entirely (visibility falls back to the fixed cone,
//! confidence to its flat floor), so an unconfigured process is **byte-identical**.
//! The functions are pure and deterministic (time + geometry only — no RNG), so
//! tests and the inspector can read them without enabling the seam.

const HEAD_SCAN_ENABLE_ENV: &str = "DD_SOCCER_ENABLE_HEAD_SCAN";

/// Angular rate (rad/s) the head/shoulders effectively sweep *past* the body's
/// instantaneous field-of-view cone, at average vision skill. This is an
/// *effective* rate: a real carrier alternates glancing back and tracking the
/// ball/play, so it is well below a raw head-turn speed. At `0.95 rad/s` the
/// ~`π/2` from the 90° cone edge to directly behind is covered in ~1.65 s.
pub const HEAD_SCAN_SWEEP_RATE_RAD_PER_S: f64 = 0.95;

/// Extra effective sweep rate at elite vision (added, scaled by `vision01`).
/// Elite scanners check their shoulders more often, so they reach the blind side
/// sooner — at `vision01 = 1` the rate is `1.65 rad/s` (~0.95 s to directly behind).
pub const HEAD_SCAN_SWEEP_RATE_VISION_BONUS_RAD_PER_S: f64 = 0.70;

/// Soft ramp width (rad) at the leading edge of the swept reach. Coverage crosses
/// from `0` to `1` over this band centred on the bearing the reach has just
/// arrived at, so a teammate fades into awareness rather than popping in. ~20°.
pub const HEAD_SCAN_COVERAGE_RAMP_RAD: f64 = 0.35;

/// Coverage at/above which the visibility seam treats a scanned bearing as seen
/// (a candidate pass target). `0.5` = the moment the swept reach arrives at it.
pub const HEAD_SCAN_VISIBLE_COVERAGE: f64 = 0.5;

/// Ceiling the confidence seam lifts a fully-swept blind-side bearing toward. Held
/// **below** the shoulder-check confidence (0.70) on purpose: a head-scan glance is
/// a peripheral, slightly stale read, so a teammate the carrier has a clean
/// shoulder look at should still out-rank one it merely glanced behind it.
pub const HEAD_SCAN_MAX_SCANNED_CONFIDENCE: f64 = 0.62;

fn env_on(name: &str) -> bool {
    std::env::var(name)
        .map(|v| {
            let v = v.trim().to_ascii_lowercase();
            matches!(v.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

/// Whether the head-scan arc-widening is folded into perception this process.
pub fn head_scan_enabled() -> bool {
    #[cfg(test)]
    {
        env_on(HEAD_SCAN_ENABLE_ENV)
    }
    #[cfg(not(test))]
    {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| env_on(HEAD_SCAN_ENABLE_ENV))
    }
}

/// Effective head-sweep rate (rad/s) for a vision ability already mapped to
/// `[0, 1]` (e.g. `ability01(skills.vision)`).
pub fn head_scan_sweep_rate_rad_per_s(vision01: f64) -> f64 {
    let v = if vision01.is_finite() {
        vision01.clamp(0.0, 1.0)
    } else {
        0.0
    };
    HEAD_SCAN_SWEEP_RATE_RAD_PER_S + v * HEAD_SCAN_SWEEP_RATE_VISION_BONUS_RAD_PER_S
}

/// How thoroughly the bearing `off_axis_radians` away from the body's forward
/// facing (`0` = dead ahead, `π` = directly behind) has been swept, in `[0, 1]`.
///
/// * Inside the instantaneous cone (`off_axis ≤ fov_half_angle`) ⇒ `1` (seen now).
/// * Beyond it, the swept *reach* is `scan_seconds · sweep_rate(vision01)`; coverage
///   ramps from `0` to `1` across [`HEAD_SCAN_COVERAGE_RAMP_RAD`] centred on the
///   bearing the reach has reached, hitting `0.5` exactly where `reach == excess`.
///
/// Pure and monotonic: more scan time ⇒ ≥ coverage; a wider off-axis bearing ⇒ ≤
/// coverage. `scan_seconds ≤ 0` (no time to look) ⇒ `0` outside the cone.
pub fn scan_coverage(
    off_axis_radians: f64,
    fov_half_angle_radians: f64,
    scan_seconds: f64,
    vision01: f64,
) -> f64 {
    let off = off_axis_radians.abs();
    let cone = fov_half_angle_radians.max(0.0);
    if !off.is_finite() || !cone.is_finite() {
        return 0.0;
    }
    if off <= cone {
        return 1.0; // already inside the instantaneous field of view
    }
    let excess = off - cone; // how far past the cone edge this bearing sits
    let reach = scan_seconds.max(0.0) * head_scan_sweep_rate_rad_per_s(vision01);
    let ramp = HEAD_SCAN_COVERAGE_RAMP_RAD.max(1e-6);
    ((reach - excess) / ramp + 0.5).clamp(0.0, 1.0)
}

/// Lift a blind-side `base_confidence` toward [`HEAD_SCAN_MAX_SCANNED_CONFIDENCE`]
/// in proportion to how thoroughly that bearing has been scanned. At `coverage = 0`
/// returns `base_confidence` unchanged; at `coverage = 1` returns the ceiling (or
/// `base_confidence` if it already exceeds the ceiling — never *lowers* it).
pub fn scanned_confidence(coverage: f64, base_confidence: f64) -> f64 {
    let c = if coverage.is_finite() {
        coverage.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let headroom = (HEAD_SCAN_MAX_SCANNED_CONFIDENCE - base_confidence).max(0.0);
    base_confidence + c * headroom
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, PI};
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn gate_defaults_off() {
        let _l = ENV_LOCK.lock().expect("env lock");
        std::env::remove_var(HEAD_SCAN_ENABLE_ENV);
        assert!(!head_scan_enabled());
    }

    #[test]
    fn inside_cone_is_fully_covered_regardless_of_time() {
        // A bearing inside the cone is seen now, even with zero scan time.
        assert_eq!(scan_coverage(0.3, FRAC_PI_2, 0.0, 0.5), 1.0);
        assert_eq!(scan_coverage(FRAC_PI_2, FRAC_PI_2, 0.0, 0.5), 1.0);
    }

    #[test]
    fn no_scan_time_means_nothing_outside_the_cone() {
        // Just past the cone edge with no time to look ⇒ not covered.
        assert_eq!(scan_coverage(FRAC_PI_2 + 0.2, FRAC_PI_2, 0.0, 1.0), 0.0);
    }

    #[test]
    fn coverage_grows_with_scan_time_and_reaches_behind_in_about_one_to_two_seconds() {
        let cone = FRAC_PI_2; // ball-holder 90° half-arc
        let behind = PI; // directly behind
        // Average vision: ~1.65 s to sweep the π/2 from cone edge to directly behind.
        let half_second = scan_coverage(behind, cone, 0.5, 0.5);
        let two_second = scan_coverage(behind, cone, 2.0, 0.5);
        assert!(
            two_second > half_second,
            "more scan time ⇒ more rear coverage: {two_second} vs {half_second}"
        );
        assert!(half_second < HEAD_SCAN_VISIBLE_COVERAGE, "behind not yet seen at 0.5s: {half_second}");
        assert!(
            two_second >= HEAD_SCAN_VISIBLE_COVERAGE,
            "behind seen by 2s: {two_second}"
        );
    }

    #[test]
    fn nearer_bearings_are_found_before_the_blind_side() {
        let cone = FRAC_PI_2;
        let t = 0.6;
        // A teammate just outside the cone is swept to well before one directly behind.
        let near = scan_coverage(cone + 0.15, cone, t, 0.5);
        let far = scan_coverage(PI, cone, t, 0.5);
        assert!(near > far, "nearer off-axis found first: {near} vs {far}");
    }

    #[test]
    fn elite_vision_sweeps_faster() {
        let cone = FRAC_PI_2;
        let bearing = cone + 1.0;
        let avg = scan_coverage(bearing, cone, 0.8, 0.0);
        let elite = scan_coverage(bearing, cone, 0.8, 1.0);
        assert!(elite > avg, "elite scans wider in the same time: {elite} vs {avg}");
    }

    #[test]
    fn scanned_confidence_lifts_floor_toward_ceiling_but_never_below() {
        let base = 0.24;
        assert_eq!(scanned_confidence(0.0, base), base);
        assert!((scanned_confidence(1.0, base) - HEAD_SCAN_MAX_SCANNED_CONFIDENCE).abs() < 1e-9);
        let mid = scanned_confidence(0.5, base);
        assert!(mid > base && mid < HEAD_SCAN_MAX_SCANNED_CONFIDENCE, "mid lift: {mid}");
        // Stays below a clean shoulder-check (0.70) even fully scanned.
        assert!(scanned_confidence(1.0, base) < 0.70);
        // Never lowers a base that already exceeds the ceiling.
        assert_eq!(scanned_confidence(1.0, 0.8), 0.8);
    }
}
