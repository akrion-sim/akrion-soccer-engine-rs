//! Centralized **lane discipline** — the single source of truth for how firmly
//! each outfield player is held in its assigned vertical channel ("lane") of the
//! pitch grid.
//!
//! The pitch is modelled as a [`VERTICAL_LANE_COUNT`]-lane × row grid (currently
//! **12 lanes × 24 rows**; see [`crate::des::general::soccer_genome`]). "Lane
//! affinity" is how strongly a player prefers targets in / near its own lane, and
//! "lane discipline" is how hard the engine *enforces* that — the thing that keeps
//! a back four spread across the width instead of collapsing onto the ball.
//!
//! ## Why this module exists
//!
//! The lane math was originally authored for a **4-lane** grid (~20yd per lane)
//! and was never rescaled when the grid tripled to 12 lanes (~6.67yd per lane).
//! Three constants silently changed meaning:
//!
//! * the dynamic lane-match divisor `|a-b| / 5.0` — lane-count-relative, so its
//!   reach went from ~3 lanes (the old max gap) to 5 of 12 lanes;
//! * the static out-of-band falloff `0.58 + 0.22 * lane_gap` — a per-*lane*
//!   coefficient, so it became ~3× steeper per yard;
//! * the clamp's relief **cliff** (`relief < 0.15` ⇒ hard-lock, else a soft
//!   blend) — a step, so any mild justified relief fully released the lane.
//!
//! The net effect was an inconsistent, drifty shape: the producer signal sharpened
//! while the bands widened and enforcement stayed a cliff. This module re-derives
//! the same intent in **yard** terms (grid-/pitch-size invariant), turns the cliff
//! into a smooth taper, and exposes one [`strength`] knob that scales the lane
//! signal every consumer reads.
//!
//! ## Parity
//!
//! Everything here is selected by [`lane_discipline_v2_enabled`]. The 12-lane
//! re-derived path is on by default; set `DD_SOCCER_DISABLE_LANE_DISCIPLINE_V2`
//! for a legacy A/B run. `DD_SOCCER_ENABLE_LANE_DISCIPLINE_V2=0` is also honored
//! for old launch scripts that already write the enable flag. The tunable weights
//! live in [`LaneDisciplineTunables`] (env-/Postgres-overridable via the
//! [`tunables`] registry).

use super::*;

/// Env controls for the centralized, 12-lane-re-derived lane-discipline path.
const LANE_DISCIPLINE_V2_ENABLE_ENV: &str = "DD_SOCCER_ENABLE_LANE_DISCIPLINE_V2";
const LANE_DISCIPLINE_V2_DISABLE_ENV: &str = "DD_SOCCER_DISABLE_LANE_DISCIPLINE_V2";

fn env_flag(name: &str) -> Option<bool> {
    std::env::var(name).ok().map(|raw| {
        let v = raw.trim().to_ascii_lowercase();
        matches!(v.as_str(), "1" | "true" | "yes" | "on")
    })
}

fn lane_discipline_v2_enabled_from_env() -> bool {
    if env_flag(LANE_DISCIPLINE_V2_DISABLE_ENV).unwrap_or(false) {
        return false;
    }
    env_flag(LANE_DISCIPLINE_V2_ENABLE_ENV).unwrap_or(true)
}

/// Whether the centralized lane-discipline path is active this process. Tests
/// re-read the env each call (so an A/B can toggle it); production caches once.
pub fn lane_discipline_v2_enabled() -> bool {
    #[cfg(test)]
    {
        lane_discipline_v2_enabled_from_env()
    }
    #[cfg(not(test))]
    {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(lane_discipline_v2_enabled_from_env)
    }
}

fn params() -> &'static LaneDisciplineTunables {
    &tunables().lane_discipline
}

/// Width of one vertical lane in yards on a pitch of `field_width`.
fn lane_width_yards(field_width: f64) -> f64 {
    field_width.max(1.0) / VERTICAL_LANE_COUNT as f64
}

/// Global lane-discipline **strength** — the one knob every consumer feels. The
/// producers multiply their affinity bonus and lane penalty by this, so a single
/// value tightens (`>1`) or loosens (`<1`) team shape everywhere at once. Returns
/// `1.0` (identity) when v2 is off so callers can multiply unconditionally.
pub fn strength() -> f64 {
    if lane_discipline_v2_enabled() {
        params().strength.max(0.0)
    } else {
        1.0
    }
}

/// Yard-based lane match in `[0, 1]`: how much a point in lane `b` still counts as
/// being in lane `a`. Credit decays linearly to zero over
/// [`LaneDisciplineTunables::lane_match_decay_yards`] of lateral gap. Replaces the
/// legacy lane-count-relative `|a-b| / 5.0`, which silently re-scaled with the
/// grid size.
pub fn lane_match(lane_a: usize, lane_b: usize, field_width: f64) -> f64 {
    let gap_yards = lane_a.abs_diff(lane_b) as f64 * lane_width_yards(field_width);
    let decay = params().lane_match_decay_yards.max(1e-6);
    (1.0 - gap_yards / decay).clamp(0.0, 1.0)
}

/// Static out-of-band affinity score in `[0, 1]` for a target `deviation_yards`
/// outside the role's home band, given the role's lane `commitment`. Yard-based
/// re-derivation of the legacy `1 - commitment * (0.58 + 0.22 * lane_gap)` (whose
/// per-*lane* coefficient became ~3× steeper per yard at 12 lanes). `1.0` when the
/// target is inside the band (`deviation_yards <= 0`).
pub fn static_affinity_score(deviation_yards: f64, commitment: f64) -> f64 {
    if deviation_yards <= 0.0 {
        return 1.0;
    }
    let p = params();
    (1.0 - commitment * (p.static_fit_base + deviation_yards * p.static_fit_per_yard))
        .clamp(0.0, 1.0)
}

/// Blend fraction in `[0, 1]` for pulling a clamped target's `x` toward its lane
/// edge (`0` = keep the target's x, `1` = snap to the lane). The **taper** that
/// replaces the legacy relief cliff: hard-lock authority ramps smoothly from full
/// (no relief) to zero (relief ≥ `relief_full_release`), gated by how committed
/// the role is, and never falls below the legacy soft-blend floor
/// (`commitment * soft_blend_max`). So a *mild*, justified relief move relaxes the
/// lane instead of abandoning it.
pub fn lane_clamp_blend(commitment: f64, relief: f64) -> f64 {
    let p = params();
    let relief_release = p.relief_full_release.max(1e-6);
    // Full hard-lock when unrelieved, tapering to zero as relief grows to release.
    let hard_authority = ((relief_release - relief) / relief_release).clamp(0.0, 1.0);
    // Ramp to full hard-lock as commitment reaches the legacy 0.65 threshold.
    let commit_gate = (commitment / p.commitment_hard_lock.max(1e-6)).clamp(0.0, 1.0);
    let lock_fraction = hard_authority * commit_gate;
    let soft_floor = (commitment * p.soft_blend_max).clamp(0.0, p.soft_blend_max);
    (lock_fraction + (1.0 - lock_fraction) * soft_floor).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // A 12-lane / 80yd pitch ⇒ 6.667yd lanes. The v2 defaults are tuned to
    // reproduce the legacy reach at this width, then stay invariant to it.
    const W: f64 = 80.0;
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn with_env_lock<R>(f: impl FnOnce() -> R) -> R {
        let _guard = ENV_LOCK.lock().expect("lane-discipline env lock poisoned");
        f()
    }

    fn with_v2<R>(f: impl FnOnce() -> R) -> R {
        with_env_lock(|| {
            std::env::remove_var(LANE_DISCIPLINE_V2_DISABLE_ENV);
            std::env::set_var(LANE_DISCIPLINE_V2_ENABLE_ENV, "1");
            let r = f();
            std::env::remove_var(LANE_DISCIPLINE_V2_ENABLE_ENV);
            r
        })
    }

    #[test]
    fn lane_discipline_v2_is_default() {
        with_env_lock(|| {
            std::env::remove_var(LANE_DISCIPLINE_V2_ENABLE_ENV);
            std::env::remove_var(LANE_DISCIPLINE_V2_DISABLE_ENV);
            assert!(lane_discipline_v2_enabled());
            assert_eq!(strength(), params().strength);
        });
    }

    #[test]
    fn strength_is_identity_when_v2_is_disabled() {
        with_env_lock(|| {
            std::env::remove_var(LANE_DISCIPLINE_V2_ENABLE_ENV);
            std::env::set_var(LANE_DISCIPLINE_V2_DISABLE_ENV, "1");
            assert!(!lane_discipline_v2_enabled());
            assert_eq!(strength(), 1.0);
            std::env::remove_var(LANE_DISCIPLINE_V2_DISABLE_ENV);
        });
    }

    #[test]
    fn enable_env_false_keeps_legacy_ab_path_available() {
        with_env_lock(|| {
            std::env::remove_var(LANE_DISCIPLINE_V2_DISABLE_ENV);
            std::env::set_var(LANE_DISCIPLINE_V2_ENABLE_ENV, "0");
            assert!(!lane_discipline_v2_enabled());
            std::env::remove_var(LANE_DISCIPLINE_V2_ENABLE_ENV);
        });
    }

    #[test]
    fn lane_match_is_yard_based_and_monotone() {
        with_v2(|| {
            assert_eq!(lane_match(5, 5, W), 1.0);
            // Same lane gap, wider pitch ⇒ larger yard gap ⇒ lower match.
            let near = lane_match(3, 5, W);
            let near_wide = lane_match(3, 5, W * 1.5);
            assert!(near_wide < near, "wider pitch widens the yard gap");
            // Far apart ⇒ no credit.
            assert_eq!(lane_match(0, 11, W), 0.0);
        });
    }

    #[test]
    fn static_affinity_full_inside_band_and_falls_off() {
        with_v2(|| {
            assert_eq!(static_affinity_score(0.0, 0.9), 1.0);
            let a = static_affinity_score(3.0, 0.9);
            let b = static_affinity_score(9.0, 0.9);
            assert!(a > b, "further out of band ⇒ weaker affinity");
            assert!((0.0..=1.0).contains(&a));
        });
    }

    #[test]
    fn clamp_blend_tapers_instead_of_cliffing() {
        with_v2(|| {
            // Committed defender (0.9): full lock unrelieved, decaying with relief,
            // monotone (no cliff), floored at the legacy soft blend.
            let none = lane_clamp_blend(0.9, 0.0);
            let mild = lane_clamp_blend(0.9, 0.15);
            let lots = lane_clamp_blend(0.9, 0.9);
            assert!((none - 1.0).abs() < 1e-9, "unrelieved ⇒ hard lock");
            assert!(none > mild && mild > lots, "monotone taper, not a cliff");
            let soft_floor = 0.9 * params().soft_blend_max;
            assert!(
                lots >= soft_floor - 1e-9,
                "never below the legacy soft blend"
            );
        });
    }
}
