//! **Winger "stay wide vs pinch in" off-ball decision.**
//!
//! As the ball moves into the opponent's attacking final third, the two wide attackers
//! (outside midfielders / wide forwards — "wingers") face a genuine positional decision that
//! the engine previously did not model as a *choice*:
//!
//! * **stay wide** — hug the touchline to stretch the back line, offer an overlap / switch
//!   outlet, and (on the ball-side flank) be the man who delivers the cross; or
//! * **pinch in** — tuck off the touchline to **get on the end of a cross**: either into the
//!   **half-space** (cut-back / second-ball / top-of-box arrival) or all the way to the **back
//!   post** to attack a delivery whipped in from the opposite flank.
//!
//! Football logic: the wide player **on the same flank as the ball** holds his width (he is
//! the deliverer / overlap threat and stretches the defence), while the **weak-side** winger
//! pinches infield to attack the box — most classically arriving at the **back post**, which
//! sits on *his* side of the goal, when the cross comes from the far flank. Box congestion and
//! the offside line steer the choice between a deep back-post run and a shallower half-space /
//! cut-back position.
//!
//! ## Where this sits in the decision space
//!
//! This is the **off-ball movement** family decision for the two winger positions (see
//! `docs/position-decision-space.md`). It is the *intermediate* shape decision that runs across
//! the whole "ball in the final third" window. The late, committed box-flood
//! ([`WorldSnapshot::crash_the_box_target_for`]) still takes precedence once a ball is genuinely
//! wide-and-high and the cross is imminent — this decision is what gets the weak-side winger
//! *already tucked toward the box* so he is there for that flood. `StayWide` returns no override
//! and falls through to the normal wide-outlet target, so width behaviour is unchanged.
//!
//! ## Discretisation (the MDP/POMDP action)
//!
//! Per the project decision schema (`mdp-pomdp-action-schema.md`), the choice is **discretised
//! into labelled buckets** — [`WingerPinchChoice`] `{StayWide, PinchHalfSpace, PinchBackPost}`
//! — each **scored from POMDP-observable features** (which flank the ball is on, whether a
//! crossing position is on, ball depth in the final third, box congestion) with the **illegal
//! (offside) back-post bucket masked out**, then argmax-selected. The realised target carries a
//! little continuity via the existing onside clamp at the call site. Outcomes flow back through
//! the existing flank cross-arrival / header-goal reward channel (see [`crash_box`]); no
//! duplicate reward term is added here, so credit for "getting on the end of the cross" is not
//! double-counted.
//!
//! ## Gating
//!
//! Behind [`winger_pinch_in_enabled`]: `gate_default_on` in production
//! (`DD_SOCCER_ENABLE_WINGER_PINCH_IN`), **default-OFF under `cfg(test)`** so the suite and the
//! current network weights stay byte-identical until flipped on for a retrain A/B. The geometry
//! and decision helpers below are pure and gate-agnostic; the gate is applied at the
//! [`WorldSnapshot`] call site.

use super::*;

/// Production kill-switch / test enable env var for the winger pinch-in decision.
pub(crate) const WINGER_PINCH_IN_ENABLE_ENV: &str = "DD_SOCCER_ENABLE_WINGER_PINCH_IN";

/// Master gate. Default-ON in production (set the env to `0`/`false`/`no`/`off` to kill it);
/// default-OFF under `cfg(test)` so the suite stays byte-identical. Tests re-read the env each
/// call (so an A/B can toggle it within the process); production caches the decision once.
pub(crate) fn winger_pinch_in_enabled() -> bool {
    #[cfg(test)]
    {
        std::env::var(WINGER_PINCH_IN_ENABLE_ENV)
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
        *V.get_or_init(|| gate_default_on(WINGER_PINCH_IN_ENABLE_ENV))
    }
}

/// The discretised winger off-ball positioning choice in the attacking final third.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WingerPinchChoice {
    /// Hold the touchline — stretch the defence, offer the overlap / switch, deliver the cross.
    StayWide,
    /// Tuck into the half-space — cut-back / second-ball / top-of-box arrival (stays onside).
    PinchHalfSpace,
    /// Crash the back post to attack a cross whipped in from the opposite flank.
    PinchBackPost,
}

impl WingerPinchChoice {
    /// Stable label for the action key / telemetry.
    pub(crate) fn label(self) -> &'static str {
        match self {
            WingerPinchChoice::StayWide => "stay-wide",
            WingerPinchChoice::PinchHalfSpace => "pinch-half-space",
            WingerPinchChoice::PinchBackPost => "pinch-back-post",
        }
    }
}

/// Learner-visible profile for the stay-wide / half-space / back-post bucket choice.
#[derive(Clone, Copy, Debug)]
pub(crate) struct WingerPinchDecisionProfile {
    pub(crate) choice: WingerPinchChoice,
    pub(crate) stay_score: f64,
    pub(crate) half_space_score: f64,
    pub(crate) back_post_score: f64,
    pub(crate) ball_on_my_flank: bool,
    pub(crate) crossing_position_on: bool,
    pub(crate) ball_final_third_depth_frac: f64,
    pub(crate) box_congestion: usize,
    pub(crate) back_post_offside: bool,
}

// --- Decision bucket scoring weights (calibrated priors; reinforced via the cross-arrival
// reward channel — see module docs). Kept as named constants so an A/B can tune them and a
// later learned head can replace the scoring while keeping the same bucket space. ---

/// Baseline pull toward holding width — the safe default for a winger.
const STAY_BASE: f64 = 1.0;
/// Extra "stay wide" weight when the ball is on this winger's own flank (he is the deliverer /
/// overlap threat and should stretch the line, not vacate the wing).
const SAME_FLANK_STAY_BONUS: f64 = 2.4;
/// Extra "stay wide" weight when no crossing position is on yet (keep width as a switch outlet).
const NO_CROSS_STAY_BONUS: f64 = 1.1;
/// Pull infield for the weak-side (far-flank) winger — the classic back-post / cut-back arrival.
const FAR_FLANK_PINCH_BONUS: f64 = 2.0;
/// Added to the back-post bucket once a genuine crossing position is on.
const CROSS_ON_BACKPOST_BONUS: f64 = 2.2;
/// Added to the half-space bucket once a genuine crossing position is on (cut-back option).
const CROSS_ON_HALFSPACE_BONUS: f64 = 1.4;
/// Per-unit ball-depth weight on the back-post bucket (deeper ball ⇒ commit to the back post).
const DEPTH_BACKPOST_WEIGHT: f64 = 2.0;
/// Per-runner weight nudging a congested box toward the half-space / cut-back instead.
const CONGESTION_HALFSPACE_WEIGHT: f64 = 0.7;
/// Per-runner penalty on the back-post bucket when the box is already crowded.
const CONGESTION_BACKPOST_PENALTY: f64 = 0.6;

// --- Realised-target geometry (yards). ---

/// Lateral offset from pitch centre, toward this winger's own side, for the half-space tuck.
pub(crate) const PINCH_HALF_SPACE_OFFSET_YARDS: f64 = 12.0;
/// Depth from the attacked goal line for the half-space / cut-back arrival (top of the box).
pub(crate) const PINCH_HALF_SPACE_DEPTH_YARDS: f64 = 16.0;
/// Lateral offset from centre, toward this winger's own side, for the back-post arrival.
pub(crate) const PINCH_BACK_POST_OFFSET_YARDS: f64 = 4.5;
/// Depth from the attacked goal line for the back-post arrival.
pub(crate) const PINCH_BACK_POST_DEPTH_YARDS: f64 = 5.5;

/// At/above this many own attackers already inside the box, a deep back-post run is treated as
/// congested and the scoring leans toward the half-space / cut-back instead.
pub(crate) const PINCH_CROWDED_BOX_RUNNERS: usize = 3;

/// Score the three buckets from POMDP-observable features and return the argmax choice.
///
/// * `ball_on_my_flank` — the ball / carrier is on the same side of centre as this winger's home.
/// * `crossing_position_on` — a genuine crossing position is on (wide & high in the final third).
/// * `ball_final_third_depth_frac` — 0 at the final-third edge → 1 at the byline (clamped).
/// * `box_congestion` — own attackers already inside the opponent box (excluding this winger).
/// * `back_post_offside` — the back-post arrival point would be offside ⇒ that bucket is masked.
pub(crate) fn winger_pinch_decision_profile(
    ball_on_my_flank: bool,
    crossing_position_on: bool,
    ball_final_third_depth_frac: f64,
    box_congestion: usize,
    back_post_offside: bool,
) -> WingerPinchDecisionProfile {
    winger_pinch_decision_profile_with_bias(
        ball_on_my_flank,
        crossing_position_on,
        ball_final_third_depth_frac,
        box_congestion,
        back_post_offside,
        0.0,
    )
}

/// As [`winger_pinch_decision_profile`], but with a learned **pinch bias** in `[-1, 1]`
/// folded into the two pinch buckets' scores (positive ⇒ lean toward pinching infield, negative
/// ⇒ hold width). See [`super::winger_pinch_decision`]. `pinch_bias == 0.0` is byte-identical to
/// [`winger_pinch_decision_profile`].
pub(crate) fn winger_pinch_decision_profile_with_bias(
    ball_on_my_flank: bool,
    crossing_position_on: bool,
    ball_final_third_depth_frac: f64,
    box_congestion: usize,
    back_post_offside: bool,
    pinch_bias: f64,
) -> WingerPinchDecisionProfile {
    let depth = ball_final_third_depth_frac.clamp(0.0, 1.0);
    let congestion = box_congestion as f64;

    let mut stay = STAY_BASE;
    let mut half = 0.0;
    let mut back = 0.0;

    // Only the **weak-side** winger with a genuine crossing position on pinches infield: the
    // ball-side winger is the deliverer / overlap and the team needs his width, and with no
    // cross threat yet width keeps the team stretched and offers the switch outlet.
    let pinch_in_play = !ball_on_my_flank && crossing_position_on;
    if ball_on_my_flank {
        stay += SAME_FLANK_STAY_BONUS;
    } else if !crossing_position_on {
        stay += NO_CROSS_STAY_BONUS;
    }

    if pinch_in_play {
        // Weak-side arriver: lean infield, the deeper back-post run weighted higher with depth,
        // a congested box pulling toward the cut-back / half-space instead.
        back += FAR_FLANK_PINCH_BONUS + CROSS_ON_BACKPOST_BONUS + depth * DEPTH_BACKPOST_WEIGHT;
        half += FAR_FLANK_PINCH_BONUS * 0.6 + CROSS_ON_HALFSPACE_BONUS;
        half += congestion * CONGESTION_HALFSPACE_WEIGHT;
        back -= congestion * CONGESTION_BACKPOST_PENALTY;
    }

    // Learned pinch appetite: a signed bias (default 0) shifts BOTH pinch buckets relative to
    // holding width — the head's contribution to the stay-vs-pinch choice. Bounded by
    // `super::winger_pinch_decision::WINGER_PINCH_APPETITE_WEIGHT` so it nudges without
    // overriding the same-flank "hold width" rule.
    let pinch_shift =
        pinch_bias.clamp(-1.0, 1.0) * super::winger_pinch_decision::WINGER_PINCH_APPETITE_WEIGHT;
    half += pinch_shift;
    back += pinch_shift;

    // Legality mask: an offside back-post arrival is illegal — prune that bucket.
    let mut masked_back = back;
    if back_post_offside {
        masked_back = f64::NEG_INFINITY;
    }

    // Argmax over the three buckets, ties resolved toward the safer (wider) option.
    let mut best = WingerPinchChoice::StayWide;
    let mut best_score = stay;
    if half > best_score {
        best = WingerPinchChoice::PinchHalfSpace;
        best_score = half;
    }
    if masked_back > best_score {
        best = WingerPinchChoice::PinchBackPost;
    }
    WingerPinchDecisionProfile {
        choice: best,
        stay_score: stay.max(0.0),
        half_space_score: half.max(0.0),
        back_post_score: if back_post_offside {
            0.0
        } else {
            back.max(0.0)
        },
        ball_on_my_flank,
        crossing_position_on,
        ball_final_third_depth_frac: depth,
        box_congestion,
        back_post_offside,
    }
}

pub(crate) fn decide_winger_pinch(
    ball_on_my_flank: bool,
    crossing_position_on: bool,
    ball_final_third_depth_frac: f64,
    box_congestion: usize,
    back_post_offside: bool,
) -> WingerPinchChoice {
    winger_pinch_decision_profile(
        ball_on_my_flank,
        crossing_position_on,
        ball_final_third_depth_frac,
        box_congestion,
        back_post_offside,
    )
    .choice
}

/// As [`decide_winger_pinch`], but with a learned **pinch bias** in `[-1, 1]` folded into the two
/// pinch buckets' scores. This is a thin compatibility wrapper around
/// [`winger_pinch_decision_profile_with_bias`] so live action selection and learner-visible bucket
/// scores stay conceptually identical.
pub(crate) fn decide_winger_pinch_with_bias(
    ball_on_my_flank: bool,
    crossing_position_on: bool,
    ball_final_third_depth_frac: f64,
    box_congestion: usize,
    back_post_offside: bool,
    pinch_bias: f64,
) -> WingerPinchChoice {
    winger_pinch_decision_profile_with_bias(
        ball_on_my_flank,
        crossing_position_on,
        ball_final_third_depth_frac,
        box_congestion,
        back_post_offside,
        pinch_bias,
    )
    .choice
}

/// Lower a pinch choice to its realised off-ball target. Returns `None` for [`WingerPinchChoice::StayWide`]
/// (the caller then keeps the normal wide-outlet target). The target is on this winger's own
/// side of the goal; the caller is responsible for the onside clamp / offside guard.
///
/// * `home_x` — the winger's home x (its side of the pitch is `sign(home_x - centre)`).
/// * `attacked_goal_y` — the y of the goal this team attacks.
/// * `attack_dir` — +1 / -1 toward that goal.
pub(crate) fn winger_pinch_target(
    choice: WingerPinchChoice,
    home_x: f64,
    attacked_goal_y: f64,
    attack_dir: f64,
    field_width: f64,
    field_length: f64,
) -> Option<Vec2> {
    if !home_x.is_finite() || !attacked_goal_y.is_finite() || field_width <= 0.0 {
        return None;
    }
    let center_x = field_width * 0.5;
    // The winger's own side: -1 = left of centre, +1 = right of centre.
    let side = (home_x - center_x).signum();
    let side = if side == 0.0 { -1.0 } else { side };
    let (offset, depth) = match choice {
        WingerPinchChoice::StayWide => return None,
        WingerPinchChoice::PinchHalfSpace => {
            (PINCH_HALF_SPACE_OFFSET_YARDS, PINCH_HALF_SPACE_DEPTH_YARDS)
        }
        WingerPinchChoice::PinchBackPost => {
            (PINCH_BACK_POST_OFFSET_YARDS, PINCH_BACK_POST_DEPTH_YARDS)
        }
    };
    let x = (center_x + side * offset).clamp(0.0, field_width);
    let y = (attacked_goal_y - attack_dir * depth).clamp(0.0, field_length);
    Some(Vec2::new(x, y))
}

/// Ball depth within the attacking final third: 0 at the final-third edge, 1 at the byline.
/// Returns 0 when the ball is short of the final third.
pub(crate) fn final_third_depth_frac(ball_y: f64, attacked_goal_y: f64, field_length: f64) -> f64 {
    if !ball_y.is_finite() || !attacked_goal_y.is_finite() || field_length <= 0.0 {
        return 0.0;
    }
    let attack_dir = (attacked_goal_y - field_length * 0.5).signum();
    let attack_dir = if attack_dir == 0.0 { 1.0 } else { attack_dir };
    let third = field_length / 3.0;
    // Distance still to run to the goal line, in [0, third] within the final third.
    let to_goal = (attacked_goal_y - ball_y) * attack_dir;
    if to_goal < 0.0 || to_goal > third {
        return 0.0;
    }
    (1.0 - to_goal / third).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: f64 = 80.0;
    const L: f64 = 120.0;

    #[test]
    fn ball_on_my_flank_holds_width() {
        // Same-flank winger is the deliverer / overlap — stay wide even with a cross on.
        let choice = decide_winger_pinch(true, true, 0.9, 0, false);
        assert_eq!(choice, WingerPinchChoice::StayWide);
    }

    #[test]
    fn weak_side_deep_cross_attacks_back_post() {
        // Far-flank winger, crossing position on, ball deep, box not crowded, onside ⇒ back post.
        let choice = decide_winger_pinch(false, true, 0.85, 1, false);
        assert_eq!(choice, WingerPinchChoice::PinchBackPost);
    }

    #[test]
    fn crowded_box_prefers_half_space_cutback() {
        // Same weak-side cross, but the box is already packed ⇒ tuck to the cut-back instead.
        let choice = decide_winger_pinch(false, true, 0.85, PINCH_CROWDED_BOX_RUNNERS + 1, false);
        assert_eq!(choice, WingerPinchChoice::PinchHalfSpace);
    }

    #[test]
    fn offside_back_post_is_masked_to_half_space() {
        // Deep weak-side cross that would be offside at the back post falls to the half-space.
        let choice = decide_winger_pinch(false, true, 0.95, 0, true);
        assert_eq!(choice, WingerPinchChoice::PinchHalfSpace);
    }

    #[test]
    fn no_cross_threat_keeps_width() {
        // Weak-side but no crossing position on yet ⇒ keep width as a switch outlet.
        let choice = decide_winger_pinch(false, false, 0.2, 0, false);
        assert_eq!(choice, WingerPinchChoice::StayWide);
    }

    #[test]
    fn back_post_target_is_on_own_side_near_goal() {
        // Right winger (home x > centre) attacks the right-hand (back) post for a left cross.
        let target = winger_pinch_target(
            WingerPinchChoice::PinchBackPost,
            W * 0.85,
            L, // Home attacks y = L
            1.0,
            W,
            L,
        )
        .expect("back-post target");
        assert!(target.x > W * 0.5, "back post is on the winger's own side");
        assert!(
            (L - target.y).abs() <= 7.0,
            "back post sits close to the goal line"
        );
        // Half-space tuck is wider of the post and shallower than the back-post run.
        let half = winger_pinch_target(WingerPinchChoice::PinchHalfSpace, W * 0.85, L, 1.0, W, L)
            .expect("half-space target");
        assert!(
            half.x > target.x,
            "half-space sits wider than the back post"
        );
        assert!(
            (L - half.y) > (L - target.y),
            "half-space sits further from goal"
        );
    }

    #[test]
    fn stay_wide_has_no_override_target() {
        assert!(winger_pinch_target(WingerPinchChoice::StayWide, W * 0.85, L, 1.0, W, L).is_none());
    }

    #[test]
    fn final_third_depth_grows_toward_the_byline() {
        // Home attacks y = L. Final-third edge at y = 80.
        assert_eq!(final_third_depth_frac(70.0, L, L), 0.0); // short of the final third
        assert!(final_third_depth_frac(100.0, L, L) > 0.4);
        assert!(final_third_depth_frac(118.0, L, L) > 0.9); // near the byline
                                                            // Away attacks y = 0: depth grows toward y = 0.
        assert!(final_third_depth_frac(2.0, 0.0, L) > 0.9);
        assert_eq!(final_third_depth_frac(50.0, 0.0, L), 0.0);
    }
}
