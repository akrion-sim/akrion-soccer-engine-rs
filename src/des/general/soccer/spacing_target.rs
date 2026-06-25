//! Learnable **attacking spacing target** — *how much space an off-ball attacker
//! should keep to the nearest teammate*.
//!
//! The "move to space" ranker ([`WorldSnapshot::open_space_for`]) and the formation
//! LP both need a *target inter-player spacing*: the distance an off-ball player
//! should hold to the nearest teammate so the team fills space evenly without two
//! players occupying the same pocket. Historically that was a single fixed band
//! (`ATTACK_SPACING_*`). In reality the right spacing is a **function of the match
//! state** — the user's spec: *"a function of the centre of mass, ball position, the
//! defending team's vector, and the direction the carrier is moving/facing; by
//! default 5-8 yards with a ~2 second grace period to converge."* This module makes
//! that function explicit and **learnable**.
//!
//! ## What it computes
//!
//! Given an [`AttackSpacingContext`] (the named state quantities) it returns an
//! [`AttackSpacingTarget`] — a `{min, ideal, max}` band in yards, seeded to the 5-8yd
//! default and modulated by context. [`AttackSpacingTarget::spacing_score`] scores a
//! realized nearest-teammate distance against that band exactly like
//! [`super::spacing_score_from_distance`] scores against the fixed band, so it is a
//! drop-in replacement at the live chokepoint.
//!
//! ## Design (mirrors [`super::back_four_line`] / [`super::support_scorer`])
//!
//! * [`AttackSpacingContext`] — the per-decision feature vector, stored raw (yards /
//!   yps / counts) so a row is human- and test-readable. The exact normalized
//!   ordering is the single source of truth ([`AttackSpacingContext::to_features`]).
//!   The fields ARE the quantities the user named:
//!     1. centre of mass — team centroid relative to the ball + self relative to it,
//!     2. ball position — progress up the pitch + how far behind it the player is,
//!     3. defending team vector — opponent centroid offset, compactness, and mean
//!        velocity (the "vector" — where the block is moving),
//!     4. carrier heading — the on-ball player's velocity (fore/aft + lateral),
//!     5. fill/occupancy — local teammate density, current nearest-teammate gap, and
//!        the team's width shortfall (so we "fill empty space" and never crowd).
//! * [`analytic_target_spacing`] — the deterministic, RNG-free seed = a closed-form
//!   5-8yd function. Live fallback + the bootstrap target a fresh head imitates.
//! * [`AttackSpacingHead`] — a `FeedForwardNetwork` (`DIM -> hidden -> 1`, linear)
//!   regressing the *ideal* spacing in yards, trained from self-play
//!   [`AttackSpacingSample`]s whose reward is the pitch-control x xT territorial
//!   advantage gained over the following window (see `pitch_value`) — i.e. "did
//!   holding that spacing improve our control of valuable space".
//!
//! ## Parity
//!
//! Gated **off** by default ([`attack_spacing_model_enabled`] /
//! `DD_SOCCER_ENABLE_LEARNED_SPACING_TARGET`). When unset the live chokepoints keep
//! the existing fixed `ATTACK_SPACING_*` band, so an unconfigured process is
//! byte-identical. Enable it in the learning / live deployment to switch to the 5-8yd
//! learnable spacing (the user's desired default) and start training the head. Feature
//! build, analytic seed, and the head's `predict`/`train` are pure and RNG-free at the
//! call site, so tests and the inspector can read them without the live seam.

use super::*;
use crate::des::general::neural_network::{ActivationName, FeedForwardNetwork, RandomNetworkSpec};
use crate::des::general::prng::mulberry32;

/// Number of features in the attacking-spacing context vector. The ordering is defined
/// once in [`AttackSpacingContext::to_features`] — change it and bump this constant and
/// any persisted head.
pub const ATTACK_SPACING_FEATURE_DIM: usize = 14;
/// Hidden width of the learned regression head.
pub const ATTACK_SPACING_HIDDEN_UNITS: usize = 20;

/// Default *ideal* nearest-teammate spacing for an off-ball attacker, in yards — the
/// centre of the user's 5-8yd band. The seed returns this in a neutral state.
pub const DEFAULT_ATTACK_SPACING_IDEAL_YARDS: f64 = 6.5;
/// Half-width of the seed band around the ideal (ideal +/- this), so the neutral seed
/// is `[5, 8]` exactly: the user's "5-8 yards by default".
pub const ATTACK_SPACING_BAND_MARGIN_YARDS: f64 = 1.5;
/// Absolute floor on the learned/served ideal — players never crowd inside this even
/// in the tightest combination (just above the hard occupancy radius).
pub const ATTACK_SPACING_ABS_MIN_YARDS: f64 = 4.5;
/// Absolute ceiling on the learned/served ideal — beyond this the "fill empty space"
/// pull would leave holes a teammate should occupy.
pub const ATTACK_SPACING_ABS_MAX_YARDS: f64 = 11.0;
/// Grace period (seconds) a player is given to converge to / from the target band
/// before the spacing correction is at full strength — the user's "~2 second grace".
/// Consumed by the formation-LP lateral-spread horizon when the model is on.
pub const ATTACK_SPACING_REGROUP_GRACE_SECONDS: f64 = 2.0;

/// How often (ticks) an attacking-spacing decision is sampled for RL while the model is on.
pub const ATTACK_SPACING_SAMPLE_INTERVAL_TICKS: u64 = 15;
/// Window (ticks) over which a sampled spacing decision's territorial reward is measured.
pub const ATTACK_SPACING_REWARD_WINDOW_TICKS: u64 = 45;
/// Cap on the rolling RL sample buffer (drained by the learner).
pub const ATTACK_SPACING_SAMPLE_CAP: usize = 4096;

// Normalization references for the network input (raw -> ~unit range).
const REF_FWD_YARDS: f64 = 30.0;
const REF_LAT_YARDS: f64 = 24.0;
const REF_VEL_YPS: f64 = 8.0;
const REF_SPREAD_YARDS: f64 = 28.0;
const REF_DENSITY: f64 = 4.0;
const REF_GAP_YARDS: f64 = 12.0;

const ATTACK_SPACING_ENABLE_ENV: &str = "DD_SOCCER_ENABLE_LEARNED_SPACING_TARGET";

/// Whether the learned attacking-spacing target is consulted at the live chokepoints
/// (`open_space_for` spacing term + the formation-LP lateral spread) this process. Off
/// (unset) ⇒ the existing fixed `ATTACK_SPACING_*` band stands, so play is byte-identical.
pub fn attack_spacing_model_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var(ATTACK_SPACING_ENABLE_ENV).is_ok())
}

/// The named state quantities the target spacing is a function of, in the team's
/// attacking frame (so the model is mirror-invariant). Stored raw (yards / yps /
/// counts) for readability; [`Self::to_features`] normalizes for the network.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttackSpacingContext {
    // 1. Centre of mass (team centroid) relative to the ball + self relative to it.
    /// Team centroid's forward offset from the ball (+ = mass ahead of the ball).
    pub centroid_fwd_from_ball: f64,
    /// Team centroid's lateral offset from the ball.
    pub centroid_lat_from_ball: f64,
    /// This player's forward offset from the team centroid (+ = ahead of the mass).
    pub self_fwd_from_centroid: f64,
    // 2. Ball position.
    /// Ball progress up the pitch in `[0, 1]` (0 = our goal line, 1 = their goal line).
    pub ball_progress: f64,
    /// How far behind the ball the player is (>= 0, forward axis) — deep runners can
    /// hold wider spacing as they climb into the channel.
    pub behind_ball_yards: f64,
    // 3. Defending team vector.
    /// Opponent centroid's forward offset from the ball (+ = block sits high / presses up).
    pub def_centroid_fwd_from_ball: f64,
    /// Opponent shape spread in yards (small = compact low block).
    pub def_spread_yards: f64,
    /// Opponent mean velocity, forward component in our attacking frame (+ = stepping up).
    pub def_mean_vel_fwd: f64,
    /// Opponent mean velocity, lateral component (block shifting across).
    pub def_mean_vel_lat: f64,
    // 4. Direction the carrier is moving / facing.
    /// Ball carrier's velocity, forward component (+ = driving at goal).
    pub carrier_vel_fwd: f64,
    /// Ball carrier's velocity, lateral component (which way play is being carried).
    pub carrier_vel_lat: f64,
    // 5. Fill / occupancy — "fill empty space, do not occupy the same space".
    /// Teammates within the local support radius (the local crowding count).
    pub local_teammate_density: f64,
    /// Current distance to the nearest teammate (yards).
    pub nearest_teammate_yards: f64,
    /// Team width shortfall in `[0, 1]` (1 = far too narrow vs the directive width).
    pub width_shortage: f64,
}

impl AttackSpacingContext {
    /// The normalized network input vector. **This ordering is the single source of
    /// truth** for the feature layout — change it and bump [`ATTACK_SPACING_FEATURE_DIM`]
    /// and any persisted head.
    pub fn to_features(&self) -> [f64; ATTACK_SPACING_FEATURE_DIM] {
        let f = |v: f64, r: f64| (v / r).clamp(-4.0, 4.0);
        [
            f(self.centroid_fwd_from_ball, REF_FWD_YARDS),
            f(self.centroid_lat_from_ball, REF_LAT_YARDS),
            f(self.self_fwd_from_centroid, REF_FWD_YARDS),
            self.ball_progress.clamp(0.0, 1.0) * 2.0 - 1.0,
            f(self.behind_ball_yards, REF_FWD_YARDS),
            f(self.def_centroid_fwd_from_ball, REF_FWD_YARDS),
            f(self.def_spread_yards, REF_SPREAD_YARDS),
            f(self.def_mean_vel_fwd, REF_VEL_YPS),
            f(self.def_mean_vel_lat, REF_VEL_YPS),
            f(self.carrier_vel_fwd, REF_VEL_YPS),
            f(self.carrier_vel_lat, REF_VEL_YPS),
            f(self.local_teammate_density, REF_DENSITY),
            f(self.nearest_teammate_yards, REF_GAP_YARDS),
            self.width_shortage.clamp(0.0, 1.0),
        ]
    }
}

/// A served target-spacing band in yards. `min`/`max` bracket the "comfortable" range
/// (the `spacing_score` is positive inside it, peaking up to `ideal`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttackSpacingTarget {
    pub min: f64,
    pub ideal: f64,
    pub max: f64,
}

impl AttackSpacingTarget {
    /// Build the `{min, ideal, max}` band from an ideal spacing, applying the seed
    /// margin and clamping to the absolute legal range. A neutral `ideal = 6.5`
    /// yields exactly `[5, 8]` — the user's default band.
    pub fn from_ideal(ideal: f64) -> Self {
        let ideal = if ideal.is_finite() {
            ideal.clamp(ATTACK_SPACING_ABS_MIN_YARDS, ATTACK_SPACING_ABS_MAX_YARDS)
        } else {
            DEFAULT_ATTACK_SPACING_IDEAL_YARDS
        };
        let min = (ideal - ATTACK_SPACING_BAND_MARGIN_YARDS).max(ATTACK_SPACING_ABS_MIN_YARDS - 0.5);
        let max = ideal + ATTACK_SPACING_BAND_MARGIN_YARDS;
        AttackSpacingTarget { min, ideal, max }
    }

    /// The neutral default band (`[5, 6.5, 8]`).
    pub fn default_band() -> Self {
        Self::from_ideal(DEFAULT_ATTACK_SPACING_IDEAL_YARDS)
    }

    /// Score a realized nearest-teammate distance against this band, in `[-1, 1]`, with
    /// the SAME shape as [`super::spacing_score_from_distance`]: full reward (`1.0`)
    /// inside `[min, ideal]`, a gentle taper to `0.55` out to `max`, and a negative
    /// ramp when crowded (`< min`, "occupying the same space") or isolated (`> max`,
    /// "empty space a teammate should fill").
    pub fn spacing_score(&self, distance_yards: f64) -> f64 {
        if !distance_yards.is_finite() {
            return -1.0;
        }
        let (min, ideal, max) = (self.min, self.ideal, self.max);
        if distance_yards < min {
            -((min - distance_yards) / min.max(1.0)).clamp(0.0, 1.0)
        } else if distance_yards > max {
            -((distance_yards - max) / max.max(1.0)).clamp(0.0, 1.0)
        } else if distance_yards <= ideal {
            1.0
        } else {
            let span = (max - ideal).max(1e-6);
            (1.0 - ((distance_yards - ideal) / span) * 0.45).clamp(0.55, 1.0)
        }
    }
}

/// Deterministic, RNG-free analytic seed: the *ideal* nearest-teammate spacing in yards
/// as a closed-form 5-8yd function of the context. This is the live fallback and the
/// bootstrap target a fresh [`AttackSpacingHead`] imitates, then beats.
///
/// Seed reasoning (kept modest so a neutral state stays ~6.5 and the served band ~5-8):
/// * **too narrow** (`width_shortage`) ⇒ spread wider — fill the unused width;
/// * **compact deep block** (small `def_spread`, low / negative `def_centroid_fwd`) ⇒
///   stretch wider to pull the block open;
/// * **locally crowded** (`local_teammate_density` high) ⇒ tighten is wrong — push the
///   target wider so the crowded player vacates into space ("don't occupy same space");
/// * **opponent stepping up hard** (`def_mean_vel_fwd` high) ⇒ tighten for quick, short
///   give-and-go combinations to play through the press;
/// * **carrier driving forward** (`carrier_vel_fwd` high) ⇒ slightly wider so runners
///   stretch the line ahead of the ball.
pub fn analytic_target_spacing(ctx: &AttackSpacingContext) -> f64 {
    let mut ideal = DEFAULT_ATTACK_SPACING_IDEAL_YARDS;

    // Too narrow ⇒ widen to fill the unused width.
    ideal += ctx.width_shortage.clamp(0.0, 1.0) * 1.6;

    // Compact, deep block ⇒ widen to stretch it. `def_spread` small and the block not
    // pressing high (centroid near/behind the ball) is the classic low block.
    let block_compactness = (1.0 - ctx.def_spread_yards / REF_SPREAD_YARDS).clamp(0.0, 1.0);
    let block_deep = (1.0 - (ctx.def_centroid_fwd_from_ball / REF_FWD_YARDS).clamp(0.0, 1.0))
        .clamp(0.0, 1.0);
    ideal += block_compactness * block_deep * 1.2;

    // Locally crowded ⇒ widen the target so the crowded player vacates into space.
    let crowd_excess = ((ctx.local_teammate_density - 1.0) / REF_DENSITY).clamp(0.0, 1.0);
    ideal += crowd_excess * 1.0;

    // Carrier driving forward ⇒ slightly wider to stretch ahead of the ball.
    ideal += (ctx.carrier_vel_fwd / REF_VEL_YPS).clamp(0.0, 1.0) * 0.5;

    // Opponent stepping up hard ⇒ tighten for quick short combinations through the press.
    ideal -= (ctx.def_mean_vel_fwd / REF_VEL_YPS).clamp(0.0, 1.0) * 0.9;

    ideal.clamp(ATTACK_SPACING_ABS_MIN_YARDS, ATTACK_SPACING_ABS_MAX_YARDS)
}

/// Resolve the served target band for a context: the trained head's prediction when one
/// is present and finite, otherwise the analytic seed. Centralized so every live
/// chokepoint (the off-ball ranker + the formation LP) reads an identical target.
pub fn resolve_attack_spacing_target(
    ctx: &AttackSpacingContext,
    head: Option<&AttackSpacingHead>,
) -> AttackSpacingTarget {
    let ideal = head
        .and_then(|h| h.predict(ctx))
        .filter(|v| v.is_finite())
        .unwrap_or_else(|| analytic_target_spacing(ctx));
    AttackSpacingTarget::from_ideal(ideal)
}

/// One self-play training row: the context of a sampled spacing decision and the
/// territorial reward attributed to the spacing held over the following window (higher
/// = the spacing improved our control of valuable space). The *target* the head
/// regresses is the realized good spacing — the nearest-teammate distance actually held
/// when the reward was positive — so the head learns the spacing that pays off. Reward
/// source: the pitch-control x xT territorial advantage delta (see `pitch_value`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttackSpacingSample {
    pub context: AttackSpacingContext,
    /// The nearest-teammate spacing actually held at the decision (the regression target).
    pub held_spacing_yards: f64,
    /// Windowed territorial-advantage change (the reward weight on this row).
    pub reward: f64,
}

/// An open spacing decision awaiting its windowed reward. Recorded with the team's
/// territorial advantage and the spacing held at decision time; resolved a fixed window
/// later by differencing into an [`AttackSpacingSample`]'s reward.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PendingAttackSpacingDecision {
    pub team: Team,
    pub context: AttackSpacingContext,
    pub held_spacing_yards: f64,
    pub decision_territorial: f64,
    pub due_tick: u64,
}

/// Learned regression head for the ideal spacing: a `FeedForwardNetwork`
/// (`DIM -> hidden -> 1`, linear output) predicting the ideal nearest-teammate spacing in
/// yards. Like the other learned heads it is not serde; it round-trips through the
/// engine's neural-network snapshot Postgres path. The live chokepoints read the
/// analytic seed until a trained head is promoted.
#[derive(Clone, Debug)]
pub struct AttackSpacingHead {
    network: FeedForwardNetwork,
    training_steps: usize,
    last_loss: Option<f64>,
}

impl AttackSpacingHead {
    pub fn new(seed: u32) -> Self {
        let mut rng = mulberry32(seed ^ 0x2C9E_7F11);
        let network = FeedForwardNetwork::random(
            &RandomNetworkSpec {
                input_dim: ATTACK_SPACING_FEATURE_DIM,
                hidden_layers: vec![ATTACK_SPACING_HIDDEN_UNITS],
                output_dim: 1,
                hidden_activation: ActivationName::Tanh,
                // Linear output: a spacing in yards (not a probability).
                output_activation: ActivationName::Linear,
                weight_scale: None,
            },
            &mut rng,
        );
        AttackSpacingHead {
            network,
            training_steps: 0,
            last_loss: None,
        }
    }

    /// Predicted ideal spacing (yards) for a context, clamped to the legal range, or
    /// `None` on a malformed (mis-dimensioned / non-finite) input.
    pub fn predict(&self, ctx: &AttackSpacingContext) -> Option<f64> {
        let x = ctx.to_features();
        if x.iter().any(|v| !v.is_finite()) {
            return None;
        }
        self.network
            .predict(&x[..])
            .first()
            .copied()
            .filter(|p| p.is_finite())
            .map(|p| p.clamp(ATTACK_SPACING_ABS_MIN_YARDS, ATTACK_SPACING_ABS_MAX_YARDS))
    }

    /// One SGD epoch regressing the head toward each row's held spacing, **weighted by
    /// the row's territorial reward** so spacings that paid off are reinforced and ones
    /// that lost ground are pushed away. Rows with non-positive reward contribute a
    /// repelling target (nudged toward the neutral default) at reduced weight. Returns
    /// the mean step loss.
    pub fn train(&mut self, samples: &[AttackSpacingSample], learning_rate: f64) -> f64 {
        let mut total = 0.0;
        let mut applied = 0usize;
        for s in samples {
            let x = s.context.to_features();
            if x.iter().any(|v| !v.is_finite())
                || !s.reward.is_finite()
                || !s.held_spacing_yards.is_finite()
            {
                continue;
            }
            // Positive reward ⇒ imitate the held spacing; non-positive ⇒ regress toward
            // the neutral default (held spacing did not earn territory), at lower weight.
            let (target_spacing, weight) = if s.reward > 0.0 {
                (s.held_spacing_yards, s.reward.min(4.0))
            } else {
                (DEFAULT_ATTACK_SPACING_IDEAL_YARDS, (-s.reward).min(4.0) * 0.25)
            };
            if weight <= 0.0 {
                continue;
            }
            let target = [target_spacing
                .clamp(ATTACK_SPACING_ABS_MIN_YARDS, ATTACK_SPACING_ABS_MAX_YARDS)];
            let result = self
                .network
                .train_sample_clipped(&x[..], &target, learning_rate * weight, 4.0);
            if result.applied && result.loss.is_finite() {
                total += result.loss;
                applied += 1;
                self.training_steps += 1;
            }
        }
        let mean = if applied > 0 { total / applied as f64 } else { 0.0 };
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

/// Summary of an attacking-spacing training pass (logged by the learner).
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct AttackSpacingTrainingReport {
    pub samples: usize,
    pub training_steps: usize,
    pub epochs: usize,
    pub final_loss: f64,
}

/// Train a fresh attacking-spacing head over an RL corpus. Returns `None` on an
/// empty/unusable corpus (no finite, positively-usable rows) or `epochs == 0`, so the
/// learner can skip cleanly.
pub fn train_attack_spacing_head(
    samples: &[AttackSpacingSample],
    seed: u32,
    epochs: usize,
    learning_rate: f64,
) -> Option<(AttackSpacingHead, AttackSpacingTrainingReport)> {
    let usable = samples
        .iter()
        .filter(|s| s.reward.is_finite() && s.held_spacing_yards.is_finite())
        .count();
    if usable == 0 || epochs == 0 {
        return None;
    }
    let mut head = AttackSpacingHead::new(seed);
    let mut final_loss = 0.0;
    for _ in 0..epochs {
        final_loss = head.train(samples, learning_rate);
    }
    let report = AttackSpacingTrainingReport {
        samples: usable,
        training_steps: head.training_steps(),
        epochs,
        final_loss,
    };
    Some((head, report))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A neutral mid-pitch in-possession state to perturb from.
    fn neutral_context() -> AttackSpacingContext {
        AttackSpacingContext {
            centroid_fwd_from_ball: 0.0,
            centroid_lat_from_ball: 0.0,
            self_fwd_from_centroid: 0.0,
            ball_progress: 0.5,
            behind_ball_yards: 0.0,
            def_centroid_fwd_from_ball: 12.0,
            def_spread_yards: REF_SPREAD_YARDS,
            def_mean_vel_fwd: 0.0,
            def_mean_vel_lat: 0.0,
            carrier_vel_fwd: 0.0,
            carrier_vel_lat: 0.0,
            local_teammate_density: 1.0,
            nearest_teammate_yards: 6.5,
            width_shortage: 0.0,
        }
    }

    #[test]
    fn neutral_seed_is_the_user_default_band_5_to_8() {
        let ctx = neutral_context();
        let ideal = analytic_target_spacing(&ctx);
        assert!(
            (ideal - DEFAULT_ATTACK_SPACING_IDEAL_YARDS).abs() < 1e-9,
            "neutral seed ideal should be the 6.5yd default, got {ideal}"
        );
        let band = AttackSpacingTarget::from_ideal(ideal);
        assert!((band.min - 5.0).abs() < 1e-9, "min should be 5, got {}", band.min);
        assert!((band.max - 8.0).abs() < 1e-9, "max should be 8, got {}", band.max);
        assert_eq!(ctx.to_features().len(), ATTACK_SPACING_FEATURE_DIM);
    }

    #[test]
    fn too_narrow_widens_and_a_high_press_tightens() {
        let mut narrow = neutral_context();
        narrow.width_shortage = 1.0;
        assert!(
            analytic_target_spacing(&narrow) > analytic_target_spacing(&neutral_context()) + 0.5,
            "a far-too-narrow team should widen its target spacing"
        );

        let mut pressing = neutral_context();
        pressing.def_mean_vel_fwd = REF_VEL_YPS; // block stepping up hard
        assert!(
            analytic_target_spacing(&pressing) < analytic_target_spacing(&neutral_context()),
            "a hard step-up press should tighten spacing for quick short combinations"
        );
    }

    #[test]
    fn seed_stays_inside_the_legal_absolute_range() {
        // Drive every widening term to its max and confirm the clamp holds.
        let mut extreme = neutral_context();
        extreme.width_shortage = 5.0;
        extreme.def_spread_yards = 0.0;
        extreme.def_centroid_fwd_from_ball = -40.0;
        extreme.local_teammate_density = 20.0;
        extreme.carrier_vel_fwd = 40.0;
        let ideal = analytic_target_spacing(&extreme);
        assert!(
            (ATTACK_SPACING_ABS_MIN_YARDS..=ATTACK_SPACING_ABS_MAX_YARDS).contains(&ideal),
            "ideal {ideal} must stay within the legal range"
        );
    }

    #[test]
    fn spacing_score_rewards_in_band_and_punishes_crowd_and_isolation() {
        let band = AttackSpacingTarget::default_band(); // [5, 6.5, 8]
        assert!((band.spacing_score(6.0) - 1.0).abs() < 1e-9, "in [min, ideal] is full reward");
        assert!(band.spacing_score(7.5) > 0.5 && band.spacing_score(7.5) < 1.0, "tapers above ideal");
        assert!(band.spacing_score(2.0) < 0.0, "crowding (occupying the same space) is punished");
        assert!(band.spacing_score(20.0) < 0.0, "isolation (an unfilled hole) is punished");
        assert!(
            band.spacing_score(2.0) < band.spacing_score(4.5),
            "the closer the crowd, the worse"
        );
        assert_eq!(band.spacing_score(f64::NAN), -1.0);
    }

    #[test]
    fn resolve_uses_head_when_present_else_seed() {
        let ctx = neutral_context();
        let seed_band = resolve_attack_spacing_target(&ctx, None);
        assert!((seed_band.ideal - DEFAULT_ATTACK_SPACING_IDEAL_YARDS).abs() < 1e-9);

        let head = AttackSpacingHead::new(11);
        let with_head = resolve_attack_spacing_target(&ctx, Some(&head));
        // A seeded (untrained) head still yields a finite, in-range band.
        assert!(
            (ATTACK_SPACING_ABS_MIN_YARDS..=ATTACK_SPACING_ABS_MAX_YARDS).contains(&with_head.ideal)
        );
    }

    #[test]
    fn head_trains_toward_rewarded_spacing() {
        let head = AttackSpacingHead::new(7);
        let ctx = neutral_context();
        assert!(head.predict(&ctx).is_some(), "a seeded head predicts a finite value");

        // A separable corpus: in this context, holding ~9yd earned territory (positive
        // reward); holding ~5yd lost it (negative). After training the head should pull
        // its prediction up toward the rewarded ~9yd.
        let before = AttackSpacingHead::new(7).predict(&ctx).unwrap();
        let samples: Vec<AttackSpacingSample> = (0..96)
            .map(|i| {
                if i % 2 == 0 {
                    AttackSpacingSample { context: ctx, held_spacing_yards: 9.0, reward: 1.0 }
                } else {
                    AttackSpacingSample { context: ctx, held_spacing_yards: 5.0, reward: -1.0 }
                }
            })
            .collect();
        let (trained, report) =
            train_attack_spacing_head(&samples, 7, 24, 0.02).expect("trains a usable corpus");
        assert_eq!(report.samples, 96);
        assert!(report.training_steps > 0 && report.final_loss.is_finite());
        assert!(
            trained.predict(&ctx).unwrap() > before,
            "head should move toward the rewarded (wider) spacing: before={before}"
        );
    }

    #[test]
    fn training_is_a_noop_on_empty_or_nonfinite() {
        assert!(train_attack_spacing_head(&[], 1, 4, 0.02).is_none());
        let mut head = AttackSpacingHead::new(3);
        let bad = vec![AttackSpacingSample {
            context: neutral_context(),
            held_spacing_yards: f64::NAN,
            reward: 1.0,
        }];
        assert_eq!(head.train(&bad, 0.05), 0.0);
        assert_eq!(head.training_steps(), 0);
    }
}
