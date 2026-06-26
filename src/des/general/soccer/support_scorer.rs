//! Learnable **off-ball support scorer** (the "move to space" candidate ranker).
//!
//! When a teammate does not have the ball, [`WorldSnapshot::open_space_for`] ranks a
//! grid of candidate destination points by a weighted sum of ~20 hand-tuned terms
//! (open space, lane clearance, spacing, width, ball-arrival, short-outlet, offside,
//! teammate-occupancy, …) and moves to the argmax. The *intent* (which off-ball action)
//! is already chosen by the neural policy; the *destination coordinate* is this fixed
//! weighted sum. This module is the seam that makes that coordinate **learnable**: each
//! candidate becomes a feature vector, and a small regression head predicts the value of
//! moving there (expected territorial / pitch-control reward), so the team can learn a
//! better "move to space" function than the hand-tuned weights — directly attacking the
//! "off-ball teammates spiral into the carrier instead of holding an open lane" failure.
//!
//! ## Design (mirrors [`super::back_four_line`])
//!
//! * [`SupportCandidateFeatures`] — the per-candidate feature vector. The exact ordering
//!   is the single source of truth ([`SupportCandidateFeatures::to_features`]); it is the
//!   SAME set of terms `open_space_for` already computes, captured as a struct so it can
//!   be logged as a training row, asserted in tests, and surfaced by the inspector.
//! * [`analytic_candidate_score`] — the deterministic, RNG-free seed = the current
//!   fixed-weight sum. This is the live fallback and the bootstrap target the learned head
//!   first imitates, then beats.
//! * [`SupportScorerHead`] — a `FeedForwardNetwork` (`DIM → hidden → 1`) regressing the
//!   candidate's windowed reward (a *value* over candidate features), trained from
//!   self-play [`SupportMoveSample`]s. The reward is the pitch-control × xT territorial
//!   advantage gained after the move (see `pitch_value`), which is exactly "did moving
//!   here improve our control of valuable space".
//!
//! ## Parity
//!
//! Gated **off** by default ([`support_scorer_enabled`] /
//! `DD_SOCCER_ENABLE_LEARNED_SUPPORT_SCORER`). The live chokepoint keeps reading the
//! analytic seed (== the existing sum) until a trained head is promoted, so an
//! unconfigured process is byte-identical. Feature build, analytic score, and the head's
//! `predict`/`train` are pure and RNG-free at call sites, so tests and the inspector can
//! read them without enabling the live seam.

use super::*;
use crate::des::general::neural_network::{ActivationName, FeedForwardNetwork, RandomNetworkSpec};
use crate::des::general::prng::mulberry32;

/// Number of features in the off-ball support candidate vector. The ordering is defined
/// once in [`SupportCandidateFeatures::to_features`] — change it and bump this constant and
/// any persisted head.
pub const SUPPORT_CANDIDATE_FEATURE_DIM: usize = 22;
/// Hidden width of the learned regression head.
pub const SUPPORT_SCORER_HIDDEN_UNITS: usize = 24;
/// Reference score magnitude used to normalize the (already-weighted) candidate terms into
/// a roughly unit range for the network input.
const SUPPORT_TERM_REF: f64 = 8.0;

const SUPPORT_SCORER_ENABLE_ENV: &str = "DD_SOCCER_ENABLE_LEARNED_SUPPORT_SCORER";

/// Whether the learned support scorer is consulted at the live `open_space_for` chokepoint
/// this process. Off (unset) ⇒ the analytic seed (the existing fixed-weight sum) stands, so
/// the ranking is byte-identical.
pub fn support_scorer_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var(SUPPORT_SCORER_ENABLE_ENV).is_ok())
}

/// The per-candidate feature vector for one off-ball destination point. Every field is the
/// SAME (already sign- and weight-bearing) term that `open_space_for` sums today, so the
/// analytic seed is exactly reproducible. Stored raw (un-normalized) so a row is human- and
/// test-readable; [`Self::to_features`] does the normalization for the network.
#[derive(Clone, Debug, PartialEq)]
pub struct SupportCandidateFeatures {
    pub open_space_score: f64,
    pub counterattack_bonus: f64,
    pub goal_directness_bonus: f64,
    pub vacuum_run_bonus: f64,
    pub forward_term: f64,
    pub forward_from_ball_term: f64,
    pub role_depth_deviation_term: f64,
    pub defensive_midfield_screen_penalty: f64,
    pub distance_from_home_term: f64,
    pub lane_bonus: f64,
    pub diagonal_support_bonus: f64,
    pub spacing_bonus: f64,
    pub width_possession_bonus: f64,
    pub wingback_flank_opening_bonus: f64,
    pub dynamic_lane_fit_bonus: f64,
    pub ball_arrival_bonus: f64,
    pub short_outlet_bonus: f64,
    pub off_ball_space_discipline: f64,
    pub offside_penalty: f64,
    pub lane_position_penalty: f64,
    pub teammate_occupation_penalty: f64,
    pub line_shape_penalty: f64,
}

impl SupportCandidateFeatures {
    /// The analytic seed score: the exact fixed-weight sum `open_space_for` uses today
    /// (penalties already carry their sign as stored). This is the live fallback and the
    /// value a fresh head is bootstrapped toward.
    pub fn analytic_score(&self) -> f64 {
        self.open_space_score
            + self.counterattack_bonus
            + self.goal_directness_bonus
            + self.vacuum_run_bonus
            + self.forward_term
            + self.forward_from_ball_term
            + self.role_depth_deviation_term
            - self.defensive_midfield_screen_penalty
            + self.distance_from_home_term
            + self.lane_bonus
            + self.diagonal_support_bonus
            + self.spacing_bonus
            + self.width_possession_bonus
            + self.wingback_flank_opening_bonus
            + self.dynamic_lane_fit_bonus
            + self.ball_arrival_bonus
            + self.short_outlet_bonus
            + self.off_ball_space_discipline
            - self.offside_penalty
            - self.lane_position_penalty
            - self.teammate_occupation_penalty
            - self.line_shape_penalty
    }

    /// The normalized network input vector. **This ordering is the single source of truth**
    /// for the feature layout — change it and bump [`SUPPORT_CANDIDATE_FEATURE_DIM`] and any
    /// persisted head.
    pub fn to_features(&self) -> [f64; SUPPORT_CANDIDATE_FEATURE_DIM] {
        let n = |v: f64| (v / SUPPORT_TERM_REF).clamp(-4.0, 4.0);
        [
            n(self.open_space_score),
            n(self.counterattack_bonus),
            n(self.goal_directness_bonus),
            n(self.vacuum_run_bonus),
            n(self.forward_term),
            n(self.forward_from_ball_term),
            n(self.role_depth_deviation_term),
            n(self.defensive_midfield_screen_penalty),
            n(self.distance_from_home_term),
            n(self.lane_bonus),
            n(self.diagonal_support_bonus),
            n(self.spacing_bonus),
            n(self.width_possession_bonus),
            n(self.wingback_flank_opening_bonus),
            n(self.dynamic_lane_fit_bonus),
            n(self.ball_arrival_bonus),
            n(self.short_outlet_bonus),
            n(self.off_ball_space_discipline),
            n(self.offside_penalty),
            n(self.lane_position_penalty),
            n(self.teammate_occupation_penalty),
            n(self.line_shape_penalty),
        ]
    }
}

/// Deterministic, RNG-free analytic seed score for a candidate (the current fixed-weight
/// sum). Live fallback + bootstrap target for [`SupportScorerHead`].
pub fn analytic_candidate_score(features: &SupportCandidateFeatures) -> f64 {
    features.analytic_score()
}

/// One self-play training row: the features of a candidate that was MOVED TO, and the
/// territorial reward attributed to that move over the following window (higher = the move
/// improved our pitch control / advanced the play). Collected during self-play, drained to
/// the learner, consumed by [`SupportScorerHead::train`]. The reward source is the
/// pitch-control × xT territorial advantage delta (see `pitch_value`).
#[derive(Clone, Debug, PartialEq)]
pub struct SupportMoveSample {
    pub features: SupportCandidateFeatures,
    pub reward: f64,
}

/// An open support-move decision awaiting its windowed reward. Recorded at the move with the
/// team's territorial advantage then; resolved a fixed window later by differencing into a
/// [`SupportMoveSample`]'s reward.
#[derive(Clone, Debug, PartialEq)]
pub struct PendingSupportDecision {
    pub team: Team,
    pub features: SupportCandidateFeatures,
    pub decision_territorial: f64,
    pub due_tick: u64,
}

/// How often (ticks) an off-ball support move is sampled for RL while the scorer is on.
pub const SUPPORT_MOVE_SAMPLE_INTERVAL_TICKS: u64 = 15;
/// Window (ticks) over which a sampled move's territorial reward is measured.
pub const SUPPORT_MOVE_REWARD_WINDOW_TICKS: u64 = 45;
/// Cap on the rolling RL sample buffer (drained by the learner).
pub const SUPPORT_MOVE_SAMPLE_CAP: usize = 4096;

/// Learned regression head for the candidate value: a `FeedForwardNetwork`
/// (`DIM → hidden → 1`, linear output) predicting the expected territorial reward of moving
/// to a candidate. Like the other learned heads it is not serde; it round-trips through the
/// engine's neural-network snapshot Postgres path. The live chokepoint reads the analytic
/// seed until a trained head is promoted.
#[derive(Clone, Debug)]
pub struct SupportScorerHead {
    network: FeedForwardNetwork,
    training_steps: usize,
    last_loss: Option<f64>,
}

impl SupportScorerHead {
    pub fn new(seed: u32) -> Self {
        let mut rng = mulberry32(seed ^ 0x51A7_0C3B);
        let network = FeedForwardNetwork::random(
            &RandomNetworkSpec {
                input_dim: SUPPORT_CANDIDATE_FEATURE_DIM,
                hidden_layers: vec![SUPPORT_SCORER_HIDDEN_UNITS],
                output_dim: 1,
                hidden_activation: ActivationName::Tanh,
                // Linear output: a raw candidate score / value (not a probability).
                output_activation: ActivationName::Linear,
                weight_scale: None,
            },
            &mut rng,
        );
        SupportScorerHead {
            network,
            training_steps: 0,
            last_loss: None,
        }
    }

    /// Predicted candidate value for a feature vector, or `None` on a malformed
    /// (mis-dimensioned / non-finite) input.
    pub fn predict(&self, features: &SupportCandidateFeatures) -> Option<f64> {
        let x = features.to_features();
        if x.iter().any(|v| !v.is_finite()) {
            return None;
        }
        self.network
            .predict(&x[..])
            .first()
            .copied()
            .filter(|p| p.is_finite())
    }

    /// One SGD epoch regressing the head toward `(features, target_value)` rows — either a
    /// supervised bootstrap toward the analytic seed or an RL territorial-reward target.
    /// Returns the mean step loss.
    pub fn train(&mut self, samples: &[SupportMoveSample], learning_rate: f64) -> f64 {
        let mut total = 0.0;
        let mut applied = 0usize;
        for s in samples {
            let x = s.features.to_features();
            if x.iter().any(|v| !v.is_finite()) || !s.reward.is_finite() {
                continue;
            }
            let target = [s.reward];
            let result = self
                .network
                .train_sample_clipped(&x[..], &target, learning_rate, 4.0);
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

/// Summary of a support-scorer training pass (logged by the learner).
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct SupportScorerTrainingReport {
    pub samples: usize,
    pub training_steps: usize,
    pub epochs: usize,
    pub final_loss: f64,
}

/// Train a fresh support-scorer head over an RL corpus. Returns `None` on an empty/unusable
/// corpus (no finite rewards) or `epochs == 0`, so the learner can skip cleanly.
pub fn train_support_scorer_head(
    samples: &[SupportMoveSample],
    seed: u32,
    epochs: usize,
    learning_rate: f64,
) -> Option<(SupportScorerHead, SupportScorerTrainingReport)> {
    let usable = samples.iter().filter(|s| s.reward.is_finite()).count();
    if usable == 0 || epochs == 0 {
        return None;
    }
    let mut head = SupportScorerHead::new(seed);
    let mut final_loss = 0.0;
    for _ in 0..epochs {
        final_loss = head.train(samples, learning_rate);
    }
    let report = SupportScorerTrainingReport {
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

    fn baseline_features() -> SupportCandidateFeatures {
        SupportCandidateFeatures {
            open_space_score: 6.0,
            counterattack_bonus: 0.0,
            goal_directness_bonus: 0.0,
            vacuum_run_bonus: 0.0,
            forward_term: 0.4,
            forward_from_ball_term: 1.2,
            role_depth_deviation_term: -0.3,
            defensive_midfield_screen_penalty: 0.0,
            distance_from_home_term: -0.1,
            lane_bonus: 1.8,
            diagonal_support_bonus: 0.0,
            spacing_bonus: 0.5,
            width_possession_bonus: 2.0,
            wingback_flank_opening_bonus: 0.0,
            dynamic_lane_fit_bonus: 0.3,
            ball_arrival_bonus: 1.0,
            short_outlet_bonus: 2.5,
            off_ball_space_discipline: 0.0,
            offside_penalty: 0.0,
            lane_position_penalty: 0.2,
            teammate_occupation_penalty: 0.0,
            line_shape_penalty: 0.0,
        }
    }

    #[test]
    fn analytic_score_matches_the_explicit_term_sum() {
        let f = baseline_features();
        // The signed sum the live chokepoint computes today.
        let expected = 6.0 + 0.4 + 1.2 - 0.3 - 0.1 + 1.8 + 0.5 + 2.0 + 0.3 + 1.0 + 2.5 - 0.2;
        assert!((analytic_candidate_score(&f) - expected).abs() < 1e-9);
        assert_eq!(f.to_features().len(), SUPPORT_CANDIDATE_FEATURE_DIM);
    }

    #[test]
    fn head_predicts_finite_and_trains_toward_reward() {
        let head = SupportScorerHead::new(7);
        let f = baseline_features();
        assert!(
            head.predict(&f).is_some(),
            "a seeded head predicts a finite value"
        );

        // A separable corpus: open space ⇒ high reward, crowded ⇒ low. After training the
        // head should rank the open candidate strictly above the crowded one.
        let mut open = baseline_features();
        open.open_space_score = 9.0;
        let mut crowded = baseline_features();
        crowded.open_space_score = -3.0;
        crowded.teammate_occupation_penalty = 8.0;
        let samples: Vec<SupportMoveSample> = (0..64)
            .map(|i| {
                if i % 2 == 0 {
                    SupportMoveSample {
                        features: open.clone(),
                        reward: 1.0,
                    }
                } else {
                    SupportMoveSample {
                        features: crowded.clone(),
                        reward: -1.0,
                    }
                }
            })
            .collect();
        let (trained, report) =
            train_support_scorer_head(&samples, 7, 12, 0.05).expect("trains a usable corpus");
        assert_eq!(report.samples, 64);
        assert!(report.training_steps > 0 && report.final_loss.is_finite());
        assert!(
            trained.predict(&open).unwrap() > trained.predict(&crowded).unwrap(),
            "trained head should value open space above a crowded, teammate-occupied pocket"
        );
    }

    #[test]
    fn training_is_a_noop_on_empty_or_nonfinite() {
        assert!(train_support_scorer_head(&[], 1, 4, 0.02).is_none());
        let mut head = SupportScorerHead::new(3);
        let bad = vec![SupportMoveSample {
            features: baseline_features(),
            reward: f64::NAN,
        }];
        assert_eq!(head.train(&bad, 0.05), 0.0);
        assert_eq!(head.training_steps(), 0);
    }
}
