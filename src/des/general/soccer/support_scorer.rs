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
pub const SUPPORT_CANDIDATE_FEATURE_DIM: usize = 25;
/// Hidden width of the learned regression head.
pub const SUPPORT_SCORER_HIDDEN_UNITS: usize = 24;
/// Reference score magnitude used to normalize the (already-weighted) candidate terms into
/// a roughly unit range for the network input.
const SUPPORT_TERM_REF: f64 = 8.0;

const SUPPORT_SCORER_ENABLE_ENV: &str = "DD_SOCCER_ENABLE_LEARNED_SUPPORT_SCORER";
const SUPPORT_SCORER_DISABLE_ENV: &str = "DD_SOCCER_DISABLE_LEARNED_SUPPORT_SCORER";

/// Opt-in: fold discrete OUTCOME events (turnover / completed forward pass / shot-on-target /
/// goal) that occur inside a support move's reward window into that move's training reward,
/// on top of the dense territorial (pitch-control × xT) delta. Off (unset) ⇒ the reward is the
/// pure territorial delta exactly as before, so the sample stream is byte-identical and this is
/// a no-op. This is the seam that teaches the off-ball "where to move" decision from what the
/// move actually *led to*, not just the space it gained. A/B-gated like the other learned heads.
const SUPPORT_OUTCOME_REWARD_ENABLE_ENV: &str = "DD_SOCCER_ENABLE_SUPPORT_OUTCOME_REWARD";
/// Env override for the outcome-blend coefficient λ ([`support_outcome_reward_weight`]).
const SUPPORT_OUTCOME_REWARD_WEIGHT_ENV: &str = "DD_SOCCER_SUPPORT_OUTCOME_REWARD_WEIGHT";

/// Whether outcome events are folded into the support-move reward this process. Off by default;
/// requires the scorer itself to be on to have any effect (it feeds the same sample stream).
pub fn support_outcome_reward_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| support_env_flag(SUPPORT_OUTCOME_REWARD_ENABLE_ENV).unwrap_or(false))
}

/// Blend coefficient λ applied to a decision's accumulated outcome term before it is added to the
/// territorial delta: `reward = territorial_delta + λ · outcome_accumulator`. The outcome term is
/// bounded per [`support_outcome_event_weight`] (a goal contributes ≈ ±1.0 × team weight), so λ
/// sets outcome shaping on the same order as the territorial base. Tunable via
/// `DD_SOCCER_SUPPORT_OUTCOME_REWARD_WEIGHT` for A/B sweeps; the default is a deliberately
/// conservative nudge (territorial stays the dominant dense signal).
pub fn support_outcome_reward_weight() -> f64 {
    use std::sync::OnceLock;
    static WEIGHT: OnceLock<f64> = OnceLock::new();
    *WEIGHT.get_or_init(|| {
        std::env::var(SUPPORT_OUTCOME_REWARD_WEIGHT_ENV)
            .ok()
            .and_then(|raw| raw.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(SUPPORT_OUTCOME_REWARD_WEIGHT_DEFAULT)
    })
}

/// Default λ for the outcome blend (see [`support_outcome_reward_weight`]).
pub const SUPPORT_OUTCOME_REWARD_WEIGHT_DEFAULT: f64 = 0.05;
/// A teammate's outcome event is credited to *this* off-ball mover's decision at this discount
/// (the mover's OWN events count at full strength). Mirrors the recency/share philosophy of
/// `BuildupChainCredit` — the whole in-possession unit shares credit for what the move enabled,
/// but the deciding player owns their own touch outright.
pub const SUPPORT_OUTCOME_TEAMMATE_DISCOUNT: f64 = 0.35;

/// The mover-weighted + team-discounted contribution of a single outcome event (already reduced to
/// its emitting player + team + normalized `weight`) to one pending move decision. Pure core of the
/// credit model, factored out so the weighting is unit-testable without a live match: the deciding
/// player's own event counts full; a teammate's counts at [`SUPPORT_OUTCOME_TEAMMATE_DISCOUNT`]; an
/// opponent's counts 0 (a move is trained only from what its OWN team's play led to).
pub(crate) fn support_outcome_decision_delta(
    decision_team: Team,
    decision_player: usize,
    event_team: Team,
    event_player: usize,
    weight: f64,
) -> f64 {
    if event_team != decision_team {
        return 0.0;
    }
    let factor = if event_player == decision_player {
        1.0
    } else {
        SUPPORT_OUTCOME_TEAMMATE_DISCOUNT
    };
    weight * factor
}

/// Normalized, signed contribution of one reward-event `kind` to an off-ball move's outcome term.
/// Decoupled from the engine's internal reward magnitudes (which live on a different scale than the
/// territorial delta) so the blend is bounded and interpretable: a goal ≈ +1.0, a completed forward
/// pass ≈ +0.2, a turnover ≈ −0.6. Kinds not tied to a move's downstream outcome return 0.0.
pub(crate) fn support_outcome_event_weight(kind: SoccerRewardEventKind) -> f64 {
    match kind {
        // Terminal / near-terminal attacking payoff.
        SoccerRewardEventKind::Goal | SoccerRewardEventKind::HeaderGoalFromCross => 1.0,
        SoccerRewardEventKind::ShotOnTarget => 0.4,
        SoccerRewardEventKind::ShotAttempt => 0.12,
        // Successful forward progression / retention of a threatening move.
        SoccerRewardEventKind::ThreePassForwardNetGain => 0.28,
        SoccerRewardEventKind::WallPassCombination => 0.22,
        SoccerRewardEventKind::BuildupChainCredit => 0.20,
        SoccerRewardEventKind::TwoForwardPasses => 0.18,
        SoccerRewardEventKind::ReleaseLongInsideOwnHalf => 0.18,
        SoccerRewardEventKind::ProgressiveCarryIntoAttack => 0.16,
        // Turnovers / move-killing blunders (the penalties the move should learn to avoid setting up).
        SoccerRewardEventKind::OverdribbleDispossession => -0.6,
        SoccerRewardEventKind::BadPassChainPenalty => -0.5,
        SoccerRewardEventKind::TurnoverChainBlame => -0.5,
        SoccerRewardEventKind::OffsideInfraction => -0.35,
        SoccerRewardEventKind::IsolatedCarrierPanicBackPass => -0.3,
        SoccerRewardEventKind::UnpressuredBackwardPass => -0.2,
        SoccerRewardEventKind::ShotOffTargetPenalty => -0.15,
        _ => 0.0,
    }
}

fn support_env_flag(name: &str) -> Option<bool> {
    std::env::var(name).ok().map(|raw| {
        let v = raw.trim().to_ascii_lowercase();
        matches!(v.as_str(), "1" | "true" | "yes" | "on")
    })
}

fn support_scorer_enabled_from_env() -> bool {
    if support_env_flag(SUPPORT_SCORER_DISABLE_ENV).unwrap_or(false) {
        return false;
    }
    support_env_flag(SUPPORT_SCORER_ENABLE_ENV).unwrap_or(true)
}

/// Whether the learned support scorer is consulted at the live `open_space_for` chokepoint
/// this process. Production default is on (analytic seed until a warm head is installed);
/// tests default off so the broad movement suite keeps byte-identical parity unless it opts in.
pub fn support_scorer_enabled() -> bool {
    #[cfg(test)]
    {
        if support_env_flag(SUPPORT_SCORER_DISABLE_ENV).unwrap_or(false) {
            return false;
        }
        support_env_flag(SUPPORT_SCORER_ENABLE_ENV).unwrap_or(false)
    }
    #[cfg(not(test))]
    {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(support_scorer_enabled_from_env)
    }
}

/// The per-candidate feature vector for one off-ball destination point. Every field is the
/// SAME (already sign- and weight-bearing) term that `open_space_for` sums today, so the
/// analytic seed is exactly reproducible. Stored raw (un-normalized) so a row is human- and
/// test-readable; [`Self::to_features`] does the normalization for the network.
#[derive(Clone, Debug, PartialEq)]
pub struct SupportCandidateFeatures {
    pub open_space_score: f64,
    pub counterattack_bonus: f64,
    pub advance_upfield_bonus: f64,
    pub goal_directness_bonus: f64,
    pub vacuum_run_bonus: f64,
    pub forward_run_bias: f64,
    pub attacking_spacing_bonus: f64,
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
            + self.advance_upfield_bonus
            + self.goal_directness_bonus
            + self.vacuum_run_bonus
            + self.forward_run_bias
            + self.attacking_spacing_bonus
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
            n(self.advance_upfield_bonus),
            n(self.goal_directness_bonus),
            n(self.vacuum_run_bonus),
            n(self.forward_run_bias),
            n(self.attacking_spacing_bonus),
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
    /// The off-ball player who made this move decision. Used to credit their OWN outcome events at
    /// full strength (teammates' at a discount) when `DD_SOCCER_ENABLE_SUPPORT_OUTCOME_REWARD` is on.
    pub player_id: usize,
    pub features: SupportCandidateFeatures,
    pub decision_territorial: f64,
    pub due_tick: u64,
    /// Running sum of normalized, mover-/team-weighted outcome-event contributions observed inside
    /// this decision's reward window. Stays 0.0 (and unused) unless the outcome-reward seam is on,
    /// preserving byte-identical samples in the default configuration.
    pub outcome_accumulator: f64,
}

/// How often (ticks) an off-ball support move is sampled for RL while the scorer is on.
pub const SUPPORT_MOVE_SAMPLE_INTERVAL_TICKS: u64 = 15;
/// Window (ticks) over which a sampled move's territorial reward is measured.
pub const SUPPORT_MOVE_REWARD_WINDOW_TICKS: u64 = 45;
/// Cap on the rolling RL sample buffer (drained by the learner).
pub const SUPPORT_MOVE_SAMPLE_CAP: usize = 4096;
/// Minimum training steps before live support ranking trusts the carried head over the
/// analytic seed. Before that, the seam records samples but keeps the known heuristic.
pub const SUPPORT_SCORER_HEAD_MIN_TRAINING_STEPS: usize = 200;

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

    pub fn from_snapshot(snapshot: &SoccerAuxiliaryHeadSnapshot) -> Result<Self, String> {
        let network = build_soccer_feed_forward_network_from_snapshot(
            &snapshot.network,
            SUPPORT_CANDIDATE_FEATURE_DIM,
            1,
            &[],
            "support-scorer head",
        )?;
        Ok(SupportScorerHead {
            network,
            training_steps: snapshot.training_steps.max(snapshot.network.training_steps),
            last_loss: snapshot.average_loss.or(snapshot.network.average_loss),
        })
    }

    pub(crate) fn to_snapshot(&self) -> SoccerAuxiliaryHeadSnapshot {
        let mut network = soccer_neural_network_snapshot(&self.network);
        network.training_steps = self.training_steps;
        network.average_loss = self.last_loss;
        SoccerAuxiliaryHeadSnapshot {
            network,
            training_steps: self.training_steps,
            average_loss: self.last_loss,
        }
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
    use std::sync::Mutex;

    static SUPPORT_SCORER_ENV_LOCK: Mutex<()> = Mutex::new(());

    fn baseline_features() -> SupportCandidateFeatures {
        SupportCandidateFeatures {
            open_space_score: 6.0,
            counterattack_bonus: 0.0,
            advance_upfield_bonus: 0.0,
            goal_directness_bonus: 0.0,
            vacuum_run_bonus: 0.0,
            forward_run_bias: 0.0,
            attacking_spacing_bonus: 0.0,
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

    #[test]
    fn installed_warm_head_is_consumed_by_snapshot_scorer() {
        let _lock = SUPPORT_SCORER_ENV_LOCK
            .lock()
            .expect("support scorer env lock");
        std::env::set_var(SUPPORT_SCORER_ENABLE_ENV, "1");
        std::env::remove_var(SUPPORT_SCORER_DISABLE_ENV);

        let features = baseline_features();
        let analytic = analytic_candidate_score(&features);
        let mut sim = SoccerMatch::default_11v11(MatchConfig::default());
        let cold_snapshot = WorldSnapshot::from_match(&sim);
        assert!(
            (cold_snapshot.support_candidate_score(&features) - analytic).abs() < 1e-9,
            "no installed head should fall back to analytic seed"
        );

        let samples: Vec<SupportMoveSample> = (0..SUPPORT_SCORER_HEAD_MIN_TRAINING_STEPS)
            .map(|_| SupportMoveSample {
                features: features.clone(),
                reward: analytic + 6.0,
            })
            .collect();
        let mut head = SupportScorerHead::new(19);
        for _ in 0..6 {
            head.train(&samples, 0.05);
        }
        assert!(head.training_steps() >= SUPPORT_SCORER_HEAD_MIN_TRAINING_STEPS);

        sim.set_support_scorer_head(head);
        let learned_snapshot = WorldSnapshot::from_match(&sim);
        let learned = learned_snapshot.support_candidate_score(&features);
        std::env::remove_var(SUPPORT_SCORER_ENABLE_ENV);
        assert!(
            (learned - analytic).abs() > 0.10,
            "warm installed head should replace the analytic support score: analytic={analytic} learned={learned}"
        );
    }

    #[test]
    fn support_scorer_head_round_trips_through_snapshot() {
        let features = baseline_features();
        let mut head = SupportScorerHead::new(23);
        let samples = vec![SupportMoveSample {
            features: features.clone(),
            reward: 1.0,
        }];
        head.train(&samples, 0.02);
        let before = head.predict(&features).expect("prediction before snapshot");
        let snapshot = head.to_snapshot();
        let restored = SupportScorerHead::from_snapshot(&snapshot).expect("restore snapshot");
        let after = restored
            .predict(&features)
            .expect("prediction after snapshot");
        assert_eq!(restored.training_steps(), head.training_steps());
        assert!(
            (before - after).abs() < 1e-9,
            "snapshot should preserve support-scorer prediction: before={before} after={after}"
        );
    }

    #[test]
    fn outcome_event_weights_rank_goal_over_pass_over_zero_over_turnover() {
        // The ladder the off-ball move learns from: scoring ≫ a completed forward pass > neutral,
        // and turnovers are strictly negative. Neutral/unrelated kinds contribute nothing.
        let goal = support_outcome_event_weight(SoccerRewardEventKind::Goal);
        let shot = support_outcome_event_weight(SoccerRewardEventKind::ShotOnTarget);
        let pass = support_outcome_event_weight(SoccerRewardEventKind::TwoForwardPasses);
        let turnover =
            support_outcome_event_weight(SoccerRewardEventKind::OverdribbleDispossession);
        assert!(goal > shot && shot > pass && pass > 0.0);
        assert!(turnover < 0.0);
        assert_eq!(
            support_outcome_event_weight(SoccerRewardEventKind::HeaderGoalFromCross),
            goal,
            "a headed goal from a cross is still a goal"
        );
        assert_eq!(
            support_outcome_event_weight(SoccerRewardEventKind::Routine),
            0.0,
            "routine/unrelated events must not shape the move reward"
        );
    }

    #[test]
    fn outcome_credit_is_mover_full_teammate_discounted_opponent_zero() {
        let w = 1.0;
        // The deciding player's own outcome: full weight.
        let own = support_outcome_decision_delta(Team::Home, 3, Team::Home, 3, w);
        // A teammate's outcome in the same window: discounted.
        let teammate = support_outcome_decision_delta(Team::Home, 3, Team::Home, 7, w);
        // An opponent's outcome: ignored entirely.
        let opponent = support_outcome_decision_delta(Team::Home, 3, Team::Away, 7, w);
        assert!((own - 1.0).abs() < 1e-9);
        assert!((teammate - SUPPORT_OUTCOME_TEAMMATE_DISCOUNT).abs() < 1e-9);
        assert_eq!(opponent, 0.0);
        assert!(own > teammate && teammate > 0.0);
        // A turnover (negative weight) flips sign but keeps the same mover/teammate ordering.
        let bad = support_outcome_event_weight(SoccerRewardEventKind::TurnoverChainBlame);
        let own_turnover = support_outcome_decision_delta(Team::Home, 3, Team::Home, 3, bad);
        assert!(own_turnover < 0.0);
    }

    #[test]
    fn outcome_reward_weight_default_is_a_bounded_nonnegative_nudge() {
        // No env override → the conservative default; territorial stays the dominant dense signal.
        let w = support_outcome_reward_weight();
        assert!(w.is_finite() && w >= 0.0);
        assert!(
            w <= 1.0,
            "default outcome blend must not swamp the territorial base: {w}"
        );
    }
}
