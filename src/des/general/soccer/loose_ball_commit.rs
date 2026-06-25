//! Learnable **loose-ball commit selector** — *which* player(s) should attack an
//! unpossessed ball to win it back, without the whole team collapsing onto it.
//!
//! When the ball is loose (a 50/50 duel, a deflection, a missed first touch) the
//! engine must pick the player(s) who break off to *attack the ball* and win
//! possession. The existing election ([`WorldSnapshot::loose_ball_retrieval_score`])
//! is a hand-tuned blend of distance, stride alignment, lane affinity, and receipt
//! shape. It says nothing about the thing a coach actually weighs: *is this the
//! player who can get there first WITHOUT being dragged out of a good
//! defensive/offensive formation position?* A centre-back that abandons the back
//! line to chase a 60/40 in midfield is the wrong man even if he is a yard closer.
//!
//! This module is the seam that makes that choice **learnable**, mirroring
//! [`super::back_four_line`]'s line-depth head:
//!
//!   * [`LooseBallCommitInputs`] — the per-candidate state the choice is a function
//!     of, in the candidate's attacking frame so it is mirror-invariant: the
//!     kinematic race to the projected loose-ball point (distance, arrival time,
//!     stride alignment), the 50/50 margin against the nearest opponent, the
//!     **formation-displacement cost** (how far the loose ball sits from the
//!     player's LP/IPM-assigned slot, vs. how far it may legally stray — the
//!     "won't be drawn out of shape" term), the territorial pitch-control value of
//!     the contest point, role, ball state, and the retained existing heuristic
//!     score (group 8: refine, never discard).
//!   * [`analytic_loose_ball_commit_priority`] — a deterministic, RNG-free seed
//!     mapping that state to a *commit priority* in `(0, 1)` (higher = this player
//!     should attack the ball). It is the bootstrap target and the live fallback.
//!   * [`LooseBallCommitHead`] — a `FeedForwardNetwork` (`DIM → hidden → 1`,
//!     sigmoid) that the cluster learner trains by **reward-weighted regression**
//!     toward the commit actions that actually won possession while keeping shape.
//!
//! The live seam is [`WorldSnapshot::loose_ball_commit_score_adjustment`], a signed
//! yards delta folded into [`WorldSnapshot::loose_ball_retrieval_score`]: a
//! high-priority candidate's retrieval score is lowered (it becomes the elected
//! attacker); a candidate the model judges would break shape is raised.
//!
//! ## Parity
//!
//! Gated **off** by default (`DD_SOCCER_ENABLE_LOOSE_BALL_COMMIT_MODEL`). When unset,
//! [`WorldSnapshot::loose_ball_commit_score_adjustment`] returns `0.0` before
//! building any features, so the retriever election — and therefore the whole
//! engine — is byte-identical to before this module existed. The feature build,
//! analytic seed, and head are pure / RNG-free, so the inspector and tests can read
//! the surface without enabling the live seam.

use super::*;
use crate::des::general::neural_network::{ActivationName, FeedForwardNetwork, RandomNetworkSpec};
use crate::des::general::prng::mulberry32;

/// Number of features in the loose-ball commit state vector. The exact ordering is
/// the single source of truth in [`LooseBallCommitInputs::to_features`].
pub const LOOSE_BALL_COMMIT_FEATURE_DIM: usize = 17;
/// Hidden width of the learned regression head.
pub const LOOSE_BALL_COMMIT_HIDDEN_UNITS: usize = 20;
/// Maximum signed swing (yards) the model may apply to a candidate's retrieval
/// score. A priority of `1.0` lowers the score (more likely to be elected) by this
/// much; `0.0` raises it by this much. Kept modest so the model *refines* the
/// well-tuned election rather than overturning it wholesale.
pub const LOOSE_BALL_COMMIT_SCORE_SWING_YARDS: f64 = 4.0;

/// Reference distance (yd) normalizing distance-to-ball features.
const REF_DISTANCE_YARDS: f64 = 30.0;
/// Reference time (s) normalizing arrival-time features.
const REF_ARRIVAL_SECONDS: f64 = 4.0;
/// Reference speed (yd/s) normalizing ball speed.
const REF_BALL_SPEED_YPS: f64 = 20.0;

/// Env gate enabling the learned commit selector at the retriever election. Off
/// (unset) by default so an unconfigured process stays byte-identical.
const LOOSE_BALL_COMMIT_MODEL_ENABLE_ENV: &str = "DD_SOCCER_ENABLE_LOOSE_BALL_COMMIT_MODEL";

fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|raw| {
            let v = raw.trim().to_ascii_lowercase();
            matches!(v.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

/// Whether the learned loose-ball commit selector is consulted at the election this
/// process. Off ⇒ the analytic retrieval election stands (parity).
pub fn loose_ball_commit_model_enabled() -> bool {
    #[cfg(test)]
    {
        env_flag_enabled(LOOSE_BALL_COMMIT_MODEL_ENABLE_ENV)
    }
    #[cfg(not(test))]
    {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| env_flag_enabled(LOOSE_BALL_COMMIT_MODEL_ENABLE_ENV))
    }
}

/// Raw (un-normalized) per-candidate state the commit selector is a function of,
/// captured in the candidate's **attacking frame** so the model is mirror-invariant.
/// A plain struct so it can be logged as a training row, asserted on in tests, and
/// surfaced by the inspector independently of the network.
#[derive(Clone, Debug, PartialEq)]
pub struct LooseBallCommitInputs {
    pub field_length: f64,
    pub field_width: f64,

    /// Distance (yd) from the candidate to the projected loose-ball point.
    pub dist_to_target: f64,
    /// Velocity-aware time (s) for the candidate to reach the point.
    pub arrival_time: f64,
    /// Stride alignment in `[0, 1]`: how well the candidate's current run already
    /// commits toward the point (`velocity · direction`, floored at 0).
    pub stride_alignment: f64,
    /// Best (smallest) arrival time among the candidate's TEAMMATES, excluding the
    /// candidate — so the model knows whether someone better placed exists.
    pub best_teammate_arrival: f64,
    /// Best arrival time among the OPPONENTS contesting the same point.
    pub best_opponent_arrival: f64,
    /// 50/50 margin (s): our best arrival minus theirs. Negative ⇒ we are favoured
    /// to win the ball; positive ⇒ they are.
    pub fifty_fifty_margin: f64,
    /// Distance (yd) from the loose-ball point to the candidate's LP/IPM-assigned
    /// formation slot — the "how far out of shape would this drag me" cost.
    pub formation_displacement: f64,
    /// How far (yd) the candidate may legally stray from its slot to contest, given
    /// the danger of the area and the race it wins. Displacement beyond this is the
    /// out-of-position signal.
    pub formation_budget: f64,
    /// Dynamic lane affinity of the candidate toward the point (existing term).
    pub lane_affinity: f64,
    /// Ball-receipt shape fit of the candidate toward the point (existing term).
    pub receipt_shape: f64,
    /// Loose-ball speed (yd/s).
    pub ball_speed: f64,
    /// Loose-ball altitude (yd).
    pub ball_altitude: f64,
    /// Probability the candidate's team controls the contest point (pitch control).
    pub team_control_at_target: f64,
    // Role one-hot (goalkeeper is never a field-retrieval candidate, so excluded).
    pub role_defender: f64,
    pub role_midfielder: f64,
    pub role_forward: f64,
    /// Group 8 retained determinant: the engine's existing analytic retrieval score
    /// for this candidate (lower = better placed). The model learns *relative to*
    /// the well-tuned election so it is refined, never discarded.
    pub heuristic_retrieval_score: f64,
}

impl LooseBallCommitInputs {
    /// The normalized network input vector. **This ordering is the single source of
    /// truth** for the feature layout — change it and bump
    /// [`LOOSE_BALL_COMMIT_FEATURE_DIM`] and any persisted head.
    pub fn to_features(&self) -> [f64; LOOSE_BALL_COMMIT_FEATURE_DIM] {
        let dist = |d: f64| (d / REF_DISTANCE_YARDS).clamp(0.0, 2.0);
        let secs = |s: f64| (s / REF_ARRIVAL_SECONDS).clamp(0.0, 2.0);
        let margin = |m: f64| (m / 2.0).clamp(-1.5, 1.5);
        let unit = |u: f64| u.clamp(0.0, 1.0);
        [
            dist(self.dist_to_target),
            secs(self.arrival_time),
            unit(self.stride_alignment),
            secs(self.best_teammate_arrival),
            secs(self.best_opponent_arrival),
            margin(self.fifty_fifty_margin),
            dist(self.formation_displacement),
            dist(self.formation_budget),
            (self.lane_affinity).clamp(-2.0, 2.0),
            (self.receipt_shape / 5.0).clamp(-2.0, 2.0),
            (self.ball_speed / REF_BALL_SPEED_YPS).clamp(0.0, 2.0),
            (self.ball_altitude / 3.0).clamp(0.0, 2.0),
            unit(self.team_control_at_target),
            self.role_defender,
            self.role_midfielder,
            self.role_forward,
            (self.heuristic_retrieval_score / REF_DISTANCE_YARDS).clamp(-2.0, 2.0),
        ]
    }
}

/// Logistic squash to `(0, 1)`.
fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

/// Analytic, deterministic, RNG-free seed for the **commit priority** in `(0, 1)`
/// (higher = this candidate should break off and attack the loose ball). It is the
/// bootstrap target and the live fallback the learned [`LooseBallCommitHead`] first
/// imitates, then beats.
///
/// The intent it encodes: commit harder when you are close and already striding
/// onto the ball, when you (or your team) win the race to it, and when it sits near
/// your formation slot; hold back when an opponent clearly wins it, or when
/// attacking it would drag you well beyond your formation budget (the
/// "out-of-position" damp).
pub fn analytic_loose_ball_commit_priority(inputs: &LooseBallCommitInputs) -> f64 {
    let mut z = 0.0;
    // Proximity: a closer candidate commits more.
    z += 2.2 * (1.0 - (inputs.dist_to_target / 12.0).clamp(0.0, 1.0));
    // Stride already pointed at the ball.
    z += 1.4 * inputs.stride_alignment.clamp(0.0, 1.0);
    // Am I the best-placed teammate? Faster than the next-best ⇒ commit; slower ⇒
    // defer (someone else has it).
    z -= 1.6 * ((inputs.arrival_time - inputs.best_teammate_arrival) / 1.0).clamp(-1.5, 1.5);
    // Do WE win the 50/50? margin < 0 ⇒ favoured ⇒ commit.
    z += 1.2 * ((-inputs.fifty_fifty_margin) / 1.0).clamp(-1.5, 1.5);
    // Formation discipline: straying beyond the budget damps the urge to commit.
    let over_budget = (inputs.formation_displacement - inputs.formation_budget).max(0.0);
    z -= 1.8 * (over_budget / 8.0).clamp(0.0, 1.5);
    // Lane / receipt shape (existing retrieval determinants).
    z += 0.6 * inputs.lane_affinity.clamp(-1.0, 1.0);
    z += 0.4 * (inputs.receipt_shape / 5.0).clamp(-1.0, 1.0);
    sigmoid(z)
}

/// One reward-weighted RL training row for the commit head: the per-candidate state
/// at a loose-ball decision, the commit ACTION the engine took (1 = this candidate
/// was elected to attack the ball, 0 = it held shape / peeled off), and the reward
/// attributed over the following window (higher = the choice led to winning/keeping
/// possession without conceding shape). Mirrors [`super::back_four_line::LineDepthSample`].
#[derive(Clone, Debug, PartialEq)]
pub struct LooseBallCommitSample {
    pub inputs: LooseBallCommitInputs,
    /// The commit action actually taken, in `[0, 1]` (1 = elected to attack).
    pub action_commit: f64,
    /// Reward attributed to the decision over the following window (any scale; only
    /// relative-to-batch matters — the trainer baselines it).
    pub reward: f64,
}

/// An open loose-ball commit decision awaiting its windowed reward. Recorded at the
/// decision tick with the deciding team's possession/territorial state then;
/// resolved `LOOSE_BALL_COMMIT_REWARD_WINDOW_TICKS` later by scoring whether the
/// team won/kept the ball.
#[derive(Clone, Debug, PartialEq)]
pub struct PendingLooseBallCommitDecision {
    pub team: Team,
    pub player_id: usize,
    pub inputs: LooseBallCommitInputs,
    pub action_commit: f64,
    /// Team territorial advantage captured at the decision tick (windowed baseline).
    pub decision_territorial: f64,
    pub due_tick: u64,
}

/// How often (ticks) loose-ball commit RL decisions are sampled while the model is on.
pub const LOOSE_BALL_COMMIT_SAMPLE_INTERVAL_TICKS: u64 = 3;
/// Window (ticks) over which a sampled decision's possession/territorial reward is measured.
pub const LOOSE_BALL_COMMIT_REWARD_WINDOW_TICKS: u64 = 30;
/// Cap on the rolling RL sample buffer (drained by the learner).
pub const LOOSE_BALL_COMMIT_SAMPLE_CAP: usize = 8192;
/// Minimum training steps before a trained head is consumed live; below this the
/// regression net is too raw, so the election falls back to the analytic seed.
pub const LOOSE_BALL_COMMIT_HEAD_MIN_TRAINING_STEPS: usize = 200;

/// Learned regression head for the per-candidate commit priority: a
/// `FeedForwardNetwork` (`DIM → hidden → 1`, sigmoid). Round-trips to Postgres via
/// the engine's `SoccerNeuralNetworkSnapshot` path like the other learned heads.
#[derive(Clone, Debug)]
pub struct LooseBallCommitHead {
    network: FeedForwardNetwork,
    training_steps: usize,
    last_loss: Option<f64>,
}

impl LooseBallCommitHead {
    pub fn new(seed: u32) -> Self {
        let mut rng = mulberry32(seed ^ 0x4B0C_5E27);
        let network = FeedForwardNetwork::random(
            &RandomNetworkSpec {
                input_dim: LOOSE_BALL_COMMIT_FEATURE_DIM,
                hidden_layers: vec![LOOSE_BALL_COMMIT_HIDDEN_UNITS],
                output_dim: 1,
                hidden_activation: ActivationName::Tanh,
                output_activation: ActivationName::Sigmoid,
                weight_scale: None,
            },
            &mut rng,
        );
        LooseBallCommitHead {
            network,
            training_steps: 0,
            last_loss: None,
        }
    }

    /// Predicted commit priority in `(0, 1)` for a captured candidate state, or
    /// `None` on a malformed (mis-dimensioned / non-finite) input.
    pub fn predict(&self, inputs: &LooseBallCommitInputs) -> Option<f64> {
        let features = inputs.to_features();
        if features.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let out = self.network.predict(&features[..]);
        out.first().copied().filter(|p| p.is_finite())
    }

    /// One SGD epoch regressing the head toward `(features, target_priority)` pairs
    /// (supervised bootstrap toward the analytic seed). Returns the mean step loss.
    pub fn train(&mut self, samples: &[(LooseBallCommitInputs, f64)], learning_rate: f64) -> f64 {
        let mut total = 0.0;
        let mut applied = 0usize;
        for (inputs, target) in samples {
            let features = inputs.to_features();
            if features.iter().any(|v| !v.is_finite()) || !target.is_finite() {
                continue;
            }
            let target = [target.clamp(0.0, 1.0)];
            let result =
                self.network
                    .train_sample_clipped(&features[..], &target, learning_rate, 4.0);
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

    /// One **reward-weighted-regression** epoch over RL samples (offline policy
    /// improvement): regress toward each sample's ACTION (commit / hold), weighted by
    /// how much its reward beat the batch baseline. Above-average decisions are
    /// imitated strongly, below-average ones barely — so the head learns the commit
    /// pattern that actually won the ball without breaking shape. Returns the mean
    /// step loss. Mirrors [`super::back_four_line::BackFourLineHead::train_reward_weighted`].
    pub fn train_reward_weighted(
        &mut self,
        samples: &[LooseBallCommitSample],
        learning_rate: f64,
    ) -> f64 {
        let finite: Vec<&LooseBallCommitSample> = samples
            .iter()
            .filter(|s| s.reward.is_finite() && s.action_commit.is_finite())
            .collect();
        if finite.is_empty() {
            return 0.0;
        }
        let n = finite.len() as f64;
        let baseline = finite.iter().map(|s| s.reward).sum::<f64>() / n;
        let std = (finite
            .iter()
            .map(|s| (s.reward - baseline).powi(2))
            .sum::<f64>()
            / n)
            .sqrt()
            .max(1e-3);
        let mut total = 0.0;
        let mut applied = 0usize;
        for s in finite {
            let features = s.inputs.to_features();
            if features.iter().any(|v| !v.is_finite()) {
                continue;
            }
            let advantage = (s.reward - baseline) / std;
            let weight = advantage.clamp(-4.0, 2.0).exp().min(7.5);
            let target = [s.action_commit.clamp(0.0, 1.0)];
            let result = self.network.train_sample_clipped(
                &features[..],
                &target,
                learning_rate * weight,
                4.0,
            );
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

/// Outcome of training a commit head on a drained RL corpus, logged by the learning
/// runner so an operator can see the corpus is loaded and learned from. Mirrors
/// [`super::back_four_line::LineDepthTrainingReport`].
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LooseBallCommitTrainingReport {
    pub samples: usize,
    pub training_steps: usize,
    pub epochs: usize,
    pub final_loss: f64,
}

/// Load-and-train entry for a commit head: given a drained RL corpus, build a fresh
/// [`LooseBallCommitHead`] and run `epochs` reward-weighted-regression passes. `None`
/// if no usable samples. Mirrors [`super::back_four_line::train_line_depth_head`].
pub fn train_loose_ball_commit_head(
    samples: &[LooseBallCommitSample],
    seed: u32,
    epochs: usize,
    learning_rate: f64,
) -> Option<(LooseBallCommitHead, LooseBallCommitTrainingReport)> {
    let usable = samples
        .iter()
        .filter(|s| s.reward.is_finite() && s.action_commit.is_finite())
        .count();
    if usable == 0 || epochs == 0 {
        return None;
    }
    let mut head = LooseBallCommitHead::new(seed);
    let mut final_loss = 0.0;
    for _ in 0..epochs {
        final_loss = head.train_reward_weighted(samples, learning_rate);
    }
    let report = LooseBallCommitTrainingReport {
        samples: usable,
        training_steps: head.training_steps(),
        epochs,
        final_loss,
    };
    Some((head, report))
}

/// Public wrapper returning ONLY the report (the head type is engine-internal). The
/// learning runner uses this to train the drained corpus and log that it learns.
pub fn report_loose_ball_commit_training(
    samples: &[LooseBallCommitSample],
    seed: u32,
    epochs: usize,
    learning_rate: f64,
) -> Option<LooseBallCommitTrainingReport> {
    train_loose_ball_commit_head(samples, seed, epochs, learning_rate).map(|(_, report)| report)
}

impl WorldSnapshot {
    /// Build the [`LooseBallCommitInputs`] for `player` contesting the loose-ball
    /// point `target`. `None` for a goalkeeper or a degenerate roster. Pure / RNG-free.
    pub(crate) fn build_loose_ball_commit_inputs(
        &self,
        player: &PlayerSnapshot,
        target: Vec2,
    ) -> Option<LooseBallCommitInputs> {
        if player.role == PlayerRole::Goalkeeper {
            return None;
        }
        let pos = self.player_snapshot_position(player);
        let to_target = target - pos;
        let dist = to_target.len();
        let stride_alignment = if player.velocity.len() > 0.5 && dist > 1e-3 {
            player
                .velocity
                .normalized()
                .dot(to_target.normalized())
                .max(0.0)
        } else {
            0.0
        };
        let my_arrival = loose_ball_arrival_time_seconds(self, player, target);

        // Race context: best teammate arrival (excluding me) and best opponent arrival.
        let mut best_teammate = f64::INFINITY;
        let mut own_best = my_arrival;
        let mut best_opponent = f64::INFINITY;
        for p in &self.players {
            if p.role == PlayerRole::Goalkeeper {
                continue;
            }
            let arrival = loose_ball_arrival_time_seconds(self, p, target);
            if p.team == player.team {
                own_best = own_best.min(arrival);
                if p.id != player.id {
                    best_teammate = best_teammate.min(arrival);
                }
            } else {
                best_opponent = best_opponent.min(arrival);
            }
        }
        let fifty_fifty_margin = if own_best.is_finite() && best_opponent.is_finite() {
            (own_best - best_opponent).clamp(-3.0, 3.0)
        } else {
            0.0
        };

        // Formation discipline: how far the ball sits from my LP/IPM slot, vs. how
        // far I may stray for it (danger of the area + how clearly I win the race).
        let slot = self.formation_slot_anchor_for(player.id, player.home_position);
        let formation_displacement = target.distance(slot);
        let formation_budget =
            self.pass_contest_position_budget_for(player, target, my_arrival, best_opponent);

        let lane_affinity = self.dynamic_lane_affinity_for_player_target(player, target);
        let receipt_shape = self.ball_receipt_shape_fit_for_player_target(player, target);

        let team_control_at_target = {
            let home = super::pitch_value::pitch_control_home(&self.players, target);
            match player.team {
                Team::Home => home,
                Team::Away => 1.0 - home,
            }
        };

        Some(LooseBallCommitInputs {
            field_length: self.field_length,
            field_width: self.field_width,
            dist_to_target: dist,
            arrival_time: my_arrival,
            stride_alignment,
            best_teammate_arrival: if best_teammate.is_finite() {
                best_teammate
            } else {
                my_arrival
            },
            best_opponent_arrival: if best_opponent.is_finite() {
                best_opponent
            } else {
                REF_ARRIVAL_SECONDS
            },
            fifty_fifty_margin,
            formation_displacement,
            formation_budget,
            lane_affinity,
            receipt_shape,
            ball_speed: self.ball.velocity.len(),
            ball_altitude: self.ball.altitude_yards,
            team_control_at_target,
            role_defender: (player.role == PlayerRole::Defender) as i32 as f64,
            role_midfielder: (player.role == PlayerRole::Midfielder) as i32 as f64,
            role_forward: (player.role == PlayerRole::Forward) as i32 as f64,
            // Defaulted to the live retrieval score by the caller (it has it in hand).
            heuristic_retrieval_score: 0.0,
        })
    }

    /// The model's **commit priority** in `(0, 1)` for `player` contesting `target`
    /// (higher = should attack the ball). The trained head once it has learned
    /// enough, else the analytic seed. `None` when the model is gated off or the
    /// inputs are degenerate. `heuristic_score` is the engine's existing retrieval
    /// score for this candidate (group 8 retained determinant).
    pub(crate) fn loose_ball_commit_priority(
        &self,
        player: &PlayerSnapshot,
        target: Vec2,
        heuristic_score: f64,
    ) -> Option<f64> {
        if !loose_ball_commit_model_enabled() {
            return None;
        }
        let mut inputs = self.build_loose_ball_commit_inputs(player, target)?;
        inputs.heuristic_retrieval_score = heuristic_score;
        let priority = self
            .loose_ball_commit_head
            .as_ref()
            .filter(|head| head.training_steps() >= LOOSE_BALL_COMMIT_HEAD_MIN_TRAINING_STEPS)
            .and_then(|head| head.predict(&inputs))
            .unwrap_or_else(|| analytic_loose_ball_commit_priority(&inputs));
        priority.is_finite().then(|| priority.clamp(0.0, 1.0))
    }

    /// Signed yards adjustment folded into [`Self::loose_ball_retrieval_score`]
    /// (lower score = better-placed retriever). A high commit priority LOWERS the
    /// score (this player becomes the elected attacker); a low priority — e.g. one
    /// the model judges would be dragged out of shape — RAISES it. Returns `0.0`
    /// (without building features) unless the model is enabled, keeping the default
    /// election byte-identical.
    pub(crate) fn loose_ball_commit_score_adjustment(
        &self,
        player: &PlayerSnapshot,
        target: Vec2,
        heuristic_score: f64,
    ) -> f64 {
        let Some(priority) = self.loose_ball_commit_priority(player, target, heuristic_score) else {
            return 0.0;
        };
        // priority 1 ⇒ −SWING (better); priority 0 ⇒ +SWING (worse).
        -(priority - 0.5) * 2.0 * LOOSE_BALL_COMMIT_SCORE_SWING_YARDS
    }
}

impl SoccerMatch {
    /// Gated per-tick RL sample collection for the loose-ball commit head. A
    /// **no-op** (zero cost, byte-identical) unless the commit model is enabled. When
    /// on: resolves decisions whose reward window has elapsed (reward = the deciding
    /// team's territorial-advantage change over the window — higher means committing
    /// these players won/kept the ball without ceding shape), and samples fresh
    /// per-candidate commit decisions on cadence whenever the ball is loose.
    pub(crate) fn collect_loose_ball_commit_rl_samples(&mut self, snapshot: &WorldSnapshot) {
        if !loose_ball_commit_model_enabled() {
            return;
        }
        let tick = self.tick;
        // Resolve due decisions.
        let mut i = 0;
        while i < self.pending_loose_ball_commit.len() {
            if self.pending_loose_ball_commit[i].due_tick <= tick {
                let decision = self.pending_loose_ball_commit.swap_remove(i);
                let now_territorial = territorial_advantage(snapshot, decision.team);
                if now_territorial.is_finite() && decision.decision_territorial.is_finite() {
                    self.loose_ball_commit_samples.push(LooseBallCommitSample {
                        inputs: decision.inputs,
                        action_commit: decision.action_commit,
                        reward: now_territorial - decision.decision_territorial,
                    });
                    if self.loose_ball_commit_samples.len() > LOOSE_BALL_COMMIT_SAMPLE_CAP {
                        let overflow =
                            self.loose_ball_commit_samples.len() - LOOSE_BALL_COMMIT_SAMPLE_CAP;
                        self.loose_ball_commit_samples.drain(0..overflow);
                    }
                }
            } else {
                i += 1;
            }
        }
        // Sample fresh per-candidate decisions on cadence while the ball is loose.
        if snapshot.ball.holder.is_some() || snapshot.pending_pass.is_some() {
            return;
        }
        if tick % LOOSE_BALL_COMMIT_SAMPLE_INTERVAL_TICKS != 0 {
            return;
        }
        let candidate_ids: Vec<usize> = snapshot
            .players
            .iter()
            .filter(|p| p.role != PlayerRole::Goalkeeper)
            .map(|p| p.id)
            .collect();
        for id in candidate_ids {
            let Some(player) = snapshot.players.iter().find(|p| p.id == id) else {
                continue;
            };
            let target = snapshot.loose_ball_recovery_target_for(id);
            let heuristic_score = snapshot.loose_ball_retrieval_score(player, target);
            let Some(mut inputs) = snapshot.build_loose_ball_commit_inputs(player, target) else {
                continue;
            };
            inputs.heuristic_retrieval_score = heuristic_score;
            // The action actually taken this tick: was this candidate one of the
            // committed attackers (the election the engine ran), or did it hold shape?
            let action_commit = if snapshot.is_committed_loose_ball_chaser(id) {
                1.0
            } else {
                0.0
            };
            let territorial = territorial_advantage(snapshot, player.team);
            if territorial.is_finite() {
                self.pending_loose_ball_commit
                    .push(PendingLooseBallCommitDecision {
                        team: player.team,
                        player_id: id,
                        inputs,
                        action_commit,
                        decision_territorial: territorial,
                        due_tick: tick + LOOSE_BALL_COMMIT_REWARD_WINDOW_TICKS,
                    });
            }
        }
    }

    /// Drain the collected commit RL samples for the cluster learner (train the head
    /// + persist). Mirrors [`Self::drain_line_depth_samples`].
    pub fn drain_loose_ball_commit_samples(&mut self) -> Vec<LooseBallCommitSample> {
        std::mem::take(&mut self.loose_ball_commit_samples)
    }

    /// Install the trained commit head for live consumption. The learner carries +
    /// trains it across games and re-installs it each game; wrapped in `Arc` so every
    /// per-tick snapshot shares it cheaply.
    pub fn set_loose_ball_commit_head(&mut self, head: LooseBallCommitHead) {
        self.loose_ball_commit_head = Some(std::sync::Arc::new(head));
    }
}

#[cfg(test)]
mod loose_ball_commit_tests {
    use super::*;

    fn baseline_inputs() -> LooseBallCommitInputs {
        LooseBallCommitInputs {
            field_length: 120.0,
            field_width: 80.0,
            dist_to_target: 8.0,
            arrival_time: 1.2,
            stride_alignment: 0.5,
            best_teammate_arrival: 1.6,
            best_opponent_arrival: 1.4,
            fifty_fifty_margin: -0.2,
            formation_displacement: 6.0,
            formation_budget: 9.0,
            lane_affinity: 0.3,
            receipt_shape: 1.0,
            ball_speed: 6.0,
            ball_altitude: 0.0,
            team_control_at_target: 0.55,
            role_defender: 0.0,
            role_midfielder: 1.0,
            role_forward: 0.0,
            heuristic_retrieval_score: 7.0,
        }
    }

    #[test]
    fn features_are_finite_bounded_and_right_dim() {
        let f = baseline_inputs().to_features();
        assert_eq!(f.len(), LOOSE_BALL_COMMIT_FEATURE_DIM);
        for v in f {
            assert!(v.is_finite());
            assert!(v.abs() <= 2.0 + 1e-9, "feature {v} out of normalized range");
        }
    }

    #[test]
    fn analytic_priority_is_a_valid_probability() {
        let p = analytic_loose_ball_commit_priority(&baseline_inputs());
        assert!((0.0..=1.0).contains(&p), "priority {p} out of [0,1]");
    }

    #[test]
    fn closer_player_commits_more_than_a_distant_one() {
        let mut near = baseline_inputs();
        near.dist_to_target = 2.0;
        near.arrival_time = 0.4;
        let mut far = baseline_inputs();
        far.dist_to_target = 22.0;
        far.arrival_time = 3.0;
        assert!(
            analytic_loose_ball_commit_priority(&near)
                > analytic_loose_ball_commit_priority(&far),
            "the closer, faster candidate should have higher commit priority"
        );
    }

    #[test]
    fn out_of_position_player_commits_less() {
        let mut in_shape = baseline_inputs();
        in_shape.formation_displacement = 4.0; // well within budget
        let mut out_of_shape = baseline_inputs();
        out_of_shape.formation_displacement = 30.0; // dragged way out of slot
        assert!(
            analytic_loose_ball_commit_priority(&in_shape)
                > analytic_loose_ball_commit_priority(&out_of_shape),
            "a candidate that would be dragged out of shape should commit less"
        );
    }

    #[test]
    fn favoured_fifty_fifty_commits_more_than_a_losing_one() {
        let mut favoured = baseline_inputs();
        favoured.fifty_fifty_margin = -0.8; // we get there well before them
        let mut losing = baseline_inputs();
        losing.fifty_fifty_margin = 0.8; // they win the race
        assert!(
            analytic_loose_ball_commit_priority(&favoured)
                > analytic_loose_ball_commit_priority(&losing),
            "winning the race should raise commit priority"
        );
    }

    #[test]
    fn head_predicts_a_bounded_priority() {
        let head = LooseBallCommitHead::new(7);
        let p = head.predict(&baseline_inputs()).expect("finite prediction");
        assert!((0.0..=1.0).contains(&p), "head priority {p} out of (0,1)");
    }

    #[test]
    fn reward_weighted_training_steers_toward_the_high_reward_action() {
        let mut head = LooseBallCommitHead::new(23);
        let inputs = baseline_inputs();
        let before = head.predict(&inputs).expect("finite");
        // Same state, two competing actions: committing (1.0) is rewarded, holding
        // (0.0) is penalized. RWR should pull the head toward committing.
        let samples: Vec<LooseBallCommitSample> = (0..32)
            .flat_map(|_| {
                [
                    LooseBallCommitSample {
                        inputs: inputs.clone(),
                        action_commit: 1.0,
                        reward: 1.0,
                    },
                    LooseBallCommitSample {
                        inputs: inputs.clone(),
                        action_commit: 0.0,
                        reward: -1.0,
                    },
                ]
            })
            .collect();
        let mut last = f64::INFINITY;
        for _ in 0..80 {
            last = head.train_reward_weighted(&samples, 0.05);
        }
        let after = head.predict(&inputs).expect("finite");
        assert!(last.is_finite());
        assert!(
            after > before && after > 0.5,
            "RWR should move the head toward the high-reward commit action: {before} -> {after}"
        );
    }

    #[test]
    fn train_entry_skips_empty_and_trains_valid() {
        assert!(train_loose_ball_commit_head(&[], 1, 4, 0.02).is_none());
        let samples: Vec<LooseBallCommitSample> = (0..16)
            .map(|i| LooseBallCommitSample {
                inputs: baseline_inputs(),
                action_commit: if i % 2 == 0 { 1.0 } else { 0.0 },
                reward: if i % 2 == 0 { 1.0 } else { -1.0 },
            })
            .collect();
        let (_, report) =
            train_loose_ball_commit_head(&samples, 1, 4, 0.02).expect("trains a usable corpus");
        assert_eq!(report.samples, 16);
        assert_eq!(report.epochs, 4);
        assert!(report.training_steps > 0 && report.final_loss.is_finite());
    }
}
