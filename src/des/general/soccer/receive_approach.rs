//! Learnable **receive-approach decision** — when a pass/ball is in flight to a
//! player, *how much should they step TOWARD the ball (meet it early, deny an
//! opponent) or ease AWAY from it (let it run onto a better foot, into space)* as it
//! approaches.
//!
//! Receiving a ball is not a fixed point: a receiver can attack the pass (shorten the
//! gap, take it sooner, win it before a defender can nick it) or settle off it (let
//! the ball travel onto their back foot and receive facing forward into space). The
//! engine already encodes a hand-tuned blend of this inside
//! [`WorldSnapshot::pending_pass_reception_target_for`] (the `pressure_early_touch_bonus`
//! / `contest_score` terms that pull the meeting point earlier along the flight). This
//! module is the seam that makes that *one continuous choice* — a signed **approach
//! bias** in `[-1, 1]` — learnable, mirroring [`super::loose_ball_commit`]'s
//! reward-weighted head:
//!
//!   * [`ReceiveApproachInputs`] — the per-receiver state the choice is a function of,
//!     captured in the receiver's attacking frame so it is mirror-invariant. The core
//!     of it is the **kinematics the user asked for**: the position gap to the ball and
//!     the ball's and the receiver's velocity / acceleration / **jerk** resolved on the
//!     approach axis, plus the decisive race term — does the nearest opponent reach the
//!     reception point before me? (if so, you must step in to win it first), pressure,
//!     ball altitude, role, and pitch control at the meeting point.
//!   * [`analytic_receive_approach_bias`] — a deterministic, RNG-free seed mapping that
//!     state to a bias in `(-1, 1)` (positive = step toward the ball). It is the
//!     bootstrap target and the live fallback, and it encodes the coach's prior: by
//!     default lean *toward* the ball, much harder when an opponent can get there first
//!     or the receiver is pressured; only when genuinely free with space behind may the
//!     receiver ease off and let the ball come.
//!   * [`ReceiveApproachHead`] — a `FeedForwardNetwork` (`DIM → hidden → 1`, **tanh**
//!     output so it is signed) the cluster learner trains by reward-weighted regression
//!     toward the approach actions that actually secured/retained possession.
//!
//! The live seam is [`WorldSnapshot::receive_approach_adjusted_target`]: it shifts the
//! elected reception point **along the line to the ball** by `bias *
//! RECEIVE_APPROACH_MAX_SHIFT_YARDS` — positive shifts the meeting point toward the
//! ball (meet it earlier), negative eases it back. The shift is small and never
//! overshoots the ball, so the model *refines* the well-tuned reception election rather
//! than overturning it.
//!
//! ## Parity
//!
//! Gated **off** by default (`DD_SOCCER_ENABLE_RECEIVE_APPROACH_MODEL`). When unset,
//! [`WorldSnapshot::receive_approach_adjusted_target`] returns the reception point
//! unchanged *before building any features*, and the RL collector early-returns, so the
//! reception election — and therefore the whole engine — is byte-identical to before
//! this module existed. The feature build, analytic seed, and head are pure / RNG-free.

use super::*;
use crate::des::general::neural_network::{ActivationName, FeedForwardNetwork, RandomNetworkSpec};
use crate::des::general::prng::mulberry32;

/// Number of features in the receive-approach state vector. The exact ordering is the
/// single source of truth in [`ReceiveApproachInputs::to_features`].
pub const RECEIVE_APPROACH_FEATURE_DIM: usize = 17;
/// Hidden width of the learned regression head.
pub const RECEIVE_APPROACH_HIDDEN_UNITS: usize = 20;
/// Maximum yards the model may shift the reception point along the line to the ball. A
/// bias of `+1` steps the meeting point this far toward the ball (meet it early); `-1`
/// eases it back this far. Kept modest so the model refines the well-tuned reception
/// election rather than overturning it.
pub const RECEIVE_APPROACH_MAX_SHIFT_YARDS: f64 = 3.5;

/// Reference distance (yd) normalizing distance features.
const REF_DISTANCE_YARDS: f64 = 30.0;
/// Reference speed (yd/s) normalizing speed features.
const REF_SPEED_YPS: f64 = 20.0;
/// Reference acceleration (yd/s²) normalizing acceleration features.
const REF_ACCEL_YPS2: f64 = 10.0;
/// Reference jerk (yd/s³) normalizing jerk features.
const REF_JERK_YPS3: f64 = 20.0;

/// Env gate enabling the learned receive-approach seam at the reception election. Off
/// (unset) by default so an unconfigured process stays byte-identical.
const RECEIVE_APPROACH_MODEL_ENABLE_ENV: &str = "DD_SOCCER_ENABLE_RECEIVE_APPROACH_MODEL";

fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|raw| {
            let v = raw.trim().to_ascii_lowercase();
            matches!(v.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

/// Whether the learned receive-approach seam is consulted this process. Off ⇒ the
/// analytic reception election stands (parity).
pub fn receive_approach_model_enabled() -> bool {
    #[cfg(test)]
    {
        env_flag_enabled(RECEIVE_APPROACH_MODEL_ENABLE_ENV)
    }
    #[cfg(not(test))]
    {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| env_flag_enabled(RECEIVE_APPROACH_MODEL_ENABLE_ENV))
    }
}

/// Raw (un-normalized) per-receiver state the approach choice is a function of, captured
/// in the receiver's **attacking frame** so the model is mirror-invariant. A plain
/// struct so it can be logged as a training row, asserted on in tests, and surfaced by
/// the inspector independently of the network.
///
/// Sign conventions are all relative to the **approach axis** = the unit vector pointing
/// from the receiver toward the ball. "Toward me" / "toward ball" components are positive
/// when they shrink the receiver↔ball gap.
#[derive(Clone, Debug, PartialEq)]
pub struct ReceiveApproachInputs {
    pub field_length: f64,
    pub field_width: f64,

    /// Current distance (yd) from the receiver to the ball.
    pub dist_to_ball: f64,
    /// Ball speed (yd/s).
    pub ball_speed: f64,
    /// Ball velocity component heading **toward the receiver** (yd/s). Positive ⇒ the
    /// ball is coming at them; negative ⇒ running away.
    pub ball_closing_speed: f64,
    /// Ball acceleration component toward the receiver (yd/s²). A rolling ball decelerates,
    /// so an incoming ball reads slightly negative ("dying short as it comes").
    pub ball_accel_toward_me: f64,
    /// Ball jerk component toward the receiver (yd/s³) — the rate of change of the ball's
    /// approach acceleration (skip / bobble / settle transition).
    pub ball_jerk_toward_me: f64,
    /// Receiver velocity component toward the ball (yd/s). Positive ⇒ already striding in.
    pub player_speed_toward_ball: f64,
    /// Receiver acceleration component toward the ball (yd/s²).
    pub player_accel_toward_ball: f64,
    /// Receiver jerk component toward the ball (yd/s³).
    pub player_jerk_toward_ball: f64,
    /// The race (s): nearest-opponent sprint arrival to the reception point MINUS the
    /// receiver's. Negative ⇒ an opponent gets there first ⇒ step in to win it. This is
    /// the decisive "don't let them reach the ball first" term.
    pub opponent_race_margin: f64,
    /// Distance (yd) of the nearest opponent to the ball **right now**.
    pub nearest_opp_dist_to_ball: f64,
    /// Distance (yd) of the nearest opponent to the **reception point** (how tight the
    /// receipt is).
    pub reception_pressure: f64,
    /// Signed forwardness (yd) of the reception point relative to the receiver, in the
    /// attacking direction. Positive ⇒ stepping forward onto it; negative ⇒ checking back.
    pub forward_reception: f64,
    /// Ball altitude (yd) at the reception point read (ground vs bouncing/aerial).
    pub ball_altitude: f64,
    /// Probability the receiver's team controls the reception point (pitch control).
    pub team_control_at_reception: f64,
    // Role one-hot (a goalkeeper is never a receive-approach candidate, so excluded).
    pub role_defender: f64,
    pub role_midfielder: f64,
    pub role_forward: f64,
}

impl ReceiveApproachInputs {
    /// The normalized network input vector. **This ordering is the single source of
    /// truth** for the feature layout — change it and bump
    /// [`RECEIVE_APPROACH_FEATURE_DIM`] and any persisted head.
    pub fn to_features(&self) -> [f64; RECEIVE_APPROACH_FEATURE_DIM] {
        let dist = |d: f64| (d / REF_DISTANCE_YARDS).clamp(0.0, 2.0);
        let spd = |s: f64| (s / REF_SPEED_YPS).clamp(-1.5, 1.5);
        let acc = |a: f64| (a / REF_ACCEL_YPS2).clamp(-1.5, 1.5);
        let jrk = |j: f64| (j / REF_JERK_YPS3).clamp(-1.5, 1.5);
        let margin = |m: f64| (m / 2.0).clamp(-1.5, 1.5);
        let unit = |u: f64| u.clamp(0.0, 1.0);
        [
            dist(self.dist_to_ball),
            (self.ball_speed / REF_SPEED_YPS).clamp(0.0, 2.0),
            spd(self.ball_closing_speed),
            acc(self.ball_accel_toward_me),
            jrk(self.ball_jerk_toward_me),
            spd(self.player_speed_toward_ball),
            acc(self.player_accel_toward_ball),
            jrk(self.player_jerk_toward_ball),
            margin(self.opponent_race_margin),
            dist(self.nearest_opp_dist_to_ball),
            dist(self.reception_pressure),
            (self.forward_reception / REF_DISTANCE_YARDS).clamp(-1.5, 1.5),
            (self.ball_altitude / 3.0).clamp(0.0, 2.0),
            unit(self.team_control_at_reception),
            self.role_defender,
            self.role_midfielder,
            self.role_forward,
        ]
    }
}

/// Analytic, deterministic, RNG-free seed for the **approach bias** in `(-1, 1)`
/// (positive = step toward the ball / meet it early; negative = ease off / let it come).
/// It is the bootstrap target and the live fallback the learned [`ReceiveApproachHead`]
/// first imitates, then beats.
///
/// The coach's prior it encodes: by default lean *toward* the ball; step in much harder
/// when an opponent can reach the reception first, when the receipt is tight, when an
/// opponent is already near the ball, or when the ball is dying short while still far
/// (so it doesn't stop in front of an onrushing defender). Only when the receiver is
/// genuinely free *and* the race is comfortably won may they ease off and take it on the
/// back foot into space.
pub fn analytic_receive_approach_bias(inputs: &ReceiveApproachInputs) -> f64 {
    let mut z = 0.0;
    // Default prior: generally move toward the ball when receiving it.
    z += 0.30;
    // THE driver: an opponent reaches the reception first (margin < 0) ⇒ step in to win it.
    z += 1.7 * ((-inputs.opponent_race_margin) / 1.0).clamp(-1.5, 1.5);
    // Tight receipt ⇒ meet it, deny the onrushing defender; lots of room ⇒ no urgency.
    z += 1.0 * (1.0 - (inputs.reception_pressure / 6.0).clamp(0.0, 1.0));
    // An opponent already close to the ball now ⇒ attack it.
    z += 0.6 * (1.0 - (inputs.nearest_opp_dist_to_ball / 8.0).clamp(0.0, 1.0));
    // A ball decelerating toward rest while still far ⇒ step in so it doesn't die short.
    z += 0.5
        * ((-inputs.ball_accel_toward_me).max(0.0) / 6.0).clamp(0.0, 1.0)
        * (inputs.dist_to_ball / 15.0).clamp(0.0, 1.0);
    // Already striding onto it ⇒ keep coming.
    z += 0.3 * (inputs.player_speed_toward_ball / 6.0).clamp(-1.0, 1.0);
    // Genuinely free WITH the race comfortably won ⇒ may ease off and receive into space.
    let comfortable = (inputs.opponent_race_margin / 1.2).clamp(0.0, 1.0)
        * (inputs.reception_pressure / 8.0).clamp(0.0, 1.0);
    z -= 1.0 * comfortable;
    z.tanh()
}

/// One reward-weighted RL training row for the approach head: the per-receiver state at
/// a reception decision, the approach ACTION the engine took (signed bias in `[-1, 1]`,
/// + = stepped toward the ball), and the reward attributed over the following window
/// (higher = the choice led to securing/keeping possession). Mirrors
/// [`super::loose_ball_commit::LooseBallCommitSample`].
#[derive(Clone, Debug, PartialEq)]
pub struct ReceiveApproachSample {
    pub inputs: ReceiveApproachInputs,
    /// The approach action actually taken, in `[-1, 1]` (+ = stepped toward the ball).
    pub action_bias: f64,
    /// Reward attributed to the decision over the following window (any scale; only
    /// relative-to-batch matters — the trainer baselines it).
    pub reward: f64,
}

/// An open receive-approach decision awaiting its windowed reward. Recorded at the
/// decision tick with the receiving team's territorial state then; resolved
/// `RECEIVE_APPROACH_REWARD_WINDOW_TICKS` later by scoring whether the team kept the ball.
#[derive(Clone, Debug, PartialEq)]
pub struct PendingReceiveApproachDecision {
    pub team: Team,
    pub player_id: usize,
    pub inputs: ReceiveApproachInputs,
    pub action_bias: f64,
    /// Team territorial advantage captured at the decision tick (windowed baseline).
    pub decision_territorial: f64,
    pub due_tick: u64,
}

/// How often (ticks) receive-approach RL decisions are sampled while the model is on.
pub const RECEIVE_APPROACH_SAMPLE_INTERVAL_TICKS: u64 = 3;
/// Window (ticks) over which a sampled decision's possession/territorial reward is measured.
pub const RECEIVE_APPROACH_REWARD_WINDOW_TICKS: u64 = 30;
/// Cap on the rolling RL sample buffer (drained by the learner).
pub const RECEIVE_APPROACH_SAMPLE_CAP: usize = 8192;
/// Minimum training steps before a trained head is consumed live; below this the
/// regression net is too raw, so the seam falls back to the analytic seed.
pub const RECEIVE_APPROACH_HEAD_MIN_TRAINING_STEPS: usize = 200;

/// Learned regression head for the signed approach bias: a `FeedForwardNetwork`
/// (`DIM → hidden → 1`, **tanh** output so it is signed in `(-1, 1)`). Round-trips to
/// Postgres via the engine's neural-snapshot path like the other learned heads.
#[derive(Clone, Debug)]
pub struct ReceiveApproachHead {
    network: FeedForwardNetwork,
    training_steps: usize,
    last_loss: Option<f64>,
}

impl ReceiveApproachHead {
    pub fn new(seed: u32) -> Self {
        let mut rng = mulberry32(seed ^ 0x52A1_C7E3);
        let network = FeedForwardNetwork::random(
            &RandomNetworkSpec {
                input_dim: RECEIVE_APPROACH_FEATURE_DIM,
                hidden_layers: vec![RECEIVE_APPROACH_HIDDEN_UNITS],
                output_dim: 1,
                hidden_activation: ActivationName::Tanh,
                output_activation: ActivationName::Tanh,
                weight_scale: None,
            },
            &mut rng,
        );
        ReceiveApproachHead {
            network,
            training_steps: 0,
            last_loss: None,
        }
    }

    /// Predicted approach bias in `(-1, 1)` for a captured receiver state, or `None` on a
    /// malformed (mis-dimensioned / non-finite) input.
    pub fn predict(&self, inputs: &ReceiveApproachInputs) -> Option<f64> {
        let features = inputs.to_features();
        if features.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let out = self.network.predict(&features[..]);
        out.first().copied().filter(|p| p.is_finite())
    }

    /// One SGD epoch regressing the head toward `(features, target_bias)` pairs
    /// (supervised bootstrap toward the analytic seed). Returns the mean step loss.
    pub fn train(&mut self, samples: &[(ReceiveApproachInputs, f64)], learning_rate: f64) -> f64 {
        let mut total = 0.0;
        let mut applied = 0usize;
        for (inputs, target) in samples {
            let features = inputs.to_features();
            if features.iter().any(|v| !v.is_finite()) || !target.is_finite() {
                continue;
            }
            let target = [target.clamp(-1.0, 1.0)];
            let result =
                self.network
                    .train_sample_clipped(&features[..], &target, learning_rate, 4.0);
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

    /// One **reward-weighted-regression** epoch over RL samples (offline policy
    /// improvement): regress toward each sample's ACTION (the signed approach bias),
    /// weighted by how much its reward beat the batch baseline. Above-average decisions
    /// are imitated strongly, below-average ones barely — so the head learns the approach
    /// pattern that actually secured the ball. Returns the mean step loss. Mirrors
    /// [`super::loose_ball_commit::LooseBallCommitHead::train_reward_weighted`].
    pub fn train_reward_weighted(
        &mut self,
        samples: &[ReceiveApproachSample],
        learning_rate: f64,
    ) -> f64 {
        let finite: Vec<&ReceiveApproachSample> = samples
            .iter()
            .filter(|s| s.reward.is_finite() && s.action_bias.is_finite())
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
            let target = [s.action_bias.clamp(-1.0, 1.0)];
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

/// Outcome of training an approach head on a drained RL corpus, logged by the learning
/// runner so an operator can see the corpus is loaded and learned from. Mirrors
/// [`super::loose_ball_commit::LooseBallCommitTrainingReport`].
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReceiveApproachTrainingReport {
    pub samples: usize,
    pub training_steps: usize,
    pub epochs: usize,
    pub final_loss: f64,
}

/// Load-and-train entry for an approach head: given a drained RL corpus, build a fresh
/// [`ReceiveApproachHead`] and run `epochs` reward-weighted-regression passes. `None` if
/// no usable samples. Mirrors [`super::loose_ball_commit::train_loose_ball_commit_head`].
pub fn train_receive_approach_head(
    samples: &[ReceiveApproachSample],
    seed: u32,
    epochs: usize,
    learning_rate: f64,
) -> Option<(ReceiveApproachHead, ReceiveApproachTrainingReport)> {
    let usable = samples
        .iter()
        .filter(|s| s.reward.is_finite() && s.action_bias.is_finite())
        .count();
    if usable == 0 || epochs == 0 {
        return None;
    }
    let mut head = ReceiveApproachHead::new(seed);
    let mut final_loss = 0.0;
    for _ in 0..epochs {
        final_loss = head.train_reward_weighted(samples, learning_rate);
    }
    let report = ReceiveApproachTrainingReport {
        samples: usable,
        training_steps: head.training_steps(),
        epochs,
        final_loss,
    };
    Some((head, report))
}

/// Public wrapper returning ONLY the report (the head type is engine-internal). The
/// learning runner uses this to train the drained corpus and log that it learns.
pub fn report_receive_approach_training(
    samples: &[ReceiveApproachSample],
    seed: u32,
    epochs: usize,
    learning_rate: f64,
) -> Option<ReceiveApproachTrainingReport> {
    train_receive_approach_head(samples, seed, epochs, learning_rate).map(|(_, report)| report)
}

impl WorldSnapshot {
    /// Build the [`ReceiveApproachInputs`] for `player` (at `current`) who is meeting the
    /// ball at the elected `reception_target`. `None` for a goalkeeper or a degenerate
    /// geometry. Pure / RNG-free.
    pub(crate) fn build_receive_approach_inputs(
        &self,
        player: &PlayerSnapshot,
        current: Vec2,
        reception_target: Vec2,
    ) -> Option<ReceiveApproachInputs> {
        if player.role == PlayerRole::Goalkeeper {
            return None;
        }
        let to_ball = self.ball.position - current;
        let dist_to_ball = to_ball.len();
        // Approach axis = unit vector from the receiver toward the ball. A degenerate
        // (on-the-ball) geometry has no meaningful approach direction.
        if dist_to_ball <= 1e-3 {
            return None;
        }
        let axis = to_ball * (1.0 / dist_to_ball);

        // Ball kinematics resolved on the approach axis. `axis` points receiver→ball, so
        // a ball coming AT the receiver (velocity pointing ball→receiver = −axis) reads as
        // positive "closing". Same sign flip for accel/jerk "toward me".
        let ball_closing_speed = -self.ball.velocity.dot(axis);
        let ball_accel_toward_me = -self.ball.acceleration.dot(axis);
        let ball_jerk_toward_me = -self.ball.jerk.dot(axis);

        // Receiver kinematics resolved on the approach axis (toward the ball is positive).
        let player_speed_toward_ball = player.velocity.dot(axis);
        let player_accel_toward_ball = player.acceleration.dot(axis);
        let player_jerk_toward_ball = player.jerk.dot(axis);

        // The race to the reception point: my sprint arrival vs the nearest opponent's.
        let my_arrival = self.snapshot_sprint_time_to(player, reception_target);
        let opponent_arrival = self
            .players
            .iter()
            .filter(|p| p.team != player.team)
            .map(|p| self.snapshot_sprint_time_to(p, reception_target))
            .fold(f64::INFINITY, f64::min);
        let opponent_race_margin = if my_arrival.is_finite() && opponent_arrival.is_finite() {
            (opponent_arrival - my_arrival).clamp(-3.0, 3.0)
        } else {
            // No opponent / degenerate ⇒ no race pressure.
            3.0
        };

        let nearest_opp_dist_to_ball =
            self.nearest_opponent_distance_at(player.team, self.ball.position);
        let reception_pressure = self.nearest_opponent_distance_at(player.team, reception_target);
        let forward_reception = (reception_target.y - current.y) * player.team.attack_dir();

        let team_control_at_reception = {
            let home = super::pitch_value::pitch_control_home(&self.players, reception_target);
            match player.team {
                Team::Home => home,
                Team::Away => 1.0 - home,
            }
        };

        Some(ReceiveApproachInputs {
            field_length: self.field_length,
            field_width: self.field_width,
            dist_to_ball,
            ball_speed: self.ball.velocity.len(),
            ball_closing_speed,
            ball_accel_toward_me,
            ball_jerk_toward_me,
            player_speed_toward_ball,
            player_accel_toward_ball,
            player_jerk_toward_ball,
            opponent_race_margin,
            nearest_opp_dist_to_ball,
            reception_pressure,
            forward_reception,
            ball_altitude: self.ball.altitude_yards,
            team_control_at_reception,
            role_defender: (player.role == PlayerRole::Defender) as i32 as f64,
            role_midfielder: (player.role == PlayerRole::Midfielder) as i32 as f64,
            role_forward: (player.role == PlayerRole::Forward) as i32 as f64,
        })
    }

    /// The model's signed **approach bias** in `(-1, 1)` for `player` meeting the ball at
    /// `reception_target` (positive = step toward the ball). The trained head once it has
    /// learned enough, else the analytic seed. `None` when the model is gated off or the
    /// inputs are degenerate.
    pub(crate) fn receive_approach_bias(
        &self,
        player: &PlayerSnapshot,
        current: Vec2,
        reception_target: Vec2,
    ) -> Option<f64> {
        if !receive_approach_model_enabled() {
            return None;
        }
        let inputs = self.build_receive_approach_inputs(player, current, reception_target)?;
        let bias = self
            .receive_approach_head
            .as_ref()
            .filter(|head| head.training_steps() >= RECEIVE_APPROACH_HEAD_MIN_TRAINING_STEPS)
            .and_then(|head| head.predict(&inputs))
            .unwrap_or_else(|| analytic_receive_approach_bias(&inputs));
        bias.is_finite().then(|| bias.clamp(-1.0, 1.0))
    }

    /// The elected reception point shifted **along the line to the ball** by the learned
    /// approach bias: a positive bias steps the meeting point toward the ball (meet it
    /// early, deny an opponent), a negative bias eases it back (let it run into space).
    /// The toward-ball shift never overshoots the ball. Returns `reception_target`
    /// unchanged (without building features) unless the model is enabled, keeping the
    /// default reception election byte-identical.
    pub(crate) fn receive_approach_adjusted_target(
        &self,
        player: &PlayerSnapshot,
        current: Vec2,
        reception_target: Vec2,
    ) -> Vec2 {
        let Some(bias) = self.receive_approach_bias(player, current, reception_target) else {
            return reception_target;
        };
        let to_ball = self.ball.position - reception_target;
        let len = to_ball.len();
        if len <= 1e-3 {
            return reception_target;
        }
        let dir = to_ball * (1.0 / len);
        let raw = bias * RECEIVE_APPROACH_MAX_SHIFT_YARDS;
        // Toward the ball: cap so the meeting point never steps past the ball itself.
        // Away from the ball: the modest negative shift, no clamp needed.
        let shift = if raw >= 0.0 {
            raw.min((len - 0.3).max(0.0))
        } else {
            raw
        };
        (reception_target + dir * shift).clamp_to_pitch(self.field_width, self.field_length)
    }
}

impl SoccerMatch {
    /// Gated per-tick RL sample collection for the receive-approach head. A **no-op**
    /// (zero cost, byte-identical) unless the model is enabled. When on: resolves
    /// decisions whose reward window has elapsed (reward = the receiving team's
    /// territorial-advantage change over the window), and samples fresh per-receiver
    /// approach decisions on cadence whenever a pass is in flight.
    pub(crate) fn collect_receive_approach_rl_samples(&mut self, snapshot: &WorldSnapshot) {
        if !receive_approach_model_enabled() {
            return;
        }
        let tick = self.tick;
        // Resolve due decisions.
        let mut i = 0;
        while i < self.pending_receive_approach.len() {
            if self.pending_receive_approach[i].due_tick <= tick {
                let decision = self.pending_receive_approach.swap_remove(i);
                let now_territorial = territorial_advantage(snapshot, decision.team);
                if now_territorial.is_finite() && decision.decision_territorial.is_finite() {
                    self.receive_approach_samples.push(ReceiveApproachSample {
                        inputs: decision.inputs,
                        action_bias: decision.action_bias,
                        reward: now_territorial - decision.decision_territorial,
                    });
                    if self.receive_approach_samples.len() > RECEIVE_APPROACH_SAMPLE_CAP {
                        let overflow =
                            self.receive_approach_samples.len() - RECEIVE_APPROACH_SAMPLE_CAP;
                        self.receive_approach_samples.drain(0..overflow);
                    }
                }
            } else {
                i += 1;
            }
        }
        // Sample fresh per-receiver decisions on cadence while a pass is in flight (the
        // only time the receive-approach choice is live).
        if snapshot.ball.holder.is_some() || snapshot.pending_pass.is_none() {
            return;
        }
        if tick % RECEIVE_APPROACH_SAMPLE_INTERVAL_TICKS != 0 {
            return;
        }
        let candidate_ids: Vec<usize> = snapshot
            .players
            .iter()
            .filter(|p| p.role != PlayerRole::Goalkeeper)
            .map(|p| p.id)
            .collect();
        for id in candidate_ids {
            // Only the elected receiver(s) of the live pass face this decision.
            let Some((target, _sprint)) = snapshot.pending_pass_reception_target_for(id) else {
                continue;
            };
            let Some(player) = snapshot.players.iter().find(|p| p.id == id) else {
                continue;
            };
            let current = snapshot.player_snapshot_position(player);
            let Some(inputs) = snapshot.build_receive_approach_inputs(player, current, target)
            else {
                continue;
            };
            // The approach action the engine used this tick (head once trained, else seed).
            let action_bias = snapshot
                .receive_approach_bias(player, current, target)
                .unwrap_or_else(|| analytic_receive_approach_bias(&inputs));
            let territorial = territorial_advantage(snapshot, player.team);
            if territorial.is_finite() {
                self.pending_receive_approach
                    .push(PendingReceiveApproachDecision {
                        team: player.team,
                        player_id: id,
                        inputs,
                        action_bias,
                        decision_territorial: territorial,
                        due_tick: tick + RECEIVE_APPROACH_REWARD_WINDOW_TICKS,
                    });
            }
        }
    }

    /// Drain the collected approach RL samples for the cluster learner (train the head +
    /// persist). Mirrors [`Self::drain_loose_ball_commit_samples`].
    pub fn drain_receive_approach_samples(&mut self) -> Vec<ReceiveApproachSample> {
        std::mem::take(&mut self.receive_approach_samples)
    }

    /// Install the trained approach head for live consumption. The learner carries +
    /// trains it across games and re-installs it each game; wrapped in `Arc` so every
    /// per-tick snapshot shares it cheaply.
    pub fn set_receive_approach_head(&mut self, head: ReceiveApproachHead) {
        self.receive_approach_head = Some(std::sync::Arc::new(head));
    }
}

#[cfg(test)]
mod receive_approach_tests {
    use super::*;

    fn baseline_inputs() -> ReceiveApproachInputs {
        ReceiveApproachInputs {
            field_length: 120.0,
            field_width: 80.0,
            dist_to_ball: 10.0,
            ball_speed: 8.0,
            ball_closing_speed: 7.0,
            ball_accel_toward_me: -1.5,
            ball_jerk_toward_me: 0.0,
            player_speed_toward_ball: 2.0,
            player_accel_toward_ball: 1.0,
            player_jerk_toward_ball: 0.0,
            opponent_race_margin: 0.4,
            nearest_opp_dist_to_ball: 9.0,
            reception_pressure: 6.0,
            forward_reception: 2.0,
            ball_altitude: 0.0,
            team_control_at_reception: 0.55,
            role_defender: 0.0,
            role_midfielder: 1.0,
            role_forward: 0.0,
        }
    }

    #[test]
    fn features_are_finite_bounded_and_right_dim() {
        let f = baseline_inputs().to_features();
        assert_eq!(f.len(), RECEIVE_APPROACH_FEATURE_DIM);
        for v in f {
            assert!(v.is_finite());
            assert!(v.abs() <= 2.0 + 1e-9, "feature {v} out of normalized range");
        }
    }

    #[test]
    fn analytic_bias_is_a_valid_signed_unit() {
        let b = analytic_receive_approach_bias(&baseline_inputs());
        assert!((-1.0..=1.0).contains(&b), "bias {b} out of [-1,1]");
    }

    #[test]
    fn opponent_first_steps_toward_ball_more_than_a_free_receiver() {
        // Opponent clearly reaches the reception first AND the receipt is tight.
        let mut contested = baseline_inputs();
        contested.opponent_race_margin = -0.8;
        contested.reception_pressure = 2.0;
        contested.nearest_opp_dist_to_ball = 3.0;
        // Genuinely free: race won by a mile, acres of space.
        let mut free = baseline_inputs();
        free.opponent_race_margin = 1.6;
        free.reception_pressure = 12.0;
        free.nearest_opp_dist_to_ball = 16.0;
        let contested_bias = analytic_receive_approach_bias(&contested);
        let free_bias = analytic_receive_approach_bias(&free);
        assert!(
            contested_bias > free_bias,
            "a contested receiver should step toward the ball more than a free one: \
             contested {contested_bias} vs free {free_bias}"
        );
        // And the coach's prior: when the opponent gets there first, actively attack the ball.
        assert!(
            contested_bias > 0.0,
            "an opponent-first receiver should bias TOWARD the ball, got {contested_bias}"
        );
    }

    #[test]
    fn a_totally_free_receiver_may_ease_off_the_ball() {
        let mut free = baseline_inputs();
        free.opponent_race_margin = 2.0;
        free.reception_pressure = 14.0;
        free.nearest_opp_dist_to_ball = 18.0;
        free.ball_accel_toward_me = 0.0;
        free.player_speed_toward_ball = 0.0;
        let b = analytic_receive_approach_bias(&free);
        assert!(
            b < 0.0,
            "a totally free receiver with space behind may ease off and let it come, got {b}"
        );
    }

    #[test]
    fn tighter_pressure_raises_the_toward_ball_bias() {
        let mut loose = baseline_inputs();
        loose.reception_pressure = 10.0;
        let mut tight = baseline_inputs();
        tight.reception_pressure = 1.5;
        assert!(
            analytic_receive_approach_bias(&tight) > analytic_receive_approach_bias(&loose),
            "tighter receipt pressure should pull the receiver harder onto the ball"
        );
    }

    #[test]
    fn head_predicts_a_bounded_signed_bias() {
        let head = ReceiveApproachHead::new(11);
        let b = head.predict(&baseline_inputs()).expect("finite prediction");
        assert!((-1.0..=1.0).contains(&b), "head bias {b} out of (-1,1)");
    }

    #[test]
    fn reward_weighted_training_steers_toward_the_high_reward_action() {
        let mut head = ReceiveApproachHead::new(29);
        let inputs = baseline_inputs();
        let before = head.predict(&inputs).expect("finite");
        // Same state, two competing actions: stepping toward the ball (+0.8) is rewarded,
        // easing off (−0.8) is penalized. RWR should pull the head toward stepping in.
        let samples: Vec<ReceiveApproachSample> = (0..32)
            .flat_map(|_| {
                [
                    ReceiveApproachSample {
                        inputs: inputs.clone(),
                        action_bias: 0.8,
                        reward: 1.0,
                    },
                    ReceiveApproachSample {
                        inputs: inputs.clone(),
                        action_bias: -0.8,
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
            after > before && after > 0.0,
            "RWR should move the head toward the high-reward step-in action: {before} -> {after}"
        );
    }

    #[test]
    fn train_entry_skips_empty_and_trains_valid() {
        assert!(train_receive_approach_head(&[], 1, 4, 0.02).is_none());
        let samples: Vec<ReceiveApproachSample> = (0..16)
            .map(|i| ReceiveApproachSample {
                inputs: baseline_inputs(),
                action_bias: if i % 2 == 0 { 0.8 } else { -0.8 },
                reward: if i % 2 == 0 { 1.0 } else { -1.0 },
            })
            .collect();
        let (_, report) =
            train_receive_approach_head(&samples, 1, 4, 0.02).expect("trains a usable corpus");
        assert_eq!(report.samples, 16);
        assert_eq!(report.epochs, 4);
        assert!(report.training_steps > 0 && report.final_loss.is_finite());
    }
}
