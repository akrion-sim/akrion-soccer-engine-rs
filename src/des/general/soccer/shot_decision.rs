//! Learnable **shot-trigger MDP/POMDP** — *when* an attacker should pull the
//! trigger — plus the small derived signals the decision reads.
//!
//! ## The problem this fixes
//!
//! The engine's hand-tuned shot-trigger stack is anchored on
//! `TEAMMATE_MUST_SHOOT_YARDS = 25.0` (and its aliases), so strikers volunteer
//! shots from ~25yd. But the *reward* only fully credits a shot inside
//! `SHOT_FULL_REWARD_DISTANCE_YARDS = 16` and tapers to **zero** by
//! `SHOT_DISTANCE_REWARD_PIVOT_YARDS = 20` (`shot_reward_distance_scale`). The
//! decision layer and the reward layer disagree by ~5-9yd — exactly the "shoots
//! from 25 when he should work it to 20/15" complaint.
//!
//! This module is the seam through which "should I shoot NOW vs. hold / drive
//! closer / pass" becomes a learnable **value function** of the real POMDP state
//! the user specified:
//!   - GK position / **% of net available** ([`open_goal_fraction`]),
//!   - shooting angle,
//!   - perceived block likelihood,
//!   - perceived **rebound / ricochet → forced-error second chance**
//!     ([`shot_rebound_second_chance_score`]),
//!   - ball x/y position, velocity, acceleration,
//!   - **body mechanics** — can the ball actually be struck properly (the MPC's
//!     `body_mechanics_fit`, fed in so the value head is coupled to executability).
//!
//! ## Closed loop with the MPC
//!
//! The shot decision is two-stage, not one-shot: the POMDP *proposes* (this value
//! head), then the shot MPC *checks execution* — `body_mechanics_fit`, foot choice,
//! `execution_probability` — and may **veto** (the existing `reselect_reason`
//! plumbing), sending control back to the decision layer to pick the next-best
//! action. `body_mechanics_fit` is both an input here AND the runtime backstop.
//!
//! ## Parity
//!
//! Default-**ON** behind a kill-switch (`DD_SOCCER_DISABLE_SHOT_TRIGGER_MDP`). When
//! the kill-switch is set, [`WorldSnapshot::shot_trigger_mdp_value`] returns `None`
//! and the live decision never reads the model ⇒ byte-identical to before. The
//! analytic seed and feature build are pure / RNG-free so tests + the inspector can
//! read them without the live seam.

use super::*;
use crate::des::general::neural_network::{ActivationName, FeedForwardNetwork, RandomNetworkSpec};
use crate::des::general::prng::mulberry32;

/// Number of features in the shot-trigger state vector. Ordering is defined once in
/// [`ShotTriggerInputs::to_features`] — change it and bump this + any persisted head.
pub const SHOT_TRIGGER_FEATURE_DIM: usize = 22;
/// Hidden width of the learned value head.
pub const SHOT_TRIGGER_HIDDEN_UNITS: usize = 24;
/// How often (ticks) a shot-trigger RL decision is sampled while the model is on.
pub const SHOT_TRIGGER_SAMPLE_INTERVAL_TICKS: u64 = 6;
/// Window (ticks) over which a sampled decision's reward is measured.
pub const SHOT_TRIGGER_REWARD_WINDOW_TICKS: u64 = 45;
/// Cap on the rolling RL sample buffer (drained by the learner).
pub const SHOT_TRIGGER_SAMPLE_CAP: usize = 4096;
/// Minimum training steps before a trained head is consumed live; below this the net
/// is too raw, so the decision falls back to the analytic seed.
pub const SHOT_TRIGGER_HEAD_MIN_TRAINING_STEPS: usize = 200;
/// Sentinel for `SoccerPomdpObservation::shot_trigger_mdp_value` meaning "not computed"
/// (the model is gated off, or the observation was built directly in a test without the
/// snapshot seam). The decision layer falls back to the analytic seed when it sees this.
pub const SHOT_TRIGGER_MDP_VALUE_UNSET: f64 = -1.0;

/// serde default for the observation's `shot_trigger_mdp_value` field.
pub fn shot_trigger_mdp_value_unset() -> f64 {
    SHOT_TRIGGER_MDP_VALUE_UNSET
}

/// Reference distance (yd) used to normalize distance features.
const REF_GOAL_DISTANCE_YARDS: f64 = 36.0;
/// Reference shooting half-angle (deg) — a wide-open central chance.
const REF_SHOT_ANGLE_DEGREES: f64 = 42.0;
/// Reference speed/accel for ball kinematics normalization.
const REF_BALL_SPEED_YPS: f64 = 24.0;
const REF_BALL_ACCEL_YPS2: f64 = 12.0;
/// Max skill rating (0-10), used to normalize per-foot power / shooting skill to `[0,1]`.
const MAX_SKILL_RATING: f64 = 10.0;

// --- Analytic shot-value distance curve (`analytic_shot_trigger_value`) ---
/// Inside this distance a shot keeps full distance value (1.0).
const SHOT_VALUE_FULL_DISTANCE_YARDS: f64 = 18.0;
/// End of the "20 or 15, not 25" taper band; distance value is `SHOT_VALUE_TAPER_FLOOR` here.
/// (Bare distance value then eases to 0 over the next 8yd, i.e. by ~30yd.)
const SHOT_VALUE_TAPER_DISTANCE_YARDS: f64 = 22.0;
/// Distance value at [`SHOT_VALUE_TAPER_DISTANCE_YARDS`] (before any long-range merit lift).
const SHOT_VALUE_TAPER_FLOOR: f64 = 0.45;

// --- Shot-score multiplier (`shot_trigger_score_multiplier`) ---
/// Soft multiplier on `shot_score` at value 0 (a far/covered chance is demoted toward this).
const SHOT_TRIGGER_MULT_FLOOR: f64 = 0.10;
/// Slope of the value→multiplier map.
const SHOT_TRIGGER_MULT_SLOPE: f64 = 1.20;
/// Ceiling of the value→multiplier map (a close/open chance is lifted toward this).
const SHOT_TRIGGER_MULT_CEILING: f64 = 1.30;

// --- RL sample collection (`collect_shot_trigger_rl_samples`) ---
/// Only sample shot decisions inside this goal distance (else the corpus is midfield carriers).
const SHOT_TRIGGER_SAMPLE_MAX_GOAL_DISTANCE_YARDS: f64 = 32.0;
/// The sampled candidate must be this close to the ball to be the one deciding to shoot.
const SHOT_TRIGGER_SAMPLE_BALL_PROXIMITY_YARDS: f64 = 4.0;
/// Analytic value at/above which the sampled action is recorded as "a shot was taken".
const SHOT_TRIGGER_SAMPLE_SHOOT_ACTION_THRESHOLD: f64 = 0.5;

/// Kill-switch: set to a truthy value to DISABLE the learnable shot trigger and keep
/// the engine byte-identical to before this module existed. Default (unset) = ON.
const SHOT_TRIGGER_MDP_DISABLE_ENV: &str = "DD_SOCCER_DISABLE_SHOT_TRIGGER_MDP";
/// Kill-switch for the MPC foot-choice (see `player.rs` shot block). Default = ON.
const SHOT_FOOT_MPC_DISABLE_ENV: &str = "DD_SOCCER_DISABLE_SHOT_FOOT_MPC";

fn env_flag_disabled(name: &str) -> bool {
    std::env::var(name)
        .map(|raw| {
            let v = raw.trim().to_ascii_lowercase();
            matches!(v.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

/// Whether the learnable shot-trigger MDP/POMDP is live this process (default ON).
pub fn shot_trigger_mdp_enabled() -> bool {
    !env_flag_disabled(SHOT_TRIGGER_MDP_DISABLE_ENV)
}

/// Whether the MPC shot foot-choice is live this process (default ON).
pub fn shot_foot_mpc_enabled() -> bool {
    !env_flag_disabled(SHOT_FOOT_MPC_DISABLE_ENV)
}

/// **% of the goal mouth available** given the keeper's position — the literal
/// "GK position / how much net is open" the user asked for. Pure function of the
/// observation: the shooting angle sets the angular width the shooter sees, and the
/// keeper's coverage (closeness on the angle + how far he is out of position) eats
/// into it. `1.0` = empty net, `0.0` = fully covered. Distance-agnostic.
pub fn open_goal_fraction(observation: &SoccerPomdpObservation) -> f64 {
    // Angular width the shooter sees (a wide central chance ⇒ near 1).
    let angle_width =
        (observation.opponent_goal_angle_degrees / REF_SHOT_ANGLE_DEGREES).clamp(0.0, 1.0);
    // Keeper coverage: a keeper sitting ON the shot angle (small angle to him) and
    // ON his line covers the most; one out of position / off the angle covers least.
    let keeper_angle_off = (observation.opposing_goalkeeper_angle_degrees.abs()
        / REF_SHOT_ANGLE_DEGREES)
        .clamp(0.0, 1.0);
    let out_of_position = observation
        .opposing_goalkeeper_out_of_position
        .clamp(0.0, 1.0);
    // Net left open by the keeper = how far he is off the shot line, lifted when he's
    // caught out of position. Covered (on the angle, in position) ⇒ near 0.
    let keeper_open = (0.30 + 0.70 * keeper_angle_off)
        .clamp(0.0, 1.0)
        .max(out_of_position);
    (angle_width * keeper_open).clamp(0.0, 1.0)
}

/// Perceived **rebound / ricochet → forced-error second-chance** value: a firm shot
/// the keeper can only parry (rather than hold) that a supporting attacker can pounce
/// on. Pure function of the observation: parry likelihood rises with shot power and a
/// keeper who is hard to beat cleanly (so he blocks rather than is beaten), and the
/// second chance only matters if we have runners up to follow it in.
pub fn shot_rebound_second_chance_score(observation: &SoccerPomdpObservation) -> f64 {
    let power = observation
        .skill_right_foot_shot_power
        .max(observation.skill_left_foot_shot_power);
    let power01 = (power / MAX_SKILL_RATING).clamp(0.0, 1.0);
    // A keeper who gets a hand to it (low beat-probability) but is reachable parries
    // a powerful strike; a wide-open net is just a goal, not a rebound.
    let stop_likelihood = (1.0 - observation.shot_beat_goalkeeper_probability.clamp(0.0, 1.0))
        * observation.shot_on_frame_probability.clamp(0.0, 1.0);
    let parry = (power01 * 0.55 + 0.25) * stop_likelihood;
    // Support to pounce: forward runners / open support outlets ahead.
    let support = ((observation.teammates_ahead.min(3) as f64) / 3.0 * 0.6
        + (observation.open_support_outlets.min(3) as f64) / 3.0 * 0.4)
        .clamp(0.0, 1.0);
    (parry * (0.35 + 0.65 * support)).clamp(0.0, 1.0)
}

/// The shot-trigger POMDP state, in the shooter's attacking frame (all signals are
/// already goal-relative in the observation, so the struct is mirror-invariant).
/// Plain struct so it logs as a training row, asserts in tests, and is surfaced by
/// the inspector independently of the network.
#[derive(Clone, Debug, PartialEq)]
pub struct ShotTriggerInputs {
    // Geometry.
    pub yards_to_goal: f64,
    pub shot_angle_degrees: f64,
    pub open_goal_fraction: f64,
    // Shot-quality perception.
    pub on_frame_probability: f64,
    pub beat_goalkeeper_probability: f64,
    pub block_probability: f64,
    pub blocker_distance_yards: f64,
    pub shot_lane_open: bool,
    pub rebound_second_chance: f64,
    // Keeper.
    pub keeper_distance_yards: f64,
    pub keeper_out_of_position: f64,
    // Ball / actor kinematics.
    pub ball_forward_velocity_yps: f64,
    pub ball_acceleration_yps2: f64,
    pub actor_acceleration_yps2: f64,
    // Pressure / urgency.
    pub nearest_opponent_distance_yards: f64,
    pub offensive_urgency: f64,
    pub decision_urgency: f64,
    pub goal_attack_window: f64,
    // Executability: can the ball be struck properly NOW (the MPC body-mechanics fit,
    // 1.0 when unknown at pure-decision time — the runtime MPC veto is the backstop).
    pub body_mechanics_fit: f64,
    // Shooter.
    pub shooting_skill: f64,
    pub strong_foot_power: f64,
    pub is_forward: bool,
}

impl ShotTriggerInputs {
    /// Build from a shooter's observation + role + shooting skill. `body_mechanics_fit`
    /// is supplied separately (the MPC computes it at sample time; the pure decision
    /// path passes `1.0`, leaning on the runtime MPC veto for executability).
    pub fn from_observation(
        observation: &SoccerPomdpObservation,
        role: PlayerRole,
        shooting_skill: f64,
        body_mechanics_fit: f64,
    ) -> Self {
        ShotTriggerInputs {
            yards_to_goal: observation.yards_to_goal,
            shot_angle_degrees: observation.opponent_goal_angle_degrees,
            open_goal_fraction: open_goal_fraction(observation),
            on_frame_probability: observation.shot_on_frame_probability,
            beat_goalkeeper_probability: observation.shot_beat_goalkeeper_probability,
            block_probability: observation.shot_block_probability,
            blocker_distance_yards: observation.shot_blocker_distance_yards,
            shot_lane_open: observation.shot_lane_open,
            rebound_second_chance: shot_rebound_second_chance_score(observation),
            keeper_distance_yards: observation.opposing_goalkeeper_distance,
            keeper_out_of_position: observation.opposing_goalkeeper_out_of_position,
            ball_forward_velocity_yps: observation.ball_forward_velocity_yps,
            ball_acceleration_yps2: observation.ball_acceleration_yps2,
            actor_acceleration_yps2: observation.actor_acceleration_yps2,
            nearest_opponent_distance_yards: observation.nearest_opponent_distance,
            offensive_urgency: observation.offensive_urgency,
            decision_urgency: observation.decision_urgency,
            goal_attack_window: observation.goal_attack_window_score,
            body_mechanics_fit,
            shooting_skill,
            strong_foot_power: observation
                .skill_right_foot_shot_power
                .max(observation.skill_left_foot_shot_power),
            is_forward: role == PlayerRole::Forward,
        }
    }

    /// The normalized network input vector. **Single source of truth** for the layout.
    pub fn to_features(&self) -> [f64; SHOT_TRIGGER_FEATURE_DIM] {
        let dist = |d: f64| (d / REF_GOAL_DISTANCE_YARDS).clamp(0.0, 1.5);
        let ang = |a: f64| (a / REF_SHOT_ANGLE_DEGREES).clamp(0.0, 1.5);
        let p = |v: f64| v.clamp(0.0, 1.0);
        let vel = |v: f64| (v / REF_BALL_SPEED_YPS).clamp(-2.0, 2.0);
        let acc = |a: f64| (a / REF_BALL_ACCEL_YPS2).clamp(-2.0, 2.0);
        let b = |flag: bool| if flag { 1.0 } else { 0.0 };
        [
            dist(self.yards_to_goal),
            ang(self.shot_angle_degrees),
            p(self.open_goal_fraction),
            p(self.on_frame_probability),
            p(self.beat_goalkeeper_probability),
            p(self.block_probability),
            dist(self.blocker_distance_yards),
            b(self.shot_lane_open),
            p(self.rebound_second_chance),
            dist(self.keeper_distance_yards),
            p(self.keeper_out_of_position),
            vel(self.ball_forward_velocity_yps),
            acc(self.ball_acceleration_yps2),
            acc(self.actor_acceleration_yps2),
            dist(self.nearest_opponent_distance_yards),
            p(self.offensive_urgency),
            p(self.decision_urgency),
            p(self.goal_attack_window),
            p(self.body_mechanics_fit),
            p(self.shooting_skill),
            (self.strong_foot_power / MAX_SKILL_RATING).clamp(0.0, 1.0),
            b(self.is_forward),
        ]
    }
}

/// Analytic, deterministic, RNG-free **seed value** of shooting NOW, in `[0, 1]`.
/// This is the bootstrap target + live fallback the learned [`ShotTriggerHead`] is
/// meant to first imitate, then beat.
///
/// Anchored to the **reward** geometry (not the 25yd heuristic): full value inside
/// ~12yd, smoothly down through the 16→20yd taper, near-zero by 22-25yd — then
/// *lifted back* for a genuine long-range chance (open net / keeper out of position
/// / live rebound), so a real worldie still gets taken. Quality gates (angle, block,
/// lane, executability) multiply it down.
pub fn analytic_shot_trigger_value(inputs: &ShotTriggerInputs) -> f64 {
    let d = inputs.yards_to_goal.max(0.0);
    // Distance base: full value inside ~18yd, ramping down through the 18→22yd band (the
    // "20 or 15, not 25" zone) and toward zero by 30yd. Tuned so a clean 15-20yd chance
    // stays high while a 25yd+ shot is low — the discipline anchor.
    // Full (1.0) inside FULL; eases to SHOT_VALUE_TAPER_FLOOR over the 4yd FULL→TAPER band
    // (slope 0.55 = 1.0 − floor); then to 0 over the 8yd TAPER→ZERO band (slope = the floor).
    // Slope/span literals are kept verbatim so the curve is bit-for-bit the original.
    let distance_base = if d <= SHOT_VALUE_FULL_DISTANCE_YARDS {
        1.0
    } else if d <= SHOT_VALUE_TAPER_DISTANCE_YARDS {
        1.0 - (d - SHOT_VALUE_FULL_DISTANCE_YARDS) / 4.0 * 0.55
    } else {
        (SHOT_VALUE_TAPER_FLOOR - (d - SHOT_VALUE_TAPER_DISTANCE_YARDS) / 8.0 * 0.45).max(0.0)
    };

    // A genuine long-range chance lifts the far tail back up: an open net / stranded
    // keeper, a live rebound to pounce on. Bounded so it can rescue a 22-26yd shot but
    // never makes distance irrelevant.
    let long_range_merit = (inputs.open_goal_fraction * 0.55
        + inputs.keeper_out_of_position * 0.30
        + inputs.rebound_second_chance * 0.25)
        .clamp(0.0, 1.0);
    let distance_value =
        (distance_base + (1.0 - distance_base) * long_range_merit * 0.6).clamp(0.0, 1.0);

    // Two hard geometric gates (a near-impossible angle or a closed/blocked lane really do
    // kill a shot) ...
    let angle_fit = (inputs.shot_angle_degrees / REF_SHOT_ANGLE_DEGREES).clamp(0.0, 1.0);
    let angle_gate = (0.45 + 0.55 * angle_fit).clamp(0.0, 1.0);
    let lane_gate = if inputs.shot_lane_open { 1.0 } else { 0.55 };

    // ... plus ONE soft quality term (additive, not a compounding product — the old
    // multiplicative stack crushed even clean 18yd chances toward zero). On-frame and a
    // beatable/exposed keeper lift; a likely block pulls down; body mechanics scale it.
    let keeper_exposure = inputs
        .beat_goalkeeper_probability
        .max(inputs.open_goal_fraction)
        .max(inputs.keeper_out_of_position)
        .max(inputs.rebound_second_chance)
        .clamp(0.0, 1.0);
    let quality =
        (0.55 + 0.30 * inputs.on_frame_probability.clamp(0.0, 1.0) + 0.20 * keeper_exposure
            - 0.35 * inputs.block_probability.clamp(0.0, 1.0))
        .clamp(0.25, 1.0);
    let mechanics = (0.55 + 0.45 * inputs.body_mechanics_fit.clamp(0.0, 1.0)).clamp(0.0, 1.0);

    (distance_value * angle_gate * lane_gate * quality * mechanics).clamp(0.0, 1.0)
}

/// Map the shot-trigger value `[0,1]` to a soft multiplier on the engine's `shot_score`
/// (and the forced-path commitment chances). A low-value (far / covered) shot is demoted
/// toward `~0.10×`, a high-value close/open chance lifted toward `~1.30×`. Deliberately
/// *linear* and not crushing in the mid-band: a clean 18-20yd chance (value ~0.6-0.8) keeps
/// near-full score so the striker still shoots from range it *should*. The hard ">22yd
/// unless the value is real" cut-off is done separately by the forced-path veto, so this
/// term only needs to nudge, not gate.
pub fn shot_trigger_score_multiplier(value: f64) -> f64 {
    let v = value.clamp(0.0, 1.0);
    (SHOT_TRIGGER_MULT_FLOOR + SHOT_TRIGGER_MULT_SLOPE * v)
        .clamp(SHOT_TRIGGER_MULT_FLOOR, SHOT_TRIGGER_MULT_CEILING)
}

/// One reward-weighted RL row for the shot-trigger head: the state at a shot decision,
/// whether a shot was actually taken (the action, `1.0`/`0.0`), and the reward over the
/// following window (goal / on-target-close = positive, wasted long shot = negative,
/// held-into-a-better-chance = positive). Mirrors [`super::back_four_line::LineDepthSample`].
#[derive(Clone, Debug, PartialEq)]
pub struct ShotTriggerSample {
    pub inputs: ShotTriggerInputs,
    /// `1.0` = a shot was taken at the decision, `0.0` = held.
    pub action_shoot: f64,
    /// Reward attributed to the decision over the window (any scale; baselined by the trainer).
    pub reward: f64,
}

/// An open shot decision awaiting its windowed reward. Mirrors
/// [`super::back_four_line::PendingLineDepthDecision`].
#[derive(Clone, Debug, PartialEq)]
pub struct PendingShotTriggerDecision {
    pub team: Team,
    pub player_id: usize,
    pub inputs: ShotTriggerInputs,
    pub action_shoot: f64,
    pub decision_pitch_value: f64,
    pub due_tick: u64,
}

/// Learned value head for the shot trigger: a `FeedForwardNetwork` (`DIM → hidden → 1`,
/// sigmoid). Round-trips through the engine's `SoccerNeuralNetworkSnapshot` like the
/// other heads. Mirrors [`super::back_four_line::BackFourLineHead`].
#[derive(Clone, Debug)]
pub struct ShotTriggerHead {
    network: FeedForwardNetwork,
    training_steps: usize,
    last_loss: Option<f64>,
}

impl ShotTriggerHead {
    pub fn new(seed: u32) -> Self {
        let mut rng = mulberry32(seed ^ 0x51A0_7E33);
        let network = FeedForwardNetwork::random(
            &RandomNetworkSpec {
                input_dim: SHOT_TRIGGER_FEATURE_DIM,
                hidden_layers: vec![SHOT_TRIGGER_HIDDEN_UNITS],
                output_dim: 1,
                hidden_activation: ActivationName::Tanh,
                output_activation: ActivationName::Sigmoid,
                weight_scale: None,
            },
            &mut rng,
        );
        ShotTriggerHead {
            network,
            training_steps: 0,
            last_loss: None,
        }
    }

    /// Predicted shot-trigger value in `(0, 1)`, or `None` on a malformed input.
    pub fn predict(&self, inputs: &ShotTriggerInputs) -> Option<f64> {
        let features = inputs.to_features();
        if features.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let out = self.network.predict(&features[..]);
        out.first().copied().filter(|p| p.is_finite())
    }

    /// One supervised SGD epoch toward `(features, target_value)` pairs (analytic-seed
    /// bootstrap). Returns the mean step loss.
    pub fn train(&mut self, samples: &[(ShotTriggerInputs, f64)], learning_rate: f64) -> f64 {
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
        let mean = if applied > 0 {
            total / applied as f64
        } else {
            0.0
        };
        self.last_loss = Some(mean);
        mean
    }

    /// One **reward-weighted-regression** epoch over RL samples: regress the head toward
    /// each sample's ACTION (shoot/held), weighted by how much its reward beat the batch
    /// baseline — good shot decisions are reinforced, bad ones barely. Mirrors
    /// [`super::back_four_line::BackFourLineHead::train_reward_weighted`].
    pub fn train_reward_weighted(
        &mut self,
        samples: &[ShotTriggerSample],
        learning_rate: f64,
    ) -> f64 {
        let finite: Vec<&ShotTriggerSample> = samples
            .iter()
            .filter(|s| s.reward.is_finite() && s.action_shoot.is_finite())
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
            let target = [s.action_shoot.clamp(0.0, 1.0)];
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

/// Outcome of training a shot-trigger head on a drained RL corpus, logged by the
/// learning runner. Mirrors `LineDepthTrainingReport`.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ShotTriggerTrainingReport {
    pub samples: usize,
    pub training_steps: usize,
    pub epochs: usize,
    pub final_loss: f64,
}

/// Load-and-train entry for a shot-trigger head over a drained RL corpus. `None` if no
/// usable samples. Mirrors `train_line_depth_head`.
pub fn train_shot_trigger_head(
    samples: &[ShotTriggerSample],
    seed: u32,
    epochs: usize,
    learning_rate: f64,
) -> Option<(ShotTriggerHead, ShotTriggerTrainingReport)> {
    let usable = samples
        .iter()
        .filter(|s| s.reward.is_finite() && s.action_shoot.is_finite())
        .count();
    if usable == 0 || epochs == 0 {
        return None;
    }
    let mut head = ShotTriggerHead::new(seed);
    let mut final_loss = 0.0;
    for _ in 0..epochs {
        final_loss = head.train_reward_weighted(samples, learning_rate);
    }
    let report = ShotTriggerTrainingReport {
        samples: usable,
        training_steps: head.training_steps(),
        epochs,
        final_loss,
    };
    Some((head, report))
}

/// Public wrapper returning ONLY the report (the head type is engine-internal). Mirrors
/// `report_line_depth_training`.
pub fn report_shot_trigger_training(
    samples: &[ShotTriggerSample],
    seed: u32,
    epochs: usize,
    learning_rate: f64,
) -> Option<ShotTriggerTrainingReport> {
    train_shot_trigger_head(samples, seed, epochs, learning_rate).map(|(_, report)| report)
}

impl WorldSnapshot {
    /// The shot-trigger MDP/POMDP value in `[0, 1]` for `player_id` given his
    /// `observation` — *how worth it is shooting NOW*. `None` when the model is gated
    /// off (kill-switch) ⇒ the caller keeps the legacy heuristic (byte-identical).
    ///
    /// Consumes the trained head once it has learned enough (rides the snapshot like the
    /// other heads); otherwise the analytic seed. `body_mechanics_fit` is unknown at this
    /// pure-decision point, so `1.0` is used here and the runtime MPC veto stays the
    /// executability backstop.
    pub(crate) fn shot_trigger_mdp_value(
        &self,
        observation: &SoccerPomdpObservation,
        role: PlayerRole,
        shooting_skill: f64,
    ) -> Option<f64> {
        if !shot_trigger_mdp_enabled() {
            return None;
        }
        let inputs = ShotTriggerInputs::from_observation(observation, role, shooting_skill, 1.0);
        let value = self
            .shot_trigger_head
            .as_ref()
            .filter(|head| head.training_steps() >= SHOT_TRIGGER_HEAD_MIN_TRAINING_STEPS)
            .and_then(|head| head.predict(&inputs))
            .unwrap_or_else(|| analytic_shot_trigger_value(&inputs));
        value.is_finite().then(|| value.clamp(0.0, 1.0))
    }
}

impl SoccerMatch {
    /// Gated per-tick RL sample collection for the shot-trigger head, driven off the
    /// already-built per-tick learning `snapshot`. A **no-op** (byte-identical) unless
    /// the model is enabled. Resolves decisions whose reward window has elapsed (reward =
    /// the attacker's pitch-value change over the window — shooting well or working into a
    /// better chance both score), and samples a fresh decision for the ball holder in a
    /// shootable state every [`SHOT_TRIGGER_SAMPLE_INTERVAL_TICKS`]. Drained by the
    /// learner via [`Self::drain_shot_trigger_samples`].
    pub(crate) fn collect_shot_trigger_rl_samples(&mut self, snapshot: &WorldSnapshot) {
        if !shot_trigger_mdp_enabled() {
            return;
        }
        let tick = self.tick;
        let mut i = 0;
        while i < self.pending_shot_trigger.len() {
            if self.pending_shot_trigger[i].due_tick <= tick {
                let decision = self.pending_shot_trigger.swap_remove(i);
                let now_value =
                    shot_decision_pitch_value(snapshot, decision.team, decision.player_id);
                if now_value.is_finite() && decision.decision_pitch_value.is_finite() {
                    self.shot_trigger_samples.push(ShotTriggerSample {
                        inputs: decision.inputs,
                        action_shoot: decision.action_shoot,
                        reward: now_value - decision.decision_pitch_value,
                    });
                    if self.shot_trigger_samples.len() > SHOT_TRIGGER_SAMPLE_CAP {
                        let overflow = self.shot_trigger_samples.len() - SHOT_TRIGGER_SAMPLE_CAP;
                        self.shot_trigger_samples.drain(0..overflow);
                    }
                }
            } else {
                i += 1;
            }
        }
        if tick % SHOT_TRIGGER_SAMPLE_INTERVAL_TICKS != 0 {
            return;
        }
        // The shot decision belongs to whoever is in control of the ball — but `holder` is
        // only set on contact ticks, so most ticks it is `None`. Fall back to the attacking
        // (last-touch) outfielder nearest the ball, i.e. the player in a shooting position.
        let candidate = snapshot.ball.holder.or_else(|| {
            let attacking_team = snapshot.ball.last_touch_team?;
            snapshot
                .players
                .iter()
                .filter(|p| p.team == attacking_team && p.role != PlayerRole::Goalkeeper)
                .min_by(|a, b| {
                    a.position
                        .distance(snapshot.ball.position)
                        .partial_cmp(&b.position.distance(snapshot.ball.position))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|p| p.id)
        });
        let Some(holder) = candidate else {
            return;
        };
        let Some(me) = snapshot.players.iter().find(|p| p.id == holder) else {
            return;
        };
        let observation = snapshot.observation_for(holder);
        // Only sample attackers in a plausible shooting window — otherwise the corpus is
        // dominated by midfield carriers who would never shoot. Require the candidate to be
        // close enough to the ball to actually be the one deciding to shoot.
        if me.role == PlayerRole::Goalkeeper
            || observation.yards_to_goal > SHOT_TRIGGER_SAMPLE_MAX_GOAL_DISTANCE_YARDS
            || me.position.distance(snapshot.ball.position)
                > SHOT_TRIGGER_SAMPLE_BALL_PROXIMITY_YARDS
        {
            return;
        }
        let shooting = ability01(me.skills.shooting);
        let inputs = ShotTriggerInputs::from_observation(&observation, me.role, shooting, 1.0);
        // Action proxy: a high analytic value means a shot is the live intent this tick.
        let action_shoot =
            if analytic_shot_trigger_value(&inputs) >= SHOT_TRIGGER_SAMPLE_SHOOT_ACTION_THRESHOLD {
                1.0
            } else {
                0.0
            };
        let pitch_value = shot_decision_pitch_value(snapshot, me.team, holder);
        if pitch_value.is_finite() {
            self.pending_shot_trigger.push(PendingShotTriggerDecision {
                team: me.team,
                player_id: holder,
                inputs,
                action_shoot,
                decision_pitch_value: pitch_value,
                due_tick: tick + SHOT_TRIGGER_REWARD_WINDOW_TICKS,
            });
        }
    }

    /// Drain the collected shot-trigger RL samples for the cluster learner. `pub` so the
    /// learner binary can drain it. Mirrors [`Self::drain_line_depth_samples`].
    pub fn drain_shot_trigger_samples(&mut self) -> Vec<ShotTriggerSample> {
        std::mem::take(&mut self.shot_trigger_samples)
    }

    /// Install the trained shot-trigger head for live consumption (carried + re-installed
    /// per game by the learner). Consumed in [`WorldSnapshot::shot_trigger_mdp_value`]
    /// once it clears [`SHOT_TRIGGER_HEAD_MIN_TRAINING_STEPS`]. Mirrors `set_line_depth_head`.
    pub fn set_shot_trigger_head(&mut self, head: ShotTriggerHead) {
        self.shot_trigger_head = Some(std::sync::Arc::new(head));
    }
}

/// The attacker's territorial / goal-threat value used as the shot-trigger reward
/// signal: how advanced + threatening our possession is. Reusing the engine's pitch
/// value keeps "shoot well OR work into a better chance" both rewarded. Returns
/// non-finite when the player/ball is gone so the sampler skips cleanly.
fn shot_decision_pitch_value(snapshot: &WorldSnapshot, team: Team, player_id: usize) -> f64 {
    // Reward = our team's goal-threat from this attacker's position: closer to goal and
    // having scored both raise it. A simple, deterministic proxy over the window.
    let Some(me) = snapshot.players.iter().find(|p| p.id == player_id) else {
        return f64::NAN;
    };
    // Forward progress from our own goal toward theirs, normalized — higher = more
    // advanced / threatening possession.
    let forward = ((me.position.y - snapshot.own_goal_y_for(team)) * team.attack_dir())
        .clamp(0.0, snapshot.field_length);
    let forward_frac = forward / snapshot.field_length.max(1.0);
    // Plus the running scoreline edge so a goal inside the window dominates (the sampler
    // differences this, so a goal shows as a big positive).
    let (our_goals, their_goals) = match team {
        Team::Home => (snapshot.score_home as f64, snapshot.score_away as f64),
        Team::Away => (snapshot.score_away as f64, snapshot.score_home as f64),
    };
    (our_goals - their_goals) * 10.0 + forward_frac
}

#[cfg(test)]
mod shot_trigger_tests {
    use super::*;

    fn baseline_inputs() -> ShotTriggerInputs {
        ShotTriggerInputs {
            yards_to_goal: 18.0,
            shot_angle_degrees: 30.0,
            open_goal_fraction: 0.5,
            on_frame_probability: 0.7,
            beat_goalkeeper_probability: 0.6,
            block_probability: 0.15,
            blocker_distance_yards: 6.0,
            shot_lane_open: true,
            rebound_second_chance: 0.2,
            keeper_distance_yards: 16.0,
            keeper_out_of_position: 0.2,
            ball_forward_velocity_yps: 0.0,
            ball_acceleration_yps2: 0.0,
            actor_acceleration_yps2: 0.0,
            nearest_opponent_distance_yards: 4.0,
            offensive_urgency: 0.4,
            decision_urgency: 0.3,
            goal_attack_window: 0.4,
            body_mechanics_fit: 1.0,
            shooting_skill: 0.7,
            strong_foot_power: 7.0,
            is_forward: true,
        }
    }

    #[test]
    fn features_are_finite_bounded_and_right_dim() {
        let f = baseline_inputs().to_features();
        assert_eq!(f.len(), SHOT_TRIGGER_FEATURE_DIM);
        for v in f {
            assert!(v.is_finite());
            assert!(v.abs() <= 2.0 + 1e-9, "feature {v} out of normalized range");
        }
    }

    #[test]
    fn analytic_value_is_in_unit_range() {
        let v = analytic_shot_trigger_value(&baseline_inputs());
        assert!((0.0..=1.0).contains(&v), "value {v} out of [0,1]");
    }

    #[test]
    fn closer_is_worth_more_than_farther_all_else_equal() {
        let mut close = baseline_inputs();
        close.yards_to_goal = 14.0;
        let mut far = baseline_inputs();
        far.yards_to_goal = 25.0;
        assert!(
            analytic_shot_trigger_value(&close) > analytic_shot_trigger_value(&far),
            "a closer shot should be worth more"
        );
    }

    #[test]
    fn covered_keeper_at_range_kills_the_shot_but_open_net_rescues_it() {
        // 24yd, keeper set and net covered ⇒ low value (work it closer).
        let mut covered = baseline_inputs();
        covered.yards_to_goal = 24.0;
        covered.open_goal_fraction = 0.1;
        covered.keeper_out_of_position = 0.0;
        covered.beat_goalkeeper_probability = 0.2;
        covered.rebound_second_chance = 0.0;
        let covered_v = analytic_shot_trigger_value(&covered);
        // Same range but the keeper is stranded and the net is gaping ⇒ a real long shot.
        let mut open = covered.clone();
        open.open_goal_fraction = 0.9;
        open.keeper_out_of_position = 0.9;
        open.beat_goalkeeper_probability = 0.85;
        let open_v = analytic_shot_trigger_value(&open);
        assert!(
            covered_v < 0.35,
            "covered 24yd shot should be low, got {covered_v}"
        );
        assert!(
            open_v > covered_v,
            "an open net at range should rescue the long shot"
        );
    }

    #[test]
    fn a_blocked_lane_suppresses_value() {
        let mut clear = baseline_inputs();
        clear.block_probability = 0.05;
        let mut blocked = baseline_inputs();
        blocked.block_probability = 0.8;
        assert!(
            analytic_shot_trigger_value(&clear) > analytic_shot_trigger_value(&blocked),
            "a blocked lane should suppress the shot value"
        );
    }

    #[test]
    fn poor_body_mechanics_suppress_value() {
        let mut good = baseline_inputs();
        good.body_mechanics_fit = 1.0;
        let mut bad = baseline_inputs();
        bad.body_mechanics_fit = 0.2;
        assert!(
            analytic_shot_trigger_value(&good) > analytic_shot_trigger_value(&bad),
            "a body that can't strike the ball should lower the shot value"
        );
    }

    #[test]
    fn score_multiplier_is_monotone_and_bounded() {
        let lo = shot_trigger_score_multiplier(0.0);
        let hi = shot_trigger_score_multiplier(1.0);
        assert!(lo >= 0.10 && lo <= 0.11 && hi <= 1.30 && hi > lo);
    }

    #[test]
    fn head_predicts_a_bounded_value() {
        let head = ShotTriggerHead::new(9);
        let p = head.predict(&baseline_inputs()).expect("finite prediction");
        assert!((0.0..=1.0).contains(&p), "head value {p} out of (0,1)");
    }

    #[test]
    fn reward_weighted_training_runs_and_counts_steps() {
        let mut head = ShotTriggerHead::new(3);
        let samples: Vec<ShotTriggerSample> = (0..64)
            .map(|i| ShotTriggerSample {
                inputs: baseline_inputs(),
                action_shoot: if i % 2 == 0 { 1.0 } else { 0.0 },
                reward: if i % 2 == 0 { 1.0 } else { -1.0 },
            })
            .collect();
        let mut last = f64::INFINITY;
        for _ in 0..20 {
            last = head.train_reward_weighted(&samples, 0.05);
        }
        assert!(last.is_finite());
        assert!(head.training_steps() > 0);
    }

    #[test]
    fn train_entry_skips_empty_and_trains_valid() {
        assert!(train_shot_trigger_head(&[], 1, 4, 0.02).is_none());
        let samples: Vec<ShotTriggerSample> = (0..16)
            .map(|_| ShotTriggerSample {
                inputs: baseline_inputs(),
                action_shoot: 1.0,
                reward: 0.5,
            })
            .collect();
        let (head, report) =
            train_shot_trigger_head(&samples, 1, 4, 0.02).expect("trains a usable corpus");
        assert_eq!(report.epochs, 4);
        assert!(report.training_steps > 0);
        assert!(head.training_steps() > 0);
    }

    #[test]
    fn open_goal_fraction_rises_when_keeper_is_out_of_position() {
        let mut obs = sample_shot_observation();
        obs.opposing_goalkeeper_out_of_position = 0.0;
        let covered = open_goal_fraction(&obs);
        obs.opposing_goalkeeper_out_of_position = 0.95;
        let open = open_goal_fraction(&obs);
        assert!(
            open > covered,
            "a stranded keeper should leave more net open"
        );
    }

    fn sample_shot_observation() -> SoccerPomdpObservation {
        let mut sim = SoccerMatch::default_11v11(MatchConfig {
            duration_seconds: 0.1,
            seed: 7,
            ..Default::default()
        });
        sim.run_time_step();
        WorldSnapshot::from_match(&sim).observation_for(0)
    }
}
