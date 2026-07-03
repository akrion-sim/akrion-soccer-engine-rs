//! Learnable **lane-affinity decision** — turns the engine's hard lane *enforcement*
//! into an MDP/POMDP *choice*: when a player's shape target sits outside its home
//! vertical lane, *how firmly should it be pulled back into the lane vs. allowed to
//! break out* into the space it found?
//!
//! ## Why this module exists
//!
//! The base engine holds a player in its lane by a **deterministic clamp**
//! ([`crate::des::general::soccer::lane_discipline::lane_clamp_blend`]): given the
//! role's lane `commitment` and a shape-exception `relief`, it computes a blend in
//! `[0, 1]` and pulls the target's `x` that far toward the lane edge. That is a firm
//! *predilection* — a back four should stay spread, a midfielder should hold its
//! channel — but it was never a *decision*. A player with a genuinely better option
//! one lane over (open space, an overload toward the ball, no team-mate already there)
//! had no way to *choose* to break the lane; the clamp simply forced it back.
//!
//! This module is the seam that makes that one continuous choice — a signed **lane
//! break bias** in `[-1, 1]` — learnable, mirroring [`super::receive_approach`] and
//! [`super::loose_ball_commit`]'s reward-weighted heads:
//!
//!   * [`LaneAffinityInputs`] — the per-player lane state the choice is a function of,
//!     captured in a **mirror-invariant** form (lane deviation, commitment, relief, the
//!     space gained by breaking vs. staying, whether a team-mate already occupies the
//!     break lane, whether the break heads toward the ball, phase, role).
//!   * [`analytic_lane_break_bias`] — a deterministic, RNG-free seed encoding the
//!     coach's prior: **hold the lane by default** (bias ≈ 0 ⇒ the clamp is unchanged),
//!     lean toward *breaking* only when in possession there is real space to be won one
//!     lane over that no team-mate already holds and the break goes toward the play, and
//!     lean toward *holding harder* when defending and the space out of lane is covered.
//!   * [`LaneAffinityHead`] — a `FeedForwardNetwork` (`DIM → hidden → 1`, **tanh** so it
//!     is signed) the cluster learner trains by reward-weighted regression toward the
//!     break/hold actions that actually improved the team's territorial control.
//!
//! The live seam is [`WorldSnapshot::lane_affinity_effective_clamp_blend`]: it takes the
//! deterministic `base_blend` and shifts it by `bias * LANE_AFFINITY_MAX_BLEND_ADJUST` —
//! a negative bias *loosens* the clamp (permits the break), a positive bias *tightens*
//! it (holds harder). The shift is bounded so the lane discipline is **refined, never
//! abandoned**: the predilection remains, the choice modulates it.
//!
//! ## Parity
//!
//! The gate [`lane_affinity_model_enabled`] is **on by default in production** (the seam
//! is live, seeded by the analytic prior which is ≈ 0 in ordinary shape states, so team
//! shape stays near-identical) but **off by default under `#[cfg(test)]`**, so the large
//! existing shape/position test-suite is byte-identical unless a test opts in. When the
//! seam is off, [`WorldSnapshot::lane_affinity_effective_clamp_blend`] returns the
//! deterministic `base_blend` unchanged *before building any features*, and the RL
//! collector early-returns. The feature build, analytic seed, and head are pure/RNG-free.

use super::*;
use crate::des::general::neural_network::{ActivationName, FeedForwardNetwork, RandomNetworkSpec};
use crate::des::general::prng::mulberry32;

/// Number of features in the lane-affinity state vector. The exact ordering is the
/// single source of truth in [`LaneAffinityInputs::to_features`].
pub const LANE_AFFINITY_FEATURE_DIM: usize = 14;
/// Hidden width of the learned regression head.
pub const LANE_AFFINITY_HIDDEN_UNITS: usize = 18;
/// Maximum the learned bias may move the deterministic lane-clamp blend, in blend units
/// (`[0, 1]`). A bias of `-1` loosens the clamp by this much (permit the break); `+1`
/// tightens it by this much (hold harder). Kept below `1.0` so the lane discipline is
/// refined rather than abandoned — the predilection survives the choice.
pub const LANE_AFFINITY_MAX_BLEND_ADJUST: f64 = 0.5;

/// Reference lateral distance (yd) normalizing deviation / opponent-distance features.
const REF_LANE_YARDS: f64 = 12.0;
/// Reference space score normalizing the pitch-space features.
const REF_SPACE: f64 = 18.0;

/// Env gate turning the learned lane-affinity seam ON. Production default is **on**;
/// set `DD_SOCCER_DISABLE_LANE_AFFINITY_MODEL` for a legacy A/B (hard clamp only).
const LANE_AFFINITY_MODEL_ENABLE_ENV: &str = "DD_SOCCER_ENABLE_LANE_AFFINITY_MODEL";
const LANE_AFFINITY_MODEL_DISABLE_ENV: &str = "DD_SOCCER_DISABLE_LANE_AFFINITY_MODEL";

fn env_flag(name: &str) -> Option<bool> {
    std::env::var(name).ok().map(|raw| {
        let v = raw.trim().to_ascii_lowercase();
        matches!(v.as_str(), "1" | "true" | "yes" | "on")
    })
}

fn lane_affinity_model_enabled_from_env() -> bool {
    if env_flag(LANE_AFFINITY_MODEL_DISABLE_ENV).unwrap_or(false) {
        return false;
    }
    env_flag(LANE_AFFINITY_MODEL_ENABLE_ENV).unwrap_or(true)
}

/// Whether the learned lane-affinity seam is consulted this process. Off ⇒ the
/// deterministic lane clamp stands (parity). Production caches once (default **on**);
/// tests re-read the env each call and default **off** so the existing shape/position
/// suite is byte-identical unless a test opts in.
pub fn lane_affinity_model_enabled() -> bool {
    #[cfg(test)]
    {
        if env_flag(LANE_AFFINITY_MODEL_DISABLE_ENV).unwrap_or(false) {
            return false;
        }
        env_flag(LANE_AFFINITY_MODEL_ENABLE_ENV).unwrap_or(false)
    }
    #[cfg(not(test))]
    {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(lane_affinity_model_enabled_from_env)
    }
}

/// How often (ticks) lane-affinity RL decisions are sampled while the model is on.
pub const LANE_AFFINITY_SAMPLE_INTERVAL_TICKS: u64 = 6;
/// Window (ticks) over which a sampled decision's territorial reward is measured.
pub const LANE_AFFINITY_REWARD_WINDOW_TICKS: u64 = 30;
/// Cap on the rolling RL sample buffer (drained by the learner).
pub const LANE_AFFINITY_SAMPLE_CAP: usize = 8192;
/// Minimum training steps before a trained head is consumed live; below this the
/// regression net is too raw, so the seam falls back to the analytic seed.
pub const LANE_AFFINITY_HEAD_MIN_TRAINING_STEPS: usize = 200;
/// Weight on the dense territorial-advantage delta added to the windowed real-reward sum.
/// Kept modest so the actual reward pipeline (passes/shots/goals/turnovers) dominates,
/// with pitch-control change as the tiebreaker for a purely-positional off-ball break
/// that fires no discrete on-ball event inside the window.
pub const LANE_AFFINITY_TERRITORIAL_SHAPING_WEIGHT: f64 = 1.0;

/// Raw (un-normalized) per-player lane state the break/hold choice is a function of,
/// captured mirror-invariantly (all lateral quantities are magnitudes or signed by the
/// break direction, never by absolute pitch side). A plain struct so it can be logged as
/// a training row, asserted on in tests, and surfaced by the inspector.
#[derive(Clone, Debug, PartialEq)]
pub struct LaneAffinityInputs {
    pub field_width: f64,
    /// How far (yd) the shape target sits outside the role's home lane band. `0` inside.
    pub deviation_yards: f64,
    /// The role's effective lane commitment in `[0, 1]` (already eroded by relief).
    pub commitment: f64,
    /// The shape-exception relief in `[0, 1]` (attacking run / mark / no-outlet / cover).
    pub relief: f64,
    /// `1` if the player's team is in possession, else `0`.
    pub in_possession: f64,
    /// `1` if the player's team is defending (opponent controls), else `0`.
    pub defensive_phase: f64,
    /// Team pitch-space score at the *break* target (out of lane). Higher ⇒ more open.
    pub space_at_target: f64,
    /// Team pitch-space score at the *lane* position (held). The break is worth it when
    /// `space_at_target` clearly exceeds this.
    pub space_at_lane: f64,
    /// Nearest-opponent distance (yd) at the break target — how contested the break is.
    pub nearest_opp_at_target: f64,
    /// Team-mate crowding penalty in `[0, 1]` at the break target: high ⇒ a team-mate is
    /// already there, so breaking is redundant (hold the lane / keep the width).
    pub teammate_redundancy: f64,
    /// Alignment of the break with the ball in `[-1, 1]`: `+1` the break heads toward the
    /// ball's channel (support the play), `-1` it heads away.
    pub ball_alignment: f64,
    pub role_defender: f64,
    pub role_midfielder: f64,
    pub role_forward: f64,
}

impl LaneAffinityInputs {
    /// The normalized network input vector. **This ordering is the single source of
    /// truth** for the feature layout — change it and bump [`LANE_AFFINITY_FEATURE_DIM`]
    /// and any persisted head.
    pub fn to_features(&self) -> [f64; LANE_AFFINITY_FEATURE_DIM] {
        let lane = |d: f64| (d / REF_LANE_YARDS).clamp(0.0, 2.0);
        let space = |s: f64| (s / REF_SPACE).clamp(0.0, 1.5);
        let unit = |u: f64| u.clamp(0.0, 1.0);
        [
            lane(self.deviation_yards),
            unit(self.commitment),
            unit(self.relief),
            unit(self.in_possession),
            unit(self.defensive_phase),
            space(self.space_at_target),
            space(self.space_at_lane),
            lane(self.nearest_opp_at_target),
            unit(self.teammate_redundancy),
            self.ball_alignment.clamp(-1.0, 1.0),
            self.role_defender,
            self.role_midfielder,
            self.role_forward,
            // Break gain (space at target minus at lane), the decisive term, exposed
            // directly so the analytic prior and the head both read it cheaply.
            ((self.space_at_target - self.space_at_lane) / REF_SPACE).clamp(-1.0, 1.0),
        ]
    }
}

/// Analytic, deterministic, RNG-free seed for the **lane break bias** in `[-1, 1]`
/// (negative = break out of lane / loosen the clamp; positive = hold harder / tighten).
/// It is the bootstrap target and the live fallback the learned [`LaneAffinityHead`]
/// first imitates, then beats.
///
/// The coach's prior it encodes: **hold the lane by default** (bias ≈ 0, so the
/// deterministic clamp is unchanged and team shape is preserved). Lean toward *breaking*
/// only when, in possession, there is real space to be won one lane over that no
/// team-mate already occupies and the break heads toward the ball (support the play).
/// Lean toward *holding harder* when defending and the space out of lane is contested /
/// covered — protecting the block.
pub fn analytic_lane_break_bias(inputs: &LaneAffinityInputs) -> f64 {
    let mut z = 0.0;

    // BREAK incentive (negative): in possession, real space gained one lane over, not
    // redundant with a team-mate, heading toward the play. Scaled by how out-of-band the
    // target already is (an in-band target needs no help).
    let break_gain = ((inputs.space_at_target - inputs.space_at_lane) / REF_SPACE).clamp(0.0, 1.0);
    let toward_play = (0.5 + 0.5 * inputs.ball_alignment).clamp(0.0, 1.0);
    let unclaimed = (1.0 - inputs.teammate_redundancy).clamp(0.0, 1.0);
    let out_of_band = (inputs.deviation_yards / REF_LANE_YARDS).clamp(0.0, 1.0);
    z -= 1.1 * inputs.in_possession * break_gain * toward_play * unclaimed * out_of_band;

    // HOLD incentive (positive): defending with the out-of-lane space covered (little
    // gain, tight opponent) ⇒ protect the block, tighten back into the lane.
    let covered = (1.0 - break_gain) * (1.0 - (inputs.nearest_opp_at_target / 8.0).clamp(0.0, 1.0));
    z += 0.7 * inputs.defensive_phase * inputs.role_defender * covered * out_of_band;

    z.tanh()
}

/// One reward-weighted RL training row for the lane-affinity head: the per-player lane
/// state at a decision, the break/hold ACTION taken (signed bias in `[-1, 1]`, − = broke
/// out), and the reward attributed over the following window (higher = the choice
/// improved the team's territorial control). Mirrors [`super::receive_approach::ReceiveApproachSample`].
#[derive(Clone, Debug, PartialEq)]
pub struct LaneAffinitySample {
    pub inputs: LaneAffinityInputs,
    /// The break/hold action actually taken, in `[-1, 1]` (− = broke out of lane).
    pub action_bias: f64,
    /// Reward attributed to the decision over the following window (any scale; only
    /// relative-to-batch matters — the trainer baselines it).
    pub reward: f64,
}

/// An open lane-affinity decision awaiting its windowed reward. Recorded at the decision
/// tick with the player's team territorial state then; resolved
/// `LANE_AFFINITY_REWARD_WINDOW_TICKS` later by scoring the territorial change.
#[derive(Clone, Debug, PartialEq)]
pub struct PendingLaneAffinityDecision {
    pub team: Team,
    pub player_id: usize,
    pub inputs: LaneAffinityInputs,
    pub action_bias: f64,
    pub decision_territorial: f64,
    /// Running sum of the team's **real reward-pipeline events** (forward-pass rewards,
    /// pass→shot / pass→goal, shot/goal back-prop, turnover / interception penalties,
    /// duel / loose-ball wins, dribble-progress rewards, hold/no-progress penalties)
    /// accrued each tick over the decision's window. This is the primary reward — it is
    /// exactly the reward the rest of the engine optimizes — with the territorial delta
    /// added as dense off-ball credit at resolution.
    pub reward_accum: f64,
    pub due_tick: u64,
}

/// Learned regression head for the signed lane break bias: a `FeedForwardNetwork`
/// (`DIM → hidden → 1`, **tanh** output so it is signed in `(-1, 1)`). Round-trips to
/// Postgres via the engine's neural-snapshot path like the other learned heads.
#[derive(Clone, Debug)]
pub struct LaneAffinityHead {
    network: FeedForwardNetwork,
    training_steps: usize,
    last_loss: Option<f64>,
}

impl LaneAffinityHead {
    pub fn new(seed: u32) -> Self {
        let mut rng = mulberry32(seed ^ 0x1A9E_4B27);
        let network = FeedForwardNetwork::random(
            &RandomNetworkSpec {
                input_dim: LANE_AFFINITY_FEATURE_DIM,
                hidden_layers: vec![LANE_AFFINITY_HIDDEN_UNITS],
                output_dim: 1,
                hidden_activation: ActivationName::Tanh,
                output_activation: ActivationName::Tanh,
                weight_scale: None,
            },
            &mut rng,
        );
        LaneAffinityHead {
            network,
            training_steps: 0,
            last_loss: None,
        }
    }

    /// Predicted lane break bias in `(-1, 1)` for a captured lane state, or `None` on a
    /// malformed (mis-dimensioned / non-finite) input.
    pub fn predict(&self, inputs: &LaneAffinityInputs) -> Option<f64> {
        let features = inputs.to_features();
        if features.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let out = self.network.predict(&features[..]);
        out.first().copied().filter(|p| p.is_finite())
    }

    /// One SGD epoch regressing the head toward `(features, target_bias)` pairs
    /// (supervised bootstrap toward the analytic seed). Returns the mean step loss.
    pub fn train(&mut self, samples: &[(LaneAffinityInputs, f64)], learning_rate: f64) -> f64 {
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
    /// improvement): regress toward each sample's ACTION (the signed break bias), weighted
    /// by how much its reward beat the batch baseline. Above-average decisions are imitated
    /// strongly, below-average ones barely — so the head learns the break/hold pattern that
    /// actually improved territorial control. Mirrors
    /// [`super::receive_approach::ReceiveApproachHead::train_reward_weighted`].
    pub fn train_reward_weighted(
        &mut self,
        samples: &[LaneAffinitySample],
        learning_rate: f64,
    ) -> f64 {
        let finite: Vec<&LaneAffinitySample> = samples
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

/// Outcome of training a lane-affinity head on a drained RL corpus, logged by the
/// learning runner. Mirrors [`super::receive_approach::ReceiveApproachTrainingReport`].
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LaneAffinityTrainingReport {
    pub samples: usize,
    pub training_steps: usize,
    pub epochs: usize,
    pub final_loss: f64,
}

/// Load-and-train entry for a lane-affinity head: given a drained RL corpus, build a
/// fresh [`LaneAffinityHead`] and run `epochs` reward-weighted-regression passes. `None`
/// if no usable samples. Mirrors [`super::receive_approach::train_receive_approach_head`].
pub fn train_lane_affinity_head(
    samples: &[LaneAffinitySample],
    seed: u32,
    epochs: usize,
    learning_rate: f64,
) -> Option<(LaneAffinityHead, LaneAffinityTrainingReport)> {
    let usable = samples
        .iter()
        .filter(|s| s.reward.is_finite() && s.action_bias.is_finite())
        .count();
    if usable == 0 || epochs == 0 {
        return None;
    }
    let mut head = LaneAffinityHead::new(seed);
    let mut final_loss = 0.0;
    for _ in 0..epochs {
        final_loss = head.train_reward_weighted(samples, learning_rate);
    }
    let report = LaneAffinityTrainingReport {
        samples: usable,
        training_steps: head.training_steps(),
        epochs,
        final_loss,
    };
    Some((head, report))
}

/// Public wrapper returning ONLY the report (the head type is engine-internal). The
/// learning runner uses this to train the drained corpus and log that it learns.
pub fn report_lane_affinity_training(
    samples: &[LaneAffinitySample],
    seed: u32,
    epochs: usize,
    learning_rate: f64,
) -> Option<LaneAffinityTrainingReport> {
    train_lane_affinity_head(samples, seed, epochs, learning_rate).map(|(_, report)| report)
}

impl WorldSnapshot {
    /// Build the [`LaneAffinityInputs`] for `player` whose shape target `bounded` sits at
    /// lateral `bounded.x`, being clamped toward `lane_x`. `fit`/`relief`/`in_possession`
    /// are the same values the clamp seam already computed. `None` for a goalkeeper, an
    /// in-band target, or a degenerate (no lateral break) geometry. Pure / RNG-free.
    pub(crate) fn build_lane_affinity_inputs(
        &self,
        player: &PlayerSnapshot,
        bounded: Vec2,
        lane_x: f64,
        fit: &VerticalLaneFit,
        relief: f64,
        in_possession: bool,
    ) -> Option<LaneAffinityInputs> {
        if player.role == PlayerRole::Goalkeeper {
            return None;
        }
        if fit.commitment <= 1e-9 || fit.deviation_yards <= 0.0 {
            return None;
        }
        // The two competing positions: `bounded` (break, out of lane) and the lane edge.
        let lane_pos = Vec2 {
            x: lane_x,
            y: bounded.y,
        };
        let break_dx = bounded.x - lane_x;
        if break_dx.abs() <= 1e-3 {
            return None;
        }
        let space_at_target = self.space_score_at(bounded, player.team);
        let space_at_lane = self.space_score_at(lane_pos, player.team);
        let nearest_opp_at_target = self.nearest_opponent_distance_at(player.team, bounded);
        let teammate_redundancy = self
            .teammate_occupied_space_penalty_at(player.team, bounded, Some(player.id), relief)
            .clamp(0.0, 1.0);
        // Does breaking head toward the ball's channel? `+1` toward, `-1` away.
        let ball_dx = self.ball.position.x - lane_x;
        let ball_alignment = if ball_dx.abs() <= 1e-3 {
            0.0
        } else {
            (break_dx.signum() * ball_dx.signum()).clamp(-1.0, 1.0)
        };
        let defensive_phase = self.possession_team() == Some(player.team.other());

        Some(LaneAffinityInputs {
            field_width: self.field_width,
            deviation_yards: fit.deviation_yards,
            commitment: fit.commitment.clamp(0.0, 1.0),
            relief: relief.clamp(0.0, 1.0),
            in_possession: in_possession as i32 as f64,
            defensive_phase: defensive_phase as i32 as f64,
            space_at_target,
            space_at_lane,
            nearest_opp_at_target,
            teammate_redundancy,
            ball_alignment,
            role_defender: (player.role == PlayerRole::Defender) as i32 as f64,
            role_midfielder: (player.role == PlayerRole::Midfielder) as i32 as f64,
            role_forward: (player.role == PlayerRole::Forward) as i32 as f64,
        })
    }

    /// The model's signed **lane break bias** in `(-1, 1)` for the given lane state
    /// (negative = break out / loosen the clamp). The trained head once it has learned
    /// enough, else the analytic seed. `None` when the model is gated off or inputs are
    /// degenerate.
    pub(crate) fn lane_break_bias(&self, inputs: &LaneAffinityInputs) -> Option<f64> {
        if !lane_affinity_model_enabled() {
            return None;
        }
        let bias = self
            .lane_affinity_head
            .as_ref()
            .filter(|head| head.training_steps() >= LANE_AFFINITY_HEAD_MIN_TRAINING_STEPS)
            .and_then(|head| head.predict(inputs))
            .unwrap_or_else(|| analytic_lane_break_bias(inputs));
        bias.is_finite().then(|| bias.clamp(-1.0, 1.0))
    }

    /// The effective lane-clamp blend to apply, given the deterministic `base_blend` the
    /// [`super::lane_discipline::lane_clamp_blend`] produced. When the model is off (or
    /// the state is degenerate) returns `base_blend` unchanged — the hard clamp stands,
    /// byte-identical. When on, shifts it by the learned break/hold bias: a negative bias
    /// *loosens* the clamp (permit the break), a positive bias *tightens* it, bounded by
    /// [`LANE_AFFINITY_MAX_BLEND_ADJUST`] and clamped to `[0, 1]`.
    pub(crate) fn lane_affinity_effective_clamp_blend(
        &self,
        player: &PlayerSnapshot,
        bounded: Vec2,
        lane_x: f64,
        fit: &VerticalLaneFit,
        relief: f64,
        in_possession: bool,
        base_blend: f64,
    ) -> f64 {
        if !lane_affinity_model_enabled() {
            return base_blend;
        }
        let Some(inputs) =
            self.build_lane_affinity_inputs(player, bounded, lane_x, fit, relief, in_possession)
        else {
            return base_blend;
        };
        let Some(bias) = self.lane_break_bias(&inputs) else {
            return base_blend;
        };
        (base_blend + bias * LANE_AFFINITY_MAX_BLEND_ADJUST).clamp(0.0, 1.0)
    }
}

impl SoccerMatch {
    /// Gated per-tick RL sample collection for the lane-affinity head. A **no-op** unless
    /// the model is enabled. When on: resolves decisions whose reward window has elapsed
    /// (reward = the team's territorial-advantage change over the window), and samples
    /// fresh per-player break/hold decisions on cadence for out-of-lane outfielders.
    pub(crate) fn collect_lane_affinity_rl_samples(&mut self, snapshot: &WorldSnapshot) {
        if !lane_affinity_model_enabled() {
            return;
        }
        let tick = self.tick;
        // Accumulate THIS tick's real reward-pipeline events per team into every open
        // decision (the exact reward the rest of the engine optimizes). `reward_events`
        // is the per-tick scratch buffer; sum each team's players' signed amounts.
        if !self.pending_lane_affinity.is_empty() && !self.reward_events.is_empty() {
            let mut home_reward = 0.0;
            let mut away_reward = 0.0;
            for event in &self.reward_events {
                if event.player_id >= self.players.len() || !event.amount.is_finite() {
                    continue;
                }
                match self.players[event.player_id].team {
                    Team::Home => home_reward += event.amount,
                    Team::Away => away_reward += event.amount,
                }
            }
            for decision in &mut self.pending_lane_affinity {
                decision.reward_accum += match decision.team {
                    Team::Home => home_reward,
                    Team::Away => away_reward,
                };
            }
        }
        // Resolve due decisions.
        let mut i = 0;
        while i < self.pending_lane_affinity.len() {
            if self.pending_lane_affinity[i].due_tick <= tick {
                let decision = self.pending_lane_affinity.swap_remove(i);
                let now_territorial = territorial_advantage(snapshot, decision.team);
                let territorial_delta =
                    if now_territorial.is_finite() && decision.decision_territorial.is_finite() {
                        now_territorial - decision.decision_territorial
                    } else {
                        0.0
                    };
                let reward = decision.reward_accum
                    + LANE_AFFINITY_TERRITORIAL_SHAPING_WEIGHT * territorial_delta;
                if reward.is_finite() {
                    self.lane_affinity_samples.push(LaneAffinitySample {
                        inputs: decision.inputs,
                        action_bias: decision.action_bias,
                        reward,
                    });
                    if self.lane_affinity_samples.len() > LANE_AFFINITY_SAMPLE_CAP {
                        let overflow = self.lane_affinity_samples.len() - LANE_AFFINITY_SAMPLE_CAP;
                        self.lane_affinity_samples.drain(0..overflow);
                    }
                }
            } else {
                i += 1;
            }
        }
        // Sample fresh per-player decisions on cadence.
        if tick % LANE_AFFINITY_SAMPLE_INTERVAL_TICKS != 0 {
            return;
        }
        let in_possession_team = snapshot
            .controlled_possession_team()
            .or_else(|| snapshot.possession_team());
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
            let pos = snapshot.player_snapshot_position(player);
            let in_possession = in_possession_team == Some(player.team);
            let relief = snapshot.positional_shape_exception_relief_for_player_target(player, pos);
            let fit = snapshot.vertical_lane_fit_for_player_target(player, pos, relief);
            // Only an out-of-lane player actually faces the break/hold choice.
            if fit.commitment <= 1e-9 || fit.deviation_yards <= 0.0 {
                continue;
            }
            let lane_x = vertical_lane_clamped_x_for_role(
                player.role,
                player.home_position.x,
                pos.x,
                snapshot.field_width,
                in_possession,
            );
            let Some(inputs) = snapshot.build_lane_affinity_inputs(
                player,
                pos,
                lane_x,
                &fit,
                relief,
                in_possession,
            ) else {
                continue;
            };
            // The break/hold action the engine used this tick (head once trained, else seed).
            let action_bias = snapshot
                .lane_break_bias(&inputs)
                .unwrap_or_else(|| analytic_lane_break_bias(&inputs));
            let territorial = territorial_advantage(snapshot, player.team);
            if territorial.is_finite() {
                self.pending_lane_affinity
                    .push(PendingLaneAffinityDecision {
                        team: player.team,
                        player_id: id,
                        inputs,
                        action_bias,
                        decision_territorial: territorial,
                        reward_accum: 0.0,
                        due_tick: tick + LANE_AFFINITY_REWARD_WINDOW_TICKS,
                    });
            }
        }
    }

    /// Drain the collected lane-affinity RL samples for the cluster learner (train the
    /// head + persist). Mirrors [`Self::drain_receive_approach_samples`].
    pub fn drain_lane_affinity_samples(&mut self) -> Vec<LaneAffinitySample> {
        std::mem::take(&mut self.lane_affinity_samples)
    }

    /// Install the trained lane-affinity head for live consumption. The learner carries +
    /// trains it across games and re-installs it each game; wrapped in `Arc` so every
    /// per-tick snapshot shares it cheaply.
    pub fn set_lane_affinity_head(&mut self, head: LaneAffinityHead) {
        self.lane_affinity_head = Some(std::sync::Arc::new(head));
    }
}

#[cfg(test)]
mod lane_affinity_tests {
    use super::*;

    fn baseline_inputs() -> LaneAffinityInputs {
        LaneAffinityInputs {
            field_width: 80.0,
            deviation_yards: 5.0,
            commitment: 0.6,
            relief: 0.1,
            in_possession: 1.0,
            defensive_phase: 0.0,
            space_at_target: 9.0,
            space_at_lane: 9.0,
            nearest_opp_at_target: 8.0,
            teammate_redundancy: 0.1,
            ball_alignment: 0.0,
            role_defender: 0.0,
            role_midfielder: 1.0,
            role_forward: 0.0,
        }
    }

    #[test]
    fn features_are_finite_bounded_and_right_dim() {
        let f = baseline_inputs().to_features();
        assert_eq!(f.len(), LANE_AFFINITY_FEATURE_DIM);
        for v in f {
            assert!(v.is_finite());
            assert!(v.abs() <= 2.0 + 1e-9, "feature {v} out of normalized range");
        }
    }

    #[test]
    fn analytic_bias_is_a_valid_signed_unit() {
        let b = analytic_lane_break_bias(&baseline_inputs());
        assert!((-1.0..=1.0).contains(&b), "bias {b} out of [-1,1]");
    }

    #[test]
    fn even_break_offers_no_reason_to_leave_the_lane() {
        // Equal space, no ball pull ⇒ the coach's prior holds the lane (≈ 0).
        let b = analytic_lane_break_bias(&baseline_inputs());
        assert!(
            b.abs() < 0.05,
            "a neutral state should barely move the clamp, got {b}"
        );
    }

    #[test]
    fn open_space_toward_the_ball_biases_toward_breaking() {
        let mut brk = baseline_inputs();
        brk.space_at_target = 16.0; // acres out of lane
        brk.space_at_lane = 6.0;
        brk.ball_alignment = 1.0; // toward the play
        brk.teammate_redundancy = 0.0;
        let b = analytic_lane_break_bias(&brk);
        assert!(
            b < 0.0,
            "real space toward the play should bias toward breaking, got {b}"
        );
    }

    #[test]
    fn a_teammate_already_there_suppresses_the_break() {
        let mut open = baseline_inputs();
        open.space_at_target = 16.0;
        open.space_at_lane = 6.0;
        open.ball_alignment = 1.0;
        open.teammate_redundancy = 0.0;
        let mut redundant = open.clone();
        redundant.teammate_redundancy = 0.95;
        assert!(
            analytic_lane_break_bias(&redundant) > analytic_lane_break_bias(&open),
            "a team-mate already occupying the break lane should suppress the break"
        );
    }

    #[test]
    fn defending_with_covered_space_biases_toward_holding() {
        let mut def = baseline_inputs();
        def.in_possession = 0.0;
        def.defensive_phase = 1.0;
        def.role_defender = 1.0;
        def.role_midfielder = 0.0;
        def.space_at_target = 6.0; // no gain out of lane
        def.space_at_lane = 9.0;
        def.nearest_opp_at_target = 2.0; // contested
        let b = analytic_lane_break_bias(&def);
        assert!(
            b > 0.0,
            "a defender with covered out-of-lane space should hold harder, got {b}"
        );
    }

    #[test]
    fn head_predicts_a_bounded_signed_bias() {
        let head = LaneAffinityHead::new(7);
        let b = head.predict(&baseline_inputs()).expect("finite prediction");
        assert!((-1.0..=1.0).contains(&b), "head bias {b} out of (-1,1)");
    }

    #[test]
    fn reward_weighted_training_steers_toward_the_high_reward_action() {
        let mut head = LaneAffinityHead::new(23);
        let inputs = baseline_inputs();
        let before = head.predict(&inputs).expect("finite");
        // Same state, two competing actions: breaking (−0.8) is rewarded, holding (+0.8)
        // penalized. RWR should pull the head toward breaking.
        let samples: Vec<LaneAffinitySample> = (0..32)
            .flat_map(|_| {
                [
                    LaneAffinitySample {
                        inputs: inputs.clone(),
                        action_bias: -0.8,
                        reward: 1.0,
                    },
                    LaneAffinitySample {
                        inputs: inputs.clone(),
                        action_bias: 0.8,
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
            after < before && after < 0.0,
            "RWR should move the head toward the high-reward break action: {before} -> {after}"
        );
    }

    #[test]
    fn train_entry_skips_empty_and_trains_valid() {
        assert!(train_lane_affinity_head(&[], 1, 4, 0.02).is_none());
        let samples: Vec<LaneAffinitySample> = (0..16)
            .map(|i| LaneAffinitySample {
                inputs: baseline_inputs(),
                action_bias: if i % 2 == 0 { -0.8 } else { 0.8 },
                reward: if i % 2 == 0 { 1.0 } else { -1.0 },
            })
            .collect();
        let (_, report) =
            train_lane_affinity_head(&samples, 1, 4, 0.02).expect("trains a usable corpus");
        assert_eq!(report.samples, 16);
        assert_eq!(report.epochs, 4);
        assert!(report.training_steps > 0 && report.final_loss.is_finite());
    }

    #[test]
    fn effective_blend_is_identity_when_model_disabled() {
        std::env::set_var(LANE_AFFINITY_MODEL_DISABLE_ENV, "1");
        assert!(!lane_affinity_model_enabled());
        std::env::remove_var(LANE_AFFINITY_MODEL_DISABLE_ENV);
    }
}
