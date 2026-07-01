//! Learnable **goal-side recovery decision** — turns the fixed lateral "shade onto the
//! ball→own-goal line" nudge into an MDP/POMDP *choice*: when a defender is recovering,
//! *how hard should they collapse onto the central screening line vs. hold their width* to
//! stay with the wide man?
//!
//! ## Why this module exists
//!
//! [`super::goal_side`] gives a recovering defender a **fixed** lateral pull toward the
//! ball→own-goal line — `guarded.x += (line_x - guarded.x) * GOAL_SIDE_LATERAL_PULL_FRACTION`
//! with `GOAL_SIDE_LATERAL_PULL_FRACTION = 0.35`. That is a sound *predilection* (screen the
//! middle, don't just stand deep on your own touchline) but it was never a *decision*. The
//! right amount is situational: a central threat with cover behind wants a hard collapse onto
//! the line; a ball wide with a winger to mark wants the full-back to hold width instead of
//! vacating it. A single constant cannot express that.
//!
//! This module makes that one continuous choice — a signed **recovery bias** in `[-1, 1]` —
//! learnable, mirroring [`super::lane_affinity_decision`] and [`super::receive_approach`]:
//!
//!   * [`GoalSideRecoveryInputs`] — the per-defender recovery state (goal-side quality, how
//!     far off the line, how central the ball/threat is, whether this defender is the natural
//!     central screener, cover behind, the value of holding width, depth, full-back vs.
//!     centre-back), captured mirror-invariantly.
//!   * [`analytic_goal_side_recovery_bias`] — the coach's prior: **bias ≈ 0 (keep the 0.35
//!     shade) by default**, lean toward a *harder* collapse onto the line when the ball is
//!     central, this is the screening defender, and there is cover behind; lean toward
//!     *holding width* when the ball is wide and this is the full-back with a wide man to mark.
//!   * [`GoalSideRecoveryHead`] — a `FeedForwardNetwork` (`DIM → hidden → 1`, tanh) trained by
//!     reward-weighted regression toward the recovery actions that best protected the goal.
//!
//! The live seam is [`WorldSnapshot::goal_side_effective_lateral_pull_fraction`]: it shifts the
//! base 0.35 fraction by `bias * GOAL_SIDE_RECOVERY_MAX_FRACTION_ADJUST`, clamped to `[0, 1]`.
//! Positive ⇒ collapse harder onto the line; negative ⇒ hold width.
//!
//! ## Parity
//!
//! Doubly inert unless live: the whole lateral pull only exists when
//! [`super::goal_side::defensive_goal_side_enabled`] is on, and this modulation is behind its
//! own gate [`goal_side_recovery_model_enabled`] — **on by default in production** (analytic
//! prior ≈ 0 ⇒ near-identical), **off under `#[cfg(test)]`** so the suite is byte-identical
//! unless a test opts in. Off ⇒ the base 0.35 fraction stands, before any features are built.

use super::*;
use crate::des::general::neural_network::{ActivationName, FeedForwardNetwork, RandomNetworkSpec};
use crate::des::general::prng::mulberry32;

/// Number of features in the goal-side recovery state vector. Single source of truth in
/// [`GoalSideRecoveryInputs::to_features`].
pub const GOAL_SIDE_RECOVERY_FEATURE_DIM: usize = 10;
/// Hidden width of the learned regression head.
pub const GOAL_SIDE_RECOVERY_HIDDEN_UNITS: usize = 16;
/// Maximum the learned bias may move the base lateral-pull fraction, in fraction units. A bias
/// of `+1` collapses `GOAL_SIDE_RECOVERY_MAX_FRACTION_ADJUST` harder onto the line; `-1` holds
/// that much more width. Kept below the room to either edge so the shade is refined, not flipped.
pub const GOAL_SIDE_RECOVERY_MAX_FRACTION_ADJUST: f64 = 0.35;

/// Reference lateral distance (yd) normalizing off-line / opponent-distance features.
const REF_LATERAL_YARDS: f64 = 20.0;
/// Reference distance (yd) normalizing ball / depth features.
const REF_DISTANCE_YARDS: f64 = 40.0;

const GOAL_SIDE_RECOVERY_MODEL_ENABLE_ENV: &str = "DD_SOCCER_ENABLE_GOAL_SIDE_RECOVERY_MODEL";
const GOAL_SIDE_RECOVERY_MODEL_DISABLE_ENV: &str = "DD_SOCCER_DISABLE_GOAL_SIDE_RECOVERY_MODEL";

fn env_flag(name: &str) -> Option<bool> {
    std::env::var(name).ok().map(|raw| {
        let v = raw.trim().to_ascii_lowercase();
        matches!(v.as_str(), "1" | "true" | "yes" | "on")
    })
}

fn goal_side_recovery_model_enabled_from_env() -> bool {
    if env_flag(GOAL_SIDE_RECOVERY_MODEL_DISABLE_ENV).unwrap_or(false) {
        return false;
    }
    env_flag(GOAL_SIDE_RECOVERY_MODEL_ENABLE_ENV).unwrap_or(true)
}

/// Whether the learned goal-side recovery modulation is consulted this process. Off ⇒ the base
/// 0.35 lateral pull stands. Production caches once (default **on**); tests re-read and default
/// **off** so the suite is byte-identical unless a test opts in.
pub fn goal_side_recovery_model_enabled() -> bool {
    #[cfg(test)]
    {
        if env_flag(GOAL_SIDE_RECOVERY_MODEL_DISABLE_ENV).unwrap_or(false) {
            return false;
        }
        env_flag(GOAL_SIDE_RECOVERY_MODEL_ENABLE_ENV).unwrap_or(false)
    }
    #[cfg(not(test))]
    {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(goal_side_recovery_model_enabled_from_env)
    }
}

/// How often (ticks) goal-side recovery RL decisions are sampled while the model is on.
pub const GOAL_SIDE_RECOVERY_SAMPLE_INTERVAL_TICKS: u64 = 6;
/// Window (ticks) over which a sampled decision's reward is measured.
pub const GOAL_SIDE_RECOVERY_REWARD_WINDOW_TICKS: u64 = 30;
/// Cap on the rolling RL sample buffer (drained by the learner).
pub const GOAL_SIDE_RECOVERY_SAMPLE_CAP: usize = 8192;
/// Minimum training steps before a trained head is consumed live; below this the seam falls
/// back to the analytic seed.
pub const GOAL_SIDE_RECOVERY_HEAD_MIN_TRAINING_STEPS: usize = 200;
/// Weight on the dense territorial-advantage delta added to the windowed real-reward sum.
pub const GOAL_SIDE_RECOVERY_TERRITORIAL_SHAPING_WEIGHT: f64 = 1.0;

/// Raw per-defender recovery state the choice is a function of, captured mirror-invariantly.
#[derive(Clone, Debug, PartialEq)]
pub struct GoalSideRecoveryInputs {
    /// Signed goal-side quality `[-1, 1]` against the ball→own-goal line (`+` genuinely
    /// goal-side and on the line, `-` caught upfield).
    pub goal_side_quality: f64,
    /// Lateral gap (yd) from the defender to the ball→own-goal line — how far off the screening
    /// line they currently sit.
    pub lateral_gap_to_line: f64,
    /// Ball centrality in `[0, 1]`: `0` the ball is central (screen the middle), `1` the ball is
    /// on the touchline (the wide man matters more than the central line).
    pub ball_centrality: f64,
    /// Distance (yd) from the defender to the ball.
    pub dist_to_ball: f64,
    /// The defender's depth from their own goal (yd) — deeper ⇒ nearer the last line.
    pub depth_from_own_goal: f64,
    /// Count (normalized) of team-mates already goal-side of the ball — the cover that frees
    /// this defender to leave width and collapse onto the line.
    pub cover_behind: f64,
    /// Whether this defender is the team's **nearest** to the ball→goal line in `[0, 1]` (`1` ⇒
    /// the natural central screener; `0` ⇒ someone else screens, so hold your job).
    pub is_primary_screener: f64,
    /// Value of holding width in `[0, 1]`: how tightly an opponent occupies the defender's
    /// current lateral zone (`1` ⇒ a wide man right there to mark, so do not vacate).
    pub width_hold_value: f64,
    /// `1` if this is a wide defender (full-back), else `0` (centre-back).
    pub is_wide_defender: f64,
    /// Ball danger in `[0, 1]`: how deep toward our own goal the ball is (`1` ⇒ deep in our
    /// third, where screening the middle is most vital).
    pub ball_danger: f64,
}

impl GoalSideRecoveryInputs {
    /// The normalized network input vector. **Single source of truth** for the feature layout.
    pub fn to_features(&self) -> [f64; GOAL_SIDE_RECOVERY_FEATURE_DIM] {
        let lat = |d: f64| (d / REF_LATERAL_YARDS).clamp(0.0, 2.0);
        let dist = |d: f64| (d / REF_DISTANCE_YARDS).clamp(0.0, 2.0);
        let unit = |u: f64| u.clamp(0.0, 1.0);
        [
            self.goal_side_quality.clamp(-1.0, 1.0),
            lat(self.lateral_gap_to_line),
            unit(self.ball_centrality),
            dist(self.dist_to_ball),
            dist(self.depth_from_own_goal),
            unit(self.cover_behind),
            unit(self.is_primary_screener),
            unit(self.width_hold_value),
            self.is_wide_defender,
            unit(self.ball_danger),
        ]
    }
}

/// Analytic, deterministic seed for the **recovery bias** in `[-1, 1]` (positive = collapse
/// harder onto the ball→goal line; negative = hold width for the wide man).
///
/// The coach's prior: **keep the 0.35 shade (bias ≈ 0) by default.** Collapse harder onto the
/// line when the ball is central, this is the screening defender, there is cover behind, and the
/// defender sits off the line. Hold width when the ball is wide and this is the full-back with an
/// opponent occupying their zone.
pub fn analytic_goal_side_recovery_bias(inputs: &GoalSideRecoveryInputs) -> f64 {
    let mut z = 0.0;
    let central = (1.0 - inputs.ball_centrality).clamp(0.0, 1.0);
    let off_line = (inputs.lateral_gap_to_line / REF_LATERAL_YARDS).clamp(0.0, 1.0);

    // COLLAPSE onto the line (positive): central threat, I'm the screener, cover exists, I'm off
    // the line, and it's dangerous.
    z += 0.9
        * central
        * inputs.is_primary_screener
        * inputs.cover_behind
        * off_line
        * (0.5 + 0.5 * inputs.ball_danger);

    // HOLD width (negative): wide ball, I'm the full-back, an opponent occupies my zone.
    z -= 0.8 * inputs.ball_centrality * inputs.is_wide_defender * inputs.width_hold_value;

    z.tanh()
}

/// One reward-weighted RL training row for the goal-side recovery head.
#[derive(Clone, Debug, PartialEq)]
pub struct GoalSideRecoverySample {
    pub inputs: GoalSideRecoveryInputs,
    /// The recovery action actually taken, in `[-1, 1]` (`+` = collapsed onto the line).
    pub action_bias: f64,
    pub reward: f64,
}

/// An open goal-side recovery decision awaiting its windowed reward.
#[derive(Clone, Debug, PartialEq)]
pub struct PendingGoalSideRecoveryDecision {
    pub team: Team,
    pub player_id: usize,
    pub inputs: GoalSideRecoveryInputs,
    pub action_bias: f64,
    pub decision_territorial: f64,
    /// Running sum of the team's real reward-pipeline events over the window (interceptions,
    /// duel/loose-ball wins, concede penalties — the reward the engine optimizes).
    pub reward_accum: f64,
    pub due_tick: u64,
}

/// Learned regression head for the signed recovery bias.
#[derive(Clone, Debug)]
pub struct GoalSideRecoveryHead {
    network: FeedForwardNetwork,
    training_steps: usize,
    last_loss: Option<f64>,
}

impl GoalSideRecoveryHead {
    pub fn new(seed: u32) -> Self {
        let mut rng = mulberry32(seed ^ 0x6D3A_11F5);
        let network = FeedForwardNetwork::random(
            &RandomNetworkSpec {
                input_dim: GOAL_SIDE_RECOVERY_FEATURE_DIM,
                hidden_layers: vec![GOAL_SIDE_RECOVERY_HIDDEN_UNITS],
                output_dim: 1,
                hidden_activation: ActivationName::Tanh,
                output_activation: ActivationName::Tanh,
                weight_scale: None,
            },
            &mut rng,
        );
        GoalSideRecoveryHead {
            network,
            training_steps: 0,
            last_loss: None,
        }
    }

    pub fn predict(&self, inputs: &GoalSideRecoveryInputs) -> Option<f64> {
        let features = inputs.to_features();
        if features.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let out = self.network.predict(&features[..]);
        out.first().copied().filter(|p| p.is_finite())
    }

    pub fn train(&mut self, samples: &[(GoalSideRecoveryInputs, f64)], learning_rate: f64) -> f64 {
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
        let mean = if applied > 0 { total / applied as f64 } else { 0.0 };
        self.last_loss = Some(mean);
        mean
    }

    /// One reward-weighted-regression epoch. Mirrors
    /// [`super::lane_affinity_decision::LaneAffinityHead::train_reward_weighted`].
    pub fn train_reward_weighted(
        &mut self,
        samples: &[GoalSideRecoverySample],
        learning_rate: f64,
    ) -> f64 {
        let finite: Vec<&GoalSideRecoverySample> = samples
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

/// Outcome of training a goal-side recovery head on a drained RL corpus.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GoalSideRecoveryTrainingReport {
    pub samples: usize,
    pub training_steps: usize,
    pub epochs: usize,
    pub final_loss: f64,
}

/// Load-and-train entry for a goal-side recovery head.
pub fn train_goal_side_recovery_head(
    samples: &[GoalSideRecoverySample],
    seed: u32,
    epochs: usize,
    learning_rate: f64,
) -> Option<(GoalSideRecoveryHead, GoalSideRecoveryTrainingReport)> {
    let usable = samples
        .iter()
        .filter(|s| s.reward.is_finite() && s.action_bias.is_finite())
        .count();
    if usable == 0 || epochs == 0 {
        return None;
    }
    let mut head = GoalSideRecoveryHead::new(seed);
    let mut final_loss = 0.0;
    for _ in 0..epochs {
        final_loss = head.train_reward_weighted(samples, learning_rate);
    }
    let report = GoalSideRecoveryTrainingReport {
        samples: usable,
        training_steps: head.training_steps(),
        epochs,
        final_loss,
    };
    Some((head, report))
}

/// Public wrapper returning ONLY the report (the head type is engine-internal).
pub fn report_goal_side_recovery_training(
    samples: &[GoalSideRecoverySample],
    seed: u32,
    epochs: usize,
    learning_rate: f64,
) -> Option<GoalSideRecoveryTrainingReport> {
    train_goal_side_recovery_head(samples, seed, epochs, learning_rate).map(|(_, report)| report)
}

impl WorldSnapshot {
    /// Centre of `team`'s own goal.
    fn goal_side_own_goal_center(&self, team: Team) -> Vec2 {
        let y = match team {
            Team::Home => 0.0,
            Team::Away => self.field_length,
        };
        Vec2 { x: self.field_width * 0.5, y }
    }

    /// Build the [`GoalSideRecoveryInputs`] for a recovering `player` whose depth-guarded target
    /// `guarded` is being shaded toward the ball→goal line crossing at `line_x`. `None` for a
    /// non-defender or degenerate geometry. Pure / RNG-free.
    pub(crate) fn build_goal_side_recovery_inputs(
        &self,
        player: &PlayerSnapshot,
        guarded: Vec2,
        line_x: f64,
    ) -> Option<GoalSideRecoveryInputs> {
        if player.role != PlayerRole::Defender {
            return None;
        }
        let own_goal = self.goal_side_own_goal_center(player.team);
        let me_pos = self.player_snapshot_position(player);
        let q = goal_side::goal_side_quality(
            me_pos,
            self.ball.position,
            own_goal,
            goal_side::GOAL_SIDE_CHANNEL_HALF_WIDTH_YARDS,
            goal_side::GOAL_SIDE_DEPTH_SATURATION_YARDS,
        );
        let lateral_gap_to_line = (line_x - guarded.x).abs();
        let half = (self.field_width * 0.5).max(1.0);
        let ball_centrality = ((self.ball.position.x - half).abs() / half).clamp(0.0, 1.0);
        let dist_to_ball = me_pos.distance(self.ball.position);
        let depth_from_own_goal = self.depth_from_own_goal_y(player.team, me_pos.y);
        let ball_depth = self.depth_from_own_goal_y(player.team, self.ball.position.y);

        // Cover behind: team-mates already goal-side of the ball (deeper toward own goal).
        let cover_behind_count = self
            .players
            .iter()
            .filter(|p| {
                p.team == player.team
                    && p.id != player.id
                    && self.depth_from_own_goal_y(player.team, self.player_snapshot_position(p).y)
                        < ball_depth
            })
            .count();
        let cover_behind = (cover_behind_count as f64 / 4.0).clamp(0.0, 1.0);

        // Primary screener: am I the team's nearest to the ball→goal line?
        let my_perp = self.goal_side_perp_distance(me_pos, self.ball.position, own_goal);
        let anyone_closer = self.players.iter().any(|p| {
            p.team == player.team
                && p.id != player.id
                && matches!(p.role, PlayerRole::Defender | PlayerRole::Midfielder)
                && self.goal_side_perp_distance(
                    self.player_snapshot_position(p),
                    self.ball.position,
                    own_goal,
                ) < my_perp - 1e-3
        });
        let is_primary_screener = if anyone_closer { 0.0 } else { 1.0 };

        // Width-hold value: how tightly an opponent occupies my current lateral zone.
        let nearest_opp_at_hold = self.nearest_opponent_distance_at(player.team, me_pos);
        let width_hold_value = (1.0 - (nearest_opp_at_hold / 12.0).clamp(0.0, 1.0)).clamp(0.0, 1.0);

        let ball_danger = (1.0 - (ball_depth / (self.field_length * 0.5).max(1.0)))
            .clamp(0.0, 1.0);

        Some(GoalSideRecoveryInputs {
            goal_side_quality: q,
            lateral_gap_to_line,
            ball_centrality,
            dist_to_ball,
            depth_from_own_goal,
            cover_behind,
            is_primary_screener,
            width_hold_value,
            is_wide_defender: self.is_wide_defender(player) as i32 as f64,
            ball_danger,
        })
    }

    /// Lateral (perpendicular) distance from `p` to the ball→own-goal line.
    fn goal_side_perp_distance(&self, p: Vec2, ball: Vec2, own_goal: Vec2) -> f64 {
        let axis = own_goal - ball;
        let len = axis.len();
        if !len.is_finite() || len < 1e-3 {
            return f64::INFINITY;
        }
        let unit = axis * (1.0 / len);
        let rel = p - ball;
        (rel - unit * rel.dot(unit)).len()
    }

    /// The model's signed recovery bias in `(-1, 1)`, or `None` when gated off / degenerate.
    pub(crate) fn goal_side_recovery_bias(&self, inputs: &GoalSideRecoveryInputs) -> Option<f64> {
        if !goal_side_recovery_model_enabled() {
            return None;
        }
        let bias = self
            .goal_side_recovery_head
            .as_ref()
            .filter(|head| head.training_steps() >= GOAL_SIDE_RECOVERY_HEAD_MIN_TRAINING_STEPS)
            .and_then(|head| head.predict(inputs))
            .unwrap_or_else(|| analytic_goal_side_recovery_bias(inputs));
        bias.is_finite().then(|| bias.clamp(-1.0, 1.0))
    }

    /// The effective lateral-pull fraction to apply, given the base `GOAL_SIDE_LATERAL_PULL_FRACTION`.
    /// Off (or degenerate) ⇒ `base_fraction` unchanged (byte-identical). On ⇒ shifted by the learned
    /// recovery bias, clamped `[0, 1]`.
    pub(crate) fn goal_side_effective_lateral_pull_fraction(
        &self,
        player: &PlayerSnapshot,
        guarded: Vec2,
        line_x: f64,
        base_fraction: f64,
    ) -> f64 {
        if !goal_side_recovery_model_enabled() {
            return base_fraction;
        }
        let Some(inputs) = self.build_goal_side_recovery_inputs(player, guarded, line_x) else {
            return base_fraction;
        };
        let Some(bias) = self.goal_side_recovery_bias(&inputs) else {
            return base_fraction;
        };
        (base_fraction + bias * GOAL_SIDE_RECOVERY_MAX_FRACTION_ADJUST).clamp(0.0, 1.0)
    }
}

impl SoccerMatch {
    /// Gated per-tick RL sample collection for the goal-side recovery head. No-op unless the
    /// model is enabled. Windows the team's real reward-pipeline events + territorial delta.
    pub(crate) fn collect_goal_side_recovery_rl_samples(&mut self, snapshot: &WorldSnapshot) {
        if !goal_side_recovery_model_enabled() || !goal_side::defensive_goal_side_enabled() {
            return;
        }
        let tick = self.tick;
        // Accumulate this tick's real reward-pipeline events per team into every open decision.
        if !self.pending_goal_side_recovery.is_empty() && !self.reward_events.is_empty() {
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
            for decision in &mut self.pending_goal_side_recovery {
                decision.reward_accum += match decision.team {
                    Team::Home => home_reward,
                    Team::Away => away_reward,
                };
            }
        }
        // Resolve due decisions.
        let mut i = 0;
        while i < self.pending_goal_side_recovery.len() {
            if self.pending_goal_side_recovery[i].due_tick <= tick {
                let decision = self.pending_goal_side_recovery.swap_remove(i);
                let now_territorial = territorial_advantage(snapshot, decision.team);
                let territorial_delta = if now_territorial.is_finite()
                    && decision.decision_territorial.is_finite()
                {
                    now_territorial - decision.decision_territorial
                } else {
                    0.0
                };
                let reward = decision.reward_accum
                    + GOAL_SIDE_RECOVERY_TERRITORIAL_SHAPING_WEIGHT * territorial_delta;
                if reward.is_finite() {
                    self.goal_side_recovery_samples.push(GoalSideRecoverySample {
                        inputs: decision.inputs,
                        action_bias: decision.action_bias,
                        reward,
                    });
                    if self.goal_side_recovery_samples.len() > GOAL_SIDE_RECOVERY_SAMPLE_CAP {
                        let overflow =
                            self.goal_side_recovery_samples.len() - GOAL_SIDE_RECOVERY_SAMPLE_CAP;
                        self.goal_side_recovery_samples.drain(0..overflow);
                    }
                }
            } else {
                i += 1;
            }
        }
        // Sample fresh per-defender decisions on cadence, only while defending.
        if tick % GOAL_SIDE_RECOVERY_SAMPLE_INTERVAL_TICKS != 0 {
            return;
        }
        let defending: Vec<usize> = snapshot
            .players
            .iter()
            .filter(|p| p.role == PlayerRole::Defender)
            .filter(|p| snapshot.possession_team() == Some(p.team.other()))
            .map(|p| p.id)
            .collect();
        for id in defending {
            let Some(player) = snapshot.players.iter().find(|p| p.id == id) else {
                continue;
            };
            let me_pos = snapshot.player_snapshot_position(player);
            let own_goal = snapshot.goal_side_own_goal_center(player.team);
            let depth = snapshot.depth_from_own_goal_y(player.team, me_pos.y);
            let Some(line_x) =
                goal_side::ball_goal_line_x_at_y(snapshot.ball.position, own_goal, me_pos.y)
            else {
                continue;
            };
            let _ = depth;
            let Some(inputs) = snapshot.build_goal_side_recovery_inputs(player, me_pos, line_x)
            else {
                continue;
            };
            let action_bias = snapshot
                .goal_side_recovery_bias(&inputs)
                .unwrap_or_else(|| analytic_goal_side_recovery_bias(&inputs));
            let territorial = territorial_advantage(snapshot, player.team);
            if territorial.is_finite() {
                self.pending_goal_side_recovery
                    .push(PendingGoalSideRecoveryDecision {
                        team: player.team,
                        player_id: id,
                        inputs,
                        action_bias,
                        decision_territorial: territorial,
                        reward_accum: 0.0,
                        due_tick: tick + GOAL_SIDE_RECOVERY_REWARD_WINDOW_TICKS,
                    });
            }
        }
    }

    /// Drain the collected goal-side recovery RL samples for the cluster learner.
    pub fn drain_goal_side_recovery_samples(&mut self) -> Vec<GoalSideRecoverySample> {
        std::mem::take(&mut self.goal_side_recovery_samples)
    }

    /// Install the trained goal-side recovery head for live consumption.
    pub fn set_goal_side_recovery_head(&mut self, head: GoalSideRecoveryHead) {
        self.goal_side_recovery_head = Some(std::sync::Arc::new(head));
    }
}

#[cfg(test)]
mod goal_side_recovery_tests {
    use super::*;

    fn baseline_inputs() -> GoalSideRecoveryInputs {
        GoalSideRecoveryInputs {
            goal_side_quality: 0.3,
            lateral_gap_to_line: 6.0,
            ball_centrality: 0.5,
            dist_to_ball: 20.0,
            depth_from_own_goal: 25.0,
            cover_behind: 0.5,
            is_primary_screener: 0.0,
            width_hold_value: 0.2,
            is_wide_defender: 0.0,
            ball_danger: 0.4,
        }
    }

    #[test]
    fn features_are_finite_bounded_and_right_dim() {
        let f = baseline_inputs().to_features();
        assert_eq!(f.len(), GOAL_SIDE_RECOVERY_FEATURE_DIM);
        for v in f {
            assert!(v.is_finite());
            assert!(v.abs() <= 2.0 + 1e-9, "feature {v} out of normalized range");
        }
    }

    #[test]
    fn analytic_bias_is_a_valid_signed_unit() {
        let b = analytic_goal_side_recovery_bias(&baseline_inputs());
        assert!((-1.0..=1.0).contains(&b), "bias {b} out of [-1,1]");
    }

    #[test]
    fn central_threat_screener_with_cover_collapses_onto_the_line() {
        let mut i = baseline_inputs();
        i.ball_centrality = 0.0; // dead central
        i.is_primary_screener = 1.0;
        i.cover_behind = 1.0;
        i.lateral_gap_to_line = 15.0; // off the line
        i.ball_danger = 1.0;
        assert!(
            analytic_goal_side_recovery_bias(&i) > 0.0,
            "the central screener with cover should collapse onto the line"
        );
    }

    #[test]
    fn wide_ball_fullback_with_a_man_to_mark_holds_width() {
        let mut i = baseline_inputs();
        i.ball_centrality = 1.0; // ball wide
        i.is_wide_defender = 1.0;
        i.width_hold_value = 1.0; // opponent right in my zone
        i.is_primary_screener = 0.0;
        assert!(
            analytic_goal_side_recovery_bias(&i) < 0.0,
            "a full-back with a wide man to mark should hold width, not vacate it"
        );
    }

    #[test]
    fn neutral_state_keeps_the_base_shade() {
        let mut i = baseline_inputs();
        i.is_primary_screener = 0.0;
        i.width_hold_value = 0.0;
        assert!(
            analytic_goal_side_recovery_bias(&i).abs() < 0.05,
            "a neutral state should barely move the base 0.35 shade"
        );
    }

    #[test]
    fn head_predicts_a_bounded_signed_bias() {
        let head = GoalSideRecoveryHead::new(5);
        let b = head.predict(&baseline_inputs()).expect("finite prediction");
        assert!((-1.0..=1.0).contains(&b));
    }

    #[test]
    fn reward_weighted_training_steers_toward_the_high_reward_action() {
        let mut head = GoalSideRecoveryHead::new(17);
        let inputs = baseline_inputs();
        let before = head.predict(&inputs).expect("finite");
        let samples: Vec<GoalSideRecoverySample> = (0..32)
            .flat_map(|_| {
                [
                    GoalSideRecoverySample { inputs: inputs.clone(), action_bias: 0.8, reward: 1.0 },
                    GoalSideRecoverySample { inputs: inputs.clone(), action_bias: -0.8, reward: -1.0 },
                ]
            })
            .collect();
        let mut last = f64::INFINITY;
        for _ in 0..80 {
            last = head.train_reward_weighted(&samples, 0.05);
        }
        let after = head.predict(&inputs).expect("finite");
        assert!(last.is_finite());
        assert!(after > before && after > 0.0, "RWR should move toward the high-reward action");
    }

    #[test]
    fn train_entry_skips_empty_and_trains_valid() {
        assert!(train_goal_side_recovery_head(&[], 1, 4, 0.02).is_none());
        let samples: Vec<GoalSideRecoverySample> = (0..16)
            .map(|i| GoalSideRecoverySample {
                inputs: baseline_inputs(),
                action_bias: if i % 2 == 0 { 0.8 } else { -0.8 },
                reward: if i % 2 == 0 { 1.0 } else { -1.0 },
            })
            .collect();
        let (_, report) =
            train_goal_side_recovery_head(&samples, 1, 4, 0.02).expect("trains a usable corpus");
        assert_eq!(report.samples, 16);
        assert_eq!(report.epochs, 4);
        assert!(report.training_steps > 0 && report.final_loss.is_finite());
    }
}
