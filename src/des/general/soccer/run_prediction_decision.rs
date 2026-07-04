//! Learnable **off-ball run selection** — makes *which run a player commits to* off the scored
//! destination surface a learned MDP/POMDP choice instead of the pure argmax.
//!
//! [`WorldSnapshot::open_space_for`] scores a grid of candidate off-ball destinations and takes the
//! argmax (then a multimodal blend). When the surface is bimodal — "cut inside" vs "hold the
//! byline", a forward gamble vs a safe show — the argmax picks by raw score alone. This module lets
//! the player express a learned **run-forward preference** in `[-1, 1]` (positive ⇒ tilt toward the
//! more forward / aggressive destination, negative ⇒ the safer one) by adding a forwardness term to
//! the candidate scores before the argmax. At bias `0` the argmax is unchanged (parity).
//!
//! Mirrors [`super::lane_affinity_decision`]. Parity: doubly inert — the multimodal run surface is
//! gated by [`super::run_prediction::multimodal_run_prediction_enabled`], and this modulation by
//! [`run_prediction_model_enabled`] (on in prod, off under `#[cfg(test)]`). Off ⇒ the argmax stands.

use super::*;
use crate::des::general::neural_network::{ActivationName, FeedForwardNetwork, RandomNetworkSpec};
use crate::des::general::prng::mulberry32;

pub const RUN_PREDICTION_FEATURE_DIM: usize = 6;
pub const RUN_PREDICTION_HIDDEN_UNITS: usize = 12;
/// Score weight per yard of forwardness the learned bias adds to a candidate before the argmax.
pub const RUN_PREDICTION_FORWARD_WEIGHT: f64 = 0.12;

const RUN_PREDICTION_MODEL_ENABLE_ENV: &str = "DD_SOCCER_ENABLE_RUN_PREDICTION_MODEL";
const RUN_PREDICTION_MODEL_DISABLE_ENV: &str = "DD_SOCCER_DISABLE_RUN_PREDICTION_MODEL";

fn env_flag(name: &str) -> Option<bool> {
    std::env::var(name).ok().map(|raw| {
        matches!(raw.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on")
    })
}

fn model_enabled_from_env() -> bool {
    if env_flag(RUN_PREDICTION_MODEL_DISABLE_ENV).unwrap_or(false) {
        return false;
    }
    env_flag(RUN_PREDICTION_MODEL_ENABLE_ENV).unwrap_or(true)
}

pub fn run_prediction_model_enabled() -> bool {
    #[cfg(test)]
    {
        if env_flag(RUN_PREDICTION_MODEL_DISABLE_ENV).unwrap_or(false) {
            return false;
        }
        env_flag(RUN_PREDICTION_MODEL_ENABLE_ENV).unwrap_or(false)
    }
    #[cfg(not(test))]
    {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(model_enabled_from_env)
    }
}

pub const RUN_PREDICTION_SAMPLE_INTERVAL_TICKS: u64 = 6;
pub const RUN_PREDICTION_REWARD_WINDOW_TICKS: u64 = 35;
pub const RUN_PREDICTION_SAMPLE_CAP: usize = 8192;
pub const RUN_PREDICTION_HEAD_MIN_TRAINING_STEPS: usize = 200;
pub const RUN_PREDICTION_TERRITORIAL_SHAPING_WEIGHT: f64 = 1.0;

#[derive(Clone, Debug, PartialEq)]
pub struct RunPredictionInputs {
    /// Pressure on this player in `[0, 1]`.
    pub pressure: f64,
    /// Space ahead (toward the attacking goal) in `[0, 1]`.
    pub space_ahead: f64,
    /// `1` if in the attacking final third.
    pub in_final_third: f64,
    /// Attacking numbers advantage ahead of the ball in `[0, 1]` (`1` ⇒ big overload).
    pub numbers_up: f64,
    /// `1` if a forward, else `0`.
    pub role_forward: f64,
    /// Offensive mindedness in `[0, 1]`.
    pub offensive_mindedness: f64,
}

impl RunPredictionInputs {
    pub fn to_features(&self) -> [f64; RUN_PREDICTION_FEATURE_DIM] {
        let unit = |u: f64| u.clamp(0.0, 1.0);
        [
            unit(self.pressure),
            unit(self.space_ahead),
            unit(self.in_final_third),
            unit(self.numbers_up),
            self.role_forward,
            unit(self.offensive_mindedness),
        ]
    }
}

/// Analytic seed for **run-forward preference** in `[-1, 1]` (positive = more forward run). Prior:
/// ≈ 0 by default; tilt forward with an overload and space ahead; safer under pressure.
pub fn analytic_run_forward_preference(inputs: &RunPredictionInputs) -> f64 {
    let mut z = 0.0;
    z += 0.8 * inputs.numbers_up * inputs.space_ahead;
    z += 0.3 * inputs.offensive_mindedness * inputs.space_ahead;
    z -= 0.6 * inputs.pressure * (1.0 - inputs.space_ahead);
    z.tanh()
}

#[derive(Clone, Debug, PartialEq)]
pub struct RunPredictionSample {
    pub inputs: RunPredictionInputs,
    pub action_bias: f64,
    pub reward: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PendingRunPredictionDecision {
    pub team: Team,
    pub player_id: usize,
    pub inputs: RunPredictionInputs,
    pub action_bias: f64,
    pub decision_territorial: f64,
    pub reward_accum: f64,
    pub due_tick: u64,
}

#[derive(Clone, Debug)]
pub struct RunPredictionHead {
    network: FeedForwardNetwork,
    training_steps: usize,
    last_loss: Option<f64>,
}

impl RunPredictionHead {
    pub fn new(seed: u32) -> Self {
        let mut rng = mulberry32(seed ^ 0x63A9_1D77);
        let network = FeedForwardNetwork::random(
            &RandomNetworkSpec {
                input_dim: RUN_PREDICTION_FEATURE_DIM,
                hidden_layers: vec![RUN_PREDICTION_HIDDEN_UNITS],
                output_dim: 1,
                hidden_activation: ActivationName::Tanh,
                output_activation: ActivationName::Tanh,
                weight_scale: None,
            },
            &mut rng,
        );
        RunPredictionHead { network, training_steps: 0, last_loss: None }
    }

    pub fn predict(&self, inputs: &RunPredictionInputs) -> Option<f64> {
        let features = inputs.to_features();
        if features.iter().any(|v| !v.is_finite()) {
            return None;
        }
        self.network.predict(&features[..]).first().copied().filter(|p| p.is_finite())
    }

    pub fn train_reward_weighted(&mut self, samples: &[RunPredictionSample], learning_rate: f64) -> f64 {
        let finite: Vec<&RunPredictionSample> = samples
            .iter()
            .filter(|s| s.reward.is_finite() && s.action_bias.is_finite())
            .collect();
        if finite.is_empty() {
            return 0.0;
        }
        let n = finite.len() as f64;
        let baseline = finite.iter().map(|s| s.reward).sum::<f64>() / n;
        let std = (finite.iter().map(|s| (s.reward - baseline).powi(2)).sum::<f64>() / n)
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
            let result =
                self.network.train_sample_clipped(&features[..], &target, learning_rate * weight, 4.0);
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

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunPredictionTrainingReport {
    pub samples: usize,
    pub training_steps: usize,
    pub epochs: usize,
    pub final_loss: f64,
}

pub fn train_run_prediction_head(
    samples: &[RunPredictionSample],
    seed: u32,
    epochs: usize,
    learning_rate: f64,
) -> Option<(RunPredictionHead, RunPredictionTrainingReport)> {
    let usable = samples
        .iter()
        .filter(|s| s.reward.is_finite() && s.action_bias.is_finite())
        .count();
    if usable == 0 || epochs == 0 {
        return None;
    }
    let mut head = RunPredictionHead::new(seed);
    let mut final_loss = 0.0;
    for _ in 0..epochs {
        final_loss = head.train_reward_weighted(samples, learning_rate);
    }
    let report = RunPredictionTrainingReport {
        samples: usable,
        training_steps: head.training_steps(),
        epochs,
        final_loss,
    };
    Some((head, report))
}

pub fn report_run_prediction_training(
    samples: &[RunPredictionSample],
    seed: u32,
    epochs: usize,
    learning_rate: f64,
) -> Option<RunPredictionTrainingReport> {
    train_run_prediction_head(samples, seed, epochs, learning_rate).map(|(_, report)| report)
}

impl WorldSnapshot {
    /// Build the [`RunPredictionInputs`] for an off-ball `player`. Pure / RNG-free.
    pub(crate) fn build_run_prediction_inputs(&self, player: &PlayerSnapshot) -> Option<RunPredictionInputs> {
        if !matches!(player.role, PlayerRole::Forward | PlayerRole::Midfielder) {
            return None;
        }
        let pos = self.player_snapshot_position(player);
        let attack_dir = player.team.attack_dir();
        let goal_y = player.team.goal_y(self.field_length);
        let pressure =
            (1.0 - (self.nearest_opponent_distance_at(player.team, pos) / 8.0).clamp(0.0, 1.0)).clamp(0.0, 1.0);
        let ahead = Vec2 { x: pos.x, y: pos.y + attack_dir * 10.0 };
        let space_ahead = (self.nearest_opponent_distance_at(player.team, ahead) / 12.0).clamp(0.0, 1.0);
        let ball_fwd = self.ball.position.y * attack_dir;
        let attackers_ahead = self
            .players
            .iter()
            .filter(|p| p.team == player.team && self.player_snapshot_position(p).y * attack_dir > ball_fwd)
            .count();
        let defenders_ahead = self
            .players
            .iter()
            .filter(|p| p.team == player.team.other() && self.player_snapshot_position(p).y * attack_dir > ball_fwd)
            .count();
        let numbers_up = ((attackers_ahead as f64 - defenders_ahead as f64 + 3.0) / 6.0).clamp(0.0, 1.0);
        let in_final_third = crash_box::ball_in_attacking_final_third(pos.y, goal_y, self.field_length);
        Some(RunPredictionInputs {
            pressure,
            space_ahead,
            in_final_third: in_final_third as i32 as f64,
            numbers_up,
            role_forward: (player.role == PlayerRole::Forward) as i32 as f64,
            offensive_mindedness: player.preferences.offensive_mindedness.clamp(0.0, 1.0),
        })
    }

    /// The model's signed run-forward preference in `(-1, 1)`, or `None` when gated off.
    pub(crate) fn run_forward_preference(&self, inputs: &RunPredictionInputs) -> Option<f64> {
        if !run_prediction_model_enabled() {
            return None;
        }
        let bias = self
            .run_prediction_head
            .as_ref()
            .filter(|head| head.training_steps() >= RUN_PREDICTION_HEAD_MIN_TRAINING_STEPS)
            .and_then(|head| head.predict(inputs))
            .unwrap_or_else(|| analytic_run_forward_preference(inputs));
        bias.is_finite().then(|| bias.clamp(-1.0, 1.0))
    }

    /// Re-select the off-ball run destination from the scored candidate surface, tilting the score
    /// by a learned forwardness preference. Returns `fallback` unchanged when the model is off / no
    /// candidates ⇒ byte-identical (at bias 0 the tilt is 0 so the argmax is unchanged).
    pub(crate) fn run_prediction_tilted_destination(
        &self,
        player_id: usize,
        scored: &[(Vec2, f64)],
        fallback: Vec2,
    ) -> Vec2 {
        if !run_prediction_model_enabled() || scored.is_empty() {
            return fallback;
        }
        let Some(player) = self.players.iter().find(|p| p.id == player_id) else {
            return fallback;
        };
        let Some(inputs) = self.build_run_prediction_inputs(player) else {
            return fallback;
        };
        let Some(bias) = self.run_forward_preference(&inputs) else {
            return fallback;
        };
        if bias.abs() < 1e-9 {
            return fallback;
        }
        let pos = self.player_snapshot_position(player);
        let attack_dir = player.team.attack_dir();
        let mut best = fallback;
        let mut best_score = f64::NEG_INFINITY;
        for &(cand, score) in scored {
            if !score.is_finite() {
                continue;
            }
            let forwardness = (cand.y - pos.y) * attack_dir;
            let tilted = score + bias * RUN_PREDICTION_FORWARD_WEIGHT * forwardness;
            if tilted > best_score {
                best_score = tilted;
                best = cand;
            }
        }
        best
    }
}

impl SoccerMatch {
    /// Gated per-tick RL sample collection for the run-selection head. No-op unless enabled.
    pub(crate) fn collect_run_prediction_rl_samples(&mut self, snapshot: &WorldSnapshot) {
        if !run_prediction_model_enabled() || !run_prediction::multimodal_run_prediction_enabled() {
            return;
        }
        let tick = self.tick;
        if !self.pending_run_prediction.is_empty() && !self.reward_events.is_empty() {
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
            for decision in &mut self.pending_run_prediction {
                decision.reward_accum += match decision.team {
                    Team::Home => home_reward,
                    Team::Away => away_reward,
                };
            }
        }
        let mut i = 0;
        while i < self.pending_run_prediction.len() {
            if self.pending_run_prediction[i].due_tick <= tick {
                let decision = self.pending_run_prediction.swap_remove(i);
                let now_territorial = territorial_advantage(snapshot, decision.team);
                let territorial_delta = if now_territorial.is_finite()
                    && decision.decision_territorial.is_finite()
                {
                    now_territorial - decision.decision_territorial
                } else {
                    0.0
                };
                let reward = decision.reward_accum
                    + RUN_PREDICTION_TERRITORIAL_SHAPING_WEIGHT * territorial_delta;
                if reward.is_finite() {
                    self.run_prediction_samples.push(RunPredictionSample {
                        inputs: decision.inputs,
                        action_bias: decision.action_bias,
                        reward,
                    });
                    if self.run_prediction_samples.len() > RUN_PREDICTION_SAMPLE_CAP {
                        let overflow =
                            self.run_prediction_samples.len() - RUN_PREDICTION_SAMPLE_CAP;
                        self.run_prediction_samples.drain(0..overflow);
                    }
                }
            } else {
                i += 1;
            }
        }
        if tick % RUN_PREDICTION_SAMPLE_INTERVAL_TICKS != 0 {
            return;
        }
        let ids: Vec<usize> = snapshot
            .players
            .iter()
            .filter(|p| matches!(p.role, PlayerRole::Forward | PlayerRole::Midfielder))
            .filter(|p| snapshot.possession_team() == Some(p.team) && snapshot.ball.holder != Some(p.id))
            .map(|p| p.id)
            .collect();
        for id in ids {
            let Some(player) = snapshot.players.iter().find(|p| p.id == id) else {
                continue;
            };
            let Some(inputs) = snapshot.build_run_prediction_inputs(player) else {
                continue;
            };
            let action_bias = snapshot
                .run_forward_preference(&inputs)
                .unwrap_or_else(|| analytic_run_forward_preference(&inputs));
            let territorial = territorial_advantage(snapshot, player.team);
            if territorial.is_finite() {
                self.pending_run_prediction.push(PendingRunPredictionDecision {
                    team: player.team,
                    player_id: id,
                    inputs,
                    action_bias,
                    decision_territorial: territorial,
                    reward_accum: 0.0,
                    due_tick: tick + RUN_PREDICTION_REWARD_WINDOW_TICKS,
                });
            }
        }
    }

    pub fn drain_run_prediction_samples(&mut self) -> Vec<RunPredictionSample> {
        std::mem::take(&mut self.run_prediction_samples)
    }

    pub fn set_run_prediction_head(&mut self, head: RunPredictionHead) {
        self.run_prediction_head = Some(std::sync::Arc::new(head));
    }
}

#[cfg(test)]
mod run_prediction_model_tests {
    use super::*;

    fn baseline_inputs() -> RunPredictionInputs {
        RunPredictionInputs {
            pressure: 0.3,
            space_ahead: 0.6,
            in_final_third: 0.0,
            numbers_up: 0.5,
            role_forward: 1.0,
            offensive_mindedness: 0.6,
        }
    }

    #[test]
    fn features_finite_bounded_right_dim() {
        let f = baseline_inputs().to_features();
        assert_eq!(f.len(), RUN_PREDICTION_FEATURE_DIM);
        for v in f {
            assert!(v.is_finite() && (0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn overload_with_space_tilts_forward() {
        let mut i = baseline_inputs();
        i.numbers_up = 1.0;
        i.space_ahead = 1.0;
        i.pressure = 0.0;
        assert!(analytic_run_forward_preference(&i) > 0.0);
    }

    #[test]
    fn pressure_no_space_plays_safe() {
        let mut i = baseline_inputs();
        i.pressure = 1.0;
        i.space_ahead = 0.0;
        i.numbers_up = 0.2;
        assert!(analytic_run_forward_preference(&i) < 0.0);
    }

    #[test]
    fn head_bounded_and_rwr_separates() {
        let inputs = baseline_inputs();
        assert!((-1.0..=1.0).contains(&RunPredictionHead::new(1).predict(&inputs).unwrap()));
        let mut hi = RunPredictionHead::new(1);
        let mut lo = RunPredictionHead::new(1);
        let hi_s: Vec<RunPredictionSample> = (0..32)
            .flat_map(|_| [
                RunPredictionSample { inputs: inputs.clone(), action_bias: 0.8, reward: 1.0 },
                RunPredictionSample { inputs: inputs.clone(), action_bias: -0.8, reward: -1.0 },
            ])
            .collect();
        let lo_s: Vec<RunPredictionSample> = (0..32)
            .flat_map(|_| [
                RunPredictionSample { inputs: inputs.clone(), action_bias: -0.8, reward: 1.0 },
                RunPredictionSample { inputs: inputs.clone(), action_bias: 0.8, reward: -1.0 },
            ])
            .collect();
        for _ in 0..80 {
            hi.train_reward_weighted(&hi_s, 0.05);
            lo.train_reward_weighted(&lo_s, 0.05);
        }
        assert!(hi.predict(&inputs).unwrap() > lo.predict(&inputs).unwrap());
    }

    #[test]
    fn train_entry_skips_empty() {
        assert!(train_run_prediction_head(&[], 1, 4, 0.02).is_none());
    }
}
