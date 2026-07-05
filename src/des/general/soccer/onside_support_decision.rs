//! Learnable **onside-support push decision** — makes *how high an idle attacking supporter holds
//! against the offside line* a learned MDP/POMDP choice instead of a hard clamp to the line.
//!
//! [`WorldSnapshot::clamp_forward_onside_support`] caps an idle forward / attacking mid exactly at
//! the last-defender line (`target.y = cap_y`) so a ball to them is never flagged offside. That is a
//! safe *predilection*, but the elite skill is timing the shoulder: sitting a touch beyond the line
//! ready to break as the ball is played (gambling the line will drop), or tucking safely behind when
//! the line is stepping up to spring the trap. This module makes that a learned **onside push** in
//! `[-1, 1]` (positive ⇒ push onto / a yard beyond the shoulder, negative ⇒ hold deeper) that shifts
//! the cap by a bounded amount, learned from whether pushing produced an in-behind chance vs. an
//! offside turnover.
//!
//! Mirrors [`super::lane_affinity_decision`]. Parity: the onside-support hold clamp is default-on, so
//! this modulation is behind its own gate [`onside_support_model_enabled`] — on in prod (analytic
//! prior ≈ 0 ⇒ still caps at the line), off under `#[cfg(test)]`. Off ⇒ push is 0 (byte-identical).

use super::*;
use crate::des::general::neural_network::{ActivationName, FeedForwardNetwork, RandomNetworkSpec};
use crate::des::general::prng::mulberry32;

pub const ONSIDE_SUPPORT_FEATURE_DIM: usize = 7;
pub const ONSIDE_SUPPORT_HIDDEN_UNITS: usize = 12;
/// Maximum yards the learned push may move the onside cap (positive = beyond the line onto the
/// shoulder, negative = deeper). Kept small so it is a shoulder-timing tune, not a reckless camp.
pub const ONSIDE_SUPPORT_MAX_PUSH_YARDS: f64 = 2.0;

const ONSIDE_SUPPORT_MODEL_ENABLE_ENV: &str = "DD_SOCCER_ENABLE_ONSIDE_SUPPORT_MODEL";
const ONSIDE_SUPPORT_MODEL_DISABLE_ENV: &str = "DD_SOCCER_DISABLE_ONSIDE_SUPPORT_MODEL";

fn env_flag(name: &str) -> Option<bool> {
    std::env::var(name).ok().map(|raw| {
        matches!(raw.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on")
    })
}

fn model_enabled_from_env() -> bool {
    if env_flag(ONSIDE_SUPPORT_MODEL_DISABLE_ENV).unwrap_or(false) {
        return false;
    }
    env_flag(ONSIDE_SUPPORT_MODEL_ENABLE_ENV).unwrap_or(true)
}

pub fn onside_support_model_enabled() -> bool {
    #[cfg(test)]
    {
        if env_flag(ONSIDE_SUPPORT_MODEL_DISABLE_ENV).unwrap_or(false) {
            return false;
        }
        env_flag(ONSIDE_SUPPORT_MODEL_ENABLE_ENV).unwrap_or(false)
    }
    #[cfg(not(test))]
    {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(model_enabled_from_env)
    }
}

pub const ONSIDE_SUPPORT_SAMPLE_INTERVAL_TICKS: u64 = 6;
pub const ONSIDE_SUPPORT_REWARD_WINDOW_TICKS: u64 = 40;
pub const ONSIDE_SUPPORT_SAMPLE_CAP: usize = 8192;
pub const ONSIDE_SUPPORT_HEAD_MIN_TRAINING_STEPS: usize = 200;
pub const ONSIDE_SUPPORT_TERRITORIAL_SHAPING_WEIGHT: f64 = 1.0;

#[derive(Clone, Debug, PartialEq)]
pub struct OnsideSupportInputs {
    /// Line stepping up to spring the trap in `[0, 1]` (`1` ⇒ defenders moving up fast ⇒ hold).
    pub line_stepping_up: f64,
    /// My forward closing speed in `[0, 1]` (pace to make a break worth it).
    pub my_forward_speed: f64,
    /// Space behind the line in `[0, 1]` (room to run into if I break).
    pub space_behind: f64,
    /// Cross / delivery imminence in `[0, 1]` (be on the shoulder for a ball in behind).
    pub cross_imminent: f64,
    /// Attacking numbers around the box in `[0, 1]`.
    pub numbers_up: f64,
    /// `1` if a forward, else `0`.
    pub role_forward: f64,
    /// Distance from me to the line (yd), normalized (`1` ⇒ far from the line).
    pub dist_to_line: f64,
}

impl OnsideSupportInputs {
    pub fn to_features(&self) -> [f64; ONSIDE_SUPPORT_FEATURE_DIM] {
        let unit = |u: f64| u.clamp(0.0, 1.0);
        [
            unit(self.line_stepping_up),
            unit(self.my_forward_speed),
            unit(self.space_behind),
            unit(self.cross_imminent),
            unit(self.numbers_up),
            self.role_forward,
            unit(self.dist_to_line),
        ]
    }
}

/// Analytic seed for **onside push** in `[-1, 1]` (positive = push onto the shoulder). Prior: ≈ 0
/// by default (hold at the line); push when a cross is imminent with pace and space behind; tuck
/// deeper when the line is stepping up to trap.
pub fn analytic_onside_support_push(inputs: &OnsideSupportInputs) -> f64 {
    let mut z = 0.0;
    z += 0.8 * inputs.cross_imminent * inputs.my_forward_speed * inputs.space_behind;
    z -= 0.9 * inputs.line_stepping_up;
    z.tanh()
}

#[derive(Clone, Debug, PartialEq)]
pub struct OnsideSupportSample {
    pub inputs: OnsideSupportInputs,
    pub action_bias: f64,
    pub reward: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PendingOnsideSupportDecision {
    pub team: Team,
    pub player_id: usize,
    pub inputs: OnsideSupportInputs,
    pub action_bias: f64,
    pub decision_territorial: f64,
    pub reward_accum: f64,
    pub due_tick: u64,
}

#[derive(Clone, Debug)]
pub struct OnsideSupportHead {
    network: FeedForwardNetwork,
    training_steps: usize,
    last_loss: Option<f64>,
}

impl OnsideSupportHead {
    pub fn new(seed: u32) -> Self {
        let mut rng = mulberry32(seed ^ 0x1D6B_3F82);
        let network = FeedForwardNetwork::random(
            &RandomNetworkSpec {
                input_dim: ONSIDE_SUPPORT_FEATURE_DIM,
                hidden_layers: vec![ONSIDE_SUPPORT_HIDDEN_UNITS],
                output_dim: 1,
                hidden_activation: ActivationName::Tanh,
                output_activation: ActivationName::Tanh,
                weight_scale: None,
            },
            &mut rng,
        );
        OnsideSupportHead { network, training_steps: 0, last_loss: None }
    }

    pub fn predict(&self, inputs: &OnsideSupportInputs) -> Option<f64> {
        let features = inputs.to_features();
        if features.iter().any(|v| !v.is_finite()) {
            return None;
        }
        self.network.predict(&features[..]).first().copied().filter(|p| p.is_finite())
    }

    pub fn train_reward_weighted(&mut self, samples: &[OnsideSupportSample], learning_rate: f64) -> f64 {
        let finite: Vec<&OnsideSupportSample> = samples
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
pub struct OnsideSupportTrainingReport {
    pub samples: usize,
    pub training_steps: usize,
    pub epochs: usize,
    pub final_loss: f64,
}

pub fn train_onside_support_head(
    samples: &[OnsideSupportSample],
    seed: u32,
    epochs: usize,
    learning_rate: f64,
) -> Option<(OnsideSupportHead, OnsideSupportTrainingReport)> {
    let usable = samples
        .iter()
        .filter(|s| s.reward.is_finite() && s.action_bias.is_finite())
        .count();
    if usable == 0 || epochs == 0 {
        return None;
    }
    let mut head = OnsideSupportHead::new(seed);
    let mut final_loss = 0.0;
    for _ in 0..epochs {
        final_loss = head.train_reward_weighted(samples, learning_rate);
    }
    let report = OnsideSupportTrainingReport {
        samples: usable,
        training_steps: head.training_steps(),
        epochs,
        final_loss,
    };
    Some((head, report))
}

pub fn report_onside_support_training(
    samples: &[OnsideSupportSample],
    seed: u32,
    epochs: usize,
    learning_rate: f64,
) -> Option<OnsideSupportTrainingReport> {
    train_onside_support_head(samples, seed, epochs, learning_rate).map(|(_, report)| report)
}

impl WorldSnapshot {
    /// Build the [`OnsideSupportInputs`] for an idle attacking supporter `player`. Pure / RNG-free.
    pub(crate) fn build_onside_support_inputs(&self, player: &PlayerSnapshot) -> Option<OnsideSupportInputs> {
        if !matches!(player.role, PlayerRole::Forward | PlayerRole::Midfielder) {
            return None;
        }
        let line_y = self.second_last_defender_line_for(player.team)?;
        let pos = self.player_snapshot_position(player);
        let attack_dir = player.team.attack_dir();
        // Line forward velocity: negative (against the attack) ⇒ stepping up to spring the trap.
        let line_fwd = self.slip_break_line_forward_velocity(player.team, line_y);
        let line_stepping_up = ((-line_fwd) / 4.0).clamp(0.0, 1.0);
        let my_forward_speed = (self
            .player_velocity(player.id)
            .map(|v| v.y * attack_dir)
            .unwrap_or(0.0)
            / 6.0)
            .clamp(0.0, 1.0);
        let behind = Vec2 { x: pos.x, y: line_y + attack_dir * 6.0 };
        let space_behind = (self.nearest_opponent_distance_at(player.team, behind) / 12.0).clamp(0.0, 1.0);
        let goal_y = player.team.goal_y(self.field_length);
        let cross_imminent = if crash_box::flank_final_third_crash_box_geometry(
            self.ball.position,
            goal_y,
            self.field_width,
            self.field_length,
        ) {
            1.0
        } else {
            0.0
        };
        let numbers_up = {
            let n = self
                .players
                .iter()
                .filter(|p| {
                    p.team == player.team
                        && p.id != player.id
                        && crash_box::ball_in_attacking_final_third(
                            self.player_snapshot_position(p).y,
                            goal_y,
                            self.field_length,
                        )
                })
                .count();
            (n as f64 / 4.0).clamp(0.0, 1.0)
        };
        let dist_to_line = ((line_y - pos.y).abs() / 15.0).clamp(0.0, 1.0);
        Some(OnsideSupportInputs {
            line_stepping_up,
            my_forward_speed,
            space_behind,
            cross_imminent,
            numbers_up,
            role_forward: (player.role == PlayerRole::Forward) as i32 as f64,
            dist_to_line,
        })
    }

    /// The model's signed onside push in `(-1, 1)`, or `None` when gated off.
    pub(crate) fn onside_support_push_bias(&self, inputs: &OnsideSupportInputs) -> Option<f64> {
        if !onside_support_model_enabled() {
            return None;
        }
        let bias = self
            .onside_support_head
            .as_ref()
            .filter(|head| head.training_steps() >= ONSIDE_SUPPORT_HEAD_MIN_TRAINING_STEPS)
            .and_then(|head| head.predict(inputs))
            .unwrap_or_else(|| analytic_onside_support_push(inputs));
        bias.is_finite().then(|| bias.clamp(-1.0, 1.0))
    }

    /// Signed yards to shift the onside-support cap for `player` (positive = beyond the line onto
    /// the shoulder). `0.0` when the model is off ⇒ byte-identical (cap stays at the line).
    pub(crate) fn onside_support_push_yards(&self, player: &PlayerSnapshot) -> f64 {
        if !onside_support_model_enabled() {
            return 0.0;
        }
        let Some(inputs) = self.build_onside_support_inputs(player) else {
            return 0.0;
        };
        let Some(bias) = self.onside_support_push_bias(&inputs) else {
            return 0.0;
        };
        (bias * ONSIDE_SUPPORT_MAX_PUSH_YARDS).clamp(-ONSIDE_SUPPORT_MAX_PUSH_YARDS, ONSIDE_SUPPORT_MAX_PUSH_YARDS)
    }
}

impl SoccerMatch {
    /// Gated per-tick RL sample collection for the onside-support head. No-op unless enabled.
    pub(crate) fn collect_onside_support_rl_samples(&mut self, snapshot: &WorldSnapshot) {
        if !onside_support_model_enabled() {
            return;
        }
        let tick = self.tick;
        if !self.pending_onside_support.is_empty() && !self.reward_events.is_empty() {
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
            for decision in &mut self.pending_onside_support {
                decision.reward_accum += match decision.team {
                    Team::Home => home_reward,
                    Team::Away => away_reward,
                };
            }
        }
        let mut i = 0;
        while i < self.pending_onside_support.len() {
            if self.pending_onside_support[i].due_tick <= tick {
                let decision = self.pending_onside_support.swap_remove(i);
                let now_territorial = territorial_advantage(snapshot, decision.team);
                let territorial_delta = if now_territorial.is_finite()
                    && decision.decision_territorial.is_finite()
                {
                    now_territorial - decision.decision_territorial
                } else {
                    0.0
                };
                let reward = decision.reward_accum
                    + ONSIDE_SUPPORT_TERRITORIAL_SHAPING_WEIGHT * territorial_delta;
                if reward.is_finite() {
                    self.onside_support_samples.push(OnsideSupportSample {
                        inputs: decision.inputs,
                        action_bias: decision.action_bias,
                        reward,
                    });
                    if self.onside_support_samples.len() > ONSIDE_SUPPORT_SAMPLE_CAP {
                        let overflow =
                            self.onside_support_samples.len() - ONSIDE_SUPPORT_SAMPLE_CAP;
                        self.onside_support_samples.drain(0..overflow);
                    }
                }
            } else {
                i += 1;
            }
        }
        if tick % ONSIDE_SUPPORT_SAMPLE_INTERVAL_TICKS != 0 {
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
            let Some(inputs) = snapshot.build_onside_support_inputs(player) else {
                continue;
            };
            let action_bias = snapshot
                .onside_support_push_bias(&inputs)
                .unwrap_or_else(|| analytic_onside_support_push(&inputs));
            let territorial = territorial_advantage(snapshot, player.team);
            if territorial.is_finite() {
                self.pending_onside_support.push(PendingOnsideSupportDecision {
                    team: player.team,
                    player_id: id,
                    inputs,
                    action_bias,
                    decision_territorial: territorial,
                    reward_accum: 0.0,
                    due_tick: tick + ONSIDE_SUPPORT_REWARD_WINDOW_TICKS,
                });
            }
        }
    }

    pub fn drain_onside_support_samples(&mut self) -> Vec<OnsideSupportSample> {
        std::mem::take(&mut self.onside_support_samples)
    }

    pub fn set_onside_support_head(&mut self, head: OnsideSupportHead) {
        self.onside_support_head = Some(std::sync::Arc::new(head));
    }
}

#[cfg(test)]
mod onside_support_model_tests {
    use super::*;

    fn baseline_inputs() -> OnsideSupportInputs {
        OnsideSupportInputs {
            line_stepping_up: 0.2,
            my_forward_speed: 0.5,
            space_behind: 0.5,
            cross_imminent: 0.5,
            numbers_up: 0.4,
            role_forward: 1.0,
            dist_to_line: 0.3,
        }
    }

    #[test]
    fn features_finite_bounded_right_dim() {
        let f = baseline_inputs().to_features();
        assert_eq!(f.len(), ONSIDE_SUPPORT_FEATURE_DIM);
        for v in f {
            assert!(v.is_finite() && (0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn imminent_cross_pace_space_pushes_the_shoulder() {
        let mut i = baseline_inputs();
        i.cross_imminent = 1.0;
        i.my_forward_speed = 1.0;
        i.space_behind = 1.0;
        i.line_stepping_up = 0.0;
        assert!(analytic_onside_support_push(&i) > 0.0);
    }

    #[test]
    fn line_stepping_up_tucks_deeper() {
        let mut i = baseline_inputs();
        i.line_stepping_up = 1.0;
        i.cross_imminent = 0.0;
        assert!(analytic_onside_support_push(&i) < 0.0);
    }

    #[test]
    fn head_bounded_and_rwr_separates() {
        let inputs = baseline_inputs();
        assert!((-1.0..=1.0).contains(&OnsideSupportHead::new(1).predict(&inputs).unwrap()));
        let mut hi = OnsideSupportHead::new(1);
        let mut lo = OnsideSupportHead::new(1);
        let hi_s: Vec<OnsideSupportSample> = (0..32)
            .flat_map(|_| [
                OnsideSupportSample { inputs: inputs.clone(), action_bias: 0.8, reward: 1.0 },
                OnsideSupportSample { inputs: inputs.clone(), action_bias: -0.8, reward: -1.0 },
            ])
            .collect();
        let lo_s: Vec<OnsideSupportSample> = (0..32)
            .flat_map(|_| [
                OnsideSupportSample { inputs: inputs.clone(), action_bias: -0.8, reward: 1.0 },
                OnsideSupportSample { inputs: inputs.clone(), action_bias: 0.8, reward: -1.0 },
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
        assert!(train_onside_support_head(&[], 1, 4, 0.02).is_none());
    }
}
