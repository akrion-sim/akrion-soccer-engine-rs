//! Learnable **head-scan effort decision** — makes *how hard a settled carrier scans* for
//! options a learned MDP/POMDP choice instead of a fixed sweep rate.
//!
//! [`super::head_scan`] turns time on the ball into a widening scan arc at a **fixed** effective
//! sweep rate (+ a vision bonus). But how much a player invests in shoulder-checks is situational:
//! a carrier under pressure with team-mates behind should scan hard (find the out-ball before the
//! tackle); one in space about to shoot barely scans. This module makes that a learned **scan
//! effort** in `[-1, 1]` (positive ⇒ scan harder, covering a wider arc in the same wall-clock time;
//! negative ⇒ scan less) that scales the effective scan time fed to
//! [`super::head_scan::scan_coverage`] at the visibility seam.
//!
//! Mirrors [`super::lane_affinity_decision`]. Parity: doubly inert — head-scan is gated by
//! [`super::head_scan::head_scan_enabled`] (off by default), and this modulation by
//! [`head_scan_model_enabled`] (on in prod, off under `#[cfg(test)]`). Off ⇒ the effort multiplier
//! is 1.0 (byte-identical).

use super::*;
use crate::des::general::neural_network::{ActivationName, FeedForwardNetwork, RandomNetworkSpec};
use crate::des::general::prng::mulberry32;

pub const HEAD_SCAN_FEATURE_DIM: usize = 7;
pub const HEAD_SCAN_HIDDEN_UNITS: usize = 12;
/// Fractional range the learned effort scales the effective scan time (a bias of `+1` scans this
/// much harder). Clamped to a sane multiplier band at the seam.
pub const HEAD_SCAN_EFFORT_MAX_FRAC: f64 = 0.8;

const HEAD_SCAN_MODEL_ENABLE_ENV: &str = "DD_SOCCER_ENABLE_HEAD_SCAN_MODEL";
const HEAD_SCAN_MODEL_DISABLE_ENV: &str = "DD_SOCCER_DISABLE_HEAD_SCAN_MODEL";

fn env_flag(name: &str) -> Option<bool> {
    std::env::var(name).ok().map(|raw| {
        matches!(raw.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on")
    })
}

fn model_enabled_from_env() -> bool {
    if env_flag(HEAD_SCAN_MODEL_DISABLE_ENV).unwrap_or(false) {
        return false;
    }
    env_flag(HEAD_SCAN_MODEL_ENABLE_ENV).unwrap_or(true)
}

pub fn head_scan_model_enabled() -> bool {
    #[cfg(test)]
    {
        if env_flag(HEAD_SCAN_MODEL_DISABLE_ENV).unwrap_or(false) {
            return false;
        }
        env_flag(HEAD_SCAN_MODEL_ENABLE_ENV).unwrap_or(false)
    }
    #[cfg(not(test))]
    {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(model_enabled_from_env)
    }
}

pub const HEAD_SCAN_SAMPLE_INTERVAL_TICKS: u64 = 4;
pub const HEAD_SCAN_REWARD_WINDOW_TICKS: u64 = 30;
pub const HEAD_SCAN_SAMPLE_CAP: usize = 8192;
pub const HEAD_SCAN_HEAD_MIN_TRAINING_STEPS: usize = 200;
pub const HEAD_SCAN_TERRITORIAL_SHAPING_WEIGHT: f64 = 1.0;

#[derive(Clone, Debug, PartialEq)]
pub struct HeadScanInputs {
    /// Pressure on the carrier in `[0, 1]` (`1` ⇒ an opponent right on top of them).
    pub pressure: f64,
    /// Time on the ball in `[0, 1]` (scan seconds normalized).
    pub time_on_ball: f64,
    /// Team-mates roughly behind the carrier (worth a shoulder-check), normalized.
    pub options_behind: f64,
    /// Space ahead of the carrier in `[0, 1]`.
    pub space_ahead: f64,
    /// `1` if in the attacking final third.
    pub in_final_third: f64,
    /// Vision ability in `[0, 1]`.
    pub vision: f64,
    /// `1` if the carrier is a forward.
    pub role_forward: f64,
}

impl HeadScanInputs {
    pub fn to_features(&self) -> [f64; HEAD_SCAN_FEATURE_DIM] {
        let unit = |u: f64| u.clamp(0.0, 1.0);
        [
            unit(self.pressure),
            unit(self.time_on_ball),
            unit(self.options_behind),
            unit(self.space_ahead),
            unit(self.in_final_third),
            unit(self.vision),
            self.role_forward,
        ]
    }
}

/// Analytic seed for **scan effort** in `[-1, 1]` (positive = scan harder). Prior: ≈ 0 by default;
/// scan harder under pressure with options behind; ease off with acres of space ahead.
pub fn analytic_head_scan_effort(inputs: &HeadScanInputs) -> f64 {
    let mut z = 0.0;
    z += 0.9 * inputs.pressure * inputs.options_behind;
    z -= 0.4 * inputs.space_ahead * (1.0 - inputs.pressure);
    z.tanh()
}

#[derive(Clone, Debug, PartialEq)]
pub struct HeadScanSample {
    pub inputs: HeadScanInputs,
    pub action_bias: f64,
    pub reward: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PendingHeadScanDecision {
    pub team: Team,
    pub player_id: usize,
    pub inputs: HeadScanInputs,
    pub action_bias: f64,
    pub decision_territorial: f64,
    pub reward_accum: f64,
    pub due_tick: u64,
}

#[derive(Clone, Debug)]
pub struct HeadScanHead {
    network: FeedForwardNetwork,
    training_steps: usize,
    last_loss: Option<f64>,
}

impl HeadScanHead {
    pub fn new(seed: u32) -> Self {
        let mut rng = mulberry32(seed ^ 0x4F1C_8B3A);
        let network = FeedForwardNetwork::random(
            &RandomNetworkSpec {
                input_dim: HEAD_SCAN_FEATURE_DIM,
                hidden_layers: vec![HEAD_SCAN_HIDDEN_UNITS],
                output_dim: 1,
                hidden_activation: ActivationName::Tanh,
                output_activation: ActivationName::Tanh,
                weight_scale: None,
            },
            &mut rng,
        );
        HeadScanHead { network, training_steps: 0, last_loss: None }
    }

    pub fn predict(&self, inputs: &HeadScanInputs) -> Option<f64> {
        let features = inputs.to_features();
        if features.iter().any(|v| !v.is_finite()) {
            return None;
        }
        self.network.predict(&features[..]).first().copied().filter(|p| p.is_finite())
    }

    pub fn train_reward_weighted(&mut self, samples: &[HeadScanSample], learning_rate: f64) -> f64 {
        let finite: Vec<&HeadScanSample> = samples
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
pub struct HeadScanTrainingReport {
    pub samples: usize,
    pub training_steps: usize,
    pub epochs: usize,
    pub final_loss: f64,
}

pub fn train_head_scan_head(
    samples: &[HeadScanSample],
    seed: u32,
    epochs: usize,
    learning_rate: f64,
) -> Option<(HeadScanHead, HeadScanTrainingReport)> {
    let usable = samples
        .iter()
        .filter(|s| s.reward.is_finite() && s.action_bias.is_finite())
        .count();
    if usable == 0 || epochs == 0 {
        return None;
    }
    let mut head = HeadScanHead::new(seed);
    let mut final_loss = 0.0;
    for _ in 0..epochs {
        final_loss = head.train_reward_weighted(samples, learning_rate);
    }
    let report = HeadScanTrainingReport {
        samples: usable,
        training_steps: head.training_steps(),
        epochs,
        final_loss,
    };
    Some((head, report))
}

pub fn report_head_scan_training(
    samples: &[HeadScanSample],
    seed: u32,
    epochs: usize,
    learning_rate: f64,
) -> Option<HeadScanTrainingReport> {
    train_head_scan_head(samples, seed, epochs, learning_rate).map(|(_, report)| report)
}

impl WorldSnapshot {
    /// Build the [`HeadScanInputs`] for the ball-carrying `observer_id`. `None` if they are not the
    /// holder. Pure / RNG-free.
    pub(crate) fn build_head_scan_inputs(&self, observer_id: usize) -> Option<HeadScanInputs> {
        if self.ball.holder != Some(observer_id) {
            return None;
        }
        let observer = self.players.iter().find(|p| p.id == observer_id)?;
        let pos = self.player_snapshot_position(observer);
        let attack_dir = observer.team.attack_dir();
        let pressure =
            (1.0 - (self.nearest_opponent_distance_at(observer.team, pos) / 8.0).clamp(0.0, 1.0))
                .clamp(0.0, 1.0);
        let time_on_ball = (self.ball_holder_possession_seconds.max(0.0) / 2.0).clamp(0.0, 1.0);
        let options_behind = {
            let n = self
                .players
                .iter()
                .filter(|p| {
                    p.team == observer.team
                        && p.id != observer_id
                        && (self.player_snapshot_position(p).y - pos.y) * attack_dir < -1.0
                })
                .count();
            (n as f64 / 4.0).clamp(0.0, 1.0)
        };
        let ahead = Vec2 { x: pos.x, y: pos.y + attack_dir * 8.0 };
        let space_ahead = (self.nearest_opponent_distance_at(observer.team, ahead) / 12.0)
            .clamp(0.0, 1.0);
        let goal_y = observer.team.goal_y(self.field_length);
        let in_final_third = crash_box::ball_in_attacking_final_third(pos.y, goal_y, self.field_length);
        Some(HeadScanInputs {
            pressure,
            time_on_ball,
            options_behind,
            space_ahead,
            in_final_third: in_final_third as i32 as f64,
            vision: ability01(observer.skills.vision),
            role_forward: (observer.role == PlayerRole::Forward) as i32 as f64,
        })
    }

    /// The model's signed scan effort in `(-1, 1)`, or `None` when gated off.
    pub(crate) fn head_scan_effort_bias(&self, inputs: &HeadScanInputs) -> Option<f64> {
        if !head_scan_model_enabled() {
            return None;
        }
        let bias = self
            .head_scan_head
            .as_ref()
            .filter(|head| head.training_steps() >= HEAD_SCAN_HEAD_MIN_TRAINING_STEPS)
            .and_then(|head| head.predict(inputs))
            .unwrap_or_else(|| analytic_head_scan_effort(inputs));
        bias.is_finite().then(|| bias.clamp(-1.0, 1.0))
    }

    /// Multiplier (default 1.0) on the effective scan time for `observer_id`. Off / non-holder ⇒
    /// 1.0 (byte-identical). Clamped to a sane band.
    pub(crate) fn head_scan_effort_multiplier(&self, observer_id: usize) -> f64 {
        if !head_scan_model_enabled() {
            return 1.0;
        }
        let Some(inputs) = self.build_head_scan_inputs(observer_id) else {
            return 1.0;
        };
        let Some(bias) = self.head_scan_effort_bias(&inputs) else {
            return 1.0;
        };
        (1.0 + bias * HEAD_SCAN_EFFORT_MAX_FRAC).clamp(0.4, 2.0)
    }
}

impl SoccerMatch {
    /// Gated per-tick RL sample collection for the head-scan effort head. No-op unless enabled.
    pub(crate) fn collect_head_scan_rl_samples(&mut self, snapshot: &WorldSnapshot) {
        if !head_scan_model_enabled() || !head_scan::head_scan_enabled() {
            return;
        }
        let tick = self.tick;
        if !self.pending_head_scan.is_empty() && !self.reward_events.is_empty() {
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
            for decision in &mut self.pending_head_scan {
                decision.reward_accum += match decision.team {
                    Team::Home => home_reward,
                    Team::Away => away_reward,
                };
            }
        }
        let mut i = 0;
        while i < self.pending_head_scan.len() {
            if self.pending_head_scan[i].due_tick <= tick {
                let decision = self.pending_head_scan.swap_remove(i);
                let now_territorial = territorial_advantage(snapshot, decision.team);
                let territorial_delta = if now_territorial.is_finite()
                    && decision.decision_territorial.is_finite()
                {
                    now_territorial - decision.decision_territorial
                } else {
                    0.0
                };
                let reward =
                    decision.reward_accum + HEAD_SCAN_TERRITORIAL_SHAPING_WEIGHT * territorial_delta;
                if reward.is_finite() {
                    self.head_scan_samples.push(HeadScanSample {
                        inputs: decision.inputs,
                        action_bias: decision.action_bias,
                        reward,
                    });
                    if self.head_scan_samples.len() > HEAD_SCAN_SAMPLE_CAP {
                        let overflow = self.head_scan_samples.len() - HEAD_SCAN_SAMPLE_CAP;
                        self.head_scan_samples.drain(0..overflow);
                    }
                }
            } else {
                i += 1;
            }
        }
        if tick % HEAD_SCAN_SAMPLE_INTERVAL_TICKS != 0 {
            return;
        }
        // Only the current ball carrier scans.
        let Some(holder) = snapshot.ball.holder else {
            return;
        };
        let Some(inputs) = snapshot.build_head_scan_inputs(holder) else {
            return;
        };
        let action_bias = snapshot
            .head_scan_effort_bias(&inputs)
            .unwrap_or_else(|| analytic_head_scan_effort(&inputs));
        let Some(player) = snapshot.players.iter().find(|p| p.id == holder) else {
            return;
        };
        let territorial = territorial_advantage(snapshot, player.team);
        if territorial.is_finite() {
            self.pending_head_scan.push(PendingHeadScanDecision {
                team: player.team,
                player_id: holder,
                inputs,
                action_bias,
                decision_territorial: territorial,
                reward_accum: 0.0,
                due_tick: tick + HEAD_SCAN_REWARD_WINDOW_TICKS,
            });
        }
    }

    pub fn drain_head_scan_samples(&mut self) -> Vec<HeadScanSample> {
        std::mem::take(&mut self.head_scan_samples)
    }

    pub fn set_head_scan_head(&mut self, head: HeadScanHead) {
        self.head_scan_head = Some(std::sync::Arc::new(head));
    }
}

#[cfg(test)]
mod head_scan_model_tests {
    use super::*;

    fn baseline_inputs() -> HeadScanInputs {
        HeadScanInputs {
            pressure: 0.4,
            time_on_ball: 0.5,
            options_behind: 0.4,
            space_ahead: 0.5,
            in_final_third: 0.0,
            vision: 0.6,
            role_forward: 0.0,
        }
    }

    #[test]
    fn features_finite_bounded_right_dim() {
        let f = baseline_inputs().to_features();
        assert_eq!(f.len(), HEAD_SCAN_FEATURE_DIM);
        for v in f {
            assert!(v.is_finite() && (0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn pressure_with_options_behind_scans_harder() {
        let mut i = baseline_inputs();
        i.pressure = 1.0;
        i.options_behind = 1.0;
        i.space_ahead = 0.0;
        assert!(analytic_head_scan_effort(&i) > 0.0);
    }

    #[test]
    fn space_ahead_no_pressure_eases_off() {
        let mut i = baseline_inputs();
        i.pressure = 0.0;
        i.space_ahead = 1.0;
        i.options_behind = 0.0;
        assert!(analytic_head_scan_effort(&i) < 0.0);
    }

    #[test]
    fn head_bounded_and_rwr_separates() {
        let inputs = baseline_inputs();
        assert!((-1.0..=1.0).contains(&HeadScanHead::new(1).predict(&inputs).unwrap()));
        let mut hi = HeadScanHead::new(1);
        let mut lo = HeadScanHead::new(1);
        let hi_s: Vec<HeadScanSample> = (0..32)
            .flat_map(|_| [
                HeadScanSample { inputs: inputs.clone(), action_bias: 0.8, reward: 1.0 },
                HeadScanSample { inputs: inputs.clone(), action_bias: -0.8, reward: -1.0 },
            ])
            .collect();
        let lo_s: Vec<HeadScanSample> = (0..32)
            .flat_map(|_| [
                HeadScanSample { inputs: inputs.clone(), action_bias: -0.8, reward: 1.0 },
                HeadScanSample { inputs: inputs.clone(), action_bias: 0.8, reward: -1.0 },
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
        assert!(train_head_scan_head(&[], 1, 4, 0.02).is_none());
    }
}
