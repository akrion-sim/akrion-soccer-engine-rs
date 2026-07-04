//! Learnable **slip-break commit decision** — makes *how eagerly a runner commits to breaking the
//! offside trap* (and the carrier to threading the slip) a learned MDP/POMDP choice.
//!
//! [`WorldSnapshot::slip_break_runner_opportunity`] scores a slip-break opportunity from the seam,
//! release timing, lane openness and the runner's speed advantage into a single **opportunity**
//! quality in `[0, 1]` that ranks/reward-shapes the cooperative move. That quality is a fixed blend.
//! This module makes the *appetite* to commit learnable: a signed **slip appetite** in `[-1, 1]`
//! (positive ⇒ back the break more readily, negative ⇒ hold, don't gamble beyond the line) that
//! scales the opportunity, learned by reward-weighted regression from whether committed slip-breaks
//! actually produced a line-breaking chance vs. ran into an offside flag / lost the ball.
//!
//! Mirrors [`super::lane_affinity_decision`]. Parity: doubly inert — the slip-break feature is gated
//! by [`super::dd_soccer_enable_slip_break_offside`] (off by default), and this modulation by
//! [`slip_break_model_enabled`] (on in prod, off under `#[cfg(test)]`). Off ⇒ the opportunity is
//! unchanged (byte-identical).

use super::*;
use crate::des::general::neural_network::{ActivationName, FeedForwardNetwork, RandomNetworkSpec};
use crate::des::general::prng::mulberry32;

pub const SLIP_BREAK_FEATURE_DIM: usize = 6;
pub const SLIP_BREAK_HIDDEN_UNITS: usize = 12;
/// Fractional range the learned appetite scales the slip-break opportunity quality.
pub const SLIP_BREAK_MAX_OPPORTUNITY_FRAC: f64 = 0.6;

const SLIP_BREAK_MODEL_ENABLE_ENV: &str = "DD_SOCCER_ENABLE_SLIP_BREAK_MODEL";
const SLIP_BREAK_MODEL_DISABLE_ENV: &str = "DD_SOCCER_DISABLE_SLIP_BREAK_MODEL";

fn env_flag(name: &str) -> Option<bool> {
    std::env::var(name).ok().map(|raw| {
        matches!(raw.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on")
    })
}

fn model_enabled_from_env() -> bool {
    if env_flag(SLIP_BREAK_MODEL_DISABLE_ENV).unwrap_or(false) {
        return false;
    }
    env_flag(SLIP_BREAK_MODEL_ENABLE_ENV).unwrap_or(true)
}

pub fn slip_break_model_enabled() -> bool {
    #[cfg(test)]
    {
        if env_flag(SLIP_BREAK_MODEL_DISABLE_ENV).unwrap_or(false) {
            return false;
        }
        env_flag(SLIP_BREAK_MODEL_ENABLE_ENV).unwrap_or(false)
    }
    #[cfg(not(test))]
    {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(model_enabled_from_env)
    }
}

pub const SLIP_BREAK_SAMPLE_INTERVAL_TICKS: u64 = 5;
pub const SLIP_BREAK_REWARD_WINDOW_TICKS: u64 = 40;
pub const SLIP_BREAK_SAMPLE_CAP: usize = 8192;
pub const SLIP_BREAK_HEAD_MIN_TRAINING_STEPS: usize = 200;
pub const SLIP_BREAK_TERRITORIAL_SHAPING_WEIGHT: f64 = 1.0;

#[derive(Clone, Debug, PartialEq)]
pub struct SlipBreakInputs {
    /// Defender seam quality in `[0, 1]`.
    pub seam_quality: f64,
    /// Release timing fit in `[0, 1]`.
    pub timing: f64,
    /// Carrier→seam lane openness in `[0, 1]`.
    pub lane_openness: f64,
    /// Runner speed advantage over the line in `[0, 1]`.
    pub speed_advantage: f64,
    /// Yards the runner is onside of the line, normalized in `[0, 1]` (`0` ⇒ right on the line).
    pub onside_margin: f64,
    /// `1` if the runner is a forward, else `0`.
    pub role_forward: f64,
}

impl SlipBreakInputs {
    pub fn to_features(&self) -> [f64; SLIP_BREAK_FEATURE_DIM] {
        let unit = |u: f64| u.clamp(0.0, 1.0);
        [
            unit(self.seam_quality),
            unit(self.timing),
            unit(self.lane_openness),
            unit(self.speed_advantage),
            unit(self.onside_margin),
            self.role_forward,
        ]
    }
}

/// Analytic seed for **slip appetite** in `[-1, 1]` (positive = back the break). Prior: ≈ 0 by
/// default; back the break when the seam, timing and pace all align; hold when the lane is blocked
/// or the runner is losing the pace race (a gamble that ends offside / dispossessed).
pub fn analytic_slip_break_appetite(inputs: &SlipBreakInputs) -> f64 {
    let mut z = 0.0;
    z += 0.7 * inputs.seam_quality * inputs.timing * inputs.speed_advantage;
    z -= 0.6 * (1.0 - inputs.lane_openness);
    z -= 0.4 * (1.0 - inputs.speed_advantage);
    z.tanh()
}

#[derive(Clone, Debug, PartialEq)]
pub struct SlipBreakSample {
    pub inputs: SlipBreakInputs,
    pub action_bias: f64,
    pub reward: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PendingSlipBreakDecision {
    pub team: Team,
    pub player_id: usize,
    pub inputs: SlipBreakInputs,
    pub action_bias: f64,
    pub decision_territorial: f64,
    pub reward_accum: f64,
    pub due_tick: u64,
}

#[derive(Clone, Debug)]
pub struct SlipBreakHead {
    network: FeedForwardNetwork,
    training_steps: usize,
    last_loss: Option<f64>,
}

impl SlipBreakHead {
    pub fn new(seed: u32) -> Self {
        let mut rng = mulberry32(seed ^ 0x7C4E_2A91);
        let network = FeedForwardNetwork::random(
            &RandomNetworkSpec {
                input_dim: SLIP_BREAK_FEATURE_DIM,
                hidden_layers: vec![SLIP_BREAK_HIDDEN_UNITS],
                output_dim: 1,
                hidden_activation: ActivationName::Tanh,
                output_activation: ActivationName::Tanh,
                weight_scale: None,
            },
            &mut rng,
        );
        SlipBreakHead { network, training_steps: 0, last_loss: None }
    }

    pub fn predict(&self, inputs: &SlipBreakInputs) -> Option<f64> {
        let features = inputs.to_features();
        if features.iter().any(|v| !v.is_finite()) {
            return None;
        }
        self.network.predict(&features[..]).first().copied().filter(|p| p.is_finite())
    }

    pub fn train_reward_weighted(&mut self, samples: &[SlipBreakSample], learning_rate: f64) -> f64 {
        let finite: Vec<&SlipBreakSample> = samples
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
pub struct SlipBreakTrainingReport {
    pub samples: usize,
    pub training_steps: usize,
    pub epochs: usize,
    pub final_loss: f64,
}

pub fn train_slip_break_head(
    samples: &[SlipBreakSample],
    seed: u32,
    epochs: usize,
    learning_rate: f64,
) -> Option<(SlipBreakHead, SlipBreakTrainingReport)> {
    let usable = samples
        .iter()
        .filter(|s| s.reward.is_finite() && s.action_bias.is_finite())
        .count();
    if usable == 0 || epochs == 0 {
        return None;
    }
    let mut head = SlipBreakHead::new(seed);
    let mut final_loss = 0.0;
    for _ in 0..epochs {
        final_loss = head.train_reward_weighted(samples, learning_rate);
    }
    let report = SlipBreakTrainingReport {
        samples: usable,
        training_steps: head.training_steps(),
        epochs,
        final_loss,
    };
    Some((head, report))
}

pub fn report_slip_break_training(
    samples: &[SlipBreakSample],
    seed: u32,
    epochs: usize,
    learning_rate: f64,
) -> Option<SlipBreakTrainingReport> {
    train_slip_break_head(samples, seed, epochs, learning_rate).map(|(_, report)| report)
}

impl WorldSnapshot {
    /// The model's signed slip appetite in `(-1, 1)`, or `None` when gated off.
    pub(crate) fn slip_break_appetite(&self, inputs: &SlipBreakInputs) -> Option<f64> {
        if !slip_break_model_enabled() {
            return None;
        }
        let bias = self
            .slip_break_head
            .as_ref()
            .filter(|head| head.training_steps() >= SLIP_BREAK_HEAD_MIN_TRAINING_STEPS)
            .and_then(|head| head.predict(inputs))
            .unwrap_or_else(|| analytic_slip_break_appetite(inputs));
        bias.is_finite().then(|| bias.clamp(-1.0, 1.0))
    }

    /// Modulate a base slip-break `opportunity` by the learned appetite for the given inputs.
    /// `opportunity` unchanged when the model is off ⇒ byte-identical.
    pub(crate) fn slip_break_effective_opportunity(
        &self,
        inputs: &SlipBreakInputs,
        opportunity: f64,
    ) -> f64 {
        let Some(bias) = self.slip_break_appetite(inputs) else {
            return opportunity;
        };
        (opportunity * (1.0 + bias * SLIP_BREAK_MAX_OPPORTUNITY_FRAC)).clamp(0.0, 1.0)
    }
}

impl SoccerMatch {
    /// Gated per-tick RL sample collection for the slip-break commit head. No-op unless enabled.
    pub(crate) fn collect_slip_break_rl_samples(&mut self, snapshot: &WorldSnapshot) {
        if !slip_break_model_enabled() || !dd_soccer_enable_slip_break_offside() {
            return;
        }
        let tick = self.tick;
        if !self.pending_slip_break.is_empty() && !self.reward_events.is_empty() {
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
            for decision in &mut self.pending_slip_break {
                decision.reward_accum += match decision.team {
                    Team::Home => home_reward,
                    Team::Away => away_reward,
                };
            }
        }
        let mut i = 0;
        while i < self.pending_slip_break.len() {
            if self.pending_slip_break[i].due_tick <= tick {
                let decision = self.pending_slip_break.swap_remove(i);
                let now_territorial = territorial_advantage(snapshot, decision.team);
                let territorial_delta = if now_territorial.is_finite()
                    && decision.decision_territorial.is_finite()
                {
                    now_territorial - decision.decision_territorial
                } else {
                    0.0
                };
                let reward =
                    decision.reward_accum + SLIP_BREAK_TERRITORIAL_SHAPING_WEIGHT * territorial_delta;
                if reward.is_finite() {
                    self.slip_break_samples.push(SlipBreakSample {
                        inputs: decision.inputs,
                        action_bias: decision.action_bias,
                        reward,
                    });
                    if self.slip_break_samples.len() > SLIP_BREAK_SAMPLE_CAP {
                        let overflow = self.slip_break_samples.len() - SLIP_BREAK_SAMPLE_CAP;
                        self.slip_break_samples.drain(0..overflow);
                    }
                }
            } else {
                i += 1;
            }
        }
        if tick % SLIP_BREAK_SAMPLE_INTERVAL_TICKS != 0 {
            return;
        }
        // Sample runners in the staging band of an attacking team that holds the ball.
        let Some(holder) = snapshot.ball.holder else {
            return;
        };
        let Some(passer) = snapshot.players.iter().find(|p| p.id == holder) else {
            return;
        };
        let attacking_team = passer.team;
        if snapshot.possession_team() != Some(attacking_team) {
            return;
        }
        let passer_pos = snapshot.player_snapshot_position(passer);
        let runner_ids: Vec<usize> = snapshot
            .players
            .iter()
            .filter(|p| p.team == attacking_team && p.id != holder)
            .filter(|p| matches!(p.role, PlayerRole::Forward | PlayerRole::Midfielder))
            .map(|p| p.id)
            .collect();
        for runner_id in runner_ids {
            let Some((inputs, _opp)) =
                snapshot.slip_break_decision_inputs(attacking_team, holder, passer_pos, runner_id)
            else {
                continue;
            };
            let action_bias = snapshot
                .slip_break_appetite(&inputs)
                .unwrap_or_else(|| analytic_slip_break_appetite(&inputs));
            let territorial = territorial_advantage(snapshot, attacking_team);
            if territorial.is_finite() {
                self.pending_slip_break.push(PendingSlipBreakDecision {
                    team: attacking_team,
                    player_id: runner_id,
                    inputs,
                    action_bias,
                    decision_territorial: territorial,
                    reward_accum: 0.0,
                    due_tick: tick + SLIP_BREAK_REWARD_WINDOW_TICKS,
                });
            }
        }
    }

    pub fn drain_slip_break_samples(&mut self) -> Vec<SlipBreakSample> {
        std::mem::take(&mut self.slip_break_samples)
    }

    pub fn set_slip_break_head(&mut self, head: SlipBreakHead) {
        self.slip_break_head = Some(std::sync::Arc::new(head));
    }
}

#[cfg(test)]
mod slip_break_model_tests {
    use super::*;

    fn baseline_inputs() -> SlipBreakInputs {
        SlipBreakInputs {
            seam_quality: 0.6,
            timing: 0.6,
            lane_openness: 0.8,
            speed_advantage: 0.6,
            onside_margin: 0.3,
            role_forward: 1.0,
        }
    }

    #[test]
    fn features_finite_bounded_right_dim() {
        let f = baseline_inputs().to_features();
        assert_eq!(f.len(), SLIP_BREAK_FEATURE_DIM);
        for v in f {
            assert!(v.is_finite() && (0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn aligned_seam_timing_pace_backs_the_break() {
        let mut i = baseline_inputs();
        i.seam_quality = 1.0;
        i.timing = 1.0;
        i.speed_advantage = 1.0;
        i.lane_openness = 1.0;
        assert!(analytic_slip_break_appetite(&i) > 0.0);
    }

    #[test]
    fn blocked_lane_losing_pace_holds() {
        let mut i = baseline_inputs();
        i.lane_openness = 0.0;
        i.speed_advantage = 0.0;
        assert!(analytic_slip_break_appetite(&i) < 0.0);
    }

    #[test]
    fn head_bounded_and_rwr_separates() {
        let inputs = baseline_inputs();
        assert!((-1.0..=1.0).contains(&SlipBreakHead::new(1).predict(&inputs).unwrap()));
        let mut hi = SlipBreakHead::new(1);
        let mut lo = SlipBreakHead::new(1);
        let hi_s: Vec<SlipBreakSample> = (0..32)
            .flat_map(|_| [
                SlipBreakSample { inputs: inputs.clone(), action_bias: 0.8, reward: 1.0 },
                SlipBreakSample { inputs: inputs.clone(), action_bias: -0.8, reward: -1.0 },
            ])
            .collect();
        let lo_s: Vec<SlipBreakSample> = (0..32)
            .flat_map(|_| [
                SlipBreakSample { inputs: inputs.clone(), action_bias: -0.8, reward: 1.0 },
                SlipBreakSample { inputs: inputs.clone(), action_bias: 0.8, reward: -1.0 },
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
        assert!(train_slip_break_head(&[], 1, 4, 0.02).is_none());
    }
}
