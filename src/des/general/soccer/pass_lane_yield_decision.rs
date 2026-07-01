//! Learnable **pass-lane yield decision** — makes the "should the middle man step out of the
//! lane to open a longer pass?" willingness of [`super::pass_lane_yield`] a *learned* choice.
//!
//! [`WorldSnapshot::pass_lane_yield_target_for`] finds when this player is the sole blocker of a
//! carrier→far-mate lane and returns a step-out target with a hand-computed **strength**
//! (`raw_value / 14`). That strength decides how strongly the yield competes with the player's
//! other support options — i.e. whether he actually vacates his position (opening the longer ball
//! but leaving his slot) or holds. This module makes that willingness learnable: a signed **yield
//! bias** in `[-1, 1]` (positive ⇒ more eager to yield, negative ⇒ hold position) that shifts the
//! strength, learned by reward-weighted regression from whether yielding actually led to a
//! rewarded longer pass / forward progress.
//!
//! Mirrors [`super::lane_affinity_decision`]. Parity: doubly inert unless live — the pass-lane
//! yield feature is gated by [`super::pass_lane_yield::pass_lane_yield_enabled`] (off by default),
//! and this modulation is behind [`pass_lane_yield_model_enabled`] (on by default in prod, off
//! under `#[cfg(test)]`). Off ⇒ the hand-computed strength stands (byte-identical).

use super::*;
use crate::des::general::neural_network::{ActivationName, FeedForwardNetwork, RandomNetworkSpec};
use crate::des::general::prng::mulberry32;

pub const PASS_LANE_YIELD_FEATURE_DIM: usize = 7;
pub const PASS_LANE_YIELD_HIDDEN_UNITS: usize = 12;
/// Maximum the learned bias may shift the yield strength (in strength units, `[0, 1]`).
pub const PASS_LANE_YIELD_MAX_STRENGTH_ADJUST: f64 = 0.5;

const PASS_LANE_YIELD_MODEL_ENABLE_ENV: &str = "DD_SOCCER_ENABLE_PASS_LANE_YIELD_MODEL";
const PASS_LANE_YIELD_MODEL_DISABLE_ENV: &str = "DD_SOCCER_DISABLE_PASS_LANE_YIELD_MODEL";

fn env_flag(name: &str) -> Option<bool> {
    std::env::var(name).ok().map(|raw| {
        matches!(raw.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on")
    })
}

fn model_enabled_from_env() -> bool {
    if env_flag(PASS_LANE_YIELD_MODEL_DISABLE_ENV).unwrap_or(false) {
        return false;
    }
    env_flag(PASS_LANE_YIELD_MODEL_ENABLE_ENV).unwrap_or(true)
}

/// Whether the learned pass-lane yield modulation is consulted this process (prod default on,
/// test default off).
pub fn pass_lane_yield_model_enabled() -> bool {
    #[cfg(test)]
    {
        if env_flag(PASS_LANE_YIELD_MODEL_DISABLE_ENV).unwrap_or(false) {
            return false;
        }
        env_flag(PASS_LANE_YIELD_MODEL_ENABLE_ENV).unwrap_or(false)
    }
    #[cfg(not(test))]
    {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(model_enabled_from_env)
    }
}

pub const PASS_LANE_YIELD_SAMPLE_INTERVAL_TICKS: u64 = 5;
pub const PASS_LANE_YIELD_REWARD_WINDOW_TICKS: u64 = 40;
pub const PASS_LANE_YIELD_SAMPLE_CAP: usize = 8192;
pub const PASS_LANE_YIELD_HEAD_MIN_TRAINING_STEPS: usize = 200;
pub const PASS_LANE_YIELD_TERRITORIAL_SHAPING_WEIGHT: f64 = 1.0;

/// Raw per-player pass-lane-yield state.
#[derive(Clone, Debug, PartialEq)]
pub struct PassLaneYieldInputs {
    /// Forward progress (yd) the far man is beyond me, normalized.
    pub forward_gain: f64,
    /// Openness of the far team-mate in `[0, 1]`.
    pub far_openness: f64,
    /// Length (yd) of the longer carrier→far lane, normalized.
    pub lane_len: f64,
    /// My distance from the carrier (yd), normalized.
    pub dist_from_carrier: f64,
    /// Open space at the step-out spot in `[0, 1]`.
    pub spot_space: f64,
    /// `1` if I am a defender (holding a back line ⇒ more costly to vacate), else `0`.
    pub is_defender: f64,
    /// Base (hand-computed) yield strength in `[0, 1]` — the analytic prior's own read.
    pub base_strength: f64,
}

impl PassLaneYieldInputs {
    pub fn to_features(&self) -> [f64; PASS_LANE_YIELD_FEATURE_DIM] {
        let unit = |u: f64| u.clamp(0.0, 1.0);
        [
            unit(self.forward_gain),
            unit(self.far_openness),
            unit(self.lane_len),
            unit(self.dist_from_carrier),
            unit(self.spot_space),
            self.is_defender,
            unit(self.base_strength),
        ]
    }
}

/// Analytic seed for the **yield bias** in `[-1, 1]` (positive = more eager to yield). Prior:
/// ≈ 0 by default (keep the hand-computed strength); lean toward yielding when the far man is
/// open and buys real forward progress; lean toward holding when I am a defender vacating the line.
pub fn analytic_pass_lane_yield_bias(inputs: &PassLaneYieldInputs) -> f64 {
    let mut z = 0.0;
    z += 0.7 * inputs.far_openness * inputs.forward_gain;
    z -= 0.5 * inputs.is_defender * (1.0 - inputs.forward_gain);
    z.tanh()
}

#[derive(Clone, Debug, PartialEq)]
pub struct PassLaneYieldSample {
    pub inputs: PassLaneYieldInputs,
    pub action_bias: f64,
    pub reward: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PendingPassLaneYieldDecision {
    pub team: Team,
    pub player_id: usize,
    pub inputs: PassLaneYieldInputs,
    pub action_bias: f64,
    pub decision_territorial: f64,
    pub reward_accum: f64,
    pub due_tick: u64,
}

#[derive(Clone, Debug)]
pub struct PassLaneYieldHead {
    network: FeedForwardNetwork,
    training_steps: usize,
    last_loss: Option<f64>,
}

impl PassLaneYieldHead {
    pub fn new(seed: u32) -> Self {
        let mut rng = mulberry32(seed ^ 0x2E71_C4B9);
        let network = FeedForwardNetwork::random(
            &RandomNetworkSpec {
                input_dim: PASS_LANE_YIELD_FEATURE_DIM,
                hidden_layers: vec![PASS_LANE_YIELD_HIDDEN_UNITS],
                output_dim: 1,
                hidden_activation: ActivationName::Tanh,
                output_activation: ActivationName::Tanh,
                weight_scale: None,
            },
            &mut rng,
        );
        PassLaneYieldHead { network, training_steps: 0, last_loss: None }
    }

    pub fn predict(&self, inputs: &PassLaneYieldInputs) -> Option<f64> {
        let features = inputs.to_features();
        if features.iter().any(|v| !v.is_finite()) {
            return None;
        }
        self.network.predict(&features[..]).first().copied().filter(|p| p.is_finite())
    }

    pub fn train_reward_weighted(&mut self, samples: &[PassLaneYieldSample], learning_rate: f64) -> f64 {
        let finite: Vec<&PassLaneYieldSample> = samples
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
pub struct PassLaneYieldTrainingReport {
    pub samples: usize,
    pub training_steps: usize,
    pub epochs: usize,
    pub final_loss: f64,
}

pub fn train_pass_lane_yield_head(
    samples: &[PassLaneYieldSample],
    seed: u32,
    epochs: usize,
    learning_rate: f64,
) -> Option<(PassLaneYieldHead, PassLaneYieldTrainingReport)> {
    let usable = samples
        .iter()
        .filter(|s| s.reward.is_finite() && s.action_bias.is_finite())
        .count();
    if usable == 0 || epochs == 0 {
        return None;
    }
    let mut head = PassLaneYieldHead::new(seed);
    let mut final_loss = 0.0;
    for _ in 0..epochs {
        final_loss = head.train_reward_weighted(samples, learning_rate);
    }
    let report = PassLaneYieldTrainingReport {
        samples: usable,
        training_steps: head.training_steps(),
        epochs,
        final_loss,
    };
    Some((head, report))
}

pub fn report_pass_lane_yield_training(
    samples: &[PassLaneYieldSample],
    seed: u32,
    epochs: usize,
    learning_rate: f64,
) -> Option<PassLaneYieldTrainingReport> {
    train_pass_lane_yield_head(samples, seed, epochs, learning_rate).map(|(_, report)| report)
}

impl WorldSnapshot {
    /// The model's signed yield bias in `(-1, 1)`, or `None` when gated off.
    pub(crate) fn pass_lane_yield_bias(&self, inputs: &PassLaneYieldInputs) -> Option<f64> {
        if !pass_lane_yield_model_enabled() {
            return None;
        }
        let bias = self
            .pass_lane_yield_head
            .as_ref()
            .filter(|head| head.training_steps() >= PASS_LANE_YIELD_HEAD_MIN_TRAINING_STEPS)
            .and_then(|head| head.predict(inputs))
            .unwrap_or_else(|| analytic_pass_lane_yield_bias(inputs));
        bias.is_finite().then(|| bias.clamp(-1.0, 1.0))
    }

    /// The effective yield strength given the hand-computed `base_strength` and the decision's
    /// inputs. `base_strength` unchanged when the model is off ⇒ byte-identical.
    pub(crate) fn pass_lane_yield_effective_strength(
        &self,
        inputs: &PassLaneYieldInputs,
        base_strength: f64,
    ) -> f64 {
        let Some(bias) = self.pass_lane_yield_bias(inputs) else {
            return base_strength;
        };
        (base_strength + bias * PASS_LANE_YIELD_MAX_STRENGTH_ADJUST).clamp(0.0, 1.0)
    }
}

impl SoccerMatch {
    /// Gated per-tick RL sample collection for the pass-lane yield head. No-op unless enabled.
    pub(crate) fn collect_pass_lane_yield_rl_samples(&mut self, snapshot: &WorldSnapshot) {
        if !pass_lane_yield_model_enabled() || !pass_lane_yield::pass_lane_yield_enabled() {
            return;
        }
        let tick = self.tick;
        if !self.pending_pass_lane_yield.is_empty() && !self.reward_events.is_empty() {
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
            for decision in &mut self.pending_pass_lane_yield {
                decision.reward_accum += match decision.team {
                    Team::Home => home_reward,
                    Team::Away => away_reward,
                };
            }
        }
        let mut i = 0;
        while i < self.pending_pass_lane_yield.len() {
            if self.pending_pass_lane_yield[i].due_tick <= tick {
                let decision = self.pending_pass_lane_yield.swap_remove(i);
                let now_territorial = territorial_advantage(snapshot, decision.team);
                let territorial_delta = if now_territorial.is_finite()
                    && decision.decision_territorial.is_finite()
                {
                    now_territorial - decision.decision_territorial
                } else {
                    0.0
                };
                let reward = decision.reward_accum
                    + PASS_LANE_YIELD_TERRITORIAL_SHAPING_WEIGHT * territorial_delta;
                if reward.is_finite() {
                    self.pass_lane_yield_samples.push(PassLaneYieldSample {
                        inputs: decision.inputs,
                        action_bias: decision.action_bias,
                        reward,
                    });
                    if self.pass_lane_yield_samples.len() > PASS_LANE_YIELD_SAMPLE_CAP {
                        let overflow =
                            self.pass_lane_yield_samples.len() - PASS_LANE_YIELD_SAMPLE_CAP;
                        self.pass_lane_yield_samples.drain(0..overflow);
                    }
                }
            } else {
                i += 1;
            }
        }
        if tick % PASS_LANE_YIELD_SAMPLE_INTERVAL_TICKS != 0 {
            return;
        }
        let ids: Vec<usize> = snapshot
            .players
            .iter()
            .filter(|p| p.role != PlayerRole::Goalkeeper)
            .map(|p| p.id)
            .collect();
        for id in ids {
            let Some((inputs, _spot, _strength)) = snapshot.pass_lane_yield_context(id) else {
                continue;
            };
            let action_bias = snapshot
                .pass_lane_yield_bias(&inputs)
                .unwrap_or_else(|| analytic_pass_lane_yield_bias(&inputs));
            let Some(player) = snapshot.players.iter().find(|p| p.id == id) else {
                continue;
            };
            let territorial = territorial_advantage(snapshot, player.team);
            if territorial.is_finite() {
                self.pending_pass_lane_yield.push(PendingPassLaneYieldDecision {
                    team: player.team,
                    player_id: id,
                    inputs,
                    action_bias,
                    decision_territorial: territorial,
                    reward_accum: 0.0,
                    due_tick: tick + PASS_LANE_YIELD_REWARD_WINDOW_TICKS,
                });
            }
        }
    }

    pub fn drain_pass_lane_yield_samples(&mut self) -> Vec<PassLaneYieldSample> {
        std::mem::take(&mut self.pass_lane_yield_samples)
    }

    pub fn set_pass_lane_yield_head(&mut self, head: PassLaneYieldHead) {
        self.pass_lane_yield_head = Some(std::sync::Arc::new(head));
    }
}

#[cfg(test)]
mod pass_lane_yield_model_tests {
    use super::*;

    fn baseline_inputs() -> PassLaneYieldInputs {
        PassLaneYieldInputs {
            forward_gain: 0.5,
            far_openness: 0.6,
            lane_len: 0.5,
            dist_from_carrier: 0.4,
            spot_space: 0.6,
            is_defender: 0.0,
            base_strength: 0.5,
        }
    }

    #[test]
    fn features_are_finite_bounded_and_right_dim() {
        let f = baseline_inputs().to_features();
        assert_eq!(f.len(), PASS_LANE_YIELD_FEATURE_DIM);
        for v in f {
            assert!(v.is_finite() && (0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn analytic_bias_valid_and_open_forward_yields() {
        let b = analytic_pass_lane_yield_bias(&baseline_inputs());
        assert!((-1.0..=1.0).contains(&b));
        let mut open = baseline_inputs();
        open.far_openness = 1.0;
        open.forward_gain = 1.0;
        assert!(analytic_pass_lane_yield_bias(&open) > 0.0);
    }

    #[test]
    fn defender_with_no_forward_gain_holds() {
        let mut d = baseline_inputs();
        d.is_defender = 1.0;
        d.forward_gain = 0.0;
        d.far_openness = 0.2;
        assert!(analytic_pass_lane_yield_bias(&d) < 0.0);
    }

    #[test]
    fn head_predicts_bounded_and_rwr_separates_directions() {
        let inputs = baseline_inputs();
        let head = PassLaneYieldHead::new(4);
        assert!((-1.0..=1.0).contains(&head.predict(&inputs).expect("finite")));
        let mut yes = PassLaneYieldHead::new(4);
        let mut no = PassLaneYieldHead::new(4);
        let yes_s: Vec<PassLaneYieldSample> = (0..32)
            .flat_map(|_| [
                PassLaneYieldSample { inputs: inputs.clone(), action_bias: 0.8, reward: 1.0 },
                PassLaneYieldSample { inputs: inputs.clone(), action_bias: -0.8, reward: -1.0 },
            ])
            .collect();
        let no_s: Vec<PassLaneYieldSample> = (0..32)
            .flat_map(|_| [
                PassLaneYieldSample { inputs: inputs.clone(), action_bias: -0.8, reward: 1.0 },
                PassLaneYieldSample { inputs: inputs.clone(), action_bias: 0.8, reward: -1.0 },
            ])
            .collect();
        for _ in 0..80 {
            yes.train_reward_weighted(&yes_s, 0.05);
            no.train_reward_weighted(&no_s, 0.05);
        }
        assert!(yes.predict(&inputs).unwrap() > no.predict(&inputs).unwrap());
    }

    #[test]
    fn train_entry_skips_empty() {
        assert!(train_pass_lane_yield_head(&[], 1, 4, 0.02).is_none());
    }
}
