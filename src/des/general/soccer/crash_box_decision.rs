//! Learnable **crash-the-box commit decision** — makes *whether an attacker gambles into the box*
//! to attack a cross a learned MDP/POMDP choice instead of a purely geometric trigger.
//!
//! [`WorldSnapshot::crash_the_box_target_for`] commits every eligible forward / attacking mid into
//! a box-flood zone whenever the flank-cross geometry is on. That is a good *predilection* but
//! crashing the box is a gamble — the attacker abandons a supporting / recycle position to attack a
//! delivery that may never come, and a box already full of team-mates does not need another body
//! (better to hold the top of the box for the cut-back / second ball). This module makes that a
//! learned **crash appetite** in `[-1, 1]` (positive ⇒ commit into the box, negative ⇒ hold): when
//! the appetite is clearly negative the box-flood target is suppressed and the attacker keeps its
//! normal shape target.
//!
//! Mirrors [`super::lane_affinity_decision`]. Parity: doubly inert — crash-the-box is gated by
//! `dd_soccer_disable_crash_the_box` / strategy, and this modulation by
//! [`crash_box_model_enabled`] (on in prod, off under `#[cfg(test)]`). Off ⇒ no suppression
//! (byte-identical).

use super::*;
use crate::des::general::neural_network::{ActivationName, FeedForwardNetwork, RandomNetworkSpec};
use crate::des::general::prng::mulberry32;

pub const CRASH_BOX_FEATURE_DIM: usize = 7;
pub const CRASH_BOX_HIDDEN_UNITS: usize = 12;
/// Appetite at/below which the box-flood target is suppressed (the attacker holds instead of
/// crashing). Negative so the default (analytic prior ≈ positive) keeps committing — only a
/// clearly-bad gamble is held back.
pub const CRASH_BOX_SUPPRESS_THRESHOLD: f64 = -0.5;

const CRASH_BOX_MODEL_ENABLE_ENV: &str = "DD_SOCCER_ENABLE_CRASH_BOX_MODEL";
const CRASH_BOX_MODEL_DISABLE_ENV: &str = "DD_SOCCER_DISABLE_CRASH_BOX_MODEL";

fn env_flag(name: &str) -> Option<bool> {
    std::env::var(name).ok().map(|raw| {
        matches!(
            raw.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn model_enabled_from_env() -> bool {
    if env_flag(CRASH_BOX_MODEL_DISABLE_ENV).unwrap_or(false) {
        return false;
    }
    env_flag(CRASH_BOX_MODEL_ENABLE_ENV).unwrap_or(true)
}

pub fn crash_box_model_enabled() -> bool {
    #[cfg(test)]
    {
        if env_flag(CRASH_BOX_MODEL_DISABLE_ENV).unwrap_or(false) {
            return false;
        }
        env_flag(CRASH_BOX_MODEL_ENABLE_ENV).unwrap_or(false)
    }
    #[cfg(not(test))]
    {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(model_enabled_from_env)
    }
}

pub const CRASH_BOX_SAMPLE_INTERVAL_TICKS: u64 = 5;
pub const CRASH_BOX_REWARD_WINDOW_TICKS: u64 = 45;
pub const CRASH_BOX_SAMPLE_CAP: usize = 8192;
pub const CRASH_BOX_HEAD_MIN_TRAINING_STEPS: usize = 200;
pub const CRASH_BOX_TERRITORIAL_SHAPING_WEIGHT: f64 = 1.0;

#[derive(Clone, Debug, PartialEq)]
pub struct CrashBoxInputs {
    /// Own attackers already in / crashing the box, normalized (`1` ⇒ box already full).
    pub box_congestion: f64,
    /// Defenders in the box, normalized (a packed box is harder to attack).
    pub defenders_in_box: f64,
    /// My distance to the box (yd), normalized (`1` ⇒ far, a long gamble).
    pub dist_to_box: f64,
    /// `1` if I am a forward (natural box-crasher), else `0` (mid).
    pub role_forward: f64,
    /// Cross imminence in `[0, 1]` (ball wide & high, delivery about to come).
    pub cross_imminent: f64,
    /// Open space at my box-arrival zone in `[0, 1]`.
    pub arrival_space: f64,
    /// Whether I am needed as a recycle / hold outlet behind in `[0, 1]` (few team-mates behind ⇒
    /// crashing leaves the team exposed / with no cut-back option).
    pub needed_behind: f64,
}

impl CrashBoxInputs {
    pub fn to_features(&self) -> [f64; CRASH_BOX_FEATURE_DIM] {
        let unit = |u: f64| u.clamp(0.0, 1.0);
        [
            unit(self.box_congestion),
            unit(self.defenders_in_box),
            unit(self.dist_to_box),
            self.role_forward,
            unit(self.cross_imminent),
            unit(self.arrival_space),
            unit(self.needed_behind),
        ]
    }
}

/// Analytic seed for **crash appetite** in `[-1, 1]` (positive = commit into the box). Prior:
/// generally commit (small positive baseline) when a cross is imminent; hold back when the box is
/// already full of team-mates or I am the only recycle outlet behind.
pub fn analytic_crash_box_appetite(inputs: &CrashBoxInputs) -> f64 {
    let mut z = 0.35 + 0.6 * inputs.cross_imminent * (0.4 + 0.6 * inputs.arrival_space);
    z -= 1.1 * inputs.box_congestion;
    z -= 0.8 * inputs.needed_behind;
    z.tanh()
}

#[derive(Clone, Debug, PartialEq)]
pub struct CrashBoxSample {
    pub inputs: CrashBoxInputs,
    pub action_bias: f64,
    pub reward: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PendingCrashBoxDecision {
    pub team: Team,
    pub player_id: usize,
    pub inputs: CrashBoxInputs,
    pub action_bias: f64,
    pub decision_territorial: f64,
    pub reward_accum: f64,
    pub due_tick: u64,
}

#[derive(Clone, Debug)]
pub struct CrashBoxHead {
    network: FeedForwardNetwork,
    training_steps: usize,
    last_loss: Option<f64>,
}

impl CrashBoxHead {
    pub fn new(seed: u32) -> Self {
        let mut rng = mulberry32(seed ^ 0x5B2D_7E13);
        let network = FeedForwardNetwork::random(
            &RandomNetworkSpec {
                input_dim: CRASH_BOX_FEATURE_DIM,
                hidden_layers: vec![CRASH_BOX_HIDDEN_UNITS],
                output_dim: 1,
                hidden_activation: ActivationName::Tanh,
                output_activation: ActivationName::Tanh,
                weight_scale: None,
            },
            &mut rng,
        );
        CrashBoxHead {
            network,
            training_steps: 0,
            last_loss: None,
        }
    }

    pub fn predict(&self, inputs: &CrashBoxInputs) -> Option<f64> {
        let features = inputs.to_features();
        if features.iter().any(|v| !v.is_finite()) {
            return None;
        }
        self.network
            .predict(&features[..])
            .first()
            .copied()
            .filter(|p| p.is_finite())
    }

    pub fn train_reward_weighted(&mut self, samples: &[CrashBoxSample], learning_rate: f64) -> f64 {
        let finite: Vec<&CrashBoxSample> = samples
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

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CrashBoxTrainingReport {
    pub samples: usize,
    pub training_steps: usize,
    pub epochs: usize,
    pub final_loss: f64,
}

pub fn train_crash_box_head(
    samples: &[CrashBoxSample],
    seed: u32,
    epochs: usize,
    learning_rate: f64,
) -> Option<(CrashBoxHead, CrashBoxTrainingReport)> {
    let usable = samples
        .iter()
        .filter(|s| s.reward.is_finite() && s.action_bias.is_finite())
        .count();
    if usable == 0 || epochs == 0 {
        return None;
    }
    let mut head = CrashBoxHead::new(seed);
    let mut final_loss = 0.0;
    for _ in 0..epochs {
        final_loss = head.train_reward_weighted(samples, learning_rate);
    }
    let report = CrashBoxTrainingReport {
        samples: usable,
        training_steps: head.training_steps(),
        epochs,
        final_loss,
    };
    Some((head, report))
}

pub fn report_crash_box_training(
    samples: &[CrashBoxSample],
    seed: u32,
    epochs: usize,
    learning_rate: f64,
) -> Option<CrashBoxTrainingReport> {
    train_crash_box_head(samples, seed, epochs, learning_rate).map(|(_, report)| report)
}

impl WorldSnapshot {
    /// Build the [`CrashBoxInputs`] for a would-be box-crasher `player`. Pure / RNG-free.
    pub(crate) fn build_crash_box_inputs(&self, player: &PlayerSnapshot) -> Option<CrashBoxInputs> {
        if !matches!(player.role, PlayerRole::Forward | PlayerRole::Midfielder) {
            return None;
        }
        let pos = self.player_snapshot_position(player);
        let opp = player.team.other();
        let goal_y = player.team.goal_y(self.field_length);
        let attack_dir = player.team.attack_dir();
        let box_congestion = {
            let n = self
                .players
                .iter()
                .filter(|p| {
                    p.team == player.team
                        && p.id != player.id
                        && matches!(p.role, PlayerRole::Forward | PlayerRole::Midfielder)
                        && self.point_in_own_penalty_area(opp, self.player_snapshot_position(p))
                })
                .count();
            (n as f64 / 3.0).clamp(0.0, 1.0)
        };
        let defenders_in_box = {
            let n = self
                .players
                .iter()
                .filter(|p| {
                    p.team == opp
                        && self.point_in_own_penalty_area(opp, self.player_snapshot_position(p))
                })
                .count();
            (n as f64 / 5.0).clamp(0.0, 1.0)
        };
        let dist_to_box =
            ((goal_y - pos.y).abs() / (self.field_length * 0.3).max(1.0)).clamp(0.0, 1.0);
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
        let arrival_zone = Vec2 {
            x: pos.x,
            y: goal_y - attack_dir * 8.0,
        };
        let arrival_space =
            (self.nearest_opponent_distance_at(player.team, arrival_zone) / 10.0).clamp(0.0, 1.0);
        let needed_behind = {
            let behind = self
                .players
                .iter()
                .filter(|p| {
                    p.team == player.team
                        && p.id != player.id
                        && (self.player_snapshot_position(p).y - pos.y) * attack_dir < -2.0
                })
                .count();
            (1.0 - (behind as f64 / 3.0).clamp(0.0, 1.0)).clamp(0.0, 1.0)
        };
        Some(CrashBoxInputs {
            box_congestion,
            defenders_in_box,
            dist_to_box,
            role_forward: (player.role == PlayerRole::Forward) as i32 as f64,
            cross_imminent,
            arrival_space,
            needed_behind,
        })
    }

    /// The model's signed crash appetite in `(-1, 1)`, or `None` when gated off.
    pub(crate) fn crash_box_appetite(&self, inputs: &CrashBoxInputs) -> Option<f64> {
        if !crash_box_model_enabled() {
            return None;
        }
        let bias = self
            .crash_box_head
            .as_ref()
            .filter(|head| head.training_steps() >= CRASH_BOX_HEAD_MIN_TRAINING_STEPS)
            .and_then(|head| head.predict(inputs))
            .unwrap_or_else(|| analytic_crash_box_appetite(inputs));
        bias.is_finite().then(|| bias.clamp(-1.0, 1.0))
    }

    /// Whether the box-flood target should be suppressed for `player` (hold instead of crashing).
    /// Always `false` when the model is off ⇒ byte-identical (the crash always fires as before).
    pub(crate) fn crash_box_should_hold(&self, player: &PlayerSnapshot) -> bool {
        if !crash_box_model_enabled() {
            return false;
        }
        let Some(inputs) = self.build_crash_box_inputs(player) else {
            return false;
        };
        matches!(self.crash_box_appetite(&inputs), Some(a) if a <= CRASH_BOX_SUPPRESS_THRESHOLD)
    }
}

impl SoccerMatch {
    /// Gated per-tick RL sample collection for the crash-box commit head. No-op unless enabled.
    pub(crate) fn collect_crash_box_rl_samples(&mut self, snapshot: &WorldSnapshot) {
        if !crash_box_model_enabled() || !crash_box::flank_crash_box_enabled() {
            return;
        }
        let tick = self.tick;
        if !self.pending_crash_box.is_empty() && !self.reward_events.is_empty() {
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
            for decision in &mut self.pending_crash_box {
                decision.reward_accum += match decision.team {
                    Team::Home => home_reward,
                    Team::Away => away_reward,
                };
            }
        }
        let mut i = 0;
        while i < self.pending_crash_box.len() {
            if self.pending_crash_box[i].due_tick <= tick {
                let decision = self.pending_crash_box.swap_remove(i);
                let now_territorial = territorial_advantage(snapshot, decision.team);
                let territorial_delta =
                    if now_territorial.is_finite() && decision.decision_territorial.is_finite() {
                        now_territorial - decision.decision_territorial
                    } else {
                        0.0
                    };
                let reward = decision.reward_accum
                    + CRASH_BOX_TERRITORIAL_SHAPING_WEIGHT * territorial_delta;
                if reward.is_finite() {
                    self.crash_box_samples.push(CrashBoxSample {
                        inputs: decision.inputs,
                        action_bias: decision.action_bias,
                        reward,
                    });
                    if self.crash_box_samples.len() > CRASH_BOX_SAMPLE_CAP {
                        let overflow = self.crash_box_samples.len() - CRASH_BOX_SAMPLE_CAP;
                        self.crash_box_samples.drain(0..overflow);
                    }
                }
            } else {
                i += 1;
            }
        }
        if tick % CRASH_BOX_SAMPLE_INTERVAL_TICKS != 0 {
            return;
        }
        // Sample would-be crashers whose team is in possession and near the final third.
        let ids: Vec<usize> = snapshot
            .players
            .iter()
            .filter(|p| matches!(p.role, PlayerRole::Forward | PlayerRole::Midfielder))
            .filter(|p| snapshot.possession_team() == Some(p.team))
            .map(|p| p.id)
            .collect();
        for id in ids {
            let Some(player) = snapshot.players.iter().find(|p| p.id == id) else {
                continue;
            };
            let Some(inputs) = snapshot.build_crash_box_inputs(player) else {
                continue;
            };
            // Only sample when a cross is genuinely on (the decision is live).
            if inputs.cross_imminent < 0.5 {
                continue;
            }
            let action_bias = snapshot
                .crash_box_appetite(&inputs)
                .unwrap_or_else(|| analytic_crash_box_appetite(&inputs));
            let territorial = territorial_advantage(snapshot, player.team);
            if territorial.is_finite() {
                self.pending_crash_box.push(PendingCrashBoxDecision {
                    team: player.team,
                    player_id: id,
                    inputs,
                    action_bias,
                    decision_territorial: territorial,
                    reward_accum: 0.0,
                    due_tick: tick + CRASH_BOX_REWARD_WINDOW_TICKS,
                });
            }
        }
    }

    pub fn drain_crash_box_samples(&mut self) -> Vec<CrashBoxSample> {
        std::mem::take(&mut self.crash_box_samples)
    }

    pub fn set_crash_box_head(&mut self, head: CrashBoxHead) {
        self.crash_box_head = Some(std::sync::Arc::new(head));
    }
}

#[cfg(test)]
mod crash_box_model_tests {
    use super::*;

    fn baseline_inputs() -> CrashBoxInputs {
        CrashBoxInputs {
            box_congestion: 0.2,
            defenders_in_box: 0.4,
            dist_to_box: 0.4,
            role_forward: 1.0,
            cross_imminent: 1.0,
            arrival_space: 0.6,
            needed_behind: 0.2,
        }
    }

    #[test]
    fn features_finite_bounded_right_dim() {
        let f = baseline_inputs().to_features();
        assert_eq!(f.len(), CRASH_BOX_FEATURE_DIM);
        for v in f {
            assert!(v.is_finite() && (0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn imminent_cross_open_space_commits() {
        assert!(analytic_crash_box_appetite(&baseline_inputs()) > 0.0);
    }

    #[test]
    fn full_box_holds() {
        let mut i = baseline_inputs();
        i.box_congestion = 1.0;
        i.needed_behind = 1.0;
        assert!(analytic_crash_box_appetite(&i) < 0.0);
    }

    #[test]
    fn head_bounded_and_rwr_separates() {
        let inputs = baseline_inputs();
        assert!((-1.0..=1.0).contains(&CrashBoxHead::new(1).predict(&inputs).unwrap()));
        let mut hi = CrashBoxHead::new(1);
        let mut lo = CrashBoxHead::new(1);
        let hi_s: Vec<CrashBoxSample> = (0..32)
            .flat_map(|_| {
                [
                    CrashBoxSample {
                        inputs: inputs.clone(),
                        action_bias: 0.8,
                        reward: 1.0,
                    },
                    CrashBoxSample {
                        inputs: inputs.clone(),
                        action_bias: -0.8,
                        reward: -1.0,
                    },
                ]
            })
            .collect();
        let lo_s: Vec<CrashBoxSample> = (0..32)
            .flat_map(|_| {
                [
                    CrashBoxSample {
                        inputs: inputs.clone(),
                        action_bias: -0.8,
                        reward: 1.0,
                    },
                    CrashBoxSample {
                        inputs: inputs.clone(),
                        action_bias: 0.8,
                        reward: -1.0,
                    },
                ]
            })
            .collect();
        for _ in 0..80 {
            hi.train_reward_weighted(&hi_s, 0.05);
            lo.train_reward_weighted(&lo_s, 0.05);
        }
        assert!(hi.predict(&inputs).unwrap() > lo.predict(&inputs).unwrap());
    }

    #[test]
    fn train_entry_skips_empty() {
        assert!(train_crash_box_head(&[], 1, 4, 0.02).is_none());
    }
}
