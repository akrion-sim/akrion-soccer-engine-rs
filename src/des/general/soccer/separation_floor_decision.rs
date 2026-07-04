//! Learnable **same-team separation decision** — makes the fixed 4-yard separation floor /
//! graduated 7→6→5→4 keep-out a *situational, learned* choice instead of one hard constant.
//!
//! ## Why this module exists
//!
//! The same-team separation floor ([`super::dd_soccer_enable_same_team_separation_floor`]) keeps
//! teammates apart with a **fixed** influence radius — the live-movement MPC keep-out inflates a
//! teammate obstacle to `SAME_TEAM_MIN_SEPARATION_YARDS + SAME_TEAM_SEPARATION_INFLUENCE_YARDS`
//! (4 + 4 = 8yd) with a single 18-yard-box exception. That is a good *predilection* (don't crowd a
//! teammate, keep the pitch stretched) but the right amount of space is situational: two players
//! combining in the attacking box (give-and-go, cut-back) want to be *tighter* than the floor
//! allows, while in build-up the team wants to be *wider* than 8yd to stretch a press. A single
//! constant with a binary box-waiver cannot express that.
//!
//! This module makes the **desired separation** a learnable MDP/POMDP choice: a signed
//! **separation bias** in `[-1, 1]` (positive ⇒ want *more* space / spread; negative ⇒ tolerate
//! *less* / combine tighter) that modulates the keep-out influence radius, learned by
//! reward-weighted regression from the real reward pipeline (the graduated same-team proximity
//! penalty is one of its terms, so crowding that never pays off is penalized while crowding that
//! produces a rewarded combination is reinforced). The hard barrier is unchanged; only the *route*
//! keep-out radius the planner uses flexes.
//!
//! Mirrors [`super::lane_affinity_decision`]: [`SeparationFloorInputs`] (POMDP obs) →
//! [`analytic_separation_bias`] (coach's prior ≈ 0) → [`SeparationFloorHead`] (tanh RWR head). The
//! live seam is [`WorldSnapshot::separation_floor_influence_radius`].
//!
//! ## Parity
//!
//! Doubly inert unless live: the separation floor itself is gated by
//! [`super::dd_soccer_enable_same_team_separation_floor`], and this modulation is behind its own
//! gate [`separation_floor_model_enabled`] — **on by default in production** (analytic prior ≈ 0
//! ⇒ near-identical 8yd radius), **off under `#[cfg(test)]`** so the suite is byte-identical.

use super::*;
use crate::des::general::neural_network::{ActivationName, FeedForwardNetwork, RandomNetworkSpec};
use crate::des::general::prng::mulberry32;

/// Number of features in the separation state vector. Single source of truth in
/// [`SeparationFloorInputs::to_features`].
pub const SEPARATION_FLOOR_FEATURE_DIM: usize = 9;
/// Hidden width of the learned regression head.
pub const SEPARATION_FLOOR_HIDDEN_UNITS: usize = 14;
/// Maximum the learned bias may move the keep-out influence radius, in yards. A bias of `+1`
/// widens the radius this much (spread), `-1` tightens it this much (combine). Bounded and
/// clamped so a learned choice flexes the 8yd radius within a football-sane `[floor, ~11]` band.
pub const SEPARATION_FLOOR_MAX_ADJUST_YARDS: f64 = 3.0;
/// Hard ceiling on the modulated influence radius (yd) so a runaway bias can't make teammates
/// avoid each other across half the pitch.
pub const SEPARATION_FLOOR_MAX_RADIUS_YARDS: f64 = 11.0;

const SEPARATION_FLOOR_MODEL_ENABLE_ENV: &str = "DD_SOCCER_ENABLE_SEPARATION_FLOOR_MODEL";
const SEPARATION_FLOOR_MODEL_DISABLE_ENV: &str = "DD_SOCCER_DISABLE_SEPARATION_FLOOR_MODEL";

fn env_flag(name: &str) -> Option<bool> {
    std::env::var(name).ok().map(|raw| {
        let v = raw.trim().to_ascii_lowercase();
        matches!(v.as_str(), "1" | "true" | "yes" | "on")
    })
}

fn separation_floor_model_enabled_from_env() -> bool {
    if env_flag(SEPARATION_FLOOR_MODEL_DISABLE_ENV).unwrap_or(false) {
        return false;
    }
    env_flag(SEPARATION_FLOOR_MODEL_ENABLE_ENV).unwrap_or(true)
}

/// Whether the learned separation modulation is consulted this process. Production caches once
/// (default **on**); tests default **off** so the suite is byte-identical.
pub fn separation_floor_model_enabled() -> bool {
    #[cfg(test)]
    {
        if env_flag(SEPARATION_FLOOR_MODEL_DISABLE_ENV).unwrap_or(false) {
            return false;
        }
        env_flag(SEPARATION_FLOOR_MODEL_ENABLE_ENV).unwrap_or(false)
    }
    #[cfg(not(test))]
    {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(separation_floor_model_enabled_from_env)
    }
}

/// How often (ticks) separation RL decisions are sampled while the model is on.
pub const SEPARATION_FLOOR_SAMPLE_INTERVAL_TICKS: u64 = 8;
/// Window (ticks) over which a sampled decision's reward is measured.
pub const SEPARATION_FLOOR_REWARD_WINDOW_TICKS: u64 = 30;
/// Cap on the rolling RL sample buffer.
pub const SEPARATION_FLOOR_SAMPLE_CAP: usize = 8192;
/// Minimum training steps before a trained head is consumed live.
pub const SEPARATION_FLOOR_HEAD_MIN_TRAINING_STEPS: usize = 200;
/// Weight on the dense territorial delta added to the windowed real-reward sum.
pub const SEPARATION_FLOOR_TERRITORIAL_SHAPING_WEIGHT: f64 = 1.0;

/// Raw per-player separation state the choice is a function of.
#[derive(Clone, Debug, PartialEq)]
pub struct SeparationFloorInputs {
    /// `1` if the player's team is in possession, else `0`.
    pub in_possession: f64,
    /// `1` if the player's team is defending (opponent controls), else `0`.
    pub defensive_phase: f64,
    /// `1` if the player is inside the attacking penalty area (a legitimate combine/congest zone).
    pub in_attacking_box: f64,
    /// `1` if the player is inside their own penalty area (defending a cross — congestion ok).
    pub in_own_box: f64,
    /// Teammate crowding in `[0, 1]`: `1` ⇒ nearest teammate is right on top of me (already tight).
    pub teammate_crowding: f64,
    /// Space around me in `[0, 1]`: opponent distance normalized (`1` ⇒ acres of room).
    pub space_around: f64,
    /// Proximity to the ball in `[0, 1]` (`1` ⇒ right on the ball; combine tighter near it).
    pub near_ball: f64,
    /// Local attacker overload in `[0, 1]` — team-mates near the ball who could combine.
    pub combine_support: f64,
    /// `1` if the player is a forward, else `0` (forwards combine in tight areas more).
    pub role_forward: f64,
}

impl SeparationFloorInputs {
    /// The normalized network input vector. **Single source of truth** for the layout.
    pub fn to_features(&self) -> [f64; SEPARATION_FLOOR_FEATURE_DIM] {
        let unit = |u: f64| u.clamp(0.0, 1.0);
        [
            unit(self.in_possession),
            unit(self.defensive_phase),
            unit(self.in_attacking_box),
            unit(self.in_own_box),
            unit(self.teammate_crowding),
            unit(self.space_around),
            unit(self.near_ball),
            unit(self.combine_support),
            unit(self.role_forward),
        ]
    }
}

/// Analytic, deterministic seed for the **separation bias** in `[-1, 1]` (positive = want more
/// space / spread; negative = tolerate less / combine tighter). Coach's prior: **≈ 0 (keep the
/// 8yd keep-out) by default.** Lean toward *tighter* (negative) when combining in the attacking
/// box near the ball with support; lean toward *wider* (positive) when in possession in open build-up
/// crowded by a teammate — stretch the pitch.
pub fn analytic_separation_bias(inputs: &SeparationFloorInputs) -> f64 {
    let mut z = 0.0;
    // COMBINE tighter (negative): attacking box, near the ball, with support to link with.
    z -= 0.9 * inputs.in_attacking_box * inputs.near_ball * (0.4 + 0.6 * inputs.combine_support);
    // SPREAD wider (positive): in possession, out of the boxes, a teammate crowding me, space to use.
    z += 0.7
        * inputs.in_possession
        * (1.0 - inputs.in_attacking_box)
        * (1.0 - inputs.in_own_box)
        * inputs.teammate_crowding
        * inputs.space_around;
    z.tanh()
}

/// One reward-weighted RL training row for the separation head.
#[derive(Clone, Debug, PartialEq)]
pub struct SeparationFloorSample {
    pub inputs: SeparationFloorInputs,
    /// The separation action taken, in `[-1, 1]` (`+` = spread wider, `-` = combined tighter).
    pub action_bias: f64,
    pub reward: f64,
}

/// An open separation decision awaiting its windowed reward.
#[derive(Clone, Debug, PartialEq)]
pub struct PendingSeparationFloorDecision {
    pub team: Team,
    pub player_id: usize,
    pub inputs: SeparationFloorInputs,
    pub action_bias: f64,
    pub decision_territorial: f64,
    pub reward_accum: f64,
    pub due_tick: u64,
}

/// Learned regression head for the signed separation bias.
#[derive(Clone, Debug)]
pub struct SeparationFloorHead {
    network: FeedForwardNetwork,
    training_steps: usize,
    last_loss: Option<f64>,
}

impl SeparationFloorHead {
    pub fn new(seed: u32) -> Self {
        let mut rng = mulberry32(seed ^ 0x71B2_9E4D);
        let network = FeedForwardNetwork::random(
            &RandomNetworkSpec {
                input_dim: SEPARATION_FLOOR_FEATURE_DIM,
                hidden_layers: vec![SEPARATION_FLOOR_HIDDEN_UNITS],
                output_dim: 1,
                hidden_activation: ActivationName::Tanh,
                output_activation: ActivationName::Tanh,
                weight_scale: None,
            },
            &mut rng,
        );
        SeparationFloorHead {
            network,
            training_steps: 0,
            last_loss: None,
        }
    }

    pub fn predict(&self, inputs: &SeparationFloorInputs) -> Option<f64> {
        let features = inputs.to_features();
        if features.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let out = self.network.predict(&features[..]);
        out.first().copied().filter(|p| p.is_finite())
    }

    pub fn train_reward_weighted(
        &mut self,
        samples: &[SeparationFloorSample],
        learning_rate: f64,
    ) -> f64 {
        let finite: Vec<&SeparationFloorSample> = samples
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

/// Outcome of training a separation head on a drained RL corpus.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SeparationFloorTrainingReport {
    pub samples: usize,
    pub training_steps: usize,
    pub epochs: usize,
    pub final_loss: f64,
}

/// Load-and-train entry for a separation head.
pub fn train_separation_floor_head(
    samples: &[SeparationFloorSample],
    seed: u32,
    epochs: usize,
    learning_rate: f64,
) -> Option<(SeparationFloorHead, SeparationFloorTrainingReport)> {
    let usable = samples
        .iter()
        .filter(|s| s.reward.is_finite() && s.action_bias.is_finite())
        .count();
    if usable == 0 || epochs == 0 {
        return None;
    }
    let mut head = SeparationFloorHead::new(seed);
    let mut final_loss = 0.0;
    for _ in 0..epochs {
        final_loss = head.train_reward_weighted(samples, learning_rate);
    }
    let report = SeparationFloorTrainingReport {
        samples: usable,
        training_steps: head.training_steps(),
        epochs,
        final_loss,
    };
    Some((head, report))
}

/// Public wrapper returning ONLY the report.
pub fn report_separation_floor_training(
    samples: &[SeparationFloorSample],
    seed: u32,
    epochs: usize,
    learning_rate: f64,
) -> Option<SeparationFloorTrainingReport> {
    train_separation_floor_head(samples, seed, epochs, learning_rate).map(|(_, report)| report)
}

impl WorldSnapshot {
    /// Build the [`SeparationFloorInputs`] for `player`. Pure / RNG-free.
    pub(crate) fn build_separation_floor_inputs(
        &self,
        player: &PlayerSnapshot,
    ) -> Option<SeparationFloorInputs> {
        let pos = self.player_snapshot_position(player);
        let possessing = self
            .controlled_possession_team()
            .or_else(|| self.possession_team());
        let in_possession = possessing == Some(player.team);
        let defensive_phase = self.possession_team() == Some(player.team.other());
        let in_attacking_box = self.point_in_own_penalty_area(player.team.other(), pos);
        let in_own_box = self.point_in_own_penalty_area(player.team, pos);

        let nearest_teammate = self
            .players
            .iter()
            .filter(|p| p.team == player.team && p.id != player.id)
            .map(|p| self.player_snapshot_position(p).distance(pos))
            .fold(f64::INFINITY, f64::min);
        let teammate_crowding = if nearest_teammate.is_finite() {
            (1.0 - (nearest_teammate / 10.0).clamp(0.0, 1.0)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let space_around = (self.nearest_opponent_distance_at(player.team, pos) / 12.0).clamp(0.0, 1.0);
        let dist_to_ball = pos.distance(self.ball.position);
        let near_ball = (1.0 - (dist_to_ball / 20.0).clamp(0.0, 1.0)).clamp(0.0, 1.0);
        let combine_support = {
            let n = self
                .players
                .iter()
                .filter(|p| {
                    p.team == player.team
                        && p.id != player.id
                        && self.player_snapshot_position(p).distance(self.ball.position) < 18.0
                })
                .count();
            (n as f64 / 3.0).clamp(0.0, 1.0)
        };

        Some(SeparationFloorInputs {
            in_possession: in_possession as i32 as f64,
            defensive_phase: defensive_phase as i32 as f64,
            in_attacking_box: in_attacking_box as i32 as f64,
            in_own_box: in_own_box as i32 as f64,
            teammate_crowding,
            space_around,
            near_ball,
            combine_support,
            role_forward: (player.role == PlayerRole::Forward) as i32 as f64,
        })
    }

    /// The model's signed separation bias in `(-1, 1)`, or `None` when gated off.
    pub(crate) fn separation_bias(&self, inputs: &SeparationFloorInputs) -> Option<f64> {
        if !separation_floor_model_enabled() {
            return None;
        }
        let bias = self
            .separation_floor_head
            .as_ref()
            .filter(|head| head.training_steps() >= SEPARATION_FLOOR_HEAD_MIN_TRAINING_STEPS)
            .and_then(|head| head.predict(inputs))
            .unwrap_or_else(|| analytic_separation_bias(inputs));
        bias.is_finite().then(|| bias.clamp(-1.0, 1.0))
    }

}

impl SoccerMatch {
    /// The effective same-team keep-out **influence radius** (yd) for `player_id`, given the base
    /// `base_influence` (the fixed 8yd floor+influence). Returns `base_influence` unchanged when
    /// the model is off ⇒ byte-identical. When on, shifts it by the learned separation bias,
    /// clamped to `[SAME_TEAM_MIN_SEPARATION_YARDS, SEPARATION_FLOOR_MAX_RADIUS_YARDS]`. Builds the
    /// inputs inline from live match state (the MPC obstacle builder is a `SoccerMatch` method with
    /// no snapshot in hand).
    pub(crate) fn separation_floor_influence_radius(
        &self,
        player_id: usize,
        base_influence: f64,
    ) -> f64 {
        if !separation_floor_model_enabled() {
            return base_influence;
        }
        let Some(player) = self.players.get(player_id) else {
            return base_influence;
        };
        let pos = player.position;
        let team = player.team;
        // Possession from the ball holder's team (cheap; no snapshot needed).
        let holder_team = self
            .ball
            .holder
            .and_then(|h| self.players.get(h))
            .map(|p| p.team);
        let in_possession = holder_team == Some(team);
        let defensive_phase = holder_team == Some(team.other());
        let in_attacking_box = self.point_in_own_penalty_area(team.other(), pos);
        let in_own_box = self.point_in_own_penalty_area(team, pos);
        let mut nearest_teammate = f64::INFINITY;
        let mut nearest_opp = f64::INFINITY;
        let mut combine = 0usize;
        for other in &self.players {
            if other.id == player_id {
                continue;
            }
            let d = other.position.distance(pos);
            if other.team == team {
                nearest_teammate = nearest_teammate.min(d);
                if other.position.distance(self.ball.position) < 18.0 {
                    combine += 1;
                }
            } else {
                nearest_opp = nearest_opp.min(d);
            }
        }
        let teammate_crowding = if nearest_teammate.is_finite() {
            (1.0 - (nearest_teammate / 10.0).clamp(0.0, 1.0)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let space_around = if nearest_opp.is_finite() {
            (nearest_opp / 12.0).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let near_ball =
            (1.0 - (pos.distance(self.ball.position) / 20.0).clamp(0.0, 1.0)).clamp(0.0, 1.0);
        let inputs = SeparationFloorInputs {
            in_possession: in_possession as i32 as f64,
            defensive_phase: defensive_phase as i32 as f64,
            in_attacking_box: in_attacking_box as i32 as f64,
            in_own_box: in_own_box as i32 as f64,
            teammate_crowding,
            space_around,
            near_ball,
            combine_support: (combine as f64 / 3.0).clamp(0.0, 1.0),
            role_forward: (player.role == PlayerRole::Forward) as i32 as f64,
        };
        let bias = self
            .separation_floor_head
            .as_ref()
            .filter(|head| head.training_steps() >= SEPARATION_FLOOR_HEAD_MIN_TRAINING_STEPS)
            .and_then(|head| head.predict(&inputs))
            .unwrap_or_else(|| analytic_separation_bias(&inputs));
        if !bias.is_finite() {
            return base_influence;
        }
        (base_influence + bias.clamp(-1.0, 1.0) * SEPARATION_FLOOR_MAX_ADJUST_YARDS)
            .clamp(SAME_TEAM_MIN_SEPARATION_YARDS, SEPARATION_FLOOR_MAX_RADIUS_YARDS)
    }
}

impl SoccerMatch {
    /// Gated per-tick RL sample collection for the separation head. No-op unless enabled.
    pub(crate) fn collect_separation_floor_rl_samples(&mut self, snapshot: &WorldSnapshot) {
        if !separation_floor_model_enabled() || !dd_soccer_enable_same_team_separation_floor() {
            return;
        }
        let tick = self.tick;
        if !self.pending_separation_floor.is_empty() && !self.reward_events.is_empty() {
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
            for decision in &mut self.pending_separation_floor {
                decision.reward_accum += match decision.team {
                    Team::Home => home_reward,
                    Team::Away => away_reward,
                };
            }
        }
        let mut i = 0;
        while i < self.pending_separation_floor.len() {
            if self.pending_separation_floor[i].due_tick <= tick {
                let decision = self.pending_separation_floor.swap_remove(i);
                let now_territorial = territorial_advantage(snapshot, decision.team);
                let territorial_delta = if now_territorial.is_finite()
                    && decision.decision_territorial.is_finite()
                {
                    now_territorial - decision.decision_territorial
                } else {
                    0.0
                };
                let reward = decision.reward_accum
                    + SEPARATION_FLOOR_TERRITORIAL_SHAPING_WEIGHT * territorial_delta;
                if reward.is_finite() {
                    self.separation_floor_samples.push(SeparationFloorSample {
                        inputs: decision.inputs,
                        action_bias: decision.action_bias,
                        reward,
                    });
                    if self.separation_floor_samples.len() > SEPARATION_FLOOR_SAMPLE_CAP {
                        let overflow =
                            self.separation_floor_samples.len() - SEPARATION_FLOOR_SAMPLE_CAP;
                        self.separation_floor_samples.drain(0..overflow);
                    }
                }
            } else {
                i += 1;
            }
        }
        if tick % SEPARATION_FLOOR_SAMPLE_INTERVAL_TICKS != 0 {
            return;
        }
        let ids: Vec<usize> = snapshot
            .players
            .iter()
            .filter(|p| p.role != PlayerRole::Goalkeeper)
            .map(|p| p.id)
            .collect();
        for id in ids {
            let Some(player) = snapshot.players.iter().find(|p| p.id == id) else {
                continue;
            };
            let Some(inputs) = snapshot.build_separation_floor_inputs(player) else {
                continue;
            };
            let action_bias = snapshot
                .separation_bias(&inputs)
                .unwrap_or_else(|| analytic_separation_bias(&inputs));
            let territorial = territorial_advantage(snapshot, player.team);
            if territorial.is_finite() {
                self.pending_separation_floor
                    .push(PendingSeparationFloorDecision {
                        team: player.team,
                        player_id: id,
                        inputs,
                        action_bias,
                        decision_territorial: territorial,
                        reward_accum: 0.0,
                        due_tick: tick + SEPARATION_FLOOR_REWARD_WINDOW_TICKS,
                    });
            }
        }
    }

    pub fn drain_separation_floor_samples(&mut self) -> Vec<SeparationFloorSample> {
        std::mem::take(&mut self.separation_floor_samples)
    }

    pub fn set_separation_floor_head(&mut self, head: SeparationFloorHead) {
        self.separation_floor_head = Some(std::sync::Arc::new(head));
    }
}

#[cfg(test)]
mod separation_floor_tests {
    use super::*;

    fn baseline_inputs() -> SeparationFloorInputs {
        SeparationFloorInputs {
            in_possession: 1.0,
            defensive_phase: 0.0,
            in_attacking_box: 0.0,
            in_own_box: 0.0,
            teammate_crowding: 0.3,
            space_around: 0.5,
            near_ball: 0.3,
            combine_support: 0.3,
            role_forward: 0.0,
        }
    }

    #[test]
    fn features_are_finite_bounded_and_right_dim() {
        let f = baseline_inputs().to_features();
        assert_eq!(f.len(), SEPARATION_FLOOR_FEATURE_DIM);
        for v in f {
            assert!(v.is_finite() && (0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn analytic_bias_is_a_valid_signed_unit() {
        let b = analytic_separation_bias(&baseline_inputs());
        assert!((-1.0..=1.0).contains(&b));
    }

    #[test]
    fn combining_in_the_box_near_the_ball_tightens() {
        let mut i = baseline_inputs();
        i.in_attacking_box = 1.0;
        i.near_ball = 1.0;
        i.combine_support = 1.0;
        assert!(
            analytic_separation_bias(&i) < 0.0,
            "combining in the box near the ball should tolerate a tighter separation"
        );
    }

    #[test]
    fn crowded_in_open_buildup_spreads() {
        let mut i = baseline_inputs();
        i.in_possession = 1.0;
        i.in_attacking_box = 0.0;
        i.in_own_box = 0.0;
        i.teammate_crowding = 1.0;
        i.space_around = 1.0;
        i.near_ball = 0.0;
        assert!(
            analytic_separation_bias(&i) > 0.0,
            "a crowded player in open build-up should spread to stretch the pitch"
        );
    }

    #[test]
    fn neutral_state_keeps_the_base_radius() {
        let mut i = baseline_inputs();
        i.teammate_crowding = 0.0;
        i.in_attacking_box = 0.0;
        assert!(analytic_separation_bias(&i).abs() < 0.15);
    }

    #[test]
    fn head_predicts_a_bounded_signed_bias() {
        let head = SeparationFloorHead::new(2);
        let b = head.predict(&baseline_inputs()).expect("finite");
        assert!((-1.0..=1.0).contains(&b));
    }

    #[test]
    fn reward_weighted_training_steers_toward_the_high_reward_action() {
        // Two heads from the SAME init: one rewarded for combining tighter (−0.8), the other for
        // spreading (+0.8). RWR must leave the tighten-rewarded head lower than the spread one —
        // robust to the random initialisation (which a single-head before/after check is not).
        let inputs = baseline_inputs();
        let mut tighten = SeparationFloorHead::new(13);
        let mut spread = SeparationFloorHead::new(13);
        let tighten_samples: Vec<SeparationFloorSample> = (0..32)
            .flat_map(|_| {
                [
                    SeparationFloorSample { inputs: inputs.clone(), action_bias: -0.8, reward: 1.0 },
                    SeparationFloorSample { inputs: inputs.clone(), action_bias: 0.8, reward: -1.0 },
                ]
            })
            .collect();
        let spread_samples: Vec<SeparationFloorSample> = (0..32)
            .flat_map(|_| {
                [
                    SeparationFloorSample { inputs: inputs.clone(), action_bias: 0.8, reward: 1.0 },
                    SeparationFloorSample { inputs: inputs.clone(), action_bias: -0.8, reward: -1.0 },
                ]
            })
            .collect();
        for _ in 0..80 {
            tighten.train_reward_weighted(&tighten_samples, 0.05);
            spread.train_reward_weighted(&spread_samples, 0.05);
        }
        let tighten_out = tighten.predict(&inputs).expect("finite");
        let spread_out = spread.predict(&inputs).expect("finite");
        assert!(
            tighten_out < spread_out,
            "tighten-rewarded head ({tighten_out}) should sit below spread-rewarded ({spread_out})"
        );
    }

    #[test]
    fn train_entry_skips_empty_and_trains_valid() {
        assert!(train_separation_floor_head(&[], 1, 4, 0.02).is_none());
        let samples: Vec<SeparationFloorSample> = (0..16)
            .map(|i| SeparationFloorSample {
                inputs: baseline_inputs(),
                action_bias: if i % 2 == 0 { -0.8 } else { 0.8 },
                reward: if i % 2 == 0 { 1.0 } else { -1.0 },
            })
            .collect();
        let (_, report) =
            train_separation_floor_head(&samples, 1, 4, 0.02).expect("trains a usable corpus");
        assert_eq!(report.samples, 16);
        assert!(report.training_steps > 0 && report.final_loss.is_finite());
    }
}
