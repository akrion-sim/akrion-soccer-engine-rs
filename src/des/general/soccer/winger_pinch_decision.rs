//! Learnable **winger pinch-appetite decision** — makes the hand-tuned "stay wide vs pinch
//! in" bucket scoring of [`super::winger_pinch`] a *learned* choice.
//!
//! [`super::winger_pinch::decide_winger_pinch`] scores three discrete buckets — `StayWide`,
//! `PinchHalfSpace`, `PinchBackPost` — from POMDP-observable features using **hand-calibrated
//! constant weights**, then argmax-selects. That captures the football logic (the weak-side
//! winger pinches to the back post / cut-back; the ball-side winger holds width) but the
//! *appetite* to pinch is a fixed prior. This module makes that appetite learnable: a signed
//! **pinch bias** in `[-1, 1]` (positive ⇒ lean toward pinching infield, negative ⇒ lean toward
//! holding width) added into the two pinch buckets' scores before the argmax, so the head can
//! learn — from the cross-arrival / header-goal reward channel — how eager a winger should be to
//! leave the touchline in a given situation, while the half-space-vs-back-post sub-choice stays
//! analytic (congestion / offside driven) and the legality mask is untouched.
//!
//! Mirrors the reward-weighted-regression heads ([`super::lane_affinity_decision`],
//! [`super::receive_approach`]): [`WingerPinchInputs`] (POMDP obs) → [`analytic_winger_pinch_bias`]
//! (coach's prior, ≈ 0 by default) → [`WingerPinchHead`] (tanh RWR head). The live seam is the
//! `pinch_bias` argument threaded into [`super::winger_pinch::decide_winger_pinch_with_bias`].
//!
//! ## Parity
//!
//! Doubly inert unless live: the winger-pinch recogniser itself is gated by
//! [`super::winger_pinch::winger_pinch_in_enabled`], and this modulation is behind its own gate
//! [`winger_pinch_model_enabled`] — **on by default in production** (analytic prior ≈ 0 ⇒
//! near-identical argmax), **off under `#[cfg(test)]`**. Off ⇒ `pinch_bias = 0` ⇒
//! `decide_winger_pinch_with_bias` is byte-identical to `decide_winger_pinch`.

use super::*;
use crate::des::general::neural_network::{ActivationName, FeedForwardNetwork, RandomNetworkSpec};
use crate::des::general::prng::mulberry32;

/// Number of features in the winger-pinch state vector. Single source of truth in
/// [`WingerPinchInputs::to_features`].
pub const WINGER_PINCH_FEATURE_DIM: usize = 8;
/// Hidden width of the learned regression head.
pub const WINGER_PINCH_HIDDEN_UNITS: usize = 14;
/// Score weight the learned bias carries into the two pinch buckets. Kept below the
/// `SAME_FLANK_STAY_BONUS` so a positive bias nudges — it cannot by itself drag a ball-side
/// winger (who must hold width) infield.
pub const WINGER_PINCH_APPETITE_WEIGHT: f64 = 1.6;

const WINGER_PINCH_MODEL_ENABLE_ENV: &str = "DD_SOCCER_ENABLE_WINGER_PINCH_MODEL";
const WINGER_PINCH_MODEL_DISABLE_ENV: &str = "DD_SOCCER_DISABLE_WINGER_PINCH_MODEL";

fn env_flag(name: &str) -> Option<bool> {
    std::env::var(name).ok().map(|raw| {
        let v = raw.trim().to_ascii_lowercase();
        matches!(v.as_str(), "1" | "true" | "yes" | "on")
    })
}

fn winger_pinch_model_enabled_from_env() -> bool {
    if env_flag(WINGER_PINCH_MODEL_DISABLE_ENV).unwrap_or(false) {
        return false;
    }
    env_flag(WINGER_PINCH_MODEL_ENABLE_ENV).unwrap_or(true)
}

/// Whether the learned winger pinch-appetite modulation is consulted this process. Production
/// caches once (default **on**); tests default **off** so the suite is byte-identical.
pub fn winger_pinch_model_enabled() -> bool {
    #[cfg(test)]
    {
        if env_flag(WINGER_PINCH_MODEL_DISABLE_ENV).unwrap_or(false) {
            return false;
        }
        env_flag(WINGER_PINCH_MODEL_ENABLE_ENV).unwrap_or(false)
    }
    #[cfg(not(test))]
    {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(winger_pinch_model_enabled_from_env)
    }
}

/// How often (ticks) winger-pinch RL decisions are sampled while the model is on.
pub const WINGER_PINCH_SAMPLE_INTERVAL_TICKS: u64 = 6;
/// Window (ticks) over which a sampled decision's reward is measured.
pub const WINGER_PINCH_REWARD_WINDOW_TICKS: u64 = 45;
/// Cap on the rolling RL sample buffer.
pub const WINGER_PINCH_SAMPLE_CAP: usize = 8192;
/// Minimum training steps before a trained head is consumed live.
pub const WINGER_PINCH_HEAD_MIN_TRAINING_STEPS: usize = 200;
/// Weight on the dense territorial delta added to the windowed real-reward sum.
pub const WINGER_PINCH_TERRITORIAL_SHAPING_WEIGHT: f64 = 1.0;

/// Raw per-winger state the pinch-appetite choice is a function of.
#[derive(Clone, Debug, PartialEq)]
pub struct WingerPinchInputs {
    /// `1` if the ball / crossing position is on this winger's own flank (⇒ he is the deliverer /
    /// overlap and should hold width), else `0`.
    pub ball_on_my_flank: f64,
    /// `1` if a genuine crossing position is on (wide & high in the final third), else `0`.
    pub crossing_position_on: f64,
    /// Ball depth in the attacking final third in `[0, 1]` (0 edge → 1 byline).
    pub depth_frac: f64,
    /// Own attackers already inside the box (excluding this winger), normalized.
    pub box_congestion: f64,
    /// `1` if the back-post arrival would be offside (that bucket is masked), else `0`.
    pub back_post_offside: f64,
    /// Openness in `[0, 1]` at the back-post arrival point (opponent distance normalized).
    pub back_post_space: f64,
    /// Openness in `[0, 1]` at the half-space arrival point.
    pub half_space_space: f64,
    /// Distance (yd) from this winger to the ball, normalized.
    pub dist_to_ball: f64,
}

impl WingerPinchInputs {
    /// The normalized network input vector. **Single source of truth** for the layout.
    pub fn to_features(&self) -> [f64; WINGER_PINCH_FEATURE_DIM] {
        let unit = |u: f64| u.clamp(0.0, 1.0);
        [
            unit(self.ball_on_my_flank),
            unit(self.crossing_position_on),
            unit(self.depth_frac),
            unit(self.box_congestion),
            unit(self.back_post_offside),
            unit(self.back_post_space),
            unit(self.half_space_space),
            unit(self.dist_to_ball),
        ]
    }
}

/// Analytic, deterministic seed for the **pinch bias** in `[-1, 1]` (positive = lean toward
/// pinching infield, negative = lean toward holding width). Coach's prior: **≈ 0 by default**
/// (keep the calibrated bucket scoring). Lean toward pinching when this is the weak-side winger,
/// a cross is on, the ball is deep, and there is space to arrive into; lean toward width when the
/// ball is on this winger's own flank (he is the deliverer).
pub fn analytic_winger_pinch_bias(inputs: &WingerPinchInputs) -> f64 {
    let mut z = 0.0;
    let weak_side = (1.0 - inputs.ball_on_my_flank).clamp(0.0, 1.0);
    // Pinch when weak-side + cross on + deep + space to arrive into.
    z += 0.8
        * weak_side
        * inputs.crossing_position_on
        * (0.4 + 0.6 * inputs.depth_frac)
        * inputs.back_post_space.max(inputs.half_space_space);
    // Hold width when the ball is on my flank (I'm the deliverer / overlap).
    z -= 0.9 * inputs.ball_on_my_flank;
    z.tanh()
}

/// One reward-weighted RL training row for the winger-pinch head.
#[derive(Clone, Debug, PartialEq)]
pub struct WingerPinchSample {
    pub inputs: WingerPinchInputs,
    /// The pinch-appetite action taken, in `[-1, 1]` (`+` = leaned toward pinching infield).
    pub action_bias: f64,
    pub reward: f64,
}

/// An open winger-pinch decision awaiting its windowed reward.
#[derive(Clone, Debug, PartialEq)]
pub struct PendingWingerPinchDecision {
    pub team: Team,
    pub player_id: usize,
    pub inputs: WingerPinchInputs,
    pub action_bias: f64,
    pub decision_territorial: f64,
    pub reward_accum: f64,
    pub due_tick: u64,
}

/// Learned regression head for the signed pinch bias.
#[derive(Clone, Debug)]
pub struct WingerPinchHead {
    network: FeedForwardNetwork,
    training_steps: usize,
    last_loss: Option<f64>,
}

impl WingerPinchHead {
    pub fn new(seed: u32) -> Self {
        let mut rng = mulberry32(seed ^ 0x3C8F_A2D1);
        let network = FeedForwardNetwork::random(
            &RandomNetworkSpec {
                input_dim: WINGER_PINCH_FEATURE_DIM,
                hidden_layers: vec![WINGER_PINCH_HIDDEN_UNITS],
                output_dim: 1,
                hidden_activation: ActivationName::Tanh,
                output_activation: ActivationName::Tanh,
                weight_scale: None,
            },
            &mut rng,
        );
        WingerPinchHead {
            network,
            training_steps: 0,
            last_loss: None,
        }
    }

    pub fn predict(&self, inputs: &WingerPinchInputs) -> Option<f64> {
        let features = inputs.to_features();
        if features.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let out = self.network.predict(&features[..]);
        out.first().copied().filter(|p| p.is_finite())
    }

    pub fn train_reward_weighted(
        &mut self,
        samples: &[WingerPinchSample],
        learning_rate: f64,
    ) -> f64 {
        let finite: Vec<&WingerPinchSample> = samples
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

/// Outcome of training a winger-pinch head on a drained RL corpus.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WingerPinchTrainingReport {
    pub samples: usize,
    pub training_steps: usize,
    pub epochs: usize,
    pub final_loss: f64,
}

/// Load-and-train entry for a winger-pinch head.
pub fn train_winger_pinch_head(
    samples: &[WingerPinchSample],
    seed: u32,
    epochs: usize,
    learning_rate: f64,
) -> Option<(WingerPinchHead, WingerPinchTrainingReport)> {
    let usable = samples
        .iter()
        .filter(|s| s.reward.is_finite() && s.action_bias.is_finite())
        .count();
    if usable == 0 || epochs == 0 {
        return None;
    }
    let mut head = WingerPinchHead::new(seed);
    let mut final_loss = 0.0;
    for _ in 0..epochs {
        final_loss = head.train_reward_weighted(samples, learning_rate);
    }
    let report = WingerPinchTrainingReport {
        samples: usable,
        training_steps: head.training_steps(),
        epochs,
        final_loss,
    };
    Some((head, report))
}

/// Public wrapper returning ONLY the report.
pub fn report_winger_pinch_training(
    samples: &[WingerPinchSample],
    seed: u32,
    epochs: usize,
    learning_rate: f64,
) -> Option<WingerPinchTrainingReport> {
    train_winger_pinch_head(samples, seed, epochs, learning_rate).map(|(_, report)| report)
}

impl WorldSnapshot {
    /// Build the [`WingerPinchInputs`] for a wide attacker whose team is attacking in the final
    /// third — the same POMDP context [`WorldSnapshot::winger_pinch_target_for`] derives, plus the
    /// openness of the two arrival points. `None` when the winger-pinch recogniser does not apply
    /// to `player` (not a wide attacker in possession in the final third off the ball).
    pub(crate) fn build_winger_pinch_inputs(
        &self,
        player: &PlayerSnapshot,
    ) -> Option<WingerPinchInputs> {
        if self.possession_team() != Some(player.team)
            || self.ball.holder == Some(player.id)
            || player.controller_slot.is_some()
            || !self.is_wide_attacker(player)
            || self.active_set_play.is_some()
        {
            return None;
        }
        let attacked_goal_y = player.team.goal_y(self.field_length);
        if !ball_in_attacking_final_third(self.ball.position.y, attacked_goal_y, self.field_length) {
            return None;
        }
        let dir = player.team.attack_dir();
        let center_x = self.field_width * 0.5;
        let crosser = self
            .ball
            .holder
            .and_then(|h| self.players.iter().find(|p| p.id == h))
            .filter(|p| p.team == player.team)
            .map(|p| self.player_snapshot_position(p))
            .unwrap_or(self.ball.position);
        let my_side = (player.home_position.x - center_x).signum();
        let ball_side = (crosser.x - center_x).signum();
        let ball_on_my_flank = my_side != 0.0 && my_side == ball_side;
        let crossing_position_on = winger_pinch::flank_final_third_crash_box_geometry(
            crosser,
            attacked_goal_y,
            self.field_width,
            self.field_length,
        );
        let depth_frac = winger_pinch::final_third_depth_frac(
            self.ball.position.y,
            attacked_goal_y,
            self.field_length,
        );
        let box_congestion = self
            .players
            .iter()
            .filter(|p| {
                p.team == player.team
                    && p.id != player.id
                    && self.ball.holder != Some(p.id)
                    && matches!(p.role, PlayerRole::Forward | PlayerRole::Midfielder)
                    && self
                        .point_in_own_penalty_area(player.team.other(), self.player_snapshot_position(p))
            })
            .count();
        let back_post_probe = winger_pinch::winger_pinch_target(
            winger_pinch::WingerPinchChoice::PinchBackPost,
            player.home_position.x,
            attacked_goal_y,
            dir,
            self.field_width,
            self.field_length,
        );
        let half_space_probe = winger_pinch::winger_pinch_target(
            winger_pinch::WingerPinchChoice::PinchHalfSpace,
            player.home_position.x,
            attacked_goal_y,
            dir,
            self.field_width,
            self.field_length,
        );
        let back_post_offside = back_post_probe
            .map(|probe| self.position_would_be_offside(player.team, probe))
            .unwrap_or(true);
        let openness = |probe: Option<Vec2>| {
            probe
                .map(|p| (self.nearest_opponent_distance_at(player.team, p) / 12.0).clamp(0.0, 1.0))
                .unwrap_or(0.0)
        };

        Some(WingerPinchInputs {
            ball_on_my_flank: ball_on_my_flank as i32 as f64,
            crossing_position_on: crossing_position_on as i32 as f64,
            depth_frac,
            box_congestion: (box_congestion as f64 / 4.0).clamp(0.0, 1.0),
            back_post_offside: back_post_offside as i32 as f64,
            back_post_space: openness(back_post_probe),
            half_space_space: openness(half_space_probe),
            dist_to_ball: (self
                .player_snapshot_position(player)
                .distance(self.ball.position)
                / 40.0)
                .clamp(0.0, 1.0),
        })
    }

    /// The model's signed pinch bias in `(-1, 1)`, or `None` when gated off / degenerate.
    pub(crate) fn winger_pinch_bias(&self, inputs: &WingerPinchInputs) -> Option<f64> {
        if !winger_pinch_model_enabled() {
            return None;
        }
        let bias = self
            .winger_pinch_head
            .as_ref()
            .filter(|head| head.training_steps() >= WINGER_PINCH_HEAD_MIN_TRAINING_STEPS)
            .and_then(|head| head.predict(inputs))
            .unwrap_or_else(|| analytic_winger_pinch_bias(inputs));
        bias.is_finite().then(|| bias.clamp(-1.0, 1.0))
    }

    /// The pinch-appetite bias to fold into the bucket scoring (0 when gated off ⇒ parity).
    pub(crate) fn winger_pinch_effective_bias(&self, inputs: &WingerPinchInputs) -> f64 {
        self.winger_pinch_bias(inputs).unwrap_or(0.0)
    }
}

impl SoccerMatch {
    /// Gated per-tick RL sample collection for the winger-pinch head. No-op unless enabled.
    pub(crate) fn collect_winger_pinch_rl_samples(&mut self, snapshot: &WorldSnapshot) {
        if !winger_pinch_model_enabled() || !winger_pinch::winger_pinch_in_enabled() {
            return;
        }
        let tick = self.tick;
        if !self.pending_winger_pinch.is_empty() && !self.reward_events.is_empty() {
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
            for decision in &mut self.pending_winger_pinch {
                decision.reward_accum += match decision.team {
                    Team::Home => home_reward,
                    Team::Away => away_reward,
                };
            }
        }
        let mut i = 0;
        while i < self.pending_winger_pinch.len() {
            if self.pending_winger_pinch[i].due_tick <= tick {
                let decision = self.pending_winger_pinch.swap_remove(i);
                let now_territorial = territorial_advantage(snapshot, decision.team);
                let territorial_delta = if now_territorial.is_finite()
                    && decision.decision_territorial.is_finite()
                {
                    now_territorial - decision.decision_territorial
                } else {
                    0.0
                };
                let reward = decision.reward_accum
                    + WINGER_PINCH_TERRITORIAL_SHAPING_WEIGHT * territorial_delta;
                if reward.is_finite() {
                    self.winger_pinch_samples.push(WingerPinchSample {
                        inputs: decision.inputs,
                        action_bias: decision.action_bias,
                        reward,
                    });
                    if self.winger_pinch_samples.len() > WINGER_PINCH_SAMPLE_CAP {
                        let overflow = self.winger_pinch_samples.len() - WINGER_PINCH_SAMPLE_CAP;
                        self.winger_pinch_samples.drain(0..overflow);
                    }
                }
            } else {
                i += 1;
            }
        }
        if tick % WINGER_PINCH_SAMPLE_INTERVAL_TICKS != 0 {
            return;
        }
        let wingers: Vec<usize> = snapshot
            .players
            .iter()
            .filter(|p| snapshot.is_wide_attacker(p))
            .map(|p| p.id)
            .collect();
        for id in wingers {
            let Some(player) = snapshot.players.iter().find(|p| p.id == id) else {
                continue;
            };
            let Some(inputs) = snapshot.build_winger_pinch_inputs(player) else {
                continue;
            };
            // The pinch-appetite action taken this tick (head once trained, else analytic seed).
            let action_bias = snapshot
                .winger_pinch_bias(&inputs)
                .unwrap_or_else(|| analytic_winger_pinch_bias(&inputs));
            let territorial = territorial_advantage(snapshot, player.team);
            if territorial.is_finite() {
                self.pending_winger_pinch.push(PendingWingerPinchDecision {
                    team: player.team,
                    player_id: id,
                    inputs,
                    action_bias,
                    decision_territorial: territorial,
                    reward_accum: 0.0,
                    due_tick: tick + WINGER_PINCH_REWARD_WINDOW_TICKS,
                });
            }
        }
    }

    pub fn drain_winger_pinch_samples(&mut self) -> Vec<WingerPinchSample> {
        std::mem::take(&mut self.winger_pinch_samples)
    }

    pub fn set_winger_pinch_head(&mut self, head: WingerPinchHead) {
        self.winger_pinch_head = Some(std::sync::Arc::new(head));
    }
}

#[cfg(test)]
mod winger_pinch_model_tests {
    use super::*;

    fn baseline_inputs() -> WingerPinchInputs {
        WingerPinchInputs {
            ball_on_my_flank: 0.0,
            crossing_position_on: 1.0,
            depth_frac: 0.7,
            box_congestion: 0.2,
            back_post_offside: 0.0,
            back_post_space: 0.6,
            half_space_space: 0.5,
            dist_to_ball: 0.6,
        }
    }

    #[test]
    fn features_are_finite_bounded_and_right_dim() {
        let f = baseline_inputs().to_features();
        assert_eq!(f.len(), WINGER_PINCH_FEATURE_DIM);
        for v in f {
            assert!(v.is_finite() && (0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn analytic_bias_is_a_valid_signed_unit() {
        let b = analytic_winger_pinch_bias(&baseline_inputs());
        assert!((-1.0..=1.0).contains(&b));
    }

    #[test]
    fn weak_side_deep_cross_with_space_biases_toward_pinching() {
        let b = analytic_winger_pinch_bias(&baseline_inputs());
        assert!(b > 0.0, "a weak-side winger with a deep cross + space should pinch, got {b}");
    }

    #[test]
    fn ball_on_my_flank_biases_toward_holding_width() {
        let mut i = baseline_inputs();
        i.ball_on_my_flank = 1.0;
        assert!(
            analytic_winger_pinch_bias(&i) < 0.0,
            "the ball-side winger is the deliverer and should hold width"
        );
    }

    #[test]
    fn head_predicts_a_bounded_signed_bias() {
        let head = WingerPinchHead::new(3);
        let b = head.predict(&baseline_inputs()).expect("finite");
        assert!((-1.0..=1.0).contains(&b));
    }

    #[test]
    fn reward_weighted_training_steers_toward_the_high_reward_action() {
        let mut head = WingerPinchHead::new(9);
        let inputs = baseline_inputs();
        let before = head.predict(&inputs).expect("finite");
        let samples: Vec<WingerPinchSample> = (0..32)
            .flat_map(|_| {
                [
                    WingerPinchSample { inputs: inputs.clone(), action_bias: 0.8, reward: 1.0 },
                    WingerPinchSample { inputs: inputs.clone(), action_bias: -0.8, reward: -1.0 },
                ]
            })
            .collect();
        for _ in 0..80 {
            head.train_reward_weighted(&samples, 0.05);
        }
        let after = head.predict(&inputs).expect("finite");
        assert!(after > before && after > 0.0);
    }

    #[test]
    fn train_entry_skips_empty_and_trains_valid() {
        assert!(train_winger_pinch_head(&[], 1, 4, 0.02).is_none());
        let samples: Vec<WingerPinchSample> = (0..16)
            .map(|i| WingerPinchSample {
                inputs: baseline_inputs(),
                action_bias: if i % 2 == 0 { 0.8 } else { -0.8 },
                reward: if i % 2 == 0 { 1.0 } else { -1.0 },
            })
            .collect();
        let (_, report) =
            train_winger_pinch_head(&samples, 1, 4, 0.02).expect("trains a usable corpus");
        assert_eq!(report.samples, 16);
        assert!(report.training_steps > 0 && report.final_loss.is_finite());
    }
}
