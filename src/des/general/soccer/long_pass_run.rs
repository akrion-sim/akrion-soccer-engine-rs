//! Learnable **long-pass option run** coordinator (the "make a forward run so the deep
//! carrier can pick you out" head).
//!
//! When a teammate holds/dribbles the ball **deep** (own half) under **low-to-mid
//! pressure** with time and space to strike a long ball,
//! [`WorldSnapshot::backfield_long_pass_run_invite_for`] scores — per attacker — how strongly
//! that forward / attacking-midfielder should break into onside space ahead to BECOME a
//! receivable long-pass target. Today that invite is a fixed weighted sum of hand-tuned
//! "fits" (backfield depth, holder pressure, holder space, dribble room, run room, lane).
//! This module is the seam that makes the per-runner invite **learnable** so the team can
//! learn *which* attacker should make the run and *when* — the multi-agent (MAPPO/CTDE)
//! piece the carrier needs to have a real receiver instead of hoofing a long ball to nobody.
//!
//! ## Design (mirrors [`super::loose_ball_commit`])
//!
//! * [`LongPassRunInputs`] — the per-runner state vector (the analytic determinants plus the
//!   raw geometry, the runner's forward stride, and how many OTHER attackers are also offering
//!   a run — the centralized coordination signal). The exact ordering is the single source of
//!   truth ([`LongPassRunInputs::to_features`]).
//! * [`analytic_long_pass_run_invite`] — the deterministic, RNG-free seed == the current
//!   fixed-weight invite. It is the live fallback and the bootstrap target the head imitates,
//!   then beats.
//! * [`LongPassRunHead`] — a `FeedForwardNetwork` (`DIM → hidden → 1`, sigmoid) regressing the
//!   per-runner invite, trained reward-weighted from self-play [`LongPassRunSample`]s (reward =
//!   the running team's windowed territorial-advantage change, so the head learns the run that
//!   actually advanced the ball).
//!
//! ## Parity
//!
//! Gated **off** by default ([`long_pass_run_model_enabled`] /
//! `DD_SOCCER_ENABLE_LEARNED_LONG_PASS_RUN`). With the gate off the invite returns the analytic
//! seed unchanged and no features are built, so an unconfigured process is byte-identical and
//! zero-cost.

use super::*;
use crate::des::general::neural_network::{ActivationName, FeedForwardNetwork, RandomNetworkSpec};
use crate::des::general::prng::mulberry32;

/// Number of features in the long-pass run state vector. The exact ordering is the single
/// source of truth in [`LongPassRunInputs::to_features`] — change it and bump this constant
/// and any persisted head.
pub const LONG_PASS_RUN_FEATURE_DIM: usize = 15;
/// Hidden width of the learned regression head.
pub const LONG_PASS_RUN_HIDDEN_UNITS: usize = 18;

/// Reference distance (yd) normalizing the yard-scale geometry features.
const REF_DISTANCE_YARDS: f64 = 30.0;
/// Reference speed (yd/s) normalizing the runner's forward stride.
const REF_SPEED_YPS: f64 = 8.0;
/// Reference count normalizing the "other attackers offering a run" coordination signal.
const REF_RUNNER_COUNT: f64 = 4.0;

/// How often (ticks) long-pass run RL decisions are sampled while the model is on.
pub const LONG_PASS_RUN_SAMPLE_INTERVAL_TICKS: u64 = 4;
/// Window (ticks) over which a sampled decision's territorial reward is measured.
pub const LONG_PASS_RUN_REWARD_WINDOW_TICKS: u64 = 36;
/// Cap on the rolling RL sample buffer (drained by the learner).
pub const LONG_PASS_RUN_SAMPLE_CAP: usize = 8192;
/// Minimum training steps before a trained head is consumed live; below this the regression
/// net is too raw, so the invite falls back to the analytic seed.
pub const LONG_PASS_RUN_HEAD_MIN_TRAINING_STEPS: usize = 200;
/// Runner forward stride (yd/s) above which the runner is judged to have COMMITTED to the
/// forward run this tick (the RL action label).
pub const LONG_PASS_RUN_COMMIT_SPEED_YPS: f64 = 2.0;

const LONG_PASS_RUN_MODEL_ENABLE_ENV: &str = "DD_SOCCER_ENABLE_LEARNED_LONG_PASS_RUN";

fn long_pass_run_env_flag(name: &str) -> bool {
    std::env::var(name)
        .map(|raw| {
            let v = raw.trim().to_ascii_lowercase();
            matches!(v.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

/// Whether the learned long-pass run invite is consulted this process. Off (unset) ⇒ the
/// analytic invite stands, so the run coordination is byte-identical.
pub fn long_pass_run_model_enabled() -> bool {
    #[cfg(test)]
    {
        long_pass_run_env_flag(LONG_PASS_RUN_MODEL_ENABLE_ENV)
    }
    #[cfg(not(test))]
    {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| long_pass_run_env_flag(LONG_PASS_RUN_MODEL_ENABLE_ENV))
    }
}

/// Raw (un-normalized) per-runner state the long-pass run invite is a function of. The first
/// seven fields are the EXACT analytic determinants the live invite sums today, so
/// [`analytic_long_pass_run_invite`] reproduces the current value bit-for-bit; the remainder
/// is extra context (raw geometry, the runner's stride, the coordination count, and the
/// analytic seed itself as a retained determinant) that the head can learn from. A plain
/// struct so a row is human- and test-readable.
#[derive(Clone, Debug, PartialEq)]
pub struct LongPassRunInputs {
    /// How deep in our half the holder is (1 = very deep, 0 = at the midfield margin).
    pub backfield_fit: f64,
    /// How unpressured the holder is (1 = free, 0 = at the max pressure threshold).
    pub pressure_fit: f64,
    /// How much open space the holder has to the nearest opponent (1 = lots).
    pub space_fit: f64,
    /// How much clear forward dribble room the holder has (1 = lots).
    pub dribble_space_fit: f64,
    /// How much ground the run gains toward the onside line (1 ≈ 20yd+).
    pub room_fit: f64,
    /// Lane clearance from holder to the run target (1 = clean, 0.55 = sighted, 0 = blocked).
    pub lane_fit: f64,
    /// Role weight (1.0 forward, 0.86 attacking midfielder).
    pub role_fit: f64,
    /// Forward yards from the holder to the runner (the runner is already ahead).
    pub forward_from_holder: f64,
    /// Forward yards the run itself gains (runner → onside run target).
    pub run_gain: f64,
    /// Forward yards from the holder to the run target (the outlet length).
    pub outlet_gain: f64,
    /// Raw holder pressure (0 = free).
    pub holder_pressure: f64,
    /// Raw nearest-opponent distance to the holder (yd).
    pub nearest_holder_opp: f64,
    /// Runner's forward stride toward the attacking goal this tick (yd/s).
    pub runner_forward_speed: f64,
    /// How many OTHER attackers are also offering a forward run for this holder (the
    /// centralized coordination signal — don't have everyone make the same run).
    pub competing_runners: f64,
    /// The analytic invite value itself (retained determinant — the head refines, never
    /// discards, the well-tuned seed).
    pub analytic_seed: f64,
}

impl LongPassRunInputs {
    /// The normalized network input vector. **This ordering is the single source of truth**
    /// for the feature layout — change it and bump [`LONG_PASS_RUN_FEATURE_DIM`] and any
    /// persisted head.
    pub fn to_features(&self) -> [f64; LONG_PASS_RUN_FEATURE_DIM] {
        let dist = |d: f64| (d / REF_DISTANCE_YARDS).clamp(-2.0, 2.0);
        let unit = |u: f64| u.clamp(0.0, 1.0);
        [
            unit(self.backfield_fit),
            unit(self.pressure_fit),
            unit(self.space_fit),
            unit(self.dribble_space_fit),
            unit(self.room_fit),
            unit(self.lane_fit),
            unit(self.role_fit),
            dist(self.forward_from_holder),
            dist(self.run_gain),
            dist(self.outlet_gain),
            unit(self.holder_pressure),
            dist(self.nearest_holder_opp),
            (self.runner_forward_speed / REF_SPEED_YPS).clamp(-2.0, 2.0),
            (self.competing_runners / REF_RUNNER_COUNT).clamp(0.0, 2.0),
            unit(self.analytic_seed),
        ]
    }
}

/// Analytic, deterministic, RNG-free seed for the per-runner long-pass run invite in
/// `[0, 1]`. **This is the exact weighted sum the live invite returns today** (see
/// [`WorldSnapshot::backfield_long_pass_run_invite_for`]); keeping it here as the single
/// definition lets the head bootstrap toward it and the test suite assert parity.
pub fn analytic_long_pass_run_invite(inputs: &LongPassRunInputs) -> f64 {
    (inputs.role_fit
        * (inputs.backfield_fit * 0.16
            + inputs.pressure_fit * 0.22
            + inputs.space_fit * 0.18
            + inputs.dribble_space_fit * 0.14
            + inputs.room_fit * 0.18
            + inputs.lane_fit * 0.12))
        .clamp(0.0, 1.0)
}

/// One reward-weighted RL training row for the long-pass run head: the per-runner state at a
/// run-decision tick, the run ACTION taken (1 = the runner broke forward, 0 = it held), and
/// the territorial reward attributed over the following window. Mirrors
/// [`super::loose_ball_commit::LooseBallCommitSample`].
#[derive(Clone, Debug, PartialEq)]
pub struct LongPassRunSample {
    pub inputs: LongPassRunInputs,
    /// The run action actually taken, in `[0, 1]` (1 = broke forward into the run).
    pub action_run: f64,
    /// Reward attributed over the following window (any scale; the trainer baselines it).
    pub reward: f64,
}

/// An open long-pass run decision awaiting its windowed reward. Recorded at the decision tick
/// with the running team's territorial state then; resolved
/// `LONG_PASS_RUN_REWARD_WINDOW_TICKS` later.
#[derive(Clone, Debug, PartialEq)]
pub struct PendingLongPassRunDecision {
    pub team: Team,
    pub player_id: usize,
    pub inputs: LongPassRunInputs,
    pub action_run: f64,
    /// Team territorial advantage captured at the decision tick (windowed baseline).
    pub decision_territorial: f64,
    pub due_tick: u64,
}

/// Learned regression head for the per-runner long-pass run invite: a `FeedForwardNetwork`
/// (`DIM → hidden → 1`, sigmoid). Round-trips to Postgres via the engine's
/// `SoccerNeuralNetworkSnapshot` path like the other learned heads.
#[derive(Clone, Debug)]
pub struct LongPassRunHead {
    network: FeedForwardNetwork,
    training_steps: usize,
    last_loss: Option<f64>,
}

impl LongPassRunHead {
    pub fn new(seed: u32) -> Self {
        let mut rng = mulberry32(seed ^ 0x51A6_3D7B);
        let network = FeedForwardNetwork::random(
            &RandomNetworkSpec {
                input_dim: LONG_PASS_RUN_FEATURE_DIM,
                hidden_layers: vec![LONG_PASS_RUN_HIDDEN_UNITS],
                output_dim: 1,
                hidden_activation: ActivationName::Tanh,
                output_activation: ActivationName::Sigmoid,
                weight_scale: None,
            },
            &mut rng,
        );
        LongPassRunHead {
            network,
            training_steps: 0,
            last_loss: None,
        }
    }

    /// Predicted invite in `(0, 1)` for a captured runner state, or `None` on a malformed
    /// (mis-dimensioned / non-finite) input.
    pub fn predict(&self, inputs: &LongPassRunInputs) -> Option<f64> {
        let features = inputs.to_features();
        if features.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let out = self.network.predict(&features[..]);
        out.first().copied().filter(|p| p.is_finite())
    }

    /// One supervised epoch regressing the head toward `(features, target_invite)` pairs
    /// (bootstrap toward the analytic seed). Returns the mean step loss.
    pub fn train(&mut self, samples: &[(LongPassRunInputs, f64)], learning_rate: f64) -> f64 {
        let mut total = 0.0;
        let mut applied = 0usize;
        for (inputs, target) in samples {
            let features = inputs.to_features();
            if features.iter().any(|v| !v.is_finite()) || !target.is_finite() {
                continue;
            }
            let target = [target.clamp(0.0, 1.0)];
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

    /// One **reward-weighted-regression** epoch over RL samples: regress toward each sample's
    /// ACTION (ran / held), weighted by how much its reward beat the batch baseline, so the
    /// head learns the run pattern that actually advanced the ball. Mirrors
    /// [`super::loose_ball_commit::LooseBallCommitHead::train_reward_weighted`].
    pub fn train_reward_weighted(
        &mut self,
        samples: &[LongPassRunSample],
        learning_rate: f64,
    ) -> f64 {
        let finite: Vec<&LongPassRunSample> = samples
            .iter()
            .filter(|s| s.reward.is_finite() && s.action_run.is_finite())
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
            let target = [s.action_run.clamp(0.0, 1.0)];
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

/// Outcome of training a long-pass run head on a drained RL corpus, logged by the learning
/// runner. Mirrors [`super::loose_ball_commit::LooseBallCommitTrainingReport`].
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LongPassRunTrainingReport {
    pub samples: usize,
    pub training_steps: usize,
    pub epochs: usize,
    pub final_loss: f64,
}

/// Load-and-train entry: given a drained RL corpus, build a fresh [`LongPassRunHead`] and run
/// `epochs` reward-weighted-regression passes. `None` if no usable samples.
pub fn train_long_pass_run_head(
    samples: &[LongPassRunSample],
    seed: u32,
    epochs: usize,
    learning_rate: f64,
) -> Option<(LongPassRunHead, LongPassRunTrainingReport)> {
    let usable = samples
        .iter()
        .filter(|s| s.reward.is_finite() && s.action_run.is_finite())
        .count();
    if usable == 0 || epochs == 0 {
        return None;
    }
    let mut head = LongPassRunHead::new(seed);
    let mut final_loss = 0.0;
    for _ in 0..epochs {
        final_loss = head.train_reward_weighted(samples, learning_rate);
    }
    let report = LongPassRunTrainingReport {
        samples: usable,
        training_steps: head.training_steps(),
        epochs,
        final_loss,
    };
    Some((head, report))
}

/// Public wrapper returning ONLY the report (the head type is engine-internal). The learning
/// runner uses this to train the drained corpus and log that it learns.
pub fn report_long_pass_run_training(
    samples: &[LongPassRunSample],
    seed: u32,
    epochs: usize,
    learning_rate: f64,
) -> Option<LongPassRunTrainingReport> {
    train_long_pass_run_head(samples, seed, epochs, learning_rate).map(|(_, report)| report)
}

impl SoccerMatch {
    /// Gated per-tick RL sample collection for the long-pass run head. A **no-op** (zero cost,
    /// byte-identical) unless the model is enabled. When on: resolves decisions whose reward
    /// window has elapsed (reward = the running team's territorial-advantage change over the
    /// window), and samples fresh per-runner decisions on cadence whenever a teammate holds the
    /// ball deep enough to invite a run.
    pub(crate) fn collect_long_pass_run_rl_samples(&mut self, snapshot: &WorldSnapshot) {
        if !long_pass_run_model_enabled() {
            return;
        }
        let tick = self.tick;
        // Resolve due decisions.
        let mut i = 0;
        while i < self.pending_long_pass_run.len() {
            if self.pending_long_pass_run[i].due_tick <= tick {
                let decision = self.pending_long_pass_run.swap_remove(i);
                let now_territorial = territorial_advantage(snapshot, decision.team);
                if now_territorial.is_finite() && decision.decision_territorial.is_finite() {
                    self.long_pass_run_samples.push(LongPassRunSample {
                        inputs: decision.inputs,
                        action_run: decision.action_run,
                        reward: now_territorial - decision.decision_territorial,
                    });
                    if self.long_pass_run_samples.len() > LONG_PASS_RUN_SAMPLE_CAP {
                        let overflow = self.long_pass_run_samples.len() - LONG_PASS_RUN_SAMPLE_CAP;
                        self.long_pass_run_samples.drain(0..overflow);
                    }
                }
            } else {
                i += 1;
            }
        }
        // Sample fresh per-runner decisions on cadence while a teammate holds the ball.
        let Some(holder_id) = snapshot.ball.holder else {
            return;
        };
        if tick % LONG_PASS_RUN_SAMPLE_INTERVAL_TICKS != 0 {
            return;
        }
        let candidate_ids: Vec<usize> = snapshot
            .players
            .iter()
            .filter(|p| {
                p.id != holder_id
                    && (p.role == PlayerRole::Forward
                        || (p.role == PlayerRole::Midfielder
                            && p.preferences.offensive_mindedness
                                >= p.preferences.defensive_mindedness))
            })
            .map(|p| p.id)
            .collect();
        for id in candidate_ids {
            let Some(inputs) = snapshot.build_long_pass_run_inputs(id) else {
                continue;
            };
            let Some(runner) = snapshot.players.iter().find(|p| p.id == id) else {
                continue;
            };
            // The action actually taken this tick: did the runner break forward (a real
            // forward stride toward the attacking goal)?
            let action_run = if inputs.runner_forward_speed >= LONG_PASS_RUN_COMMIT_SPEED_YPS {
                1.0
            } else {
                0.0
            };
            let territorial = territorial_advantage(snapshot, runner.team);
            if territorial.is_finite() {
                self.pending_long_pass_run.push(PendingLongPassRunDecision {
                    team: runner.team,
                    player_id: id,
                    inputs,
                    action_run,
                    decision_territorial: territorial,
                    due_tick: tick + LONG_PASS_RUN_REWARD_WINDOW_TICKS,
                });
            }
        }
    }

    /// Drain the collected long-pass run RL samples for the cluster learner (train + persist).
    pub fn drain_long_pass_run_samples(&mut self) -> Vec<LongPassRunSample> {
        std::mem::take(&mut self.long_pass_run_samples)
    }

    /// Install the trained long-pass run head for live consumption. The learner carries +
    /// trains it across games and re-installs it each game; wrapped in `Arc` so every per-tick
    /// snapshot shares it cheaply.
    pub fn set_long_pass_run_head(&mut self, head: LongPassRunHead) {
        self.long_pass_run_head = Some(std::sync::Arc::new(head));
    }
}

#[cfg(test)]
mod long_pass_run_tests {
    use super::*;

    fn baseline_inputs() -> LongPassRunInputs {
        let mut inputs = LongPassRunInputs {
            backfield_fit: 0.6,
            pressure_fit: 0.7,
            space_fit: 0.6,
            dribble_space_fit: 0.5,
            room_fit: 0.7,
            lane_fit: 1.0,
            role_fit: 1.0,
            forward_from_holder: 22.0,
            run_gain: 14.0,
            outlet_gain: 30.0,
            holder_pressure: 0.2,
            nearest_holder_opp: 9.0,
            runner_forward_speed: 4.0,
            competing_runners: 1.0,
            analytic_seed: 0.0,
        };
        inputs.analytic_seed = analytic_long_pass_run_invite(&inputs);
        inputs
    }

    #[test]
    fn features_are_finite_bounded_and_right_dim() {
        let f = baseline_inputs().to_features();
        assert_eq!(f.len(), LONG_PASS_RUN_FEATURE_DIM);
        for v in f {
            assert!(v.is_finite());
            assert!(v.abs() <= 2.0 + 1e-9, "feature {v} out of normalized range");
        }
    }

    #[test]
    fn analytic_invite_is_a_valid_probability() {
        let p = analytic_long_pass_run_invite(&baseline_inputs());
        assert!((0.0..=1.0).contains(&p), "invite {p} out of [0,1]");
    }

    #[test]
    fn analytic_invite_matches_the_historical_weighted_sum() {
        // The analytic seed MUST equal the exact inline sum the live invite used before this
        // head existed, so the gate-off path is byte-identical.
        let i = baseline_inputs();
        let expected = (i.role_fit
            * (i.backfield_fit * 0.16
                + i.pressure_fit * 0.22
                + i.space_fit * 0.18
                + i.dribble_space_fit * 0.14
                + i.room_fit * 0.18
                + i.lane_fit * 0.12))
            .clamp(0.0, 1.0);
        assert_eq!(analytic_long_pass_run_invite(&i), expected);
    }

    #[test]
    fn unpressured_holder_with_room_invites_more_than_a_pressed_cramped_one() {
        let good = baseline_inputs();
        let mut poor = baseline_inputs();
        poor.pressure_fit = 0.05;
        poor.space_fit = 0.05;
        poor.room_fit = 0.05;
        assert!(
            analytic_long_pass_run_invite(&good) > analytic_long_pass_run_invite(&poor),
            "a calm holder with room should invite the run more strongly"
        );
    }

    #[test]
    fn attacking_midfielder_role_weight_trims_the_invite() {
        let forward = baseline_inputs();
        let mut amid = baseline_inputs();
        amid.role_fit = 0.86;
        assert!(
            analytic_long_pass_run_invite(&forward) > analytic_long_pass_run_invite(&amid),
            "the forward (role_fit 1.0) should invite at least as strongly as the att-mid (0.86)"
        );
    }

    #[test]
    fn head_predicts_a_bounded_invite() {
        let head = LongPassRunHead::new(11);
        let p = head.predict(&baseline_inputs()).expect("finite prediction");
        assert!((0.0..=1.0).contains(&p), "head invite {p} out of (0,1)");
    }

    #[test]
    fn reward_weighted_training_steers_toward_the_high_reward_run() {
        let mut head = LongPassRunHead::new(29);
        let inputs = baseline_inputs();
        let before = head.predict(&inputs).expect("finite");
        // Same state, two competing actions: running (1.0) is rewarded, holding (0.0)
        // penalized. RWR should pull the head toward making the run.
        let samples: Vec<LongPassRunSample> = (0..32)
            .flat_map(|_| {
                [
                    LongPassRunSample {
                        inputs: inputs.clone(),
                        action_run: 1.0,
                        reward: 1.0,
                    },
                    LongPassRunSample {
                        inputs: inputs.clone(),
                        action_run: 0.0,
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
            after > before && after > 0.5,
            "RWR should move the head toward the high-reward run: {before} -> {after}"
        );
    }

    #[test]
    fn train_entry_skips_empty_and_trains_valid() {
        assert!(train_long_pass_run_head(&[], 1, 4, 0.02).is_none());
        let samples: Vec<LongPassRunSample> = (0..16)
            .map(|_| LongPassRunSample {
                inputs: baseline_inputs(),
                action_run: 1.0,
                reward: 0.5,
            })
            .collect();
        let (head, report) =
            train_long_pass_run_head(&samples, 3, 4, 0.02).expect("trains a usable corpus");
        assert_eq!(report.epochs, 4);
        assert!(report.samples > 0);
        assert!(head.training_steps() > 0);
    }

    #[test]
    fn model_is_gated_off_by_default() {
        // No env set in the default test process ⇒ the learned invite is not consulted, so the
        // analytic seed stands (parity).
        assert!(!long_pass_run_model_enabled());
    }
}
