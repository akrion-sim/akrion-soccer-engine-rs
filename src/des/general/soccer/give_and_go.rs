//! Learnable **give-and-go / wall-pass** coordinator (the "lay it off and burst past the man
//! for the first-time return" head).
//!
//! When a carrier in the attacking half has a man to beat and a side-on teammate to combine
//! around, [`WorldSnapshot::wall_pass_option_for`] already recognises the one-two analytically
//! and scores its `quality`; the carrier's appetite to attempt it is then a fixed weighted sum
//! of hand-tuned multipliers (`WALL_PASS_BASE_APPETITE` × pressure × goal-proximity × …). This
//! module is the seam that makes that appetite **learnable** so the team can learn *which*
//! give-and-go to commit to and *when* — the multi-agent (MAPPO/CTDE) piece, because the two
//! cooperating agents (the carrier-turned-runner and the wall) only earn the shared
//! [`super::SoccerMatch`]-level wall-pass reward when the combination actually breaks a line.
//!
//! ## Design (mirrors [`super::long_pass_run`])
//!
//! * [`GiveAndGoInputs`] — the per-decision state vector: the analytic `quality` determinants
//!   ([`wall_pass_option_for`] computes them per candidate) plus raw geometry, the carrier's
//!   forward stride (it becomes the runner — the "go"), and the centralized coordination signals
//!   (`competing_walls`, `competing_runners`). The exact ordering is the single source of truth
//!   ([`GiveAndGoInputs::to_features`]).
//! * [`analytic_give_and_go_invite`] — the deterministic, RNG-free seed == the exact `quality`
//!   blend [`wall_pass_option_for`] returns today. It is the live fallback and the bootstrap
//!   target the head imitates, then beats.
//! * [`GiveAndGoHead`] — a `FeedForwardNetwork` (`DIM → hidden → 1`, sigmoid) regressing the
//!   per-decision invite, trained reward-weighted from self-play [`GiveAndGoSample`]s (reward =
//!   the attacking team's windowed territorial-advantage change, so the head learns the one-two
//!   that actually advanced the ball — the shared MAPPO outcome).
//!
//! ## Parity
//!
//! Gated **off** by default ([`give_and_go_model_enabled`] /
//! `DD_SOCCER_ENABLE_LEARNED_GIVE_AND_GO`). With the gate off the appetite uses the analytic
//! seed unchanged and no features are built, so an unconfigured process is byte-identical and
//! zero-cost.

use super::*;
use crate::des::general::neural_network::{ActivationName, FeedForwardNetwork, RandomNetworkSpec};
use crate::des::general::prng::mulberry32;

/// Number of features in the give-and-go state vector. The exact ordering is the single source
/// of truth in [`GiveAndGoInputs::to_features`] — change it and bump this constant and any
/// persisted head.
pub const GIVE_AND_GO_FEATURE_DIM: usize = 14;
/// Hidden width of the learned regression head.
pub const GIVE_AND_GO_HIDDEN_UNITS: usize = 16;

/// Reference distance (yd) normalizing the yard-scale geometry features.
const REF_DISTANCE_YARDS: f64 = 20.0;
/// Reference speed (yd/s) normalizing the carrier's forward stride into the burst.
const REF_SPEED_YPS: f64 = 8.0;
/// Reference count normalizing the coordination-signal counts.
const REF_COUNT: f64 = 4.0;

/// How often (ticks) give-and-go RL decisions are sampled while the model is on.
pub const GIVE_AND_GO_SAMPLE_INTERVAL_TICKS: u64 = 4;
/// Window (ticks) over which a sampled decision's territorial reward is measured.
pub const GIVE_AND_GO_REWARD_WINDOW_TICKS: u64 = 36;
/// Cap on the rolling RL sample buffer (drained by the learner).
pub const GIVE_AND_GO_SAMPLE_CAP: usize = 8192;
/// Minimum training steps before a trained head is consumed live; below this the regression net
/// is too raw, so the appetite falls back to the analytic seed.
pub const GIVE_AND_GO_HEAD_MIN_TRAINING_STEPS: usize = 200;
/// Carrier forward stride (yd/s) above which the carrier is judged to have COMMITTED to bursting
/// forward off the give this tick (the RL action label proxy when the one-two is taken).
pub const GIVE_AND_GO_COMMIT_SPEED_YPS: f64 = 2.0;

const GIVE_AND_GO_MODEL_ENABLE_ENV: &str = "DD_SOCCER_ENABLE_LEARNED_GIVE_AND_GO";

fn give_and_go_env_flag(name: &str) -> bool {
    std::env::var(name)
        .map(|raw| {
            let v = raw.trim().to_ascii_lowercase();
            matches!(v.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

/// Whether the learned give-and-go appetite is consulted this process. Off (unset) ⇒ the
/// analytic appetite stands, so the one-two propensity is byte-identical.
pub fn give_and_go_model_enabled() -> bool {
    #[cfg(test)]
    {
        give_and_go_env_flag(GIVE_AND_GO_MODEL_ENABLE_ENV)
    }
    #[cfg(not(test))]
    {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| give_and_go_env_flag(GIVE_AND_GO_MODEL_ENABLE_ENV))
    }
}

/// Raw (un-normalized) per-decision state the give-and-go appetite is a function of. The first
/// five fields are the EXACT analytic `quality` determinants [`WorldSnapshot::wall_pass_option_for`]
/// blends today, so [`analytic_give_and_go_invite`] reproduces that `quality` bit-for-bit; the
/// remainder is extra context (raw geometry, the carrier's forward stride, the coordination
/// counts, and the analytic seed itself as a retained determinant) that the head can learn from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GiveAndGoInputs {
    /// Open space the led return run attacks (1 = lots of grass behind the man).
    pub run_space: f64,
    /// How beatable the man-to-beat is — closer ⇒ more beatable by combining around him (1 = on
    /// top of the carrier, 0 = at the trigger-range edge).
    pub man_beatable: f64,
    /// How free the wall is to turn the give around first-time (1 = lots of space).
    pub partner_open: f64,
    /// Soft backward-traffic risk on the give leg (carrier → wall), demotes a long/back give.
    pub give_backward_risk: f64,
    /// Soft backward-traffic risk on the return leg (wall → led run), demotes a backward return.
    pub return_backward_risk: f64,
    /// Lay-off distance carrier → wall (yd).
    pub give_dist: f64,
    /// Forward yards the burst run gains (carrier → return target).
    pub run_forward_gain: f64,
    /// Distance carrier → man-to-beat (yd).
    pub man_dist: f64,
    /// Goal proximity of the carrier (1 = on the opponent goal, 0 = ref distance away).
    pub goal_proximity: f64,
    /// Pressure on the carrier (1 = a defender right on top, 0 = free).
    pub carrier_pressure: f64,
    /// The carrier's forward stride toward the attacking goal this tick (yd/s) — the "go".
    pub runner_forward_speed: f64,
    /// How many OTHER viable walls the carrier could combine with (centralized coordination —
    /// having options makes the one-two safer to commit to).
    pub competing_walls: f64,
    /// How many teammates are also offering a forward run (coordination — don't all break).
    pub competing_runners: f64,
    /// The analytic invite (== `quality`) itself: the head refines, never discards, the seed.
    pub analytic_seed: f64,
}

impl GiveAndGoInputs {
    /// The normalized network input vector. **This ordering is the single source of truth** for
    /// the feature layout — change it and bump [`GIVE_AND_GO_FEATURE_DIM`] and any persisted head.
    pub fn to_features(&self) -> [f64; GIVE_AND_GO_FEATURE_DIM] {
        let dist = |d: f64| (d / REF_DISTANCE_YARDS).clamp(-2.0, 2.0);
        let unit = |u: f64| u.clamp(0.0, 1.0);
        [
            unit(self.run_space),
            unit(self.man_beatable),
            unit(self.partner_open),
            unit(self.give_backward_risk),
            unit(self.return_backward_risk),
            dist(self.give_dist),
            dist(self.run_forward_gain),
            dist(self.man_dist),
            unit(self.goal_proximity),
            unit(self.carrier_pressure),
            (self.runner_forward_speed / REF_SPEED_YPS).clamp(-2.0, 2.0),
            (self.competing_walls / REF_COUNT).clamp(0.0, 2.0),
            (self.competing_runners / REF_COUNT).clamp(0.0, 2.0),
            unit(self.analytic_seed),
        ]
    }
}

/// Analytic, deterministic, RNG-free seed for the give-and-go invite in `[0, 1]`. **This is the
/// exact `quality` blend [`WorldSnapshot::wall_pass_option_for`] returns today**; keeping it here
/// as the single definition lets the head bootstrap toward it and the test suite assert parity.
pub fn analytic_give_and_go_invite(inputs: &GiveAndGoInputs) -> f64 {
    (inputs.run_space * 0.42 + inputs.man_beatable * 0.34 + inputs.partner_open * 0.24
        - inputs.give_backward_risk
        - inputs.return_backward_risk)
        .clamp(0.0, 1.0)
}

/// One reward-weighted RL training row for the give-and-go head: the per-decision state at a
/// one-two-decision tick, the ACTION taken (1 = the carrier committed the give-and-go, 0 = it
/// held / chose another option), and the territorial reward attributed over the following window.
/// Mirrors [`super::long_pass_run::LongPassRunSample`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GiveAndGoSample {
    pub inputs: GiveAndGoInputs,
    /// The action actually taken, in `[0, 1]` (1 = committed the give-and-go).
    pub action_commit: f64,
    /// Reward attributed over the following window (any scale; the trainer baselines it).
    pub reward: f64,
}

/// An open give-and-go decision awaiting its windowed reward. Recorded at the decision tick with
/// the attacking team's territorial state then; resolved `GIVE_AND_GO_REWARD_WINDOW_TICKS` later.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PendingGiveAndGoDecision {
    pub team: Team,
    pub player_id: usize,
    pub inputs: GiveAndGoInputs,
    pub action_commit: f64,
    /// Team territorial advantage captured at the decision tick (windowed baseline).
    pub decision_territorial: f64,
    pub due_tick: u64,
}

/// Learned regression head for the give-and-go invite: a `FeedForwardNetwork`
/// (`DIM → hidden → 1`, sigmoid). Round-trips to Postgres via the engine's
/// `SoccerNeuralNetworkSnapshot` path like the other learned heads.
#[derive(Clone, Debug)]
pub struct GiveAndGoHead {
    network: FeedForwardNetwork,
    training_steps: usize,
    last_loss: Option<f64>,
}

impl GiveAndGoHead {
    pub fn new(seed: u32) -> Self {
        let mut rng = mulberry32(seed ^ 0x6E1A_77C3);
        let network = FeedForwardNetwork::random(
            &RandomNetworkSpec {
                input_dim: GIVE_AND_GO_FEATURE_DIM,
                hidden_layers: vec![GIVE_AND_GO_HIDDEN_UNITS],
                output_dim: 1,
                hidden_activation: ActivationName::Tanh,
                output_activation: ActivationName::Sigmoid,
                weight_scale: None,
            },
            &mut rng,
        );
        GiveAndGoHead {
            network,
            training_steps: 0,
            last_loss: None,
        }
    }

    /// Predicted invite in `(0, 1)` for a captured decision state, or `None` on a malformed
    /// (mis-dimensioned / non-finite) input.
    pub fn predict(&self, inputs: &GiveAndGoInputs) -> Option<f64> {
        let features = inputs.to_features();
        if features.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let out = self.network.predict(&features[..]);
        out.first().copied().filter(|p| p.is_finite())
    }

    /// One supervised epoch regressing the head toward `(features, target_invite)` pairs
    /// (bootstrap toward the analytic seed). Returns the mean step loss.
    pub fn train(&mut self, samples: &[(GiveAndGoInputs, f64)], learning_rate: f64) -> f64 {
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
    /// ACTION (committed / held), weighted by how much its reward beat the batch baseline, so the
    /// head learns the one-two that actually advanced the ball. Mirrors
    /// [`super::long_pass_run::LongPassRunHead::train_reward_weighted`].
    pub fn train_reward_weighted(&mut self, samples: &[GiveAndGoSample], learning_rate: f64) -> f64 {
        let finite: Vec<&GiveAndGoSample> = samples
            .iter()
            .filter(|s| s.reward.is_finite() && s.action_commit.is_finite())
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
            let target = [s.action_commit.clamp(0.0, 1.0)];
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

/// Outcome of training a give-and-go head on a drained RL corpus, logged by the learning runner.
/// Mirrors [`super::long_pass_run::LongPassRunTrainingReport`].
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GiveAndGoTrainingReport {
    pub samples: usize,
    pub training_steps: usize,
    pub epochs: usize,
    pub final_loss: f64,
}

/// Load-and-train entry: given a drained RL corpus, build a fresh [`GiveAndGoHead`] and run
/// `epochs` reward-weighted-regression passes. `None` if no usable samples.
pub fn train_give_and_go_head(
    samples: &[GiveAndGoSample],
    seed: u32,
    epochs: usize,
    learning_rate: f64,
) -> Option<(GiveAndGoHead, GiveAndGoTrainingReport)> {
    let usable = samples
        .iter()
        .filter(|s| s.reward.is_finite() && s.action_commit.is_finite())
        .count();
    if usable == 0 || epochs == 0 {
        return None;
    }
    let mut head = GiveAndGoHead::new(seed);
    let mut final_loss = 0.0;
    for _ in 0..epochs {
        final_loss = head.train_reward_weighted(samples, learning_rate);
    }
    let report = GiveAndGoTrainingReport {
        samples: usable,
        training_steps: head.training_steps(),
        epochs,
        final_loss,
    };
    Some((head, report))
}

/// Public wrapper returning ONLY the report (the head type is engine-internal). The learning
/// runner uses this to train the drained corpus and log that it learns.
pub fn report_give_and_go_training(
    samples: &[GiveAndGoSample],
    seed: u32,
    epochs: usize,
    learning_rate: f64,
) -> Option<GiveAndGoTrainingReport> {
    train_give_and_go_head(samples, seed, epochs, learning_rate).map(|(_, report)| report)
}

impl SoccerMatch {
    /// Gated per-tick RL sample collection for the give-and-go head. A **no-op** (zero cost,
    /// byte-identical) unless the model is enabled. When on: resolves decisions whose reward
    /// window has elapsed (reward = the attacking team's territorial-advantage change over the
    /// window), and samples the ball-holder's give-and-go decision on cadence whenever a wall-pass
    /// option is available.
    pub(crate) fn collect_give_and_go_rl_samples(&mut self, snapshot: &WorldSnapshot) {
        if !give_and_go_model_enabled() {
            return;
        }
        let tick = self.tick;
        // Resolve due decisions.
        let mut i = 0;
        while i < self.pending_give_and_go.len() {
            if self.pending_give_and_go[i].due_tick <= tick {
                let decision = self.pending_give_and_go.swap_remove(i);
                let now_territorial = territorial_advantage(snapshot, decision.team);
                if now_territorial.is_finite() && decision.decision_territorial.is_finite() {
                    self.give_and_go_samples.push(GiveAndGoSample {
                        inputs: decision.inputs,
                        action_commit: decision.action_commit,
                        reward: now_territorial - decision.decision_territorial,
                    });
                    if self.give_and_go_samples.len() > GIVE_AND_GO_SAMPLE_CAP {
                        let overflow = self.give_and_go_samples.len() - GIVE_AND_GO_SAMPLE_CAP;
                        self.give_and_go_samples.drain(0..overflow);
                    }
                }
            } else {
                i += 1;
            }
        }
        // Sample the holder's give-and-go decision on cadence whenever a wall-pass is on.
        let Some(holder_id) = snapshot.ball.holder else {
            return;
        };
        if tick % GIVE_AND_GO_SAMPLE_INTERVAL_TICKS != 0 {
            return;
        }
        let Some(inputs) = snapshot.build_give_and_go_inputs(holder_id) else {
            return;
        };
        let Some(holder) = snapshot.players.iter().find(|p| p.id == holder_id) else {
            return;
        };
        // The action actually taken this tick: did the carrier commit to the give-and-go (it laid
        // the ball off and is bursting forward — its `one_two` is set to a wall partner)?
        let action_commit = if holder.one_two.is_some()
            || inputs.runner_forward_speed >= GIVE_AND_GO_COMMIT_SPEED_YPS
        {
            1.0
        } else {
            0.0
        };
        let territorial = territorial_advantage(snapshot, holder.team);
        if territorial.is_finite() {
            self.pending_give_and_go.push(PendingGiveAndGoDecision {
                team: holder.team,
                player_id: holder_id,
                inputs,
                action_commit,
                decision_territorial: territorial,
                due_tick: tick + GIVE_AND_GO_REWARD_WINDOW_TICKS,
            });
        }
    }

    /// Drain the collected give-and-go RL samples for the cluster learner (train + persist).
    pub fn drain_give_and_go_samples(&mut self) -> Vec<GiveAndGoSample> {
        std::mem::take(&mut self.give_and_go_samples)
    }

    /// Install the trained give-and-go head for live consumption. The learner carries + trains it
    /// across games and re-installs it each game; wrapped in `Arc` so every per-tick snapshot
    /// shares it cheaply.
    pub fn set_give_and_go_head(&mut self, head: GiveAndGoHead) {
        self.give_and_go_head = Some(std::sync::Arc::new(head));
    }
}

#[cfg(test)]
mod give_and_go_tests {
    use super::*;

    fn baseline_inputs() -> GiveAndGoInputs {
        let mut inputs = GiveAndGoInputs {
            run_space: 0.7,
            man_beatable: 0.6,
            partner_open: 0.5,
            give_backward_risk: 0.05,
            return_backward_risk: 0.05,
            give_dist: 9.0,
            run_forward_gain: 12.0,
            man_dist: 5.0,
            goal_proximity: 0.5,
            carrier_pressure: 0.6,
            runner_forward_speed: 3.0,
            competing_walls: 1.0,
            competing_runners: 1.0,
            analytic_seed: 0.0,
        };
        inputs.analytic_seed = analytic_give_and_go_invite(&inputs);
        inputs
    }

    #[test]
    fn features_are_finite_bounded_and_right_dim() {
        let f = baseline_inputs().to_features();
        assert_eq!(f.len(), GIVE_AND_GO_FEATURE_DIM);
        for v in f {
            assert!(v.is_finite());
            assert!(v.abs() <= 2.0 + 1e-9, "feature {v} out of normalized range");
        }
    }

    #[test]
    fn analytic_invite_is_a_valid_probability() {
        let p = analytic_give_and_go_invite(&baseline_inputs());
        assert!((0.0..=1.0).contains(&p), "invite {p} out of [0,1]");
    }

    #[test]
    fn analytic_invite_matches_the_wall_pass_quality_blend() {
        // The seed MUST equal the exact quality blend `wall_pass_option_for` returns, so the
        // gate-off path is byte-identical.
        let i = baseline_inputs();
        let expected = (i.run_space * 0.42 + i.man_beatable * 0.34 + i.partner_open * 0.24
            - i.give_backward_risk
            - i.return_backward_risk)
            .clamp(0.0, 1.0);
        assert_eq!(analytic_give_and_go_invite(&i), expected);
    }

    #[test]
    fn more_run_space_and_a_beatable_man_invite_more_than_a_blocked_one() {
        let good = baseline_inputs();
        let mut poor = baseline_inputs();
        poor.run_space = 0.05;
        poor.man_beatable = 0.05;
        poor.partner_open = 0.05;
        assert!(
            analytic_give_and_go_invite(&good) > analytic_give_and_go_invite(&poor),
            "open grass + a beatable man should invite the one-two more strongly"
        );
    }

    #[test]
    fn backward_traffic_demotes_the_invite() {
        let clean = baseline_inputs();
        let mut risky = baseline_inputs();
        risky.give_backward_risk = 0.3;
        risky.return_backward_risk = 0.3;
        assert!(
            analytic_give_and_go_invite(&risky) < analytic_give_and_go_invite(&clean),
            "backward give/return traffic should demote (never lift) the one-two"
        );
    }

    #[test]
    fn head_predicts_a_bounded_invite() {
        let head = GiveAndGoHead::new(13);
        let p = head.predict(&baseline_inputs()).expect("finite prediction");
        assert!((0.0..=1.0).contains(&p), "head invite {p} out of (0,1)");
    }

    #[test]
    fn reward_weighted_training_steers_toward_the_high_reward_give() {
        let mut head = GiveAndGoHead::new(31);
        let inputs = baseline_inputs();
        let before = head.predict(&inputs).expect("finite");
        // Same state, two competing actions: committing the give-and-go (1.0) is rewarded,
        // holding (0.0) penalized. RWR should pull the head toward making the one-two.
        let samples: Vec<GiveAndGoSample> = (0..32)
            .flat_map(|_| {
                [
                    GiveAndGoSample {
                        inputs,
                        action_commit: 1.0,
                        reward: 1.0,
                    },
                    GiveAndGoSample {
                        inputs,
                        action_commit: 0.0,
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
            "RWR should move the head toward the high-reward give: {before} -> {after}"
        );
    }

    #[test]
    fn train_entry_skips_empty_and_trains_valid() {
        assert!(train_give_and_go_head(&[], 1, 4, 0.02).is_none());
        let samples: Vec<GiveAndGoSample> = (0..16)
            .map(|_| GiveAndGoSample {
                inputs: baseline_inputs(),
                action_commit: 1.0,
                reward: 0.5,
            })
            .collect();
        let (head, report) =
            train_give_and_go_head(&samples, 3, 4, 0.02).expect("trains a usable corpus");
        assert_eq!(report.epochs, 4);
        assert!(report.samples > 0);
        assert!(head.training_steps() > 0);
    }

    #[test]
    fn model_is_gated_off_by_default() {
        // No env set in the default test process ⇒ the learned appetite is not consulted, so the
        // analytic seed stands (parity).
        assert!(!give_and_go_model_enabled());
    }
}
