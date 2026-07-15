//! Learnable **per-context MPC reject threshold** — *how low* an action's predicted
//! probability of success has to be, in this exact situation, before MPC rejects the
//! POMDP's chosen action and kicks the decision back for the next-best one.
//!
//! The kickback machinery already exists: the POMDP ranks actions, MPC scores the
//! chosen one's execution probability `p` via the family execution-probability
//! functions, and [`SoccerMatch::learned_plan_needs_mpc_replan`] rejects it when
//! `p < threshold`, walking the ranking for a feasible replacement (see
//! [`SoccerMatch::mpc_reconciled_learned_plan`] and the receiver-level kickback in
//! [`WorldSnapshot::learned_pass_receiver_selection`]). What it lacked is the thing a
//! coach actually weighs: *"too low" is not a single number.* A 20 % pass is fine to
//! attempt with acres of space at 1-0 down; the same 20 % into a packed box, under
//! pressure, protecting a lead, is a giveaway you should refuse. The reject bar is a
//! function of the field vector, and until now it was a **global constant per family**
//! ([`LearnedMpcReplanThresholds`]).
//!
//! This module makes that bar **context-dependent and learnable**, mirroring
//! [`super::loose_ball_commit`]'s commit head:
//!
//!   * [`MpcRejectFamily`] — pass / dribble / shot, the action families MPC gates.
//!   * [`MpcRejectThresholdInputs`] — the per-decision context the bar is a function
//!     of, in the actor's attacking frame so it is mirror-invariant: which family,
//!     the pitch threat / EPV of the actor's position, how much pressure the nearest
//!     opponent applies, the actor's role, ball state, and the retained analytic base
//!     threshold for the family (group 8: refine, never discard).
//!   * [`analytic_mpc_reject_threshold`] — a deterministic, RNG-free seed that
//!     modulates the family base bar by context (raise it under pressure / low team
//!     control, relax it in space). It is the bootstrap target and the live fallback,
//!     so even before a head is trained the bar is already context-aware.
//!   * [`MpcRejectThresholdHead`] — a `FeedForwardNetwork` (`DIM → hidden → 1`,
//!     sigmoid) the cluster learner trains by **reward-weighted regression** toward
//!     the bar that best separated the commits that *worked out* from the ones that
//!     didn't: an above-baseline outcome after committing at probability `p` pulls the
//!     bar below `p` (keep attempting these), a below-baseline outcome pushes it above
//!     `p` (refuse them next time).
//!
//! The live seam is [`WorldSnapshot::learned_mpc_reject_threshold`], returning the
//! effective reject bar for a family in the current context.
//!
//! ## Parity
//!
//! Gated **off** by default (`DD_SOCCER_ENABLE_MPC_REJECT_THRESHOLD_MODEL`). When
//! unset, [`WorldSnapshot::learned_mpc_reject_threshold`] returns the pre-existing
//! constant base threshold *before building any features*, so the reject decision —
//! and therefore the whole engine — is byte-identical to before this module existed.
//! The feature build, analytic seed, and head are pure / RNG-free.
//!
//! ## Scope
//!
//! The live seam and automated RL corpus both cover **all three families** (pass /
//! dribble / shot): when the POMDP/MPC pipeline actually commits one of those actions,
//! that exact plan is scored by the same MPC execution-probability function used at
//! the live reject gate, then resolved against its windowed outcome. Hypothetical
//! alternatives are deliberately not labelled with the committed action's result.
//! Family is an explicit feature, so the learned bar can differ by action as well as
//! field context.

use super::*;
use crate::des::general::neural_network::{ActivationName, FeedForwardNetwork, RandomNetworkSpec};
use crate::des::general::prng::mulberry32;

/// Number of features in the reject-threshold context vector. The exact ordering is
/// the single source of truth in [`MpcRejectThresholdInputs::to_features`].
pub const MPC_REJECT_THRESHOLD_FEATURE_DIM: usize = 11;
/// Hidden width of the learned regression head.
pub const MPC_REJECT_THRESHOLD_HIDDEN_UNITS: usize = 16;

/// Absolute floor on any effective reject bar — below this we never refuse an action
/// (a valid probability, keeps the seam from ever gating everything off).
pub const MPC_REJECT_THRESHOLD_FLOOR: f64 = 0.02;
/// Absolute ceiling on any effective reject bar — a refined bar may raise the base
/// substantially in a hostile context but never demand near-certainty.
pub const MPC_REJECT_THRESHOLD_CEIL: f64 = 0.75;

/// Reference distance (yd) at which nearest-opponent pressure has decayed to `1/e`.
const PRESSURE_REF_YARDS: f64 = 4.5;
/// Reference speed (yd/s) normalizing ball speed.
const REF_BALL_SPEED_YPS: f64 = 20.0;
/// How far (in threshold units) context may move the analytic bar off its base.
const ANALYTIC_CONTEXT_SWING: f64 = 0.22;
/// How far a below/above-baseline outcome shifts the RWR target off the committed
/// probability when learning where the bar belongs.
const THRESHOLD_LEARN_MARGIN: f64 = 0.14;

/// Env gate enabling the learned reject-threshold at the MPC gate this process. Off
/// (unset) by default so an unconfigured process stays byte-identical.
const MPC_REJECT_THRESHOLD_MODEL_ENABLE_ENV: &str = "DD_SOCCER_ENABLE_MPC_REJECT_THRESHOLD_MODEL";

fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|raw| {
            let v = raw.trim().to_ascii_lowercase();
            matches!(v.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

/// Whether the learned reject-threshold is consulted at the MPC gate this process.
/// Off ⇒ the constant per-family base thresholds stand (parity).
pub fn mpc_reject_threshold_model_enabled() -> bool {
    #[cfg(test)]
    {
        env_flag_enabled(MPC_REJECT_THRESHOLD_MODEL_ENABLE_ENV)
    }
    #[cfg(not(test))]
    {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| env_flag_enabled(MPC_REJECT_THRESHOLD_MODEL_ENABLE_ENV))
    }
}

/// The action families whose MPC execution the reject bar gates. One-hot encoded into
/// the feature vector so a single head serves all three.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MpcRejectFamily {
    Pass,
    Dribble,
    Shot,
}

impl MpcRejectFamily {
    fn one_hot(self) -> (f64, f64, f64) {
        match self {
            MpcRejectFamily::Pass => (1.0, 0.0, 0.0),
            MpcRejectFamily::Dribble => (0.0, 1.0, 0.0),
            MpcRejectFamily::Shot => (0.0, 0.0, 1.0),
        }
    }
}

/// Classify a concrete soccer action label into the MPC execution family whose
/// contextual reject bar governs it. Non-technical hold/support/defensive actions do
/// not pass through this gate.
pub(crate) fn mpc_reject_family_for_action_label(label: &str) -> Option<MpcRejectFamily> {
    let normalized = normalize_soccer_action_label(label);
    if pass_like_action_flight(normalized).is_some() {
        Some(MpcRejectFamily::Pass)
    } else if is_dribble_action_label(normalized) {
        Some(MpcRejectFamily::Dribble)
    } else if matches!(
        normalized,
        "shoot" | "first-time-shot" | "first-time-header"
    ) {
        Some(MpcRejectFamily::Shot)
    } else {
        None
    }
}

/// Raw (un-normalized) per-decision context the reject bar is a function of, captured
/// in the actor's **attacking frame** so the bar is mirror-invariant. A plain struct
/// so it can be logged as a training row, asserted on in tests, and surfaced by the
/// inspector independently of the network.
#[derive(Clone, Debug, PartialEq)]
pub struct MpcRejectThresholdInputs {
    /// Which family MPC is gating.
    pub family: MpcRejectFamily,
    /// Pitch threat / EPV of the actor's position for the attacking team, in `[0, 1]`
    /// (higher = more dangerous area — a spot where refusing a marginal move costs a
    /// promising situation, so the bar should relax).
    pub actor_threat: f64,
    /// Nearest-opponent pressure on the actor in `[0, 1]` (1 = an opponent on top of
    /// them). Under pressure a marginal move is more likely a giveaway ⇒ raise the bar.
    pub nearest_opponent_pressure: f64,
    /// Probability the actor's team controls the actor's cell (pitch control).
    pub team_control_at_actor: f64,
    /// Ball speed (yd/s) — a moving/bouncing ball makes execution harder.
    pub ball_speed: f64,
    /// Ball altitude (yd).
    pub ball_altitude: f64,
    // Role one-hot (goalkeeper folds into the all-zero case).
    pub role_defender: f64,
    pub role_midfielder: f64,
    pub role_forward: f64,
    /// Group 8 retained determinant: the family's analytic base reject bar (the
    /// constant the global path uses). The model learns *relative to* it so the bar is
    /// refined, never discarded.
    pub base_threshold: f64,
}

impl MpcRejectThresholdInputs {
    /// The normalized network input vector. **This ordering is the single source of
    /// truth** for the feature layout — change it and bump
    /// [`MPC_REJECT_THRESHOLD_FEATURE_DIM`] and any persisted head.
    pub fn to_features(&self) -> [f64; MPC_REJECT_THRESHOLD_FEATURE_DIM] {
        let unit = |u: f64| u.clamp(0.0, 1.0);
        let (fam_pass, fam_dribble, fam_shot) = self.family.one_hot();
        [
            fam_pass,
            fam_dribble,
            fam_shot,
            unit(self.actor_threat),
            unit(self.nearest_opponent_pressure),
            unit(self.team_control_at_actor),
            (self.ball_speed / REF_BALL_SPEED_YPS).clamp(0.0, 2.0),
            (self.ball_altitude / 3.0).clamp(0.0, 2.0),
            self.role_defender,
            self.role_midfielder,
            self.role_forward,
        ]
    }
}

/// Logistic squash to `(0, 1)`.
fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

/// Analytic, deterministic, RNG-free seed for the **effective reject bar** in
/// `[FLOOR, CEIL]`. It is the bootstrap target and the live fallback the learned
/// [`MpcRejectThresholdHead`] first imitates, then beats.
///
/// The intent it encodes: start from the family's tuned base bar, then RAISE it when
/// the move is likely to be a giveaway — high opponent pressure, low team control of
/// the actor's cell — and RELAX it in space and in dangerous attacking areas where
/// refusing a marginal-but-promising move throws away the chance. The swing is modest
/// so the bar is refined around its well-tuned base, not overturned.
pub fn analytic_mpc_reject_threshold(inputs: &MpcRejectThresholdInputs) -> f64 {
    // Signed context term in roughly [-1.5, 1.5]: positive ⇒ raise the bar (refuse
    // marginal moves), negative ⇒ relax it (attempt them).
    let pressure = inputs.nearest_opponent_pressure.clamp(0.0, 1.0);
    let control = inputs.team_control_at_actor.clamp(0.0, 1.0);
    let threat = inputs.actor_threat.clamp(0.0, 1.0);
    let mut context = 0.0;
    // Under pressure a marginal execution is more likely a turnover: raise the bar.
    context += 1.3 * (pressure - 0.5);
    // Losing control of your own cell: raise the bar.
    context += 0.9 * (0.5 - control);
    // In a dangerous attacking spot, a refused marginal move wastes a real chance:
    // relax the bar.
    context -= 1.1 * (threat - 0.5);
    // A shot in a promising spot is the move you least want to talk yourself out of.
    if inputs.family == MpcRejectFamily::Shot {
        context -= 0.4 * threat;
    }
    let swing = ANALYTIC_CONTEXT_SWING * context.tanh();
    (inputs.base_threshold + swing).clamp(MPC_REJECT_THRESHOLD_FLOOR, MPC_REJECT_THRESHOLD_CEIL)
}

/// One reward-weighted RL training row for the reject-threshold head: the per-decision
/// context, the execution probability `p` at which the engine COMMITTED the action
/// (i.e. the bar was below `p`), and the reward attributed over the following window
/// (higher = committing at `p` in this context worked out). Mirrors
/// [`super::loose_ball_commit::LooseBallCommitSample`].
#[derive(Clone, Debug, PartialEq)]
pub struct MpcRejectThresholdSample {
    pub inputs: MpcRejectThresholdInputs,
    /// The execution probability the action was committed at, in `[0, 1]`.
    pub committed_probability: f64,
    /// Reward attributed to the decision over the following window (any scale; only
    /// relative-to-batch matters — the trainer baselines it).
    pub reward: f64,
}

/// An open reject-threshold decision awaiting its windowed reward. Recorded at the
/// decision tick with the deciding team's territorial state then; resolved
/// `MPC_REJECT_THRESHOLD_REWARD_WINDOW_TICKS` later.
#[derive(Clone, Debug, PartialEq)]
pub struct PendingMpcRejectThresholdDecision {
    pub team: Team,
    pub inputs: MpcRejectThresholdInputs,
    pub committed_probability: f64,
    /// Team territorial advantage captured at the decision tick (windowed baseline).
    pub decision_territorial: f64,
    pub due_tick: u64,
}

/// Window (ticks) over which a sampled decision's territorial reward is measured.
pub const MPC_REJECT_THRESHOLD_REWARD_WINDOW_TICKS: u64 = 30;
/// Cap on the rolling RL sample buffer (drained by the learner).
pub const MPC_REJECT_THRESHOLD_SAMPLE_CAP: usize = 8192;
/// Minimum training steps before a trained head is consumed live; below this the
/// regression net is too raw, so the gate falls back to the analytic seed.
pub const MPC_REJECT_THRESHOLD_HEAD_MIN_TRAINING_STEPS: usize = 200;

/// Learned regression head for the per-context reject bar: a `FeedForwardNetwork`
/// (`DIM → hidden → 1`, sigmoid). Process-local, carried + trained across games by the
/// learner like the other auxiliary heads.
#[derive(Clone, Debug)]
pub struct MpcRejectThresholdHead {
    network: FeedForwardNetwork,
    training_steps: usize,
    last_loss: Option<f64>,
}

impl MpcRejectThresholdHead {
    pub fn new(seed: u32) -> Self {
        let mut rng = mulberry32(seed ^ 0x9D2B_4A17);
        let network = FeedForwardNetwork::random(
            &RandomNetworkSpec {
                input_dim: MPC_REJECT_THRESHOLD_FEATURE_DIM,
                hidden_layers: vec![MPC_REJECT_THRESHOLD_HIDDEN_UNITS],
                output_dim: 1,
                hidden_activation: ActivationName::Tanh,
                output_activation: ActivationName::Sigmoid,
                weight_scale: None,
            },
            &mut rng,
        );
        MpcRejectThresholdHead {
            network,
            training_steps: 0,
            last_loss: None,
        }
    }

    /// Predicted effective reject bar in `[FLOOR, CEIL]` for a captured context, or
    /// `None` on a malformed (mis-dimensioned / non-finite) input. The sigmoid output
    /// is affine-mapped onto the valid bar band.
    pub fn predict(&self, inputs: &MpcRejectThresholdInputs) -> Option<f64> {
        let features = inputs.to_features();
        if features.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let out = self.network.predict(&features[..]);
        let raw = out.first().copied().filter(|p| p.is_finite())?;
        // Sigmoid ∈ (0,1) → bar band [FLOOR, CEIL].
        let bar = MPC_REJECT_THRESHOLD_FLOOR
            + raw.clamp(0.0, 1.0) * (MPC_REJECT_THRESHOLD_CEIL - MPC_REJECT_THRESHOLD_FLOOR);
        Some(bar)
    }

    /// Map an effective bar in `[FLOOR, CEIL]` back to the sigmoid target in `[0, 1]`
    /// the network regresses toward.
    fn bar_to_target(bar: f64) -> f64 {
        ((bar - MPC_REJECT_THRESHOLD_FLOOR)
            / (MPC_REJECT_THRESHOLD_CEIL - MPC_REJECT_THRESHOLD_FLOOR))
            .clamp(0.0, 1.0)
    }

    /// One SGD epoch regressing the head toward `(context, target_bar)` pairs
    /// (supervised bootstrap toward the analytic seed). Returns the mean step loss.
    pub fn train(
        &mut self,
        samples: &[(MpcRejectThresholdInputs, f64)],
        learning_rate: f64,
    ) -> f64 {
        let mut total = 0.0;
        let mut applied = 0usize;
        for (inputs, target_bar) in samples {
            let features = inputs.to_features();
            if features.iter().any(|v| !v.is_finite()) || !target_bar.is_finite() {
                continue;
            }
            let target = [Self::bar_to_target(*target_bar)];
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

    /// One **reward-weighted-regression** epoch over RL samples (offline policy
    /// improvement). For each committed decision at probability `p`, the target bar is
    /// pushed to the correct side of `p` by the sign of the (baselined) reward: an
    /// above-baseline outcome pulls the bar BELOW `p` (keep committing such moves), a
    /// below-baseline outcome pushes it ABOVE `p` (refuse them). Both directions are
    /// informative, so the weight rises with the reward's magnitude off baseline.
    /// Returns the mean step loss.
    pub fn train_reward_weighted(
        &mut self,
        samples: &[MpcRejectThresholdSample],
        learning_rate: f64,
    ) -> f64 {
        let finite: Vec<&MpcRejectThresholdSample> = samples
            .iter()
            .filter(|s| s.reward.is_finite() && s.committed_probability.is_finite())
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
            // Both good and bad outcomes teach where the bar belongs; weight by how
            // far the outcome sat from baseline.
            let weight = advantage.abs().clamp(0.0, 4.0).exp().min(7.5);
            // adv > 0 (worked out) ⇒ bar below p; adv < 0 (backfired) ⇒ bar above p.
            let target_bar = (s.committed_probability - THRESHOLD_LEARN_MARGIN * advantage.tanh())
                .clamp(MPC_REJECT_THRESHOLD_FLOOR, MPC_REJECT_THRESHOLD_CEIL);
            let target = [Self::bar_to_target(target_bar)];
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

/// Outcome of training a reject-threshold head on a drained RL corpus, logged by the
/// learning runner so an operator can see the corpus is loaded and learned from.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MpcRejectThresholdTrainingReport {
    pub samples: usize,
    pub training_steps: usize,
    pub epochs: usize,
    pub final_loss: f64,
}

/// Load-and-train entry for a reject-threshold head: given a drained RL corpus, build
/// a fresh [`MpcRejectThresholdHead`] and run `epochs` reward-weighted-regression
/// passes. `None` if no usable samples.
pub fn train_mpc_reject_threshold_head(
    samples: &[MpcRejectThresholdSample],
    seed: u32,
    epochs: usize,
    learning_rate: f64,
) -> Option<(MpcRejectThresholdHead, MpcRejectThresholdTrainingReport)> {
    let usable = samples
        .iter()
        .filter(|s| s.reward.is_finite() && s.committed_probability.is_finite())
        .count();
    if usable == 0 || epochs == 0 {
        return None;
    }
    let mut head = MpcRejectThresholdHead::new(seed);
    let mut final_loss = 0.0;
    for _ in 0..epochs {
        final_loss = head.train_reward_weighted(samples, learning_rate);
    }
    let report = MpcRejectThresholdTrainingReport {
        samples: usable,
        training_steps: head.training_steps(),
        epochs,
        final_loss,
    };
    Some((head, report))
}

/// Public wrapper returning ONLY the report (the head type is engine-internal). The
/// learning runner uses this to train the drained corpus and log that it learns.
pub fn report_mpc_reject_threshold_training(
    samples: &[MpcRejectThresholdSample],
    seed: u32,
    epochs: usize,
    learning_rate: f64,
) -> Option<MpcRejectThresholdTrainingReport> {
    train_mpc_reject_threshold_head(samples, seed, epochs, learning_rate).map(|(_, report)| report)
}

impl WorldSnapshot {
    /// Build the [`MpcRejectThresholdInputs`] for `player` committing a `family` action
    /// with analytic base bar `base_threshold`. `None` for a degenerate roster. Pure /
    /// RNG-free.
    pub(crate) fn build_mpc_reject_threshold_inputs(
        &self,
        player: &PlayerSnapshot,
        family: MpcRejectFamily,
        base_threshold: f64,
    ) -> Option<MpcRejectThresholdInputs> {
        let pos = self.player_snapshot_position(player);

        // Nearest-opponent pressure: exponential decay in distance to the closest
        // opponent (goalkeepers included — a keeper closing you down is pressure too).
        let mut nearest_opponent = f64::INFINITY;
        for p in &self.players {
            if p.team == player.team {
                continue;
            }
            let d = self.player_snapshot_position(p).distance(pos);
            nearest_opponent = nearest_opponent.min(d);
        }
        let nearest_opponent_pressure = if nearest_opponent.is_finite() {
            (-nearest_opponent / PRESSURE_REF_YARDS)
                .exp()
                .clamp(0.0, 1.0)
        } else {
            0.0
        };

        let actor_threat =
            super::pitch_value::threat_at(player.team, pos, self.field_width, self.field_length)
                .clamp(0.0, 1.0);

        let team_control_at_actor = {
            let home = super::pitch_value::pitch_control_home(&self.players, pos);
            match player.team {
                Team::Home => home,
                Team::Away => 1.0 - home,
            }
        };

        Some(MpcRejectThresholdInputs {
            family,
            actor_threat,
            nearest_opponent_pressure,
            team_control_at_actor,
            ball_speed: self.ball.velocity.len(),
            ball_altitude: self.ball.altitude_yards,
            role_defender: (player.role == PlayerRole::Defender) as i32 as f64,
            role_midfielder: (player.role == PlayerRole::Midfielder) as i32 as f64,
            role_forward: (player.role == PlayerRole::Forward) as i32 as f64,
            base_threshold,
        })
    }

    /// The **effective MPC reject bar** for `player_id` committing a `family` action in
    /// the current context. Returns `base_threshold` (without building any features)
    /// unless the model is enabled, keeping the default gate byte-identical. When on:
    /// the trained head once it has learned enough, else the context-aware analytic
    /// seed; either way clamped to a valid probability band.
    pub(crate) fn learned_mpc_reject_threshold(
        &self,
        player_id: usize,
        family: MpcRejectFamily,
        base_threshold: f64,
    ) -> f64 {
        if !mpc_reject_threshold_model_enabled() {
            return base_threshold;
        }
        let Some(player) = self.players.iter().find(|p| p.id == player_id) else {
            return base_threshold;
        };
        let Some(inputs) = self.build_mpc_reject_threshold_inputs(player, family, base_threshold)
        else {
            return base_threshold;
        };
        let bar = self
            .mpc_reject_threshold_head
            .as_ref()
            .filter(|head| head.training_steps() >= MPC_REJECT_THRESHOLD_HEAD_MIN_TRAINING_STEPS)
            .and_then(|head| head.predict(&inputs))
            .unwrap_or_else(|| analytic_mpc_reject_threshold(&inputs));
        if bar.is_finite() {
            bar.clamp(MPC_REJECT_THRESHOLD_FLOOR, MPC_REJECT_THRESHOLD_CEIL)
        } else {
            base_threshold
        }
    }
}

impl SoccerMatch {
    /// Resolve committed actions whose reward window has elapsed. A **no-op** (zero
    /// cost, byte-identical) unless the model is enabled. New decisions are enqueued by
    /// [`Self::enqueue_mpc_reject_threshold_committed_plan`] at the point where the
    /// POMDP/MPC pipeline has selected the action that will actually execute. Keeping
    /// enqueue and resolve separate prevents a hypothetical pass, dribble, and shot
    /// from all inheriting the outcome of whichever one was really played.
    pub(crate) fn collect_mpc_reject_threshold_rl_samples(&mut self, snapshot: &WorldSnapshot) {
        if !mpc_reject_threshold_model_enabled() {
            return;
        }
        let tick = self.tick;
        // Resolve due decisions.
        let mut i = 0;
        while i < self.pending_mpc_reject_threshold.len() {
            if self.pending_mpc_reject_threshold[i].due_tick <= tick {
                let decision = self.pending_mpc_reject_threshold.swap_remove(i);
                let now_territorial = territorial_advantage(snapshot, decision.team);
                if now_territorial.is_finite() && decision.decision_territorial.is_finite() {
                    self.mpc_reject_threshold_samples
                        .push(MpcRejectThresholdSample {
                            inputs: decision.inputs,
                            committed_probability: decision.committed_probability,
                            reward: now_territorial - decision.decision_territorial,
                        });
                    if self.mpc_reject_threshold_samples.len() > MPC_REJECT_THRESHOLD_SAMPLE_CAP {
                        let overflow = self.mpc_reject_threshold_samples.len()
                            - MPC_REJECT_THRESHOLD_SAMPLE_CAP;
                        self.mpc_reject_threshold_samples.drain(0..overflow);
                    }
                }
            } else {
                i += 1;
            }
        }
    }

    /// Enqueue the exact learned action that survived POMDP ranking and MPC
    /// feasibility reconciliation. The later reward therefore belongs to the action
    /// represented by `family`, `inputs`, and `committed_probability`; it is not a
    /// counterfactual label copied onto unplayed alternatives.
    pub(crate) fn enqueue_mpc_reject_threshold_committed_plan(
        &mut self,
        snapshot: &WorldSnapshot,
        player_id: usize,
        plan: &SoccerLearnedPlan,
    ) {
        if !mpc_reject_threshold_model_enabled()
            || self.tick % MPC_REJECT_THRESHOLD_SAMPLE_INTERVAL_TICKS != 0
            || snapshot.ball.holder != Some(player_id)
        {
            return;
        }
        let Some(player) = snapshot.players.iter().find(|p| p.id == player_id) else {
            return;
        };
        if player.role == PlayerRole::Goalkeeper {
            return;
        }
        let Some(family) = mpc_reject_family_for_action_label(&plan.action) else {
            return;
        };
        let Some(committed_probability) =
            Self::learned_plan_mpc_execution_probability(snapshot, player_id, plan)
        else {
            return;
        };
        let territorial = territorial_advantage(snapshot, player.team);
        if !territorial.is_finite() {
            return;
        }
        let base_threshold = mpc_reject_base_threshold(family);
        let Some(inputs) =
            snapshot.build_mpc_reject_threshold_inputs(player, family, base_threshold)
        else {
            return;
        };
        self.pending_mpc_reject_threshold
            .push(PendingMpcRejectThresholdDecision {
                team: player.team,
                inputs,
                committed_probability: committed_probability.clamp(0.0, 1.0),
                decision_territorial: territorial,
                due_tick: self.tick + MPC_REJECT_THRESHOLD_REWARD_WINDOW_TICKS,
            });
    }

    /// Drain the collected reject-threshold RL samples for the cluster learner (train
    /// the head + carry it across games).
    pub fn drain_mpc_reject_threshold_samples(&mut self) -> Vec<MpcRejectThresholdSample> {
        std::mem::take(&mut self.mpc_reject_threshold_samples)
    }

    /// Install the trained reject-threshold head for live consumption. The learner
    /// carries + trains it across games and re-installs it each game; wrapped in `Arc`
    /// so every per-tick snapshot shares it cheaply.
    pub fn set_mpc_reject_threshold_head(&mut self, head: MpcRejectThresholdHead) {
        self.mpc_reject_threshold_head = Some(std::sync::Arc::new(head));
    }
}

#[cfg(test)]
mod mpc_reject_threshold_tests {
    use super::*;

    fn baseline_inputs() -> MpcRejectThresholdInputs {
        MpcRejectThresholdInputs {
            family: MpcRejectFamily::Pass,
            actor_threat: 0.4,
            nearest_opponent_pressure: 0.4,
            team_control_at_actor: 0.55,
            ball_speed: 6.0,
            ball_altitude: 0.0,
            role_defender: 0.0,
            role_midfielder: 1.0,
            role_forward: 0.0,
            base_threshold: 0.18,
        }
    }

    #[test]
    fn features_are_finite_bounded_and_right_dim() {
        let f = baseline_inputs().to_features();
        assert_eq!(f.len(), MPC_REJECT_THRESHOLD_FEATURE_DIM);
        for v in f {
            assert!(v.is_finite());
            assert!(v.abs() <= 2.0 + 1e-9, "feature {v} out of normalized range");
        }
    }

    #[test]
    fn analytic_bar_is_a_valid_probability() {
        let bar = analytic_mpc_reject_threshold(&baseline_inputs());
        assert!(
            (MPC_REJECT_THRESHOLD_FLOOR..=MPC_REJECT_THRESHOLD_CEIL).contains(&bar),
            "bar {bar} out of band"
        );
    }

    #[test]
    fn pressure_raises_the_bar_space_relaxes_it() {
        let mut under_pressure = baseline_inputs();
        under_pressure.nearest_opponent_pressure = 0.95;
        under_pressure.team_control_at_actor = 0.2;
        under_pressure.actor_threat = 0.3;
        let mut in_space = baseline_inputs();
        in_space.nearest_opponent_pressure = 0.05;
        in_space.team_control_at_actor = 0.85;
        in_space.actor_threat = 0.7;
        assert!(
            analytic_mpc_reject_threshold(&under_pressure)
                > analytic_mpc_reject_threshold(&in_space),
            "the reject bar should be higher under pressure than in space"
        );
    }

    #[test]
    fn dangerous_area_relaxes_the_bar_below_base() {
        let mut dangerous = baseline_inputs();
        dangerous.actor_threat = 0.95;
        dangerous.nearest_opponent_pressure = 0.4;
        dangerous.team_control_at_actor = 0.5;
        // In a promising spot with neutral pressure/control the bar should sit at or
        // below the family base (attempt the marginal, promising move).
        assert!(
            analytic_mpc_reject_threshold(&dangerous) <= dangerous.base_threshold + 1e-9,
            "a dangerous attacking spot should not raise the reject bar above base"
        );
    }

    #[test]
    fn head_predicts_a_bar_in_band() {
        let head = MpcRejectThresholdHead::new(11);
        let bar = head.predict(&baseline_inputs()).expect("finite prediction");
        assert!(
            (MPC_REJECT_THRESHOLD_FLOOR..=MPC_REJECT_THRESHOLD_CEIL).contains(&bar),
            "head bar {bar} out of band"
        );
    }

    #[test]
    fn rwr_lowers_the_bar_when_low_p_commits_pay_off() {
        // In the target context, committing marginal (p = 0.2) moves keeps working out
        // (reward above the batch baseline) ⇒ the learned bar for this context should
        // drop below the committed p, i.e. keep attempting them. The below-baseline
        // mixer sits in a DISTINCT context (so it sets the batch baseline without the
        // head having to reconcile two targets at one feature vector).
        let target_ctx = baseline_inputs();
        let mut other_ctx = baseline_inputs();
        other_ctx.nearest_opponent_pressure = 0.95;
        other_ctx.team_control_at_actor = 0.15;
        other_ctx.role_midfielder = 0.0;
        other_ctx.role_forward = 1.0;
        let good_p = 0.2;
        let samples: Vec<MpcRejectThresholdSample> = (0..48)
            .flat_map(|_| {
                [
                    // low-p commits that paid off, in the target context
                    MpcRejectThresholdSample {
                        inputs: target_ctx.clone(),
                        committed_probability: good_p,
                        reward: 1.0,
                    },
                    // a commit that did poorly, in a different context (baseline mixer)
                    MpcRejectThresholdSample {
                        inputs: other_ctx.clone(),
                        committed_probability: 0.6,
                        reward: -1.0,
                    },
                ]
            })
            .collect();
        let mut head = MpcRejectThresholdHead::new(29);
        let mut last = f64::INFINITY;
        for _ in 0..200 {
            last = head.train_reward_weighted(&samples, 0.05);
        }
        let bar = head.predict(&target_ctx).expect("finite");
        assert!(last.is_finite());
        assert!(
            bar < good_p,
            "when low-p commits pay off the learned bar should fall below {good_p}, got {bar}"
        );
    }

    #[test]
    fn train_entry_skips_empty_and_trains_valid() {
        assert!(train_mpc_reject_threshold_head(&[], 1, 4, 0.02).is_none());
        let samples: Vec<MpcRejectThresholdSample> = (0..16)
            .map(|i| MpcRejectThresholdSample {
                inputs: baseline_inputs(),
                committed_probability: if i % 2 == 0 { 0.2 } else { 0.6 },
                reward: if i % 2 == 0 { 1.0 } else { -1.0 },
            })
            .collect();
        let (_, report) =
            train_mpc_reject_threshold_head(&samples, 1, 4, 0.02).expect("trains a usable corpus");
        assert_eq!(report.samples, 16);
        assert_eq!(report.epochs, 4);
        assert!(report.training_steps > 0 && report.final_loss.is_finite());
    }
}
