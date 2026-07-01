//! Per-defender **individual line decision** — the MAPPO layer on top of the
//! collective [`super::back_four_line`] centre.
//!
//! [`super::back_four_line`] decides the back four's **group** line-CENTRE (one
//! scalar: push the whole line up vs. drop it back). But the four are not a rigid
//! bar: at any tick one defender may need to step UP to squeeze a runner onside or
//! DROP a yard to cover a ball played into its channel, independently of where the
//! collective centre sits. That per-agent push/drop is what this module models.
//!
//! ## MAPPO shape
//!
//! The four defenders are **four cooperating agents** trained under MAPPO with
//! **parameter sharing**: one [`DefenderLinePolicyHead`] serves all four (the agent
//! is distinguished only by its own identity + egocentric observation — lane, is-
//! wingback, its slot among the four), executed **decentrally** (each defender reads
//! its own observation and picks its own delta), and credited by a **centralized,
//! shared team reward** (the defending team's windowed territorial advantage — the
//! same critic signal the group line head uses). Parameter sharing across
//! homogeneous agents is the standard MAPPO efficiency; the shared reward is what
//! makes the four cooperate (hold a connected line) rather than each greedily
//! covering its own man.
//!
//! ## Observation — the full 22-body field vector
//!
//! Unlike the group head (which compresses the opponents to a centroid + foremost
//! stats), the individual head reads the **whole field egocentrically**: the ball,
//! the defender itself, its three back-four teammates individually (coordination),
//! **every opponent outfielder individually** (so a lone shoulder-runner peeling off
//! the centroid is visible — the exact thing a defender must react to), the keeper,
//! and the own attacking unit's centroid. All in the defending team's attacking
//! frame (mirror-invariant), position + velocity + acceleration throughout.
//!
//! ## Output & safety
//!
//! A single signed **push delta** in yards (`+` = step up toward the attackers, `−`
//! = drop to cover), bounded to `±`[`DEFENDER_MAX_PUSH_YARDS`]. It is applied on top
//! of the group centre at the live per-defender chokepoint and **re-clamped to the
//! same legal band + offside cap** as the group target, so the individual layer can
//! never step a defender illegally ahead of the line (playing a runner onside) — it
//! only perturbs where inside the legal band the defender eases to.
//!
//! ## Parity
//!
//! Gated **off** by default (`DD_SOCCER_ENABLE_BACK_FOUR_INDIVIDUAL_MODEL`), exactly
//! like the group model: unset ⇒ the builder returns `None` before any features are
//! built and the live seam is byte-identical. Feature build + analytic seed are pure
//! and RNG-free, so tests / the inspector can read them without enabling the seam.

use super::*;
use crate::des::general::neural_network::{ActivationName, FeedForwardNetwork, RandomNetworkSpec};
use crate::des::general::prng::mulberry32;

/// Max opponents encoded individually in the egocentric observation (an outfield
/// side is 10 non-keepers). Fewer ⇒ zero-padded; degenerate rosters stay finite.
pub const DEFENDER_OBS_MAX_OPPONENTS: usize = 10;
/// Back-four teammates encoded individually (the other three defenders).
pub const DEFENDER_OBS_TEAMMATES: usize = 3;

/// Feature count in the per-defender observation. Layout (see
/// [`DefenderAgentInputs::to_features`]):
/// ball 6 + self 6 + identity 3 + context 6 + teammates 3×4 + opponents 10×6 +
/// own-attack centroid 4 = 97.
pub const DEFENDER_AGENT_FEATURE_DIM: usize =
    6 + 6 + 3 + 6 + DEFENDER_OBS_TEAMMATES * 4 + DEFENDER_OBS_MAX_OPPONENTS * 6 + 4;

/// Hidden width of the shared per-defender policy head.
pub const DEFENDER_LINE_HIDDEN_UNITS: usize = 32;

/// Magnitude cap (yd) on the per-defender push/drop delta the head may request. Kept
/// modest: the group centre does the heavy lifting; the individual layer only nudges
/// a single defender up to squeeze or back to cover.
pub const DEFENDER_MAX_PUSH_YARDS: f64 = 8.0;

/// Reference top speed (yd/s) normalizing velocity features.
const REF_TOP_SPEED_YPS: f64 = 9.0;
/// Reference acceleration (yd/s²) normalizing acceleration features.
const REF_ACCEL_YPS2: f64 = 8.0;

/// Env gate enabling the per-defender individual model at the live chokepoint. Off
/// (unset) by default so an unconfigured / test process is byte-identical.
const BACK_FOUR_INDIVIDUAL_MODEL_ENABLE_ENV: &str = "DD_SOCCER_ENABLE_BACK_FOUR_INDIVIDUAL_MODEL";

/// How often (ticks) a per-defender RL decision is sampled while the model is on.
pub const DEFENDER_LINE_SAMPLE_INTERVAL_TICKS: u64 = 15;
/// Window (ticks) over which a sampled decision's territorial reward is measured.
pub const DEFENDER_LINE_REWARD_WINDOW_TICKS: u64 = 45;
/// Cap on the rolling RL sample buffer (4 defenders × 2 teams per cadence ⇒ larger
/// than the single-row-per-team group buffer).
pub const DEFENDER_LINE_SAMPLE_CAP: usize = 8192;
/// Minimum training steps before the head is consumed live; below it the decision
/// falls back to the analytic seed.
pub const DEFENDER_LINE_HEAD_MIN_TRAINING_STEPS: usize = 400;

fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|raw| {
            let v = raw.trim().to_ascii_lowercase();
            matches!(v.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

/// Whether the per-defender individual line model is consulted at the live per-
/// defender chokepoint this process. Off ⇒ the group centre stands alone (parity).
pub fn back_four_individual_model_enabled() -> bool {
    #[cfg(test)]
    {
        env_flag_enabled(BACK_FOUR_INDIVIDUAL_MODEL_ENABLE_ENV)
    }
    #[cfg(not(test))]
    {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| env_flag_enabled(BACK_FOUR_INDIVIDUAL_MODEL_ENABLE_ENV))
    }
}

/// One egocentrically-encoded neighbour (teammate or opponent) relative to the
/// deciding defender, in the defending team's attacking frame.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct NeighbourKinematics {
    /// Forward offset from the deciding defender (`+` = ahead toward the attackers).
    pub rel_fwd: f64,
    /// Lateral offset from the deciding defender (`+` = toward the same wing).
    pub rel_lat: f64,
    pub vel_fwd: f64,
    pub vel_lat: f64,
    pub acc_fwd: f64,
    pub acc_lat: f64,
    /// `true` when this slot is filled (a real neighbour); `false` = zero-padding.
    pub present: bool,
}

/// The full per-defender observation: the ball, self, the three back-four teammates,
/// every opponent outfielder, the keeper, and the own attacking centroid — all in the
/// defending team's attacking frame. Plain struct so it can be logged as a training
/// row, asserted in tests, and surfaced by the inspector independently of the net.
#[derive(Clone, Debug, PartialEq)]
pub struct DefenderAgentInputs {
    pub field_length: f64,
    pub field_width: f64,

    // 1. Ball.
    pub ball_fwd_from_own_goal: f64,
    pub ball_lat: f64,
    pub ball_vel_fwd: f64,
    pub ball_vel_lat: f64,
    pub ball_acc_fwd: f64,
    pub ball_acc_lat: f64,

    // 2. Self (the deciding defender).
    pub self_fwd_from_own_goal: f64,
    pub self_lat: f64,
    pub self_vel_fwd: f64,
    pub self_vel_lat: f64,
    pub self_acc_fwd: f64,
    pub self_acc_lat: f64,

    // 3. Identity (what distinguishes this MAPPO agent under parameter sharing).
    pub is_wide: bool,
    /// Signed lane fraction of this defender's home slot (`−1` left touchline …
    /// `+1` right, from the attacking frame).
    pub home_lane_frac: f64,
    /// The group line-centre depth (yd from own goal) this defender perturbs around.
    pub group_centre_fwd_from_own_goal: f64,

    // 4. Context.
    pub we_control: bool,
    pub they_control: bool,
    pub gk_fwd_from_own_goal: f64,
    pub gk_lat: f64,
    pub offside_in_force: bool,
    pub trap_active_or_imminent: bool,

    // 5. Back-four teammates (the other three), ordered by lateral (left→right).
    pub teammates: [NeighbourKinematics; DEFENDER_OBS_TEAMMATES],

    // 6. Opponent outfielders, ordered by threat (deepest into our territory first).
    pub opponents: [NeighbourKinematics; DEFENDER_OBS_MAX_OPPONENTS],

    // 7. Own attacking unit (non-back-four, non-keeper) centroid, for shape context.
    pub own_attack_fwd_from_own_goal: f64,
    pub own_attack_lat: f64,
    pub own_attack_vel_fwd: f64,
    pub own_attack_vel_lat: f64,
}

impl DefenderAgentInputs {
    /// The normalized network input vector. **This ordering is the single source of
    /// truth** for the feature layout — change it and bump
    /// [`DEFENDER_AGENT_FEATURE_DIM`] and any persisted head.
    pub fn to_features(&self) -> [f64; DEFENDER_AGENT_FEATURE_DIM] {
        let l = self.field_length.max(1.0);
        let half_w = (self.field_width * 0.5).max(1.0);
        let fwd = |y: f64| (y / l).clamp(-0.2, 1.2);
        let lat = |x: f64| (x / half_w).clamp(-1.5, 1.5);
        // Relative offsets span the whole pitch, so allow a wider clamp.
        let rel = |d: f64| (d / l).clamp(-1.2, 1.2);
        let vel = |v: f64| (v / REF_TOP_SPEED_YPS).clamp(-2.0, 2.0);
        let acc = |a: f64| (a / REF_ACCEL_YPS2).clamp(-2.0, 2.0);
        let b = |flag: bool| if flag { 1.0 } else { 0.0 };

        let mut f = [0.0; DEFENDER_AGENT_FEATURE_DIM];
        let mut i = 0;
        let mut push = |slot: &mut usize, buf: &mut [f64; DEFENDER_AGENT_FEATURE_DIM], v: f64| {
            buf[*slot] = v;
            *slot += 1;
        };

        // 1. Ball.
        push(&mut i, &mut f, fwd(self.ball_fwd_from_own_goal));
        push(&mut i, &mut f, lat(self.ball_lat));
        push(&mut i, &mut f, vel(self.ball_vel_fwd));
        push(&mut i, &mut f, vel(self.ball_vel_lat));
        push(&mut i, &mut f, acc(self.ball_acc_fwd));
        push(&mut i, &mut f, acc(self.ball_acc_lat));
        // 2. Self.
        push(&mut i, &mut f, fwd(self.self_fwd_from_own_goal));
        push(&mut i, &mut f, lat(self.self_lat));
        push(&mut i, &mut f, vel(self.self_vel_fwd));
        push(&mut i, &mut f, vel(self.self_vel_lat));
        push(&mut i, &mut f, acc(self.self_acc_fwd));
        push(&mut i, &mut f, acc(self.self_acc_lat));
        // 3. Identity.
        push(&mut i, &mut f, b(self.is_wide));
        push(&mut i, &mut f, self.home_lane_frac.clamp(-1.5, 1.5));
        push(&mut i, &mut f, fwd(self.group_centre_fwd_from_own_goal));
        // 4. Context.
        push(&mut i, &mut f, b(self.we_control));
        push(&mut i, &mut f, b(self.they_control));
        push(&mut i, &mut f, fwd(self.gk_fwd_from_own_goal));
        push(&mut i, &mut f, lat(self.gk_lat));
        push(&mut i, &mut f, b(self.offside_in_force));
        push(&mut i, &mut f, b(self.trap_active_or_imminent));
        // 5. Teammates.
        for t in &self.teammates {
            push(&mut i, &mut f, rel(t.rel_fwd));
            push(&mut i, &mut f, rel(t.rel_lat));
            push(&mut i, &mut f, vel(t.vel_fwd));
            push(&mut i, &mut f, vel(t.vel_lat));
        }
        // 6. Opponents.
        for o in &self.opponents {
            push(&mut i, &mut f, rel(o.rel_fwd));
            push(&mut i, &mut f, rel(o.rel_lat));
            push(&mut i, &mut f, vel(o.vel_fwd));
            push(&mut i, &mut f, vel(o.vel_lat));
            push(&mut i, &mut f, acc(o.acc_fwd));
            push(&mut i, &mut f, acc(o.acc_lat));
        }
        // 7. Own attack centroid.
        push(&mut i, &mut f, fwd(self.own_attack_fwd_from_own_goal));
        push(&mut i, &mut f, lat(self.own_attack_lat));
        push(&mut i, &mut f, vel(self.own_attack_vel_fwd));
        push(&mut i, &mut f, vel(self.own_attack_vel_lat));

        debug_assert_eq!(i, DEFENDER_AGENT_FEATURE_DIM);
        f
    }
}

/// Analytic, deterministic, RNG-free **seed policy** for the per-defender push delta
/// (yd, `+` = step up, `−` = drop to cover), the bootstrap target + live fallback the
/// learned [`DefenderLinePolicyHead`] first imitates then beats.
///
/// Intent, egocentric: DROP to cover when this defender has an opponent on its
/// shoulder threatening to run in behind (an opponent near its channel that is
/// goal-side of / level with it, or accelerating toward our goal) — the classic
/// runner the collective centroid misses. STEP UP to squeeze when this defender's
/// channel is clear and it sits deeper than its teammates (a lagger dragging the
/// line ragged), so the four stay flat + connected. Bounded to
/// `±`[`DEFENDER_MAX_PUSH_YARDS`].
pub fn analytic_defender_push_delta(inputs: &DefenderAgentInputs) -> f64 {
    let mut delta = 0.0;

    // --- Cover a shoulder runner (DROP, negative) ---
    // The most threatening opponent in this defender's lateral channel: near in
    // lateral, and either already goal-side (rel_fwd < 0, behind us toward our goal)
    // or driving toward our goal (negative fwd velocity). The closer + faster, the
    // deeper we drop to stay goal-side of the run.
    let mut worst_threat = 0.0_f64;
    for o in inputs.opponents.iter().filter(|o| o.present) {
        // Channel proximity: within ~6yd laterally is "our man".
        let lateral_closeness = (1.0 - (o.rel_lat.abs() / 6.0)).clamp(0.0, 1.0);
        if lateral_closeness <= 0.0 {
            continue;
        }
        // Goal-side threat: opponent level-with or behind us (rel_fwd ≤ ~2yd ahead)
        // is a runner we must stay goal-side of; weight by how goal-side it is.
        let goal_side = ((2.0 - o.rel_fwd) / 8.0).clamp(0.0, 1.0);
        // Ball-ward acceleration/velocity: a man accelerating in behind is urgent.
        let breaking = (-o.vel_fwd / REF_TOP_SPEED_YPS).clamp(0.0, 1.0);
        let threat = lateral_closeness * (0.7 * goal_side + 0.3 * breaking);
        worst_threat = worst_threat.max(threat);
    }
    delta -= DEFENDER_MAX_PUSH_YARDS * worst_threat;

    // --- Squeeze a lagging, unthreatened defender UP (positive) ---
    // If this defender sits DEEPER than the group centre (a lagger) and its channel
    // is clear, step up toward the line so the four stay flat + connected. Never when
    // a live threat pulled us back above.
    if worst_threat < 0.15 {
        let behind_centre =
            (inputs.group_centre_fwd_from_own_goal - inputs.self_fwd_from_own_goal).max(0.0);
        let step_up = (behind_centre / DEFENDER_MAX_PUSH_YARDS).clamp(0.0, 1.0);
        // In possession the line steps up more eagerly; when they control, gently.
        let eagerness = if inputs.we_control {
            0.8
        } else if inputs.they_control {
            0.4
        } else {
            0.6
        };
        delta += DEFENDER_MAX_PUSH_YARDS * step_up * eagerness;
    }

    delta.clamp(-DEFENDER_MAX_PUSH_YARDS, DEFENDER_MAX_PUSH_YARDS)
}

/// One reward-weighted RL training row for the per-defender head: the egocentric
/// state at a decision, the push delta actually used (the action), and the shared-
/// team windowed reward attributed to it. Mirrors
/// [`super::back_four_line::LineDepthSample`] but the action is a signed delta.
#[derive(Clone, Debug, PartialEq)]
pub struct DefenderLineSample {
    pub inputs: DefenderAgentInputs,
    /// The push delta actually used (the action), yd in `±DEFENDER_MAX_PUSH_YARDS`.
    pub action_push_yards: f64,
    /// Shared-team reward over the following window (any scale; RWR baselines it).
    pub reward: f64,
}

/// An open per-defender decision awaiting its windowed reward. Recorded at the
/// decision tick with the team's territorial advantage then; resolved
/// [`DEFENDER_LINE_REWARD_WINDOW_TICKS`] later by differencing it.
#[derive(Clone, Debug, PartialEq)]
pub struct PendingDefenderLineDecision {
    pub team: Team,
    pub player_id: usize,
    pub inputs: DefenderAgentInputs,
    pub action_push_yards: f64,
    pub decision_territorial: f64,
    pub due_tick: u64,
}

/// Shared (parameter-sharing) MAPPO policy head for the per-defender push delta: a
/// `FeedForwardNetwork` (`DIM → hidden → 1`, **tanh** output = a signed delta scaled
/// by [`DEFENDER_MAX_PUSH_YARDS`]). Mirrors [`super::back_four_line::BackFourLineHead`]
/// but regresses a signed action rather than a `[0,1]` fraction.
#[derive(Clone, Debug)]
pub struct DefenderLinePolicyHead {
    network: FeedForwardNetwork,
    training_steps: usize,
    last_loss: Option<f64>,
}

impl DefenderLinePolicyHead {
    pub fn new(seed: u32) -> Self {
        let mut rng = mulberry32(seed ^ 0x51ED_270B);
        let network = FeedForwardNetwork::random(
            &RandomNetworkSpec {
                input_dim: DEFENDER_AGENT_FEATURE_DIM,
                hidden_layers: vec![DEFENDER_LINE_HIDDEN_UNITS],
                output_dim: 1,
                hidden_activation: ActivationName::Tanh,
                // Tanh output = a signed delta fraction in (−1, 1).
                output_activation: ActivationName::Tanh,
                weight_scale: None,
            },
            &mut rng,
        );
        DefenderLinePolicyHead {
            network,
            training_steps: 0,
            last_loss: None,
        }
    }

    /// Predicted push delta (yd, signed, bounded to `±DEFENDER_MAX_PUSH_YARDS`) for a
    /// captured state, or `None` on a malformed (mis-dimensioned / non-finite) input.
    pub fn predict(&self, inputs: &DefenderAgentInputs) -> Option<f64> {
        let features = inputs.to_features();
        if features.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let out = self.network.predict(&features[..]);
        out.first()
            .copied()
            .filter(|p| p.is_finite())
            .map(|frac| (frac.clamp(-1.0, 1.0)) * DEFENDER_MAX_PUSH_YARDS)
    }

    /// One SGD epoch regressing the head toward `(features, target_delta_yards)`
    /// pairs (supervised bootstrap toward the analytic seed, or an RL target). The
    /// target delta is mapped back to the tanh `(−1, 1)` action space. Returns the
    /// mean step loss.
    pub fn train(&mut self, samples: &[(DefenderAgentInputs, f64)], learning_rate: f64) -> f64 {
        let mut total = 0.0;
        let mut applied = 0usize;
        for (inputs, target_yards) in samples {
            let features = inputs.to_features();
            if features.iter().any(|v| !v.is_finite()) || !target_yards.is_finite() {
                continue;
            }
            let target = [(target_yards / DEFENDER_MAX_PUSH_YARDS).clamp(-1.0, 1.0)];
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
    /// improvement, the MAPPO update with a shared-team advantage). Regress toward each
    /// sample's ACTION (the delta actually used), weighted by how much its shared-team
    /// reward beat the batch baseline: above-average decisions are imitated strongly,
    /// below-average ones barely. Returns the mean step loss.
    pub fn train_reward_weighted(
        &mut self,
        samples: &[DefenderLineSample],
        learning_rate: f64,
    ) -> f64 {
        let finite: Vec<&DefenderLineSample> = samples
            .iter()
            .filter(|s| s.reward.is_finite() && s.action_push_yards.is_finite())
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
            let target = [(s.action_push_yards / DEFENDER_MAX_PUSH_YARDS).clamp(-1.0, 1.0)];
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

/// Outcome of training a per-defender head on a drained RL corpus, logged by the
/// learning runner. Mirrors [`super::back_four_line::LineDepthTrainingReport`].
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DefenderLineTrainingReport {
    pub samples: usize,
    pub training_steps: usize,
    pub epochs: usize,
    pub final_loss: f64,
}

/// Load-and-train entry for a per-defender head: build a fresh
/// [`DefenderLinePolicyHead`] and run `epochs` reward-weighted-regression passes over
/// a drained RL corpus. `None` if no usable samples. Mirrors
/// [`super::back_four_line::train_line_depth_head`].
pub fn train_defender_line_head(
    samples: &[DefenderLineSample],
    seed: u32,
    epochs: usize,
    learning_rate: f64,
) -> Option<(DefenderLinePolicyHead, DefenderLineTrainingReport)> {
    let usable = samples
        .iter()
        .filter(|s| s.reward.is_finite() && s.action_push_yards.is_finite())
        .count();
    if usable == 0 || epochs == 0 {
        return None;
    }
    let mut head = DefenderLinePolicyHead::new(seed);
    let mut final_loss = 0.0;
    for _ in 0..epochs {
        final_loss = head.train_reward_weighted(samples, learning_rate);
    }
    let report = DefenderLineTrainingReport {
        samples: usable,
        training_steps: head.training_steps(),
        epochs,
        final_loss,
    };
    Some((head, report))
}

impl WorldSnapshot {
    /// Build the [`DefenderAgentInputs`] egocentric observation for `me` (a back-four
    /// defender) in its team's attacking frame. `None` when the model is gated off,
    /// `me` is not a defender, or the roster is degenerate (no keeper / no opponents).
    /// Pure / RNG-free.
    pub(crate) fn build_defender_agent_inputs(
        &self,
        me: &PlayerSnapshot,
    ) -> Option<DefenderAgentInputs> {
        if me.role != PlayerRole::Defender {
            return None;
        }
        let team = me.team;
        let attack = team.attack_dir();
        let mid_x = self.field_width * 0.5;
        let own_goal_fwd = self.own_goal_y_for(team) * attack;
        let lat = |x: f64| (x - mid_x) * attack;
        let fwd_from_goal = |y: f64| y * attack - own_goal_fwd;

        let me_pos = self.player_snapshot_position(me);
        let self_fwd = fwd_from_goal(me_pos.y);
        let self_lat = lat(me_pos.x);

        // Teammates (other back-four defenders), ordered by lateral.
        let mut teammate_rows: Vec<(f64, NeighbourKinematics)> = Vec::new();
        // Opponents (outfield), collected with their depth-from-our-goal for threat sort.
        let mut opponent_rows: Vec<(f64, NeighbourKinematics)> = Vec::new();
        // Own attacking unit (non-defender, non-keeper) centroid.
        let mut own_attack_pos = Vec2::zero();
        let mut own_attack_vel = Vec2::zero();
        let mut own_attack_n = 0.0;

        for p in &self.players {
            let p_pos = self.player_snapshot_position(p);
            if p.team == team {
                if p.role == PlayerRole::Defender && p.id != me.id {
                    teammate_rows.push((
                        lat(p_pos.x),
                        NeighbourKinematics {
                            rel_fwd: fwd_from_goal(p_pos.y) - self_fwd,
                            rel_lat: lat(p_pos.x) - self_lat,
                            vel_fwd: p.velocity.y * attack,
                            vel_lat: p.velocity.x * attack,
                            acc_fwd: p.acceleration.y * attack,
                            acc_lat: p.acceleration.x * attack,
                            present: true,
                        },
                    ));
                } else if p.role != PlayerRole::Goalkeeper && p.role != PlayerRole::Defender {
                    own_attack_pos = own_attack_pos + p_pos;
                    own_attack_vel = own_attack_vel + p.velocity;
                    own_attack_n += 1.0;
                }
            } else if p.role != PlayerRole::Goalkeeper {
                let depth = fwd_from_goal(p_pos.y);
                opponent_rows.push((
                    depth,
                    NeighbourKinematics {
                        rel_fwd: depth - self_fwd,
                        rel_lat: lat(p_pos.x) - self_lat,
                        vel_fwd: p.velocity.y * attack,
                        vel_lat: p.velocity.x * attack,
                        acc_fwd: p.acceleration.y * attack,
                        acc_lat: p.acceleration.x * attack,
                        present: true,
                    },
                ));
            }
        }
        if opponent_rows.is_empty() {
            return None;
        }
        let gk = self
            .players
            .iter()
            .find(|p| p.team == team && p.role == PlayerRole::Goalkeeper)?;
        let gk_pos = self.player_snapshot_position(gk);

        // Teammates left→right; opponents deepest-into-our-territory first (threat).
        teammate_rows.sort_by(|a, b| a.0.total_cmp(&b.0));
        opponent_rows.sort_by(|a, b| a.0.total_cmp(&b.0));

        let mut teammates = [NeighbourKinematics::default(); DEFENDER_OBS_TEAMMATES];
        for (slot, (_, k)) in teammate_rows.into_iter().take(DEFENDER_OBS_TEAMMATES).enumerate() {
            teammates[slot] = k;
        }
        let mut opponents = [NeighbourKinematics::default(); DEFENDER_OBS_MAX_OPPONENTS];
        for (slot, (_, k)) in opponent_rows.into_iter().take(DEFENDER_OBS_MAX_OPPONENTS).enumerate()
        {
            opponents[slot] = k;
        }

        let (own_attack_fwd, own_attack_lat, own_attack_vfwd, own_attack_vlat) = if own_attack_n
            > 0.0
        {
            let inv = 1.0 / own_attack_n;
            let c = own_attack_pos * inv;
            let v = own_attack_vel * inv;
            (fwd_from_goal(c.y), lat(c.x), v.y * attack, v.x * attack)
        } else {
            (self_fwd, self_lat, 0.0, 0.0)
        };

        let controlled = self.controlled_possession_team();
        let trap_active_or_imminent =
            !self.offside_currently_suspended() && controlled != Some(team) && {
                let ball_from_own_goal = fwd_from_goal(self.ball.position.y).max(0.0);
                ball_from_own_goal > self.field_length / 3.0
            };
        let group_centre_fwd = self.back_four_line_v2_centre_fwd(team);

        Some(DefenderAgentInputs {
            field_length: self.field_length,
            field_width: self.field_width,

            ball_fwd_from_own_goal: fwd_from_goal(self.ball.position.y),
            ball_lat: lat(self.ball.position.x),
            ball_vel_fwd: self.ball.velocity.y * attack,
            ball_vel_lat: self.ball.velocity.x * attack,
            ball_acc_fwd: self.ball.acceleration.y * attack,
            ball_acc_lat: self.ball.acceleration.x * attack,

            self_fwd_from_own_goal: self_fwd,
            self_lat,
            self_vel_fwd: me.velocity.y * attack,
            self_vel_lat: me.velocity.x * attack,
            self_acc_fwd: me.acceleration.y * attack,
            self_acc_lat: me.acceleration.x * attack,

            is_wide: self.is_wide_defender(me),
            home_lane_frac: ((me.home_position.x - mid_x) / mid_x.max(1.0) * attack)
                .clamp(-1.5, 1.5),
            group_centre_fwd_from_own_goal: group_centre_fwd - own_goal_fwd,

            we_control: controlled == Some(team),
            they_control: controlled == Some(team.other()),
            gk_fwd_from_own_goal: fwd_from_goal(gk_pos.y),
            gk_lat: lat(gk_pos.x),
            offside_in_force: self.offside_currently_suspended(),
            trap_active_or_imminent,

            teammates,
            opponents,

            own_attack_fwd_from_own_goal: own_attack_fwd,
            own_attack_lat,
            own_attack_vel_fwd: own_attack_vfwd,
            own_attack_vel_lat: own_attack_vlat,
        })
    }

    /// The per-defender push delta (yd, `+` = step up, `−` = drop) for `me` this
    /// tick: the trained head once it clears
    /// [`DEFENDER_LINE_HEAD_MIN_TRAINING_STEPS`], else the analytic seed. `None` when
    /// the model is gated off or the observation is degenerate — the caller then
    /// applies no per-defender adjustment (the group centre stands alone, parity).
    pub(crate) fn back_four_individual_push_delta(&self, me: &PlayerSnapshot) -> Option<f64> {
        if !back_four_individual_model_enabled() {
            return None;
        }
        let inputs = self.build_defender_agent_inputs(me)?;
        let delta = self
            .defender_line_head
            .as_ref()
            .filter(|head| head.training_steps() >= DEFENDER_LINE_HEAD_MIN_TRAINING_STEPS)
            .and_then(|head| head.predict(&inputs))
            .unwrap_or_else(|| analytic_defender_push_delta(&inputs));
        delta
            .is_finite()
            .then(|| delta.clamp(-DEFENDER_MAX_PUSH_YARDS, DEFENDER_MAX_PUSH_YARDS))
    }
}

impl SoccerMatch {
    /// Gated per-tick RL sample collection for the per-defender head, driven off the
    /// per-tick learning `snapshot`. A **no-op** (zero cost, byte-identical) unless the
    /// individual model is enabled. When on: resolves decisions whose reward window has
    /// elapsed (reward = the defending team's territorial-advantage change over the
    /// window) and samples a fresh decision per back-four defender every
    /// [`DEFENDER_LINE_SAMPLE_INTERVAL_TICKS`]. Drained via
    /// [`Self::drain_defender_line_samples`].
    pub(crate) fn collect_defender_line_rl_samples(&mut self, snapshot: &WorldSnapshot) {
        if !back_four_individual_model_enabled() {
            return;
        }
        let tick = self.tick;
        let mut i = 0;
        while i < self.pending_defender_line.len() {
            if self.pending_defender_line[i].due_tick <= tick {
                let decision = self.pending_defender_line.swap_remove(i);
                let now_territorial = territorial_advantage(snapshot, decision.team);
                if now_territorial.is_finite() && decision.decision_territorial.is_finite() {
                    let reward = now_territorial - decision.decision_territorial;
                    self.defender_line_samples.push(DefenderLineSample {
                        inputs: decision.inputs,
                        action_push_yards: decision.action_push_yards,
                        reward,
                    });
                    if self.defender_line_samples.len() > DEFENDER_LINE_SAMPLE_CAP {
                        let overflow = self.defender_line_samples.len() - DEFENDER_LINE_SAMPLE_CAP;
                        self.defender_line_samples.drain(0..overflow);
                    }
                }
            } else {
                i += 1;
            }
        }
        if tick % DEFENDER_LINE_SAMPLE_INTERVAL_TICKS == 0 {
            // Sample one decision per back-four defender (the MAPPO agents). Collect the
            // ids first so the immutable snapshot borrow does not overlap the push.
            let defender_ids: Vec<(Team, usize)> = snapshot
                .players
                .iter()
                .filter(|p| p.role == PlayerRole::Defender)
                .map(|p| (p.team, p.id))
                .collect();
            for (team, id) in defender_ids {
                let Some(me) = snapshot.players.iter().find(|p| p.id == id) else {
                    continue;
                };
                let Some(inputs) = snapshot.build_defender_agent_inputs(me) else {
                    continue;
                };
                let action = snapshot
                    .back_four_individual_push_delta(me)
                    .unwrap_or_else(|| analytic_defender_push_delta(&inputs));
                let territorial = territorial_advantage(snapshot, team);
                if action.is_finite() && territorial.is_finite() {
                    self.pending_defender_line.push(PendingDefenderLineDecision {
                        team,
                        player_id: id,
                        inputs,
                        action_push_yards: action,
                        decision_territorial: territorial,
                        due_tick: tick + DEFENDER_LINE_REWARD_WINDOW_TICKS,
                    });
                }
            }
        }
    }

    /// Drain the collected per-defender RL samples for the cluster learner. `pub` so
    /// the learner binary can drain it. Mirrors [`Self::drain_line_depth_samples`].
    pub fn drain_defender_line_samples(&mut self) -> Vec<DefenderLineSample> {
        std::mem::take(&mut self.defender_line_samples)
    }

    /// Install the trained per-defender head for live consumption. Wrapped in `Arc` so
    /// every per-tick snapshot shares it cheaply. Mirrors [`Self::set_line_depth_head`].
    pub fn set_defender_line_head(&mut self, head: DefenderLinePolicyHead) {
        self.defender_line_head = Some(std::sync::Arc::new(head));
    }
}

#[cfg(test)]
mod back_four_individual_tests {
    use super::*;

    fn neighbour(rel_fwd: f64, rel_lat: f64, vel_fwd: f64) -> NeighbourKinematics {
        NeighbourKinematics {
            rel_fwd,
            rel_lat,
            vel_fwd,
            vel_lat: 0.0,
            acc_fwd: 0.0,
            acc_lat: 0.0,
            present: true,
        }
    }

    fn baseline_inputs() -> DefenderAgentInputs {
        DefenderAgentInputs {
            field_length: 120.0,
            field_width: 80.0,
            ball_fwd_from_own_goal: 60.0,
            ball_lat: 0.0,
            ball_vel_fwd: 0.0,
            ball_vel_lat: 0.0,
            ball_acc_fwd: 0.0,
            ball_acc_lat: 0.0,
            self_fwd_from_own_goal: 30.0,
            self_lat: 0.0,
            self_vel_fwd: 0.0,
            self_vel_lat: 0.0,
            self_acc_fwd: 0.0,
            self_acc_lat: 0.0,
            is_wide: false,
            home_lane_frac: 0.2,
            group_centre_fwd_from_own_goal: 30.0,
            we_control: false,
            they_control: true,
            gk_fwd_from_own_goal: 6.0,
            gk_lat: 0.0,
            offside_in_force: false,
            trap_active_or_imminent: true,
            teammates: [NeighbourKinematics::default(); DEFENDER_OBS_TEAMMATES],
            opponents: [NeighbourKinematics::default(); DEFENDER_OBS_MAX_OPPONENTS],
            own_attack_fwd_from_own_goal: 70.0,
            own_attack_lat: 0.0,
            own_attack_vel_fwd: 0.0,
            own_attack_vel_lat: 0.0,
        }
    }

    #[test]
    fn features_are_finite_bounded_and_right_dim() {
        let mut inp = baseline_inputs();
        inp.opponents[0] = neighbour(-4.0, 2.0, -6.0);
        inp.teammates[0] = neighbour(0.0, -20.0, 0.0);
        let f = inp.to_features();
        assert_eq!(f.len(), DEFENDER_AGENT_FEATURE_DIM);
        for v in f {
            assert!(v.is_finite());
            assert!(v.abs() <= 2.0 + 1e-9, "feature {v} out of normalized range");
        }
    }

    #[test]
    fn analytic_delta_is_bounded() {
        let d = analytic_defender_push_delta(&baseline_inputs());
        assert!(d.abs() <= DEFENDER_MAX_PUSH_YARDS + 1e-9);
    }

    #[test]
    fn shoulder_runner_drops_the_defender() {
        // An opponent on our shoulder (level with us, in our channel, breaking toward
        // our goal) should pull the defender DEEPER (negative delta).
        let mut inp = baseline_inputs();
        inp.opponents[0] = neighbour(-1.0, 1.0, -7.0); // goal-side, close, breaking
        let d = analytic_defender_push_delta(&inp);
        assert!(d < 0.0, "a shoulder runner should drop the defender: {d}");
    }

    #[test]
    fn clear_lagging_defender_steps_up() {
        // No threat, and we sit 8yd DEEPER than the group centre ⇒ step UP (positive).
        let mut inp = baseline_inputs();
        inp.self_fwd_from_own_goal = 22.0;
        inp.group_centre_fwd_from_own_goal = 30.0;
        // Opponents all far away laterally (no channel threat).
        inp.opponents[0] = neighbour(20.0, 40.0, 0.0);
        let d = analytic_defender_push_delta(&inp);
        assert!(d > 0.0, "a clear lagging defender should step up: {d}");
    }

    #[test]
    fn head_predicts_a_bounded_delta() {
        let head = DefenderLinePolicyHead::new(7);
        let d = head.predict(&baseline_inputs()).expect("finite prediction");
        assert!(d.abs() <= DEFENDER_MAX_PUSH_YARDS + 1e-9, "delta {d} unbounded");
    }

    #[test]
    fn reward_weighted_training_steers_toward_high_reward_action() {
        let mut head = DefenderLinePolicyHead::new(23);
        let inputs = baseline_inputs();
        // Same state: a DROP (−6) is rewarded, a STEP-UP (+6) penalized. RWR should
        // pull the head toward the negative (drop) action.
        let samples: Vec<DefenderLineSample> = (0..32)
            .flat_map(|_| {
                [
                    DefenderLineSample {
                        inputs: inputs.clone(),
                        action_push_yards: -6.0,
                        reward: 1.0,
                    },
                    DefenderLineSample {
                        inputs: inputs.clone(),
                        action_push_yards: 6.0,
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
        assert!(after < 0.0, "RWR should steer toward the rewarded drop: {after}");
    }

    #[test]
    fn train_entry_skips_empty_and_trains_valid() {
        assert!(train_defender_line_head(&[], 1, 4, 0.02).is_none());
        let samples: Vec<DefenderLineSample> = (0..16)
            .map(|i| DefenderLineSample {
                inputs: baseline_inputs(),
                action_push_yards: if i % 2 == 0 { 4.0 } else { -4.0 },
                reward: if i % 2 == 0 { 1.0 } else { -1.0 },
            })
            .collect();
        let (_, report) =
            train_defender_line_head(&samples, 1, 4, 0.02).expect("trains a usable corpus");
        assert_eq!(report.samples, 16);
        assert_eq!(report.epochs, 4);
        assert!(report.training_steps > 0 && report.final_loss.is_finite());
    }

    #[test]
    fn reward_weighted_training_is_a_noop_on_empty_or_nonfinite() {
        let mut head = DefenderLinePolicyHead::new(5);
        assert_eq!(head.train_reward_weighted(&[], 0.05), 0.0);
        let bad = vec![DefenderLineSample {
            inputs: baseline_inputs(),
            action_push_yards: f64::NAN,
            reward: f64::NAN,
        }];
        assert_eq!(head.train_reward_weighted(&bad, 0.05), 0.0);
        assert_eq!(head.training_steps(), 0);
    }
}
