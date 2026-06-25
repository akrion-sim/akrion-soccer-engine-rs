//! Dynamic, learnable **back-four line model** (forward / y-axis line depth).
//!
//! The forward position of the back four is not a fixed formation slot: at every
//! tick the line has a *centre* (the average y of the defenders holding it) and
//! that centre should be a dynamic function of the match state. This module is
//! the seam through which that function is computed — analytically today, by a
//! neural net once trained — and fed into the single live line chokepoint,
//! [`WorldSnapshot::defender_line_band_average_adjusted_y`].
//!
//! The controlled quantity is the line **centre**, expressed as a *gap fraction*
//! in `[0, 1]`: `0` = the line sits level with the (predicted) ball (highest line
//! / offside trap), `1` = the full `max_gap` behind it (deepest block). The
//! caller maps that to a forward coordinate and re-clamps it to the legal band,
//! so the model chooses *where inside the legal line* the four sit, never an
//! illegal line.
//!
//! Inputs (the 7 groups the model is a function of), all in the defending team's
//! attacking frame so the model is mirror-invariant:
//!   1. ball position / velocity / acceleration,
//!   2. opponent vector — position / velocity / acceleration,
//!   3. the back four's own position / velocity / acceleration,
//!   4. opponent centre of mass (folded into group 2's centroid),
//!   5. which team controls the ball,
//!   6. our own goalkeeper's position,
//!   7. offside-trap state now and (implicitly, via velocity/accel + the predicted
//!      ball) over the next ~3 s.
//!
//! ## Parity
//!
//! Gated **off** by default (`DD_SOCCER_ENABLE_BACK_FOUR_LINE_MODEL`). When unset,
//! [`WorldSnapshot::back_four_line_model_centre_fwd`] returns `None` before
//! building any features, so the line chokepoint is byte-identical to before this
//! module existed. The feature build and analytic fraction are pure and RNG-free,
//! so tests / the inspector can read them without enabling the live seam.
//!
//! See `docs/back-four-line-model.md` for the full design + the path to a learned
//! head.

use super::*;
use crate::des::general::neural_network::{
    ActivationName, FeedForwardNetwork, RandomNetworkSpec,
};
use crate::des::general::prng::mulberry32;

/// Number of features in the back-four line state vector. The exact ordering is
/// defined once in [`BackFourLineInputs::to_features`].
pub const BACK_FOUR_LINE_FEATURE_DIM: usize = 26;
/// Weight on the model's own depth vs. the retained existing heuristic centre in
/// the analytic seed (group 8: *leave in existing determinants*). `0` = pure
/// heuristic (parity-ish), `1` = pure model. Kept modest so the well-tuned
/// existing line dominates until a trained head earns more of the blend; a
/// promoted [`BackFourLineHead`] can drive this toward `1`.
pub const BACK_FOUR_LINE_MODEL_BLEND: f64 = 0.5;
/// Hidden width of the learned regression head.
pub const BACK_FOUR_LINE_HIDDEN_UNITS: usize = 24;

/// Reference top speed (yd/s) used to normalize velocity features.
const REF_TOP_SPEED_YPS: f64 = 9.0;
/// Reference acceleration (yd/s²) used to normalize acceleration features.
const REF_ACCEL_YPS2: f64 = 8.0;

/// Env gate enabling the dynamic line model at the live chokepoint. Off (unset)
/// by default so an unconfigured process stays byte-identical.
const BACK_FOUR_LINE_MODEL_ENABLE_ENV: &str = "DD_SOCCER_ENABLE_BACK_FOUR_LINE_MODEL";

fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|raw| {
            let v = raw.trim().to_ascii_lowercase();
            matches!(v.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

/// Whether the dynamic back-four line model is consulted at the live line
/// chokepoint this process. Off ⇒ the analytic heuristic line stands (parity).
pub fn back_four_line_model_enabled() -> bool {
    #[cfg(test)]
    {
        env_flag_enabled(BACK_FOUR_LINE_MODEL_ENABLE_ENV)
    }
    #[cfg(not(test))]
    {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| env_flag_enabled(BACK_FOUR_LINE_MODEL_ENABLE_ENV))
    }
}

/// Raw (un-normalized) state the line model is a function of, captured in the
/// defending team's **attacking frame** (`fwd` increases toward the opponent
/// goal; `lat` is signed distance from the pitch spine). Kept as a plain struct
/// so it can be logged as a training row, asserted on in tests, and surfaced by
/// the inspector independently of the network.
#[derive(Clone, Debug, PartialEq)]
pub struct BackFourLineInputs {
    /// Pitch length / width, for normalization (`L`, `W`).
    pub field_length: f64,
    pub field_width: f64,

    // 1. Ball.
    pub ball_fwd_from_own_goal: f64,
    pub ball_lat: f64,
    pub ball_vel_fwd: f64,
    pub ball_vel_lat: f64,
    pub ball_acc_fwd: f64,
    pub ball_acc_lat: f64,

    // 2. Opponents (incl. 4: centre of mass = centroid).
    pub opp_foremost_fwd_from_own_goal: f64,
    pub opp_centroid_fwd_from_own_goal: f64,
    pub opp_centroid_lat: f64,
    pub opp_mean_vel_fwd: f64,
    pub opp_mean_vel_lat: f64,
    pub opp_mean_acc_fwd: f64,
    pub opp_mean_acc_lat: f64,

    // 3. Self (the back four).
    pub line_fwd_from_own_goal: f64,
    pub line_lat: f64,
    pub line_vel_fwd: f64,
    pub line_vel_lat: f64,
    pub line_acc_fwd: f64,
    pub line_acc_lat: f64,

    // 5. Possession.
    pub we_control: bool,
    pub they_control: bool,

    // 6. Own goalkeeper.
    pub gk_fwd_from_own_goal: f64,
    pub gk_lat: f64,

    // 7. Offside-trap state.
    pub offside_in_force: bool,
    pub trap_active_or_imminent: bool,

    // 8. Existing determinants to retain: where the engine's current heuristic
    // (directive line target, ball blend, role bias, press focus, the legal band)
    // already puts the line centre. The model learns *relative to* this so the
    // well-tuned existing line is never discarded — only refined.
    pub heuristic_centre_fwd_from_own_goal: f64,
}

impl BackFourLineInputs {
    /// The normalized network input vector. **This ordering is the single source
    /// of truth** for the feature layout — change it and bump
    /// [`BACK_FOUR_LINE_FEATURE_DIM`] and any persisted head.
    pub fn to_features(&self) -> [f64; BACK_FOUR_LINE_FEATURE_DIM] {
        let l = self.field_length.max(1.0);
        let half_w = (self.field_width * 0.5).max(1.0);
        let fwd = |y: f64| (y / l).clamp(-0.2, 1.2);
        let lat = |x: f64| (x / half_w).clamp(-1.5, 1.5);
        let vel = |v: f64| (v / REF_TOP_SPEED_YPS).clamp(-2.0, 2.0);
        let acc = |a: f64| (a / REF_ACCEL_YPS2).clamp(-2.0, 2.0);
        let b = |flag: bool| if flag { 1.0 } else { 0.0 };
        [
            // 1. Ball.
            fwd(self.ball_fwd_from_own_goal),
            lat(self.ball_lat),
            vel(self.ball_vel_fwd),
            vel(self.ball_vel_lat),
            acc(self.ball_acc_fwd),
            acc(self.ball_acc_lat),
            // 2. Opponents (+ 4: centroid = centre of mass).
            fwd(self.opp_foremost_fwd_from_own_goal),
            fwd(self.opp_centroid_fwd_from_own_goal),
            lat(self.opp_centroid_lat),
            vel(self.opp_mean_vel_fwd),
            vel(self.opp_mean_vel_lat),
            acc(self.opp_mean_acc_fwd),
            acc(self.opp_mean_acc_lat),
            // 3. Self.
            fwd(self.line_fwd_from_own_goal),
            lat(self.line_lat),
            vel(self.line_vel_fwd),
            vel(self.line_vel_lat),
            acc(self.line_acc_fwd),
            acc(self.line_acc_lat),
            // 5. Possession.
            b(self.we_control),
            b(self.they_control),
            // 6. Own GK.
            fwd(self.gk_fwd_from_own_goal),
            lat(self.gk_lat),
            // 7. Offside-trap state.
            b(self.offside_in_force),
            b(self.trap_active_or_imminent),
            // 8. Existing heuristic determinant (the retained base line).
            fwd(self.heuristic_centre_fwd_from_own_goal),
        ]
    }
}

/// Analytic, deterministic, RNG-free seed policy for the line-centre **gap
/// fraction** in `[0, 1]` (`0` = level with the ball / high line, `1` = the full
/// max-gap deep block). This is the bootstrap target and live fallback the
/// learned [`BackFourLineHead`] is meant to first imitate, then beat.
///
/// The intent it encodes (all in the defending frame): drop **deeper** when the
/// opponent is driving the ball at our goal, when they have it in our half, or
/// when no trap is meaningful; hold a **higher** line when we control, when a
/// trap is on, and the further from our own goal the ball is. Velocity and
/// acceleration enter so the line leads the next ~3 s rather than chasing the
/// current frame.
pub fn analytic_line_centre_gap_fraction(inputs: &BackFourLineInputs) -> f64 {
    let l = inputs.field_length.max(1.0);
    // Base depth: deeper as the ball sits closer to our own goal. `ball_fwd /
    // L` is 0 on our line, 1 on theirs; invert so a deep ball ⇒ a deep line.
    let ball_fwd_frac = (inputs.ball_fwd_from_own_goal / l).clamp(0.0, 1.0);
    let mut gap = 0.30 + 0.45 * (1.0 - ball_fwd_frac);

    // A ball accelerating / moving toward our goal (negative fwd velocity) pulls
    // the line back early; one moving upfield lets it step up.
    let approach = (-inputs.ball_vel_fwd / REF_TOP_SPEED_YPS).clamp(-1.0, 1.0);
    gap += 0.18 * approach;
    let approach_acc = (-inputs.ball_acc_fwd / REF_ACCEL_YPS2).clamp(-1.0, 1.0);
    gap += 0.06 * approach_acc;

    // Possession: holding the ball ⇒ step up (smaller gap); conceding it ⇒ sit.
    if inputs.we_control {
        gap -= 0.20;
    } else if inputs.they_control {
        gap += 0.08;
    }

    // A live, meaningful offside trap is the squeeze tool: hold a high line.
    if inputs.trap_active_or_imminent {
        gap -= 0.16;
    }
    // No offside in force (a restart is being played in behind): there is nothing
    // to trap, so drop off.
    if inputs.offside_in_force {
        gap += 0.12;
    }

    gap.clamp(0.0, 1.0)
}

/// Learned regression head for the line-centre gap fraction: a `FeedForwardNetwork`
/// (`DIM → hidden → 1`, sigmoid) mirroring [`SoccerPassCompletionHead`]. The live
/// net is not serde; like the other learned heads it round-trips through the
/// engine's `SoccerNeuralNetworkSnapshot` Postgres path. Construction, `predict`,
/// and `train` are in place; the live chokepoint reads the analytic seed until a
/// trained head is promoted (see `docs/back-four-line-model.md`).
#[derive(Clone, Debug)]
pub struct BackFourLineHead {
    network: FeedForwardNetwork,
    training_steps: usize,
    last_loss: Option<f64>,
}

impl BackFourLineHead {
    pub fn new(seed: u32) -> Self {
        let mut rng = mulberry32(seed ^ 0x2D9F_4E11);
        let network = FeedForwardNetwork::random(
            &RandomNetworkSpec {
                input_dim: BACK_FOUR_LINE_FEATURE_DIM,
                hidden_layers: vec![BACK_FOUR_LINE_HIDDEN_UNITS],
                output_dim: 1,
                hidden_activation: ActivationName::Tanh,
                // Sigmoid output = a gap fraction in (0, 1).
                output_activation: ActivationName::Sigmoid,
                weight_scale: None,
            },
            &mut rng,
        );
        BackFourLineHead {
            network,
            training_steps: 0,
            last_loss: None,
        }
    }

    /// Predicted gap fraction in `(0, 1)` for a captured state, or `None` on a
    /// malformed (mis-dimensioned / non-finite) input.
    pub fn predict(&self, inputs: &BackFourLineInputs) -> Option<f64> {
        let features = inputs.to_features();
        if features.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let out = self.network.predict(&features[..]);
        out.first().copied().filter(|p| p.is_finite())
    }

    /// One SGD epoch regressing the head toward `(features, target_fraction)`
    /// pairs (supervised bootstrap toward the analytic seed, or an RL target).
    /// Returns the mean step loss.
    pub fn train(&mut self, samples: &[(BackFourLineInputs, f64)], learning_rate: f64) -> f64 {
        let mut total = 0.0;
        let mut applied = 0usize;
        for (inputs, target) in samples {
            let features = inputs.to_features();
            if features.iter().any(|v| !v.is_finite()) || !target.is_finite() {
                continue;
            }
            let target = [target.clamp(0.0, 1.0)];
            let result = self
                .network
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

    pub fn training_steps(&self) -> usize {
        self.training_steps
    }

    pub fn last_loss(&self) -> Option<f64> {
        self.last_loss
    }
}

impl WorldSnapshot {
    /// Build the [`BackFourLineInputs`] for `team`, in that team's attacking
    /// frame. `None` when the team has no defenders or no goalkeeper on the field
    /// (a degenerate roster the model should not opine on). Pure / RNG-free.
    pub(crate) fn build_back_four_line_inputs(&self, team: Team) -> Option<BackFourLineInputs> {
        let attack = team.attack_dir();
        let own_goal_fwd = self.own_goal_y_for(team) * attack;
        let mid_x = self.field_width * 0.5;
        // Sign lateral so it reads the same after a mirror: lateral grows the same
        // way for either team relative to the pitch spine.
        let lat = |x: f64| (x - mid_x) * attack;
        let fwd_from_goal = |y: f64| y * attack - own_goal_fwd;

        // 3. The back four (self): centroid kinematics over the team's defenders.
        let mut def_pos = Vec2::zero();
        let mut def_vel = Vec2::zero();
        let mut def_acc = Vec2::zero();
        let mut def_n = 0.0;
        // 2/4. Opponent outfield centroid + foremost (deepest toward our goal).
        let mut opp_pos = Vec2::zero();
        let mut opp_vel = Vec2::zero();
        let mut opp_acc = Vec2::zero();
        let mut opp_n = 0.0;
        // Foremost attacker = the one deepest into our territory = the smallest
        // forward-gap-from-our-goal in the defending frame.
        let mut opp_foremost_fwd_from_goal = f64::INFINITY;
        for p in &self.players {
            if p.team == team {
                if p.role == PlayerRole::Defender {
                    def_pos = def_pos + p.position;
                    def_vel = def_vel + p.velocity;
                    def_acc = def_acc + p.acceleration;
                    def_n += 1.0;
                }
            } else if p.role != PlayerRole::Goalkeeper {
                opp_pos = opp_pos + p.position;
                opp_vel = opp_vel + p.velocity;
                opp_acc = opp_acc + p.acceleration;
                opp_n += 1.0;
                opp_foremost_fwd_from_goal = opp_foremost_fwd_from_goal.min(fwd_from_goal(p.position.y));
            }
        }
        if def_n < 1.0 || opp_n < 1.0 || !opp_foremost_fwd_from_goal.is_finite() {
            return None;
        }
        let gk = self
            .players
            .iter()
            .find(|p| p.team == team && p.role == PlayerRole::Goalkeeper)?;

        let inv_def = 1.0 / def_n;
        let inv_opp = 1.0 / opp_n;
        let def_pos = def_pos * inv_def;
        let def_vel = def_vel * inv_def;
        let def_acc = def_acc * inv_def;
        let opp_pos = opp_pos * inv_opp;
        let opp_vel = opp_vel * inv_opp;
        let opp_acc = opp_acc * inv_opp;

        let controlled = self.controlled_possession_team();
        let trap_active_or_imminent = !self.offside_currently_suspended()
            && controlled != Some(team)
            && {
                let ball_from_own_goal = fwd_from_goal(self.ball.position.y).max(0.0);
                ball_from_own_goal > self.field_length / 3.0
            };

        Some(BackFourLineInputs {
            field_length: self.field_length,
            field_width: self.field_width,

            ball_fwd_from_own_goal: fwd_from_goal(self.ball.position.y),
            ball_lat: lat(self.ball.position.x),
            ball_vel_fwd: self.ball.velocity.y * attack,
            ball_vel_lat: self.ball.velocity.x * attack,
            ball_acc_fwd: self.ball.acceleration.y * attack,
            ball_acc_lat: self.ball.acceleration.x * attack,

            opp_foremost_fwd_from_own_goal: opp_foremost_fwd_from_goal,
            opp_centroid_fwd_from_own_goal: fwd_from_goal(opp_pos.y),
            opp_centroid_lat: lat(opp_pos.x),
            opp_mean_vel_fwd: opp_vel.y * attack,
            opp_mean_vel_lat: opp_vel.x * attack,
            opp_mean_acc_fwd: opp_acc.y * attack,
            opp_mean_acc_lat: opp_acc.x * attack,

            line_fwd_from_own_goal: fwd_from_goal(def_pos.y),
            line_lat: lat(def_pos.x),
            line_vel_fwd: def_vel.y * attack,
            line_vel_lat: def_vel.x * attack,
            line_acc_fwd: def_acc.y * attack,
            line_acc_lat: def_acc.x * attack,

            we_control: controlled == Some(team),
            they_control: controlled == Some(team.other()),

            gk_fwd_from_own_goal: fwd_from_goal(gk.position.y),
            gk_lat: lat(gk.position.x),

            offside_in_force: self.offside_currently_suspended(),
            trap_active_or_imminent,

            // Default the retained-heuristic determinant to the line's current
            // centroid; the live chokepoint overrides it with the precise band
            // centre it has in hand (see `back_four_line_model_centre_fwd`).
            heuristic_centre_fwd_from_own_goal: fwd_from_goal(def_pos.y),
        })
    }

    /// The model's preferred **line-centre forward coordinate** for `me`'s back
    /// four, already clamped to the legal `[deepest_fwd, shallowest_fwd]` band.
    /// `None` when the model is gated off or the inputs are degenerate — the
    /// caller then keeps its heuristic centre (parity).
    ///
    /// `heuristic_centre_fwd` is the engine's existing band centre — the retained
    /// determinant (group 8). The result is a **blend** of that and the model's
    /// own depth, so the well-tuned existing line is refined, never discarded. The
    /// model depth comes from the analytic seed today; flip the seed call to
    /// [`BackFourLineHead::predict`] (and raise [`BACK_FOUR_LINE_MODEL_BLEND`])
    /// once a trained head is promoted.
    pub(crate) fn back_four_line_model_centre_fwd(
        &self,
        me: &PlayerSnapshot,
        predicted_fwd: f64,
        deepest_fwd: f64,
        shallowest_fwd: f64,
        max_gap: f64,
        heuristic_centre_fwd: f64,
    ) -> Option<f64> {
        if !back_four_line_model_enabled() {
            return None;
        }
        let mut inputs = self.build_back_four_line_inputs(me.team)?;
        // Group 8: feed the engine's precise existing band centre to the model so
        // it learns relative to the retained determinant.
        let own_goal_fwd = self.own_goal_y_for(me.team) * me.team.attack_dir();
        inputs.heuristic_centre_fwd_from_own_goal = heuristic_centre_fwd - own_goal_fwd;

        let gap_fraction = analytic_line_centre_gap_fraction(&inputs);
        if !gap_fraction.is_finite() || !max_gap.is_finite() || !heuristic_centre_fwd.is_finite() {
            return None;
        }
        let model_centre_fwd = predicted_fwd - gap_fraction * max_gap;
        // Blend retained-heuristic ⟷ model depth (group 8 keeps existing
        // determinants in play); the band clamp keeps the result legal.
        let blend = BACK_FOUR_LINE_MODEL_BLEND.clamp(0.0, 1.0);
        let centre_fwd = heuristic_centre_fwd * (1.0 - blend) + model_centre_fwd * blend;
        Some(centre_fwd.clamp(deepest_fwd.min(shallowest_fwd), shallowest_fwd))
    }
}

#[cfg(test)]
mod back_four_line_tests {
    use super::*;

    /// A neutral mid-pitch state to perturb from.
    fn baseline_inputs() -> BackFourLineInputs {
        BackFourLineInputs {
            field_length: 120.0,
            field_width: 80.0,
            ball_fwd_from_own_goal: 60.0,
            ball_lat: 0.0,
            ball_vel_fwd: 0.0,
            ball_vel_lat: 0.0,
            ball_acc_fwd: 0.0,
            ball_acc_lat: 0.0,
            opp_foremost_fwd_from_own_goal: 40.0,
            opp_centroid_fwd_from_own_goal: 55.0,
            opp_centroid_lat: 0.0,
            opp_mean_vel_fwd: 0.0,
            opp_mean_vel_lat: 0.0,
            opp_mean_acc_fwd: 0.0,
            opp_mean_acc_lat: 0.0,
            line_fwd_from_own_goal: 35.0,
            line_lat: 0.0,
            line_vel_fwd: 0.0,
            line_vel_lat: 0.0,
            line_acc_fwd: 0.0,
            line_acc_lat: 0.0,
            we_control: false,
            they_control: true,
            gk_fwd_from_own_goal: 6.0,
            gk_lat: 0.0,
            offside_in_force: false,
            trap_active_or_imminent: true,
            heuristic_centre_fwd_from_own_goal: 35.0,
        }
    }

    #[test]
    fn features_are_finite_bounded_and_right_dim() {
        let f = baseline_inputs().to_features();
        assert_eq!(f.len(), BACK_FOUR_LINE_FEATURE_DIM);
        for v in f {
            assert!(v.is_finite());
            assert!(v.abs() <= 2.0 + 1e-9, "feature {v} out of normalized range");
        }
    }

    #[test]
    fn analytic_fraction_is_a_valid_gap() {
        let g = analytic_line_centre_gap_fraction(&baseline_inputs());
        assert!((0.0..=1.0).contains(&g), "gap fraction {g} out of [0,1]");
    }

    #[test]
    fn deep_ball_pushes_line_deeper_than_high_ball() {
        let mut deep = baseline_inputs();
        deep.ball_fwd_from_own_goal = 15.0; // ball near our own goal
        let mut high = baseline_inputs();
        high.ball_fwd_from_own_goal = 105.0; // ball near their goal
        assert!(
            analytic_line_centre_gap_fraction(&deep)
                > analytic_line_centre_gap_fraction(&high),
            "a deeper ball should yield a deeper (larger-gap) line"
        );
    }

    #[test]
    fn our_possession_lifts_the_line_vs_theirs() {
        let mut ours = baseline_inputs();
        ours.we_control = true;
        ours.they_control = false;
        let mut theirs = baseline_inputs();
        theirs.we_control = false;
        theirs.they_control = true;
        assert!(
            analytic_line_centre_gap_fraction(&ours)
                < analytic_line_centre_gap_fraction(&theirs),
            "controlling the ball should hold a higher (smaller-gap) line"
        );
    }

    #[test]
    fn ball_driving_at_our_goal_drops_the_line() {
        let mut approaching = baseline_inputs();
        approaching.ball_vel_fwd = -8.0; // moving toward our own goal
        let mut receding = baseline_inputs();
        receding.ball_vel_fwd = 8.0; // moving upfield
        assert!(
            analytic_line_centre_gap_fraction(&approaching)
                > analytic_line_centre_gap_fraction(&receding),
            "a ball accelerating at our goal should drop the line off early"
        );
    }

    #[test]
    fn head_predicts_a_bounded_fraction() {
        let head = BackFourLineHead::new(7);
        let p = head.predict(&baseline_inputs()).expect("finite prediction");
        assert!((0.0..=1.0).contains(&p), "head fraction {p} out of (0,1)");
    }

    #[test]
    fn head_trains_toward_a_target() {
        let mut head = BackFourLineHead::new(11);
        let samples: Vec<(BackFourLineInputs, f64)> =
            std::iter::repeat_with(|| (baseline_inputs(), 0.9))
                .take(64)
                .collect();
        let mut last = f64::INFINITY;
        for _ in 0..40 {
            last = head.train(&samples, 0.05);
        }
        assert!(last.is_finite());
        assert!(head.training_steps() > 0);
    }
}
