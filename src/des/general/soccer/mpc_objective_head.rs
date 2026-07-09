//! Neural MPC execution-objective head — the *executor* half of the selector-vs-executor ceiling.
//!
//! The POMDP/actor picks the action FAMILY and semantic destination; the analytic layer +
//! point-mass MPC execute it, so pass/shoot/dribble *quality* (aim, lead point, arrival pocket)
//! is not in the learnable space. This head learns a BOUNDED RESIDUAL on the execution target
//! from the whole-field 256-d config embedding + a few execution-context scalars, so chance
//! creation QUALITY becomes learnable while the MPC still guarantees dynamic feasibility.
//!
//! Design (locked with Codex, round 9): output ONLY a bounded (forward, lateral) target residual
//! (±`MPC_OBJECTIVE_MAX_RESIDUAL_YARDS`), applied to the ball-carrier's analytic shot-aim /
//! pass-lead target and re-clamped through the existing pitch/onside/speed guards. This preserves
//! the "policy owns WHERE" reconcile contract: the head may nudge the aim/lead a yard, never
//! change the receiver, the action family, or send the actor to an unrelated point. Trained by
//! reward-weighted regression (contextual bandit) on the delayed chance/outcome reward — the same
//! terminal signal the chance-quality reward uses. Default-on in production once installed, with
//! the gate's falsey values as the kill switch; tests stay env-explicit for parity.
//!
//! Mirrors [`super::SoccerPassCompletionHead`]: a shared `FeedForwardNetwork` over the config
//! embedding, live net not serde (round-tripped through the neural-snapshot path for persistence).

use crate::des::general::des_base::neural_network::NeuralNetworkLike;
use crate::des::general::neural_network::{ActivationName, FeedForwardNetwork, RandomNetworkSpec};
use crate::des::general::prng::mulberry32;

use super::{
    build_soccer_feed_forward_network_from_snapshot, soccer_neural_network_snapshot,
    SoccerAuxiliaryHeadSnapshot, Vec2, SOCCER_MOMENT_EMBEDDING_DIM,
};

/// Execution-context scalars appended to the 256-d field embedding: action-family flags
/// (shot/pass/dribble), normalized distance-to-target, forward/lateral components of the analytic
/// target delta, angle-to-goal, and pressure proxy. Kept small and slow-varying.
pub const MPC_OBJECTIVE_EXEC_FEATURES: usize = 8;
pub const MPC_OBJECTIVE_FEATURE_DIM: usize =
    SOCCER_MOMENT_EMBEDDING_DIM + MPC_OBJECTIVE_EXEC_FEATURES;
const MPC_OBJECTIVE_HIDDEN_UNITS: usize = 32;
/// Hard bound (yards) on the residual — the guardrail that keeps the executor inside the
/// "policy owns WHERE" contract. Codex round-9: ~1.0–1.5 yd for a first cut.
pub const MPC_OBJECTIVE_MAX_RESIDUAL_YARDS: f64 = 1.5;
/// Hard bound (yards) on the learned SIGNED bend — the 3rd output dim, present only when the head
/// is built bend-enabled (`DD_SOCCER_ENABLE_LEARNED_CURVE`). Positive = curl one way, negative the
/// other; `|value|` = lateral yards the flight bows off the straight chord. Sized so a warmed head
/// can steer a pass/shot AROUND a defender on the straight lane (the "knight-move"), while the ball
/// physics still clamp the realised curl to `MAX_BALL_CURL_YPS2`.
pub const MPC_OBJECTIVE_MAX_BEND_YARDS: f64 = 6.0;
/// Exploration std-dev (yards) for the bend axis. Larger than the aim-residual sigma because bend
/// only helps in the minority of states with a blocked straight lane, so it needs a wider search to
/// find the states where curling pays.
pub const MPC_OBJECTIVE_BEND_EXPLORE_SIGMA_YARDS: f64 = 1.2;
/// Default exploration std-dev (yards) added to the greedy residual at capture time. Without
/// exploration the RWR loop only ever imitates the residual it already took, so it can never
/// DISCOVER a better aim — the jitter is what gives the contextual bandit a gradient to climb.
/// Kept below the hard bound so a jittered residual rarely rails. Decays toward 0 as the head
/// warms (see the exec-side schedule); this is the un-warmed ceiling.
pub const MPC_OBJECTIVE_EXPLORE_SIGMA_YARDS: f64 = 0.6;
/// Live use falls back to the pure analytic target until the head has warmed this many steps
/// (mirrors the support-scorer / pass-completion warm thresholds).
pub const MPC_OBJECTIVE_MIN_TRAINING_STEPS: usize = 200;
/// Rolling cap on retained RWR samples.
pub const MPC_OBJECTIVE_SAMPLE_CAP: usize = 4096;

/// One reward-weighted-regression example: the captured features, the (already-bounded) residual
/// that was actually applied at execution (the "action"), and the delayed advantage it earned
/// (the RWR weight). A positive advantage reinforces the residual taken; non-positive is skipped.
#[derive(Clone, Debug)]
pub struct MpcObjectiveSample {
    pub features: Vec<f32>,
    pub applied_residual: Vec2,
    /// Signed bend (yards) actually applied at execution, or `0.0` when the head is not
    /// bend-enabled. Trained as the 3rd output dim exactly like the aim residual: a completed pass
    /// reinforces the bend taken, an interception is skipped (RWR keeps positive advantage only).
    pub applied_bend: f64,
    pub reward: f64,
}

/// Which execution family the residual is shaping. Sets the one-hot flags in the exec-context
/// feature block so a SINGLE head can serve shots, passes, and dribbles without conflating them —
/// the analytic seed and reward attribution differ per family, but the geometry (bounded aim/lead
/// nudge under the optimizer's feasibility guard) is identical.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MpcObjectiveFamily {
    Shot,
    Pass,
    Dribble,
}

impl MpcObjectiveFamily {
    /// `(is_shot, is_pass, is_dribble)` one-hot flags, in that fixed order.
    pub fn one_hot(self) -> [f32; 3] {
        match self {
            MpcObjectiveFamily::Shot => [1.0, 0.0, 0.0],
            MpcObjectiveFamily::Pass => [0.0, 1.0, 0.0],
            MpcObjectiveFamily::Dribble => [0.0, 0.0, 1.0],
        }
    }
}

/// Learned execution-objective head: MLP over `[field embedding ++ exec features]` → a bounded
/// `(forward, lateral)` target residual (Tanh output × `MAX_RESIDUAL` = hard-bounded yards).
#[derive(Clone, Debug)]
pub struct SoccerMpcObjectiveHead {
    network: FeedForwardNetwork,
    training_steps: usize,
    last_loss: Option<f64>,
    /// When true the network has a 3rd output dim = signed bend (yards). When false the head is
    /// byte-identical to the original 2-output aim/lead head — same shape, same RNG consumption,
    /// so the `DD_SOCCER_ENABLE_LEARNED_CURVE`-off path is unchanged.
    bend_enabled: bool,
}

impl SoccerMpcObjectiveHead {
    /// Original 2-output (aim/lead only) head. Bend disabled — byte-identical to prior behavior.
    pub fn new(seed: u32) -> Self {
        Self::new_with_bend(seed, false)
    }

    /// Build the head with an optional 3rd learnable bend output. `bend_enabled=false` reproduces
    /// [`Self::new`] exactly (output_dim 2, same init draw order).
    pub fn new_with_bend(seed: u32, bend_enabled: bool) -> Self {
        let mut rng = mulberry32(seed ^ 0x4D50_4301);
        let output_dim = if bend_enabled { 3 } else { 2 };
        let network = FeedForwardNetwork::random(
            &RandomNetworkSpec {
                input_dim: MPC_OBJECTIVE_FEATURE_DIM,
                hidden_layers: vec![MPC_OBJECTIVE_HIDDEN_UNITS],
                output_dim,
                hidden_activation: ActivationName::Relu,
                // Tanh → each output in (-1, 1); scaled by MAX_RESIDUAL / MAX_BEND for a hard bound.
                output_activation: ActivationName::Tanh,
                weight_scale: None,
            },
            &mut rng,
        );
        SoccerMpcObjectiveHead {
            network,
            training_steps: 0,
            last_loss: None,
            bend_enabled,
        }
    }

    /// Whether this head carries the learnable bend (3rd output) axis.
    pub fn bend_enabled(&self) -> bool {
        self.bend_enabled
    }

    pub fn training_steps(&self) -> usize {
        self.training_steps
    }

    pub fn last_loss(&self) -> Option<f64> {
        self.last_loss
    }

    /// Persist the learned executor objective alongside the main neural snapshot.
    pub fn to_snapshot(&self) -> SoccerAuxiliaryHeadSnapshot {
        let mut network = soccer_neural_network_snapshot(&self.network);
        network.training_steps = self.training_steps;
        network.average_loss = self.last_loss;
        SoccerAuxiliaryHeadSnapshot {
            network,
            training_steps: self.training_steps,
            average_loss: self.last_loss,
        }
    }

    /// Restore a persisted executor objective head. Output dim 2 is the legacy aim/lead
    /// head; output dim 3 carries the learned bend axis.
    pub fn from_snapshot(snapshot: &SoccerAuxiliaryHeadSnapshot) -> Result<Self, String> {
        let output_dim = snapshot.network.output_dim;
        let bend_enabled = match output_dim {
            2 => false,
            3 => true,
            _ => {
                return Err(format!(
                    "mpc objective snapshot output_dim {output_dim} must be 2 or 3"
                ));
            }
        };
        let network = build_soccer_feed_forward_network_from_snapshot(
            &snapshot.network,
            MPC_OBJECTIVE_FEATURE_DIM,
            output_dim,
            &[],
            "mpc objective",
        )?;
        Ok(SoccerMpcObjectiveHead {
            network,
            training_steps: snapshot.training_steps.max(snapshot.network.training_steps),
            last_loss: snapshot.average_loss.or(snapshot.network.average_loss),
            bend_enabled,
        })
    }

    /// Whether the head is warm enough for live use (else callers keep the pure analytic target).
    pub fn is_warm(&self) -> bool {
        self.training_steps >= MPC_OBJECTIVE_MIN_TRAINING_STEPS
    }

    /// Bounded `(forward=y, lateral=x)` residual in yards for the captured features, or `None` on
    /// malformed input. Callers add this to the analytic target, then re-clamp through the
    /// pitch/onside/speed guards — the residual never bypasses them.
    pub fn predict_residual(&self, features: &[f32]) -> Option<Vec2> {
        if features.len() != MPC_OBJECTIVE_FEATURE_DIM {
            return None;
        }
        let input: Vec<f64> = features.iter().map(|&v| f64::from(v)).collect();
        let out = self.network.predict(&input);
        let fwd = *out.first()?;
        let lat = *out.get(1)?;
        if !fwd.is_finite() || !lat.is_finite() {
            return None;
        }
        Some(Vec2 {
            x: lat.clamp(-1.0, 1.0) * MPC_OBJECTIVE_MAX_RESIDUAL_YARDS,
            y: fwd.clamp(-1.0, 1.0) * MPC_OBJECTIVE_MAX_RESIDUAL_YARDS,
        })
    }

    /// Exploration wrapper around [`predict_residual`]. Adds Gaussian jitter — the caller supplies
    /// the two standard-normal draws (`noise_fwd`, `noise_lat`) from the match RNG so the head
    /// stays deterministic and rng-free, exactly like the rest of the engine threads randomness —
    /// of std-dev `sigma_yards`, then RE-CLAMPS to the hard `MAX_RESIDUAL` bound so an explored
    /// residual can never breach the "policy owns WHERE" contract. **Capture the RETURNED residual
    /// as the RWR `applied_residual`**: reinforcing the explored action (not the greedy prediction)
    /// is what lets the contextual bandit climb. `None` on malformed input, like `predict_residual`.
    /// Non-finite noise is treated as zero (falls back to the greedy residual) rather than poisoning
    /// the executor with a NaN target.
    pub fn explore_residual(
        &self,
        features: &[f32],
        sigma_yards: f64,
        noise_fwd: f64,
        noise_lat: f64,
    ) -> Option<Vec2> {
        let base = self.predict_residual(features)?;
        let sigma = if sigma_yards.is_finite() {
            sigma_yards.max(0.0)
        } else {
            0.0
        };
        let jitter_y = if noise_fwd.is_finite() {
            noise_fwd * sigma
        } else {
            0.0
        };
        let jitter_x = if noise_lat.is_finite() {
            noise_lat * sigma
        } else {
            0.0
        };
        Some(Vec2 {
            x: (base.x + jitter_x).clamp(
                -MPC_OBJECTIVE_MAX_RESIDUAL_YARDS,
                MPC_OBJECTIVE_MAX_RESIDUAL_YARDS,
            ),
            y: (base.y + jitter_y).clamp(
                -MPC_OBJECTIVE_MAX_RESIDUAL_YARDS,
                MPC_OBJECTIVE_MAX_RESIDUAL_YARDS,
            ),
        })
    }

    /// Greedy signed bend (yards) for the captured features, or `None` when the head is not
    /// bend-enabled or on malformed input. Positive/negative = curl side; `|value|` = lateral yards
    /// the flight bows off the straight chord. The caller maps this to the kick's curve direction +
    /// `curve_bend_yards`, which the existing lowering turns into `curl_acceleration`.
    pub fn predict_bend(&self, features: &[f32]) -> Option<f64> {
        if !self.bend_enabled || features.len() != MPC_OBJECTIVE_FEATURE_DIM {
            return None;
        }
        let input: Vec<f64> = features.iter().map(|&v| f64::from(v)).collect();
        let out = self.network.predict(&input);
        let bend = *out.get(2)?;
        if !bend.is_finite() {
            return None;
        }
        Some(bend.clamp(-1.0, 1.0) * MPC_OBJECTIVE_MAX_BEND_YARDS)
    }

    /// Exploration wrapper around [`predict_bend`] — one standard-normal draw from the match RNG,
    /// scaled by `sigma_yards`, re-clamped to the hard bend bound. Capture the RETURNED bend as the
    /// RWR `applied_bend`. `None` when the head is not bend-enabled or on malformed input; non-finite
    /// noise falls back to the greedy bend (never a NaN target).
    pub fn explore_bend(&self, features: &[f32], sigma_yards: f64, noise: f64) -> Option<f64> {
        let base = self.predict_bend(features)?;
        let sigma = if sigma_yards.is_finite() {
            sigma_yards.max(0.0)
        } else {
            0.0
        };
        let jitter = if noise.is_finite() {
            noise * sigma
        } else {
            0.0
        };
        Some((base + jitter).clamp(-MPC_OBJECTIVE_MAX_BEND_YARDS, MPC_OBJECTIVE_MAX_BEND_YARDS))
    }

    /// Reward-weighted regression: nudge the head toward residuals that earned positive advantage,
    /// with the learning rate scaled by a squashed advantage (a contextual bandit, not full policy
    /// gradient — lower variance, harder to destabilize). Returns the number of steps applied.
    pub fn train_rwr(&mut self, samples: &[MpcObjectiveSample], base_lr: f64) -> usize {
        let mut trained = 0usize;
        for sample in samples {
            if sample.features.len() != MPC_OBJECTIVE_FEATURE_DIM {
                continue;
            }
            let weight = sample.reward.tanh();
            if weight <= 0.0 {
                continue;
            }
            let input: Vec<f64> = sample.features.iter().map(|&v| f64::from(v)).collect();
            let target_y =
                (sample.applied_residual.y / MPC_OBJECTIVE_MAX_RESIDUAL_YARDS).clamp(-0.999, 0.999);
            let target_x =
                (sample.applied_residual.x / MPC_OBJECTIVE_MAX_RESIDUAL_YARDS).clamp(-0.999, 0.999);
            let lr = (base_lr * weight).min(base_lr);
            // Target vector matches the network's output_dim: 2 (aim only) or 3 (aim + bend).
            let targets: Vec<f64> = if self.bend_enabled {
                let target_bend =
                    (sample.applied_bend / MPC_OBJECTIVE_MAX_BEND_YARDS).clamp(-0.999, 0.999);
                vec![target_y, target_x, target_bend]
            } else {
                vec![target_y, target_x]
            };
            let _ = self.network.train_sample_clipped(&input, &targets, lr, 4.0);
            self.training_steps += 1;
            trained += 1;
        }
        trained
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predict_residual_is_hard_bounded() {
        let head = SoccerMpcObjectiveHead::new(7);
        let features = vec![0.5f32; MPC_OBJECTIVE_FEATURE_DIM];
        let residual = head.predict_residual(&features).expect("residual");
        assert!(residual.x.abs() <= MPC_OBJECTIVE_MAX_RESIDUAL_YARDS + 1e-9);
        assert!(residual.y.abs() <= MPC_OBJECTIVE_MAX_RESIDUAL_YARDS + 1e-9);
    }

    #[test]
    fn malformed_input_returns_none() {
        let head = SoccerMpcObjectiveHead::new(7);
        assert!(head.predict_residual(&[0.1f32, 0.2]).is_none());
    }

    #[test]
    fn rwr_only_reinforces_positive_advantage() {
        let mut head = SoccerMpcObjectiveHead::new(9);
        let good = MpcObjectiveSample {
            features: vec![0.3f32; MPC_OBJECTIVE_FEATURE_DIM],
            applied_residual: Vec2 { x: 0.8, y: 1.0 },
            applied_bend: 0.0,
            reward: 1.5,
        };
        let bad = MpcObjectiveSample {
            features: vec![0.3f32; MPC_OBJECTIVE_FEATURE_DIM],
            applied_residual: Vec2 { x: -0.8, y: -1.0 },
            applied_bend: 0.0,
            reward: -1.5,
        };
        let trained = head.train_rwr(&[good, bad], 0.05);
        assert_eq!(trained, 1, "only the positive-advantage sample trains");
        assert!(head.training_steps() == 1);
    }

    #[test]
    fn explore_residual_stays_hard_bounded_under_large_jitter() {
        let head = SoccerMpcObjectiveHead::new(11);
        let features = vec![0.4f32; MPC_OBJECTIVE_FEATURE_DIM];
        // Huge noise * huge sigma must still re-clamp inside the bound (never breach the contract).
        let residual = head
            .explore_residual(&features, 100.0, 50.0, -50.0)
            .expect("residual");
        assert!(residual.x.abs() <= MPC_OBJECTIVE_MAX_RESIDUAL_YARDS + 1e-9);
        assert!(residual.y.abs() <= MPC_OBJECTIVE_MAX_RESIDUAL_YARDS + 1e-9);
    }

    #[test]
    fn bend_axis_only_present_when_enabled() {
        // Default head has no bend axis: predict_bend is None, byte-identical 2-output behavior.
        let plain = SoccerMpcObjectiveHead::new(21);
        assert!(!plain.bend_enabled());
        let features = vec![0.4f32; MPC_OBJECTIVE_FEATURE_DIM];
        assert!(plain.predict_bend(&features).is_none());

        // Bend-enabled head yields a hard-bounded signed bend.
        let bent = SoccerMpcObjectiveHead::new_with_bend(21, true);
        assert!(bent.bend_enabled());
        let bend = bent.predict_bend(&features).expect("bend");
        assert!(bend.abs() <= MPC_OBJECTIVE_MAX_BEND_YARDS + 1e-9);
        // Explore stays inside the bound under absurd jitter.
        let explored = bent
            .explore_bend(&features, 100.0, 50.0)
            .expect("explored bend");
        assert!(explored.abs() <= MPC_OBJECTIVE_MAX_BEND_YARDS + 1e-9);
    }

    #[test]
    fn rwr_trains_bend_axis_on_positive_advantage() {
        let mut head = SoccerMpcObjectiveHead::new_with_bend(23, true);
        let sample = MpcObjectiveSample {
            features: vec![0.3f32; MPC_OBJECTIVE_FEATURE_DIM],
            applied_residual: Vec2 { x: 0.5, y: 0.5 },
            applied_bend: MPC_OBJECTIVE_MAX_BEND_YARDS * 0.8,
            reward: 1.5,
        };
        let before = head.predict_bend(&sample.features).expect("bend before");
        // Many RWR steps toward a strong positive bend should move the prediction that way.
        for _ in 0..200 {
            head.train_rwr(std::slice::from_ref(&sample), 0.1);
        }
        let after = head.predict_bend(&sample.features).expect("bend after");
        assert!(
            after > before,
            "positive-advantage bend target should raise the predicted bend: {before} -> {after}"
        );
    }

    #[test]
    fn snapshot_roundtrip_preserves_training_progress_and_shape() {
        let mut head = SoccerMpcObjectiveHead::new_with_bend(29, true);
        let sample = MpcObjectiveSample {
            features: vec![0.2f32; MPC_OBJECTIVE_FEATURE_DIM],
            applied_residual: Vec2 { x: 0.25, y: 0.75 },
            applied_bend: MPC_OBJECTIVE_MAX_BEND_YARDS * 0.25,
            reward: 1.0,
        };
        head.train_rwr(&[sample], 0.05);

        let snapshot = head.to_snapshot();
        let restored = SoccerMpcObjectiveHead::from_snapshot(&snapshot).expect("restore");

        assert!(restored.bend_enabled());
        assert_eq!(restored.training_steps(), head.training_steps());
        assert_eq!(snapshot.network.output_dim, 3);
    }

    #[test]
    fn explore_residual_nan_noise_falls_back_to_greedy() {
        let head = SoccerMpcObjectiveHead::new(13);
        let features = vec![0.4f32; MPC_OBJECTIVE_FEATURE_DIM];
        let greedy = head.predict_residual(&features).expect("greedy");
        let explored = head
            .explore_residual(&features, f64::NAN, f64::NAN, f64::NAN)
            .expect("residual");
        // Non-finite sigma/noise ⇒ zero jitter ⇒ identical to the greedy residual (no NaN target).
        assert!((explored.x - greedy.x).abs() < 1e-9);
        assert!((explored.y - greedy.y).abs() < 1e-9);
    }
}
