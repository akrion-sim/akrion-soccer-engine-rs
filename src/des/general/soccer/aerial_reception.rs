//! Anticipating and controlling a **long aerial pass as it lands** — the descent
//! geometry, the MDP/POMDP *attack-the-ball* decision, and a learnable control head.
//!
//! ## The gap this closes
//!
//! A lofted ball spends most of its flight high overhead (apex ~20-27 ft) and only
//! becomes *receivable* in the last fraction of a second, as it drops back through the
//! chest/head band (~8 ft) down to the foot (~5 ft and below). The existing receiver
//! election ([`WorldSnapshot::pending_pass_reception_target_for`]) samples points along
//! a **flat horizontal projection** of the ball velocity and pulls the receiver to meet
//! it *as early as possible* — which, for a high ball, means camping under a ball that
//! is still sailing 20 ft overhead. A real player instead reads the **arc**: they run to
//! where the ball will drop into reach and time the touch, choosing to *attack it in the
//! air* (step up and head/chest it before a defender) or *settle under it* (let it drop
//! to the feet) based on pressure and how cleanly they can get there.
//!
//! ## What this module computes
//!
//! * [`aerial_descent_plan`] — pure projectile geometry. From the arc's apex + elapsed
//!   flight time (the vertical schedule) and the **current** ball position + horizontal
//!   velocity (the horizontal projection, so it inherits the drag already bled off in
//!   flight rather than assuming the launch pace), it returns the two horizontal points
//!   where the **descending** ball passes through the control band: the *top* (~8 ft —
//!   where a header/jump meets it) and the *sweet* height (~5 ft — where a chest/thigh
//!   touch settles it), plus the times the ball reaches each and the bare landing point.
//!   This is the trajectory the receiver anticipates.
//! * [`decide_aerial_reception`] — the per-receiver **MDP/POMDP** decision. Given the
//!   descent plan and the receiver's kinematics + the nearest opponent's race to the drop
//!   zone, it picks an [`AerialReceptionDecision`] (attack it high, settle under it, or
//!   chase a ball dropping in front) and the exact point + height to take it at. Reaching
//!   the drop zone in time under speed/accel limits is an individual-MPC-style reachability
//!   check; the choice of *which* height to take it at trades contest-margin against clean
//!   control — the partially-observed part is the opponent's intent, proxied by their race.
//! * [`AerialReceptionControlHead`] — a learnable `FeedForwardNetwork` head that refines
//!   the one free knob of that decision: the **attack height blend** in `[0, 1]` (0 = settle
//!   it at the feet, 1 = attack it at the top of the band). Trained from self-play
//!   [`AerialReceptionSample`]s whose reward is *whether the aerial pass was actually
//!   brought under clean control* — so the team learns how high to take a dropping ball
//!   to control it given the trajectory, their tools, and the pressure.
//!
//! ## Parity
//!
//! The descent geometry + analytic decision are pure and RNG-free; they are wired into the
//! receiver election behind [`aerial_reception_anticipation_enabled`]
//! (`DD_SOCCER_DISABLE_AERIAL_RECEPTION_ANTICIPATION` to turn **off** ⇒ byte-identical to
//! the old flat-projection election). The learnable head is consulted only when
//! [`aerial_reception_model_enabled`] (`DD_SOCCER_ENABLE_LEARNED_AERIAL_RECEPTION`) is set
//! and a trained head is present; unset ⇒ the analytic seed stands. Mirrors
//! [`super::spacing_target`] / [`super::loose_ball_commit`].

use super::*;
use crate::des::general::neural_network::{ActivationName, FeedForwardNetwork, RandomNetworkSpec};
use crate::des::general::prng::mulberry32;

/// Top of the controllable descent band, in yards (~8 ft): the height at which a
/// descending ball first comes into chest/head reach. A player who *attacks* the ball
/// meets it here, in the air, before it drops to a defender.
pub const AERIAL_CONTROL_BAND_TOP_YARDS: f64 = 2.667; // 8 ft
/// The "settle" height, in yards (~5 ft): chest/thigh height where an unpressured
/// receiver lets the ball drop to take a clean, cushioned first touch toward space.
pub const AERIAL_CONTROL_BAND_SWEET_YARDS: f64 = 1.667; // 5 ft

/// Minimum apex (yards) for a pass to count as a genuine *lofted* ball whose descent
/// must be anticipated. Below this the ball never climbs out of the control band, so the
/// flat-projection election already meets it correctly and this module stands aside.
pub const AERIAL_ANTICIPATION_MIN_APEX_YARDS: f64 = AERIAL_CONTROL_BAND_TOP_YARDS + 0.4;

/// How much earlier than the ball the receiver must arrive at the drop zone to count as
/// comfortably under it (and so free to settle it rather than attack it on the move).
const AERIAL_SETTLE_SLACK_SECONDS: f64 = 0.35;
/// Contest margin (seconds the receiver beats the nearest opponent to the drop zone) at
/// or below which the reception is treated as pressured — attack the ball high to win it.
const AERIAL_CONTEST_MARGIN_SECONDS: f64 = 0.55;
/// Grace beyond the sweet-height arrival time within which a slightly-late receiver can
/// still chase the dropping ball (attack on the move) rather than abandon it.
const AERIAL_CHASE_TOLERANCE_SECONDS: f64 = 0.55;

const AERIAL_RECEPTION_ENABLE_ENV: &str = "DD_SOCCER_ENABLE_LEARNED_AERIAL_RECEPTION";
const AERIAL_RECEPTION_DISABLE_ENV: &str = "DD_SOCCER_DISABLE_AERIAL_RECEPTION_ANTICIPATION";

/// Whether descent-aware aerial reception anticipation is applied at the receiver
/// election. **On by default** (the realistic behaviour); set
/// `DD_SOCCER_DISABLE_AERIAL_RECEPTION_ANTICIPATION` to fall back to the flat-projection
/// election (byte-identical baseline / A-B).
pub fn aerial_reception_anticipation_enabled() -> bool {
    use std::sync::OnceLock;
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var(AERIAL_RECEPTION_DISABLE_ENV).is_err())
}

/// Whether the learned aerial-reception control head refines the attack-height blend this
/// process. Off (unset) ⇒ the analytic seed stands, so play is byte-identical to the
/// pure-geometry decision.
pub fn aerial_reception_model_enabled() -> bool {
    use std::sync::OnceLock;
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var(AERIAL_RECEPTION_ENABLE_ENV).is_ok())
}

/// Number of features in the aerial-reception context vector. The ordering is the single
/// source of truth in [`AerialReceptionContext::to_features`] — change it and bump this
/// constant and any persisted head.
pub const AERIAL_RECEPTION_FEATURE_DIM: usize = 11;
/// Hidden width of the learned regression head.
pub const AERIAL_RECEPTION_HIDDEN_UNITS: usize = 16;

/// How often (ticks) an aerial-reception decision is sampled for RL while the model is on.
pub const AERIAL_RECEPTION_SAMPLE_INTERVAL_TICKS: u64 = 6;
/// Cap on the rolling RL sample buffer (drained by the learner).
pub const AERIAL_RECEPTION_SAMPLE_CAP: usize = 4096;

// Normalization references for the network input (raw -> ~unit range).
const REF_APEX_YARDS: f64 = 9.0;
const REF_DIST_YARDS: f64 = 60.0;
const REF_TIME_SECONDS: f64 = 1.2;
const REF_RUN_YARDS: f64 = 18.0;
const REF_BALL_SPEED_YPS: f64 = 24.0;

/// The two horizontal points (and their timing) where a descending lofted ball passes
/// through the control band, plus the bare landing point — the geometry a receiver
/// anticipates. All times are **seconds from now** (the moment of the decision).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AerialDescentPlan {
    /// Horizontal point where the ball drops through the **top** of the band (~8 ft) —
    /// the spot to meet it in the air (header/jump).
    pub attack_point: Vec2,
    /// Horizontal point where the ball reaches the **sweet** settle height (~5 ft) — the
    /// spot to take a cushioned touch as it drops to the feet.
    pub settle_point: Vec2,
    /// Bare landing point (altitude 0) — where the ball would hit the ground untouched.
    pub landing_point: Vec2,
    /// Seconds from now until the ball descends to the top of the band.
    pub time_to_attack: f64,
    /// Seconds from now until the ball descends to the sweet height.
    pub time_to_settle: f64,
    /// Seconds from now until the ball lands (altitude 0).
    pub time_to_land: f64,
    /// Apex of the arc (yards) — how high the ball climbed.
    pub apex_yards: f64,
}

impl AerialDescentPlan {
    /// Interpolate the take-on point for an attack-height blend in `[0, 1]`: 0 = the
    /// settle (foot) point, 1 = the attack (header) point. The MDP decision + the learned
    /// head both express their choice as this single blend.
    pub fn point_for_blend(&self, blend: f64) -> Vec2 {
        let b = blend.clamp(0.0, 1.0);
        self.settle_point * (1.0 - b) + self.attack_point * b
    }

    /// Seconds from now to take the ball at a blend (the ball reaches the top of the band
    /// before the sweet height, so a higher blend is sooner).
    pub fn time_for_blend(&self, blend: f64) -> f64 {
        let b = blend.clamp(0.0, 1.0);
        self.time_to_settle * (1.0 - b) + self.time_to_attack * b
    }
}

/// Descending-root time at which a gravity-timed loft of hang time `hang` reaches height
/// `h` (yards). `None` when the arc never climbs that high (apex `< h`).
fn descent_time_to_height(hang: f64, h: f64) -> Option<f64> {
    // altitude(t) = ½·g·t·(hang − t); solve = h ⇒ t² − hang·t + 2h/g = 0.
    // Descending root = (hang + √(hang² − 8h/g)) / 2.
    let disc = hang * hang - 8.0 * h / GRAVITY_YPS2;
    if disc < 0.0 {
        return None;
    }
    Some((hang + disc.sqrt()) * 0.5)
}

/// How far past the intended landing spot a descent point may be projected before it is
/// capped, in yards. A small slack so a slightly led/over-struck ball isn't truncated to
/// the aim point, while a wildly over-hit ball isn't chased off the park.
const AERIAL_OVERSHOOT_MARGIN_YARDS: f64 = 4.0;

/// Pure projectile geometry of a descending lofted ball. The vertical schedule is the
/// engine's own gravity-timed loft (`apex_yards` + `time_aloft_seconds` ⇒ when the ball
/// passes each control height); the **horizontal** is projected from the **current** ball
/// state (`ball_now` + `ball_horizontal_vel`) over the remaining time, so it inherits the
/// drag already bled off in flight instead of assuming the launch pace for the whole arc
/// (a long/fast aerial loses real horizontal pace to air + linear drag, so the constant-
/// launch model over-shoots the drop zone). `intended_target` only bounds the projection
/// so an over-hit ball isn't chased off the pitch. Returns `None` when the ball is not a
/// genuine loft (apex below the band) — the caller then keeps the flat-projection election.
pub fn aerial_descent_plan(
    apex_yards: f64,
    time_aloft_seconds: f64,
    ball_now: Vec2,
    ball_horizontal_vel: Vec2,
    intended_target: Vec2,
    field_width: f64,
    field_length: f64,
) -> Option<AerialDescentPlan> {
    if !apex_yards.is_finite() || apex_yards < AERIAL_ANTICIPATION_MIN_APEX_YARDS {
        return None;
    }
    if !ball_now.x.is_finite() || !ball_now.y.is_finite() {
        return None;
    }
    // A non-finite velocity ⇒ treat the ball as dropping vertically (project off the aim
    // bearing only), never propagate a NaN into the target.
    let ball_horizontal_vel = if ball_horizontal_vel.x.is_finite() && ball_horizontal_vel.y.is_finite()
    {
        ball_horizontal_vel
    } else {
        Vec2::new(0.0, 0.0)
    };
    let hang = 2.0 * (2.0 * apex_yards / GRAVITY_YPS2).sqrt();
    if !hang.is_finite() || hang <= 1e-3 {
        return None;
    }
    let t_attack = descent_time_to_height(hang, AERIAL_CONTROL_BAND_TOP_YARDS)?;
    // The sweet height is below the top, so it always exists when the top does.
    let t_settle =
        descent_time_to_height(hang, AERIAL_CONTROL_BAND_SWEET_YARDS).unwrap_or(t_attack);
    // Remaining time (from now) until the ball passes each height on the way down.
    let r_attack = (t_attack - time_aloft_seconds).max(0.0);
    let r_settle = (t_settle - time_aloft_seconds).max(0.0);
    let r_land = (hang - time_aloft_seconds).max(0.0);
    // Horizontal projection from the CURRENT ball state. Use the current horizontal pace
    // and heading (drag-aware to date); fall back to the bearing to the aim point when the
    // ball is dropping near-vertically (negligible horizontal speed).
    let speed_now = ball_horizontal_vel.len();
    let to_target = intended_target - ball_now;
    let dist_to_target = to_target.len();
    let dir = if speed_now > 1e-3 {
        ball_horizontal_vel / speed_now
    } else if dist_to_target > 1e-3 {
        to_target / dist_to_target
    } else {
        Vec2::new(0.0, 0.0)
    };
    let max_forward = dist_to_target + AERIAL_OVERSHOOT_MARGIN_YARDS;
    let point_at = |r: f64| -> Vec2 {
        let traveled = (speed_now * r).clamp(0.0, max_forward);
        (ball_now + dir * traveled).clamp_to_pitch(field_width, field_length)
    };
    Some(AerialDescentPlan {
        attack_point: point_at(r_attack),
        settle_point: point_at(r_settle),
        landing_point: point_at(r_land),
        time_to_attack: r_attack,
        time_to_settle: r_settle,
        time_to_land: r_land,
        apex_yards,
    })
}

/// The MDP/POMDP choice of how to take a dropping aerial ball.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AerialReceptionDecision {
    /// Comfortably first to the drop zone and unpressured: stand under it and cushion it
    /// down to the feet (take it at the sweet height).
    SettleUnder,
    /// A defender is racing the drop zone: step up and meet the ball high (header / chest)
    /// to win it before it drops.
    AttackInFront,
    /// Reachable only on the move / slightly late: chase the ball dropping in front and
    /// take it as it lands.
    ChaseDrop,
}

/// The resolved per-receiver aerial reception: where to go, how to take it, whether to
/// sprint, and the analytic estimate of how cleanly it can be controlled.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AerialReceptionPlan {
    pub decision: AerialReceptionDecision,
    /// The point to move to (the take-on point for the chosen attack-height blend).
    pub target: Vec2,
    /// Chosen attack-height blend in `[0, 1]` (0 = settle at feet, 1 = attack at the top).
    pub attack_blend: f64,
    pub sprint: bool,
    /// Analytic estimate in `[0, 1]` of bringing it under clean control (the RL signal the
    /// head is trained against; also a useful decision-trace value).
    pub control_estimate: f64,
}

/// Kinematic + contest inputs to the reception decision, kept raw for readability.
#[derive(Clone, Copy, Debug)]
pub struct AerialReceptionInputs {
    /// Receiver's fatigue-adjusted sprint speed (yps).
    pub receiver_speed_yps: f64,
    /// Receiver's current position.
    pub receiver_position: Vec2,
    /// Nearest opponent's arrival time (seconds) to the drop zone (settle point).
    pub opponent_time_to_drop: f64,
    /// Nearest opponent's distance (yards) to the drop zone — a pressure proxy.
    pub opponent_distance_to_drop: f64,
    /// Receiver's aerial-duel tool in `[0, 1]` (header/strength/height).
    pub aerial_tool: f64,
    /// Receiver's first-touch tool in `[0, 1]` (cushioned settle).
    pub first_touch_tool: f64,
}

/// Closed-form analytic seed of the reception decision: the live fallback and the
/// bootstrap the learned head refines. RNG-free and pure.
pub fn analytic_aerial_reception(
    plan: &AerialDescentPlan,
    inputs: &AerialReceptionInputs,
    field_width: f64,
    field_length: f64,
) -> AerialReceptionPlan {
    let speed = inputs.receiver_speed_yps.max(1.0);
    // Reachability of the drop zone (settle point) under the receiver's own speed limit —
    // the individual reach the rest of the decision conditions on.
    let settle_run = inputs.receiver_position.distance(plan.settle_point);
    let settle_arrival = settle_run / speed;
    let slack = plan.time_to_settle - settle_arrival; // > 0 ⇒ waiting under it.
    let contest = inputs.opponent_time_to_drop - settle_arrival; // > 0 ⇒ we win the race.

    // Pressured when a defender can reach the drop zone within the contest margin, or is
    // simply close to it. Then the ball must be attacked high before it drops to him.
    let pressured = contest <= AERIAL_CONTEST_MARGIN_SECONDS
        || inputs.opponent_distance_to_drop <= AERIAL_CONTROL_BAND_TOP_YARDS + 1.5;

    let (decision, mut attack_blend) = if settle_arrival > plan.time_to_settle + AERIAL_CHASE_TOLERANCE_SECONDS
    {
        // Can't get under it in time: chase the ball down as it lands.
        (AerialReceptionDecision::ChaseDrop, 0.0)
    } else if pressured {
        // Win it in the air. How high we attack scales with how hard we are pressed and
        // our aerial tool — a strong header attacks the very top of the band.
        let press = (1.0 - (contest / AERIAL_CONTEST_MARGIN_SECONDS).clamp(0.0, 1.0)).clamp(0.0, 1.0);
        let blend = (0.55 + press * 0.30 + inputs.aerial_tool * 0.15).clamp(0.0, 1.0);
        (AerialReceptionDecision::AttackInFront, blend)
    } else if slack >= AERIAL_SETTLE_SLACK_SECONDS {
        // Comfortably under it and free: let it drop to the feet for a clean touch.
        (AerialReceptionDecision::SettleUnder, 0.0)
    } else {
        // Reachable but tight: take it slightly above the feet to be sure of it.
        (AerialReceptionDecision::AttackInFront, 0.30)
    };

    // Clean-control estimate: settling under a dropping ball with a good first touch is
    // the most secure; attacking it high under pressure is less so. This is the analytic
    // proxy for the RL reward the head learns from.
    let tool = (inputs.first_touch_tool * (1.0 - attack_blend) + inputs.aerial_tool * attack_blend)
        .clamp(0.0, 1.0);
    let lateness = (settle_arrival - plan.time_to_settle).max(0.0);
    let control_estimate = (0.40 + tool * 0.45 + contest.clamp(-0.6, 0.8) * 0.18
        - lateness * 0.30)
        .clamp(0.0, 1.0);

    // A chase runs onto where the ball actually comes down (the landing point), not the
    // settle spot it can't reach in time — keep the full attack blend at 0 for the trace.
    let target = if matches!(decision, AerialReceptionDecision::ChaseDrop) {
        attack_blend = 0.0;
        plan.landing_point
    } else {
        plan.point_for_blend(attack_blend)
    }
    .clamp_to_pitch(field_width, field_length);
    let sprint = !matches!(decision, AerialReceptionDecision::SettleUnder)
        || slack < AERIAL_SETTLE_SLACK_SECONDS * 2.0;
    AerialReceptionPlan {
        decision,
        target,
        attack_blend,
        sprint,
        control_estimate,
    }
}

/// Resolve the full reception plan, letting the learned head refine the attack-height
/// blend when present (and enabled). The head moves only the blend knob; the decision
/// label, target geometry, and reachability all come from the analytic seed, so an absent
/// head is byte-identical to [`analytic_aerial_reception`].
pub fn decide_aerial_reception(
    plan: &AerialDescentPlan,
    inputs: &AerialReceptionInputs,
    head: Option<&AerialReceptionControlHead>,
    field_width: f64,
    field_length: f64,
) -> AerialReceptionPlan {
    let mut resolved = analytic_aerial_reception(plan, inputs, field_width, field_length);
    // The head only ever refines an attacking take-on (a settle/chase is geometry, not a
    // height trade-off), and only while the model is enabled.
    if matches!(resolved.decision, AerialReceptionDecision::AttackInFront) {
        if let Some(head) = head.filter(|_| aerial_reception_model_enabled()) {
            let ctx = AerialReceptionContext::build(plan, inputs, resolved.attack_blend);
            if let Some(blend) = head.predict(&ctx) {
                resolved.attack_blend = blend.clamp(0.0, 1.0);
                resolved.target = plan
                    .point_for_blend(resolved.attack_blend)
                    .clamp_to_pitch(field_width, field_length);
            }
        }
    }
    resolved
}

/// Feature vector for the learnable control head, in the receiver's reception frame so
/// the head is mirror-invariant. Stored raw; [`Self::to_features`] normalizes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AerialReceptionContext {
    /// Arc apex (yards) — how steeply the ball is dropping.
    pub apex_yards: f64,
    /// Horizontal distance (yards) from the receiver to the drop zone (settle point).
    pub run_to_drop_yards: f64,
    /// Seconds until the ball reaches the sweet settle height.
    pub time_to_settle: f64,
    /// Receiver's slack (seconds early) at the drop zone (> 0 ⇒ waiting under it).
    pub slack_seconds: f64,
    /// Contest margin (seconds the receiver beats the nearest opponent to the drop zone).
    pub contest_seconds: f64,
    /// Nearest opponent's distance (yards) to the drop zone.
    pub opponent_distance_yards: f64,
    /// Receiver's aerial-duel tool in `[0, 1]`.
    pub aerial_tool: f64,
    /// Receiver's first-touch tool in `[0, 1]`.
    pub first_touch_tool: f64,
    /// Ball's horizontal pace (yps) along the arc.
    pub ball_pace_yps: f64,
    /// The analytic seed blend (so the head learns a *correction* to the seed).
    pub seed_blend: f64,
    /// Receiver's sprint speed (yps).
    pub receiver_speed_yps: f64,
}

impl AerialReceptionContext {
    /// Build the context from the descent plan + inputs (the seed blend is the analytic
    /// choice the head refines).
    pub fn build(
        plan: &AerialDescentPlan,
        inputs: &AerialReceptionInputs,
        seed_blend: f64,
    ) -> Self {
        let speed = inputs.receiver_speed_yps.max(1.0);
        let run = inputs.receiver_position.distance(plan.settle_point);
        let settle_arrival = run / speed;
        let pace = plan.settle_point.distance(plan.landing_point)
            / (plan.time_to_land - plan.time_to_settle).max(1e-3);
        AerialReceptionContext {
            apex_yards: plan.apex_yards,
            run_to_drop_yards: run,
            time_to_settle: plan.time_to_settle,
            slack_seconds: plan.time_to_settle - settle_arrival,
            contest_seconds: inputs.opponent_time_to_drop - settle_arrival,
            opponent_distance_yards: inputs.opponent_distance_to_drop,
            aerial_tool: inputs.aerial_tool,
            first_touch_tool: inputs.first_touch_tool,
            ball_pace_yps: pace,
            seed_blend,
            receiver_speed_yps: inputs.receiver_speed_yps,
        }
    }

    /// The normalized network input. **This ordering is the single source of truth** —
    /// change it and bump [`AERIAL_RECEPTION_FEATURE_DIM`] and any persisted head.
    pub fn to_features(&self) -> [f64; AERIAL_RECEPTION_FEATURE_DIM] {
        let f = |v: f64, r: f64| (v / r).clamp(-4.0, 4.0);
        [
            f(self.apex_yards, REF_APEX_YARDS),
            f(self.run_to_drop_yards, REF_RUN_YARDS),
            f(self.time_to_settle, REF_TIME_SECONDS),
            f(self.slack_seconds, REF_TIME_SECONDS),
            f(self.contest_seconds, REF_TIME_SECONDS),
            f(self.opponent_distance_yards, REF_RUN_YARDS),
            self.aerial_tool.clamp(0.0, 1.0) * 2.0 - 1.0,
            self.first_touch_tool.clamp(0.0, 1.0) * 2.0 - 1.0,
            f(self.ball_pace_yps, REF_BALL_SPEED_YPS),
            self.seed_blend.clamp(0.0, 1.0) * 2.0 - 1.0,
            f(self.receiver_speed_yps, REF_BALL_SPEED_YPS),
        ]
    }
}

/// One self-play training row: the reception context, the attack-height blend actually
/// used, and whether the ball was brought under clean control (the reward). The head
/// regresses toward the blend that controlled the ball, weighted by the reward.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AerialReceptionSample {
    pub context: AerialReceptionContext,
    /// The attack-height blend used at the decision (the regression target).
    pub used_blend: f64,
    /// Clean-control reward in `[0, 1]` (1 = settled/won cleanly, 0 = lost / headed away
    /// under duress). Positive reinforces the used blend; low pushes toward the seed.
    pub reward: f64,
}

/// An aerial reception awaiting its control outcome. Recorded when the receiver commits to
/// the dropping ball; resolved a few ticks later by whether they cleanly controlled it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PendingAerialReception {
    pub receiver: usize,
    pub team: Team,
    pub context: AerialReceptionContext,
    pub used_blend: f64,
    pub due_tick: u64,
}

/// Learned regression head for the attack-height blend: a `FeedForwardNetwork`
/// (`DIM -> hidden -> 1`, sigmoid output in `[0, 1]`). Not serde; round-trips through the
/// engine's neural-network snapshot path like the other learned heads.
#[derive(Clone, Debug)]
pub struct AerialReceptionControlHead {
    network: FeedForwardNetwork,
    training_steps: usize,
    last_loss: Option<f64>,
}

impl AerialReceptionControlHead {
    pub fn new(seed: u32) -> Self {
        let mut rng = mulberry32(seed ^ 0x5BD1_E995);
        let network = FeedForwardNetwork::random(
            &RandomNetworkSpec {
                input_dim: AERIAL_RECEPTION_FEATURE_DIM,
                hidden_layers: vec![AERIAL_RECEPTION_HIDDEN_UNITS],
                output_dim: 1,
                hidden_activation: ActivationName::Tanh,
                // Sigmoid: the output is a blend in [0, 1].
                output_activation: ActivationName::Sigmoid,
                weight_scale: None,
            },
            &mut rng,
        );
        AerialReceptionControlHead {
            network,
            training_steps: 0,
            last_loss: None,
        }
    }

    /// Predicted attack-height blend in `[0, 1]`, or `None` on a malformed input.
    pub fn predict(&self, ctx: &AerialReceptionContext) -> Option<f64> {
        let x = ctx.to_features();
        if x.iter().any(|v| !v.is_finite()) {
            return None;
        }
        self.network
            .predict(&x[..])
            .first()
            .copied()
            .filter(|p| p.is_finite())
            .map(|p| p.clamp(0.0, 1.0))
    }

    /// One SGD epoch regressing the head toward each row's used blend, **weighted by the
    /// clean-control reward** so blends that controlled the ball are reinforced and ones
    /// that lost it are pushed toward the analytic seed. Returns the mean step loss.
    pub fn train(&mut self, samples: &[AerialReceptionSample], learning_rate: f64) -> f64 {
        let mut total = 0.0;
        let mut applied = 0usize;
        for s in samples {
            let x = s.context.to_features();
            if x.iter().any(|v| !v.is_finite())
                || !s.reward.is_finite()
                || !s.used_blend.is_finite()
            {
                continue;
            }
            let (target_blend, weight) = if s.reward > 0.5 {
                (s.used_blend, (s.reward).min(1.0))
            } else {
                // Lost it: regress toward the analytic seed blend at reduced weight.
                (s.context.seed_blend, (0.5 - s.reward).max(0.0).min(0.5))
            };
            if weight <= 0.0 {
                continue;
            }
            let target = [target_blend.clamp(0.0, 1.0)];
            let result =
                self.network
                    .train_sample_clipped(&x[..], &target, learning_rate * weight, 4.0);
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

/// Summary of an aerial-reception training pass (logged by the learner).
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct AerialReceptionTrainingReport {
    pub samples: usize,
    pub training_steps: usize,
    pub epochs: usize,
    pub final_loss: f64,
}

/// Train a fresh aerial-reception head over an RL corpus. `None` on an empty/unusable
/// corpus or `epochs == 0`, so the learner can skip cleanly.
pub fn train_aerial_reception_head(
    samples: &[AerialReceptionSample],
    seed: u32,
    epochs: usize,
    learning_rate: f64,
) -> Option<(AerialReceptionControlHead, AerialReceptionTrainingReport)> {
    let usable = samples
        .iter()
        .filter(|s| s.reward.is_finite() && s.used_blend.is_finite())
        .count();
    if usable == 0 || epochs == 0 {
        return None;
    }
    let mut head = AerialReceptionControlHead::new(seed);
    let mut final_loss = 0.0;
    for _ in 0..epochs {
        final_loss = head.train(samples, learning_rate);
    }
    let report = AerialReceptionTrainingReport {
        samples: usable,
        training_steps: head.training_steps(),
        epochs,
        final_loss,
    };
    Some((head, report))
}

/// Control-settling grace (seconds) added past the ball's landing time before an aerial
/// reception's outcome is scored — long enough for the receiver to take the touch.
pub const AERIAL_RECEPTION_SETTLE_GRACE_SECONDS: f64 = 0.45;
/// Bounds (ticks) on the adaptive outcome window, so a degenerate flight time can't make
/// the reward resolve instantly or hang indefinitely.
pub const AERIAL_RECEPTION_MIN_WINDOW_TICKS: u64 = 4;
pub const AERIAL_RECEPTION_MAX_WINDOW_TICKS: u64 = 60;

impl SoccerMatch {
    /// Gated per-tick RL sample collection for the learnable aerial-reception control
    /// head, driven off the already-built per-tick `snapshot`. A **no-op** (zero cost,
    /// byte-identical) unless the model is enabled. When on: resolves receptions whose
    /// outcome window has elapsed (reward = whether the receiving team holds the ball
    /// under control — at/below the control band — by then) and, when a lofted ball is
    /// in flight with a named receiver who is genuinely anticipating its descent, samples
    /// that reception's context + the attack-height blend it used. Drained by the learner
    /// via [`Self::drain_aerial_reception_samples`].
    pub(crate) fn collect_aerial_reception_rl_samples(&mut self, snapshot: &WorldSnapshot) {
        if !aerial_reception_model_enabled() {
            return;
        }
        let tick = self.tick;
        // Resolve receptions whose outcome window has elapsed.
        let mut i = 0;
        while i < self.pending_aerial_reception.len() {
            if self.pending_aerial_reception[i].due_tick <= tick {
                let decision = self.pending_aerial_reception.swap_remove(i);
                let holder_team = snapshot
                    .ball
                    .holder
                    .and_then(|h| snapshot.players.iter().find(|p| p.id == h))
                    .map(|p| p.team);
                let reward = if holder_team == Some(decision.team) {
                    // Held by us: fully clean if it has been brought down into control,
                    // a partial win if it is still being contested in the air.
                    if snapshot.ball.altitude_yards <= AERIAL_CONTROL_BAND_TOP_YARDS {
                        1.0
                    } else {
                        0.6
                    }
                } else {
                    0.0
                };
                self.aerial_reception_samples.push(AerialReceptionSample {
                    context: decision.context,
                    used_blend: decision.used_blend,
                    reward,
                });
                if self.aerial_reception_samples.len() > AERIAL_RECEPTION_SAMPLE_CAP {
                    let overflow = self.aerial_reception_samples.len() - AERIAL_RECEPTION_SAMPLE_CAP;
                    self.aerial_reception_samples.drain(0..overflow);
                }
            } else {
                i += 1;
            }
        }
        // Sample a fresh in-flight aerial reception for its named receiver.
        if tick % AERIAL_RECEPTION_SAMPLE_INTERVAL_TICKS != 0 {
            return;
        }
        let Some(pass) = snapshot.pending_pass.as_ref() else {
            return;
        };
        if !pass.flight.is_aerial() {
            return;
        }
        let Some(receiver_id) = pass.nearest_receiver else {
            return;
        };
        // Don't double-sample the same in-flight reception for the same receiver.
        if self
            .pending_aerial_reception
            .iter()
            .any(|p| p.receiver == receiver_id)
        {
            return;
        }
        let Some(me) = snapshot.players.iter().find(|p| p.id == receiver_id) else {
            return;
        };
        let current = snapshot.player_snapshot_position(me);
        if let Some((descent, inputs, plan)) = snapshot.aerial_reception_resolve(me, pass, current) {
            let context = AerialReceptionContext::build(&descent, &inputs, plan.attack_blend);
            // Resolve the outcome shortly AFTER the ball actually lands (flight time + a
            // touch-settling grace), not a fixed window — a long lob is still airborne a
            // fixed window later, which would score every high ball as "not controlled".
            let dt = snapshot.dt_seconds.max(1e-3);
            let window = (((descent.time_to_land + AERIAL_RECEPTION_SETTLE_GRACE_SECONDS) / dt)
                .ceil() as u64)
                .clamp(
                    AERIAL_RECEPTION_MIN_WINDOW_TICKS,
                    AERIAL_RECEPTION_MAX_WINDOW_TICKS,
                );
            self.pending_aerial_reception.push(PendingAerialReception {
                receiver: receiver_id,
                team: pass.team,
                context,
                used_blend: plan.attack_blend,
                due_tick: tick + window,
            });
        }
    }

    /// Drain the collected aerial-reception RL samples for the cluster learner (train the
    /// head + persist). `pub` so the learner binary (a separate crate) can drain it.
    pub fn drain_aerial_reception_samples(&mut self) -> Vec<AerialReceptionSample> {
        std::mem::take(&mut self.aerial_reception_samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn long_loft_plan() -> AerialDescentPlan {
        // A long DRIVEN aerial (apex ~4.5yd, ~26yps) launched +y: still travelling
        // forward as it drops through the band, so the 8ft/5ft control points are
        // horizontally distinct (a steep lob would drop vertically onto one spot —
        // see `steep_lob_band_points_coincide_horizontally`).
        aerial_descent_plan(
            4.5,
            0.0,
            Vec2::new(30.0, 10.0),
            Vec2::new(0.0, 26.0),
            Vec2::new(30.0, 70.0),
            68.0,
            105.0,
        )
        .expect("a 4.5yd-apex driven aerial is a genuine aerial ball")
    }

    #[test]
    fn descent_plan_attacks_higher_and_sooner_than_it_settles() {
        let plan = long_loft_plan();
        // The ball passes the 8ft band before it drops to 5ft, so the attack point is
        // reached sooner and is short of (behind, up the path) the settle point.
        assert!(plan.time_to_attack < plan.time_to_settle);
        assert!(plan.time_to_settle < plan.time_to_land);
        // Both control points are on the path between origin and the landing point.
        assert!(plan.attack_point.y < plan.settle_point.y + 1e-6);
        assert!(plan.settle_point.y <= plan.landing_point.y + 1e-6);
    }

    #[test]
    fn steep_lob_band_points_coincide_horizontally() {
        // A high, slow lob (apex ~7yd, 16yps) drops near-vertically onto its target:
        // the ball reaches the landing spot horizontally before descending into the
        // band, so the 8ft and 5ft control points are the same spot — only their timing
        // differs (the lever for the receiver is *when* to jump, not where to step).
        let plan = aerial_descent_plan(
            7.0,
            0.0,
            Vec2::new(30.0, 10.0),
            Vec2::new(0.0, 16.0),
            Vec2::new(30.0, 38.0),
            68.0,
            105.0,
        )
        .expect("a steep lob is a genuine aerial ball");
        assert!((plan.attack_point.distance(plan.settle_point)) < 1e-6);
        assert!(plan.time_to_attack < plan.time_to_settle);
    }

    #[test]
    fn a_flat_ball_has_no_descent_plan() {
        // Apex inside the control band ⇒ never a genuine loft to anticipate.
        assert!(aerial_descent_plan(
            1.0,
            0.0,
            Vec2::new(0.0, 0.0),
            Vec2::new(0.0, 18.0),
            Vec2::new(0.0, 20.0),
            68.0,
            105.0,
        )
        .is_none());
    }

    #[test]
    fn point_for_blend_interpolates_settle_to_attack() {
        let plan = long_loft_plan();
        assert_eq!(plan.point_for_blend(0.0), plan.settle_point);
        assert_eq!(plan.point_for_blend(1.0), plan.attack_point);
        let mid = plan.point_for_blend(0.5);
        assert!((mid.y - (plan.settle_point.y + plan.attack_point.y) * 0.5).abs() < 1e-6);
    }

    #[test]
    fn drag_aware_projection_shortens_the_drop_zone_vs_launch_pace() {
        // Same arc sampled mid-flight at two horizontal paces: a ball that has bled pace
        // to drag (slower current velocity) must drop SHORTER than one still at full pace.
        let fast = aerial_descent_plan(
            5.0,
            0.3,
            Vec2::new(30.0, 25.0),
            Vec2::new(0.0, 24.0),
            Vec2::new(30.0, 80.0),
            68.0,
            105.0,
        )
        .expect("aerial");
        let dragged = aerial_descent_plan(
            5.0,
            0.3,
            Vec2::new(30.0, 25.0),
            Vec2::new(0.0, 16.0),
            Vec2::new(30.0, 80.0),
            68.0,
            105.0,
        )
        .expect("aerial");
        assert!(
            dragged.settle_point.y < fast.settle_point.y,
            "a dragged ball ({}) drops shorter than a full-pace one ({})",
            dragged.settle_point.y,
            fast.settle_point.y
        );
    }

    #[test]
    fn overhit_ball_drop_zone_is_capped_near_the_aim() {
        // A very fast ball over a short remaining distance must not project far past the
        // intended target (the receiver shouldn't chase it off the park).
        let plan = aerial_descent_plan(
            5.0,
            0.2,
            Vec2::new(30.0, 60.0),
            Vec2::new(0.0, 40.0),
            Vec2::new(30.0, 66.0),
            68.0,
            105.0,
        )
        .expect("aerial");
        assert!(plan.landing_point.y <= 66.0 + AERIAL_OVERSHOOT_MARGIN_YARDS + 1e-6);
    }

    #[test]
    fn unreachable_ball_chases_the_landing_point() {
        let plan = long_loft_plan();
        // Receiver far behind the drop zone and slow: can't get under it ⇒ chase where it
        // lands, not the settle spot it can't reach.
        let inputs = AerialReceptionInputs {
            receiver_speed_yps: 6.0,
            receiver_position: plan.settle_point + Vec2::new(0.0, -40.0),
            opponent_time_to_drop: 5.0,
            opponent_distance_to_drop: 30.0,
            aerial_tool: 0.5,
            first_touch_tool: 0.6,
        };
        let resolved = analytic_aerial_reception(&plan, &inputs, 68.0, 105.0);
        assert_eq!(resolved.decision, AerialReceptionDecision::ChaseDrop);
        assert_eq!(resolved.target, plan.landing_point);
    }

    #[test]
    fn free_receiver_under_it_settles_at_the_feet() {
        let plan = long_loft_plan();
        // Receiver already standing at the drop zone, no opponent near.
        let inputs = AerialReceptionInputs {
            receiver_speed_yps: 8.0,
            receiver_position: plan.settle_point,
            opponent_time_to_drop: 5.0,
            opponent_distance_to_drop: 30.0,
            aerial_tool: 0.5,
            first_touch_tool: 0.7,
        };
        let resolved = analytic_aerial_reception(&plan, &inputs, 68.0, 105.0);
        assert_eq!(resolved.decision, AerialReceptionDecision::SettleUnder);
        assert_eq!(resolved.attack_blend, 0.0);
        assert_eq!(resolved.target, plan.settle_point);
    }

    #[test]
    fn pressured_receiver_attacks_the_ball_high() {
        let plan = long_loft_plan();
        // Receiver and a defender both arriving around the same time ⇒ contest ⇒ attack.
        let inputs = AerialReceptionInputs {
            receiver_speed_yps: 8.0,
            receiver_position: plan.settle_point + Vec2::new(0.0, -4.0),
            opponent_time_to_drop: 0.2,
            opponent_distance_to_drop: 2.0,
            aerial_tool: 0.8,
            first_touch_tool: 0.5,
        };
        let resolved = analytic_aerial_reception(&plan, &inputs, 68.0, 105.0);
        assert_eq!(resolved.decision, AerialReceptionDecision::AttackInFront);
        assert!(resolved.attack_blend > 0.5);
        assert!(resolved.sprint);
        // The take-on point steps UP the path toward the incoming ball (higher contact).
        assert!(resolved.target.y < plan.settle_point.y);
    }

    #[test]
    fn head_only_moves_the_blend_when_enabled_and_attacking() {
        let plan = long_loft_plan();
        let inputs = AerialReceptionInputs {
            receiver_speed_yps: 8.0,
            receiver_position: plan.settle_point,
            opponent_time_to_drop: 5.0,
            opponent_distance_to_drop: 30.0,
            aerial_tool: 0.5,
            first_touch_tool: 0.7,
        };
        // A settle decision is geometry — even with a head, the blend stays 0.
        let head = AerialReceptionControlHead::new(7);
        let resolved = decide_aerial_reception(&plan, &inputs, Some(&head), 68.0, 105.0);
        assert_eq!(resolved.decision, AerialReceptionDecision::SettleUnder);
        assert_eq!(resolved.attack_blend, 0.0);
    }

    #[test]
    fn head_trains_toward_a_controlled_blend() {
        let plan = long_loft_plan();
        let inputs = AerialReceptionInputs {
            receiver_speed_yps: 8.0,
            receiver_position: plan.settle_point + Vec2::new(0.0, -4.0),
            opponent_time_to_drop: 0.2,
            opponent_distance_to_drop: 2.0,
            aerial_tool: 0.8,
            first_touch_tool: 0.5,
        };
        let ctx = AerialReceptionContext::build(&plan, &inputs, 0.7);
        let samples: Vec<_> = (0..64)
            .map(|_| AerialReceptionSample {
                context: ctx,
                used_blend: 0.9,
                reward: 1.0,
            })
            .collect();
        let (head, report) =
            train_aerial_reception_head(&samples, 11, 30, 0.05).expect("trainable corpus");
        assert!(report.training_steps > 0);
        let pred = head.predict(&ctx).expect("finite prediction");
        // It should move toward the rewarded high blend (well above neutral 0.5).
        assert!(pred > 0.6, "predicted blend {pred} should learn toward 0.9");
    }
}
