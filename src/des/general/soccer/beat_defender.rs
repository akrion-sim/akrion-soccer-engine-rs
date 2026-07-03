//! Player-level **beat-the-defender duel** — the dribble carrier exploiting a
//! *committed / over-committed* defender, modelled as an MDP option over a POMDP
//! (reaction-latency) perception of the defender and executed by the existing
//! single-player MPC. This is the "accelerate past the man" / feint / nutmeg
//! behaviour that the skill-only [`super::dribble_beat_probability`] never
//! captured, because that model is blind to the defender's *velocity* — to
//! whether they are leaning the wrong way or diving in.
//!
//! ## What is modelled
//!
//! 1. **Defender perception latency (POMDP)** — every player reacts on a
//!    100–250 ms lag derived from their traits
//!    ([`defender_perception_lag_seconds`]). During that window the defender is
//!    still acting on a *stale* read of the carrier, so a sharp change of
//!    direction buys the carrier free yards. This is *duel-scoped*: only the one
//!    defender being taken on is modelled with the lag, not a global perception
//!    rewrite.
//! 2. **Over-commitment (the exploit)** — [`DefenderCommitmentState`] reads the
//!    defender's momentum in the carrier's attacking frame: a lateral lean opens
//!    the *opposite* side; a hard closing run can be turned/rounded because the
//!    momentum carries the defender past. `over_commit` is the yards the defender
//!    drifts off the ideal interception line before they can arrest/redirect
//!    (their reaction lag + turn inertia).
//! 3. **The move (MDP option)** — [`BeatMove`]: go opposite the momentum, feint
//!    to *induce* a lean then break opposite, or nutmeg a planted/square man.
//!    Scored by a kinematics-aware [`analytic_beat_probability`] seed, with an
//!    optional learnable [`BeatDefenderHead`] blended in once trained.
//!
//! ## Parity
//!
//! The behaviour is **on by default** but fully gated: [`beat_defender_duel_enabled`]
//! is false only when `DD_SOCCER_DISABLE_BEAT_DEFENDER_DUEL` is set, and every
//! live consumer (the run-around side/precondition, the carrier launch appetite,
//! the anti-freeze peel-away carry) returns its pre-existing value when the gate
//! is off — so a disabled process is byte-identical. The learnable head is a
//! *separate* opt-in (`DD_SOCCER_ENABLE_LEARNED_BEAT_DEFENDER`, off by default);
//! until enabled the analytic seed alone is used. All functions here are pure and
//! RNG-free (the head aside), so tests and the inspector can read them without
//! enabling any live seam.

use super::*;
// `ability01` is a private parent helper (not re-exported), so a glob `use
// super::*` does not bring it in — name it explicitly.
use super::ability01;
use crate::des::general::neural_network::{ActivationName, FeedForwardNetwork, RandomNetworkSpec};
use crate::des::general::prng::mulberry32;

/// Env gate that **disables** the beat-the-defender duel (default behaviour is ON).
/// Set to any value ⇒ every duel consumer falls back to its pre-existing path.
const BEAT_DEFENDER_DISABLE_ENV: &str = "DD_SOCCER_DISABLE_BEAT_DEFENDER_DUEL";
/// Env gate that **enables** the learnable beat-probability head (default OFF ⇒
/// the analytic seed is used alone). Independent of the behaviour gate.
const BEAT_DEFENDER_LEARNED_ENABLE_ENV: &str = "DD_SOCCER_ENABLE_LEARNED_BEAT_DEFENDER";

/// Perception/reaction latency band (seconds) for the duel POMDP — the human
/// "see it, then move" lag. 100 ms for the sharpest reactor, 250 ms for the
/// slowest, before fatigue.
const PERCEPTION_LAG_MIN_SECONDS: f64 = 0.10;
const PERCEPTION_LAG_MAX_SECONDS: f64 = 0.25;
/// Extra seconds of "can't change direction instantly" the defender's existing
/// momentum keeps carrying them, on top of the perception lag, when computing
/// how far they over-commit. A planted defender (~0 speed) over-commits ~0 yd;
/// a lunging one drifts `(lag + turn_inertia) * speed` yd off line.
const TURN_INERTIA_SECONDS: f64 = 0.16;
/// How much a fast *closing* run (diving in) counts toward over-commitment
/// relative to a pure lateral lean. Closing momentum is rounded/turned rather
/// than side-stepped, so it is a real — but slightly discounted — exploit.
const CLOSING_OVERCOMMIT_WEIGHT: f64 = 0.55;
/// Minimum lateral closing speed (yd/s) before the defender is read as having
/// *leaned* to one side (below this they are square ⇒ the open side is chosen by
/// space/cover, not by momentum).
const LATERAL_COMMIT_MIN_YPS: f64 = 1.1;
/// Reference over-commit (yards off line) that saturates the kinematic exploit
/// term of the beat probability. ~2 yd of drift is a clean beat.
const OVER_COMMIT_REF_YARDS: f64 = 2.2;

/// Number of features in the duel state vector consumed by [`BeatDefenderHead`].
/// Ordering is defined once in [`BeatDefenderInputs::to_features`].
pub const BEAT_DEFENDER_FEATURE_DIM: usize = 9;
/// Hidden width of the learnable head.
pub const BEAT_DEFENDER_HIDDEN_UNITS: usize = 16;
/// Blend of the learned head's beat probability against the analytic seed when
/// the head is enabled. `0` = pure analytic (parity-ish), `1` = pure head. Modest
/// until a trained head earns more of the blend.
pub const BEAT_DEFENDER_HEAD_BLEND: f64 = 0.5;

fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|raw| {
            let v = raw.trim().to_ascii_lowercase();
            matches!(v.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

/// Whether the beat-the-defender duel behaviour is live this process. ON unless
/// `DD_SOCCER_DISABLE_BEAT_DEFENDER_DUEL` is set ⇒ off restores the pre-existing
/// run-around / shield / dribble paths byte-for-byte.
pub fn beat_defender_duel_enabled() -> bool {
    #[cfg(test)]
    {
        // Read each call under test so a #[serial] test can toggle the gate.
        !std::env::var(BEAT_DEFENDER_DISABLE_ENV)
            .map(|v| !v.trim().is_empty())
            .unwrap_or(false)
    }
    #[cfg(not(test))]
    {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| {
            !std::env::var(BEAT_DEFENDER_DISABLE_ENV)
                .map(|v| !v.trim().is_empty())
                .unwrap_or(false)
        })
    }
}

/// Whether the learnable beat-probability head is consulted (off ⇒ analytic seed
/// only). Independent opt-in so the behaviour can ship before the head is trained.
pub fn learned_beat_defender_enabled() -> bool {
    #[cfg(test)]
    {
        env_flag_enabled(BEAT_DEFENDER_LEARNED_ENABLE_ENV)
    }
    #[cfg(not(test))]
    {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| env_flag_enabled(BEAT_DEFENDER_LEARNED_ENABLE_ENV))
    }
}

/// The duel POMDP perception/reaction latency (seconds) for a defender, in the
/// [`PERCEPTION_LAG_MIN_SECONDS`, `PERCEPTION_LAG_MAX_SECONDS`] band. A quick,
/// switched-on defender (high acceleration / tracking / awareness) sits near the
/// floor; a slow or tired one drifts toward the ceiling. Mirrors the trait
/// composite of [`super::player_reaction_time_seconds_from_traits`] but rescaled
/// to the human *perception* band (that function returns the larger
/// decision+initiation latency).
pub fn defender_perception_lag_seconds(skills: &SkillProfile, fatigue: f64) -> f64 {
    let quickness = ability01(skills.acceleration) * 0.30
        + ability01(skills.defensive_tracking) * 0.22
        + ability01(skills.defending) * 0.18
        + ability01(skills.aggression) * 0.10
        + ability01(skills.vision) * 0.20;
    let span = PERCEPTION_LAG_MAX_SECONDS - PERCEPTION_LAG_MIN_SECONDS;
    (PERCEPTION_LAG_MAX_SECONDS - quickness * span + fatigue.clamp(0.0, 1.0) * 0.05)
        .clamp(PERCEPTION_LAG_MIN_SECONDS, PERCEPTION_LAG_MAX_SECONDS)
}

/// The defender's committed motion as the carrier reads it, in the carrier's
/// attacking frame (so the model is mirror-invariant). Pure function of the two
/// kinematic states + the defender's traits; no world access, fully testable.
#[derive(Clone, Copy, Debug)]
pub struct DefenderCommitmentState {
    /// Perception/reaction lag (s) — the window the defender acts on a stale read.
    pub reaction_seconds: f64,
    /// Distance carrier→defender (yd).
    pub separation: f64,
    /// Defender speed (yd/s).
    pub speed: f64,
    /// Closing speed of the defender toward the carrier (yd/s; +ve = closing).
    pub closing_speed: f64,
    /// Signed lateral momentum in the attacking frame (yd/s; +x = toward +x line).
    pub lateral_commit: f64,
    /// Yards the defender drifts off the ideal interception line before they can
    /// arrest/redirect (their lag + turn inertia). The exploit magnitude.
    pub over_commit: f64,
    /// Side that the momentum leaves OPEN: `-1`/`+1` = beat on the −x/+x side,
    /// `0` = square defender (caller picks the side from space/cover).
    pub open_side: f64,
}

impl DefenderCommitmentState {
    /// Read the defender's commitment from the live kinematics. `attack_dir` is
    /// the carrier team's attacking y-direction (`±1`).
    pub fn evaluate(
        attack_dir: f64,
        carrier_pos: Vec2,
        defender_pos: Vec2,
        defender_vel: Vec2,
        defender_skills: &SkillProfile,
        defender_fatigue: f64,
    ) -> Self {
        let reaction_seconds = defender_perception_lag_seconds(defender_skills, defender_fatigue);
        let to_carrier = carrier_pos - defender_pos;
        let separation = to_carrier.len();
        let speed = defender_vel.len();
        let closing_speed = if separation > 1e-6 {
            defender_vel.dot(to_carrier * (1.0 / separation))
        } else {
            0.0
        };
        // Lateral lean is the x-component of the defender's velocity (x = width).
        let lateral_commit = defender_vel.x;
        // The defender keeps moving on momentum for the lag + turn-inertia window
        // before they can change direction; that is the drift off the line.
        let drift_window = reaction_seconds + TURN_INERTIA_SECONDS;
        let lateral_over = lateral_commit.abs() * drift_window;
        let closing_over = closing_speed.max(0.0) * drift_window * CLOSING_OVERCOMMIT_WEIGHT;
        let over_commit = lateral_over + closing_over;
        let open_side = if lateral_commit.abs() >= LATERAL_COMMIT_MIN_YPS {
            // Beat them on the side AWAY from their lean.
            -lateral_commit.signum()
        } else {
            0.0
        };
        // `attack_dir` is retained in the frame so callers can fold forward
        // progress in; the lateral exploit itself is mirror-invariant in x.
        let _ = attack_dir;
        DefenderCommitmentState {
            reaction_seconds,
            separation,
            speed,
            closing_speed,
            lateral_commit,
            over_commit,
            open_side,
        }
    }

    /// `true` when the defender has meaningfully committed their momentum (leaning
    /// or diving in) such that the take-on is *on* even without a raw pace edge.
    pub fn is_over_committed(&self) -> bool {
        self.over_commit >= OVER_COMMIT_REF_YARDS * 0.5
    }
}

/// The exploit move the carrier plays in the duel (an MDP action over the
/// committed-defender state). Each maps onto an existing `DribbleMoveKind` when
/// executed, so no new animation/label vocabulary is introduced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BeatMove {
    /// Accelerate past on the side away from the defender's momentum (run-around).
    GoOppositeMomentum,
    /// Feint toward the momentum side to *induce* a lean, then break opposite.
    FeintThenOpposite,
    /// Push the ball through a planted/square defender's legs.
    Nutmeg,
}

/// A viable beat-the-defender option: which defender, which move, the open side,
/// and the (analytic or learned) probability the beat comes off.
#[derive(Clone, Copy, Debug)]
pub struct BeatDefenderDuel {
    /// Index into `WorldSnapshot::players` of the defender being beaten.
    pub defender: usize,
    pub beat_move: BeatMove,
    /// `-1`/`+1` lateral side the carrier exits past the defender.
    pub open_side: f64,
    pub beat_probability: f64,
    pub commitment: DefenderCommitmentState,
}

/// Feature vector for the learnable head. Skill edges + the duel kinematics,
/// already normalised to ~`[-1, 1]`/`[0, 1]` and mirror-invariant.
#[derive(Clone, Copy, Debug)]
pub struct BeatDefenderInputs {
    pub attacker_dribbling: f64,
    pub attacker_acceleration: f64,
    pub attacker_first_touch: f64,
    pub defender_defending: f64,
    pub defender_tracking: f64,
    pub reaction_seconds: f64,
    pub over_commit: f64,
    pub closing_speed: f64,
    pub separation: f64,
}

impl BeatDefenderInputs {
    pub fn to_features(&self) -> [f64; BEAT_DEFENDER_FEATURE_DIM] {
        [
            ability01_raw(self.attacker_dribbling),
            ability01_raw(self.attacker_acceleration),
            ability01_raw(self.attacker_first_touch),
            ability01_raw(self.defender_defending),
            ability01_raw(self.defender_tracking),
            ((self.reaction_seconds - PERCEPTION_LAG_MIN_SECONDS)
                / (PERCEPTION_LAG_MAX_SECONDS - PERCEPTION_LAG_MIN_SECONDS))
                .clamp(0.0, 1.0),
            (self.over_commit / OVER_COMMIT_REF_YARDS).clamp(0.0, 2.0),
            (self.closing_speed / 9.0).clamp(-1.0, 1.0),
            (self.separation / 6.0).clamp(0.0, 1.0),
        ]
    }
}

/// Local copy of the parent's skill normaliser, kept here so the pure feature
/// builder has no hidden dependency on call order. Identical mapping to
/// [`super::ability01`].
fn ability01_raw(value: f64) -> f64 {
    ability01(value)
}

/// Per-move multiplier on the skill term — feinting to induce a lean beats a
/// committed man slightly more often, a nutmeg slightly less (it needs a gap).
fn beat_move_multiplier(beat_move: BeatMove) -> f64 {
    match beat_move {
        BeatMove::GoOppositeMomentum => 1.0,
        BeatMove::FeintThenOpposite => 1.06,
        BeatMove::Nutmeg => 0.86,
    }
}

/// Closed-form seed for the probability a beat move comes off, combining the
/// existing skill duel with the new kinematic exploit (over-commit + reaction
/// lag). Rises monotonically with the defender's over-commitment and lag, so a
/// leaning/diving defender is beaten more often than a planted one of equal
/// skill. Range `[0.05, 0.95]`.
pub fn analytic_beat_probability(
    beat_move: BeatMove,
    commitment: &DefenderCommitmentState,
    attacker: &SkillProfile,
    defender: &SkillProfile,
) -> f64 {
    let attacker_evasion = ability01(attacker.dribbling) * 0.50
        + ability01(attacker.acceleration) * 0.30
        + ability01(attacker.first_touch) * 0.20;
    let defender_balance = ability01(defender.defending) * 0.58
        + ability01(defender.defensive_tracking) * 0.27
        + ability01(defender.aggression) * 0.15;
    let skill = attacker_evasion / (attacker_evasion + defender_balance * 0.94);
    let exploit = (commitment.over_commit / OVER_COMMIT_REF_YARDS).clamp(0.0, 1.0);
    let lag = ((commitment.reaction_seconds - PERCEPTION_LAG_MIN_SECONDS)
        / (PERCEPTION_LAG_MAX_SECONDS - PERCEPTION_LAG_MIN_SECONDS))
        .clamp(0.0, 1.0);
    let move_mult = beat_move_multiplier(beat_move);
    let raw = skill * (0.55 + 0.30 * move_mult) + exploit * 0.42 + lag * 0.10;
    raw.clamp(0.05, 0.95)
}

/// Additional beat probability — in `[0, 0.48]` — that the defender's *over-commitment* and
/// perception lag confer, independent of the specific move kind. For blending into a standing-duel
/// resolution where the defender has lunged / dived in (the classic case: a missed tackle leaves
/// them wrong-footed with momentum), so the carrier accelerates past more often than the
/// skill-only model alone would allow. `0` for a planted, switched-on defender ⇒ no uplift.
pub fn over_commit_beat_uplift(commitment: &DefenderCommitmentState) -> f64 {
    let exploit = (commitment.over_commit / OVER_COMMIT_REF_YARDS).clamp(0.0, 1.0);
    let lag = ((commitment.reaction_seconds - PERCEPTION_LAG_MIN_SECONDS)
        / (PERCEPTION_LAG_MAX_SECONDS - PERCEPTION_LAG_MIN_SECONDS))
        .clamp(0.0, 1.0);
    (exploit * 0.40 + lag * 0.08).clamp(0.0, 0.48)
}

/// Pick the strongest beat move for a committed-defender state, plus its analytic
/// (or learned, when enabled) probability and the side to exit. Returns `None`
/// when no move clears a minimum viability — the caller then keeps the ball.
///
/// `central` is `true` when the defender is roughly in front (low |lateral gap|),
/// which is the picture for a nutmeg; `space_open_side` is the side with more
/// room (used when the defender is square and `open_side == 0`).
pub fn select_beat_move(
    commitment: &DefenderCommitmentState,
    attacker: &SkillProfile,
    defender: &SkillProfile,
    central: bool,
    space_open_side: f64,
    head: Option<&BeatDefenderHead>,
) -> Option<(BeatMove, f64, f64)> {
    let attacker_flair = ability01(attacker.flair_passing);
    let attacker_dribble = ability01(attacker.dribbling);
    // Candidate moves, each with the side it exits on.
    let mut best: Option<(BeatMove, f64, f64)> = None;
    let mut consider = |beat_move: BeatMove, side: f64| {
        let mut p = analytic_beat_probability(beat_move, commitment, attacker, defender);
        if let Some(head) = head {
            if learned_beat_defender_enabled() {
                if let Some(learned) = head.predict(&BeatDefenderInputs {
                    attacker_dribbling: attacker.dribbling,
                    attacker_acceleration: attacker.acceleration,
                    attacker_first_touch: attacker.first_touch,
                    defender_defending: defender.defending,
                    defender_tracking: defender.defensive_tracking,
                    reaction_seconds: commitment.reaction_seconds,
                    over_commit: commitment.over_commit,
                    closing_speed: commitment.closing_speed,
                    separation: commitment.separation,
                }) {
                    p = (p * (1.0 - BEAT_DEFENDER_HEAD_BLEND) + learned * BEAT_DEFENDER_HEAD_BLEND)
                        .clamp(0.05, 0.95);
                }
            }
        }
        if best.map_or(true, |(_, bp, _)| p > bp) {
            best = Some((beat_move, p, side));
        }
    };

    // Exit side: away from the lean, else into the more-open side.
    let exit_side = if commitment.open_side.abs() > 0.5 {
        commitment.open_side
    } else {
        space_open_side.signum()
    };
    let exit_side = if exit_side.abs() < 0.5 {
        1.0
    } else {
        exit_side
    };

    if commitment.is_over_committed() {
        // Already leaning/diving — exploit it directly.
        consider(BeatMove::GoOppositeMomentum, exit_side);
    } else {
        // Square/planted defender: either induce a lean (feint) if the carrier can
        // sell it, or nutmeg through a central planted man.
        if attacker_dribble >= 0.45 {
            consider(BeatMove::FeintThenOpposite, exit_side);
        }
        if central && commitment.separation <= 2.6 && attacker_flair >= 0.35 {
            consider(BeatMove::Nutmeg, exit_side);
        }
    }

    best.filter(|(_, p, _)| *p >= 0.30)
}

/// Learnable beat-probability head — a small regression net, seeded random and
/// trainable toward the analytic seed (supervised bootstrap) or RL beat outcomes
/// (reward-weighted). Mirrors [`super::back_four_line::BackFourLineHead`].
pub struct BeatDefenderHead {
    network: FeedForwardNetwork,
    training_steps: usize,
    last_loss: Option<f64>,
}

impl BeatDefenderHead {
    pub fn new(seed: u32) -> Self {
        let mut rng = mulberry32(seed ^ 0x51A3_9C7B);
        let network = FeedForwardNetwork::random(
            &RandomNetworkSpec {
                input_dim: BEAT_DEFENDER_FEATURE_DIM,
                hidden_layers: vec![BEAT_DEFENDER_HIDDEN_UNITS],
                output_dim: 1,
                hidden_activation: ActivationName::Tanh,
                // Sigmoid output = a probability in (0, 1).
                output_activation: ActivationName::Sigmoid,
                weight_scale: None,
            },
            &mut rng,
        );
        BeatDefenderHead {
            network,
            training_steps: 0,
            last_loss: None,
        }
    }

    /// Predicted beat probability in `(0, 1)`, or `None` on a malformed input.
    pub fn predict(&self, inputs: &BeatDefenderInputs) -> Option<f64> {
        let features = inputs.to_features();
        if features.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let out = self.network.predict(&features[..]);
        out.first().copied().filter(|p| p.is_finite())
    }

    /// One SGD epoch regressing the head toward `(inputs, target_probability)`
    /// pairs — a supervised bootstrap toward the analytic seed, or an RL target
    /// (1 = beat succeeded, 0 = dispossessed). Returns the mean step loss.
    pub fn train(&mut self, samples: &[(BeatDefenderInputs, f64)], learning_rate: f64) -> f64 {
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

    pub fn training_steps(&self) -> usize {
        self.training_steps
    }

    pub fn last_loss(&self) -> Option<f64> {
        self.last_loss
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn balanced_skills() -> SkillProfile {
        SkillProfile::default()
    }

    #[test]
    fn leaning_defender_opens_opposite_side() {
        let attack_dir = 1.0;
        let carrier = Vec2::new(30.0, 50.0);
        let defender = Vec2::new(30.0, 56.0);
        // Defender lunging toward +x.
        let state = DefenderCommitmentState::evaluate(
            attack_dir,
            carrier,
            defender,
            Vec2::new(4.0, 0.0),
            &balanced_skills(),
            0.0,
        );
        assert!(state.lateral_commit > 0.0);
        // Open side is the opposite of the lean (−x).
        assert!(state.open_side < 0.0, "open_side={}", state.open_side);
        assert!(
            state.is_over_committed(),
            "over_commit={}",
            state.over_commit
        );
    }

    #[test]
    fn over_committed_defender_is_more_beatable_than_planted() {
        let attack_dir = 1.0;
        let carrier = Vec2::new(30.0, 50.0);
        let defender = Vec2::new(30.0, 56.0);
        let attacker = balanced_skills();
        let defender_skills = balanced_skills();
        let leaning = DefenderCommitmentState::evaluate(
            attack_dir,
            carrier,
            defender,
            Vec2::new(5.0, 0.0),
            &defender_skills,
            0.0,
        );
        let planted = DefenderCommitmentState::evaluate(
            attack_dir,
            carrier,
            defender,
            Vec2::zero(),
            &defender_skills,
            0.0,
        );
        let p_lean = analytic_beat_probability(
            BeatMove::GoOppositeMomentum,
            &leaning,
            &attacker,
            &defender_skills,
        );
        let p_planted = analytic_beat_probability(
            BeatMove::GoOppositeMomentum,
            &planted,
            &attacker,
            &defender_skills,
        );
        assert!(
            p_lean > p_planted + 0.05,
            "leaning {p_lean} should beat planted {p_planted} more often"
        );
    }

    #[test]
    fn slow_reactor_is_easier_to_beat_than_quick_one() {
        // A defender with weak quickness traits has a longer perception lag, so a
        // sharp change of direction buys more free yards.
        // Skills are on a ~0–10 scale (see `SkillProfile::default`).
        let mut quick = balanced_skills();
        quick.acceleration = 9.0;
        quick.defensive_tracking = 9.5;
        quick.vision = 9.5;
        let mut slow = balanced_skills();
        slow.acceleration = 3.0;
        slow.defensive_tracking = 2.0;
        slow.vision = 2.0;
        let lag_quick = defender_perception_lag_seconds(&quick, 0.0);
        let lag_slow = defender_perception_lag_seconds(&slow, 0.0);
        assert!(lag_quick >= PERCEPTION_LAG_MIN_SECONDS - 1e-9);
        assert!(lag_slow <= PERCEPTION_LAG_MAX_SECONDS + 1e-9);
        assert!(
            lag_slow > lag_quick,
            "slow reactor lag {lag_slow} should exceed quick {lag_quick}"
        );
    }

    #[test]
    fn select_returns_none_when_gate_default_but_no_viable_move() {
        // A planted, square, distant defender with a poor dribbler => no move.
        let attack_dir = 1.0;
        // Skills on a ~0–10 scale: a weak dribbler vs an elite, planted defender.
        let mut poor = balanced_skills();
        poor.dribbling = 2.0;
        poor.flair_passing = 2.0;
        let strong_defender = {
            let mut s = balanced_skills();
            s.defending = 9.5;
            s.defensive_tracking = 9.5;
            s
        };
        let planted = DefenderCommitmentState::evaluate(
            attack_dir,
            Vec2::new(30.0, 50.0),
            Vec2::new(30.0, 55.0),
            Vec2::zero(),
            &strong_defender,
            0.0,
        );
        let pick = select_beat_move(&planted, &poor, &strong_defender, true, 1.0, None);
        assert!(pick.is_none());
    }

    #[test]
    fn head_predict_is_bounded_probability() {
        let head = BeatDefenderHead::new(11);
        let p = head
            .predict(&BeatDefenderInputs {
                attacker_dribbling: 70.0,
                attacker_acceleration: 7.0,
                attacker_first_touch: 70.0,
                defender_defending: 60.0,
                defender_tracking: 60.0,
                reaction_seconds: 0.18,
                over_commit: 1.5,
                closing_speed: 3.0,
                separation: 2.5,
            })
            .expect("finite prediction");
        assert!((0.0..=1.0).contains(&p), "p={p}");
    }

    /// Park every player well clear, then put a Home outfielder on the ball at
    /// the centre spot — a clean 1v1 canvas for the duel wiring. Returns the
    /// match, the carrier index, and the carrier's attacking y-direction.
    fn clean_1v1() -> (SoccerMatch, usize, f64) {
        let mut sim = SoccerMatch::default_11v11(MatchConfig::default());
        let fw = sim.config.field_width_yards;
        let fl = sim.config.field_length_yards;
        let carrier = sim
            .players
            .iter()
            .position(|p| p.team == Team::Home && p.role != PlayerRole::Goalkeeper)
            .expect("a Home outfielder");
        let attack = sim.players[carrier].team.attack_dir();
        let far = Vec2::new(fw * 0.5, fl * 0.5 - attack * 42.0);
        for p in sim.players.iter_mut() {
            p.position = far;
            p.velocity = Vec2::zero();
        }
        let carrier_pos = Vec2::new(fw * 0.5, fl * 0.5);
        sim.players[carrier].position = carrier_pos;
        sim.ball.holder = Some(carrier);
        sim.ball.position = carrier_pos;
        sim.ball.last_touch_team = Some(Team::Home);
        (sim, carrier, attack)
    }

    #[test]
    fn runaround_knocks_to_open_side_against_a_leaning_defender() {
        // Gate is ON by default. Carrier drives forward at a defender who is leaning hard to +x:
        // the open side is −x, and the duel should accept the take-on (over-commit relaxes the
        // pace gate) and knock the ball away from the lean. Skip if the duel was forced off.
        if !beat_defender_duel_enabled() {
            return;
        }
        let (mut sim, carrier, attack) = clean_1v1();
        let carrier_pos = sim.players[carrier].position;
        sim.players[carrier].velocity = Vec2::new(0.0, attack * 5.0);
        let defender = sim
            .players
            .iter()
            .position(|p| p.team == Team::Away && p.role != PlayerRole::Goalkeeper)
            .expect("an Away outfielder");
        sim.players[defender].position =
            Vec2::new(carrier_pos.x + 0.5, carrier_pos.y + attack * 3.0);
        sim.players[defender].velocity = Vec2::new(4.0, 0.0);
        let snapshot = WorldSnapshot::from_match(&sim);
        let plan = snapshot
            .runaround_dribble_option_for(carrier)
            .expect("an over-committed defender makes the take-on viable");
        assert_eq!(plan.defender, defender);
        let defender_x = sim.players[defender].position.x;
        assert!(
            plan.push_target.x < defender_x,
            "knock x={} should be on the open (−x) side of defender x={}",
            plan.push_target.x,
            defender_x
        );
    }

    #[test]
    fn shielding_carrier_peels_away_to_keep_distance() {
        // Gate ON by default. A carrier shielding (ProtectBall) with a defender tight at its back
        // and open space ahead should peel AWAY to keep distance, not freeze in place. Skip if the
        // duel was forced off.
        if !beat_defender_duel_enabled() {
            return;
        }
        let (mut sim, carrier, attack) = clean_1v1();
        let carrier_pos = sim.players[carrier].position;
        let defender = sim
            .players
            .iter()
            .position(|p| p.team == Team::Away && p.role != PlayerRole::Goalkeeper)
            .expect("an Away outfielder");
        let defender_pos = Vec2::new(carrier_pos.x, carrier_pos.y - attack * 2.0);
        sim.players[defender].position = defender_pos;
        let snapshot = WorldSnapshot::from_match(&sim);
        let target = snapshot
            .dribble_away_from_pressure_target_for(
                carrier,
                carrier_pos,
                Some(DribbleMoveKind::ProtectBall),
            )
            .expect("a pressured shielding carrier with space should peel away");
        assert!(
            target.distance(defender_pos) > carrier_pos.distance(defender_pos),
            "peel-away target {target:?} should keep more distance from the presser \
             {defender_pos:?} than holding at {carrier_pos:?}"
        );
    }
}
