//! Team brain: per-team coordination (home/away). The discrete strategy model
//! (attack/defense maneuvers + shapes), the team tactical directive, the
//! formation LP solver brain, and the [`CentralBrain`] that drives both teams
//! plus its shape/snapshot belief types. Extracted from the soccer god-module;
//! see `super` for shared primitives, the world snapshot, and reward helpers.

use super::*;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum FlankAttackPolicy {
    #[default]
    None,
    PlayDownFlankLowCross,
    PlayDownFlankHighCross,
}

impl FlankAttackPolicy {
    pub(crate) fn is_flank(self) -> bool {
        !matches!(self, FlankAttackPolicy::None)
    }

    pub(crate) fn prefers_low_cross(self) -> bool {
        matches!(self, FlankAttackPolicy::PlayDownFlankLowCross)
    }

    pub(crate) fn prefers_high_cross(self) -> bool {
        matches!(self, FlankAttackPolicy::PlayDownFlankHighCross)
    }
}

// ---------------------------------------------------------------------------
// Team-level strategy modes (semi-MDP options).
//
// These are *team-level* (not per-player) short tactical maneuvers — "options"
// in the hierarchical-RL sense — that the MDP/POMDP/neural brain selects from to
// create or deny openings. They are deliberately short-horizon and fully
// interruptible: most resolve within ~3 passes (`max_passes`), a few patient
// ones run 5-7, and the brain may abandon/switch the active strategy on any tick
// ("change on a dime"). The horizon is the *intended* lifetime, not a lock-in.
// ---------------------------------------------------------------------------

/// Coarse horizontal lane a maneuver pulls the ball into / resolves through.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum StrategyLane {
    Left,
    #[default]
    Center,
    Right,
}

impl StrategyLane {
    /// -1.0 left .. 0.0 central .. +1.0 right (drives directive flank bias).
    fn bias(self) -> f64 {
        match self {
            StrategyLane::Left => -1.0,
            StrategyLane::Center => 0.0,
            StrategyLane::Right => 1.0,
        }
    }
}

/// Strategic shape of an attacking maneuver: where it baits the defense first,
/// where it tries to resolve the opening, how patient it is, and how direct.
#[derive(Clone, Copy, Debug)]
pub struct AttackStrategyShape {
    pub start_lane: StrategyLane,
    pub resolve_lane: StrategyLane,
    pub max_passes: u8,
    pub directness: f64,
}

impl AttackStrategyShape {
    /// A maneuver that finishes in a different lane than it started is a switch.
    pub fn switches_lane(&self) -> bool {
        self.start_lane != self.resolve_lane
    }
}

/// Team-level attacking maneuvers (create openings). Short and interruptible.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TeamAttackStrategy {
    #[default]
    PullWideLeftThenCenter,
    PullWideRightThenCenter,
    PullWideLeftSwitchRight,
    PullWideRightSwitchLeft,
    HoldUpfieldUntilOpening,
    GiveAndGoCentral,
    OneTwoLeftRelease,
    OneTwoRightRelease,
    OverloadLeftIsolateRight,
    OverloadRightIsolateLeft,
    ThirdManRunCentral,
    QuickVerticalThroughBall,
    DrawPressThenPlayThrough,
    BaitOffsideThenThroughBall,
    WingOverlapLeftCross,
    WingOverlapRightCross,
    UnderlapLeftCutback,
    UnderlapRightCutback,
    SwitchPlayDiagonalLeftRight,
    SwitchPlayDiagonalRightLeft,
    RecycleViaKeeperReset,
    DragDefenderOpenChannelLeft,
    DragDefenderOpenChannelRight,
    DecoyFarSideCutbackLeft,
    DecoyFarSideCutbackRight,
    CentralDoubleOneTwo,
    DirectLongDiagonalLeft,
    DirectLongDiagonalRight,
    PatientPossessionProbe,
    CounterTransitionVertical,
    HalfSpaceComboLeft,
    HalfSpaceComboRight,
    /// In possession in the upper 2/3 (not near our own box) when play is too compact: play the
    /// ball BACKWARDS to bait the opponent into stepping up to win it, which opens the space to
    /// then play forward. Very patient / low directness; the inviting backward ball is the point.
    PlayBackwardsToGoad,
}

impl TeamAttackStrategy {
    pub const ALL: [TeamAttackStrategy; 33] = [
        TeamAttackStrategy::PullWideLeftThenCenter,
        TeamAttackStrategy::PullWideRightThenCenter,
        TeamAttackStrategy::PullWideLeftSwitchRight,
        TeamAttackStrategy::PullWideRightSwitchLeft,
        TeamAttackStrategy::HoldUpfieldUntilOpening,
        TeamAttackStrategy::GiveAndGoCentral,
        TeamAttackStrategy::OneTwoLeftRelease,
        TeamAttackStrategy::OneTwoRightRelease,
        TeamAttackStrategy::OverloadLeftIsolateRight,
        TeamAttackStrategy::OverloadRightIsolateLeft,
        TeamAttackStrategy::ThirdManRunCentral,
        TeamAttackStrategy::QuickVerticalThroughBall,
        TeamAttackStrategy::DrawPressThenPlayThrough,
        TeamAttackStrategy::BaitOffsideThenThroughBall,
        TeamAttackStrategy::WingOverlapLeftCross,
        TeamAttackStrategy::WingOverlapRightCross,
        TeamAttackStrategy::UnderlapLeftCutback,
        TeamAttackStrategy::UnderlapRightCutback,
        TeamAttackStrategy::SwitchPlayDiagonalLeftRight,
        TeamAttackStrategy::SwitchPlayDiagonalRightLeft,
        TeamAttackStrategy::RecycleViaKeeperReset,
        TeamAttackStrategy::DragDefenderOpenChannelLeft,
        TeamAttackStrategy::DragDefenderOpenChannelRight,
        TeamAttackStrategy::DecoyFarSideCutbackLeft,
        TeamAttackStrategy::DecoyFarSideCutbackRight,
        TeamAttackStrategy::CentralDoubleOneTwo,
        TeamAttackStrategy::DirectLongDiagonalLeft,
        TeamAttackStrategy::DirectLongDiagonalRight,
        TeamAttackStrategy::PatientPossessionProbe,
        TeamAttackStrategy::CounterTransitionVertical,
        TeamAttackStrategy::HalfSpaceComboLeft,
        TeamAttackStrategy::HalfSpaceComboRight,
        TeamAttackStrategy::PlayBackwardsToGoad,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            TeamAttackStrategy::PullWideLeftThenCenter => "pull-wide-left-then-center",
            TeamAttackStrategy::PullWideRightThenCenter => "pull-wide-right-then-center",
            TeamAttackStrategy::PullWideLeftSwitchRight => "pull-wide-left-switch-right",
            TeamAttackStrategy::PullWideRightSwitchLeft => "pull-wide-right-switch-left",
            TeamAttackStrategy::HoldUpfieldUntilOpening => "hold-upfield-until-opening",
            TeamAttackStrategy::GiveAndGoCentral => "give-and-go-central",
            TeamAttackStrategy::OneTwoLeftRelease => "one-two-left-release",
            TeamAttackStrategy::OneTwoRightRelease => "one-two-right-release",
            TeamAttackStrategy::OverloadLeftIsolateRight => "overload-left-isolate-right",
            TeamAttackStrategy::OverloadRightIsolateLeft => "overload-right-isolate-left",
            TeamAttackStrategy::ThirdManRunCentral => "third-man-run-central",
            TeamAttackStrategy::QuickVerticalThroughBall => "quick-vertical-through-ball",
            TeamAttackStrategy::DrawPressThenPlayThrough => "draw-press-then-play-through",
            TeamAttackStrategy::BaitOffsideThenThroughBall => "bait-offside-then-through-ball",
            TeamAttackStrategy::WingOverlapLeftCross => "wing-overlap-left-cross",
            TeamAttackStrategy::WingOverlapRightCross => "wing-overlap-right-cross",
            TeamAttackStrategy::UnderlapLeftCutback => "underlap-left-cutback",
            TeamAttackStrategy::UnderlapRightCutback => "underlap-right-cutback",
            TeamAttackStrategy::SwitchPlayDiagonalLeftRight => "switch-play-diagonal-left-right",
            TeamAttackStrategy::SwitchPlayDiagonalRightLeft => "switch-play-diagonal-right-left",
            TeamAttackStrategy::RecycleViaKeeperReset => "recycle-via-keeper-reset",
            TeamAttackStrategy::DragDefenderOpenChannelLeft => "drag-defender-open-channel-left",
            TeamAttackStrategy::DragDefenderOpenChannelRight => "drag-defender-open-channel-right",
            TeamAttackStrategy::DecoyFarSideCutbackLeft => "decoy-far-side-cutback-left",
            TeamAttackStrategy::DecoyFarSideCutbackRight => "decoy-far-side-cutback-right",
            TeamAttackStrategy::CentralDoubleOneTwo => "central-double-one-two",
            TeamAttackStrategy::DirectLongDiagonalLeft => "direct-long-diagonal-left",
            TeamAttackStrategy::DirectLongDiagonalRight => "direct-long-diagonal-right",
            TeamAttackStrategy::PatientPossessionProbe => "patient-possession-probe",
            TeamAttackStrategy::CounterTransitionVertical => "counter-transition-vertical",
            TeamAttackStrategy::HalfSpaceComboLeft => "half-space-combo-left",
            TeamAttackStrategy::HalfSpaceComboRight => "half-space-combo-right",
            TeamAttackStrategy::PlayBackwardsToGoad => "play-backwards-to-goad",
        }
    }

    pub fn shape(self) -> AttackStrategyShape {
        use StrategyLane::{Center, Left, Right};
        let s = |start, resolve, max_passes, directness| AttackStrategyShape {
            start_lane: start,
            resolve_lane: resolve,
            max_passes,
            directness,
        };
        match self {
            TeamAttackStrategy::PullWideLeftThenCenter => s(Left, Center, 3, 0.30),
            TeamAttackStrategy::PullWideRightThenCenter => s(Right, Center, 3, 0.30),
            TeamAttackStrategy::PullWideLeftSwitchRight => s(Left, Right, 3, 0.40),
            TeamAttackStrategy::PullWideRightSwitchLeft => s(Right, Left, 3, 0.40),
            TeamAttackStrategy::HoldUpfieldUntilOpening => s(Center, Center, 7, 0.20),
            TeamAttackStrategy::GiveAndGoCentral => s(Center, Center, 2, 0.55),
            TeamAttackStrategy::OneTwoLeftRelease => s(Left, Left, 2, 0.55),
            TeamAttackStrategy::OneTwoRightRelease => s(Right, Right, 2, 0.55),
            TeamAttackStrategy::OverloadLeftIsolateRight => s(Left, Right, 4, 0.35),
            TeamAttackStrategy::OverloadRightIsolateLeft => s(Right, Left, 4, 0.35),
            TeamAttackStrategy::ThirdManRunCentral => s(Center, Center, 3, 0.50),
            TeamAttackStrategy::QuickVerticalThroughBall => s(Center, Center, 2, 0.80),
            TeamAttackStrategy::DrawPressThenPlayThrough => s(Center, Center, 3, 0.45),
            TeamAttackStrategy::BaitOffsideThenThroughBall => s(Center, Center, 2, 0.78),
            TeamAttackStrategy::WingOverlapLeftCross => s(Left, Left, 3, 0.45),
            TeamAttackStrategy::WingOverlapRightCross => s(Right, Right, 3, 0.45),
            TeamAttackStrategy::UnderlapLeftCutback => s(Left, Center, 3, 0.40),
            TeamAttackStrategy::UnderlapRightCutback => s(Right, Center, 3, 0.40),
            TeamAttackStrategy::SwitchPlayDiagonalLeftRight => s(Left, Right, 2, 0.60),
            TeamAttackStrategy::SwitchPlayDiagonalRightLeft => s(Right, Left, 2, 0.60),
            TeamAttackStrategy::RecycleViaKeeperReset => s(Center, Center, 4, 0.10),
            TeamAttackStrategy::DragDefenderOpenChannelLeft => s(Left, Center, 3, 0.40),
            TeamAttackStrategy::DragDefenderOpenChannelRight => s(Right, Center, 3, 0.40),
            TeamAttackStrategy::DecoyFarSideCutbackLeft => s(Left, Center, 3, 0.45),
            TeamAttackStrategy::DecoyFarSideCutbackRight => s(Right, Center, 3, 0.45),
            TeamAttackStrategy::CentralDoubleOneTwo => s(Center, Center, 3, 0.58),
            TeamAttackStrategy::DirectLongDiagonalLeft => s(Center, Left, 1, 0.90),
            TeamAttackStrategy::DirectLongDiagonalRight => s(Center, Right, 1, 0.90),
            TeamAttackStrategy::PatientPossessionProbe => s(Center, Center, 6, 0.18),
            TeamAttackStrategy::CounterTransitionVertical => s(Center, Center, 2, 0.85),
            TeamAttackStrategy::HalfSpaceComboLeft => s(Left, Center, 3, 0.52),
            TeamAttackStrategy::HalfSpaceComboRight => s(Right, Center, 3, 0.52),
            // Very patient, lowest directness: the whole point is an inviting backward ball.
            TeamAttackStrategy::PlayBackwardsToGoad => s(Center, Center, 7, 0.06),
        }
    }
}

/// Strategic shape of a defensive maneuver.
#[derive(Clone, Copy, Debug)]
pub struct DefenseStrategyShape {
    /// 0.0 deep low block .. 1.0 high line.
    pub block_height: f64,
    /// 0.0 pure zonal .. 1.0 tight man-marking.
    pub man_orientation: f64,
    /// 0.0 passive contain .. 1.0 aggressive press.
    pub press: f64,
    /// Lane the maneuver tries to herd the ball into, if any.
    pub force_lane: Option<StrategyLane>,
    /// Intended lifetime in seconds before re-evaluation (still interruptible).
    pub max_seconds: f64,
}

/// Team-level defensive maneuvers (deny openings). Short and interruptible.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TeamDefenseStrategy {
    #[default]
    MidBlockZonal,
    HighPressManOriented,
    HighPressZonalTrigger,
    MidBlockManMark,
    LowBlockCompact,
    LowBlockParkBus,
    OffsideTrapStepUp,
    ForceWideLeftTrap,
    ForceWideRightTrap,
    PressTriggerOnBackPass,
    DropAndScreenCentral,
    ManMarkKeyReceiver,
    DoubleTeamBallCarrier,
    ContainAndDelayCounter,
    CounterPressGegenpress,
    RetreatRegroupHalfway,
    ShowInsideClampCentral,
    WidePressOverloadBall,
}

impl TeamDefenseStrategy {
    pub const ALL: [TeamDefenseStrategy; 18] = [
        TeamDefenseStrategy::MidBlockZonal,
        TeamDefenseStrategy::HighPressManOriented,
        TeamDefenseStrategy::HighPressZonalTrigger,
        TeamDefenseStrategy::MidBlockManMark,
        TeamDefenseStrategy::LowBlockCompact,
        TeamDefenseStrategy::LowBlockParkBus,
        TeamDefenseStrategy::OffsideTrapStepUp,
        TeamDefenseStrategy::ForceWideLeftTrap,
        TeamDefenseStrategy::ForceWideRightTrap,
        TeamDefenseStrategy::PressTriggerOnBackPass,
        TeamDefenseStrategy::DropAndScreenCentral,
        TeamDefenseStrategy::ManMarkKeyReceiver,
        TeamDefenseStrategy::DoubleTeamBallCarrier,
        TeamDefenseStrategy::ContainAndDelayCounter,
        TeamDefenseStrategy::CounterPressGegenpress,
        TeamDefenseStrategy::RetreatRegroupHalfway,
        TeamDefenseStrategy::ShowInsideClampCentral,
        TeamDefenseStrategy::WidePressOverloadBall,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            TeamDefenseStrategy::MidBlockZonal => "mid-block-zonal",
            TeamDefenseStrategy::HighPressManOriented => "high-press-man-oriented",
            TeamDefenseStrategy::HighPressZonalTrigger => "high-press-zonal-trigger",
            TeamDefenseStrategy::MidBlockManMark => "mid-block-man-mark",
            TeamDefenseStrategy::LowBlockCompact => "low-block-compact",
            TeamDefenseStrategy::LowBlockParkBus => "low-block-park-bus",
            TeamDefenseStrategy::OffsideTrapStepUp => "offside-trap-step-up",
            TeamDefenseStrategy::ForceWideLeftTrap => "force-wide-left-trap",
            TeamDefenseStrategy::ForceWideRightTrap => "force-wide-right-trap",
            TeamDefenseStrategy::PressTriggerOnBackPass => "press-trigger-on-back-pass",
            TeamDefenseStrategy::DropAndScreenCentral => "drop-and-screen-central",
            TeamDefenseStrategy::ManMarkKeyReceiver => "man-mark-key-receiver",
            TeamDefenseStrategy::DoubleTeamBallCarrier => "double-team-ball-carrier",
            TeamDefenseStrategy::ContainAndDelayCounter => "contain-and-delay-counter",
            TeamDefenseStrategy::CounterPressGegenpress => "counter-press-gegenpress",
            TeamDefenseStrategy::RetreatRegroupHalfway => "retreat-regroup-halfway",
            TeamDefenseStrategy::ShowInsideClampCentral => "show-inside-clamp-central",
            TeamDefenseStrategy::WidePressOverloadBall => "wide-press-overload-ball",
        }
    }

    pub fn shape(self) -> DefenseStrategyShape {
        use StrategyLane::{Left, Right};
        let d =
            |block_height, man_orientation, press, force_lane, max_seconds| DefenseStrategyShape {
                block_height,
                man_orientation,
                press,
                force_lane,
                max_seconds,
            };
        match self {
            TeamDefenseStrategy::MidBlockZonal => d(0.50, 0.20, 0.45, None, 8.0),
            TeamDefenseStrategy::HighPressManOriented => d(0.85, 0.85, 0.95, None, 4.0),
            TeamDefenseStrategy::HighPressZonalTrigger => d(0.82, 0.30, 0.90, None, 4.0),
            TeamDefenseStrategy::MidBlockManMark => d(0.50, 0.80, 0.55, None, 7.0),
            TeamDefenseStrategy::LowBlockCompact => d(0.22, 0.35, 0.30, None, 10.0),
            TeamDefenseStrategy::LowBlockParkBus => d(0.10, 0.30, 0.20, None, 12.0),
            TeamDefenseStrategy::OffsideTrapStepUp => d(0.92, 0.25, 0.60, None, 3.0),
            TeamDefenseStrategy::ForceWideLeftTrap => d(0.60, 0.40, 0.75, Some(Left), 4.0),
            TeamDefenseStrategy::ForceWideRightTrap => d(0.60, 0.40, 0.75, Some(Right), 4.0),
            TeamDefenseStrategy::PressTriggerOnBackPass => d(0.70, 0.45, 0.85, None, 3.0),
            TeamDefenseStrategy::DropAndScreenCentral => d(0.35, 0.25, 0.35, None, 8.0),
            TeamDefenseStrategy::ManMarkKeyReceiver => d(0.55, 0.95, 0.60, None, 6.0),
            TeamDefenseStrategy::DoubleTeamBallCarrier => d(0.55, 0.70, 0.88, None, 3.0),
            TeamDefenseStrategy::ContainAndDelayCounter => d(0.40, 0.30, 0.25, None, 6.0),
            TeamDefenseStrategy::CounterPressGegenpress => d(0.75, 0.55, 0.98, None, 3.0),
            TeamDefenseStrategy::RetreatRegroupHalfway => d(0.45, 0.20, 0.20, None, 5.0),
            TeamDefenseStrategy::ShowInsideClampCentral => d(0.58, 0.40, 0.70, None, 4.0),
            TeamDefenseStrategy::WidePressOverloadBall => d(0.62, 0.50, 0.82, None, 4.0),
        }
    }
}

#[cfg(test)]
mod team_strategy_mode_tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn defines_thirty_plus_attack_and_fifteen_plus_defense_modes() {
        // Team-level strategy catalogue: ~30-50 modes total (attack + defense).
        assert!(
            TeamAttackStrategy::ALL.len() >= 30,
            "expected >=30 attack modes, got {}",
            TeamAttackStrategy::ALL.len()
        );
        assert!(
            TeamDefenseStrategy::ALL.len() >= 15,
            "expected >=15 defense modes, got {}",
            TeamDefenseStrategy::ALL.len()
        );
        let total = TeamAttackStrategy::ALL.len() + TeamDefenseStrategy::ALL.len();
        assert!(
            (30..=80).contains(&total),
            "total team-level modes should land in the 30-50+ range, got {total}"
        );
    }

    #[test]
    fn strategy_labels_are_unique_and_shapes_are_sane() {
        let mut labels = HashSet::new();
        for strat in TeamAttackStrategy::ALL {
            assert!(labels.insert(strat.as_str()), "duplicate label {strat:?}");
            let shape = strat.shape();
            // Maneuvers are short-horizon: ~3 passes typical, up to 7 for the
            // patient "hold until an opening" plays.
            assert!(
                (1..=7).contains(&shape.max_passes),
                "{strat:?} horizon out of range: {}",
                shape.max_passes
            );
            assert!((0.0..=1.0).contains(&shape.directness));
        }
        for strat in TeamDefenseStrategy::ALL {
            assert!(labels.insert(strat.as_str()), "duplicate label {strat:?}");
            let shape = strat.shape();
            assert!((0.0..=1.0).contains(&shape.block_height));
            assert!((0.0..=1.0).contains(&shape.man_orientation));
            assert!((0.0..=1.0).contains(&shape.press));
            assert!(shape.max_seconds > 0.0 && shape.max_seconds <= 20.0);
        }
    }

    #[test]
    fn offside_recovery_reward_can_never_exceed_accrued_lingering_penalty() {
        // Hardening invariant: stepping back onside must never net positive, so a
        // player can't farm the recovery reward by briefly dipping offside.
        let dt = 0.1_f64;
        for steps_over_grace in 1..=120usize {
            // Simulate `steps_over_grace` ticks spent past the grace window.
            let mut clock = OFFSIDE_GRACE_SECONDS;
            let mut accrued_penalty = 0.0_f64;
            for _ in 0..steps_over_grace {
                clock += dt;
                let over = (clock - OFFSIDE_GRACE_SECONDS).min(OFFSIDE_LINGER_PENALTY_CAP_SECONDS);
                // Time term only — a strict lower bound on the real penalty, which
                // also adds an independent distance term.
                accrued_penalty += OFFSIDE_TIME_PENALTY_PER_SECOND * over * dt;
            }
            let lingered = (clock - OFFSIDE_GRACE_SECONDS).min(OFFSIDE_LINGER_PENALTY_CAP_SECONDS);
            let recovery = (0.4 * lingered * lingered).min(OFFSIDE_RECOVERY_REWARD);
            assert!(
                recovery <= accrued_penalty + 1e-9,
                "offside recovery {recovery} exceeded accrued penalty {accrued_penalty} after {steps_over_grace} ticks"
            );
        }
    }

    #[test]
    fn attack_switch_maneuvers_change_lane() {
        // A "switch" maneuver must resolve in a different lane than it starts.
        assert!(TeamAttackStrategy::PullWideLeftSwitchRight
            .shape()
            .switches_lane());
        assert!(TeamAttackStrategy::SwitchPlayDiagonalRightLeft
            .shape()
            .switches_lane());
        assert!(!TeamAttackStrategy::HoldUpfieldUntilOpening
            .shape()
            .switches_lane());
        // The patient hold-up option is the long-horizon one.
        assert_eq!(
            TeamAttackStrategy::HoldUpfieldUntilOpening
                .shape()
                .max_passes,
            7
        );
    }

    #[test]
    fn rapid_head_turning_accrues_dizziness_and_stable_facing_recovers() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig {
            dt_seconds: 0.1,
            seed: 7,
            ..Default::default()
        });
        let p = 5;
        sim.players[p].position = Vec2::new(40.0, 60.0);
        sim.players[p].facing_yaw = 0.0;
        sim.players[p].dizziness = 0.0;
        sim.ball.holder = None;
        // Force repeated large reversals: the ball jumps to opposite sides so the
        // player whips its head back and forth. Each side is held a few ticks so the
        // head actually winds up to a fast spin before reversing — with realistic
        // rotational inertia a body cannot build speed if the target flips EVERY tick
        // (it just jitters slowly); a real whip-around is a fast spin one way, then the
        // other, every few tenths of a second.
        for _ in 0..10 {
            sim.ball.position = Vec2::new(40.0, 95.0);
            for _ in 0..3 {
                sim.update_player_facing_dizziness_energy();
            }
            sim.ball.position = Vec2::new(40.0, 25.0);
            for _ in 0..3 {
                sim.update_player_facing_dizziness_energy();
            }
        }
        let dizzy = sim.players[p].dizziness;
        assert!(
            dizzy > 0.05,
            "rapid head reversals should accrue dizziness, got {dizzy}"
        );
        // Now settle facing the ball and rest: dizziness must recover.
        let to_ball = sim.ball.position - sim.players[p].position;
        sim.players[p].facing_yaw = to_ball.y.atan2(to_ball.x);
        let before_rest = sim.players[p].dizziness;
        for _ in 0..30 {
            sim.update_player_facing_dizziness_energy();
        }
        assert!(
            sim.players[p].dizziness < before_rest,
            "stable ball-watching should let dizziness recover: {} -> {}",
            before_rest,
            sim.players[p].dizziness
        );
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TeamTacticalDirective {
    pub team: Team,
    pub defensive_line_y: f64,
    pub defensive_cover_target: usize,
    pub defensive_cover_actual: usize,
    pub foremost_attacker_y: Option<f64>,
    #[serde(default)]
    pub attacking_overload_attackers: usize,
    #[serde(default)]
    pub attacking_overload_defenders: usize,
    #[serde(default)]
    pub attacking_numbers_advantage: i32,
    #[serde(default)]
    pub attacking_overload_score: f64,
    pub support_depth_yards: f64,
    pub width_yards: f64,
    pub press_intensity: f64,
    pub pass_priority: f64,
    pub carry_priority: f64,
    pub shot_threshold_yards: f64,
    pub risk_tolerance: f64,
    #[serde(default)]
    pub flank_attack_policy: FlankAttackPolicy,
    /// Active team-level attacking maneuver (semi-MDP option). Interruptible.
    #[serde(default)]
    pub attack_strategy: TeamAttackStrategy,
    /// Active team-level defensive maneuver (semi-MDP option). Interruptible.
    #[serde(default)]
    pub defense_strategy: TeamDefenseStrategy,
    #[serde(default)]
    pub flank_overlap_run_probability: f64,
    #[serde(default)]
    pub adversarial_embedding_attack_score: f64,
    #[serde(default)]
    pub adversarial_embedding_defense_score: f64,
    #[serde(default)]
    pub adversarial_embedding_hits: usize,
    /// Value-head forward-intent for this team's whole-field shape, in [0, 1]:
    /// how much the trained neural net judges that pushing the formation forward
    /// is worth right now. Folded into the formation-LP forward-pull coefficient,
    /// exactly like `adversarial_embedding_attack_score`. Zero unless a run opts
    /// into the neural blend and the value head has warmed up.
    #[serde(default)]
    pub neural_formation_attack_score: f64,
    /// Value-head defensive-restraint intent in [0, 1]: how much the net judges
    /// this team should hold shape / sit rather than commit forward.
    #[serde(default)]
    pub neural_formation_defense_score: f64,
}

impl TeamTacticalDirective {
    pub(crate) fn neutral(team: Team, field_width: f64, field_length: f64) -> Self {
        TeamTacticalDirective {
            team,
            defensive_line_y: match team {
                Team::Home => field_length * 0.40,
                Team::Away => field_length * 0.60,
            },
            defensive_cover_target: 2,
            defensive_cover_actual: 0,
            foremost_attacker_y: None,
            attacking_overload_attackers: 0,
            attacking_overload_defenders: 0,
            attacking_numbers_advantage: 0,
            attacking_overload_score: 0.0,
            support_depth_yards: 11.0,
            width_yards: field_width * 0.62,
            press_intensity: 0.48,
            pass_priority: 1.0,
            carry_priority: 1.0,
            shot_threshold_yards: 20.0,
            risk_tolerance: 0.50,
            flank_attack_policy: FlankAttackPolicy::None,
            attack_strategy: TeamAttackStrategy::default(),
            defense_strategy: TeamDefenseStrategy::default(),
            flank_overlap_run_probability: 0.0,
            adversarial_embedding_attack_score: 0.0,
            adversarial_embedding_defense_score: 0.0,
            adversarial_embedding_hits: 0,
            neural_formation_attack_score: 0.0,
            neural_formation_defense_score: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SoccerFormationLpPlayerVars {
    target_x: usize,
    target_y: usize,
    target_vx: usize,
    target_vy: usize,
    target_ax: usize,
    target_ay: usize,
    dynamics_slack_x: usize,
    dynamics_slack_y: usize,
    dynamics_slack_vx: usize,
    dynamics_slack_vy: usize,
    anchor_slack_x: usize,
    anchor_slack_y: usize,
    movement_slack_x: usize,
    movement_slack_y: usize,
    pressure_slack_x: usize,
    pressure_slack_y: usize,
    speed_match_slack_y: usize,
    acceleration_slack_x: usize,
    acceleration_slack_y: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SoccerFormationLpPlayerRows {
    anchor_x_upper: usize,
    anchor_x_lower: usize,
    anchor_y_upper: usize,
    anchor_y_lower: usize,
    movement_x_upper: usize,
    movement_x_lower: usize,
    movement_y_upper: usize,
    movement_y_lower: usize,
    reach_x_upper: usize,
    reach_x_lower: usize,
    reach_y_upper: usize,
    reach_y_lower: usize,
    pressure_x_upper: usize,
    pressure_x_lower: usize,
    pressure_y_upper: usize,
    pressure_y_lower: usize,
    speed_y_upper: usize,
    speed_y_lower: usize,
    dynamics_x_upper: usize,
    dynamics_x_lower: usize,
    dynamics_y_upper: usize,
    dynamics_y_lower: usize,
    dynamics_vx_upper: usize,
    dynamics_vx_lower: usize,
    dynamics_vy_upper: usize,
    dynamics_vy_lower: usize,
    acceleration_x_upper: usize,
    acceleration_x_lower: usize,
    acceleration_y_upper: usize,
    acceleration_y_lower: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SoccerFormationLpPairVars {
    pub(crate) a: usize,
    pub(crate) b: usize,
    pub(crate) separation_slack_x: usize,
    pub(crate) separation_slack_y: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SoccerFormationLpPairRows {
    separation_x_upper: usize,
    separation_x_lower: usize,
    separation_y_upper: usize,
    separation_y_lower: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SoccerFormationLpStateVars {
    players_start: usize,
    ball_start: usize,
    context_start: usize,
    count: usize,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerFormationLpObjectiveWeights {
    pub expected_goal: f64,
    pub progression: f64,
    pub retention: f64,
    pub expected_goals_against: f64,
    pub fatigue: f64,
    pub passing_lane_quality: f64,
    pub space_occupation: f64,
    pub numerical_superiority: f64,
    pub press_resistance: f64,
    pub defensive_compactness: f64,
    pub opponent_progression: f64,
    pub transition_risk: f64,
    #[serde(default)]
    pub adversarial_embedding_attack: f64,
    #[serde(default)]
    pub adversarial_embedding_defense: f64,
    /// Value-head forward-intent pull on the whole-field shape (see
    /// [`TeamTacticalDirective::neural_formation_attack_score`]). Enters the
    /// per-slot forward-pull coefficient alongside `adversarial_embedding_attack`.
    #[serde(default)]
    pub neural_attack: f64,
    /// Value-head defensive-restraint pull (holds the shape back).
    #[serde(default)]
    pub neural_defense: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerFormationLpPlayerGuidance {
    pub team: Team,
    pub player_id: usize,
    pub slot: usize,
    pub current: Vec2,
    pub target: Vec2,
    pub formation_anchor: Vec2,
    #[serde(default)]
    pub pressure_target: Option<Vec2>,
    pub recommended_move: Vec2,
    #[serde(default)]
    pub target_velocity: Vec2,
    #[serde(default)]
    pub target_acceleration: Vec2,
    #[serde(default)]
    pub local_mpc_guidance: bool,
    pub recommended_move_yards: f64,
    #[serde(default)]
    pub recommended_speed_yps: f64,
    #[serde(default)]
    pub recommended_acceleration_yps2: f64,
    pub formation_error_yards: f64,
    pub movement_error_yards: f64,
    pub pair_error_yards: f64,
    #[serde(default)]
    pub role_line_error_yards: f64,
    pub pressure_error_yards: f64,
    pub fore_aft_speed_error_yps: f64,
    pub pressure_weight: f64,
    pub speed_match_weight: f64,
    pub alignment_weight: f64,
    pub reduced_cost_target_x: f64,
    pub reduced_cost_target_y: f64,
    pub solver_status: String,
    pub solver_objective: f64,
    pub solver_iterations: Option<usize>,
    pub solver_elapsed_ms: f64,
    pub variable_count: usize,
    pub constraint_count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerFormationLpTeamSnapshot {
    pub team: Team,
    pub status: String,
    pub objective: f64,
    pub variables: usize,
    pub constraints: usize,
    pub guided_players: usize,
    pub iterations: Option<usize>,
    pub elapsed_ms: f64,
    pub mean_pair_error_yards: f64,
    #[serde(default)]
    pub mean_role_line_error_yards: f64,
    pub mean_pressure_error_yards: f64,
    pub objective_weights: SoccerFormationLpObjectiveWeights,
}

impl Default for SoccerFormationLpTeamSnapshot {
    fn default() -> Self {
        Self {
            team: Team::Home,
            status: "not-solved".to_string(),
            objective: 0.0,
            variables: 0,
            constraints: 0,
            guided_players: 0,
            iterations: None,
            elapsed_ms: 0.0,
            mean_pair_error_yards: 0.0,
            mean_role_line_error_yards: 0.0,
            mean_pressure_error_yards: 0.0,
            objective_weights: SoccerFormationLpObjectiveWeights::default(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SoccerFormationLpSlotInput {
    pub(crate) active: bool,
    pub(crate) player_id: usize,
    pub(crate) role: PlayerRole,
    pub(crate) current: Vec2,
    pub(crate) velocity: Vec2,
    pub(crate) acceleration: Vec2,
    pub(crate) home_position: Vec2,
    pub(crate) top_speed: f64,
    pub(crate) max_acceleration: f64,
    pub(crate) fatigue: f64,
    pub(crate) anchor: Vec2,
    pub(crate) pressure_target: Option<Vec2>,
    pub(crate) speed_match_velocity_yps: f64,
    pub(crate) pressure_weight: f64,
    pub(crate) speed_match_weight: f64,
    pub(crate) pair_weight: f64,
    pub(crate) alignment_weight: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct SoccerFormationLpBrain {
    pub(crate) team: Team,
    pub(crate) problem: LPProblem,
    pub(crate) player_vars: Vec<SoccerFormationLpPlayerVars>,
    pub(crate) player_rows: Vec<SoccerFormationLpPlayerRows>,
    pub(crate) pair_vars: Vec<SoccerFormationLpPairVars>,
    pub(crate) pair_rows: Vec<SoccerFormationLpPairRows>,
    pub(crate) state_vars: SoccerFormationLpStateVars,
    pub(crate) basis_start: Option<LPBasisWarmStart>,
    pub(crate) last_guidance: Vec<SoccerFormationLpPlayerGuidance>,
    pub(crate) last_snapshot: SoccerFormationLpTeamSnapshot,
    /// Performance guardrails + telemetry: the iteration cap adapts down when a
    /// solve runs over the per-tick budget (so we bail within time constraints)
    /// and recovers when fast; counters surface slow/suboptimal solves.
    pub(crate) adaptive_max_iter: usize,
    pub(crate) last_solve_micros: u128,
    pub(crate) last_solve_status: String,
    pub(crate) solve_count: u64,
    pub(crate) solve_timeout_count: u64,
    pub(crate) suboptimal_count: u64,
    pub(crate) consecutive_timeouts: u32,
    pub(crate) simplex_circuit_open: bool,
}

fn soccer_formation_lp_offensive_overlap_relief(
    team: Team,
    runner: &SoccerFormationLpSlotInput,
    teammate: &SoccerFormationLpSlotInput,
) -> f64 {
    if !runner.active
        || !teammate.active
        || matches!(runner.role, PlayerRole::Goalkeeper)
        || matches!(teammate.role, PlayerRole::Goalkeeper)
    {
        return 0.0;
    }
    let distance = runner.current.distance(teammate.current);
    if distance > OFFENSIVE_OVERLAP_RELIEF_TEAMMATE_RADIUS_YARDS {
        return 0.0;
    }
    let speed = runner.velocity.len();
    let forward_speed = (runner.velocity.y * team.attack_dir()).max(0.0);
    let projected = runner.current + runner.velocity * OFFENSIVE_OVERLAP_RELIEF_LOOKAHEAD_SECONDS;
    let current_vs_teammate = (runner.current.y - teammate.current.y) * team.attack_dir();
    let projected_vs_teammate = (projected.y - teammate.current.y) * team.attack_dir();
    let passing_fit = ((projected_vs_teammate.max(current_vs_teammate)
        - OFFENSIVE_OVERLAP_RELIEF_MIN_FORWARD_YARDS)
        / 3.0)
        .clamp(0.0, 1.0);
    let close_fit = ((OFFENSIVE_OVERLAP_RELIEF_TEAMMATE_RADIUS_YARDS - distance)
        / (OFFENSIVE_OVERLAP_RELIEF_TEAMMATE_RADIUS_YARDS
            - TEAMMATE_OCCUPIED_SPACE_HARD_RADIUS_YARDS)
            .max(1e-6))
    .clamp(0.0, 1.0);
    let speed_fit = ((speed - 4.5) / 3.0).clamp(0.0, 1.0);
    let forward_fit = ((forward_speed - 3.0) / 6.0).clamp(0.0, 1.0);
    (passing_fit * close_fit * speed_fit * forward_fit * 0.55).clamp(0.0, 0.55)
}

pub(crate) fn soccer_formation_lp_pair_shape_relief(
    snapshot: &WorldSnapshot,
    team: Team,
    a: &SoccerFormationLpSlotInput,
    b: &SoccerFormationLpSlotInput,
) -> f64 {
    let possession = snapshot
        .controlled_possession_team()
        .or_else(|| snapshot.possession_team());
    let defensive_tracking_relief = if possession == Some(team.other()) {
        soccer_lp_unit((a.pressure_weight + b.pressure_weight) * 0.5 * 0.58)
    } else {
        0.0
    };
    let offensive_overlap_relief = if possession == Some(team) {
        soccer_formation_lp_offensive_overlap_relief(team, a, b)
            .max(soccer_formation_lp_offensive_overlap_relief(team, b, a))
    } else {
        0.0
    };
    defensive_tracking_relief
        .max(offensive_overlap_relief)
        .clamp(0.0, 0.85)
}

/// `SOCCER_LP_DEBUG=1` logs slow/non-optimal LP solves to stderr.
fn soccer_lp_debug_enabled() -> bool {
    std::env::var("SOCCER_LP_DEBUG")
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(false)
}

fn soccer_formation_lp_internal_simplex_enabled() -> bool {
    std::env::var("SOCCER_FORMATION_LP_INTERNAL_SIMPLEX")
        .map(|value| {
            matches!(
                value.as_str(),
                "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"
            )
        })
        .unwrap_or(false)
}

fn soccer_formation_lp_budgeted_fallback_solution() -> crate::des::general::lp::LPSolution {
    crate::des::general::lp::LPSolution {
        status: LPStatus::IterLimit,
        x: Vec::new(),
        objective: f64::NAN,
        dual_ub: None,
        dual_eq: None,
        reduced_costs: None,
        var_basis: None,
        row_basis: None,
        unbounded_ray: None,
        infeasibility_certificate: None,
        iters: Some(0),
        solver: "formation-lp-budgeted-fallback".to_string(),
        elapsed_ms: 0.0,
        message: Some(
            "internal simplex disabled for realtime soccer tick; set SOCCER_FORMATION_LP_INTERNAL_SIMPLEX=1 for exact solve"
                .to_string(),
        ),
    }
}

impl SoccerFormationLpBrain {
    pub(crate) fn new(team: Team) -> Self {
        let mut c = Vec::new();
        let mut lb = Vec::new();
        let mut ub = Vec::new();
        let mut var_names = Vec::new();
        let mut player_vars = Vec::new();
        let mut pair_vars = Vec::new();

        fn add_var(
            c: &mut Vec<f64>,
            lb: &mut Vec<Option<f64>>,
            ub: &mut Vec<Option<f64>>,
            var_names: &mut Vec<String>,
            name: String,
            lower: Option<f64>,
            upper: Option<f64>,
        ) -> usize {
            let idx = c.len();
            c.push(0.0);
            lb.push(lower);
            ub.push(upper);
            var_names.push(name);
            idx
        }

        for slot in 0..SOCCER_FORMATION_LP_PLAYER_CAPACITY {
            let prefix = format!("{}:slot{slot}", team.label().to_ascii_lowercase());
            player_vars.push(SoccerFormationLpPlayerVars {
                target_x: add_var(
                    &mut c,
                    &mut lb,
                    &mut ub,
                    &mut var_names,
                    format!("{prefix}:target_x"),
                    Some(0.0),
                    Some(DEFAULT_FIELD_WIDTH_YARDS),
                ),
                target_y: add_var(
                    &mut c,
                    &mut lb,
                    &mut ub,
                    &mut var_names,
                    format!("{prefix}:target_y"),
                    Some(0.0),
                    Some(DEFAULT_FIELD_LENGTH_YARDS),
                ),
                target_vx: add_var(
                    &mut c,
                    &mut lb,
                    &mut ub,
                    &mut var_names,
                    format!("{prefix}:target_vx"),
                    Some(-20.0),
                    Some(20.0),
                ),
                target_vy: add_var(
                    &mut c,
                    &mut lb,
                    &mut ub,
                    &mut var_names,
                    format!("{prefix}:target_vy"),
                    Some(-20.0),
                    Some(20.0),
                ),
                target_ax: add_var(
                    &mut c,
                    &mut lb,
                    &mut ub,
                    &mut var_names,
                    format!("{prefix}:target_ax"),
                    Some(-30.0),
                    Some(30.0),
                ),
                target_ay: add_var(
                    &mut c,
                    &mut lb,
                    &mut ub,
                    &mut var_names,
                    format!("{prefix}:target_ay"),
                    Some(-30.0),
                    Some(30.0),
                ),
                dynamics_slack_x: add_var(
                    &mut c,
                    &mut lb,
                    &mut ub,
                    &mut var_names,
                    format!("{prefix}:dynamics_slack_x"),
                    Some(0.0),
                    None,
                ),
                dynamics_slack_y: add_var(
                    &mut c,
                    &mut lb,
                    &mut ub,
                    &mut var_names,
                    format!("{prefix}:dynamics_slack_y"),
                    Some(0.0),
                    None,
                ),
                dynamics_slack_vx: add_var(
                    &mut c,
                    &mut lb,
                    &mut ub,
                    &mut var_names,
                    format!("{prefix}:dynamics_slack_vx"),
                    Some(0.0),
                    None,
                ),
                dynamics_slack_vy: add_var(
                    &mut c,
                    &mut lb,
                    &mut ub,
                    &mut var_names,
                    format!("{prefix}:dynamics_slack_vy"),
                    Some(0.0),
                    None,
                ),
                anchor_slack_x: add_var(
                    &mut c,
                    &mut lb,
                    &mut ub,
                    &mut var_names,
                    format!("{prefix}:anchor_slack_x"),
                    Some(0.0),
                    None,
                ),
                anchor_slack_y: add_var(
                    &mut c,
                    &mut lb,
                    &mut ub,
                    &mut var_names,
                    format!("{prefix}:anchor_slack_y"),
                    Some(0.0),
                    None,
                ),
                movement_slack_x: add_var(
                    &mut c,
                    &mut lb,
                    &mut ub,
                    &mut var_names,
                    format!("{prefix}:movement_slack_x"),
                    Some(0.0),
                    None,
                ),
                movement_slack_y: add_var(
                    &mut c,
                    &mut lb,
                    &mut ub,
                    &mut var_names,
                    format!("{prefix}:movement_slack_y"),
                    Some(0.0),
                    None,
                ),
                pressure_slack_x: add_var(
                    &mut c,
                    &mut lb,
                    &mut ub,
                    &mut var_names,
                    format!("{prefix}:pressure_slack_x"),
                    Some(0.0),
                    None,
                ),
                pressure_slack_y: add_var(
                    &mut c,
                    &mut lb,
                    &mut ub,
                    &mut var_names,
                    format!("{prefix}:pressure_slack_y"),
                    Some(0.0),
                    None,
                ),
                speed_match_slack_y: add_var(
                    &mut c,
                    &mut lb,
                    &mut ub,
                    &mut var_names,
                    format!("{prefix}:speed_match_slack_y"),
                    Some(0.0),
                    None,
                ),
                acceleration_slack_x: add_var(
                    &mut c,
                    &mut lb,
                    &mut ub,
                    &mut var_names,
                    format!("{prefix}:acceleration_slack_x"),
                    Some(0.0),
                    None,
                ),
                acceleration_slack_y: add_var(
                    &mut c,
                    &mut lb,
                    &mut ub,
                    &mut var_names,
                    format!("{prefix}:acceleration_slack_y"),
                    Some(0.0),
                    None,
                ),
            });
        }

        for a in 0..SOCCER_FORMATION_LP_PLAYER_CAPACITY {
            for b in (a + 1)..SOCCER_FORMATION_LP_PLAYER_CAPACITY {
                let prefix = format!("{}:pair{a}_{b}", team.label().to_ascii_lowercase());
                pair_vars.push(SoccerFormationLpPairVars {
                    a,
                    b,
                    separation_slack_x: add_var(
                        &mut c,
                        &mut lb,
                        &mut ub,
                        &mut var_names,
                        format!("{prefix}:separation_slack_x"),
                        Some(0.0),
                        None,
                    ),
                    separation_slack_y: add_var(
                        &mut c,
                        &mut lb,
                        &mut ub,
                        &mut var_names,
                        format!("{prefix}:separation_slack_y"),
                        Some(0.0),
                        None,
                    ),
                });
            }
        }

        let players_start = c.len();
        for slot in 0..SOCCER_FORMATION_LP_WORLD_PLAYER_CAPACITY {
            for component in ["pos_x", "pos_y", "vel_x", "vel_y", "acc_x", "acc_y"] {
                add_var(
                    &mut c,
                    &mut lb,
                    &mut ub,
                    &mut var_names,
                    format!("world:player{slot}:{component}"),
                    Some(-2.0),
                    Some(2.0),
                );
            }
        }
        let ball_start = c.len();
        for component in ["pos_x", "pos_y", "vel_x", "vel_y", "acc_x", "acc_y"] {
            add_var(
                &mut c,
                &mut lb,
                &mut ub,
                &mut var_names,
                format!("world:ball:{component}"),
                Some(-2.0),
                Some(2.0),
            );
        }
        let context_start = c.len();
        for idx in 0..SOCCER_FORMATION_LP_CONTEXT_FEATURES {
            add_var(
                &mut c,
                &mut lb,
                &mut ub,
                &mut var_names,
                format!("context:feature{idx}"),
                Some(-2.0),
                Some(2.0),
            );
        }
        let state_count = c.len() - players_start;
        let state_vars = SoccerFormationLpStateVars {
            players_start,
            ball_start,
            context_start,
            count: state_count,
        };

        let n = c.len();
        let mut a_ub = Vec::new();
        let mut b_ub = Vec::new();
        let mut con_names = Vec::new();
        let mut player_rows = Vec::new();
        let mut pair_rows = Vec::new();

        fn add_ub_row(
            a_ub: &mut Vec<Vec<f64>>,
            b_ub: &mut Vec<f64>,
            con_names: &mut Vec<String>,
            n: usize,
            terms: &[(usize, f64)],
            name: String,
        ) -> usize {
            let idx = a_ub.len();
            let mut row = vec![0.0; n];
            for (var, coef) in terms {
                row[*var] = *coef;
            }
            a_ub.push(row);
            b_ub.push(0.0);
            con_names.push(name);
            idx
        }

        for (slot, vars) in player_vars.iter().enumerate() {
            let prefix = format!("{}:slot{slot}", team.label().to_ascii_lowercase());
            player_rows.push(SoccerFormationLpPlayerRows {
                anchor_x_upper: add_ub_row(
                    &mut a_ub,
                    &mut b_ub,
                    &mut con_names,
                    n,
                    &[(vars.target_x, 1.0), (vars.anchor_slack_x, -1.0)],
                    format!("{prefix}:anchor_x_upper"),
                ),
                anchor_x_lower: add_ub_row(
                    &mut a_ub,
                    &mut b_ub,
                    &mut con_names,
                    n,
                    &[(vars.target_x, -1.0), (vars.anchor_slack_x, -1.0)],
                    format!("{prefix}:anchor_x_lower"),
                ),
                anchor_y_upper: add_ub_row(
                    &mut a_ub,
                    &mut b_ub,
                    &mut con_names,
                    n,
                    &[(vars.target_y, 1.0), (vars.anchor_slack_y, -1.0)],
                    format!("{prefix}:anchor_y_upper"),
                ),
                anchor_y_lower: add_ub_row(
                    &mut a_ub,
                    &mut b_ub,
                    &mut con_names,
                    n,
                    &[(vars.target_y, -1.0), (vars.anchor_slack_y, -1.0)],
                    format!("{prefix}:anchor_y_lower"),
                ),
                movement_x_upper: add_ub_row(
                    &mut a_ub,
                    &mut b_ub,
                    &mut con_names,
                    n,
                    &[(vars.target_x, 1.0), (vars.movement_slack_x, -1.0)],
                    format!("{prefix}:movement_x_upper"),
                ),
                movement_x_lower: add_ub_row(
                    &mut a_ub,
                    &mut b_ub,
                    &mut con_names,
                    n,
                    &[(vars.target_x, -1.0), (vars.movement_slack_x, -1.0)],
                    format!("{prefix}:movement_x_lower"),
                ),
                movement_y_upper: add_ub_row(
                    &mut a_ub,
                    &mut b_ub,
                    &mut con_names,
                    n,
                    &[(vars.target_y, 1.0), (vars.movement_slack_y, -1.0)],
                    format!("{prefix}:movement_y_upper"),
                ),
                movement_y_lower: add_ub_row(
                    &mut a_ub,
                    &mut b_ub,
                    &mut con_names,
                    n,
                    &[(vars.target_y, -1.0), (vars.movement_slack_y, -1.0)],
                    format!("{prefix}:movement_y_lower"),
                ),
                reach_x_upper: add_ub_row(
                    &mut a_ub,
                    &mut b_ub,
                    &mut con_names,
                    n,
                    &[(vars.target_x, 1.0)],
                    format!("{prefix}:reach_x_upper"),
                ),
                reach_x_lower: add_ub_row(
                    &mut a_ub,
                    &mut b_ub,
                    &mut con_names,
                    n,
                    &[(vars.target_x, -1.0)],
                    format!("{prefix}:reach_x_lower"),
                ),
                reach_y_upper: add_ub_row(
                    &mut a_ub,
                    &mut b_ub,
                    &mut con_names,
                    n,
                    &[(vars.target_y, 1.0)],
                    format!("{prefix}:reach_y_upper"),
                ),
                reach_y_lower: add_ub_row(
                    &mut a_ub,
                    &mut b_ub,
                    &mut con_names,
                    n,
                    &[(vars.target_y, -1.0)],
                    format!("{prefix}:reach_y_lower"),
                ),
                pressure_x_upper: add_ub_row(
                    &mut a_ub,
                    &mut b_ub,
                    &mut con_names,
                    n,
                    &[(vars.target_x, 1.0), (vars.pressure_slack_x, -1.0)],
                    format!("{prefix}:pressure_x_upper"),
                ),
                pressure_x_lower: add_ub_row(
                    &mut a_ub,
                    &mut b_ub,
                    &mut con_names,
                    n,
                    &[(vars.target_x, -1.0), (vars.pressure_slack_x, -1.0)],
                    format!("{prefix}:pressure_x_lower"),
                ),
                pressure_y_upper: add_ub_row(
                    &mut a_ub,
                    &mut b_ub,
                    &mut con_names,
                    n,
                    &[(vars.target_y, 1.0), (vars.pressure_slack_y, -1.0)],
                    format!("{prefix}:pressure_y_upper"),
                ),
                pressure_y_lower: add_ub_row(
                    &mut a_ub,
                    &mut b_ub,
                    &mut con_names,
                    n,
                    &[(vars.target_y, -1.0), (vars.pressure_slack_y, -1.0)],
                    format!("{prefix}:pressure_y_lower"),
                ),
                speed_y_upper: add_ub_row(
                    &mut a_ub,
                    &mut b_ub,
                    &mut con_names,
                    n,
                    &[(vars.target_y, 1.0), (vars.speed_match_slack_y, -1.0)],
                    format!("{prefix}:speed_y_upper"),
                ),
                speed_y_lower: add_ub_row(
                    &mut a_ub,
                    &mut b_ub,
                    &mut con_names,
                    n,
                    &[(vars.target_y, -1.0), (vars.speed_match_slack_y, -1.0)],
                    format!("{prefix}:speed_y_lower"),
                ),
                dynamics_x_upper: add_ub_row(
                    &mut a_ub,
                    &mut b_ub,
                    &mut con_names,
                    n,
                    &[
                        (vars.target_x, 1.0),
                        (vars.target_vx, 0.0),
                        (vars.dynamics_slack_x, -1.0),
                    ],
                    format!("{prefix}:dynamics_x_upper"),
                ),
                dynamics_x_lower: add_ub_row(
                    &mut a_ub,
                    &mut b_ub,
                    &mut con_names,
                    n,
                    &[
                        (vars.target_x, -1.0),
                        (vars.target_vx, 0.0),
                        (vars.dynamics_slack_x, -1.0),
                    ],
                    format!("{prefix}:dynamics_x_lower"),
                ),
                dynamics_y_upper: add_ub_row(
                    &mut a_ub,
                    &mut b_ub,
                    &mut con_names,
                    n,
                    &[
                        (vars.target_y, 1.0),
                        (vars.target_vy, 0.0),
                        (vars.dynamics_slack_y, -1.0),
                    ],
                    format!("{prefix}:dynamics_y_upper"),
                ),
                dynamics_y_lower: add_ub_row(
                    &mut a_ub,
                    &mut b_ub,
                    &mut con_names,
                    n,
                    &[
                        (vars.target_y, -1.0),
                        (vars.target_vy, 0.0),
                        (vars.dynamics_slack_y, -1.0),
                    ],
                    format!("{prefix}:dynamics_y_lower"),
                ),
                dynamics_vx_upper: add_ub_row(
                    &mut a_ub,
                    &mut b_ub,
                    &mut con_names,
                    n,
                    &[
                        (vars.target_vx, 1.0),
                        (vars.target_ax, 0.0),
                        (vars.dynamics_slack_vx, -1.0),
                    ],
                    format!("{prefix}:dynamics_vx_upper"),
                ),
                dynamics_vx_lower: add_ub_row(
                    &mut a_ub,
                    &mut b_ub,
                    &mut con_names,
                    n,
                    &[
                        (vars.target_vx, -1.0),
                        (vars.target_ax, 0.0),
                        (vars.dynamics_slack_vx, -1.0),
                    ],
                    format!("{prefix}:dynamics_vx_lower"),
                ),
                dynamics_vy_upper: add_ub_row(
                    &mut a_ub,
                    &mut b_ub,
                    &mut con_names,
                    n,
                    &[
                        (vars.target_vy, 1.0),
                        (vars.target_ay, 0.0),
                        (vars.dynamics_slack_vy, -1.0),
                    ],
                    format!("{prefix}:dynamics_vy_upper"),
                ),
                dynamics_vy_lower: add_ub_row(
                    &mut a_ub,
                    &mut b_ub,
                    &mut con_names,
                    n,
                    &[
                        (vars.target_vy, -1.0),
                        (vars.target_ay, 0.0),
                        (vars.dynamics_slack_vy, -1.0),
                    ],
                    format!("{prefix}:dynamics_vy_lower"),
                ),
                acceleration_x_upper: add_ub_row(
                    &mut a_ub,
                    &mut b_ub,
                    &mut con_names,
                    n,
                    &[(vars.target_ax, 1.0), (vars.acceleration_slack_x, -1.0)],
                    format!("{prefix}:acceleration_x_upper"),
                ),
                acceleration_x_lower: add_ub_row(
                    &mut a_ub,
                    &mut b_ub,
                    &mut con_names,
                    n,
                    &[(vars.target_ax, -1.0), (vars.acceleration_slack_x, -1.0)],
                    format!("{prefix}:acceleration_x_lower"),
                ),
                acceleration_y_upper: add_ub_row(
                    &mut a_ub,
                    &mut b_ub,
                    &mut con_names,
                    n,
                    &[(vars.target_ay, 1.0), (vars.acceleration_slack_y, -1.0)],
                    format!("{prefix}:acceleration_y_upper"),
                ),
                acceleration_y_lower: add_ub_row(
                    &mut a_ub,
                    &mut b_ub,
                    &mut con_names,
                    n,
                    &[(vars.target_ay, -1.0), (vars.acceleration_slack_y, -1.0)],
                    format!("{prefix}:acceleration_y_lower"),
                ),
            });
        }

        for pair in &pair_vars {
            let vars_a = player_vars[pair.a];
            let vars_b = player_vars[pair.b];
            let prefix = format!(
                "{}:pair{}_{}",
                team.label().to_ascii_lowercase(),
                pair.a,
                pair.b
            );
            pair_rows.push(SoccerFormationLpPairRows {
                separation_x_upper: add_ub_row(
                    &mut a_ub,
                    &mut b_ub,
                    &mut con_names,
                    n,
                    &[
                        (vars_a.target_x, 1.0),
                        (vars_b.target_x, -1.0),
                        (pair.separation_slack_x, -1.0),
                    ],
                    format!("{prefix}:separation_x_upper"),
                ),
                separation_x_lower: add_ub_row(
                    &mut a_ub,
                    &mut b_ub,
                    &mut con_names,
                    n,
                    &[
                        (vars_a.target_x, -1.0),
                        (vars_b.target_x, 1.0),
                        (pair.separation_slack_x, -1.0),
                    ],
                    format!("{prefix}:separation_x_lower"),
                ),
                separation_y_upper: add_ub_row(
                    &mut a_ub,
                    &mut b_ub,
                    &mut con_names,
                    n,
                    &[
                        (vars_a.target_y, 1.0),
                        (vars_b.target_y, -1.0),
                        (pair.separation_slack_y, -1.0),
                    ],
                    format!("{prefix}:separation_y_upper"),
                ),
                separation_y_lower: add_ub_row(
                    &mut a_ub,
                    &mut b_ub,
                    &mut con_names,
                    n,
                    &[
                        (vars_a.target_y, -1.0),
                        (vars_b.target_y, 1.0),
                        (pair.separation_slack_y, -1.0),
                    ],
                    format!("{prefix}:separation_y_lower"),
                ),
            });
        }

        let mut a_eq = Vec::new();
        let mut b_eq = Vec::new();
        for var in state_vars.players_start..(state_vars.players_start + state_vars.count) {
            let mut row = vec![0.0; n];
            row[var] = 1.0;
            a_eq.push(row);
            b_eq.push(0.0);
            con_names.push(format!("state:{}:fixed", var_names[var]));
        }

        let problem = LPProblem {
            sense: Sense::Max,
            c,
            a_ub: Some(a_ub),
            b_ub: Some(b_ub),
            a_eq: Some(a_eq),
            b_eq: Some(b_eq),
            lb: Some(lb),
            ub: Some(ub),
            var_names: Some(var_names),
            con_names: Some(con_names),
        };
        let variables = problem.c.len();
        let constraints =
            problem.a_ub.as_ref().map_or(0, Vec::len) + problem.a_eq.as_ref().map_or(0, Vec::len);
        SoccerFormationLpBrain {
            team,
            problem,
            player_vars,
            player_rows,
            pair_vars,
            pair_rows,
            state_vars,
            basis_start: None,
            last_guidance: Vec::new(),
            last_snapshot: SoccerFormationLpTeamSnapshot {
                team,
                status: "not-solved".to_string(),
                objective: 0.0,
                variables,
                constraints,
                guided_players: 0,
                iterations: None,
                elapsed_ms: 0.0,
                mean_pair_error_yards: 0.0,
                mean_role_line_error_yards: 0.0,
                mean_pressure_error_yards: 0.0,
                objective_weights: SoccerFormationLpObjectiveWeights::default(),
            },
            adaptive_max_iter: SOCCER_FORMATION_LP_INTERNAL_SIMPLEX_MAX_ITER,
            last_solve_micros: 0,
            last_solve_status: "not-solved".to_string(),
            solve_count: 0,
            solve_timeout_count: 0,
            suboptimal_count: 0,
            consecutive_timeouts: 0,
            simplex_circuit_open: false,
        }
    }

    /// Exact per-tick formation solve via the sparse interior-point (Clarabel).
    ///
    /// The formation LP only needs the optimal *point* (player target positions),
    /// not a vertex, so there is no warm-start basis to carry between ticks and no
    /// crossover/polish: the sparse IPM solves the whole problem from scratch each
    /// tick in a few ms — cheaper here than a dense simplex (~400ms / ~1000 pivots,
    /// warm or cold) or the dense internal IPM. The realtime guardrails (adaptive
    /// budget + circuit breaker) live in [`Self::solve_tick`].
    pub(crate) fn solve_exact_formation_lp(&mut self) -> crate::des::general::lp::LPSolution {
        // The dense internal simplex needs ~1000 pivots (~400ms) on this 521-var/
        // 750-constraint degenerate LP whether cold OR warm, and the dense internal
        // IPM is worse still. The SPARSE interior-point (Clarabel) solves it to
        // optimality directly. The formation only needs an optimal *point* (player
        // positions), not a vertex, so we skip the IPM->simplex crossover/polish
        // entirely — that polish was the ~400ms bottleneck for a vertex we never use.
        // The circuit breaker in `solve_tick` still guards realtime.
        solve_lp_clarabel(&self.problem)
    }

    fn solve_tick(&mut self, snapshot: &WorldSnapshot, directive: &TeamTacticalDirective) {
        let objective_weights =
            soccer_formation_lp_objective_weights(snapshot, self.team, directive);
        let slots =
            soccer_formation_lp_slot_inputs(snapshot, self.team, directive, &objective_weights);
        self.update_problem_for_tick(snapshot, directive, &objective_weights, &slots);
        let simplex = soccer_formation_lp_internal_simplex_enabled() && !self.simplex_circuit_open;
        let started = std::time::Instant::now();
        let solution = if simplex {
            self.solve_exact_formation_lp()
        } else {
            soccer_formation_lp_budgeted_fallback_solution()
        };
        let elapsed_micros = started.elapsed().as_micros();
        self.solve_count = self.solve_count.saturating_add(1);
        self.last_solve_micros = elapsed_micros;
        self.last_solve_status = solution.status.as_str().to_string();
        if soccer_lp_debug_enabled() && self.solve_count <= 2 {
            let ineq = self.problem.a_ub.as_ref().map(|m| m.len()).unwrap_or(0);
            let eq = self.problem.a_eq.as_ref().map(|m| m.len()).unwrap_or(0);
            eprintln!(
                "[soccer-lp] {:?} DIAG vars={} ineq_rows={} eq_rows={} iters={:?} status={} solve={}us",
                self.team, self.problem.c.len(), ineq, eq, solution.iters, solution.status.as_str(), elapsed_micros
            );
        }
        // Adaptive time-budgeting + failure telemetry only apply to a real solve;
        // the heuristic fallback is intentionally suboptimal (not a failure).
        if simplex {
            if elapsed_micros > SOCCER_FORMATION_LP_SOLVE_BUDGET_MICROS {
                self.solve_timeout_count = self.solve_timeout_count.saturating_add(1);
                self.consecutive_timeouts = self.consecutive_timeouts.saturating_add(1);
                self.adaptive_max_iter =
                    (self.adaptive_max_iter / 2).max(SOCCER_FORMATION_LP_MIN_ITER);
                if self.consecutive_timeouts >= SOCCER_FORMATION_LP_TIMEOUT_CIRCUIT_LIMIT {
                    // Repeatedly too slow for realtime: trip the breaker and run the
                    // fast heuristic fallback for the rest of the match.
                    self.simplex_circuit_open = true;
                    if soccer_lp_debug_enabled() {
                        eprintln!(
                            "[soccer-lp] {:?} circuit OPEN after {} consecutive timeouts -> heuristic fallback",
                            self.team, self.consecutive_timeouts
                        );
                    }
                }
                if soccer_lp_debug_enabled() {
                    eprintln!(
                        "[soccer-lp] {:?} solve {}us over budget ({}us); max_iter->{}",
                        self.team,
                        elapsed_micros,
                        SOCCER_FORMATION_LP_SOLVE_BUDGET_MICROS,
                        self.adaptive_max_iter
                    );
                }
            } else if elapsed_micros * 3 < SOCCER_FORMATION_LP_SOLVE_BUDGET_MICROS {
                self.consecutive_timeouts = 0;
                self.adaptive_max_iter = (self.adaptive_max_iter
                    + SOCCER_FORMATION_LP_ITER_RECOVER_STEP)
                    .min(SOCCER_FORMATION_LP_INTERNAL_SIMPLEX_MAX_ITER);
            }
            if solution.status == LPStatus::Optimal {
                self.basis_start = LPBasisWarmStart::from_solution(&solution);
            } else {
                self.basis_start = None;
                self.suboptimal_count = self.suboptimal_count.saturating_add(1);
                if soccer_lp_debug_enabled() {
                    eprintln!(
                        "[soccer-lp] {:?} non-optimal status={} ({}us) - capture degrades to anchors",
                        self.team, self.last_solve_status, elapsed_micros
                    );
                }
            }
        }
        // capture_solution degrades gracefully (formation anchors) when the solution
        // is non-optimal/empty, so a bail-out never applies a garbage formation.
        self.capture_solution(snapshot, objective_weights, &slots, &solution);
    }

    pub(crate) fn update_problem_for_tick(
        &mut self,
        snapshot: &WorldSnapshot,
        _directive: &TeamTacticalDirective,
        weights: &SoccerFormationLpObjectiveWeights,
        slots: &[SoccerFormationLpSlotInput],
    ) {
        let (width, length) = sane_pitch_dimensions(snapshot.field_width, snapshot.field_length);
        let center = Vec2::new(width * 0.5, length * 0.5);
        let dt = sane_dt_seconds(snapshot.dt_seconds, DEFAULT_DT_SECONDS).max(1e-6);
        let expected_goal = soccer_lp_unit(weights.expected_goal);
        let progression = soccer_lp_unit(weights.progression);
        let retention = soccer_lp_unit(weights.retention);
        let fatigue = soccer_lp_unit(weights.fatigue);
        let passing_lane_quality = soccer_lp_unit(weights.passing_lane_quality);
        let space_occupation = soccer_lp_unit(weights.space_occupation);
        let defensive_compactness = soccer_lp_unit(weights.defensive_compactness);
        let transition_risk = soccer_lp_unit(weights.transition_risk);
        let embedding_attack = soccer_lp_unit(weights.adversarial_embedding_attack);
        let embedding_defense = soccer_lp_unit(weights.adversarial_embedding_defense);
        let neural_attack = soccer_lp_unit(weights.neural_attack);
        let neural_defense = soccer_lp_unit(weights.neural_defense);
        if let (Some(lb), Some(ub)) = (self.problem.lb.as_mut(), self.problem.ub.as_mut()) {
            for (slot, vars) in self.player_vars.iter().enumerate() {
                let input = slots[slot];
                let velocity = finite_vec2(input.velocity, Vec2::zero());
                let acceleration = finite_vec2(input.acceleration, Vec2::zero());
                let max_speed = if input.active {
                    soccer_lp_clamped(input.top_speed, 0.0, 20.0, 0.0)
                        .max(velocity.len())
                        .max(0.0)
                } else {
                    0.0
                };
                let max_acceleration = if input.active {
                    soccer_lp_clamped(input.max_acceleration, 0.0, 30.0, 0.0)
                        .max(acceleration.len())
                        .max(0.0)
                } else {
                    0.0
                };
                ub[vars.target_x] = Some(width);
                ub[vars.target_y] = Some(length);
                lb[vars.target_vx] = Some(-max_speed);
                ub[vars.target_vx] = Some(max_speed);
                lb[vars.target_vy] = Some(-max_speed);
                ub[vars.target_vy] = Some(max_speed);
                lb[vars.target_ax] = Some(-max_acceleration);
                ub[vars.target_ax] = Some(max_acceleration);
                lb[vars.target_ay] = Some(-max_acceleration);
                ub[vars.target_ay] = Some(max_acceleration);
            }
        }
        if let Some(a_ub) = self.problem.a_ub.as_mut() {
            for (slot, vars) in self.player_vars.iter().enumerate() {
                let rows = self.player_rows[slot];
                a_ub[rows.dynamics_x_upper][vars.target_vx] = -dt;
                a_ub[rows.dynamics_x_lower][vars.target_vx] = dt;
                a_ub[rows.dynamics_y_upper][vars.target_vy] = -dt;
                a_ub[rows.dynamics_y_lower][vars.target_vy] = dt;
                a_ub[rows.dynamics_vx_upper][vars.target_ax] = -dt;
                a_ub[rows.dynamics_vx_lower][vars.target_ax] = dt;
                a_ub[rows.dynamics_vy_upper][vars.target_ay] = -dt;
                a_ub[rows.dynamics_vy_lower][vars.target_ay] = dt;
            }
        }
        for coef in &mut self.problem.c {
            *coef = 0.0;
        }
        let b_ub = self
            .problem
            .b_ub
            .as_mut()
            .expect("formation LP has upper rows");
        let b_eq = self
            .problem
            .b_eq
            .as_mut()
            .expect("formation LP has state equality rows");

        for (slot, input) in slots.iter().enumerate() {
            let vars = self.player_vars[slot];
            let rows = self.player_rows[slot];
            let current = finite_pitch_point(input.current, width, length, center);
            let anchor = finite_pitch_point(input.anchor, width, length, current);
            let pressure_target = input
                .pressure_target
                .map(|target| finite_pitch_point(target, width, length, anchor))
                .unwrap_or(anchor);
            let speed_match_velocity_yps = if input.speed_match_velocity_yps.is_finite() {
                input.speed_match_velocity_yps
            } else {
                0.0
            };
            let velocity = finite_vec2(input.velocity, Vec2::zero());
            let speed_match_y = (current.y + speed_match_velocity_yps * dt).clamp(0.0, length);
            let reach = soccer_formation_lp_reach_yards(input, dt);

            b_ub[rows.anchor_x_upper] = anchor.x;
            b_ub[rows.anchor_x_lower] = -anchor.x;
            b_ub[rows.anchor_y_upper] = anchor.y;
            b_ub[rows.anchor_y_lower] = -anchor.y;
            b_ub[rows.movement_x_upper] = current.x;
            b_ub[rows.movement_x_lower] = -current.x;
            b_ub[rows.movement_y_upper] = current.y;
            b_ub[rows.movement_y_lower] = -current.y;
            b_ub[rows.reach_x_upper] = current.x + reach;
            b_ub[rows.reach_x_lower] = reach - current.x;
            b_ub[rows.reach_y_upper] = current.y + reach;
            b_ub[rows.reach_y_lower] = reach - current.y;
            b_ub[rows.pressure_x_upper] = pressure_target.x;
            b_ub[rows.pressure_x_lower] = -pressure_target.x;
            b_ub[rows.pressure_y_upper] = pressure_target.y;
            b_ub[rows.pressure_y_lower] = -pressure_target.y;
            b_ub[rows.speed_y_upper] = speed_match_y;
            b_ub[rows.speed_y_lower] = -speed_match_y;
            b_ub[rows.dynamics_x_upper] = current.x;
            b_ub[rows.dynamics_x_lower] = -current.x;
            b_ub[rows.dynamics_y_upper] = current.y;
            b_ub[rows.dynamics_y_lower] = -current.y;
            b_ub[rows.dynamics_vx_upper] = velocity.x;
            b_ub[rows.dynamics_vx_lower] = -velocity.x;
            b_ub[rows.dynamics_vy_upper] = velocity.y;
            b_ub[rows.dynamics_vy_lower] = -velocity.y;
            b_ub[rows.acceleration_x_upper] = 0.0;
            b_ub[rows.acceleration_x_lower] = 0.0;
            b_ub[rows.acceleration_y_upper] = 0.0;
            b_ub[rows.acceleration_y_lower] = 0.0;

            if !input.active {
                continue;
            }
            let role_attack = match input.role {
                PlayerRole::Goalkeeper => 0.05,
                PlayerRole::Defender => 0.30,
                PlayerRole::Midfielder => 0.82,
                PlayerRole::Forward => 1.0,
            };
            let role_rest_defense = match input.role {
                PlayerRole::Goalkeeper => 0.76,
                PlayerRole::Defender => 1.0,
                PlayerRole::Midfielder => 0.62,
                PlayerRole::Forward => 0.24,
            };
            let input_fatigue = soccer_lp_unit(input.fatigue);
            let anchor_weight = 0.42
                + space_occupation * 0.30
                + passing_lane_quality * 0.18
                + defensive_compactness * role_rest_defense * 0.26
                + progression * role_attack * 0.12;
            let movement_weight = 0.18 + fatigue * (0.45 + input_fatigue * 0.75);
            let acceleration_weight = 0.06 + fatigue * 0.18 + input_fatigue * 0.16;
            let dynamics_weight =
                1.35 + retention * 0.28 + defensive_compactness * role_rest_defense * 0.22;
            let pressure_weight = soccer_lp_clamped(input.pressure_weight, 0.0, 1.4, 0.0);
            let speed_weight = soccer_lp_unit(input.speed_match_weight);
            self.problem.c[vars.anchor_slack_x] -= anchor_weight;
            self.problem.c[vars.anchor_slack_y] -= anchor_weight;
            self.problem.c[vars.movement_slack_x] -= movement_weight;
            self.problem.c[vars.movement_slack_y] -= movement_weight;
            self.problem.c[vars.pressure_slack_x] -= pressure_weight;
            self.problem.c[vars.pressure_slack_y] -= pressure_weight;
            self.problem.c[vars.speed_match_slack_y] -= speed_weight;
            self.problem.c[vars.acceleration_slack_x] -= acceleration_weight;
            self.problem.c[vars.acceleration_slack_y] -= acceleration_weight;
            self.problem.c[vars.dynamics_slack_x] -= dynamics_weight;
            self.problem.c[vars.dynamics_slack_y] -= dynamics_weight;
            self.problem.c[vars.dynamics_slack_vx] -= dynamics_weight;
            self.problem.c[vars.dynamics_slack_vy] -= dynamics_weight;
            self.problem.c[vars.target_y] += self.team.attack_dir()
                * (expected_goal * 0.010
                    + progression * role_attack * 0.012
                    + embedding_attack * role_attack * 0.008
                    + neural_attack * role_attack * 0.008
                    - transition_risk * role_rest_defense * 0.006
                    - embedding_defense * role_rest_defense * 0.006
                    - neural_defense * role_rest_defense * 0.006);
        }

        for (idx, pair) in self.pair_vars.iter().enumerate() {
            let rows = self.pair_rows[idx];
            let a = slots[pair.a];
            let b = slots[pair.b];
            let a_anchor = finite_pitch_point(a.anchor, width, length, center);
            let b_anchor = finite_pitch_point(b.anchor, width, length, center);
            let desired_dx = a_anchor.x - b_anchor.x;
            let desired_dy = a_anchor.y - b_anchor.y;
            b_ub[rows.separation_x_upper] = desired_dx;
            b_ub[rows.separation_x_lower] = -desired_dx;
            b_ub[rows.separation_y_upper] = desired_dy;
            b_ub[rows.separation_y_lower] = -desired_dy;
            if a.active && b.active {
                let same_role_spacing_weight = match (a.role, b.role) {
                    (PlayerRole::Defender, PlayerRole::Defender) => 0.18,
                    (PlayerRole::Midfielder, PlayerRole::Midfielder) => 0.10,
                    _ => 0.0,
                };
                let same_role_line_y_weight = match (a.role, b.role) {
                    (PlayerRole::Defender, PlayerRole::Defender) => {
                        FORMATION_LP_DEFENDER_LINE_Y_WEIGHT
                    }
                    (PlayerRole::Midfielder, PlayerRole::Midfielder) => {
                        FORMATION_LP_MIDFIELDER_LINE_Y_WEIGHT
                    }
                    _ => 0.0,
                };
                let shape_relief =
                    soccer_formation_lp_pair_shape_relief(snapshot, self.team, &a, &b);
                let base_pair_weight = soccer_lp_unit((a.pair_weight + b.pair_weight) * 0.5);
                let spacing_shape_weight =
                    same_role_spacing_weight * (1.0 - shape_relief).clamp(0.15, 1.0);
                let line_shape_weight =
                    same_role_line_y_weight * (1.0 - shape_relief).clamp(0.15, 1.0);
                let pair_x_weight = soccer_lp_unit(base_pair_weight + spacing_shape_weight);
                let pair_y_weight =
                    soccer_lp_unit(base_pair_weight + spacing_shape_weight + line_shape_weight);
                self.problem.c[pair.separation_slack_x] -= pair_x_weight;
                self.problem.c[pair.separation_slack_y] -= pair_y_weight;
            }
        }

        let state_values = soccer_formation_lp_state_values(snapshot, self.team, weights);
        for (idx, value) in state_values.into_iter().enumerate() {
            b_eq[idx] = soccer_lp_clamped(value, -2.0, 2.0, 0.0);
        }
    }

    pub(crate) fn capture_solution(
        &mut self,
        snapshot: &WorldSnapshot,
        weights: SoccerFormationLpObjectiveWeights,
        slots: &[SoccerFormationLpSlotInput],
        solution: &crate::des::general::lp::LPSolution,
    ) {
        let variables = self.problem.c.len();
        let constraints = self.problem.a_ub.as_ref().map_or(0, Vec::len)
            + self.problem.a_eq.as_ref().map_or(0, Vec::len);
        let (width, length) = sane_pitch_dimensions(snapshot.field_width, snapshot.field_length);
        let center = Vec2::new(width * 0.5, length * 0.5);
        let dt = sane_dt_seconds(snapshot.dt_seconds, DEFAULT_DT_SECONDS).max(1e-6);
        let solver_elapsed_ms = if solution.elapsed_ms.is_finite() && solution.elapsed_ms >= 0.0 {
            solution.elapsed_ms
        } else {
            0.0
        };
        let mut status = solution.status.as_str().to_string();
        let mut guidance = Vec::new();
        let mut pair_error_sum_by_slot = vec![0.0; SOCCER_FORMATION_LP_PLAYER_CAPACITY];
        let mut pair_count_by_slot = vec![0usize; SOCCER_FORMATION_LP_PLAYER_CAPACITY];
        let has_complete_finite_solution = solution.status == LPStatus::Optimal
            && solution.x.len() >= variables
            && solution
                .x
                .iter()
                .take(variables)
                .all(|value| value.is_finite());
        if has_complete_finite_solution {
            for pair in &self.pair_vars {
                let pair_error = solution.x[pair.separation_slack_x].abs()
                    + solution.x[pair.separation_slack_y].abs();
                pair_error_sum_by_slot[pair.a] += pair_error;
                pair_error_sum_by_slot[pair.b] += pair_error;
                pair_count_by_slot[pair.a] += 1;
                pair_count_by_slot[pair.b] += 1;
            }
            for (slot, input) in slots.iter().enumerate() {
                if !input.active {
                    continue;
                }
                let vars = self.player_vars[slot];
                let current = finite_pitch_point(input.current, width, length, center);
                let anchor = finite_pitch_point(input.anchor, width, length, current);
                let pressure_target = input
                    .pressure_target
                    .map(|target| finite_pitch_point(target, width, length, anchor));
                let velocity = finite_vec2(input.velocity, Vec2::zero());
                let acceleration = finite_vec2(input.acceleration, Vec2::zero());
                let pressure_weight = soccer_lp_clamped(input.pressure_weight, 0.0, 1.4, 0.0);
                let speed_match_weight = soccer_lp_unit(input.speed_match_weight);
                let speed_match_velocity_yps = if input.speed_match_velocity_yps.is_finite() {
                    input.speed_match_velocity_yps
                } else {
                    0.0
                };
                let alignment_weight = soccer_lp_clamped(input.alignment_weight, 0.0, 2.0, 0.0);
                let target = finite_pitch_point(
                    Vec2::new(solution.x[vars.target_x], solution.x[vars.target_y]),
                    width,
                    length,
                    current,
                );
                let recommended_move = target - current;
                let target_velocity = limit_vec2_len(
                    finite_vec2(
                        Vec2::new(solution.x[vars.target_vx], solution.x[vars.target_vy]),
                        velocity,
                    ),
                    soccer_lp_clamped(input.top_speed, 0.0, 20.0, 0.0)
                        .max(velocity.len())
                        .max(0.0),
                );
                let target_acceleration = limit_vec2_len(
                    finite_vec2(
                        Vec2::new(solution.x[vars.target_ax], solution.x[vars.target_ay]),
                        acceleration,
                    ),
                    soccer_lp_clamped(input.max_acceleration, 0.0, 30.0, 0.0)
                        .max(acceleration.len())
                        .max(0.0),
                );
                let pair_error_yards = if pair_count_by_slot[slot] > 0 {
                    pair_error_sum_by_slot[slot] / pair_count_by_slot[slot] as f64
                } else {
                    0.0
                };
                let pressure_error_yards = pressure_target
                    .map(|pressure_target| target.distance(pressure_target))
                    .unwrap_or(0.0);
                let fore_aft_speed_error_yps = if speed_match_weight > 1e-9 && dt > 1e-9 {
                    (target_velocity.y - speed_match_velocity_yps).abs()
                } else {
                    0.0
                };
                let reduced_costs = solution.reduced_costs.as_ref();
                let reduced_cost_target_x = reduced_costs
                    .and_then(|costs| costs.get(vars.target_x))
                    .copied()
                    .filter(|value| value.is_finite())
                    .unwrap_or(0.0);
                let reduced_cost_target_y = reduced_costs
                    .and_then(|costs| costs.get(vars.target_y))
                    .copied()
                    .filter(|value| value.is_finite())
                    .unwrap_or(0.0);
                guidance.push(SoccerFormationLpPlayerGuidance {
                    team: self.team,
                    player_id: input.player_id,
                    slot,
                    current,
                    target,
                    formation_anchor: anchor,
                    pressure_target,
                    recommended_move,
                    target_velocity,
                    target_acceleration,
                    local_mpc_guidance: false,
                    recommended_move_yards: recommended_move.len(),
                    recommended_speed_yps: target_velocity.len(),
                    recommended_acceleration_yps2: target_acceleration.len(),
                    formation_error_yards: target.distance(anchor),
                    movement_error_yards: target.distance(current),
                    pair_error_yards,
                    role_line_error_yards: 0.0,
                    pressure_error_yards,
                    fore_aft_speed_error_yps,
                    pressure_weight,
                    speed_match_weight,
                    alignment_weight,
                    reduced_cost_target_x,
                    reduced_cost_target_y,
                    solver_status: status.clone(),
                    solver_objective: if solution.objective.is_finite() {
                        solution.objective
                    } else {
                        0.0
                    },
                    solver_iterations: solution.iters,
                    solver_elapsed_ms,
                    variable_count: variables,
                    constraint_count: constraints,
                });
            }
        } else {
            status = if solution.status == LPStatus::Optimal {
                "fallback-malformed-optimal".to_string()
            } else {
                format!("fallback-{status}")
            };
            for (slot, input) in slots.iter().enumerate() {
                if !input.active {
                    continue;
                }
                let current = finite_pitch_point(input.current, width, length, center);
                let anchor = finite_pitch_point(input.anchor, width, length, current);
                let pressure_target = input
                    .pressure_target
                    .map(|target| finite_pitch_point(target, width, length, anchor));
                let velocity = finite_vec2(input.velocity, Vec2::zero());
                let acceleration = finite_vec2(input.acceleration, Vec2::zero());
                let pressure_weight = soccer_lp_clamped(input.pressure_weight, 0.0, 1.4, 0.0);
                let speed_match_weight = soccer_lp_unit(input.speed_match_weight);
                let speed_match_velocity_yps = if input.speed_match_velocity_yps.is_finite() {
                    input.speed_match_velocity_yps
                } else {
                    0.0
                };
                let alignment_weight = soccer_lp_clamped(input.alignment_weight, 0.0, 2.0, 0.0);
                let pressure_blend = soccer_lp_unit(pressure_weight);
                let fallback_anchor = pressure_target
                    .map(|pressure_target| {
                        anchor * (1.0 - pressure_blend) + pressure_target * pressure_blend
                    })
                    .unwrap_or(anchor)
                    .clamp_to_pitch(width, length);
                let reach = soccer_formation_lp_reach_yards(input, dt);
                let desired = fallback_anchor - current;
                let desired_len = desired.len();
                let recommended_move = if desired_len > reach && reach > 1e-9 {
                    desired.normalized() * reach
                } else {
                    desired
                };
                let target = (current + recommended_move).clamp_to_pitch(width, length);
                let raw_target_velocity = (target - current) / dt;
                let speed_cap = soccer_lp_clamped(input.top_speed, 0.0, 20.0, 0.0)
                    .max(velocity.len())
                    .max(0.0);
                let target_velocity = limit_vec2_len(raw_target_velocity, speed_cap);
                let raw_target_acceleration = (target_velocity - velocity) / dt;
                let acceleration_cap = soccer_lp_clamped(input.max_acceleration, 0.0, 30.0, 0.0)
                    .max(acceleration.len())
                    .max(0.0);
                let target_acceleration = limit_vec2_len(raw_target_acceleration, acceleration_cap);
                let pressure_error_yards = pressure_target
                    .map(|pressure_target| target.distance(pressure_target))
                    .unwrap_or(0.0);
                let fore_aft_speed_error_yps = if speed_match_weight > 1e-9 && dt > 1e-9 {
                    (target_velocity.y - speed_match_velocity_yps).abs()
                } else {
                    0.0
                };
                guidance.push(SoccerFormationLpPlayerGuidance {
                    team: self.team,
                    player_id: input.player_id,
                    slot,
                    current,
                    target,
                    formation_anchor: anchor,
                    pressure_target,
                    recommended_move: target - current,
                    target_velocity,
                    target_acceleration,
                    local_mpc_guidance: false,
                    recommended_move_yards: (target - current).len(),
                    recommended_speed_yps: target_velocity.len(),
                    recommended_acceleration_yps2: target_acceleration.len(),
                    formation_error_yards: target.distance(anchor),
                    movement_error_yards: target.distance(current),
                    pair_error_yards: 0.0,
                    role_line_error_yards: 0.0,
                    pressure_error_yards,
                    fore_aft_speed_error_yps,
                    pressure_weight,
                    speed_match_weight,
                    alignment_weight,
                    reduced_cost_target_x: 0.0,
                    reduced_cost_target_y: 0.0,
                    solver_status: status.clone(),
                    solver_objective: 0.0,
                    solver_iterations: solution.iters,
                    solver_elapsed_ms,
                    variable_count: variables,
                    constraint_count: constraints,
                });
            }
        }
        annotate_formation_lp_role_line_errors(&mut guidance, slots);
        let guided_players = guidance.len();
        let mean_pair_error_yards = if guided_players > 0 {
            guidance
                .iter()
                .map(|entry| entry.pair_error_yards)
                .sum::<f64>()
                / guided_players as f64
        } else {
            0.0
        };
        let role_line_guidance = guidance
            .iter()
            .filter(|entry| entry.role_line_error_yards.is_finite())
            .collect::<Vec<_>>();
        let mean_role_line_error_yards = if role_line_guidance.is_empty() {
            0.0
        } else {
            role_line_guidance
                .iter()
                .map(|entry| entry.role_line_error_yards.max(0.0))
                .sum::<f64>()
                / role_line_guidance.len() as f64
        };
        let pressure_guidance = guidance
            .iter()
            .filter(|entry| entry.pressure_weight > 1e-9)
            .collect::<Vec<_>>();
        let mean_pressure_error_yards = if pressure_guidance.is_empty() {
            0.0
        } else {
            pressure_guidance
                .iter()
                .map(|entry| entry.pressure_error_yards)
                .sum::<f64>()
                / pressure_guidance.len() as f64
        };
        self.last_guidance = guidance;
        self.last_snapshot = SoccerFormationLpTeamSnapshot {
            team: self.team,
            status,
            objective: if solution.objective.is_finite() {
                solution.objective
            } else {
                0.0
            },
            variables,
            constraints,
            guided_players,
            iterations: solution.iters,
            elapsed_ms: solver_elapsed_ms,
            mean_pair_error_yards,
            mean_role_line_error_yards,
            mean_pressure_error_yards,
            objective_weights: weights,
        };
    }

    fn disable_tick(&mut self) {
        self.last_guidance.clear();
        self.last_snapshot.status = "disabled".to_string();
        self.last_snapshot.objective = 0.0;
        self.last_snapshot.guided_players = 0;
        self.last_snapshot.iterations = None;
        self.last_snapshot.elapsed_ms = 0.0;
        self.last_snapshot.mean_pair_error_yards = 0.0;
        self.last_snapshot.mean_role_line_error_yards = 0.0;
        self.last_snapshot.mean_pressure_error_yards = 0.0;
    }

    fn apply_local_mpc_tick(&mut self, snapshot: &WorldSnapshot, max_players: usize) -> usize {
        if max_players == 0 || self.last_guidance.is_empty() {
            return 0;
        }
        let mut candidates = self
            .last_guidance
            .iter()
            .enumerate()
            .filter_map(|(guidance_index, guidance)| {
                soccer_local_mpc_candidate_priority(snapshot, guidance).map(|priority| {
                    SoccerLocalMpcCandidate {
                        guidance_index,
                        priority,
                    }
                })
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|a, b| {
            b.priority
                .partial_cmp(&a.priority)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut solved = 0usize;
        for candidate in candidates.into_iter().take(max_players) {
            let original = self.last_guidance[candidate.guidance_index].clone();
            if let Some(refined) = soccer_local_mpc_refined_guidance(snapshot, &original) {
                self.last_guidance[candidate.guidance_index] = refined;
                solved = solved.saturating_add(1);
            }
        }
        solved
    }
}

#[derive(Clone, Copy, Debug)]
struct SoccerLocalMpcCandidate {
    guidance_index: usize,
    priority: f64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SoccerLocalMpcWholeFieldContext {
    pub(crate) target: Vec2,
    pub(crate) desired_velocity: Vec2,
    pub(crate) position_weight_bonus: f64,
    pub(crate) velocity_weight_bonus: f64,
}

fn soccer_local_mpc_candidate_priority(
    snapshot: &WorldSnapshot,
    guidance: &SoccerFormationLpPlayerGuidance,
) -> Option<f64> {
    let player = snapshot
        .players
        .iter()
        .find(|player| player.id == guidance.player_id && player.team == guidance.team)?;
    if player.role == PlayerRole::Goalkeeper || snapshot.ball.holder == Some(player.id) {
        return None;
    }
    let has_ball = snapshot
        .controlled_possession_team()
        .or_else(|| snapshot.possession_team())
        == Some(guidance.team);
    let recovery_bonus = if has_ball { 0.0 } else { 0.85 };
    let urgency = guidance.recommended_move_yards * 0.48
        + guidance.pressure_weight * 1.55
        + guidance.speed_match_weight * 0.85
        + guidance.pair_error_yards * 0.28
        + recovery_bonus;
    (urgency.is_finite() && urgency > 0.10).then_some(urgency)
}

pub(crate) fn soccer_local_mpc_whole_field_context(
    snapshot: &WorldSnapshot,
    player: &PlayerSnapshot,
    current: Vec2,
    velocity: Vec2,
    target: Vec2,
    desired_velocity: Vec2,
    dt: f64,
    speed_cap: f64,
    width: f64,
    length: f64,
) -> SoccerLocalMpcWholeFieldContext {
    let half_dt2 = 0.5 * dt * dt;
    let mut target_influence = Vec2::zero();
    let mut velocity_influence = Vec2::zero();
    let mut context_pressure = 0.0;
    let self_direction = if (target - current).len() > 1e-6 {
        (target - current).normalized()
    } else if velocity.len() > 1e-6 {
        velocity.normalized()
    } else {
        Vec2::new(0.0, player.team.attack_dir())
    };

    for other in &snapshot.players {
        let other_position = finite_pitch_point(other.position, width, length, current);
        let other_velocity = finite_vec2(other.velocity, Vec2::zero());
        let other_acceleration = finite_vec2(other.acceleration, Vec2::zero());
        let projected = finite_pitch_point(
            other_position + other_velocity * dt + other_acceleration * half_dt2,
            width,
            length,
            other_position,
        );
        if other.id == player.id {
            velocity_influence += other_acceleration * (0.04 * dt);
            continue;
        }

        let gap = target - projected;
        let distance = gap.len();
        let direction = if distance > 1e-6 {
            gap / distance
        } else {
            self_direction
        };
        let same_team = other.team == player.team;
        let base_radius = if same_team { 3.2 } else { 4.2 };
        let holder_bonus = if snapshot.ball.holder == Some(other.id) {
            1.35
        } else {
            0.0
        };
        let influence_radius = base_radius + holder_bonus;
        if distance < influence_radius {
            let closing_speed = (other_velocity - desired_velocity).len().clamp(0.0, 14.0);
            let acceleration_pressure = other_acceleration.len().clamp(0.0, 12.0);
            let team_weight = if same_team { 0.95 } else { 1.35 };
            let pressure = ((influence_radius - distance) / influence_radius).clamp(0.0, 1.0)
                * team_weight
                * (1.0 + closing_speed / 32.0 + acceleration_pressure / 40.0);
            target_influence += direction * pressure * 2.4;
            velocity_influence += direction * pressure * 0.85;
            context_pressure += pressure;
        }
    }

    let ball_position = finite_pitch_point(snapshot.ball.position, width, length, target);
    let ball_velocity = finite_vec2(snapshot.ball.velocity, Vec2::zero());
    let ball_acceleration = finite_vec2(snapshot.ball.acceleration, Vec2::zero());
    let ball_projected = finite_pitch_point(
        ball_position + ball_velocity * dt + ball_acceleration * half_dt2,
        width,
        length,
        ball_position,
    );
    let ball_gap = target - ball_projected;
    let ball_distance = ball_gap.len();
    if snapshot.ball.holder != Some(player.id) && ball_distance < 3.4 {
        let direction = if ball_distance > 1e-6 {
            ball_gap / ball_distance
        } else {
            self_direction * -1.0
        };
        let pressure = ((3.4 - ball_distance) / 3.4).clamp(0.0, 1.0)
            * (1.0 + ball_velocity.len().clamp(0.0, 24.0) / 36.0)
            * (1.0 + ball_acceleration.len().clamp(0.0, 24.0) / 48.0);
        target_influence += direction * pressure * 1.8;
        velocity_influence += direction * pressure * 0.65;
        context_pressure += pressure;
    }

    let target = (target + limit_vec2_len(target_influence, 4.5)).clamp_to_pitch(width, length);
    let desired_velocity =
        limit_vec2_len(desired_velocity + velocity_influence, speed_cap.max(0.5));
    SoccerLocalMpcWholeFieldContext {
        target,
        desired_velocity,
        position_weight_bonus: context_pressure.clamp(0.0, 5.0),
        velocity_weight_bonus: (context_pressure * 0.35).clamp(0.0, 2.0),
    }
}

pub(crate) fn soccer_local_mpc_planar_obstacles(
    snapshot: &WorldSnapshot,
    player: &PlayerSnapshot,
    dt: f64,
    width: f64,
    length: f64,
) -> Vec<PlanarObstacle> {
    let half_dt2 = 0.5 * dt * dt;
    let mut obstacles = Vec::with_capacity(snapshot.players.len().saturating_add(1));
    for other in &snapshot.players {
        if other.id == player.id {
            continue;
        }
        let position = finite_pitch_point(other.position, width, length, player.position);
        let velocity = finite_vec2(other.velocity, Vec2::zero());
        let acceleration = finite_vec2(other.acceleration, Vec2::zero());
        let center = finite_pitch_point(
            position + velocity * dt + acceleration * half_dt2,
            width,
            length,
            position,
        );
        let obstacle_velocity = limit_vec2_len(velocity + acceleration * dt, 24.0);
        let same_team = other.team == player.team;
        let holder_bonus = if snapshot.ball.holder == Some(other.id) {
            0.55
        } else {
            0.0
        };
        let role_radius_bonus = match other.role {
            PlayerRole::Goalkeeper => 0.20,
            PlayerRole::Defender => 0.10,
            PlayerRole::Midfielder => 0.05,
            PlayerRole::Forward => 0.0,
        };
        let radius = if same_team { 1.35 } else { 1.75 } + holder_bonus + role_radius_bonus;
        let kinematic_pressure = 1.0
            + velocity.len().clamp(0.0, 16.0) / 48.0
            + acceleration.len().clamp(0.0, 14.0) / 56.0;
        let weight = if same_team { 7.0 } else { 12.0 } * kinematic_pressure
            + if snapshot.ball.holder == Some(other.id) {
                4.0
            } else {
                0.0
            };
        obstacles.push(PlanarObstacle {
            center: [center.x, center.y],
            velocity: [obstacle_velocity.x, obstacle_velocity.y],
            radius,
            weight,
        });
    }

    let ball_position = finite_pitch_point(snapshot.ball.position, width, length, player.position);
    let ball_velocity = finite_vec2(snapshot.ball.velocity, Vec2::zero());
    let ball_acceleration = finite_vec2(snapshot.ball.acceleration, Vec2::zero());
    let ball_center = finite_pitch_point(
        ball_position + ball_velocity * dt + ball_acceleration * half_dt2,
        width,
        length,
        ball_position,
    );
    let ball_obstacle_velocity = limit_vec2_len(ball_velocity + ball_acceleration * dt, 32.0);
    let ball_weight = 4.5
        + ball_velocity.len().clamp(0.0, 28.0) / 7.0
        + ball_acceleration.len().clamp(0.0, 28.0) / 14.0;
    obstacles.push(PlanarObstacle {
        center: [ball_center.x, ball_center.y],
        velocity: [ball_obstacle_velocity.x, ball_obstacle_velocity.y],
        radius: 0.85,
        weight: ball_weight,
    });

    obstacles
}

fn soccer_local_mpc_refined_guidance(
    snapshot: &WorldSnapshot,
    guidance: &SoccerFormationLpPlayerGuidance,
) -> Option<SoccerFormationLpPlayerGuidance> {
    let player = snapshot
        .players
        .iter()
        .find(|player| player.id == guidance.player_id && player.team == guidance.team)?;
    let (width, length) = sane_pitch_dimensions(snapshot.field_width, snapshot.field_length);
    let dt = sane_dt_seconds(snapshot.dt_seconds, DEFAULT_DT_SECONDS).max(1e-6);
    let current = finite_pitch_point(guidance.current, width, length, player.position);
    let velocity = finite_vec2(player.velocity, guidance.target_velocity);
    let target = finite_pitch_point(guidance.target, width, length, current);
    let desired_velocity = if guidance.target_velocity.len() > 1e-9 {
        guidance.target_velocity
    } else {
        (target - current) / dt
    };
    let speed_cap = player_top_speed_yps(player.role, &player.skills)
        .max(guidance.recommended_speed_yps)
        .max(velocity.len())
        .clamp(0.5, 20.0);
    let desired_velocity = limit_vec2_len(desired_velocity, speed_cap);
    let whole_field_context = soccer_local_mpc_whole_field_context(
        snapshot,
        player,
        current,
        velocity,
        target,
        desired_velocity,
        dt,
        speed_cap,
        width,
        length,
    );
    let target = whole_field_context.target;
    let desired_velocity = whole_field_context.desired_velocity;
    let accel_cap = acceleration_yps2_from_score(player.skills.acceleration)
        .max(guidance.recommended_acceleration_yps2.min(12.0))
        .clamp(2.5, 12.0);
    let (lb_x, ub_x) =
        soccer_local_mpc_axis_bounds(current.x, velocity.x, dt, 0.0, width, accel_cap, speed_cap)?;
    let (lb_y, ub_y) =
        soccer_local_mpc_axis_bounds(current.y, velocity.y, dt, 0.0, length, accel_cap, speed_cap)?;
    if lb_x > ub_x || lb_y > ub_y {
        return None;
    }

    let half_dt2 = 0.5 * dt * dt;
    let pos_weight = 8.0
        + guidance.pressure_weight * 6.0
        + guidance.pair_error_yards.min(6.0)
        + whole_field_context.position_weight_bonus;
    let vel_weight =
        1.25 + guidance.speed_match_weight * 3.0 + whole_field_context.velocity_weight_bonus;
    let horizon = (2.25 / dt).round().clamp(8.0, 45.0) as usize;
    let obstacle_decay_per_step = 0.5_f64.powf(dt / 1.5).clamp(0.70, 1.0);
    let mut controller = PlanarPointMassMpc::new(PlanarMpcConfig {
        horizon,
        dt,
        q_pos: pos_weight,
        q_vel: vel_weight,
        qf_pos: pos_weight * 3.0,
        qf_vel: vel_weight * 2.0,
        r: 0.10,
        a_max: accel_cap,
        iters: (horizon.saturating_mul(5)).clamp(48, 160),
        obstacle_decay_per_step,
    })
    .ok()?;
    let reference = PlanarReference::through(
        [target.x, target.y],
        [desired_velocity.x, desired_velocity.y],
    );
    let obstacles = soccer_local_mpc_planar_obstacles(snapshot, player, dt, width, length);
    let planned_acceleration = controller.control_with_obstacles(
        PlanarState {
            pos: [current.x, current.y],
            vel: [velocity.x, velocity.y],
        },
        &[reference],
        &obstacles,
    );
    let planned_acceleration = Vec2::new(planned_acceleration[0], planned_acceleration[1]);
    if !planned_acceleration.x.is_finite() || !planned_acceleration.y.is_finite() {
        return None;
    }
    let acceleration = limit_vec2_len(
        Vec2::new(
            planned_acceleration.x.clamp(lb_x, ub_x),
            planned_acceleration.y.clamp(lb_y, ub_y),
        ),
        accel_cap,
    );
    if !acceleration.x.is_finite() || !acceleration.y.is_finite() {
        return None;
    }
    let target_velocity = limit_vec2_len(velocity + acceleration * dt, speed_cap);
    let target = finite_pitch_point(
        current + velocity * dt + acceleration * half_dt2,
        width,
        length,
        target,
    );
    let recommended_move = target - current;
    let pressure_error_yards = guidance
        .pressure_target
        .map(|pressure_target| target.distance(pressure_target))
        .unwrap_or(0.0);
    let fore_aft_speed_error_yps = if guidance.speed_match_weight > 1e-9 {
        (target_velocity.y - guidance.target_velocity.y).abs()
    } else {
        guidance.fore_aft_speed_error_yps
    };

    let mut refined = guidance.clone();
    refined.target = target;
    refined.recommended_move = recommended_move;
    refined.target_velocity = target_velocity;
    refined.target_acceleration = acceleration;
    refined.local_mpc_guidance = true;
    refined.recommended_move_yards = recommended_move.len();
    refined.recommended_speed_yps = target_velocity.len();
    refined.recommended_acceleration_yps2 = acceleration.len();
    refined.formation_error_yards = target.distance(refined.formation_anchor);
    refined.movement_error_yards = target.distance(current);
    refined.pressure_error_yards = pressure_error_yards;
    refined.fore_aft_speed_error_yps = fore_aft_speed_error_yps;
    Some(refined)
}

fn soccer_local_mpc_axis_bounds(
    position: f64,
    velocity: f64,
    dt: f64,
    min_position: f64,
    max_position: f64,
    accel_cap: f64,
    speed_cap: f64,
) -> Option<(f64, f64)> {
    let mut lower = -accel_cap;
    let mut upper = accel_cap;
    if dt > 1e-9 {
        lower = lower.max((-speed_cap - velocity) / dt);
        upper = upper.min((speed_cap - velocity) / dt);
    }
    let half_dt2 = 0.5 * dt * dt;
    if half_dt2 > 1e-12 {
        let base = position + velocity * dt;
        lower = lower.max((min_position - base) / half_dt2);
        upper = upper.min((max_position - base) / half_dt2);
    }
    if lower.is_finite() && upper.is_finite() && lower <= upper {
        Some((lower, upper))
    } else {
        None
    }
}

pub(crate) fn annotate_formation_lp_role_line_errors(
    guidance: &mut [SoccerFormationLpPlayerGuidance],
    slots: &[SoccerFormationLpSlotInput],
) {
    let role_by_player = slots
        .iter()
        .filter(|slot| slot.active)
        .map(|slot| (slot.player_id, slot.role))
        .collect::<HashMap<_, _>>();
    let targets = guidance
        .iter()
        .map(|entry| (entry.player_id, entry.target))
        .collect::<HashMap<_, _>>();
    for entry in guidance {
        let Some(role) = role_by_player.get(&entry.player_id).copied() else {
            entry.role_line_error_yards = 0.0;
            continue;
        };
        let tolerance = match role {
            PlayerRole::Defender => DEFENDER_LINE_COHESION_TOLERANCE_YARDS,
            PlayerRole::Midfielder => MIDFIELDER_LINE_COHESION_TOLERANCE_YARDS,
            PlayerRole::Goalkeeper | PlayerRole::Forward => {
                entry.role_line_error_yards = 0.0;
                continue;
            }
        };
        let mut same_line_total_y = 0.0;
        let mut same_line_count = 0usize;
        for slot in slots {
            if !slot.active || slot.player_id == entry.player_id || slot.role != role {
                continue;
            }
            if let Some(target) = targets.get(&slot.player_id).copied() {
                same_line_total_y += target.y;
                same_line_count += 1;
            }
        }
        entry.role_line_error_yards = if same_line_count < 2 {
            0.0
        } else {
            ((entry.target.y - same_line_total_y / same_line_count as f64).abs() - tolerance)
                .max(0.0)
        };
    }
}

pub(crate) fn apply_formation_lp_directive_signal(
    directive: &mut TeamTacticalDirective,
    snapshot: &SoccerFormationLpTeamSnapshot,
    guidance: &[SoccerFormationLpPlayerGuidance],
) {
    if snapshot.status != "optimal" || guidance.is_empty() {
        return;
    }
    let finite_guidance = guidance
        .iter()
        .filter(|entry| {
            entry.pressure_weight.is_finite()
                && entry.pair_error_yards.is_finite()
                && entry.role_line_error_yards.is_finite()
        })
        .collect::<Vec<_>>();
    if finite_guidance.is_empty() {
        return;
    }
    let pressure_guidance = finite_guidance
        .iter()
        .filter(|entry| entry.pressure_weight.clamp(0.0, 1.0) > 0.12)
        .count();
    directive.defensive_cover_actual = directive.defensive_cover_actual.max(pressure_guidance);
    let mean_pressure_weight = finite_guidance
        .iter()
        .map(|entry| entry.pressure_weight.clamp(0.0, 1.0))
        .sum::<f64>()
        / finite_guidance.len() as f64;
    let mean_pair_error = finite_guidance
        .iter()
        .map(|entry| entry.pair_error_yards.max(0.0))
        .sum::<f64>()
        / finite_guidance.len() as f64;
    let mean_role_line_error = finite_guidance
        .iter()
        .map(|entry| entry.role_line_error_yards.max(0.0))
        .sum::<f64>()
        / finite_guidance.len() as f64;
    let current_press = if directive.press_intensity.is_finite() {
        directive.press_intensity
    } else {
        TeamTacticalDirective::neutral(
            directive.team,
            DEFAULT_FIELD_WIDTH_YARDS,
            DEFAULT_FIELD_LENGTH_YARDS,
        )
        .press_intensity
    };
    let current_risk = if directive.risk_tolerance.is_finite() {
        directive.risk_tolerance
    } else {
        TeamTacticalDirective::neutral(
            directive.team,
            DEFAULT_FIELD_WIDTH_YARDS,
            DEFAULT_FIELD_LENGTH_YARDS,
        )
        .risk_tolerance
    };
    directive.press_intensity =
        (current_press * 0.92 + mean_pressure_weight * 0.08).clamp(0.0, 1.0);
    let shape_risk_penalty =
        (mean_pair_error / 40.0 + mean_role_line_error / 36.0).clamp(0.0, 0.10);
    directive.risk_tolerance = (current_risk - shape_risk_penalty).clamp(0.0, 1.0);
}

fn soccer_lp_scaled(value: f64, scale: f64) -> f64 {
    if !value.is_finite() || scale.abs() <= 1e-9 {
        0.0
    } else {
        (value / scale).clamp(-2.0, 2.0)
    }
}

pub(crate) fn soccer_lp_unit(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn soccer_adversarial_embedding_confidence(hits: usize) -> f64 {
    let search_limit = (ADVERSARIAL_EMBEDDING_SEARCH_LIMIT as f64).max(1.0);
    smoothstep_unit(hits as f64 / search_limit)
}

pub(crate) fn soccer_lp_clamped(value: f64, min: f64, max: f64, fallback: f64) -> f64 {
    if value.is_finite() {
        value.clamp(min, max)
    } else {
        fallback.clamp(min, max)
    }
}

fn soccer_formation_lp_ball_holder_elapsed_seconds(snapshot: &WorldSnapshot) -> f64 {
    let Some(holder_id) = snapshot.ball.holder else {
        return 0.0;
    };
    let clock_seconds = if snapshot.clock_seconds.is_finite() {
        snapshot.clock_seconds.max(0.0)
    } else {
        0.0
    };
    let mut oldest_same_holder = clock_seconds;
    for sample in snapshot.ball_history.iter().rev() {
        if sample.holder == Some(holder_id) {
            if sample.clock_seconds.is_finite() {
                oldest_same_holder = sample.clock_seconds.max(0.0);
            }
        } else {
            break;
        }
    }
    (clock_seconds - oldest_same_holder).max(0.0)
}

fn soccer_formation_lp_average_fatigue(snapshot: &WorldSnapshot, team: Team) -> f64 {
    let mut total = 0.0;
    let mut count = 0usize;
    for player in snapshot.players.iter().filter(|player| player.team == team) {
        total += soccer_lp_unit(player.fatigue);
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

fn soccer_formation_lp_rest_defense_score(snapshot: &WorldSnapshot, team: Team) -> f64 {
    let (width, length) = sane_pitch_dimensions(snapshot.field_width, snapshot.field_length);
    let attack_dir = team.attack_dir();
    let center = Vec2::new(width * 0.5, length * 0.5);
    let ball_y = finite_pitch_point(snapshot.ball.position, width, length, center).y;
    let count = snapshot
        .players
        .iter()
        .filter(|player| player.team == team)
        .filter(|player| !matches!(player.role, PlayerRole::Forward))
        .filter(|player| {
            let position = finite_pitch_point(
                snapshot
                    .player_position(player.id)
                    .unwrap_or(player.position),
                width,
                length,
                player.home_position,
            );
            (ball_y - position.y) * attack_dir >= -2.0
        })
        .count();
    (count as f64 / 5.0).clamp(0.0, 1.0)
}

pub(crate) fn soccer_formation_lp_objective_weights(
    snapshot: &WorldSnapshot,
    team: Team,
    directive: &TeamTacticalDirective,
) -> SoccerFormationLpObjectiveWeights {
    let possession = snapshot
        .controlled_possession_team()
        .or_else(|| snapshot.possession_team());
    let has_ball = possession == Some(team);
    let defending = possession == Some(team.other());
    let (width, length) = sane_pitch_dimensions(snapshot.field_width, snapshot.field_length);
    let center = Vec2::new(width * 0.5, length * 0.5);
    let ball = finite_pitch_point(snapshot.ball.position, width, length, center);
    let opponent_goal = Vec2::new(width * 0.5, team.goal_y(length));
    let own_goal = Vec2::new(width * 0.5, team.other().goal_y(length));
    let ball_to_opponent_goal = ball.distance(opponent_goal).min(length);
    let ball_to_own_goal = ball.distance(own_goal).min(length);
    let attacking_depth = (1.0 - ball_to_opponent_goal / length).clamp(0.0, 1.0);
    let own_goal_danger = (1.0 - ball_to_own_goal / length).clamp(0.0, 1.0);
    let shape = team_shape_observation_from_snapshot(snapshot, team, ball);
    let opponent_shape = team_shape_observation_from_snapshot(snapshot, team.other(), ball);
    let holder_pressure = snapshot
        .ball
        .holder
        .and_then(|holder_id| {
            let holder = snapshot
                .players
                .iter()
                .find(|player| player.id == holder_id)?;
            let holder_position = finite_pitch_point(
                snapshot
                    .player_position(holder.id)
                    .unwrap_or(holder.position),
                width,
                length,
                holder.position,
            );
            Some(pressure_from_nearest_distance(
                snapshot.nearest_opponent_distance_at(holder.team, holder_position),
            ))
        })
        .unwrap_or(0.0);
    let team_possession_seconds = soccer_lp_clamped(
        snapshot.possession_elapsed_seconds_for_team(team),
        0.0,
        f64::MAX,
        0.0,
    );
    let settled = soccer_lp_unit(team_possession_seconds / SETTLED_POSSESSION_SECONDS);
    let overload = soccer_lp_unit(directive.attacking_overload_score);
    let numbers = ((directive.attacking_numbers_advantage as f64 + 2.0) / 4.0).clamp(0.0, 1.0);
    let field_width_usage =
        soccer_lp_unit(soccer_lp_clamped(directive.width_yards, 0.0, width, width * 0.62) / width);
    let pass_priority = soccer_lp_clamped(directive.pass_priority, 0.0, 2.0, 1.0);
    let carry_priority = soccer_lp_clamped(directive.carry_priority, 0.0, 2.0, 1.0);
    let press_intensity = soccer_lp_unit(directive.press_intensity);
    let embedding_confidence =
        soccer_adversarial_embedding_confidence(directive.adversarial_embedding_hits);
    let embedding_attack = soccer_lp_unit(directive.adversarial_embedding_attack_score)
        * (0.55 + embedding_confidence * 0.45);
    let embedding_defense = soccer_lp_unit(directive.adversarial_embedding_defense_score)
        * (0.55 + embedding_confidence * 0.45);
    // Value-head forward-intent enters the same forward-pull channel as the
    // moment-store embedding. Zero unless a run opted into the neural blend.
    let neural_attack = soccer_lp_unit(directive.neural_formation_attack_score);
    let neural_defense = soccer_lp_unit(directive.neural_formation_defense_score);
    let central_pressure =
        pressure_from_nearest_distance(snapshot.nearest_opponent_distance_at(team, ball));
    let rest_defense = soccer_formation_lp_rest_defense_score(snapshot, team);
    let fatigue = soccer_formation_lp_average_fatigue(snapshot, team);

    let mut weights = SoccerFormationLpObjectiveWeights {
        expected_goal: if has_ball {
            (attacking_depth * 0.62 + overload * 0.18 + numbers * 0.08 + embedding_attack * 0.12)
                .clamp(0.0, 1.0)
        } else {
            0.0
        },
        progression: if has_ball {
            (attacking_depth * 0.48
                + pass_priority * 0.15
                + carry_priority * 0.11
                + settled * 0.16
                + embedding_attack * 0.10)
                .clamp(0.0, 1.0)
        } else {
            0.0
        },
        retention: if has_ball {
            (1.0 - holder_pressure * 0.65
                + settled * 0.18
                + rest_defense * 0.13
                + embedding_defense * 0.04)
                .clamp(0.0, 1.0)
        } else {
            0.12
        },
        expected_goals_against: if defending || possession.is_none() {
            (own_goal_danger * 0.64
                + central_pressure * 0.17
                + (1.0 - rest_defense) * 0.09
                + embedding_defense * 0.10)
                .clamp(0.0, 1.0)
        } else {
            ((1.0 - rest_defense) * 0.36 + own_goal_danger * 0.22 + embedding_defense * 0.12)
                .clamp(0.0, 1.0)
        },
        fatigue,
        passing_lane_quality: if has_ball {
            (field_width_usage * 0.38
                + (shape.players_near_ball as f64 / 4.0).clamp(0.0, 1.0) * 0.24
                + (1.0 - holder_pressure) * 0.20
                + overload * 0.12
                + embedding_attack * 0.06)
                .clamp(0.0, 1.0)
        } else {
            0.0
        },
        space_occupation: (field_width_usage * 0.42
            + soccer_lp_unit(shape.spread_yards / 24.0) * 0.30
            + settled * 0.24
            + embedding_attack * 0.04)
            .clamp(0.0, 1.0),
        numerical_superiority: numbers,
        press_resistance: if has_ball {
            (1.0 - holder_pressure).clamp(0.0, 1.0)
        } else {
            0.0
        },
        defensive_compactness: if defending || possession.is_none() {
            ((1.0 - soccer_lp_unit(shape.spread_yards / 28.0)) * 0.58
                + press_intensity * 0.22
                + (opponent_shape.players_near_ball as f64 / 4.0).clamp(0.0, 1.0) * 0.16
                + embedding_defense * 0.04)
                .clamp(0.0, 1.0)
        } else {
            (rest_defense * 0.48 + embedding_defense * 0.10).clamp(0.0, 1.0)
        },
        opponent_progression: if defending {
            (own_goal_danger * 0.57
                + attacking_depth * 0.11
                + holder_pressure * 0.20
                + embedding_defense * 0.12)
                .clamp(0.0, 1.0)
        } else {
            0.0
        },
        transition_risk: if has_ball {
            ((1.0 - rest_defense) * 0.48
                + holder_pressure * 0.27
                + attacking_depth * 0.13
                + embedding_defense * 0.12)
                .clamp(0.0, 1.0)
        } else {
            (own_goal_danger * 0.88 + embedding_defense * 0.12).clamp(0.0, 1.0)
        },
        adversarial_embedding_attack: embedding_attack,
        adversarial_embedding_defense: embedding_defense,
        neural_attack,
        neural_defense,
    };
    soccer_formation_lp_apply_strategy_profile(&mut weights, directive, has_ball, defending);
    weights
}

/// Specialize the (single, guardrailed) LP objective toward the team's COMMITTED
/// play, so it optimizes the right thing per strategy: a pressing trap maximizes
/// high turnovers, a defensive shape prizes compactness/xGA and conserves energy,
/// a back-unit trap holds a tight high line, and attacking profiles weight direct
/// progression vs patient retention vs overloads vs wide play.
fn soccer_formation_lp_apply_strategy_profile(
    weights: &mut SoccerFormationLpObjectiveWeights,
    directive: &TeamTacticalDirective,
    has_ball: bool,
    defending: bool,
) {
    if defending {
        use TeamDefenseStrategy::*;
        match directive.defense_strategy {
            HighPressManOriented
            | HighPressZonalTrigger
            | CounterPressGegenpress
            | PressTriggerOnBackPass
            | WidePressOverloadBall => {
                weights.opponent_progression *= 1.5;
                weights.press_resistance *= 1.35;
                weights.numerical_superiority *= 1.25;
                weights.defensive_compactness *= 0.85;
                weights.fatigue *= 0.8;
            }
            LowBlockCompact
            | LowBlockParkBus
            | MidBlockZonal
            | MidBlockManMark
            | DropAndScreenCentral
            | ContainAndDelayCounter
            | RetreatRegroupHalfway
            | ShowInsideClampCentral
            | ManMarkKeyReceiver
            | DoubleTeamBallCarrier => {
                weights.defensive_compactness *= 1.5;
                weights.expected_goals_against *= 1.35;
                weights.opponent_progression *= 1.2;
                weights.progression *= 0.85;
                weights.fatigue *= 1.15;
            }
            OffsideTrapStepUp | ForceWideLeftTrap | ForceWideRightTrap => {
                weights.defensive_compactness *= 1.35;
                weights.opponent_progression *= 1.25;
            }
        }
    } else if has_ball {
        use TeamAttackStrategy::*;
        match directive.attack_strategy {
            CounterTransitionVertical
            | QuickVerticalThroughBall
            | DirectLongDiagonalLeft
            | DirectLongDiagonalRight => {
                weights.progression *= 1.3;
                weights.expected_goal *= 1.2;
                weights.retention *= 0.85;
            }
            PatientPossessionProbe | RecycleViaKeeperReset | HoldUpfieldUntilOpening => {
                weights.retention *= 1.3;
                weights.passing_lane_quality *= 1.2;
                weights.transition_risk *= 1.2;
            }
            PlayBackwardsToGoad => {
                // Keep the ball and deliberately DON'T rush forward — an inviting backward ball
                // baits the press up; only then is the forward space worth taking.
                weights.retention *= 1.45;
                weights.passing_lane_quality *= 1.25;
                weights.progression *= 0.6;
                weights.transition_risk *= 1.3;
            }
            OverloadLeftIsolateRight | OverloadRightIsolateLeft => {
                weights.numerical_superiority *= 1.4;
                weights.space_occupation *= 1.2;
            }
            WingOverlapLeftCross
            | WingOverlapRightCross
            | UnderlapLeftCutback
            | UnderlapRightCutback
            | PullWideLeftThenCenter
            | PullWideRightThenCenter => {
                weights.space_occupation *= 1.3;
                weights.expected_goal *= 1.15;
            }
            _ => {}
        }
    }
}

pub(crate) fn soccer_defensive_mark_target(
    snapshot: &WorldSnapshot,
    team: Team,
    defender_role: PlayerRole,
    zone: Vec2,
) -> Option<Vec2> {
    let (width, length) = sane_pitch_dimensions(snapshot.field_width, snapshot.field_length);
    let zone = finite_pitch_point(zone, width, length, Vec2::new(width * 0.5, length * 0.5));
    let own_goal = Vec2::new(width * 0.5, team.other().goal_y(length));
    let mut best_mark = None;
    let mut best_score = f64::INFINITY;
    for opponent in snapshot
        .players
        .iter()
        .filter(|player| player.team == team.other() && player.role != PlayerRole::Goalkeeper)
    {
        let opponent_position = finite_pitch_point(
            snapshot
                .player_position(opponent.id)
                .unwrap_or(opponent.position),
            width,
            length,
            opponent.position,
        );
        let zone_distance = opponent_position.distance(zone);
        let distance_to_goal = opponent_position.distance(own_goal);
        let role_threat = match opponent.role {
            PlayerRole::Forward => 2.4,
            PlayerRole::Midfielder => 1.4,
            PlayerRole::Defender => 0.4,
            PlayerRole::Goalkeeper => 0.0,
        };
        let holder_bonus = if snapshot.ball.holder == Some(opponent.id) {
            5.0
        } else {
            0.0
        };
        let goal_lane_bonus = if snapshot.clear_line(opponent_position, own_goal, team, 2.4) {
            1.2
        } else {
            0.0
        };
        let goal_threat_bonus = (70.0 - distance_to_goal).max(0.0) * 0.025;
        let score =
            zone_distance - role_threat - holder_bonus - goal_lane_bonus - goal_threat_bonus;
        if score < best_score {
            best_mark = Some((opponent_position, opponent.role));
            best_score = score;
        }
    }

    if let Some((cross_mark, cross_score)) =
        soccer_defensive_cross_arrival_mark_target(snapshot, team, defender_role, zone)
    {
        let cross_priority_margin = match defender_role {
            PlayerRole::Defender => 3.0,
            PlayerRole::Midfielder => 1.8,
            PlayerRole::Forward => 0.6,
            PlayerRole::Goalkeeper => 0.0,
        };
        if best_mark.is_none() || cross_score <= best_score + cross_priority_margin {
            return Some(cross_mark);
        }
    }

    let (mark, mark_role) = best_mark?;
    let goal_side = (own_goal - mark).normalized();
    let mark_distance = match defender_role {
        PlayerRole::Defender if mark_role == PlayerRole::Midfielder => {
            DEFENDER_PRESS_MIDFIELDER_IDEAL_YARDS
        }
        PlayerRole::Defender => 2.4,
        PlayerRole::Midfielder => 3.4,
        PlayerRole::Forward => 4.8,
        PlayerRole::Goalkeeper => 1.0,
    };
    Some((mark + goal_side * mark_distance).clamp_to_pitch(width, length))
}

pub(crate) fn soccer_defensive_cross_arrival_mark_target(
    snapshot: &WorldSnapshot,
    team: Team,
    defender_role: PlayerRole,
    zone: Vec2,
) -> Option<(Vec2, f64)> {
    if !matches!(defender_role, PlayerRole::Defender | PlayerRole::Midfielder)
        || snapshot.controlled_possession_team() != Some(team.other())
    {
        return None;
    }

    let (width, length) = sane_pitch_dimensions(snapshot.field_width, snapshot.field_length);
    let zone = finite_pitch_point(zone, width, length, Vec2::new(width * 0.5, length * 0.5));
    let attacking_team = team.other();
    let directive = snapshot.tactical_directive(attacking_team);
    if !directive.flank_attack_policy.is_flank() {
        return None;
    }

    let crosser_position = snapshot
        .ball
        .holder
        .and_then(|holder| {
            snapshot
                .players
                .iter()
                .find(|player| player.id == holder && player.team == attacking_team)
        })
        .and_then(|player| snapshot.player_position(player.id))
        .unwrap_or(snapshot.ball.position);
    if flank_lane_score(crosser_position, width) < 0.42 {
        return None;
    }

    let own_goal = Vec2::new(width * 0.5, team.other().goal_y(length));
    let center_x = width * 0.5;
    let mut best = None;
    let mut best_score = f64::INFINITY;

    for opponent in snapshot.players.iter().filter(|player| {
        player.team == attacking_team
            && matches!(player.role, PlayerRole::Forward | PlayerRole::Midfielder)
    }) {
        let Some(arrival) =
            snapshot.flank_cross_arrival_target_for(opponent.id, opponent.home_position)
        else {
            continue;
        };
        let arrival = finite_pitch_point(arrival, width, length, arrival);
        let distance_to_goal = arrival.distance(own_goal);
        if distance_to_goal > 24.0 {
            continue;
        }

        let runner_position = snapshot
            .player_position(opponent.id)
            .unwrap_or(opponent.position);
        let runner_distance = runner_position.distance(arrival);
        let goal_side = (own_goal - arrival).normalized();
        let mark_distance = match defender_role {
            PlayerRole::Defender => {
                if directive.flank_attack_policy.prefers_high_cross() {
                    1.8
                } else {
                    2.2
                }
            }
            PlayerRole::Midfielder => 3.0,
            PlayerRole::Forward | PlayerRole::Goalkeeper => 4.0,
        };
        let coverage = (arrival + goal_side * mark_distance).clamp_to_pitch(width, length);
        let lane_clear_threat = if snapshot.clear_line(
            crosser_position,
            arrival,
            team,
            if directive.flank_attack_policy.prefers_high_cross() {
                1.2
            } else {
                2.1
            },
        ) {
            1.35
        } else if directive.flank_attack_policy.prefers_high_cross() {
            0.55
        } else {
            -0.18
        };
        let role_threat = match opponent.role {
            PlayerRole::Forward => 3.0,
            PlayerRole::Midfielder => 1.65,
            PlayerRole::Defender => 0.45,
            PlayerRole::Goalkeeper => 0.0,
        };
        let centrality = (1.0 - ((arrival.x - center_x).abs() / center_x.max(1.0))).clamp(0.0, 1.0);
        let goalmouth_threat = ((24.0 - distance_to_goal) / 24.0).clamp(0.0, 1.0);
        let runner_commitment = ((18.0 - runner_distance) / 18.0).clamp(0.0, 1.0);
        let policy_threat = if directive.flank_attack_policy.prefers_high_cross() {
            match opponent.role {
                PlayerRole::Forward => 1.15,
                PlayerRole::Midfielder => 0.62,
                _ => 0.0,
            }
        } else {
            match opponent.role {
                PlayerRole::Forward => 0.78,
                PlayerRole::Midfielder => 0.90,
                _ => 0.0,
            }
        };
        let role_coverage_fit = match defender_role {
            PlayerRole::Defender => 1.10,
            PlayerRole::Midfielder => 0.52,
            PlayerRole::Forward | PlayerRole::Goalkeeper => 0.0,
        };
        let score = coverage.distance(zone) * 0.76 + runner_distance * 0.055
            - role_threat
            - policy_threat
            - role_coverage_fit
            - lane_clear_threat
            - centrality * 0.48
            - goalmouth_threat * 1.25
            - runner_commitment * 0.55;
        if score < best_score {
            best = Some(coverage);
            best_score = score;
        }
    }

    best.map(|target| (target, best_score))
}

/// The position that keeps a player's offset to its formation neighbours equal to its
/// ideal (base-formation) offset, anchored to where those neighbours actually are right
/// now. Because the ideal offset is the difference of the two players' home slots, this
/// reproduces every relative-shape relationship (CB-to-CB lateral gap, midfield-ahead-
/// of-defence depth, strikers ahead of midfield, …) without a hand-coded table, and it
/// slides with the neighbours so the block moves up and down field as a unit.
///
/// `roster` is `(id, team, current_position, home_position)` for all players; keepers
/// are excluded from being neighbours (they anchor to their box, not the outfield
/// shape). Returns `None` when the player has no outfield teammates.
// --- Fore-aft (depth) inter-line padding -------------------------------------
// `relational_shape_cohesion_target` only pulls a player toward its nearest HOME
// neighbours — which, in a back-four / midfield-three / front-three, are
// same-line teammates. That preserves LATERAL (intra-line) spacing but leaves
// the FORE-AFT gaps BETWEEN the Defender / Midfielder / Forward lines with no
// restoring force, so the lines drift together and compact vertically (the
// "lines too close fore/aft" symptom). These bound a soft, capped, deadzoned
// y-only nudge that keeps adjacent lines at least PADDING yards apart in depth.
// A nudge, never a hard constraint — a ball pinging end-to-end must stay
// chaseable, so we never pin a player against this.
pub(crate) const FORE_AFT_LINE_PADDING_YARDS: f64 = 9.0;
pub(crate) const FORE_AFT_LINE_PADDING_MAX_NUDGE_YARDS: f64 = 1.6;
pub(crate) const FORE_AFT_LINE_PADDING_MIN_LINE_MEMBERS: usize = 2;

pub(crate) fn relational_shape_cohesion_target(
    roster: &[(usize, Team, PlayerRole, Vec2, Vec2)],
    me_id: usize,
    me_team: Team,
    me_home: Vec2,
    params: &SoccerSpacingParams,
) -> Option<Vec2> {
    if !me_home.x.is_finite() || !me_home.y.is_finite() {
        return None;
    }
    // Only finite same-team outfield teammates can anchor the shape; a non-finite
    // position must never poison the average (it would NaN the whole signal).
    let mut neighbors: Vec<(f64, Vec2, Vec2)> = roster
        .iter()
        .filter(|&&(id, team, role, current, home)| {
            team == me_team
                && id != me_id
                && role != PlayerRole::Goalkeeper
                && current.x.is_finite()
                && current.y.is_finite()
                && home.x.is_finite()
                && home.y.is_finite()
        })
        .map(|&(_, _, _, current, home)| (me_home.distance(home), current, home))
        .collect();
    if neighbors.is_empty() {
        return None;
    }
    // Tie-break equal home distances by the (already-stable) ordering so the neighbour
    // set is deterministic.
    neighbors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let k = neighbors.len().min(params.relational_neighbors.max(1));
    let mut acc = Vec2::zero();
    for &(_, current, home) in &neighbors[..k] {
        // Where I should be to sit at my ideal offset from this neighbour.
        acc = acc + (current + (me_home - home));
    }
    let target = acc / k as f64;
    (target.x.is_finite() && target.y.is_finite()).then_some(target)
}

/// Out-of-band relational-shape error in yards: how far the player sits from its
/// relationally-implied position [`relational_shape_cohesion_target`] beyond the
/// tolerance band. Zero inside the band (statistical distance, not a hard rule).
fn relational_shape_error_yards(
    roster: &[(usize, Team, PlayerRole, Vec2, Vec2)],
    me_id: usize,
    me_team: Team,
    me_current: Vec2,
    me_home: Vec2,
    params: &SoccerSpacingParams,
) -> f64 {
    match relational_shape_cohesion_target(roster, me_id, me_team, me_home, params) {
        Some(target) => (me_current.distance(target) - params.relational_tolerance_yards).max(0.0),
        None => 0.0,
    }
}

/// MDP/POMDP shaping reward (integration #3) for relational shape: rewards shrinking a
/// player's out-of-band relational error across the decision interval, with a light
/// standing penalty on whatever error remains, so the learned policy develops a
/// *proclivity* to hold its relative offsets (cover ground as a unit) rather than only
/// reacting after drifting out of shape. Bounded by the error cap.
pub(crate) fn relational_shape_learning_reward(
    player_id: usize,
    team: Team,
    before: &WorldSnapshot,
    after: &WorldSnapshot,
    before_pos: Vec2,
    after_pos: Vec2,
) -> f64 {
    let Some(me_home) = after
        .players
        .iter()
        .chain(before.players.iter())
        .find(|p| p.id == player_id)
        .map(|p| p.home_position)
    else {
        return 0.0;
    };
    let roster = |snapshot: &WorldSnapshot| -> Vec<(usize, Team, PlayerRole, Vec2, Vec2)> {
        snapshot
            .players
            .iter()
            .map(|p| {
                (
                    p.id,
                    p.team,
                    p.role,
                    snapshot.player_snapshot_position(p),
                    p.home_position,
                )
            })
            .collect()
    };
    let params = after.spacing_params;
    let cap = params.relational_reward_cap_yards;
    let err_before = relational_shape_error_yards(
        &roster(before),
        player_id,
        team,
        before_pos,
        me_home,
        &params,
    )
    .min(cap);
    let err_after =
        relational_shape_error_yards(&roster(after), player_id, team, after_pos, me_home, &params)
            .min(cap);
    // Potential-based: + for shrinking the out-of-band error, - for growing it, zero for
    // holding station. Policy-invariant shaping — a proclivity, not a new objective.
    (err_before - err_after) * params.relational_reward_per_yard
}

fn soccer_formation_lp_anchor(
    snapshot: &WorldSnapshot,
    team: Team,
    directive: &TeamTacticalDirective,
    player: &PlayerSnapshot,
    weights: &SoccerFormationLpObjectiveWeights,
) -> Vec2 {
    let (width, length) = sane_pitch_dimensions(snapshot.field_width, snapshot.field_length);
    let center = Vec2::new(width * 0.5, length * 0.5);
    let attack_dir = team.attack_dir();
    let own_goal_y = team.other().goal_y(length);
    let home_position = finite_pitch_point(player.home_position, width, length, center);
    let home_lane = ((home_position.x - width * 0.5) / (width * 0.5)).clamp(-1.0, 1.0);
    let directive_width = soccer_lp_clamped(directive.width_yards, 14.0, width, width * 0.62);
    let defensive_line_y = soccer_lp_clamped(
        directive.defensive_line_y,
        0.0,
        length,
        match team {
            Team::Home => length * 0.40,
            Team::Away => length * 0.60,
        },
    );
    let support_depth_yards =
        soccer_lp_clamped(directive.support_depth_yards, 2.0, length * 0.5, 11.0);
    let press_intensity = soccer_lp_unit(directive.press_intensity);
    let mut x = width * 0.5 + home_lane * directive_width * 0.5;
    let ball = finite_pitch_point(snapshot.ball.position, width, length, center);
    let possession = snapshot
        .controlled_possession_team()
        .or_else(|| snapshot.possession_team());
    let y = match possession {
        Some(possessing) if possessing == team => match player.role {
            PlayerRole::Goalkeeper => own_goal_y + attack_dir * 7.0,
            PlayerRole::Defender => defensive_line_y - attack_dir * 2.0,
            PlayerRole::Midfielder => ball.y - attack_dir * support_depth_yards * 0.42,
            PlayerRole::Forward => ball.y + attack_dir * support_depth_yards * 0.72,
        },
        Some(possessing) if possessing == team.other() => match player.role {
            PlayerRole::Goalkeeper => own_goal_y + attack_dir * 6.0,
            PlayerRole::Defender => defensive_line_y,
            PlayerRole::Midfielder => {
                let drop = 4.0 + (1.0 - press_intensity) * 8.0;
                ball.y - attack_dir * drop
            }
            PlayerRole::Forward => {
                let drop = 1.5 + (1.0 - press_intensity) * 5.0;
                ball.y - attack_dir * drop
            }
        },
        _ => home_position.y * 0.55 + ball.y * 0.45,
    };

    if player.role != PlayerRole::Goalkeeper {
        let ball_lane_pull = ((ball.x - width * 0.5) / (width * 0.5)).clamp(-1.0, 1.0);
        let in_possession = possession == Some(team);
        let lane_commitment = role_vertical_lane_commitment(player.role, in_possession);
        let base_pull = if in_possession {
            0.18 + weights.progression * 0.10
        } else {
            0.12 + weights.defensive_compactness * 0.16
        };
        let mut pull = base_pull * (1.0 - lane_commitment * 0.46).clamp(0.30, 1.0);
        // The back four tracks the ball laterally more than lane discipline alone; wide
        // defenders (wing-backs) shuttle the most.
        if player.role == PlayerRole::Defender {
            let bonus = if home_lane.abs() > 0.44 {
                WINGBACK_BALL_SIDE_PULL_BONUS
            } else {
                DEFENDER_BALL_SIDE_PULL_BONUS
            };
            pull = (pull + bonus).min(0.9);
        }
        x = x * (1.0 - pull) + (width * 0.5 + ball_lane_pull * width * 0.28) * pull;
        let lane_x =
            vertical_lane_clamped_x_for_role(player.role, home_position.x, x, width, in_possession);
        x = x * (1.0 - lane_commitment * 0.28) + lane_x * (lane_commitment * 0.28);
    }

    let mut zone = Vec2::new(x, y).clamp_to_pitch(width, length);
    // Relational-shape proclivity (integration #2): when this player is *out* of its
    // relative shape, tilt the LP slot toward the position that restores its ideal
    // offsets to its formation neighbours, so the solved block keeps statistical
    // inter-role distances and travels as a unit. Gated on the current out-of-band error
    // and ramped with it, so an in-shape player is left fully ball-responsive (we never
    // fight a good shape) and only a drifting one is pulled back.
    if player.role != PlayerRole::Goalkeeper {
        let roster: Vec<(usize, Team, PlayerRole, Vec2, Vec2)> = snapshot
            .players
            .iter()
            .map(|p| {
                (
                    p.id,
                    p.team,
                    p.role,
                    snapshot.player_snapshot_position(p),
                    p.home_position,
                )
            })
            .collect();
        let current_pos = snapshot.player_snapshot_position(player);
        let params = snapshot.spacing_params;
        if let Some(relational) =
            relational_shape_cohesion_target(&roster, player.id, team, home_position, &params)
        {
            let out_of_band =
                (current_pos.distance(relational) - params.relational_tolerance_yards).max(0.0);
            if out_of_band > params.relational_lp_anchor_deadzone_yards {
                let ramp = out_of_band - params.relational_lp_anchor_deadzone_yards;
                let weight = (ramp / params.relational_lp_anchor_full_error_yards.max(1e-3)
                    * params.relational_lp_anchor_weight)
                    .min(params.relational_lp_anchor_weight);
                zone = (zone * (1.0 - weight) + relational * weight).clamp_to_pitch(width, length);
            }
        }
    }
    if possession == Some(team.other()) && player.role != PlayerRole::Goalkeeper {
        if let Some(mark_target) = soccer_defensive_mark_target(snapshot, team, player.role, zone) {
            return (mark_target * 0.5 + zone * 0.5).clamp_to_pitch(width, length);
        }
    }
    zone
}

pub(crate) fn soccer_formation_lp_slot_inputs(
    snapshot: &WorldSnapshot,
    team: Team,
    directive: &TeamTacticalDirective,
    weights: &SoccerFormationLpObjectiveWeights,
) -> Vec<SoccerFormationLpSlotInput> {
    let mut players = snapshot
        .players
        .iter()
        .filter(|player| player.team == team)
        .collect::<Vec<_>>();
    players.sort_by_key(|player| {
        let role_rank = match player.role {
            PlayerRole::Goalkeeper => 0usize,
            PlayerRole::Defender => 1,
            PlayerRole::Midfielder => 2,
            PlayerRole::Forward => 3,
        };
        (role_rank, player.id)
    });
    let (width, length) = sane_pitch_dimensions(snapshot.field_width, snapshot.field_length);
    let center = Vec2::new(width * 0.5, length * 0.5);
    let dt = sane_dt_seconds(snapshot.dt_seconds, DEFAULT_DT_SECONDS);

    let opponent_holder = snapshot.ball.holder.and_then(|holder_id| {
        let holder = snapshot
            .players
            .iter()
            .find(|player| player.id == holder_id)?;
        (holder.team == team.other()).then(|| {
            let holder_position = finite_pitch_point(
                snapshot
                    .player_position(holder.id)
                    .unwrap_or(holder.position),
                width,
                length,
                holder.position,
            );
            let holder_velocity = finite_vec2(holder.velocity, Vec2::zero());
            (holder.id, holder_position, holder_velocity)
        })
    });
    let mut pressure_ranks = HashMap::new();
    if let Some((_holder_id, holder_position, _holder_velocity)) = opponent_holder {
        let mut ranked = players
            .iter()
            .filter(|player| player.role != PlayerRole::Goalkeeper)
            .map(|player| {
                let position = finite_pitch_point(
                    snapshot
                        .player_position(player.id)
                        .unwrap_or(player.position),
                    width,
                    length,
                    player.position,
                );
                (player.id, position.distance(holder_position))
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        for (rank, (player_id, _)) in ranked.into_iter().enumerate() {
            pressure_ranks.insert(player_id, rank);
        }
    }

    let mut slots = Vec::with_capacity(SOCCER_FORMATION_LP_PLAYER_CAPACITY);
    for slot in 0..SOCCER_FORMATION_LP_PLAYER_CAPACITY {
        if let Some(player) = players.get(slot).copied() {
            let current = finite_pitch_point(
                snapshot
                    .player_position(player.id)
                    .unwrap_or(player.position),
                width,
                length,
                player.position,
            );
            let velocity = finite_vec2(player.velocity, Vec2::zero());
            let acceleration = finite_vec2(player.acceleration, Vec2::zero());
            let home_position = finite_pitch_point(player.home_position, width, length, center);
            let anchor = soccer_formation_lp_anchor(snapshot, team, directive, player, weights);
            let mut pressure_target = None;
            let mut pressure_weight = 0.0;
            let mut speed_match_weight = 0.0;
            let mut speed_match_velocity_yps = velocity.y;
            if let Some((_holder_id, holder_position, holder_velocity)) = opponent_holder {
                if let Some(rank) = pressure_ranks.get(&player.id).copied() {
                    let target_count = directive.defensive_cover_target.clamp(1, 4);
                    let near_rank = rank < target_count;
                    let role_factor = match player.role {
                        PlayerRole::Goalkeeper => 0.0,
                        PlayerRole::Defender => 1.0,
                        PlayerRole::Midfielder => 0.90,
                        PlayerRole::Forward => 0.46,
                    };
                    let rank_factor = if near_rank {
                        1.0 - rank as f64 / (target_count as f64 + 1.0)
                    } else {
                        0.12 / (rank as f64 + 1.0)
                    };
                    let pressure_need = soccer_lp_clamped(
                        0.36 + soccer_lp_unit(weights.expected_goals_against) * 0.34
                            + soccer_lp_unit(weights.opponent_progression) * 0.22
                            + soccer_lp_unit(directive.press_intensity) * 0.24,
                        0.0,
                        1.2,
                        0.0,
                    );
                    pressure_weight =
                        soccer_lp_clamped(rank_factor * role_factor * pressure_need, 0.0, 1.4, 0.0);
                    speed_match_weight = soccer_lp_unit(
                        pressure_weight * (0.42 + soccer_lp_unit(weights.transition_risk) * 0.28),
                    );
                    speed_match_velocity_yps = holder_velocity.y;
                    let holder_to_defender = (current - holder_position).normalized();
                    let approach = if holder_to_defender.len() > 1e-9 {
                        holder_to_defender
                    } else {
                        Vec2::new(0.0, -team.attack_dir())
                    };
                    pressure_target = Some(
                        (holder_position
                            + approach * SOCCER_FORMATION_LP_PRESS_DISTANCE_YARDS
                            + holder_velocity * (dt.max(0.0) * 0.45))
                            .clamp_to_pitch(width, length),
                    );
                }
            }
            let pair_weight = soccer_lp_clamped(
                0.08 + soccer_lp_unit(weights.space_occupation) * 0.12
                    + soccer_lp_unit(weights.defensive_compactness) * 0.16
                    + soccer_lp_unit(weights.transition_risk) * 0.08,
                0.04,
                0.42,
                0.08,
            );
            slots.push(SoccerFormationLpSlotInput {
                active: true,
                player_id: player.id,
                role: player.role,
                current,
                velocity,
                acceleration,
                home_position,
                top_speed: soccer_lp_clamped(
                    player_top_speed_yps(player.role, &player.skills),
                    0.0,
                    20.0,
                    0.0,
                ),
                max_acceleration: soccer_lp_clamped(
                    acceleration_yps2_from_score(player.skills.acceleration)
                        * strength_to_weight_acceleration_multiplier(&player.skills),
                    0.0,
                    30.0,
                    0.0,
                ),
                fatigue: soccer_lp_unit(player.fatigue),
                anchor,
                pressure_target,
                speed_match_velocity_yps,
                pressure_weight,
                speed_match_weight,
                pair_weight,
                alignment_weight: soccer_lp_clamped(
                    0.46 + pressure_weight * 0.42 + pair_weight * 0.38,
                    0.2,
                    1.6,
                    0.46,
                ),
            });
        } else {
            let anchor = center;
            slots.push(SoccerFormationLpSlotInput {
                active: false,
                player_id: usize::MAX - slot,
                role: PlayerRole::Midfielder,
                current: anchor,
                velocity: Vec2::zero(),
                acceleration: Vec2::zero(),
                home_position: anchor,
                top_speed: 0.0,
                max_acceleration: 0.0,
                fatigue: 0.0,
                anchor,
                pressure_target: None,
                speed_match_velocity_yps: 0.0,
                pressure_weight: 0.0,
                speed_match_weight: 0.0,
                pair_weight: 0.0,
                alignment_weight: 0.0,
            });
        }
    }
    // Seed the shape with ideal *relative* role positions (mids ahead of defs,
    // forwards ahead of mids, lateral spread within each line) so the team holds a
    // spread, cohesive block. A proclivity, not a command: only violations are
    // nudged, and the result is just the LP's anchor target, balanced downstream.
    if std::env::var("DD_SOCCER_DISABLE_FORMATION_STAGGER").is_err() {
        soccer_formation_lp_stagger_role_layers(&mut slots, team, width, length);
    }
    slots
}

/// Gentle, seed-valued staggering of the formation-LP anchors so the team keeps
/// ideal *relative* role positions and travels up/down the pitch as a unit:
/// midfielders ahead of defenders, forwards ahead of midfielders, and lateral
/// spread within the back line and the midfield. Only *violations* are corrected,
/// and only partway (a nudge toward the seed band, not a snap onto it), so a
/// compliant shape is left untouched. Anchors are a soft LP target, so the LP
/// still balances these against pressure, dynamics, and movement cost.
pub(crate) fn soccer_formation_lp_stagger_role_layers(
    slots: &mut [SoccerFormationLpSlotInput],
    team: Team,
    width: f64,
    length: f64,
) {
    let attack_dir = team.attack_dir();
    // Fraction of each deficit we correct per tick — keeps it a nudge.
    const STAGGER_CORRECTION: f64 = 0.6;

    let layer_mean = |slots: &[SoccerFormationLpSlotInput], role: PlayerRole| -> Option<f64> {
        let (sum, count) = slots
            .iter()
            .filter(|s| s.active && s.role == role)
            .fold((0.0_f64, 0usize), |(sum, count), s| {
                (sum + s.anchor.y * attack_dir, count + 1)
            });
        (count > 0).then(|| sum / count as f64)
    };

    // Forward layering (advancement measured along the attack direction). Lift a
    // whole layer *uniformly* — by its mean's deficit — only when that mean sits
    // too close to (or behind) the line behind it. This preserves each player's
    // intra-layer spread; lifting per-player would collapse the layer onto one
    // thin line and cause the very bunching this is meant to prevent.
    if let (Some(def_mean), Some(mid_mean)) = (
        layer_mean(slots, PlayerRole::Defender),
        layer_mean(slots, PlayerRole::Midfielder),
    ) {
        let gap = mid_mean - def_mean;
        if gap < MID_AHEAD_OF_DEF_MIN_YARDS {
            let lift = (MID_AHEAD_OF_DEF_MIN_YARDS - gap) * STAGGER_CORRECTION;
            for s in slots
                .iter_mut()
                .filter(|s| s.active && s.role == PlayerRole::Midfielder)
            {
                s.anchor.y += attack_dir * lift;
            }
        }
    }
    // Recompute the midfield line after the lift so forwards stagger off it.
    if let (Some(mid_mean), Some(fwd_mean)) = (
        layer_mean(slots, PlayerRole::Midfielder),
        layer_mean(slots, PlayerRole::Forward),
    ) {
        let gap = fwd_mean - mid_mean;
        if gap < STRIKER_AHEAD_OF_MID_MIN_YARDS {
            let lift = (STRIKER_AHEAD_OF_MID_MIN_YARDS - gap) * STAGGER_CORRECTION;
            for s in slots
                .iter_mut()
                .filter(|s| s.active && s.role == PlayerRole::Forward)
            {
                s.anchor.y += attack_dir * lift;
            }
        }
    }

    // Lateral spread within a line (identity = static home-x order, so a momentary
    // overlap does not reshuffle the line).
    soccer_formation_lp_spread_line_x(
        slots,
        PlayerRole::Defender,
        BACKLINE_LATERAL_MIN_YARDS,
        STAGGER_CORRECTION,
    );
    soccer_formation_lp_spread_line_x(
        slots,
        PlayerRole::Midfielder,
        MIDLINE_LATERAL_MIN_YARDS,
        STAGGER_CORRECTION,
    );

    for s in slots.iter_mut().filter(|s| s.active) {
        s.anchor = s.anchor.clamp_to_pitch(width, length);
    }
}

/// Push adjacent same-line anchors apart (in x) when they are closer than
/// `min_gap`, symmetrically and scaled by `correction`. Adjacency is by the
/// static home-x identity order. Idempotent on an already-spread line.
fn soccer_formation_lp_spread_line_x(
    slots: &mut [SoccerFormationLpSlotInput],
    role: PlayerRole,
    min_gap: f64,
    correction: f64,
) {
    let mut idx: Vec<usize> = (0..slots.len())
        .filter(|&i| slots[i].active && slots[i].role == role)
        .collect();
    if idx.len() < 2 {
        return;
    }
    idx.sort_by(|&a, &b| {
        slots[a]
            .home_position
            .x
            .partial_cmp(&slots[b].home_position.x)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for w in 1..idx.len() {
        let left = idx[w - 1];
        let right = idx[w];
        let dx = slots[right].anchor.x - slots[left].anchor.x;
        if dx < min_gap {
            let push = (min_gap - dx) * 0.5 * correction;
            slots[left].anchor.x -= push;
            slots[right].anchor.x += push;
        }
    }
}

fn soccer_formation_lp_reach_yards(input: &SoccerFormationLpSlotInput, dt_seconds: f64) -> f64 {
    let dt = if dt_seconds.is_finite() {
        dt_seconds.max(0.0)
    } else {
        0.0
    };
    let velocity = finite_vec2(input.velocity, Vec2::zero());
    let acceleration = finite_vec2(input.acceleration, Vec2::zero());
    let fatigue_factor = (1.0 - soccer_lp_unit(input.fatigue) * 0.38).clamp(0.35, 1.0);
    let speed_reach = soccer_lp_clamped(input.top_speed, 0.0, 20.0, 0.0)
        .max(velocity.len())
        .max(0.0)
        * fatigue_factor
        * dt;
    let acceleration_reach = 0.5
        * soccer_lp_clamped(input.max_acceleration, 0.0, 30.0, 0.0)
            .max(acceleration.len())
            .max(0.0)
        * dt
        * dt;
    (speed_reach + acceleration_reach + 0.12).clamp(0.0, 4.5)
}

pub(crate) fn soccer_formation_lp_state_values(
    snapshot: &WorldSnapshot,
    team: Team,
    weights: &SoccerFormationLpObjectiveWeights,
) -> Vec<f64> {
    let mut values = vec![
        0.0;
        SOCCER_FORMATION_LP_WORLD_PLAYER_CAPACITY * 6
            + 6
            + SOCCER_FORMATION_LP_CONTEXT_FEATURES
    ];
    let (width, length) = sane_pitch_dimensions(snapshot.field_width, snapshot.field_length);
    let center = Vec2::new(width * 0.5, length * 0.5);
    let mut players = snapshot.players.iter().collect::<Vec<_>>();
    players.sort_by_key(|player| player.id);
    for (slot, player) in players
        .into_iter()
        .take(SOCCER_FORMATION_LP_WORLD_PLAYER_CAPACITY)
        .enumerate()
    {
        let base = slot * 6;
        let position = finite_pitch_point(
            snapshot
                .player_position(player.id)
                .unwrap_or(player.position),
            width,
            length,
            player.home_position,
        );
        let velocity = finite_vec2(player.velocity, Vec2::zero());
        let acceleration = finite_vec2(player.acceleration, Vec2::zero());
        values[base] = (position.x / width * 2.0 - 1.0).clamp(-2.0, 2.0);
        values[base + 1] = (position.y / length * 2.0 - 1.0).clamp(-2.0, 2.0);
        values[base + 2] = soccer_lp_scaled(velocity.x, 12.0);
        values[base + 3] = soccer_lp_scaled(velocity.y, 12.0);
        values[base + 4] = soccer_lp_scaled(acceleration.x, 18.0);
        values[base + 5] = soccer_lp_scaled(acceleration.y, 18.0);
    }
    let ball_base = SOCCER_FORMATION_LP_WORLD_PLAYER_CAPACITY * 6;
    let ball_position = finite_pitch_point(snapshot.ball.position, width, length, center);
    let ball_velocity = finite_vec2(snapshot.ball.velocity, Vec2::zero());
    let ball_acceleration = finite_vec2(snapshot.ball.acceleration, Vec2::zero());
    values[ball_base] = (ball_position.x / width * 2.0 - 1.0).clamp(-2.0, 2.0);
    values[ball_base + 1] = (ball_position.y / length * 2.0 - 1.0).clamp(-2.0, 2.0);
    values[ball_base + 2] = soccer_lp_scaled(ball_velocity.x, 24.0);
    values[ball_base + 3] = soccer_lp_scaled(ball_velocity.y, 24.0);
    values[ball_base + 4] = soccer_lp_scaled(ball_acceleration.x, 36.0);
    values[ball_base + 5] = soccer_lp_scaled(ball_acceleration.y, 36.0);

    let context_base = ball_base + 6;
    let possession = snapshot
        .controlled_possession_team()
        .or_else(|| snapshot.possession_team());
    let shape = team_shape_observation_from_snapshot(snapshot, team, ball_position);
    let opp_shape = team_shape_observation_from_snapshot(snapshot, team.other(), ball_position);
    let directive = snapshot.tactical_directive(team);
    let holder_pressure = snapshot
        .ball
        .holder
        .and_then(|holder_id| {
            let holder = snapshot
                .players
                .iter()
                .find(|player| player.id == holder_id)?;
            let position = finite_pitch_point(
                snapshot
                    .player_position(holder.id)
                    .unwrap_or(holder.position),
                width,
                length,
                holder.position,
            );
            Some(pressure_from_nearest_distance(
                snapshot.nearest_opponent_distance_at(holder.team, position),
            ))
        })
        .unwrap_or(0.0);
    let mut context = [0.0; SOCCER_FORMATION_LP_CONTEXT_FEATURES];
    context[0] = team.attack_dir();
    context[1] = soccer_lp_unit((possession == Some(team)) as u8 as f64);
    context[2] = soccer_lp_unit((possession == Some(team.other())) as u8 as f64);
    context[3] = soccer_lp_scaled(snapshot.clock_seconds, DEFAULT_DURATION_SECONDS);
    context[4] = soccer_lp_scaled(snapshot.score_home as f64 - snapshot.score_away as f64, 5.0);
    context[5] = soccer_lp_scaled(ball_position.x - width * 0.5, width * 0.5);
    context[6] = soccer_lp_scaled(
        (ball_position.y - length * 0.5) * team.attack_dir(),
        length * 0.5,
    );
    context[7] = soccer_lp_unit(weights.expected_goal);
    context[8] = soccer_lp_unit(weights.progression);
    context[9] = soccer_lp_unit(weights.retention);
    context[10] = soccer_lp_unit(weights.expected_goals_against);
    context[11] = soccer_lp_unit(weights.fatigue);
    context[12] = soccer_lp_unit(weights.passing_lane_quality);
    context[13] = soccer_lp_unit(weights.space_occupation);
    context[14] = soccer_lp_unit(weights.numerical_superiority);
    context[15] = soccer_lp_unit(weights.press_resistance);
    context[16] = soccer_lp_unit(weights.defensive_compactness);
    context[17] = soccer_lp_unit(weights.opponent_progression);
    context[18] = soccer_lp_unit(weights.transition_risk);
    context[19] = soccer_lp_scaled(snapshot.possession_elapsed_seconds_for_team(team), 30.0);
    context[20] = soccer_lp_scaled(
        soccer_formation_lp_ball_holder_elapsed_seconds(snapshot),
        10.0,
    );
    context[21] = soccer_lp_scaled(
        ball_position.distance(Vec2::new(width * 0.5, team.goal_y(length))),
        length,
    );
    context[22] = soccer_lp_scaled(
        ball_position.distance(Vec2::new(width * 0.5, team.other().goal_y(length))),
        length,
    );
    context[23] = soccer_lp_scaled(shape.centroid_to_ball_yards, 40.0);
    context[24] = soccer_lp_scaled(shape.spread_yards, 35.0);
    context[25] = soccer_lp_scaled(shape.players_near_ball as f64, 6.0);
    context[26] = soccer_lp_scaled(shape.forward_velocity_yps, 8.0);
    context[27] = soccer_lp_scaled(opp_shape.centroid_to_ball_yards, 40.0);
    context[28] = soccer_lp_scaled(opp_shape.spread_yards, 35.0);
    context[29] = soccer_lp_scaled(opp_shape.players_near_ball as f64, 6.0);
    context[30] = soccer_lp_unit(directive.press_intensity);
    context[31] = soccer_lp_unit(directive.risk_tolerance);
    context[32] = soccer_lp_scaled(directive.defensive_cover_target as f64, 5.0);
    context[33] = soccer_lp_scaled(directive.defensive_cover_actual as f64, 5.0);
    context[34] = soccer_lp_scaled(directive.width_yards, width);
    context[35] = soccer_lp_scaled(directive.support_depth_yards, 24.0);
    context[36] = soccer_lp_unit(holder_pressure);
    context[37] = soccer_lp_unit(soccer_formation_lp_rest_defense_score(snapshot, team));
    context[38] = soccer_lp_unit(soccer_formation_lp_average_fatigue(snapshot, team));
    context[39] = soccer_lp_scaled(directive.attacking_numbers_advantage as f64, 4.0);
    context[40] = soccer_lp_unit(directive.attacking_overload_score);
    context[41] = soccer_lp_unit(weights.adversarial_embedding_attack);
    context[42] = soccer_lp_unit(weights.adversarial_embedding_defense);
    for (offset, value) in context.into_iter().enumerate() {
        values[context_base + offset] = soccer_lp_clamped(value, -2.0, 2.0, 0.0);
    }
    values
}

fn default_home_formation_lp_brain() -> SoccerFormationLpBrain {
    SoccerFormationLpBrain::new(Team::Home)
}

fn default_away_formation_lp_brain() -> SoccerFormationLpBrain {
    SoccerFormationLpBrain::new(Team::Away)
}

impl Default for SoccerFormationLpBrain {
    fn default() -> Self {
        default_home_formation_lp_brain()
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct DefensiveCoverProfile {
    pub(crate) target: usize,
    pub(crate) actual: usize,
    pub(crate) foremost_attacker_y: Option<f64>,
}

impl Default for DefensiveCoverProfile {
    fn default() -> Self {
        DefensiveCoverProfile {
            target: 2,
            actual: 2,
            foremost_attacker_y: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct AttackingOverloadProfile {
    pub(crate) attackers: usize,
    pub(crate) defenders: usize,
    pub(crate) advantage: i32,
    pub(crate) score: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SoccerAdversarialEmbeddingSignal {
    pub(crate) attack_score: f64,
    pub(crate) defense_score: f64,
    pub(crate) hits: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SoccerAdversarialEmbeddingSignals {
    pub(crate) home: SoccerAdversarialEmbeddingSignal,
    pub(crate) away: SoccerAdversarialEmbeddingSignal,
}

/// Upper bound on how much the NN→LP coupling can move a single team's
/// attack/defense embedding contribution in one tick. Small by design: the
/// critic biases the whole-field LP, it must never dominate it.
pub(crate) const SOCCER_NEURAL_LP_SIGNAL_CAP: f64 = 0.35;

/// Combine the moment-memory embedding signal with the neural-critic signal by
/// summing (clamped) their per-team scores. Either side may be absent.
pub(crate) fn merge_soccer_adversarial_embedding_signals(
    memory: Option<SoccerAdversarialEmbeddingSignals>,
    neural: Option<SoccerAdversarialEmbeddingSignals>,
) -> Option<SoccerAdversarialEmbeddingSignals> {
    match (memory, neural) {
        (Some(mut base), Some(extra)) => {
            for team in [Team::Home, Team::Away] {
                let add = extra.signal_for(team);
                let into = base.signal_for_mut(team);
                into.attack_score = (into.attack_score + add.attack_score).clamp(0.0, 1.0);
                into.defense_score = (into.defense_score + add.defense_score).clamp(0.0, 1.0);
                into.hits = into.hits.saturating_add(add.hits);
            }
            Some(base)
        }
        (Some(base), None) => Some(base),
        (None, Some(extra)) => Some(extra),
        (None, None) => None,
    }
}

impl SoccerAdversarialEmbeddingSignals {
    pub(crate) fn signal_for(&self, team: Team) -> SoccerAdversarialEmbeddingSignal {
        match team {
            Team::Home => self.home,
            Team::Away => self.away,
        }
    }

    pub(crate) fn signal_for_mut(&mut self, team: Team) -> &mut SoccerAdversarialEmbeddingSignal {
        match team {
            Team::Home => &mut self.home,
            Team::Away => &mut self.away,
        }
    }
}

/// Per-team value-head forward-intent applied to the formation LP each tick. The
/// trained value head says, for each team, how much it judges advancing the whole
/// shape is worth right now (`attack`) versus holding it back (`defense`); these
/// land on the directive and feed the LP's forward-pull coefficient, so the net
/// governs whole-field configuration through the LP — not just per-player action
/// choice. All-zero (the default) leaves the LP untouched.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SoccerNeuralFormationIntent {
    pub(crate) home_attack: f64,
    pub(crate) home_defense: f64,
    pub(crate) away_attack: f64,
    pub(crate) away_defense: f64,
}

impl SoccerNeuralFormationIntent {
    fn is_active(&self) -> bool {
        self.home_attack > 1e-9
            || self.home_defense > 1e-9
            || self.away_attack > 1e-9
            || self.away_defense > 1e-9
    }

    /// `(attack, defense)` intent for one team.
    fn for_team(&self, team: Team) -> (f64, f64) {
        match team {
            Team::Home => (self.home_attack, self.home_defense),
            Team::Away => (self.away_attack, self.away_defense),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CentralBrainTeamShapeTrace {
    pub centroid_to_ball_yards: f64,
    pub spread_yards: f64,
    /// Same-team players within the 18yd wide support band (NOT a swarm metric).
    pub players_near_ball: usize,
    /// Same-team players in the tight on-ball rings — the real congestion: expect
    /// ~1 within 1yd and 3yd, 1-3 within 5yd; more is a swarm.
    #[serde(default)]
    pub players_within_1yd: usize,
    #[serde(default)]
    pub players_within_3yd: usize,
    #[serde(default)]
    pub players_within_5yd: usize,
    pub forward_velocity_yps: f64,
}

impl Default for CentralBrainTeamShapeTrace {
    fn default() -> Self {
        Self {
            centroid_to_ball_yards: 0.0,
            spread_yards: 0.0,
            players_near_ball: 0,
            players_within_1yd: 0,
            players_within_3yd: 0,
            players_within_5yd: 0,
            forward_velocity_yps: 0.0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CentralBrainDecisionTrace {
    pub tick: u64,
    #[serde(default)]
    pub scheduled_index: Option<usize>,
    pub action: String,
    pub phase: TacticalPhase,
    pub possession_team: Option<Team>,
    pub ball_holder: Option<usize>,
    #[serde(default)]
    pub ball_position: Vec2,
    #[serde(default)]
    pub ball_velocity: Vec2,
    #[serde(default)]
    pub ball_acceleration: Vec2,
    #[serde(default)]
    pub ball_jerk: Vec2,
    #[serde(default)]
    pub ball_altitude_yards: f64,
    #[serde(default)]
    pub ball_scheduled_index: Option<usize>,
    #[serde(default)]
    pub home_shape: CentralBrainTeamShapeTrace,
    #[serde(default)]
    pub away_shape: CentralBrainTeamShapeTrace,
    pub tracked_players: usize,
    pub tracked_officials: usize,
    #[serde(default)]
    pub controlled_human_players: usize,
    #[serde(default)]
    pub controller_assignments: Vec<ControllerAssignment>,
    pub operation_order: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CentralBrain {
    pub phase: TacticalPhase,
    pub possession_team: Option<Team>,
    pub pressure_line_home: f64,
    pub pressure_line_away: f64,
    pub home_directive: TeamTacticalDirective,
    pub away_directive: TeamTacticalDirective,
    #[serde(skip)]
    pub(crate) home_strategy_commitment: TeamStrategyCommitment,
    #[serde(skip)]
    pub(crate) away_strategy_commitment: TeamStrategyCommitment,
    #[serde(skip)]
    pub(crate) home_attack_value: std::collections::HashMap<(u8, TeamAttackStrategy), f64>,
    #[serde(skip)]
    pub(crate) away_attack_value: std::collections::HashMap<(u8, TeamAttackStrategy), f64>,
    #[serde(default)]
    pub last_decision: Option<CentralBrainDecisionTrace>,
    #[serde(skip, default = "default_home_formation_lp_brain")]
    pub(crate) home_formation_lp: SoccerFormationLpBrain,
    #[serde(skip, default = "default_away_formation_lp_brain")]
    pub(crate) away_formation_lp: SoccerFormationLpBrain,
    /// Latest value-head forward-intent, set by the owning `SoccerMatch` each tick
    /// before the brain runs. Transient (not serialized); zero leaves the LP shape
    /// untouched.
    #[serde(skip)]
    pub(crate) neural_formation_intent: SoccerNeuralFormationIntent,
}

impl Default for CentralBrain {
    fn default() -> Self {
        CentralBrain {
            phase: TacticalPhase::Kickoff,
            possession_team: None,
            pressure_line_home: DEFAULT_FIELD_LENGTH_YARDS * 0.55,
            pressure_line_away: DEFAULT_FIELD_LENGTH_YARDS * 0.45,
            home_directive: TeamTacticalDirective::neutral(
                Team::Home,
                DEFAULT_FIELD_WIDTH_YARDS,
                DEFAULT_FIELD_LENGTH_YARDS,
            ),
            away_directive: TeamTacticalDirective::neutral(
                Team::Away,
                DEFAULT_FIELD_WIDTH_YARDS,
                DEFAULT_FIELD_LENGTH_YARDS,
            ),
            last_decision: None,
            home_strategy_commitment: TeamStrategyCommitment::default(),
            away_strategy_commitment: TeamStrategyCommitment::default(),
            home_attack_value: std::collections::HashMap::new(),
            away_attack_value: std::collections::HashMap::new(),
            home_formation_lp: default_home_formation_lp_brain(),
            away_formation_lp: default_away_formation_lp_brain(),
            neural_formation_intent: SoccerNeuralFormationIntent::default(),
        }
    }
}

impl CentralBrain {
    /// Pre-solve both formation LPs on the kickoff snapshot during match setup,
    /// before the clock starts. The first solve is the expensive cold one (IPM +
    /// crossover); paying it here primes each brain's warm-start basis so the
    /// first in-game tick already hits the cheap warm path — no kickoff hitch. A
    /// no-op when the exact solver is disabled (the heuristic carries no basis).
    pub fn prime_formation_lp(&mut self, snapshot: &WorldSnapshot) {
        if !snapshot.formation_lp_enabled {
            return;
        }
        let home_directive = self.home_directive.clone();
        self.home_formation_lp.solve_tick(snapshot, &home_directive);
        let away_directive = self.away_directive.clone();
        self.away_formation_lp.solve_tick(snapshot, &away_directive);
    }

    /// Hold the committed discrete strategy for the team, re-committing only when
    /// the commit window expires or possession flips — no per-tick flip-flop and no
    /// interpolation between strategies (they are exclusive maneuvers).
    fn apply_strategy_commitment(&mut self, team: Team, snapshot: &WorldSnapshot) {
        let tick = snapshot.tick;
        let has_ball = snapshot
            .controlled_possession_team()
            .or_else(|| snapshot.possession_team())
            == Some(team);
        let advantage = central_brain_team_advantage(snapshot, team);
        let ball_x = snapshot.ball.position.x;
        let width = snapshot.field_width;
        // Context bucket: possession phase + ball third + whether the opponent presses
        // high. Learned strategy value is keyed by this, so a team can have different
        // dominant strategies in different situations / vs different opponents.
        let opp_press = match team {
            Team::Home => self.away_directive.press_intensity,
            Team::Away => self.home_directive.press_intensity,
        };
        let length = snapshot.field_length.max(1.0);
        let to_opp_goal = (snapshot.ball.position.y - team.goal_y(length)).abs() / length;
        let third: u8 = if to_opp_goal < 0.34 {
            2
        } else if to_opp_goal < 0.67 {
            1
        } else {
            0
        };
        let ctx: u8 = ((has_ball as u8) << 3) | (third << 1) | u8::from(opp_press > 0.6);
        let (directive, commitment, value) = match team {
            Team::Home => (
                &mut self.home_directive,
                &mut self.home_strategy_commitment,
                &mut self.home_attack_value,
            ),
            Team::Away => (
                &mut self.away_directive,
                &mut self.away_strategy_commitment,
                &mut self.away_attack_value,
            ),
        };
        let possession_flip = commitment.set && commitment.committed_has_ball != has_ball;
        if !commitment.set || tick >= commitment.review_tick || possession_flip {
            // Learn: credit the just-finished commitment with the field advantage it
            // gained while running (only while we held the ball — attacking value).
            if commitment.set && commitment.committed_has_ball {
                let window_reward = advantage - commitment.advantage_at_commit;
                let entry = value
                    .entry((commitment.context, commitment.attack))
                    .or_insert(0.0);
                *entry = *entry * (1.0 - STRATEGY_VALUE_EMA_ALPHA)
                    + window_reward * STRATEGY_VALUE_EMA_ALPHA;
            }
            // Discrete argmax: the fresh rule candidate (fits the current state) vs the
            // proven held one (continuity bonus), each lifted by its learned value. A
            // held strategy with high learned value resists switching (the ensemble of
            // heuristic prior + learning + hysteresis), but we never interpolate.
            let rule_candidate = directive.attack_strategy;
            let held = commitment.attack;
            let chosen = if !commitment.set || rule_candidate == held {
                rule_candidate
            } else {
                let v_rule = value.get(&(ctx, rule_candidate)).copied().unwrap_or(0.0);
                let v_held = value.get(&(ctx, held)).copied().unwrap_or(0.0);
                let rule_score = STRATEGY_RULE_PRIOR + STRATEGY_VALUE_WEIGHT * v_rule;
                let held_score = STRATEGY_HELD_HYSTERESIS + STRATEGY_VALUE_WEIGHT * v_held;
                if held_score > rule_score {
                    held
                } else {
                    rule_candidate
                }
            };
            directive.attack_strategy = chosen;
            commitment.attack = chosen;
            commitment.defense = directive.defense_strategy;
            commitment.sub = select_nested_sub_maneuver(chosen, ball_x, width, has_ball);
            commitment.review_tick = tick.saturating_add(STRATEGY_COMMIT_TICKS);
            commitment.committed_has_ball = has_ball;
            commitment.advantage_at_commit = advantage;
            commitment.context = ctx;
            commitment.set = true;
        } else {
            directive.attack_strategy = commitment.attack;
            directive.defense_strategy = commitment.defense;
        }
        // Execute the nested Pair maneuver concurrently with the Team shape by
        // overlaying its effect on the directive params the players read.
        if let Some(sub) = commitment.sub {
            use TeamAttackStrategy::*;
            match sub {
                WingOverlapLeftCross
                | WingOverlapRightCross
                | UnderlapLeftCutback
                | UnderlapRightCutback => {
                    directive.flank_overlap_run_probability = directive
                        .flank_overlap_run_probability
                        .max(NESTED_OVERLAP_RUN_PROBABILITY);
                }
                GiveAndGoCentral | OneTwoLeftRelease | OneTwoRightRelease => {
                    directive.pass_priority *= NESTED_GIVE_AND_GO_PASS_BOOST;
                }
                _ => {}
            }
        }
    }
}

/// A team's field advantage right now: how far the ball is advanced toward the
/// opponent goal while we hold it, or the negative danger of it sitting near our
/// own goal while we don't. Drives the learned strategy-value signal.
/// Pick a lower-layer Pair maneuver to nest inside a Team-layer shape (in
/// possession): a wide ball invites a flank overlap, a central ball a give-and-go.
/// Enforces the declared rule — the nested maneuver must be a different, co-existing
/// layer than the primary.
pub(crate) fn select_nested_sub_maneuver(
    primary: TeamAttackStrategy,
    ball_x: f64,
    width: f64,
    has_ball: bool,
) -> Option<TeamAttackStrategy> {
    use TeamAttackStrategy::*;
    if !has_ball || primary.layer() != StrategyLayer::Team {
        return None;
    }
    let center = width * 0.5;
    let sub = if ball_x < center - width * 0.12 {
        WingOverlapLeftCross
    } else if ball_x > center + width * 0.12 {
        WingOverlapRightCross
    } else {
        GiveAndGoCentral
    };
    strategy_layers_can_coexist(primary.layer(), sub.layer()).then_some(sub)
}

fn central_brain_team_advantage(snapshot: &WorldSnapshot, team: Team) -> f64 {
    let length = snapshot.field_length.max(1.0);
    let ball = snapshot.ball.position;
    let has_ball = snapshot
        .controlled_possession_team()
        .or_else(|| snapshot.possession_team())
        == Some(team);
    let opp_progress = (1.0 - (ball.y - team.goal_y(length)).abs() / length).clamp(0.0, 1.0);
    let own_danger = (1.0 - (ball.y - team.other().goal_y(length)).abs() / length).clamp(0.0, 1.0);
    if has_ball {
        opp_progress
    } else {
        -own_danger
    }
}

pub(crate) fn central_brain_formation_lp_operation_label(
    formation_lp_enabled: bool,
) -> &'static str {
    if formation_lp_enabled {
        "solve-formation-lp"
    } else {
        "skip-formation-lp"
    }
}

pub(crate) fn central_brain_local_mpc_operation_label(
    local_mpc_enabled: bool,
    solved_players: usize,
) -> &'static str {
    if local_mpc_enabled && solved_players > 0 {
        "solve-local-mpc"
    } else if local_mpc_enabled {
        "fallback-local-mpc"
    } else {
        "skip-local-mpc"
    }
}

impl CentralBrain {
    pub fn run_time_step(&mut self, snapshot: &WorldSnapshot, rng: &mut SeededRandom) {
        self.run_time_step_with_adversarial_embeddings(snapshot, rng, None);
    }

    pub(crate) fn run_time_step_with_adversarial_embeddings(
        &mut self,
        snapshot: &WorldSnapshot,
        rng: &mut SeededRandom,
        embedding_signals: Option<&SoccerAdversarialEmbeddingSignals>,
    ) {
        self.possession_team = snapshot.possession_team();
        let y = snapshot.ball.position.y;
        self.phase = match self.possession_team {
            Some(Team::Home) if y > snapshot.field_length * 0.68 => TacticalPhase::HomeAttack,
            Some(Team::Home) => TacticalPhase::HomeBuildUp,
            Some(Team::Away) if y < snapshot.field_length * 0.32 => TacticalPhase::AwayAttack,
            Some(Team::Away) => TacticalPhase::AwayBuildUp,
            None if snapshot.tick < 5 => TacticalPhase::Kickoff,
            None => TacticalPhase::Transition,
        };
        let score_diff_home = snapshot.score_home as i32 - snapshot.score_away as i32;
        let home_cover =
            defensive_cover_profile(snapshot, Team::Home, sample_defensive_cover_target(rng));
        let away_cover =
            defensive_cover_profile(snapshot, Team::Away, sample_defensive_cover_target(rng));
        let home_overload = attacking_overload_profile(snapshot, Team::Home);
        let away_overload = attacking_overload_profile(snapshot, Team::Away);
        self.home_directive = tactical_directive_for_team(
            Team::Home,
            self.phase,
            self.possession_team,
            snapshot.ball.position,
            score_diff_home,
            snapshot.field_width,
            snapshot.field_length,
            home_cover,
            home_overload,
        );
        self.away_directive = tactical_directive_for_team(
            Team::Away,
            self.phase,
            self.possession_team,
            snapshot.ball.position,
            -score_diff_home,
            snapshot.field_width,
            snapshot.field_length,
            away_cover,
            away_overload,
        );
        // Commit to the freshly-selected strategy and hold it for a window so teams
        // run a maneuver out instead of flip-flopping the discrete strategy each tick.
        self.apply_strategy_commitment(Team::Home, snapshot);
        self.apply_strategy_commitment(Team::Away, snapshot);
        if let Some(signals) = embedding_signals {
            apply_adversarial_embedding_signal(
                &mut self.home_directive,
                signals.signal_for(Team::Home),
                snapshot.field_width,
                snapshot.field_length,
            );
            apply_adversarial_embedding_signal(
                &mut self.away_directive,
                signals.signal_for(Team::Away),
                snapshot.field_width,
                snapshot.field_length,
            );
        }
        // Value-head forward-intent into the whole-field shape (set by the match
        // before this tick). Lands on the directive so it flows into the formation
        // LP objective; zero leaves the LP shape unchanged.
        if self.neural_formation_intent.is_active() {
            let (home_attack, home_defense) = self.neural_formation_intent.for_team(Team::Home);
            self.home_directive.neural_formation_attack_score = home_attack;
            self.home_directive.neural_formation_defense_score = home_defense;
            let (away_attack, away_defense) = self.neural_formation_intent.for_team(Team::Away);
            self.away_directive.neural_formation_attack_score = away_attack;
            self.away_directive.neural_formation_defense_score = away_defense;
        }
        let mut local_mpc_solved_players = 0usize;
        let formation_lp_operation = if snapshot.formation_lp_enabled {
            let home_directive = self.home_directive.clone();
            self.home_formation_lp.solve_tick(snapshot, &home_directive);
            apply_formation_lp_directive_signal(
                &mut self.home_directive,
                &self.home_formation_lp.last_snapshot,
                &self.home_formation_lp.last_guidance,
            );
            let away_directive = self.away_directive.clone();
            self.away_formation_lp.solve_tick(snapshot, &away_directive);
            apply_formation_lp_directive_signal(
                &mut self.away_directive,
                &self.away_formation_lp.last_snapshot,
                &self.away_formation_lp.last_guidance,
            );
            if snapshot.local_mpc_enabled {
                let max_players = snapshot
                    .local_mpc_max_players_per_team
                    .min(SOCCER_MATCH_TEAM_PLAYER_COUNT);
                local_mpc_solved_players = local_mpc_solved_players.saturating_add(
                    self.home_formation_lp
                        .apply_local_mpc_tick(snapshot, max_players),
                );
                local_mpc_solved_players = local_mpc_solved_players.saturating_add(
                    self.away_formation_lp
                        .apply_local_mpc_tick(snapshot, max_players),
                );
            }
            central_brain_formation_lp_operation_label(true)
        } else {
            self.home_formation_lp.disable_tick();
            self.away_formation_lp.disable_tick();
            central_brain_formation_lp_operation_label(false)
        };
        let local_mpc_operation = central_brain_local_mpc_operation_label(
            snapshot.formation_lp_enabled && snapshot.local_mpc_enabled,
            local_mpc_solved_players,
        );
        self.pressure_line_home = self.home_directive.defensive_line_y;
        self.pressure_line_away = self.away_directive.defensive_line_y;
        let overload_weight = if self.possession_team.is_some() {
            1.15
        } else {
            0.72
        };
        let formation_weight = if snapshot.formation_lp_enabled {
            1.04
        } else {
            0.46
        };
        let mut central_operations = vec![
            ("sense-possession".to_string(), 1.32),
            ("classify-phase".to_string(), 1.18),
            ("sample-defensive-cover".to_string(), 0.92),
            ("measure-overloads".to_string(), overload_weight),
            ("issue-team-directives".to_string(), 1.24),
            (formation_lp_operation.to_string(), formation_weight),
        ];
        if snapshot.formation_lp_enabled && snapshot.local_mpc_enabled {
            let local_mpc_weight = if local_mpc_solved_players > 0 {
                0.86
            } else {
                0.32
            };
            central_operations.push((local_mpc_operation.to_string(), local_mpc_weight));
        }
        let operation_order = weighted_fisher_yates_order(central_operations, rng);
        let mut controller_assignments = snapshot
            .players
            .iter()
            .filter_map(|player| {
                player
                    .controller_slot
                    .map(|controller_slot| ControllerAssignment {
                        controller_slot,
                        player_id: player.id,
                        player_name: player.name.clone(),
                        team: player.team,
                    })
            })
            .collect::<Vec<_>>();
        controller_assignments.sort_by_key(|assignment| assignment.controller_slot);
        let ball_scheduled_index = snapshot.scheduled_ball_index();
        let home_shape =
            team_shape_observation_from_snapshot(snapshot, Team::Home, snapshot.ball.position);
        let away_shape =
            team_shape_observation_from_snapshot(snapshot, Team::Away, snapshot.ball.position);
        self.last_decision = Some(CentralBrainDecisionTrace {
            tick: snapshot.tick,
            scheduled_index: Some(0),
            action: format!("directives-{}", self.phase.action_label()),
            phase: self.phase,
            possession_team: self.possession_team,
            ball_holder: snapshot.ball.holder,
            ball_position: snapshot.ball.position,
            ball_velocity: snapshot.ball.velocity,
            ball_acceleration: snapshot.ball.acceleration,
            ball_jerk: snapshot.ball.jerk,
            ball_altitude_yards: snapshot.ball.altitude_yards,
            ball_scheduled_index,
            home_shape: central_brain_team_shape_trace(home_shape),
            away_shape: central_brain_team_shape_trace(away_shape),
            tracked_players: snapshot.players.len(),
            tracked_officials: snapshot.shared_positions.official_latest.len(),
            controlled_human_players: controller_assignments.len(),
            controller_assignments,
            operation_order,
        });
    }

    pub fn directive_for(&self, team: Team) -> &TeamTacticalDirective {
        match team {
            Team::Home => &self.home_directive,
            Team::Away => &self.away_directive,
        }
    }

    pub(crate) fn formation_lp_guidance(&self) -> Vec<SoccerFormationLpPlayerGuidance> {
        self.home_formation_lp
            .last_guidance
            .iter()
            .chain(self.away_formation_lp.last_guidance.iter())
            .cloned()
            .collect()
    }

    pub(crate) fn formation_lp_team_snapshots(&self) -> Vec<SoccerFormationLpTeamSnapshot> {
        vec![
            self.home_formation_lp.last_snapshot.clone(),
            self.away_formation_lp.last_snapshot.clone(),
        ]
    }

    pub fn to_snapshot(
        &self,
        snapshot: &WorldSnapshot,
        officials: &[OfficialAgent],
    ) -> CentralBrainSnapshot {
        let tracked_official_awareness = officials
            .iter()
            .map(|official| CentralBrainOfficialAwareness {
                id: official.id,
                kind: official.kind,
                position: official.position,
                velocity: official.velocity,
                acceleration: official.acceleration,
                scheduled_index: snapshot.scheduled_official_index(official.id),
                offside_line: assistant_offside_line_snapshot(snapshot, official.kind),
            })
            .collect::<Vec<_>>();
        let tracked_officials = tracked_official_awareness.len();
        let center_of_play = center_of_play_centroids(snapshot);
        let ball_scheduled_index = snapshot.scheduled_ball_index();
        let scheduled_index = snapshot.scheduled_central_brain_index();
        let tracked_players = snapshot
            .players
            .iter()
            .map(|player| CentralBrainPlayerAwareness {
                id: player.id,
                team: player.team,
                position: player.position,
                velocity: player.velocity,
                acceleration: player.acceleration,
                scheduled_index: player.scheduled_index,
                controller_slot: player.controller_slot,
            })
            .collect::<Vec<_>>();
        let mut controller_assignments = snapshot
            .players
            .iter()
            .filter_map(|player| {
                player
                    .controller_slot
                    .map(|controller_slot| ControllerAssignment {
                        controller_slot,
                        player_id: player.id,
                        player_name: player.name.clone(),
                        team: player.team,
                    })
            })
            .collect::<Vec<_>>();
        controller_assignments.sort_by_key(|assignment| assignment.controller_slot);
        let controlled_human_players = controller_assignments.len();
        CentralBrainSnapshot {
            scheduled_index,
            phase: self.phase,
            possession_team: self.possession_team.or_else(|| snapshot.possession_team()),
            ball_position: snapshot.ball.position,
            ball_velocity: snapshot.ball.velocity,
            ball_acceleration: snapshot.ball.acceleration,
            ball_jerk: snapshot.ball.jerk,
            ball_altitude_yards: snapshot.ball.altitude_yards,
            ball_holder: snapshot.ball.holder,
            ball_scheduled_index,
            ball_last_action: snapshot
                .ball
                .last_decision
                .as_ref()
                .map(|decision| decision.action.clone()),
            last_decision: self.last_decision.clone(),
            player_centroid: center_of_play.player_centroid,
            live_play_centroid: center_of_play.live_play_centroid,
            center_of_play: center_of_play.center_of_play,
            pressure_line_home: self.pressure_line_home,
            pressure_line_away: self.pressure_line_away,
            controlled_human_players,
            controller_assignments,
            tracked_players,
            tracked_officials,
            tracked_official_awareness,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TeamBrainMode {
    Kickoff,
    Attack,
    BuildUp,
    Defend,
    #[default]
    Transition,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TeamBrainSnapshot {
    pub team: Team,
    pub mode: TeamBrainMode,
    pub phase: TacticalPhase,
    pub possession_team: Option<Team>,
    pub in_possession: bool,
    pub ball_holder: Option<usize>,
    #[serde(default)]
    pub ball_position: Vec2,
    #[serde(default)]
    pub ball_velocity: Vec2,
    #[serde(default)]
    pub ball_acceleration: Vec2,
    #[serde(default)]
    pub ball_jerk: Vec2,
    #[serde(default)]
    pub ball_altitude_yards: f64,
    #[serde(default)]
    pub ball_scheduled_index: Option<usize>,
    #[serde(default)]
    pub ball_last_action: Option<String>,
    pub tracked_team_players: usize,
    pub controlled_human_players: usize,
    #[serde(default)]
    pub team_centroid: Vec2,
    #[serde(default)]
    pub team_average_velocity: Vec2,
    #[serde(default)]
    pub team_spread_yards: f64,
    #[serde(default)]
    pub players_near_ball: usize,
    pub directive: TeamTacticalDirective,
    pub policy_action_entries: usize,
    pub policy_target_entries: usize,
    pub attacking_numbers_advantage: i32,
    pub attacking_overload_score: f64,
    pub defensive_cover_target: usize,
    pub defensive_cover_actual: usize,
    pub press_intensity: f64,
    pub risk_tolerance: f64,
}

fn default_team_brain_snapshot(team: Team) -> TeamBrainSnapshot {
    let directive =
        TeamTacticalDirective::neutral(team, DEFAULT_FIELD_WIDTH_YARDS, DEFAULT_FIELD_LENGTH_YARDS);
    TeamBrainSnapshot {
        team,
        mode: TeamBrainMode::Transition,
        phase: TacticalPhase::Transition,
        possession_team: None,
        in_possession: false,
        ball_holder: None,
        ball_position: Vec2::zero(),
        ball_velocity: Vec2::zero(),
        ball_acceleration: Vec2::zero(),
        ball_jerk: Vec2::zero(),
        ball_altitude_yards: 0.0,
        ball_scheduled_index: None,
        ball_last_action: None,
        tracked_team_players: 0,
        controlled_human_players: 0,
        team_centroid: Vec2::zero(),
        team_average_velocity: Vec2::zero(),
        team_spread_yards: 0.0,
        players_near_ball: 0,
        policy_action_entries: 0,
        policy_target_entries: 0,
        attacking_numbers_advantage: directive.attacking_numbers_advantage,
        attacking_overload_score: directive.attacking_overload_score,
        defensive_cover_target: directive.defensive_cover_target,
        defensive_cover_actual: directive.defensive_cover_actual,
        press_intensity: directive.press_intensity,
        risk_tolerance: directive.risk_tolerance,
        directive,
    }
}

pub(crate) fn default_home_team_brain_snapshot() -> TeamBrainSnapshot {
    default_team_brain_snapshot(Team::Home)
}

pub(crate) fn default_away_team_brain_snapshot() -> TeamBrainSnapshot {
    default_team_brain_snapshot(Team::Away)
}

pub(crate) fn team_brain_mode_for(
    team: Team,
    phase: TacticalPhase,
    possession_team: Option<Team>,
) -> TeamBrainMode {
    if phase == TacticalPhase::Kickoff {
        return TeamBrainMode::Kickoff;
    }
    match possession_team {
        Some(possession) if possession == team => match (team, phase) {
            (Team::Home, TacticalPhase::HomeAttack) | (Team::Away, TacticalPhase::AwayAttack) => {
                TeamBrainMode::Attack
            }
            _ => TeamBrainMode::BuildUp,
        },
        Some(_) => TeamBrainMode::Defend,
        None => TeamBrainMode::Transition,
    }
}

pub(crate) fn team_brain_shape_from_awareness(
    players: &[CentralBrainPlayerAwareness],
    team: Team,
    ball_position: Vec2,
) -> (Vec2, Vec2, f64, usize) {
    let mut count = 0_usize;
    let mut position_sum = Vec2::zero();
    let mut velocity_sum = Vec2::zero();
    let mut players_near_ball = 0_usize;
    for player in players.iter().filter(|player| player.team == team) {
        count += 1;
        position_sum += player.position;
        velocity_sum += player.velocity;
        if player.position.distance(ball_position) <= TEAM_SHAPE_NEAR_BALL_RADIUS_YARDS {
            players_near_ball += 1;
        }
    }
    if count == 0 {
        return (Vec2::zero(), Vec2::zero(), 0.0, 0);
    }
    let count_f = count as f64;
    let centroid = position_sum / count_f;
    let average_velocity = velocity_sum / count_f;
    let spread = players
        .iter()
        .filter(|player| player.team == team)
        .map(|player| player.position.distance(centroid))
        .sum::<f64>()
        / count_f;
    (centroid, average_velocity, spread, players_near_ball)
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct TeamShapeObservation {
    pub(crate) centroid_to_ball_yards: f64,
    pub(crate) spread_yards: f64,
    pub(crate) players_near_ball: usize,
    // Tight same-team congestion rings around the ball (1/3/5yd) — the real
    // on-ball cluster size, as opposed to `players_near_ball` (18yd support band).
    players_within_1yd: usize,
    players_within_3yd: usize,
    players_within_5yd: usize,
    pub(crate) forward_velocity_yps: f64,
}

fn central_brain_team_shape_trace(shape: TeamShapeObservation) -> CentralBrainTeamShapeTrace {
    CentralBrainTeamShapeTrace {
        centroid_to_ball_yards: shape.centroid_to_ball_yards,
        spread_yards: shape.spread_yards,
        players_near_ball: shape.players_near_ball,
        players_within_1yd: shape.players_within_1yd,
        players_within_3yd: shape.players_within_3yd,
        players_within_5yd: shape.players_within_5yd,
        forward_velocity_yps: shape.forward_velocity_yps,
    }
}

pub(crate) fn team_shape_observation_from_snapshot(
    snapshot: &WorldSnapshot,
    team: Team,
    ball_position: Vec2,
) -> TeamShapeObservation {
    let (width, length) = sane_pitch_dimensions(snapshot.field_width, snapshot.field_length);
    let center = Vec2::new(width * 0.5, length * 0.5);
    let ball_position = finite_pitch_point(ball_position, width, length, center);
    let mut count = 0_usize;
    let mut position_sum = Vec2::zero();
    let mut velocity_sum = Vec2::zero();
    let mut players_near_ball = 0_usize;
    let mut players_within_1yd = 0_usize;
    let mut players_within_3yd = 0_usize;
    let mut players_within_5yd = 0_usize;

    for player in snapshot.players.iter().filter(|player| player.team == team) {
        let position = finite_pitch_point(
            snapshot
                .player_position(player.id)
                .unwrap_or(player.position),
            width,
            length,
            player.home_position,
        );
        count += 1;
        position_sum += position;
        velocity_sum += finite_vec2(player.velocity, Vec2::zero());
        let ball_distance = position.distance(ball_position);
        if ball_distance <= TEAM_SHAPE_NEAR_BALL_RADIUS_YARDS {
            players_near_ball += 1;
        }
        if ball_distance <= TEAM_BALL_RING_NEAR_YARDS {
            players_within_5yd += 1;
        }
        if ball_distance <= TEAM_BALL_RING_CLOSE_YARDS {
            players_within_3yd += 1;
        }
        if ball_distance <= TEAM_BALL_RING_TIGHT_YARDS {
            players_within_1yd += 1;
        }
    }

    if count == 0 {
        return TeamShapeObservation::default();
    }

    let count_f = count as f64;
    let centroid = position_sum / count_f;
    let average_velocity = velocity_sum / count_f;
    let spread_yards = snapshot
        .players
        .iter()
        .filter(|player| player.team == team)
        .map(|player| {
            finite_pitch_point(
                snapshot
                    .player_position(player.id)
                    .unwrap_or(player.position),
                width,
                length,
                player.home_position,
            )
            .distance(centroid)
        })
        .sum::<f64>()
        / count_f;

    TeamShapeObservation {
        centroid_to_ball_yards: centroid.distance(ball_position),
        spread_yards,
        players_near_ball,
        players_within_1yd,
        players_within_3yd,
        players_within_5yd,
        forward_velocity_yps: average_velocity.y * team.attack_dir(),
    }
}

/// A team's currently committed strategy. Strategies are discrete and exclusive
/// (no interpolation between them), so the team commits to one attack + one
/// defense maneuver and HOLDS it for a window, re-evaluating only when the window
/// expires or possession flips — rather than flip-flopping every tick.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TeamStrategyCommitment {
    attack: TeamAttackStrategy,
    defense: TeamDefenseStrategy,
    review_tick: u64,
    committed_has_ball: bool,
    advantage_at_commit: f64,
    /// Game-context bucket (phase/zone/opponent-press) the strategy was committed in,
    /// so learned value is situational — teams develop several context-dominant
    /// strategies rather than one global one.
    context: u8,
    /// A lower-layer Pair/Triangle maneuver nested inside the (Team-layer) primary
    /// — runs concurrently for the involved players while the rest hold the shape.
    sub: Option<TeamAttackStrategy>,
    set: bool,
}

impl Default for TeamStrategyCommitment {
    fn default() -> Self {
        Self {
            attack: TeamAttackStrategy::default(),
            defense: TeamDefenseStrategy::default(),
            review_tick: 0,
            committed_has_ball: false,
            advantage_at_commit: 0.0,
            context: 0,
            sub: None,
            set: false,
        }
    }
}
