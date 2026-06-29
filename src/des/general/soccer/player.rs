//! Player domain: the on-pitch [`PlayerAgent`] and its decision/skill model
//! (skills, preferences, the action space, intents, learned plans). Extracted
//! from the soccer god-module; see `super` for shared primitives and the world.

use super::*;

const DEFENDER_DRIBBLE_CLOSING_PASS_LIFT: f64 = 0.48;
const STEAL_RISK_GOOD_OUTLET_PASS_LIFT: f64 = 1.05;
const STEAL_RISK_BAD_OUTLET_ESCAPE_LIFT: f64 = 0.82;
// main and this branch convergently fixed the pressured-holder panic pass from opposite sides
// of the escape-vs-pass comparison: main strengthened the panic-pass damp (suppress the bad
// pass), this branch floored the body-shield independent of dribble skill (lift the correct
// action). Both fire only under genuine steal-pressure-with-no-outlet and reinforce each other,
// so keep both.
const STEAL_RISK_BAD_OUTLET_PANIC_PASS_DAMP_STRENGTH: f64 = 10.40;
// Turning your body to SHIELD a pinned ball needs no dribbling skill, so under genuine steal
// pressure with no good outlet the protect-ball score gets a floor INDEPENDENT of the
// (pressure-damped) dribble base — otherwise the one correct action collapses with the
// carrier's dribble score and loses to a release-urgency-lifted panic pass into traffic.
const PINNED_SHIELD_FLOOR_LIFT: f64 = 1.20;
// First-touch escape (selector side): a pressed receiver with a real, DEFENDER-AWARE lateral lane
// (observation `pressured_escape_lane_yards`, not pitch-edge room) should break contact with the
// dedicated evade rather than dwell on a static shield. A full lane is this many yards.
const FIRST_TOUCH_ESCAPE_LANE_MIN_YARDS: f64 = 4.0;
// Floor lift for the side-step escape (analogue of PINNED_SHIELD_FLOOR_LIFT): sized so a strong
// fresh-receipt-escape signal floors the evade above the pinned no-outlet shield.
const FIRST_TOUCH_ESCAPE_SIDE_STEP_FLOOR_LIFT: f64 = 2.7;
// A ball-holder under close pressure who is barely moving is usually just shielding with their
// back to goal. When a defender-aware escape lane exists, prefer breaking the cushion over
// dwelling on that static shield.
const PRESSURED_SLOW_HOLDER_SPEED_YPS: f64 = 1.65;
const PRESSURED_SLOW_HOLDER_ESCAPE_FLOOR_LIFT: f64 = 1.70;
const PRESSURED_SLOW_HOLDER_SHIELD_DAMP: f64 = 0.34;
/// Selector-side mirror of the first-touch-escape gate (same env var so the whole behaviour toggles
/// together). Default-off (escape ON); OFF zeroes the lift so the side-step score is byte-identical.
fn dd_soccer_disable_first_touch_escape() -> bool {
    use std::sync::OnceLock;
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("DD_SOCCER_DISABLE_FIRST_TOUCH_ESCAPE").is_ok())
}
/// Gate (default-ON) for the crowded won-ball escape floor: a player who has just won possession in
/// traffic gets a probability floor on the break-into-space action family so it accelerates AWAY
/// from the nearest presser into the open lane instead of settling into a shield/recycle. Set
/// `DD_SOCCER_DISABLE_WON_BALL_PRESSURE_ESCAPE=1` to disable (byte-identical to baseline).
fn dd_soccer_disable_won_ball_pressure_escape() -> bool {
    use std::sync::OnceLock;
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("DD_SOCCER_DISABLE_WON_BALL_PRESSURE_ESCAPE").is_ok())
}
const COMMITTED_LOOSE_BALL_SPRINT_MAX_DISTANCE_YARDS: f64 = 18.0;
const COMMITTED_LOOSE_BALL_SPRINT_BALL_SPEED_YPS: f64 = 1.15;
const MAX_PLAYER_BODY_YAW_TURN_PER_TICK_RAD: f64 = std::f64::consts::PI / 6.0;
const WIDE_OUTLET_WIDTH_TARGET_FRACTION: f64 = 0.92;
const WIDE_OUTLET_WIDTH_SHORTAGE_SCORE_LIFT: f64 = 0.42;
const WIDE_OUTLET_WIDTH_SHORTAGE_FLOOR_LIFT: f64 = 0.20;
const WIDE_OUTLET_NARROW_SHAPE_SUPPORT_DAMPING: f64 = 0.18;
const OPEN_PASS_LANE_MIN_DRIBBLE_YARDS: f64 = 1.0;
const OPEN_PASS_LANE_BASE_MAX_DRIBBLE_YARDS: f64 = 4.0;
const OPEN_PASS_LANE_MAX_DRIBBLE_YARDS: f64 = 8.0;
const OPEN_PASS_LANE_RADIUS_YARDS: f64 = 1.8;
const OPEN_PASS_LANE_ALREADY_SAFE_COMPLETION: f64 = 0.62;
const OPEN_PASS_LANE_ALREADY_SAFE_RISK: f64 = 0.28;
const OPEN_PASS_LANE_MIN_RECEIVER_OPENNESS: f64 = 0.24;
const OPEN_PASS_LANE_MIN_NEW_COMPLETION: f64 = 0.42;
const OPEN_PASS_LANE_MIN_LANE_GAIN: f64 = 0.16;
const OPEN_PASS_LANE_MIN_COMPLETION_GAIN: f64 = 0.10;
const OPEN_PASS_LANE_MIN_RISK_DROP: f64 = 0.14;
const OPEN_PASS_LANE_MAX_BACKWARD_STEP_YARDS: f64 = 0.65;
const OPEN_PASS_LANE_MIN_CANDIDATE_OPPONENT_SPACE: f64 = 1.35;
const OPEN_PASS_LANE_TARGET_SEARCH_LIMIT: usize = 8;
const OPEN_PASS_LANE_MAX_BACKFIELD_RECEIVER_YARDS: f64 = 18.0;
const OPEN_PASS_LANE_TRACKING_RADIUS_YARDS: f64 = 12.0;
const OPEN_PASS_LANE_TRACKING_REF_SPEED_YPS: f64 = 7.0;
const OPEN_PASS_LANE_SPRINT_PRESSURE: f64 = 0.62;
const OPEN_PASS_LANE_SPRINT_TRACKING: f64 = 0.42;
const OPEN_PASS_LANE_SPRINT_CLOSE_OPPONENT_YARDS: f64 = 3.6;
const ROUND_GOALKEEPER_MIN_YARDS_TO_GOAL: f64 = 5.5;
const ROUND_GOALKEEPER_MAX_YARDS_TO_GOAL: f64 = 26.0;
const ROUND_GOALKEEPER_MIN_DRIBBLE_YARDS: f64 = 1.2;
const ROUND_GOALKEEPER_MAX_DRIBBLE_YARDS: f64 = OPEN_PASS_LANE_MAX_DRIBBLE_YARDS;
const ROUND_GOALKEEPER_MIN_TARGET_DEPTH_YARDS: f64 = 4.8;
const ROUND_GOALKEEPER_MIN_KEEPER_CLEARANCE_YARDS: f64 = 2.15;
const ROUND_GOALKEEPER_MIN_PATH_KEEPER_GAP_YARDS: f64 = 1.05;
const ROUND_GOALKEEPER_MIN_CANDIDATE_OPPONENT_SPACE: f64 = 1.15;
const ROUND_GOALKEEPER_LONG_SHOT_YARDS: f64 = 20.0;

#[derive(Clone, Copy, Debug)]
struct OpenPassLaneSituation {
    pressure: f64,
    tracking: f64,
    extension: f64,
}

#[derive(Clone, Copy, Debug)]
struct OpenPassLaneEval {
    clear_now: bool,
    clear_through: bool,
    risk: f64,
    blockers: usize,
    receiver_openness: f64,
    expected_completion: f64,
    lane_score: f64,
}

#[derive(Clone, Copy, Debug)]
struct OpenPassLaneDribblePlan {
    target: Vec2,
    kind: DribbleMoveKind,
    touch: DribbleTouchDecision,
    score: f64,
}

#[derive(Clone, Copy, Debug)]
struct RoundGoalkeeperDribblePlan {
    target: Vec2,
    kind: DribbleMoveKind,
    touch: DribbleTouchDecision,
    score: f64,
    sprint: bool,
}

fn open_pass_lane_touch_for_target(
    team: Team,
    origin: Vec2,
    target: Vec2,
) -> Option<DribbleTouchDecision> {
    let delta = target - origin;
    let distance = delta.len();
    if !(OPEN_PASS_LANE_MIN_DRIBBLE_YARDS..=OPEN_PASS_LANE_MAX_DRIBBLE_YARDS + 1e-6)
        .contains(&distance)
    {
        return None;
    }
    let direction = delta.normalized();
    if direction.len() <= 1e-6 {
        return None;
    }
    let lateral = direction.x * team.attack_dir();
    let forward = direction.y * team.attack_dir();
    let mut angle_degrees = lateral.atan2(forward).to_degrees();
    if angle_degrees < 0.0 {
        angle_degrees += 360.0;
    }
    let bucket = ((angle_degrees / DRIBBLE_TOUCH_BUCKET_DEGREES).round() as u8)
        % DRIBBLE_TOUCH_ANGLE_BUCKETS;
    Some(DribbleTouchDecision::new(bucket, distance))
}

fn open_pass_lane_kind_for_target(team: Team, origin: Vec2, target: Vec2) -> DribbleMoveKind {
    let delta = target - origin;
    let forward = delta.y * team.attack_dir();
    if forward > delta.x.abs() * 1.25 {
        return DribbleMoveKind::CarryForward;
    }
    let left_x_sign = -team.attack_dir();
    if delta.x.signum() == left_x_sign.signum() {
        DribbleMoveKind::LeftCut
    } else {
        DribbleMoveKind::RightCut
    }
}

fn round_goalkeeper_kind_for_target(team: Team, origin: Vec2, target: Vec2) -> DribbleMoveKind {
    let delta = target - origin;
    let forward = delta.y * team.attack_dir();
    if forward > delta.x.abs() * 0.95 {
        return DribbleMoveKind::CarryForward;
    }
    let left_x_sign = -team.attack_dir();
    if delta.x.signum() == left_x_sign.signum() {
        DribbleMoveKind::LeftCut
    } else {
        DribbleMoveKind::RightCut
    }
}

fn open_pass_lane_pressure_signal(observation: &SoccerPomdpObservation) -> f64 {
    observation
        .pressure_urgency
        .max(observation.perceived_pressure)
        .max(observation.immediate_dispossession_risk)
        .max(pressure_rising_signal(observation) * 0.92)
        .clamp(0.0, 1.0)
}

fn open_pass_lane_tracking_signal(
    snapshot: &WorldSnapshot,
    team: Team,
    current: Vec2,
    target: Vec2,
) -> f64 {
    let dribble_vector = target - current;
    if dribble_vector.len() <= 1e-6 {
        return 0.0;
    }
    let dribble_dir = dribble_vector.normalized();
    let mut best = 0.0_f64;
    for opponent in snapshot
        .players
        .iter()
        .filter(|opponent| opponent.team == team.other())
    {
        let position = snapshot
            .player_position(opponent.id)
            .unwrap_or(opponent.position);
        let velocity = snapshot
            .player_velocity(opponent.id)
            .unwrap_or(opponent.velocity);
        if !velocity.x.is_finite() || !velocity.y.is_finite() {
            continue;
        }
        let current_distance = position.distance(current);
        let target_distance = position.distance(target);
        let proximity = (1.0
            - current_distance.min(target_distance) / OPEN_PASS_LANE_TRACKING_RADIUS_YARDS)
            .clamp(0.0, 1.0);
        if proximity <= 0.0 {
            continue;
        }
        let closing_current = if current_distance > 1e-6 {
            velocity.dot((current - position).normalized()).max(0.0)
        } else {
            velocity.len()
        };
        let closing_target = if target_distance > 1e-6 {
            velocity.dot((target - position).normalized()).max(0.0)
        } else {
            velocity.len()
        };
        let matching_run = velocity.dot(dribble_dir).max(0.0) * 0.82;
        let tracking_speed = closing_current.max(closing_target).max(matching_run);
        let speed_fit = (tracking_speed / OPEN_PASS_LANE_TRACKING_REF_SPEED_YPS).clamp(0.0, 1.0);
        best = best.max(speed_fit * proximity);
    }
    best.clamp(0.0, 1.0)
}

fn open_pass_lane_situation_for_target(
    snapshot: &WorldSnapshot,
    player: &PlayerAgent,
    observation: &SoccerPomdpObservation,
    current: Vec2,
    target: Vec2,
    lane_blocker_signal: f64,
) -> OpenPassLaneSituation {
    let pressure = open_pass_lane_pressure_signal(observation)
        .max(
            pressure_from_nearest_distance(
                snapshot.nearest_opponent_distance_at(player.team, current),
            ) * 0.88,
        )
        .clamp(0.0, 1.0);
    let tracking = open_pass_lane_tracking_signal(snapshot, player.team, current, target);
    let extension = pressure
        .max(tracking)
        .max(lane_blocker_signal.clamp(0.0, 1.0) * 0.92)
        .clamp(0.0, 1.0);
    OpenPassLaneSituation {
        pressure,
        tracking,
        extension,
    }
}

fn open_pass_lane_dynamic_max_dribble_yards(situation: OpenPassLaneSituation) -> f64 {
    (OPEN_PASS_LANE_BASE_MAX_DRIBBLE_YARDS
        + (OPEN_PASS_LANE_MAX_DRIBBLE_YARDS - OPEN_PASS_LANE_BASE_MAX_DRIBBLE_YARDS)
            * situation.extension)
        .clamp(
            OPEN_PASS_LANE_BASE_MAX_DRIBBLE_YARDS,
            OPEN_PASS_LANE_MAX_DRIBBLE_YARDS,
        )
}

fn open_pass_lane_should_sprint(
    snapshot: &WorldSnapshot,
    player: &PlayerAgent,
    observation: &SoccerPomdpObservation,
    current: Vec2,
    target: Vec2,
) -> bool {
    let situation =
        open_pass_lane_situation_for_target(snapshot, player, observation, current, target, 0.0);
    situation.pressure >= OPEN_PASS_LANE_SPRINT_PRESSURE
        || situation.tracking >= OPEN_PASS_LANE_SPRINT_TRACKING
        || observation.nearest_opponent_distance <= OPEN_PASS_LANE_SPRINT_CLOSE_OPPONENT_YARDS
}

fn open_pass_lane_sprint_for_action(
    snapshot: &WorldSnapshot,
    player: &PlayerAgent,
    observation: &SoccerPomdpObservation,
    action_label: &str,
    action: &SoccerAction,
) -> bool {
    if action_label != OPEN_PASS_LANE_ACTION_LABEL {
        return false;
    }
    let SoccerAction::DribbleMove { target, .. } = action else {
        return false;
    };
    let current = snapshot
        .player_position(player.id)
        .unwrap_or(player.position)
        .clamp_to_pitch(snapshot.field_width, snapshot.field_length);
    open_pass_lane_should_sprint(snapshot, player, observation, current, *target)
}

fn open_pass_lane_eval_for_target(
    snapshot: &WorldSnapshot,
    passer: &PlayerSnapshot,
    from: Vec2,
    target: &PlayerSnapshot,
    target_position: Vec2,
) -> OpenPassLaneEval {
    let is_cross = pass_would_be_cross(
        from,
        target_position,
        passer.team,
        snapshot.field_width,
        snapshot.field_length,
    );
    let speed = pass_speed_yps_from_power(0.68, PassFlight::Floor, is_cross, &passer.skills);
    let (clear_now, clear_through) = snapshot.pass_lane_clearance(
        from,
        target_position,
        passer.team.other(),
        OPEN_PASS_LANE_RADIUS_YARDS,
        speed,
    );
    let risk = snapshot
        .pass_lane_interception_risk(
            from,
            target_position,
            passer.team.other(),
            OPEN_PASS_LANE_RADIUS_YARDS,
            speed,
            PASS_LANE_DECISION_LOOKAHEAD_SECONDS,
        )
        .risk
        .clamp(0.0, 1.0);
    let blockers = snapshot.opponents_on_pass_path(
        from,
        target_position,
        passer.team.other(),
        OPEN_PASS_LANE_RADIUS_YARDS,
    );
    let quality = pass_target_quality_for_snapshot(
        snapshot,
        passer,
        from,
        target,
        target_position,
        PassFlight::Floor,
    );
    let clear_fit = if clear_through {
        1.0
    } else if clear_now {
        0.66
    } else {
        0.08
    };
    let traffic_fit = (1.0 - blockers as f64 * 0.24).clamp(0.0, 1.0);
    let lane_score = (clear_fit * 0.38
        + (1.0 - risk) * 0.30
        + quality.expected_completion.clamp(0.0, 1.0) * 0.24
        + traffic_fit * 0.08)
        .clamp(0.0, 1.0);
    OpenPassLaneEval {
        clear_now,
        clear_through,
        risk,
        blockers,
        receiver_openness: quality.receiver_openness.clamp(0.0, 1.0),
        expected_completion: quality.expected_completion.clamp(0.0, 1.0),
        lane_score,
    }
}

fn snapshot_team_lateral_width_yards(snapshot: &WorldSnapshot, team: Team) -> f64 {
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut count = 0usize;
    for player in snapshot
        .players
        .iter()
        .filter(|player| player.team == team && player.role != PlayerRole::Goalkeeper)
    {
        min_x = min_x.min(player.position.x);
        max_x = max_x.max(player.position.x);
        count += 1;
    }
    if count < 2 {
        0.0
    } else {
        (max_x - min_x).clamp(0.0, snapshot.field_width.max(0.0))
    }
}

fn possession_width_shortage(snapshot: &WorldSnapshot, directive: &TeamTacticalDirective) -> f64 {
    let target_width = directive
        .width_yards
        .max(snapshot.field_width * WIDE_OUTLET_WIDTH_TARGET_FRACTION)
        .min(snapshot.field_width * 0.99);
    if target_width <= 1e-6 {
        return 0.0;
    }
    ((target_width - snapshot_team_lateral_width_yards(snapshot, directive.team)) / target_width)
        .clamp(0.0, 1.0)
}

fn dd_soccer_disable_power_duration_ceiling() -> bool {
    use std::sync::OnceLock;
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("DD_SOCCER_DISABLE_POWER_DURATION_CEILING").is_ok())
}

/// Demo lever (mirrors the gate's `SOCCER_FORCE_RUNAROUND`): commit the run-around dribble whenever
/// the (relaxed) gate offers one, so it can be watched locally. Unset = the realistic appetite.
fn soccer_force_runaround_commit() -> bool {
    use std::sync::OnceLock;
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("SOCCER_FORCE_RUNAROUND").is_ok())
}

// Viability floor for PRE-EMPT-committing a `long-ball-in-behind` — an immediate, single-option
// aerial release that bypasses the ranked POMDP decision. Introspection of live play showed ~70%
// of ALL passes were this forced commit, most with ~0.0 modelled completion to a covered/■ runner
// (reading as "passed to nobody / the other team"), and because it is NOT a ranked option the
// learned policy could never down-weight it. Below these floors the long ball is a hopeful hoof, so
// we fall through to the full ranked decision instead. Disable via the flag below for parity / A-B.
const LONG_BALL_IN_BEHIND_MIN_AERIAL_COMPLETION: f64 = 0.40;
const LONG_BALL_IN_BEHIND_MIN_RUNNER_OPENNESS: f64 = 0.30;
const LONG_BALL_IN_BEHIND_MAX_INTERCEPTION_RISK: f64 = 0.55;

fn dd_soccer_disable_long_ball_viability_gate() -> bool {
    use std::sync::OnceLock;
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("DD_SOCCER_DISABLE_LONG_BALL_VIABILITY_GATE").is_ok())
}

/// Whether a `long-ball-in-behind` aerial release is on enough to PRE-EMPT-commit (skipping the
/// ranked decision). Gated ON by default; the disable flag restores the old always-fire behaviour
/// byte-for-byte. Reads only already-computed observation fields (no RNG, no mutation), so enabling
/// changes only WHICH deterministic decision path runs — a hopeless long ball now defers to the
/// full ranked POMDP decision (where ground passes / dribble / hold compete and the policy learns).
fn long_ball_in_behind_is_viable(observation: &SoccerPomdpObservation) -> bool {
    if dd_soccer_disable_long_ball_viability_gate() {
        return true;
    }
    observation.expected_aerial_pass_completion >= LONG_BALL_IN_BEHIND_MIN_AERIAL_COMPLETION
        && observation.best_aerial_pass_receiver_openness >= LONG_BALL_IN_BEHIND_MIN_RUNNER_OPENNESS
        && observation.aerial_pass_interception_risk <= LONG_BALL_IN_BEHIND_MAX_INTERCEPTION_RISK
}

fn is_give_and_go_strategy(strategy: TeamAttackStrategy) -> bool {
    matches!(
        strategy,
        TeamAttackStrategy::GiveAndGoCentral
            | TeamAttackStrategy::OneTwoLeftRelease
            | TeamAttackStrategy::OneTwoRightRelease
            | TeamAttackStrategy::CentralDoubleOneTwo
            | TeamAttackStrategy::BackheelDisguisedRelease
            // The "outside mid attack defender" play wants the wide man to either take the isolated
            // full-back on OR beat him with a give-and-go — so it requests the one-two appetite too.
            | TeamAttackStrategy::OutsideMidAttackDefenderLeft
            | TeamAttackStrategy::OutsideMidAttackDefenderRight
    )
}

fn directive_requests_give_and_go(directive: &TeamTacticalDirective) -> bool {
    is_give_and_go_strategy(directive.attack_strategy)
        || directive
            .pair_attack_strategy
            .map(is_give_and_go_strategy)
            .unwrap_or(false)
}

fn is_threaded_release_strategy(strategy: TeamAttackStrategy) -> bool {
    matches!(
        strategy,
        TeamAttackStrategy::QuickVerticalThroughBall
            | TeamAttackStrategy::DrawPressThenPlayThrough
            | TeamAttackStrategy::BaitOffsideThenThroughBall
            | TeamAttackStrategy::ThirdManRunCentral
            | TeamAttackStrategy::CentralDoubleOneTwo
            | TeamAttackStrategy::HalfSpaceComboLeft
            | TeamAttackStrategy::HalfSpaceComboRight
            | TeamAttackStrategy::ExploitSpace
            | TeamAttackStrategy::TransitionBurstOnRegain
    )
}

fn directive_requests_threaded_release(directive: &TeamTacticalDirective) -> bool {
    is_threaded_release_strategy(directive.attack_strategy)
        || directive
            .pair_attack_strategy
            .map(is_threaded_release_strategy)
            .unwrap_or(false)
}

fn is_lob_or_scoop_strategy(strategy: TeamAttackStrategy) -> bool {
    matches!(
        strategy,
        TeamAttackStrategy::QuickVerticalThroughBall
            | TeamAttackStrategy::DrawPressThenPlayThrough
            | TeamAttackStrategy::BaitOffsideThenThroughBall
            | TeamAttackStrategy::DirectLongDiagonalLeft
            | TeamAttackStrategy::DirectLongDiagonalRight
            | TeamAttackStrategy::AerialFlickOnRelease
            | TeamAttackStrategy::TransitionBurstOnRegain
    )
}

fn directive_requests_lob_or_scoop(directive: &TeamTacticalDirective) -> bool {
    is_lob_or_scoop_strategy(directive.attack_strategy)
        || directive
            .pair_attack_strategy
            .map(is_lob_or_scoop_strategy)
            .unwrap_or(false)
}

/// Strategies that commit a wide carrier to DRIVING the byline corner and crossing — the carrier
/// should keep driving (not slow / recycle) and then cross, while team-mates crash the box.
fn is_byline_drive_strategy(strategy: TeamAttackStrategy) -> bool {
    matches!(
        strategy,
        TeamAttackStrategy::BylineCrossLeftToPenaltySpot
            | TeamAttackStrategy::BylineCrossRightToPenaltySpot
            | TeamAttackStrategy::WingOverlapLeftCross
            | TeamAttackStrategy::WingOverlapRightCross
            | TeamAttackStrategy::CrashTheBox
    )
}

fn directive_requests_byline_drive(directive: &TeamTacticalDirective) -> bool {
    is_byline_drive_strategy(directive.attack_strategy)
        || directive
            .pair_attack_strategy
            .map(is_byline_drive_strategy)
            .unwrap_or(false)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SkillProfile {
    #[serde(alias = "topSpeedYps")]
    pub top_speed: f64,
    #[serde(alias = "accelerationYps2")]
    pub acceleration: f64,
    #[serde(default = "default_skill_strength")]
    pub strength: f64,
    #[serde(default = "default_skill_height", alias = "heightInches")]
    pub height: f64,
    #[serde(default = "default_player_weight_pounds", alias = "weightLbs")]
    pub weight_pounds: f64,
    #[serde(default = "default_skill_score")]
    pub shooting: f64,
    #[serde(default = "default_skill_score")]
    pub right_foot_shot_power: f64,
    #[serde(default = "default_skill_score")]
    pub left_foot_shot_power: f64,
    #[serde(default = "default_skill_score")]
    pub passing: f64,
    #[serde(default = "default_skill_score")]
    pub passing_completion_rate: f64,
    #[serde(default = "default_skill_score")]
    pub flair_passing: f64,
    #[serde(
        default = "default_skill_score",
        alias = "crossingAbilityWithLeftRoot",
        alias = "leftFootCrossingAbility"
    )]
    pub crossing_left: f64,
    #[serde(default = "default_skill_score", alias = "rightFootCrossingAbility")]
    pub crossing_right: f64,
    #[serde(default = "default_skill_score")]
    pub dribbling: f64,
    #[serde(default = "default_skill_score")]
    pub first_touch: f64,
    #[serde(default = "default_skill_score", alias = "defensiveAbility")]
    pub defending: f64,
    #[serde(default = "default_goalkeeping_skill", alias = "abilityInGoal")]
    pub goalkeeping: f64,
    #[serde(default = "default_skill_score", alias = "trackingBack")]
    pub defensive_tracking: f64,
    #[serde(default = "default_skill_score")]
    pub stamina: f64,
    #[serde(default = "default_player_vision_skill")]
    pub vision: f64,
    pub decision_noise: f64,
    pub aggression: f64,
}

/// Per-shirt position accent: small, deterministic shaping deltas applied to the
/// role baseline so each of the 11 shirts plays to type (a #9 finisher vs a #6
/// holder) while every player stays the *same overall standard*. The deltas are
/// hand-balanced to net ≈0 per shirt, so no position is globally stronger — only
/// differently shaped. Fields left at 0.0 inherit the role baseline unchanged.
#[derive(Clone, Copy, Default)]
struct ShirtAccent {
    top_speed: f64,
    acceleration: f64,
    strength: f64,
    height: f64,
    shooting: f64,
    passing: f64,
    dribbling: f64,
    first_touch: f64,
    defending: f64,
    vision: f64,
    crossing: f64,
    stamina: f64,
    aggression: f64,
}

impl SkillProfile {
    /// Balanced per-role attribute baseline (14-tuple): top_speed, acceleration,
    /// strength, height, shooting, passing, dribbling, first_touch, defending,
    /// stamina, vision, aggression, goalkeeping, defensive_tracking_base. Shared by
    /// the randomized [`Self::blended`] and the deterministic [`Self::for_shirt`].
    fn role_bias_tuple(
        role: PlayerRole,
    ) -> (
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
    ) {
        match role {
            PlayerRole::Goalkeeper => (
                5.8, 6.0, 8.0, 8.6, 3.6, 6.2, 4.0, 7.4, 7.2, 7.8, 7.0, 4.2, 9.2, 7.2,
            ),
            PlayerRole::Defender => (
                6.9, 6.8, 8.0, 7.6, 5.0, 6.8, 5.6, 6.5, 8.2, 8.2, 7.4, 7.0, 2.0, 8.6,
            ),
            PlayerRole::Midfielder => (
                7.4, 7.5, 6.8, 6.0, 6.7, 8.1, 7.7, 7.7, 6.8, 8.7, 8.4, 6.2, 1.7, 7.4,
            ),
            PlayerRole::Forward => (
                9.3, 8.4, 7.5, 7.0, 8.4, 7.1, 8.6, 7.5, 4.9, 8.0, 7.8, 7.4, 1.5, 5.3,
            ),
        }
    }

    /// Deterministic, RNG-free skill profile for a given shirt number (1–11): every
    /// player is the *same overall standard*, distinguished only by a
    /// position-shaped accent, and identical between the two teams (Home #7 ==
    /// Away #7). This is the "all players basically equal besides their position
    /// 1–11" model — no per-player or per-team randomness, so neither side starts
    /// with a skill edge and both learn from an equal footing.
    pub(crate) fn for_shirt(shirt: u8, role: PlayerRole) -> Self {
        let bias = Self::role_bias_tuple(role);
        // Position accent per shirt (1 GK … 11 LW). Deltas net ≈0 so the overall
        // standard is equal; they only re-shape toward the position's job.
        let a = match shirt {
            // GK
            1 => ShirtAccent::default(),
            // Right-back: pace + crossing for the overlap, a touch less aerial.
            2 => ShirtAccent {
                top_speed: 0.5,
                acceleration: 0.4,
                crossing: 0.7,
                stamina: 0.4,
                height: -0.6,
                strength: -0.4,
                ..Default::default()
            },
            // Left-back: mirror of the right-back (footedness handled below).
            3 => ShirtAccent {
                top_speed: 0.5,
                acceleration: 0.4,
                crossing: 0.7,
                stamina: 0.4,
                height: -0.6,
                strength: -0.4,
                ..Default::default()
            },
            // Right centre-back: dominant aerial/strength, slower.
            4 => ShirtAccent {
                strength: 0.6,
                height: 0.7,
                defending: 0.5,
                aggression: 0.3,
                top_speed: -0.6,
                dribbling: -0.5,
                ..Default::default()
            },
            // Left centre-back: same shape as the #4.
            5 => ShirtAccent {
                strength: 0.6,
                height: 0.7,
                defending: 0.5,
                aggression: 0.3,
                top_speed: -0.6,
                dribbling: -0.5,
                ..Default::default()
            },
            // Holding midfielder: screen + distribute, less attacking thrust.
            6 => ShirtAccent {
                defending: 0.7,
                passing: 0.4,
                stamina: 0.3,
                shooting: -0.6,
                dribbling: -0.4,
                ..Default::default()
            },
            // Right winger: pace + dribbling + crossing, lighter defensively.
            7 => ShirtAccent {
                top_speed: 0.7,
                acceleration: 0.6,
                dribbling: 0.7,
                crossing: 0.6,
                defending: -0.8,
                strength: -0.4,
                ..Default::default()
            },
            // Box-to-box centre midfielder: passing, vision, engine.
            8 => ShirtAccent {
                passing: 0.5,
                vision: 0.6,
                stamina: 0.6,
                strength: 0.2,
                top_speed: -0.4,
                ..Default::default()
            },
            // Centre-forward: finishing + strength, less creation/defending.
            9 => ShirtAccent {
                shooting: 0.8,
                strength: 0.5,
                first_touch: 0.3,
                passing: -0.6,
                defending: -0.6,
                ..Default::default()
            },
            // Attacking midfielder / second striker: flair, vision, dribbling.
            10 => ShirtAccent {
                vision: 0.7,
                dribbling: 0.6,
                passing: 0.5,
                first_touch: 0.3,
                defending: -0.7,
                strength: -0.6,
                ..Default::default()
            },
            // Left winger: mirror of the right winger.
            11 => ShirtAccent {
                top_speed: 0.7,
                acceleration: 0.6,
                dribbling: 0.7,
                crossing: 0.6,
                defending: -0.8,
                strength: -0.4,
                ..Default::default()
            },
            _ => ShirtAccent::default(),
        };

        let clamp = |v: f64| v.clamp(1.0, 10.0);
        let shooting = clamp(bias.4 + a.shooting);
        let passing = clamp(bias.5 + a.passing);
        let dribbling = clamp(bias.6 + a.dribbling);
        let first_touch = clamp(bias.7 + a.first_touch);
        let defending = clamp(bias.8 + a.defending);
        let strength = clamp(bias.2 + a.strength);
        let height = clamp(bias.3 + a.height);
        // Footedness is positional, not random: the left-flank shirts are
        // left-footed; everyone else is right-footed. Deterministic and mirrored.
        let left_footed = matches!(shirt, 3 | 11);
        let strong_foot = clamp(shooting + 0.3);
        let weak_foot = clamp(shooting - 1.1);
        let (right_foot_shot_power, left_foot_shot_power) = if left_footed {
            (weak_foot, strong_foot)
        } else {
            (strong_foot, weak_foot)
        };
        let crossing_strong = clamp(passing * 0.76 + first_touch * 0.16 + a.crossing);
        let crossing_weak = clamp(passing * 0.72 + first_touch * 0.18 + a.crossing);
        let (crossing_right, crossing_left) = if left_footed {
            (crossing_weak, crossing_strong)
        } else {
            (crossing_strong, crossing_weak)
        };
        let weight_pounds = default_player_weight_for_traits(role, strength, height, 0.0);
        SkillProfile {
            top_speed: clamp(bias.0 + a.top_speed),
            acceleration: clamp(bias.1 + a.acceleration),
            strength,
            height,
            weight_pounds,
            shooting,
            right_foot_shot_power,
            left_foot_shot_power,
            passing,
            passing_completion_rate: clamp(passing * 0.86 + first_touch * 0.12),
            flair_passing: clamp(passing * 0.58 + dribbling * 0.30),
            crossing_left,
            crossing_right,
            dribbling,
            first_touch,
            defending,
            goalkeeping: clamp(bias.12),
            defensive_tracking: clamp(bias.13 * 0.78 + defending * 0.22),
            stamina: clamp(bias.9 + a.stamina),
            vision: clamp(bias.10 + a.vision),
            // Equal decision discipline for everyone — no per-player noise edge.
            decision_noise: 0.08,
            aggression: clamp(bias.11 + a.aggression),
        }
    }

    pub(crate) fn blended(seed: usize, role: PlayerRole, _rng: &mut SeededRandom) -> Self {
        let role_bias = Self::role_bias_tuple(role);
        let role_salt = match role {
            PlayerRole::Goalkeeper => 101,
            PlayerRole::Defender => 211,
            PlayerRole::Midfielder => 307,
            PlayerRole::Forward => 401,
        };
        let jitter = |salt: u64, scale: f64| {
            (deterministic_unit_draw(seed as u64, role_salt, salt) - 0.5) * scale
        };
        let shooting = (role_bias.4 + jitter(1, 1.1)).clamp(1.0, 10.0);
        let passing = (role_bias.5 + jitter(2, 1.0)).clamp(1.0, 10.0);
        let dribbling = (role_bias.6 + jitter(3, 1.0)).clamp(1.0, 10.0);
        let first_touch = (role_bias.7 + jitter(4, 0.9)).clamp(1.0, 10.0);
        let defending = (role_bias.8 + jitter(5, 1.0)).clamp(1.0, 10.0);
        let dominant_right = seed % 5 != 0;
        let strong_foot = (shooting + jitter(6, 0.6)).clamp(1.0, 10.0);
        let weak_foot = (shooting - 1.1 + jitter(7, 0.8)).clamp(1.0, 10.0);
        let (right_foot_shot_power, left_foot_shot_power) = if dominant_right {
            (strong_foot, weak_foot)
        } else {
            (weak_foot, strong_foot)
        };
        let crossing_left = (passing * 0.72 + first_touch * 0.18 + jitter(8, 0.9)).clamp(1.0, 10.0);
        let crossing_right =
            (passing * 0.76 + first_touch * 0.16 + jitter(9, 0.9)).clamp(1.0, 10.0);
        let strength = (role_bias.2 + jitter(10, 0.9)).clamp(1.0, 10.0);
        let height = (role_bias.3 + jitter(11, 0.9)).clamp(1.0, 10.0);
        let weight_pounds =
            default_player_weight_for_traits(role, strength, height, jitter(12, 0.55));
        SkillProfile {
            top_speed: role_bias.0 + jitter(13, 0.9) + (seed % 3) as f64 * 0.08,
            acceleration: role_bias.1 + jitter(14, 0.8),
            strength,
            height,
            weight_pounds,
            shooting,
            right_foot_shot_power,
            left_foot_shot_power,
            passing,
            passing_completion_rate: (passing * 0.86 + first_touch * 0.12 + jitter(15, 0.45))
                .clamp(1.0, 10.0),
            flair_passing: (passing * 0.58 + dribbling * 0.30 + jitter(16, 0.9)).clamp(1.0, 10.0),
            crossing_left,
            crossing_right,
            dribbling,
            first_touch,
            defending,
            goalkeeping: (role_bias.12 + jitter(17, 0.8)).clamp(1.0, 10.0),
            defensive_tracking: (role_bias.13 * 0.78 + defending * 0.22 + jitter(18, 0.7))
                .clamp(1.0, 10.0),
            stamina: (role_bias.9 + jitter(19, 0.8)).clamp(1.0, 10.0),
            vision: (role_bias.10 + jitter(20, 0.8)).clamp(1.0, 10.0),
            decision_noise: (0.08 + jitter(21, 0.08)).clamp(0.01, 0.18),
            aggression: (role_bias.11 + jitter(22, 0.9)).clamp(1.0, 10.0),
        }
    }
}

impl Default for SkillProfile {
    fn default() -> Self {
        SkillProfile {
            top_speed: 7.4,
            acceleration: 7.2,
            strength: 6.8,
            height: 6.6,
            weight_pounds: default_player_weight_pounds(),
            shooting: 6.2,
            right_foot_shot_power: 6.6,
            left_foot_shot_power: 5.4,
            passing: 7.2,
            passing_completion_rate: 7.2,
            flair_passing: 4.2,
            crossing_left: 6.6,
            crossing_right: 7.0,
            dribbling: 6.8,
            first_touch: 7.0,
            defending: 6.2,
            goalkeeping: 2.0,
            defensive_tracking: 6.2,
            stamina: 8.2,
            vision: DEFAULT_PLAYER_VISION_SKILL,
            decision_noise: 0.08,
            aggression: 5.5,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentPreferences {
    pub shoot_bias: f64,
    pub pass_bias: f64,
    pub dribble_bias: f64,
    pub open_space_bias: f64,
    pub offensive_mindedness: f64,
    pub defensive_mindedness: f64,
}

impl Default for AgentPreferences {
    fn default() -> Self {
        AgentPreferences {
            shoot_bias: 0.40,
            pass_bias: 0.55,
            dribble_bias: 0.45,
            open_space_bias: 0.84,
            offensive_mindedness: 0.50,
            defensive_mindedness: 0.50,
        }
    }
}

/// A live one-two / wall-pass commitment carried by the player who just played the
/// "give" pass. It marks them as the runner in a give-and-go: having laid the ball
/// off to `wall_partner`, they immediately burst forward into `return_target` (the
/// onside space past the man they are combining around — the "go"), expecting a
/// first-time return into that run. Set when the carrier elects the one-two, read
/// by the runner's off-ball movement (to sprint the threaded run) and by the wall
/// partner (to play the first-time return led into the run). Cleared on receipt,
/// on a turnover, or after a short time-out if the return never comes.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OneTwoRun {
    /// The teammate the ball was laid off to (the "wall").
    pub wall_partner: usize,
    /// Match clock when the give pass was played; drives the time-out.
    pub launch_clock_seconds: f64,
    /// The onside space ahead/past the man-to-beat that the runner attacks.
    pub return_target: Vec2,
}

/// A committed pace-dribble that beats a defender by SPLITTING them: the ball is knocked past on
/// one side (`push_target`) while the attacker's body runs the OTHER side around the defender and
/// re-collects behind at `recollect_point`. Like [`OneTwoRun`], it is a short-lived commitment that
/// makes the carrier sprint to recover the knocked ball instead of treating it as a turnover.
/// Cleared on re-collection, a turnover, the ball leaving play, or a time-out. Gated by
/// `DD_SOCCER_DISABLE_RUNAROUND_DRIBBLE`.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunaroundDribble {
    /// The defender being run around.
    pub defender: usize,
    /// Match clock when the ball was knocked past; drives the time-out.
    pub launch_clock_seconds: f64,
    /// Where the ball was knocked (just past the defender, on the ball side).
    pub push_target: Vec2,
    /// The rendezvous behind the defender where the attacker re-collects the ball.
    pub recollect_point: Vec2,
    /// `true` once the knock has been played and the attacker is sprinting to re-collect.
    #[serde(default)]
    pub knocked: bool,
}

/// What a carrier holding the ball is *signalling* its teammates to do while it waits for
/// the play to set up (see [`HoldForSupport`]). Both are a delay-before-pass: the carrier
/// keeps the ball calmly and summons support rather than forcing a poor outlet.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SupportSummonKind {
    /// "Push forward into space" — summoned teammates make an advanced/in-behind run so a
    /// forward outlet opens up.
    MoveForward,
    /// "Take up your position" — summoned teammates step into an open passing lane / their
    /// structural slot so a clean pass becomes available.
    AssumePosition,
}

/// Active "hold-and-summon support" commitment on the carrier: it has elected to delay the
/// release, keep the ball (shield/slow), and signal `summoned` teammates to improve their
/// position (`kind`) before passing. Cleared on a pass / turnover, by the bounded max-hold
/// in the gate, or by the safety TTL — see
/// [`SoccerMatch::maintain_hold_for_support_commitments`]. Mirrored onto the snapshot so the
/// summoned teammates can read the signal from their own decision.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HoldForSupport {
    /// What the carrier is asking its teammates to do.
    pub kind: SupportSummonKind,
    /// Match clock when the wait began; drives the bounded max-hold and the safety TTL. NOT
    /// reset while the carrier keeps re-electing the same wait, so the pause is genuinely
    /// bounded rather than able to renew forever.
    pub launch_clock_seconds: f64,
    /// The teammates being signalled (the best-placed candidates, capped to a few).
    pub summoned: Vec<usize>,
}

/// A viable "hold-and-summon support" the carrier can elect right now: keep the ball and
/// signal `summoned` teammates to improve their position (`kind`) before releasing. `quality`
/// in `[0, 1]` scores how worthwhile the wait is and drives the appetite to elect it. Produced
/// by [`WorldSnapshot::hold_for_support_option_for`].
#[derive(Clone, Debug)]
pub struct HoldForSupportPlan {
    pub kind: SupportSummonKind,
    pub summoned: Vec<usize>,
    pub quality: f64,
}

/// A viable wall-pass the carrier can elect right now: lay the ball off to
/// `wall_partner` (the "give"), then burst into `return_target` (the "go").
/// `quality` in [0, 1] scores how good the combination is (lane to the wall, space
/// to run into, how beatable the man is) and drives the appetite to attempt it.
#[derive(Clone, Copy, Debug)]
pub struct WallPassPlan {
    pub wall_partner: usize,
    pub return_target: Vec2,
    pub quality: f64,
}

/// Short-horizon locomotion commitment: the movement momentum/inertia that stops a
/// player from issuing contradictory decisions tick-to-tick. A committed gait must be
/// held a minimum dwell before it may be downshifted (you carry a sprint's momentum;
/// you can't bleed it off instantly), and a committed travel heading must be held a
/// minimum dwell before it may be reversed (you can't cut back on yourself instantly).
/// Upshifting effort and gentle steering are always free — committing harder and
/// bending a run are decisive, instantaneous acts; only the *reversal* of a
/// commitment has inertia. See `commit_gait` / `commit_travel_heading`.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LocomotionCommitment {
    /// Seconds the player has continuously held the current gait effort tier, summed
    /// per movement step from `dt`. Reset when the tier changes; gates downshifts.
    pub gait_held_seconds: f64,
    /// Unit travel heading the player is committed to (`None` until first moving).
    pub heading: Option<Vec2>,
    /// Seconds since `heading` last changed sharply (a reversal), summed per step;
    /// gates how soon the player may reverse direction again.
    pub heading_held_seconds: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlayerAgent {
    pub id: usize,
    pub name: String,
    pub team: Team,
    pub role: PlayerRole,
    pub shirt: u8,
    pub home_position: Vec2,
    pub position: Vec2,
    pub velocity: Vec2,
    pub acceleration: Vec2,
    pub jerk: Vec2,
    #[serde(default)]
    pub movement_gait: MovementGait,
    /// Locomotion momentum/commitment state (anti-jitter); see [`LocomotionCommitment`].
    #[serde(default)]
    pub locomotion: LocomotionCommitment,
    pub position_history: VecDeque<Vec2>,
    #[serde(default)]
    pub receive_facing: FacingBucket,
    #[serde(default)]
    pub action_facing: FacingBucket,
    /// Continuous body heading about the z axis (radians), rate-limited by
    /// biomechanics. Drives smooth facing and the dizziness model.
    #[serde(default)]
    pub facing_yaw: f64,
    /// Current angular velocity (rad/s) about the z axis. A decision is instant,
    /// but the body's turn rate can only change within an angular-acceleration
    /// limit each step — it cannot start or stop spinning instantaneously.
    #[serde(default)]
    pub yaw_rate: f64,
    /// Accumulated vestibular disorientation from rapid repeated turning; decays
    /// with rest, degrades control/perception while high.
    #[serde(default)]
    pub dizziness: f64,
    /// Short-burst (anaerobic) energy load: 0 = fresh, 1 = gassed. Spent quickly
    /// by sprints/accelerations/duels and recovered over ~tens of seconds —
    /// distinct from `fatigue`, which is the slow aerobic drain over the match.
    #[serde(default)]
    pub anaerobic_load: f64,
    /// Metabolic locomotion cost spent THIS tick (J) — the di Prampero running model
    /// (run + accel power × dt) the W′ update already uses, surfaced for the learning
    /// layer. Read only by the wasted-energy reward (`remember_recent_learning_transitions`);
    /// it never feeds a decision, so it leaves the simulation trajectory byte-identical.
    #[serde(default)]
    pub last_tick_locomotion_joules: f64,
    /// Consecutive seconds the player has been holding a flat-out sprint (≥
    /// `SUSTAINED_EFFORT_SPEED_FRACTION` of top speed); reset the instant pace drops below it.
    /// Maintained only while the sustained-effort-no-outcome penalty is enabled; otherwise it
    /// stays 0 and is read by nothing, so the simulation trajectory is byte-identical.
    #[serde(default)]
    pub sustained_sprint_seconds: f64,
    /// Distance (yards) covered within the current sustained flat-out sprint; companion to
    /// `sustained_sprint_seconds` (reset together). See its note for the gating contract.
    #[serde(default)]
    pub sustained_sprint_distance_yards: f64,
    #[serde(default)]
    pub incoming_ball: Option<IncomingBallContext>,
    pub skills: SkillProfile,
    pub fatigue: f64,
    pub controller_slot: Option<usize>,
    pub preferences: AgentPreferences,
    pub last_decision: Option<AgentDecisionTrace>,
    /// Transient: the behaviour-policy probability of the action the stochastic
    /// top-k selector promoted this tick (renormalised rank weight), or `None`
    /// when the deterministic argmax was used. Reset at the start of each
    /// decision and folded into [`AgentDecisionTrace::behavior_policy_probability`]
    /// by `decision_trace`. Never serialized — purely a within-tick relay so the
    /// learning layer can read the true behaviour probability. Inert (always
    /// `None`) unless `DD_SOCCER_ENABLE_STOCHASTIC_POLICY_TOPK` is on.
    #[serde(skip)]
    pub pending_policy_behavior_probability: Option<f64>,
    /// The player's belief-grounded confidence in its current action, in `[0.2, 1.0]`
    /// (the engine sees truth; this is the player's read). Drives decisiveness (move
    /// speed) and proaction-vs-reaction. Recomputed each decision in `run_time_step`.
    #[serde(default = "default_unit_confidence")]
    pub decision_confidence: f64,
    /// Active one-two commitment after this player has played the "give" pass — see
    /// [`OneTwoRun`]. `None` when not engaged in a give-and-go.
    #[serde(default)]
    pub one_two: Option<OneTwoRun>,
    /// Active run-around-dribble commitment after this player has knocked the ball past a
    /// committed defender — see [`RunaroundDribble`]. `None` when not running one.
    #[serde(default)]
    pub runaround: Option<RunaroundDribble>,
    /// Active "hold-and-summon support" commitment: this player is on the ball and has
    /// elected to delay the pass while signalling teammates to get into a better position —
    /// see [`HoldForSupport`]. `None` when not waiting on support.
    #[serde(default)]
    pub hold_for_support: Option<HoldForSupport>,
    /// Seconds remaining flat on the ground after a MISSED slide tackle. While `> 0`
    /// the player is prone: it makes no decision, holds its position, and cannot
    /// challenge — so the attacker carries the ball past it unopposed. Drawn uniformly
    /// from `[SLIDE_TACKLE_RECOVERY_MIN_SECONDS, SLIDE_TACKLE_RECOVERY_MAX_SECONDS]`
    /// on the miss and bled down by `dt` each live tick. `0.0` = on its feet.
    #[serde(default)]
    pub slide_recovery_seconds: f64,
    /// Ticks at which this player last *committed a changed deliberative decision* (most recent
    /// last, capped at the two the refractory gate needs). Drives
    /// [`PlayerAgent::apply_decision_refractory`]. Empty while the refractory is disabled, so it
    /// is skipped from serialization and leaves snapshots byte-identical.
    #[serde(default, skip_serializing_if = "VecDeque::is_empty")]
    pub decision_commit_ticks: VecDeque<u64>,
    /// The intent emitted last tick, replayed to continue a held decision when a change is
    /// refractory-blocked. `None` until the first decision / while the refractory is disabled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_intent: Option<PlayerIntent>,
}

fn soccer_pressured_contested_pass_damp_enabled() -> bool {
    use std::sync::OnceLock;
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("DD_SOCCER_DISABLE_PRESSURED_PASS_DAMP").is_err())
}

fn mpc_reselect_candidate_label(label: &str) -> bool {
    matches!(
        label,
        "shoot"
            | "wall-pass"
            | "corner-flag-cross"
            | "vertical-attack"
            | "vacate-space"
            | "surprise-pass"
            | "flick-on"
            | "turnover-burst"
            | "flank-low-cross"
            | "flank-high-cross"
            | "aerial-pass"
            | "aerial-pass1"
            | "pass"
            | "pass1"
            | "first-time-pass"
            | "killer-pass"
            | "switch-play"
            | "recycle-reset"
            | "first-time-shot"
            | "first-time-header"
            | "dribble"
            | OPEN_PASS_LANE_ACTION_LABEL
            | ROUND_GOALKEEPER_ACTION_LABEL
            | "carry-forward"
            | "carry-out-left"
            | "carry-out-right"
            | "left-cut"
            | "right-cut"
            | "nutmeg"
            | "xavi-turn"
            | "open-passing-lane"
            | "round-the-keeper"
            | "fake-left-cut-right"
            | "fake-right-cut-left"
            | "protect-ball"
            | "side-step"
            | "hold-up-flank"
            | "run-in-behind"
            | "shot-creation-run"
            | "overlap-run"
            | "support-push-up"
            | "support-roam"
            | "support-shape"
    )
}

fn mpc_execution_skill_for_label(player: &PlayerAgent, label: &str) -> f64 {
    match label {
        "corner-flag-cross" | "flank-low-cross" | "flank-high-cross" => {
            ability01(
                player
                    .skills
                    .crossing_left
                    .max(player.skills.crossing_right),
            ) * 0.58
                + ability01(player.skills.passing_completion_rate) * 0.28
                + ability01(player.skills.vision) * 0.14
        }
        "surprise-pass" | "flick-on" => {
            ability01(player.skills.flair_passing) * 0.42
                + ability01(player.skills.first_touch) * 0.34
                + ability01(player.skills.passing_completion_rate) * 0.24
        }
        "wall-pass" | "pass" | "pass1" | "first-time-pass" | "killer-pass" | "switch-play"
        | "recycle-reset" | "aerial-pass" | "aerial-pass1" => {
            ability01(player.skills.passing_completion_rate) * 0.56
                + ability01(player.skills.vision) * 0.28
                + ability01(player.skills.first_touch) * 0.16
        }
        "shoot" | "first-time-shot" | "first-time-header" => {
            ability01(player.skills.shooting) * 0.42
                + ability01(
                    player
                        .skills
                        .right_foot_shot_power
                        .max(player.skills.left_foot_shot_power),
                ) * 0.32
                + ability01(player.skills.first_touch) * 0.26
        }
        _ => {
            ability01(player.skills.acceleration) * 0.42
                + ability01(player.skills.top_speed) * 0.34
                + ability01(player.skills.dribbling) * 0.24
        }
    }
    .clamp(0.0, 1.0)
}

const SHORT_UPFIELD_GROUND_PASS_MIN_YARDS: f64 = 5.47;
const SHORT_UPFIELD_GROUND_PASS_IDEAL_MAX_YARDS: f64 = 8.75;
const SHORT_UPFIELD_GROUND_PASS_TAPER_YARDS: f64 = 8.0;

fn short_upfield_ground_pass_fit(observation: &SoccerPomdpObservation) -> f64 {
    if observation.visible_forward_pass_options == 0 {
        return 0.0;
    }
    let distance = observation.nearest_forward_teammate_distance_yards;
    if !distance.is_finite() || distance <= 0.0 {
        return 0.0;
    }
    let distance_fit = if distance < SHORT_UPFIELD_GROUND_PASS_MIN_YARDS {
        (distance / SHORT_UPFIELD_GROUND_PASS_MIN_YARDS).clamp(0.0, 1.0) * 0.82
    } else if distance <= SHORT_UPFIELD_GROUND_PASS_IDEAL_MAX_YARDS {
        1.0
    } else {
        (1.0 - (distance - SHORT_UPFIELD_GROUND_PASS_IDEAL_MAX_YARDS)
            / SHORT_UPFIELD_GROUND_PASS_TAPER_YARDS)
            .clamp(0.0, 1.0)
    };
    let outlet_fit = (observation
        .best_forward_pass_receiver_openness
        .clamp(0.0, 1.0)
        * 0.38
        + observation.expected_pass_completion.clamp(0.0, 1.0) * 0.30
        + observation.floor_pass_lane_score.clamp(0.0, 1.0) * 0.32)
        .clamp(0.0, 1.0);
    (distance_fit * (0.42 + outlet_fit * 0.58)).clamp(0.0, 1.0)
}

#[derive(Clone, Debug, Default)]
struct MpcExecutionEstimate {
    ball_urgency: f64,
    touch_window_seconds: f64,
    body_mechanics_fit: f64,
    execution_probability: f64,
    pass_completion_probability: f64,
    dribble_success_probability: f64,
    dribble_left_success_probability: f64,
    dribble_right_success_probability: f64,
    recommended_speed_yps: f64,
    recommended_angle_degrees: f64,
    recommended_curve_bend_yards: f64,
    recommended_spin_rps: f64,
    recommended_curve: &'static str,
    horizon_seconds: f64,
    reselect_reason: &'static str,
    /// MPC shot foot choice: which foot the strike is taken with ("right"/"left"), and
    /// that foot's power rating. Empty / `0.0` for non-shot actions or when the foot-choice
    /// MPC is disabled (`DD_SOCCER_DISABLE_SHOT_FOOT_MPC`).
    recommended_foot: &'static str,
    recommended_foot_power: f64,
}

// Shot-trigger discipline + MPC foot-choice thresholds are centralized in the runtime
// config: `tunables().shooting.{shot_trigger_long_range_yards, shot_trigger_long_range_min_value,
// shot_trigger_volunteer_floor, shot_foot_weak_foot_execution_damp}` (see tunables.rs).

/// Weight on the ball-vs-body offset (vs the target sweep) in the foot-choice geometry score.
const SHOT_FOOT_BALL_OFFSET_WEIGHT: f64 = 0.7;
/// Weight on the strike's lateral sweep in the foot-choice geometry score.
const SHOT_FOOT_TARGET_SWEEP_WEIGHT: f64 = 0.3;
/// Max skill rating, used to normalize the per-foot power asymmetry to `[-1, 1]`.
const SHOT_FOOT_MAX_SKILL: f64 = 10.0;
/// How strongly the normalized power asymmetry tilts the geometry-vs-power decision.
const SHOT_FOOT_POWER_BIAS_WEIGHT: f64 = 2.0;
/// Power margin (rating points) below the strong foot that flags the chosen foot as "weak".
const SHOT_FOOT_WEAK_MARGIN: f64 = 1.0;

/// MPC shot foot choice: pick which foot to strike with from the ball-vs-body geometry
/// and the per-foot power. The ball sitting on a given side of the body, and a target
/// swept toward that side, both favor the same-side foot; the choice is then weighted by
/// each foot's power so a player swivels onto a stronger foot when geometry is neutral.
/// Returns `(foot, power, weak_foot)` where `weak_foot` is true when forced onto the
/// weaker foot (used to damp execution probability).
fn shot_foot_choice(
    skills: &SkillProfile,
    ball_lateral_offset: f64,
    target_lateral_sweep: f64,
) -> (&'static str, f64, bool) {
    let right_power = skills.right_foot_shot_power.max(0.0);
    let left_power = skills.left_foot_shot_power.max(0.0);
    // Geometry score: positive ⇒ right foot is naturally placed, negative ⇒ left. The ball
    // on the right of the body (+x offset) and a strike swept to the right both favor right.
    let geometry = ball_lateral_offset * SHOT_FOOT_BALL_OFFSET_WEIGHT
        + target_lateral_sweep * SHOT_FOOT_TARGET_SWEEP_WEIGHT;
    // Power asymmetry (normalized): how much stronger the right foot is than the left.
    let power_bias = (right_power - left_power) / SHOT_FOOT_MAX_SKILL;
    // Combine: geometry dominates when pronounced, power breaks a near-neutral stance.
    let decision = geometry + power_bias * SHOT_FOOT_POWER_BIAS_WEIGHT;
    let strong_foot_power = right_power.max(left_power);
    let (foot, power) = if decision >= 0.0 {
        ("right", right_power)
    } else {
        ("left", left_power)
    };
    // Weak foot = the chosen foot is materially weaker than the player's strong foot.
    let weak_foot = power + SHOT_FOOT_WEAK_MARGIN < strong_foot_power;
    (foot, power, weak_foot)
}

#[derive(Clone, Copy, Debug, Default)]
struct MpcBallExecutionContext {
    urgency: f64,
    touch_window_seconds: f64,
    body_mechanics_fit: f64,
}

fn mpc_ball_execution_context_for_player(
    snapshot: &WorldSnapshot,
    player: &PlayerAgent,
    current: Vec2,
    movement_execution_confidence: f64,
) -> MpcBallExecutionContext {
    let holder = snapshot.ball.holder;
    let ball_altitude = finite_metric(snapshot.ball.altitude_yards).max(0.0);
    let ball_speed = snapshot.ball.velocity.len();
    let ball_distance = current.distance(snapshot.ball.position);
    let top_speed = player_top_speed_yps(player.role, &player.skills)
        * MovementGait::Sprint.speed_multiplier()
        * fatigue_speed_factor(player.skills.stamina, player.fatigue);
    let fallback_time_to_ball = ball_distance / top_speed.max(0.85);
    let loose_race = if holder.is_none() {
        loose_ball_race_context_for_snapshot(snapshot, player.id)
    } else {
        LooseBallRaceContext::default()
    };
    let player_time_to_ball = if holder == Some(player.id) {
        0.0
    } else if loose_race.loose_ball {
        loose_race.player_time_to_ball_seconds
    } else {
        fallback_time_to_ball
    }
    .clamp(0.0, 8.0);
    let nearest_opponent_distance = snapshot
        .players
        .iter()
        .filter(|opponent| opponent.team == player.team.other())
        .map(|opponent| opponent.position.distance(current))
        .fold(f64::INFINITY, f64::min);
    let pressure_urgency = pressure_from_nearest_distance(nearest_opponent_distance);
    let altitude_urgency = if ball_altitude <= BALL_ROLLING_ALTITUDE_YARDS {
        0.0
    } else if ball_altitude <= AERIAL_HEADER_MAX_ALTITUDE_YARDS {
        (0.30 + ball_altitude / AERIAL_HEADER_MAX_ALTITUDE_YARDS * 0.70).clamp(0.0, 1.0)
    } else {
        (0.54 - (ball_altitude - AERIAL_HEADER_MAX_ALTITUDE_YARDS) / 8.0 * 0.34).clamp(0.14, 0.54)
    };
    let speed_urgency =
        (ball_speed / 24.0).clamp(0.0, 1.0) * if ball_distance <= 3.0 { 1.0 } else { 0.65 };
    let race_urgency = if loose_race.loose_ball {
        let time_pressure = (1.0 - player_time_to_ball / 1.8).clamp(0.0, 1.0);
        let contest_pressure = if loose_race.fifty_fifty {
            0.35
        } else if loose_race.team_time_advantage_seconds < 0.0 {
            0.25
        } else {
            0.0
        };
        (time_pressure * 0.55 + contest_pressure).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let first_touch_tool = (ability01(player.skills.first_touch) * 0.42
        + ability01(player.skills.strength) * 0.18
        + ability01(player.skills.dribbling) * 0.18
        + ability01(player.skills.flair_passing) * 0.12
        + movement_execution_confidence.clamp(0.0, 1.0) * 0.10)
        .clamp(0.0, 1.0);
    let aerial_tool = aerial_duel_skill_from_agent(player);
    let touch_tool = if ball_altitude >= AERIAL_HEADER_MIN_ALTITUDE_YARDS {
        (first_touch_tool * 0.52 + aerial_tool * 0.48).clamp(0.0, 1.0)
    } else {
        first_touch_tool
    };
    let base_window = if ball_altitude <= BALL_ROLLING_ALTITUDE_YARDS {
        1.35
    } else if ball_altitude <= AERIAL_HEADER_MIN_ALTITUDE_YARDS {
        0.46 + touch_tool * 0.46
    } else if ball_altitude <= AERIAL_HEADER_MAX_ALTITUDE_YARDS {
        0.34 + touch_tool * 0.42 + aerial_tool * 0.14
    } else {
        0.68 + aerial_tool * 0.50
    };
    let travel_loss = if holder == Some(player.id) {
        0.0
    } else {
        (player_time_to_ball / 2.4).clamp(0.0, 1.0) * 0.32
    };
    let speed_loss = if ball_altitude > BALL_ROLLING_ALTITUDE_YARDS || holder.is_none() {
        (ball_speed / 30.0).clamp(0.0, 1.0) * 0.24
    } else {
        (ball_speed / 35.0).clamp(0.0, 1.0) * 0.10
    };
    let touch_window_seconds =
        (base_window - travel_loss - speed_loss - pressure_urgency * 0.12).clamp(0.06, 1.60);
    let window_urgency = (1.0 - touch_window_seconds / 1.05).clamp(0.0, 1.0);
    let urgency = (altitude_urgency * 0.48
        + speed_urgency * 0.20
        + race_urgency * 0.24
        + pressure_urgency * 0.08)
        .clamp(0.0, 1.0)
        .max(window_urgency);
    let mechanics_baseline = (0.70 + touch_tool * 0.30).clamp(0.70, 1.0);
    let urgency_damp = (1.0 - urgency * (0.20 + (1.0 - touch_tool) * 0.70)).clamp(0.20, 1.0);
    let window_fit = (touch_window_seconds / 0.75).clamp(0.25, 1.0);
    let body_mechanics_fit = if urgency <= 0.05 {
        1.0
    } else {
        (mechanics_baseline * urgency_damp)
            .min(window_fit)
            .clamp(0.20, 1.0)
    };
    MpcBallExecutionContext {
        urgency,
        touch_window_seconds,
        body_mechanics_fit,
    }
}

/// Whether the **full per-action MPC coverage** is live this process. OFF unless
/// `DD_SOCCER_ENABLE_FULL_ACTION_MPC_COVERAGE` is set ⇒ off, the ball-contact
/// actions that lack a bespoke striking/winning execution model (clearance,
/// route-one, control-touch, tackle, slide-tackle) fall through to the plain
/// movement reach-model exactly as before, so the trajectory is byte-identical.
/// On, each of those actions gets its own execution model in
/// [`mpc_execution_estimate_for_action`] — a clearance/route-one strike, a
/// control-touch cushion, or a (slide-)tackle ball-win — so every `SoccerAction`
/// carries an explicit "how to execute" definition rather than being treated as a
/// bare move-to-a-point.
fn full_action_mpc_coverage_enabled() -> bool {
    #[cfg(test)]
    {
        std::env::var("DD_SOCCER_ENABLE_FULL_ACTION_MPC_COVERAGE")
            .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
            .unwrap_or(false)
    }
    #[cfg(not(test))]
    {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var("DD_SOCCER_ENABLE_FULL_ACTION_MPC_COVERAGE")
                .map(|v| {
                    matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on")
                })
                .unwrap_or(false)
        })
    }
}

fn mpc_execution_estimate_for_action(
    snapshot: &WorldSnapshot,
    player: &PlayerAgent,
    action_target: &Option<AgentActionTargetTrace>,
    action_label: &str,
    current: Vec2,
    dt: f64,
    target_delta_yards: f64,
    reachable_delta_yards: f64,
    movement_execution_confidence: f64,
) -> MpcExecutionEstimate {
    let label = normalize_soccer_action_label(action_label);
    let mut estimate = MpcExecutionEstimate {
        execution_probability: movement_execution_confidence,
        horizon_seconds: (dt * 2.0).clamp(2.0 / 15.0, 0.24),
        ..MpcExecutionEstimate::default()
    };
    let ball_context = mpc_ball_execution_context_for_player(
        snapshot,
        player,
        current,
        movement_execution_confidence,
    );
    estimate.ball_urgency = ball_context.urgency;
    estimate.touch_window_seconds = ball_context.touch_window_seconds;
    estimate.body_mechanics_fit = ball_context.body_mechanics_fit;
    if let Some(target) = action_target.as_ref().and_then(|target| target.point) {
        let to_target = target - current;
        if to_target.len() > 1e-6 {
            estimate.recommended_angle_degrees = to_target.y.atan2(to_target.x).to_degrees();
        }
    }

    let pass_flight = pass_like_action_flight(label);
    if let Some(flight) = pass_flight {
        let target_point = action_target
            .as_ref()
            .and_then(|target| target.point)
            .or_else(|| {
                action_target
                    .as_ref()
                    .and_then(|target| target.player_id)
                    .and_then(|id| snapshot.player_position(id))
            });
        let Some(target_point) = target_point else {
            estimate.execution_probability = 0.0;
            estimate.pass_completion_probability = 0.0;
            estimate.reselect_reason = "mpc-pass-has-no-target";
            return estimate;
        };
        let distance = current.distance(target_point);
        let is_cross = pass_would_be_cross(
            current,
            target_point,
            player.team,
            snapshot.field_width,
            snapshot.field_length,
        );
        let power = match label {
            "wall-pass" => WALL_PASS_GIVE_POWER + 0.20 * ability01(player.skills.passing),
            "killer-pass" => {
                0.66 + 0.24
                    * ability01(player.skills.passing_completion_rate)
                        .max(ability01(player.skills.vision))
            }
            "switch-play" => {
                0.62 + 0.24
                    * ability01(player.skills.passing_completion_rate)
                        .max(ability01(player.skills.flair_passing))
            }
            "recycle-reset" => 0.46 + 0.22 * ability01(player.skills.passing_completion_rate),
            "flank-low-cross" | "corner-flag-cross" => {
                0.60 + 0.28
                    * ability01(
                        player
                            .skills
                            .crossing_left
                            .max(player.skills.crossing_right),
                    )
                    .max(ability01(player.skills.passing_completion_rate))
            }
            "flank-high-cross" | "flick-on" => {
                0.64 + 0.28
                    * ability01(
                        player
                            .skills
                            .crossing_left
                            .max(player.skills.crossing_right),
                    )
                    .max(ability01(player.skills.passing_completion_rate))
            }
            "surprise-pass" => {
                0.50 + 0.22
                    * ability01(player.skills.passing_completion_rate)
                        .max(ability01(player.skills.flair_passing))
            }
            _ => 0.58 + 0.32 * ability01(player.skills.passing_completion_rate),
        }
        .clamp(0.25, 1.0);
        let speed = pass_speed_yps_from_power(power, flight, is_cross, &player.skills);
        let lane_radius = if flight.is_aerial() { 1.25 } else { 1.8 };
        let (clear_now, clear_through_flight) = snapshot.pass_lane_clearance(
            current,
            target_point,
            player.team.other(),
            lane_radius,
            speed,
        );
        let lane_interception_risk = snapshot
            .pass_lane_interception_risk(
                current,
                target_point,
                player.team.other(),
                lane_radius,
                speed,
                PASS_LANE_DECISION_LOOKAHEAD_SECONDS,
            )
            .risk;
        let nearest_opponent_to_lane = snapshot
            .players
            .iter()
            .filter(|opponent| opponent.team == player.team.other())
            .map(|opponent| {
                let position = snapshot
                    .player_position(opponent.id)
                    .unwrap_or(opponent.position);
                segment_distance_to_point(current, target_point, position)
            })
            .fold(f64::INFINITY, f64::min);
        let base_lane_fit = if clear_now {
            if clear_through_flight {
                1.0
            } else {
                0.46
            }
        } else {
            0.08
        };
        let dynamic_lane_gate = (1.0
            - lane_interception_risk.clamp(0.0, 1.0) * PASS_LANE_DYNAMIC_RISK_GATE_STRENGTH)
            .clamp(0.12, 1.0);
        let lane_fit = base_lane_fit * dynamic_lane_gate;
        let receiver_fit = action_target
            .as_ref()
            .and_then(|target| target.player_id)
            .and_then(|id| snapshot.player_position(id).map(|position| (id, position)))
            .map(|(id, receiver_position)| {
                let receiver_pressure = snapshot
                    .players
                    .iter()
                    .filter(|opponent| opponent.team == player.team.other())
                    .map(|opponent| opponent.position.distance(receiver_position))
                    .fold(f64::INFINITY, f64::min);
                let moving_target_bonus = snapshot
                    .player_velocity(id)
                    .map(|velocity| {
                        (velocity.y * player.team.attack_dir())
                            .max(0.0)
                            .clamp(0.0, 7.0)
                            / 7.0
                    })
                    .unwrap_or(0.0);
                (0.50
                    + (receiver_pressure / 7.0).clamp(0.0, 1.0) * 0.32
                    + moving_target_bonus * 0.18)
                    .clamp(0.16, 1.0)
            })
            .unwrap_or(0.62);
        let speed_fit = (1.0 - ((distance / speed.max(1.0)) - 1.0).abs() / 3.4).clamp(0.35, 1.0);
        let skill = pass_execution_skill(&player.skills, flight, is_cross);
        let body_fit =
            ((0.36 + movement_execution_confidence * 0.40 + player.decision_confidence * 0.24)
                .clamp(0.0, 1.0)
                * ball_context.body_mechanics_fit)
                .clamp(0.0, 1.0);
        let lane_margin_fit = if nearest_opponent_to_lane.is_finite() {
            ((nearest_opponent_to_lane - lane_radius) / 4.0).clamp(0.0, 1.0)
        } else {
            1.0
        }
        .min((1.0 - lane_interception_risk * 0.58).clamp(0.18, 1.0));
        let pass_probability = (0.05
            + skill * 0.30
            + lane_fit * 0.30
            + receiver_fit * 0.15
            + speed_fit * 0.08
            + lane_margin_fit * 0.12)
            * body_fit;
        estimate.pass_completion_probability = pass_probability.clamp(0.0, 0.99);
        estimate.execution_probability = estimate.pass_completion_probability;
        estimate.recommended_speed_yps = speed;
        let curl_probability = pass_curl_probability_for_player(
            &player.skills,
            flight,
            is_cross,
            distance,
            1.0 - lane_margin_fit,
        );
        let curve = if curl_probability >= 0.38 && discretized_kick_curve_allowed(distance, speed) {
            if target_point.x >= snapshot.field_width * 0.5 {
                DiscretizedKickCurve::Right
            } else {
                DiscretizedKickCurve::Left
            }
        } else {
            DiscretizedKickCurve::None
        };
        estimate.recommended_curve = match curve {
            DiscretizedKickCurve::Left => "left",
            DiscretizedKickCurve::Right => "right",
            DiscretizedKickCurve::None => "none",
        };
        estimate.recommended_curve_bend_yards = if curve == DiscretizedKickCurve::None {
            0.0
        } else {
            (0.55 + distance / 24.0).clamp(0.55, 2.8)
                * (0.72 + ability01(player.skills.flair_passing) * 0.56)
        };
        estimate.recommended_spin_rps =
            (estimate.recommended_curve_bend_yards / distance.max(1.0) * speed.max(1.0) * 0.42)
                .clamp(0.0, 9.0);
        estimate.reselect_reason = if !clear_now {
            "mpc-pass-lane-blocked-now"
        } else if lane_interception_risk >= PASS_LANE_DYNAMIC_RISK_HIGH {
            "mpc-pass-lane-risk-next-two-seconds"
        } else if !clear_through_flight {
            "mpc-pass-cut-out-during-flight"
        } else if ball_context.urgency >= 0.50 && ball_context.body_mechanics_fit <= 0.60 {
            "mpc-ball-touch-window-closing"
        } else if estimate.pass_completion_probability
            < tunables()
                .decision_mpc
                .reselect_min_ball_execution_probability
        {
            "mpc-pass-low-completion"
        } else {
            ""
        };
        return estimate;
    }

    if matches!(label, "shoot" | "first-time-shot" | "first-time-header") {
        let observation = snapshot.observation_for(player.id);
        let first_time = matches!(label, "first-time-shot" | "first-time-header");
        // MPC shot foot choice (default-ON; kill-switch `DD_SOCCER_DISABLE_SHOT_FOOT_MPC`).
        // A header has no foot; otherwise pick the foot from ball-vs-body geometry and the
        // per-foot power, instead of blindly taking the stronger foot's power. Disabled ⇒
        // `None` ⇒ the legacy `.max()` power ⇒ byte-identical.
        let foot_choice = if shot_foot_mpc_enabled() && label != "first-time-header" {
            let attack_dir = player.team.attack_dir();
            let ball_lateral_offset = (snapshot.ball.position.x - current.x) * attack_dir;
            let target_lateral_sweep = (snapshot.field_width * 0.5 - current.x) * attack_dir;
            Some(shot_foot_choice(
                &player.skills,
                ball_lateral_offset,
                target_lateral_sweep,
            ))
        } else {
            None
        };
        let chosen_foot_power = foot_choice.map(|(_, power, _)| power).unwrap_or_else(|| {
            player
                .skills
                .right_foot_shot_power
                .max(player.skills.left_foot_shot_power)
        });
        let finish_skill = if label == "first-time-header" {
            aerial_duel_skill_from_agent(player)
        } else {
            (ability01(chosen_foot_power) * 0.62
                + ability01(player.skills.shooting) * 0.26
                + ability01(player.skills.first_touch) * 0.12)
                .clamp(0.0, 1.0)
        };
        let shot_power = if first_time {
            shot_power_for_finish_skill(finish_skill)
        } else {
            shot_power_for_skill(ability01(player.skills.shooting))
        };
        let shot_speed = shot_speed_yps_from_power(shot_power, &player.skills);
        let shot_mpc = shot_mpc_accuracy_estimate_for_snapshot(
            snapshot,
            player.id,
            player.team,
            &player.skills,
            player.fatigue,
            current,
            player.velocity,
            shot_speed,
            first_time,
            observation.perceived_pressure,
        );
        let frame_fit = (observation.shot_on_frame_probability.clamp(0.0, 1.0) * 0.64
            + shot_mpc.accuracy_probability * 0.36)
            .clamp(0.0, 1.0);
        let keeper_fit = (observation.shot_beat_goalkeeper_probability.clamp(0.0, 1.0) * 0.58
            + shot_mpc.goal_probability * 0.42)
            .clamp(0.0, 1.0);
        let block_fit = ((1.0 - observation.shot_block_probability.clamp(0.0, 1.0)) * 0.72
            + (1.0 - shot_mpc.block_probability) * 0.28)
            .clamp(0.0, 1.0);
        let angle_fit = (observation.opponent_goal_angle_degrees / 42.0).clamp(0.0, 1.0);
        let first_touch_fit = if first_time {
            observation.first_time_shot_score.clamp(0.0, 1.0)
        } else {
            0.72
        };
        let mechanics_fit = if first_time {
            ball_context.body_mechanics_fit
        } else {
            (0.72 + ball_context.body_mechanics_fit * 0.28).clamp(0.0, 1.0)
        };
        let body_fit = mechanics_fit * (0.76 + shot_mpc.qp_target_fit * 0.24).clamp(0.55, 1.0);
        estimate.execution_probability = ((0.04
            + frame_fit * 0.26
            + keeper_fit * 0.22
            + block_fit * 0.20
            + angle_fit * 0.08
            + finish_skill * 0.10
            + first_touch_fit * 0.10)
            * body_fit)
            .clamp(0.0, 0.99);
        // Record the MPC foot choice and damp execution when geometry forces the weaker
        // foot (a less reliable strike). Skipped when the foot-choice MPC is off (parity).
        if let Some((foot, power, weak_foot)) = foot_choice {
            estimate.recommended_foot = foot;
            estimate.recommended_foot_power = power;
            if weak_foot {
                estimate.execution_probability =
                    (estimate.execution_probability
                        * tunables().shooting.shot_foot_weak_foot_execution_damp)
                        .clamp(0.0, 0.99);
            }
        }
        estimate.recommended_speed_yps = shot_speed;
        let shot_target_delta = shot_mpc.target_point - current;
        if shot_target_delta.len() > 1e-6 {
            estimate.recommended_angle_degrees =
                shot_target_delta.y.atan2(shot_target_delta.x).to_degrees();
        }
        estimate.recommended_curve = if observation.shot_curl_probability >= 0.36 {
            if current.x >= snapshot.field_width * 0.5 {
                "left"
            } else {
                "right"
            }
        } else {
            "none"
        };
        estimate.recommended_curve_bend_yards = if estimate.recommended_curve == "none" {
            0.0
        } else {
            (0.45 + observation.yards_to_goal / 28.0).clamp(0.45, 2.2)
        };
        estimate.recommended_spin_rps = (estimate.recommended_curve_bend_yards
            / observation.yards_to_goal.max(1.0)
            * shot_speed.max(1.0)
            * 0.38)
            .clamp(0.0, 8.0);
        estimate.reselect_reason = if first_time
            && ball_context.urgency >= 0.50
            && ball_context.body_mechanics_fit <= 0.60
        {
            "mpc-ball-touch-window-closing"
        } else if first_time && observation.first_time_shot_score < 0.20 {
            "mpc-first-time-shot-low-contact"
        } else if shot_mpc.qp_target_fit < 0.22 {
            "mpc-shot-target-uncontrolled"
        } else if observation.shot_block_probability
            >= tunables().shooting.shot_block_bailout_max_probability
        {
            "mpc-shot-blocked"
        } else if estimate.execution_probability
            < tunables()
                .decision_mpc
                .reselect_min_ball_execution_probability
        {
            "mpc-shot-low-execution"
        } else {
            ""
        };
        return estimate;
    }

    if matches!(
        label,
        "dribble"
            | "carry-forward"
            | "vertical-attack"
            | "turnover-burst"
            | "carry-out-left"
            | "carry-out-right"
            | "side-step"
            | "left-cut"
            | "right-cut"
            | OPEN_PASS_LANE_ACTION_LABEL
            | ROUND_GOALKEEPER_ACTION_LABEL
            | "nutmeg"
            | "xavi-turn"
            | "open-passing-lane"
            | "round-the-keeper"
            | "fake-left-cut-right"
            | "fake-right-cut-left"
            | "protect-ball"
            | "hold-up-flank"
    ) {
        let target = action_target
            .as_ref()
            .and_then(|target| target.point)
            .unwrap_or(current);
        let open_pass_lane = label == OPEN_PASS_LANE_ACTION_LABEL;
        let nearest_opponent = snapshot
            .players
            .iter()
            .filter(|opponent| opponent.team == player.team.other())
            .min_by(|a, b| {
                a.position
                    .distance(current)
                    .total_cmp(&b.position.distance(current))
            });
        let nearest_distance = nearest_opponent
            .map(|opponent| opponent.position.distance(current))
            .unwrap_or(24.0);
        let target_clearance = snapshot
            .players
            .iter()
            .filter(|opponent| opponent.team == player.team.other())
            .map(|opponent| opponent.position.distance(target))
            .fold(f64::INFINITY, f64::min);
        let to_target = target - current;
        if open_pass_lane {
            let sprint_horizon = to_target.len()
                / (player_top_speed_yps(player.role, &player.skills)
                    * MovementGait::Sprint.speed_multiplier())
                .max(1.0);
            estimate.horizon_seconds = sprint_horizon.clamp(estimate.horizon_seconds, 0.82);
        }
        let forward_dir = Vec2::new(0.0, player.team.attack_dir());
        let left_dir = Vec2::new(-player.team.attack_dir(), 0.0);
        let right_dir = Vec2::new(player.team.attack_dir(), 0.0);
        let normalized_target = to_target.normalized();
        let lateral_left_fit = normalized_target.dot(left_dir).clamp(0.0, 1.0);
        let lateral_right_fit = normalized_target.dot(right_dir).clamp(0.0, 1.0);
        let opponent_lateral = nearest_opponent
            .map(|opponent| (opponent.position - current).dot(left_dir))
            .unwrap_or(0.0);
        let space_left = if opponent_lateral > 0.0 { 0.34 } else { 1.0 };
        let space_right = if opponent_lateral < 0.0 { 0.34 } else { 1.0 };
        let dribble_skill = (ability01(player.skills.dribbling) * 0.42
            + ability01(player.skills.acceleration) * 0.30
            + ability01(player.skills.first_touch) * 0.18
            + ability01(player.skills.flair_passing) * 0.10)
            .clamp(0.0, 1.0);
        let nearest_closing_rate = nearest_opponent
            .map(|opponent| {
                let opponent_position = snapshot
                    .player_position(opponent.id)
                    .unwrap_or(opponent.position);
                let opponent_velocity = snapshot
                    .player_velocity(opponent.id)
                    .unwrap_or(opponent.velocity);
                dribble_closing_rate_toward_point(opponent_position, opponent_velocity, current)
            })
            .unwrap_or(0.0);
        let pressure = dribble_spacing_pressure_for_state(
            player.role,
            pass_origin_in_own_half(player.team, current, snapshot.field_length),
            nearest_distance,
            nearest_closing_rate,
        )
        .max(pressure_from_nearest_distance(nearest_distance));
        let dribble_mpc = dribble_mpc_control_estimate_for_snapshot(
            snapshot,
            player.id,
            player.team,
            player.role,
            &player.skills,
            player.fatigue,
            current,
            player.velocity,
            target,
            action_target
                .as_ref()
                .and_then(|target| target.dribble_touch)
                .map(|touch| touch.distance_yards)
                .unwrap_or_else(|| to_target.len()),
        );
        let target_space_fit = ((target_clearance / 5.5).clamp(0.0, 1.0) * 0.62
            + dribble_mpc.control_probability * 0.38)
            .clamp(0.0, 1.0);
        let open_lane_situation = if open_pass_lane {
            Some(open_pass_lane_situation_for_target(
                snapshot,
                player,
                &snapshot.observation_for(player.id),
                current,
                target,
                0.0,
            ))
        } else {
            None
        };
        let touch_fit = action_target
            .as_ref()
            .and_then(|target| target.dribble_touch)
            .map(|touch| {
                if open_pass_lane && touch.distance_yards > OPEN_PASS_LANE_BASE_MAX_DRIBBLE_YARDS {
                    let extra = ((touch.distance_yards - OPEN_PASS_LANE_BASE_MAX_DRIBBLE_YARDS)
                        / (OPEN_PASS_LANE_MAX_DRIBBLE_YARDS
                            - OPEN_PASS_LANE_BASE_MAX_DRIBBLE_YARDS)
                            .max(1e-6))
                    .clamp(0.0, 1.0);
                    let situation_fit = open_lane_situation
                        .map(|situation| 0.70 + situation.extension * 0.34)
                        .unwrap_or(0.70);
                    ((0.90 - extra * 0.18) * situation_fit).clamp(0.34, 0.98)
                } else {
                    (1.0 - (touch.distance_yards - 2.2).abs() / 3.4).clamp(0.25, 1.0)
                }
            })
            .unwrap_or(0.72);
        let qp_control_gate = (0.70 + dribble_mpc.qp_accel_fit * 0.30).clamp(0.50, 1.0);
        let forward_fit = normalized_target.dot(forward_dir).clamp(-0.4, 1.0);
        let left_prob = (0.16
            + dribble_skill * 0.34
            + target_space_fit * 0.22
            + touch_fit * 0.12
            + lateral_left_fit * 0.12
            + space_left * 0.16
            - pressure * 0.24)
            .clamp(0.02, 0.96);
        let right_prob = (0.16
            + dribble_skill * 0.34
            + target_space_fit * 0.22
            + touch_fit * 0.12
            + lateral_right_fit * 0.12
            + space_right * 0.16
            - pressure * 0.24)
            .clamp(0.02, 0.96);
        let direct_prob = (0.14
            + dribble_skill * 0.36
            + target_space_fit * 0.24
            + touch_fit * 0.12
            + forward_fit.max(0.0) * 0.14
            - pressure * 0.28)
            .clamp(0.02, 0.96);
        let selected = match label {
            "carry-out-left" | "left-cut" => left_prob,
            "carry-out-right" | "right-cut" => right_prob,
            "side-step" | "fake-left-cut-right" | "fake-right-cut-left" => {
                left_prob.max(right_prob)
            }
            "xavi-turn" => {
                (0.28 + dribble_skill * 0.32 + pressure * 0.26 + target_space_fit * 0.12)
                    .clamp(0.12, 0.96)
            }
            "open-passing-lane" => {
                // A short carry into a clear, opponent-free spot (chosen to have space) — inherently
                // feasible; scales with control + the space at the spot, lightly damped by pressure.
                (0.46 + dribble_skill * 0.26 + target_space_fit * 0.18 - pressure * 0.12)
                    .clamp(0.18, 0.97)
            }
            ROUND_GOALKEEPER_ACTION_LABEL => {
                // A short carry to a closer spot with a clear strike past the keeper — feasible;
                // scales with control + the space at the spot, lightly damped by pressure. (Ours'
                // tuned estimate, folded onto the unified round-goalkeeper label during the merge.)
                (0.48 + dribble_skill * 0.24 + target_space_fit * 0.18 - pressure * 0.10)
                    .clamp(0.18, 0.97)
            }
            "protect-ball" => (0.34 + dribble_skill * 0.28 + pressure * 0.22).clamp(0.10, 0.94),
            OPEN_PASS_LANE_ACTION_LABEL => {
                let situational_boost = open_lane_situation
                    .map(|situation| situation.pressure.max(situation.tracking) * 0.12)
                    .unwrap_or(0.0);
                direct_prob.max(left_prob).max(right_prob) * (0.88 + situational_boost)
            }
            _ => direct_prob.max(if left_prob > right_prob {
                left_prob * 0.92
            } else {
                right_prob * 0.92
            }),
        };
        estimate.dribble_left_success_probability = left_prob;
        estimate.dribble_right_success_probability = right_prob;
        estimate.dribble_success_probability = (selected
            * (0.62 + movement_execution_confidence * 0.38)
            * ball_context.body_mechanics_fit
            * qp_control_gate)
            .clamp(0.0, 0.99);
        estimate.execution_probability = estimate.dribble_success_probability;
        estimate.recommended_speed_yps = (to_target.len() / estimate.horizon_seconds.max(1e-6))
            .clamp(
                0.0,
                player_top_speed_yps(player.role, &player.skills) * 1.16,
            );
        estimate.recommended_curve = if left_prob > right_prob + 0.08 {
            "left"
        } else if right_prob > left_prob + 0.08 {
            "right"
        } else {
            "none"
        };
        estimate.recommended_curve_bend_yards = 0.0;
        estimate.recommended_spin_rps = 0.0;
        estimate.reselect_reason =
            if ball_context.urgency >= 0.50 && ball_context.body_mechanics_fit <= 0.60 {
                "mpc-ball-touch-window-closing"
            } else if dribble_mpc.control_probability < 0.18 {
                "mpc-dribble-target-uncontrolled"
            } else if estimate.dribble_success_probability
                < tunables()
                    .decision_mpc
                    .reselect_min_ball_execution_probability
            {
                "mpc-dribble-low-success"
            } else {
                ""
            };
        return estimate;
    }

    // Ball-contact actions that don't match the pass / shot / dribble families above
    // (clearance, route-one, control-touch, tackle, slide-tackle) would otherwise
    // fall through to the plain movement reach-model below — a "move to a point"
    // definition that ignores the strike / cushion / challenge the action actually
    // is. Give each its own execution model so every action carries an explicit
    // "how to execute" definition. Gated (default OFF ⇒ byte-identical): when off
    // these still use the movement default exactly as before.
    if full_action_mpc_coverage_enabled() {
        let contact_fit = (movement_execution_confidence * 0.5
            + ball_context.body_mechanics_fit * 0.5)
            .clamp(0.0, 1.0);
        match label {
            "clearance" | "route-one" => {
                // A clearing strike: get a foot through the ball with enough power to
                // send it to the target zone. Execution scales with leg power +
                // technique, gated by whether the body can reach/strike it cleanly.
                let strike = (ability01(player.skills.strength) * 0.40
                    + ability01(player.skills.passing) * 0.30
                    + ability01(player.skills.shooting) * 0.30)
                    .clamp(0.0, 1.0);
                estimate.execution_probability =
                    ((0.58 + strike * 0.34) * contact_fit).clamp(0.0, 0.99);
                // Recommended strike pace: a firm hit, harder from a stronger player.
                estimate.recommended_speed_yps = 26.0 + ability01(player.skills.strength) * 16.0;
                if estimate.execution_probability < 0.20 {
                    estimate.reselect_reason = "mpc-clearance-strike-unreachable";
                }
            }
            "control-touch" => {
                // A cushioned first touch / settle: dominated by touch technique and
                // a calm body shape, not power.
                let cushion = (ability01(player.skills.first_touch) * 0.62
                    + ability01(player.skills.passing) * 0.20
                    + ability01(player.skills.acceleration) * 0.18)
                    .clamp(0.0, 1.0);
                estimate.execution_probability =
                    ((0.50 + cushion * 0.45) * contact_fit).clamp(0.0, 0.99);
                // Settle the ball under control: a soft touch, not a drive.
                estimate.recommended_speed_yps = (3.5 + cushion * 3.0).min(player.velocity.len() + 4.0);
                if estimate.execution_probability < 0.20 {
                    estimate.reselect_reason = "mpc-control-touch-unsettled";
                }
            }
            "tackle" | "slide-tackle" | ACTION_LABEL_RECOVER => {
                // A ball-winning challenge: reach the carrier's ball and come away
                // with it. Slide tackles trade a little execution certainty (commit /
                // recovery risk) for their extra reach.
                let win = (ability01(player.skills.defending) * 0.50
                    + ability01(player.skills.defensive_tracking) * 0.30
                    + ability01(player.skills.aggression) * 0.20)
                    .clamp(0.0, 1.0);
                let slide_penalty = if label == "slide-tackle" { 0.90 } else { 1.0 };
                estimate.execution_probability =
                    ((0.34 + win * 0.50) * contact_fit * slide_penalty).clamp(0.0, 0.99);
                // Close to the ball at pace.
                estimate.recommended_speed_yps =
                    player_top_speed_yps(player.role, &player.skills) * MovementGait::Sprint.speed_multiplier();
                if estimate.execution_probability < 0.18 {
                    estimate.reselect_reason = "mpc-tackle-cannot-win-ball";
                }
            }
            _ => {}
        }
        if !estimate.reselect_reason.is_empty() {
            return estimate;
        }
    }

    if target_delta_yards > reachable_delta_yards {
        estimate.reselect_reason = "mpc-target-beyond-reachable-horizon";
    }
    estimate
}

fn direct_mpc_execution_estimate_for_candidate(
    snapshot: &WorldSnapshot,
    player: &PlayerAgent,
    action_target: &Option<AgentActionTargetTrace>,
    action_label: &str,
) -> Option<MpcExecutionEstimate> {
    let label = normalize_soccer_action_label(action_label);
    if !mpc_reselect_candidate_label(label) {
        return None;
    }
    let current = snapshot
        .player_position(player.id)
        .unwrap_or(player.position)
        .clamp_to_pitch(snapshot.field_width, snapshot.field_length);
    let target_delta_yards = action_target
        .as_ref()
        .and_then(|target| target.point)
        .map(|target| target.distance(current))
        .unwrap_or(0.0);
    let dt = sane_dt_seconds(snapshot.dt_seconds, DEFAULT_DT_SECONDS).max(1e-6);
    let execution_skill = mpc_execution_skill_for_label(player, label);
    let mpc_horizon_seconds = (dt * 2.0).clamp(2.0 / 15.0, 0.24);
    let reachable_delta_yards = (player.velocity.len() * mpc_horizon_seconds
        + 0.5
            * acceleration_yps2_from_score(player.skills.acceleration)
            * mpc_horizon_seconds
            * mpc_horizon_seconds
        + execution_skill * 1.6)
        .max(0.75);
    let execution_gap = (target_delta_yards - reachable_delta_yards).max(0.0);
    let mpc_cfg = &tunables().decision_mpc;
    let execution_confidence = ((1.0 - execution_gap / mpc_cfg.reselect_max_target_delta_yards)
        .clamp(0.0, 1.0)
        * (0.42 + execution_skill * 0.58)
        * player.decision_confidence.clamp(0.0, 1.0))
    .clamp(0.0, 1.0);
    Some(mpc_execution_estimate_for_action(
        snapshot,
        player,
        action_target,
        action_label,
        current,
        dt,
        target_delta_yards,
        reachable_delta_yards,
        execution_confidence,
    ))
}

pub(crate) fn player_mdp_mpc_comparison_trace(
    snapshot: &WorldSnapshot,
    player: &PlayerAgent,
    action_target: &Option<AgentActionTargetTrace>,
    action_label: &str,
) -> Option<SoccerMdpMpcComparisonTrace> {
    let guidance = snapshot.formation_lp_guidance_for(player.id)?;
    if !guidance.local_mpc_guidance {
        return None;
    }
    let mdp_target = action_target.as_ref().and_then(|target| target.point);
    let mpc_target = guidance.target;
    let current = snapshot
        .player_position(player.id)
        .unwrap_or(player.position)
        .clamp_to_pitch(snapshot.field_width, snapshot.field_length);
    let target_delta_yards = mdp_target
        .map(|target| target.distance(mpc_target))
        .unwrap_or(0.0);
    let dt = sane_dt_seconds(snapshot.dt_seconds, DEFAULT_DT_SECONDS).max(1e-6);
    let mdp_velocity = mdp_target
        .map(|target| (target - current) / dt)
        .unwrap_or(Vec2::zero());
    let velocity_delta_yps = (mdp_velocity - guidance.target_velocity).len();
    let label = normalize_soccer_action_label(action_label);
    let mpc_cfg = &tunables().decision_mpc;
    let semantic_mismatch = mdp_target.is_none()
        || snapshot.ball.holder == Some(player.id)
        || player.role == PlayerRole::Goalkeeper
        || matches!(
            label,
            "shoot"
                | "pass"
                | "pass1"
                | "switch-play"
                | "recycle-reset"
                | "aerial-pass"
                | "aerial-pass1"
                | "clearance"
                | "route-one"
                | "tackle"
                | "slide-tackle"
                | "control-touch"
        );
    let blend_eligible = !semantic_mismatch
        && target_delta_yards.is_finite()
        && target_delta_yards <= mpc_cfg.blend_max_target_delta_yards
        && matches!(
            label,
            "support-shape"
                | "support-roam"
                | "check-to-ball"
                | "wide-outlet"
                | "overlap-run"
                | "run-in-behind"
                | "shot-creation-run"
                | "support-push-up"
                | "press-cover"
                | "defend"
                | "hold"
                | ACTION_LABEL_RECOVER
                | "move"
        );
    let mpc_reselect_candidate = mpc_reselect_candidate_label(label);
    let execution_skill = mpc_execution_skill_for_label(player, label);
    let mpc_horizon_seconds = (dt * 2.0).clamp(2.0 / 15.0, 0.24);
    let reachable_delta_yards = (player.velocity.len() * mpc_horizon_seconds
        + 0.5
            * acceleration_yps2_from_score(player.skills.acceleration)
            * mpc_horizon_seconds
            * mpc_horizon_seconds
        + execution_skill * 1.6)
        .max(0.75);
    let execution_gap = (target_delta_yards - reachable_delta_yards).max(0.0);
    let mpc_cfg = &tunables().decision_mpc;
    let execution_confidence = ((1.0 - execution_gap / mpc_cfg.reselect_max_target_delta_yards)
        .clamp(0.0, 1.0)
        * (0.42 + execution_skill * 0.58)
        * player.decision_confidence.clamp(0.0, 1.0))
    .clamp(0.0, 1.0);
    let execution_estimate = mpc_execution_estimate_for_action(
        snapshot,
        player,
        action_target,
        action_label,
        current,
        dt,
        target_delta_yards,
        reachable_delta_yards,
        execution_confidence,
    );
    let reselect_after_mpc_low_confidence = mpc_reselect_candidate
        && ((target_delta_yards.is_finite()
            && target_delta_yards >= mpc_cfg.reselect_max_target_delta_yards
            && velocity_delta_yps.is_finite()
            && velocity_delta_yps >= reachable_delta_yards / mpc_horizon_seconds.max(1e-6)
            && execution_confidence < mpc_cfg.reselect_min_execution_confidence)
            || (!execution_estimate.reselect_reason.is_empty()
                && execution_estimate.execution_probability
                    < mpc_cfg.reselect_min_ball_execution_probability));
    let blended_target = if blend_eligible {
        mdp_target.map(|target| {
            ((target + mpc_target) * 0.5)
                .clamp_to_pitch(snapshot.field_width, snapshot.field_length)
        })
    } else {
        None
    };
    let deviation_recorded = reselect_after_mpc_low_confidence
        || target_delta_yards >= mpc_cfg.deviation_trace_threshold_yards
        || velocity_delta_yps >= 1.0
        || semantic_mismatch;
    let decision = if reselect_after_mpc_low_confidence {
        "reselect-mdp-after-mpc-low-confidence"
    } else if blend_eligible {
        "blend-continuous-candidate"
    } else if semantic_mismatch {
        "reject-blend-semantic-mismatch"
    } else {
        "record-deviation-only"
    };

    Some(SoccerMdpMpcComparisonTrace {
        player_id: player.id,
        action: action_label.to_string(),
        mdp_target,
        mpc_target: Some(mpc_target),
        blended_target,
        target_delta_yards: finite_metric(target_delta_yards),
        velocity_delta_yps: finite_metric(velocity_delta_yps),
        mdp_confidence: player.decision_confidence.clamp(0.0, 1.0),
        mpc_ball_urgency: finite_metric(execution_estimate.ball_urgency),
        mpc_touch_window_seconds: finite_metric(execution_estimate.touch_window_seconds),
        mpc_body_mechanics_fit: finite_metric(execution_estimate.body_mechanics_fit),
        mpc_execution_probability: finite_metric(execution_estimate.execution_probability),
        mpc_pass_completion_probability: finite_metric(
            execution_estimate.pass_completion_probability,
        ),
        mpc_dribble_success_probability: finite_metric(
            execution_estimate.dribble_success_probability,
        ),
        mpc_dribble_left_success_probability: finite_metric(
            execution_estimate.dribble_left_success_probability,
        ),
        mpc_dribble_right_success_probability: finite_metric(
            execution_estimate.dribble_right_success_probability,
        ),
        mpc_recommended_speed_yps: finite_metric(execution_estimate.recommended_speed_yps),
        mpc_recommended_angle_degrees: finite_metric(execution_estimate.recommended_angle_degrees),
        mpc_recommended_curve_bend_yards: finite_metric(
            execution_estimate.recommended_curve_bend_yards,
        ),
        mpc_recommended_spin_rps: finite_metric(execution_estimate.recommended_spin_rps),
        mpc_recommended_curve: execution_estimate.recommended_curve.to_string(),
        mpc_execution_horizon_seconds: finite_metric(execution_estimate.horizon_seconds),
        mpc_reselect_reason: execution_estimate.reselect_reason.to_string(),
        mpc_guidance_present: true,
        blend_eligible,
        deviation_recorded,
        decision: decision.to_string(),
    })
}

fn mpc_reselects_candidate(
    snapshot: &WorldSnapshot,
    player: &PlayerAgent,
    action: &SoccerAction,
    action_label: &str,
) -> bool {
    let action_target = player.action_target_trace(action, snapshot);
    if !snapshot
        .formation_lp_guidance_for(player.id)
        .is_some_and(|guidance| guidance.local_mpc_guidance)
    {
        return false;
    }
    if player_mdp_mpc_comparison_trace(snapshot, player, &action_target, action_label)
        .is_some_and(|trace| trace.decision == "reselect-mdp-after-mpc-low-confidence")
    {
        return true;
    }
    direct_mpc_execution_estimate_for_candidate(snapshot, player, &action_target, action_label)
        .is_some_and(|estimate| {
            !estimate.reselect_reason.is_empty()
                && estimate.execution_probability
                    < tunables()
                        .decision_mpc
                        .reselect_min_ball_execution_probability
        })
}

/// Minimum ticks between two consecutive deliberative decisions (≈0.20 s at 15 Hz): at most one
/// changed decision per 3 ticks. See [`PlayerAgent::apply_decision_refractory`].
const DECISION_REFRACTORY_MIN_GAP_TICKS: u64 = 3;
/// The second-to-last decision must be this many ticks old before a third may commit (≈0.47 s):
/// at most two changed decisions per 7-tick window. Combined with the 3-tick min gap this gives
/// the 3,4,3,4-tick cadence (decisions on ticks 1, 4, 8, 11, 15…).
const DECISION_REFRACTORY_WINDOW_TICKS: u64 = 7;

/// Master switch for the deliberative-decision refractory. OFF by default ⇒ players re-decide
/// freely every tick exactly as before (byte-identical); set `DD_SOCCER_ENABLE_DECISION_REFRACTORY`
/// to rate-limit how often a player may *change its mind* (human reaction time + processing +
/// momentum). The new `PlayerAgent` continuity fields stay empty while disabled, so serialized
/// snapshots are unchanged too.
pub(crate) fn decision_refractory_enabled() -> bool {
    #[cfg(test)]
    {
        use std::sync::OnceLock;
        static V: OnceLock<bool> = OnceLock::new();
        *V.get_or_init(|| std::env::var("DD_SOCCER_ENABLE_DECISION_REFRACTORY").is_ok())
    }
    #[cfg(not(test))]
    {
        use std::sync::OnceLock;
        static V: OnceLock<bool> = OnceLock::new();
        *V.get_or_init(|| gate_default_on("DD_SOCCER_ENABLE_DECISION_REFRACTORY"))
    }
}

/// Whether the **forward-pass-first release** is active this process. Default-ON in
/// production (env `DD_SOCCER_ENABLE_FORWARD_PASS_FIRST=0/false` is the kill switch);
/// default-OFF under test so the option-scoring parity suite stays byte-identical.
pub(crate) fn forward_pass_first_enabled() -> bool {
    #[cfg(test)]
    {
        std::env::var("DD_SOCCER_ENABLE_FORWARD_PASS_FIRST").is_ok()
    }
    #[cfg(not(test))]
    {
        use std::sync::OnceLock;
        static V: OnceLock<bool> = OnceLock::new();
        *V.get_or_init(|| gate_default_on("DD_SOCCER_ENABLE_FORWARD_PASS_FIRST"))
    }
}

/// Whether the **isolated attacking-carrier drive** is active this process. This is the fix for
/// the "no teammate ahead ⇒ panic backward pass" blunder: the carrier instead either drives at
/// goal solo or holds the ball up moving forward while support arrives, with a backward recycle
/// as the very last resort. Default-ON in production (env
/// `DD_SOCCER_ENABLE_ISOLATED_CARRIER_DRIVE=0/false` is the kill switch); default-OFF under test
/// so the option-scoring parity suite stays byte-identical. See
/// [`isolated_attacking_carrier_drive_mode`].
pub(crate) fn isolated_carrier_drive_enabled() -> bool {
    #[cfg(test)]
    {
        std::env::var("DD_SOCCER_ENABLE_ISOLATED_CARRIER_DRIVE").is_ok()
    }
    #[cfg(not(test))]
    {
        use std::sync::OnceLock;
        static V: OnceLock<bool> = OnceLock::new();
        *V.get_or_init(|| gate_default_on("DD_SOCCER_ENABLE_ISOLATED_CARRIER_DRIVE"))
    }
}

/// What an isolated attacking carrier (no teammate ahead, no open forward pass) should do
/// INSTEAD of panicking into a backward pass. See [`isolated_attacking_carrier_drive_mode`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum IsolatedCarrierDriveMode {
    /// Few defenders are goal-side of the ball, OR there's space + a speed advantage to run into:
    /// drive at goal at a sprint and shoot once inside [`ISOLATED_CARRIER_SHOOT_YARDS`].
    SoloGoalDrive,
    /// A solo goal does not look promising: keep the ball, carry it forward / on an angle at a
    /// CONTROLLED pace (not a sprint), and wait for teammates to push up into the attack. A
    /// backward/square recycle is the very last resort.
    HoldUp,
}

/// "Fewer than 4 defenders behind the ball" (the user's threshold): with this many or more
/// opponents goal-side of the ball a solo run is into a packed block, so prefer hold-up.
const ISOLATED_CARRIER_MAX_DEFENDERS_BEHIND_BALL: usize = 4;
/// Forward dribble space (yards) that counts as "a lot of space to run into" for the solo drive.
const ISOLATED_CARRIER_SOLO_OPEN_SPACE_YARDS: f64 = 12.0;
/// Nearest-opponent distance (yards) clear of the carrier that counts as runway for the solo run.
const ISOLATED_CARRIER_SOLO_RUNWAY_YARDS: f64 = 6.0;
/// Closing rate (yards/sec) at/below which the nearest opponent is NOT catching the carrier — a
/// favourable speed differential. Negative means the defender is falling behind.
const ISOLATED_CARRIER_SOLO_CLOSING_CUTOFF_YPS: f64 = 0.5;
/// Once the solo driver is within this range of goal he should look to shoot (the user's 25 yards).
const ISOLATED_CARRIER_SHOOT_YARDS: f64 = 25.0;
/// A forward pass option this open means the carrier is NOT actually isolated — let the normal
/// forward-pass-first / killer-pass logic play it instead of forcing a drive/hold-up.
const ISOLATED_CARRIER_FORWARD_OPTION_OPEN_CUTOFF: f64 = 0.45;

/// A clear runway + favourable speed differential for the solo goal drive: lots of forward space,
/// the nearest opponent is not on top of the carrier, and is not closing the gap. Pure + RNG-free.
pub(crate) fn isolated_carrier_solo_speed_space_advantage(
    observation: &SoccerPomdpObservation,
) -> bool {
    let big_space =
        observation.forward_dribble_space_yards >= ISOLATED_CARRIER_SOLO_OPEN_SPACE_YARDS;
    let runway = observation.nearest_opponent_distance >= ISOLATED_CARRIER_SOLO_RUNWAY_YARDS;
    let pulling_away = observation
        .neural_extended
        .nearest_opponent_closing_rate_yps
        <= ISOLATED_CARRIER_SOLO_CLOSING_CUTOFF_YPS;
    big_space && runway && pulling_away
}

/// Recognise the "isolated attacking carrier" picture and choose the response. Returns `None`
/// unless THIS carrier is an attacker (a Forward, or a wide / outside midfielder — i.e. a winger)
/// who has the ball in the ATTACKING half with NO teammate ahead of him AND no genuinely open
/// forward pass on. That is precisely the state that used to deteriorate into a panicked backward
/// pass. When it holds, the carrier should DRIVE at goal ([`SoloGoalDrive`]) when few defenders
/// are behind the ball or there's space + a speed advantage, otherwise hold the ball up
/// ([`HoldUp`]) and wait for support — never recycle backwards as a first instinct. Pure +
/// RNG-free so it can be unit-tested and shared by the scoring and gait paths.
///
/// [`SoloGoalDrive`]: IsolatedCarrierDriveMode::SoloGoalDrive
/// [`HoldUp`]: IsolatedCarrierDriveMode::HoldUp
pub(crate) fn isolated_attacking_carrier_drive_mode(
    observation: &SoccerPomdpObservation,
    role: PlayerRole,
    is_outside_midfielder: bool,
) -> Option<IsolatedCarrierDriveMode> {
    // Only the attacking carriers who actually get stranded up top: strikers AND wide midfielders
    // (wingers), per the report that outside mids exhibit the same bug.
    if !(role == PlayerRole::Forward || is_outside_midfielder) {
        return None;
    }
    if !observation.has_ball {
        return None;
    }
    // Only when attacking (in the opponent half / past the half-way line). Never override the
    // own-half retention / build-up / clearance logic, where a safe backward ball is correct.
    if observation.yards_to_goal >= observation.yards_to_own_goal {
        return None;
    }
    // The bug's precondition: nobody ahead of the carrier to play, and no genuinely open forward
    // pass option on. (When a forward man IS open, the forward-pass-first / killer-pass paths
    // handle it — this recogniser stays out of the way.)
    let no_teammate_ahead =
        observation.field_teammates_ahead == 0 && observation.teammates_ahead == 0;
    if !no_teammate_ahead {
        return None;
    }
    let forward_option_open = observation.visible_forward_pass_options > 0
        && observation
            .best_forward_pass_option_quality
            .max(observation.best_forward_pass_receiver_openness)
            >= ISOLATED_CARRIER_FORWARD_OPTION_OPEN_CUTOFF;
    if forward_option_open {
        return None;
    }
    // Solo goal vs hold-up: few defenders goal-side of the ball, OR space + a speed advantage.
    let defenders_behind_ball = observation.field_opponents_ahead;
    let solo = defenders_behind_ball < ISOLATED_CARRIER_MAX_DEFENDERS_BEHIND_BALL
        || isolated_carrier_solo_speed_space_advantage(observation);
    Some(if solo {
        IsolatedCarrierDriveMode::SoloGoalDrive
    } else {
        IsolatedCarrierDriveMode::HoldUp
    })
}

/// Whether `role` at home-x `home_x` is an outside (wide) midfielder — a winger. Matches the
/// `outside_midfielder` test used inside [`possession_action_options`], factored out so the gait
/// resolver can reuse it for [`isolated_attacking_carrier_drive_mode`].
pub(crate) fn is_outside_midfielder_role(role: PlayerRole, home_x: f64, field_width: f64) -> bool {
    role == PlayerRole::Midfielder
        && (home_x < field_width * 0.30 || home_x > field_width * 0.70)
}

/// Re-weight the carrier's possession options for the isolated-carrier `mode` so a forward DRIVE
/// (or hold-up carry) floats above the panicked backward/square recycle. `pass1` is a
/// backward/square ball in this state by construction (no teammate is ahead), so it is damped too;
/// nothing is made illegal, so a backward outlet still survives when nothing else is on. Pure +
/// RNG-free (operates only on the option scores) so it is unit-testable without the env gate.
pub(crate) fn apply_isolated_carrier_drive_bias(
    options: &mut Vec<AgentActionOptionTrace>,
    mode: IsolatedCarrierDriveMode,
    in_shot_range: bool,
    shot_legal: bool,
) {
    // Square / backward outlets that become a last resort in both modes.
    let backward_outlets = ["recycle-reset", "switch-play", "route-one"];
    match mode {
        IsolatedCarrierDriveMode::SoloGoalDrive => {
            let mut drive_family = vec!["vertical-attack", "turnover-burst", "carry-forward"];
            if in_shot_range && shot_legal {
                drive_family.push("shoot");
            }
            ensure_min_legal_option_family_probability(options, &drive_family, 0.72);
            for label in backward_outlets {
                scale_legal_option_score(options, label, 0.28);
            }
            // Don't square it to a covered man, and don't stall on a shield/turn when the lane to
            // goal is there to be driven.
            scale_legal_option_score(options, "pass1", 0.40);
            for label in ["protect-ball", "hold-up-flank", "side-step"] {
                scale_legal_option_score(options, label, 0.58);
            }
        }
        IsolatedCarrierDriveMode::HoldUp => {
            // Carry forward / on an angle at a controlled pace (the sprint resolver keeps it off a
            // sprint) and wait for support to arrive.
            ensure_min_legal_option_family_probability(
                options,
                &["carry-forward", "carry-out-left", "carry-out-right"],
                0.60,
            );
            // Backward is the very last resort: damp it hard. Shielding while you wait
            // (protect-ball) is fine, so it is intentionally left alone.
            for label in backward_outlets {
                scale_legal_option_score(options, label, 0.20);
            }
            scale_legal_option_score(options, "pass1", 0.32);
        }
    }
}

/// Forward-receiver openness at/above which a forward pass is "sufficiently open" to be
/// released early instead of dwelt on (the user's "if that option is forward and sufficiently
/// open"). Below this the carrier keeps his normal patience.
const FORWARD_PASS_FIRST_OPEN_THRESHOLD: f64 = 0.45;
/// Seconds of dwell on the ball at which the forward-release urgency saturates: a carrier
/// sitting this long on an open forward option is pushed fully toward releasing it ("passing
/// sooner rather than later"). The bias already applies (weaker) the instant he has the ball.
const FORWARD_PASS_FIRST_DWELL_FULL_SECONDS: f64 = 0.9;

/// Release strength in `[0, 1]` for the forward-pass-first bias: how strongly to prioritise the
/// forward ball (floor it) and damp the hold/dribble/shield family. Returns 0 below the openness
/// threshold; otherwise rises with how open the forward receiver is *and* with how long the
/// carrier has already dwelt on the ball, so deliberation actively converts into a forward
/// release rather than a late, pressured square/back blunder. Pure + RNG-free for unit testing.
pub(crate) fn forward_pass_first_release_strength(
    forward_open: f64,
    time_on_ball_seconds: f64,
) -> f64 {
    let fwd = forward_open.clamp(0.0, 1.0);
    if fwd < FORWARD_PASS_FIRST_OPEN_THRESHOLD {
        return 0.0;
    }
    let open_strength = ((fwd - FORWARD_PASS_FIRST_OPEN_THRESHOLD)
        / (1.0 - FORWARD_PASS_FIRST_OPEN_THRESHOLD))
        .clamp(0.0, 1.0);
    let dwell = (time_on_ball_seconds.max(0.0) / FORWARD_PASS_FIRST_DWELL_FULL_SECONDS).clamp(0.0, 1.0);
    ((0.45 + 0.55 * dwell) * (0.55 + 0.45 * open_strength)).clamp(0.0, 1.0)
}

/// True when the refractory forbids committing a changed decision at tick `now`, given the ticks
/// of the last (up to two) committed decisions. Allowed iff it has been ≥3 ticks since the last
/// decision **and** ≥7 ticks since the second-to-last.
fn decision_refractory_blocks(commit_ticks: &VecDeque<u64>, now: u64) -> bool {
    let len = commit_ticks.len();
    if let Some(&last) = commit_ticks.back() {
        if now.saturating_sub(last) < DECISION_REFRACTORY_MIN_GAP_TICKS {
            return true;
        }
    }
    if len >= 2 {
        let second_last = commit_ticks[len - 2];
        if now.saturating_sub(second_last) < DECISION_REFRACTORY_WINDOW_TICKS {
            return true;
        }
    }
    false
}

fn agentic_action_commitment(
    score: f64,
    dt_seconds: f64,
    observation: &SoccerPomdpObservation,
    role: PlayerRole,
) -> bool {
    let score = score.clamp(0.0, 1.0);
    if score <= 0.0 {
        return false;
    }
    let urgency = observation
        .decision_urgency
        .max(observation.offensive_urgency)
        .max(observation.pressure_urgency)
        .max(observation.immediate_dispossession_risk)
        .clamp(0.0, 1.0);
    let pressure = observation.perceived_pressure.clamp(0.0, 1.0);
    let role_patience = match role {
        PlayerRole::Goalkeeper => 0.018,
        PlayerRole::Defender => 0.016,
        PlayerRole::Midfielder => 0.014,
        PlayerRole::Forward => 0.012,
    };
    let threshold = (role_patience - urgency * 0.010 - pressure * 0.004).clamp(0.003, 0.024);
    time_window_probability(score, dt_seconds) >= threshold
}

fn decision_cadence_labels_match(committed_label: &str, candidate_label: &str) -> bool {
    let committed_label = normalize_soccer_action_label(committed_label);
    let candidate_label = normalize_soccer_action_label(candidate_label);
    committed_label == candidate_label
        || (committed_label == "defend"
            && matches!(candidate_label, "defend-shape" | "defend-roam"))
}

fn decision_cadence_immediate_label(label: &str) -> bool {
    matches!(
        normalize_soccer_action_label(label),
        "tackle" | "slide-tackle" | ACTION_LABEL_RECOVER
    )
}

fn decision_cadence_hold_label_from_weights(
    observation: &SoccerPomdpObservation,
    weighted_ops: &[(String, f64)],
    tick: u64,
) -> Option<String> {
    let carryover = observation.previous_tick_carryover.as_ref()?;
    if !soccer_decision_commitment_switch_locked_for_history(&carryover.commitment_ticks, tick) {
        return None;
    }
    if decision_cadence_immediate_label(&carryover.action) {
        return None;
    }
    weighted_ops
        .iter()
        .filter(|(_, score)| score.is_finite() && *score > 0.0)
        .find(|(label, _)| decision_cadence_labels_match(&carryover.action, label))
        .map(|(label, _)| label.clone())
}

fn apply_decision_cadence_hold_weights(weighted_ops: &mut [(String, f64)], hold_label: &str) {
    let best_non_immediate_score = weighted_ops
        .iter()
        .filter(|(label, _)| !decision_cadence_immediate_label(label))
        .map(|(_, score)| if score.is_finite() { *score } else { 0.0 })
        .fold(0.0_f64, f64::max);
    for (label, score) in weighted_ops.iter_mut() {
        if decision_cadence_labels_match(hold_label, label) {
            *score = (*score).max(best_non_immediate_score + 1.0).max(1.0);
        } else if decision_cadence_immediate_label(label) {
            *score = (*score).max(0.0);
        } else {
            *score *= 0.01;
        }
    }
}

#[cfg(test)]
mod decision_cadence_player_tests {
    use super::*;

    fn score_for(weighted_ops: &[(String, f64)], label: &str) -> f64 {
        weighted_ops
            .iter()
            .find(|(candidate, _)| candidate == label)
            .map(|(_, score)| *score)
            .expect("weighted label")
    }

    #[test]
    fn decision_cadence_hold_boosts_committed_non_immediate_label() {
        let mut weighted_ops = vec![
            ("support-shape".to_string(), 0.15),
            ("run-in-behind".to_string(), 0.72),
            ("wide-outlet".to_string(), 0.44),
        ];

        apply_decision_cadence_hold_weights(&mut weighted_ops, "support-shape");

        assert!(
            score_for(&weighted_ops, "support-shape") > score_for(&weighted_ops, "run-in-behind"),
            "cadence-held support shape should outrank ordinary support switches"
        );
        assert!(
            score_for(&weighted_ops, "wide-outlet") < 0.01,
            "non-held ordinary switches should be damped while cadence-locked"
        );
    }

    #[test]
    fn decision_cadence_hold_does_not_outboost_high_immediate_tackle() {
        let mut weighted_ops = vec![
            ("defend-shape".to_string(), 0.20),
            ("press-cover".to_string(), 0.80),
            ("tackle".to_string(), 4.00),
        ];

        apply_decision_cadence_hold_weights(&mut weighted_ops, "defend");

        assert_eq!(score_for(&weighted_ops, "tackle"), 4.00);
        assert!(
            score_for(&weighted_ops, "tackle") > score_for(&weighted_ops, "defend-shape"),
            "urgent tackle scores must remain able to outrank a cadence-held defensive shape"
        );
        assert!(
            score_for(&weighted_ops, "defend-shape") > score_for(&weighted_ops, "press-cover"),
            "held defensive shape should still beat ordinary non-immediate defensive switches"
        );
    }
}

fn agentic_action_order<T>(items: Vec<(T, f64)>) -> Vec<T> {
    agentic_action_order_scored(items).0
}

/// Like [`agentic_action_order`] but also returns the post-transform ordering
/// weight of each item in ranked order, so a value-weighted (Boltzmann) policy
/// draw can score candidates by how good they are rather than by rank alone. The
/// item order is identical to [`agentic_action_order`]; the weights are exactly
/// the keys it sorts on (so `weights[0]` is the argmax weight).
fn agentic_action_order_scored<T>(items: Vec<(T, f64)>) -> (Vec<T>, Vec<f64>) {
    let mut keyed = Vec::with_capacity(items.len());
    for (ordinal, (item, weight)) in items.into_iter().enumerate() {
        let weight = weighted_agentic_order_weight(weight);
        keyed.push((weight, ordinal, item));
    }
    keyed.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    let mut ordered = Vec::with_capacity(keyed.len());
    let mut weights = Vec::with_capacity(keyed.len());
    for (weight, _, item) in keyed {
        ordered.push(item);
        weights.push(weight);
    }
    (ordered, weights)
}

impl PlayerAgent {
    pub(crate) fn record_position_history(&mut self) {
        self.position_history.push_back(self.position);
        while self.position_history.len() > PLAYER_POSITION_HISTORY_LIMIT {
            self.position_history.pop_front();
        }
    }

    pub fn history_velocity_estimate(&self, dt_seconds: f64) -> Vec2 {
        if dt_seconds <= 0.0 || self.position_history.len() < 2 {
            return self.velocity;
        }
        let last = self.position_history.len() - 1;
        (self.position_history[last] - self.position_history[last - 1]) / dt_seconds
    }

    pub fn history_acceleration_estimate(&self, dt_seconds: f64) -> Vec2 {
        if dt_seconds <= 0.0 || self.position_history.len() < 3 {
            return self.acceleration;
        }
        let last = self.position_history.len() - 1;
        let v0 = (self.position_history[last - 1] - self.position_history[last - 2]) / dt_seconds;
        let v1 = (self.position_history[last] - self.position_history[last - 1]) / dt_seconds;
        (v1 - v0) / dt_seconds
    }

    pub fn history_jerk_estimate(&self, dt_seconds: f64) -> Vec2 {
        if dt_seconds <= 0.0 || self.position_history.len() < 4 {
            return self.jerk;
        }
        let last = self.position_history.len() - 1;
        let v0 = (self.position_history[last - 2] - self.position_history[last - 3]) / dt_seconds;
        let v1 = (self.position_history[last - 1] - self.position_history[last - 2]) / dt_seconds;
        let v2 = (self.position_history[last] - self.position_history[last - 1]) / dt_seconds;
        let a0 = (v1 - v0) / dt_seconds;
        let a1 = (v2 - v1) / dt_seconds;
        (a1 - a0) / dt_seconds
    }

    /// THE single point where additive per-tick aerobic `fatigue` is applied: one clamp
    /// policy, one place to audit and tune the whole fatigue budget. Every additive
    /// contributor funnels through here (this resolves audit finding #2 — fatigue was written
    /// from five scattered sites with independent clamps and no overview):
    ///   • locomotion gait cost — `MovementGait::fatigue_delta`, charged on the movement tick
    ///     in `SoccerMatch::move_player_towards`. Run/sprint deplete; walk/stand RECOVER, so
    ///     `delta` is signed (can be negative).
    ///   • rotation cost — `actual_turn * YAW_ENERGY_FATIGUE_PER_RAD` (this fn).
    ///   • ball-involvement cost — `BALL_INVOLVEMENT_FATIGUE_PER_SECOND` within the 12yd ball
    ///     radius, scaled by gait intensity (this fn).
    ///   • supra-CP aerobic bleed — the slice of above-critical-power work (this fn).
    ///   • extra chase load — `SoccerMatch::add_extra_chase_fatigue` (event-driven).
    ///   • period-break recovery — `SoccerMatch::start_new_period` (negative `Stand` delta).
    /// (Half-time recovery is a MULTIPLICATIVE reset, not an additive delta, so it stays inline
    /// in `SoccerMatch`.) `fatigue` is the slow aerobic channel; the fast anaerobic W′ battery
    /// (`anaerobic_load`) is the SEPARATE channel updated only below in this function.
    pub(crate) fn add_fatigue(&mut self, delta: f64) {
        self.fatigue = (finite_metric(self.fatigue).clamp(0.0, 1.0) + finite_metric(delta))
            .clamp(0.0, 1.0);
    }

    pub(crate) fn update_body_orientation_dizziness_energy(
        &mut self,
        dt_seconds: f64,
        ball_position: Vec2,
        ball_holder: Option<usize>,
    ) {
        use std::f64::consts::{PI, TAU};

        let dt = sane_dt_seconds(dt_seconds, DEFAULT_DT_SECONDS).max(1e-3);
        let desired_dir = if ball_holder == Some(self.id) {
            let raw = facing_bucket_to_vector(self.action_facing)
                .filter(|v| v.len() > 1e-6)
                .unwrap_or_else(|| Vec2::new(0.0, self.team.attack_dir()));
            let forward = Vec2::new(0.0, self.team.attack_dir());
            let blended = raw + forward * CARRIER_FORWARD_FACING_NUDGE;
            let blended = if blended.len() > 1e-6 {
                blended.normalized()
            } else {
                raw
            };
            let to_ball = ball_position - self.position;
            let follow = if to_ball.len() > 1e-3 {
                let to_ball_dir = to_ball.normalized();
                blended * (1.0 - CARRIER_BALL_FACING_FOLLOW)
                    + to_ball_dir * CARRIER_BALL_FACING_FOLLOW
            } else {
                blended
            };
            if follow.len() > 1e-6 {
                follow.normalized()
            } else {
                blended
            }
        } else {
            let to_ball = ball_position - self.position;
            let ball_dir = if to_ball.len() > 1e-3 {
                to_ball.normalized()
            } else {
                Vec2::new(0.0, self.team.attack_dir())
            };
            let velocity = self.velocity;
            let speed = finite_metric(velocity.len()).max(0.0);
            if speed < MOVING_FACING_MIN_SPEED_YPS {
                ball_dir
            } else {
                let move_dir = velocity.normalized();
                use std::f64::consts::FRAC_PI_2;
                match self.movement_gait {
                    MovementGait::Walk | MovementGait::Jog => {
                        clamp_dir_to_cone(ball_dir, move_dir, FRAC_PI_2)
                    }
                    MovementGait::Run => clamp_dir_to_cone(ball_dir, move_dir, FRAC_PI_2 * 0.80),
                    MovementGait::Sprint => clamp_dir_to_cone(ball_dir, move_dir, FRAC_PI_2 * 0.55),
                    MovementGait::BackWalk | MovementGait::BackJog | MovementGait::BackSkip => {
                        clamp_dir_to_cone(ball_dir, move_dir * -1.0, FRAC_PI_2)
                    }
                    MovementGait::Skip | MovementGait::SideStep => {
                        clamp_dir_to_cone(ball_dir, move_dir, FRAC_PI_2)
                    }
                    MovementGait::Stand => ball_dir,
                }
            }
        };

        self.facing_yaw = finite_metric(self.facing_yaw);
        let desired_yaw = finite_metric(desired_dir.y.atan2(desired_dir.x));
        let mut delta = desired_yaw - self.facing_yaw;
        while delta > PI {
            delta -= TAU;
        }
        while delta < -PI {
            delta += TAU;
        }

        let max_rate_change = MAX_BODY_YAW_ACCEL_RAD_S2 * dt;
        let top_speed = finite_metric(player_top_speed_yps(self.role, &self.skills)).max(1e-6);
        let speed_frac =
            (finite_metric(self.velocity.len()).max(0.0) / top_speed).clamp(0.0, 1.0);
        let max_yaw_rate = if ball_holder == Some(self.id) {
            POSSESSION_MAX_YAW_RATE_SLOW_RAD_S
                - speed_frac
                    * (POSSESSION_MAX_YAW_RATE_SLOW_RAD_S - POSSESSION_MAX_YAW_RATE_FAST_RAD_S)
        } else {
            OFF_BALL_MAX_YAW_RATE_SLOW_RAD_S
                - speed_frac * (OFF_BALL_MAX_YAW_RATE_SLOW_RAD_S - OFF_BALL_MAX_YAW_RATE_FAST_RAD_S)
        };
        let stopping_rate = (2.0 * MAX_BODY_YAW_ACCEL_RAD_S2 * delta.abs()).sqrt() * delta.signum();
        let target_rate = stopping_rate.clamp(-max_yaw_rate, max_yaw_rate);
        let yaw_rate = finite_metric(self.yaw_rate);
        let new_rate = target_rate.clamp(
            yaw_rate - max_rate_change,
            yaw_rate + max_rate_change,
        );
        self.yaw_rate = new_rate;
        let proposed_turn = (new_rate * dt).clamp(
            -MAX_PLAYER_BODY_YAW_TURN_PER_TICK_RAD,
            MAX_PLAYER_BODY_YAW_TURN_PER_TICK_RAD,
        );
        let actual_turn = if proposed_turn.abs() > delta.abs() {
            delta
        } else {
            proposed_turn
        };
        self.facing_yaw += actual_turn;
        while self.facing_yaw > PI {
            self.facing_yaw -= TAU;
        }
        while self.facing_yaw < -PI {
            self.facing_yaw += TAU;
        }

        let turn_rate = actual_turn.abs() / dt;
        let accrual = (turn_rate - COMFORT_YAW_RATE_RAD_S).max(0.0) * DIZZINESS_GAIN_PER_RAD * dt;
        self.dizziness = ((finite_metric(self.dizziness).clamp(0.0, DIZZINESS_MAX) + accrual)
            * (1.0 - DIZZINESS_RECOVERY_PER_SECOND * dt).max(0.0))
        .clamp(0.0, DIZZINESS_MAX);

        let rotation_fatigue = actual_turn.abs() * YAW_ENERGY_FATIGUE_PER_RAD;
        let involvement_fatigue =
            if self.position.distance(ball_position) <= BALL_INVOLVEMENT_RADIUS_YARDS {
                let intensity = match self.movement_gait {
                    MovementGait::Sprint => 1.0,
                    MovementGait::Run => 0.7,
                    _ => 0.35,
                };
                BALL_INVOLVEMENT_FATIGUE_PER_SECOND * intensity * dt
            } else {
                0.0
            };
        self.add_fatigue(rotation_fatigue + involvement_fatigue);

        // Per-tick locomotion metabolic cost (J) for the wasted-energy reward. Computed
        // every tick from the same speed/forward-accel running model the W′ branch uses,
        // independent of `dd_soccer_disable_power_duration_ceiling()` so the metric exists
        // under either energy model. Read-only metric: never changes a decision.
        {
            let speed_yps = finite_metric(self.velocity.len()).max(0.0);
            let accel_yps2 = finite_metric(self.acceleration.len()).max(0.0);
            let accel_forward_yps2 = if speed_yps > 0.5 {
                finite_metric(dot(self.acceleration, self.velocity / speed_yps)).max(0.0)
            } else {
                accel_yps2
            };
            let locomotion_joules =
                metabolic_power_demand_w(&self.skills, speed_yps, accel_forward_yps2) * dt;
            self.last_tick_locomotion_joules = finite_metric(locomotion_joules).max(0.0);
            // Sustained flat-out sprint tracking for the sustained-effort-no-outcome penalty:
            // accumulate held time + distance while at ≥98% of top speed, and reset the moment
            // the player eases below it — so only a long, all-out run ever qualifies. Gated; the
            // fields stay 0 and unread when off, leaving the trajectory byte-identical.
            if dd_soccer_enable_sustained_effort_no_outcome_penalty() {
                let top_speed = player_top_speed_yps(self.role, &self.skills).max(1e-6);
                if speed_yps >= top_speed * SUSTAINED_EFFORT_SPEED_FRACTION {
                    self.sustained_sprint_seconds += dt;
                    self.sustained_sprint_distance_yards += speed_yps * dt;
                } else {
                    self.sustained_sprint_seconds = 0.0;
                    self.sustained_sprint_distance_yards = 0.0;
                }
            }
        }

        if dd_soccer_disable_power_duration_ceiling() {
            let current_load = finite_metric(self.anaerobic_load).clamp(0.0, 1.0);
            let accel_load = finite_metric(self.acceleration.len()).max(0.0)
                * ANAEROBIC_ACCEL_LOAD_PER_YPS2
                * dt;
            let burst_load = match self.movement_gait {
                MovementGait::Sprint => ANAEROBIC_SPRINT_LOAD_PER_SECOND,
                MovementGait::Run => ANAEROBIC_SPRINT_LOAD_PER_SECOND * 0.45,
                _ => 0.0,
            } * dt
                + accel_load;
            self.anaerobic_load =
                (current_load + burst_load - ANAEROBIC_RECOVERY_PER_SECOND * dt).clamp(0.0, 1.0);
        } else {
            let cp = finite_metric(player_critical_power_w(&self.skills)).max(1.0);
            let w_prime_max = finite_metric(player_anaerobic_capacity_j(&self.skills)).max(1.0);
            let speed_yps = finite_metric(self.velocity.len()).max(0.0);
            let accel_yps2 = finite_metric(self.acceleration.len()).max(0.0);
            let accel_forward_yps2 = if speed_yps > 0.5 {
                finite_metric(dot(self.acceleration, self.velocity / speed_yps)).max(0.0)
            } else {
                accel_yps2
            };
            let demand =
                finite_metric(metabolic_power_demand_w(&self.skills, speed_yps, accel_forward_yps2))
                    .max(0.0);
            let current_load = finite_metric(self.anaerobic_load).clamp(0.0, 1.0);
            let mut w_prime = (1.0 - current_load) * w_prime_max;
            if demand >= cp {
                let overshoot = demand - cp;
                w_prime -= overshoot * dt;
                let supra_cp_kj = overshoot * dt / 1000.0;
                self.add_fatigue(supra_cp_kj * SUPRA_CP_AEROBIC_FATIGUE_PER_KJ);
            } else {
                let mass_kg =
                    finite_metric(player_weight_pounds(self.skills.weight_pounds) * KG_PER_POUND)
                        .max(1.0);
                let restore = ((cp - demand) * ANAEROBIC_RECOVERY_GAIN)
                    .min(ANAEROBIC_RECOVERY_CAP_W_PER_KG * mass_kg);
                w_prime += restore * dt;
            }
            w_prime = w_prime.clamp(0.0, w_prime_max);
            self.anaerobic_load = (1.0 - w_prime / w_prime_max).clamp(0.0, 1.0);
        }
    }

    fn open_pass_lane_dribble_plan_for(
        &self,
        snapshot: &WorldSnapshot,
        observation: &SoccerPomdpObservation,
        pass_targets: &[usize],
    ) -> Option<OpenPassLaneDribblePlan> {
        if !observation.has_ball
            || self.role == PlayerRole::Goalkeeper
            || goal_attack_shot_blocks_alternatives(observation, self.role)
        {
            return None;
        }

        let current = snapshot
            .player_position(self.id)
            .unwrap_or(self.position)
            .clamp_to_pitch(snapshot.field_width, snapshot.field_length);
        let Some(me_snapshot) = snapshot.players.iter().find(|player| player.id == self.id) else {
            return None;
        };
        let attack_dir = self.team.attack_dir();
        let distances = [
            1.15,
            1.65,
            2.25,
            3.0,
            3.65,
            OPEN_PASS_LANE_BASE_MAX_DRIBBLE_YARDS,
            5.0,
            6.25,
            OPEN_PASS_LANE_MAX_DRIBBLE_YARDS,
        ];
        let candidate_dirs = [
            Vec2::new(-1.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(-0.94, 0.28 * attack_dir).normalized(),
            Vec2::new(0.94, 0.28 * attack_dir).normalized(),
            Vec2::new(-0.78, -0.18 * attack_dir).normalized(),
            Vec2::new(0.78, -0.18 * attack_dir).normalized(),
            Vec2::new(-0.42, 0.78 * attack_dir).normalized(),
            Vec2::new(0.42, 0.78 * attack_dir).normalized(),
        ];

        let mut target_candidates: Vec<(usize, f64)> = snapshot
            .players
            .iter()
            .filter(|player| {
                player.team == self.team
                    && player.id != self.id
                    && player.role != PlayerRole::Goalkeeper
            })
            .filter_map(|player| {
                let target_position = snapshot
                    .player_position(player.id)
                    .unwrap_or(player.position)
                    .clamp_to_pitch(snapshot.field_width, snapshot.field_length);
                let pass_distance = current.distance(target_position);
                if !(5.0..=44.0).contains(&pass_distance) {
                    return None;
                }
                let receiver_forward = (target_position.y - current.y) * attack_dir;
                if receiver_forward < -OPEN_PASS_LANE_MAX_BACKFIELD_RECEIVER_YARDS {
                    return None;
                }
                let ranked_bonus = pass_targets
                    .iter()
                    .position(|target_id| *target_id == player.id)
                    .map(|rank| {
                        if rank == 0 {
                            48.0
                        } else {
                            5.0 - rank.min(5) as f64 * 0.45
                        }
                    })
                    .unwrap_or(0.0);
                let forward_bonus = receiver_forward.max(0.0).min(24.0) * 0.18;
                Some((
                    player.id,
                    pass_distance + (-receiver_forward).max(0.0) * 3.0
                        - ranked_bonus
                        - forward_bonus,
                ))
            })
            .collect();
        target_candidates
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let target_ids: Vec<usize> = target_candidates
            .into_iter()
            .take(OPEN_PASS_LANE_TARGET_SEARCH_LIMIT)
            .map(|(target_id, _)| target_id)
            .collect();

        let mut best: Option<OpenPassLaneDribblePlan> = None;
        for target_id in target_ids {
            let Some(target) = snapshot.players.iter().find(|player| {
                player.id == target_id && player.team == self.team && player.id != self.id
            }) else {
                continue;
            };
            if target.role == PlayerRole::Goalkeeper
                || snapshot
                    .pending_offside_for_pass(self.id, target.id)
                    .is_some()
            {
                continue;
            }
            let target_position = snapshot
                .player_position(target.id)
                .unwrap_or(target.position)
                .clamp_to_pitch(snapshot.field_width, snapshot.field_length);
            let pass_distance = current.distance(target_position);
            if !(5.0..=44.0).contains(&pass_distance) {
                continue;
            }
            let receiver_forward = (target_position.y - current.y) * attack_dir;
            if receiver_forward < -OPEN_PASS_LANE_MAX_BACKFIELD_RECEIVER_YARDS {
                continue;
            }

            let current_eval = open_pass_lane_eval_for_target(
                snapshot,
                me_snapshot,
                current,
                target,
                target_position,
            );
            if current_eval.receiver_openness < OPEN_PASS_LANE_MIN_RECEIVER_OPENNESS {
                continue;
            }
            let current_lane_already_safe = current_eval.clear_through
                && current_eval.blockers == 0
                && current_eval.risk <= OPEN_PASS_LANE_ALREADY_SAFE_RISK
                && current_eval.expected_completion >= OPEN_PASS_LANE_ALREADY_SAFE_COMPLETION;
            if current_lane_already_safe {
                continue;
            }
            let current_blocked_fit = if current_eval.clear_now {
                0.0_f64
            } else {
                0.72_f64
            }
            .max(((current_eval.risk - OPEN_PASS_LANE_ALREADY_SAFE_RISK) / 0.50).clamp(0.0, 1.0))
            .max((current_eval.blockers as f64 / 2.0).clamp(0.0, 1.0) * 0.72);
            let lane_blocker_extension = if current_eval.clear_now {
                0.0
            } else {
                (current_eval.blockers as f64 / 2.0).clamp(0.0, 1.0)
            };

            for dir in candidate_dirs {
                if dir.len() <= 1e-6 {
                    continue;
                }
                for distance in distances {
                    let raw_candidate = current + dir * distance;
                    let candidate =
                        raw_candidate.clamp_to_pitch(snapshot.field_width, snapshot.field_length);
                    let actual_distance = current.distance(candidate);
                    if !(OPEN_PASS_LANE_MIN_DRIBBLE_YARDS..=OPEN_PASS_LANE_MAX_DRIBBLE_YARDS + 1e-6)
                        .contains(&actual_distance)
                    {
                        continue;
                    }
                    let situation = open_pass_lane_situation_for_target(
                        snapshot,
                        self,
                        observation,
                        current,
                        candidate,
                        lane_blocker_extension,
                    );
                    let dynamic_max_yards = open_pass_lane_dynamic_max_dribble_yards(situation);
                    if actual_distance > dynamic_max_yards + 1e-6 {
                        continue;
                    }
                    let carrier_step_forward = (candidate.y - current.y) * attack_dir;
                    if carrier_step_forward < -OPEN_PASS_LANE_MAX_BACKWARD_STEP_YARDS {
                        continue;
                    }
                    let opponent_space =
                        snapshot.nearest_opponent_distance_at(self.team, candidate);
                    if opponent_space < OPEN_PASS_LANE_MIN_CANDIDATE_OPPONENT_SPACE {
                        continue;
                    }
                    let occupancy =
                        snapshot.candidate_occupancy_at(self.team, candidate, Some(self.id));
                    let teammate_penalty = occupancy.teammate_occupied_space_penalty(0.0);
                    if teammate_penalty >= 0.78 {
                        continue;
                    }

                    let candidate_eval = open_pass_lane_eval_for_target(
                        snapshot,
                        me_snapshot,
                        candidate,
                        target,
                        target_position,
                    );
                    let lane_gain = candidate_eval.lane_score - current_eval.lane_score;
                    let completion_gain =
                        candidate_eval.expected_completion - current_eval.expected_completion;
                    let risk_drop = current_eval.risk - candidate_eval.risk;
                    let blocker_drop = current_eval
                        .blockers
                        .saturating_sub(candidate_eval.blockers)
                        as f64;
                    let materially_opens = candidate_eval.clear_now
                        && (!current_eval.clear_now
                            || blocker_drop >= 1.0
                            || risk_drop >= OPEN_PASS_LANE_MIN_RISK_DROP);
                    if !materially_opens
                        && lane_gain < OPEN_PASS_LANE_MIN_LANE_GAIN
                        && completion_gain < OPEN_PASS_LANE_MIN_COMPLETION_GAIN
                        && risk_drop < OPEN_PASS_LANE_MIN_RISK_DROP
                    {
                        continue;
                    }
                    if candidate_eval.expected_completion < OPEN_PASS_LANE_MIN_NEW_COMPLETION
                        && !candidate_eval.clear_now
                    {
                        continue;
                    }

                    let distance_fit = if actual_distance <= OPEN_PASS_LANE_BASE_MAX_DRIBBLE_YARDS {
                        (1.0 - (actual_distance - 2.4).abs() / 2.3).clamp(0.42, 1.0)
                    } else {
                        let extra = ((actual_distance - OPEN_PASS_LANE_BASE_MAX_DRIBBLE_YARDS)
                            / (OPEN_PASS_LANE_MAX_DRIBBLE_YARDS
                                - OPEN_PASS_LANE_BASE_MAX_DRIBBLE_YARDS)
                                .max(1e-6))
                        .clamp(0.0, 1.0);
                        ((0.92 - extra * 0.20) * (0.70 + situation.extension * 0.36))
                            .clamp(0.38, 0.98)
                    };
                    let space_fit = (opponent_space / 7.0).clamp(0.0, 1.0);
                    let receiver_forward_fit = (receiver_forward / 26.0).clamp(0.0, 1.0);
                    let receiver_backward_penalty = ((-receiver_forward).max(0.0)
                        / OPEN_PASS_LANE_MAX_BACKFIELD_RECEIVER_YARDS)
                        .clamp(0.0, 1.0)
                        * 0.08;
                    let long_touch_penalty =
                        (actual_distance - OPEN_PASS_LANE_BASE_MAX_DRIBBLE_YARDS).max(0.0)
                            * 0.055
                            * (1.0 - situation.extension).clamp(0.0, 1.0);
                    let score = (0.10
                        + current_blocked_fit * 0.18
                        + lane_gain.max(0.0) * 0.56
                        + completion_gain.max(0.0) * 0.52
                        + risk_drop.max(0.0) * 0.28
                        + blocker_drop.min(3.0) * 0.08
                        + space_fit * 0.08
                        + receiver_forward_fit * 0.06
                        + situation.pressure.max(situation.tracking) * 0.05
                        - teammate_penalty * 0.20
                        - receiver_backward_penalty
                        - long_touch_penalty
                        - (-carrier_step_forward).max(0.0) * 0.08)
                        * distance_fit;
                    let score = score.clamp(0.0, 0.92);
                    if score <= 0.0 {
                        continue;
                    }
                    let Some(touch) =
                        open_pass_lane_touch_for_target(self.team, current, candidate)
                    else {
                        continue;
                    };
                    let kind = open_pass_lane_kind_for_target(self.team, current, candidate);
                    let plan = OpenPassLaneDribblePlan {
                        target: candidate,
                        kind,
                        touch,
                        score,
                    };
                    if best.is_none_or(|best| plan.score > best.score) {
                        best = Some(plan);
                    }
                }
            }
        }
        best
    }

    fn round_goalkeeper_dribble_plan_for(
        &self,
        snapshot: &WorldSnapshot,
        observation: &SoccerPomdpObservation,
    ) -> Option<RoundGoalkeeperDribblePlan> {
        if !observation.has_ball || self.role == PlayerRole::Goalkeeper {
            return None;
        }
        let current = snapshot
            .player_position(self.id)
            .unwrap_or(self.position)
            .clamp_to_pitch(snapshot.field_width, snapshot.field_length);
        let Some(me_snapshot) = snapshot.players.iter().find(|player| player.id == self.id) else {
            return None;
        };
        let attack_dir = self.team.attack_dir();
        if (current.y - snapshot.field_length * 0.5) * attack_dir < -2.0 {
            return None;
        }
        let goal_y = self.team.goal_y(snapshot.field_length);
        let yards_to_goal = (goal_y - current.y).abs();
        if !(ROUND_GOALKEEPER_MIN_YARDS_TO_GOAL..=ROUND_GOALKEEPER_MAX_YARDS_TO_GOAL)
            .contains(&yards_to_goal)
        {
            return None;
        }
        if observation.forward_dribble_space_yards < 0.75
            && observation.nearest_opponent_distance < 2.4
        {
            return None;
        }

        let goal_center = Vec2::new(snapshot.field_width * 0.5, goal_y);
        let keeper = snapshot
            .players
            .iter()
            .filter(|player| {
                player.team == self.team.other() && player.role == PlayerRole::Goalkeeper
            })
            .min_by(|a, b| {
                snapshot
                    .player_snapshot_position(a)
                    .distance(goal_center)
                    .total_cmp(&snapshot.player_snapshot_position(b).distance(goal_center))
            })?;
        let keeper_position = snapshot
            .player_snapshot_position(keeper)
            .clamp_to_pitch(snapshot.field_width, snapshot.field_length);
        let keeper_depth = (goal_y - keeper_position.y).abs();
        let direct_keeper_gap = segment_distance_to_point(current, goal_center, keeper_position);
        let keeper_blocks_line = (1.0 - direct_keeper_gap / 4.5).clamp(0.0, 1.0);
        let shot_block = observation.shot_block_probability.clamp(0.0, 1.0);
        let keeper_beat_problem =
            ((0.30 - observation.shot_beat_goalkeeper_probability) / 0.30).clamp(0.0, 1.0);
        let long_shot_need = ((yards_to_goal - 17.0)
            / (ROUND_GOALKEEPER_MAX_YARDS_TO_GOAL - 17.0).max(1.0))
        .clamp(0.0, 1.0);
        let direct_shot_problem = shot_block.max(keeper_blocks_line).max(keeper_beat_problem);
        let direct_shot_clean = observation.shot_lane_open
            && shot_block <= 0.22
            && observation.shot_beat_goalkeeper_probability >= 0.30
            && yards_to_goal <= ROUND_GOALKEEPER_LONG_SHOT_YARDS;
        if direct_shot_clean && direct_shot_problem < 0.30 {
            return None;
        }

        let direct_window = snapshot
            .shooting_window_score_at(me_snapshot, current)
            .clamp(0.0, 1.0);
        let side_preference = if (current.x - keeper_position.x).abs() <= 0.75 {
            if self.id % 2 == 0 {
                -1.0
            } else {
                1.0
            }
        } else {
            (current.x - keeper_position.x).signum()
        };
        let sides = [side_preference, -side_preference];
        let lateral_offsets = [2.8, 3.8, 5.0, 6.2];
        let depth_candidates = [
            (yards_to_goal - 3.6).max(ROUND_GOALKEEPER_MIN_TARGET_DEPTH_YARDS),
            (yards_to_goal - 5.6).max(ROUND_GOALKEEPER_MIN_TARGET_DEPTH_YARDS),
            (yards_to_goal - 7.2).max(ROUND_GOALKEEPER_MIN_TARGET_DEPTH_YARDS),
            (keeper_depth + 2.0).max(ROUND_GOALKEEPER_MIN_TARGET_DEPTH_YARDS),
            (keeper_depth - 1.4).max(ROUND_GOALKEEPER_MIN_TARGET_DEPTH_YARDS),
        ];

        let mut best: Option<RoundGoalkeeperDribblePlan> = None;
        for side in sides {
            for lateral_offset in lateral_offsets {
                let target_x = (keeper_position.x + side * lateral_offset)
                    .clamp(1.5, snapshot.field_width - 1.5);
                for target_depth in depth_candidates {
                    if target_depth >= yards_to_goal - 0.9 {
                        continue;
                    }
                    let candidate = Vec2::new(target_x, goal_y - attack_dir * target_depth)
                        .clamp_to_pitch(snapshot.field_width, snapshot.field_length);
                    let actual_distance = current.distance(candidate);
                    if !(ROUND_GOALKEEPER_MIN_DRIBBLE_YARDS
                        ..=ROUND_GOALKEEPER_MAX_DRIBBLE_YARDS + 1e-6)
                        .contains(&actual_distance)
                    {
                        continue;
                    }
                    let goalward_gain = (candidate.y - current.y) * attack_dir;
                    if goalward_gain < 1.0 {
                        continue;
                    }
                    let target_yards_to_goal = (goal_y - candidate.y).abs();
                    if target_yards_to_goal < ROUND_GOALKEEPER_MIN_TARGET_DEPTH_YARDS
                        || target_yards_to_goal >= yards_to_goal - 0.75
                    {
                        continue;
                    }
                    let keeper_clearance = candidate.distance(keeper_position);
                    if keeper_clearance < ROUND_GOALKEEPER_MIN_KEEPER_CLEARANCE_YARDS {
                        continue;
                    }
                    let path_keeper_gap =
                        segment_distance_to_point(current, candidate, keeper_position);
                    if path_keeper_gap < ROUND_GOALKEEPER_MIN_PATH_KEEPER_GAP_YARDS {
                        continue;
                    }
                    let opponent_space =
                        snapshot.nearest_opponent_distance_at(self.team, candidate);
                    if opponent_space < ROUND_GOALKEEPER_MIN_CANDIDATE_OPPONENT_SPACE {
                        continue;
                    }
                    let occupancy =
                        snapshot.candidate_occupancy_at(self.team, candidate, Some(self.id));
                    let teammate_penalty = occupancy.teammate_occupied_space_penalty(0.0);
                    if teammate_penalty >= 0.76 {
                        continue;
                    }

                    let candidate_window = snapshot
                        .shooting_window_score_at(me_snapshot, candidate)
                        .clamp(0.0, 1.0);
                    let window_gain = candidate_window - direct_window;
                    let candidate_keeper_gap =
                        segment_distance_to_point(candidate, goal_center, keeper_position);
                    let keeper_gap_gain =
                        ((candidate_keeper_gap - direct_keeper_gap) / 4.5).clamp(-1.0, 1.0);
                    let angle_gain = ((snapshot.goal_angle_degrees(candidate, self.team)
                        - observation.opponent_goal_angle_degrees)
                        / 30.0)
                        .clamp(-1.0, 1.0);
                    let material_improvement = window_gain >= 0.035
                        || keeper_gap_gain >= 0.18
                        || (long_shot_need >= 0.48 && goalward_gain >= 3.0);
                    if !material_improvement {
                        continue;
                    }

                    let distance_fit = (1.0 - (actual_distance - 4.8).abs() / 4.8).clamp(0.38, 1.0);
                    let depth_gain =
                        (goalward_gain / ROUND_GOALKEEPER_MAX_DRIBBLE_YARDS).clamp(0.0, 1.0);
                    let space_fit = (opponent_space / 7.0).clamp(0.0, 1.0);
                    let skill_fit = (ability01(self.skills.dribbling) * 0.64
                        + ability01(self.skills.first_touch) * 0.20
                        + ability01(self.skills.acceleration) * 0.16)
                        .clamp(0.0, 1.0);
                    let round_keeper_fit =
                        (1.0 - (target_yards_to_goal - keeper_depth).abs() / 8.0).clamp(0.0, 1.0);
                    let lane_clear_bonus =
                        if snapshot.clear_line(candidate, goal_center, self.team.other(), 1.35) {
                            0.06
                        } else {
                            0.0
                        };
                    let mut score = (0.14
                        + long_shot_need * 0.22
                        + direct_shot_problem * 0.20
                        + window_gain.max(0.0) * 0.58
                        + keeper_gap_gain.max(0.0) * 0.25
                        + angle_gain.max(0.0) * 0.10
                        + depth_gain * 0.16
                        + space_fit * 0.06
                        + skill_fit * 0.10
                        + round_keeper_fit * 0.05
                        + lane_clear_bonus
                        + observation.offensive_urgency.clamp(0.0, 1.0) * 0.06
                        - teammate_penalty * 0.18
                        - (actual_distance - 6.0).max(0.0) * 0.018)
                        * distance_fit;
                    if yards_to_goal >= ROUND_GOALKEEPER_LONG_SHOT_YARDS && window_gain >= 0.02 {
                        score = score.max(0.88 + long_shot_need * 0.08);
                    }
                    if goal_attack_shot_blocks_alternatives(observation, self.role)
                        && shot_block >= 0.42
                        && window_gain >= 0.06
                    {
                        score = score.max(0.985);
                    }
                    let score = score.clamp(0.0, 0.995);
                    if score <= 0.0 {
                        continue;
                    }
                    let Some(touch) =
                        open_pass_lane_touch_for_target(self.team, current, candidate)
                    else {
                        continue;
                    };
                    let kind = round_goalkeeper_kind_for_target(self.team, current, candidate);
                    let sprint = observation.perceived_pressure >= 0.30
                        || observation.forward_dribble_space_yards >= 3.0
                        || long_shot_need >= 0.50;
                    let plan = RoundGoalkeeperDribblePlan {
                        target: candidate,
                        kind,
                        touch,
                        score,
                        sprint,
                    };
                    if best.is_none_or(|best| plan.score > best.score) {
                        best = Some(plan);
                    }
                }
            }
        }
        best
    }

    pub(crate) fn possession_action_options(
        &self,
        observation: &SoccerPomdpObservation,
        directive: &TeamTacticalDirective,
        pass_target_count: usize,
        aerial_pass_target_count: usize,
        hold_up_flank_available: bool,
        dt_seconds: f64,
        field_width_yards: f64,
    ) -> Vec<AgentActionOptionTrace> {
        let shooting = ability01(self.skills.shooting);
        let dribbling = ability01(self.skills.dribbling);
        let passing = ability01(self.skills.passing_completion_rate);
        let crossing = ability01(self.skills.crossing_left.max(self.skills.crossing_right));
        let pressure = observation.perceived_pressure.clamp(0.0, 1.0);
        let pressure_urgency = observation.pressure_urgency.clamp(0.0, 1.0);
        // d(pressure)/dt: a defender closing fast. Drives "release sooner" (via
        // excessive_hold) AND "body-shield now" (the protect-ball boost below).
        let pressure_rising = pressure_rising_signal(observation);
        // In our own half, retention (shielding / safe dribbling) takes priority over
        // risky attacking dribbling — keep the ball rather than forcing it forward.
        let own_half = observation.yards_to_own_goal < observation.yards_to_goal;
        let field_width = if field_width_yards.is_finite() && field_width_yards > 0.0 {
            field_width_yards
        } else {
            DEFAULT_FIELD_WIDTH_YARDS
        };
        let lateral_position = self.position.x.clamp(0.0, field_width);
        let flank_lane_fit = ((lateral_position - field_width * 0.5).abs()
            / (field_width * 0.5).max(1.0))
        .clamp(0.0, 1.0);
        let outside_midfielder = self.role == PlayerRole::Midfielder
            && (self.home_position.x < field_width * 0.30
                || self.home_position.x > field_width * 0.70);
        let eligible_byline_carrier = self.role == PlayerRole::Forward || outside_midfielder;
        let byline_drive_active = eligible_byline_carrier
            && flank_lane_fit >= BYLINE_DRIVE_MIN_WIDTH_FROM_CENTER
            && is_byline_cross_drive_strategy(directive.attack_strategy);
        let byline_cross_release = eligible_byline_carrier
            && flank_lane_fit >= BYLINE_DRIVE_MIN_WIDTH_FROM_CENTER
            && directive.attack_strategy == TeamAttackStrategy::CrashTheBox
            && directive.flank_attack_policy.is_flank();
        // A ball-carrying keeper under proximate pressure should strongly prefer a
        // release, but it is no longer illegal to carry with the feet. The risk is
        // modeled as a score damp on carry/dribble options plus a lift to release
        // options, so rare high-value carries can still survive the policy.
        let keeper_release_pressure = if self.role == PlayerRole::Goalkeeper
            && observation.nearest_opponent_distance.is_finite()
        {
            ((GOALKEEPER_RELEASE_PRESSURE_RADIUS_YARDS + 3.0
                - observation.nearest_opponent_distance)
                / 3.0)
                .clamp(0.0, 1.0)
        } else {
            0.0
        };
        let keeper_carry_under_pressure_damp =
            (1.0 - keeper_release_pressure * 0.88).clamp(0.10, 1.0);
        let keeper_release_under_pressure_lift =
            (1.0 + keeper_release_pressure * 0.34).clamp(1.0, 1.34);
        let offensive_urgency = observation.offensive_urgency.clamp(0.0, 1.0);
        let defensive_urgency = observation.defensive_urgency.clamp(0.0, 1.0);
        let decision_urgency = observation.decision_urgency.clamp(0.0, 1.0);
        let goal_attack = observation.goal_attack_window_score.clamp(0.0, 1.0);
        let goal_entry_pressure = goal_entry_decisive_pressure_score(observation);
        let shot_choice_learning =
            tactical_learning_multiplier(directive.shot_choice_learning_weight, 0.28);
        let goal_entry_pass_learning =
            tactical_learning_multiplier(directive.goal_entry_pass_learning_weight, 0.34);
        let pressure_release_learning =
            tactical_learning_multiplier(directive.pressure_release_learning_weight, 0.36);
        let speculative_long_shot =
            speculative_long_shot_is_qualified(observation, self.role, shooting);
        let contested_final_third_shot =
            contested_final_third_shot_is_qualified(observation, self.role, shooting);
        let shot_legal = shot_decision_is_qualified_for_role(observation, self.role)
            || speculative_long_shot
            || contested_final_third_shot;
        let shot_quality_weight = (observation.shot_on_frame_probability * 0.72
            + observation.shot_beat_goalkeeper_probability * 0.48
            + observation.shot_curl_probability * 0.12)
            .clamp(0.0, 1.25);
        let shot_block_penalty =
            (1.0 - observation.shot_block_probability.clamp(0.0, 1.0) * 0.58).clamp(0.30, 1.0);
        let goal_attack_shot_blocks_alternatives =
            goal_attack_shot_blocks_alternatives(observation, self.role);
        let threaded_goal_pass_overrides_shot =
            threaded_goal_pass_can_override_forced_shot(observation, self.role);
        let close_shot_attempt =
            close_clear_shot_attempt_probability(observation, self.role, shooting);
        let striker_shot_bonus = striker_legal_shot_attempt_bonus(observation, self.role);
        let goal_proximity_shot_pressure =
            goal_proximity_shot_pressure_score(observation, self.role, shooting);
        let decisive_goal_pressure = observation
            .decisive_goal_action_pressure
            .max(near_goal_decisive_action_pressure_score(
                observation,
                self.role,
            ))
            .clamp(0.0, 1.0);
        let final_ball_goal_decision_fit = goal_entry_pressure
            .max(goal_proximity_shot_pressure)
            .max(decisive_goal_pressure * 0.72)
            .clamp(0.0, 1.0);
        let shot_score_base = (self.preferences.shoot_bias
            * (0.52 + shooting * 0.62)
            * (1.0 + directive.risk_tolerance * 0.35)
            * (0.78 + (observation.opponent_goal_angle_degrees / 42.0).clamp(0.0, 1.0) * 0.44)
            // Acute angles are near-impossible to score from: drive shot value to
            // zero as the goal-mouth angle closes below ~30 degrees (e.g. by the
            // byline or at a corner), forcing a cutback/pass instead of a shot.
            * (observation.opponent_goal_angle_degrees / LOW_SHOT_ANGLE_CUTOFF_DEGREES)
                .clamp(0.0, 1.0)
            * (0.34 + shot_quality_weight)
            * shot_block_penalty
            * (1.0 + offensive_urgency * 2.45 + pressure_urgency * 0.42)
            * (1.0 + goal_attack * 1.36)
            * (1.0 + goal_proximity_shot_pressure * 1.55)
            * (1.0 + goal_entry_pressure * 0.74)
            * (1.0 + striker_shot_bonus * 1.35)
            * (1.0 + decisive_goal_pressure * 1.18)
            * (1.0 + final_ball_goal_decision_fit * 0.42)
            * shot_choice_learning
            * 0.042)
            .clamp(
                0.004,
                (0.12
                    + offensive_urgency * 0.30
                    + striker_shot_bonus * 0.18
                    + goal_attack * 0.22
                    + goal_proximity_shot_pressure * 0.36
                    + goal_entry_pressure * 0.20
                    + decisive_goal_pressure * 0.30)
                    * shot_choice_learning.clamp(0.82, 1.24),
            )
            .max(close_shot_attempt * shot_choice_learning.clamp(0.88, 1.16))
            .max(
                goal_proximity_shot_pressure_floor(observation, self.role, shooting)
                    * shot_choice_learning.clamp(0.88, 1.16),
            )
            .max(
                attacking_goal_pressure_shot_attempt_probability(observation, self.role, shooting)
                    * shot_choice_learning.clamp(0.88, 1.16),
            )
            .max(
                speculative_long_shot_attempt_probability(observation, self.role, shooting)
                    * shot_choice_learning,
            )
            .max(
                contested_final_third_shot_attempt_probability(observation, self.role, shooting)
                    * shot_choice_learning,
            )
            .max(striker_shot_bonus * shot_choice_learning.clamp(0.88, 1.16))
            .max(if goal_attack_shot_blocks_alternatives {
                0.99 * shot_choice_learning.clamp(0.90, 1.10)
            } else {
                0.0
            });
        // Learnable shot-trigger MDP/POMDP: scale the whole shot score (floors included)
        // by the value of shooting NOW. This is what fixes "shoots from 25 when he should
        // work it to 20/15": a far, covered, low-value chance is suppressed toward ~0.2×
        // while a close, open, high-value one is lifted toward ~1.35×. Default-ON; the
        // kill-switch (`DD_SOCCER_DISABLE_SHOT_TRIGGER_MDP`) skips this entirely ⇒
        // byte-identical. The MPC then still re-checks executability and can veto/reselect.
        let shot_trigger_value = if shot_trigger_mdp_enabled() {
            Some(if observation.shot_trigger_mdp_value >= 0.0 {
                // Populated by the snapshot seam (trained head or analytic seed).
                observation.shot_trigger_mdp_value
            } else {
                // Observation built without the snapshot seam (e.g. a direct unit test):
                // recompute the analytic seed so the discipline still applies.
                analytic_shot_trigger_value(&ShotTriggerInputs::from_observation(
                    observation,
                    self.role,
                    shooting,
                    1.0,
                ))
            })
        } else {
            None
        };
        // The shot stays a LEGAL option from range (a human / genuine speculative chance can
        // still take it) — the discipline is to make the AI not *volunteer* low-value long
        // shots. Inside range, a mild value-scaled demotion; beyond
        // `shooting.shot_trigger_long_range_yards` with a low MDP value (a covered long shot), the
        // score is crushed below the carrying/passing alternatives so the carrier works it
        // closer — but `shot_legal` is untouched, so it stays a selectable option.
        let shot_trigger_long_range_yards = tunables().shooting.shot_trigger_long_range_yards;
        let shot_trigger_long_range_min_value =
            tunables().shooting.shot_trigger_long_range_min_value;
        let shot_score = match shot_trigger_value {
            Some(v)
                if observation.yards_to_goal > shot_trigger_long_range_yards
                    && v < shot_trigger_long_range_min_value =>
            {
                shot_score_base.min(tunables().shooting.shot_trigger_volunteer_floor)
            }
            Some(v) => shot_score_base * shot_trigger_score_multiplier(v),
            None => shot_score_base,
        };
        // Whether the stop-volunteering discipline vetoes a long shot here. Also used below
        // to skip the near-goal shoot *floors* (which would otherwise re-boost shoot above
        // the crush and defeat the discipline). `false` when the kill-switch is set.
        let shot_trigger_long_range_veto = matches!(
            shot_trigger_value,
            Some(v) if observation.yards_to_goal > shot_trigger_long_range_yards
                && v < shot_trigger_long_range_min_value
        );
        let fatigue_dribble = fatigue_dribble_multiplier(observation);
        let shot_creation_carry = shot_creation_carry_multiplier(observation);
        let patience_factor = low_pressure_patience_factor(observation);
        let floor_pass_quality = pass_quality_for_patience(observation, PassFlight::Floor);
        let aerial_pass_quality = pass_quality_for_patience(observation, PassFlight::Aerial);
        let pressured_release_signal =
            (pressure_release_signal(observation) * pressure_release_learning).clamp(0.0, 1.0);
        let pressured_release_multiplier = (1.0
            + (pressured_release_multiplier(observation) - 1.0) * pressure_release_learning)
            .clamp(1.0, 2.72);
        let open_support_fit = open_support_outlet_fit(observation);
        let pressured_good_outlet = (pressured_release_signal
            * (floor_pass_quality.max(aerial_pass_quality * 0.86) + open_support_fit * 0.18))
            .clamp(0.0, 1.0);
        let hold_penalty_multiplier = dribble_hold_score_multiplier(observation, dribbling);
        let mut hold_release_multiplier = release_after_hold_multiplier(observation, dribbling);
        let poor_floor_pass = (1.0 - floor_pass_quality).clamp(0.0, 1.0);
        let forward_space_fit = (observation.forward_dribble_space_yards / 18.0).clamp(0.0, 1.0);
        let patient_dribble_lift = (1.0
            + patience_factor * poor_floor_pass * (0.42 + forward_space_fit * 0.50))
            .clamp(1.0, 1.38);
        let goal_approach_window = goal_approach_carry_window_yards(self.role);
        let goal_approach_fit = if observation.yards_to_goal <= goal_approach_window {
            ((goal_approach_window - observation.yards_to_goal)
                / (goal_approach_window - TEAMMATE_MUST_SHOOT_YARDS).max(1.0))
            .clamp(0.0, 1.0)
        } else {
            0.0
        };
        let goalmouth_carry_bias = (goal_approach_fit
            * (observation.forward_dribble_space_yards / 10.0).clamp(0.0, 1.0))
        .clamp(0.0, 1.0);
        let goalmouth_carry_forced = goal_approach_forces_goalmouth_carry(observation, self.role);
        let patient_carry_multiplier = low_pressure_patient_carry_multiplier(observation);
        let striker_carry_boost = if self.role == PlayerRole::Forward && pass_target_count == 0 {
            1.48
        } else if self.role == PlayerRole::Forward {
            1.24
        } else {
            1.0
        };
        // Don't dribble straight into an opponent for no reason: keep 2yd where possible,
        // 3yd in our own half, and relax only when the nearest opponent is moving away.
        let dribble_into_opponent_penalty = dribble_into_opponent_penalty(
            observation.forward_dribble_space_yards,
            observation.yards_to_goal,
            observation.yards_to_own_goal,
            observation
                .neural_extended
                .nearest_opponent_closing_rate_yps,
        );
        // A clearly-open teammate ahead is an outlet the carrier should USE rather than
        // keep dribbling — even with time on the ball (the "dribbles too long while a
        // forward runner is open" complaint). Gated on a forward option actually existing
        // so a lone striker isn't told to stop carrying.
        let open_forward_outlet = if observation.visible_forward_pass_options > 0 {
            observation
                .best_forward_pass_receiver_openness
                .clamp(0.0, 1.0)
        } else {
            0.0
        };
        let short_upfield_ground_pass_fit = short_upfield_ground_pass_fit(observation);
        let short_upfield_ground_pass_multiplier = (1.0
            + short_upfield_ground_pass_fit
                * (0.22
                    + decision_urgency * 0.14
                    + pressured_release_signal * 0.18
                    + open_forward_outlet * 0.18))
            .clamp(1.0, 1.62);
        // Critical spacing: most carriers react inside the 2-yard floor; everyone
        // wants 3yd in their own half, and defenders keep the wider 3-4yd cushion.
        // Moving-away opponents are less urgent. The response stays soft: dribbling
        // is damped and passing lifted rather than making carry options illegal.
        let defender_crowding = dribble_spacing_pressure_for_state(
            self.role,
            own_half,
            observation.nearest_opponent_distance,
            observation
                .neural_extended
                .nearest_opponent_closing_rate_yps,
        );
        let crowded_dribble_damp =
            (1.0 - defender_crowding * DRIBBLE_CROWDED_SPACE_DAMP).clamp(0.30, 1.0);
        // The flip side: when a defender closes the spacing gap AND a receiver is open,
        // lift passing so the carrier releases the ball before the gap collapses. For
        // defenders in possession, a fast-closing opponent increases that release lift
        // even before the gap has dropped below the 2-yard floor.
        let open_outlet_fit = observation.best_pass_receiver_openness.clamp(0.0, 1.0);
        let steal_pressure = pressure_urgency
            .max(pressure)
            .max(observation.immediate_dispossession_risk)
            .max(pressure_rising * 0.90)
            .max(defender_crowding)
            .clamp(0.0, 1.0);
        let good_escape_outlet_fit = floor_pass_quality
            .max(aerial_pass_quality * 0.86)
            .max(open_outlet_fit)
            .max(open_support_fit * 0.72)
            .clamp(0.0, 1.0);
        let bad_escape_outlet_fit = (1.0 - good_escape_outlet_fit).clamp(0.0, 1.0);
        let steal_risk_escape_lift = (steal_pressure * bad_escape_outlet_fit).clamp(0.0, 1.0);
        let steal_risk_good_outlet_pass_lift = (1.0
            + steal_pressure * good_escape_outlet_fit * STEAL_RISK_GOOD_OUTLET_PASS_LIFT)
            .clamp(1.0, 1.90);
        let panic_pass_damp = (1.0
            / (1.0
                + steal_pressure
                    * bad_escape_outlet_fit.powi(2)
                    * STEAL_RISK_BAD_OUTLET_PANIC_PASS_DAMP_STRENGTH))
            .clamp(0.14, 1.0);
        let closing_good_outlet_pass_lift =
            (1.0 + pressure_rising * open_outlet_fit * 0.44).clamp(1.0, 1.38);
        let defender_closing_pass_lift = if self.role == PlayerRole::Defender {
            (1.0 + pressure_rising * DEFENDER_DRIBBLE_CLOSING_PASS_LIFT * open_outlet_fit)
                .clamp(1.0, 1.42)
        } else {
            1.0
        };
        let crowded_pass_lift = (1.0
            + defender_crowding * PASS_CROWDED_RELEASE_LIFT * open_outlet_fit)
            * defender_closing_pass_lift
            * closing_good_outlet_pass_lift
            * steal_risk_good_outlet_pass_lift;
        // --- Positional dribble-risk appetite -----------------------------------------
        // Driving INTO pressure (taking a man on, carrying into traffic) is only worth the
        // risk for the most-advanced players — the three furthest forward. See
        // `advanced_dribbler_fit_from_rank` for the rank→appetite curve.
        let advanced_dribbler_fit = advanced_dribbler_fit_from_rank(observation.teammates_ahead);
        // How risky a dribble is right now: real pressure, a defender on top of the ball, or
        // a live chance of being dispossessed.
        let dribble_risk = steal_pressure
            .max(observation.perceived_pressure)
            .clamp(0.0, 1.0);
        // Deep carriers pay an escalating penalty for dribbling while at risk; the three
        // most-forward players keep their full take-on appetite (fit = 1 ⇒ no damp).
        let deep_pressure_dribble_damp =
            (1.0 - dribble_risk * (1.0 - advanced_dribbler_fit) * 0.62).clamp(0.40, 1.0);
        // ...and a deep carrier should give the ball up sooner the longer that risk persists,
        // so the release (pass/shot) gets a positional urgency boost on top of the existing
        // hold-time model. For defenders the wider comfort buffer already feeds this via
        // `defender_crowding` → `dribble_risk`, so no extra term is needed here.
        let deep_release_urgency =
            (1.0 + dribble_risk * (1.0 - advanced_dribbler_fit) * 0.34).clamp(1.0, 1.45);
        // Stack the positional/keeper release lifts onto the hold-time release model, but
        // cap the product so several "release sooner" signals firing at once (e.g. a
        // pressured keeper, who is also a deep carrier) can't blow the multiplier past a
        // sane ceiling. Downstream pass/shot caps re-clamp it too, so this is a guardrail.
        hold_release_multiplier =
            (hold_release_multiplier * deep_release_urgency * keeper_release_under_pressure_lift)
                .clamp(1.0, 3.6);
        let pre_fatigue_dribble_score = (self.preferences.dribble_bias
            * (0.62 + dribbling * 0.48)
            * directive.carry_priority
            * (0.70 + (observation.forward_dribble_space_yards / 18.0).clamp(0.0, 1.0) * 0.58)
            * shot_creation_carry
            * striker_carry_boost
            // The pressure-driven dribble lift ("take him on now") only applies to the
            // most-advanced players; for everyone else mounting pressure must not make
            // dribbling MORE attractive.
            * (1.0 + offensive_urgency * 0.30 + pressure_urgency * 0.20 * advanced_dribbler_fit)
            * (1.0 - pressured_release_signal * open_support_fit * 0.18).clamp(0.70, 1.0)
            * (1.0 - pressured_good_outlet * 0.30).clamp(0.62, 1.0)
            * (1.0 - (open_forward_outlet * 0.34 + short_upfield_ground_pass_fit * 0.18))
                .clamp(0.44, 1.0)
            * dribble_into_opponent_penalty
            * deep_pressure_dribble_damp
            * crowded_dribble_damp)
            .clamp(0.02, 1.36);
        let dribble_score = (pre_fatigue_dribble_score
            * fatigue_dribble
            * patient_dribble_lift
            * hold_penalty_multiplier
            * keeper_carry_under_pressure_damp)
            .clamp(0.02, 1.75);
        let patient_carry_score_base = (dribble_score * patient_carry_multiplier).clamp(0.02, 1.58);
        // Central defenders must stay back: they do NOT dribble the ball forward when
        // an opponent is within ~5 yds in front of them (little open space ahead).
        // Wing-backs / full-backs are exempt — they are the ones meant to push on.
        let is_central_defender = self.role == PlayerRole::Defender
            && (self.home_position.x - field_width_yards * 0.5).abs()
                <= field_width_yards * CENTRAL_DEFENDER_LANE_HALF_FRACTION;
        let central_defender_forward_blocked = is_central_defender
            && observation.forward_dribble_space_yards
                < CENTRAL_DEFENDER_FORWARD_DRIBBLE_MIN_SPACE_YARDS;
        let carry_forward_min_space =
            if observation.yards_to_goal <= DRIBBLE_FINAL_THIRD_YARDS_TO_GOAL {
                1.2
            } else {
                DRIBBLE_OPEN_PLAY_MIN_FORWARD_SPACE_YARDS
            };
        let carry_forward_legal = (observation.forward_dribble_space_yards
            >= carry_forward_min_space
            || goalmouth_carry_forced)
            && !goal_attack_shot_blocks_alternatives
            && !central_defender_forward_blocked;
        let carry_forward_score = (patient_carry_score_base
            * (0.34 + patience_factor * 0.62 + poor_floor_pass * 0.36 + forward_space_fit * 0.34)
            * (1.0
                + goalmouth_carry_bias * 0.78
                + if goalmouth_carry_forced { 0.42 } else { 0.0 })
            * (1.0 + goal_attack * 0.30)
            * (1.0 - pressure * 0.24).clamp(0.70, 1.0)
            * (1.0 - pressured_good_outlet * 0.44).clamp(0.50, 1.0)
            // Carrying forward INTO pressure is the worst option for a deep player — damp it
            // hard for anyone outside the three most-forward when there's real risk ahead.
            * (1.0 - dribble_risk * (1.0 - advanced_dribbler_fit) * 0.40).clamp(0.45, 1.0)
            // Own-half retention: risky forward dribbling is de-emphasised in our own
            // half (more so when pressured) in favour of keeping the ball.
            * if own_half { (1.0 - 0.24 - pressure * 0.18).clamp(0.55, 1.0) } else { 1.0 }
            * keeper_carry_under_pressure_damp)
            .clamp(0.01, 1.46)
            .max(if byline_drive_active {
                (1.10 + forward_space_fit * 0.18 + crossing * 0.12).clamp(1.10, 1.38)
            } else {
                0.0
            });
        let vertical_attack_legal = carry_forward_legal
            && self.role != PlayerRole::Goalkeeper
            && observation.yards_to_goal < observation.yards_to_own_goal
            && observation.forward_dribble_space_yards >= 2.0
            && !goal_attack_shot_blocks_alternatives;
        let vertical_attack_score = (carry_forward_score
            * (0.62
                + offensive_urgency * 0.42
                + goal_attack * 0.34
                + forward_space_fit * 0.30
                + self.preferences.offensive_mindedness.clamp(0.0, 1.0) * 0.18)
            * (1.0 - pressure * 0.16).clamp(0.76, 1.0))
        .clamp(0.01, 1.48);
        let fresh_turnover_fit =
            (1.0 - observation.perceived_time_on_ball_seconds / 0.85).clamp(0.0, 1.0);
        let turnover_burst_legal = carry_forward_legal
            && fresh_turnover_fit > 0.0
            && observation.forward_dribble_space_yards >= 1.0
            && self.role != PlayerRole::Goalkeeper;
        let turnover_burst_score = (carry_forward_score
            * (0.52
                + fresh_turnover_fit * 0.76
                + pressure_rising * 0.22
                + forward_space_fit * 0.26
                + offensive_urgency * 0.16)
            * (1.0 + ability01(self.skills.acceleration) * 0.24))
            .clamp(0.01, 1.60);
        let carry_out_legal = observation.forward_dribble_space_yards >= 0.8
            && !goal_attack_shot_blocks_alternatives
            && !goalmouth_carry_forced;
        let carry_out_score = (patient_carry_score_base
            * (0.20
                + patience_factor * 0.42
                + poor_floor_pass * 0.30
                + (1.0 - observation.best_pass_receiver_openness.clamp(0.0, 1.0)) * 0.20)
            * (1.0 - goalmouth_carry_bias * 0.52 - goal_attack * 0.34).clamp(0.30, 1.0)
            * (1.0 - pressured_good_outlet * 0.36).clamp(0.55, 1.0)
            // Escaping pressure sideways / out is exactly what a pressured deep carrier SHOULD
            // do (dribble OUT of trouble, not into it) — lift it relative to carrying forward.
            * (1.0 + dribble_risk * (1.0 - advanced_dribbler_fit) * 0.42).clamp(1.0, 1.50)
            * keeper_carry_under_pressure_damp)
            .clamp(0.01, 0.92);
        // Drive-to-corner commitment: once the team has committed to a byline drive (the wide
        // carrier in the opponent half), LIFT the forward/wide carry so the carrier keeps driving
        // the corner instead of slowing 32-38yd out — the carry itself is steered at the corner
        // flag by `byline_corner_drive_target_for`, and hands off to the cross at the byline.
        let byline_drive_boost = if directive_requests_byline_drive(directive)
            && observation.yards_to_goal < observation.yards_to_own_goal
            && matches!(self.role, PlayerRole::Forward | PlayerRole::Midfielder)
        {
            1.0 + observation.offensive_urgency.clamp(0.0, 1.0) * 0.14 + 0.30
        } else {
            1.0
        };
        let carry_forward_score = (carry_forward_score * byline_drive_boost).clamp(0.01, 1.85);
        let carry_out_score = (carry_out_score * byline_drive_boost).clamp(0.01, 1.30);
        let field_width = if field_width_yards.is_finite() && field_width_yards > 0.0 {
            field_width_yards
        } else {
            DEFAULT_FIELD_WIDTH_YARDS
        };
        let lateral_position = self.position.x.clamp(0.0, field_width);
        let left_room = if self.team == Team::Home {
            lateral_position
        } else {
            field_width - lateral_position
        };
        let right_room = if self.team == Team::Home {
            field_width - lateral_position
        } else {
            lateral_position
        };
        let flank_lane_fit = ((lateral_position - field_width * 0.5).abs()
            / (field_width * 0.5).max(1.0))
        .clamp(0.0, 1.0);
        let flank_policy_active = directive.flank_attack_policy.is_flank();
        let flank_drive_multiplier = if flank_policy_active {
            (1.0 + (1.0 - flank_lane_fit) * 0.26 + directive.flank_overlap_run_probability * 0.20)
                .clamp(1.0, 1.48)
        } else {
            1.0
        };
        let low_cross_multiplier = if directive.flank_attack_policy.prefers_low_cross() {
            (1.0 + flank_lane_fit * 0.34
                + crossing * 0.16
                + directive.flank_overlap_run_probability * 0.18)
                .clamp(1.0, 1.72)
        } else if flank_policy_active {
            (1.0 + flank_lane_fit * 0.08).clamp(1.0, 1.14)
        } else {
            1.0
        };
        let high_cross_multiplier = if directive.flank_attack_policy.prefers_high_cross() {
            (1.0 + flank_lane_fit * 0.42
                + crossing * 0.20
                + directive.flank_overlap_run_probability * 0.18)
                .clamp(1.0, 1.84)
        } else if flank_policy_active {
            (1.0 + flank_lane_fit * 0.10).clamp(1.0, 1.16)
        } else {
            1.0
        };
        // Forward path blocked (a defender goal-side) ⇒ low forward space. Under
        // pressure that calls for an ACTIVE escape: carry/side-step AWAY into space,
        // dragging the ball out of the locking zone away from the defender, rather
        // than holding still and feeding a ping-pong. `escape_urgency` scales those
        // away-moves; it collapses to ~0 in open play so normal carrying is untouched.
        // BUT never when a goalmouth carry is forced — a deep-and-wide attacker with a
        // clear mandate to drive at goal must take the ball IN, not shield or peel away
        // (escape/shield are anti-progression; the goalmouth drive is the whole point).
        let forward_blocked = if goalmouth_carry_forced {
            0.0
        } else {
            (1.0 - observation.forward_dribble_space_yards / 6.0).clamp(0.0, 1.0)
        };
        let escape_urgency = (pressure_urgency.max(pressure) * forward_blocked).clamp(0.0, 1.0);
        let steal_escape_urgency = escape_urgency.max(steal_risk_escape_lift).clamp(0.0, 1.0);
        let escape_lane_fit = (observation.pressured_escape_lane_yards
            / FIRST_TOUCH_ESCAPE_LANE_MIN_YARDS)
            .clamp(0.0, 1.0);
        let slow_holder_fit =
            (1.0 - observation.actor_speed_yps / PRESSURED_SLOW_HOLDER_SPEED_YPS).clamp(0.0, 1.0);
        let slow_holder_escape_signal = if dd_soccer_disable_first_touch_escape() {
            0.0
        } else {
            let close_pressure =
                pressure_urgency
                    .max(pressure)
                    .max(pressure_from_nearest_distance(
                        observation.nearest_opponent_distance,
                    ));
            (slow_holder_fit
                * close_pressure
                * escape_lane_fit
                * (0.42 + 0.58 * forward_blocked.max(steal_risk_escape_lift)))
            .clamp(0.0, 1.0)
        };
        // Lift the away-carry ceiling under escape pressure so a decisive break into
        // lateral space can outscore options that keep the holder pinned in the duel.
        let carry_out_escape_ceiling =
            (0.96 + steal_escape_urgency * 0.26 + slow_holder_escape_signal * 0.36)
                .clamp(0.96, 1.36);
        let carry_out_left_score = (carry_out_score
            * flank_drive_multiplier
            * (0.76 + (left_room / 18.0).clamp(0.0, 1.0) * 0.32)
            * (1.0
                + steal_escape_urgency * (left_room / 12.0).clamp(0.0, 1.0) * 0.66
                + slow_holder_escape_signal * (left_room / 10.0).clamp(0.0, 1.0) * 0.70))
            .clamp(0.01, carry_out_escape_ceiling);
        let carry_out_right_score = (carry_out_score
            * flank_drive_multiplier
            * (0.76 + (right_room / 18.0).clamp(0.0, 1.0) * 0.32)
            * (1.0
                + steal_escape_urgency * (right_room / 12.0).clamp(0.0, 1.0) * 0.66
                + slow_holder_escape_signal * (right_room / 10.0).clamp(0.0, 1.0) * 0.70))
            .clamp(0.01, carry_out_escape_ceiling);
        let carry_out_left_legal = carry_out_legal && left_room > 2.0;
        let carry_out_right_legal = carry_out_legal && right_room > 2.0;
        let protect_ball_legal = observation.nearest_opponent_distance <= 5.2
            || pressure_urgency.max(pressure) >= 0.24
            // A fast-closing defender makes shielding legal a touch earlier (before
            // they're already on top of you) so the body gets between in time.
            || (pressure_rising >= 0.30 && observation.nearest_opponent_distance <= 7.5);
        // Body-shield to SECURE possession when a defender is goal-side and the
        // forward path is blocked (same `forward_blocked` signal as the escape
        // carry above): with the opponent between you and goal you have few
        // attacking options, so keeping the ball with your body beats forcing past
        // them and coughing it up. Gated to genuine pressure (`goal_side_shield`)
        // so it never dampens progression in open play, where it collapses to ~0.
        let goal_side_shield = forward_blocked * pressure_urgency.max(pressure);
        // When truly blocked-and-pressured (a livelock duel) lift the ceiling so the
        // shield can outscore a now-suppressed forward dribble and the holder keeps
        // it rather than feeding the ping-pong; otherwise it stays at the old cap.
        // Just received the ball with a defender closing — shield with the body to
        // secure it instead of knocking it in front of them to be stolen. Gated to
        // genuine pressure so receiving in space still plays out normally.
        let fresh_receipt_shield = (1.0 - observation.perceived_time_on_ball_seconds / 1.5)
            .clamp(0.0, 1.0)
            * pressure_urgency.max(pressure);
        let own_half_retention = if own_half { 0.30 } else { 0.0 };
        // Sustained pressure with NO good outlet: shield to keep the ball rather than
        // force a low-percentage pass into traffic. This is what stops the "dwell under
        // pressure, then play a terrible pass" failure — when nothing is on, protect
        // possession instead of gifting it away.
        let no_good_outlet = (1.0
            - floor_pass_quality
                .max(aerial_pass_quality)
                .max(open_support_fit))
        .clamp(0.0, 1.0);
        let pressured_no_outlet_shield = pressured_release_signal * no_good_outlet;
        // Shield harder both as pressure RISES (a man bearing down — HEAD's term) and
        // when there is NO good outlet (origin's term); take the higher ceiling.
        let protect_ball_ceiling = (0.88
            + goal_side_shield * 0.34
            + own_half_retention
            + pressure_rising * 0.34
            + pressured_no_outlet_shield * 0.30
            + steal_risk_escape_lift * STEAL_RISK_BAD_OUTLET_ESCAPE_LIFT)
            .clamp(0.88, 1.68);
        let attack_escape_bias = if own_half { 0.55 } else { 1.0 };
        let protect_ball_escape_damp = (1.0
            - slow_holder_escape_signal * PRESSURED_SLOW_HOLDER_SHIELD_DAMP * attack_escape_bias)
            .clamp(0.62, 1.0);
        // A pinned holder with no good outlet can always turn and shield, regardless of
        // dribbling skill — so floor the shield on the pressured-no-outlet / steal-risk signal
        // INDEPENDENTLY of the (pressure-damped) dribble base, capped at the same ceiling.
        // This keeps the one correct action (protect possession) above a panic pass into
        // traffic instead of collapsing with the carrier's dribble score. Collapses to ~0 in
        // open play (no steal risk / a real outlet), so normal possession is unaffected.
        let pinned_shield_floor = (pressured_no_outlet_shield.max(steal_risk_escape_lift)
            * PINNED_SHIELD_FLOOR_LIFT)
            .clamp(0.0, protect_ball_ceiling)
            * protect_ball_escape_damp;
        let protect_ball_score = (dribble_score
            * (0.12
                + pressure_urgency.max(pressure) * 0.76
                + observation.immediate_dispossession_risk.clamp(0.0, 1.0) * 0.44
                + goal_side_shield * 0.46
                + fresh_receipt_shield * 0.55
                + pressured_no_outlet_shield * 0.60
                + steal_risk_escape_lift * 0.82
                + own_half_retention
                // Rising pressure (a man bearing down) is exactly when you turn your
                // body between him and the ball -- lift the shield's urgency hard.
                + pressure_rising * 0.80
                + (1.0 - observation.perceived_time_on_ball_seconds / 2.8).clamp(0.0, 1.0) * 0.24)
            * protect_ball_escape_damp
            * keeper_carry_under_pressure_damp)
            .clamp(0.01, protect_ball_ceiling)
            .max(pinned_shield_floor);
        // Calm on the ball: when a clean pass is on and the holder is NOT under real
        // heat, they should settle and pass rather than throw flashy feints / side-steps
        // / swivels. This damps the evade moves by how good the available pass is, scaled
        // by how unpressured they are (and how slowly pressure is rising) -- it melts away
        // the instant a defender is genuinely on them. Floored so a legal evade is never
        // fully zeroed.
        let calm_quiet =
            ((1.0 - pressure_urgency.max(pressure)) * (1.0 - pressure_rising)).clamp(0.0, 1.0);
        let calm_pass_focus = (1.0
            - calm_quiet * floor_pass_quality.clamp(0.0, 1.0) * CALM_PASS_FOCUS_DAMP)
            .clamp(CALM_PASS_FOCUS_FLOOR, 1.0);
        let side_step_legal = (observation.nearest_opponent_distance <= 4.6
            && pressure_urgency.max(pressure) >= 0.28)
            || slow_holder_escape_signal >= 0.18;
        // First-touch escape: a player who just received under pressure WITH a defender-aware
        // lateral lane open should break contact with the side-step now instead of freezing on the
        // shield. Collapses to 0 when boxed in (lane ~0), in open play, off fresh receipt, or when
        // the gate is disabled — so it only fires in exactly the "freeze on receipt" picture.
        let first_touch_escape_signal = if dd_soccer_disable_first_touch_escape() {
            0.0
        } else {
            (1.0 - observation.perceived_time_on_ball_seconds / 1.5).clamp(0.0, 1.0)
                * pressure_urgency.max(pressure)
                * escape_lane_fit
        };
        let active_escape_signal = first_touch_escape_signal.max(slow_holder_escape_signal);
        // The side-step is the dedicated evade — knock the ball away from the
        // defender and step past. Under escape pressure it gets a real urgency boost
        // (and a lifted ceiling) so a pinned holder breaks contact instead of dwelling.
        let side_step_ceiling =
            (0.82 + steal_escape_urgency * 0.52 + active_escape_signal * 0.52).clamp(0.82, 1.48);
        // Escape FLOOR (analogue of pinned_shield_floor): under pressure the dribble base is damped
        // to ~0, so a multiplier alone can't lift the evade above the floored shield. Skill-gate it
        // (as the shield/hold gates do) so an elite dribbler keeps his better moves (xavi-turn/cuts)
        // and only the mid/low-skill carrier — who would otherwise just freeze — gets floored.
        let skilled_escape_relief =
            ((dribbling - (NON_ELITE_DRIBBLE_HOLD_SKILL_CUTOFF - 0.28)) / 0.28).clamp(0.0, 1.0);
        let side_step_escape_floor = ((first_touch_escape_signal
            * FIRST_TOUCH_ESCAPE_SIDE_STEP_FLOOR_LIFT
            + slow_holder_escape_signal * PRESSURED_SLOW_HOLDER_ESCAPE_FLOOR_LIFT)
            * (1.0 - skilled_escape_relief))
            .clamp(0.0, side_step_ceiling);
        let side_step_score = (dribble_score
            * (0.30 + pressure_urgency.max(pressure) * 0.88)
            * (1.0
                + (1.0 - observation.forward_dribble_space_yards / 14.0).clamp(0.0, 1.0) * 0.24
                + steal_escape_urgency * 1.02
                + active_escape_signal * 1.24)
            * calm_pass_focus
            * keeper_carry_under_pressure_damp)
            .clamp(0.01, side_step_ceiling)
            .max(side_step_escape_floor);
        let technical_escape_pressure = pressure_urgency
            .max(pressure)
            .max(observation.immediate_dispossession_risk.clamp(0.0, 1.0))
            .max(pressure_from_nearest_distance(
                observation.nearest_opponent_distance,
            ))
            .clamp(0.0, 1.0);
        let defender_overcommit = observation
            .neural_extended
            .nearest_defender_overcommit_score
            .clamp(0.0, 1.0);
        let perception = &tunables().pomdp_perception;
        let defender_reaction_delay_fit = ((observation
            .neural_extended
            .nearest_defender_reaction_delay_seconds
            - perception.player_reaction_min_seconds)
            / perception.player_reaction_span_seconds())
        .clamp(0.0, 1.0);
        let momentum_escape_side = observation
            .neural_extended
            .dribble_momentum_escape_side
            .clamp(-1.0, 1.0);
        let nutmeg_window = observation
            .neural_extended
            .dribble_nutmeg_window_score
            .clamp(0.0, 1.0);
        let dribble_beat_pressure = technical_escape_pressure
            .max(defender_overcommit)
            .max(nutmeg_window * 0.86)
            .max(slow_holder_escape_signal * 0.84)
            .clamp(0.0, 1.0);
        let cut_legal = observation.nearest_opponent_distance <= 4.8
            && dribble_beat_pressure >= 0.30
            && !goalmouth_carry_forced
            && !goal_attack_shot_blocks_alternatives;
        let left_cut_legal = cut_legal && left_room > 1.5;
        let right_cut_legal = cut_legal && right_room > 1.5;
        let cut_score_base = (dribble_score
            * (0.18
                + technical_escape_pressure * 0.68
                + defender_overcommit * 0.34
                + defender_reaction_delay_fit * 0.08
                + decision_urgency * 0.16
                + steal_escape_urgency * 0.52
                + steal_risk_escape_lift * 0.42)
            * (0.74 + dribbling * 0.36 + ability01(self.skills.first_touch) * 0.16)
            * calm_pass_focus
            * keeper_carry_under_pressure_damp)
            .clamp(0.01, (0.66 + steal_escape_urgency * 0.32).clamp(0.66, 0.98));
        let left_momentum_fit = if momentum_escape_side < -0.25 {
            (-momentum_escape_side).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let right_momentum_fit = if momentum_escape_side > 0.25 {
            momentum_escape_side.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let left_cut_score = (cut_score_base
            * (0.82
                + (left_room / 12.0).clamp(0.0, 1.0) * 0.28
                + defender_overcommit * left_momentum_fit * 0.20))
            .clamp(0.01, 1.08);
        let right_cut_score = (cut_score_base
            * (0.82
                + (right_room / 12.0).clamp(0.0, 1.0) * 0.28
                + defender_overcommit * right_momentum_fit * 0.20))
            .clamp(0.01, 1.08);
        let flair = ability01(self.skills.flair_passing);
        let nutmeg_legal = observation.nearest_opponent_distance <= 3.2
            && dribble_beat_pressure.max(nutmeg_window) >= 0.34
            && self.role != PlayerRole::Goalkeeper
            && !own_half
            && !goalmouth_carry_forced
            && !goal_attack_shot_blocks_alternatives;
        let nutmeg_score = (dribble_score
            * (0.08
                + technical_escape_pressure * 0.52
                + defender_overcommit * 0.36
                + nutmeg_window * 0.58
                + defender_reaction_delay_fit * 0.10
                + decision_urgency * 0.18
                + steal_escape_urgency * 0.34)
            * (0.52 + dribbling * 0.34 + flair * 0.30 + ability01(self.skills.first_touch) * 0.14)
            * calm_pass_focus
            * keeper_carry_under_pressure_damp)
            .clamp(
                0.01,
                (0.42 + steal_escape_urgency * 0.24 + flair * 0.18 + nutmeg_window * 0.16)
                    .clamp(0.42, 0.96),
            );
        let feint_legal =
            observation.nearest_opponent_distance <= 5.2 && dribble_beat_pressure >= 0.30;
        // Feints are already pressure-gated (feint_legal). Keep the ceiling low so a
        // carrier does not throw a feint every other tick — fewer feints per second,
        // reserved for when they're genuinely needed under pressure. The calm-pass-focus
        // damp suppresses them further when an unpressured holder has a clean pass on.
        let feint_score = (dribble_score
            * (0.22
                + pressure_urgency * 0.66
                + defender_overcommit * 0.30
                + defender_reaction_delay_fit * 0.08
                + decision_urgency * 0.18
                + steal_risk_escape_lift * 0.52)
            * (0.78 + dribbling * 0.28)
            * calm_pass_focus
            * keeper_carry_under_pressure_damp)
            .clamp(
                0.01,
                (0.55 + steal_escape_urgency * 0.28 + defender_overcommit * 0.14).clamp(0.55, 0.94),
            );
        let hold_up_flank_score = ((self.preferences.dribble_bias
            * (0.46 + dribbling * 0.42 + ability01(self.skills.strength) * 0.18)
            * (0.74 + pressure * 0.18)
            * (1.0 + (1.0 - (pass_target_count as f64 / 2.0).clamp(0.0, 1.0)) * 0.28))
            * hold_penalty_multiplier
            * (1.0 - pressured_good_outlet * 0.26).clamp(0.62, 1.0)
            * keeper_carry_under_pressure_damp)
            .clamp(0.03, 0.78);
        let mut options = vec![
            AgentActionOptionTrace::new("shoot", shot_score, shot_legal),
            AgentActionOptionTrace::new("dribble", dribble_score, true),
            AgentActionOptionTrace::new("carry-forward", carry_forward_score, carry_forward_legal),
            AgentActionOptionTrace::new(
                "vertical-attack",
                vertical_attack_score,
                vertical_attack_legal,
            ),
            AgentActionOptionTrace::new(
                "turnover-burst",
                turnover_burst_score,
                turnover_burst_legal,
            ),
            AgentActionOptionTrace::new(
                "carry-out-left",
                carry_out_left_score,
                carry_out_left_legal,
            ),
            AgentActionOptionTrace::new(
                "carry-out-right",
                carry_out_right_score,
                carry_out_right_legal,
            ),
            AgentActionOptionTrace::new("protect-ball", protect_ball_score, protect_ball_legal),
            AgentActionOptionTrace::new("side-step", side_step_score, side_step_legal),
            AgentActionOptionTrace::new("left-cut", left_cut_score, left_cut_legal),
            AgentActionOptionTrace::new("right-cut", right_cut_score, right_cut_legal),
            AgentActionOptionTrace::new("nutmeg", nutmeg_score, nutmeg_legal),
            AgentActionOptionTrace::new("fake-left-cut-right", feint_score, feint_legal),
            AgentActionOptionTrace::new("fake-right-cut-left", feint_score, feint_legal),
            AgentActionOptionTrace::new(
                "hold-up-flank",
                hold_up_flank_score,
                hold_up_flank_available && !goalmouth_carry_forced && goal_attack < 0.18,
            ),
        ];
        let own_half = observation.yards_to_own_goal < observation.yards_to_goal;
        let defensive_third_pressure =
            (1.0 - observation.yards_to_own_goal / 42.0).clamp(0.0, 1.0) * pressure;
        let clearance_role_bias = match self.role {
            PlayerRole::Goalkeeper => 0.82,
            PlayerRole::Defender => 1.0,
            PlayerRole::Midfielder => 0.24,
            PlayerRole::Forward => 0.08,
        };
        let emergency_clearance_pressure = pressure_urgency
            .max(pressure)
            .max(defensive_urgency)
            .max(observation.immediate_dispossession_risk)
            .clamp(0.0, 1.0);
        let clearance_pressure_signal = pressure_urgency
            .max(pressure)
            .max(observation.immediate_dispossession_risk)
            .clamp(0.0, 1.0);
        let clearance_legal = own_half
            && (clearance_pressure_signal >= 0.34
                || (defensive_urgency >= 0.42 && clearance_pressure_signal >= 0.24)
                || (defensive_urgency >= 0.58
                    && observation.yards_to_own_goal <= 40.0
                    && (clearance_pressure_signal >= 0.18
                        || pass_target_count == 0
                        || (clearance_pressure_signal >= 0.12
                            && observation.expected_pass_completion < 0.32)))
                || (clearance_pressure_signal >= 0.14
                    && excessive_hold_pressure(observation, dribbling) >= 0.18))
            // A clearance is a release under PROXIMATE pressure — never the choice
            // for a defender with all day on the ball. Require an opponent actually
            // closing OR a genuine dispossession threat (the live bug: center-backs
            // hoofing it at dispossession-risk 0.1 with the nearest opponent
            // 30-68yd away). The own-box emergency still clears regardless.
            && (observation.nearest_opponent_distance <= CLEARANCE_MAX_OPPONENT_DISTANCE_YARDS
                || observation.immediate_dispossession_risk >= 0.5
                || observation.yards_to_own_goal <= 20.0)
            && matches!(self.role, PlayerRole::Goalkeeper | PlayerRole::Defender);
        let clearance_score = ((pressure * 0.48
            + defensive_urgency * 0.56
            + defensive_third_pressure * 0.48
            + observation.immediate_dispossession_risk.clamp(0.0, 1.0) * 0.36
            + excessive_hold_pressure(observation, dribbling) * 0.46
            + (1.0 - observation.expected_pass_completion.clamp(0.0, 1.0)) * 0.20)
            * clearance_role_bias
            * (0.78 + ability01(self.skills.defending) * 0.22)
            * hold_release_multiplier
            * (1.0 + emergency_clearance_pressure * 0.28))
            .clamp(0.01, 1.78);
        let route_one_role_bias = match self.role {
            PlayerRole::Goalkeeper => 0.76,
            PlayerRole::Defender => 0.92,
            PlayerRole::Midfielder => 0.72,
            PlayerRole::Forward => 0.24,
        };
        let route_one_legal = own_half;
        let route_one_score = ((0.05
            + directive.risk_tolerance * 0.12
            + passing * 0.12
            + defensive_urgency * 0.22
            + excessive_hold_pressure(observation, dribbling) * 0.18
            + observation.attacking_overload_score.clamp(0.0, 1.0) * 0.22
            + (1.0 - observation.expected_pass_completion.clamp(0.0, 1.0)) * 0.10
            + (1.0 - (pass_target_count as f64 / 3.0).clamp(0.0, 1.0)) * 0.08)
            * route_one_role_bias
            * hold_release_multiplier)
            .clamp(0.02, 1.12);
        options.push(AgentActionOptionTrace::new(
            "clearance",
            clearance_score,
            clearance_legal,
        ));
        options.push(AgentActionOptionTrace::new(
            "route-one",
            route_one_score,
            route_one_legal,
        ));
        let near_goal_pass_multiplier = near_goal_pass_release_multiplier(observation, self.role);
        let floor_pass_patience_multiplier =
            low_pressure_pass_release_multiplier(observation, PassFlight::Floor);
        let aerial_pass_patience_multiplier =
            low_pressure_pass_release_multiplier(observation, PassFlight::Aerial);
        let hold_release_score_cap = 1.22 * hold_release_multiplier.clamp(1.0, 1.38);
        let aerial_forward_runner_multiplier = observation
            .aerial_forward_runner_pass_multiplier
            .clamp(1.0, 1.50);
        let reset_quality = (observation.expected_pass_completion.clamp(0.0, 1.0) * 0.42
            + observation.best_pass_receiver_openness.clamp(0.0, 1.0) * 0.24
            + open_support_fit * 0.22
            + pressured_release_signal * 0.12)
            .clamp(0.0, 1.0);
        let recycle_reset_score = (self.preferences.pass_bias
            * directive.pass_priority
            * (0.36 + passing * 0.34)
            * (0.74 + reset_quality * 0.38)
            * (1.0
                + if own_half { 0.16 } else { 0.0 }
                + pressure_urgency.max(pressure) * 0.18
                + excessive_hold_pressure(observation, dribbling) * 0.22)
            * floor_pass_patience_multiplier
            * hold_release_multiplier
            * crowded_pass_lift
            * pressured_release_multiplier)
            .clamp(0.004, hold_release_score_cap);
        options.push(AgentActionOptionTrace::new(
            "recycle-reset",
            recycle_reset_score,
            pass_target_count > 0
                && reset_quality >= 0.20
                && !goal_attack_shot_blocks_alternatives
                // A reset (square/backward) is a build-up tool, never the right call when a
                // killer ball to goal is on or a shot is on — it must not drain the final
                // ball or produce a backward outlet in the shooting window.
                && !observation.threaded_goal_pass_available
                && !must_shoot_near_goal(observation, self.role),
        ));
        let switch_context = (flank_lane_fit * 0.36
            + observation.attacking_overload_score.clamp(0.0, 1.0) * 0.18
            + observation.team_spread_yards / field_width.max(1.0) * 0.18
            + pressure_urgency.max(pressure) * 0.16
            + observation.pass_curl_probability.clamp(0.0, 1.0) * 0.12)
            .clamp(0.0, 1.0);
        let switch_play_score = (self.preferences.pass_bias
            * directive.pass_priority
            * (0.34 + passing * 0.36 + ability01(self.skills.vision) * 0.18)
            * (0.70
                + observation.expected_pass_completion.clamp(0.0, 1.0) * 0.22
                + observation.best_pass_receiver_openness.clamp(0.0, 1.0) * 0.16
                + switch_context * 0.34)
            * floor_pass_patience_multiplier
            * hold_release_multiplier
            * crowded_pass_lift
            * pressured_release_multiplier)
            .clamp(0.004, 0.98 * hold_release_multiplier.clamp(1.0, 1.28));
        options.push(AgentActionOptionTrace::new(
            "switch-play",
            switch_play_score,
            pass_target_count > 0
                && observation.expected_pass_completion >= 0.24
                && !goal_attack_shot_blocks_alternatives
                // Don't switch the play when a killer ball to goal is on or a shot is on —
                // take the goal threat, don't drain it across the pitch.
                && !observation.threaded_goal_pass_available
                && !must_shoot_near_goal(observation, self.role),
        ));
        let flank_cross_context =
            flank_cross_context_score(observation, self.position, field_width);
        let committed_byline_cross_release =
            byline_cross_release && directive.flank_attack_policy.is_flank();
        let flank_cross_legal_context =
            flank_cross_context_is_legal(observation, self.position, field_width)
                && (!goal_attack_shot_blocks_alternatives || committed_byline_cross_release)
                && (!goalmouth_carry_forced || directive.flank_attack_policy.is_flank())
                && !byline_drive_active;
        let low_cross_score = (self.preferences.pass_bias
            * directive.pass_priority
            * (0.42 + passing * 0.24 + crossing * 0.42)
            // Use the width: an advanced wide player should look to cross into the box, so the
            // base preference is lifted (don't let a good crossing position fizzle out centrally).
            * (0.74 + flank_cross_context * 0.74)
            * (0.78
                + observation.expected_pass_completion.clamp(0.0, 1.0) * 0.20
                + observation.best_pass_receiver_openness.clamp(0.0, 1.0) * 0.18
                + goal_attack * 0.22)
            * near_goal_pass_multiplier
            * floor_pass_patience_multiplier
            * hold_release_multiplier
            * pressured_release_multiplier
            * low_cross_multiplier)
            .clamp(0.01, 0.86)
            .max(
                if byline_cross_release && directive.flank_attack_policy.prefers_low_cross() {
                    (0.90 + crossing * 0.08).clamp(0.90, 0.98)
                } else {
                    0.0
                },
            );
        options.push(AgentActionOptionTrace::new(
            "flank-low-cross",
            low_cross_score,
            pass_target_count > 0 && flank_cross_legal_context,
        ));
        let high_cross_score = (self.preferences.pass_bias
            * directive.pass_priority
            * (0.36
                + passing * 0.18
                + crossing * 0.48
                + ability01(self.skills.flair_passing) * 0.10)
            * (0.56 + flank_cross_context * 0.78)
            * (0.76
                + observation
                    .expected_aerial_pass_completion
                    .max(observation.expected_pass_completion * 0.72)
                    .clamp(0.0, 1.0)
                    * 0.20
                + observation
                    .best_aerial_pass_receiver_openness
                    .max(observation.best_pass_receiver_openness * 0.70)
                    .clamp(0.0, 1.0)
                    * 0.18
                + goal_attack * 0.26)
            * (1.0 - observation.aerial_pass_interception_risk.clamp(0.0, 1.0) * 0.28)
                .clamp(0.62, 1.0)
            * near_goal_pass_multiplier
            * aerial_pass_patience_multiplier
            * aerial_forward_runner_multiplier
            * hold_release_multiplier
            * pressured_release_multiplier
            * high_cross_multiplier)
            .clamp(0.01, 0.78)
            .max(
                if byline_cross_release && directive.flank_attack_policy.prefers_high_cross() {
                    (0.90 + crossing * 0.08).clamp(0.90, 0.98)
                } else {
                    0.0
                },
            );
        options.push(AgentActionOptionTrace::new(
            "flank-high-cross",
            high_cross_score,
            aerial_pass_target_count > 0 && flank_cross_legal_context,
        ));
        let corner_flag_cross_score = (low_cross_score.max(high_cross_score)
            * (0.72
                + flank_cross_context * 0.34
                + flank_lane_fit * 0.18
                + goal_attack * 0.12
                + directive.flank_overlap_run_probability * 0.10))
            .clamp(0.01, 0.96);
        options.push(AgentActionOptionTrace::new(
            "corner-flag-cross",
            corner_flag_cross_score,
            (pass_target_count > 0 || aerial_pass_target_count > 0) && flank_cross_legal_context,
        ));
        let surprise_pass_score = (self.preferences.pass_bias
            * directive.pass_priority
            * (0.18
                + ability01(self.skills.flair_passing) * 0.46
                + passing * 0.18
                + ability01(self.skills.vision) * 0.16)
            * (0.74
                + pressure_urgency.max(pressure) * 0.22
                + goal_attack * 0.18
                + observation.best_pass_receiver_openness.clamp(0.0, 1.0) * 0.18)
            * floor_pass_patience_multiplier
            * hold_release_multiplier
            * pressured_release_multiplier)
            .clamp(0.01, 0.84);
        options.push(AgentActionOptionTrace::new(
            "surprise-pass",
            surprise_pass_score,
            pass_target_count > 0
                && !goal_attack_shot_blocks_alternatives
                && (pressure_urgency.max(pressure) >= 0.18
                    || goal_attack >= 0.10
                    || ability01(self.skills.flair_passing) >= 0.54),
        ));
        let incoming_aerial = matches!(
            observation.incoming_ball_kind,
            IncomingBallKind::AerialCross | IncomingBallKind::AerialPass
        );
        let flick_on_score = (self.preferences.pass_bias
            * directive.pass_priority
            * (0.20
                + ability01(self.skills.first_touch) * 0.34
                + ability01(self.skills.flair_passing) * 0.20
                + aerial_duel_skill_from_agent(self) * 0.22)
            * (0.72
                + observation.expected_aerial_pass_completion.clamp(0.0, 1.0) * 0.18
                + observation
                    .best_aerial_pass_receiver_openness
                    .clamp(0.0, 1.0)
                    * 0.18
                + observation
                    .aerial_forward_runner_pass_multiplier
                    .clamp(1.0, 1.50)
                    * 0.10)
            * aerial_pass_patience_multiplier
            * aerial_forward_runner_multiplier
            * pressured_release_multiplier)
            .clamp(0.01, 0.88);
        options.push(AgentActionOptionTrace::new(
            "flick-on",
            flick_on_score,
            aerial_pass_target_count > 0 && observation.first_touch_available && incoming_aerial,
        ));
        let killer_pass_range_fit = killer_pass_goal_range_fit(observation);
        let killer_pass_goal_pressure = observation
            .killer_pass_goal_pressure
            .max(killer_pass_goal_pressure_score(observation))
            .clamp(0.0, 1.0);
        let single_thread_goal_pressure =
            single_pass_goal_thread_pressure_score(observation).clamp(0.0, 1.0);
        let threaded_goal_receiver_available = observation.threaded_goal_pass_available;
        let threaded_goal_pass_quality = threaded_goal_pass_quality_fit(observation);
        let approaching_threaded_goal_fit =
            if threaded_goal_receiver_available && observation.visible_forward_pass_options > 0 {
                killer_pass_goal_range_fit(observation).powf(0.44)
            } else {
                0.0
            };
        // A confirmed threaded receiver IS a forward option, so it supersedes the safe-pass
        // count / visible-forward-option preconditions (those use the reception-gated set,
        // which is empty through a blocked lane — exactly when a killer ball is wanted).
        // No generic floor_pass_lane_score gate here: a confirmed threaded receiver already
        // passed the killer assessment's own lane vetting (it rejects lane_fit < 0.30 to the
        // specific receiver). The generic score is computed over the reception-gated visible
        // set, which is empty through a blocked lane — exactly when the killer ball is wanted.
        let killer_pass_legal = threaded_goal_receiver_available
            && (!goal_attack_shot_blocks_alternatives || threaded_goal_pass_overrides_shot)
            && observation.yards_to_goal <= KILLER_PASS_MAX_YARDS_TO_GOAL;
        let killer_pass_score = (self.preferences.pass_bias
            * directive.pass_priority
            * (0.48 + passing * 0.44 + ability01(self.skills.vision) * 0.20)
            * (0.72
                + observation.expected_pass_completion.clamp(0.0, 1.0) * 0.28
                + observation
                    .threaded_goal_pass_expected_completion
                    .clamp(0.0, 1.0)
                    * 0.16
                + observation
                    .best_forward_pass_receiver_openness
                    .clamp(0.0, 1.0)
                    * 0.22
                + observation
                    .threaded_goal_pass_receiver_openness
                    .clamp(0.0, 1.0)
                    * 0.14
                + observation.best_pass_stride_fit.clamp(0.0, 1.0) * 0.24
                + observation.threaded_goal_pass_stride_fit.clamp(0.0, 1.0) * 0.14
                + observation.floor_pass_lane_score.clamp(0.0, 1.0) * 0.24)
            * (1.0
                + killer_pass_range_fit * 1.05
                + killer_pass_goal_pressure * 1.42
                + single_thread_goal_pressure * 1.50
                + threaded_goal_pass_quality * 0.62
                + approaching_threaded_goal_fit * 0.38
                + goal_entry_pressure * 0.78
                + goal_attack * 0.42)
            * (1.0 + offensive_urgency * 0.42)
            * (1.0 + decisive_goal_pressure * 0.72)
            * (1.0 + final_ball_goal_decision_fit * 0.28)
            * goal_entry_pass_learning
            * near_goal_pass_multiplier.max(0.72)
            * floor_pass_patience_multiplier
            * hold_release_multiplier
            * crowded_pass_lift
            * pressured_release_multiplier)
            .clamp(0.01, 1.68 * goal_entry_pass_learning.clamp(0.86, 1.20));
        options.push(AgentActionOptionTrace::new(
            "killer-pass",
            killer_pass_score,
            killer_pass_legal,
        ));
        let pass_and_move_forward_lift = (1.0
            + observation
                .pass_and_move_forward_opportunity
                .clamp(0.0, 1.0)
                * 0.24
            + observation.pass_and_move_numbers_advantage.clamp(0.0, 1.0) * 0.16
            + observation.pass_and_move_run_lane_score.clamp(0.0, 1.0) * 0.10)
            .clamp(1.0, 1.50);
        for rank in 0..pass_target_count.min(3) {
            let rank_weight = match rank {
                0 => 1.00,
                1 => 0.74,
                _ => 0.56,
            };
            let quick_release = (1.35 - observation.perceived_time_on_ball_seconds)
                .max(0.0)
                .min(1.0);
            let completion_bonus = {
                let base = 0.82
                    + observation.expected_pass_completion.clamp(0.0, 1.0) * 0.24
                    + observation.best_pass_receiver_openness.clamp(0.0, 1.0) * 0.16;
                // Under real pressure a CONTESTED pass (low expected completion) is penalised much
                // harder so a pinned carrier shields/holds rather than forcing the ball into traffic
                // (the "passed it straight to the other team" turnover). Calm play and a genuinely
                // open pass are untouched (heat or contested ≈ 0).
                let damp = if soccer_pressured_contested_pass_damp_enabled() {
                    let heat = observation.perceived_pressure.clamp(0.0, 1.0);
                    let contested = 1.0 - observation.expected_pass_completion.clamp(0.0, 1.0);
                    heat * contested * PRESSURED_CONTESTED_PASS_DAMP
                } else {
                    0.0
                };
                (base - damp).clamp(0.40, 1.22)
            };
            // Own-half short-pass discipline (Layer 2): when the best floor pass on offer is a
            // short ball played from our own half, lean the carrier toward keeping the dribble /
            // holding rather than playing the square ball. Scales with how short the ball is and
            // eases for a wide-open receiver; floored so it demotes without hard-vetoing. Off (or
            // outside the own half / with a 4+yd ball available) ⇒ 1.0 (no change).
            let own_half_short_pass_dribble_bias = if own_half
                && !dd_soccer_disable_own_half_short_pass_liability()
                && observation.best_floor_pass_distance_yards > 0.05
                && observation.best_floor_pass_distance_yards
                    < OWN_HALF_SHORT_PASS_LIABILITY_MAX_YARDS
            {
                let shortness = (1.0
                    - observation.best_floor_pass_distance_yards
                        / OWN_HALF_SHORT_PASS_LIABILITY_MAX_YARDS)
                    .clamp(0.0, 1.0);
                let openness_relief = observation.best_pass_receiver_openness.clamp(0.0, 1.0);
                let damp = OWN_HALF_SHORT_PASS_DRIBBLE_DAMP
                    * (0.5 + 0.5 * shortness)
                    * (1.0 - 0.5 * openness_relief);
                (1.0 - damp).clamp(OWN_HALF_SHORT_PASS_DRIBBLE_MIN_MULT, 1.0)
            } else {
                1.0
            };
            let pass_score = (self.preferences.pass_bias
                * directive.pass_priority
                * (0.70 + passing * 0.42)
                * (1.0 + quick_release * 0.22)
                * completion_bonus
                * (1.0 + observation.pass_curl_probability * 0.055)
                * near_goal_pass_multiplier
                * floor_pass_patience_multiplier
                * hold_release_multiplier
                * low_cross_multiplier
                * short_upfield_ground_pass_multiplier
                * own_half_short_pass_dribble_bias
                * pass_and_move_forward_lift
                * crowded_pass_lift
                * pressured_release_multiplier
                * panic_pass_damp
                * rank_weight)
                .clamp(0.004, hold_release_score_cap);
            options.push(AgentActionOptionTrace::new(
                format!("pass{}", rank + 1),
                pass_score,
                true,
            ));
        }
        // An over-the-top flick-on is the committed maneuver — actively prize the early
        // aerial ball into the target man so he can flick it on to the runner beyond.
        let aerial_flick_strategy_multiplier = if matches!(
            directive.attack_strategy,
            TeamAttackStrategy::AerialFlickOnRelease
        ) {
            1.35
        } else {
            1.0
        };
        for rank in 0..aerial_pass_target_count.min(3) {
            let rank_weight = match rank {
                0 => 0.82,
                1 => 0.62,
                _ => 0.48,
            };
            let bypass_bonus = (observation.perceived_pressure * 0.20
                + (1.0 - observation.forward_dribble_space_yards / 16.0).clamp(0.0, 1.0) * 0.16
                + observation.aerial_pass_bypass_score * 0.34)
                .clamp(0.0, 0.42);
            // A ball cut out in the flight lane is a turnover no matter how open
            // the receiver is, so interception risk should nearly VETO the aerial
            // ball, not merely shave it (was 1-0.38*risk floored at 0.58, which
            // left a 0.9-interception hoof at 66% of its score — the live aerial
            // giveaways). Near-linear to ~0 so a high-risk lane collapses below
            // the carry/shield/floor-pass alternatives.
            let interception_penalty =
                (1.0 - observation.aerial_pass_interception_risk.clamp(0.0, 1.0)).clamp(0.08, 1.0);
            let aerial_completion_bonus = (0.80
                + observation.expected_aerial_pass_completion.clamp(0.0, 1.0) * 0.18
                + observation
                    .best_aerial_pass_receiver_openness
                    .clamp(0.0, 1.0)
                    * 0.12)
                .clamp(0.74, 1.14);
            let aerial_score = (self.preferences.pass_bias
                * directive.pass_priority
                * (0.48
                    + passing * 0.26
                    + crossing * 0.22
                    + ability01(self.skills.flair_passing) * 0.10)
                * (1.0 + bypass_bonus)
                * interception_penalty
                * aerial_completion_bonus
                * (1.0 + observation.pass_curl_probability * 0.075)
                * near_goal_pass_multiplier
                * aerial_pass_patience_multiplier
                * aerial_forward_runner_multiplier
                * hold_release_multiplier
                * high_cross_multiplier
                * pressured_release_multiplier
                * aerial_flick_strategy_multiplier
                * rank_weight)
                .clamp(0.004, 0.98 * hold_release_multiplier.clamp(1.0, 1.24));
            options.push(AgentActionOptionTrace::new(
                format!("aerial-pass{}", rank + 1),
                aerial_score,
                true,
            ));
        }
        if directive.flank_attack_policy.prefers_low_cross() {
            ensure_min_legal_option_probability(
                &mut options,
                "flank-low-cross",
                (0.18 + directive.flank_overlap_run_probability * 0.10).clamp(0.18, 0.30),
            );
        } else if directive.flank_attack_policy.prefers_high_cross() {
            ensure_min_legal_option_probability(
                &mut options,
                "flank-high-cross",
                (0.18 + directive.flank_overlap_run_probability * 0.10).clamp(0.18, 0.30),
            );
        }
        if killer_pass_legal && directive_requests_threaded_release(directive) {
            let threaded_strategy_floor = (0.50
                + threaded_goal_pass_quality * 0.18
                + single_thread_goal_pressure * 0.18
                + killer_pass_goal_pressure * 0.14
                + approaching_threaded_goal_fit * 0.10)
                .clamp(0.56, 0.88);
            ensure_min_legal_option_probability(
                &mut options,
                "killer-pass",
                threaded_strategy_floor,
            );
        }
        if clearance_legal {
            let hold_pressure = excessive_hold_pressure(observation, dribbling);
            // Pressure-driven floor only — NO unconditional base. An unpressured
            // defender that can still legally clear (opponent ~10yd out) is not
            // FORCED to; it can play out. The floor rises only as real pressure /
            // hold-pressure / dispossession-risk climbs.
            let min_clearance_probability = (clearance_pressure_signal * 0.34
                + defensive_urgency * 0.18
                + defensive_third_pressure * 0.14
                + observation.immediate_dispossession_risk.clamp(0.0, 1.0) * 0.16
                + hold_pressure * 0.40)
                .clamp(0.0, 0.82);
            ensure_min_legal_option_probability(
                &mut options,
                "clearance",
                min_clearance_probability,
            );
        }
        let hold_pressure = excessive_hold_pressure(observation, dribbling);
        let release_pressure = pressure_urgency
            .max(pressure)
            .max(observation.immediate_dispossession_risk)
            .max(defender_crowding)
            .clamp(0.0, 1.0);
        if pass_target_count > 0
            && hold_pressure >= 0.12
            && release_pressure >= 0.38
            && dribbling < NON_ELITE_DRIBBLE_HOLD_SKILL_CUTOFF
            && !goal_attack_shot_blocks_alternatives
        {
            let release_floor = (0.20
                + hold_pressure * 0.28
                + release_pressure * 0.18
                + open_support_fit * 0.18
                + observation.best_pass_receiver_openness.clamp(0.0, 1.0) * 0.12
                + observation.expected_pass_completion.clamp(0.0, 1.0) * 0.10)
                .clamp(0.28, 0.68 + open_support_fit * 0.16);
            ensure_min_legal_option_probability(&mut options, "pass1", release_floor);
        }
        if pass_target_count > 0
            && defender_crowding >= 0.30
            && open_outlet_fit >= 0.55
            && !goal_attack_shot_blocks_alternatives
        {
            let spacing_release_floor = (0.42
                + defender_crowding * 0.34
                + open_outlet_fit * 0.16
                + observation.expected_pass_completion.clamp(0.0, 1.0) * 0.08)
                .clamp(0.50, 0.78);
            ensure_min_legal_option_probability(&mut options, "pass1", spacing_release_floor);
        }
        if pass_target_count > 0
            && short_upfield_ground_pass_fit >= 0.42
            && !goal_attack_shot_blocks_alternatives
        {
            let short_forward_release_floor = (0.24
                + short_upfield_ground_pass_fit * 0.34
                + decision_urgency * 0.08
                + pressured_release_signal * 0.10
                + open_forward_outlet * 0.08)
                .clamp(0.34, 0.76);
            ensure_min_legal_option_probability(&mut options, "pass1", short_forward_release_floor);
        }
        if pass_target_count > 0
            && observation.pass_and_move_forward_opportunity >= 0.35
            && !goal_attack_shot_blocks_alternatives
        {
            let pass_and_move_floor = (0.22
                + observation
                    .pass_and_move_forward_opportunity
                    .clamp(0.0, 1.0)
                    * 0.32
                + observation.pass_and_move_numbers_advantage.clamp(0.0, 1.0) * 0.16
                + observation.pass_and_move_run_lane_score.clamp(0.0, 1.0) * 0.10)
                .clamp(0.34, 0.74);
            ensure_min_legal_option_probability(&mut options, "pass1", pass_and_move_floor);
        }
        if clearance_legal && hold_pressure >= 0.16 && release_pressure >= 0.42 {
            let danger_clearance_floor =
                (0.30 + hold_pressure * 0.22 + release_pressure * 0.10).clamp(0.34, 0.72);
            ensure_min_legal_option_probability(&mut options, "clearance", danger_clearance_floor);
        }
        let shot_floor = near_goal_shot_pressure_floor(observation, self.role, shooting).max(
            goal_proximity_shot_pressure_floor(observation, self.role, shooting),
        );
        if shot_floor > 0.0 && !shot_trigger_long_range_veto {
            let decisive_shot_floor = if shot_legal && decisive_goal_pressure > 0.0 {
                let shot_lane_fit = (observation.shot_on_frame_probability.clamp(0.0, 1.0) * 0.42
                    + observation.shot_beat_goalkeeper_probability.clamp(0.0, 1.0) * 0.28
                    + (1.0 - observation.shot_block_probability.clamp(0.0, 1.0)) * 0.30)
                    .clamp(0.0, 1.0);
                (0.15 + decisive_goal_pressure * 0.50 + shot_lane_fit * 0.18)
                    .clamp(0.0, 0.84)
                    .max(
                        (0.13 + goal_entry_pressure * 0.46 + shot_lane_fit * 0.20).clamp(0.0, 0.82),
                    )
            } else {
                0.0
            };
            ensure_min_legal_option_probability(
                &mut options,
                "shoot",
                shot_floor.max(decisive_shot_floor),
            );
        }
        if killer_pass_legal && killer_pass_goal_pressure >= 0.24 {
            let threaded_lane_fit = threaded_goal_pass_quality.max(
                (observation
                    .best_forward_pass_receiver_openness
                    .clamp(0.0, 1.0)
                    * 0.34
                    + observation.best_pass_stride_fit.clamp(0.0, 1.0) * 0.30
                    + observation.floor_pass_lane_score.clamp(0.0, 1.0) * 0.36)
                    .clamp(0.0, 1.0),
            );
            let close_threaded_goal_fit =
                ((36.0 - observation.yards_to_goal) / 18.0).clamp(0.0, 1.0);
            let killer_floor = killer_pass_goal_pressure_floor(observation).max(
                (0.10
                    + decisive_goal_pressure * 0.42
                    + threaded_lane_fit * 0.16
                    + close_threaded_goal_fit * 0.14
                    + goal_entry_pressure * 0.16
                    + single_thread_goal_pressure * 0.18
                    + approaching_threaded_goal_fit * 0.08)
                    .clamp(0.0, 0.84),
            );
            ensure_min_legal_option_probability(&mut options, "killer-pass", killer_floor);
        }
        let release_pressure_carry_fit = (1.0
            - release_pressure * (0.34 + open_outlet_fit * floor_pass_quality * 0.24))
            .clamp(0.38, 1.0);
        let bad_outlet_escape_fit = (1.0 - open_outlet_fit * floor_pass_quality).clamp(0.0, 1.0);
        let pressure_escape_progression_lift =
            (1.0 + steal_pressure * bad_outlet_escape_fit * 0.16).clamp(1.0, 1.16);
        if carry_forward_legal
            && !goal_attack_shot_blocks_alternatives
            && observation.yards_to_goal < observation.yards_to_own_goal
            && observation.yards_to_goal <= 58.0
            && observation.forward_dribble_space_yards >= 3.0
        {
            let final_third_ramp = ((58.0 - observation.yards_to_goal) / 24.0).clamp(0.0, 1.0);
            let grass_fit = (observation.forward_dribble_space_yards / 14.0).clamp(0.0, 1.0);
            let role_floor = match self.role {
                PlayerRole::Forward => 0.30,
                PlayerRole::Midfielder => 0.24,
                PlayerRole::Defender => 0.14,
                PlayerRole::Goalkeeper => 0.0,
            };
            let progression_floor = (role_floor
                + final_third_ramp * 0.12
                + grass_fit * 0.10
                + offensive_urgency * 0.06
                + goal_attack * 0.05)
                * release_pressure_carry_fit
                * pressure_escape_progression_lift.clamp(0.0, 0.54);
            ensure_min_legal_option_probability(&mut options, "carry-forward", progression_floor);
        }
        // Won the ball with grass to escape into: DRIVE out. MDP/POMDP elects the forward-drive
        // family here (the MPC dribble then sprints into the open space). A freshly-won carrier
        // (low real time on the ball) should not be damped merely because the win happened under
        // pressure — that is the moment when the first carry away from the crowd matters most.
        // Pressure still prefers a good outlet when one exists; otherwise it lifts the transition
        // carry/turnover-burst floor so the next decision keeps accelerating instead of settling.
        let fresh_escape = &tunables().fresh_possession_escape;
        let fresh_window = fresh_escape.fresh_seconds.max(f64::EPSILON);
        let just_won_fit =
            (1.0 - observation.actual_time_on_ball_seconds / fresh_window).clamp(0.0, 1.0);
        let pressure_escape_fit = if fresh_escape.enabled {
            ((steal_pressure - fresh_escape.min_pressure)
                / (1.0 - fresh_escape.min_pressure).max(f64::EPSILON))
            .clamp(0.0, 1.0)
        } else {
            0.0
        };
        let pressure_drive_fit =
            (1.0 - pressure * (1.0 - pressure_escape_fit * bad_escape_outlet_fit))
                .clamp(0.0, 1.0);
        let pressure_escape_drive_lift =
            (1.0 + pressure_escape_fit * bad_escape_outlet_fit * 0.70).clamp(1.0, 1.45);
        if self.role != PlayerRole::Goalkeeper
            && carry_forward_legal
            && fresh_escape.enabled
            && just_won_fit > 0.0
            && observation.forward_dribble_space_yards >= fresh_escape.min_forward_space_yards
            && !goal_attack_shot_blocks_alternatives
        {
            let role_floor = match self.role {
                PlayerRole::Forward => 0.42,
                PlayerRole::Midfielder => 0.34,
                PlayerRole::Defender => 0.16,
                PlayerRole::Goalkeeper => 0.0,
            };
            let won_ball_drive_fit = (just_won_fit
                * forward_space_fit
                * pressure_drive_fit
                * pressure_escape_drive_lift)
                .clamp(0.0, 1.0);
            let drive_floor = (role_floor * won_ball_drive_fit)
                .clamp(0.0, fresh_escape.decision_floor_max_probability);
            ensure_min_legal_option_family_probability(
                &mut options,
                &["turnover-burst", "carry-forward", "vertical-attack"],
                drive_floor,
            );
        }
        // Won the ball in a CROWD under pressure: BREAK AWAY into space. The forward-drive floor
        // above only fires with clear grass ahead and little pressure (using
        // `fresh_escape.min_forward_space_yards`) — precisely the situation a
        // freshly-won ball in traffic is NOT in, so nothing currently turns the genuine win moment
        // into an explosive break. There the real-soccer move is to accelerate the ball AWAY from
        // the nearest presser into whatever lane exists (usually lateral/diagonal, rarely straight
        // ahead) rather than settle into a shield and recycle. This is the pressured complement of
        // the drive block: same fresh-win trigger (`just_won_fit`, off the genuine elapsed
        // possession time — NOT the pressure-proxy perceived time the steady-state escape signals
        // use), but it fires UNDER pressure and routes the floor onto the break-into-space family
        // (turnover-burst / carry-out-left|right / side-step), whose execution already steers the
        // ball into the best escape lane (`first_touch_escape_carry_target_for`). Scaled by
        // acceleration skill so it reads as an explosive break, by the escape-lane room so it stands
        // its ground when truly boxed in (lane ~0), and by pressure so it collapses to nothing in
        // calm possession. Default-ON; `DD_SOCCER_DISABLE_WON_BALL_PRESSURE_ESCAPE` restores parity.
        if self.role != PlayerRole::Goalkeeper
            && !dd_soccer_disable_won_ball_pressure_escape()
            && just_won_fit > 0.0
            && escape_lane_fit > 0.0
            && !goalmouth_carry_forced
            && !goal_attack_shot_blocks_alternatives
        {
            let crowd_pressure = pressure_urgency
                .max(pressure)
                .max(pressure_from_nearest_distance(
                    observation.nearest_opponent_distance,
                ))
                .clamp(0.0, 1.0);
            if crowd_pressure >= WON_BALL_PRESSURE_ESCAPE_MIN_PRESSURE {
                let role_floor = match self.role {
                    PlayerRole::Forward => 0.40,
                    PlayerRole::Midfielder => 0.36,
                    PlayerRole::Defender => 0.24,
                    PlayerRole::Goalkeeper => 0.0,
                };
                let accel_fit = 0.70 + ability01(self.skills.acceleration) * 0.30;
                // Only when there is no easy way out: an open, makeable outlet means the carrier can
                // lay it off, so don't force a break into the crowd. `bad_outlet_escape_fit` collapses
                // to ~0 with a good pass available and ~1 when the carrier is genuinely trapped.
                let escape_floor = (role_floor
                    * just_won_fit
                    * escape_lane_fit
                    * crowd_pressure
                    * accel_fit
                    * bad_outlet_escape_fit)
                    .clamp(0.0, 0.50);
                ensure_min_legal_option_family_probability(
                    &mut options,
                    &[
                        "turnover-burst",
                        "carry-out-left",
                        "carry-out-right",
                        "side-step",
                    ],
                    escape_floor,
                );
            }
        }
        if decisive_goal_pressure >= 0.12 && (shot_legal || killer_pass_legal) {
            let recycle_multiplier = (1.0
                - decisive_goal_pressure * 0.82
                - if killer_pass_legal {
                    single_thread_goal_pressure * 0.34
                } else {
                    0.0
                }
                - goal_entry_pressure * 0.16
                - goal_attack * 0.10
                - final_ball_goal_decision_fit * 0.16)
                .clamp(0.16, 1.0);
            for label in [
                "pass1",
                "pass2",
                "pass3",
                "aerial-pass1",
                "aerial-pass2",
                "aerial-pass3",
                "dribble",
                "carry-forward",
                "vertical-attack",
                "turnover-burst",
                "carry-out-left",
                "carry-out-right",
                "protect-ball",
                "xavi-turn",
                "side-step",
                "fake-left-cut-right",
                "fake-right-cut-left",
                "hold-up-flank",
                "corner-flag-cross",
                "surprise-pass",
                "flick-on",
            ] {
                scale_legal_option_score(&mut options, label, recycle_multiplier);
            }
        }
        if decisive_goal_pressure >= 0.08 && (shot_legal || killer_pass_legal) {
            let decisive_family_floor = (0.08
                + decisive_goal_pressure * 0.56
                + goal_entry_pressure * 0.18
                + goal_attack * 0.10
                + offensive_urgency * 0.06
                + single_thread_goal_pressure * 0.10
                + approaching_threaded_goal_fit * 0.08
                + final_ball_goal_decision_fit * 0.10)
                .clamp(
                    0.0,
                    if goal_attack_shot_blocks_alternatives {
                        0.88
                    } else {
                        0.78
                    },
                );
            // When the long shot is vetoed (stop-volunteering), don't let the decisive
            // near-goal floor re-boost SHOOT — keep only the killer-pass in the family so
            // the carrier threads / works it closer instead of hammering from range.
            let decisive_family: &[&str] = if shot_trigger_long_range_veto {
                &["killer-pass"]
            } else {
                &["shoot", "killer-pass"]
            };
            ensure_min_legal_option_family_probability(
                &mut options,
                decisive_family,
                decisive_family_floor,
            );
        }
        // Quick forward ground-pass priority: a short progressive ball (≈5–8 m) to an OPEN,
        // advanced teammate should be RELEASED quickly, not dwelt on. Floor the primary pass
        // option and trim the dwell options (carry/dribble/shield/hold) in proportion to its
        // value so the carrier knocks it forward and keeps the ball moving — "passing sooner."
        // Deferred to the goal-threat logic above (don't drain a real shot/killer ball). Gate
        // OFF ⇒ `quick_forward_pass_value` is 0 ⇒ byte-identical no-op.
        if pass_target_count > 0
            && observation.quick_forward_pass_value > 0.0
            && !goal_attack_shot_blocks_alternatives
            && !observation.threaded_goal_pass_available
            && !must_shoot_near_goal(observation, self.role)
        {
            let quick_value = observation.quick_forward_pass_value.clamp(0.0, 1.0);
            let urgency = (0.62 + decision_urgency * 0.38).clamp(0.62, 1.0);
            let quick_floor = ((0.40 + quick_value * 0.46) * urgency).clamp(0.36, 0.88);
            ensure_min_legal_option_probability(&mut options, "pass1", quick_floor);
            let dwell_damp = (1.0 - quick_value * 0.34).clamp(0.62, 1.0);
            for label in ["dribble", "protect-ball", "hold-up-flank", "side-step"] {
                scale_legal_option_score(&mut options, label, dwell_damp);
            }
        }
        // Forward-pass-first release: when a forward teammate is sufficiently open, prioritise
        // releasing the ball forward and do it SOONER the longer the carrier has dwelt on it — so
        // he plays the open forward man instead of deliberating into a late, pressured square/back
        // pass (the reported "held forever then passed backward to the opponent" blunder). Floors
        // the best (forward-biased ranking) pass above the hold/dribble family and damps that dwell
        // family in proportion to openness × dwell. Broader than the narrow ~5-8m quick-forward
        // band above (any open forward option qualifies). Deferred to real shot/killer logic. Gate
        // OFF (default under test) ⇒ strength clamps to 0 ⇒ byte-identical no-op.
        if forward_pass_first_enabled()
            && pass_target_count > 0
            && !goal_attack_shot_blocks_alternatives
            && !observation.threaded_goal_pass_available
            && !must_shoot_near_goal(observation, self.role)
        {
            // Recognise a good forward option by its lane-aware quality, not just the
            // receiver's nearest-opponent openness, so a clear ball to an advanced runner
            // triggers the early release instead of a deliberated backward recycle. The
            // recognition value is `>=` the legacy openness, so it can only ADD releases.
            let forward_open = if dd_soccer_enable_forward_option_recognition() {
                observation
                    .best_forward_pass_option_quality
                    .max(observation.best_forward_pass_receiver_openness)
            } else {
                observation.best_forward_pass_receiver_openness
            };
            let strength = forward_pass_first_release_strength(
                forward_open,
                observation.perceived_time_on_ball_seconds,
            );
            if strength > 0.0 {
                let floor = (0.42 + 0.50 * strength).clamp(0.40, 0.95);
                ensure_min_legal_option_probability(&mut options, "pass1", floor);
                let dwell_damp = (1.0 - 0.50 * strength).clamp(0.45, 1.0);
                for label in ["dribble", "protect-ball", "hold-up-flank", "side-step"] {
                    scale_legal_option_score(&mut options, label, dwell_damp);
                }
            }
        }
        // Isolated attacking carrier (gated `DD_SOCCER_ENABLE_ISOLATED_CARRIER_DRIVE`; OFF ⇒
        // no-op, byte-identical). When a Forward / winger has the ball in the attacking half with
        // NO teammate ahead and no open forward pass, he used to deliberate into a panicked
        // BACKWARD pass. Instead: DRIVE at goal (sprint, then shoot in range) when few defenders
        // are behind the ball or there's space + a speed advantage; otherwise HOLD the ball up,
        // carrying it forward / on an angle at a controlled pace while teammates push up to join
        // the attack (the `front_line_carrier_support_cue` does the calling-forward). A
        // backward/square recycle is the very last resort in BOTH modes — damped hard, never
        // made illegal so it survives when nothing else is on.
        if isolated_carrier_drive_enabled() {
            if let Some(mode) =
                isolated_attacking_carrier_drive_mode(observation, self.role, outside_midfielder)
            {
                apply_isolated_carrier_drive_bias(
                    &mut options,
                    mode,
                    observation.yards_to_goal <= ISOLATED_CARRIER_SHOOT_YARDS,
                    shot_legal,
                );
            }
        }
        let mut options = normalize_action_options(options);
        annotate_tick_probabilities_from_scores(&mut options, dt_seconds);
        options
    }

    pub(crate) fn first_touch_action_options(
        &self,
        observation: &SoccerPomdpObservation,
        pass_target_count: usize,
        aerial_pass_target_count: usize,
    ) -> Vec<AgentActionOptionTrace> {
        let is_aerial = matches!(
            observation.incoming_ball_kind,
            IncomingBallKind::AerialCross | IncomingBallKind::AerialPass
        );
        let shot_label = if is_aerial {
            "header"
        } else {
            "first-time-shot"
        };
        let control_label = if is_aerial {
            "chest-control"
        } else {
            "control-touch"
        };
        let shot_legal = first_time_shot_decision_is_qualified_for_role(observation, self.role);
        let pass_legal = pass_target_count > 0;
        let flick_on_legal = is_aerial && aerial_pass_target_count > 0;
        let quick_pressure_bonus = 1.0
            + observation.perceived_pressure.clamp(0.0, 1.0) * 0.24
            + observation.decision_urgency.clamp(0.0, 1.0) * 0.18;
        // TWO complementary first-touch feasibilities, combined:
        //  (1) PHYSICAL/contact gate — can this ball actually be struck/redirected in one
        //      touch (pace, height, turn angle)? Near-identity (≈1.0) for a routine ground
        //      ball; HARD-suppresses (≈veto) awkward/fast/aerial ones so a settling touch wins.
        let pass_feasibility = observation.one_touch_pass_feasibility.clamp(0.0, 1.0);
        let shot_feasibility = observation.one_touch_shot_feasibility.clamp(0.0, 1.0);
        let pass_feasibility_gate = if pass_feasibility < ONE_TOUCH_VETO_FEASIBILITY {
            pass_feasibility * ONE_TOUCH_VETO_SUPPRESSION
        } else {
            pass_feasibility
        };
        let shot_feasibility_gate = if shot_feasibility < ONE_TOUCH_VETO_FEASIBILITY {
            shot_feasibility * ONE_TOUCH_VETO_SUPPRESSION
        } else {
            shot_feasibility
        };
        let one_touch_pass_blocked = pass_feasibility < ONE_TOUCH_VETO_FEASIBILITY;
        //  (2) FIELD-CONFIGURATION fit — does the whole-field shape (vector/embeddings/LP)
        //      support a first-time pass/shot here, vs. settling first? A softer multiplier.
        let pass_field_fit = observation
            .first_time_pass_field_feasibility
            .clamp(0.0, 1.0);
        let shot_field_fit = observation
            .first_time_shot_field_feasibility
            .clamp(0.0, 1.0);
        let shape_prior = observation.first_touch_shape_prior.clamp(0.0, 1.0);
        let pass_field_multiplier =
            (0.72 + pass_field_fit * 0.30 + shape_prior * 0.08).clamp(0.46, 1.10);
        let shot_field_multiplier =
            (0.70 + shot_field_fit * 0.32 + shape_prior * 0.08).clamp(0.44, 1.10);
        let control_field_multiplier =
            (1.0 + (1.0 - pass_field_fit.max(shot_field_fit)) * 0.18).clamp(1.0, 1.18);
        let short_upfield_ground_pass_fit = short_upfield_ground_pass_fit(observation);
        let first_time_forward_pass_multiplier =
            (1.0 + short_upfield_ground_pass_fit * (0.26 + pass_field_fit * 0.20)).clamp(1.0, 1.46);
        let killer_pressure = observation
            .killer_pass_goal_pressure
            .max(killer_pass_goal_pressure_score(observation))
            .max(single_pass_goal_thread_pressure_score(observation))
            .clamp(0.0, 1.0);
        let threaded_quality = threaded_goal_pass_quality_fit(observation);
        // A confirmed threaded receiver supersedes the reception-gated safe-pass count /
        // visible-forward-option preconditions (empty through a blocked lane — exactly when a
        // killer ball is the right call).
        let killer_pass_legal = observation.threaded_goal_pass_available
            && observation.yards_to_goal <= KILLER_PASS_MAX_YARDS_TO_GOAL
            && !clean_twenty_yard_shot_is_qualified(observation, self.role)
            && killer_pressure >= 0.32;
        let killer_pass_score = (observation.first_time_pass_score
            * quick_pressure_bonus
            * pass_field_multiplier
            * (0.72
                + killer_pressure * 0.66
                + threaded_quality * 0.28
                + observation
                    .threaded_goal_pass_expected_completion
                    .clamp(0.0, 1.0)
                    * 0.16
                + observation.threaded_goal_pass_stride_fit.clamp(0.0, 1.0) * 0.14))
            .clamp(0.02, 1.24);
        let mut options = vec![
            AgentActionOptionTrace::new(
                shot_label,
                (observation.first_time_shot_score
                    * quick_pressure_bonus
                    * shot_field_multiplier
                    * 0.52
                    * shot_feasibility_gate)
                    .clamp(0.0, 0.64),
                shot_legal,
            ),
            AgentActionOptionTrace::new(
                "killer-pass",
                killer_pass_score * pass_feasibility_gate,
                killer_pass_legal,
            ),
            AgentActionOptionTrace::new(
                "first-time-pass",
                (observation.first_time_pass_score
                    * quick_pressure_bonus
                    * pass_field_multiplier
                    * first_time_forward_pass_multiplier
                    * 0.86
                    * pass_feasibility_gate)
                    .clamp(0.0, 0.88),
                pass_legal,
            ),
            AgentActionOptionTrace::new(
                "flick-on",
                (observation.first_time_pass_score
                    * quick_pressure_bonus
                    * pass_field_multiplier
                    * (0.34
                        + ability01(self.skills.first_touch) * 0.30
                        + ability01(self.skills.flair_passing) * 0.18
                        + aerial_duel_skill_from_agent(self) * 0.18)
                    * pass_feasibility_gate)
                    .clamp(0.0, 0.82),
                flick_on_legal,
            ),
            AgentActionOptionTrace::new(
                control_label,
                (observation.control_touch_score
                    * control_field_multiplier
                    * (1.12 - observation.perceived_pressure * 0.18))
                    .clamp(0.02, 0.95),
                true,
            ),
        ];
        if killer_pass_legal {
            // Only floor the killer ball up if it is physically on; a coin-flip
            // one-touch through-ball must not be force-promoted over a settling touch.
            if !one_touch_pass_blocked {
                let floor =
                    (0.24 + killer_pressure * 0.50 + threaded_quality * 0.12).clamp(0.34, 0.82);
                ensure_min_legal_option_probability(&mut options, "killer-pass", floor);
            }
            if killer_pressure >= 0.48 {
                let recycle_multiplier = (1.0 - killer_pressure * 0.44).clamp(0.46, 1.0);
                scale_legal_option_score(&mut options, "first-time-pass", recycle_multiplier);
                scale_legal_option_score(&mut options, control_label, recycle_multiplier);
            }
        } else if pass_legal {
            // Under real pressure, taking a settling touch is dangerous — a
            // defender is right on you. If a clean first-time pass to an open
            // receiver is on, it must DOMINATE the controlling touch, not lose a
            // weighted coin-flip to it (the live bug: control-touch chosen over an
            // available first-time-pass to a wide-open teammate at openness 1.0 /
            // completion 0.97 under a 3yd press with zero forward space).
            let pass_quality = observation
                .best_pass_receiver_openness
                .max(observation.expected_pass_completion)
                .clamp(0.0, 1.0);
            let pressure = observation
                .perceived_pressure
                .max(observation.immediate_dispossession_risk)
                .clamp(0.0, 1.0);
            if pressure >= 0.45 && pass_quality >= 0.5 && !one_touch_pass_blocked {
                let dominance = (pressure * pass_quality).clamp(0.0, 1.0);
                let first_time_floor = (0.40 + dominance * 0.50).clamp(0.40, 0.92);
                ensure_min_legal_option_probability(
                    &mut options,
                    "first-time-pass",
                    first_time_floor,
                );
                let control_multiplier = (1.0 - dominance * 0.85).clamp(0.10, 1.0);
                scale_legal_option_score(&mut options, control_label, control_multiplier);
            }
            if short_upfield_ground_pass_fit >= 0.42 && !one_touch_pass_blocked {
                let forward_release_floor =
                    (0.34 + short_upfield_ground_pass_fit * 0.42 + pass_field_fit * 0.10)
                        .clamp(0.42, 0.86);
                ensure_min_legal_option_probability(
                    &mut options,
                    "first-time-pass",
                    forward_release_floor,
                );
                let control_multiplier =
                    (1.0 - short_upfield_ground_pass_fit * 0.46).clamp(0.36, 1.0);
                scale_legal_option_score(&mut options, control_label, control_multiplier);
            }
        }
        // Quick forward ground-pass priority on the first touch: when a short progressive ball
        // (≈5–8 m) to an OPEN, advanced teammate is available AND the one-touch is physically on,
        // play it FIRST-TIME rather than settling — floor the first-time-pass and trim the
        // controlling touch in proportion to the value. The chosen pass is delivered through the
        // MPC pass path. Gate OFF ⇒ value is 0 ⇒ byte-identical no-op.
        if pass_legal && !one_touch_pass_blocked && observation.quick_forward_pass_value > 0.0 {
            let quick_value = observation.quick_forward_pass_value.clamp(0.0, 1.0);
            let quick_floor = (0.44 + quick_value * 0.44).clamp(0.44, 0.88);
            ensure_min_legal_option_probability(&mut options, "first-time-pass", quick_floor);
            let control_multiplier = (1.0 - quick_value * 0.45).clamp(0.45, 1.0);
            scale_legal_option_score(&mut options, control_label, control_multiplier);
        }
        normalize_action_options(options)
    }

    pub(crate) fn support_action_options(
        &self,
        snapshot: &WorldSnapshot,
    ) -> Vec<AgentActionOptionTrace> {
        self.support_action_context(snapshot).options
    }

    fn support_action_context(&self, snapshot: &WorldSnapshot) -> SupportActionContext {
        let directive = snapshot.tactical_directive(self.team);
        let exploit_space_strategy_active =
            matches!(directive.attack_strategy, TeamAttackStrategy::ExploitSpace);
        let crash_box_strategy_active =
            matches!(directive.attack_strategy, TeamAttackStrategy::CrashTheBox);
        let possession_team = snapshot.controlled_possession_team();
        let width_shortage = if possession_team == Some(self.team) {
            possession_width_shortage(snapshot, directive)
        } else {
            0.0
        };
        let flank_policy_active =
            possession_team == Some(self.team) && directive.flank_attack_policy.is_flank();
        // A striker holding up in the opponent half, a midfielder/striker carrying at goal,
        // or a calm backfield holder lining up a long pass all pull attackers into roaming
        // runs to find space in behind.
        let backfield_long_pass_run = snapshot
            .backfield_long_pass_run_invite_for(self.id)
            .clamp(0.0, 1.0);
        let run_prediction =
            snapshot.off_ball_run_prediction_profile_for(self.id, self.home_position);
        let striker_attack = snapshot
            .striker_holder_in_opponent_half(self.team)
            .is_some()
            || snapshot
                .attacking_carrier_driving_at_goal(self.team)
                .is_some()
            || backfield_long_pass_run >= BACKFIELD_LONG_PASS_RUN_MIN_INVITE
            || snapshot.forward_attacking_momentum(self.team);
        let shape_support_urgency = attacking_shape_support_urgency(snapshot, self.team);
        let holder_pressure_urgency = holder_pressure_support_urgency(snapshot, self.team);
        let support_urgency =
            (shape_support_urgency * 0.55 + holder_pressure_urgency * 0.72).clamp(0.0, 1.0);
        let attacking_midfielder = self.role == PlayerRole::Midfielder
            && self.preferences.offensive_mindedness >= self.preferences.defensive_mindedness;
        let current_x = self.position.x.clamp(0.0, snapshot.field_width);
        let wide_midfielder = attacking_midfielder
            && (current_x < snapshot.field_width * 0.30 || current_x > snapshot.field_width * 0.70);
        let mut roam_weight = if striker_attack && wide_midfielder {
            0.46
        } else if striker_attack && attacking_midfielder {
            0.34
        } else if striker_attack && self.role == PlayerRole::Forward {
            0.22
        } else {
            0.10
        };
        if flank_policy_active {
            roam_weight =
                (roam_weight + directive.flank_overlap_run_probability * 0.18).clamp(0.10, 0.72);
        }
        if holder_pressure_urgency >= 0.22 {
            roam_weight = (roam_weight + holder_pressure_urgency * 0.22).clamp(0.10, 0.76);
        }
        if run_prediction.entropy > 0.0 {
            roam_weight = (roam_weight + run_prediction.entropy * 0.08).clamp(0.10, 0.80);
        }
        let special_targets = SupportSpecialTargets {
            check_to_ball: snapshot.check_to_ball_target_for(self.id, self.home_position),
            in_behind: snapshot.in_behind_run_target_for(self.id),
            exploit_space: snapshot.exploit_space_run_target_for(self.id, self.home_position),
            wide_outlet: snapshot.wide_possession_outlet_target_for(self.id, self.home_position),
            flank_cross_arrival: snapshot
                .flank_cross_arrival_target_for(self.id, self.home_position),
        };
        let current = snapshot.player_position(self.id).unwrap_or(self.position);
        let open = snapshot.open_space_for(self.id, self.home_position);
        let clear_lane_holder_context = snapshot.ball.holder.and_then(|holder_id| {
            if holder_id == self.id {
                return None;
            }
            let holder = snapshot
                .players
                .iter()
                .find(|player| player.id == holder_id)?;
            if holder.team != self.team || !snapshot.player_can_see_player(self.id, holder.id) {
                return None;
            }
            let holder_position = snapshot.player_snapshot_position(holder);
            let distance = current.distance(holder_position);
            let lane_open = snapshot.clear_line(holder_position, current, self.team.other(), 2.0);
            (lane_open && distance > SUPPORT_HOLDER_CLEAR_LANE_START_DISTANCE_YARDS)
                .then_some(distance)
        });
        let clear_lane_support_hold_signal = clear_lane_holder_context
            .map(|distance| {
                let distance_fit = ((distance - SUPPORT_HOLDER_CLEAR_LANE_START_DISTANCE_YARDS)
                    / 10.0)
                    .clamp(0.0, 1.0);
                let pressure_relief = (1.0
                    - holder_pressure_urgency / PRESSURED_SUPPORT_SPRINT_URGENCY.max(1e-6))
                .clamp(0.0, 1.0);
                distance_fit * pressure_relief
            })
            .unwrap_or(0.0);
        let guarded_special_target = |target: Vec2| {
            snapshot.shape_guarded_support_point(
                self.id,
                target,
                &[open, self.home_position],
                self.home_position,
                false,
            )
        };
        let guarded_exploit_space_target = |target: Vec2| {
            snapshot.shape_guarded_support_point(self.id, target, &[], self.home_position, false)
        };
        let special_score = |target: Vec2, base: f64| {
            let forward = ((target.y - current.y) * self.team.attack_dir()).clamp(-8.0, 24.0);
            let openness = pass_receiver_openness_for_snapshots_with_teammates(
                &snapshot.players,
                self.team,
                target,
                Some(self.id),
            );
            let pass_lane =
                if snapshot.clear_line(snapshot.ball.position, target, self.team.other(), 2.0) {
                    1.0
                } else {
                    0.0
                };
            let space_fit = (snapshot.space_score_at(target, self.team) / 14.0).clamp(0.0, 1.0);
            (base
                + forward.max(0.0) * 0.034
                + openness * 0.26
                + pass_lane * 0.22
                + space_fit * 0.24)
                * self.preferences.open_space_bias.clamp(0.25, 1.0)
        };
        let mut options = vec![
            AgentActionOptionTrace::new(
                "support-shape",
                (1.0 - roam_weight)
                    * self.preferences.open_space_bias
                    * (1.0
                        + support_urgency * 0.14
                        + run_prediction.weight_for_label("support-shape") * 0.18),
                true,
            ),
            AgentActionOptionTrace::new(
                "support-roam",
                roam_weight
                    * self.preferences.open_space_bias
                    * (1.0 + support_urgency * 0.18 + run_prediction.entropy * 0.16),
                true,
            ),
        ];
        if let Some(target) = special_targets.check_to_ball {
            let target = guarded_special_target(target);
            let ball_gain = (current.distance(snapshot.ball.position)
                - target.distance(snapshot.ball.position))
            .max(0.0);
            options.push(AgentActionOptionTrace::new(
                "check-to-ball",
                special_score(
                    target,
                    0.42 + ball_gain.min(8.0) * 0.030
                        + shape_support_urgency * 2.6
                        + holder_pressure_urgency * 0.58
                        + run_prediction.weight_for_label("check-to-ball") * 0.42
                        - clear_lane_support_hold_signal * 1.05,
                ),
                true,
            ));
        }
        if let Some(target) = special_targets.in_behind {
            let target = snapshot.shape_guarded_support_point(
                self.id,
                target,
                &[],
                self.home_position,
                false,
            );
            options.push(AgentActionOptionTrace::new(
                "run-in-behind",
                special_score(
                    target,
                    0.68 + shape_support_urgency * 0.10
                        + holder_pressure_urgency * 0.18
                        + backfield_long_pass_run * 0.46
                        + run_prediction.weight_for_label("run-in-behind") * 0.58
                        + run_prediction.off_ball_pitch_value_score_lift(),
                ),
                true,
            ));
        }
        if let Some(target) = special_targets.exploit_space {
            let target = guarded_exploit_space_target(target);
            options.push(AgentActionOptionTrace::new(
                "exploit-space-run",
                special_score(
                    target,
                    0.66 + shape_support_urgency * 0.16
                        + holder_pressure_urgency * 0.14
                        + run_prediction.weight_for_label("exploit-space-run") * 0.54
                        + run_prediction.off_ball_pitch_value_score_lift(),
                ),
                true,
            ));
        }
        if let Some(target) = special_targets.wide_outlet {
            let target = guarded_special_target(target);
            let touchline_fit = ((target.x - snapshot.field_width * 0.5).abs()
                / (snapshot.field_width * 0.5).max(1.0))
            .clamp(0.0, 1.0);
            options.push(AgentActionOptionTrace::new(
                "wide-outlet",
                special_score(
                    // Wingers getting open on the flank is a priority — lift the base
                    // preference and the wider-is-better term so wide attackers push to space
                    // on the wing more readily than drifting inside.
                    target,
                    0.72 + touchline_fit * 0.34
                        + width_shortage * WIDE_OUTLET_WIDTH_SHORTAGE_SCORE_LIFT
                        + shape_support_urgency * 0.20
                        + holder_pressure_urgency * 0.34
                        + run_prediction.weight_for_label("wide-outlet") * 0.46
                        + if flank_policy_active {
                            0.28 + directive.flank_overlap_run_probability * 0.22
                        } else {
                            0.0
                        },
                ),
                true,
            ));
        }
        if let Some(target) = special_targets.flank_cross_arrival {
            let target = guarded_special_target(target);
            let goalward = ((target.y - current.y) * self.team.attack_dir()).clamp(-4.0, 24.0);
            let centrality = (1.0
                - ((target.x - snapshot.field_width * 0.5).abs()
                    / (snapshot.field_width * 0.5).max(1.0)))
            .clamp(0.0, 1.0);
            let crash_box_bonus = if crash_box_strategy_active {
                if self.role == PlayerRole::Forward {
                    0.54
                } else {
                    0.22
                }
            } else {
                0.0
            };
            options.push(AgentActionOptionTrace::new(
                "shot-creation-run",
                special_score(
                    target,
                    0.64 + goalward.max(0.0) * 0.026
                        + centrality * 0.14
                        + shape_support_urgency * 0.20
                        + holder_pressure_urgency * 0.18
                        + directive.flank_overlap_run_probability * 0.22
                        + run_prediction.weight_for_label("shot-creation-run") * 0.48
                        + run_prediction.off_ball_pitch_value_score_lift() * 0.70
                        + crash_box_bonus,
                ),
                true,
            ));
        }
        let natural_support = snapshot.attacking_support_movement_for_with_targets(
            self.id,
            self.home_position,
            false,
            &special_targets,
        );
        let mut vacate_support = snapshot.attacking_support_movement_for_with_targets(
            self.id,
            self.home_position,
            true,
            &special_targets,
        );
        vacate_support.action_label = "vacate-space";
        let vacate_target = snapshot.shape_guarded_support_point(
            self.id,
            vacate_support.point,
            &[open, self.home_position],
            self.home_position,
            true,
        );
        let current_teammate_pressure =
            snapshot.teammate_occupied_space_pressure_at(self.team, current, Some(self.id));
        let target_teammate_pressure =
            snapshot.teammate_occupied_space_pressure_at(self.team, vacate_target, Some(self.id));
        let vacate_relief = (current_teammate_pressure - target_teammate_pressure).max(0.0);
        if self.role != PlayerRole::Goalkeeper
            && (current_teammate_pressure >= 0.18
                || holder_pressure_urgency >= 0.16
                || shape_support_urgency >= 0.14)
        {
            options.push(AgentActionOptionTrace::new(
                "vacate-space",
                special_score(
                    vacate_target,
                    0.34 + vacate_relief * 0.82
                        + current_teammate_pressure * 0.28
                        + shape_support_urgency * 0.20
                        + holder_pressure_urgency * 0.24,
                ),
                true,
            ));
        }
        // Lane-yield: I'm the middle of a three-in-a-line and the only thing blocking a
        // longer pass from the carrier to a farther team-mate. Step out of the lane to
        // space (forward when attacking, back for a deep defender) so the carrier can
        // prefer the longer ball. Scored to win only when the opening is genuinely valuable.
        if self.role != PlayerRole::Goalkeeper {
            if let Some((lane_yield_target, lane_yield_strength)) =
                snapshot.pass_lane_yield_target_for(self.id, self.home_position)
            {
                options.push(AgentActionOptionTrace::new(
                    "lane-yield",
                    special_score(lane_yield_target, 0.30 + lane_yield_strength * 0.95),
                    true,
                ));
            }
        }
        let home_x = self.home_position.x.clamp(0.0, snapshot.field_width);
        let wide_home =
            home_x < snapshot.field_width * 0.34 || home_x > snapshot.field_width * 0.66;
        let policy_overlap_candidate = flank_policy_active
            && self.role != PlayerRole::Goalkeeper
            && (wide_home
                || wide_midfielder
                || attacking_midfielder
                || (self.role == PlayerRole::Forward && !crash_box_strategy_active));
        if natural_support.action_label == "overlap-run" || policy_overlap_candidate {
            let overlap_support = if natural_support.action_label == "overlap-run" {
                natural_support
            } else {
                let mut target = snapshot.attacking_support_movement_for_with_targets(
                    self.id,
                    self.home_position,
                    true,
                    &special_targets,
                );
                target.action_label = "overlap-run";
                target
            };
            let touchline_fit = ((overlap_support.point.x - snapshot.field_width * 0.5).abs()
                / (snapshot.field_width * 0.5).max(1.0))
            .clamp(0.0, 1.0);
            let forward =
                ((overlap_support.point.y - current.y) * self.team.attack_dir()).clamp(-4.0, 24.0);
            options.push(AgentActionOptionTrace::new(
                "overlap-run",
                special_score(
                    overlap_support.point,
                    0.56 + shape_support_urgency * 0.18
                        + holder_pressure_urgency * 0.22
                        + if flank_policy_active {
                            touchline_fit * 0.12
                                + forward.max(0.0) * 0.010
                                + directive.flank_overlap_run_probability * 0.30
                        } else {
                            0.0
                        },
                ),
                true,
            ));
            let min_share = if crash_box_strategy_active && self.role == PlayerRole::Forward {
                FLANK_OVERLAP_MIN_OPTION_SHARE
            } else if flank_policy_active {
                directive
                    .flank_overlap_run_probability
                    .clamp(FLANK_OVERLAP_MIN_OPTION_SHARE, 0.72)
            } else {
                FLANK_OVERLAP_MIN_OPTION_SHARE
            };
            ensure_min_legal_option_probability(&mut options, "overlap-run", min_share);
        }
        if options
            .iter()
            .any(|option| option.legal && option.label == "run-in-behind")
        {
            let run_floor = (0.14
                + if striker_attack { 0.06 } else { 0.0 }
                + shape_support_urgency * 0.10
                + holder_pressure_urgency * 0.14)
                .clamp(0.18, 0.38);
            ensure_min_legal_option_probability(&mut options, "run-in-behind", run_floor);
        }
        if crash_box_strategy_active
            && options
                .iter()
                .any(|option| option.legal && option.label == "shot-creation-run")
        {
            let crash_floor = (0.36
                + if self.role == PlayerRole::Forward {
                    0.20
                } else {
                    0.06
                }
                + directive.flank_overlap_run_probability * 0.12
                + holder_pressure_urgency * 0.08)
                .clamp(0.42, 0.68);
            if self.role == PlayerRole::Forward {
                scale_legal_option_score(&mut options, "overlap-run", 0.34);
            }
            ensure_min_legal_option_probability(&mut options, "shot-creation-run", crash_floor);
        }
        if options
            .iter()
            .any(|option| option.legal && option.label == "exploit-space-run")
        {
            let strategy_floor = if exploit_space_strategy_active {
                0.42
            } else {
                0.0
            };
            let exploit_floor = (0.16
                + strategy_floor
                + shape_support_urgency * 0.10
                + holder_pressure_urgency * 0.12)
                .clamp(
                    0.26,
                    if exploit_space_strategy_active {
                        0.66
                    } else {
                        0.54
                    },
                );
            ensure_min_legal_option_probability(&mut options, "exploit-space-run", exploit_floor);
        }
        if options
            .iter()
            .any(|option| option.legal && option.label == "wide-outlet")
        {
            let wide_floor = (0.12
                + width_shortage * WIDE_OUTLET_WIDTH_SHORTAGE_FLOOR_LIFT
                + shape_support_urgency * 0.08
                + holder_pressure_urgency * 0.10
                + if flank_policy_active { 0.10 } else { 0.0 })
            .clamp(0.18, 0.50);
            ensure_min_legal_option_probability(&mut options, "wide-outlet", wide_floor);
            if width_shortage > 0.0 {
                let support_shape_multiplier = (1.0
                    - width_shortage * WIDE_OUTLET_NARROW_SHAPE_SUPPORT_DAMPING)
                    .clamp(0.76, 1.0);
                let support_roam_multiplier = (1.0
                    - width_shortage * WIDE_OUTLET_NARROW_SHAPE_SUPPORT_DAMPING * 0.55)
                    .clamp(0.84, 1.0);
                scale_legal_option_score(&mut options, "support-shape", support_shape_multiplier);
                scale_legal_option_score(&mut options, "support-roam", support_roam_multiplier);
            }
        }
        if clear_lane_support_hold_signal > 0.0 {
            scale_legal_option_score(
                &mut options,
                "check-to-ball",
                (1.0 - clear_lane_support_hold_signal * 0.78).clamp(0.18, 1.0),
            );
            scale_legal_option_score(
                &mut options,
                "support-shape",
                1.0 + clear_lane_support_hold_signal * 0.38,
            );
            scale_legal_option_score(
                &mut options,
                "support-roam",
                1.0 + clear_lane_support_hold_signal * 0.12,
            );
            scale_legal_option_score(
                &mut options,
                "wide-outlet",
                1.0 + clear_lane_support_hold_signal * 0.42,
            );
            scale_legal_option_score(
                &mut options,
                "exploit-space-run",
                1.0 + clear_lane_support_hold_signal * 0.24,
            );
            scale_legal_option_score(
                &mut options,
                "run-in-behind",
                1.0 + clear_lane_support_hold_signal * 0.18,
            );
            ensure_min_legal_option_family_probability(
                &mut options,
                &[
                    "support-shape",
                    "wide-outlet",
                    "exploit-space-run",
                    "run-in-behind",
                ],
                (0.30 + clear_lane_support_hold_signal * 0.28).clamp(0.30, 0.62),
            );
        }
        if holder_pressure_urgency >= PRESSURED_SUPPORT_SPRINT_URGENCY {
            if options
                .iter()
                .any(|option| option.legal && option.label == "check-to-ball")
            {
                let check_floor =
                    (0.24 + holder_pressure_urgency * 0.26 + shape_support_urgency * 0.08)
                        .clamp(0.30, 0.58);
                ensure_min_legal_option_probability(&mut options, "check-to-ball", check_floor);
            } else if options
                .iter()
                .any(|option| option.legal && option.label == "wide-outlet")
            {
                let wide_floor = (0.20
                    + width_shortage * WIDE_OUTLET_WIDTH_SHORTAGE_FLOOR_LIFT
                    + holder_pressure_urgency * 0.20
                    + if flank_policy_active { 0.08 } else { 0.0 })
                .clamp(0.28, 0.58);
                ensure_min_legal_option_probability(&mut options, "wide-outlet", wide_floor);
            }
        }
        if backfield_long_pass_run >= BACKFIELD_LONG_PASS_RUN_MIN_INVITE
            && options
                .iter()
                .any(|option| option.legal && option.label == "run-in-behind")
        {
            let run_floor = (BACKFIELD_LONG_PASS_RUN_OPTION_FLOOR_BASE
                + backfield_long_pass_run * BACKFIELD_LONG_PASS_RUN_OPTION_FLOOR_LIFT)
                .clamp(BACKFIELD_LONG_PASS_RUN_OPTION_FLOOR_BASE, 0.72);
            ensure_min_legal_option_probability(&mut options, "run-in-behind", run_floor);
            scale_legal_option_score(&mut options, "support-shape", 0.82);
            scale_legal_option_score(&mut options, "support-roam", 0.90);
        }
        if exploit_space_strategy_active
            && options
                .iter()
                .any(|option| option.legal && option.label == "exploit-space-run")
        {
            scale_legal_option_score(&mut options, "wide-outlet", 0.72);
            scale_legal_option_score(&mut options, "support-shape", 0.82);
            scale_legal_option_score(&mut options, "support-roam", 0.88);
            ensure_min_legal_option_probability(&mut options, "exploit-space-run", 0.58);
        }
        if flank_policy_active
            && !(crash_box_strategy_active && self.role == PlayerRole::Forward)
            && options
                .iter()
                .any(|option| option.legal && option.label == "overlap-run")
        {
            ensure_min_legal_option_probability(
                &mut options,
                "overlap-run",
                directive
                    .flank_overlap_run_probability
                    .clamp(FLANK_OVERLAP_MIN_OPTION_SHARE, 0.72),
            );
        }
        // Hold-and-summon support (off-ball half): if this player's on-ball teammate is waiting
        // and has summoned it, make the matching response the priority in its OWN decision — a
        // POMDP option-score boost, not a world movement override. The summoned-forward urgency
        // also feeds `in_behind_run_target_for`, so the run target itself exists to be boosted.
        if let Some(summon) = snapshot.support_summon_for(self.id) {
            match summon {
                SupportSummonKind::MoveForward => {
                    // "Make the run, I'll find you" — push forward into space.
                    for label in ["run-in-behind", "exploit-space-run"] {
                        if options.iter().any(|o| o.legal && o.label == label) {
                            ensure_min_legal_option_probability(
                                &mut options,
                                label,
                                HOLD_FOR_SUPPORT_SUMMON_RUN_FLOOR,
                            );
                        }
                    }
                }
                SupportSummonKind::AssumePosition => {
                    // "Take up your position" — find the open passing lane / settle into the slot.
                    if options
                        .iter()
                        .any(|o| o.legal && o.label == "check-to-ball")
                    {
                        ensure_min_legal_option_probability(
                            &mut options,
                            "check-to-ball",
                            HOLD_FOR_SUPPORT_SUMMON_POSITION_FLOOR,
                        );
                    }
                    scale_legal_option_score(
                        &mut options,
                        "support-shape",
                        HOLD_FOR_SUPPORT_SUMMON_SHAPE_SCALE,
                    );
                }
            }
        }
        let options = normalize_action_options(options);
        SupportActionContext {
            options,
            special_targets,
            open,
        }
    }

    pub(crate) fn defensive_action_options(
        &self,
        snapshot: &WorldSnapshot,
        directive: &TeamTacticalDirective,
        dt_seconds: f64,
    ) -> Vec<AgentActionOptionTrace> {
        let holder_context = snapshot.ball.holder.and_then(|holder| {
            let holder_player = snapshot.players.iter().find(|p| p.id == holder)?;
            (holder_player.team == self.team.other()).then(|| {
                let holder_position = snapshot.player_snapshot_position(holder_player);
                let distance = self.position.distance(holder_position);
                let my_depth = snapshot.depth_from_own_goal_y(self.team, self.position.y);
                let holder_depth = snapshot.depth_from_own_goal_y(self.team, holder_position.y);
                let goal_side_or_level = my_depth <= holder_depth + 0.85;
                let defender_steal_skill = ability01(self.skills.defending) * 0.50
                    + ability01(self.skills.aggression) * 0.16
                    + ability01(self.skills.acceleration) * 0.16
                    + ability01(self.skills.strength) * 0.18;
                let holder_security = ability01(holder_player.skills.dribbling) * 0.44
                    + ability01(holder_player.skills.first_touch) * 0.26
                    + ability01(holder_player.skills.strength) * 0.18
                    + ability01(holder_player.skills.acceleration) * 0.12;
                (
                    distance,
                    goal_side_or_level,
                    defender_steal_skill,
                    holder_security,
                )
            })
        });
        let mut tackle_legal = holder_context
            .map(
                |(distance, goal_side_or_level, defender_steal_skill, holder_security)| {
                    goal_side_or_level
                        && distance <= 1.85
                        && (distance <= 1.35 || defender_steal_skill + 0.16 >= holder_security)
                },
            )
            .unwrap_or(false);
        if self.role == PlayerRole::Goalkeeper {
            tackle_legal = tackle_legal && snapshot.goalkeeper_direct_intervention_is_safe(self.id);
        }
        let defending = ability01(self.skills.defending);
        let aggression = ability01(self.skills.aggression);
        let tackle_contact_fit = holder_context
            .map(|(distance, _, defender_steal_skill, holder_security)| {
                let closeness = (1.0 - (distance - 1.0) / 1.75).clamp(0.0, 1.0);
                let skill_edge = (defender_steal_skill - holder_security + 0.26).clamp(0.0, 0.72);
                (closeness * 0.72 + skill_edge * 0.28).clamp(0.0, 1.0)
            })
            .unwrap_or(0.0);
        let tackle_score = ((defending * 0.6 + aggression * 0.4)
            * directive.press_intensity
            * tackle_contact_fit
            * 0.38)
            .clamp(0.005, 0.22);
        let defensive_mindedness = self.preferences.defensive_mindedness.clamp(0.0, 1.0);
        let offensive_mindedness = self.preferences.offensive_mindedness.clamp(0.0, 1.0);
        let press = directive.press_intensity.clamp(0.0, 1.0);
        let (shape_score, roam_score) = if self.role == PlayerRole::Goalkeeper {
            let ball_distance = self.position.distance(snapshot.ball.position);
            let close_ball = (1.0 - ball_distance / 12.0).clamp(0.0, 1.0);
            let line_recovery = snapshot.goalkeeper_line_recovery_sprint_active(self.id);
            (
                (4.70 + defensive_mindedness * 0.44 + if line_recovery { 0.85 } else { 0.0 })
                    .clamp(4.70, 6.05),
                if line_recovery {
                    0.02
                } else {
                    (0.04 + close_ball * 0.10 + press * 0.015).clamp(0.04, 0.16)
                },
            )
        } else {
            (
                (0.44 + defensive_mindedness * 0.46 + (1.0 - press) * 0.18).clamp(0.10, 1.18),
                (0.08 + offensive_mindedness * 0.22 + aggression * 0.14 + press * 0.30)
                    .clamp(0.04, 0.82),
            )
        };
        let press_cover_legal = self.role != PlayerRole::Goalkeeper && holder_context.is_some();
        let press_cover_score = ((0.10
            + press * 0.32
            + defending * 0.18
            + aggression * 0.12
            + tackle_contact_fit * 0.18)
            * (0.70 + defensive_mindedness * 0.30))
            .clamp(0.02, 0.92);
        let mut options = vec![
            AgentActionOptionTrace::new("tackle", tackle_score, tackle_legal),
            AgentActionOptionTrace::new("defend-shape", shape_score, true),
            AgentActionOptionTrace::new("defend-roam", roam_score, true),
            AgentActionOptionTrace::new("press-cover", press_cover_score, press_cover_legal),
        ];
        if self.role == PlayerRole::Goalkeeper {
            ensure_min_legal_option_probability(&mut options, "defend-shape", 0.97);
        }
        let mut options = normalize_action_options(options);
        annotate_tick_probability_from_score(&mut options, "tackle", dt_seconds);
        options
    }

    /// The opponent carrier this defender would throw a committed SLIDE tackle at this
    /// tick, plus the per-second appetite for doing so, or `None`. A slide is the
    /// last-ditch challenge — reserved for a carrier moving too fast to contain on foot
    /// who is driving at our goal — so the gate is deliberately narrow:
    ///
    /// * keeper never slides (outfield); a keeper holding the ball in hand is untouchable;
    /// * the carrier must be inside the slide-lunge reach band but not at point-blank
    ///   (where a standing steal already serves);
    /// * the defender must NOT be behind the carrier — no tackle wins a ball that is led
    ///   in front of the holder's body (the user's hard rule);
    /// * the carrier must actually be running fast (you do not slide a shielding holder);
    /// * the advancing-carrier urgency model must favour COMMITTING over containing — a
    ///   lone last defender told to contain must stay on his feet (a missed slide there
    ///   springs a clean run on goal).
    ///
    /// Returns `None` (drawing no RNG) whenever any gate fails, so a defender not in a
    /// genuine last-ditch window never perturbs the decision RNG stream.
    pub(crate) fn slide_tackle_decision(&self, snapshot: &WorldSnapshot) -> Option<(usize, f64)> {
        if !snapshot.slide_tackle_enabled || self.role == PlayerRole::Goalkeeper {
            return None;
        }
        let holder_id = snapshot.ball.holder?;
        let holder = snapshot.players.iter().find(|p| p.id == holder_id)?;
        if holder.team != self.team.other() {
            return None;
        }
        let holder_pos = snapshot.player_snapshot_position(holder);
        if holder.role == PlayerRole::Goalkeeper
            && snapshot.point_in_own_penalty_area(holder.team, holder_pos)
        {
            return None;
        }
        let distance = self.position.distance(holder_pos);
        if distance < SLIDE_TACKLE_MIN_REACH_YARDS || distance > SLIDE_TACKLE_MAX_REACH_YARDS {
            return None;
        }
        // No tackle wins the ball from behind the carrier — the body shields the led ball.
        if snapshot.tackle_blocked_from_behind(holder_id, self.position) {
            return None;
        }
        // You do not slide a stationary/shielding holder — only one running away from
        // (or at) you faster than you can stay with on your feet.
        let holder_speed = snapshot
            .player_velocity(holder_id)
            .unwrap_or(holder.velocity)
            .len();
        if holder_speed < SLIDE_TACKLE_MIN_HOLDER_SPEED_YPS {
            return None;
        }
        // Commit only when the model rates committing over containing (the same urgency
        // that drives the standing lunge): >1.0 = nearest presser to a carrier driving at
        // our goal with cover behind, or deep danger near our own goal.
        let urgency = snapshot.advancing_carrier_steal_urgency(self.id);
        if urgency < SLIDE_TACKLE_MIN_COMMIT_URGENCY {
            return None;
        }
        let commit01 = ((urgency - SLIDE_TACKLE_MIN_COMMIT_URGENCY) / CARRIER_ADVANCE_STEAL_BOOST)
            .clamp(0.0, 1.0);
        if commit01 <= 0.0 {
            return None;
        }
        let defending = ability01(self.skills.defending);
        let aggression = ability01(self.skills.aggression);
        let directive = snapshot.tactical_directive(self.team);
        let appetite =
            ((defending * 0.34 + aggression * 0.50) * directive.press_intensity * commit01)
                .clamp(0.0, SLIDE_TACKLE_MAX_DECISION_PROBABILITY);
        (appetite > 0.0).then_some((holder_id, appetite))
    }

    pub(crate) fn immediate_defensive_steal_target(
        &self,
        snapshot: &WorldSnapshot,
    ) -> Option<usize> {
        if self.role == PlayerRole::Goalkeeper {
            return None;
        }
        let holder = snapshot.ball.holder?;
        let holder_player = snapshot
            .players
            .iter()
            .find(|player| player.id == holder && player.team == self.team.other())?;
        let holder_position = snapshot.player_snapshot_position(holder_player);
        // Can't steal from a goalkeeper holding the ball in his own box — it is in
        // his hands (he is bound only by the handling time limit instead).
        if holder_player.role == PlayerRole::Goalkeeper
            && snapshot.point_in_own_penalty_area(holder_player.team, holder_position)
        {
            return None;
        }
        let distance = self.position.distance(holder_position);
        if distance > DEFENSIVE_IMMEDIATE_STEAL_RADIUS_YARDS {
            return None;
        }

        let my_depth = snapshot.depth_from_own_goal_y(self.team, self.position.y);
        let holder_depth = snapshot.depth_from_own_goal_y(self.team, holder_position.y);
        let goal_side_or_level = my_depth <= holder_depth + 0.65;

        let defender_steal_skill = ability01(self.skills.defending) * 0.46
            + ability01(self.skills.aggression) * 0.18
            + ability01(self.skills.acceleration) * 0.16
            + ability01(self.skills.strength) * 0.20;
        let holder_security = ability01(holder_player.skills.dribbling) * 0.42
            + ability01(holder_player.skills.first_touch) * 0.28
            + ability01(holder_player.skills.strength) * 0.18
            + ability01(holder_player.skills.acceleration) * 0.12;
        let wrong_side_emergency_contact = !goal_side_or_level
            && distance <= DEFENSIVE_IMMEDIATE_STEAL_CLEAN_RADIUS_YARDS
            && defender_steal_skill + 0.20 >= holder_security;
        if !goal_side_or_level && !wrong_side_emergency_contact {
            return None;
        }
        let clean_contact = distance <= DEFENSIVE_IMMEDIATE_STEAL_CLEAN_RADIUS_YARDS
            && defender_steal_skill + if goal_side_or_level { 0.10 } else { 0.20 }
                >= holder_security;

        let exposed_dribble = holder_player
            .last_decision
            .as_ref()
            .map(|decision| {
                let action = normalize_soccer_action_label(&decision.action);
                is_dribble_action_label(action)
                    && (decision.observation.immediate_dispossession_risk >= 0.58
                        || decision.observation.perceived_pressure >= 0.56
                        || decision.observation.excessive_hold_pressure >= 0.36)
            })
            .unwrap_or(false)
            && distance <= DEFENSIVE_IMMEDIATE_STEAL_RADIUS_YARDS
            && goal_side_or_level
            && defender_steal_skill + 0.18 >= holder_security;

        // Ball protection: a ball shielded at the holder's feet (body between ball
        // and this defender, low pace) is hard to win; a ball pushed ahead at
        // speed toward us is exposed. A contestable (low-protection) ball within a
        // yard of us is stolen with high urgency regardless of holder skill.
        let ball_protection =
            snapshot.ball_protection_against(holder_player, holder_position, self.position);
        let ball_distance = self.position.distance(snapshot.ball.position);
        let contestable_close = ball_distance <= CONTESTABLE_STEAL_RADIUS_YARDS
            && ball_protection < CONTESTABLE_PROTECTION_THRESHOLD
            && goal_side_or_level;

        (clean_contact || exposed_dribble || contestable_close).then_some(holder)
    }

    fn loose_ball_action_options(
        &self,
        my_distance: f64,
        closer_teammates: usize,
        goalkeeper_can_recover: bool,
        fifty_fifty_duel: bool,
        must_attack: bool,
    ) -> Vec<AgentActionOptionTrace> {
        if fifty_fifty_duel {
            return single_action_option(FIFTY_FIFTY_DUEL_ACTION_LABEL);
        }

        let recover_legal =
            (must_attack || closer_teammates < 2) && my_distance <= 46.0 && goalkeeper_can_recover;
        let distance_fit = (1.0 - my_distance / 46.0).clamp(0.0, 1.0);
        let role_fit = match self.role {
            PlayerRole::Goalkeeper => 0.42,
            PlayerRole::Defender => 0.72,
            PlayerRole::Midfielder => 0.92,
            PlayerRole::Forward => 0.84,
        };
        let teammate_priority = match closer_teammates {
            0 => 1.0,
            1 => 0.70,
            _ if must_attack => 0.82,
            _ => 0.0,
        };
        let recover_score = if recover_legal {
            (0.34
                + distance_fit * 0.46
                + teammate_priority * 0.22
                + self.preferences.offensive_mindedness.clamp(0.0, 1.0) * 0.08
                + if must_attack { 0.36 } else { 0.0 })
                * role_fit
        } else {
            0.0
        }
        .clamp(0.0, 1.10);
        let hold_score = if must_attack {
            0.02
        } else if recover_legal {
            (0.16
                + self.preferences.defensive_mindedness.clamp(0.0, 1.0) * 0.10
                + (1.0 - teammate_priority) * 0.12)
                .clamp(0.04, 0.34)
        } else {
            1.0
        };

        normalize_action_options(vec![
            AgentActionOptionTrace::new(ACTION_LABEL_RECOVER, recover_score, recover_legal),
            AgentActionOptionTrace::new("hold", hold_score, true),
        ])
    }

    fn decision_trace(
        &self,
        snapshot: &WorldSnapshot,
        mdp_state: SoccerMdpState,
        observation: SoccerPomdpObservation,
        belief: BeliefState,
        operation_order: Vec<String>,
        action_options: Vec<AgentActionOptionTrace>,
        action: &SoccerAction,
        action_label: impl Into<String>,
    ) -> AgentDecisionTrace {
        let scheduled_index = snapshot.scheduled_player_index(self.id);
        let action_label = action_label.into();
        let action_target = self.action_target_trace(action, snapshot);
        let mdp_mpc_comparison = if snapshot.trace_mdp_mpc_comparison {
            player_mdp_mpc_comparison_trace(snapshot, self, &action_target, &action_label)
        } else {
            None
        };
        AgentDecisionTrace {
            mdp_state,
            observation,
            belief,
            operation_order,
            scheduled_index,
            action_options,
            action_target,
            mdp_mpc_comparison,
            learned_mpc_replan: None,
            behavior_policy_probability: self.pending_policy_behavior_probability,
            action: action_label,
        }
    }

    pub(crate) fn action_target_trace(
        &self,
        action: &SoccerAction,
        snapshot: &WorldSnapshot,
    ) -> Option<AgentActionTargetTrace> {
        let (point, player_id, dribble_touch) = match action {
            SoccerAction::HoldShape => (self.home_position, None, None),
            SoccerAction::MoveTo(target)
            | SoccerAction::Dribble(target)
            | SoccerAction::ControlTouch { target } => (*target, None, None),
            SoccerAction::DribbleMove { target, touch, .. } => (*target, None, Some(*touch)),
            SoccerAction::Pass {
                target_player,
                power,
                flight,
                ..
            } => {
                let resolved_target = target_player
                    .or_else(|| {
                        if flight.is_aerial() {
                            snapshot.best_aerial_pass_target(self.id)
                        } else {
                            snapshot.best_visible_pass_target(self.id)
                        }
                    })
                    .or_else(|| snapshot.best_pass_target(self.id));
                let passer_position = snapshot.player_position(self.id).unwrap_or(self.position);
                let point = resolved_target
                    .and_then(|id| {
                        let target_position = snapshot.player_position(id)?;
                        let is_cross = pass_would_be_cross(
                            passer_position,
                            target_position,
                            self.team,
                            snapshot.field_width,
                            snapshot.field_length,
                        );
                        let speed =
                            pass_speed_yps_from_power(*power, *flight, is_cross, &self.skills);
                        snapshot
                            .anticipated_pass_reception_point(self.id, id, *flight, speed)
                            .or(Some(target_position))
                    })
                    .unwrap_or_else(|| {
                        Vec2::new(
                            self.position.x,
                            self.position.y + 18.0 * self.team.attack_dir(),
                        )
                        .clamp_to_pitch(snapshot.field_width, snapshot.field_length)
                    });
                (point, resolved_target, None)
            }
            SoccerAction::Clearance { target, .. } | SoccerAction::RouteOne { target, .. } => {
                (*target, None, None)
            }
            SoccerAction::Shoot { .. } => (
                Vec2::new(
                    snapshot.field_width * 0.5,
                    self.team.goal_y(snapshot.field_length),
                ),
                None,
                None,
            ),
            SoccerAction::Tackle { target_player }
            | SoccerAction::SlideTackle { target_player } => {
                let point = snapshot
                    .player_position(*target_player)
                    .unwrap_or(snapshot.ball.position);
                (point, Some(*target_player), None)
            }
        };
        let point = point.clamp_to_pitch(snapshot.field_width, snapshot.field_length);
        let facing = facing_bucket_from_vector(point - self.position);
        Some(AgentActionTargetTrace {
            point: Some(point),
            player_id,
            grid: Some(pitch_grid_address(
                point,
                snapshot.field_width,
                snapshot.field_length,
            )),
            facing,
            dribble_touch,
        })
    }

    pub(crate) fn restart_release_action(
        &self,
        snapshot: &WorldSnapshot,
        observation: &SoccerPomdpObservation,
        restart_label: &str,
    ) -> Option<(SoccerAction, String)> {
        // On a goal kick the ball must leave the penalty area with the first touch
        // (Law 16): a pass to a teammate still inside our own box is illegal. This guard
        // applies to both the planned set-play release and the freely-computed restart
        // pass below — the keeper plays to an outlet that has stepped outside the area, or
        // clears it long, rather than tapping it short to a defender in the box.
        let goal_kick_receiver_legal = |target: Option<usize>| -> bool {
            if restart_label != "goal-kick" {
                return true;
            }
            match target {
                None => true,
                Some(id) => snapshot
                    .player_position(id)
                    .map(|pos| !snapshot.point_in_own_penalty_area(self.team, pos))
                    .unwrap_or(true),
            }
        };

        if let Some(planned_release) = snapshot.active_set_play_release_for(self.id, restart_label)
        {
            let release_legal = match &planned_release.0 {
                SoccerAction::Pass { target_player, .. } => {
                    goal_kick_receiver_legal(*target_player)
                }
                _ => true,
            };
            if release_legal {
                return Some(planned_release);
            }
        }

        // Goal kick: the keeper may hold the ball up to ~7 s to let teammates reset
        // and find open positions, only releasing once someone is genuinely open (or
        // the cap forces it). Returning None makes the caller hold the restart.
        if restart_label == "goal-kick" {
            let best_openness = observation
                .best_aerial_pass_receiver_openness
                .max(observation.best_pass_receiver_openness);
            let held_seconds = snapshot.possession_elapsed_seconds_for_team(self.team);
            if best_openness < GK_GOAL_KICK_RELEASE_OPENNESS
                && held_seconds < GK_GOAL_KICK_MAX_HOLD_SECONDS
            {
                return None;
            }
        }

        let passing_skill = ability01(self.skills.passing_completion_rate);
        let crossing_skill = ability01(self.skills.crossing_left.max(self.skills.crossing_right));
        if restart_label == "free-kick"
            && shot_decision_is_qualified_for_role(observation, self.role)
            && observation.yards_to_goal <= 32.0
        {
            return Some((
                SoccerAction::Shoot {
                    power: shot_power_for_skill(ability01(self.skills.shooting)),
                },
                restart_label.to_string(),
            ));
        }

        let prefer_aerial = matches!(restart_label, "throw-in" | "corner-kick" | "goal-kick");
        let mut visible_targets = if prefer_aerial {
            snapshot.ranked_visible_aerial_pass_targets(self.id, 11)
        } else {
            snapshot.ranked_visible_pass_targets(self.id, 11)
        };
        // Drop any in-box receiver on a goal kick so the chosen outlet is outside the area.
        visible_targets.retain(|&id| goal_kick_receiver_legal(Some(id)));
        let mut target = visible_targets
            .first()
            .copied()
            .or_else(|| snapshot.best_pass_target(self.id));
        // If the only remaining option is still inside our box, clear it long (no target)
        // so the ball is guaranteed to leave the penalty area on the first touch.
        if !goal_kick_receiver_legal(target) {
            target = None;
        }
        let flight = if prefer_aerial {
            PassFlight::Aerial
        } else {
            PassFlight::Floor
        };
        let power = match restart_label {
            "throw-in" => 0.48 + 0.18 * passing_skill,
            "corner-kick" => 0.62 + 0.24 * crossing_skill.max(passing_skill),
            "goal-kick" => 0.70 + 0.22 * passing_skill.max(ability01(self.skills.strength)),
            "kickoff" => 0.42 + 0.22 * passing_skill,
            "free-kick" => 0.56 + 0.26 * passing_skill.max(crossing_skill),
            _ => 0.58 + 0.24 * passing_skill,
        }
        .clamp(0.32, 0.96);
        match target {
            Some(id) => Some((
                SoccerAction::Pass {
                    target_player: Some(id),
                    power,
                    flight,
                },
                restart_label.to_string(),
            )),
            // No team-mate to aim at and the taker must release now (any hold window above has
            // already elapsed): hoof it into the attacking corner channel, wide of the keeper,
            // rather than tapping it to nobody / straight out of play.
            None => {
                let (aim, hoof_power) = snapshot.restart_corner_hoof_for(self.id);
                Some((
                    SoccerAction::Clearance {
                        target: aim,
                        power: hoof_power,
                    },
                    restart_label.to_string(),
                ))
            }
        }
    }

    fn control_touch_target_for_goal_context(
        &self,
        snapshot: &WorldSnapshot,
        observation: &SoccerPomdpObservation,
    ) -> Vec2 {
        if goal_approach_forces_goalmouth_carry(observation, self.role) {
            return snapshot.dribble_move_target_for_touch(
                self.id,
                self.home_position,
                DribbleMoveKind::CarryForward,
                DribbleTouchDecision::new(0, 1.35),
            );
        }
        // A neutral controlling touch carries the ball a little ahead. If a marker
        // is already tight, make that default safer by nudging away from pressure
        // and toward a visible forward support lane. With pass-anticipation on, the
        // snapshot helper can still override this with a runner flick or a stronger
        // retaining touch against a marker bearing down.
        let neutral = (self.position + carried_ball_lead(self))
            .clamp_to_pitch(snapshot.field_width, snapshot.field_length);
        let pressured_neutral = snapshot
            .players
            .iter()
            .filter(|player| player.team != self.team)
            .map(|player| (player.position, player.position.distance(self.position)))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .filter(|(_, distance)| *distance <= 6.5)
            .and_then(|(marker, _)| {
                let away = self.position - marker;
                if away.len() <= 1e-6 {
                    return None;
                }
                let support_lane = snapshot
                    .ranked_visible_pass_targets(self.id, 1)
                    .first()
                    .and_then(|target_id| snapshot.player_position(*target_id))
                    .filter(|target| (target.y - self.position.y) * self.team.attack_dir() > 1.0)
                    .map(|target| (target - self.position).normalized())
                    .unwrap_or_else(|| Vec2::new(0.0, self.team.attack_dir()));
                let direction = (away.normalized() * 0.72 + support_lane * 0.28).normalized();
                let touch_yards = 1.05 + ability01(self.skills.first_touch) * 1.10;
                Some(
                    (self.position + direction * touch_yards)
                        .clamp_to_pitch(snapshot.field_width, snapshot.field_length),
                )
            })
            .unwrap_or(neutral);
        snapshot.first_touch_directional_target_for(self.id, pressured_neutral)
    }

    pub(crate) fn agentic_fallback_dribble_kind(
        &self,
        snapshot: &WorldSnapshot,
        observation: &SoccerPomdpObservation,
        patient_carry: bool,
        carry_to_create: bool,
    ) -> DribbleMoveKind {
        if goal_approach_forces_goalmouth_carry(observation, self.role) {
            return DribbleMoveKind::CarryForward;
        }

        let field_width = if snapshot.field_width.is_finite() && snapshot.field_width > 0.0 {
            snapshot.field_width
        } else {
            DEFAULT_FIELD_WIDTH_YARDS
        };
        let lateral = self.position.x.clamp(0.0, field_width);
        let left_room = if self.team == Team::Home {
            lateral
        } else {
            field_width - lateral
        };
        let right_room = if self.team == Team::Home {
            field_width - lateral
        } else {
            lateral
        };
        let left_room_fit = (left_room / 16.0).clamp(0.0, 1.0);
        let right_room_fit = (right_room / 16.0).clamp(0.0, 1.0);
        let nearest_pressure =
            pressure_from_nearest_distance(observation.nearest_opponent_distance);
        let pressure = observation
            .perceived_pressure
            .max(observation.pressure_urgency)
            .max(observation.immediate_dispossession_risk)
            .max(nearest_pressure)
            .clamp(0.0, 1.0);
        let forward_space_fit = (observation.forward_dribble_space_yards / 18.0).clamp(0.0, 1.0);
        let forward_blocked = (1.0 - observation.forward_dribble_space_yards / 6.0).clamp(0.0, 1.0);
        let no_outlet_fit = if observation.visible_pass_options == 0 {
            1.0
        } else {
            (1.0 - observation.best_pass_receiver_openness.clamp(0.0, 1.0)).clamp(0.0, 1.0)
        };
        let left_x_sign = -self.team.attack_dir();
        let nearest_defender = snapshot
            .players
            .iter()
            .filter(|player| player.team != self.team)
            .map(|defender| {
                let distance = defender.position.distance(self.position);
                (defender, distance)
            })
            .min_by(|a, b| a.1.total_cmp(&b.1));
        let (defender_left_pressure, defender_right_pressure, closing_fit) = nearest_defender
            .map(|(defender, defender_distance)| {
                let lateral_delta = defender.position.x - self.position.x;
                let side_fit = (lateral_delta.abs() / 5.0).clamp(0.0, 1.0);
                let close_fit = (1.0_f64 - defender_distance / 7.0).clamp(0.0, 1.0);
                let pressure_fit = (side_fit * close_fit).clamp(0.0, 1.0);
                let closing_direction = (self.position - defender.position).normalized();
                let closing_fit = if closing_direction.len() > 1e-6 {
                    ((defender.velocity - self.velocity).dot(closing_direction) / 6.0)
                        .clamp(0.0, 1.0)
                } else {
                    0.0
                };
                if lateral_delta * left_x_sign > 0.0 {
                    (pressure_fit, 0.0, closing_fit)
                } else if lateral_delta * left_x_sign < 0.0 {
                    (0.0, pressure_fit, closing_fit)
                } else {
                    (pressure_fit * 0.5, pressure_fit * 0.5, closing_fit)
                }
            })
            .unwrap_or((0.0, 0.0, 0.0));
        let escape_pressure = pressure.max(forward_blocked * 0.76).max(closing_fit);
        let patient_fit = if patient_carry { 1.0 } else { 0.0 };
        let creation_fit = if carry_to_create { 1.0 } else { 0.0 };
        let carry_forward_score = 0.46
            + forward_space_fit * 0.46
            + creation_fit * 0.24
            + patient_fit * 0.08
            + no_outlet_fit * 0.08
            - pressure * 0.18
            - forward_blocked * 0.26;
        let carry_left_score = 0.36
            + left_room_fit * 0.30
            + defender_right_pressure * 0.36
            + escape_pressure * 0.22
            + patient_fit * 0.08
            - defender_left_pressure * 0.24;
        let carry_right_score = 0.36
            + right_room_fit * 0.30
            + defender_left_pressure * 0.36
            + escape_pressure * 0.22
            + patient_fit * 0.08
            - defender_right_pressure * 0.24;
        let protect_score = 0.26
            + pressure * 0.30
            + forward_blocked * 0.22
            + (1.0 - left_room_fit.max(right_room_fit)) * 0.18
            - forward_space_fit * 0.12;
        let xavi_turn_score = 0.30
            + pressure * 0.34
            + forward_blocked * 0.26
            + closing_fit * 0.22
            + no_outlet_fit * 0.10
            - forward_space_fit * 0.08;

        // Scored candidate carry kinds. The deterministic argmax is the first
        // element with the highest score; with the top-k gate on, the stochastic
        // selector draws among the best three (70/20/10) over this same set.
        let kind_candidates = [
            (DribbleMoveKind::CarryForward, carry_forward_score),
            (DribbleMoveKind::CarryOutLeft, carry_left_score),
            (DribbleMoveKind::CarryOutRight, carry_right_score),
            (DribbleMoveKind::ProtectBall, protect_score),
            (DribbleMoveKind::XaviTurn, xavi_turn_score),
        ];
        let kind_scores: Vec<f64> = kind_candidates.iter().map(|(_, s)| *s).collect();
        let best = sampled_index_by_score(
            &kind_scores,
            snapshot.decision_seed,
            self.id,
            snapshot.tick,
            site::DRIBBLE_KIND,
        )
        .map(|idx| kind_candidates[idx])
        .unwrap_or(kind_candidates[0]);

        if best.0 == DribbleMoveKind::CarryOutLeft
            && pressure >= 0.36
            && defender_right_pressure > defender_left_pressure
            && left_room_fit >= 0.25
        {
            DribbleMoveKind::FakeRightCutLeft
        } else if best.0 == DribbleMoveKind::CarryOutRight
            && pressure >= 0.36
            && defender_left_pressure > defender_right_pressure
            && right_room_fit >= 0.25
        {
            DribbleMoveKind::FakeLeftCutRight
        } else if best.0 == DribbleMoveKind::CarryOutLeft
            && pressure >= 0.54
            && left_room_fit >= 0.22
        {
            DribbleMoveKind::LeftCut
        } else if best.0 == DribbleMoveKind::CarryOutRight
            && pressure >= 0.54
            && right_room_fit >= 0.22
        {
            DribbleMoveKind::RightCut
        } else {
            best.0
        }
    }

    fn human_shoot_or_carry_action(
        &self,
        snapshot: &WorldSnapshot,
        observation: &SoccerPomdpObservation,
    ) -> (SoccerAction, String) {
        if shot_decision_is_qualified_for_role(observation, self.role) {
            return (SoccerAction::Shoot { power: 1.0 }, "shoot".to_string());
        }

        self.human_carry_or_protect_action(snapshot, observation)
    }

    fn human_carry_or_protect_action(
        &self,
        snapshot: &WorldSnapshot,
        observation: &SoccerPomdpObservation,
    ) -> (SoccerAction, String) {
        let mut kind = if observation.perceived_pressure >= 0.42
            && observation.forward_dribble_space_yards < 1.25
        {
            DribbleMoveKind::ProtectBall
        } else {
            DribbleMoveKind::CarryForward
        };
        kind = goal_approach_dribble_kind(kind, observation, self.role);
        let touch = snapshot.deterministic_dribble_touch_decision_for(self.id, kind);
        (
            SoccerAction::DribbleMove {
                target: snapshot.dribble_move_target_for_touch(
                    self.id,
                    self.home_position,
                    kind,
                    touch,
                ),
                kind,
                touch,
            },
            kind.label().to_string(),
        )
    }

    fn human_pass_or_carry_action(
        &self,
        input: &HumanInputFrame,
        snapshot: &WorldSnapshot,
        observation: &SoccerPomdpObservation,
        flight: PassFlight,
        power: f64,
    ) -> (SoccerAction, String) {
        let visible_targets = if flight.is_aerial() {
            snapshot.ranked_visible_aerial_pass_targets(self.id, 11)
        } else {
            snapshot.ranked_visible_pass_targets(self.id, 11)
        };
        if let Some(target) = input
            .target_player
            .filter(|&target| {
                // Honour an explicit human target as long as it is a VISIBLE, ONSIDE
                // teammate — the controller is choosing deliberately and must not be
                // overridden by the autonomous lane / set-interceptor guards (those only
                // gate the AI's own target selection, not the human's intent).
                snapshot
                    .players
                    .iter()
                    .any(|p| p.id == target && p.team == self.team && p.id != self.id)
                    && snapshot.player_can_see_player(self.id, target)
                    && snapshot.pending_offside_for_pass(self.id, target).is_none()
            })
            .or_else(|| visible_targets.first().copied())
        {
            return (
                SoccerAction::Pass {
                    target_player: Some(target),
                    power,
                    flight,
                },
                if flight.is_aerial() {
                    "aerial-pass"
                } else {
                    "pass"
                }
                .to_string(),
            );
        }

        self.human_carry_or_protect_action(snapshot, observation)
    }

    fn sanitized_off_ball_human_move_target(
        &self,
        snapshot: &WorldSnapshot,
        input: &HumanInputFrame,
    ) -> Vec2 {
        let axis = input.axis.normalized();
        if axis.len() <= 1e-6 {
            // Released (no movement input): hold position. The direct human-control
            // path brakes the player to a stop — never drift toward the ball.
            return self.position;
        }
        let step_yards = if input.sprint { 7.0 } else { 4.5 };
        (self.position + axis * step_yards)
            .clamp_to_pitch(snapshot.field_width, snapshot.field_length)
    }

    fn explicit_human_action(
        &self,
        input: &HumanInputFrame,
        snapshot: &WorldSnapshot,
        observation: &SoccerPomdpObservation,
    ) -> Option<(SoccerAction, String)> {
        let action = normalize_soccer_action_label(input.action.as_deref()?);
        match action {
            "shoot" if observation.has_ball => {
                Some(self.human_shoot_or_carry_action(snapshot, observation))
            }
            "pass" if observation.has_ball => Some(self.human_pass_or_carry_action(
                input,
                snapshot,
                observation,
                PassFlight::Floor,
                0.78,
            )),
            "aerial-pass" if observation.has_ball => Some(self.human_pass_or_carry_action(
                input,
                snapshot,
                observation,
                PassFlight::Aerial,
                0.78,
            )),
            "killer-pass" | "clearance" | "route-one" if observation.has_ball => self
                .action_from_learned_plan(
                    &SoccerLearnedPlan {
                        action: action.to_string(),
                        target_player: input.target_player,
                        target_point: None,
                        mpc_replan: None,
                    },
                    snapshot,
                    observation,
                ),
            _ if observation.has_ball => dribble_move_kind_for_action_label(action).map(|kind| {
                let kind = goal_approach_dribble_kind(kind, observation, self.role);
                let touch = snapshot.deterministic_dribble_touch_decision_for(self.id, kind);
                (
                    SoccerAction::DribbleMove {
                        target: snapshot.dribble_move_target_for_touch(
                            self.id,
                            self.home_position,
                            kind,
                            touch,
                        ),
                        kind,
                        touch,
                    },
                    kind.label().to_string(),
                )
            }),
            _ => None,
        }
    }

    pub fn run_time_step(
        &mut self,
        snapshot: &WorldSnapshot,
        human_input: Option<&HumanInputFrame>,
        learned_plan: Option<&SoccerLearnedPlan>,
        rng: &mut SeededRandom,
    ) -> PlayerIntent {
        let mdp_state = snapshot.mdp_state_for_player(self.id);
        let observation = snapshot.observation_for(self.id);
        let intent = self.run_time_step_with_context(
            snapshot,
            mdp_state,
            observation,
            human_input,
            learned_plan,
            rng,
        );
        self.committed_chaser_sprint_guarantee(snapshot, intent)
    }

    /// A committed loose-ball chaser already steers to the intercept point via the normal
    /// recovery logic — make it SPRINT there when the ball is live enough to reward it (attack
    /// the ball before settling into support shape). Kept as a post-step so it doesn't bypass the
    /// weighted recovery options / 50-50 two-competitor logic the way an early-return short-circuit
    /// would, but bounded so tired players do not spend extra accelerations on long, low-value
    /// chases.
    pub(crate) fn committed_chaser_sprint_guarantee(
        &self,
        snapshot: &WorldSnapshot,
        mut intent: PlayerIntent,
    ) -> PlayerIntent {
        if let SoccerAction::MoveTo(target) = intent.action {
            let distance = self.position.distance(target);
            let lane_interceptor = snapshot
                .on_path_ball_intercept_target_for(self.id)
                .is_some();
            let fifty_fifty_duel = loose_ball_fifty_fifty_duel_for(snapshot, self.id);
            let moving_ball_to_attack =
                snapshot.ball.velocity.len() >= COMMITTED_LOOSE_BALL_SPRINT_BALL_SPEED_YPS;
            let economical_distance =
                distance <= COMMITTED_LOOSE_BALL_SPRINT_MAX_DISTANCE_YARDS || lane_interceptor;
            let fatigue_allows =
                self.fatigue <= SPRINT_CONSERVE_FATIGUE || lane_interceptor || fifty_fifty_duel;
            if self.role != PlayerRole::Goalkeeper
                && snapshot.ball.holder.is_none()
                && snapshot.is_committed_loose_ball_chaser(self.id)
                && distance > 1.0
                && economical_distance
                && fatigue_allows
                && (moving_ball_to_attack || lane_interceptor || fifty_fifty_duel)
            {
                intent.sprint = true;
            }
        }
        intent
    }

    fn rolling_ball_arrival_time_to_target(&self, snapshot: &WorldSnapshot, target: Vec2) -> f64 {
        let speed = snapshot.ball.velocity.len();
        if speed <= REACTIVE_GROUND_PASS_NUMERIC_EPSILON {
            return f64::INFINITY;
        }
        let ball_dir = snapshot.ball.velocity.normalized();
        let along = (target - snapshot.ball.position).dot(ball_dir).max(0.0);
        if along <= REACTIVE_GROUND_PASS_NUMERIC_EPSILON {
            return 0.0;
        }
        let accel = dot(snapshot.ball.acceleration, ball_dir);
        if accel.abs() <= REACTIVE_GROUND_PASS_NUMERIC_EPSILON {
            return along / speed;
        }
        let discriminant = speed * speed + 2.0 * accel * along;
        if discriminant < 0.0 {
            return f64::INFINITY;
        }
        let root = discriminant.sqrt();
        let t = (-speed + root) / accel;
        if t.is_finite() && t >= 0.0 {
            t
        } else {
            along / speed
        }
    }

    fn pomdp_mpc_loose_ball_recovery_target(
        &self,
        snapshot: &WorldSnapshot,
        early_control_target: Vec2,
    ) -> Vec2 {
        if snapshot.ball.holder.is_some()
            || snapshot.pending_pass.is_some()
            || snapshot.ball.altitude_yards > BALL_ROLLING_ALTITUDE_YARDS
            || snapshot.ball.velocity.len() < REACTIVE_GROUND_PASS_MIN_SPEED_YPS
        {
            return early_control_target;
        }
        let run_on_target = snapshot
            .projected_loose_ball_target()
            .unwrap_or(early_control_target);
        if run_on_target.distance(early_control_target)
            <= POMDP_BALL_CONTROL_LET_RUN_MIN_AHEAD_YARDS
        {
            return early_control_target;
        }
        let ball_arrival = self.rolling_ball_arrival_time_to_target(snapshot, early_control_target);
        if !ball_arrival.is_finite() {
            return early_control_target;
        }
        let player_velocity = snapshot.player_velocity(self.id).unwrap_or(self.velocity);
        let facing_fit = player_facing_ball_control_multiplier(
            self,
            early_control_target,
            snapshot.ball.velocity.len(),
        );
        let mpc_fit = mpc_ball_control_execution_fit(
            self.role,
            &self.skills,
            self.fatigue,
            self.position,
            player_velocity,
            facing_fit,
            early_control_target,
            snapshot.ball.velocity,
            ball_arrival,
            REACTIVE_GROUND_PASS_CONTROL_RADIUS_YARDS,
        );
        if mpc_fit.qp_accel_fit < MPC_BALL_CONTROL_MIN_QP_ACCEL_FIT {
            return run_on_target.clamp_to_pitch(snapshot.field_width, snapshot.field_length);
        }
        let control_choice = pomdp_loose_ball_control_decision(
            &self.skills,
            self.decision_confidence,
            self.position,
            early_control_target,
            run_on_target,
            snapshot.ball.velocity,
            snapshot.nearest_opponent_distance_at(self.team, early_control_target),
            mpc_fit,
            false,
        );
        control_choice
            .target
            .clamp_to_pitch(snapshot.field_width, snapshot.field_length)
    }

    pub(crate) fn apply_post_decision_movement_discipline(
        &self,
        snapshot: &WorldSnapshot,
        match_state: &SoccerMatch,
        intent: PlayerIntent,
    ) -> PlayerIntent {
        debug_assert_eq!(
            intent.player_id, self.id,
            "movement-discipline intent must belong to the player applying it"
        );
        if intent.player_id != self.id {
            return intent;
        }

        let intent = self.committed_chaser_sprint_guarantee(snapshot, intent);
        let live_ball_attacker =
            snapshot.is_live_ball_attacker_for_movement_guards(intent.player_id);

        let mut intent = intent;
        if !dd_soccer_disable_anti_bunch() {
            intent = snapshot.discipline_intent_against_bunchball(intent);
        }
        if !(dd_soccer_disable_spacing_nudge() || live_ball_attacker) {
            intent = snapshot.nudge_intent_for_teammate_spacing(intent);
        }

        // Team-line structure. These are global geometry constraints, but the
        // player accepts them after its own MDP/POMDP choice and before execution.
        let intent = snapshot.defensive_line_cushion_adjusted_intent(intent);
        let intent = snapshot.back_four_shape_adjusted_intent(intent);
        let intent = snapshot.midfield_line_band_adjusted_intent(intent);
        let intent = snapshot.forward_line_band_adjusted_intent(intent);
        let intent = snapshot.offside_standing_recovery_adjusted_intent(intent);
        let intent = snapshot.goalside_anticipation_adjusted_intent(intent);
        let intent = snapshot.offside_trap_cover_adjusted_intent(intent);

        let mut intent = intent;
        if !live_ball_attacker {
            intent = match_state.teammate_spacing_disciplined_intent(intent);
            intent = match_state.relational_shape_disciplined_intent(intent);
            intent = snapshot.teammate_lane_guard_adjusted_intent(intent);
            intent = snapshot.ball_proximity_adjusted_intent(intent);
        }

        let intent = snapshot.back_four_shape_adjusted_intent(intent);
        let intent = snapshot.ball_in_behind_recovery_adjusted_intent(intent);
        let intent = snapshot.cross_through_disciplined_intent(intent);
        let intent = snapshot.teammate_spacing_path_adjusted_intent(intent);
        let intent = snapshot.weak_side_width_hold_adjusted_intent(intent);
        let intent = snapshot.defensive_line_cushion_adjusted_intent(intent);
        let intent = snapshot.midfield_line_band_adjusted_intent(intent);
        let intent = snapshot.forward_line_band_adjusted_intent(intent);
        let intent = snapshot.wingback_width_adjusted_intent(intent);
        snapshot.possession_wide_lane_floor_adjusted_intent(intent)
    }

    /// Per-tick decision entry. Thin wrapper over [`Self::run_time_step_with_context_inner`]
    /// that enforces the *deliberative decision refractory* (see
    /// [`decision_refractory_enabled`]): a player re-derives its intent every tick (15 Hz),
    /// but human reaction time + mental processing + momentum mean it cannot keep *changing
    /// its mind* that fast. The inner pass produces the freshly-ranked intent; this wrapper
    /// then decides whether the player is allowed to *commit a changed deliberative decision*
    /// this tick or must continue the one it already committed to. Re-selecting the same
    /// intent (steering/re-aiming it with fresh perception) is free and never throttled — we
    /// rate-limit the *switch*, not the *tracking*. Gate is OFF by default ⇒ byte-identical.
    pub fn run_time_step_with_context(
        &mut self,
        snapshot: &WorldSnapshot,
        mdp_state: SoccerMdpState,
        observation: SoccerPomdpObservation,
        human_input: Option<&HumanInputFrame>,
        learned_plan: Option<&SoccerLearnedPlan>,
        rng: &mut SeededRandom,
    ) -> PlayerIntent {
        if !decision_refractory_enabled() {
            return self.run_time_step_with_context_inner(
                snapshot,
                mdp_state,
                observation,
                human_input,
                learned_plan,
                rng,
            );
        }
        // The decision currently committed (held from a prior tick), captured before the
        // inner pass overwrites `last_decision` with the freshly-ranked one.
        let held_decision = self.last_decision.clone();
        let held_label = held_decision
            .as_ref()
            .map(|d| normalize_soccer_action_label(&d.action).to_string());
        let now = snapshot.tick;

        let intent = self.run_time_step_with_context_inner(
            snapshot,
            mdp_state,
            observation,
            human_input,
            learned_plan,
            rng,
        );

        let intent = self.apply_decision_refractory(now, held_label, held_decision, intent);
        self.last_intent = Some(intent.clone());
        intent
    }

    /// Enforce the deliberative-decision refractory on the freshly-ranked intent.
    ///
    /// A *changed* deliberative decision (the canonical action label differs from the one the
    /// player was already committed to) is only allowed to commit when **both** rate limits
    /// clear: at least [`DECISION_REFRACTORY_MIN_GAP_TICKS`] (3) ticks since the last decision,
    /// **and** at least [`DECISION_REFRACTORY_WINDOW_TICKS`] (7) ticks since the second-to-last.
    /// That yields a 3,4,3,4-tick cadence — decisions land on ticks 1, 4, 8, 11, 15… — i.e. at
    /// most 2 deliberative decisions in any 7-tick window, with the third possible on the 8th
    /// tick. Re-selecting the same label is not a decision and passes through untouched. When a
    /// change is blocked the player *continues the intent it already committed to* (replayed
    /// from `last_intent`) so it does not freeze in no-man's-land; if the held intent cannot be
    /// continued (a ball-release one-shot already played, a challenge), the change is allowed
    /// through rather than emitting something illegal.
    fn apply_decision_refractory(
        &mut self,
        now: u64,
        held_label: Option<String>,
        held_decision: Option<AgentDecisionTrace>,
        intent: PlayerIntent,
    ) -> PlayerIntent {
        let Some(proposed_label) = self
            .last_decision
            .as_ref()
            .map(|d| normalize_soccer_action_label(&d.action).to_string())
        else {
            return intent;
        };

        let is_change = held_label.as_deref() != Some(proposed_label.as_str());
        if !is_change {
            // Continuing the same deliberative decision — free, never throttled.
            return intent;
        }

        // The very first decision (nothing held yet) commits immediately and arms the timer.
        if held_label.is_none() {
            self.push_decision_commit_tick(now);
            return intent;
        }

        if !decision_refractory_blocks(&self.decision_commit_ticks, now) {
            self.push_decision_commit_tick(now);
            return intent;
        }

        // Blocked: keep executing the previously committed decision if it can be continued.
        if let Some(held_intent) = self.continuation_intent() {
            self.last_decision = held_decision;
            return held_intent;
        }

        // Cannot replay the held intent (released ball / committed challenge): allow the change
        // rather than emit an illegal continuation, and re-arm the timer on the new commitment.
        self.push_decision_commit_tick(now);
        intent
    }

    /// Replay the previously emitted intent when a decision change is refractory-blocked.
    /// Only *sustainable* held actions can be continued tick-to-tick; a ball-release one-shot
    /// (pass/shot/clearance) or a committed challenge (tackle) returns `None` — the ball state
    /// has moved on, so the new decision must be allowed instead.
    fn continuation_intent(&self) -> Option<PlayerIntent> {
        let held = self.last_intent.as_ref()?;
        match held.action {
            SoccerAction::HoldShape
            | SoccerAction::MoveTo(_)
            | SoccerAction::Dribble(_)
            | SoccerAction::DribbleMove { .. }
            | SoccerAction::ControlTouch { .. } => Some(held.clone()),
            SoccerAction::Pass { .. }
            | SoccerAction::Clearance { .. }
            | SoccerAction::RouteOne { .. }
            | SoccerAction::Shoot { .. }
            | SoccerAction::Tackle { .. }
            | SoccerAction::SlideTackle { .. } => None,
        }
    }

    /// Record the tick at which a changed deliberative decision was committed, keeping only the
    /// last two (all the refractory gate needs).
    fn push_decision_commit_tick(&mut self, now: u64) {
        self.decision_commit_ticks.push_back(now);
        while self.decision_commit_ticks.len() > 2 {
            self.decision_commit_ticks.pop_front();
        }
    }

    fn run_time_step_with_context_inner(
        &mut self,
        snapshot: &WorldSnapshot,
        mut mdp_state: SoccerMdpState,
        mut observation: SoccerPomdpObservation,
        human_input: Option<&HumanInputFrame>,
        learned_plan: Option<&SoccerLearnedPlan>,
        _rng: &mut SeededRandom,
    ) -> PlayerIntent {
        let human_input = human_input
            .filter(|input| human_input_matches_player(input, self.id, self.controller_slot));
        annotate_human_control_observation(&mut observation, self.controller_slot, human_input);
        Self::apply_field_vector_context(self, snapshot, &mut mdp_state, &mut observation);
        // Clear last tick's stochastic-selection relay; each ranker that fires
        // this tick sets it to the behaviour probability of the option it
        // promoted (or leaves it `None` for a deterministic argmax).
        self.pending_policy_behavior_probability = None;
        let belief = belief_from_observation(&observation);
        // Belief-grounded confidence for this decision: drives decisiveness (move speed)
        // in movement and proaction. Engine=truth, brain=strong belief, player=this.
        self.decision_confidence = player_decision_confidence(&observation, &self.skills);
        let directive = snapshot.tactical_directive(self.team);
        let has_ball = observation.has_ball;
        let shooting_skill = ability01(self.skills.shooting);
        let passing_skill = ability01(self.skills.passing_completion_rate);
        let defending_skill = ability01(self.skills.defending);
        let aggression_skill = ability01(self.skills.aggression);
        let held_restart = held_restart_action_for_snapshot(snapshot, self.id);
        let learned_action_label =
            learned_plan.map(|plan| normalize_soccer_action_label(&plan.action));
        let learned_open_pass_lane_requested =
            learned_action_label == Some(OPEN_PASS_LANE_ACTION_LABEL);

        if let Some(input) = human_input {
            let (action, action_label) = if let Some(restart_label) = held_restart {
                if input.pass || input.shoot {
                    if input.pass {
                        let flight = if restart_label == "throw-in" {
                            PassFlight::Aerial
                        } else {
                            input.pass_flight
                        };
                        (
                            SoccerAction::Pass {
                                target_player: input.target_player,
                                power: if restart_label == "throw-in" {
                                    0.58
                                } else {
                                    0.78
                                },
                                flight,
                            },
                            restart_label.to_string(),
                        )
                    } else {
                        self.restart_release_action(snapshot, &observation, restart_label)
                            .unwrap_or((
                                SoccerAction::MoveTo(self.position),
                                "hold-restart".to_string(),
                            ))
                    }
                } else {
                    (
                        SoccerAction::MoveTo(self.position),
                        "hold-restart".to_string(),
                    )
                }
            } else if let Some(explicit) = self.explicit_human_action(input, snapshot, &observation)
            {
                explicit
            } else if input.shoot {
                if has_ball {
                    self.human_shoot_or_carry_action(snapshot, &observation)
                } else {
                    (
                        SoccerAction::MoveTo(
                            self.sanitized_off_ball_human_move_target(snapshot, input),
                        ),
                        "human-move".to_string(),
                    )
                }
            } else if input.pass {
                if has_ball {
                    let pass_flight = input.pass_flight;
                    self.human_pass_or_carry_action(
                        input,
                        snapshot,
                        &observation,
                        pass_flight,
                        0.78,
                    )
                } else {
                    (
                        SoccerAction::MoveTo(
                            self.sanitized_off_ball_human_move_target(snapshot, input),
                        ),
                        "human-move".to_string(),
                    )
                }
            } else {
                let dir = input.axis.normalized();
                (
                    SoccerAction::MoveTo(
                        self.position + dir * if input.sprint { 7.0 } else { 4.5 },
                    ),
                    "human-move".to_string(),
                )
            };
            self.last_decision = Some(self.decision_trace(
                snapshot,
                mdp_state,
                observation,
                belief,
                vec!["human-input".to_string()],
                single_action_option(&action_label),
                &action,
                action_label,
            ));
            return PlayerIntent {
                player_id: self.id,
                action,
                sprint: input.sprint,
            };
        }

        if let Some(restart_label) = held_restart {
            let (action, action_label) = self
                .restart_release_action(snapshot, &observation, restart_label)
                .unwrap_or((
                    SoccerAction::MoveTo(self.position),
                    "hold-restart".to_string(),
                ));
            self.last_decision = Some(self.decision_trace(
                snapshot,
                mdp_state,
                observation,
                belief,
                vec![restart_label.to_string(), "restart-release".to_string()],
                single_action_option(&action_label),
                &action,
                action_label,
            ));
            return PlayerIntent {
                player_id: self.id,
                action,
                sprint: false,
            };
        }

        if !has_ball {
            if let Some(assignment) = snapshot.active_set_play_assignment_for(self.id) {
                // The taker's set-play assignment target is the dead-ball spot itself.
                // While they actually hold the ball, the held-restart branch above governs
                // them (deliver after the routine delay). Reaching here as the taker means
                // they no longer hold it — they've already played the restart. Do NOT drag
                // them back to the spot: that pins the kicker at the corner flag, shuffling
                // around a ball they are barred from re-touching by the no-double-touch
                // guard (the "ghost kick" at the corner). They've delivered; let them fall
                // through and rejoin open play.
                let is_taker = assignment.role == SoccerSetPlayAssignmentRole::Taker;
                let distance = self.position.distance(assignment.target);
                if !is_taker && distance > 0.35 {
                    let action = SoccerAction::MoveTo(assignment.target);
                    self.last_decision = Some(self.decision_trace(
                        snapshot,
                        mdp_state,
                        observation,
                        belief,
                        vec![
                            "set-play-routine".to_string(),
                            assignment.role.as_label().to_string(),
                        ],
                        single_action_option("set-play-run"),
                        &action,
                        "set-play-run",
                    ));
                    return PlayerIntent {
                        player_id: self.id,
                        action,
                        sprint: matches!(
                            assignment.role,
                            SoccerSetPlayAssignmentRole::PrimaryRunner
                                | SoccerSetPlayAssignmentRole::SecondaryRunner
                                | SoccerSetPlayAssignmentRole::Screen
                        ),
                    };
                }
            }
        }

        // Ball-at-feet reflex: a player must NEVER ignore a controllable ball right
        // at its feet. If the ball is within control range and not in a teammate's
        // possession, settle a loose ball (or fall through to the steal logic for an
        // opponent's). Checked every step so this can never be missed regardless of
        // the player's other intentions. Humans/restarts/set-plays already returned
        // above and keep precedence.
        if !has_ball {
            let ball_pos = snapshot.ball.position;
            let ball_distance = self.position.distance(ball_pos);
            if ball_distance.is_finite() && ball_distance <= PLAYER_CONTROL_RADIUS_YARDS {
                // A loose ball is, by definition, in nobody's possession. (An
                // opponent's ball at the feet is handled by the steal logic below.)
                if snapshot.ball.holder.is_none() {
                    let target = self.control_touch_target_for_goal_context(snapshot, &observation);
                    let action = SoccerAction::ControlTouch { target };
                    let keeper_outside_box = self.role == PlayerRole::Goalkeeper
                        && !snapshot.goalkeeper_can_use_hands_at(self.team, self.position);
                    let (operation_order, action_label) = if keeper_outside_box {
                        (
                            vec![
                                "keeper-outside-box".to_string(),
                                "pomdp-foot-control".to_string(),
                                "ball-at-feet-reflex".to_string(),
                            ],
                            "keeper-foot-control-outside-box",
                        )
                    } else {
                        (
                            vec!["ball-at-feet-reflex".to_string()],
                            "control-ball-at-feet",
                        )
                    };
                    self.last_decision = Some(self.decision_trace(
                        snapshot,
                        mdp_state,
                        observation,
                        belief,
                        operation_order,
                        single_action_option(action_label),
                        &action,
                        action_label,
                    ));
                    return PlayerIntent {
                        player_id: self.id,
                        action,
                        sprint: false,
                    };
                }
            }
        }

        if !has_ball {
            if let Some(target) = snapshot.teammate_dummy_let_run_target_for(self.id) {
                let action = SoccerAction::MoveTo(target);
                self.last_decision = Some(self.decision_trace(
                    snapshot,
                    mdp_state,
                    observation,
                    belief,
                    vec![
                        "pomdp-dummy-through".to_string(),
                        DUMMY_LET_RUN_ACTION_LABEL.to_string(),
                    ],
                    single_action_option(DUMMY_LET_RUN_ACTION_LABEL),
                    &action,
                    DUMMY_LET_RUN_ACTION_LABEL,
                ));
                return PlayerIntent {
                    player_id: self.id,
                    action,
                    sprint: false,
                };
            }
        }

        // Loose-ball awareness reflex (every tick, cheap): a player must recognize a LIVE
        // loose ball that is close OR rolling toward it and pounce on it directly — drive at
        // the kinematic trajectory intercept and sprint — instead of waiting for the broad
        // decision match below (where shape/support branches can divert it, so it reacts a
        // beat late). Only the committed contester(s) pounce, so the anti-swarm election
        // still holds; everyone else falls through to normal shape. This is what makes a
        // 1-2yd 50-50 get attacked at once rather than strolled to.
        if !has_ball
            && snapshot.ball.holder.is_none()
            && self.role != PlayerRole::Goalkeeper
            && snapshot.active_set_play.is_none()
        {
            let ball_gap = self.position.distance(snapshot.ball.position);
            let approaching = {
                let to_me = self.position - snapshot.ball.position;
                let d = to_me.len();
                d > 1e-6
                    && snapshot.ball.velocity.dot(to_me * (1.0 / d))
                        >= LOOSE_BALL_POUNCE_CLOSING_YPS
            };
            let fifty_fifty_duel = loose_ball_fifty_fifty_duel_for(snapshot, self.id);
            if (ball_gap <= LOOSE_BALL_POUNCE_RADIUS_YARDS || approaching || fifty_fifty_duel)
                && snapshot.is_committed_loose_ball_chaser(self.id)
            {
                let (intercept, _trap_now, _ball_speed) =
                    snapshot.loose_ball_control_plan_for(self.id);
                let my_distance = self.position.distance(intercept);
                let my_distance_sq = (self.position - intercept).dot(self.position - intercept);
                let closer_teammates = snapshot
                    .players
                    .iter()
                    .filter(|player| player.team == self.team && player.id != self.id)
                    .filter(|player| {
                        let delta = player.position - intercept;
                        delta.dot(delta) < my_distance_sq
                    })
                    .count();
                let options = self.loose_ball_action_options(
                    my_distance,
                    closer_teammates,
                    true,
                    fifty_fifty_duel,
                    true,
                );
                // The decision LABEL stays "recover" (the loose-ball recovery move), but the
                // leading operation reflects the contest type — an explicit "fifty-fifty-duel"
                // when both teams are right on it, else the ordinary "recover" pursuit.
                let first_op = if fifty_fifty_duel {
                    FIFTY_FIFTY_DUEL_ACTION_LABEL
                } else {
                    ACTION_LABEL_RECOVER
                };
                let action = SoccerAction::MoveTo(intercept);
                let intent = self.committed_chaser_sprint_guarantee(
                    snapshot,
                    PlayerIntent {
                        player_id: self.id,
                        action: action.clone(),
                        sprint: false,
                    },
                );
                self.last_decision = Some(self.decision_trace(
                    snapshot,
                    mdp_state,
                    observation,
                    belief,
                    vec![first_op.to_string()],
                    options,
                    &action,
                    "recover",
                ));
                return intent;
            }
        }

        // Keeper-collection build-up: when a loose ball is safely running to our own
        // keeper (deep, fast, no opponent within 15 yds), the back four trust the
        // keeper 100% — they stop tracking the ball and fan into an open position in
        // their own lane to offer a short build-up pass.
        if !has_ball
            && self.role == PlayerRole::Defender
            && snapshot.goalkeeper_will_safely_collect(self.team)
        {
            let target = snapshot.goalkeeper_buildup_lane_target_for(self.id, self.home_position);
            let action = SoccerAction::MoveTo(target);
            self.last_decision = Some(self.decision_trace(
                snapshot,
                mdp_state,
                observation,
                belief,
                vec![
                    "gk-will-collect".to_string(),
                    "buildup-lane-receive".to_string(),
                ],
                single_action_option("buildup-receive"),
                &action,
                "buildup-receive",
            ));
            return PlayerIntent {
                player_id: self.id,
                action,
                sprint: false,
            };
        }

        if has_ball {
            if self.role == PlayerRole::Goalkeeper {
                let can_use_hands_here =
                    snapshot.goalkeeper_can_use_hands_at(self.team, self.position);
                let handling_held_seconds = snapshot.keeper_handling_held_seconds(self.id);
                let handling_hold_window =
                    handling_held_seconds.is_some_and(|held| held < GK_HANDLING_HOLD_LIMIT_SECONDS);
                let handling_clock_expired = handling_held_seconds
                    .is_some_and(|held| held >= GK_HANDLING_HOLD_LIMIT_SECONDS);
                let under_urgent_pressure = observation.nearest_opponent_distance.is_finite()
                    && observation.nearest_opponent_distance
                        <= GOALKEEPER_OUTSIDE_BOX_URGENT_PRESSURE_YARDS;
                if let Some(plan) = snapshot.goalkeeper_mpc_play_out_plan(self.id, None) {
                    let action = if let Some(target_player) = plan.target_player {
                        SoccerAction::Pass {
                            target_player: Some(target_player),
                            power: plan.power,
                            flight: plan.flight,
                        }
                    } else {
                        SoccerAction::Clearance {
                            target: plan.target,
                            power: plan.power,
                        }
                    };
                    let release_intelligent = snapshot.keeper_handling_release_is_intelligent(
                        self.id,
                        plan.target_player,
                        plan.target,
                    );
                    let outside_box_release = !can_use_hands_here
                        && (plan.target_player.is_some() || under_urgent_pressure);
                    let in_box_pass_release = can_use_hands_here
                        && plan.target_player.is_some()
                        && (!handling_hold_window || release_intelligent);
                    let pressured_in_box_clearance = can_use_hands_here
                        && plan.target_player.is_none()
                        && handling_clock_expired
                        && release_intelligent;
                    let learned_keeper_release_requested = matches!(
                        learned_action_label,
                        Some(
                            "keeper-mpc-floor-pass"
                                | "keeper-mpc-aerial-pass"
                                | "keeper-mpc-clearance"
                                | "pass"
                                | "aerial-pass"
                                | "clearance"
                                | "route-one"
                                | "recycle-reset"
                        )
                    ) && release_intelligent;
                    let learned_keeper_hold_requested =
                        matches!(learned_action_label, Some("keeper-survey-hands"))
                            && can_use_hands_here
                            && !handling_clock_expired;
                    if !learned_keeper_hold_requested
                        && (outside_box_release
                            || in_box_pass_release
                            || pressured_in_box_clearance
                            || learned_keeper_release_requested)
                    {
                        let mut operation_order = vec![
                            "keeper-play-out".to_string(),
                            "pomdp-release-choice".to_string(),
                            "mpc-execution-plan".to_string(),
                        ];
                        if learned_keeper_release_requested {
                            operation_order.push("learned-keeper-release".to_string());
                        }
                        if outside_box_release {
                            operation_order.push("outside-box-no-hands".to_string());
                        }
                        if can_use_hands_here && handling_clock_expired {
                            operation_order.push("handling-clock-expired".to_string());
                        }
                        self.last_decision = Some(self.decision_trace(
                            snapshot,
                            mdp_state.clone(),
                            observation.clone(),
                            belief.clone(),
                            operation_order,
                            single_action_option(plan.label),
                            &action,
                            plan.label,
                        ));
                        return PlayerIntent {
                            player_id: self.id,
                            action,
                            sprint: false,
                        };
                    }
                }
                if can_use_hands_here {
                    let action = SoccerAction::MoveTo(self.position);
                    self.last_decision = Some(self.decision_trace(
                        snapshot,
                        mdp_state.clone(),
                        observation.clone(),
                        belief.clone(),
                        vec![
                            "keeper-play-out".to_string(),
                            "pomdp-survey".to_string(),
                            "hands-allowed".to_string(),
                        ],
                        single_action_option("keeper-survey-hands"),
                        &action,
                        "keeper-survey-hands",
                    ));
                    return PlayerIntent {
                        player_id: self.id,
                        action,
                        sprint: false,
                    };
                }
                let target = self.control_touch_target_for_goal_context(snapshot, &observation);
                let action = SoccerAction::ControlTouch { target };
                self.last_decision = Some(self.decision_trace(
                    snapshot,
                    mdp_state.clone(),
                    observation.clone(),
                    belief.clone(),
                    vec![
                        "keeper-play-out".to_string(),
                        "pomdp-foot-control".to_string(),
                        "outside-box-no-hands".to_string(),
                    ],
                    single_action_option("keeper-foot-control-outside-box"),
                    &action,
                    "keeper-foot-control-outside-box",
                ));
                return PlayerIntent {
                    player_id: self.id,
                    action,
                    sprint: false,
                };
            }

            // WALL RETURN (the "two" of a one-two): return the lay-off before
            // generic first-touch logic turns the combination into an ordinary pass.
            if let Some(runner) = snapshot.wall_return_pass_target_for(self.id) {
                // `wall_return_pass_target_for` has already checked the live commitment,
                // offside, and both the led and feet lanes. Once that coordinated pattern is
                // legal, do not roll another appetite check: deferring for generic first-touch
                // logic was turning most chosen wall passes into unrelated possessions.
                let action = SoccerAction::Pass {
                    target_player: Some(runner),
                    power: WALL_PASS_GIVE_POWER + 0.22 * ability01(self.skills.passing),
                    flight: PassFlight::Floor,
                };
                self.last_decision = Some(self.decision_trace(
                    snapshot,
                    mdp_state.clone(),
                    observation.clone(),
                    belief.clone(),
                    vec!["wall-return".to_string()],
                    single_action_option("wall-return"),
                    &action,
                    "wall-return",
                ));
                return PlayerIntent {
                    player_id: self.id,
                    action,
                    sprint: false,
                };
            }

            // Named one-two / give-and-go strategies commit to the wall pass before
            // first-touch, killer/scoop, or generic pass selection can swallow it.
            if let Some(plan) = snapshot.wall_pass_option_for(self.id) {
                let give_and_go_strategy = directive_requests_give_and_go(directive);
                if give_and_go_strategy && plan.quality >= WALL_PASS_STRATEGY_COMMIT_MIN_QUALITY {
                    let action = SoccerAction::Pass {
                        target_player: Some(plan.wall_partner),
                        power: WALL_PASS_GIVE_POWER + 0.20 * passing_skill,
                        flight: PassFlight::Floor,
                    };
                    self.one_two = Some(OneTwoRun {
                        wall_partner: plan.wall_partner,
                        launch_clock_seconds: snapshot.clock_seconds,
                        return_target: plan.return_target,
                    });
                    self.last_decision = Some(self.decision_trace(
                        snapshot,
                        mdp_state.clone(),
                        observation.clone(),
                        belief.clone(),
                        vec!["wall-pass".to_string()],
                        single_action_option("wall-pass"),
                        &action,
                        "wall-pass",
                    ));
                    return PlayerIntent {
                        player_id: self.id,
                        action,
                        sprint: false,
                    };
                }
            }
        }

        if has_ball && observation.first_touch_available {
            let pass_targets = snapshot.ranked_visible_pass_targets(self.id, 3);
            let aerial_pass_targets = snapshot.ranked_visible_aerial_pass_targets(self.id, 3);
            let action_options = self.first_touch_action_options(
                &observation,
                pass_targets.len(),
                aerial_pass_targets.len(),
            );
            let mut forced_first_touch_order = Vec::new();
            if goal_attack_shot_blocks_alternatives(&observation, self.role) {
                let is_aerial = matches!(
                    observation.incoming_ball_kind,
                    IncomingBallKind::AerialCross | IncomingBallKind::AerialPass
                );
                let finish_skill = if is_aerial {
                    aerial_duel_skill_from_agent(self)
                } else {
                    ability01(
                        self.skills
                            .right_foot_shot_power
                            .max(self.skills.left_foot_shot_power),
                    )
                };
                let action = SoccerAction::Shoot {
                    power: shot_power_for_finish_skill(finish_skill),
                };
                let action_label = if is_aerial {
                    "first-time-header"
                } else {
                    "first-time-shot"
                };
                if mpc_reselects_candidate(snapshot, self, &action, action_label) {
                    forced_first_touch_order.extend([
                        "first-touch-goalmouth-window".to_string(),
                        "must-shoot".to_string(),
                        format!("first-touch-mpc-reselect:{action_label}"),
                    ]);
                } else {
                    self.last_decision = Some(self.decision_trace(
                        snapshot,
                        mdp_state,
                        observation,
                        belief,
                        vec![
                            "first-touch-goalmouth-window".to_string(),
                            "must-shoot".to_string(),
                        ],
                        single_action_option(action_label),
                        &action,
                        action_label,
                    ));
                    return PlayerIntent {
                        player_id: self.id,
                        action,
                        sprint: false,
                    };
                }
            }
            let weighted_ops = action_options
                .iter()
                .map(|option| {
                    (
                        option.label.clone(),
                        if option.legal { option.score } else { 0.0 },
                    )
                })
                .collect::<Vec<_>>();
            let ops = {
                let (ordered, ordered_scores) = agentic_action_order_scored(weighted_ops);
                let sampled = reorder_with_scores_traced(
                    ordered,
                    &ordered_scores,
                    snapshot.decision_seed,
                    self.id,
                    snapshot.tick,
                    site::FIRST_TOUCH,
                );
                self.pending_policy_behavior_probability = sampled.behavior_probability;
                sampled.ordered
            };
            let full_ops = ops.clone();
            let mut order_names = forced_first_touch_order;
            order_names.reserve(ops.len());
            let mut chosen = None;
            for op in ops {
                order_names.push(op.clone());
                match normalize_soccer_action_label(&op) {
                    "first-time-shot" | "first-time-header"
                        if first_time_shot_decision_is_qualified_for_role(
                            &observation,
                            self.role,
                        ) =>
                    {
                        let finish_skill = if matches!(
                            observation.incoming_ball_kind,
                            IncomingBallKind::AerialCross | IncomingBallKind::AerialPass
                        ) {
                            aerial_duel_skill_from_agent(self)
                        } else {
                            ability01(
                                self.skills
                                    .right_foot_shot_power
                                    .max(self.skills.left_foot_shot_power),
                            )
                        };
                        let candidate_label = normalize_soccer_action_label(&op).to_string();
                        let candidate_action = SoccerAction::Shoot {
                            power: shot_power_for_finish_skill(finish_skill),
                        };
                        if mpc_reselects_candidate(
                            snapshot,
                            self,
                            &candidate_action,
                            &candidate_label,
                        ) {
                            order_names.push(format!("first-touch-mpc-reselect:{candidate_label}"));
                            continue;
                        }
                        chosen = Some((candidate_action, candidate_label));
                        break;
                    }
                    "killer-pass" if !pass_targets.is_empty() => {
                        if let Some(target) =
                            snapshot.killer_pass_target_for(self.id, &pass_targets)
                        {
                            let candidate_label = "killer-pass".to_string();
                            let candidate_action = SoccerAction::Pass {
                                target_player: Some(target),
                                power: 0.66
                                    + 0.24 * passing_skill.max(ability01(self.skills.vision)),
                                flight: snapshot.killer_pass_flight_for(self.id, &pass_targets),
                            };
                            if mpc_reselects_candidate(
                                snapshot,
                                self,
                                &candidate_action,
                                &candidate_label,
                            ) {
                                order_names
                                    .push(format!("first-touch-mpc-reselect:{candidate_label}"));
                                continue;
                            }
                            chosen = Some((candidate_action, candidate_label));
                            break;
                        }
                    }
                    "first-time-pass" if !pass_targets.is_empty() => {
                        let target = pass_targets[0];
                        let candidate_label = "first-time-pass".to_string();
                        let candidate_action = SoccerAction::Pass {
                            target_player: Some(target),
                            power: 0.54 + 0.30 * passing_skill,
                            flight: PassFlight::Floor,
                        };
                        if mpc_reselects_candidate(
                            snapshot,
                            self,
                            &candidate_action,
                            &candidate_label,
                        ) {
                            order_names.push(format!("first-touch-mpc-reselect:{candidate_label}"));
                            continue;
                        }
                        chosen = Some((candidate_action, candidate_label));
                        break;
                    }
                    "flick-on" if !aerial_pass_targets.is_empty() => {
                        let target = aerial_pass_targets[0];
                        let candidate_label = "flick-on".to_string();
                        let candidate_action = SoccerAction::Pass {
                            target_player: Some(target),
                            power: 0.46
                                + 0.26
                                    * aerial_duel_skill_from_agent(self)
                                        .max(ability01(self.skills.first_touch)),
                            flight: PassFlight::Aerial,
                        };
                        if mpc_reselects_candidate(
                            snapshot,
                            self,
                            &candidate_action,
                            &candidate_label,
                        ) {
                            order_names.push(format!("first-touch-mpc-reselect:{candidate_label}"));
                            continue;
                        }
                        chosen = Some((candidate_action, candidate_label));
                        break;
                    }
                    "control-touch" => {
                        let target =
                            self.control_touch_target_for_goal_context(snapshot, &observation);
                        chosen = Some((
                            SoccerAction::ControlTouch { target },
                            normalize_soccer_action_label(&op).to_string(),
                        ));
                        break;
                    }
                    _ => {}
                }
            }
            for op in full_ops {
                if !order_names.iter().any(|entry| entry == &op) {
                    order_names.push(op);
                }
            }
            let (action, action_label) = chosen.unwrap_or_else(|| {
                let target = self.control_touch_target_for_goal_context(snapshot, &observation);
                (
                    SoccerAction::ControlTouch { target },
                    "control-touch".to_string(),
                )
            });
            let first_touch_escape_sprint = matches!(action, SoccerAction::ControlTouch { .. })
                && observation.first_touch_escape_pressure >= FIRST_TOUCH_ESCAPE_SPRINT_PRESSURE;
            self.last_decision = Some(self.decision_trace(
                snapshot,
                mdp_state,
                observation,
                belief,
                order_names,
                action_options,
                &action,
                action_label,
            ));
            return PlayerIntent {
                player_id: self.id,
                action,
                sprint: first_touch_escape_sprint,
            };
        }

        // Learnable shot-trigger MDP/POMDP value of shooting NOW. The *forced* shot rule
        // paths below fire before `possession_action_options` and are the ones pulling the
        // trigger from 25yd; the veto derived from this value makes a striker work a far,
        // covered ball closer instead of hammering it from distance. `None` (kill-switch) ⇒
        // no veto ⇒ byte-identical.
        let shot_trigger_value = if shot_trigger_mdp_enabled() {
            Some(if observation.shot_trigger_mdp_value >= 0.0 {
                observation.shot_trigger_mdp_value
            } else {
                analytic_shot_trigger_value(&ShotTriggerInputs::from_observation(
                    &observation,
                    self.role,
                    shooting_skill,
                    1.0,
                ))
            })
        } else {
            None
        };
        // Stop-volunteering discipline: beyond `shooting.shot_trigger_long_range_yards` a *forced*
        // shot needs a real MDP justification (open net / stranded keeper / live rebound),
        // otherwise the rule paths below are skipped and the carrier works it closer. Inside
        // that range the forced paths fire exactly as before (a clean 18-22yd must-shoot is
        // untouched). `None` (kill-switch) ⇒ no veto ⇒ byte-identical.
        let shot_trigger_long_range_veto = match shot_trigger_value {
            Some(v) => {
                observation.yards_to_goal > tunables().shooting.shot_trigger_long_range_yards
                    && v < tunables().shooting.shot_trigger_long_range_min_value
            }
            None => false,
        };
        let shot_trigger_allows_forced_shot = !shot_trigger_long_range_veto;

        if has_ball
            && goal_attack_shot_blocks_alternatives(&observation, self.role)
            && shot_trigger_allows_forced_shot
            && !threaded_goal_pass_can_override_forced_shot(&observation, self.role)
        {
            let action = SoccerAction::Shoot {
                power: shot_power_for_skill(shooting_skill),
            };
            let action_label = action.label();
            self.last_decision = Some(self.decision_trace(
                snapshot,
                mdp_state,
                observation,
                belief,
                vec!["goalmouth-window".to_string(), "must-shoot".to_string()],
                single_action_option("shoot"),
                &action,
                action_label,
            ));
            return PlayerIntent {
                player_id: self.id,
                action,
                sprint: false,
            };
        } else if has_ball
            && !shot_trigger_long_range_veto
            && striker_shot_window_is_qualified(&observation, self.role)
        {
            let striker_shot_bonus = striker_legal_shot_attempt_bonus(&observation, self.role);
            let window_shot_chance = (0.66 + shooting_skill * 0.14 + striker_shot_bonus * 0.18
                - observation.shot_block_probability.clamp(0.0, 1.0) * 0.08)
                .clamp(0.68, 0.94);
            if agentic_action_commitment(
                window_shot_chance,
                snapshot.dt_seconds,
                &observation,
                self.role,
            ) {
                let action = SoccerAction::Shoot {
                    power: shot_power_for_skill(shooting_skill),
                };
                let action_label = action.label();
                self.last_decision = Some(self.decision_trace(
                    snapshot,
                    mdp_state,
                    observation,
                    belief,
                    vec!["striker-window".to_string(), "shoot".to_string()],
                    single_action_option("shoot"),
                    &action,
                    action_label,
                ));
                return PlayerIntent {
                    player_id: self.id,
                    action,
                    sprint: false,
                };
            }
        } else if has_ball
            && !shot_trigger_long_range_veto
            && (shot_decision_is_qualified_for_role(&observation, self.role)
                || contested_final_third_shot_is_qualified(&observation, self.role, shooting_skill))
            && !threaded_goal_pass_can_override_forced_shot(&observation, self.role)
            && !goal_attack_shot_can_yield_to_support(&observation, self.role)
        {
            let striker_shot_bonus = striker_legal_shot_attempt_bonus(&observation, self.role);
            let finish_chance = (0.018
                + shooting_skill * 0.050
                + observation.shot_on_frame_probability * 0.060
                + observation.shot_beat_goalkeeper_probability * 0.034
                + observation.shot_curl_probability * 0.012
                + striker_shot_bonus * 0.46
                - observation.shot_block_probability * 0.030
                - self.skills.decision_noise * 0.12)
                .clamp(0.012, 0.12 + striker_shot_bonus * 0.34)
                .max(close_clear_shot_attempt_probability(
                    &observation,
                    self.role,
                    shooting_skill,
                ))
                .max(attacking_goal_pressure_shot_attempt_probability(
                    &observation,
                    self.role,
                    shooting_skill,
                ))
                .max(contested_final_third_shot_attempt_probability(
                    &observation,
                    self.role,
                    shooting_skill,
                ))
                .max(striker_shot_bonus);
            if agentic_action_commitment(
                finish_chance,
                snapshot.dt_seconds,
                &observation,
                self.role,
            ) {
                let action = SoccerAction::Shoot {
                    power: shot_power_for_skill(shooting_skill),
                };
                let action_label = action.label();
                self.last_decision = Some(self.decision_trace(
                    snapshot,
                    mdp_state,
                    observation,
                    belief,
                    vec!["finish".to_string()],
                    single_action_option("shoot"),
                    &action,
                    action_label,
                ));
                return PlayerIntent {
                    player_id: self.id,
                    action,
                    sprint: false,
                };
            }
        }

        if has_ball {
            // Run-around dribble (knock the ball past a committed defender on one side, sprint the
            // body the OTHER side, re-collect behind). Gated pre-empt — parity-safe when disabled,
            // fires only on a genuine pace-beat opportunity (or always under SOCCER_FORCE_RUNAROUND).
            if let Some(plan) = snapshot.runaround_dribble_option_for(self.id) {
                let outside_mid_takeon = observation.outside_mid_takeon_isolation.clamp(0.0, 1.0);
                let outside_mid_attack_defender = outside_mid_takeon >= 0.55;
                let appetite = (0.15 + 0.60 * plan.quality + 0.24 * outside_mid_takeon).clamp(
                    0.05,
                    if outside_mid_attack_defender {
                        0.96
                    } else {
                        0.85
                    },
                );
                if soccer_force_runaround_commit()
                    || agentic_action_commitment(
                        appetite,
                        snapshot.dt_seconds,
                        &observation,
                        self.role,
                    )
                {
                    self.runaround = Some(RunaroundDribble {
                        defender: plan.defender,
                        launch_clock_seconds: snapshot.clock_seconds,
                        push_target: plan.push_target,
                        recollect_point: plan.recollect_point,
                        knocked: false,
                    });
                    // The body runs to the re-collect point; the knock itself is applied in
                    // `apply_player_intent`, and the MPC routes this MoveTo around the defender.
                    let action = SoccerAction::MoveTo(plan.recollect_point);
                    let operation_order = if outside_mid_attack_defender {
                        vec![
                            "outside-mid-attack-defender".to_string(),
                            "runaround-dribble".to_string(),
                        ]
                    } else {
                        vec!["runaround-dribble".to_string()]
                    };
                    self.last_decision = Some(self.decision_trace(
                        snapshot,
                        mdp_state,
                        observation,
                        belief,
                        operation_order,
                        single_action_option("runaround-dribble"),
                        &action,
                        "runaround-dribble",
                    ));
                    return PlayerIntent {
                        player_id: self.id,
                        action,
                        sprint: true,
                    };
                }
            }
            let ground_killer_priority = observation.threaded_goal_pass_available
                && observation.visible_forward_pass_options > 0
                && observation.yards_to_goal <= 42.0
                && snapshot
                    .killer_pass_target_for(
                        self.id,
                        &snapshot.ranked_visible_pass_targets(self.id, 3),
                    )
                    .is_some()
                && killer_pass_goal_pressure_score(&observation)
                    .max(single_pass_goal_thread_pressure_score(&observation))
                    .max(threaded_goal_pass_quality_fit(&observation) * 0.82)
                    >= 0.40;
            if !ground_killer_priority && !learned_open_pass_lane_requested {
                // Only PRE-EMPT-commit the long ball when it is genuinely on (an open onside runner
                // the aerial pass can actually reach). Otherwise fall through to the full ranked
                // decision instead of forcing a 0%-completion hoof. Gated; see the helper.
                if let Some(target) = snapshot
                    .long_ball_in_behind_target(self.id)
                    .filter(|_| long_ball_in_behind_is_viable(&observation))
                    .filter(|&t| {
                        // Part B: a DEEP carrier only lofts the long ball to a teammate who
                        // actually made the run — otherwise defer to the ranked decision
                        // (keep / short outlet / dribble) instead of hoofing to nobody.
                        snapshot.deep_long_aerial_target_is_committed_runner(self.id, t)
                    })
                {
                    let crossing =
                        ability01(self.skills.crossing_left.max(self.skills.crossing_right));
                    let runner_multiplier = observation
                        .aerial_forward_runner_pass_multiplier
                        .clamp(1.0, 1.50);
                    let long_ball_chance = ((0.10
                        + passing_skill * 0.07
                        + crossing * 0.05
                        + directive.risk_tolerance * 0.03)
                        .clamp(0.05, 0.22)
                        * runner_multiplier)
                        .clamp(0.05, 0.34);
                    if agentic_action_commitment(
                        long_ball_chance,
                        snapshot.dt_seconds,
                        &observation,
                        self.role,
                    ) {
                        let action = SoccerAction::Pass {
                            target_player: Some(target),
                            power: 0.80 + 0.14 * crossing.max(passing_skill),
                            flight: PassFlight::Aerial,
                        };
                        let decision = self.decision_trace(
                            snapshot,
                            mdp_state,
                            observation,
                            belief,
                            vec!["long-ball-in-behind".to_string()],
                            single_action_option("aerial-pass"),
                            &action,
                            "aerial-pass",
                        );
                        self.last_decision = Some(decision);
                        return PlayerIntent {
                            player_id: self.id,
                            action,
                            sprint: false,
                        };
                    }
                }
            }
        }

        if !has_ball {
            // Run-around re-collect: after the knock, sprint the body to the rendezvous behind the
            // defender to recover the loose ball — the MPC routes this MoveTo around the defender as
            // a moving keep-out. Takes precedence over receiving a pass.
            if let Some(run) = self.runaround {
                if run.knocked {
                    let action = SoccerAction::MoveTo(run.recollect_point);
                    self.last_decision = Some(self.decision_trace(
                        snapshot,
                        mdp_state,
                        observation,
                        belief,
                        vec!["runaround-collect".to_string()],
                        single_action_option("runaround-collect"),
                        &action,
                        "runaround-collect",
                    ));
                    return PlayerIntent {
                        player_id: self.id,
                        action,
                        sprint: true,
                    };
                }
            }
            if let Some(target) = snapshot.teammate_pass_lane_clearance_target_for(self.id) {
                let action = SoccerAction::MoveTo(target);
                self.last_decision = Some(self.decision_trace(
                    snapshot,
                    mdp_state,
                    observation,
                    belief,
                    vec![
                        "pomdp-clear-pass-lane".to_string(),
                        DUMMY_CLEAR_LANE_ACTION_LABEL.to_string(),
                    ],
                    single_action_option(DUMMY_CLEAR_LANE_ACTION_LABEL),
                    &action,
                    DUMMY_CLEAR_LANE_ACTION_LABEL,
                ));
                return PlayerIntent {
                    player_id: self.id,
                    action,
                    sprint: false,
                };
            }
            if let Some((target, sprint)) = snapshot.pending_pass_reception_target_for(self.id) {
                let action = SoccerAction::MoveTo(target);
                self.last_decision = Some(self.decision_trace(
                    snapshot,
                    mdp_state,
                    observation,
                    belief,
                    vec!["receive-pending-pass".to_string()],
                    single_action_option(ACTION_LABEL_RECOVER),
                    &action,
                    ACTION_LABEL_RECOVER,
                ));
                return PlayerIntent {
                    player_id: self.id,
                    action,
                    sprint,
                };
            }
        }

        if !has_ball {
            if let Some((target, sprint)) =
                snapshot.controllable_ground_pass_trajectory_for(self.id)
            {
                let action = SoccerAction::MoveTo(target);
                self.last_decision = Some(self.decision_trace(
                    snapshot,
                    mdp_state,
                    observation,
                    belief,
                    vec![
                        TRACE_REACTIVE_GROUND_PASS.to_string(),
                        TRACE_TRAP_CONTROLLABLE_TRAJECTORY.to_string(),
                    ],
                    single_action_option(ACTION_LABEL_RECOVER),
                    &action,
                    ACTION_LABEL_RECOVER,
                ));
                return PlayerIntent {
                    player_id: self.id,
                    action,
                    sprint,
                };
            }
        }

        // Anticipate where the in-flight pass is going and race for it: a defender jumps
        // the lane to intercept (gated by an LP/IPM position budget so it stays in shape),
        // or a second attacker crashes a contested reception. Off by default; pure
        // movement intent (no RNG), so baseline play is unchanged when disabled.
        if !has_ball {
            if let Some((target, sprint)) = snapshot.pass_contest_intent_for(self.id) {
                let action = SoccerAction::MoveTo(target);
                self.last_decision = Some(self.decision_trace(
                    snapshot,
                    mdp_state,
                    observation,
                    belief,
                    vec![
                        "anticipate-pass".to_string(),
                        "contest-reception".to_string(),
                    ],
                    single_action_option(ACTION_LABEL_RECOVER),
                    &action,
                    ACTION_LABEL_RECOVER,
                ));
                return PlayerIntent {
                    player_id: self.id,
                    action,
                    sprint,
                };
            }
        }

        if !has_ball && snapshot.controlled_possession_team() == Some(self.team.other()) {
            if let Some(holder) = self.immediate_defensive_steal_target(snapshot) {
                let action = SoccerAction::Tackle {
                    target_player: holder,
                };
                let action_options =
                    self.defensive_action_options(snapshot, &directive, snapshot.dt_seconds);
                self.last_decision = Some(self.decision_trace(
                    snapshot,
                    mdp_state,
                    observation,
                    belief,
                    vec!["immediate-steal".to_string(), "tackle".to_string()],
                    action_options,
                    &action,
                    "tackle",
                ));
                return PlayerIntent {
                    player_id: self.id,
                    action,
                    sprint: false,
                };
            }
        }

        // Last-ditch committed challenge: throw a SLIDE tackle at a fast carrier driving
        // at our goal that this defender can reach but not contain on its feet. The gate
        // (`slide_tackle_decision`) is narrow and draws from the decision RNG ONLY when a
        // genuine slide window is open, so ordinary play stays parity-preserved (and is
        // byte-identical when the mechanic is disabled). Placed after the standing
        // immediate-steal so a clean, risk-free steal is always preferred to a gamble.
        if !has_ball && snapshot.controlled_possession_team() == Some(self.team.other()) {
            if let Some((holder, appetite)) = self.slide_tackle_decision(snapshot) {
                if _rng.next_float() < time_window_probability(appetite, snapshot.dt_seconds) {
                    let action = SoccerAction::SlideTackle {
                        target_player: holder,
                    };
                    let action_options =
                        self.defensive_action_options(snapshot, &directive, snapshot.dt_seconds);
                    self.last_decision = Some(self.decision_trace(
                        snapshot,
                        mdp_state,
                        observation,
                        belief,
                        vec!["last-ditch".to_string(), "slide-tackle".to_string()],
                        action_options,
                        &action,
                        "slide-tackle",
                    ));
                    return PlayerIntent {
                        player_id: self.id,
                        action,
                        sprint: true,
                    };
                }
            }
        }

        if !has_ball {
            if let Some((target, sprint)) = snapshot.loose_long_ball_recovery_intent_for(self.id) {
                let action = SoccerAction::MoveTo(target);
                self.last_decision = Some(self.decision_trace(
                    snapshot,
                    mdp_state,
                    observation,
                    belief,
                    vec!["long-ball-duel".to_string(), ACTION_LABEL_RECOVER.to_string()],
                    single_action_option(ACTION_LABEL_RECOVER),
                    &action,
                    ACTION_LABEL_RECOVER,
                ));
                return PlayerIntent {
                    player_id: self.id,
                    action,
                    sprint,
                };
            }
        }

        let mut learned_mpc_reselect_label: Option<String> = None;
        if let Some(plan) = learned_plan {
            if let Some((action, action_label)) =
                self.action_from_learned_plan(plan, snapshot, &observation)
            {
                if matches!(action, SoccerAction::Shoot { .. }) {
                    let striker_shot_bonus =
                        striker_legal_shot_attempt_bonus(&observation, self.role);
                    let learned_shot_chance = (0.025
                        + shooting_skill * 0.050
                        + observation.shot_curl_probability * 0.010
                        + striker_shot_bonus * 0.42
                        - observation.shot_block_probability * 0.026
                        - self.skills.decision_noise * 0.10)
                        .clamp(0.008, 0.08 + striker_shot_bonus * 0.30)
                        .max(close_clear_shot_attempt_probability(
                            &observation,
                            self.role,
                            shooting_skill,
                        ))
                        .max(attacking_goal_pressure_shot_attempt_probability(
                            &observation,
                            self.role,
                            shooting_skill,
                        ))
                        .max(speculative_long_shot_attempt_probability(
                            &observation,
                            self.role,
                            shooting_skill,
                        ))
                        .max(striker_shot_bonus);
                    if !goal_attack_shot_blocks_alternatives(&observation, self.role)
                        && !agentic_action_commitment(
                            learned_shot_chance,
                            snapshot.dt_seconds,
                            &observation,
                            self.role,
                        )
                    {
                        let kind = if goal_approach_carry_preferred(&observation, self.role) {
                            DribbleMoveKind::CarryForward
                        } else {
                            self.agentic_fallback_dribble_kind(snapshot, &observation, false, true)
                        };
                        let touch =
                            snapshot.deterministic_dribble_touch_decision_for(self.id, kind);
                        let defer_action = SoccerAction::DribbleMove {
                            target: snapshot.dribble_move_target_for_touch(
                                self.id,
                                self.home_position,
                                kind,
                                touch,
                            ),
                            kind,
                            touch,
                        };
                        let mut decision = self.decision_trace(
                            snapshot,
                            mdp_state,
                            observation,
                            belief,
                            vec![
                                "learned-policy".to_string(),
                                plan.action.clone(),
                                "defer-shot".to_string(),
                            ],
                            single_action_option(kind.label()),
                            &defer_action,
                            kind.label(),
                        );
                        decision.learned_mpc_replan = plan.mpc_replan.clone();
                        self.last_decision = Some(decision);
                        return PlayerIntent {
                            player_id: self.id,
                            action: defer_action,
                            sprint: false,
                        };
                    }
                }
                if mpc_reselects_candidate(snapshot, self, &action, &action_label) {
                    learned_mpc_reselect_label = Some(action_label.clone());
                } else {
                    if action_label == "wall-pass" {
                        if let Some(plan) = snapshot.wall_pass_option_for(self.id) {
                            self.one_two = Some(OneTwoRun {
                                wall_partner: plan.wall_partner,
                                launch_clock_seconds: snapshot.clock_seconds,
                                return_target: plan.return_target,
                            });
                        }
                    }
                    if action_label == "runaround-dribble" {
                        if let Some(plan) = snapshot.runaround_dribble_option_for(self.id) {
                            self.runaround = Some(RunaroundDribble {
                                defender: plan.defender,
                                launch_clock_seconds: snapshot.clock_seconds,
                                push_target: plan.push_target,
                                recollect_point: plan.recollect_point,
                                knocked: false,
                            });
                        }
                    }
                    let first_touch_escape_sprint =
                        matches!(action, SoccerAction::ControlTouch { .. })
                            && observation.first_touch_escape_pressure
                                >= FIRST_TOUCH_ESCAPE_SPRINT_PRESSURE;
                    let sprint = if first_touch_escape_sprint {
                        true
                    } else if action_label == "runaround-dribble" {
                        true
                    } else {
                        open_pass_lane_sprint_for_action(
                            snapshot,
                            self,
                            &observation,
                            &action_label,
                            &action,
                        )
                    };
                    let mut decision = self.decision_trace(
                        snapshot,
                        mdp_state,
                        observation,
                        belief,
                        vec!["learned-policy".to_string(), plan.action.clone()],
                        single_action_option(&action_label),
                        &action,
                        action_label,
                    );
                    decision.learned_mpc_replan = plan.mpc_replan.clone();
                    self.last_decision = Some(decision);
                    return PlayerIntent {
                        player_id: self.id,
                        action,
                        sprint,
                    };
                }
            }
        }

        if has_ball {
            let pass_targets = snapshot.ranked_visible_pass_targets(self.id, 3);
            let strategic_pass_targets = snapshot.ranked_visible_pass_targets(self.id, 11);
            let aerial_pass_targets = snapshot.ranked_visible_aerial_pass_targets(self.id, 3);
            let hold_up_flank_target = snapshot.striker_hold_up_flank_target_for(self.id);
            let mut action_options = self.possession_action_options(
                &observation,
                &directive,
                pass_targets.len(),
                aerial_pass_targets.len(),
                hold_up_flank_target.is_some(),
                snapshot.dt_seconds,
                snapshot.field_width,
            );
            let open_pass_lane_plan = self.open_pass_lane_dribble_plan_for(
                snapshot,
                &observation,
                &strategic_pass_targets,
            );
            if let Some(plan) = open_pass_lane_plan {
                action_options.push(AgentActionOptionTrace::new(
                    OPEN_PASS_LANE_ACTION_LABEL,
                    plan.score,
                    true,
                ));
            }
            let round_goalkeeper_plan =
                self.round_goalkeeper_dribble_plan_for(snapshot, &observation);
            if let Some(plan) = round_goalkeeper_plan {
                action_options.push(AgentActionOptionTrace::new(
                    ROUND_GOALKEEPER_ACTION_LABEL,
                    plan.score,
                    true,
                ));
            }
            let runaround_dribble_option =
                snapshot.runaround_dribble_option_for(self.id).map(|plan| {
                    let outside_mid_takeon =
                        observation.outside_mid_takeon_isolation.clamp(0.0, 1.0);
                    let defender_overcommit = observation
                        .neural_extended
                        .nearest_defender_overcommit_score
                        .clamp(0.0, 1.0);
                    let perception = &tunables().pomdp_perception;
                    let defender_reaction_delay_fit = ((observation
                        .neural_extended
                        .nearest_defender_reaction_delay_seconds
                        - perception.player_reaction_min_seconds)
                        / perception.player_reaction_span_seconds())
                    .clamp(0.0, 1.0);
                    let appetite = (0.18
                        + plan.quality * 0.58
                        + outside_mid_takeon * 0.28
                        + defender_overcommit * 0.26
                        + defender_reaction_delay_fit * 0.08)
                        .clamp(
                            0.05,
                            if outside_mid_takeon >= 0.55 || defender_overcommit >= 0.45 {
                                0.97
                            } else {
                                0.86
                            },
                        );
                    let strategy_commits = (outside_mid_takeon >= 0.75 && plan.quality >= 0.60)
                        || (defender_overcommit >= 0.72 && plan.quality >= 0.55);
                    (plan, appetite, strategy_commits)
                });
            if let Some((_, appetite, strategy_commits)) = runaround_dribble_option {
                action_options.push(AgentActionOptionTrace::new(
                    "runaround-dribble",
                    if strategy_commits {
                        appetite.max(1.0)
                    } else {
                        appetite
                    },
                    true,
                ));
            }
            let killer_pass_preempt_chance = if snapshot
                .killer_pass_target_for(self.id, &pass_targets)
                .is_some()
            {
                final_third_killer_pass_preemption_probability(
                    &observation,
                    self.role,
                    &action_options,
                )
            } else {
                0.0
            };
            if killer_pass_preempt_chance > 0.0 {
                if let Some(option) = action_options
                    .iter_mut()
                    .find(|option| option.legal && option.label == "killer-pass")
                {
                    option.score = option.score.max(killer_pass_preempt_chance);
                }
            }
            let scoop_strategy_requested = directive_requests_lob_or_scoop(directive);
            let scoop_pass_option = snapshot.scoop_pass_target_for(self.id).map(|target| {
                let technique = ability01(self.skills.flair_passing) * 0.5
                    + ability01(self.skills.passing) * 0.3
                    + ability01(self.skills.vision) * 0.2;
                let scoop_chance = (0.22 + technique * 0.56)
                    .max(if scoop_strategy_requested {
                        0.78 + technique * 0.10
                    } else {
                        0.0
                    })
                    .clamp(0.0, if scoop_strategy_requested { 0.94 } else { 0.80 });
                (target, scoop_chance)
            });
            if let Some((_, scoop_chance)) = scoop_pass_option {
                action_options.push(AgentActionOptionTrace::new(
                    "scoop-pass",
                    scoop_chance,
                    true,
                ));
                if scoop_strategy_requested {
                    ensure_min_legal_option_probability(&mut action_options, "scoop-pass", 0.64);
                }
            }
            let wall_pass_option = snapshot.wall_pass_option_for(self.id).map(|plan| {
                let give_and_go_strategy = directive_requests_give_and_go(directive);
                let goal_proximity = (1.0
                    - (observation.yards_to_goal / WALL_PASS_GOAL_PROXIMITY_REFERENCE_YARDS)
                        .clamp(0.0, 1.0))
                .clamp(0.0, 1.0);
                let ambition_appetite = if attack_ambition_enabled() { 1.45 } else { 1.0 };
                let raw_wall_appetite = WALL_PASS_BASE_APPETITE
                    * ambition_appetite
                    * self.preferences.pass_bias.clamp(0.4, 1.0)
                    * (0.55 + passing_skill * 0.45 + ability01(self.skills.vision) * 0.25)
                    * (0.5 + plan.quality)
                    * (1.0 + observation.perceived_pressure.clamp(0.0, 1.0) * 0.9)
                    * (1.0 + goal_proximity * 0.8)
                    * if give_and_go_strategy {
                        WALL_PASS_STRATEGY_APPETITE_BOOST
                    } else {
                        1.0
                    };
                let strategy_commitment_floor = if give_and_go_strategy {
                    (WALL_PASS_STRATEGY_MIN_APPETITE + plan.quality * 0.14).clamp(0.0, 0.95)
                } else {
                    0.0
                };
                let wall_appetite = raw_wall_appetite
                    .max(strategy_commitment_floor)
                    .clamp(0.0, if give_and_go_strategy { 0.97 } else { 0.92 });
                let strategy_commits =
                    give_and_go_strategy && plan.quality >= WALL_PASS_STRATEGY_COMMIT_MIN_QUALITY;
                (plan, wall_appetite, strategy_commits)
            });
            if let Some((_, wall_appetite, strategy_commits)) = wall_pass_option {
                action_options.push(AgentActionOptionTrace::new(
                    "wall-pass",
                    if strategy_commits {
                        wall_appetite.max(1.0)
                    } else {
                        wall_appetite
                    },
                    true,
                ));
            }
            // Hold-and-summon support: a calm carrier with no good forward outlet may DELAY the
            // release and signal teammates to improve their position (see
            // `WorldSnapshot::hold_for_support_option_for`). Added conditionally (like the
            // wall-pass / scoop options) so the option set is byte-identical to baseline outside
            // this window. Never offered when something decisive is already on — it is the patient
            // "let the play set up" choice, not a substitute for a shot or a killer ball.
            let hold_for_support_option = snapshot.hold_for_support_option_for(self.id);
            let byline_corner_drive = snapshot.byline_cross_drive_active_for(self.id);
            let mut hold_for_support_offered = false;
            if let Some(plan) = hold_for_support_option.as_ref() {
                let decisive_on = observation.threaded_goal_pass_available
                    || goal_attack_shot_blocks_alternatives(&observation, self.role)
                    || must_shoot_near_goal(&observation, self.role);
                if !decisive_on {
                    let patience = low_pressure_patience_factor(&observation);
                    let calm = ((1.0
                        - observation
                            .pressure_urgency
                            .max(observation.perceived_pressure))
                        * (1.0 - pressure_rising_signal(&observation)))
                    .clamp(0.0, 1.0);
                    let appetite = (HOLD_FOR_SUPPORT_BASE_APPETITE
                        * (0.40 + plan.quality * 0.90)
                        * (0.60 + patience * 0.60)
                        * (0.50 + calm * 0.70))
                        .clamp(0.0, HOLD_FOR_SUPPORT_MAX_APPETITE);
                    action_options.push(AgentActionOptionTrace::new(
                        "wait-for-support",
                        appetite,
                        true,
                    ));
                    hold_for_support_offered = true;
                }
            }
            // XAVI TURN: a high-skill shielded pirouette that turns a tight defender the
            // long way around, keeping the ball on the far side throughout. Offered (like
            // the wall-pass / wait-for-support options) only in its window — a defender
            // right on top with the forward path blocked — and only when live
            // (`xavi_turn_enabled`). Added conditionally so the option set is byte-identical
            // to baseline outside this window, which keeps a disabled match in lock-step.
            // DRIBBLE TO OPEN A PASSING LANE: the ideal receiver's direct lane is blocked, so a
            // short (1-8yd) quick carry to one side shifts the angle off the blocker and opens the
            // pass. Computed once; offered as its own option and executed as an MPC-driven carry to
            // the spot. Returns the spot, the receiver it opens, and whether to sprint (very high
            // pressure / a fast-tracking opponent) vs run.
            let open_lane_dribble = if self.role != PlayerRole::Goalkeeper {
                snapshot.dribble_to_open_passing_lane_for(self.id)
            } else {
                None
            };
            let mut open_lane_offered = false;
            if let Some((_spot, receiver_id, _sprint)) = open_lane_dribble {
                let press = observation
                    .perceived_pressure
                    .max(observation.pressure_urgency)
                    .clamp(0.0, 1.0);
                // MDP/POMDP value: worth more when it opens a PROGRESSIVE (upfield) pass, and
                // lifted by pressure (the more boxed in, the more the angle is worth making).
                let upfield = snapshot
                    .players
                    .iter()
                    .find(|p| p.id == receiver_id)
                    .map(|r| {
                        ((snapshot.player_snapshot_position(r).y - self.position.y)
                            * self.team.attack_dir()
                            / 20.0)
                            .clamp(0.0, 1.0)
                    })
                    .unwrap_or(0.0);
                let appetite = (OPEN_LANE_DRIBBLE_BASE_APPETITE
                    * self.preferences.dribble_bias.clamp(0.5, 1.2)
                    * (0.70 + press * 0.50)
                    + OPEN_LANE_DRIBBLE_UPFIELD_APPETITE_BONUS * upfield)
                    .clamp(0.0, OPEN_LANE_DRIBBLE_MAX_APPETITE);
                if let Some(option) = action_options
                    .iter_mut()
                    .find(|option| option.label == "open-passing-lane")
                {
                    option.legal = true;
                    option.score = option.score.max(appetite);
                } else {
                    action_options.push(AgentActionOptionTrace::new(
                        "open-passing-lane",
                        appetite,
                        true,
                    ));
                }
                open_lane_offered = true;
            }
            // ROUND THE KEEPER is generated above as `round_goalkeeper_plan` (the unified,
            // plan-based implementation); ours' separate `dribble_round_the_keeper_for` generator
            // was folded into it during the merge to avoid offering the same carry twice.
            let mut xavi_turn_offered = false;
            if snapshot.xavi_turn_enabled
                && self.role != PlayerRole::Goalkeeper
                && observation.nearest_opponent_distance <= XAVI_TURN_MAX_DEFENDER_YARDS
                // Never wheel and turn your back deep in your own defensive third.
                && observation.yards_to_own_goal >= XAVI_TURN_MIN_OWN_GOAL_YARDS
                && !goal_approach_forces_goalmouth_carry(&observation, self.role)
            {
                let press = observation
                    .perceived_pressure
                    .max(observation.pressure_urgency)
                    .clamp(0.0, 1.0);
                let forward_blocked = (1.0
                    - observation.forward_dribble_space_yards / XAVI_TURN_FORWARD_BLOCK_YARDS)
                    .clamp(0.0, 1.0);
                if press >= XAVI_TURN_MIN_PRESSURE && forward_blocked >= 0.25 {
                    // Heavily skill-weighted: only a genuinely technical carrier reaches an
                    // appetite that can edge out a side-step / cut and rival a static shield;
                    // a journeyman stays well below them and keeps the ball more simply. The
                    // skill gate is superlinear (`^1.6`) so the elite/average gap is wide
                    // enough that, even when totally boxed in, an average carrier shields
                    // rather than attempting a turn it cannot reliably pull off.
                    let technique = (ability01(self.skills.dribbling) * 0.62
                        + ability01(self.skills.flair_passing) * 0.20
                        + ability01(self.skills.first_touch) * 0.18)
                        .clamp(0.0, 1.0);
                    let skill_gate = 0.10 + technique.powf(1.6) * 1.85;
                    let appetite = (XAVI_TURN_BASE_APPETITE
                        * self.preferences.dribble_bias.clamp(0.4, 1.2)
                        * skill_gate
                        * (0.55 + press * 0.85)
                        * (0.55 + forward_blocked * 0.80))
                        .clamp(0.0, XAVI_TURN_MAX_APPETITE);
                    if let Some(option) = action_options
                        .iter_mut()
                        .find(|option| option.label == "xavi-turn")
                    {
                        option.legal = true;
                        option.score = option.score.max(appetite);
                    } else {
                        action_options.push(AgentActionOptionTrace::new(
                            "xavi-turn",
                            appetite,
                            true,
                        ));
                    }
                    xavi_turn_offered = true;
                }
            }
            action_options = normalize_action_options(action_options);
            annotate_tick_probabilities_from_scores(&mut action_options, snapshot.dt_seconds);
            let mut weighted_ops = vec![
                (
                    "shoot".to_string(),
                    action_option_score(&action_options, "shoot"),
                ),
                (
                    "dribble".to_string(),
                    action_option_score(&action_options, "dribble"),
                ),
                (
                    "carry-forward".to_string(),
                    action_option_score(&action_options, "carry-forward"),
                ),
                (
                    "vertical-attack".to_string(),
                    action_option_score(&action_options, "vertical-attack"),
                ),
                (
                    "turnover-burst".to_string(),
                    action_option_score(&action_options, "turnover-burst"),
                ),
                (
                    "carry-out-left".to_string(),
                    action_option_score(&action_options, "carry-out-left"),
                ),
                (
                    "carry-out-right".to_string(),
                    action_option_score(&action_options, "carry-out-right"),
                ),
                (
                    "protect-ball".to_string(),
                    action_option_score(&action_options, "protect-ball"),
                ),
                (
                    "side-step".to_string(),
                    action_option_score(&action_options, "side-step"),
                ),
                (
                    "left-cut".to_string(),
                    action_option_score(&action_options, "left-cut"),
                ),
                (
                    "right-cut".to_string(),
                    action_option_score(&action_options, "right-cut"),
                ),
                (
                    "nutmeg".to_string(),
                    action_option_score(&action_options, "nutmeg"),
                ),
                (
                    "fake-left-cut-right".to_string(),
                    action_option_score(&action_options, "fake-left-cut-right"),
                ),
                (
                    "fake-right-cut-left".to_string(),
                    action_option_score(&action_options, "fake-right-cut-left"),
                ),
                (
                    "hold-up-flank".to_string(),
                    action_option_score(&action_options, "hold-up-flank"),
                ),
                (
                    "clearance".to_string(),
                    action_option_score(&action_options, "clearance"),
                ),
                (
                    "route-one".to_string(),
                    action_option_score(&action_options, "route-one"),
                ),
                (
                    "flank-low-cross".to_string(),
                    action_option_score(&action_options, "flank-low-cross"),
                ),
                (
                    "flank-high-cross".to_string(),
                    action_option_score(&action_options, "flank-high-cross"),
                ),
                (
                    "corner-flag-cross".to_string(),
                    action_option_score(&action_options, "corner-flag-cross"),
                ),
                (
                    "killer-pass".to_string(),
                    action_option_score(&action_options, "killer-pass"),
                ),
                (
                    "recycle-reset".to_string(),
                    action_option_score(&action_options, "recycle-reset"),
                ),
                (
                    "switch-play".to_string(),
                    action_option_score(&action_options, "switch-play"),
                ),
                (
                    "surprise-pass".to_string(),
                    action_option_score(&action_options, "surprise-pass"),
                ),
                (
                    "scoop-pass".to_string(),
                    action_option_score(&action_options, "scoop-pass"),
                ),
                (
                    "flick-on".to_string(),
                    action_option_score(&action_options, "flick-on"),
                ),
            ];
            if wall_pass_option.is_some() {
                weighted_ops.push((
                    "wall-pass".to_string(),
                    action_option_score(&action_options, "wall-pass"),
                ));
            }
            if hold_for_support_offered {
                weighted_ops.push((
                    "wait-for-support".to_string(),
                    action_option_score(&action_options, "wait-for-support"),
                ));
            }
            if open_pass_lane_plan.is_some() {
                weighted_ops.push((
                    OPEN_PASS_LANE_ACTION_LABEL.to_string(),
                    action_option_score(&action_options, OPEN_PASS_LANE_ACTION_LABEL),
                ));
            }
            if runaround_dribble_option.is_some() {
                weighted_ops.push((
                    "runaround-dribble".to_string(),
                    action_option_score(&action_options, "runaround-dribble"),
                ));
            }
            if round_goalkeeper_plan.is_some() {
                weighted_ops.push((
                    ROUND_GOALKEEPER_ACTION_LABEL.to_string(),
                    action_option_score(&action_options, ROUND_GOALKEEPER_ACTION_LABEL),
                ));
            }
            if xavi_turn_offered {
                weighted_ops.push((
                    "xavi-turn".to_string(),
                    action_option_score(&action_options, "xavi-turn"),
                ));
            }
            if open_lane_offered {
                weighted_ops.push((
                    "open-passing-lane".to_string(),
                    action_option_score(&action_options, "open-passing-lane"),
                ));
            }
            for rank in 0..pass_targets.len() {
                let label = format!("pass{}", rank + 1);
                weighted_ops.push((label.clone(), action_option_score(&action_options, &label)));
            }
            for rank in 0..aerial_pass_targets.len() {
                let label = format!("aerial-pass{}", rank + 1);
                weighted_ops.push((label.clone(), action_option_score(&action_options, &label)));
            }
            let cadence_hold_label = decision_cadence_hold_label_from_weights(
                &observation,
                &weighted_ops,
                snapshot.tick,
            );
            if let Some(label) = cadence_hold_label.as_deref() {
                apply_decision_cadence_hold_weights(&mut weighted_ops, label);
            }
            let ops = {
                let (ordered, ordered_scores) = agentic_action_order_scored(weighted_ops);
                let sampled = reorder_with_scores_traced(
                    ordered,
                    &ordered_scores,
                    snapshot.decision_seed,
                    self.id,
                    snapshot.tick,
                    site::POSSESSION,
                );
                self.pending_policy_behavior_probability = sampled.behavior_probability;
                sampled.ordered
            };
            let full_ops = ops.clone();
            let mut order_names = Vec::with_capacity(ops.len());
            if let Some(label) = cadence_hold_label.as_deref() {
                order_names.push(format!("decision-cadence-hold:{label}"));
            }
            if let Some(label) = learned_mpc_reselect_label.as_deref() {
                order_names.push(format!("learned-mpc-reselect:{}", label));
            }
            let mut chosen = None;
            // Set by the open-passing-lane arm to its precise sprint decision (very high pressure /
            // a fast-tracking opponent), overriding the generic carry sprint rule below.
            let mut open_lane_sprint: Option<bool> = None;
            let mut runaround_sprint = false;
            let mut round_goalkeeper_sprint: Option<bool> = None;
            for op in ops {
                match op.as_str() {
                    "shoot" => {
                        order_names.push("shoot".to_string());
                        let shot_chance = action_option_score(&action_options, "shoot");
                        if (shot_decision_is_qualified_for_role(&observation, self.role)
                            || speculative_long_shot_is_qualified(
                                &observation,
                                self.role,
                                shooting_skill,
                            ))
                            && agentic_action_commitment(
                                shot_chance,
                                snapshot.dt_seconds,
                                &observation,
                                self.role,
                            )
                        {
                            chosen = Some((
                                SoccerAction::Shoot {
                                    power: shot_power_for_skill(shooting_skill),
                                },
                                "shoot".to_string(),
                            ));
                            break;
                        }
                    }
                    "dribble" => {
                        order_names.push("dribble".to_string());
                        let dribble_chance = action_option_score(&action_options, "dribble");
                        if agentic_action_commitment(
                            dribble_chance,
                            snapshot.dt_seconds,
                            &observation,
                            self.role,
                        ) {
                            let kind =
                                if goal_approach_forces_goalmouth_carry(&observation, self.role) {
                                    DribbleMoveKind::CarryForward
                                } else {
                                    self.agentic_fallback_dribble_kind(
                                        snapshot,
                                        &observation,
                                        false,
                                        true,
                                    )
                                };
                            let touch =
                                snapshot.deterministic_dribble_touch_decision_for(self.id, kind);
                            let target = snapshot.dribble_move_target_for_touch(
                                self.id,
                                self.home_position,
                                kind,
                                touch,
                            );
                            chosen = Some((
                                SoccerAction::DribbleMove {
                                    target,
                                    kind,
                                    touch,
                                },
                                kind.label().to_string(),
                            ));
                            break;
                        }
                    }
                    "vertical-attack" | "turnover-burst" => {
                        let kind = DribbleMoveKind::CarryForward;
                        order_names.push(op.to_string());
                        let carry_chance = action_option_score(&action_options, op.as_str());
                        if agentic_action_commitment(
                            carry_chance,
                            snapshot.dt_seconds,
                            &observation,
                            self.role,
                        ) {
                            let touch =
                                snapshot.deterministic_dribble_touch_decision_for(self.id, kind);
                            let target = snapshot.dribble_move_target_for_touch(
                                self.id,
                                self.home_position,
                                kind,
                                touch,
                            );
                            chosen = Some((
                                SoccerAction::DribbleMove {
                                    target,
                                    kind,
                                    touch,
                                },
                                op.to_string(),
                            ));
                            break;
                        }
                    }
                    "carry-forward" | "carry-out-left" | "carry-out-right" | "protect-ball" => {
                        let kind = match op.as_str() {
                            "carry-forward" => DribbleMoveKind::CarryForward,
                            "carry-out-left" => DribbleMoveKind::CarryOutLeft,
                            "carry-out-right" => DribbleMoveKind::CarryOutRight,
                            _ => DribbleMoveKind::ProtectBall,
                        };
                        let kind = goal_approach_dribble_kind(kind, &observation, self.role);
                        order_names.push(kind.label().to_string());
                        let carry_chance = action_option_score(&action_options, kind.label());
                        if agentic_action_commitment(
                            carry_chance,
                            snapshot.dt_seconds,
                            &observation,
                            self.role,
                        ) {
                            let touch =
                                snapshot.deterministic_dribble_touch_decision_for(self.id, kind);
                            let target = snapshot.dribble_move_target_for_touch(
                                self.id,
                                self.home_position,
                                kind,
                                touch,
                            );
                            chosen = Some((
                                SoccerAction::DribbleMove {
                                    target,
                                    kind,
                                    touch,
                                },
                                kind.label().to_string(),
                            ));
                            break;
                        }
                    }
                    "left-cut"
                    | "right-cut"
                    | "nutmeg"
                    | "fake-left-cut-right"
                    | "fake-right-cut-left" => {
                        let kind = match op.as_str() {
                            "left-cut" => DribbleMoveKind::LeftCut,
                            "right-cut" => DribbleMoveKind::RightCut,
                            "nutmeg" => DribbleMoveKind::Nutmeg,
                            "fake-left-cut-right" => DribbleMoveKind::FakeLeftCutRight,
                            _ => DribbleMoveKind::FakeRightCutLeft,
                        };
                        order_names.push(kind.label().to_string());
                        let technical_chance = action_option_score(&action_options, kind.label());
                        if agentic_action_commitment(
                            technical_chance,
                            snapshot.dt_seconds,
                            &observation,
                            self.role,
                        ) {
                            let touch =
                                snapshot.deterministic_dribble_touch_decision_for(self.id, kind);
                            let target = snapshot.dribble_move_target_for_touch(
                                self.id,
                                self.home_position,
                                kind,
                                touch,
                            );
                            chosen = Some((
                                SoccerAction::DribbleMove {
                                    target,
                                    kind,
                                    touch,
                                },
                                kind.label().to_string(),
                            ));
                            break;
                        }
                    }
                    "xavi-turn" => {
                        order_names.push("xavi-turn".to_string());
                        let xavi_chance = action_option_score(&action_options, "xavi-turn");
                        if agentic_action_commitment(
                            xavi_chance,
                            snapshot.dt_seconds,
                            &observation,
                            self.role,
                        ) {
                            let kind = DribbleMoveKind::XaviTurn;
                            let touch =
                                snapshot.deterministic_dribble_touch_decision_for(self.id, kind);
                            let target = snapshot.dribble_move_target_for_touch(
                                self.id,
                                self.home_position,
                                kind,
                                touch,
                            );
                            chosen = Some((
                                SoccerAction::DribbleMove {
                                    target,
                                    kind,
                                    touch,
                                },
                                "xavi-turn".to_string(),
                            ));
                            break;
                        }
                    }
                    "open-passing-lane" => {
                        order_names.push("open-passing-lane".to_string());
                        let open_lane_chance =
                            action_option_score(&action_options, "open-passing-lane");
                        if let Some((spot, _receiver, sprint_flag)) = open_lane_dribble {
                            if agentic_action_commitment(
                                open_lane_chance,
                                snapshot.dt_seconds,
                                &observation,
                                self.role,
                            ) {
                                // A controlled carry to the spot, executed through the per-player
                                // MPC like any dribble; sprint per the maneuver's own decision.
                                let kind = DribbleMoveKind::CarryForward;
                                let touch = snapshot
                                    .deterministic_dribble_touch_decision_for(self.id, kind);
                                open_lane_sprint = Some(sprint_flag);
                                chosen = Some((
                                    SoccerAction::DribbleMove {
                                        target: spot,
                                        kind,
                                        touch,
                                    },
                                    "open-passing-lane".to_string(),
                                ));
                                break;
                            }
                        }
                    }
                    "runaround-dribble" => {
                        if observation.outside_mid_takeon_isolation >= 0.55 {
                            order_names.push("outside-mid-attack-defender".to_string());
                        }
                        order_names.push("runaround-dribble".to_string());
                        if let Some((plan, appetite, strategy_commits)) = runaround_dribble_option {
                            if strategy_commits
                                || agentic_action_commitment(
                                    appetite,
                                    snapshot.dt_seconds,
                                    &observation,
                                    self.role,
                                )
                            {
                                self.runaround = Some(RunaroundDribble {
                                    defender: plan.defender,
                                    launch_clock_seconds: snapshot.clock_seconds,
                                    push_target: plan.push_target,
                                    recollect_point: plan.recollect_point,
                                    knocked: false,
                                });
                                runaround_sprint = true;
                                chosen = Some((
                                    SoccerAction::MoveTo(plan.recollect_point),
                                    "runaround-dribble".to_string(),
                                ));
                                break;
                            }
                        }
                    }
                    "side-step" => {
                        order_names.push("side-step".to_string());
                        let side_step_chance = action_option_score(&action_options, "side-step");
                        if agentic_action_commitment(
                            side_step_chance,
                            snapshot.dt_seconds,
                            &observation,
                            self.role,
                        ) {
                            let kind = snapshot.side_step_dribble_kind_for(self.id);
                            let touch =
                                snapshot.deterministic_dribble_touch_decision_for(self.id, kind);
                            let target = snapshot.dribble_move_target_for_touch(
                                self.id,
                                self.home_position,
                                kind,
                                touch,
                            );
                            chosen = Some((
                                SoccerAction::DribbleMove {
                                    target,
                                    kind,
                                    touch,
                                },
                                kind.label().to_string(),
                            ));
                            break;
                        }
                    }
                    "hold-up-flank" => {
                        order_names.push("hold-up-flank".to_string());
                        let hold_up_chance = action_option_score(&action_options, "hold-up-flank");
                        if let Some(target) = hold_up_flank_target {
                            if agentic_action_commitment(
                                hold_up_chance,
                                snapshot.dt_seconds,
                                &observation,
                                self.role,
                            ) {
                                let kind = snapshot.side_step_dribble_kind_for(self.id);
                                let bucket = match dribble_final_cut_kind(kind) {
                                    DribbleMoveKind::CarryForward
                                    | DribbleMoveKind::ProtectBall => 0,
                                    DribbleMoveKind::CarryOutLeft | DribbleMoveKind::LeftCut => 9,
                                    DribbleMoveKind::CarryOutRight | DribbleMoveKind::RightCut => 3,
                                    DribbleMoveKind::Nutmeg => 0,
                                    DribbleMoveKind::XaviTurn => 10,
                                    DribbleMoveKind::FakeLeftCutRight
                                    | DribbleMoveKind::FakeRightCutLeft => 0,
                                };
                                chosen = Some((
                                    SoccerAction::DribbleMove {
                                        target,
                                        kind,
                                        touch: DribbleTouchDecision::new(bucket, 1.35),
                                    },
                                    "hold-up-flank".to_string(),
                                ));
                                break;
                            }
                        }
                    }
                    "clearance" => {
                        order_names.push("clearance".to_string());
                        let clearance_chance = action_option_score(&action_options, "clearance");
                        if agentic_action_commitment(
                            clearance_chance,
                            snapshot.dt_seconds,
                            &observation,
                            self.role,
                        ) {
                            chosen = Some((
                                SoccerAction::Clearance {
                                    target: snapshot
                                        .pressure_clearance_target_for(self.id)
                                        .unwrap_or_else(|| {
                                            clearance_target_for_player(
                                                self.team,
                                                self.position,
                                                snapshot.field_width,
                                                snapshot.field_length,
                                            )
                                        }),
                                    power: 0.86
                                        + observation.perceived_pressure.clamp(0.0, 1.0) * 0.12,
                                },
                                "clearance".to_string(),
                            ));
                            break;
                        }
                    }
                    "route-one" => {
                        order_names.push("route-one".to_string());
                        let route_one_chance = action_option_score(&action_options, "route-one");
                        if agentic_action_commitment(
                            route_one_chance,
                            snapshot.dt_seconds,
                            &observation,
                            self.role,
                        ) {
                            chosen = Some((
                                SoccerAction::RouteOne {
                                    target: snapshot.route_one_target_for(self.id).unwrap_or_else(
                                        || {
                                            route_one_target_for_actor(
                                                self.team,
                                                self.position,
                                                snapshot.field_width,
                                                snapshot.field_length,
                                                self.role,
                                            )
                                        },
                                    ),
                                    power: 0.76
                                        + 0.18 * passing_skill.max(ability01(self.skills.strength)),
                                },
                                "route-one".to_string(),
                            ));
                            break;
                        }
                    }
                    "flank-low-cross" => {
                        order_names.push("flank-low-cross".to_string());
                        let cross_chance = action_option_score(&action_options, "flank-low-cross");
                        if let Some(target) = pass_targets.first().copied() {
                            if agentic_action_commitment(
                                cross_chance,
                                snapshot.dt_seconds,
                                &observation,
                                self.role,
                            ) {
                                let crossing = ability01(
                                    self.skills.crossing_left.max(self.skills.crossing_right),
                                );
                                chosen = Some((
                                    SoccerAction::Pass {
                                        target_player: Some(target),
                                        power: 0.60 + 0.26 * crossing.max(passing_skill),
                                        flight: PassFlight::Floor,
                                    },
                                    "flank-low-cross".to_string(),
                                ));
                                break;
                            }
                        }
                    }
                    "flank-high-cross" => {
                        order_names.push("flank-high-cross".to_string());
                        let cross_chance = action_option_score(&action_options, "flank-high-cross");
                        if let Some(target) = aerial_pass_targets.first().copied() {
                            if agentic_action_commitment(
                                cross_chance,
                                snapshot.dt_seconds,
                                &observation,
                                self.role,
                            ) {
                                let crossing = ability01(
                                    self.skills.crossing_left.max(self.skills.crossing_right),
                                );
                                chosen = Some((
                                    SoccerAction::Pass {
                                        target_player: Some(target),
                                        power: 0.64 + 0.28 * crossing.max(passing_skill),
                                        flight: PassFlight::Aerial,
                                    },
                                    "flank-high-cross".to_string(),
                                ));
                                break;
                            }
                        }
                    }
                    "corner-flag-cross" => {
                        order_names.push("corner-flag-cross".to_string());
                        let cross_chance =
                            action_option_score(&action_options, "corner-flag-cross");
                        if agentic_action_commitment(
                            cross_chance,
                            snapshot.dt_seconds,
                            &observation,
                            self.role,
                        ) {
                            let crossing = ability01(
                                self.skills.crossing_left.max(self.skills.crossing_right),
                            );
                            if let Some(target) = pass_targets.first().copied() {
                                chosen = Some((
                                    SoccerAction::Pass {
                                        target_player: Some(target),
                                        power: 0.62 + 0.28 * crossing.max(passing_skill),
                                        flight: PassFlight::Floor,
                                    },
                                    "corner-flag-cross".to_string(),
                                ));
                                break;
                            } else if let Some(target) = aerial_pass_targets.first().copied() {
                                chosen = Some((
                                    SoccerAction::Pass {
                                        target_player: Some(target),
                                        power: 0.66 + 0.28 * crossing.max(passing_skill),
                                        flight: PassFlight::Aerial,
                                    },
                                    "corner-flag-cross".to_string(),
                                ));
                                break;
                            }
                        }
                    }
                    "killer-pass" => {
                        order_names.push("killer-pass".to_string());
                        let killer_chance = action_option_score(&action_options, "killer-pass");
                        if let Some(target) =
                            snapshot.killer_pass_target_for(self.id, &pass_targets)
                        {
                            if agentic_action_commitment(
                                killer_chance,
                                snapshot.dt_seconds,
                                &observation,
                                self.role,
                            ) {
                                chosen = Some((
                                    SoccerAction::Pass {
                                        target_player: Some(target),
                                        power: 0.66
                                            + 0.24
                                                * passing_skill.max(ability01(self.skills.vision)),
                                        flight: snapshot
                                            .killer_pass_flight_for(self.id, &pass_targets),
                                    },
                                    "killer-pass".to_string(),
                                ));
                                break;
                            }
                        }
                    }
                    "recycle-reset" => {
                        order_names.push("recycle-reset".to_string());
                        let reset_chance = action_option_score(&action_options, "recycle-reset");
                        if let Some(target) =
                            self.recycle_reset_target_for(snapshot, None, &strategic_pass_targets)
                        {
                            if agentic_action_commitment(
                                reset_chance,
                                snapshot.dt_seconds,
                                &observation,
                                self.role,
                            ) {
                                chosen = Some((
                                    SoccerAction::Pass {
                                        target_player: Some(target),
                                        power: 0.46 + 0.22 * passing_skill,
                                        flight: PassFlight::Floor,
                                    },
                                    "recycle-reset".to_string(),
                                ));
                                break;
                            }
                        }
                    }
                    "switch-play" => {
                        order_names.push("switch-play".to_string());
                        let switch_chance = action_option_score(&action_options, "switch-play");
                        if let Some(target) =
                            self.switch_play_target_for(snapshot, None, &strategic_pass_targets)
                        {
                            if agentic_action_commitment(
                                switch_chance,
                                snapshot.dt_seconds,
                                &observation,
                                self.role,
                            ) {
                                chosen = Some((
                                    SoccerAction::Pass {
                                        target_player: Some(target),
                                        power: 0.62
                                            + 0.24
                                                * passing_skill
                                                    .max(ability01(self.skills.flair_passing)),
                                        flight: PassFlight::Floor,
                                    },
                                    "switch-play".to_string(),
                                ));
                                break;
                            }
                        }
                    }
                    "scoop-pass" => {
                        order_names.push("scoop-pass".to_string());
                        let scoop_chance = action_option_score(&action_options, "scoop-pass");
                        if let Some((scoop_target, _)) = scoop_pass_option {
                            if agentic_action_commitment(
                                scoop_chance,
                                snapshot.dt_seconds,
                                &observation,
                                self.role,
                            ) {
                                chosen = Some((
                                    SoccerAction::Pass {
                                        target_player: Some(scoop_target),
                                        power: 0.5,
                                        flight: PassFlight::Scoop,
                                    },
                                    "scoop-pass".to_string(),
                                ));
                                break;
                            }
                        }
                    }
                    "surprise-pass" => {
                        order_names.push("surprise-pass".to_string());
                        let surprise_chance = action_option_score(&action_options, "surprise-pass");
                        if let Some(target) = pass_targets.first().copied() {
                            if agentic_action_commitment(
                                surprise_chance,
                                snapshot.dt_seconds,
                                &observation,
                                self.role,
                            ) {
                                chosen = Some((
                                    SoccerAction::Pass {
                                        target_player: Some(target),
                                        power: 0.50
                                            + 0.22
                                                * passing_skill
                                                    .max(ability01(self.skills.flair_passing)),
                                        flight: PassFlight::Floor,
                                    },
                                    "surprise-pass".to_string(),
                                ));
                                break;
                            }
                        }
                    }
                    "flick-on" => {
                        order_names.push("flick-on".to_string());
                        let flick_chance = action_option_score(&action_options, "flick-on");
                        if let Some(target) = aerial_pass_targets.first().copied() {
                            if agentic_action_commitment(
                                flick_chance,
                                snapshot.dt_seconds,
                                &observation,
                                self.role,
                            ) {
                                chosen = Some((
                                    SoccerAction::Pass {
                                        target_player: Some(target),
                                        power: 0.46
                                            + 0.26
                                                * aerial_duel_skill_from_agent(self)
                                                    .max(ability01(self.skills.first_touch)),
                                        flight: PassFlight::Aerial,
                                    },
                                    "flick-on".to_string(),
                                ));
                                break;
                            }
                        }
                    }
                    "wall-pass" => {
                        order_names.push("wall-pass".to_string());
                        if let Some((plan, wall_appetite, strategy_commits)) = wall_pass_option {
                            if strategy_commits
                                || agentic_action_commitment(
                                    wall_appetite,
                                    snapshot.dt_seconds,
                                    &observation,
                                    self.role,
                                )
                            {
                                chosen = Some((
                                    SoccerAction::Pass {
                                        target_player: Some(plan.wall_partner),
                                        power: WALL_PASS_GIVE_POWER + 0.20 * passing_skill,
                                        flight: PassFlight::Floor,
                                    },
                                    "wall-pass".to_string(),
                                ));
                                self.one_two = Some(OneTwoRun {
                                    wall_partner: plan.wall_partner,
                                    launch_clock_seconds: snapshot.clock_seconds,
                                    return_target: plan.return_target,
                                });
                                break;
                            }
                        }
                    }
                    "wait-for-support" => {
                        order_names.push("wait-for-support".to_string());
                        if let Some(plan) = hold_for_support_option.as_ref() {
                            let chance = action_option_score(&action_options, "wait-for-support");
                            if agentic_action_commitment(
                                chance,
                                snapshot.dt_seconds,
                                &observation,
                                self.role,
                            ) {
                                // Keep the ball while summoned teammates get into position. A wide
                                // attacker executing the byline program keeps MOVING toward the
                                // corner; other carriers retain the controlled body-shield.
                                let kind = if byline_corner_drive {
                                    DribbleMoveKind::CarryForward
                                } else {
                                    DribbleMoveKind::ProtectBall
                                };
                                let touch = snapshot
                                    .deterministic_dribble_touch_decision_for(self.id, kind);
                                let target = snapshot.dribble_move_target_for_touch(
                                    self.id,
                                    self.home_position,
                                    kind,
                                    touch,
                                );
                                // Preserve the original wait start so the bounded max-hold counts
                                // total elapsed time, not just since the latest re-election.
                                let launch_clock_seconds = self
                                    .hold_for_support
                                    .as_ref()
                                    .map(|h| h.launch_clock_seconds)
                                    .unwrap_or(snapshot.clock_seconds);
                                self.hold_for_support = Some(HoldForSupport {
                                    kind: plan.kind,
                                    launch_clock_seconds,
                                    summoned: plan.summoned.clone(),
                                });
                                chosen = Some((
                                    SoccerAction::DribbleMove {
                                        target,
                                        kind,
                                        touch,
                                    },
                                    "wait-for-support".to_string(),
                                ));
                                break;
                            }
                        }
                    }
                    OPEN_PASS_LANE_ACTION_LABEL => {
                        order_names.push(OPEN_PASS_LANE_ACTION_LABEL.to_string());
                        if let Some(plan) = open_pass_lane_plan {
                            let chance =
                                action_option_score(&action_options, OPEN_PASS_LANE_ACTION_LABEL);
                            if agentic_action_commitment(
                                chance,
                                snapshot.dt_seconds,
                                &observation,
                                self.role,
                            ) {
                                chosen = Some((
                                    SoccerAction::DribbleMove {
                                        target: plan.target,
                                        kind: plan.kind,
                                        touch: plan.touch,
                                    },
                                    OPEN_PASS_LANE_ACTION_LABEL.to_string(),
                                ));
                                break;
                            }
                        }
                    }
                    ROUND_GOALKEEPER_ACTION_LABEL => {
                        order_names.push(ROUND_GOALKEEPER_ACTION_LABEL.to_string());
                        if let Some(plan) = round_goalkeeper_plan {
                            let chance =
                                action_option_score(&action_options, ROUND_GOALKEEPER_ACTION_LABEL);
                            if agentic_action_commitment(
                                chance,
                                snapshot.dt_seconds,
                                &observation,
                                self.role,
                            ) {
                                round_goalkeeper_sprint = Some(plan.sprint);
                                chosen = Some((
                                    SoccerAction::DribbleMove {
                                        target: plan.target,
                                        kind: plan.kind,
                                        touch: plan.touch,
                                    },
                                    ROUND_GOALKEEPER_ACTION_LABEL.to_string(),
                                ));
                                break;
                            }
                        }
                    }
                    pass_label if pass_label.starts_with("pass") => {
                        let rank = pass_label
                            .trim_start_matches("pass")
                            .parse::<usize>()
                            .ok()
                            .and_then(|n| n.checked_sub(1))
                            .unwrap_or(0);
                        if rank >= pass_targets.len() {
                            continue;
                        }
                        order_names.push(format!("pass{}", rank + 1));
                        let pass_chance = action_option_score(&action_options, pass_label);
                        if agentic_action_commitment(
                            pass_chance,
                            snapshot.dt_seconds,
                            &observation,
                            self.role,
                        ) {
                            // The primary pass adopts the quick forward (≈5–8 m, open, advanced)
                            // teammate when one qualifies, so the floored "pass sooner" release
                            // actually goes forward to feet. None when the gate is off ⇒ the
                            // ranked target is used unchanged.
                            let target = if rank == 0 {
                                observation
                                    .quick_forward_pass_target
                                    .filter(|id| pass_targets.contains(id))
                                    .unwrap_or(pass_targets[rank])
                            } else {
                                pass_targets[rank]
                            };
                            chosen = Some((
                                SoccerAction::Pass {
                                    target_player: Some(target),
                                    power: 0.58 + 0.32 * passing_skill,
                                    flight: PassFlight::Floor,
                                },
                                format!("pass{}", rank + 1),
                            ));
                            break;
                        }
                    }
                    pass_label if pass_label.starts_with("aerial-pass") => {
                        let rank = pass_label
                            .trim_start_matches("aerial-pass")
                            .parse::<usize>()
                            .ok()
                            .and_then(|n| n.checked_sub(1))
                            .unwrap_or(0);
                        if rank >= aerial_pass_targets.len() {
                            continue;
                        }
                        order_names.push(format!("aerial-pass{}", rank + 1));
                        let crossing =
                            ability01(self.skills.crossing_left.max(self.skills.crossing_right));
                        let pass_chance = action_option_score(&action_options, pass_label);
                        if agentic_action_commitment(
                            pass_chance,
                            snapshot.dt_seconds,
                            &observation,
                            self.role,
                        ) {
                            let target = aerial_pass_targets[rank];
                            chosen = Some((
                                SoccerAction::Pass {
                                    target_player: Some(target),
                                    power: 0.56 + 0.28 * crossing.max(passing_skill),
                                    flight: PassFlight::Aerial,
                                },
                                format!("aerial-pass{}", rank + 1),
                            ));
                            break;
                        }
                    }
                    _ => {}
                }
            }
            for op in full_ops {
                if !order_names.iter().any(|entry| entry == &op) {
                    order_names.push(op);
                }
            }

            let forced_killer_pass_target =
                if killer_pass_forced_by_goal_pressure(&observation, self.role) {
                    snapshot.killer_pass_target_for(self.id, &pass_targets)
                } else {
                    None
                };
            let (mut action, mut action_label) = chosen.unwrap_or_else(|| {
                let carry_to_create = shot_creation_carry_multiplier(&observation) > 1.28
                    && observation.forward_dribble_space_yards > 2.2;
                let patient_carry = low_pressure_patient_carry_preferred(&observation);
                let clearance_score = action_option_score(&action_options, "clearance");
                let clearance_safety_valve =
                    matches!(self.role, PlayerRole::Goalkeeper | PlayerRole::Defender)
                        && clearance_score >= 0.46
                        && observation.yards_to_own_goal < observation.yards_to_goal
                        && observation
                            .pressure_urgency
                            .max(observation.defensive_urgency)
                            .max(observation.immediate_dispossession_risk)
                            .max(observation.excessive_hold_pressure)
                            >= 0.42;
                if clearance_safety_valve {
                    (
                        SoccerAction::Clearance {
                            target: snapshot
                                .pressure_clearance_target_for(self.id)
                                .unwrap_or_else(|| {
                                    clearance_target_for_player(
                                        self.team,
                                        self.position,
                                        snapshot.field_width,
                                        snapshot.field_length,
                                    )
                                }),
                            power: 0.88 + observation.perceived_pressure.clamp(0.0, 1.0) * 0.10,
                        },
                        "clearance".to_string(),
                    )
                } else if let Some(target) = forced_killer_pass_target.filter(|_| {
                    threaded_goal_pass_can_override_forced_shot(&observation, self.role)
                        || goal_attack_shot_can_yield_to_support(&observation, self.role)
                }) {
                    (
                        SoccerAction::Pass {
                            target_player: Some(target),
                            power: 0.66 + 0.24 * passing_skill.max(ability01(self.skills.vision)),
                            flight: snapshot.killer_pass_flight_for(self.id, &pass_targets),
                        },
                        "killer-pass".to_string(),
                    )
                } else if goal_attack_shot_blocks_alternatives(&observation, self.role)
                    || striker_shot_window_is_qualified(&observation, self.role)
                {
                    (
                        SoccerAction::Shoot {
                            power: shot_power_for_skill(shooting_skill),
                        },
                        "shoot".to_string(),
                    )
                } else if let Some(target) = forced_killer_pass_target {
                    (
                        SoccerAction::Pass {
                            target_player: Some(target),
                            power: 0.66 + 0.24 * passing_skill.max(ability01(self.skills.vision)),
                            flight: snapshot.killer_pass_flight_for(self.id, &pass_targets),
                        },
                        "killer-pass".to_string(),
                    )
                } else if let Some(target) = hold_up_flank_target {
                    let kind = snapshot.side_step_dribble_kind_for(self.id);
                    let bucket = match dribble_final_cut_kind(kind) {
                        DribbleMoveKind::CarryForward | DribbleMoveKind::ProtectBall => 0,
                        DribbleMoveKind::CarryOutLeft | DribbleMoveKind::LeftCut => 9,
                        DribbleMoveKind::CarryOutRight | DribbleMoveKind::RightCut => 3,
                        DribbleMoveKind::Nutmeg => 0,
                        DribbleMoveKind::XaviTurn => 10,
                        DribbleMoveKind::FakeLeftCutRight | DribbleMoveKind::FakeRightCutLeft => 0,
                    };
                    (
                        SoccerAction::DribbleMove {
                            target,
                            kind,
                            touch: DribbleTouchDecision::new(bucket, 1.35),
                        },
                        "hold-up-flank".to_string(),
                    )
                } else if patient_carry || carry_to_create || pass_targets.is_empty() {
                    let kind = if goal_approach_forces_goalmouth_carry(&observation, self.role) {
                        DribbleMoveKind::CarryForward
                    } else {
                        self.agentic_fallback_dribble_kind(
                            snapshot,
                            &observation,
                            patient_carry,
                            carry_to_create,
                        )
                    };
                    let touch = snapshot.deterministic_dribble_touch_decision_for(self.id, kind);
                    (
                        SoccerAction::DribbleMove {
                            target: snapshot.dribble_move_target_for_touch(
                                self.id,
                                self.home_position,
                                kind,
                                touch,
                            ),
                            kind,
                            touch,
                        },
                        kind.label().to_string(),
                    )
                } else if let Some(plan) = open_pass_lane_plan {
                    (
                        SoccerAction::DribbleMove {
                            target: plan.target,
                            kind: plan.kind,
                            touch: plan.touch,
                        },
                        OPEN_PASS_LANE_ACTION_LABEL.to_string(),
                    )
                } else if let Some(target) = pass_targets.first() {
                    (
                        SoccerAction::Pass {
                            target_player: Some(*target),
                            power: 0.58 + 0.32 * passing_skill,
                            flight: PassFlight::Floor,
                        },
                        "pass1".to_string(),
                    )
                } else {
                    let kind =
                        self.agentic_fallback_dribble_kind(snapshot, &observation, false, false);
                    let touch = snapshot.deterministic_dribble_touch_decision_for(self.id, kind);
                    (
                        SoccerAction::DribbleMove {
                            target: snapshot.dribble_move_target_for_touch(
                                self.id,
                                self.home_position,
                                kind,
                                touch,
                            ),
                            kind,
                            touch,
                        },
                        kind.label().to_string(),
                    )
                }
            });
            if mpc_reselects_candidate(snapshot, self, &action, &action_label) {
                let rejected_label = normalize_soccer_action_label(&action_label).to_string();
                order_names.push(format!("mpc-reselect:{}", action_label));
                if rejected_label == "wall-pass" {
                    self.one_two = None;
                }
                let fallback = action_options
                    .iter()
                    .filter(|option| option.legal && option.score > 0.0)
                    .filter(|option| {
                        let label = normalize_soccer_action_label(&option.label);
                        label != rejected_label && label != "wall-pass"
                    })
                    .filter_map(|option| {
                        let label = if option.label == "scoop-pass" {
                            "scoop-pass"
                        } else {
                            normalize_soccer_action_label(&option.label)
                        };
                        let candidate = match label {
                            "shoot"
                                if shot_decision_is_qualified_for_role(&observation, self.role)
                                    || speculative_long_shot_is_qualified(
                                        &observation,
                                        self.role,
                                        shooting_skill,
                                    ) =>
                            {
                                Some((
                                    SoccerAction::Shoot {
                                        power: shot_power_for_skill(shooting_skill),
                                    },
                                    "shoot".to_string(),
                                ))
                            }
                            "pass" => {
                                let rank = option
                                    .label
                                    .trim_start_matches("pass")
                                    .parse::<usize>()
                                    .ok()
                                    .and_then(|n| n.checked_sub(1))
                                    .unwrap_or(0);
                                pass_targets.get(rank).copied().map(|target| {
                                    (
                                        SoccerAction::Pass {
                                            target_player: Some(target),
                                            power: 0.58 + 0.32 * passing_skill,
                                            flight: PassFlight::Floor,
                                        },
                                        format!("pass{}", rank + 1),
                                    )
                                })
                            }
                            "aerial-pass" => {
                                let rank = option
                                    .label
                                    .trim_start_matches("aerial-pass")
                                    .parse::<usize>()
                                    .ok()
                                    .and_then(|n| n.checked_sub(1))
                                    .unwrap_or(0);
                                aerial_pass_targets.get(rank).copied().map(|target| {
                                    let crossing = ability01(
                                        self.skills.crossing_left.max(self.skills.crossing_right),
                                    );
                                    (
                                        SoccerAction::Pass {
                                            target_player: Some(target),
                                            power: 0.56 + 0.28 * crossing.max(passing_skill),
                                            flight: PassFlight::Aerial,
                                        },
                                        format!("aerial-pass{}", rank + 1),
                                    )
                                })
                            }
                            "killer-pass" => snapshot
                                .killer_pass_target_for(self.id, &pass_targets)
                                .map(|target| {
                                    (
                                        SoccerAction::Pass {
                                            target_player: Some(target),
                                            power: 0.66
                                                + 0.24
                                                    * passing_skill
                                                        .max(ability01(self.skills.vision)),
                                            flight: snapshot
                                                .killer_pass_flight_for(self.id, &pass_targets),
                                        },
                                        "killer-pass".to_string(),
                                    )
                                }),
                            "recycle-reset" => self
                                .recycle_reset_target_for(snapshot, None, &strategic_pass_targets)
                                .map(|target| {
                                    (
                                        SoccerAction::Pass {
                                            target_player: Some(target),
                                            power: 0.46 + 0.22 * passing_skill,
                                            flight: PassFlight::Floor,
                                        },
                                        "recycle-reset".to_string(),
                                    )
                                }),
                            "switch-play" => self
                                .switch_play_target_for(snapshot, None, &strategic_pass_targets)
                                .map(|target| {
                                    (
                                        SoccerAction::Pass {
                                            target_player: Some(target),
                                            power: 0.62
                                                + 0.24
                                                    * passing_skill
                                                        .max(ability01(self.skills.flair_passing)),
                                            flight: PassFlight::Floor,
                                        },
                                        "switch-play".to_string(),
                                    )
                                }),
                            "flank-low-cross" => pass_targets.first().copied().map(|target| {
                                let crossing = ability01(
                                    self.skills.crossing_left.max(self.skills.crossing_right),
                                );
                                (
                                    SoccerAction::Pass {
                                        target_player: Some(target),
                                        power: 0.60 + 0.26 * crossing.max(passing_skill),
                                        flight: PassFlight::Floor,
                                    },
                                    "flank-low-cross".to_string(),
                                )
                            }),
                            "flank-high-cross" => {
                                aerial_pass_targets.first().copied().map(|target| {
                                    let crossing = ability01(
                                        self.skills.crossing_left.max(self.skills.crossing_right),
                                    );
                                    (
                                        SoccerAction::Pass {
                                            target_player: Some(target),
                                            power: 0.64 + 0.28 * crossing.max(passing_skill),
                                            flight: PassFlight::Aerial,
                                        },
                                        "flank-high-cross".to_string(),
                                    )
                                })
                            }
                            "corner-flag-cross" => pass_targets
                                .first()
                                .copied()
                                .map(|target| {
                                    let crossing = ability01(
                                        self.skills.crossing_left.max(self.skills.crossing_right),
                                    );
                                    (
                                        SoccerAction::Pass {
                                            target_player: Some(target),
                                            power: 0.62 + 0.28 * crossing.max(passing_skill),
                                            flight: PassFlight::Floor,
                                        },
                                        "corner-flag-cross".to_string(),
                                    )
                                })
                                .or_else(|| {
                                    aerial_pass_targets.first().copied().map(|target| {
                                        let crossing = ability01(
                                            self.skills
                                                .crossing_left
                                                .max(self.skills.crossing_right),
                                        );
                                        (
                                            SoccerAction::Pass {
                                                target_player: Some(target),
                                                power: 0.66 + 0.28 * crossing.max(passing_skill),
                                                flight: PassFlight::Aerial,
                                            },
                                            "corner-flag-cross".to_string(),
                                        )
                                    })
                                }),
                            "scoop-pass" => scoop_pass_option.map(|(target, _)| {
                                (
                                    SoccerAction::Pass {
                                        target_player: Some(target),
                                        power: 0.5,
                                        flight: PassFlight::Scoop,
                                    },
                                    "scoop-pass".to_string(),
                                )
                            }),
                            "surprise-pass" => pass_targets.first().copied().map(|target| {
                                (
                                    SoccerAction::Pass {
                                        target_player: Some(target),
                                        power: 0.50
                                            + 0.22
                                                * passing_skill
                                                    .max(ability01(self.skills.flair_passing)),
                                        flight: PassFlight::Floor,
                                    },
                                    "surprise-pass".to_string(),
                                )
                            }),
                            "flick-on" => aerial_pass_targets.first().copied().map(|target| {
                                (
                                    SoccerAction::Pass {
                                        target_player: Some(target),
                                        power: 0.46
                                            + 0.26
                                                * aerial_duel_skill_from_agent(self)
                                                    .max(ability01(self.skills.first_touch)),
                                        flight: PassFlight::Aerial,
                                    },
                                    "flick-on".to_string(),
                                )
                            }),
                            "vertical-attack" | "turnover-burst" => {
                                let kind = DribbleMoveKind::CarryForward;
                                let touch = snapshot
                                    .deterministic_dribble_touch_decision_for(self.id, kind);
                                Some((
                                    SoccerAction::DribbleMove {
                                        target: snapshot.dribble_move_target_for_touch(
                                            self.id,
                                            self.home_position,
                                            kind,
                                            touch,
                                        ),
                                        kind,
                                        touch,
                                    },
                                    label.to_string(),
                                ))
                            }
                            OPEN_PASS_LANE_ACTION_LABEL => open_pass_lane_plan.map(|plan| {
                                (
                                    SoccerAction::DribbleMove {
                                        target: plan.target,
                                        kind: plan.kind,
                                        touch: plan.touch,
                                    },
                                    OPEN_PASS_LANE_ACTION_LABEL.to_string(),
                                )
                            }),
                            ROUND_GOALKEEPER_ACTION_LABEL => round_goalkeeper_plan.map(|plan| {
                                (
                                    SoccerAction::DribbleMove {
                                        target: plan.target,
                                        kind: plan.kind,
                                        touch: plan.touch,
                                    },
                                    ROUND_GOALKEEPER_ACTION_LABEL.to_string(),
                                )
                            }),
                            "dribble" => {
                                let kind = self.agentic_fallback_dribble_kind(
                                    snapshot,
                                    &observation,
                                    false,
                                    true,
                                );
                                let touch = snapshot
                                    .deterministic_dribble_touch_decision_for(self.id, kind);
                                Some((
                                    SoccerAction::DribbleMove {
                                        target: snapshot.dribble_move_target_for_touch(
                                            self.id,
                                            self.home_position,
                                            kind,
                                            touch,
                                        ),
                                        kind,
                                        touch,
                                    },
                                    kind.label().to_string(),
                                ))
                            }
                            "carry-forward" | "carry-out-left" | "carry-out-right"
                            | "protect-ball" | "xavi-turn" => {
                                let kind = match label {
                                    "carry-forward" => DribbleMoveKind::CarryForward,
                                    "carry-out-left" => DribbleMoveKind::CarryOutLeft,
                                    "carry-out-right" => DribbleMoveKind::CarryOutRight,
                                    "xavi-turn" => DribbleMoveKind::XaviTurn,
                                    _ => DribbleMoveKind::ProtectBall,
                                };
                                let kind =
                                    goal_approach_dribble_kind(kind, &observation, self.role);
                                let touch = snapshot
                                    .deterministic_dribble_touch_decision_for(self.id, kind);
                                Some((
                                    SoccerAction::DribbleMove {
                                        target: snapshot.dribble_move_target_for_touch(
                                            self.id,
                                            self.home_position,
                                            kind,
                                            touch,
                                        ),
                                        kind,
                                        touch,
                                    },
                                    kind.label().to_string(),
                                ))
                            }
                            "clearance" => Some((
                                SoccerAction::Clearance {
                                    target: snapshot
                                        .pressure_clearance_target_for(self.id)
                                        .unwrap_or_else(|| {
                                            clearance_target_for_player(
                                                self.team,
                                                self.position,
                                                snapshot.field_width,
                                                snapshot.field_length,
                                            )
                                        }),
                                    power: 0.88
                                        + observation.perceived_pressure.clamp(0.0, 1.0) * 0.10,
                                },
                                "clearance".to_string(),
                            )),
                            "route-one" => Some((
                                SoccerAction::RouteOne {
                                    target: snapshot.route_one_target_for(self.id).unwrap_or_else(
                                        || {
                                            route_one_target_for_actor(
                                                self.team,
                                                self.position,
                                                snapshot.field_width,
                                                snapshot.field_length,
                                                self.role,
                                            )
                                        },
                                    ),
                                    power: 0.80,
                                },
                                "route-one".to_string(),
                            )),
                            _ => None,
                        };
                        candidate
                            .filter(|(candidate_action, candidate_label)| {
                                !mpc_reselects_candidate(
                                    snapshot,
                                    self,
                                    candidate_action,
                                    candidate_label,
                                )
                            })
                            .map(|candidate| (option.score, candidate))
                    })
                    .max_by(|a, b| a.0.total_cmp(&b.0))
                    .map(|(_, candidate)| candidate);
                let (fallback_action, fallback_label) = fallback.unwrap_or_else(|| {
                    let kind = DribbleMoveKind::ProtectBall;
                    let touch = snapshot.deterministic_dribble_touch_decision_for(self.id, kind);
                    (
                        SoccerAction::DribbleMove {
                            target: snapshot.dribble_move_target_for_touch(
                                self.id,
                                self.home_position,
                                kind,
                                touch,
                            ),
                            kind,
                            touch,
                        },
                        "protect-ball".to_string(),
                    )
                });
                action = fallback_action;
                action_label = fallback_label;
            }
            // Each special carry decides sprint-vs-run on its own terms; every other carry uses
            // the generic role/urgency rule. This unifies both convergent open-lane features:
            // ours ("open-passing-lane") sets `open_lane_sprint` in its arm (very high pressure /
            // fast-tracking opponent → burst, else a controlled run to conserve energy), and
            // theirs ("open-pass-lane", `OPEN_PASS_LANE_ACTION_LABEL`) decides via
            // `open_pass_lane_sprint_for_action`. The round-goalkeeper carry keeps its own
            // plan-specific burst decision (pressure/space decides glide-around vs explode-past).
            let is_dribble_action = matches!(action, SoccerAction::DribbleMove { .. });
            let sprint = if runaround_sprint {
                true
            } else if !is_dribble_action {
                false
            } else if let Some(open_lane_sprint) = open_lane_sprint {
                open_lane_sprint
            } else if let Some(round_goalkeeper_sprint) = round_goalkeeper_sprint {
                round_goalkeeper_sprint
            } else if action_label == OPEN_PASS_LANE_ACTION_LABEL {
                open_pass_lane_sprint_for_action(
                    snapshot,
                    self,
                    &observation,
                    &action_label,
                    &action,
                )
            } else {
                self.role == PlayerRole::Forward
                    || observation.forward_dribble_space_yards > 3.0
                    || observation.perceived_pressure > 0.35
                    || observation.decision_urgency > 0.46
                    || observation.offensive_urgency > 0.34
            };
            self.last_decision = Some(self.decision_trace(
                snapshot,
                mdp_state,
                observation,
                belief,
                order_names,
                action_options,
                &action,
                action_label,
            ));
            return PlayerIntent {
                player_id: self.id,
                action,
                sprint,
            };
        }

        let possession_team = snapshot.controlled_possession_team();
        let mut order_names = Vec::new();
        let action_options: Vec<AgentActionOptionTrace>;
        // Mechanism 5: set when the off-ball target was lifted upfield far enough to
        // warrant a sprint (run, not jog) into the space ahead. Read by the sprint
        // resolver below for the "defend"/"hold" off-ball moves.
        let mut push_up_sprint = false;
        // Set when recovering a loose ball that got away under pressure (e.g. the
        // player's own long/heavy touch with an opponent bearing down) — read by the
        // sprint resolver below so the chase is a sprint, not a jog.
        let mut loose_ball_pressured_sprint = false;
        let (action, action_label) = if let Some(burst_target) = (possession_team
            == Some(self.team))
        .then(|| snapshot.one_two_run_target_for(self.id))
        .flatten()
        {
            // THE "GO": this player just played the give in a one-two and is bursting
            // onside past his man to receive the first-time return. This threaded run
            // overrides ordinary support shape and is sprinted (see the sprint resolver).
            action_options = single_action_option("one-two-run");
            order_names.push("one-two-run".to_string());
            (
                SoccerAction::MoveTo(burst_target),
                "one-two-run".to_string(),
            )
        } else if possession_team == Some(self.team) {
            let support_context = self.support_action_context(snapshot);
            action_options = support_context.options.clone();
            let mut support_weighted_ops = action_options
                .iter()
                .map(|option| {
                    (
                        option.label.clone(),
                        if option.legal { option.score } else { 0.0 },
                    )
                })
                .collect::<Vec<_>>();
            let cadence_hold_label = decision_cadence_hold_label_from_weights(
                &observation,
                &support_weighted_ops,
                snapshot.tick,
            );
            if let Some(label) = cadence_hold_label.as_deref() {
                apply_decision_cadence_hold_weights(&mut support_weighted_ops, label);
                order_names.push(format!("decision-cadence-hold:{label}"));
            }
            let support_order = {
                let (ordered, ordered_scores) =
                    agentic_action_order_scored(support_weighted_ops);
                let sampled = reorder_with_scores_traced(
                    ordered,
                    &ordered_scores,
                    snapshot.decision_seed,
                    self.id,
                    snapshot.tick,
                    site::SUPPORT,
                );
                self.pending_policy_behavior_probability = sampled.behavior_probability;
                sampled.ordered
            };
            let first_support_label = support_order
                .first()
                .map(|label| normalize_soccer_action_label(label.as_str()))
                .unwrap_or("support-shape")
                .to_string();
            order_names.extend(support_order);
            let support_open = support_context.open;
            let guarded_support_special = |point: Vec2| {
                snapshot.shape_guarded_support_point(
                    self.id,
                    point,
                    &[support_open, self.home_position],
                    self.home_position,
                    false,
                )
            };
            let support_target =
                match first_support_label.as_str() {
                    "check-to-ball" => support_context.special_targets.check_to_ball.map(|point| {
                        SupportMovementTarget {
                            point: guarded_support_special(point),
                            action_label: "check-to-ball",
                        }
                    }),
                    "run-in-behind" => support_context.special_targets.in_behind.map(|point| {
                        SupportMovementTarget {
                            point: guarded_support_special(point),
                            action_label: "run-in-behind",
                        }
                    }),
                    "exploit-space-run" => {
                        support_context.special_targets.exploit_space.map(|point| {
                            SupportMovementTarget {
                                point: guarded_support_special(point),
                                action_label: "exploit-space-run",
                            }
                        })
                    }
                    "wide-outlet" => support_context.special_targets.wide_outlet.map(|point| {
                        SupportMovementTarget {
                            point: guarded_support_special(point),
                            action_label: "wide-outlet",
                        }
                    }),
                    "overlap-run" => {
                        let mut target = snapshot.attacking_support_movement_for_with_targets(
                            self.id,
                            self.home_position,
                            true,
                            &support_context.special_targets,
                        );
                        target.action_label = "overlap-run";
                        Some(target)
                    }
                    "vacate-space" => {
                        let mut target = snapshot.attacking_support_movement_for_with_targets(
                            self.id,
                            self.home_position,
                            true,
                            &support_context.special_targets,
                        );
                        target.action_label = "vacate-space";
                        Some(target)
                    }
                    "lane-yield" => snapshot
                        .pass_lane_yield_target_for(self.id, self.home_position)
                        .map(|(point, _)| SupportMovementTarget {
                            point: guarded_support_special(point),
                            action_label: "lane-yield",
                        }),
                    "support-roam" => Some(snapshot.attacking_support_movement_for_with_targets(
                        self.id,
                        self.home_position,
                        true,
                        &support_context.special_targets,
                    )),
                    _ => Some(snapshot.attacking_support_movement_for_with_targets(
                        self.id,
                        self.home_position,
                        false,
                        &support_context.special_targets,
                    )),
                }
                .unwrap_or_else(|| {
                    snapshot.attacking_support_movement_for_with_targets(
                        self.id,
                        self.home_position,
                        matches!(
                            first_support_label.as_str(),
                            "support-roam" | "overlap-run" | "vacate-space"
                        ),
                        &support_context.special_targets,
                    )
                });
            let support_target = if self.role != PlayerRole::Goalkeeper
                && matches!(
                    first_support_label.as_str(),
                    "support-shape" | "support-roam"
                ) {
                snapshot
                    .formation_lp_guidance_for(self.id)
                    .map(|guidance| SupportMovementTarget {
                        point: snapshot.clamp_to_role_position(
                            self.id,
                            guidance.target,
                            self.home_position,
                            false,
                        ),
                        action_label: support_target.action_label,
                    })
                    .unwrap_or(support_target)
            } else {
                support_target
            };
            (
                SoccerAction::MoveTo(support_target.point),
                support_target.action_label.to_string(),
            )
        } else if possession_team == Some(self.team.other()) {
            action_options =
                self.defensive_action_options(snapshot, &directive, snapshot.dt_seconds);
            let mut defensive_weighted_ops = vec![
                (
                    "tackle".to_string(),
                    action_option_score(&action_options, "tackle"),
                ),
                (
                    "defend-shape".to_string(),
                    action_option_score(&action_options, "defend-shape"),
                ),
                (
                    "defend-roam".to_string(),
                    action_option_score(&action_options, "defend-roam"),
                ),
                (
                    "press-cover".to_string(),
                    action_option_score(&action_options, "press-cover"),
                ),
            ];
            let cadence_hold_label = decision_cadence_hold_label_from_weights(
                &observation,
                &defensive_weighted_ops,
                snapshot.tick,
            );
            if let Some(label) = cadence_hold_label.as_deref() {
                apply_decision_cadence_hold_weights(&mut defensive_weighted_ops, label);
                order_names.push(format!("decision-cadence-hold:{label}"));
            }
            let ops = {
                let (ordered, ordered_scores) =
                    agentic_action_order_scored(defensive_weighted_ops);
                let sampled = reorder_with_scores_traced(
                    ordered,
                    &ordered_scores,
                    snapshot.decision_seed,
                    self.id,
                    snapshot.tick,
                    site::DEFENSIVE,
                );
                self.pending_policy_behavior_probability = sampled.behavior_probability;
                sampled.ordered
            };
            let mut chosen = None;
            for op in ops {
                match op.as_str() {
                    "tackle" => {
                        order_names.push("tackle".to_string());
                        if let Some(holder) = snapshot.ball.holder {
                            let holder_is_opponent = snapshot
                                .players
                                .iter()
                                .find(|p| p.id == holder)
                                .is_some_and(|p| p.team == self.team.other());
                            let tackle_commitment = ((defending_skill * 0.6
                                + aggression_skill * 0.4)
                                * directive.press_intensity
                                * snapshot.advancing_carrier_steal_urgency(self.id))
                            .clamp(0.02, 0.97);
                            if holder_is_opponent
                                && self.position.distance(snapshot.ball.position) < 3.1
                                && (self.role != PlayerRole::Goalkeeper
                                    || snapshot.goalkeeper_direct_intervention_is_safe(self.id))
                                && agentic_action_commitment(
                                    tackle_commitment,
                                    snapshot.dt_seconds,
                                    &observation,
                                    self.role,
                                )
                            {
                                chosen = Some((
                                    SoccerAction::Tackle {
                                        target_player: holder,
                                    },
                                    "tackle".to_string(),
                                ));
                                break;
                            }
                        }
                    }
                    "defend-shape" | "defend-roam" | "press-cover" => {
                        let press_cover = op.as_str() == "press-cover";
                        let roam = op.as_str() == "defend-roam" || press_cover;
                        order_names.push(op.clone());
                        let dist = self.position.distance(snapshot.ball.position);
                        let defend_radius = 3.0 + directive.press_intensity * 3.0;
                        let target = if self.role == PlayerRole::Goalkeeper {
                            snapshot.defensive_assignment_for(self.id, self.home_position, roam)
                        } else if press_cover {
                            snapshot.defensive_assignment_for(self.id, self.home_position, true)
                        } else if roam && dist < defend_radius {
                            snapshot.goal_side_defensive_target_for(self.id, snapshot.ball.position)
                        } else {
                            let target = snapshot
                                .formation_lp_guidance_for(self.id)
                                .map(|guidance| guidance.target)
                                .unwrap_or_else(|| {
                                    snapshot.defensive_assignment_for(
                                        self.id,
                                        self.home_position,
                                        roam,
                                    )
                                });
                            if self.role == PlayerRole::Defender {
                                if let Some((holder_position, line_gap)) =
                                    snapshot.opponent_breakthrough_ball_carrier(self.team)
                                {
                                    snapshot.clamp_defensive_line_break_retreat(
                                        self.team,
                                        holder_position,
                                        line_gap,
                                        target,
                                    )
                                } else {
                                    target
                                }
                            } else {
                                target
                            }
                        };
                        let target = snapshot.goal_side_defensive_target_for(self.id, target);
                        // Push up to fill the space ahead when the ball is far upfield.
                        let (target, push_up) =
                            snapshot.defensive_push_up_adjustment(self.id, target);
                        push_up_sprint = push_up;
                        let label = if press_cover { "press-cover" } else { "defend" };
                        chosen = Some((SoccerAction::MoveTo(target), label.to_string()));
                        break;
                    }
                    _ => {}
                }
            }
            chosen.unwrap_or_else(|| {
                (
                    SoccerAction::MoveTo(snapshot.defensive_assignment_for(
                        self.id,
                        self.home_position,
                        false,
                    )),
                    "defend".to_string(),
                )
            })
        } else {
            let loose_ball_target = snapshot.loose_ball_recovery_target_for(self.id);
            let my_distance = self.position.distance(loose_ball_target);
            let my_distance_sq =
                (self.position - loose_ball_target).dot(self.position - loose_ball_target);
            let fifty_fifty_duel = loose_ball_fifty_fifty_duel_for(snapshot, self.id);
            let closer_teammates = snapshot
                .players
                .iter()
                .filter(|player| player.team == self.team && player.id != self.id)
                .filter(|player| {
                    let delta = player.position - loose_ball_target;
                    delta.dot(delta) < my_distance_sq
                })
                .count();
            // The keeper only leaves its box for a loose ball when near-certain to win it:
            // clearly first ahead of any opponent AND beating its own nearest teammate to
            // it by more than 0.5s — otherwise it holds its line and lets the defender deal
            // with it. Inside its own box it always claims.
            let goalkeeper_can_recover = self.role != PlayerRole::Goalkeeper
                || snapshot.goalkeeper_should_commit_to_loose_ball(self.id, loose_ball_target);
            let loose_options = self.loose_ball_action_options(
                my_distance,
                closer_teammates,
                goalkeeper_can_recover,
                fifty_fifty_duel,
                snapshot
                    .loose_ball_attack_profile_for(self.id)
                    .attack_candidate
                    >= 0.65,
            );
            let loose_order = {
                let (ordered, ordered_scores) = agentic_action_order_scored(
                    loose_options
                        .iter()
                        .map(|option| {
                            (
                                option.label.clone(),
                                if option.legal { option.score } else { 0.0 },
                            )
                        })
                        .collect(),
                );
                let sampled = reorder_with_scores_traced(
                    ordered,
                    &ordered_scores,
                    snapshot.decision_seed,
                    self.id,
                    snapshot.tick,
                    site::LOOSE_BALL,
                );
                self.pending_policy_behavior_probability = sampled.behavior_probability;
                sampled.ordered
            };
            // A defender genuinely placed to cut an in-flight ground pass on its lane ALWAYS
            // steps to intercept — it is not subject to the anti-swarm "recover" legality cap
            // (closer-teammates), which otherwise let a ground pass roll right past a defender
            // 1–3yd off the lane while it deferred to a team-mate and held shape. `loose_ball_target`
            // already resolves to the lane cut-point for such a player.
            let lane_interceptor = self.role != PlayerRole::Goalkeeper
                && snapshot
                    .on_path_ball_intercept_target_for(self.id)
                    .is_some();
            let should_recover = lane_interceptor
                || fifty_fifty_duel
                || snapshot
                    .loose_ball_attack_profile_for(self.id)
                    .attack_candidate
                    >= 0.65
                || action_option_score(&loose_options, ACTION_LABEL_RECOVER) > 0.0;
            if should_recover {
                action_options = loose_options;
                order_names.extend(loose_order);
                let movement_target =
                    self.pomdp_mpc_loose_ball_recovery_target(snapshot, loose_ball_target);
                // Win the race back to a ball that got away under pressure (notably
                // the carrier's own long/heavy touch): if it's meaningfully ahead and
                // an opponent is contesting / closing on it, sprint to it.
                let nearest_opponent_to_ball = snapshot
                    .players
                    .iter()
                    .filter(|player| {
                        player.team != self.team && player.role != PlayerRole::Goalkeeper
                    })
                    .map(|player| player.position.distance(movement_target))
                    .fold(f64::INFINITY, f64::min);
                loose_ball_pressured_sprint = self.role != PlayerRole::Goalkeeper
                    && self.position.distance(movement_target)
                        > LOOSE_BALL_PRESSURED_SPRINT_MIN_DISTANCE_YARDS
                    && (fifty_fifty_duel
                        || nearest_opponent_to_ball
                            <= LOOSE_BALL_PRESSURED_SPRINT_OPPONENT_RADIUS_YARDS);
                (
                    SoccerAction::MoveTo(movement_target),
                    ACTION_LABEL_RECOVER.to_string(),
                )
            } else if let Some((support_target, sprint)) =
                snapshot.aerial_attacking_support_intent_for(self.id, self.home_position)
            {
                action_options = single_action_option("support-push-up");
                order_names.push("support-push-up".to_string());
                push_up_sprint = sprint;
                (
                    SoccerAction::MoveTo(support_target),
                    "support-push-up".to_string(),
                )
            } else {
                action_options = loose_options;
                order_names.extend(loose_order);
                let shape = snapshot.defensive_shape_for(self.id, self.home_position);
                // Push up to fill the space ahead when the ball is far upfield.
                let (shape, push_up) = snapshot.defensive_push_up_adjustment(self.id, shape);
                push_up_sprint = push_up;
                (SoccerAction::MoveTo(shape), "hold".to_string())
            }
        };

        self.last_decision = Some(self.decision_trace(
            snapshot,
            mdp_state,
            observation,
            belief,
            order_names,
            action_options,
            &action,
            action_label.clone(),
        ));
        let sprint = match &action {
            SoccerAction::MoveTo(target) if is_attacking_support_action_label(&action_label) => {
                if self.role == PlayerRole::Goalkeeper {
                    snapshot.goalkeeper_line_recovery_sprint_active(self.id)
                        && self.position.distance(*target)
                            > GOALKEEPER_LINE_RECOVERY_SPRINT_DISTANCE_YARDS
                } else {
                    // Energy conservation: when the anaerobic reserve is low, only the PEAK
                    // attacking moments (a run in behind, urgent support for a pressured
                    // holder, or a teammate carrying the ball at goal) are worth a sprint —
                    // routine shape/support runs jog to save the reserve for when it matters.
                    let peak_moment = holder_pressure_support_urgency(snapshot, self.team)
                        >= PRESSURED_SUPPORT_SPRINT_URGENCY
                        || matches!(
                            normalize_soccer_action_label(&action_label),
                            "run-in-behind" | "exploit-space-run" | "one-two-run"
                        )
                        || snapshot
                            .attacking_carrier_driving_at_goal(self.team)
                            .is_some();
                    let conserving = self.fatigue > SPRINT_CONSERVE_FATIGUE && !peak_moment;
                    !conserving
                        && (snapshot.attacking_support_sprint_active(self.team)
                            || holder_pressure_support_urgency(snapshot, self.team)
                                >= PRESSURED_SUPPORT_SPRINT_URGENCY
                            || matches!(
                                normalize_soccer_action_label(&action_label),
                                "check-to-ball"
                                    | "wide-outlet"
                                    | "run-in-behind"
                                    | "exploit-space-run"
                                    | "shot-creation-run"
                                    | "overlap-run"
                                    | "support-push-up"
                                    | "vertical-attack"
                                    | "vacate-space"
                                    | "one-two-run"
                            ))
                        && matches!(self.role, PlayerRole::Midfielder | PlayerRole::Forward)
                        && self.position.distance(*target) > 3.5
                }
            }
            SoccerAction::MoveTo(target) if action_label == "defend" => {
                if self.role == PlayerRole::Goalkeeper {
                    snapshot.goalkeeper_line_recovery_sprint_active(self.id)
                        && self.position.distance(*target)
                            > GOALKEEPER_LINE_RECOVERY_SPRINT_DISTANCE_YARDS
                } else {
                    // Sprint to track a threat OR to proactively fill the space ahead
                    // when the line is pushing up (mechanism 5) — run, not jog.
                    push_up_sprint
                        || (snapshot.defensive_tracking_sprint_active(self.team)
                            && matches!(self.role, PlayerRole::Defender | PlayerRole::Midfielder)
                            && self.position.distance(*target) > 2.5)
                }
            }
            SoccerAction::MoveTo(target) if action_label == ACTION_LABEL_RECOVER => {
                // A loose ball that got away under pressure (notably the carrier's own
                // long/heavy touch with a defender closing) must be SPRINTED to, not
                // jogged — win the race back to it.
                loose_ball_pressured_sprint
                    || ((snapshot.rebound_recovery_sprint_active(self.id)
                        || (self.role == PlayerRole::Goalkeeper
                            && snapshot.goalkeeper_line_recovery_sprint_active(self.id)))
                        && self.position.distance(*target)
                            > GOALKEEPER_LINE_RECOVERY_SPRINT_DISTANCE_YARDS)
            }
            SoccerAction::MoveTo(target) if action_label == "hold" => {
                push_up_sprint
                    || (self.role == PlayerRole::Goalkeeper
                        && snapshot.goalkeeper_line_recovery_sprint_active(self.id)
                        && self.position.distance(*target)
                            > GOALKEEPER_LINE_RECOVERY_SPRINT_DISTANCE_YARDS)
            }
            _ => false,
        };
        PlayerIntent {
            player_id: self.id,
            action,
            sprint,
        }
    }

    fn apply_field_vector_context(
        &self,
        snapshot: &WorldSnapshot,
        mdp_state: &mut SoccerMdpState,
        observation: &mut SoccerPomdpObservation,
    ) {
        let canonical_state = snapshot.mdp_state_for_player(self.id);
        let canonical = snapshot.observation_for(self.id);
        let mut field_player_motion = canonical.field_player_motion.clone();
        if field_player_motion.len() != SOCCER_NEURAL_FIELD_MOTION_DIM
            || field_player_motion.iter().any(|value| !value.is_finite())
        {
            field_player_motion = soccer_field_player_motion_block(snapshot, self.id, self.team);
        }
        debug_assert_eq!(field_player_motion.len(), SOCCER_NEURAL_FIELD_MOTION_DIM);
        debug_assert!(field_player_motion.iter().all(|value| value.is_finite()));

        mdp_state.ball_zone_x = canonical_state.ball_zone_x;
        mdp_state.ball_zone_y = canonical_state.ball_zone_y;
        mdp_state.ball_grid = canonical_state.ball_grid;
        mdp_state.player_grid = canonical_state.player_grid;
        mdp_state.receive_facing = canonical_state.receive_facing;
        mdp_state.action_facing = canonical_state.action_facing;

        observation.player_grid = canonical.player_grid;
        observation.ball_grid = canonical.ball_grid;
        observation.receive_facing = canonical.receive_facing;
        observation.action_facing = canonical.action_facing;
        observation.field_player_motion = field_player_motion;
        observation.field_players_ahead = canonical.field_players_ahead;
        observation.field_players_behind = canonical.field_players_behind;
        observation.field_teammates_ahead = canonical.field_teammates_ahead;
        observation.field_teammates_behind = canonical.field_teammates_behind;
        observation.field_opponents_ahead = canonical.field_opponents_ahead;
        observation.field_opponents_behind = canonical.field_opponents_behind;
        observation.field_ahead_teammate_ratio =
            canonical.field_ahead_teammate_ratio.clamp(0.0, 1.0);
        observation.field_behind_teammate_ratio =
            canonical.field_behind_teammate_ratio.clamp(0.0, 1.0);
        observation.own_team_center_of_mass_forward_yards =
            canonical.own_team_center_of_mass_forward_yards;
        observation.own_team_center_of_mass_lateral_yards =
            canonical.own_team_center_of_mass_lateral_yards;
        observation.opponent_team_center_of_mass_forward_yards =
            canonical.opponent_team_center_of_mass_forward_yards;
        observation.opponent_team_center_of_mass_lateral_yards =
            canonical.opponent_team_center_of_mass_lateral_yards;
        observation.team_center_of_mass_forward_gap_yards =
            canonical.team_center_of_mass_forward_gap_yards;
        observation.team_center_of_mass_lateral_gap_yards =
            canonical.team_center_of_mass_lateral_gap_yards;
        observation.own_team_center_velocity_forward_yps =
            canonical.own_team_center_velocity_forward_yps;
        observation.own_team_center_velocity_lateral_yps =
            canonical.own_team_center_velocity_lateral_yps;
        observation.opponent_team_center_velocity_forward_yps =
            canonical.opponent_team_center_velocity_forward_yps;
        observation.opponent_team_center_velocity_lateral_yps =
            canonical.opponent_team_center_velocity_lateral_yps;
        observation.team_center_velocity_forward_gap_yps =
            canonical.team_center_velocity_forward_gap_yps;
        observation.team_center_velocity_lateral_gap_yps =
            canonical.team_center_velocity_lateral_gap_yps;
        observation.own_team_center_acceleration_forward_yps2 =
            canonical.own_team_center_acceleration_forward_yps2;
        observation.own_team_center_acceleration_lateral_yps2 =
            canonical.own_team_center_acceleration_lateral_yps2;
        observation.opponent_team_center_acceleration_forward_yps2 =
            canonical.opponent_team_center_acceleration_forward_yps2;
        observation.opponent_team_center_acceleration_lateral_yps2 =
            canonical.opponent_team_center_acceleration_lateral_yps2;
        observation.team_center_acceleration_forward_gap_yps2 =
            canonical.team_center_acceleration_forward_gap_yps2;
        observation.team_center_acceleration_lateral_gap_yps2 =
            canonical.team_center_acceleration_lateral_gap_yps2;

        observation.offensive_urgency = observation
            .offensive_urgency
            .max(finite_metric(canonical.offensive_urgency).clamp(0.0, 1.0));
        observation.defensive_urgency = observation
            .defensive_urgency
            .max(finite_metric(canonical.defensive_urgency).clamp(0.0, 1.0));
        observation.decision_urgency = observation
            .decision_urgency
            .max(finite_metric(canonical.decision_urgency).clamp(0.0, 1.0))
            .max(observation.offensive_urgency)
            .max(observation.defensive_urgency)
            .clamp(0.0, 1.0);
    }

    pub(crate) fn action_from_learned_plan(
        &self,
        plan: &SoccerLearnedPlan,
        snapshot: &WorldSnapshot,
        observation: &SoccerPomdpObservation,
    ) -> Option<(SoccerAction, String)> {
        let label = normalize_soccer_action_label(&plan.action);
        if !observation.has_ball
            && snapshot.controlled_possession_team() == Some(self.team.other())
            && !matches!(label, "tackle" | "slide-tackle" | "press-cover")
        {
            let assignment = snapshot.defensive_assignment_for(self.id, self.home_position, false);
            let target = plan
                .target_point
                .map(|target| snapshot.goal_side_defensive_target_for(self.id, target))
                .unwrap_or(assignment);
            return Some((SoccerAction::MoveTo(target), "defend".to_string()));
        }
        match label {
            "shoot"
                if observation.has_ball
                    && (shot_decision_is_qualified_for_role(observation, self.role)
                        || speculative_long_shot_is_qualified(
                            observation,
                            self.role,
                            ability01(self.skills.shooting),
                        )) =>
            {
                Some((
                    SoccerAction::Shoot {
                        power: shot_power_for_skill(ability01(self.skills.shooting)),
                    },
                    "shoot".to_string(),
                ))
            }
            "wall-pass" if observation.has_ball => {
                snapshot.wall_pass_option_for(self.id).map(|plan| {
                    (
                        SoccerAction::Pass {
                            target_player: Some(plan.wall_partner),
                            power: WALL_PASS_GIVE_POWER
                                + 0.20 * ability01(self.skills.passing_completion_rate),
                            flight: PassFlight::Floor,
                        },
                        "wall-pass".to_string(),
                    )
                })
            }
            "corner-flag-cross" if observation.has_ball => {
                let floor_targets = snapshot.ranked_visible_pass_targets(self.id, 11);
                let aerial_targets = snapshot.ranked_visible_aerial_pass_targets(self.id, 11);
                let crossing = ability01(self.skills.crossing_left.max(self.skills.crossing_right));
                plan.target_player
                    .filter(|target| floor_targets.contains(target))
                    .or_else(|| floor_targets.first().copied())
                    .map(|target| {
                        (
                            SoccerAction::Pass {
                                target_player: Some(target),
                                power: 0.62
                                    + 0.28
                                        * crossing
                                            .max(ability01(self.skills.passing_completion_rate)),
                                flight: PassFlight::Floor,
                            },
                            "corner-flag-cross".to_string(),
                        )
                    })
                    .or_else(|| {
                        plan.target_player
                            .filter(|target| aerial_targets.contains(target))
                            .or_else(|| aerial_targets.first().copied())
                            .map(|target| {
                                (
                                    SoccerAction::Pass {
                                        target_player: Some(target),
                                        power: 0.66
                                            + 0.28
                                                * crossing.max(ability01(
                                                    self.skills.passing_completion_rate,
                                                )),
                                        flight: PassFlight::Aerial,
                                    },
                                    "corner-flag-cross".to_string(),
                                )
                            })
                    })
            }
            "flank-low-cross" if observation.has_ball => {
                let visible_targets = snapshot.ranked_visible_pass_targets(self.id, 11);
                let target = plan
                    .target_player
                    .filter(|target| visible_targets.contains(target))
                    .or_else(|| {
                        plan.target_point.and_then(|point| {
                            visible_targets.iter().copied().min_by(|a, b| {
                                let a_dist = snapshot
                                    .player_position(*a)
                                    .map(|position| position.distance(point))
                                    .unwrap_or(f64::INFINITY);
                                let b_dist = snapshot
                                    .player_position(*b)
                                    .map(|position| position.distance(point))
                                    .unwrap_or(f64::INFINITY);
                                a_dist
                                    .partial_cmp(&b_dist)
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            })
                        })
                    })
                    .or_else(|| visible_targets.first().copied());
                target.map(|target| {
                    let crossing =
                        ability01(self.skills.crossing_left.max(self.skills.crossing_right));
                    (
                        SoccerAction::Pass {
                            target_player: Some(target),
                            power: 0.60
                                + 0.26
                                    * crossing.max(ability01(self.skills.passing_completion_rate)),
                            flight: PassFlight::Floor,
                        },
                        "flank-low-cross".to_string(),
                    )
                })
            }
            "flank-high-cross" if observation.has_ball => {
                let visible_targets = snapshot.ranked_visible_aerial_pass_targets(self.id, 11);
                let target = plan
                    .target_player
                    .filter(|target| visible_targets.contains(target))
                    .or_else(|| visible_targets.first().copied());
                target.map(|target| {
                    let crossing =
                        ability01(self.skills.crossing_left.max(self.skills.crossing_right));
                    (
                        SoccerAction::Pass {
                            target_player: Some(target),
                            power: 0.64
                                + 0.28
                                    * crossing.max(ability01(self.skills.passing_completion_rate)),
                            flight: PassFlight::Aerial,
                        },
                        "flank-high-cross".to_string(),
                    )
                })
            }
            "switch-play" if observation.has_ball => {
                if !Self::learned_pass_viable(observation, PassFlight::Floor) {
                    return None;
                }
                let visible_targets = snapshot.ranked_visible_pass_targets(self.id, 11);
                let target =
                    self.switch_play_target_for(snapshot, plan.target_player, &visible_targets);
                target.map(|target| {
                    (
                        SoccerAction::Pass {
                            target_player: Some(target),
                            power: 0.62
                                + 0.24
                                    * ability01(self.skills.passing_completion_rate)
                                        .max(ability01(self.skills.flair_passing)),
                            flight: PassFlight::Floor,
                        },
                        "switch-play".to_string(),
                    )
                })
            }
            "recycle-reset" if observation.has_ball => {
                if !Self::learned_pass_viable(observation, PassFlight::Floor)
                    || goal_attack_shot_blocks_alternatives(observation, self.role)
                {
                    return None;
                }
                let visible_targets = snapshot.ranked_visible_pass_targets(self.id, 11);
                let target =
                    self.recycle_reset_target_for(snapshot, plan.target_player, &visible_targets);
                target.map(|target| {
                    (
                        SoccerAction::Pass {
                            target_player: Some(target),
                            power: 0.46 + 0.22 * ability01(self.skills.passing_completion_rate),
                            flight: PassFlight::Floor,
                        },
                        "recycle-reset".to_string(),
                    )
                })
            }
            "scoop-pass" if observation.has_ball => {
                snapshot.scoop_pass_target_for(self.id).map(|target| {
                    (
                        SoccerAction::Pass {
                            target_player: Some(target),
                            power: 0.5,
                            flight: PassFlight::Scoop,
                        },
                        "scoop-pass".to_string(),
                    )
                })
            }
            "surprise-pass" if observation.has_ball => {
                let visible_targets = snapshot.ranked_visible_pass_targets(self.id, 11);
                plan.target_player
                    .filter(|target| visible_targets.contains(target))
                    .or_else(|| visible_targets.first().copied())
                    .map(|target| {
                        (
                            SoccerAction::Pass {
                                target_player: Some(target),
                                power: 0.50
                                    + 0.22
                                        * ability01(self.skills.passing_completion_rate)
                                            .max(ability01(self.skills.flair_passing)),
                                flight: PassFlight::Floor,
                            },
                            "surprise-pass".to_string(),
                        )
                    })
            }
            "pass" if observation.has_ball => {
                // Veto a junk pass (covered/absent receiver) — defer to heuristic.
                if !Self::learned_pass_viable(observation, PassFlight::Floor) {
                    return None;
                }
                let visible_targets = snapshot.ranked_visible_pass_targets(self.id, 11);
                if let Some(target) = learned_generic_pass_killer_upgrade_target_for(
                    self,
                    snapshot,
                    observation,
                    &visible_targets,
                ) {
                    return Some((
                        SoccerAction::Pass {
                            target_player: Some(target),
                            power: 0.66
                                + 0.24
                                    * ability01(self.skills.passing_completion_rate)
                                        .max(ability01(self.skills.vision)),
                            flight: snapshot.killer_pass_flight_for(self.id, &visible_targets),
                        },
                        "killer-pass".to_string(),
                    ));
                }
                let target = plan
                    .target_player
                    .filter(|target| visible_targets.contains(target))
                    .or_else(|| {
                        plan.target_point.and_then(|point| {
                            visible_targets.iter().copied().min_by(|a, b| {
                                let a_dist = snapshot
                                    .player_position(*a)
                                    .map(|position| position.distance(point))
                                    .unwrap_or(f64::INFINITY);
                                let b_dist = snapshot
                                    .player_position(*b)
                                    .map(|position| position.distance(point))
                                    .unwrap_or(f64::INFINITY);
                                a_dist
                                    .partial_cmp(&b_dist)
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            })
                        })
                    })
                    .or_else(|| visible_targets.first().copied());
                target.map(|target| {
                    (
                        SoccerAction::Pass {
                            target_player: Some(target),
                            power: 0.58 + 0.32 * ability01(self.skills.passing_completion_rate),
                            flight: PassFlight::Floor,
                        },
                        "pass".to_string(),
                    )
                })
            }
            "killer-pass" if observation.has_ball => {
                // A killer-pass is only on if there's a REAL threaded lane to a
                // goalward receiver. The learned policy otherwise fires it down a
                // covered lane — a through-ball straight to nobody (the live
                // "passed to nobody" giveaways). Mirror the heuristic's own
                // legality (`threaded_goal_pass_available`) and require the lane to
                // actually be completable; otherwise defer to the heuristic.
                if !observation.threaded_goal_pass_available
                    || observation.threaded_goal_pass_expected_completion
                        < LEARNED_PASS_MIN_COMPLETION
                {
                    return None;
                }
                let visible_targets = snapshot.ranked_visible_pass_targets(self.id, 11);
                let target = plan
                    .target_player
                    .filter(|target| visible_targets.contains(target))
                    .filter(|target| {
                        snapshot
                            .killer_pass_target_for(self.id, &[*target])
                            .is_some()
                    })
                    .or_else(|| snapshot.killer_pass_target_for(self.id, &visible_targets));
                target.map(|target| {
                    (
                        SoccerAction::Pass {
                            target_player: Some(target),
                            power: 0.66
                                + 0.24
                                    * ability01(self.skills.passing_completion_rate)
                                        .max(ability01(self.skills.vision)),
                            flight: snapshot.killer_pass_flight_for(self.id, &visible_targets),
                        },
                        "killer-pass".to_string(),
                    )
                })
            }
            "aerial-pass" if observation.has_ball => {
                // Veto a junk aerial hoof (covered/absent receiver) — the live
                // giveaways were aerial-pass at completion 0.0 / openness 0.0
                // straight to an opponent. Defer to the heuristic scorer instead.
                if !Self::learned_pass_viable(observation, PassFlight::Aerial) {
                    return None;
                }
                let visible_targets = snapshot.ranked_visible_aerial_pass_targets(self.id, 11);
                let target = plan
                    .target_player
                    .filter(|target| visible_targets.contains(target))
                    .or_else(|| visible_targets.first().copied());
                target.map(|target| {
                    let crossing =
                        ability01(self.skills.crossing_left.max(self.skills.crossing_right));
                    (
                        SoccerAction::Pass {
                            target_player: Some(target),
                            power: 0.56
                                + 0.28
                                    * crossing.max(ability01(self.skills.passing_completion_rate)),
                            flight: PassFlight::Aerial,
                        },
                        "aerial-pass".to_string(),
                    )
                })
            }
            "flick-on" if observation.has_ball => {
                let visible_targets = snapshot.ranked_visible_aerial_pass_targets(self.id, 11);
                plan.target_player
                    .filter(|target| visible_targets.contains(target))
                    .or_else(|| visible_targets.first().copied())
                    .map(|target| {
                        (
                            SoccerAction::Pass {
                                target_player: Some(target),
                                power: 0.46
                                    + 0.26
                                        * aerial_duel_skill_from_agent(self)
                                            .max(ability01(self.skills.first_touch)),
                                flight: PassFlight::Aerial,
                            },
                            "flick-on".to_string(),
                        )
                    })
            }
            "first-time-shot" | "first-time-header"
                if observation.has_ball
                    && observation.first_touch_available
                    && first_time_shot_decision_is_qualified_for_role(observation, self.role) =>
            {
                let finish_skill = if label == "first-time-header" {
                    aerial_duel_skill_from_agent(self)
                } else {
                    ability01(
                        self.skills
                            .right_foot_shot_power
                            .max(self.skills.left_foot_shot_power),
                    )
                };
                Some((
                    SoccerAction::Shoot {
                        power: shot_power_for_finish_skill(finish_skill),
                    },
                    label.to_string(),
                ))
            }
            "first-time-pass" if observation.has_ball && observation.first_touch_available => {
                let visible = snapshot.ranked_visible_pass_targets(self.id, 11);
                // Prefer the quick forward (≈5–8 m, open, advanced) teammate when one is on —
                // played first-time and delivered through the MPC pass path. None when the gate
                // is off ⇒ falls back to the top-ranked visible target (unchanged).
                let target = observation
                    .quick_forward_pass_target
                    .filter(|id| visible.contains(id))
                    .or_else(|| visible.first().copied());
                target.map(|target| {
                    (
                        SoccerAction::Pass {
                            target_player: Some(target),
                            power: 0.54 + 0.30 * ability01(self.skills.passing_completion_rate),
                            flight: PassFlight::Floor,
                        },
                        "first-time-pass".to_string(),
                    )
                })
            }
            "control-touch" if observation.has_ball && observation.first_touch_available => {
                let target = self.control_touch_target_for_goal_context(snapshot, observation);
                Some((
                    SoccerAction::ControlTouch { target },
                    "control-touch".to_string(),
                ))
            }
            "clearance" if observation.has_ball => Some((
                SoccerAction::Clearance {
                    target: snapshot
                        .pressure_clearance_target_for(self.id)
                        .unwrap_or_else(|| {
                            clearance_target_for_player(
                                self.team,
                                self.position,
                                snapshot.field_width,
                                snapshot.field_length,
                            )
                        }),
                    power: 0.92,
                },
                "clearance".to_string(),
            )),
            "route-one" if observation.has_ball => Some((
                SoccerAction::RouteOne {
                    target: snapshot.route_one_target_for(self.id).unwrap_or_else(|| {
                        route_one_target_for_actor(
                            self.team,
                            self.position,
                            snapshot.field_width,
                            snapshot.field_length,
                            self.role,
                        )
                    }),
                    power: 0.88,
                },
                "route-one".to_string(),
            )),
            "hold-up-flank" if observation.has_ball => {
                if goal_approach_forces_goalmouth_carry(observation, self.role) {
                    let kind = DribbleMoveKind::CarryForward;
                    let touch = snapshot.deterministic_dribble_touch_decision_for(self.id, kind);
                    return Some((
                        SoccerAction::DribbleMove {
                            target: snapshot.dribble_move_target_for_touch(
                                self.id,
                                self.home_position,
                                kind,
                                touch,
                            ),
                            kind,
                            touch,
                        },
                        kind.label().to_string(),
                    ));
                }
                let target = plan
                    .target_point
                    .or_else(|| snapshot.striker_hold_up_flank_target_for(self.id))?;
                let kind = snapshot.side_step_dribble_kind_for(self.id);
                let bucket = match dribble_final_cut_kind(kind) {
                    DribbleMoveKind::CarryForward | DribbleMoveKind::ProtectBall => 0,
                    DribbleMoveKind::CarryOutLeft | DribbleMoveKind::LeftCut => 9,
                    DribbleMoveKind::CarryOutRight | DribbleMoveKind::RightCut => 3,
                    DribbleMoveKind::Nutmeg => 0,
                    DribbleMoveKind::XaviTurn => 10,
                    DribbleMoveKind::FakeLeftCutRight | DribbleMoveKind::FakeRightCutLeft => 0,
                };
                Some((
                    SoccerAction::DribbleMove {
                        target,
                        kind,
                        touch: DribbleTouchDecision::new(bucket, 1.35),
                    },
                    "hold-up-flank".to_string(),
                ))
            }
            ROUND_GOALKEEPER_ACTION_LABEL if observation.has_ball => self
                .round_goalkeeper_dribble_plan_for(snapshot, observation)
                .map(|round_plan| {
                    (
                        SoccerAction::DribbleMove {
                            target: round_plan.target,
                            kind: round_plan.kind,
                            touch: round_plan.touch,
                        },
                        ROUND_GOALKEEPER_ACTION_LABEL.to_string(),
                    )
                }),
            "runaround-dribble" if observation.has_ball => snapshot
                .runaround_dribble_option_for(self.id)
                .map(|runaround_plan| {
                    (
                        SoccerAction::MoveTo(runaround_plan.recollect_point),
                        "runaround-dribble".to_string(),
                    )
                }),
            OPEN_PASS_LANE_ACTION_LABEL if observation.has_ball => {
                let visible_targets = snapshot.ranked_visible_pass_targets(self.id, 11);
                let mut lane_targets = Vec::with_capacity(visible_targets.len() + 1);
                if let Some(target_player) = plan.target_player {
                    lane_targets.push(target_player);
                }
                for target_id in visible_targets {
                    if !lane_targets.contains(&target_id) {
                        lane_targets.push(target_id);
                    }
                }
                let current = snapshot
                    .player_position(self.id)
                    .unwrap_or(self.position)
                    .clamp_to_pitch(snapshot.field_width, snapshot.field_length);
                let learned_target = plan.target_point.and_then(|target| {
                    let target = target.clamp_to_pitch(snapshot.field_width, snapshot.field_length);
                    let step = target - current;
                    let step_distance = step.len();
                    let forward = step.y * self.team.attack_dir();
                    let situation = open_pass_lane_situation_for_target(
                        snapshot,
                        self,
                        observation,
                        current,
                        target,
                        0.0,
                    );
                    let dynamic_max_yards = open_pass_lane_dynamic_max_dribble_yards(situation);
                    if !(OPEN_PASS_LANE_MIN_DRIBBLE_YARDS..=OPEN_PASS_LANE_MAX_DRIBBLE_YARDS + 1e-6)
                        .contains(&step_distance)
                        || step_distance > dynamic_max_yards + 1e-6
                        || forward < -OPEN_PASS_LANE_MAX_BACKWARD_STEP_YARDS
                        || snapshot.nearest_opponent_distance_at(self.team, target)
                            < OPEN_PASS_LANE_MIN_CANDIDATE_OPPONENT_SPACE
                    {
                        return None;
                    }
                    let touch = open_pass_lane_touch_for_target(self.team, current, target)?;
                    let kind = open_pass_lane_kind_for_target(self.team, current, target);
                    Some(OpenPassLaneDribblePlan {
                        target,
                        kind,
                        touch,
                        score: 0.0,
                    })
                });
                self.open_pass_lane_dribble_plan_for(snapshot, observation, &lane_targets)
                    .or(learned_target)
                    .map(|lane_plan| {
                        (
                            SoccerAction::DribbleMove {
                                target: lane_plan.target,
                                kind: lane_plan.kind,
                                touch: lane_plan.touch,
                            },
                            OPEN_PASS_LANE_ACTION_LABEL.to_string(),
                        )
                    })
            }
            "dribble"
            | "vertical-attack"
            | "turnover-burst"
            | "carry-forward"
            | "carry-out-left"
            | "carry-out-right"
            | "protect-ball"
            | "xavi-turn"
            | "left-cut"
            | "right-cut"
            | "nutmeg"
            | "fake-left-cut-right"
            | "fake-right-cut-left"
                if observation.has_ball =>
            {
                if let Some(release) =
                    self.learned_stale_dribble_release_override(snapshot, observation)
                {
                    return Some(release);
                }
                let kind = match label {
                    "vertical-attack" | "turnover-burst" => DribbleMoveKind::CarryForward,
                    "carry-forward" => DribbleMoveKind::CarryForward,
                    "carry-out-left" => DribbleMoveKind::CarryOutLeft,
                    "carry-out-right" => DribbleMoveKind::CarryOutRight,
                    "protect-ball" => DribbleMoveKind::ProtectBall,
                    "xavi-turn" => DribbleMoveKind::XaviTurn,
                    "left-cut" => DribbleMoveKind::LeftCut,
                    "right-cut" => DribbleMoveKind::RightCut,
                    "nutmeg" => DribbleMoveKind::Nutmeg,
                    "fake-left-cut-right" => DribbleMoveKind::FakeLeftCutRight,
                    "fake-right-cut-left" => DribbleMoveKind::FakeRightCutLeft,
                    _ => self.agentic_fallback_dribble_kind(snapshot, observation, false, true),
                };
                let force_goal_drive = goal_approach_forces_goalmouth_carry(observation, self.role);
                let kind = goal_approach_dribble_kind(kind, observation, self.role);
                let touch = snapshot.deterministic_dribble_touch_decision_for(self.id, kind);
                Some((
                    SoccerAction::DribbleMove {
                        target: if force_goal_drive {
                            snapshot.dribble_move_target_for_touch(
                                self.id,
                                self.home_position,
                                kind,
                                touch,
                            )
                        } else {
                            plan.target_point.unwrap_or_else(|| {
                                snapshot.dribble_move_target_for_touch(
                                    self.id,
                                    self.home_position,
                                    kind,
                                    touch,
                                )
                            })
                        },
                        kind,
                        touch,
                    },
                    if matches!(label, "vertical-attack" | "turnover-burst") {
                        label.to_string()
                    } else {
                        kind.label().to_string()
                    },
                ))
            }
            "side-step" if observation.has_ball => {
                if let Some(release) =
                    self.learned_stale_dribble_release_override(snapshot, observation)
                {
                    return Some(release);
                }
                let kind = snapshot.side_step_dribble_kind_for(self.id);
                let force_goal_drive = goal_approach_forces_goalmouth_carry(observation, self.role);
                let kind = goal_approach_dribble_kind(kind, observation, self.role);
                let touch = snapshot.deterministic_dribble_touch_decision_for(self.id, kind);
                Some((
                    SoccerAction::DribbleMove {
                        target: if force_goal_drive {
                            snapshot.dribble_move_target_for_touch(
                                self.id,
                                self.home_position,
                                kind,
                                touch,
                            )
                        } else {
                            plan.target_point.unwrap_or_else(|| {
                                snapshot.dribble_move_target_for_touch(
                                    self.id,
                                    self.home_position,
                                    kind,
                                    touch,
                                )
                            })
                        },
                        kind,
                        touch,
                    },
                    kind.label().to_string(),
                ))
            }
            "defend" if snapshot.controlled_possession_team() == Some(self.team.other()) => Some((
                SoccerAction::MoveTo(snapshot.defensive_assignment_for(
                    self.id,
                    self.home_position,
                    false,
                )),
                "defend".to_string(),
            )),
            ACTION_LABEL_RECOVER if snapshot.controlled_possession_team().is_none() => Some((
                SoccerAction::MoveTo(snapshot.loose_ball_recovery_target_for(self.id)),
                ACTION_LABEL_RECOVER.to_string(),
            )),
            "tackle" => snapshot.ball.holder.and_then(|holder| {
                if self.role == PlayerRole::Goalkeeper
                    && !snapshot.goalkeeper_direct_intervention_is_safe(self.id)
                {
                    return None;
                }
                let holder_player = snapshot.players.iter().find(|p| p.id == holder)?;
                if holder_player.team == self.team.other()
                    && self.position.distance(holder_player.position) < 3.2
                {
                    Some((
                        SoccerAction::Tackle {
                            target_player: holder,
                        },
                        "tackle".to_string(),
                    ))
                } else {
                    None
                }
            }),
            // A learned/retrieved plan can elect a slide, but it must clear the same
            // last-ditch legality gate as the heuristic (enabled, in reach, not from
            // behind, fast carrier, commit-warranted) — no summoning an illegal slide.
            "slide-tackle" => self.slide_tackle_decision(snapshot).map(|(holder, _)| {
                (
                    SoccerAction::SlideTackle {
                        target_player: holder,
                    },
                    "slide-tackle".to_string(),
                )
            }),
            "press-cover" if snapshot.controlled_possession_team() == Some(self.team.other()) => {
                let assignment =
                    snapshot.defensive_assignment_for(self.id, self.home_position, true);
                Some((
                    SoccerAction::MoveTo(
                        snapshot.goal_side_defensive_target_for(self.id, assignment),
                    ),
                    "press-cover".to_string(),
                ))
            }
            DUMMY_LET_RUN_ACTION_LABEL if !observation.has_ball => snapshot
                .teammate_dummy_let_run_target_for(self.id)
                .map(|target| {
                    (
                        SoccerAction::MoveTo(target),
                        DUMMY_LET_RUN_ACTION_LABEL.to_string(),
                    )
                }),
            DUMMY_CLEAR_LANE_ACTION_LABEL if !observation.has_ball => snapshot
                .teammate_pass_lane_clearance_target_for(self.id)
                .map(|target| {
                    (
                        SoccerAction::MoveTo(target),
                        DUMMY_CLEAR_LANE_ACTION_LABEL.to_string(),
                    )
                }),
            "check-to-ball" if !observation.has_ball => snapshot
                .check_to_ball_target_for(self.id, self.home_position)
                .map(|target| {
                    let open = snapshot.open_space_for(self.id, self.home_position);
                    (
                        SoccerAction::MoveTo(snapshot.shape_guarded_support_point(
                            self.id,
                            target,
                            &[open, self.home_position],
                            self.home_position,
                            false,
                        )),
                        "check-to-ball".to_string(),
                    )
                }),
            "run-in-behind" if !observation.has_ball => {
                snapshot.in_behind_run_target_for(self.id).map(|target| {
                    (
                        SoccerAction::MoveTo(snapshot.shape_guarded_support_point(
                            self.id,
                            target,
                            &[],
                            self.home_position,
                            false,
                        )),
                        "run-in-behind".to_string(),
                    )
                })
            }
            "wide-outlet" if !observation.has_ball => snapshot
                .wide_possession_outlet_target_for(self.id, self.home_position)
                .map(|target| {
                    let open = snapshot.open_space_for(self.id, self.home_position);
                    (
                        SoccerAction::MoveTo(snapshot.shape_guarded_support_point(
                            self.id,
                            target,
                            &[open, self.home_position],
                            self.home_position,
                            false,
                        )),
                        "wide-outlet".to_string(),
                    )
                }),
            "exploit-space-run" | "shot-creation-run" | "overlap-run" | "support-push-up"
            | "support-screen" | "support-shape" | "support-roam" | "vacate-space"
            | "vertical-attack"
                if !observation.has_ball =>
            {
                let roam = matches!(label, "support-roam" | "overlap-run" | "vacate-space");
                let support_target =
                    snapshot.attacking_support_movement_for(self.id, self.home_position, roam);
                let target = if self.role == PlayerRole::Goalkeeper {
                    support_target.point
                } else {
                    plan.target_point
                        .map(|target| {
                            snapshot.shape_guarded_learned_support_target(
                                self.id,
                                target,
                                support_target.point,
                                self.home_position,
                            )
                        })
                        .unwrap_or(support_target.point)
                };
                Some((SoccerAction::MoveTo(target), label.to_string()))
            }
            "space" if !observation.has_ball => {
                let target = if snapshot.controlled_possession_team() == Some(self.team) {
                    if self.role == PlayerRole::Goalkeeper {
                        snapshot.goalkeeper_ball_goal_tracking_target(self.team)
                    } else {
                        snapshot.positional_open_space_for(self.id, self.home_position, false)
                    }
                } else if snapshot.controlled_possession_team() == Some(self.team.other()) {
                    snapshot.defensive_assignment_for(self.id, self.home_position, false)
                } else {
                    snapshot.defensive_shape_for(self.id, self.home_position)
                };
                Some((SoccerAction::MoveTo(target), "space".to_string()))
            }
            "hold" if self.role == PlayerRole::Goalkeeper && !observation.has_ball => Some((
                SoccerAction::MoveTo(snapshot.goalkeeper_ball_goal_tracking_target(self.team)),
                "hold".to_string(),
            )),
            "hold" => Some((SoccerAction::MoveTo(self.home_position), "hold".to_string())),
            _ => None,
        }
    }

    fn switch_play_target_for(
        &self,
        snapshot: &WorldSnapshot,
        preferred: Option<usize>,
        candidates: &[usize],
    ) -> Option<usize> {
        let passer_position = snapshot.player_position(self.id).unwrap_or(self.position);
        let center_x = snapshot.field_width * 0.5;
        let attack_dir = self.team.attack_dir();
        let min_lateral = (snapshot.field_width * 0.24).max(12.0);
        let score = |target_id: usize| -> Option<f64> {
            let target_position = snapshot.player_position(target_id)?;
            let lateral = (target_position.x - passer_position.x).abs();
            let forward = (target_position.y - passer_position.y) * attack_dir;
            let distance = target_position.distance(passer_position);
            let crosses_midline =
                (passer_position.x - center_x) * (target_position.x - center_x) < -6.0;
            if distance < 10.0 || forward < -8.0 || (!crosses_midline && lateral < min_lateral) {
                return None;
            }
            Some(
                lateral * 0.070
                    + forward.clamp(-4.0, 18.0) * 0.020
                    + if crosses_midline { 1.0 } else { 0.0 },
            )
        };
        if preferred.is_some_and(|target| score(target).is_some()) {
            return preferred;
        }
        candidates
            .iter()
            .copied()
            .filter_map(|target| score(target).map(|s| (target, s)))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(target, _)| target)
    }

    fn recycle_reset_target_for(
        &self,
        snapshot: &WorldSnapshot,
        preferred: Option<usize>,
        candidates: &[usize],
    ) -> Option<usize> {
        let passer_position = snapshot.player_position(self.id).unwrap_or(self.position);
        let attack_dir = self.team.attack_dir();
        let score = |target_id: usize| -> Option<f64> {
            let target_position = snapshot.player_position(target_id)?;
            let forward = (target_position.y - passer_position.y) * attack_dir;
            let distance = target_position.distance(passer_position);
            if !(2.0..=10.0).contains(&distance)
                || forward > 1.5
                || forward < -BACKWARD_PASS_SHORT_RESET_MAX_YARDS - 0.75
            {
                return None;
            }
            let backward_yards = (-forward).max(0.0);
            let short_reset_fit = (1.0 - (backward_yards - 4.0).abs() / 4.0).clamp(0.0, 1.0);
            Some(short_reset_fit + (1.0 - distance / 10.0).clamp(0.0, 1.0) * 0.35)
        };
        if preferred.is_some_and(|target| score(target).is_some()) {
            return preferred;
        }
        candidates
            .iter()
            .copied()
            .filter_map(|target| score(target).map(|s| (target, s)))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(target, _)| target)
    }

    /// Whether a learned-policy pass of `flight` is worth executing. A pass is
    /// viable if it has a real completion estimate OR an open receiver; only when
    /// BOTH are below the floor is it a giveaway (a hoof at a covered/absent
    /// target). Used to veto junk learned passes and defer to the heuristic.
    fn learned_pass_viable(observation: &SoccerPomdpObservation, flight: PassFlight) -> bool {
        let (completion, openness) = if flight.is_aerial() {
            (
                observation.expected_aerial_pass_completion,
                observation.best_aerial_pass_receiver_openness,
            )
        } else {
            (
                observation.expected_pass_completion,
                observation.best_pass_receiver_openness,
            )
        };
        // A receiver can be wide open yet a defender sits in the flight lane and
        // cuts it out — `expected_*_completion` measures space AROUND the receiver,
        // NOT whether the ball reaches them. An aerial ball with high interception
        // risk is a giveaway no matter how open the target is (the live aerial
        // hoofs were openness ~1.0 / completion ~0.9 with interception 0.5-0.9).
        if flight.is_aerial()
            && observation.aerial_pass_interception_risk > LEARNED_PASS_MAX_INTERCEPTION_RISK
        {
            return false;
        }
        completion >= LEARNED_PASS_MIN_COMPLETION || openness >= LEARNED_PASS_MIN_OPENNESS
    }

    fn learned_stale_dribble_release_override(
        &self,
        snapshot: &WorldSnapshot,
        observation: &SoccerPomdpObservation,
    ) -> Option<(SoccerAction, String)> {
        if !observation.has_ball {
            return None;
        }

        // Open advanced outlet — pressure-independent. If a teammate is clearly
        // open with a high-completion lane, move the ball to them instead of
        // dribbling on: you do NOT carry into traffic when a man is open ahead.
        // (The heuristic possession scorer already prefers this; the learned
        // policy bypasses it, which is how the carrier ends up dribbling past an
        // obviously open pass.) A hard-blocking close shot still takes priority.
        if observation.best_pass_receiver_openness >= OPEN_OUTLET_RELEASE_OPENNESS
            && observation.expected_pass_completion >= OPEN_OUTLET_RELEASE_COMPLETION
            && !goal_attack_shot_blocks_alternatives(observation, self.role)
        {
            if let Some(target) = snapshot
                .ranked_visible_pass_targets(self.id, 1)
                .first()
                .copied()
            {
                return Some((
                    SoccerAction::Pass {
                        target_player: Some(target),
                        power: 0.58 + 0.30 * ability01(self.skills.passing_completion_rate),
                        flight: PassFlight::Floor,
                    },
                    "pass1".to_string(),
                ));
            }
        }

        let dribbling = ability01(self.skills.dribbling);
        let hold_pressure = observation
            .excessive_hold_pressure
            .max(excessive_hold_pressure(observation, dribbling));
        let danger_pressure = observation
            .pressure_urgency
            .max(observation.defensive_urgency)
            .max(observation.immediate_dispossession_risk)
            .max(observation.perceived_pressure)
            .clamp(0.0, 1.0);
        if hold_pressure < 0.18 || danger_pressure < 0.42 {
            return None;
        }

        if matches!(self.role, PlayerRole::Goalkeeper | PlayerRole::Defender)
            && observation.yards_to_own_goal < observation.yards_to_goal
            // Only clear under PROXIMATE pressure / genuine dispossession threat (or
            // from our own box) — don't hoof it when the nearest opponent is yards
            // away and we can play out.
            && (observation.nearest_opponent_distance <= CLEARANCE_MAX_OPPONENT_DISTANCE_YARDS
                || observation.immediate_dispossession_risk >= 0.5
                || observation.yards_to_own_goal <= 20.0)
        {
            return Some((
                SoccerAction::Clearance {
                    target: snapshot
                        .pressure_clearance_target_for(self.id)
                        .unwrap_or_else(|| {
                            clearance_target_for_player(
                                self.team,
                                self.position,
                                snapshot.field_width,
                                snapshot.field_length,
                            )
                        }),
                    power: (0.88 + danger_pressure * 0.10 + hold_pressure * 0.06).clamp(0.88, 0.98),
                },
                "clearance".to_string(),
            ));
        }

        // Forward room needed before "carry on" is even an option. Mirrors the
        // heuristic carry-forward gate: relaxed inside the final third where
        // taking a man on pays, stricter in open play where it just walks the
        // ball into the tackle.
        let carry_min_forward_space =
            if observation.yards_to_goal <= DRIBBLE_FINAL_THIRD_YARDS_TO_GOAL {
                1.2
            } else {
                DRIBBLE_OPEN_PLAY_MIN_FORWARD_SPACE_YARDS
            };

        // Even an elite dribbler is NOT exempt when boxed in: with no forward
        // space ahead, carrying on simply walks the ball into the defender.
        // Only exempt the elite when there is genuine room to attack. He ALSO
        // yields the carry when he is genuinely about to be dispossessed AND a
        // forward teammate is clearly open — there, carrying on just gifts the ball
        // away, so the open forward pass is the better ball. (Falls through to the
        // floor-pass release below, which is guaranteed viable since overall
        // openness >= the open forward openness here.) In moderate pressure the
        // elite keeps the carry — the whole point of the exemption.
        let yield_to_open_forward_outlet = observation.visible_forward_pass_options > 0
            && observation.best_forward_pass_receiver_openness
                >= ELITE_HOLD_OPEN_FORWARD_OUTLET_OPENNESS
            && observation.immediate_dispossession_risk >= ELITE_HOLD_YIELD_DISPOSSESSION_RISK;
        if dribbling >= NON_ELITE_DRIBBLE_HOLD_SKILL_CUTOFF
            && hold_pressure < 0.64
            && observation.forward_dribble_space_yards >= carry_min_forward_space
            && !yield_to_open_forward_outlet
        {
            return None;
        }

        if shot_decision_is_qualified_for_role(observation, self.role)
            && (goal_attack_shot_blocks_alternatives(observation, self.role)
                || observation.yards_to_goal <= TEAMMATE_MUST_SHOOT_YARDS)
        {
            return Some((
                SoccerAction::Shoot {
                    power: shot_power_for_skill(ability01(self.skills.shooting)),
                },
                "shoot".to_string(),
            ));
        }

        let floor_target = snapshot
            .ranked_visible_pass_targets(self.id, 1)
            .first()
            .copied()
            .filter(|_| {
                observation.expected_pass_completion >= 0.34
                    || observation.best_pass_receiver_openness >= 0.34
            });
        if let Some(target) = floor_target {
            return Some((
                SoccerAction::Pass {
                    target_player: Some(target),
                    power: 0.60 + 0.30 * ability01(self.skills.passing_completion_rate),
                    flight: PassFlight::Floor,
                },
                "pass1".to_string(),
            ));
        }

        let aerial_target = snapshot
            .ranked_visible_aerial_pass_targets(self.id, 1)
            .first()
            .copied()
            .filter(|_| {
                observation.expected_aerial_pass_completion >= 0.34
                    || observation.best_aerial_pass_receiver_openness >= 0.34
            });
        if let Some(target) = aerial_target {
            return Some((
                SoccerAction::Pass {
                    target_player: Some(target),
                    power: 0.62
                        + 0.28
                            * ability01(
                                self.skills
                                    .passing_completion_rate
                                    .max(self.skills.crossing_left.max(self.skills.crossing_right)),
                            ),
                    flight: PassFlight::Aerial,
                },
                "aerial-pass1".to_string(),
            ));
        }

        // Pinned under real pressure, can't shoot, and no open teammate to pass
        // to. Carrying on here just dribbles the ball into the defender (or
        // forces a panic ball straight to an opponent). When there's no forward
        // room left, shield the ball instead — turn the body between ball and
        // defender to retain possession and buy time for support — rather than
        // gifting it away.
        if observation.forward_dribble_space_yards < carry_min_forward_space {
            let kind = DribbleMoveKind::ProtectBall;
            let touch = snapshot.deterministic_dribble_touch_decision_for(self.id, kind);
            return Some((
                SoccerAction::DribbleMove {
                    target: snapshot.dribble_move_target_for_touch(
                        self.id,
                        self.home_position,
                        kind,
                        touch,
                    ),
                    kind,
                    touch,
                },
                "protect-ball".to_string(),
            ));
        }

        None
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SoccerAction {
    HoldShape,
    MoveTo(Vec2),
    Dribble(Vec2),
    DribbleMove {
        target: Vec2,
        kind: DribbleMoveKind,
        #[serde(default)]
        touch: DribbleTouchDecision,
    },
    ControlTouch {
        target: Vec2,
    },
    Pass {
        target_player: Option<usize>,
        power: f64,
        #[serde(default)]
        flight: PassFlight,
    },
    Clearance {
        target: Vec2,
        power: f64,
    },
    RouteOne {
        target: Vec2,
        power: f64,
    },
    Shoot {
        power: f64,
    },
    Tackle {
        target_player: usize,
    },
    /// A committed, off-the-feet challenge: higher chance to win the ball than a
    /// standing tackle, but a miss leaves the defender on the ground (see
    /// [`PlayerAgent::slide_recovery_seconds`]) for ~2s while the attacker advances.
    SlideTackle {
        target_player: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DribbleMoveKind {
    CarryForward,
    CarryOutLeft,
    CarryOutRight,
    ProtectBall,
    LeftCut,
    RightCut,
    Nutmeg,
    FakeLeftCutRight,
    FakeRightCutLeft,
    /// The "Xavi turn": a near-full (~280-300°) shielded pirouette. The carrier keeps
    /// the ball on the FAR side of the nearest defender for the whole arc and wheels
    /// the long way AROUND them — a longer but safer path that turns the defender
    /// rather than trying to knock the ball past. While turning, the ball is so well
    /// protected that a clean steal is capped near 10% (see
    /// [`XAVI_TURN_TACKLE_SUCCESS_CAP`]). Gated by [`xavi_turn_enabled`].
    XaviTurn,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DribbleTouchDecision {
    pub angle_bucket: u8,
    pub angle_degrees: f64,
    pub distance_yards: f64,
}

impl Default for DribbleTouchDecision {
    fn default() -> Self {
        Self::new(0, DRIBBLE_TOUCH_LEAD_YARDS)
    }
}

impl DribbleTouchDecision {
    pub(crate) fn new(angle_bucket: u8, distance_yards: f64) -> Self {
        let bucket = angle_bucket % DRIBBLE_TOUCH_ANGLE_BUCKETS;
        Self {
            angle_bucket: bucket,
            angle_degrees: bucket as f64 * DRIBBLE_TOUCH_BUCKET_DEGREES,
            distance_yards: distance_yards.clamp(DRIBBLE_MIN_TOUCH_YARDS, DRIBBLE_MAX_TOUCH_YARDS),
        }
    }

    pub(crate) fn direction_for_team(self, team: Team) -> Vec2 {
        let radians = self.angle_degrees.to_radians();
        Vec2::new(
            radians.sin() * team.attack_dir(),
            radians.cos() * team.attack_dir(),
        )
        .normalized()
    }
}

impl DribbleMoveKind {
    pub(crate) fn label(self) -> &'static str {
        match self {
            DribbleMoveKind::CarryForward => "carry-forward",
            DribbleMoveKind::CarryOutLeft => "carry-out-left",
            DribbleMoveKind::CarryOutRight => "carry-out-right",
            DribbleMoveKind::ProtectBall => "protect-ball",
            DribbleMoveKind::LeftCut => "left-cut",
            DribbleMoveKind::RightCut => "right-cut",
            DribbleMoveKind::Nutmeg => "nutmeg",
            DribbleMoveKind::XaviTurn => "xavi-turn",
            DribbleMoveKind::FakeLeftCutRight => "fake-left-cut-right",
            DribbleMoveKind::FakeRightCutLeft => "fake-right-cut-left",
        }
    }

    pub(crate) fn event_kind(self) -> &'static str {
        self.label()
    }

    pub(crate) fn beat_reward_points(self) -> f64 {
        match self {
            DribbleMoveKind::CarryForward
            | DribbleMoveKind::CarryOutLeft
            | DribbleMoveKind::CarryOutRight => DRIBBLE_BEAT_REWARD_POINTS * 0.72,
            DribbleMoveKind::ProtectBall => DRIBBLE_BEAT_REWARD_POINTS * 0.42,
            DribbleMoveKind::LeftCut | DribbleMoveKind::RightCut => DRIBBLE_BEAT_REWARD_POINTS,
            DribbleMoveKind::Nutmeg => NUTMEG_BEAT_REWARD_POINTS,
            DribbleMoveKind::FakeLeftCutRight | DribbleMoveKind::FakeRightCutLeft => {
                DRIBBLE_BEAT_REWARD_POINTS * 1.08
            }
            // Wheeling a defender with a Xavi turn cleanly beats them and buys time on
            // the ball — a full-value beat.
            DribbleMoveKind::XaviTurn => DRIBBLE_BEAT_REWARD_POINTS,
        }
    }
}

impl SoccerAction {
    fn label(&self) -> String {
        match self {
            SoccerAction::HoldShape => "hold".to_string(),
            SoccerAction::MoveTo(_) => "move".to_string(),
            SoccerAction::Dribble(_) => "dribble".to_string(),
            SoccerAction::DribbleMove { kind, .. } => kind.label().to_string(),
            SoccerAction::ControlTouch { .. } => "control-touch".to_string(),
            SoccerAction::Pass { flight, .. } => {
                if flight.is_aerial() {
                    "aerial-pass".to_string()
                } else {
                    "pass".to_string()
                }
            }
            SoccerAction::Clearance { .. } => "clearance".to_string(),
            SoccerAction::RouteOne { .. } => "route-one".to_string(),
            SoccerAction::Shoot { .. } => "shoot".to_string(),
            SoccerAction::Tackle { .. } => "tackle".to_string(),
            SoccerAction::SlideTackle { .. } => "slide-tackle".to_string(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlayerIntent {
    pub player_id: usize,
    pub action: SoccerAction,
    pub sprint: bool,
}

#[derive(Clone, Debug, Default)]
pub struct SoccerLearnedPlan {
    pub action: String,
    pub target_player: Option<usize>,
    pub target_point: Option<Vec2>,
    pub mpc_replan: Option<SoccerLearnedMpcReplanTrace>,
}

#[cfg(test)]
mod decision_refractory_tests {
    use super::{
        decision_refractory_blocks, DECISION_REFRACTORY_MIN_GAP_TICKS,
        DECISION_REFRACTORY_WINDOW_TICKS,
    };
    use std::collections::VecDeque;

    /// Drive the gate the way `apply_decision_refractory` does — at every tick the player *wants*
    /// to change its mind; record the ticks on which a changed decision is actually allowed to
    /// commit. The spec: at most 1 decision per 3 ticks AND 2 per 7-tick window, i.e. a
    /// 3,4,3,4-tick cadence with the third decision possible on the 8th tick.
    fn committed_ticks(span: u64) -> Vec<u64> {
        let mut commits: VecDeque<u64> = VecDeque::new();
        let mut allowed = Vec::new();
        for now in 0..span {
            // First decision (empty history) always commits; otherwise honour the gate.
            if commits.is_empty() || !decision_refractory_blocks(&commits, now) {
                allowed.push(now);
                commits.push_back(now);
                while commits.len() > 2 {
                    commits.pop_front();
                }
            }
        }
        allowed
    }

    #[test]
    fn refractory_yields_3_4_cadence_two_per_seven_ticks() {
        // Starting from tick 0: 0, +3, +4, +3, +4 ... = 0,3,7,10,14,17 over 18 ticks.
        assert_eq!(committed_ticks(18), vec![0, 3, 7, 10, 14, 17]);
    }

    #[test]
    fn refractory_caps_two_decisions_per_seven_tick_window() {
        let commits = committed_ticks(30);
        for &t in &commits {
            let in_window = commits
                .iter()
                .filter(|&&u| u <= t && t.saturating_sub(u) < DECISION_REFRACTORY_WINDOW_TICKS)
                .count();
            assert!(
                in_window <= 2,
                "tick {t}: {in_window} decisions within a {DECISION_REFRACTORY_WINDOW_TICKS}-tick window"
            );
        }
    }

    #[test]
    fn refractory_enforces_min_gap_between_consecutive_decisions() {
        let commits = committed_ticks(30);
        for pair in commits.windows(2) {
            assert!(
                pair[1] - pair[0] >= DECISION_REFRACTORY_MIN_GAP_TICKS,
                "consecutive decisions {} and {} closer than the min gap",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn third_decision_lands_on_the_eighth_tick_after_a_first_tick_start() {
        // Mirror the user's framing with a 1-based first decision on tick 1: 1, 4, 8.
        let mut commits: VecDeque<u64> = VecDeque::new();
        let mut allowed = Vec::new();
        for now in 1..=8 {
            if commits.is_empty() || !decision_refractory_blocks(&commits, now) {
                allowed.push(now);
                commits.push_back(now);
                while commits.len() > 2 {
                    commits.pop_front();
                }
            }
        }
        assert_eq!(allowed, vec![1, 4, 8]);
    }
}

#[cfg(test)]
mod forward_pass_first_tests {
    use super::{
        forward_pass_first_release_strength, FORWARD_PASS_FIRST_DWELL_FULL_SECONDS,
        FORWARD_PASS_FIRST_OPEN_THRESHOLD,
    };

    #[test]
    fn below_open_threshold_is_inert() {
        // A covered / square forward option carries no early-release pressure.
        assert_eq!(
            forward_pass_first_release_strength(FORWARD_PASS_FIRST_OPEN_THRESHOLD - 0.01, 2.0),
            0.0
        );
        assert_eq!(forward_pass_first_release_strength(0.0, 5.0), 0.0);
    }

    #[test]
    fn release_strength_climbs_with_dwell_and_openness() {
        let open = 0.8;
        let early = forward_pass_first_release_strength(open, 0.0);
        let mid = forward_pass_first_release_strength(open, FORWARD_PASS_FIRST_DWELL_FULL_SECONDS * 0.5);
        let late = forward_pass_first_release_strength(open, FORWARD_PASS_FIRST_DWELL_FULL_SECONDS);
        assert!(early > 0.0, "an open forward option pressures release immediately: {early}");
        assert!(mid > early && late > mid, "dwelling raises urgency: {early} {mid} {late}");
        // Saturates (doesn't keep climbing past the full-dwell point).
        let beyond = forward_pass_first_release_strength(open, FORWARD_PASS_FIRST_DWELL_FULL_SECONDS * 3.0);
        assert!((beyond - late).abs() < 1e-9, "dwell saturates: {late} vs {beyond}");
        // A more open receiver is a stronger release than a marginal one.
        let marginal = forward_pass_first_release_strength(FORWARD_PASS_FIRST_OPEN_THRESHOLD + 0.02, 1.0);
        let wide_open = forward_pass_first_release_strength(0.95, 1.0);
        assert!(wide_open > marginal, "more open ⇒ stronger: {marginal} vs {wide_open}");
        assert!((0.0..=1.0).contains(&wide_open));
    }
}

#[cfg(test)]
mod forward_option_recognition_tests {
    use crate::des::general::soccer::forward_pass_option_quality;

    #[test]
    fn never_reports_below_raw_openness() {
        // The revision only ADDS recognition; a legacy-open receiver is never downgraded,
        // whatever the completability (so no existing forward read is lost).
        for &open in &[0.0, 0.2, 0.45, 0.7, 1.0] {
            for &completion in &[0.0, 0.3, 0.6, 1.0] {
                let q = forward_pass_option_quality(open, completion);
                assert!(q >= open - 1e-9, "open={open} completion={completion} q={q}");
                assert!((0.0..=1.0).contains(&q), "q out of range: {q}");
            }
        }
    }

    #[test]
    fn clear_lane_lifts_a_moderately_open_runner_above_threshold() {
        // The defect case: an advanced runner with only MODERATE nearest-opponent openness
        // but a clear, completable lane should now read as a GOOD forward option (>= 0.5),
        // whereas raw openness alone (0.30) would have it dismissed.
        let raw_open = 0.30;
        let recognised = forward_pass_option_quality(raw_open, 0.95);
        assert!(raw_open < 0.45, "precondition: raw openness is below the release threshold");
        assert!(
            recognised >= 0.5,
            "a clear lane to an advanced runner is recognised: {recognised}"
        );
    }

    #[test]
    fn blocked_lane_does_not_manufacture_a_false_option() {
        // A receiver with space around them but a BLOCKED lane (low completion) must not be
        // inflated into a good option beyond what their own openness already implies.
        let open = 0.40;
        let q = forward_pass_option_quality(open, 0.05);
        assert!((q - open).abs() < 1e-9, "blocked lane keeps the raw openness: {q}");
    }

    #[test]
    fn rises_monotonically_with_completability() {
        let open = 0.30;
        let low = forward_pass_option_quality(open, 0.2);
        let mid = forward_pass_option_quality(open, 0.6);
        let high = forward_pass_option_quality(open, 0.95);
        assert!(low <= mid && mid <= high, "more completable ⇒ better: {low} {mid} {high}");
    }
}

#[cfg(test)]
mod shot_foot_choice_tests {
    use super::shot_foot_choice;
    use crate::des::general::soccer::{tunables, SkillProfile};

    fn right_footed() -> SkillProfile {
        let mut s = SkillProfile::default();
        s.right_foot_shot_power = 9.0;
        s.left_foot_shot_power = 4.0;
        s
    }

    #[test]
    fn neutral_geometry_picks_the_stronger_foot() {
        let skills = right_footed();
        // Ball centred on the body, no sweep ⇒ the power bias should win → right.
        let (foot, power, weak) = shot_foot_choice(&skills, 0.0, 0.0);
        assert_eq!(foot, "right");
        assert!((power - 9.0).abs() < 1e-9);
        assert!(!weak, "strong foot is never the weak foot");
    }

    #[test]
    fn ball_far_on_the_weak_side_forces_the_weak_foot_and_flags_it() {
        let skills = right_footed();
        // Ball a long way to the left of the body overrides the modest power bias.
        let (foot, power, weak) = shot_foot_choice(&skills, -20.0, -5.0);
        assert_eq!(foot, "left");
        assert!((power - 4.0).abs() < 1e-9);
        assert!(weak, "a forced weaker foot should be flagged for the execution damp");
    }

    #[test]
    fn weak_foot_damp_is_a_reduction() {
        let damp = tunables().shooting.shot_foot_weak_foot_execution_damp;
        assert!(damp > 0.0 && damp < 1.0);
    }
}
