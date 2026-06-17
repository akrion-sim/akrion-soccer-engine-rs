//! Player domain: the on-pitch [`PlayerAgent`] and its decision/skill model
//! (skills, preferences, the action space, intents, learned plans). Extracted
//! from the soccer god-module; see `super` for shared primitives and the world.

use super::*;

const DEFENDER_DRIBBLE_CLOSING_PASS_LIFT: f64 = 0.48;

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

    pub(crate) fn blended(seed: usize, role: PlayerRole, rng: &mut SeededRandom) -> Self {
        let role_bias = Self::role_bias_tuple(role);
        let jitter = |rng: &mut SeededRandom, scale: f64| (rng.next_float() - 0.5) * scale;
        let shooting = (role_bias.4 + jitter(rng, 1.1)).clamp(1.0, 10.0);
        let passing = (role_bias.5 + jitter(rng, 1.0)).clamp(1.0, 10.0);
        let dribbling = (role_bias.6 + jitter(rng, 1.0)).clamp(1.0, 10.0);
        let first_touch = (role_bias.7 + jitter(rng, 0.9)).clamp(1.0, 10.0);
        let defending = (role_bias.8 + jitter(rng, 1.0)).clamp(1.0, 10.0);
        let dominant_right = seed % 5 != 0;
        let strong_foot = (shooting + jitter(rng, 0.6)).clamp(1.0, 10.0);
        let weak_foot = (shooting - 1.1 + jitter(rng, 0.8)).clamp(1.0, 10.0);
        let (right_foot_shot_power, left_foot_shot_power) = if dominant_right {
            (strong_foot, weak_foot)
        } else {
            (weak_foot, strong_foot)
        };
        let crossing_left =
            (passing * 0.72 + first_touch * 0.18 + jitter(rng, 0.9)).clamp(1.0, 10.0);
        let crossing_right =
            (passing * 0.76 + first_touch * 0.16 + jitter(rng, 0.9)).clamp(1.0, 10.0);
        let strength = (role_bias.2 + jitter(rng, 0.9)).clamp(1.0, 10.0);
        let height = (role_bias.3 + jitter(rng, 0.9)).clamp(1.0, 10.0);
        let weight_pounds =
            default_player_weight_for_traits(role, strength, height, jitter(rng, 0.55));
        SkillProfile {
            top_speed: role_bias.0 + jitter(rng, 0.9) + (seed % 3) as f64 * 0.08,
            acceleration: role_bias.1 + jitter(rng, 0.8),
            strength,
            height,
            weight_pounds,
            shooting,
            right_foot_shot_power,
            left_foot_shot_power,
            passing,
            passing_completion_rate: (passing * 0.86 + first_touch * 0.12 + jitter(rng, 0.45))
                .clamp(1.0, 10.0),
            flair_passing: (passing * 0.58 + dribbling * 0.30 + jitter(rng, 0.9)).clamp(1.0, 10.0),
            crossing_left,
            crossing_right,
            dribbling,
            first_touch,
            defending,
            goalkeeping: (role_bias.12 + jitter(rng, 0.8)).clamp(1.0, 10.0),
            defensive_tracking: (role_bias.13 * 0.78 + defending * 0.22 + jitter(rng, 0.7))
                .clamp(1.0, 10.0),
            stamina: (role_bias.9 + jitter(rng, 0.8)).clamp(1.0, 10.0),
            vision: (role_bias.10 + jitter(rng, 0.8)).clamp(1.0, 10.0),
            decision_noise: (0.08 + jitter(rng, 0.08)).clamp(0.01, 0.18),
            aggression: (role_bias.11 + jitter(rng, 0.9)).clamp(1.0, 10.0),
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
    #[serde(default)]
    pub incoming_ball: Option<IncomingBallContext>,
    pub skills: SkillProfile,
    pub fatigue: f64,
    pub controller_slot: Option<usize>,
    pub preferences: AgentPreferences,
    pub last_decision: Option<AgentDecisionTrace>,
    /// The player's belief-grounded confidence in its current action, in `[0.2, 1.0]`
    /// (the engine sees truth; this is the player's read). Drives decisiveness (move
    /// speed) and proaction-vs-reaction. Recomputed each decision in `run_time_step`.
    #[serde(default = "default_unit_confidence")]
    pub decision_confidence: f64,
    /// Active one-two commitment after this player has played the "give" pass — see
    /// [`OneTwoRun`]. `None` when not engaged in a give-and-go.
    #[serde(default)]
    pub one_two: Option<OneTwoRun>,
}

const MDP_MPC_DEVIATION_TRACE_THRESHOLD_YARDS: f64 = 0.75;
const MDP_MPC_BLEND_MAX_TARGET_DELTA_YARDS: f64 = 6.0;

fn player_mdp_mpc_comparison_trace(
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
    let semantic_mismatch = mdp_target.is_none()
        || snapshot.ball.holder == Some(player.id)
        || player.role == PlayerRole::Goalkeeper
        || matches!(
            label,
            "shoot"
                | "pass"
                | "pass1"
                | "aerial-pass"
                | "aerial-pass1"
                | "clearance"
                | "route-one"
                | "tackle"
                | "control-touch"
        );
    let blend_eligible = !semantic_mismatch
        && target_delta_yards.is_finite()
        && target_delta_yards <= MDP_MPC_BLEND_MAX_TARGET_DELTA_YARDS
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
                | "defend"
                | "hold"
                | "recover"
                | "move"
        );
    let blended_target = if blend_eligible {
        mdp_target.map(|target| {
            ((target + mpc_target) * 0.5)
                .clamp_to_pitch(snapshot.field_width, snapshot.field_length)
        })
    } else {
        None
    };
    let deviation_recorded = target_delta_yards >= MDP_MPC_DEVIATION_TRACE_THRESHOLD_YARDS
        || velocity_delta_yps >= 1.0
        || semantic_mismatch;
    let decision = if blend_eligible {
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
        mpc_guidance_present: true,
        blend_eligible,
        deviation_recorded,
        decision: decision.to_string(),
    })
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
        let speculative_long_shot =
            speculative_long_shot_is_qualified(observation, self.role, shooting);
        let shot_legal =
            shot_decision_is_qualified_for_role(observation, self.role) || speculative_long_shot;
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
        let shot_score = (self.preferences.shoot_bias
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
            * 0.042)
            .clamp(
                0.004,
                0.12 + offensive_urgency * 0.30
                    + striker_shot_bonus * 0.18
                    + goal_attack * 0.22
                    + goal_proximity_shot_pressure * 0.36
                    + goal_entry_pressure * 0.20
                    + decisive_goal_pressure * 0.30,
            )
            .max(close_shot_attempt)
            .max(goal_proximity_shot_pressure_floor(
                observation,
                self.role,
                shooting,
            ))
            .max(attacking_goal_pressure_shot_attempt_probability(
                observation,
                self.role,
                shooting,
            ))
            .max(speculative_long_shot_attempt_probability(
                observation,
                self.role,
                shooting,
            ))
            .max(striker_shot_bonus)
            .max(if goal_attack_shot_blocks_alternatives {
                0.99
            } else {
                0.0
            });
        let fatigue_dribble = fatigue_dribble_multiplier(observation);
        let shot_creation_carry = shot_creation_carry_multiplier(observation);
        let patience_factor = low_pressure_patience_factor(observation);
        let floor_pass_quality = pass_quality_for_patience(observation, PassFlight::Floor);
        let aerial_pass_quality = pass_quality_for_patience(observation, PassFlight::Aerial);
        let pressured_release_signal = pressure_release_signal(observation);
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
        // Don't dribble straight into an opponent for no reason: OUTSIDE the final third
        // the carrier should keep ~2 yds of space ahead (1.5-2 yd buffer). In the final
        // attacking third more risk is worthwhile, so the guard relaxes there.
        let dribble_into_opponent_penalty = dribble_into_opponent_penalty(
            observation.forward_dribble_space_yards,
            observation.yards_to_goal,
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
        // Critical spacing: most carriers react once the nearest defender gets inside
        // the 2-yard floor; defenders on the ball should prefer a larger 3-4 yard buffer.
        // The response stays soft: dribbling is damped and passing lifted rather than
        // making carry options illegal.
        let defender_crowding =
            dribble_spacing_pressure_for_role(self.role, observation.nearest_opponent_distance);
        let crowded_dribble_damp =
            (1.0 - defender_crowding * DRIBBLE_CROWDED_SPACE_DAMP).clamp(0.30, 1.0);
        // The flip side: when a defender closes the spacing gap AND a receiver is open,
        // lift passing so the carrier releases the ball before the gap collapses. For
        // defenders in possession, a fast-closing opponent increases that release lift
        // even before the gap has dropped below the 2-yard floor.
        let open_outlet_fit = observation.best_pass_receiver_openness.clamp(0.0, 1.0);
        let defender_closing_pass_lift = if self.role == PlayerRole::Defender {
            (1.0 + pressure_rising * DEFENDER_DRIBBLE_CLOSING_PASS_LIFT * open_outlet_fit)
                .clamp(1.0, 1.42)
        } else {
            1.0
        };
        let crowded_pass_lift = (1.0
            + defender_crowding * PASS_CROWDED_RELEASE_LIFT * open_outlet_fit)
            * defender_closing_pass_lift;
        // --- Positional dribble-risk appetite -----------------------------------------
        // Driving INTO pressure (taking a man on, carrying into traffic) is only worth the
        // risk for the most-advanced players — the three furthest forward. See
        // `advanced_dribbler_fit_from_rank` for the rank→appetite curve.
        let advanced_dribbler_fit = advanced_dribbler_fit_from_rank(observation.teammates_ahead);
        // How risky a dribble is right now: real pressure, a defender on top of the ball, or
        // a live chance of being dispossessed.
        let dribble_risk = pressure_urgency
            .max(observation.perceived_pressure)
            .max(observation.immediate_dispossession_risk)
            .max(defender_crowding)
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
            * (1.0 - open_forward_outlet * 0.34).clamp(0.58, 1.0)
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
            .clamp(0.01, 1.46);
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
        // Lift the away-carry ceiling under escape pressure so a decisive break into
        // lateral space can outscore options that keep the holder pinned in the duel.
        let carry_out_escape_ceiling = (0.96 + escape_urgency * 0.20).clamp(0.96, 1.16);
        let carry_out_left_score = (carry_out_score
            * flank_drive_multiplier
            * (0.76 + (left_room / 18.0).clamp(0.0, 1.0) * 0.32)
            * (1.0 + escape_urgency * (left_room / 12.0).clamp(0.0, 1.0) * 0.55))
            .clamp(0.01, carry_out_escape_ceiling);
        let carry_out_right_score = (carry_out_score
            * flank_drive_multiplier
            * (0.76 + (right_room / 18.0).clamp(0.0, 1.0) * 0.32)
            * (1.0 + escape_urgency * (right_room / 12.0).clamp(0.0, 1.0) * 0.55))
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
            + pressured_no_outlet_shield * 0.30)
            .clamp(0.88, 1.55);
        let protect_ball_score = (dribble_score
            * (0.12
                + pressure_urgency.max(pressure) * 0.76
                + observation.immediate_dispossession_risk.clamp(0.0, 1.0) * 0.44
                + goal_side_shield * 0.46
                + fresh_receipt_shield * 0.55
                + pressured_no_outlet_shield * 0.60
                + own_half_retention
                // Rising pressure (a man bearing down) is exactly when you turn your
                // body between him and the ball -- lift the shield's urgency hard.
                + pressure_rising * 0.80
                + (1.0 - observation.perceived_time_on_ball_seconds / 2.8).clamp(0.0, 1.0) * 0.24)
            * keeper_carry_under_pressure_damp)
            .clamp(0.01, protect_ball_ceiling);
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
        let side_step_legal =
            observation.nearest_opponent_distance <= 4.6 && pressure_urgency.max(pressure) >= 0.28;
        // The side-step is the dedicated evade — knock the ball away from the
        // defender and step past. Under escape pressure it gets a real urgency boost
        // (and a lifted ceiling) so a pinned holder breaks contact instead of dwelling.
        let side_step_score = (dribble_score
            * (0.30 + pressure_urgency.max(pressure) * 0.88)
            * (1.0
                + (1.0 - observation.forward_dribble_space_yards / 14.0).clamp(0.0, 1.0) * 0.24
                + escape_urgency * 0.55)
            * calm_pass_focus
            * keeper_carry_under_pressure_damp)
            .clamp(0.01, (0.82 + escape_urgency * 0.26).clamp(0.82, 1.08));
        let feint_legal =
            observation.nearest_opponent_distance <= 5.2 && pressure_urgency.max(pressure) >= 0.34;
        // Feints are already pressure-gated (feint_legal). Keep the ceiling low so a
        // carrier does not throw a feint every other tick — fewer feints per second,
        // reserved for when they're genuinely needed under pressure. The calm-pass-focus
        // damp suppresses them further when an unpressured holder has a clean pass on.
        let feint_score = (dribble_score
            * (0.22 + pressure_urgency * 0.66 + decision_urgency * 0.18)
            * (0.78 + dribbling * 0.28)
            * calm_pass_focus
            * keeper_carry_under_pressure_damp)
            .clamp(0.01, 0.55);
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
        let flank_cross_context =
            flank_cross_context_score(observation, self.position, field_width);
        let flank_cross_legal_context =
            flank_cross_context_is_legal(observation, self.position, field_width)
                && !goal_attack_shot_blocks_alternatives
                && (!goalmouth_carry_forced || directive.flank_attack_policy.is_flank());
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
            * pressured_release_multiplier(observation)
            * low_cross_multiplier)
            .clamp(0.01, 0.86);
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
            * pressured_release_multiplier(observation)
            * high_cross_multiplier)
            .clamp(0.01, 0.78);
        options.push(AgentActionOptionTrace::new(
            "flank-high-cross",
            high_cross_score,
            aerial_pass_target_count > 0 && flank_cross_legal_context,
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
            * near_goal_pass_multiplier.max(0.72)
            * floor_pass_patience_multiplier
            * hold_release_multiplier
            * crowded_pass_lift
            * pressured_release_multiplier(observation))
        .clamp(0.01, 1.68);
        options.push(AgentActionOptionTrace::new(
            "killer-pass",
            killer_pass_score,
            killer_pass_legal,
        ));
        for rank in 0..pass_target_count.min(3) {
            let rank_weight = match rank {
                0 => 1.00,
                1 => 0.74,
                _ => 0.56,
            };
            let quick_release = (1.35 - observation.perceived_time_on_ball_seconds)
                .max(0.0)
                .min(1.0);
            let completion_bonus = (0.82
                + observation.expected_pass_completion.clamp(0.0, 1.0) * 0.24
                + observation.best_pass_receiver_openness.clamp(0.0, 1.0) * 0.16)
                .clamp(0.76, 1.22);
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
                * crowded_pass_lift
                * pressured_release_multiplier(observation)
                * rank_weight)
                .clamp(0.004, hold_release_score_cap);
            options.push(AgentActionOptionTrace::new(
                format!("pass{}", rank + 1),
                pass_score,
                true,
            ));
        }
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
                * pressured_release_multiplier(observation)
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
        if clearance_legal && hold_pressure >= 0.16 && release_pressure >= 0.42 {
            let danger_clearance_floor =
                (0.30 + hold_pressure * 0.22 + release_pressure * 0.10).clamp(0.34, 0.72);
            ensure_min_legal_option_probability(&mut options, "clearance", danger_clearance_floor);
        }
        let shot_floor = near_goal_shot_pressure_floor(observation, self.role, shooting).max(
            goal_proximity_shot_pressure_floor(observation, self.role, shooting),
        );
        if shot_floor > 0.0 {
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
        if carry_forward_legal
            && !goal_attack_shot_blocks_alternatives
            && observation.yards_to_goal < observation.yards_to_own_goal
            && observation.yards_to_goal <= 58.0
            && observation.forward_dribble_space_yards >= 3.0
            && release_pressure < 0.50
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
                .clamp(0.0, 0.54);
            ensure_min_legal_option_probability(&mut options, "carry-forward", progression_floor);
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
                "carry-out-left",
                "carry-out-right",
                "protect-ball",
                "side-step",
                "fake-left-cut-right",
                "fake-right-cut-left",
                "hold-up-flank",
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
            ensure_min_legal_option_family_probability(
                &mut options,
                &["shoot", "killer-pass"],
                decisive_family_floor,
            );
        }
        let mut options = normalize_action_options(options);
        annotate_tick_probabilities_from_scores(&mut options, dt_seconds);
        options
    }

    pub(crate) fn first_touch_action_options(
        &self,
        observation: &SoccerPomdpObservation,
        pass_target_count: usize,
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
        let quick_pressure_bonus = 1.0
            + observation.perceived_pressure.clamp(0.0, 1.0) * 0.24
            + observation.decision_urgency.clamp(0.0, 1.0) * 0.18;
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
                (observation.first_time_shot_score * quick_pressure_bonus * 0.52).clamp(0.01, 0.64),
                shot_legal,
            ),
            AgentActionOptionTrace::new("killer-pass", killer_pass_score, killer_pass_legal),
            AgentActionOptionTrace::new(
                "first-time-pass",
                (observation.first_time_pass_score * quick_pressure_bonus * 0.86).clamp(0.02, 0.88),
                pass_legal,
            ),
            AgentActionOptionTrace::new(
                control_label,
                (observation.control_touch_score * (1.12 - observation.perceived_pressure * 0.18))
                    .clamp(0.02, 0.95),
                true,
            ),
        ];
        if killer_pass_legal {
            let floor = (0.24 + killer_pressure * 0.50 + threaded_quality * 0.12).clamp(0.34, 0.82);
            ensure_min_legal_option_probability(&mut options, "killer-pass", floor);
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
            if pressure >= 0.45 && pass_quality >= 0.5 {
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
        let possession_team = snapshot.controlled_possession_team();
        let flank_policy_active =
            possession_team == Some(self.team) && directive.flank_attack_policy.is_flank();
        // A striker holding up in the opponent half, OR a midfielder/striker carrying the
        // ball and driving at goal, both pull the other attackers into roaming runs to find
        // space in behind — the latter is the urgency cue for a teammate running at the
        // defence (not just a settled forward).
        let striker_attack = snapshot
            .striker_holder_in_opponent_half(self.team)
            .is_some()
            || snapshot
                .attacking_carrier_driving_at_goal(self.team)
                .is_some()
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
        let special_targets = SupportSpecialTargets {
            check_to_ball: snapshot.check_to_ball_target_for(self.id, self.home_position),
            in_behind: snapshot.in_behind_run_target_for(self.id),
            wide_outlet: snapshot.wide_possession_outlet_target_for(self.id, self.home_position),
            flank_cross_arrival: snapshot
                .flank_cross_arrival_target_for(self.id, self.home_position),
        };
        let current = snapshot.player_position(self.id).unwrap_or(self.position);
        let open = snapshot.open_space_for(self.id, self.home_position);
        let guarded_special_target = |target: Vec2| {
            snapshot.shape_guarded_support_point(
                self.id,
                target,
                &[open, self.home_position],
                self.home_position,
                false,
            )
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
                    * (1.0 + support_urgency * 0.14),
                true,
            ),
            AgentActionOptionTrace::new(
                "support-roam",
                roam_weight * self.preferences.open_space_bias * (1.0 + support_urgency * 0.18),
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
                        + shape_support_urgency * 2.0
                        + holder_pressure_urgency * 0.58,
                ),
                true,
            ));
        }
        if let Some(target) = special_targets.in_behind {
            let target = guarded_special_target(target);
            options.push(AgentActionOptionTrace::new(
                "run-in-behind",
                special_score(
                    target,
                    0.68 + shape_support_urgency * 0.10 + holder_pressure_urgency * 0.18,
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
                        + shape_support_urgency * 0.20
                        + holder_pressure_urgency * 0.34
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
            options.push(AgentActionOptionTrace::new(
                "shot-creation-run",
                special_score(
                    target,
                    0.64 + goalward.max(0.0) * 0.026
                        + centrality * 0.14
                        + shape_support_urgency * 0.20
                        + holder_pressure_urgency * 0.18
                        + directive.flank_overlap_run_probability * 0.22,
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
        let home_x = self.home_position.x.clamp(0.0, snapshot.field_width);
        let wide_home =
            home_x < snapshot.field_width * 0.34 || home_x > snapshot.field_width * 0.66;
        let policy_overlap_candidate = flank_policy_active
            && self.role != PlayerRole::Goalkeeper
            && (wide_home
                || wide_midfielder
                || attacking_midfielder
                || self.role == PlayerRole::Forward);
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
            let min_share = if flank_policy_active {
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
        if options
            .iter()
            .any(|option| option.legal && option.label == "wide-outlet")
        {
            let wide_floor = (0.12
                + shape_support_urgency * 0.08
                + holder_pressure_urgency * 0.10
                + if flank_policy_active { 0.10 } else { 0.0 })
            .clamp(0.16, 0.34);
            ensure_min_legal_option_probability(&mut options, "wide-outlet", wide_floor);
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
                    + holder_pressure_urgency * 0.20
                    + if flank_policy_active { 0.08 } else { 0.0 })
                .clamp(0.26, 0.48);
                ensure_min_legal_option_probability(&mut options, "wide-outlet", wide_floor);
            }
        }
        let options = normalize_action_options(options);
        SupportActionContext {
            options,
            special_targets,
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
        let mut options = vec![
            AgentActionOptionTrace::new("tackle", tackle_score, tackle_legal),
            AgentActionOptionTrace::new("defend-shape", shape_score, true),
            AgentActionOptionTrace::new("defend-roam", roam_score, true),
        ];
        if self.role == PlayerRole::Goalkeeper {
            ensure_min_legal_option_probability(&mut options, "defend-shape", 0.97);
        }
        let mut options = normalize_action_options(options);
        annotate_tick_probability_from_score(&mut options, "tackle", dt_seconds);
        options
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
    ) -> Vec<AgentActionOptionTrace> {
        if fifty_fifty_duel {
            return single_action_option("fifty-fifty-duel");
        }

        let recover_legal = closer_teammates < 2 && my_distance <= 46.0 && goalkeeper_can_recover;
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
            _ => 0.0,
        };
        let recover_score = if recover_legal {
            (0.34
                + distance_fit * 0.46
                + teammate_priority * 0.22
                + self.preferences.offensive_mindedness.clamp(0.0, 1.0) * 0.08)
                * role_fit
        } else {
            0.0
        }
        .clamp(0.0, 1.10);
        let hold_score = if recover_legal {
            (0.16
                + self.preferences.defensive_mindedness.clamp(0.0, 1.0) * 0.10
                + (1.0 - teammate_priority) * 0.12)
                .clamp(0.04, 0.34)
        } else {
            1.0
        };

        normalize_action_options(vec![
            AgentActionOptionTrace::new("recover", recover_score, recover_legal),
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
        let mdp_mpc_comparison =
            player_mdp_mpc_comparison_trace(snapshot, self, &action_target, &action_label);
        AgentDecisionTrace {
            mdp_state,
            observation,
            belief,
            operation_order,
            scheduled_index,
            action_options,
            action_target,
            mdp_mpc_comparison,
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
            SoccerAction::Tackle { target_player } => {
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
        Some((
            SoccerAction::Pass {
                target_player: target,
                power,
                flight,
            },
            restart_label.to_string(),
        ))
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
        self.run_time_step_with_context(
            snapshot,
            mdp_state,
            observation,
            human_input,
            learned_plan,
            rng,
        )
    }

    pub fn run_time_step_with_context(
        &mut self,
        snapshot: &WorldSnapshot,
        mdp_state: SoccerMdpState,
        mut observation: SoccerPomdpObservation,
        human_input: Option<&HumanInputFrame>,
        learned_plan: Option<&SoccerLearnedPlan>,
        rng: &mut SeededRandom,
    ) -> PlayerIntent {
        let human_input = human_input
            .filter(|input| human_input_matches_player(input, self.id, self.controller_slot));
        annotate_human_control_observation(&mut observation, self.controller_slot, human_input);
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
                let distance = self.position.distance(assignment.target);
                if distance > 0.35 {
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
                    self.last_decision = Some(self.decision_trace(
                        snapshot,
                        mdp_state,
                        observation,
                        belief,
                        vec!["ball-at-feet-reflex".to_string()],
                        single_action_option("control-ball-at-feet"),
                        &action,
                        "control-ball-at-feet",
                    ));
                    return PlayerIntent {
                        player_id: self.id,
                        action,
                        sprint: false,
                    };
                }
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
            // WALL RETURN (the "two" of a one-two): return the lay-off before
            // generic first-touch logic turns the combination into an ordinary pass.
            if let Some(runner) = snapshot.wall_return_pass_target_for(self.id) {
                let return_reliability = (0.70
                    + ability01(self.skills.first_touch) * 0.20
                    + ability01(self.skills.passing) * 0.10)
                    .clamp(0.0, 0.97);
                if rng.next_float()
                    < time_window_probability(return_reliability, snapshot.dt_seconds)
                {
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
            }

            // Named one-two / give-and-go strategies commit to the wall pass before
            // first-touch, killer/scoop, or generic pass selection can swallow it.
            if let Some(plan) = snapshot.wall_pass_option_for(self.id) {
                let give_and_go_strategy = matches!(
                    directive.attack_strategy,
                    TeamAttackStrategy::GiveAndGoCentral
                        | TeamAttackStrategy::OneTwoLeftRelease
                        | TeamAttackStrategy::OneTwoRightRelease
                        | TeamAttackStrategy::CentralDoubleOneTwo
                );
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
            let action_options = self.first_touch_action_options(&observation, pass_targets.len());
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
            let weighted_ops = action_options
                .iter()
                .map(|option| {
                    (
                        option.label.clone(),
                        if option.legal { option.score } else { 0.0 },
                    )
                })
                .collect::<Vec<_>>();
            let ops = weighted_fisher_yates_order(weighted_ops, rng);
            let mut order_names = Vec::with_capacity(ops.len());
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
                        chosen = Some((
                            SoccerAction::Shoot {
                                power: shot_power_for_finish_skill(finish_skill),
                            },
                            normalize_soccer_action_label(&op).to_string(),
                        ));
                        break;
                    }
                    "killer-pass" if !pass_targets.is_empty() => {
                        if let Some(target) =
                            snapshot.killer_pass_target_for(self.id, &pass_targets)
                        {
                            chosen = Some((
                                SoccerAction::Pass {
                                    target_player: Some(target),
                                    power: 0.66
                                        + 0.24 * passing_skill.max(ability01(self.skills.vision)),
                                    flight: PassFlight::Floor,
                                },
                                "killer-pass".to_string(),
                            ));
                            break;
                        }
                    }
                    "first-time-pass" if !pass_targets.is_empty() => {
                        let target = pass_targets[0];
                        chosen = Some((
                            SoccerAction::Pass {
                                target_player: Some(target),
                                power: 0.54 + 0.30 * passing_skill,
                                flight: PassFlight::Floor,
                            },
                            "first-time-pass".to_string(),
                        ));
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
            let (action, action_label) = chosen.unwrap_or_else(|| {
                let target = self.control_touch_target_for_goal_context(snapshot, &observation);
                (
                    SoccerAction::ControlTouch { target },
                    "control-touch".to_string(),
                )
            });
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
                sprint: false,
            };
        }

        if has_ball
            && goal_attack_shot_blocks_alternatives(&observation, self.role)
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
        } else if has_ball && striker_shot_window_is_qualified(&observation, self.role) {
            let striker_shot_bonus = striker_legal_shot_attempt_bonus(&observation, self.role);
            let window_shot_chance = (0.66 + shooting_skill * 0.14 + striker_shot_bonus * 0.18
                - observation.shot_block_probability.clamp(0.0, 1.0) * 0.08)
                .clamp(0.68, 0.94);
            if rng.next_float() < window_shot_chance {
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
            && shot_decision_is_qualified_for_role(&observation, self.role)
            && !threaded_goal_pass_can_override_forced_shot(&observation, self.role)
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
                .max(striker_shot_bonus);
            if rng.next_float() < time_window_probability(finish_chance, snapshot.dt_seconds) {
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
            if let Some(target) = snapshot.long_ball_in_behind_target(self.id) {
                let crossing = ability01(self.skills.crossing_left.max(self.skills.crossing_right));
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
                if rng.next_float() < time_window_probability(long_ball_chance, snapshot.dt_seconds)
                {
                    let action = SoccerAction::Pass {
                        target_player: Some(target),
                        power: 0.80 + 0.14 * crossing.max(passing_skill),
                        flight: PassFlight::Aerial,
                    };
                    self.last_decision = Some(self.decision_trace(
                        snapshot,
                        mdp_state,
                        observation,
                        belief,
                        vec!["long-ball-in-behind".to_string()],
                        single_action_option("aerial-pass"),
                        &action,
                        "aerial-pass",
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
            if let Some((target, sprint)) = snapshot.pending_pass_reception_target_for(self.id) {
                let action = SoccerAction::MoveTo(target);
                self.last_decision = Some(self.decision_trace(
                    snapshot,
                    mdp_state,
                    observation,
                    belief,
                    vec!["receive-pending-pass".to_string()],
                    single_action_option("recover"),
                    &action,
                    "recover",
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
                    vec!["anticipate-pass".to_string(), "contest-reception".to_string()],
                    single_action_option("recover"),
                    &action,
                    "recover",
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

        if !has_ball {
            if let Some((target, sprint)) = snapshot.loose_long_ball_recovery_intent_for(self.id) {
                let action = SoccerAction::MoveTo(target);
                self.last_decision = Some(self.decision_trace(
                    snapshot,
                    mdp_state,
                    observation,
                    belief,
                    vec!["long-ball-duel".to_string(), "recover".to_string()],
                    single_action_option("recover"),
                    &action,
                    "recover",
                ));
                return PlayerIntent {
                    player_id: self.id,
                    action,
                    sprint,
                };
            }
        }

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
                        && rng.next_float()
                            >= time_window_probability(learned_shot_chance, snapshot.dt_seconds)
                    {
                        let kind = if goal_approach_carry_preferred(&observation, self.role) {
                            DribbleMoveKind::CarryForward
                        } else {
                            deterministic_dribble_move_kind(snapshot.tick, self.id)
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
                        self.last_decision = Some(self.decision_trace(
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
                        ));
                        return PlayerIntent {
                            player_id: self.id,
                            action: defer_action,
                            sprint: false,
                        };
                    }
                }
                self.last_decision = Some(self.decision_trace(
                    snapshot,
                    mdp_state,
                    observation,
                    belief,
                    vec!["learned-policy".to_string(), plan.action.clone()],
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
        }

        if has_ball {
            let pass_targets = snapshot.ranked_visible_pass_targets(self.id, 3);
            let aerial_pass_targets = snapshot.ranked_visible_aerial_pass_targets(self.id, 3);
            let hold_up_flank_target = snapshot.striker_hold_up_flank_target_for(self.id);
            let action_options = self.possession_action_options(
                &observation,
                &directive,
                pass_targets.len(),
                aerial_pass_targets.len(),
                hold_up_flank_target.is_some(),
                snapshot.dt_seconds,
                snapshot.field_width,
            );
            if let Some(target) = snapshot.killer_pass_target_for(self.id, &pass_targets) {
                let preempt_chance = final_third_killer_pass_preemption_probability(
                    &observation,
                    self.role,
                    &action_options,
                );
                if preempt_chance > 0.0 && rng.next_float() < preempt_chance {
                    let action = SoccerAction::Pass {
                        target_player: Some(target),
                        power: 0.68 + 0.24 * passing_skill.max(ability01(self.skills.vision)),
                        flight: PassFlight::Floor,
                    };
                    self.last_decision = Some(self.decision_trace(
                        snapshot,
                        mdp_state,
                        observation,
                        belief,
                        vec![
                            "decisive-goal-pressure".to_string(),
                            "killer-pass".to_string(),
                        ],
                        action_options,
                        &action,
                        "killer-pass",
                    ));
                    return PlayerIntent {
                        player_id: self.id,
                        action,
                        sprint: false,
                    };
                }
            }
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
                    "killer-pass".to_string(),
                    action_option_score(&action_options, "killer-pass"),
                ),
            ];
            for rank in 0..pass_targets.len() {
                let label = format!("pass{}", rank + 1);
                weighted_ops.push((label.clone(), action_option_score(&action_options, &label)));
            }
            for rank in 0..aerial_pass_targets.len() {
                let label = format!("aerial-pass{}", rank + 1);
                weighted_ops.push((label.clone(), action_option_score(&action_options, &label)));
            }
            // SCOOP: a clear lofted-dink chance — the carrier is surrounded and an open
            // teammate sits 5-8yd away, in the facing cone, behind a defender in the lane.
            // Take it ahead of the general option roll, gated by technique so only capable
            // players attempt the delicate chip.
            if let Some(scoop_target) = snapshot.scoop_pass_target_for(self.id) {
                let technique = ability01(self.skills.flair_passing) * 0.5
                    + ability01(self.skills.passing) * 0.3
                    + ability01(self.skills.vision) * 0.2;
                let scoop_chance = (0.22 + technique * 0.56).clamp(0.0, 0.80);
                if rng.next_float() < scoop_chance {
                    let action = SoccerAction::Pass {
                        target_player: Some(scoop_target),
                        power: 0.5,
                        flight: PassFlight::Scoop,
                    };
                    self.last_decision = Some(self.decision_trace(
                        snapshot,
                        mdp_state.clone(),
                        observation.clone(),
                        belief.clone(),
                        vec!["scoop-pass".to_string()],
                        single_action_option("scoop-pass"),
                        &action,
                        "scoop-pass",
                    ));
                    return PlayerIntent {
                        player_id: self.id,
                        action,
                        sprint: false,
                    };
                }
            }
            // WALL PASS (one-two / give-and-go): with a beatable man in front, a free
            // side-on teammate, and onside space to run into, lay the ball off and burst
            // past him — a combination to a shot is usually the better route than going it
            // alone. Genuinely tried (not rare): appetite scales with the quality of the
            // combination, with pressure on the carrier (relief + penetration), and with
            // goal proximity, and is lifted hard when the team's active maneuver IS a
            // give-and-go / one-two. On commit, the give is played AND this player takes on
            // the runner role (the "go"), bursting onside to receive the first-time return.
            if let Some(plan) = snapshot.wall_pass_option_for(self.id) {
                let give_and_go_strategy = matches!(
                    directive.attack_strategy,
                    TeamAttackStrategy::GiveAndGoCentral
                        | TeamAttackStrategy::OneTwoLeftRelease
                        | TeamAttackStrategy::OneTwoRightRelease
                        | TeamAttackStrategy::CentralDoubleOneTwo
                );
                let goal_proximity = (1.0
                    - (observation.yards_to_goal / WALL_PASS_GOAL_PROXIMITY_REFERENCE_YARDS)
                        .clamp(0.0, 1.0))
                .clamp(0.0, 1.0);
                let raw_wall_appetite = WALL_PASS_BASE_APPETITE
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
                if strategy_commits
                    || rng.next_float() < time_window_probability(wall_appetite, snapshot.dt_seconds)
                {
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
            let ops = weighted_fisher_yates_order(weighted_ops, rng);
            let mut order_names = Vec::with_capacity(ops.len());
            let mut chosen = None;
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
                            && rng.next_float()
                                < time_window_probability(shot_chance, snapshot.dt_seconds)
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
                        if rng.next_float()
                            < time_window_probability(dribble_chance, snapshot.dt_seconds)
                        {
                            let kind =
                                if goal_approach_forces_goalmouth_carry(&observation, self.role) {
                                    DribbleMoveKind::CarryForward
                                } else {
                                    choose_dribble_move_kind(rng)
                                };
                            let touch =
                                snapshot.choose_dribble_touch_decision_for(self.id, kind, rng);
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
                        if rng.next_float()
                            < time_window_probability(carry_chance, snapshot.dt_seconds)
                        {
                            let touch =
                                snapshot.choose_dribble_touch_decision_for(self.id, kind, rng);
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
                    "fake-left-cut-right" | "fake-right-cut-left" => {
                        let kind = if op == "fake-left-cut-right" {
                            DribbleMoveKind::FakeLeftCutRight
                        } else {
                            DribbleMoveKind::FakeRightCutLeft
                        };
                        order_names.push(kind.label().to_string());
                        let feint_chance = action_option_score(&action_options, kind.label());
                        if rng.next_float()
                            < time_window_probability(feint_chance, snapshot.dt_seconds)
                        {
                            let touch =
                                snapshot.choose_dribble_touch_decision_for(self.id, kind, rng);
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
                    "side-step" => {
                        order_names.push("side-step".to_string());
                        let side_step_chance = action_option_score(&action_options, "side-step");
                        if rng.next_float()
                            < time_window_probability(side_step_chance, snapshot.dt_seconds)
                        {
                            let kind = snapshot.side_step_dribble_kind_for(self.id);
                            let touch =
                                snapshot.choose_dribble_touch_decision_for(self.id, kind, rng);
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
                            if rng.next_float()
                                < time_window_probability(hold_up_chance, snapshot.dt_seconds)
                            {
                                let kind = snapshot.side_step_dribble_kind_for(self.id);
                                let bucket = match dribble_final_cut_kind(kind) {
                                    DribbleMoveKind::CarryForward
                                    | DribbleMoveKind::ProtectBall => 0,
                                    DribbleMoveKind::CarryOutLeft | DribbleMoveKind::LeftCut => 9,
                                    DribbleMoveKind::CarryOutRight | DribbleMoveKind::RightCut => 3,
                                    DribbleMoveKind::Nutmeg => 0,
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
                        if rng.next_float()
                            < time_window_probability(clearance_chance, snapshot.dt_seconds)
                        {
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
                        if rng.next_float()
                            < time_window_probability(route_one_chance, snapshot.dt_seconds)
                        {
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
                            if rng.next_float()
                                < time_window_probability(cross_chance, snapshot.dt_seconds)
                            {
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
                            if rng.next_float()
                                < time_window_probability(cross_chance, snapshot.dt_seconds)
                            {
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
                    "killer-pass" => {
                        order_names.push("killer-pass".to_string());
                        let killer_chance = action_option_score(&action_options, "killer-pass");
                        if let Some(target) =
                            snapshot.killer_pass_target_for(self.id, &pass_targets)
                        {
                            if rng.next_float()
                                < time_window_probability(killer_chance, snapshot.dt_seconds)
                            {
                                chosen = Some((
                                    SoccerAction::Pass {
                                        target_player: Some(target),
                                        power: 0.66
                                            + 0.24
                                                * passing_skill.max(ability01(self.skills.vision)),
                                        flight: PassFlight::Floor,
                                    },
                                    "killer-pass".to_string(),
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
                        if rng.next_float()
                            < time_window_probability(pass_chance, snapshot.dt_seconds)
                        {
                            let target = pass_targets[rank];
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
                        if rng.next_float()
                            < time_window_probability(pass_chance, snapshot.dt_seconds)
                        {
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

            let forced_killer_pass_target =
                if killer_pass_forced_by_goal_pressure(&observation, self.role) {
                    snapshot.killer_pass_target_for(self.id, &pass_targets)
                } else {
                    None
                };
            let (action, action_label) = chosen.unwrap_or_else(|| {
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
                            power: 0.68 + 0.24 * passing_skill.max(ability01(self.skills.vision)),
                            flight: PassFlight::Floor,
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
                            power: 0.68 + 0.24 * passing_skill.max(ability01(self.skills.vision)),
                            flight: PassFlight::Floor,
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
                    } else if patient_carry {
                        patient_carry_dribble_kind(snapshot.tick, self.id)
                    } else {
                        deterministic_dribble_move_kind(snapshot.tick, self.id)
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
                    let kind = deterministic_dribble_move_kind(snapshot.tick, self.id);
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
            let sprint = matches!(action, SoccerAction::DribbleMove { .. })
                && (self.role == PlayerRole::Forward
                    || observation.forward_dribble_space_yards > 3.0
                    || observation.perceived_pressure > 0.35
                    || observation.decision_urgency > 0.46
                    || observation.offensive_urgency > 0.34);
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
        // Initialized empty so every decision path is definitely-assigned (one of the
        // branches below overwrites it); a path that doesn't just records no options.
        let mut action_options: Vec<AgentActionOptionTrace> = Vec::new();
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
            (SoccerAction::MoveTo(burst_target), "one-two-run".to_string())
        } else if possession_team == Some(self.team) {
            let support_context = self.support_action_context(snapshot);
            action_options = support_context.options.clone();
            let support_order = weighted_fisher_yates_order(
                action_options
                    .iter()
                    .map(|option| {
                        (
                            option.label.clone(),
                            if option.legal { option.score } else { 0.0 },
                        )
                    })
                    .collect(),
                rng,
            );
            let first_support_label = support_order
                .first()
                .map(|label| normalize_soccer_action_label(label.as_str()))
                .unwrap_or("support-shape")
                .to_string();
            order_names.extend(support_order);
            let support_open = snapshot.open_space_for(self.id, self.home_position);
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
                        matches!(first_support_label.as_str(), "support-roam" | "overlap-run"),
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
            let ops = weighted_fisher_yates_order(
                vec![
                    ("tackle", action_option_score(&action_options, "tackle")),
                    (
                        "defend-shape",
                        action_option_score(&action_options, "defend-shape"),
                    ),
                    (
                        "defend-roam",
                        action_option_score(&action_options, "defend-roam"),
                    ),
                ],
                rng,
            );
            let mut chosen = None;
            for op in ops {
                match op {
                    "tackle" => {
                        order_names.push("tackle".to_string());
                        if let Some(holder) = snapshot.ball.holder {
                            let holder_is_opponent = snapshot
                                .players
                                .iter()
                                .find(|p| p.id == holder)
                                .is_some_and(|p| p.team == self.team.other());
                            if holder_is_opponent
                                && self.position.distance(snapshot.ball.position) < 3.1
                                && (self.role != PlayerRole::Goalkeeper
                                    || snapshot.goalkeeper_direct_intervention_is_safe(self.id))
                                && rng.next_float()
                                    < time_window_probability(
                                        ((defending_skill * 0.6 + aggression_skill * 0.4)
                                            * directive.press_intensity
                                            * snapshot.advancing_carrier_steal_urgency(self.id))
                                        .clamp(0.02, 0.97),
                                        snapshot.dt_seconds,
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
                    "defend-shape" | "defend-roam" => {
                        let roam = op == "defend-roam";
                        order_names.push(op.to_string());
                        let dist = self.position.distance(snapshot.ball.position);
                        let defend_radius = 3.0 + directive.press_intensity * 3.0;
                        let target = if self.role == PlayerRole::Goalkeeper {
                            snapshot.defensive_assignment_for(self.id, self.home_position, roam)
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
                        chosen = Some((SoccerAction::MoveTo(target), "defend".to_string()));
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
            );
            let loose_order = weighted_fisher_yates_order(
                loose_options
                    .iter()
                    .map(|option| {
                        (
                            option.label.clone(),
                            if option.legal { option.score } else { 0.0 },
                        )
                    })
                    .collect(),
                rng,
            );
            let should_recover =
                fifty_fifty_duel || action_option_score(&loose_options, "recover") > 0.0;
            action_options = loose_options;
            order_names.extend(loose_order);
            if should_recover {
                // Win the race back to a ball that got away under pressure (notably
                // the carrier's own long/heavy touch): if it's meaningfully ahead and
                // an opponent is contesting / closing on it, sprint to it.
                let nearest_opponent_to_ball = snapshot
                    .players
                    .iter()
                    .filter(|player| {
                        player.team != self.team && player.role != PlayerRole::Goalkeeper
                    })
                    .map(|player| player.position.distance(loose_ball_target))
                    .fold(f64::INFINITY, f64::min);
                loose_ball_pressured_sprint = self.role != PlayerRole::Goalkeeper
                    && my_distance > LOOSE_BALL_PRESSURED_SPRINT_MIN_DISTANCE_YARDS
                    && (fifty_fifty_duel
                        || nearest_opponent_to_ball
                            <= LOOSE_BALL_PRESSURED_SPRINT_OPPONENT_RADIUS_YARDS);
                (
                    SoccerAction::MoveTo(loose_ball_target),
                    "recover".to_string(),
                )
            } else {
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
                            "run-in-behind" | "one-two-run"
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
                                    | "shot-creation-run"
                                    | "overlap-run"
                                    | "support-push-up"
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
            SoccerAction::MoveTo(target) if action_label == "recover" => {
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

    pub(crate) fn action_from_learned_plan(
        &self,
        plan: &SoccerLearnedPlan,
        snapshot: &WorldSnapshot,
        observation: &SoccerPomdpObservation,
    ) -> Option<(SoccerAction, String)> {
        let label = normalize_soccer_action_label(&plan.action);
        if !observation.has_ball
            && snapshot.controlled_possession_team() == Some(self.team.other())
            && label != "tackle"
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
                            flight: PassFlight::Floor,
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
                            flight: PassFlight::Floor,
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
                let target = snapshot
                    .ranked_visible_pass_targets(self.id, 1)
                    .first()
                    .copied();
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
            "dribble"
            | "carry-forward"
            | "carry-out-left"
            | "carry-out-right"
            | "protect-ball"
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
                    "carry-forward" => DribbleMoveKind::CarryForward,
                    "carry-out-left" => DribbleMoveKind::CarryOutLeft,
                    "carry-out-right" => DribbleMoveKind::CarryOutRight,
                    "protect-ball" => DribbleMoveKind::ProtectBall,
                    "left-cut" => DribbleMoveKind::LeftCut,
                    "right-cut" => DribbleMoveKind::RightCut,
                    "nutmeg" => DribbleMoveKind::Nutmeg,
                    "fake-left-cut-right" => DribbleMoveKind::FakeLeftCutRight,
                    "fake-right-cut-left" => DribbleMoveKind::FakeRightCutLeft,
                    _ => deterministic_dribble_move_kind(snapshot.tick, self.id),
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
                    kind.label().to_string(),
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
            "recover" if snapshot.controlled_possession_team().is_none() => Some((
                SoccerAction::MoveTo(snapshot.loose_ball_recovery_target_for(self.id)),
                "recover".to_string(),
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
                    let open = snapshot.open_space_for(self.id, self.home_position);
                    (
                        SoccerAction::MoveTo(snapshot.shape_guarded_support_point(
                            self.id,
                            target,
                            &[open, self.home_position],
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
            "shot-creation-run" | "overlap-run" | "support-push-up" | "support-screen"
            | "support-shape" | "support-roam"
                if !observation.has_ball =>
            {
                let roam = matches!(label, "support-roam" | "overlap-run");
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
}
