//! Central registry of **tunable behavioral weights** for the soccer engine.
//!
//! Historically the engine was full of bare numeric literals — thresholds,
//! multipliers, margins — scattered across ~47k lines. Many are genuine
//! *tuning knobs* (how far is "close enough" to count a tackle, how much
//! movement counts as a dribble, ...) that we want to adjust without a
//! recompile and to override from the learning store.
//!
//! [`Tunables`] groups those knobs into one serde-serializable tree with three
//! override layers, applied lowest-to-highest:
//!
//! 1. **Compile-time defaults** — [`Tunables::default`], which reproduces the
//!    historical literal values exactly (so an unconfigured process is
//!    behaviorally identical to before this registry existed).
//! 2. **Environment** — a whole-tree `SOCCER_TUNABLES_JSON` blob *and/or*
//!    per-field `DD_SOCCER_TUNABLE__<dotted.path>=<json>` variables.
//! 3. **Postgres** — a JSON overlay supplied by the learning layer via
//!    [`init_tunables_with_overrides`] (e.g. the active policy's tuning row).
//!
//! Overlays are merged as JSON, so each layer only needs to specify the fields
//! it changes. Access the live values through [`tunables()`]; the first access
//! lazily materialises the env layers, so free functions deep in the engine can
//! read a knob with `tunables().tracking.moved_dt_multiplier` and no plumbing.
//!
//! Per-*match* knobs that must vary within a single process still belong in
//! [`crate::des::general::soccer::MatchConfig`]; this registry is the
//! process-wide home for the long tail of global literals.

use std::sync::OnceLock;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Root of the tunable-weights tree. Grouped by subsystem; each group is a
/// plain serde struct with `#[serde(default)]` fields so a partial override
/// only names what it changes.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Tunables {
    /// Knobs for the frame-diff action *inference* used to label tracking data
    /// (`infer_tracking_action` and friends).
    pub tracking: TrackingInferenceTunables,
    /// Flank-cross legality + context-scoring weights
    /// (`flank_cross_context_score` / `_is_legal`).
    pub flank_cross: FlankCrossTunables,
    /// Discrete-event learning reward magnitudes (goal scored / conceded).
    pub reward: RewardTunables,
    /// MDP/POMDP ↔ MPC execution-reselect thresholds.
    pub decision_mpc: DecisionMpcTunables,
    /// Shot/carry tactical decision thresholds.
    pub shooting: ShootingTunables,
    /// Defensive line and back-four shape thresholds.
    pub defensive_shape: DefensiveShapeTunables,
    /// Goalkeeper positioning shaping (resting line, alignment score, buildup
    /// fan-out, loose-ball commit). See [`GoalkeeperTunables`].
    pub goalkeeper: GoalkeeperTunables,
    /// Fine-grid lane/row affinity used by formation, support, and retrieval.
    pub lane_affinity: LaneAffinityTunables,
    /// Centralized lane-discipline weights (12-lane grid). Read only when
    /// `DD_SOCCER_ENABLE_LANE_DISCIPLINE_V2` is set; see
    /// [`crate::des::general::soccer::lane_discipline`].
    pub lane_discipline: LaneDisciplineTunables,
}

impl Default for Tunables {
    fn default() -> Self {
        Tunables {
            tracking: TrackingInferenceTunables::default(),
            flank_cross: FlankCrossTunables::default(),
            reward: RewardTunables::default(),
            decision_mpc: DecisionMpcTunables::default(),
            shooting: ShootingTunables::default(),
            defensive_shape: DefensiveShapeTunables::default(),
            goalkeeper: GoalkeeperTunables::default(),
            lane_affinity: LaneAffinityTunables::default(),
            lane_discipline: LaneDisciplineTunables::default(),
        }
    }
}

/// Centralized **lane-discipline** weights — the single source of truth for how
/// firmly outfield players are held in their assigned vertical channel (lane) of
/// the 12-lane × 24-row pitch grid. Consumed exclusively by
/// [`crate::des::general::soccer::lane_discipline`], and only when its gate
/// (`DD_SOCCER_ENABLE_LANE_DISCIPLINE_V2`) is on; with the gate off the engine
/// runs the legacy inline formulas unchanged, so an unconfigured process is
/// byte-identical to before this group existed.
///
/// The defaults **re-derive** the historical lane constants for the *current*
/// 12-lane grid. The originals (`/5.0` lane-match divisor, `0.22`-per-lane
/// static falloff, `0.15` relief cliff) were authored when the grid had **4**
/// lanes (~20yd each) and were never rescaled when it tripled to 12 (~6.67yd
/// each), so the lane signal silently sharpened ~3x and the enforcement stayed a
/// cliff. These knobs express the same intent in **yard** terms (grid-/pitch-size
/// invariant) plus a smooth relief taper.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct LaneDisciplineTunables {
    /// Global lane-discipline **strength**: the one knob every consumer feels. It
    /// scales the affinity bonus *and* the lane penalty the producer functions
    /// emit, so a single value tightens or loosens shape everywhere at once.
    /// `1.0` = the re-derived baseline; `>1` holds lanes harder, `<1` looser.
    pub strength: f64,
    /// Lane-match credit (how much a target in a *neighbouring* channel still
    /// counts as "in lane") decays linearly to zero over this many **yards** of
    /// lateral gap. Replaces the legacy `|a-b| lanes / 5.0` divisor, whose reach
    /// was lane-count-relative and so silently shrank from ~3 lanes to ~5 lanes of
    /// meaning when the grid changed. `33.3` ≈ the legacy 5-lane reach at the
    /// standard 80yd width, now invariant to pitch size.
    pub lane_match_decay_yards: f64,
    /// Static out-of-band affinity falloff base: `affinity = 1 - commitment *
    /// (base + per_yard * deviation_yards)` for a target outside the role band.
    pub static_fit_base: f64,
    /// Per-**yard**-outside-band coefficient of the static falloff. Replaces the
    /// legacy per-*lane* `0.22` (which became ~3x steeper per yard at 12 lanes);
    /// `0.033` ≈ `0.22 / (80/12)` reproduces the original yard-rate at standard
    /// width.
    pub static_fit_per_yard: f64,
    /// Relief→clamp **taper** width. The legacy clamp was a cliff: a player was
    /// hard-locked to its lane only while `relief < 0.15`, then dropped straight
    /// to a soft blend. Instead the hard-lock *authority* ramps smoothly from full
    /// (relief 0) to zero (relief ≥ this), so a *mild*, justified relief move
    /// relaxes the lane rather than abandoning it — the core "hardening".
    pub relief_full_release: f64,
    /// Commitment at/above which a fully-unrelieved player is hard-locked to lane
    /// (the legacy `0.65` threshold, now the top of the taper ramp).
    pub commitment_hard_lock: f64,
    /// Soft-blend ceiling the lane pull never falls below even under full relief
    /// (`commitment * this`) — matches the legacy else-branch blend so a relieved
    /// player still carries the same residual lane pull it always did.
    pub soft_blend_max: f64,
}

impl Default for LaneDisciplineTunables {
    fn default() -> Self {
        LaneDisciplineTunables {
            strength: 1.0,
            lane_match_decay_yards: 33.3,
            static_fit_base: 0.58,
            static_fit_per_yard: 0.033,
            relief_full_release: 0.55,
            commitment_hard_lock: 0.65,
            soft_blend_max: 0.72,
        }
    }
}

/// Thresholds the tracking-frame action inference uses to classify what a
/// player did between two frames. Defaults reproduce the previous inline
/// literals in `infer_tracking_action`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct TrackingInferenceTunables {
    /// A player is judged to have actively *moved* (dribbled / repositioned)
    /// when their displacement exceeds `dt_seconds * this`. Previously the bare
    /// `1.2` in `moved > before.dt_seconds * 1.2`.
    pub moved_dt_multiplier: f64,
    /// Max yards-to-goal for a goalward ball move to be read as a `shoot`
    /// rather than a dribble. Previously `yards_to_goal <= 25.0`.
    pub shot_lane_max_yards_to_goal: f64,
    /// A defender who ends a frame this close to the ball after the opponent
    /// held it is credited with a `tackle`. Previously `< 3.8`.
    pub tackle_recover_max_distance_yards: f64,
    /// How many yards a defender must *close* on the ball within a frame to be
    /// credited with `defend`. Previously `after + 0.35 < before`.
    pub defend_closing_margin_yards: f64,
    /// Minimum open-space-score improvement for an on-possession off-ball player
    /// to be credited with finding `space`. Previously `+ 0.25`.
    pub space_improvement_threshold: f64,
}

impl Default for TrackingInferenceTunables {
    fn default() -> Self {
        TrackingInferenceTunables {
            moved_dt_multiplier: 1.2,
            shot_lane_max_yards_to_goal: 25.0,
            tackle_recover_max_distance_yards: 3.8,
            defend_closing_margin_yards: 0.35,
            space_improvement_threshold: 0.25,
        }
    }
}

/// Flank-cross legality threshold + the weights of the flank-cross context
/// score. Defaults reproduce the former `FLANK_CROSS_*` consts and the inline
/// literals in `flank_cross_context_score`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct FlankCrossTunables {
    /// Minimum flank-lane score for a flank cross to be legal. Was
    /// `FLANK_CROSS_MIN_FLANK_SCORE`.
    pub min_flank_score: f64,
    /// Maximum yards-to-goal for a flank cross (legality gate + attacking-depth
    /// scale). Was `FLANK_CROSS_MAX_YARDS_TO_GOAL`.
    pub max_yards_to_goal: f64,
    /// Weight of perceived pressure in the pressure-release term.
    pub perceived_pressure_weight: f64,
    /// Weight of pressure urgency in the pressure-release term.
    pub pressure_urgency_weight: f64,
    /// Upper cap on the pressure-release term.
    pub pressure_release_cap: f64,
    /// Weight of the flank-lane component in the context score.
    pub flank_weight: f64,
    /// Weight of the attacking-depth component in the context score.
    pub attacking_depth_weight: f64,
}

impl Default for FlankCrossTunables {
    fn default() -> Self {
        FlankCrossTunables {
            min_flank_score: 0.52,
            max_yards_to_goal: 54.0,
            perceived_pressure_weight: 0.35,
            pressure_urgency_weight: 0.25,
            pressure_release_cap: 0.40,
            flank_weight: 0.62,
            attacking_depth_weight: 0.30,
        }
    }
}

/// Discrete-event reward magnitudes folded into the learning signal when a goal
/// is scored or conceded. Defaults reproduce the former inline literals in
/// `soccer_transition_reward_with_tactics`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct RewardTunables {
    /// Reward added per goal the actor's team scores this transition. Was `100.0`.
    pub goal_scored_points: f64,
    /// Penalty when conceding, for a goalkeeper or defender (heavier — it's their
    /// job to prevent it). Was `8.0`.
    pub concede_keeper_defender_penalty: f64,
    /// Penalty when conceding, for an outfield non-defender. Was `2.0`.
    pub concede_outfield_penalty: f64,
    /// Shaping reward for easing out of a sustained teammate overlap. Was the
    /// `TEAMMATE_SPACING_OVERLAP_RELIEF_REWARD` const.
    pub teammate_overlap_relief_reward: f64,
    /// Shaping penalty for camping in a teammate overlap. Was
    /// `TEAMMATE_SPACING_OVERLAP_CAMP_PENALTY`.
    pub teammate_overlap_camp_penalty: f64,
    /// Penalty per yard a central defender advances ahead of the wing-backs. Was
    /// `CENTER_BACK_AHEAD_OF_WINGBACK_PENALTY_PER_YARD`.
    pub center_back_ahead_of_wingback_penalty_per_yard: f64,
    /// Penalty for a floor pass played into a blocked lane. Was
    /// `BLOCKED_LANE_FLOOR_PASS_PENALTY_POINTS`.
    pub blocked_lane_floor_pass_penalty_points: f64,
    /// Penalty for a forced pass played under low pressure (no need to rush). Was
    /// `LOW_PRESSURE_FORCED_PASS_PENALTY_POINTS`.
    pub low_pressure_forced_pass_penalty_points: f64,
    /// Scale on the dense **territorial pitch-control × expected-threat** delta
    /// reward (see [`crate::des::general::soccer::pitch_value`]). Multiplies the
    /// net change in the acting team's controlled threat between the before/after
    /// snapshots. Only applied when the reward is enabled by
    /// `DD_SOCCER_ENABLE_PITCH_VALUE_REWARD`; the gate keeps an unconfigured
    /// process byte-identical to before this term existed.
    pub pitch_value_threat_delta_points: f64,
}

impl Default for RewardTunables {
    fn default() -> Self {
        RewardTunables {
            goal_scored_points: 100.0,
            concede_keeper_defender_penalty: 8.0,
            concede_outfield_penalty: 2.0,
            teammate_overlap_relief_reward: 0.06,
            teammate_overlap_camp_penalty: 0.03,
            center_back_ahead_of_wingback_penalty_per_yard: 0.11,
            blocked_lane_floor_pass_penalty_points: 6.0,
            low_pressure_forced_pass_penalty_points: 1.75,
            pitch_value_threat_delta_points: 12.0,
        }
    }
}

/// MDP/POMDP action selection proposes an action; MPC prices whether the body
/// and ball can execute it. These thresholds decide when execution is "bad
/// enough" to send the candidate back for reselection.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct DecisionMpcTunables {
    pub deviation_trace_threshold_yards: f64,
    pub blend_max_target_delta_yards: f64,
    pub reselect_max_target_delta_yards: f64,
    pub reselect_min_execution_confidence: f64,
    pub reselect_min_ball_execution_probability: f64,
}

impl Default for DecisionMpcTunables {
    fn default() -> Self {
        DecisionMpcTunables {
            deviation_trace_threshold_yards: 0.75,
            blend_max_target_delta_yards: 6.0,
            reselect_max_target_delta_yards: 16.0,
            reselect_min_execution_confidence: 0.18,
            reselect_min_ball_execution_probability: 0.34,
        }
    }
}

/// Global shot and goal-approach decision thresholds. Defaults preserve the
/// historical literals in `soccer.rs`, `player.rs`, and `world.rs`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ShootingTunables {
    pub shot_on_frame_min_probability: f64,
    pub striker_shot_window_yards: f64,
    pub shot_block_bailout_max_probability: f64,
    pub goal_approach_carry_yards: f64,
    pub striker_hold_up_min_goal_distance_yards: f64,
    /// Shot-trigger MDP/POMDP discipline: beyond this distance a shot is only
    /// *volunteered* by the AI when the trigger value clears
    /// [`Self::shot_trigger_long_range_min_value`] (an open net / stranded keeper /
    /// live rebound). Inside it, the legacy near-goal shoot logic stands. See
    /// `shot_decision.rs`.
    pub shot_trigger_long_range_yards: f64,
    /// Minimum shot-trigger value `[0,1]` for a long (`> shot_trigger_long_range_yards`)
    /// shot to be volunteered. A covered long shot falls below it and is worked closer.
    pub shot_trigger_long_range_min_value: f64,
    /// Score a vetoed long shot is crushed to in the possession ranker: tiny but
    /// non-zero so it ranks below carrying/passing while staying a LEGAL option.
    pub shot_trigger_volunteer_floor: f64,
    /// Execution-probability damp applied when the MPC foot choice forces the weaker foot.
    pub shot_foot_weak_foot_execution_damp: f64,
}

impl Default for ShootingTunables {
    fn default() -> Self {
        ShootingTunables {
            shot_on_frame_min_probability: 0.60,
            striker_shot_window_yards: 30.0,
            shot_block_bailout_max_probability: 0.86,
            goal_approach_carry_yards: 45.0,
            striker_hold_up_min_goal_distance_yards: 45.0,
            shot_trigger_long_range_yards: 22.0,
            shot_trigger_long_range_min_value: 0.25,
            shot_trigger_volunteer_floor: 0.002,
            shot_foot_weak_foot_execution_damp: 0.88,
        }
    }
}

/// Defensive line and back-four spacing knobs. These are intentionally narrow:
/// they cover the line/wingback shape rules that must stay configurable without
/// allowing an overlay to make the back four collapse into nonsense.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct DefensiveShapeTunables {
    pub defensive_line_max_into_opp_half_yards: f64,
    pub back_four_block_width_yards: f64,
    pub back_four_horizontal_min_gap_yards: f64,
    pub back_four_horizontal_max_gap_yards: f64,
    pub wingback_defensive_pinch_target_seconds: f64,
    pub wingback_defensive_pinch_opponent_half_margin_yards: f64,
    pub defensive_goal_side_min_yards: f64,
}

impl Default for DefensiveShapeTunables {
    fn default() -> Self {
        DefensiveShapeTunables {
            defensive_line_max_into_opp_half_yards: 5.0,
            back_four_block_width_yards: 22.0,
            back_four_horizontal_min_gap_yards: 1.5,
            back_four_horizontal_max_gap_yards: 8.0,
            wingback_defensive_pinch_target_seconds: 3.0,
            wingback_defensive_pinch_opponent_half_margin_yards: 8.0,
            defensive_goal_side_min_yards: 1.5,
        }
    }
}

/// **Goalkeeper positioning** knobs — the shaping numbers behind where the keeper
/// rests on its ball↔goal tracking line, how its line alignment is scored, how it
/// fans defenders out for buildup, and how confidently it commits to a loose ball.
///
/// These were previously bare numeric literals inside the keeper positioning
/// functions in `world.rs` / `soccer.rs`. The defaults reproduce those historical
/// literals exactly, so an unconfigured process is byte-identical to before this
/// group existed; override per-field via `DD_SOCCER_TUNABLE__goalkeeper.<field>` or
/// the Postgres tuning overlay. Hard geometry (six-yard box depth/width, leave-box
/// confidence thresholds, line-recovery deviations) stays in the named `const`
/// block in `soccer.rs`; this group is only the soft behavioral shaping.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct GoalkeeperTunables {
    // --- resting line on the ball↔goal segment (goalkeeper_ball_goal_tracking_target) ---
    /// Ball→goal distance (yds) at which an advancing ball reaches full "ball
    /// pressure" (keeper pushed to its max forward depth). 0 at this range, 1 at the goal.
    pub tracking_ball_pressure_reference_yards: f64,
    /// Holder→goal-line distance (yds) at which holder pressure saturates.
    pub tracking_holder_pressure_reference_yards: f64,
    /// Resting depth off the goal line (yds) with no pressure.
    pub tracking_resting_depth_yards: f64,
    /// Extra depth (yds) added at full ball pressure.
    pub tracking_ball_pressure_depth_gain_yards: f64,
    /// Extra depth (yds) added at full holder pressure.
    pub tracking_holder_pressure_depth_gain_yards: f64,
    /// Minimum resting depth (yds) before the genome line-height shift.
    pub tracking_min_depth_yards: f64,
    /// Full ± span (yds) the `gk_line_height` genome shifts the resting line (0/1 vs 0.5).
    pub tracking_line_height_depth_span_yards: f64,
    /// Minimum resting depth (yds) after the line-height shift.
    pub tracking_line_height_min_depth_yards: f64,
    /// Standoff (yds) kept short of the ball so the keeper never overruns it.
    pub tracking_ball_standoff_yards: f64,

    // --- ball↔goal line ALIGNMENT score (goalkeeper_ball_goal_line_alignment_score) ---
    /// Perpendicular off-line distance (yds) that drops the line-alignment score by 1.0.
    pub alignment_line_distance_reference_yards: f64,
    /// Projection overshoot past the goal/ball endpoints that drops the projection score by 1.0.
    pub alignment_projection_reference: f64,
    /// Distance (yds) from the ideal tracking point that drops the depth score by 1.0.
    pub alignment_depth_reference_yards: f64,
    /// Weight of the on-line term in the blended alignment score.
    pub alignment_line_weight: f64,
    /// Weight of the projection (between-the-endpoints) term.
    pub alignment_projection_weight: f64,
    /// Weight of the depth (distance-to-ideal) term.
    pub alignment_depth_weight: f64,

    // --- buildup fan-out lane target (goalkeeper_buildup_lane_target_for) ---
    /// Depth (yds) off the own goal line a WIDE defender fans out to receive a short pass.
    pub buildup_wide_defender_depth_yards: f64,
    /// Depth (yds) off the own goal line a CENTRAL defender fans out to.
    pub buildup_central_defender_depth_yards: f64,
    /// Lateral probe offset (yds) used to pick the most-open in-lane buildup spot (±this, and 0).
    pub buildup_lane_probe_offset_yards: f64,

    // --- leave-six-yard confidence blend (goalkeeper_leave_six_yard_confidence_from_times) ---
    /// Weight of goalkeeping ability in the keeper's sweep "tool" term.
    pub leave_confidence_goalkeeping_weight: f64,
    /// Weight of acceleration ability in the sweep tool term.
    pub leave_confidence_acceleration_weight: f64,
    /// Weight of perceived-position confidence in the sweep tool term.
    pub leave_confidence_perception_weight: f64,
    /// Weight of the race-dominance term in the overall leave confidence.
    pub leave_confidence_dominance_weight: f64,
    /// Weight of the keeper-tool term in the overall leave confidence.
    pub leave_confidence_tool_weight: f64,

    // --- loose-ball commit, own-ball backpass case (goalkeeper_should_commit_to_loose_ball) ---
    /// On a ball WE last touched (can't handle), the keeper only charges out if it beats the
    /// nearest opponent to the ball by this fraction of the opponent's arrival time.
    pub backpass_commit_opponent_time_fraction: f64,
    /// ...and additionally reaches it at least this many seconds before the nearest teammate.
    pub backpass_commit_teammate_margin_seconds: f64,
}

impl Default for GoalkeeperTunables {
    fn default() -> Self {
        GoalkeeperTunables {
            tracking_ball_pressure_reference_yards: 72.0,
            tracking_holder_pressure_reference_yards: 42.0,
            tracking_resting_depth_yards: 3.0,
            tracking_ball_pressure_depth_gain_yards: 3.0,
            tracking_holder_pressure_depth_gain_yards: 1.2,
            tracking_min_depth_yards: 2.0,
            tracking_line_height_depth_span_yards: 2.0,
            tracking_line_height_min_depth_yards: 1.5,
            tracking_ball_standoff_yards: 0.85,
            alignment_line_distance_reference_yards: 1.0,
            alignment_projection_reference: 0.35,
            alignment_depth_reference_yards: 4.5,
            alignment_line_weight: 0.82,
            alignment_projection_weight: 0.08,
            alignment_depth_weight: 0.10,
            buildup_wide_defender_depth_yards: 24.0,
            buildup_central_defender_depth_yards: 18.0,
            buildup_lane_probe_offset_yards: 4.0,
            leave_confidence_goalkeeping_weight: 0.72,
            leave_confidence_acceleration_weight: 0.18,
            leave_confidence_perception_weight: 0.10,
            leave_confidence_dominance_weight: 0.90,
            leave_confidence_tool_weight: 0.10,
            backpass_commit_opponent_time_fraction: 0.65,
            backpass_commit_teammate_margin_seconds: 0.5,
        }
    }
}

/// Fine-grid lane/row affinity knobs. Defaults harden the existing 12-lane /
/// 24-row shape signal while keeping forwards soft enough to make real runs.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct LaneAffinityTunables {
    pub goalkeeper_neutral_score: f64,
    pub defender_lane_radius_possession: usize,
    pub defender_lane_radius_defense: usize,
    pub midfielder_lane_radius_possession: usize,
    pub midfielder_lane_radius_defense: usize,
    pub forward_lane_radius: usize,
    pub defender_commitment_possession: f64,
    pub defender_commitment_defense: f64,
    pub midfielder_commitment_possession: f64,
    pub midfielder_commitment_defense: f64,
    pub forward_commitment_possession: f64,
    pub forward_commitment_defense: f64,
    pub commitment_relief_max: f64,
    pub lane_gap_base_penalty: f64,
    pub lane_gap_step_penalty: f64,
    pub possession_factor: f64,
    pub markov_relief_max: f64,
    pub ball_speed_clamp_yps: f64,
    pub ball_acceleration_clamp_yps2: f64,
    pub lookahead_base_seconds: f64,
    pub lookahead_speed_scale_yps: f64,
    pub lookahead_acceleration_scale_yps2: f64,
    pub lookahead_min_seconds: f64,
    pub lookahead_max_seconds: f64,
    pub lane_match_span_lanes: f64,
    pub row_match_span_rows: f64,
    pub player_predicted_lane_weight: f64,
    pub home_predicted_lane_weight: f64,
    pub player_current_lane_weight: f64,
    pub target_predicted_lane_weight: f64,
    pub target_current_lane_weight: f64,
    pub player_predicted_row_weight: f64,
    pub home_predicted_row_weight: f64,
    pub target_predicted_row_weight: f64,
    pub target_current_row_weight: f64,
    pub flow_base_score: f64,
    pub flow_gap_weight: f64,
    pub flow_min_score: f64,
    pub flow_max_score: f64,
    pub field_teammate_space_weight: f64,
    pub field_open_space_weight: f64,
    pub field_open_space_normalizer_yards: f64,
    pub forward_static_fit_weight: f64,
    pub forward_player_ball_weight: f64,
    pub forward_target_ball_weight: f64,
    pub forward_row_coherence_weight: f64,
    pub forward_flow_weight: f64,
    pub forward_field_config_weight: f64,
    pub role_markov_weight: f64,
    pub role_static_fit_weight: f64,
    pub role_player_ball_weight: f64,
    pub role_target_ball_weight: f64,
    pub role_row_coherence_weight: f64,
    pub role_flow_weight: f64,
    pub role_field_config_weight: f64,
    pub open_space_dynamic_lane_bonus_weight: f64,
    pub movement_shape_dynamic_lane_weight: f64,
}

impl Default for LaneAffinityTunables {
    fn default() -> Self {
        LaneAffinityTunables {
            goalkeeper_neutral_score: 0.5,
            defender_lane_radius_possession: 2,
            defender_lane_radius_defense: 1,
            midfielder_lane_radius_possession: 2,
            midfielder_lane_radius_defense: 1,
            forward_lane_radius: 3,
            defender_commitment_possession: 0.80,
            defender_commitment_defense: 0.92,
            midfielder_commitment_possession: 0.80,
            midfielder_commitment_defense: 0.90,
            forward_commitment_possession: 0.36,
            forward_commitment_defense: 0.40,
            commitment_relief_max: 0.65,
            lane_gap_base_penalty: 0.58,
            lane_gap_step_penalty: 0.22,
            possession_factor: 0.80,
            markov_relief_max: 0.85,
            ball_speed_clamp_yps: 42.0,
            ball_acceleration_clamp_yps2: 72.0,
            lookahead_base_seconds: 0.45,
            lookahead_speed_scale_yps: 42.0,
            lookahead_acceleration_scale_yps2: 96.0,
            lookahead_min_seconds: 0.45,
            lookahead_max_seconds: 1.35,
            lane_match_span_lanes: 5.0,
            row_match_span_rows: 8.0,
            player_predicted_lane_weight: 0.48,
            home_predicted_lane_weight: 0.34,
            player_current_lane_weight: 0.18,
            target_predicted_lane_weight: 0.68,
            target_current_lane_weight: 0.32,
            player_predicted_row_weight: 0.42,
            home_predicted_row_weight: 0.18,
            target_predicted_row_weight: 0.28,
            target_current_row_weight: 0.12,
            flow_base_score: 0.5,
            flow_gap_weight: 0.18,
            flow_min_score: 0.15,
            flow_max_score: 0.85,
            field_teammate_space_weight: 0.62,
            field_open_space_weight: 0.38,
            field_open_space_normalizer_yards: 18.0,
            forward_static_fit_weight: 0.38,
            forward_player_ball_weight: 0.18,
            forward_target_ball_weight: 0.14,
            forward_row_coherence_weight: 0.08,
            forward_flow_weight: 0.06,
            forward_field_config_weight: 0.10,
            role_markov_weight: 0.34,
            role_static_fit_weight: 0.30,
            role_player_ball_weight: 0.16,
            role_target_ball_weight: 0.08,
            role_row_coherence_weight: 0.06,
            role_flow_weight: 0.03,
            role_field_config_weight: 0.03,
            open_space_dynamic_lane_bonus_weight: 1.05,
            movement_shape_dynamic_lane_weight: 0.36,
        }
    }
}

impl Tunables {
    /// Build the tunables from compile-time defaults plus the environment
    /// layers (`SOCCER_TUNABLES_JSON` then per-field `DD_SOCCER_TUNABLE__*`).
    /// Does not consult Postgres — see [`init_tunables_with_overrides`].
    pub fn from_env() -> Tunables {
        Tunables::from_overlays(env_overlays())
    }

    /// Build from defaults, then fold each JSON overlay on top in order
    /// (later overlays win). Each overlay only needs to name the fields it
    /// changes. Malformed overlays are skipped rather than panicking.
    pub fn from_overlays<I: IntoIterator<Item = Value>>(overlays: I) -> Tunables {
        let mut merged = serde_json::to_value(Tunables::default())
            .unwrap_or_else(|_| Value::Object(Default::default()));
        for overlay in overlays {
            merge_json(&mut merged, overlay);
        }
        match serde_json::from_value::<Tunables>(merged) {
            Ok(tunables) => tunables.sanitized(),
            Err(err) => {
                eprintln!(
                    "soccer tunables: overlay type mismatch ({err}); using sanitized defaults"
                );
                Tunables::default().sanitized()
            }
        }
    }

    /// Return a copy with every numeric field finite and inside its hard safety
    /// bounds. Hard outliers are clamped with a warning; softer "that looks
    /// weird" ranges also warn but keep the value.
    pub fn sanitized(mut self) -> Tunables {
        self.tracking.sanitize();
        self.flank_cross.sanitize();
        self.reward.sanitize();
        self.decision_mpc.sanitize();
        self.shooting.sanitize();
        self.defensive_shape.sanitize();
        self.lane_affinity.sanitize();
        self
    }

    /// Strict validation for callers/tests that want an error instead of runtime
    /// warning+clamp behavior.
    pub fn validate_strict(&self) -> Result<(), String> {
        let mut errors = Vec::new();
        self.tracking.validate_strict("tracking", &mut errors);
        self.flank_cross.validate_strict("flank_cross", &mut errors);
        self.reward.validate_strict("reward", &mut errors);
        self.decision_mpc
            .validate_strict("decision_mpc", &mut errors);
        self.shooting.validate_strict("shooting", &mut errors);
        self.defensive_shape
            .validate_strict("defensive_shape", &mut errors);
        self.lane_affinity
            .validate_strict("lane_affinity", &mut errors);
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors.join("; "))
        }
    }
}

impl LaneAffinityTunables {
    fn sanitize(&mut self) {
        let default = LaneAffinityTunables::default();
        sanitize_f64(
            "lane_affinity.goalkeeper_neutral_score",
            &mut self.goalkeeper_neutral_score,
            default.goalkeeper_neutral_score,
            0.0,
            1.0,
            0.25,
            0.75,
        );
        sanitize_usize(
            "lane_affinity.defender_lane_radius_possession",
            &mut self.defender_lane_radius_possession,
            default.defender_lane_radius_possession,
            0,
            6,
            1,
            3,
        );
        sanitize_usize(
            "lane_affinity.defender_lane_radius_defense",
            &mut self.defender_lane_radius_defense,
            default.defender_lane_radius_defense,
            0,
            6,
            1,
            2,
        );
        sanitize_usize(
            "lane_affinity.midfielder_lane_radius_possession",
            &mut self.midfielder_lane_radius_possession,
            default.midfielder_lane_radius_possession,
            0,
            6,
            1,
            3,
        );
        sanitize_usize(
            "lane_affinity.midfielder_lane_radius_defense",
            &mut self.midfielder_lane_radius_defense,
            default.midfielder_lane_radius_defense,
            0,
            6,
            1,
            2,
        );
        sanitize_usize(
            "lane_affinity.forward_lane_radius",
            &mut self.forward_lane_radius,
            default.forward_lane_radius,
            0,
            6,
            2,
            4,
        );
        sanitize_f64(
            "lane_affinity.defender_commitment_possession",
            &mut self.defender_commitment_possession,
            default.defender_commitment_possession,
            0.0,
            1.0,
            0.50,
            0.95,
        );
        sanitize_f64(
            "lane_affinity.defender_commitment_defense",
            &mut self.defender_commitment_defense,
            default.defender_commitment_defense,
            0.0,
            1.0,
            0.60,
            1.0,
        );
        sanitize_f64(
            "lane_affinity.midfielder_commitment_possession",
            &mut self.midfielder_commitment_possession,
            default.midfielder_commitment_possession,
            0.0,
            1.0,
            0.45,
            0.95,
        );
        sanitize_f64(
            "lane_affinity.midfielder_commitment_defense",
            &mut self.midfielder_commitment_defense,
            default.midfielder_commitment_defense,
            0.0,
            1.0,
            0.55,
            1.0,
        );
        sanitize_f64(
            "lane_affinity.forward_commitment_possession",
            &mut self.forward_commitment_possession,
            default.forward_commitment_possession,
            0.0,
            1.0,
            0.15,
            0.65,
        );
        sanitize_f64(
            "lane_affinity.forward_commitment_defense",
            &mut self.forward_commitment_defense,
            default.forward_commitment_defense,
            0.0,
            1.0,
            0.15,
            0.70,
        );
        sanitize_f64(
            "lane_affinity.commitment_relief_max",
            &mut self.commitment_relief_max,
            default.commitment_relief_max,
            0.0,
            1.0,
            0.30,
            0.85,
        );
        sanitize_f64(
            "lane_affinity.lane_gap_base_penalty",
            &mut self.lane_gap_base_penalty,
            default.lane_gap_base_penalty,
            0.0,
            2.0,
            0.25,
            1.0,
        );
        sanitize_f64(
            "lane_affinity.lane_gap_step_penalty",
            &mut self.lane_gap_step_penalty,
            default.lane_gap_step_penalty,
            0.0,
            1.0,
            0.05,
            0.45,
        );
        sanitize_f64(
            "lane_affinity.possession_factor",
            &mut self.possession_factor,
            default.possession_factor,
            0.0,
            1.0,
            0.60,
            0.90,
        );
        sanitize_f64(
            "lane_affinity.markov_relief_max",
            &mut self.markov_relief_max,
            default.markov_relief_max,
            0.0,
            1.0,
            0.40,
            0.95,
        );
        sanitize_f64(
            "lane_affinity.ball_speed_clamp_yps",
            &mut self.ball_speed_clamp_yps,
            default.ball_speed_clamp_yps,
            1.0,
            80.0,
            20.0,
            55.0,
        );
        sanitize_f64(
            "lane_affinity.ball_acceleration_clamp_yps2",
            &mut self.ball_acceleration_clamp_yps2,
            default.ball_acceleration_clamp_yps2,
            1.0,
            160.0,
            30.0,
            110.0,
        );
        sanitize_f64(
            "lane_affinity.lookahead_base_seconds",
            &mut self.lookahead_base_seconds,
            default.lookahead_base_seconds,
            0.0,
            5.0,
            0.20,
            1.25,
        );
        sanitize_f64(
            "lane_affinity.lookahead_speed_scale_yps",
            &mut self.lookahead_speed_scale_yps,
            default.lookahead_speed_scale_yps,
            1.0,
            120.0,
            20.0,
            70.0,
        );
        sanitize_f64(
            "lane_affinity.lookahead_acceleration_scale_yps2",
            &mut self.lookahead_acceleration_scale_yps2,
            default.lookahead_acceleration_scale_yps2,
            1.0,
            220.0,
            40.0,
            140.0,
        );
        sanitize_f64(
            "lane_affinity.lookahead_min_seconds",
            &mut self.lookahead_min_seconds,
            default.lookahead_min_seconds,
            0.0,
            5.0,
            0.20,
            1.0,
        );
        sanitize_f64(
            "lane_affinity.lookahead_max_seconds",
            &mut self.lookahead_max_seconds,
            default.lookahead_max_seconds,
            0.0,
            5.0,
            0.75,
            2.5,
        );
        sanitize_f64(
            "lane_affinity.lane_match_span_lanes",
            &mut self.lane_match_span_lanes,
            default.lane_match_span_lanes,
            1.0,
            12.0,
            3.0,
            8.0,
        );
        sanitize_f64(
            "lane_affinity.row_match_span_rows",
            &mut self.row_match_span_rows,
            default.row_match_span_rows,
            1.0,
            24.0,
            4.0,
            12.0,
        );
        sanitize_f64(
            "lane_affinity.player_predicted_lane_weight",
            &mut self.player_predicted_lane_weight,
            default.player_predicted_lane_weight,
            0.0,
            5.0,
            0.0,
            1.5,
        );
        sanitize_f64(
            "lane_affinity.home_predicted_lane_weight",
            &mut self.home_predicted_lane_weight,
            default.home_predicted_lane_weight,
            0.0,
            5.0,
            0.0,
            1.5,
        );
        sanitize_f64(
            "lane_affinity.player_current_lane_weight",
            &mut self.player_current_lane_weight,
            default.player_current_lane_weight,
            0.0,
            5.0,
            0.0,
            1.5,
        );
        sanitize_f64(
            "lane_affinity.target_predicted_lane_weight",
            &mut self.target_predicted_lane_weight,
            default.target_predicted_lane_weight,
            0.0,
            5.0,
            0.0,
            1.5,
        );
        sanitize_f64(
            "lane_affinity.target_current_lane_weight",
            &mut self.target_current_lane_weight,
            default.target_current_lane_weight,
            0.0,
            5.0,
            0.0,
            1.5,
        );
        sanitize_f64(
            "lane_affinity.player_predicted_row_weight",
            &mut self.player_predicted_row_weight,
            default.player_predicted_row_weight,
            0.0,
            5.0,
            0.0,
            1.5,
        );
        sanitize_f64(
            "lane_affinity.home_predicted_row_weight",
            &mut self.home_predicted_row_weight,
            default.home_predicted_row_weight,
            0.0,
            5.0,
            0.0,
            1.5,
        );
        sanitize_f64(
            "lane_affinity.target_predicted_row_weight",
            &mut self.target_predicted_row_weight,
            default.target_predicted_row_weight,
            0.0,
            5.0,
            0.0,
            1.5,
        );
        sanitize_f64(
            "lane_affinity.target_current_row_weight",
            &mut self.target_current_row_weight,
            default.target_current_row_weight,
            0.0,
            5.0,
            0.0,
            1.5,
        );
        sanitize_f64(
            "lane_affinity.flow_base_score",
            &mut self.flow_base_score,
            default.flow_base_score,
            0.0,
            1.0,
            0.20,
            0.80,
        );
        sanitize_f64(
            "lane_affinity.flow_gap_weight",
            &mut self.flow_gap_weight,
            default.flow_gap_weight,
            0.0,
            1.0,
            0.05,
            0.40,
        );
        sanitize_f64(
            "lane_affinity.flow_min_score",
            &mut self.flow_min_score,
            default.flow_min_score,
            0.0,
            1.0,
            0.0,
            0.45,
        );
        sanitize_f64(
            "lane_affinity.flow_max_score",
            &mut self.flow_max_score,
            default.flow_max_score,
            0.0,
            1.0,
            0.55,
            1.0,
        );
        sanitize_f64(
            "lane_affinity.field_teammate_space_weight",
            &mut self.field_teammate_space_weight,
            default.field_teammate_space_weight,
            0.0,
            5.0,
            0.0,
            1.5,
        );
        sanitize_f64(
            "lane_affinity.field_open_space_weight",
            &mut self.field_open_space_weight,
            default.field_open_space_weight,
            0.0,
            5.0,
            0.0,
            1.5,
        );
        sanitize_f64(
            "lane_affinity.field_open_space_normalizer_yards",
            &mut self.field_open_space_normalizer_yards,
            default.field_open_space_normalizer_yards,
            1.0,
            80.0,
            8.0,
            35.0,
        );
        sanitize_f64(
            "lane_affinity.forward_static_fit_weight",
            &mut self.forward_static_fit_weight,
            default.forward_static_fit_weight,
            0.0,
            5.0,
            0.0,
            1.5,
        );
        sanitize_f64(
            "lane_affinity.forward_player_ball_weight",
            &mut self.forward_player_ball_weight,
            default.forward_player_ball_weight,
            0.0,
            5.0,
            0.0,
            1.5,
        );
        sanitize_f64(
            "lane_affinity.forward_target_ball_weight",
            &mut self.forward_target_ball_weight,
            default.forward_target_ball_weight,
            0.0,
            5.0,
            0.0,
            1.5,
        );
        sanitize_f64(
            "lane_affinity.forward_row_coherence_weight",
            &mut self.forward_row_coherence_weight,
            default.forward_row_coherence_weight,
            0.0,
            5.0,
            0.0,
            1.5,
        );
        sanitize_f64(
            "lane_affinity.forward_flow_weight",
            &mut self.forward_flow_weight,
            default.forward_flow_weight,
            0.0,
            5.0,
            0.0,
            1.5,
        );
        sanitize_f64(
            "lane_affinity.forward_field_config_weight",
            &mut self.forward_field_config_weight,
            default.forward_field_config_weight,
            0.0,
            5.0,
            0.0,
            1.5,
        );
        sanitize_f64(
            "lane_affinity.role_markov_weight",
            &mut self.role_markov_weight,
            default.role_markov_weight,
            0.0,
            5.0,
            0.0,
            1.5,
        );
        sanitize_f64(
            "lane_affinity.role_static_fit_weight",
            &mut self.role_static_fit_weight,
            default.role_static_fit_weight,
            0.0,
            5.0,
            0.0,
            1.5,
        );
        sanitize_f64(
            "lane_affinity.role_player_ball_weight",
            &mut self.role_player_ball_weight,
            default.role_player_ball_weight,
            0.0,
            5.0,
            0.0,
            1.5,
        );
        sanitize_f64(
            "lane_affinity.role_target_ball_weight",
            &mut self.role_target_ball_weight,
            default.role_target_ball_weight,
            0.0,
            5.0,
            0.0,
            1.5,
        );
        sanitize_f64(
            "lane_affinity.role_row_coherence_weight",
            &mut self.role_row_coherence_weight,
            default.role_row_coherence_weight,
            0.0,
            5.0,
            0.0,
            1.5,
        );
        sanitize_f64(
            "lane_affinity.role_flow_weight",
            &mut self.role_flow_weight,
            default.role_flow_weight,
            0.0,
            5.0,
            0.0,
            1.5,
        );
        sanitize_f64(
            "lane_affinity.role_field_config_weight",
            &mut self.role_field_config_weight,
            default.role_field_config_weight,
            0.0,
            5.0,
            0.0,
            1.5,
        );
        sanitize_f64(
            "lane_affinity.open_space_dynamic_lane_bonus_weight",
            &mut self.open_space_dynamic_lane_bonus_weight,
            default.open_space_dynamic_lane_bonus_weight,
            0.0,
            5.0,
            0.20,
            2.0,
        );
        sanitize_f64(
            "lane_affinity.movement_shape_dynamic_lane_weight",
            &mut self.movement_shape_dynamic_lane_weight,
            default.movement_shape_dynamic_lane_weight,
            0.0,
            5.0,
            0.10,
            1.0,
        );
        if self.lookahead_min_seconds > self.lookahead_max_seconds {
            eprintln!("soccer tunables: lane_affinity lookahead min exceeded max; swapping");
            std::mem::swap(
                &mut self.lookahead_min_seconds,
                &mut self.lookahead_max_seconds,
            );
        }
        if self.flow_min_score > self.flow_max_score {
            eprintln!("soccer tunables: lane_affinity flow min exceeded max; swapping");
            std::mem::swap(&mut self.flow_min_score, &mut self.flow_max_score);
        }
    }

    fn validate_strict(&self, prefix: &str, errors: &mut Vec<String>) {
        validate_f64(
            prefix,
            "goalkeeper_neutral_score",
            self.goalkeeper_neutral_score,
            0.0,
            1.0,
            errors,
        );
        validate_usize(
            prefix,
            "defender_lane_radius_possession",
            self.defender_lane_radius_possession,
            0,
            6,
            errors,
        );
        validate_usize(
            prefix,
            "defender_lane_radius_defense",
            self.defender_lane_radius_defense,
            0,
            6,
            errors,
        );
        validate_usize(
            prefix,
            "midfielder_lane_radius_possession",
            self.midfielder_lane_radius_possession,
            0,
            6,
            errors,
        );
        validate_usize(
            prefix,
            "midfielder_lane_radius_defense",
            self.midfielder_lane_radius_defense,
            0,
            6,
            errors,
        );
        validate_usize(
            prefix,
            "forward_lane_radius",
            self.forward_lane_radius,
            0,
            6,
            errors,
        );
        validate_f64(
            prefix,
            "defender_commitment_possession",
            self.defender_commitment_possession,
            0.0,
            1.0,
            errors,
        );
        validate_f64(
            prefix,
            "defender_commitment_defense",
            self.defender_commitment_defense,
            0.0,
            1.0,
            errors,
        );
        validate_f64(
            prefix,
            "midfielder_commitment_possession",
            self.midfielder_commitment_possession,
            0.0,
            1.0,
            errors,
        );
        validate_f64(
            prefix,
            "midfielder_commitment_defense",
            self.midfielder_commitment_defense,
            0.0,
            1.0,
            errors,
        );
        validate_f64(
            prefix,
            "forward_commitment_possession",
            self.forward_commitment_possession,
            0.0,
            1.0,
            errors,
        );
        validate_f64(
            prefix,
            "forward_commitment_defense",
            self.forward_commitment_defense,
            0.0,
            1.0,
            errors,
        );
        validate_f64(
            prefix,
            "commitment_relief_max",
            self.commitment_relief_max,
            0.0,
            1.0,
            errors,
        );
        validate_f64(
            prefix,
            "lane_gap_base_penalty",
            self.lane_gap_base_penalty,
            0.0,
            2.0,
            errors,
        );
        validate_f64(
            prefix,
            "lane_gap_step_penalty",
            self.lane_gap_step_penalty,
            0.0,
            1.0,
            errors,
        );
        validate_f64(
            prefix,
            "possession_factor",
            self.possession_factor,
            0.0,
            1.0,
            errors,
        );
        validate_f64(
            prefix,
            "markov_relief_max",
            self.markov_relief_max,
            0.0,
            1.0,
            errors,
        );
        validate_f64(
            prefix,
            "ball_speed_clamp_yps",
            self.ball_speed_clamp_yps,
            1.0,
            80.0,
            errors,
        );
        validate_f64(
            prefix,
            "ball_acceleration_clamp_yps2",
            self.ball_acceleration_clamp_yps2,
            1.0,
            160.0,
            errors,
        );
        validate_f64(
            prefix,
            "lookahead_base_seconds",
            self.lookahead_base_seconds,
            0.0,
            5.0,
            errors,
        );
        validate_f64(
            prefix,
            "lookahead_speed_scale_yps",
            self.lookahead_speed_scale_yps,
            1.0,
            120.0,
            errors,
        );
        validate_f64(
            prefix,
            "lookahead_acceleration_scale_yps2",
            self.lookahead_acceleration_scale_yps2,
            1.0,
            220.0,
            errors,
        );
        validate_f64(
            prefix,
            "lookahead_min_seconds",
            self.lookahead_min_seconds,
            0.0,
            5.0,
            errors,
        );
        validate_f64(
            prefix,
            "lookahead_max_seconds",
            self.lookahead_max_seconds,
            0.0,
            5.0,
            errors,
        );
        validate_f64(
            prefix,
            "lane_match_span_lanes",
            self.lane_match_span_lanes,
            1.0,
            12.0,
            errors,
        );
        validate_f64(
            prefix,
            "row_match_span_rows",
            self.row_match_span_rows,
            1.0,
            24.0,
            errors,
        );
        validate_f64(
            prefix,
            "player_predicted_lane_weight",
            self.player_predicted_lane_weight,
            0.0,
            5.0,
            errors,
        );
        validate_f64(
            prefix,
            "home_predicted_lane_weight",
            self.home_predicted_lane_weight,
            0.0,
            5.0,
            errors,
        );
        validate_f64(
            prefix,
            "player_current_lane_weight",
            self.player_current_lane_weight,
            0.0,
            5.0,
            errors,
        );
        validate_f64(
            prefix,
            "target_predicted_lane_weight",
            self.target_predicted_lane_weight,
            0.0,
            5.0,
            errors,
        );
        validate_f64(
            prefix,
            "target_current_lane_weight",
            self.target_current_lane_weight,
            0.0,
            5.0,
            errors,
        );
        validate_f64(
            prefix,
            "player_predicted_row_weight",
            self.player_predicted_row_weight,
            0.0,
            5.0,
            errors,
        );
        validate_f64(
            prefix,
            "home_predicted_row_weight",
            self.home_predicted_row_weight,
            0.0,
            5.0,
            errors,
        );
        validate_f64(
            prefix,
            "target_predicted_row_weight",
            self.target_predicted_row_weight,
            0.0,
            5.0,
            errors,
        );
        validate_f64(
            prefix,
            "target_current_row_weight",
            self.target_current_row_weight,
            0.0,
            5.0,
            errors,
        );
        validate_f64(
            prefix,
            "flow_base_score",
            self.flow_base_score,
            0.0,
            1.0,
            errors,
        );
        validate_f64(
            prefix,
            "flow_gap_weight",
            self.flow_gap_weight,
            0.0,
            1.0,
            errors,
        );
        validate_f64(
            prefix,
            "flow_min_score",
            self.flow_min_score,
            0.0,
            1.0,
            errors,
        );
        validate_f64(
            prefix,
            "flow_max_score",
            self.flow_max_score,
            0.0,
            1.0,
            errors,
        );
        validate_f64(
            prefix,
            "field_teammate_space_weight",
            self.field_teammate_space_weight,
            0.0,
            5.0,
            errors,
        );
        validate_f64(
            prefix,
            "field_open_space_weight",
            self.field_open_space_weight,
            0.0,
            5.0,
            errors,
        );
        validate_f64(
            prefix,
            "field_open_space_normalizer_yards",
            self.field_open_space_normalizer_yards,
            1.0,
            80.0,
            errors,
        );
        validate_f64(
            prefix,
            "forward_static_fit_weight",
            self.forward_static_fit_weight,
            0.0,
            5.0,
            errors,
        );
        validate_f64(
            prefix,
            "forward_player_ball_weight",
            self.forward_player_ball_weight,
            0.0,
            5.0,
            errors,
        );
        validate_f64(
            prefix,
            "forward_target_ball_weight",
            self.forward_target_ball_weight,
            0.0,
            5.0,
            errors,
        );
        validate_f64(
            prefix,
            "forward_row_coherence_weight",
            self.forward_row_coherence_weight,
            0.0,
            5.0,
            errors,
        );
        validate_f64(
            prefix,
            "forward_flow_weight",
            self.forward_flow_weight,
            0.0,
            5.0,
            errors,
        );
        validate_f64(
            prefix,
            "forward_field_config_weight",
            self.forward_field_config_weight,
            0.0,
            5.0,
            errors,
        );
        validate_f64(
            prefix,
            "role_markov_weight",
            self.role_markov_weight,
            0.0,
            5.0,
            errors,
        );
        validate_f64(
            prefix,
            "role_static_fit_weight",
            self.role_static_fit_weight,
            0.0,
            5.0,
            errors,
        );
        validate_f64(
            prefix,
            "role_player_ball_weight",
            self.role_player_ball_weight,
            0.0,
            5.0,
            errors,
        );
        validate_f64(
            prefix,
            "role_target_ball_weight",
            self.role_target_ball_weight,
            0.0,
            5.0,
            errors,
        );
        validate_f64(
            prefix,
            "role_row_coherence_weight",
            self.role_row_coherence_weight,
            0.0,
            5.0,
            errors,
        );
        validate_f64(
            prefix,
            "role_flow_weight",
            self.role_flow_weight,
            0.0,
            5.0,
            errors,
        );
        validate_f64(
            prefix,
            "role_field_config_weight",
            self.role_field_config_weight,
            0.0,
            5.0,
            errors,
        );
        validate_f64(
            prefix,
            "open_space_dynamic_lane_bonus_weight",
            self.open_space_dynamic_lane_bonus_weight,
            0.0,
            5.0,
            errors,
        );
        validate_f64(
            prefix,
            "movement_shape_dynamic_lane_weight",
            self.movement_shape_dynamic_lane_weight,
            0.0,
            5.0,
            errors,
        );
        if self.lookahead_min_seconds > self.lookahead_max_seconds {
            errors.push(format!(
                "{prefix}.lookahead_min_seconds > {prefix}.lookahead_max_seconds"
            ));
        }
        if self.flow_min_score > self.flow_max_score {
            errors.push(format!("{prefix}.flow_min_score > {prefix}.flow_max_score"));
        }
    }
}

impl TrackingInferenceTunables {
    fn sanitize(&mut self) {
        let default = TrackingInferenceTunables::default();
        sanitize_f64(
            "tracking.moved_dt_multiplier",
            &mut self.moved_dt_multiplier,
            default.moved_dt_multiplier,
            0.05,
            12.0,
            0.30,
            3.0,
        );
        sanitize_f64(
            "tracking.shot_lane_max_yards_to_goal",
            &mut self.shot_lane_max_yards_to_goal,
            default.shot_lane_max_yards_to_goal,
            1.0,
            120.0,
            8.0,
            45.0,
        );
        sanitize_f64(
            "tracking.tackle_recover_max_distance_yards",
            &mut self.tackle_recover_max_distance_yards,
            default.tackle_recover_max_distance_yards,
            0.2,
            20.0,
            1.0,
            8.0,
        );
        sanitize_f64(
            "tracking.defend_closing_margin_yards",
            &mut self.defend_closing_margin_yards,
            default.defend_closing_margin_yards,
            0.0,
            8.0,
            0.05,
            2.0,
        );
        sanitize_f64(
            "tracking.space_improvement_threshold",
            &mut self.space_improvement_threshold,
            default.space_improvement_threshold,
            0.0,
            20.0,
            0.0,
            3.0,
        );
    }

    fn validate_strict(&self, prefix: &str, errors: &mut Vec<String>) {
        validate_f64(
            prefix,
            "moved_dt_multiplier",
            self.moved_dt_multiplier,
            0.05,
            12.0,
            errors,
        );
        validate_f64(
            prefix,
            "shot_lane_max_yards_to_goal",
            self.shot_lane_max_yards_to_goal,
            1.0,
            120.0,
            errors,
        );
        validate_f64(
            prefix,
            "tackle_recover_max_distance_yards",
            self.tackle_recover_max_distance_yards,
            0.2,
            20.0,
            errors,
        );
        validate_f64(
            prefix,
            "defend_closing_margin_yards",
            self.defend_closing_margin_yards,
            0.0,
            8.0,
            errors,
        );
        validate_f64(
            prefix,
            "space_improvement_threshold",
            self.space_improvement_threshold,
            0.0,
            20.0,
            errors,
        );
    }
}

impl FlankCrossTunables {
    fn sanitize(&mut self) {
        let default = FlankCrossTunables::default();
        sanitize_f64(
            "flank_cross.min_flank_score",
            &mut self.min_flank_score,
            default.min_flank_score,
            0.0,
            1.0,
            0.10,
            0.90,
        );
        sanitize_f64(
            "flank_cross.max_yards_to_goal",
            &mut self.max_yards_to_goal,
            default.max_yards_to_goal,
            1.0,
            120.0,
            20.0,
            80.0,
        );
        sanitize_f64(
            "flank_cross.perceived_pressure_weight",
            &mut self.perceived_pressure_weight,
            default.perceived_pressure_weight,
            0.0,
            5.0,
            0.0,
            1.5,
        );
        sanitize_f64(
            "flank_cross.pressure_urgency_weight",
            &mut self.pressure_urgency_weight,
            default.pressure_urgency_weight,
            0.0,
            5.0,
            0.0,
            1.5,
        );
        sanitize_f64(
            "flank_cross.pressure_release_cap",
            &mut self.pressure_release_cap,
            default.pressure_release_cap,
            0.0,
            1.0,
            0.0,
            0.85,
        );
        sanitize_f64(
            "flank_cross.flank_weight",
            &mut self.flank_weight,
            default.flank_weight,
            0.0,
            5.0,
            0.0,
            1.5,
        );
        sanitize_f64(
            "flank_cross.attacking_depth_weight",
            &mut self.attacking_depth_weight,
            default.attacking_depth_weight,
            0.0,
            5.0,
            0.0,
            1.5,
        );
    }

    fn validate_strict(&self, prefix: &str, errors: &mut Vec<String>) {
        validate_f64(
            prefix,
            "min_flank_score",
            self.min_flank_score,
            0.0,
            1.0,
            errors,
        );
        validate_f64(
            prefix,
            "max_yards_to_goal",
            self.max_yards_to_goal,
            1.0,
            120.0,
            errors,
        );
        validate_f64(
            prefix,
            "perceived_pressure_weight",
            self.perceived_pressure_weight,
            0.0,
            5.0,
            errors,
        );
        validate_f64(
            prefix,
            "pressure_urgency_weight",
            self.pressure_urgency_weight,
            0.0,
            5.0,
            errors,
        );
        validate_f64(
            prefix,
            "pressure_release_cap",
            self.pressure_release_cap,
            0.0,
            1.0,
            errors,
        );
        validate_f64(prefix, "flank_weight", self.flank_weight, 0.0, 5.0, errors);
        validate_f64(
            prefix,
            "attacking_depth_weight",
            self.attacking_depth_weight,
            0.0,
            5.0,
            errors,
        );
    }
}

impl RewardTunables {
    fn sanitize(&mut self) {
        let default = RewardTunables::default();
        sanitize_f64(
            "reward.goal_scored_points",
            &mut self.goal_scored_points,
            default.goal_scored_points,
            -1000.0,
            1000.0,
            0.0,
            250.0,
        );
        sanitize_f64(
            "reward.concede_keeper_defender_penalty",
            &mut self.concede_keeper_defender_penalty,
            default.concede_keeper_defender_penalty,
            0.0,
            200.0,
            0.0,
            40.0,
        );
        sanitize_f64(
            "reward.concede_outfield_penalty",
            &mut self.concede_outfield_penalty,
            default.concede_outfield_penalty,
            0.0,
            200.0,
            0.0,
            25.0,
        );
        sanitize_f64(
            "reward.teammate_overlap_relief_reward",
            &mut self.teammate_overlap_relief_reward,
            default.teammate_overlap_relief_reward,
            -10.0,
            10.0,
            0.0,
            2.0,
        );
        sanitize_f64(
            "reward.teammate_overlap_camp_penalty",
            &mut self.teammate_overlap_camp_penalty,
            default.teammate_overlap_camp_penalty,
            0.0,
            10.0,
            0.0,
            2.0,
        );
        sanitize_f64(
            "reward.center_back_ahead_of_wingback_penalty_per_yard",
            &mut self.center_back_ahead_of_wingback_penalty_per_yard,
            default.center_back_ahead_of_wingback_penalty_per_yard,
            0.0,
            10.0,
            0.0,
            2.0,
        );
        sanitize_f64(
            "reward.blocked_lane_floor_pass_penalty_points",
            &mut self.blocked_lane_floor_pass_penalty_points,
            default.blocked_lane_floor_pass_penalty_points,
            0.0,
            200.0,
            0.0,
            40.0,
        );
        sanitize_f64(
            "reward.low_pressure_forced_pass_penalty_points",
            &mut self.low_pressure_forced_pass_penalty_points,
            default.low_pressure_forced_pass_penalty_points,
            0.0,
            200.0,
            0.0,
            25.0,
        );
        sanitize_f64(
            "reward.pitch_value_threat_delta_points",
            &mut self.pitch_value_threat_delta_points,
            default.pitch_value_threat_delta_points,
            0.0,
            500.0,
            0.0,
            60.0,
        );
    }

    fn validate_strict(&self, prefix: &str, errors: &mut Vec<String>) {
        validate_f64(
            prefix,
            "goal_scored_points",
            self.goal_scored_points,
            -1000.0,
            1000.0,
            errors,
        );
        validate_f64(
            prefix,
            "concede_keeper_defender_penalty",
            self.concede_keeper_defender_penalty,
            0.0,
            200.0,
            errors,
        );
        validate_f64(
            prefix,
            "concede_outfield_penalty",
            self.concede_outfield_penalty,
            0.0,
            200.0,
            errors,
        );
        validate_f64(
            prefix,
            "teammate_overlap_relief_reward",
            self.teammate_overlap_relief_reward,
            -10.0,
            10.0,
            errors,
        );
        validate_f64(
            prefix,
            "teammate_overlap_camp_penalty",
            self.teammate_overlap_camp_penalty,
            0.0,
            10.0,
            errors,
        );
        validate_f64(
            prefix,
            "center_back_ahead_of_wingback_penalty_per_yard",
            self.center_back_ahead_of_wingback_penalty_per_yard,
            0.0,
            10.0,
            errors,
        );
        validate_f64(
            prefix,
            "blocked_lane_floor_pass_penalty_points",
            self.blocked_lane_floor_pass_penalty_points,
            0.0,
            200.0,
            errors,
        );
        validate_f64(
            prefix,
            "low_pressure_forced_pass_penalty_points",
            self.low_pressure_forced_pass_penalty_points,
            0.0,
            200.0,
            errors,
        );
        validate_f64(
            prefix,
            "pitch_value_threat_delta_points",
            self.pitch_value_threat_delta_points,
            0.0,
            500.0,
            errors,
        );
    }
}

impl DecisionMpcTunables {
    fn sanitize(&mut self) {
        let default = DecisionMpcTunables::default();
        sanitize_f64(
            "decision_mpc.deviation_trace_threshold_yards",
            &mut self.deviation_trace_threshold_yards,
            default.deviation_trace_threshold_yards,
            0.0,
            60.0,
            0.1,
            8.0,
        );
        sanitize_f64(
            "decision_mpc.blend_max_target_delta_yards",
            &mut self.blend_max_target_delta_yards,
            default.blend_max_target_delta_yards,
            0.0,
            120.0,
            1.0,
            25.0,
        );
        sanitize_f64(
            "decision_mpc.reselect_max_target_delta_yards",
            &mut self.reselect_max_target_delta_yards,
            default.reselect_max_target_delta_yards,
            0.1,
            120.0,
            3.0,
            45.0,
        );
        sanitize_f64(
            "decision_mpc.reselect_min_execution_confidence",
            &mut self.reselect_min_execution_confidence,
            default.reselect_min_execution_confidence,
            0.0,
            1.0,
            0.0,
            0.8,
        );
        sanitize_f64(
            "decision_mpc.reselect_min_ball_execution_probability",
            &mut self.reselect_min_ball_execution_probability,
            default.reselect_min_ball_execution_probability,
            0.0,
            1.0,
            0.0,
            0.8,
        );
    }

    fn validate_strict(&self, prefix: &str, errors: &mut Vec<String>) {
        validate_f64(
            prefix,
            "deviation_trace_threshold_yards",
            self.deviation_trace_threshold_yards,
            0.0,
            60.0,
            errors,
        );
        validate_f64(
            prefix,
            "blend_max_target_delta_yards",
            self.blend_max_target_delta_yards,
            0.0,
            120.0,
            errors,
        );
        validate_f64(
            prefix,
            "reselect_max_target_delta_yards",
            self.reselect_max_target_delta_yards,
            0.1,
            120.0,
            errors,
        );
        validate_f64(
            prefix,
            "reselect_min_execution_confidence",
            self.reselect_min_execution_confidence,
            0.0,
            1.0,
            errors,
        );
        validate_f64(
            prefix,
            "reselect_min_ball_execution_probability",
            self.reselect_min_ball_execution_probability,
            0.0,
            1.0,
            errors,
        );
    }
}

impl ShootingTunables {
    fn sanitize(&mut self) {
        let default = ShootingTunables::default();
        sanitize_f64(
            "shooting.shot_on_frame_min_probability",
            &mut self.shot_on_frame_min_probability,
            default.shot_on_frame_min_probability,
            0.0,
            1.0,
            0.05,
            0.95,
        );
        sanitize_f64(
            "shooting.striker_shot_window_yards",
            &mut self.striker_shot_window_yards,
            default.striker_shot_window_yards,
            1.0,
            120.0,
            8.0,
            55.0,
        );
        sanitize_f64(
            "shooting.shot_block_bailout_max_probability",
            &mut self.shot_block_bailout_max_probability,
            default.shot_block_bailout_max_probability,
            0.0,
            1.0,
            0.10,
            0.98,
        );
        sanitize_f64(
            "shooting.goal_approach_carry_yards",
            &mut self.goal_approach_carry_yards,
            default.goal_approach_carry_yards,
            1.0,
            120.0,
            15.0,
            80.0,
        );
        sanitize_f64(
            "shooting.striker_hold_up_min_goal_distance_yards",
            &mut self.striker_hold_up_min_goal_distance_yards,
            default.striker_hold_up_min_goal_distance_yards,
            1.0,
            120.0,
            15.0,
            80.0,
        );
        sanitize_f64(
            "shooting.shot_trigger_long_range_yards",
            &mut self.shot_trigger_long_range_yards,
            default.shot_trigger_long_range_yards,
            1.0,
            40.0,
            12.0,
            35.0,
        );
        sanitize_f64(
            "shooting.shot_trigger_long_range_min_value",
            &mut self.shot_trigger_long_range_min_value,
            default.shot_trigger_long_range_min_value,
            0.0,
            1.0,
            0.0,
            0.9,
        );
        sanitize_f64(
            "shooting.shot_trigger_volunteer_floor",
            &mut self.shot_trigger_volunteer_floor,
            default.shot_trigger_volunteer_floor,
            0.0,
            1.0,
            0.0,
            0.5,
        );
        sanitize_f64(
            "shooting.shot_foot_weak_foot_execution_damp",
            &mut self.shot_foot_weak_foot_execution_damp,
            default.shot_foot_weak_foot_execution_damp,
            0.0,
            1.0,
            0.5,
            1.0,
        );
    }

    fn validate_strict(&self, prefix: &str, errors: &mut Vec<String>) {
        validate_f64(
            prefix,
            "shot_on_frame_min_probability",
            self.shot_on_frame_min_probability,
            0.0,
            1.0,
            errors,
        );
        validate_f64(
            prefix,
            "striker_shot_window_yards",
            self.striker_shot_window_yards,
            1.0,
            120.0,
            errors,
        );
        validate_f64(
            prefix,
            "shot_block_bailout_max_probability",
            self.shot_block_bailout_max_probability,
            0.0,
            1.0,
            errors,
        );
        validate_f64(
            prefix,
            "goal_approach_carry_yards",
            self.goal_approach_carry_yards,
            1.0,
            120.0,
            errors,
        );
        validate_f64(
            prefix,
            "striker_hold_up_min_goal_distance_yards",
            self.striker_hold_up_min_goal_distance_yards,
            1.0,
            120.0,
            errors,
        );
        validate_f64(
            prefix,
            "shot_trigger_long_range_yards",
            self.shot_trigger_long_range_yards,
            1.0,
            40.0,
            errors,
        );
        validate_f64(
            prefix,
            "shot_trigger_long_range_min_value",
            self.shot_trigger_long_range_min_value,
            0.0,
            1.0,
            errors,
        );
        validate_f64(
            prefix,
            "shot_trigger_volunteer_floor",
            self.shot_trigger_volunteer_floor,
            0.0,
            1.0,
            errors,
        );
        validate_f64(
            prefix,
            "shot_foot_weak_foot_execution_damp",
            self.shot_foot_weak_foot_execution_damp,
            0.0,
            1.0,
            errors,
        );
    }
}

impl DefensiveShapeTunables {
    fn sanitize(&mut self) {
        let default = DefensiveShapeTunables::default();
        sanitize_f64(
            "defensive_shape.defensive_line_max_into_opp_half_yards",
            &mut self.defensive_line_max_into_opp_half_yards,
            default.defensive_line_max_into_opp_half_yards,
            0.0,
            30.0,
            0.0,
            12.0,
        );
        sanitize_f64(
            "defensive_shape.back_four_block_width_yards",
            &mut self.back_four_block_width_yards,
            default.back_four_block_width_yards,
            4.0,
            80.0,
            12.0,
            45.0,
        );
        sanitize_f64(
            "defensive_shape.back_four_horizontal_min_gap_yards",
            &mut self.back_four_horizontal_min_gap_yards,
            default.back_four_horizontal_min_gap_yards,
            0.5,
            20.0,
            1.0,
            6.0,
        );
        sanitize_f64(
            "defensive_shape.back_four_horizontal_max_gap_yards",
            &mut self.back_four_horizontal_max_gap_yards,
            default.back_four_horizontal_max_gap_yards,
            0.5,
            30.0,
            3.0,
            16.0,
        );
        sanitize_f64(
            "defensive_shape.wingback_defensive_pinch_target_seconds",
            &mut self.wingback_defensive_pinch_target_seconds,
            default.wingback_defensive_pinch_target_seconds,
            0.0,
            20.0,
            0.5,
            8.0,
        );
        sanitize_f64(
            "defensive_shape.wingback_defensive_pinch_opponent_half_margin_yards",
            &mut self.wingback_defensive_pinch_opponent_half_margin_yards,
            default.wingback_defensive_pinch_opponent_half_margin_yards,
            0.0,
            40.0,
            2.0,
            20.0,
        );
        sanitize_f64(
            "defensive_shape.defensive_goal_side_min_yards",
            &mut self.defensive_goal_side_min_yards,
            default.defensive_goal_side_min_yards,
            0.0,
            10.0,
            0.5,
            4.0,
        );
        if self.back_four_horizontal_min_gap_yards > self.back_four_horizontal_max_gap_yards {
            eprintln!(
                "soccer tunables: defensive_shape back-four min gap exceeded max gap; swapping"
            );
            std::mem::swap(
                &mut self.back_four_horizontal_min_gap_yards,
                &mut self.back_four_horizontal_max_gap_yards,
            );
        }
    }

    fn validate_strict(&self, prefix: &str, errors: &mut Vec<String>) {
        validate_f64(
            prefix,
            "defensive_line_max_into_opp_half_yards",
            self.defensive_line_max_into_opp_half_yards,
            0.0,
            30.0,
            errors,
        );
        validate_f64(
            prefix,
            "back_four_block_width_yards",
            self.back_four_block_width_yards,
            4.0,
            80.0,
            errors,
        );
        validate_f64(
            prefix,
            "back_four_horizontal_min_gap_yards",
            self.back_four_horizontal_min_gap_yards,
            0.5,
            20.0,
            errors,
        );
        validate_f64(
            prefix,
            "back_four_horizontal_max_gap_yards",
            self.back_four_horizontal_max_gap_yards,
            0.5,
            30.0,
            errors,
        );
        validate_f64(
            prefix,
            "wingback_defensive_pinch_target_seconds",
            self.wingback_defensive_pinch_target_seconds,
            0.0,
            20.0,
            errors,
        );
        validate_f64(
            prefix,
            "wingback_defensive_pinch_opponent_half_margin_yards",
            self.wingback_defensive_pinch_opponent_half_margin_yards,
            0.0,
            40.0,
            errors,
        );
        validate_f64(
            prefix,
            "defensive_goal_side_min_yards",
            self.defensive_goal_side_min_yards,
            0.0,
            10.0,
            errors,
        );
        if self.back_four_horizontal_min_gap_yards > self.back_four_horizontal_max_gap_yards {
            errors.push(format!("{prefix}.back_four_horizontal_min_gap_yards > {prefix}.back_four_horizontal_max_gap_yards"));
        }
    }
}

fn sanitize_f64(
    path: &str,
    value: &mut f64,
    default: f64,
    hard_min: f64,
    hard_max: f64,
    sane_min: f64,
    sane_max: f64,
) {
    if !value.is_finite() {
        eprintln!("soccer tunables: {path} is non-finite; using default {default}");
        *value = default;
        return;
    }
    if *value < hard_min || *value > hard_max {
        let before = *value;
        *value = (*value).clamp(hard_min, hard_max);
        eprintln!(
            "soccer tunables: {path}={before} outside hard range [{hard_min}, {hard_max}]; clamped to {}",
            *value
        );
        return;
    }
    if *value < sane_min || *value > sane_max {
        eprintln!(
            "soccer tunables: {path}={} outside sane range [{sane_min}, {sane_max}]",
            *value
        );
    }
}

fn validate_f64(
    prefix: &str,
    field: &str,
    value: f64,
    hard_min: f64,
    hard_max: f64,
    errors: &mut Vec<String>,
) {
    if !value.is_finite() {
        errors.push(format!("{prefix}.{field} is non-finite"));
    } else if value < hard_min || value > hard_max {
        errors.push(format!(
            "{prefix}.{field}={value} outside [{hard_min}, {hard_max}]"
        ));
    }
}

fn sanitize_usize(
    path: &str,
    value: &mut usize,
    default: usize,
    hard_min: usize,
    hard_max: usize,
    sane_min: usize,
    sane_max: usize,
) {
    if *value < hard_min || *value > hard_max {
        let before = *value;
        *value = (*value).clamp(hard_min, hard_max);
        eprintln!(
            "soccer tunables: {path}={before} outside hard range [{hard_min}, {hard_max}]; clamped to {}",
            *value
        );
        return;
    }
    if *value < sane_min || *value > sane_max {
        eprintln!(
            "soccer tunables: {path}={} outside sane range [{sane_min}, {sane_max}], default {default}",
            *value
        );
    }
}

fn validate_usize(
    prefix: &str,
    field: &str,
    value: usize,
    hard_min: usize,
    hard_max: usize,
    errors: &mut Vec<String>,
) {
    if value < hard_min || value > hard_max {
        errors.push(format!(
            "{prefix}.{field}={value} outside [{hard_min}, {hard_max}]"
        ));
    }
}

/// Recursively merge `overlay` into `base`: objects are merged key-by-key, any
/// other value (scalar, array) replaces wholesale. Mirrors how a config patch
/// should behave.
fn merge_json(base: &mut Value, overlay: Value) {
    match (base, overlay) {
        (Value::Object(base_map), Value::Object(overlay_map)) => {
            for (key, value) in overlay_map {
                merge_json(base_map.entry(key).or_insert(Value::Null), value);
            }
        }
        (base_slot, overlay) => *base_slot = overlay,
    }
}

/// Collect the environment override layers, lowest-priority first:
/// 1. `SOCCER_TUNABLES_JSON` — a full or partial JSON object.
/// 2. each `DD_SOCCER_TUNABLE__<dotted.path>=<json-or-string>` as a one-field
///    overlay (so `DD_SOCCER_TUNABLE__tracking.moved_dt_multiplier=1.25` sets
///    exactly that leaf).
fn env_overlays() -> Vec<Value> {
    let mut overlays = Vec::new();

    if let Ok(blob) = std::env::var("SOCCER_TUNABLES_JSON") {
        match serde_json::from_str::<Value>(&blob) {
            Ok(value) => overlays.push(value),
            Err(err) => {
                eprintln!("soccer tunables: ignoring malformed SOCCER_TUNABLES_JSON: {err}")
            }
        }
    }

    const FIELD_PREFIX: &str = "DD_SOCCER_TUNABLE__";
    let mut field_overrides: Vec<(String, String)> = std::env::vars()
        .filter_map(|(key, value)| {
            key.strip_prefix(FIELD_PREFIX)
                .map(|p| (p.to_string(), value))
        })
        .collect();
    // Deterministic application order regardless of env iteration order.
    field_overrides.sort();
    for (path, raw) in field_overrides {
        // Parse the value as JSON (numbers/bools/objects), falling back to a
        // bare string for unquoted text like `DD_SOCCER_TUNABLE__x=foo`.
        let leaf = serde_json::from_str::<Value>(&raw).unwrap_or(Value::String(raw));
        overlays.push(nest_dotted_path(&path, leaf));
    }

    overlays
}

/// Turn a dotted path + leaf value into a nested JSON object,
/// e.g. `("tracking.moved_dt_multiplier", 1.25)` =>
/// `{"tracking": {"moved_dt_multiplier": 1.25}}`.
fn nest_dotted_path(path: &str, leaf: Value) -> Value {
    let mut value = leaf;
    for segment in path.split('.').rev() {
        let mut map = serde_json::Map::new();
        map.insert(segment.to_string(), value);
        value = Value::Object(map);
    }
    value
}

static TUNABLES: OnceLock<Tunables> = OnceLock::new();

/// The live, process-wide tunables. Lazily materialised from the environment
/// layers on first access. If Postgres overrides are wanted they must be
/// installed via [`init_tunables_with_overrides`] *before* the first call here.
pub fn tunables() -> &'static Tunables {
    TUNABLES.get_or_init(Tunables::from_env)
}

/// Install the process tunables built from defaults + environment + the given
/// Postgres/extra JSON overlays (applied last, highest priority). Returns
/// `true` if this call performed the initialisation, `false` if the registry
/// was already initialised (in which case the overrides are ignored — call
/// this once at startup before any `tunables()` read).
pub fn init_tunables_with_overrides<I: IntoIterator<Item = Value>>(pg_overlays: I) -> bool {
    let mut overlays = env_overlays();
    overlays.extend(pg_overlays);
    let built = Tunables::from_overlays(overlays);
    let mut installed = false;
    let _ = TUNABLES.get_or_init(|| {
        installed = true;
        built
    });
    installed
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn defaults_match_historical_literals() {
        let t = Tunables::default();
        assert_eq!(t.tracking.moved_dt_multiplier, 1.2);
        assert_eq!(t.tracking.shot_lane_max_yards_to_goal, 25.0);
        assert_eq!(t.tracking.tackle_recover_max_distance_yards, 3.8);
        assert_eq!(t.tracking.defend_closing_margin_yards, 0.35);
        assert_eq!(t.tracking.space_improvement_threshold, 0.25);
        assert_eq!(t.decision_mpc.deviation_trace_threshold_yards, 0.75);
        assert_eq!(t.decision_mpc.blend_max_target_delta_yards, 6.0);
        assert_eq!(t.decision_mpc.reselect_max_target_delta_yards, 16.0);
        assert_eq!(t.decision_mpc.reselect_min_execution_confidence, 0.18);
        assert_eq!(t.decision_mpc.reselect_min_ball_execution_probability, 0.34);
        assert_eq!(t.shooting.shot_on_frame_min_probability, 0.60);
        assert_eq!(t.shooting.striker_shot_window_yards, 30.0);
        assert_eq!(t.shooting.shot_block_bailout_max_probability, 0.86);
        assert_eq!(t.shooting.goal_approach_carry_yards, 45.0);
        assert_eq!(t.shooting.striker_hold_up_min_goal_distance_yards, 45.0);
        assert_eq!(
            t.defensive_shape.defensive_line_max_into_opp_half_yards,
            5.0
        );
        assert_eq!(t.defensive_shape.back_four_block_width_yards, 22.0);
        assert_eq!(t.defensive_shape.back_four_horizontal_min_gap_yards, 1.5);
        assert_eq!(t.defensive_shape.back_four_horizontal_max_gap_yards, 8.0);
        assert_eq!(
            t.defensive_shape.wingback_defensive_pinch_target_seconds,
            3.0
        );
        assert_eq!(
            t.defensive_shape
                .wingback_defensive_pinch_opponent_half_margin_yards,
            8.0
        );
        assert_eq!(t.defensive_shape.defensive_goal_side_min_yards, 1.5);
        assert_eq!(t.lane_affinity.forward_lane_radius, 3);
        assert_eq!(t.lane_affinity.defender_commitment_possession, 0.80);
        assert_eq!(t.lane_affinity.possession_factor, 0.80);
        assert_eq!(t.lane_affinity.row_match_span_rows, 8.0);
        assert_eq!(t.lane_affinity.open_space_dynamic_lane_bonus_weight, 1.05);
        assert_eq!(t.lane_affinity.movement_shape_dynamic_lane_weight, 0.36);
    }

    #[test]
    fn partial_overlay_only_touches_named_fields() {
        let t = Tunables::from_overlays([json!({
            "tracking": { "moved_dt_multiplier": 1.5 },
            "decision_mpc": { "reselect_min_ball_execution_probability": 0.42 },
            "defensive_shape": { "defensive_line_max_into_opp_half_yards": 4.0 },
            "lane_affinity": {
                "possession_factor": 0.74,
                "open_space_dynamic_lane_bonus_weight": 1.3
            }
        })]);
        assert_eq!(t.tracking.moved_dt_multiplier, 1.5);
        assert_eq!(t.decision_mpc.reselect_min_ball_execution_probability, 0.42);
        assert_eq!(
            t.defensive_shape.defensive_line_max_into_opp_half_yards,
            4.0
        );
        assert_eq!(t.lane_affinity.possession_factor, 0.74);
        assert_eq!(t.lane_affinity.open_space_dynamic_lane_bonus_weight, 1.3);
        // Untouched fields keep their defaults.
        assert_eq!(t.tracking.tackle_recover_max_distance_yards, 3.8);
        assert_eq!(t.shooting.shot_block_bailout_max_probability, 0.86);
        assert_eq!(t.lane_affinity.movement_shape_dynamic_lane_weight, 0.36);
    }

    #[test]
    fn later_overlay_wins() {
        let t = Tunables::from_overlays([
            json!({ "tracking": { "moved_dt_multiplier": 1.5 } }),
            json!({ "tracking": { "moved_dt_multiplier": 2.0 } }),
        ]);
        assert_eq!(t.tracking.moved_dt_multiplier, 2.0);
    }

    #[test]
    fn dotted_path_nests_correctly() {
        let nested = nest_dotted_path("tracking.moved_dt_multiplier", json!(1.25));
        assert_eq!(
            nested,
            json!({ "tracking": { "moved_dt_multiplier": 1.25 } })
        );
    }

    #[test]
    fn merge_replaces_scalars_and_merges_objects() {
        let mut base = json!({ "a": { "x": 1, "y": 2 }, "b": 3 });
        merge_json(&mut base, json!({ "a": { "y": 9 }, "c": 4 }));
        assert_eq!(base, json!({ "a": { "x": 1, "y": 9 }, "b": 3, "c": 4 }));
    }

    #[test]
    fn overlays_are_sanitized_to_hard_bounds() {
        let t = Tunables::from_overlays([json!({
            "decision_mpc": { "reselect_min_execution_confidence": -1.0 },
            "shooting": { "shot_block_bailout_max_probability": 4.0 },
            "defensive_shape": {
                "back_four_horizontal_min_gap_yards": 12.0,
                "back_four_horizontal_max_gap_yards": 4.0
            },
            "lane_affinity": {
                "possession_factor": 4.0,
                "forward_lane_radius": 99,
                "lookahead_min_seconds": 2.0,
                "lookahead_max_seconds": 0.5
            }
        })]);
        assert_eq!(t.decision_mpc.reselect_min_execution_confidence, 0.0);
        assert_eq!(t.shooting.shot_block_bailout_max_probability, 1.0);
        assert_eq!(t.defensive_shape.back_four_horizontal_min_gap_yards, 4.0);
        assert_eq!(t.defensive_shape.back_four_horizontal_max_gap_yards, 12.0);
        assert_eq!(t.lane_affinity.possession_factor, 1.0);
        assert_eq!(t.lane_affinity.forward_lane_radius, 6);
        assert_eq!(t.lane_affinity.lookahead_min_seconds, 0.5);
        assert_eq!(t.lane_affinity.lookahead_max_seconds, 2.0);
    }

    #[test]
    fn strict_validation_reports_bad_direct_configs() {
        let mut t = Tunables::default();
        t.shooting.shot_on_frame_min_probability = 1.2;
        t.defensive_shape.back_four_horizontal_min_gap_yards = 9.0;
        t.defensive_shape.back_four_horizontal_max_gap_yards = 6.0;
        t.lane_affinity.possession_factor = 1.2;
        t.lane_affinity.forward_lane_radius = 9;

        let err = t.validate_strict().expect_err("config should be invalid");
        assert!(err.contains("shooting.shot_on_frame_min_probability"));
        assert!(err.contains("defensive_shape.back_four_horizontal_min_gap_yards >"));
        assert!(err.contains("lane_affinity.possession_factor"));
        assert!(err.contains("lane_affinity.forward_lane_radius"));
    }
}
