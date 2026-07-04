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

pub(crate) const KILLER_PASS_OVER_TOP_PASS_FLIGHT_ALIASES: &[&str] = &[
    "overtop",
    "overhitbackfour",
    "killerlob",
    "killerloft",
    "threadedlob",
];
pub(crate) const KILLER_PASS_OVER_TOP_DISTANCE_BINS_YARDS: [f64; 4] = [25.0, 28.0, 32.0, 35.0];
pub(crate) const KILLER_PASS_OVER_TOP_HEIGHT_BINS_YARDS: [f64; 4] = [2.0, 3.4, 4.6, 6.0];
pub(crate) const KILLER_PASS_OVER_TOP_LATERAL_OFFSET_BINS_YARDS: [f64; 4] =
    [1.0, 3.0, 5.0, 8.0];
pub(crate) const KILLER_PASS_OVER_TOP_BACK_LINE_CLEARANCE_BINS_YARDS: [f64; 4] =
    [0.5, 2.0, 4.0, 7.0];
pub(crate) const KILLER_PASS_OVER_TOP_GOALKEEPER_AVOIDANCE_BINS_YARDS: [f64; 4] =
    [2.0, 4.0, 6.0, 9.0];
pub(crate) const KILLER_PASS_OVER_TOP_NEURAL_LATERAL_NORMALIZER_FACTOR: f64 = 2.0;
pub(crate) const KILLER_PASS_OVER_TOP_NEURAL_BACK_LINE_CLEARANCE_NORMALIZER_YARDS: f64 = 10.0;
pub(crate) const KILLER_PASS_OVER_TOP_NEURAL_GOALKEEPER_AVOIDANCE_NORMALIZER_YARDS: f64 = 16.0;
pub(crate) const KILLER_PASS_OVER_TOP_PITCH_MARGIN_CAP_FACTOR: f64 = 0.45;
pub(crate) const KILLER_PASS_OVER_TOP_NUMERIC_EPSILON: f64 = 1e-6;

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
    /// Ball-carrier anti-stutter / keep-rolling thresholds.
    pub carrier_keep_rolling: CarrierKeepRollingTunables,
    /// Good-dribbling knobs: touch DIRECTION weighting, touch LENGTH (how far the ball rolls per
    /// touch), and carry SPEED (the forward-drive gait floor). Read by both the deterministic/MPC
    /// touch-decision path and tunable from the learning store's policy overlay (the POMDP tuning
    /// row), so dribbling can be tuned via MPC and POMDP both. See [`DribbleTuning`].
    pub dribble: DribbleTuning,
    /// Freshly-won possession escape burst thresholds.
    pub fresh_possession_escape: FreshPossessionEscapeTunables,
    /// Goalkeeper positioning shaping (resting line, alignment score, buildup
    /// fan-out, loose-ball commit). See [`GoalkeeperTunables`].
    pub goalkeeper: GoalkeeperTunables,
    /// 25-35yd clipped killer-pass over the opponent back four.
    pub killer_pass_over_top: KillerPassOverTopTunables,
    /// Fine-grid lane/row affinity used by formation, support, and retrieval.
    pub lane_affinity: LaneAffinityTunables,
    /// Centralized lane-discipline weights (12-lane grid). Read only when
    /// `DD_SOCCER_ENABLE_LANE_DISCIPLINE_V2` is set; see
    /// [`crate::des::general::soccer::lane_discipline`].
    pub lane_discipline: LaneDisciplineTunables,
    /// Per-rank weights for stochastic top-k policy selection. Read only when
    /// `DD_SOCCER_ENABLE_STOCHASTIC_POLICY_TOPK` is set; see
    /// [`crate::des::general::soccer::policy_select`].
    pub policy_selection: PolicySelectionTunables,
    /// POMDP perception model for ball-holder head turns, shoulder checks, and
    /// delayed 360-degree pass recognition.
    pub pomdp_perception: PomdpPerceptionTunables,
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
            carrier_keep_rolling: CarrierKeepRollingTunables::default(),
            dribble: DribbleTuning::default(),
            fresh_possession_escape: FreshPossessionEscapeTunables::default(),
            goalkeeper: GoalkeeperTunables::default(),
            killer_pass_over_top: KillerPassOverTopTunables::default(),
            lane_affinity: LaneAffinityTunables::default(),
            lane_discipline: LaneDisciplineTunables::default(),
            policy_selection: PolicySelectionTunables::default(),
            pomdp_perception: PomdpPerceptionTunables::default(),
        }
    }
}

/// Ball-holder POMDP perception knobs. A carrier can see the core body-facing
/// cone immediately, shoulder-check wider lanes, and head-scan the rest of the
/// pitch with lower confidence plus a realistic delay.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct PomdpPerceptionTunables {
    /// Minimum bounded human reaction delay carried in the POMDP observation.
    pub player_reaction_min_seconds: f64,
    /// Maximum bounded human reaction delay carried in the POMDP observation.
    pub player_reaction_max_seconds: f64,
    /// Full body-facing vision cone, in degrees. Previously
    /// `BALL_HOLDER_CORE_VISION_DEGREES`.
    pub ball_holder_core_degrees: f64,
    /// Additional degrees reachable by a quick shoulder check on either side.
    /// Previously `BALL_HOLDER_SHOULDER_VISION_DEGREES`.
    pub ball_holder_shoulder_degrees: f64,
    /// Position confidence inside the core body-facing cone.
    pub ball_holder_core_confidence: f64,
    /// Position confidence inside shoulder-check range.
    pub ball_holder_shoulder_confidence: f64,
    /// Base position confidence for side head scans.
    pub ball_holder_side_scan_confidence: f64,
    /// Base position confidence for rear head scans.
    pub ball_holder_rear_scan_confidence: f64,
    /// Minimum added perception delay for non-forward head scans.
    pub ball_holder_head_scan_min_seconds: f64,
    /// Maximum added perception delay for non-forward head scans.
    pub ball_holder_head_scan_max_seconds: f64,
    /// Amount high-vision players shave from the maximum scan delay.
    pub ball_holder_head_scan_vision_relief_seconds: f64,
    /// Extra confidence high-vision players add while scanning side/rear lanes.
    pub ball_holder_scan_confidence_vision_bonus: f64,
    /// Maximum confidence for side/rear head-scan recognition.
    pub ball_holder_scan_confidence_cap: f64,
    /// Baseline drift risk recorded in the POMDP observation for any head scan.
    pub ball_holder_head_scan_drift_risk_base: f64,
    /// Extra drift risk added as the scan delay approaches its maximum.
    pub ball_holder_head_scan_drift_risk_span: f64,
    /// Radius for matching a perceived point against recent Kalman history.
    pub kalman_point_match_radius_yards: f64,
    /// Minimum history samples before the Kalman confidence bridge is trusted.
    pub kalman_min_history_samples: usize,
    /// Distance treated as a current-position sample rather than a prediction.
    pub kalman_current_sample_epsilon_yards: f64,
    /// Baseline uncertainty for stale/predicted player positions.
    pub kalman_base_sigma_yards: f64,
    /// Additional uncertainty per yard-per-second of target speed.
    pub kalman_speed_sigma_per_yps: f64,
    /// Additional uncertainty for velocity disagreement between history samples.
    pub kalman_velocity_disagreement_sigma_yards: f64,
    /// Confidence cap for currently visible, history-supported positions.
    pub kalman_visible_max_confidence: f64,
    /// Confidence cap for forward-but-not-currently-visible positions.
    pub kalman_front_max_confidence: f64,
    /// Confidence cap for generally occluded positions.
    pub kalman_occluded_max_confidence: f64,
    /// Confidence cap for ball-holder side/rear scanned but occluded positions.
    pub kalman_ball_holder_occluded_max_confidence: f64,
}

impl Default for PomdpPerceptionTunables {
    fn default() -> Self {
        PomdpPerceptionTunables {
            player_reaction_min_seconds: 0.10,
            player_reaction_max_seconds: 0.25,
            ball_holder_core_degrees: 100.0,
            ball_holder_shoulder_degrees: 40.0,
            ball_holder_core_confidence: 0.90,
            ball_holder_shoulder_confidence: 0.70,
            ball_holder_side_scan_confidence: 0.52,
            ball_holder_rear_scan_confidence: 0.38,
            ball_holder_head_scan_min_seconds: 0.75,
            ball_holder_head_scan_max_seconds: 1.85,
            ball_holder_head_scan_vision_relief_seconds: 0.35,
            ball_holder_scan_confidence_vision_bonus: 0.08,
            ball_holder_scan_confidence_cap: 0.62,
            ball_holder_head_scan_drift_risk_base: 0.24,
            ball_holder_head_scan_drift_risk_span: 0.42,
            kalman_point_match_radius_yards: 0.35,
            kalman_min_history_samples: 2,
            kalman_current_sample_epsilon_yards: 0.15,
            kalman_base_sigma_yards: 0.85,
            kalman_speed_sigma_per_yps: 0.10,
            kalman_velocity_disagreement_sigma_yards: 0.14,
            kalman_visible_max_confidence: 0.96,
            kalman_front_max_confidence: 0.88,
            kalman_occluded_max_confidence: 0.68,
            kalman_ball_holder_occluded_max_confidence: 0.58,
        }
    }
}

impl PomdpPerceptionTunables {
    pub fn ball_holder_shoulder_scan_limit_degrees(&self) -> f64 {
        self.ball_holder_core_degrees * 0.5 + self.ball_holder_shoulder_degrees
    }

    pub fn player_reaction_span_seconds(&self) -> f64 {
        (self.player_reaction_max_seconds - self.player_reaction_min_seconds).max(f64::EPSILON)
    }

    pub fn perception_latency_upper_bound_seconds(&self) -> f64 {
        self.player_reaction_max_seconds + self.ball_holder_head_scan_max_seconds
    }
}

/// Per-rank weights for **stochastic top-k policy selection** (see
/// [`crate::des::general::soccer::policy_select`]). When its gate
/// (`DD_SOCCER_ENABLE_STOCHASTIC_POLICY_TOPK`) is on, each MDP/POMDP decision
/// draws among its best three candidate actions with these probabilities
/// (renormalised over however many candidates exist), or — when
/// `boltzmann_temperature > 0` — proportional to `exp(score / temperature)` over
/// all candidates (value-weighted). With the gate off the engine takes the
/// deterministic argmax and these values are never read, so an unconfigured
/// process is byte-identical to before this group existed.
///
/// Defaults are the requested **70 / 20 / 10** rank split with the value-weighted
/// path off (`boltzmann_temperature = 0`).
pub const POLICY_SELECTION_TOP_RANK_LIMIT: usize = 3;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct PolicySelectionTunables {
    /// Probability weight of committing to the **best** (rank-0) candidate.
    pub top1_weight: f64,
    /// Probability weight of committing to the **2nd-best** (rank-1) candidate.
    pub top2_weight: f64,
    /// Probability weight of committing to the **3rd-best** (rank-2) candidate.
    pub top3_weight: f64,
    /// Boltzmann (softmax) temperature for **value-weighted** selection, in the
    /// same score units the ranker scores candidates in. When `> 0` *and* the
    /// stochastic gate is on, the decision draws over **all** candidates with
    /// probability `∝ exp(score / temperature)` instead of the fixed-by-rank
    /// `top{1,2,3}_weight` split — so a near-tie explores readily while a runaway
    /// best is taken almost always. A **non-positive** value (the default `0.0`)
    /// disables this path and falls back to the rank-weighted top-k, keeping an
    /// unconfigured process byte-identical. Smaller ⇒ greedier, larger ⇒ flatter.
    pub boltzmann_temperature: f64,
}

impl PolicySelectionTunables {
    pub fn rank_weights(&self) -> [f64; POLICY_SELECTION_TOP_RANK_LIMIT] {
        [self.top1_weight, self.top2_weight, self.top3_weight]
    }
}

impl Default for PolicySelectionTunables {
    fn default() -> Self {
        PolicySelectionTunables {
            top1_weight: 0.70,
            top2_weight: 0.20,
            top3_weight: 0.10,
            boltzmann_temperature: 0.0,
        }
    }
}

impl PolicySelectionTunables {
    /// Clamp obvious garbage to safe ranges with a warning. The selection code is
    /// already defensive (non-positive / non-finite weights are floored, and a
    /// non-positive / non-finite temperature falls back to the rank-weighted
    /// split), so this only turns silent misconfiguration into a visible warning
    /// and keeps the values in a sensible band. The per-rank weights are
    /// renormalised at use, so any non-negative magnitude is valid.
    fn sanitize(&mut self) {
        let default = PolicySelectionTunables::default();
        // Weights are renormalised at use, so any non-negative magnitude is valid
        // (e.g. [2,1,1] == 50/25/25). Clamp only negatives/non-finite (a hard
        // floor of 0 matches the downstream flooring of non-positive weights);
        // warn — but keep — values above 1.0 as "that looks unusual".
        sanitize_f64(
            "policy_selection.top1_weight",
            &mut self.top1_weight,
            default.top1_weight,
            0.0,
            f64::MAX,
            0.0,
            1.0,
        );
        sanitize_f64(
            "policy_selection.top2_weight",
            &mut self.top2_weight,
            default.top2_weight,
            0.0,
            f64::MAX,
            0.0,
            1.0,
        );
        sanitize_f64(
            "policy_selection.top3_weight",
            &mut self.top3_weight,
            default.top3_weight,
            0.0,
            f64::MAX,
            0.0,
            1.0,
        );
        // Temperature: 0 disables value-weighting; negatives are meaningless
        // (treated as disabled downstream) so clamp them up to 0. The hard upper
        // bound only catches absurd values — a very large temperature is a valid
        // "near-uniform exploration" choice, hence the wide sane band.
        sanitize_f64(
            "policy_selection.boltzmann_temperature",
            &mut self.boltzmann_temperature,
            default.boltzmann_temperature,
            0.0,
            1_000.0,
            0.0,
            100.0,
        );
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
    /// Flat penalty for giving the ball straight to the opponent (a turnover with
    /// no intentional-long-ball exemption) from a hold in the actor's OWN half —
    /// the most expensive giveaway, as it exposes the goal. The danger-scaled
    /// follow-on cost (the opponent then advancing) is priced separately and
    /// continuously by the net-Φ pitch-value term; this is the flat decision-step
    /// price of the giveaway itself. Was the inline `3.5`.
    pub giveaway_to_opponent_own_half_penalty: f64,
    /// Flat penalty for giving the ball straight to the opponent from a hold in
    /// the OPPONENT's half (less exposed than an own-half giveaway). Was the inline
    /// `2.2`.
    pub giveaway_to_opponent_opp_half_penalty: f64,
    /// Flat penalty for losing the ball into a loose/contested state (no team in
    /// settled possession after the touch) from the actor's OWN half — softer than
    /// a clean giveaway because the ball is still up for grabs. Was the inline
    /// `0.85`.
    pub giveaway_to_loose_own_half_penalty: f64,
    /// Flat penalty for losing the ball into a loose/contested state from the
    /// OPPONENT's half. Was the inline `0.55`.
    pub giveaway_to_loose_opp_half_penalty: f64,
    /// Scale on the dense **territorial pitch-control × expected-threat** delta
    /// reward (see [`crate::des::general::soccer::pitch_value`]). Multiplies the
    /// net change in the acting team's controlled threat between the before/after
    /// snapshots. Only applied when the reward is enabled by
    /// `DD_SOCCER_ENABLE_PITCH_VALUE_REWARD`; the gate keeps an unconfigured
    /// process byte-identical to before this term existed.
    pub pitch_value_threat_delta_points: f64,
    /// Per-step **dense shaping budget** (P4 / audit follow-up #3): the symmetric
    /// `±` cap applied to the dense per-step reward so accreted dense shards can't
    /// dominate the sparse match-outcome signal on a single step. Only active when
    /// `DD_SOCCER_ENABLE_SHAPING_DISCIPLINE` is set (off ⇒ byte-identical). The
    /// default is generous (catches pathological spikes without clipping a normal
    /// stacked action price); tighten it in an A/B to lean on the budget harder.
    pub dense_shaping_budget_points: f64,
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
            giveaway_to_opponent_own_half_penalty: 3.5,
            giveaway_to_opponent_opp_half_penalty: 2.2,
            giveaway_to_loose_own_half_penalty: 0.85,
            giveaway_to_loose_opp_half_penalty: 0.55,
            pitch_value_threat_delta_points: 12.0,
            dense_shaping_budget_points: 12.0,
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
            back_four_horizontal_min_gap_yards: 6.0,
            back_four_horizontal_max_gap_yards: 15.0,
            wingback_defensive_pinch_target_seconds: 3.0,
            wingback_defensive_pinch_opponent_half_margin_yards: 8.0,
            defensive_goal_side_min_yards: 1.5,
        }
    }
}

/// Ball-carrier keep-rolling thresholds. These govern the execution-layer guard
/// that turns an unpressured holder's short settle target into a real carry so
/// the carrier does not flicker walk/stop/walk while the ball would naturally
/// keep rolling.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct CarrierKeepRollingTunables {
    pub enabled: bool,
    pub stop_target_yards: f64,
    pub min_opponent_distance_yards: f64,
    pub min_space_yards: f64,
    pub carry_target_yards: f64,
    pub carry_min_step_yards: f64,
    pub momentum_yps: f64,
}

impl Default for CarrierKeepRollingTunables {
    fn default() -> Self {
        CarrierKeepRollingTunables {
            enabled: true,
            stop_target_yards: 2.75,
            min_opponent_distance_yards: 3.0,
            min_space_yards: 5.0,
            carry_target_yards: 4.2,
            carry_min_step_yards: 2.25,
            momentum_yps: 0.6,
        }
    }
}

/// Good-dribbling knobs grouped along the three axes the carrier controls: which DIRECTION the ball
/// is knocked (touch angle), how far it rolls per touch (touch LENGTH), and how fast the carrier
/// drives with it (carry SPEED / gait floor). The deterministic touch-decision path and the MPC
/// control estimate read these, and the learning store can override them through the policy tuning
/// overlay — so dribbling is tunable from MPC and POMDP/learned both. Defaults reproduce the prior
/// hard-coded literals exactly (an unconfigured process is byte-identical).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct DribbleTuning {
    // --- DIRECTION: touch-angle bucket scoring in `agentic_dribble_touch_bucket_for`. ---
    /// Weight on the MPC ball-control probability when scoring a touch direction (how much MPC
    /// execution feasibility steers which way the ball is knocked).
    pub direction_mpc_control_weight: f64,
    /// Weight on the MPC QP acceleration fit in the direction score.
    pub direction_mpc_accel_weight: f64,
    /// Weight on the kind-appropriate angle prior in the direction score.
    pub direction_angle_fit_weight: f64,
    /// Open-grass scale on the forward-component reward in the direction score (a clearer lane ahead
    /// pulls the touch more forward).
    pub direction_forward_open_grass_scale: f64,
    // --- LENGTH: touch distance in `dribble_touch_distance_for` (how far the ball rolls). ---
    /// Base touch length in yards before skill/space/intent terms.
    pub touch_length_base_yards: f64,
    /// Yards of touch length added per unit of ball-control skill.
    pub touch_length_control_scale: f64,
    /// Yards of touch length added per unit of open grass ahead.
    pub touch_length_open_grass_scale: f64,
    /// Yards of touch length added per unit of touch intent.
    pub touch_length_intent_scale: f64,
    // --- SPEED: carrier forward-drive gait floor (see `carrier_forward_drive_gait_floor`). ---
    /// An opponent at/inside this radius keeps the carrier on close control (no speed floor).
    pub carry_tight_pressure_yards: f64,
    /// Open forward space (yards) at/above which the carrier is floored to at least a jog.
    pub carry_jog_space_yards: f64,
    /// Open forward space (yards) at/above which the carrier is floored to a run.
    pub carry_run_space_yards: f64,
    /// Open forward space (yards) at/above which the carrier is floored to a sprint.
    pub carry_sprint_space_yards: f64,
    /// Above this fatigue the open-field carry floor caps at a run rather than a sprint.
    pub carry_sprint_max_fatigue: f64,
    /// Minimum forward component (yards) of the intended move for the carry floor to engage.
    pub carry_min_forward_yards: f64,
    /// Half-width (yards) of the carry lane used to measure open forward space.
    pub carry_lane_half_width_yards: f64,
}

impl Default for DribbleTuning {
    fn default() -> Self {
        DribbleTuning {
            direction_mpc_control_weight: 0.48,
            direction_mpc_accel_weight: 0.30,
            direction_angle_fit_weight: 1.18,
            direction_forward_open_grass_scale: 0.68,
            touch_length_base_yards: 0.78,
            touch_length_control_scale: 0.86,
            touch_length_open_grass_scale: 1.05,
            touch_length_intent_scale: 0.52,
            carry_tight_pressure_yards: 2.5,
            carry_jog_space_yards: 3.0,
            carry_run_space_yards: 9.0,
            carry_sprint_space_yards: 18.0,
            carry_sprint_max_fatigue: 0.80,
            carry_min_forward_yards: 1.0,
            carry_lane_half_width_yards: 4.0,
        }
    }
}

/// Fresh ball-winner pressure escape knobs. These govern the transition touch
/// after an outfield player wins possession in a crowd: accelerate away from
/// the nearest pressure into a landing point that gains cushion and stays clear.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct FreshPossessionEscapeTunables {
    pub enabled: bool,
    pub fresh_seconds: f64,
    pub min_pressure: f64,
    pub crowded_radius_yards: f64,
    pub crowded_min_opponents: usize,
    pub min_forward_space_yards: f64,
    pub target_yards: f64,
    pub corridor_half_width_yards: f64,
    pub min_cushion_gain_yards: f64,
    pub min_landing_clearance_yards: f64,
    pub initial_push_yps: f64,
    pub decision_floor_max_probability: f64,
}

impl Default for FreshPossessionEscapeTunables {
    fn default() -> Self {
        FreshPossessionEscapeTunables {
            enabled: true,
            fresh_seconds: 0.8,
            min_pressure: 0.55,
            crowded_radius_yards: 6.0,
            crowded_min_opponents: 2,
            min_forward_space_yards: 4.0,
            target_yards: 4.8,
            corridor_half_width_yards: 3.0,
            min_cushion_gain_yards: 0.85,
            min_landing_clearance_yards: 2.2,
            initial_push_yps: 1.45,
            decision_floor_max_probability: 0.56,
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

/// Central knobs for the killer-pass variant clipped over an opponent back four:
/// 25-35yd, ~12ft apex, and angled away from the goalkeeper. Defaults reproduce
/// the original feature constants; override with
/// `DD_SOCCER_TUNABLE__killer_pass_over_top.<field>` or `SOCCER_TUNABLES_JSON`.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct KillerPassOverTopTunables {
    pub min_distance_yards: f64,
    pub max_distance_yards: f64,
    pub target_distance_yards: f64,
    pub height_yards: f64,
    pub back_line_margin_yards: f64,
    pub lateral_offset_yards: f64,
    pub keeper_avoid_radius_yards: f64,
    pub lane_fit: f64,
    pub target_line_slack_yards: f64,
    pub side_basis_epsilon_yards: f64,
    pub touchline_margin_yards: f64,
    pub byline_margin_yards: f64,
    pub target_lateral_velocity_projection_seconds: f64,
    pub target_lateral_velocity_cap_yards: f64,
    pub central_gap_keeper_radius_factor: f64,
    pub secondary_lateral_offset_factor: f64,
    pub keeper_avoidance_min_factor: f64,
    pub fit_distance_weight: f64,
    pub fit_angle_weight: f64,
    pub fit_line_weight: f64,
    pub fit_keeper_weight: f64,
    pub fit_line_normalizer_yards: f64,
    pub fit_keeper_normalizer_yards: f64,
    pub score_bonus_weight: f64,
}

impl Default for KillerPassOverTopTunables {
    fn default() -> Self {
        KillerPassOverTopTunables {
            min_distance_yards: 25.0,
            max_distance_yards: 35.0,
            target_distance_yards: 30.0,
            height_yards: 4.0,
            back_line_margin_yards: 2.5,
            lateral_offset_yards: 4.5,
            keeper_avoid_radius_yards: 5.0,
            lane_fit: 0.72,
            target_line_slack_yards: 6.0,
            side_basis_epsilon_yards: 0.75,
            touchline_margin_yards: 3.0,
            byline_margin_yards: 2.0,
            target_lateral_velocity_projection_seconds: 0.35,
            target_lateral_velocity_cap_yards: 3.0,
            central_gap_keeper_radius_factor: 0.55,
            secondary_lateral_offset_factor: 0.45,
            keeper_avoidance_min_factor: 0.70,
            fit_distance_weight: 0.34,
            fit_angle_weight: 0.18,
            fit_line_weight: 0.26,
            fit_keeper_weight: 0.22,
            fit_line_normalizer_yards: 8.0,
            fit_keeper_normalizer_yards: 10.0,
            score_bonus_weight: 1.45,
        }
    }
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
    pub goalkeeper_home_lane_weight: f64,
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
    pub home_lane_match_span_lanes: f64,
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
    pub forward_home_lane_weight: f64,
    pub role_markov_weight: f64,
    pub role_static_fit_weight: f64,
    pub role_player_ball_weight: f64,
    pub role_target_ball_weight: f64,
    pub role_row_coherence_weight: f64,
    pub role_flow_weight: f64,
    pub role_field_config_weight: f64,
    pub role_home_lane_weight: f64,
    pub open_space_dynamic_lane_bonus_weight: f64,
    pub movement_shape_dynamic_lane_weight: f64,
}

impl Default for LaneAffinityTunables {
    fn default() -> Self {
        LaneAffinityTunables {
            goalkeeper_neutral_score: 0.5,
            goalkeeper_home_lane_weight: 0.35,
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
            home_lane_match_span_lanes: 3.0,
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
            forward_home_lane_weight: 0.16,
            role_markov_weight: 0.34,
            role_static_fit_weight: 0.30,
            role_player_ball_weight: 0.16,
            role_target_ball_weight: 0.08,
            role_row_coherence_weight: 0.06,
            role_flow_weight: 0.03,
            role_field_config_weight: 0.03,
            role_home_lane_weight: 0.20,
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
        self.carrier_keep_rolling.sanitize();
        self.fresh_possession_escape.sanitize();
        self.killer_pass_over_top.sanitize();
        self.lane_affinity.sanitize();
        self.pomdp_perception.sanitize();
        self.policy_selection.sanitize();
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
        self.carrier_keep_rolling
            .validate_strict("carrier_keep_rolling", &mut errors);
        self.fresh_possession_escape
            .validate_strict("fresh_possession_escape", &mut errors);
        self.killer_pass_over_top
            .validate_strict("killer_pass_over_top", &mut errors);
        self.lane_affinity
            .validate_strict("lane_affinity", &mut errors);
        self.pomdp_perception
            .validate_strict("pomdp_perception", &mut errors);
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors.join("; "))
        }
    }
}

impl PomdpPerceptionTunables {
    fn sanitize(&mut self) {
        let default = PomdpPerceptionTunables::default();
        sanitize_f64(
            "pomdp_perception.player_reaction_min_seconds",
            &mut self.player_reaction_min_seconds,
            default.player_reaction_min_seconds,
            0.0,
            1.0,
            0.05,
            0.25,
        );
        sanitize_f64(
            "pomdp_perception.player_reaction_max_seconds",
            &mut self.player_reaction_max_seconds,
            default.player_reaction_max_seconds,
            0.0,
            1.0,
            0.12,
            0.50,
        );
        sanitize_f64(
            "pomdp_perception.ball_holder_core_degrees",
            &mut self.ball_holder_core_degrees,
            default.ball_holder_core_degrees,
            1.0,
            240.0,
            60.0,
            160.0,
        );
        sanitize_f64(
            "pomdp_perception.ball_holder_shoulder_degrees",
            &mut self.ball_holder_shoulder_degrees,
            default.ball_holder_shoulder_degrees,
            0.0,
            140.0,
            20.0,
            80.0,
        );
        sanitize_f64(
            "pomdp_perception.ball_holder_core_confidence",
            &mut self.ball_holder_core_confidence,
            default.ball_holder_core_confidence,
            0.0,
            1.0,
            0.50,
            1.0,
        );
        sanitize_f64(
            "pomdp_perception.ball_holder_shoulder_confidence",
            &mut self.ball_holder_shoulder_confidence,
            default.ball_holder_shoulder_confidence,
            0.0,
            1.0,
            0.35,
            0.95,
        );
        sanitize_f64(
            "pomdp_perception.ball_holder_side_scan_confidence",
            &mut self.ball_holder_side_scan_confidence,
            default.ball_holder_side_scan_confidence,
            0.0,
            1.0,
            0.05,
            0.85,
        );
        sanitize_f64(
            "pomdp_perception.ball_holder_rear_scan_confidence",
            &mut self.ball_holder_rear_scan_confidence,
            default.ball_holder_rear_scan_confidence,
            0.0,
            1.0,
            0.05,
            0.85,
        );
        sanitize_f64(
            "pomdp_perception.ball_holder_head_scan_min_seconds",
            &mut self.ball_holder_head_scan_min_seconds,
            default.ball_holder_head_scan_min_seconds,
            0.0,
            3.0,
            0.25,
            1.5,
        );
        sanitize_f64(
            "pomdp_perception.ball_holder_head_scan_max_seconds",
            &mut self.ball_holder_head_scan_max_seconds,
            default.ball_holder_head_scan_max_seconds,
            0.0,
            4.0,
            1.0,
            2.5,
        );
        sanitize_f64(
            "pomdp_perception.ball_holder_head_scan_vision_relief_seconds",
            &mut self.ball_holder_head_scan_vision_relief_seconds,
            default.ball_holder_head_scan_vision_relief_seconds,
            0.0,
            2.0,
            0.0,
            0.8,
        );
        sanitize_f64(
            "pomdp_perception.ball_holder_scan_confidence_vision_bonus",
            &mut self.ball_holder_scan_confidence_vision_bonus,
            default.ball_holder_scan_confidence_vision_bonus,
            0.0,
            0.25,
            0.0,
            0.12,
        );
        sanitize_f64(
            "pomdp_perception.ball_holder_scan_confidence_cap",
            &mut self.ball_holder_scan_confidence_cap,
            default.ball_holder_scan_confidence_cap,
            0.0,
            1.0,
            0.45,
            0.80,
        );
        sanitize_f64(
            "pomdp_perception.ball_holder_head_scan_drift_risk_base",
            &mut self.ball_holder_head_scan_drift_risk_base,
            default.ball_holder_head_scan_drift_risk_base,
            0.0,
            1.0,
            0.0,
            0.5,
        );
        sanitize_f64(
            "pomdp_perception.ball_holder_head_scan_drift_risk_span",
            &mut self.ball_holder_head_scan_drift_risk_span,
            default.ball_holder_head_scan_drift_risk_span,
            0.0,
            1.0,
            0.10,
            0.75,
        );
        sanitize_f64(
            "pomdp_perception.kalman_point_match_radius_yards",
            &mut self.kalman_point_match_radius_yards,
            default.kalman_point_match_radius_yards,
            0.0,
            5.0,
            0.05,
            1.0,
        );
        sanitize_usize(
            "pomdp_perception.kalman_min_history_samples",
            &mut self.kalman_min_history_samples,
            default.kalman_min_history_samples,
            1,
            10,
            2,
            4,
        );
        sanitize_f64(
            "pomdp_perception.kalman_current_sample_epsilon_yards",
            &mut self.kalman_current_sample_epsilon_yards,
            default.kalman_current_sample_epsilon_yards,
            0.0,
            2.0,
            0.02,
            0.50,
        );
        sanitize_f64(
            "pomdp_perception.kalman_base_sigma_yards",
            &mut self.kalman_base_sigma_yards,
            default.kalman_base_sigma_yards,
            0.01,
            10.0,
            0.20,
            2.50,
        );
        sanitize_f64(
            "pomdp_perception.kalman_speed_sigma_per_yps",
            &mut self.kalman_speed_sigma_per_yps,
            default.kalman_speed_sigma_per_yps,
            0.0,
            2.0,
            0.0,
            0.50,
        );
        sanitize_f64(
            "pomdp_perception.kalman_velocity_disagreement_sigma_yards",
            &mut self.kalman_velocity_disagreement_sigma_yards,
            default.kalman_velocity_disagreement_sigma_yards,
            0.0,
            2.0,
            0.0,
            0.50,
        );
        sanitize_f64(
            "pomdp_perception.kalman_visible_max_confidence",
            &mut self.kalman_visible_max_confidence,
            default.kalman_visible_max_confidence,
            0.0,
            1.0,
            0.75,
            1.0,
        );
        sanitize_f64(
            "pomdp_perception.kalman_front_max_confidence",
            &mut self.kalman_front_max_confidence,
            default.kalman_front_max_confidence,
            0.0,
            1.0,
            0.60,
            0.98,
        );
        sanitize_f64(
            "pomdp_perception.kalman_occluded_max_confidence",
            &mut self.kalman_occluded_max_confidence,
            default.kalman_occluded_max_confidence,
            0.0,
            1.0,
            0.35,
            0.85,
        );
        sanitize_f64(
            "pomdp_perception.kalman_ball_holder_occluded_max_confidence",
            &mut self.kalman_ball_holder_occluded_max_confidence,
            default.kalman_ball_holder_occluded_max_confidence,
            0.0,
            1.0,
            0.25,
            0.80,
        );
        if self.player_reaction_max_seconds < self.player_reaction_min_seconds {
            eprintln!(
                "soccer tunables: pomdp_perception reaction min seconds exceeded max seconds; raising max"
            );
            self.player_reaction_max_seconds = self.player_reaction_min_seconds;
        }
        if self.ball_holder_head_scan_max_seconds < self.ball_holder_head_scan_min_seconds {
            eprintln!(
                "soccer tunables: pomdp_perception head-scan min seconds exceeded max seconds; raising max"
            );
            self.ball_holder_head_scan_max_seconds = self.ball_holder_head_scan_min_seconds;
        }
    }

    fn validate_strict(&self, prefix: &str, errors: &mut Vec<String>) {
        validate_f64(
            prefix,
            "player_reaction_min_seconds",
            self.player_reaction_min_seconds,
            0.0,
            1.0,
            errors,
        );
        validate_f64(
            prefix,
            "player_reaction_max_seconds",
            self.player_reaction_max_seconds,
            0.0,
            1.0,
            errors,
        );
        validate_f64(
            prefix,
            "ball_holder_core_degrees",
            self.ball_holder_core_degrees,
            1.0,
            240.0,
            errors,
        );
        validate_f64(
            prefix,
            "ball_holder_shoulder_degrees",
            self.ball_holder_shoulder_degrees,
            0.0,
            140.0,
            errors,
        );
        validate_f64(
            prefix,
            "ball_holder_core_confidence",
            self.ball_holder_core_confidence,
            0.0,
            1.0,
            errors,
        );
        validate_f64(
            prefix,
            "ball_holder_shoulder_confidence",
            self.ball_holder_shoulder_confidence,
            0.0,
            1.0,
            errors,
        );
        validate_f64(
            prefix,
            "ball_holder_side_scan_confidence",
            self.ball_holder_side_scan_confidence,
            0.0,
            1.0,
            errors,
        );
        validate_f64(
            prefix,
            "ball_holder_rear_scan_confidence",
            self.ball_holder_rear_scan_confidence,
            0.0,
            1.0,
            errors,
        );
        validate_f64(
            prefix,
            "ball_holder_head_scan_min_seconds",
            self.ball_holder_head_scan_min_seconds,
            0.0,
            3.0,
            errors,
        );
        validate_f64(
            prefix,
            "ball_holder_head_scan_max_seconds",
            self.ball_holder_head_scan_max_seconds,
            0.0,
            4.0,
            errors,
        );
        validate_f64(
            prefix,
            "ball_holder_head_scan_vision_relief_seconds",
            self.ball_holder_head_scan_vision_relief_seconds,
            0.0,
            2.0,
            errors,
        );
        validate_f64(
            prefix,
            "ball_holder_scan_confidence_vision_bonus",
            self.ball_holder_scan_confidence_vision_bonus,
            0.0,
            0.25,
            errors,
        );
        validate_f64(
            prefix,
            "ball_holder_scan_confidence_cap",
            self.ball_holder_scan_confidence_cap,
            0.0,
            1.0,
            errors,
        );
        validate_f64(
            prefix,
            "ball_holder_head_scan_drift_risk_base",
            self.ball_holder_head_scan_drift_risk_base,
            0.0,
            1.0,
            errors,
        );
        validate_f64(
            prefix,
            "ball_holder_head_scan_drift_risk_span",
            self.ball_holder_head_scan_drift_risk_span,
            0.0,
            1.0,
            errors,
        );
        validate_f64(
            prefix,
            "kalman_point_match_radius_yards",
            self.kalman_point_match_radius_yards,
            0.0,
            5.0,
            errors,
        );
        validate_usize(
            prefix,
            "kalman_min_history_samples",
            self.kalman_min_history_samples,
            1,
            10,
            errors,
        );
        validate_f64(
            prefix,
            "kalman_current_sample_epsilon_yards",
            self.kalman_current_sample_epsilon_yards,
            0.0,
            2.0,
            errors,
        );
        validate_f64(
            prefix,
            "kalman_base_sigma_yards",
            self.kalman_base_sigma_yards,
            0.01,
            10.0,
            errors,
        );
        validate_f64(
            prefix,
            "kalman_speed_sigma_per_yps",
            self.kalman_speed_sigma_per_yps,
            0.0,
            2.0,
            errors,
        );
        validate_f64(
            prefix,
            "kalman_velocity_disagreement_sigma_yards",
            self.kalman_velocity_disagreement_sigma_yards,
            0.0,
            2.0,
            errors,
        );
        validate_f64(
            prefix,
            "kalman_visible_max_confidence",
            self.kalman_visible_max_confidence,
            0.0,
            1.0,
            errors,
        );
        validate_f64(
            prefix,
            "kalman_front_max_confidence",
            self.kalman_front_max_confidence,
            0.0,
            1.0,
            errors,
        );
        validate_f64(
            prefix,
            "kalman_occluded_max_confidence",
            self.kalman_occluded_max_confidence,
            0.0,
            1.0,
            errors,
        );
        validate_f64(
            prefix,
            "kalman_ball_holder_occluded_max_confidence",
            self.kalman_ball_holder_occluded_max_confidence,
            0.0,
            1.0,
            errors,
        );
        if self.player_reaction_max_seconds < self.player_reaction_min_seconds {
            errors.push(format!(
                "{prefix}.player_reaction_max_seconds < {prefix}.player_reaction_min_seconds"
            ));
        }
        if self.ball_holder_head_scan_max_seconds < self.ball_holder_head_scan_min_seconds {
            errors.push(format!(
                "{prefix}.ball_holder_head_scan_max_seconds < {prefix}.ball_holder_head_scan_min_seconds"
            ));
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
        sanitize_f64(
            "lane_affinity.goalkeeper_home_lane_weight",
            &mut self.goalkeeper_home_lane_weight,
            default.goalkeeper_home_lane_weight,
            0.0,
            1.0,
            0.0,
            0.70,
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
            "lane_affinity.home_lane_match_span_lanes",
            &mut self.home_lane_match_span_lanes,
            default.home_lane_match_span_lanes,
            1.0,
            12.0,
            2.0,
            6.0,
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
            "lane_affinity.forward_home_lane_weight",
            &mut self.forward_home_lane_weight,
            default.forward_home_lane_weight,
            0.0,
            1.0,
            0.0,
            0.45,
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
            "lane_affinity.role_home_lane_weight",
            &mut self.role_home_lane_weight,
            default.role_home_lane_weight,
            0.0,
            1.0,
            0.0,
            0.55,
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
        validate_f64(
            prefix,
            "goalkeeper_home_lane_weight",
            self.goalkeeper_home_lane_weight,
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
            "home_lane_match_span_lanes",
            self.home_lane_match_span_lanes,
            1.0,
            12.0,
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
            "forward_home_lane_weight",
            self.forward_home_lane_weight,
            0.0,
            1.0,
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
            "role_home_lane_weight",
            self.role_home_lane_weight,
            0.0,
            1.0,
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
            "reward.giveaway_to_opponent_own_half_penalty",
            &mut self.giveaway_to_opponent_own_half_penalty,
            default.giveaway_to_opponent_own_half_penalty,
            0.0,
            200.0,
            0.0,
            25.0,
        );
        sanitize_f64(
            "reward.giveaway_to_opponent_opp_half_penalty",
            &mut self.giveaway_to_opponent_opp_half_penalty,
            default.giveaway_to_opponent_opp_half_penalty,
            0.0,
            200.0,
            0.0,
            25.0,
        );
        sanitize_f64(
            "reward.giveaway_to_loose_own_half_penalty",
            &mut self.giveaway_to_loose_own_half_penalty,
            default.giveaway_to_loose_own_half_penalty,
            0.0,
            200.0,
            0.0,
            25.0,
        );
        sanitize_f64(
            "reward.giveaway_to_loose_opp_half_penalty",
            &mut self.giveaway_to_loose_opp_half_penalty,
            default.giveaway_to_loose_opp_half_penalty,
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
        sanitize_f64(
            "reward.dense_shaping_budget_points",
            &mut self.dense_shaping_budget_points,
            default.dense_shaping_budget_points,
            0.0,
            1000.0,
            0.0,
            200.0,
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
            "giveaway_to_opponent_own_half_penalty",
            self.giveaway_to_opponent_own_half_penalty,
            0.0,
            200.0,
            errors,
        );
        validate_f64(
            prefix,
            "giveaway_to_opponent_opp_half_penalty",
            self.giveaway_to_opponent_opp_half_penalty,
            0.0,
            200.0,
            errors,
        );
        validate_f64(
            prefix,
            "giveaway_to_loose_own_half_penalty",
            self.giveaway_to_loose_own_half_penalty,
            0.0,
            200.0,
            errors,
        );
        validate_f64(
            prefix,
            "giveaway_to_loose_opp_half_penalty",
            self.giveaway_to_loose_opp_half_penalty,
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
        validate_f64(
            prefix,
            "dense_shaping_budget_points",
            self.dense_shaping_budget_points,
            0.0,
            1000.0,
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

impl CarrierKeepRollingTunables {
    fn sanitize(&mut self) {
        let default = CarrierKeepRollingTunables::default();
        sanitize_f64(
            "carrier_keep_rolling.stop_target_yards",
            &mut self.stop_target_yards,
            default.stop_target_yards,
            0.0,
            6.0,
            0.5,
            3.25,
        );
        sanitize_f64(
            "carrier_keep_rolling.min_opponent_distance_yards",
            &mut self.min_opponent_distance_yards,
            default.min_opponent_distance_yards,
            0.0,
            15.0,
            2.0,
            6.0,
        );
        sanitize_f64(
            "carrier_keep_rolling.min_space_yards",
            &mut self.min_space_yards,
            default.min_space_yards,
            0.0,
            40.0,
            3.0,
            12.0,
        );
        sanitize_f64(
            "carrier_keep_rolling.carry_target_yards",
            &mut self.carry_target_yards,
            default.carry_target_yards,
            0.5,
            25.0,
            2.0,
            8.0,
        );
        sanitize_f64(
            "carrier_keep_rolling.carry_min_step_yards",
            &mut self.carry_min_step_yards,
            default.carry_min_step_yards,
            0.0,
            12.0,
            1.0,
            5.0,
        );
        sanitize_f64(
            "carrier_keep_rolling.momentum_yps",
            &mut self.momentum_yps,
            default.momentum_yps,
            0.0,
            12.0,
            0.2,
            3.0,
        );
        if self.carry_min_step_yards > self.carry_target_yards {
            eprintln!(
                "soccer tunables: carrier_keep_rolling min step exceeded target; clamping min step"
            );
            self.carry_min_step_yards = self.carry_target_yards;
        }
        if self.min_space_yards < self.carry_min_step_yards {
            eprintln!(
                "soccer tunables: carrier_keep_rolling min space below min step; raising min space"
            );
            self.min_space_yards = self.carry_min_step_yards;
        }
    }

    fn validate_strict(&self, prefix: &str, errors: &mut Vec<String>) {
        validate_f64(prefix, "stop_target_yards", self.stop_target_yards, 0.0, 6.0, errors);
        validate_f64(
            prefix,
            "min_opponent_distance_yards",
            self.min_opponent_distance_yards,
            0.0,
            15.0,
            errors,
        );
        validate_f64(prefix, "min_space_yards", self.min_space_yards, 0.0, 40.0, errors);
        validate_f64(
            prefix,
            "carry_target_yards",
            self.carry_target_yards,
            0.5,
            25.0,
            errors,
        );
        validate_f64(
            prefix,
            "carry_min_step_yards",
            self.carry_min_step_yards,
            0.0,
            12.0,
            errors,
        );
        validate_f64(prefix, "momentum_yps", self.momentum_yps, 0.0, 12.0, errors);
        if self.carry_min_step_yards > self.carry_target_yards {
            errors.push(format!(
                "{prefix}.carry_min_step_yards > {prefix}.carry_target_yards"
            ));
        }
        if self.min_space_yards < self.carry_min_step_yards {
            errors.push(format!(
                "{prefix}.min_space_yards < {prefix}.carry_min_step_yards"
            ));
        }
    }
}

impl FreshPossessionEscapeTunables {
    fn sanitize(&mut self) {
        let default = FreshPossessionEscapeTunables::default();
        sanitize_f64(
            "fresh_possession_escape.fresh_seconds",
            &mut self.fresh_seconds,
            default.fresh_seconds,
            0.0,
            5.0,
            0.2,
            1.5,
        );
        sanitize_f64(
            "fresh_possession_escape.min_pressure",
            &mut self.min_pressure,
            default.min_pressure,
            0.0,
            1.0,
            0.25,
            0.80,
        );
        sanitize_f64(
            "fresh_possession_escape.crowded_radius_yards",
            &mut self.crowded_radius_yards,
            default.crowded_radius_yards,
            0.0,
            20.0,
            3.0,
            10.0,
        );
        sanitize_usize(
            "fresh_possession_escape.crowded_min_opponents",
            &mut self.crowded_min_opponents,
            default.crowded_min_opponents,
            0,
            8,
            1,
            4,
        );
        sanitize_f64(
            "fresh_possession_escape.min_forward_space_yards",
            &mut self.min_forward_space_yards,
            default.min_forward_space_yards,
            0.0,
            20.0,
            2.0,
            8.0,
        );
        sanitize_f64(
            "fresh_possession_escape.target_yards",
            &mut self.target_yards,
            default.target_yards,
            0.5,
            12.0,
            2.5,
            7.0,
        );
        sanitize_f64(
            "fresh_possession_escape.corridor_half_width_yards",
            &mut self.corridor_half_width_yards,
            default.corridor_half_width_yards,
            0.5,
            8.0,
            1.5,
            4.5,
        );
        sanitize_f64(
            "fresh_possession_escape.min_cushion_gain_yards",
            &mut self.min_cushion_gain_yards,
            default.min_cushion_gain_yards,
            0.0,
            4.0,
            0.4,
            1.8,
        );
        sanitize_f64(
            "fresh_possession_escape.min_landing_clearance_yards",
            &mut self.min_landing_clearance_yards,
            default.min_landing_clearance_yards,
            0.0,
            8.0,
            1.2,
            4.0,
        );
        sanitize_f64(
            "fresh_possession_escape.initial_push_yps",
            &mut self.initial_push_yps,
            default.initial_push_yps,
            0.0,
            8.0,
            0.5,
            3.5,
        );
        sanitize_f64(
            "fresh_possession_escape.decision_floor_max_probability",
            &mut self.decision_floor_max_probability,
            default.decision_floor_max_probability,
            0.0,
            1.0,
            0.20,
            0.75,
        );
        if self.target_yards < self.min_cushion_gain_yards {
            eprintln!(
                "soccer tunables: fresh_possession_escape target below cushion gain; raising target"
            );
            self.target_yards = self.min_cushion_gain_yards;
        }
    }

    fn validate_strict(&self, prefix: &str, errors: &mut Vec<String>) {
        validate_f64(prefix, "fresh_seconds", self.fresh_seconds, 0.0, 5.0, errors);
        validate_f64(prefix, "min_pressure", self.min_pressure, 0.0, 1.0, errors);
        validate_f64(
            prefix,
            "crowded_radius_yards",
            self.crowded_radius_yards,
            0.0,
            20.0,
            errors,
        );
        validate_usize(
            prefix,
            "crowded_min_opponents",
            self.crowded_min_opponents,
            0,
            8,
            errors,
        );
        validate_f64(
            prefix,
            "min_forward_space_yards",
            self.min_forward_space_yards,
            0.0,
            20.0,
            errors,
        );
        validate_f64(prefix, "target_yards", self.target_yards, 0.5, 12.0, errors);
        validate_f64(
            prefix,
            "corridor_half_width_yards",
            self.corridor_half_width_yards,
            0.5,
            8.0,
            errors,
        );
        validate_f64(
            prefix,
            "min_cushion_gain_yards",
            self.min_cushion_gain_yards,
            0.0,
            4.0,
            errors,
        );
        validate_f64(
            prefix,
            "min_landing_clearance_yards",
            self.min_landing_clearance_yards,
            0.0,
            8.0,
            errors,
        );
        validate_f64(
            prefix,
            "initial_push_yps",
            self.initial_push_yps,
            0.0,
            8.0,
            errors,
        );
        validate_f64(
            prefix,
            "decision_floor_max_probability",
            self.decision_floor_max_probability,
            0.0,
            1.0,
            errors,
        );
        if self.target_yards < self.min_cushion_gain_yards {
            errors.push(format!(
                "{prefix}.target_yards < {prefix}.min_cushion_gain_yards"
            ));
        }
    }
}

impl KillerPassOverTopTunables {
    fn sanitize(&mut self) {
        let default = KillerPassOverTopTunables::default();
        macro_rules! sanitize_field {
            ($field:ident, $hard_min:expr, $hard_max:expr, $sane_min:expr, $sane_max:expr) => {
                sanitize_f64(
                    concat!("killer_pass_over_top.", stringify!($field)),
                    &mut self.$field,
                    default.$field,
                    $hard_min,
                    $hard_max,
                    $sane_min,
                    $sane_max,
                );
            };
        }

        sanitize_field!(min_distance_yards, 5.0, 60.0, 18.0, 32.0);
        sanitize_field!(max_distance_yards, 10.0, 80.0, 30.0, 45.0);
        sanitize_field!(target_distance_yards, 5.0, 80.0, 25.0, 35.0);
        sanitize_field!(height_yards, 0.5, 12.0, 3.0, 6.0);
        sanitize_field!(back_line_margin_yards, 0.0, 10.0, 1.5, 5.0);
        sanitize_field!(lateral_offset_yards, 0.0, 15.0, 2.0, 7.0);
        sanitize_field!(keeper_avoid_radius_yards, 0.0, 20.0, 3.0, 8.0);
        sanitize_field!(lane_fit, 0.0, 1.0, 0.45, 0.90);
        sanitize_field!(target_line_slack_yards, 0.0, 20.0, 3.0, 9.0);
        sanitize_field!(side_basis_epsilon_yards, 0.05, 5.0, 0.25, 1.5);
        sanitize_field!(touchline_margin_yards, 0.5, 12.0, 2.0, 5.0);
        sanitize_field!(byline_margin_yards, 0.5, 12.0, 1.5, 5.0);
        sanitize_field!(
            target_lateral_velocity_projection_seconds,
            0.0,
            2.0,
            0.15,
            0.70
        );
        sanitize_field!(target_lateral_velocity_cap_yards, 0.0, 12.0, 1.0, 5.0);
        sanitize_field!(central_gap_keeper_radius_factor, 0.0, 2.0, 0.25, 0.85);
        sanitize_field!(secondary_lateral_offset_factor, 0.0, 1.0, 0.25, 0.65);
        sanitize_field!(keeper_avoidance_min_factor, 0.0, 1.0, 0.55, 0.85);
        sanitize_field!(fit_distance_weight, 0.0, 1.0, 0.05, 0.55);
        sanitize_field!(fit_angle_weight, 0.0, 1.0, 0.05, 0.40);
        sanitize_field!(fit_line_weight, 0.0, 1.0, 0.05, 0.50);
        sanitize_field!(fit_keeper_weight, 0.0, 1.0, 0.05, 0.45);
        sanitize_field!(fit_line_normalizer_yards, 0.5, 40.0, 5.0, 16.0);
        sanitize_field!(fit_keeper_normalizer_yards, 0.5, 40.0, 6.0, 18.0);
        sanitize_field!(score_bonus_weight, 0.0, 5.0, 0.5, 2.5);

        if self.min_distance_yards > self.max_distance_yards {
            eprintln!("soccer tunables: killer_pass_over_top min distance exceeded max; swapping");
            std::mem::swap(&mut self.min_distance_yards, &mut self.max_distance_yards);
        }
        if self.target_distance_yards < self.min_distance_yards
            || self.target_distance_yards > self.max_distance_yards
        {
            eprintln!(
                "soccer tunables: killer_pass_over_top target distance outside min/max; clamping"
            );
            self.target_distance_yards = self
                .target_distance_yards
                .clamp(self.min_distance_yards, self.max_distance_yards);
        }
        let fit_weight_sum = self.fit_distance_weight
            + self.fit_angle_weight
            + self.fit_line_weight
            + self.fit_keeper_weight;
        if fit_weight_sum <= 1e-9 {
            eprintln!("soccer tunables: killer_pass_over_top fit weights sum to zero; using defaults");
            self.fit_distance_weight = default.fit_distance_weight;
            self.fit_angle_weight = default.fit_angle_weight;
            self.fit_line_weight = default.fit_line_weight;
            self.fit_keeper_weight = default.fit_keeper_weight;
        }
    }

    fn validate_strict(&self, prefix: &str, errors: &mut Vec<String>) {
        macro_rules! validate_field {
            ($field:ident, $hard_min:expr, $hard_max:expr) => {
                validate_f64(
                    prefix,
                    stringify!($field),
                    self.$field,
                    $hard_min,
                    $hard_max,
                    errors,
                );
            };
        }

        validate_field!(min_distance_yards, 5.0, 60.0);
        validate_field!(max_distance_yards, 10.0, 80.0);
        validate_field!(target_distance_yards, 5.0, 80.0);
        validate_field!(height_yards, 0.5, 12.0);
        validate_field!(back_line_margin_yards, 0.0, 10.0);
        validate_field!(lateral_offset_yards, 0.0, 15.0);
        validate_field!(keeper_avoid_radius_yards, 0.0, 20.0);
        validate_field!(lane_fit, 0.0, 1.0);
        validate_field!(target_line_slack_yards, 0.0, 20.0);
        validate_field!(side_basis_epsilon_yards, 0.05, 5.0);
        validate_field!(touchline_margin_yards, 0.5, 12.0);
        validate_field!(byline_margin_yards, 0.5, 12.0);
        validate_field!(target_lateral_velocity_projection_seconds, 0.0, 2.0);
        validate_field!(target_lateral_velocity_cap_yards, 0.0, 12.0);
        validate_field!(central_gap_keeper_radius_factor, 0.0, 2.0);
        validate_field!(secondary_lateral_offset_factor, 0.0, 1.0);
        validate_field!(keeper_avoidance_min_factor, 0.0, 1.0);
        validate_field!(fit_distance_weight, 0.0, 1.0);
        validate_field!(fit_angle_weight, 0.0, 1.0);
        validate_field!(fit_line_weight, 0.0, 1.0);
        validate_field!(fit_keeper_weight, 0.0, 1.0);
        validate_field!(fit_line_normalizer_yards, 0.5, 40.0);
        validate_field!(fit_keeper_normalizer_yards, 0.5, 40.0);
        validate_field!(score_bonus_weight, 0.0, 5.0);

        if self.min_distance_yards > self.max_distance_yards {
            errors.push(format!(
                "{prefix}.min_distance_yards > {prefix}.max_distance_yards"
            ));
        }
        if self.target_distance_yards < self.min_distance_yards
            || self.target_distance_yards > self.max_distance_yards
        {
            errors.push(format!(
                "{prefix}.target_distance_yards outside {prefix}.min_distance_yards..={prefix}.max_distance_yards"
            ));
        }
        let fit_weight_sum = self.fit_distance_weight
            + self.fit_angle_weight
            + self.fit_line_weight
            + self.fit_keeper_weight;
        if fit_weight_sum <= 1e-9 {
            errors.push(format!("{prefix}.fit weights sum to zero"));
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
    fn reward_giveaway_penalty_defaults_match_historical_literals() {
        // The four flat turnover-penalty magnitudes were bare literals in
        // `dense_soccer_transition_reward` before extraction; defaults must
        // reproduce them exactly so the training baseline stays byte-identical.
        let r = RewardTunables::default();
        assert_eq!(r.giveaway_to_opponent_own_half_penalty, 3.5);
        assert_eq!(r.giveaway_to_opponent_opp_half_penalty, 2.2);
        assert_eq!(r.giveaway_to_loose_own_half_penalty, 0.85);
        assert_eq!(r.giveaway_to_loose_opp_half_penalty, 0.55);
        // The ordering the turnover model encodes: an own-half giveaway straight to
        // the opponent is the most expensive; a loose ball lost in the opponent's
        // half the least. Locking it in stops a future retune from inverting it.
        assert!(
            r.giveaway_to_opponent_own_half_penalty > r.giveaway_to_opponent_opp_half_penalty
        );
        assert!(
            r.giveaway_to_opponent_opp_half_penalty > r.giveaway_to_loose_own_half_penalty
        );
        assert!(r.giveaway_to_loose_own_half_penalty > r.giveaway_to_loose_opp_half_penalty);
    }

    #[test]
    fn reward_giveaway_penalties_are_sanitized_and_validated() {
        // Defaults pass through sanitize untouched and validate cleanly.
        let mut r = RewardTunables::default();
        r.sanitize();
        assert_eq!(r.giveaway_to_opponent_own_half_penalty, 3.5);
        assert_eq!(r.giveaway_to_loose_opp_half_penalty, 0.55);
        let mut errors = Vec::new();
        r.validate_strict("reward", &mut errors);
        assert!(errors.is_empty(), "default reward tunables must validate: {errors:?}");

        // A degenerate config can never inject a non-finite or wild penalty into a
        // gradient: non-finite is repaired to the default, out-of-range is clamped
        // to the hard bound.
        let mut bad = RewardTunables::default();
        bad.giveaway_to_opponent_own_half_penalty = f64::NAN;
        bad.giveaway_to_loose_opp_half_penalty = 10_000.0;
        bad.sanitize();
        assert_eq!(bad.giveaway_to_opponent_own_half_penalty, 3.5, "NaN -> default");
        assert_eq!(bad.giveaway_to_loose_opp_half_penalty, 200.0, "out-of-range -> hard cap");
    }

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
        assert_eq!(t.defensive_shape.back_four_horizontal_min_gap_yards, 6.0);
        assert_eq!(t.defensive_shape.back_four_horizontal_max_gap_yards, 15.0);
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
        assert!(t.carrier_keep_rolling.enabled);
        assert_eq!(t.carrier_keep_rolling.stop_target_yards, 2.75);
        assert_eq!(t.carrier_keep_rolling.min_opponent_distance_yards, 3.0);
        assert_eq!(t.carrier_keep_rolling.min_space_yards, 5.0);
        assert_eq!(t.carrier_keep_rolling.carry_target_yards, 4.2);
        assert_eq!(t.carrier_keep_rolling.carry_min_step_yards, 2.25);
        assert_eq!(t.carrier_keep_rolling.momentum_yps, 0.6);
        assert!(t.fresh_possession_escape.enabled);
        assert_eq!(t.fresh_possession_escape.fresh_seconds, 0.8);
        assert_eq!(t.fresh_possession_escape.min_pressure, 0.55);
        assert_eq!(t.fresh_possession_escape.crowded_radius_yards, 6.0);
        assert_eq!(t.fresh_possession_escape.crowded_min_opponents, 2);
        assert_eq!(t.fresh_possession_escape.min_forward_space_yards, 4.0);
        assert_eq!(t.fresh_possession_escape.target_yards, 4.8);
        assert_eq!(t.fresh_possession_escape.corridor_half_width_yards, 3.0);
        assert_eq!(t.fresh_possession_escape.min_cushion_gain_yards, 0.85);
        assert_eq!(t.fresh_possession_escape.min_landing_clearance_yards, 2.2);
        assert_eq!(t.fresh_possession_escape.initial_push_yps, 1.45);
        assert_eq!(
            t.fresh_possession_escape.decision_floor_max_probability,
            0.56
        );
        assert_eq!(t.lane_affinity.forward_lane_radius, 3);
        assert_eq!(t.lane_affinity.defender_commitment_possession, 0.80);
        assert_eq!(t.lane_affinity.possession_factor, 0.80);
        assert_eq!(t.lane_affinity.row_match_span_rows, 8.0);
        assert_eq!(t.lane_affinity.home_lane_match_span_lanes, 3.0);
        assert_eq!(t.lane_affinity.forward_home_lane_weight, 0.16);
        assert_eq!(t.lane_affinity.role_home_lane_weight, 0.20);
        assert_eq!(t.lane_affinity.open_space_dynamic_lane_bonus_weight, 1.05);
        assert_eq!(t.lane_affinity.movement_shape_dynamic_lane_weight, 0.36);
        assert_eq!(t.policy_selection.top1_weight, 0.70);
        assert_eq!(t.policy_selection.top2_weight, 0.20);
        assert_eq!(t.policy_selection.top3_weight, 0.10);
        assert_eq!(t.policy_selection.rank_weights(), [0.70, 0.20, 0.10]);
        assert_eq!(t.policy_selection.boltzmann_temperature, 0.0);
        assert_eq!(t.pomdp_perception.player_reaction_min_seconds, 0.10);
        assert_eq!(t.pomdp_perception.player_reaction_max_seconds, 0.25);
        assert_eq!(t.pomdp_perception.ball_holder_core_degrees, 100.0);
        assert_eq!(t.pomdp_perception.ball_holder_shoulder_degrees, 40.0);
        assert_eq!(t.pomdp_perception.ball_holder_core_confidence, 0.90);
        assert_eq!(t.pomdp_perception.ball_holder_shoulder_confidence, 0.70);
        assert_eq!(t.pomdp_perception.ball_holder_side_scan_confidence, 0.52);
        assert_eq!(t.pomdp_perception.ball_holder_rear_scan_confidence, 0.38);
        assert_eq!(
            t.pomdp_perception.ball_holder_head_scan_min_seconds,
            0.75
        );
        assert_eq!(
            t.pomdp_perception.ball_holder_head_scan_max_seconds,
            1.85
        );
        assert_eq!(
            t.pomdp_perception
                .ball_holder_head_scan_vision_relief_seconds,
            0.35
        );
        assert_eq!(
            t.pomdp_perception
                .ball_holder_scan_confidence_vision_bonus,
            0.08
        );
        assert_eq!(t.pomdp_perception.ball_holder_scan_confidence_cap, 0.62);
        assert_eq!(
            t.pomdp_perception
                .ball_holder_head_scan_drift_risk_base,
            0.24
        );
        assert_eq!(
            t.pomdp_perception
                .ball_holder_head_scan_drift_risk_span,
            0.42
        );
        assert_eq!(t.pomdp_perception.kalman_point_match_radius_yards, 0.35);
        assert_eq!(t.pomdp_perception.kalman_min_history_samples, 2);
        assert_eq!(
            t.pomdp_perception.kalman_current_sample_epsilon_yards,
            0.15
        );
        assert_eq!(t.pomdp_perception.kalman_base_sigma_yards, 0.85);
        assert_eq!(t.pomdp_perception.kalman_speed_sigma_per_yps, 0.10);
        assert_eq!(
            t.pomdp_perception
                .kalman_velocity_disagreement_sigma_yards,
            0.14
        );
        assert_eq!(t.pomdp_perception.kalman_visible_max_confidence, 0.96);
        assert_eq!(t.pomdp_perception.kalman_front_max_confidence, 0.88);
        assert_eq!(t.pomdp_perception.kalman_occluded_max_confidence, 0.68);
        assert_eq!(
            t.pomdp_perception
                .kalman_ball_holder_occluded_max_confidence,
            0.58
        );
    }

    #[test]
    fn partial_overlay_only_touches_named_fields() {
        let t = Tunables::from_overlays([json!({
            "tracking": { "moved_dt_multiplier": 1.5 },
            "decision_mpc": { "reselect_min_ball_execution_probability": 0.42 },
            "defensive_shape": { "defensive_line_max_into_opp_half_yards": 4.0 },
            "carrier_keep_rolling": {
                "enabled": false,
                "stop_target_yards": 1.4
            },
            "fresh_possession_escape": {
                "min_pressure": 0.48,
                "initial_push_yps": 1.8,
                "min_forward_space_yards": 3.5,
                "corridor_half_width_yards": 2.6
            },
            "lane_affinity": {
                "goalkeeper_home_lane_weight": 0.42,
                "possession_factor": 0.74,
                "role_home_lane_weight": 0.44,
                "open_space_dynamic_lane_bonus_weight": 1.3
            },
            "pomdp_perception": {
                "player_reaction_max_seconds": 0.30,
                "ball_holder_head_scan_max_seconds": 2.0,
                "ball_holder_side_scan_confidence": 0.55,
                "ball_holder_head_scan_drift_risk_span": 0.50,
                "kalman_min_history_samples": 3
            }
        })]);
        assert_eq!(t.tracking.moved_dt_multiplier, 1.5);
        assert_eq!(t.decision_mpc.reselect_min_ball_execution_probability, 0.42);
        assert_eq!(
            t.defensive_shape.defensive_line_max_into_opp_half_yards,
            4.0
        );
        assert!(!t.carrier_keep_rolling.enabled);
        assert_eq!(t.carrier_keep_rolling.stop_target_yards, 1.4);
        assert_eq!(t.fresh_possession_escape.min_pressure, 0.48);
        assert_eq!(t.fresh_possession_escape.initial_push_yps, 1.8);
        assert_eq!(t.fresh_possession_escape.min_forward_space_yards, 3.5);
        assert_eq!(t.fresh_possession_escape.corridor_half_width_yards, 2.6);
        assert_eq!(t.lane_affinity.goalkeeper_home_lane_weight, 0.42);
        assert_eq!(t.lane_affinity.possession_factor, 0.74);
        assert_eq!(t.lane_affinity.role_home_lane_weight, 0.44);
        assert_eq!(t.lane_affinity.open_space_dynamic_lane_bonus_weight, 1.3);
        assert_eq!(t.pomdp_perception.player_reaction_max_seconds, 0.30);
        assert_eq!(t.pomdp_perception.ball_holder_head_scan_max_seconds, 2.0);
        assert_eq!(t.pomdp_perception.ball_holder_side_scan_confidence, 0.55);
        assert_eq!(
            t.pomdp_perception
                .ball_holder_head_scan_drift_risk_span,
            0.50
        );
        assert_eq!(t.pomdp_perception.kalman_min_history_samples, 3);
        // Untouched fields keep their defaults.
        assert_eq!(t.tracking.tackle_recover_max_distance_yards, 3.8);
        assert_eq!(t.shooting.shot_block_bailout_max_probability, 0.86);
        assert_eq!(t.carrier_keep_rolling.carry_target_yards, 4.2);
        assert_eq!(t.fresh_possession_escape.target_yards, 4.8);
        assert_eq!(t.lane_affinity.home_lane_match_span_lanes, 3.0);
        assert_eq!(t.lane_affinity.forward_home_lane_weight, 0.16);
        assert_eq!(t.lane_affinity.movement_shape_dynamic_lane_weight, 0.36);
        assert_eq!(t.pomdp_perception.player_reaction_min_seconds, 0.10);
        assert_eq!(t.pomdp_perception.ball_holder_rear_scan_confidence, 0.38);
        assert_eq!(t.pomdp_perception.kalman_base_sigma_yards, 0.85);
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
            "carrier_keep_rolling": {
                "stop_target_yards": 99.0,
                "carry_target_yards": 2.0,
                "carry_min_step_yards": 4.0,
                "min_space_yards": 1.0
            },
            "fresh_possession_escape": {
                "fresh_seconds": 99.0,
                "min_forward_space_yards": 99.0,
                "target_yards": 0.5,
                "corridor_half_width_yards": 99.0,
                "min_cushion_gain_yards": 2.0,
                "initial_push_yps": 99.0,
                "crowded_min_opponents": 99
            },
            "lane_affinity": {
                "goalkeeper_home_lane_weight": 9.0,
                "possession_factor": 4.0,
                "forward_lane_radius": 99,
                "home_lane_match_span_lanes": 99.0,
                "role_home_lane_weight": -1.0,
                "lookahead_min_seconds": 2.0,
                "lookahead_max_seconds": 0.5
            },
            "policy_selection": {
                "top1_weight": -0.5,
                "boltzmann_temperature": 5000.0
            },
            "pomdp_perception": {
                "player_reaction_min_seconds": 2.0,
                "player_reaction_max_seconds": -1.0,
                "ball_holder_core_degrees": -4.0,
                "ball_holder_side_scan_confidence": 2.0,
                "ball_holder_head_scan_min_seconds": 99.0,
                "ball_holder_head_scan_max_seconds": -1.0,
                "ball_holder_scan_confidence_vision_bonus": 9.0,
                "ball_holder_scan_confidence_cap": -1.0,
                "ball_holder_head_scan_drift_risk_base": 5.0,
                "ball_holder_head_scan_drift_risk_span": -5.0,
                "kalman_point_match_radius_yards": 99.0,
                "kalman_min_history_samples": 99,
                "kalman_current_sample_epsilon_yards": -2.0,
                "kalman_base_sigma_yards": 0.0,
                "kalman_visible_max_confidence": 3.0
            }
        })]);
        assert_eq!(t.decision_mpc.reselect_min_execution_confidence, 0.0);
        assert_eq!(t.shooting.shot_block_bailout_max_probability, 1.0);
        assert_eq!(t.defensive_shape.back_four_horizontal_min_gap_yards, 4.0);
        assert_eq!(t.defensive_shape.back_four_horizontal_max_gap_yards, 12.0);
        assert_eq!(t.carrier_keep_rolling.stop_target_yards, 6.0);
        assert_eq!(t.carrier_keep_rolling.carry_target_yards, 2.0);
        assert_eq!(t.carrier_keep_rolling.carry_min_step_yards, 2.0);
        assert_eq!(t.carrier_keep_rolling.min_space_yards, 2.0);
        assert_eq!(t.fresh_possession_escape.fresh_seconds, 5.0);
        assert_eq!(t.fresh_possession_escape.min_forward_space_yards, 20.0);
        assert_eq!(t.fresh_possession_escape.target_yards, 2.0);
        assert_eq!(t.fresh_possession_escape.corridor_half_width_yards, 8.0);
        assert_eq!(t.fresh_possession_escape.min_cushion_gain_yards, 2.0);
        assert_eq!(t.fresh_possession_escape.initial_push_yps, 8.0);
        assert_eq!(t.fresh_possession_escape.crowded_min_opponents, 8);
        assert_eq!(t.lane_affinity.goalkeeper_home_lane_weight, 1.0);
        assert_eq!(t.lane_affinity.possession_factor, 1.0);
        assert_eq!(t.lane_affinity.forward_lane_radius, 6);
        assert_eq!(t.lane_affinity.home_lane_match_span_lanes, 12.0);
        assert_eq!(t.lane_affinity.role_home_lane_weight, 0.0);
        assert_eq!(t.lane_affinity.lookahead_min_seconds, 0.5);
        assert_eq!(t.lane_affinity.lookahead_max_seconds, 2.0);
        // Negative rank weight clamps up to the 0.0 hard floor; an absurd
        // temperature clamps down to the 1000.0 hard ceiling.
        assert_eq!(t.policy_selection.top1_weight, 0.0);
        assert_eq!(t.policy_selection.boltzmann_temperature, 1_000.0);
        assert_eq!(t.pomdp_perception.player_reaction_min_seconds, 1.0);
        assert_eq!(t.pomdp_perception.player_reaction_max_seconds, 1.0);
        assert_eq!(t.pomdp_perception.ball_holder_core_degrees, 1.0);
        assert_eq!(t.pomdp_perception.ball_holder_side_scan_confidence, 1.0);
        assert_eq!(
            t.pomdp_perception.ball_holder_head_scan_min_seconds,
            3.0
        );
        assert_eq!(
            t.pomdp_perception.ball_holder_head_scan_max_seconds,
            3.0
        );
        assert_eq!(
            t.pomdp_perception
                .ball_holder_scan_confidence_vision_bonus,
            0.25
        );
        assert_eq!(t.pomdp_perception.ball_holder_scan_confidence_cap, 0.0);
        assert_eq!(
            t.pomdp_perception
                .ball_holder_head_scan_drift_risk_base,
            1.0
        );
        assert_eq!(
            t.pomdp_perception
                .ball_holder_head_scan_drift_risk_span,
            0.0
        );
        assert_eq!(t.pomdp_perception.kalman_point_match_radius_yards, 5.0);
        assert_eq!(t.pomdp_perception.kalman_min_history_samples, 10);
        assert_eq!(
            t.pomdp_perception.kalman_current_sample_epsilon_yards,
            0.0
        );
        assert_eq!(t.pomdp_perception.kalman_base_sigma_yards, 0.01);
        assert_eq!(t.pomdp_perception.kalman_visible_max_confidence, 1.0);
    }

    #[test]
    fn strict_validation_reports_bad_direct_configs() {
        let mut t = Tunables::default();
        t.shooting.shot_on_frame_min_probability = 1.2;
        t.defensive_shape.back_four_horizontal_min_gap_yards = 9.0;
        t.defensive_shape.back_four_horizontal_max_gap_yards = 6.0;
        t.carrier_keep_rolling.carry_min_step_yards = 6.0;
        t.carrier_keep_rolling.carry_target_yards = 4.0;
        t.fresh_possession_escape.target_yards = 0.6;
        t.fresh_possession_escape.min_cushion_gain_yards = 1.2;
        t.lane_affinity.possession_factor = 1.2;
        t.lane_affinity.goalkeeper_home_lane_weight = -0.1;
        t.lane_affinity.forward_lane_radius = 9;
        t.lane_affinity.home_lane_match_span_lanes = 0.5;
        t.lane_affinity.forward_home_lane_weight = 6.0;
        t.pomdp_perception.player_reaction_min_seconds = 0.8;
        t.pomdp_perception.player_reaction_max_seconds = 0.2;
        t.pomdp_perception.ball_holder_head_scan_min_seconds = 2.5;
        t.pomdp_perception.ball_holder_head_scan_max_seconds = 1.5;
        t.pomdp_perception.ball_holder_side_scan_confidence = 1.2;
        t.pomdp_perception.ball_holder_head_scan_drift_risk_base = -0.1;
        t.pomdp_perception.kalman_min_history_samples = 0;

        let err = t.validate_strict().expect_err("config should be invalid");
        assert!(err.contains("shooting.shot_on_frame_min_probability"));
        assert!(err.contains("defensive_shape.back_four_horizontal_min_gap_yards >"));
        assert!(err.contains("carrier_keep_rolling.carry_min_step_yards >"));
        assert!(err.contains("fresh_possession_escape.target_yards <"));
        assert!(err.contains("lane_affinity.possession_factor"));
        assert!(err.contains("lane_affinity.goalkeeper_home_lane_weight"));
        assert!(err.contains("lane_affinity.forward_lane_radius"));
        assert!(err.contains("lane_affinity.home_lane_match_span_lanes"));
        assert!(err.contains("lane_affinity.forward_home_lane_weight"));
        assert!(err.contains(
            "pomdp_perception.player_reaction_max_seconds < pomdp_perception.player_reaction_min_seconds"
        ));
        assert!(err.contains("pomdp_perception.ball_holder_side_scan_confidence"));
        assert!(err.contains(
            "pomdp_perception.ball_holder_head_scan_drift_risk_base"
        ));
        assert!(err.contains("pomdp_perception.kalman_min_history_samples"));
        assert!(err.contains(
            "pomdp_perception.ball_holder_head_scan_max_seconds < pomdp_perception.ball_holder_head_scan_min_seconds"
        ));
    }
}
