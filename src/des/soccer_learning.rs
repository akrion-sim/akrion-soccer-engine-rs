//! Durable soccer self-play learning primitives.
//!
//! The soccer simulator owns the MDP/POMDP update mechanics. This module owns
//! the cross-run layer: outcome scoring, policy deltas, weighted merge, simple
//! evolutionary spawning, and a queue runner that keeps worker slots full.

use std::collections::{hash_map::DefaultHasher, BTreeMap, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::{mpsc, Arc, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::des::general::prng::SeededRandom;
use crate::des::general::soccer::{
    MatchConfig, MatchSummary, SoccerConfigMomentInsert, SoccerMatch, SoccerNeuralBlendConfig,
    SoccerNeuralLearningConfig, SoccerNeuralNetworkSnapshot, SoccerPassOutcomeSample, SoccerQEntry,
    SoccerQPolicy, SoccerQPolicyOptions, SoccerQStateKey, SoccerQTargetEntry,
    SoccerSelfPlayEpisodeSummary, SoccerSelfPlayTrainingArtifact, SoccerTacticalLearningSummary,
    SoccerTacticalLearningWeights, SoccerTeamQPolicies, Team, DEFAULT_FIELD_LENGTH_YARDS,
    DEFAULT_FIELD_WIDTH_YARDS, MAX_SOCCER_NEURAL_LEARNING_RATE,
};
use crate::des::shared::capabilities::RandomSource;

pub const SOCCER_LEARNING_FIXED_SCALE: i64 = 1_000_000;
pub const SOCCER_POLICY_STATUS_ACTIVE: &str = "active";
pub const SOCCER_POLICY_STATUS_ARCHIVED: &str = "archived";
pub const SOCCER_EVOLUTION_MAX_POPULATION_SIZE: usize = 4096;
const SOCCER_MAPPO_MIN_CLIP_EPSILON: f64 = 0.01;
const SOCCER_NEURAL_MCTS_MAX_SIMULATIONS: usize = 32;
const SOCCER_NEURAL_MCTS_MAX_CANDIDATES: usize = 8;
const SOCCER_NEURAL_MCTS_MAX_DEPTH: usize = 3;
const SOCCER_NEURAL_MCTS_MAX_EXPLORATION: f64 = 4.0;
const SOCCER_MATCH_FITNESS_WINNER_WEIGHT: f64 = 0.72;
const SOCCER_MATCH_FITNESS_QUALITY_WEIGHT: f64 = 0.28;
const SOCCER_MATCH_FITNESS_DECISIVE_MARGIN_WEIGHT: f64 = 0.06;
const SOCCER_MATCH_FITNESS_MAX_DECISIVE_MARGIN: f64 = 6.0;
const SOCCER_MATCH_FITNESS_MIN: f64 = -8.0;
const SOCCER_MATCH_FITNESS_MAX: f64 = 12.0;
const SOCCER_POLICY_PLATEAU_FITNESS_SPREAD: f64 = 0.075;
const SOCCER_POLICY_LOW_CEILING_FITNESS: f64 = 1.50;
const SOCCER_POLICY_SEARCH_MAX_ADAPTED_POPULATION: usize = 128;
const SOCCER_POLICY_SEARCH_MAX_ADAPTED_POPULATION_ENV: &str =
    "SOCCER_POLICY_SEARCH_MAX_ADAPTED_POPULATION";
static SOCCER_POLICY_SEARCH_MAX_ADAPTED_POPULATION_OVERRIDE: OnceLock<usize> = OnceLock::new();
const SOCCER_POLICY_NOVELTY_MAX_STATES: usize = 256;
const SOCCER_POLICY_NOVELTY_MAX_ACTIONS_PER_STATE: usize = 3;
const SOCCER_POLICY_NOVELTY_VISIT_WEIGHT_MAX: f64 = 3.0;
const SOCCER_POLICY_STATE_ACTION_COVERAGE_WEIGHT: f64 = 0.14;
const SOCCER_POLICY_TECHNICAL_COVERAGE_WEIGHT: f64 = 0.05;
const SOCCER_POLICY_TARGET_COVERAGE_WEIGHT: f64 = 0.07;
const SOCCER_POLICY_DP_RESIDUAL_PENALTY_WEIGHT: f64 = 0.018;
const SOCCER_CURRICULUM_FULL_PLAYERS_PER_TEAM: usize = 11;
const SOCCER_POLICY_DRIBBLE_NOVELTY_ACTIONS: &[&str] = &[
    "dribble",
    "carry-forward",
    "carry-out-left",
    "carry-out-right",
    "protect-ball",
    "side-step",
    "left-cut",
    "right-cut",
    "nutmeg",
    "fake-left-cut-right",
    "fake-right-cut-left",
];
const SOCCER_POLICY_PASS_NOVELTY_ACTIONS: &[&str] = &[
    "pass",
    "aerial-pass",
    "killer-pass",
    "switch-play",
    "recycle-reset",
    "wall-pass",
    "surprise-pass",
    "flick-on",
    "scoop-pass",
    "first-time-pass",
];
const SOCCER_POLICY_SHOT_NOVELTY_ACTIONS: &[&str] = &["shoot", "first-time-shot"];
const SOCCER_POLICY_DEFENSE_NOVELTY_ACTIONS: &[&str] =
    &["tackle", "press-cover", "clearance", "route-one"];
const SOCCER_POLICY_SUPPORT_NOVELTY_ACTIONS: &[&str] = &[
    "space",
    "control-touch",
    "vertical-attack",
    "turnover-burst",
    "vacate-space",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SoccerLearningPolicyEntryKind {
    Action,
    Target,
}

impl SoccerLearningPolicyEntryKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Action => "action",
            Self::Target => "target",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SoccerLearningOutcome {
    Win,
    Draw,
    Loss,
}

impl SoccerLearningOutcome {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Win => "win",
            Self::Draw => "draw",
            Self::Loss => "loss",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerLearningTeamScore {
    pub team: Team,
    pub goals_for: u32,
    pub goals_against: u32,
    pub goal_diff: i32,
    pub outcome: SoccerLearningOutcome,
    pub merge_weight: f64,
    pub merge_weight_micros: i64,
    pub fitness: f64,
    pub fitness_micros: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerLearningRunScore {
    pub home: SoccerLearningTeamScore,
    pub away: SoccerLearningTeamScore,
    pub match_fitness: f64,
    pub match_fitness_micros: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerLearningPolicyDeltaEntry {
    pub team: Team,
    pub entry_kind: SoccerLearningPolicyEntryKind,
    pub state_hash: String,
    pub state_key: SoccerQStateKey,
    pub state_json: Value,
    pub action: String,
    pub target_fine_cell_id: i32,
    pub target_tactical_cell_id: i32,
    pub target_macro_cell_id: i32,
    pub target_root_cell_id: i32,
    pub before_value: f64,
    pub after_value: f64,
    pub value_delta: f64,
    pub before_value_micros: i64,
    pub after_value_micros: i64,
    pub value_delta_micros: i64,
    pub visit_delta: u32,
    pub merge_weight: f64,
    pub merge_weight_micros: i64,
    pub effective_visit_weight: f64,
    pub effective_visit_micros: i64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerLearningPolicyDelta {
    pub entries: Vec<SoccerLearningPolicyDeltaEntry>,
}

#[derive(Clone, Debug)]
pub struct SoccerLearningCompletedGame {
    pub episode: usize,
    pub seed: u64,
    pub summary: MatchSummary,
    pub episode_summary: SoccerSelfPlayEpisodeSummary,
    pub tactical_summary: SoccerTacticalLearningSummary,
    pub starting_tactical_learning: SoccerTacticalLearningWeights,
    pub policies: SoccerTeamQPolicies,
    pub score: SoccerLearningRunScore,
    pub delta: SoccerLearningPolicyDelta,
    pub config_moments: Vec<SoccerConfigMomentInsert>,
    /// Learned pass-completion training samples captured this game (config embedding + pass
    /// features, labelled completed/intercepted). Persisted to Postgres alongside the config
    /// moments so the cluster learner can train [`SoccerPassCompletionHead`] on the pooled corpus.
    pub pass_outcome_samples: Vec<SoccerPassOutcomeSample>,
    pub neural_network: Option<SoccerNeuralNetworkSnapshot>,
    pub elapsed_seconds: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerLearningQueueRunnerConfig {
    pub games: usize,
    pub parallel_games: usize,
    pub base_seed: u32,
    pub match_config: MatchConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial_neural_network: Option<SoccerNeuralNetworkSnapshot>,
    pub neural_drain_timeout: Duration,
    pub options: SoccerQPolicyOptions,
    pub prune_action_entries_per_team: usize,
    pub prune_target_entries_per_team: usize,
    pub min_policy_visits: u32,
}

#[derive(Clone, Debug)]
pub struct SoccerLearningQueueReport {
    pub completed_games: usize,
    pub failed_games: usize,
    pub elapsed_seconds: f64,
    pub total_home_goals: u32,
    pub total_away_goals: u32,
    pub tactical_summary: SoccerTacticalLearningSummary,
    pub final_policy_entries: usize,
    pub final_target_entries: usize,
    pub episode_summaries: Vec<SoccerSelfPlayEpisodeSummary>,
    pub final_policies: SoccerTeamQPolicies,
    pub latest_neural_network: Option<SoccerNeuralNetworkSnapshot>,
}

pub enum SoccerLearningQueueEvent<'a> {
    StartingBatch {
        next_episode: usize,
        match_config: &'a mut MatchConfig,
        policies: &'a mut SoccerTeamQPolicies,
        neural_network: &'a mut Option<SoccerNeuralNetworkSnapshot>,
    },
    CompletedGame {
        game: &'a SoccerLearningCompletedGame,
        merged_policies: &'a mut SoccerTeamQPolicies,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerEvolutionOptions {
    pub mutation_rate: f64,
    pub mutation_scale: f64,
    pub crossover_rate: f64,
    pub exploration_rate: f64,
    pub exploration_scale: f64,
    pub elite_weight_floor: f64,
    pub population_size: usize,
    pub seed: u64,
}

impl Default for SoccerEvolutionOptions {
    fn default() -> Self {
        Self {
            mutation_rate: 0.04,
            mutation_scale: 0.22,
            crossover_rate: 0.45,
            exploration_rate: 0.06,
            exploration_scale: 0.55,
            elite_weight_floor: 0.05,
            population_size: 8,
            seed: 2026,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SoccerLearningCurriculumStage {
    Locomotion,
    BallSkills,
    Duels,
    SmallSided,
    TeamShape,
    FullMatch,
}

impl SoccerLearningCurriculumStage {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Locomotion => "locomotion",
            Self::BallSkills => "ball_skills",
            Self::Duels => "duels",
            Self::SmallSided => "small_sided",
            Self::TeamShape => "team_shape",
            Self::FullMatch => "full_match",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerLearningCurriculumConfig {
    pub ball_skills_after_games: usize,
    pub duels_after_games: usize,
    pub small_sided_after_games: usize,
    pub team_shape_after_games: usize,
    pub full_match_after_games: usize,
}

impl Default for SoccerLearningCurriculumConfig {
    fn default() -> Self {
        Self {
            ball_skills_after_games: 8,
            duels_after_games: 24,
            small_sided_after_games: 64,
            team_shape_after_games: 128,
            full_match_after_games: 256,
        }
    }
}

pub fn validate_soccer_learning_curriculum_config_for_learning_run(
    config: &SoccerLearningCurriculumConfig,
) -> Result<(), String> {
    let thresholds = [
        config.ball_skills_after_games,
        config.duels_after_games,
        config.small_sided_after_games,
        config.team_shape_after_games,
        config.full_match_after_games,
    ];
    if thresholds.windows(2).any(|window| window[0] > window[1]) {
        return Err("curriculum thresholds must be monotonic".to_string());
    }
    Ok(())
}

pub fn soccer_learning_curriculum_stage_for_completed_games(
    completed_games: usize,
    config: &SoccerLearningCurriculumConfig,
) -> SoccerLearningCurriculumStage {
    if completed_games >= config.full_match_after_games {
        SoccerLearningCurriculumStage::FullMatch
    } else if completed_games >= config.team_shape_after_games {
        SoccerLearningCurriculumStage::TeamShape
    } else if completed_games >= config.small_sided_after_games {
        SoccerLearningCurriculumStage::SmallSided
    } else if completed_games >= config.duels_after_games {
        SoccerLearningCurriculumStage::Duels
    } else if completed_games >= config.ball_skills_after_games {
        SoccerLearningCurriculumStage::BallSkills
    } else {
        SoccerLearningCurriculumStage::Locomotion
    }
}

/// Per-stage shaping of a single training game for the progressive "2v2 → 11v11"
/// curriculum.
///
/// The engine roster is a hard 11v11 invariant (`SOCCER_MATCH_TEAM_PLAYER_COUNT`, with
/// `home != 11 / away != 11` asserted across the frame + accounting checks), so the
/// curriculum deliberately does **not** change the player *count*. Instead it scales the
/// parts of a match that ARE configurable so early games behave like isolated skills /
/// small-sided reps and later games are the full contest:
///   - **pitch size** — a tight box → the full pitch (`pitch_fraction`),
///   - **match length** — short reps → the full duration (`duration_fraction`),
///   - **MAPPO team-reward share** — individual skill learning → the shared team return
///     (`team_reward_share_fraction`, scaling the configured `mappo_team_reward_share`),
///   - **team-shape machinery** — formation LP only engages once shape matters
///     (`formation_lp`).
///
/// Every numeric field is a fraction in `[0, 1]` of the configured full-match value, and
/// `formation_lp` is only ever used to turn the formation LP *off* for the individual
/// stages (never to force it on). `FullMatch` is therefore the identity — all fractions
/// `1.0`, formation untouched — so the final stage trains on exactly the configured game
/// and a run pinned at `FullMatch` (or with the curriculum disabled) is byte-identical.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoccerCurriculumStageShaping {
    pub pitch_fraction: f64,
    pub duration_fraction: f64,
    pub team_reward_share_fraction: f64,
    pub formation_lp: bool,
}

/// The fixed stage ramp. Tuned so the pitch and duration grow smoothly while the actor's
/// objective shifts from purely individual (ball skills, 1v1 duels) to the shared team
/// return, and formation shape only switches on at the small-sided stage and beyond.
pub fn soccer_curriculum_stage_shaping(
    stage: SoccerLearningCurriculumStage,
) -> SoccerCurriculumStageShaping {
    let (pitch, duration, share, formation) = match stage {
        SoccerLearningCurriculumStage::Locomotion => (0.25, 0.25, 0.0, false),
        SoccerLearningCurriculumStage::BallSkills => (0.30, 0.30, 0.0, false),
        SoccerLearningCurriculumStage::Duels => (0.40, 0.40, 0.10, false),
        SoccerLearningCurriculumStage::SmallSided => (0.60, 0.60, 0.35, true),
        SoccerLearningCurriculumStage::TeamShape => (0.85, 0.85, 0.60, true),
        SoccerLearningCurriculumStage::FullMatch => (1.0, 1.0, 1.0, true),
    };
    SoccerCurriculumStageShaping {
        pitch_fraction: pitch,
        duration_fraction: duration,
        team_reward_share_fraction: share,
        formation_lp: formation,
    }
}

/// Smallest playable pitch dimensions the curriculum will shrink a stage to (a tight
/// training box). The fractions never *increase* a configured value, so these floors only
/// bite for the early stages and never alter a full-size pitch.
const SOCCER_CURRICULUM_MIN_PITCH_LENGTH_YARDS: f64 = 40.0;
const SOCCER_CURRICULUM_MIN_PITCH_WIDTH_YARDS: f64 = 30.0;
const SOCCER_CURRICULUM_MIN_DURATION_SECONDS: f64 = 20.0;

/// Reshape `config` in place for `stage`, treating the config's current pitch / duration /
/// `mappo_team_reward_share` as the full-match (`stage == FullMatch`) targets. `FullMatch`
/// leaves the config byte-identical. Floors keep early stages playable but never raise a
/// value above what was configured.
pub fn apply_soccer_curriculum_stage_to_match_config(
    stage: SoccerLearningCurriculumStage,
    config: &mut MatchConfig,
) {
    let shaping = soccer_curriculum_stage_shaping(stage);

    // Pitch — scale both dimensions together (aspect ratio preserved), with a minimum box.
    if config.field_length_yards.is_finite() && config.field_length_yards > 0.0 {
        let floor = SOCCER_CURRICULUM_MIN_PITCH_LENGTH_YARDS.min(config.field_length_yards);
        config.field_length_yards = (config.field_length_yards * shaping.pitch_fraction).max(floor);
    }
    if config.field_width_yards.is_finite() && config.field_width_yards > 0.0 {
        let floor = SOCCER_CURRICULUM_MIN_PITCH_WIDTH_YARDS.min(config.field_width_yards);
        config.field_width_yards = (config.field_width_yards * shaping.pitch_fraction).max(floor);
    }

    // Match length — scale both the single-duration and per-half paths so the shorter game
    // takes effect regardless of which one `effective_duration_seconds` reads.
    let scale_duration = |seconds: f64| -> f64 {
        if seconds.is_finite() && seconds > 0.0 {
            let floor = SOCCER_CURRICULUM_MIN_DURATION_SECONDS.min(seconds);
            (seconds * shaping.duration_fraction).max(floor)
        } else {
            seconds
        }
    };
    config.duration_seconds = scale_duration(config.duration_seconds);
    config.half_duration_seconds = scale_duration(config.half_duration_seconds);

    // Objective — ramp the shared team return from individual (early) to fully shared.
    if config.neural_learning.mappo_team_reward_share.is_finite() {
        config.neural_learning.mappo_team_reward_share *= shaping.team_reward_share_fraction;
    }

    // Team shape — disengage the formation LP for the individual stages (never force it on).
    config.formation_lp_enabled &= shaping.formation_lp;
}

fn soccer_curriculum_env_flag(name: &str) -> Option<bool> {
    std::env::var(name).ok().map(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn soccer_curriculum_env_usize_alias(primary: &str, alias: &str, fallback: usize) -> usize {
    for name in [primary, alias] {
        if let Ok(value) = std::env::var(name) {
            if let Ok(parsed) = value.trim().parse::<usize>() {
                return parsed;
            }
        }
    }
    fallback
}

/// Whether the per-game curriculum is engaged. Off unless `SOCCER_CURRICULUM_ENABLED`
/// (or its `SOCCER_BATCH_*` alias) is truthy, so an unset environment is byte-identical to
/// the pre-curriculum learner.
pub fn soccer_curriculum_enabled_from_env() -> bool {
    soccer_curriculum_env_flag("SOCCER_CURRICULUM_ENABLED")
        .or_else(|| soccer_curriculum_env_flag("SOCCER_BATCH_CURRICULUM_ENABLED"))
        .unwrap_or(false)
}

/// The curriculum stage thresholds from the environment, reusing the same env names the
/// learning-run binary already parses for labeling so application and labeling agree.
pub fn soccer_curriculum_config_from_env() -> SoccerLearningCurriculumConfig {
    let defaults = SoccerLearningCurriculumConfig::default();
    SoccerLearningCurriculumConfig {
        ball_skills_after_games: soccer_curriculum_env_usize_alias(
            "SOCCER_BATCH_CURRICULUM_BALL_SKILLS_AFTER_GAMES",
            "SOCCER_CURRICULUM_BALL_SKILLS_AFTER_GAMES",
            defaults.ball_skills_after_games,
        ),
        duels_after_games: soccer_curriculum_env_usize_alias(
            "SOCCER_BATCH_CURRICULUM_DUELS_AFTER_GAMES",
            "SOCCER_CURRICULUM_DUELS_AFTER_GAMES",
            defaults.duels_after_games,
        ),
        small_sided_after_games: soccer_curriculum_env_usize_alias(
            "SOCCER_BATCH_CURRICULUM_SMALL_SIDED_AFTER_GAMES",
            "SOCCER_CURRICULUM_SMALL_SIDED_AFTER_GAMES",
            defaults.small_sided_after_games,
        ),
        team_shape_after_games: soccer_curriculum_env_usize_alias(
            "SOCCER_BATCH_CURRICULUM_TEAM_SHAPE_AFTER_GAMES",
            "SOCCER_CURRICULUM_TEAM_SHAPE_AFTER_GAMES",
            defaults.team_shape_after_games,
        ),
        full_match_after_games: soccer_curriculum_env_usize_alias(
            "SOCCER_BATCH_CURRICULUM_FULL_MATCH_AFTER_GAMES",
            "SOCCER_CURRICULUM_FULL_MATCH_AFTER_GAMES",
            defaults.full_match_after_games,
        ),
    }
}

/// Apply the curriculum to a single game's `config` given how many games have already
/// completed (`episode`), when enabled via the environment. Returns the stage applied (for
/// logging), or `None` when the curriculum is off or the env thresholds are invalid — in
/// which case `config` is untouched.
pub fn maybe_apply_soccer_curriculum_for_episode(
    episode: usize,
    config: &mut MatchConfig,
) -> Option<SoccerLearningCurriculumStage> {
    if !soccer_curriculum_enabled_from_env() {
        return None;
    }
    let curriculum = soccer_curriculum_config_from_env();
    if validate_soccer_learning_curriculum_config_for_learning_run(&curriculum).is_err() {
        return None;
    }
    let stage = soccer_learning_curriculum_stage_for_completed_games(episode, &curriculum);
    apply_soccer_curriculum_stage_to_match_config(stage, config);
    Some(stage)
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerLearningCurriculumEpisodeSpec {
    pub stage: SoccerLearningCurriculumStage,
    pub drill_players_per_team: usize,
    pub field_width_yards: f64,
    pub field_length_yards: f64,
    pub duration_seconds: f64,
    pub local_mpc_max_players_per_team: usize,
}

fn soccer_learning_curriculum_stage_players_per_team(
    stage: SoccerLearningCurriculumStage,
) -> usize {
    match stage {
        SoccerLearningCurriculumStage::Locomotion => 2,
        SoccerLearningCurriculumStage::BallSkills => 3,
        SoccerLearningCurriculumStage::Duels => 4,
        SoccerLearningCurriculumStage::SmallSided => 5,
        SoccerLearningCurriculumStage::TeamShape => 8,
        SoccerLearningCurriculumStage::FullMatch => SOCCER_CURRICULUM_FULL_PLAYERS_PER_TEAM,
    }
}

fn soccer_learning_curriculum_stage_pitch_scale(stage: SoccerLearningCurriculumStage) -> f64 {
    match stage {
        SoccerLearningCurriculumStage::Locomotion => 0.45,
        SoccerLearningCurriculumStage::BallSkills => 0.55,
        SoccerLearningCurriculumStage::Duels => 0.65,
        SoccerLearningCurriculumStage::SmallSided => 0.75,
        SoccerLearningCurriculumStage::TeamShape => 0.88,
        SoccerLearningCurriculumStage::FullMatch => 1.0,
    }
}

fn soccer_learning_curriculum_stage_duration_cap_seconds(
    stage: SoccerLearningCurriculumStage,
) -> Option<f64> {
    match stage {
        SoccerLearningCurriculumStage::Locomotion => Some(45.0),
        SoccerLearningCurriculumStage::BallSkills => Some(60.0),
        SoccerLearningCurriculumStage::Duels => Some(90.0),
        SoccerLearningCurriculumStage::SmallSided => Some(150.0),
        SoccerLearningCurriculumStage::TeamShape => Some(240.0),
        SoccerLearningCurriculumStage::FullMatch => None,
    }
}

pub fn soccer_learning_curriculum_episode_config(
    base_config: &MatchConfig,
    completed_games: usize,
    curriculum_config: &SoccerLearningCurriculumConfig,
) -> (MatchConfig, SoccerLearningCurriculumEpisodeSpec) {
    let stage =
        soccer_learning_curriculum_stage_for_completed_games(completed_games, curriculum_config);
    let mut episode_config = base_config.sanitized_for_runtime();
    let players_per_team = soccer_learning_curriculum_stage_players_per_team(stage)
        .clamp(1, SOCCER_CURRICULUM_FULL_PLAYERS_PER_TEAM);
    let pitch_scale = soccer_learning_curriculum_stage_pitch_scale(stage);
    let min_field_width_yards =
        (DEFAULT_FIELD_WIDTH_YARDS * 0.25).min(episode_config.field_width_yards);
    let min_field_length_yards =
        (DEFAULT_FIELD_LENGTH_YARDS * 0.25).min(episode_config.field_length_yards);
    episode_config.field_width_yards =
        (episode_config.field_width_yards * pitch_scale).max(min_field_width_yards);
    episode_config.field_length_yards =
        (episode_config.field_length_yards * pitch_scale).max(min_field_length_yards);
    if let Some(duration_cap_seconds) = soccer_learning_curriculum_stage_duration_cap_seconds(stage)
    {
        let capped_duration = episode_config
            .effective_duration_seconds()
            .min(duration_cap_seconds)
            .max(0.0);
        episode_config.duration_seconds = capped_duration;
        if episode_config.half_duration_seconds > 0.0 {
            episode_config.half_duration_seconds =
                capped_duration / f64::from(episode_config.half_count());
        }
    }
    episode_config.local_mpc_max_players_per_team = episode_config
        .local_mpc_max_players_per_team
        .clamp(1, players_per_team);
    episode_config = episode_config.sanitized_for_runtime();
    let spec = SoccerLearningCurriculumEpisodeSpec {
        stage,
        drill_players_per_team: players_per_team,
        field_width_yards: episode_config.field_width_yards,
        field_length_yards: episode_config.field_length_yards,
        duration_seconds: episode_config.effective_duration_seconds(),
        local_mpc_max_players_per_team: episode_config.local_mpc_max_players_per_team,
    };
    (episode_config, spec)
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerPolicyPromotionGateConfig {
    pub enabled: bool,
    pub min_sample_games: usize,
    pub min_mean_match_fitness: f64,
    pub min_best_match_fitness: f64,
    pub min_mean_play_quality: f64,
    pub max_mean_conceded_goals: f64,
    pub max_mean_goal_margin: f64,
    pub max_mean_chain_net_loss: f64,
}

impl Default for SoccerPolicyPromotionGateConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_sample_games: 1,
            min_mean_match_fitness: -0.25,
            min_best_match_fitness: 0.0,
            min_mean_play_quality: 0.0,
            max_mean_conceded_goals: 6.0,
            max_mean_goal_margin: 8.0,
            max_mean_chain_net_loss: 20.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerPolicyPromotionGateEvaluation {
    pub enabled: bool,
    pub eligible: bool,
    pub sample_games: usize,
    pub min_sample_games: usize,
    pub mean_match_fitness: f64,
    pub best_match_fitness: f64,
    pub mean_play_quality: f64,
    pub mean_conceded_goals: f64,
    pub mean_goal_margin: f64,
    pub mean_chain_net_loss: f64,
    pub rejection_reasons: Vec<String>,
}

pub fn validate_soccer_policy_promotion_gate_config_for_learning_run(
    config: &SoccerPolicyPromotionGateConfig,
) -> Result<(), String> {
    if config.min_sample_games == 0 {
        return Err("promotion gate minSampleGames must be at least 1".to_string());
    }
    for (name, value) in [
        ("minMeanMatchFitness", config.min_mean_match_fitness),
        ("minBestMatchFitness", config.min_best_match_fitness),
        ("minMeanPlayQuality", config.min_mean_play_quality),
        ("maxMeanConcededGoals", config.max_mean_conceded_goals),
        ("maxMeanGoalMargin", config.max_mean_goal_margin),
        ("maxMeanChainNetLoss", config.max_mean_chain_net_loss),
    ] {
        if !value.is_finite() {
            return Err(format!("promotion gate {name} must be finite"));
        }
    }
    if config.min_mean_play_quality < 0.0 {
        return Err("promotion gate minMeanPlayQuality must be non-negative".to_string());
    }
    if config.max_mean_conceded_goals < 0.0 {
        return Err("promotion gate maxMeanConcededGoals must be non-negative".to_string());
    }
    if config.max_mean_goal_margin < 0.0 {
        return Err("promotion gate maxMeanGoalMargin must be non-negative".to_string());
    }
    if config.max_mean_chain_net_loss < 0.0 {
        return Err("promotion gate maxMeanChainNetLoss must be non-negative".to_string());
    }
    Ok(())
}

pub fn evaluate_soccer_policy_promotion_gate<'a, I>(
    summaries: I,
    config: SoccerPolicyPromotionGateConfig,
) -> SoccerPolicyPromotionGateEvaluation
where
    I: IntoIterator<Item = &'a MatchSummary>,
{
    let mut sample_games = 0usize;
    let mut match_fitness_sum = 0.0;
    let mut best_match_fitness = f64::NEG_INFINITY;
    let mut play_quality_sum = 0.0;
    let mut conceded_goals_sum = 0.0;
    let mut goal_margin_sum = 0.0;
    let mut chain_net_loss_sum = 0.0;
    let mut non_finite_metric_samples = 0usize;

    for summary in summaries {
        sample_games = sample_games.saturating_add(1);
        let mut sample_has_non_finite = soccer_summary_has_non_finite_learning_metrics(summary);
        let match_fitness = soccer_learning_run_score(summary).match_fitness;
        if !match_fitness.is_finite() {
            sample_has_non_finite = true;
        }
        let match_fitness = finite_promotion_gate_metric(match_fitness, SOCCER_MATCH_FITNESS_MIN);
        let play_quality = soccer_learning_match_play_quality(summary);
        if !play_quality.is_finite() {
            sample_has_non_finite = true;
        }
        let play_quality = finite_promotion_gate_metric(play_quality, 0.0);
        let chain_net_loss = (summary.stats.pass_chains_net_loss_home
            + summary.stats.pass_chains_net_loss_away) as f64
            * 0.5;
        if !chain_net_loss.is_finite() {
            sample_has_non_finite = true;
        }
        let chain_net_loss = finite_promotion_gate_metric(chain_net_loss, 0.0);
        if sample_has_non_finite {
            non_finite_metric_samples = non_finite_metric_samples.saturating_add(1);
        }
        match_fitness_sum += match_fitness;
        best_match_fitness = best_match_fitness.max(match_fitness);
        play_quality_sum += play_quality;
        conceded_goals_sum += (summary.score_home + summary.score_away) as f64 * 0.5;
        goal_margin_sum += (summary.score_home as i32 - summary.score_away as i32).abs() as f64;
        chain_net_loss_sum += chain_net_loss;
    }

    let denominator = sample_games.max(1) as f64;
    let mean_match_fitness = match_fitness_sum / denominator;
    let mean_play_quality = play_quality_sum / denominator;
    let mean_conceded_goals = conceded_goals_sum / denominator;
    let mean_goal_margin = goal_margin_sum / denominator;
    let mean_chain_net_loss = chain_net_loss_sum / denominator;
    let best_match_fitness = if best_match_fitness.is_finite() {
        best_match_fitness
    } else {
        SOCCER_MATCH_FITNESS_MIN
    };
    let mut rejection_reasons = Vec::new();

    if config.enabled {
        if sample_games < config.min_sample_games {
            rejection_reasons.push(format!(
                "sample_games {sample_games} below min_sample_games {}",
                config.min_sample_games
            ));
        }
        if non_finite_metric_samples > 0 {
            rejection_reasons.push(format!(
                "non_finite_learning_metrics {non_finite_metric_samples} samples"
            ));
        }
        if mean_match_fitness < config.min_mean_match_fitness {
            rejection_reasons.push(format!(
                "mean_match_fitness {mean_match_fitness:.4} below min_mean_match_fitness {:.4}",
                config.min_mean_match_fitness
            ));
        }
        if best_match_fitness < config.min_best_match_fitness {
            rejection_reasons.push(format!(
                "best_match_fitness {best_match_fitness:.4} below min_best_match_fitness {:.4}",
                config.min_best_match_fitness
            ));
        }
        if mean_play_quality < config.min_mean_play_quality {
            rejection_reasons.push(format!(
                "mean_play_quality {mean_play_quality:.4} below min_mean_play_quality {:.4}",
                config.min_mean_play_quality
            ));
        }
        if mean_conceded_goals > config.max_mean_conceded_goals {
            rejection_reasons.push(format!(
                "mean_conceded_goals {mean_conceded_goals:.4} above max_mean_conceded_goals {:.4}",
                config.max_mean_conceded_goals
            ));
        }
        if mean_goal_margin > config.max_mean_goal_margin {
            rejection_reasons.push(format!(
                "mean_goal_margin {mean_goal_margin:.4} above max_mean_goal_margin {:.4}",
                config.max_mean_goal_margin
            ));
        }
        if mean_chain_net_loss > config.max_mean_chain_net_loss {
            rejection_reasons.push(format!(
                "mean_chain_net_loss {mean_chain_net_loss:.4} above max_mean_chain_net_loss {:.4}",
                config.max_mean_chain_net_loss
            ));
        }
    }

    SoccerPolicyPromotionGateEvaluation {
        enabled: config.enabled,
        eligible: !config.enabled || rejection_reasons.is_empty(),
        sample_games,
        min_sample_games: config.min_sample_games,
        mean_match_fitness,
        best_match_fitness,
        mean_play_quality,
        mean_conceded_goals,
        mean_goal_margin,
        mean_chain_net_loss,
        rejection_reasons,
    }
}

fn finite_promotion_gate_metric(value: f64, fallback: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

fn soccer_summary_has_non_finite_learning_metrics(summary: &MatchSummary) -> bool {
    let stats = &summary.stats;
    [
        stats.completed_pass_gain_yards_home,
        stats.completed_pass_gain_yards_away,
        stats.pass_chain_gain_yards_home,
        stats.pass_chain_gain_yards_away,
        stats.defensive_chase_load_home,
        stats.defensive_chase_load_away,
        stats.possession_chase_advantage_home,
        stats.possession_chase_advantage_away,
        stats.teamwork_upfield_progress_home,
        stats.teamwork_upfield_progress_away,
        stats.teamwork_near_ball_progress_home,
        stats.teamwork_near_ball_progress_away,
    ]
    .into_iter()
    .any(|value| !value.is_finite())
}

pub fn validate_soccer_evolution_options_for_learning_run(
    options: &SoccerEvolutionOptions,
) -> Result<(), String> {
    for (name, value) in [
        ("mutationRate", options.mutation_rate),
        ("crossoverRate", options.crossover_rate),
        ("explorationRate", options.exploration_rate),
    ] {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(format!("{name} must be finite and in [0, 1]"));
        }
    }
    for (name, value) in [
        ("mutationScale", options.mutation_scale),
        ("explorationScale", options.exploration_scale),
        ("eliteWeightFloor", options.elite_weight_floor),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(format!("{name} must be finite and non-negative"));
        }
    }
    if !(1..=SOCCER_EVOLUTION_MAX_POPULATION_SIZE).contains(&options.population_size) {
        return Err(format!(
            "populationSize must be in [1, {SOCCER_EVOLUTION_MAX_POPULATION_SIZE}]"
        ));
    }
    Ok(())
}

pub fn validate_soccer_q_policy_options_for_learning_run(
    options: &SoccerQPolicyOptions,
) -> Result<(), String> {
    for (name, value) in [
        ("alpha", options.alpha),
        ("gamma", options.gamma),
        ("explorationEpsilon", options.exploration_epsilon),
    ] {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(format!("{name} must be finite and in [0, 1]"));
        }
    }
    Ok(())
}

pub fn validate_soccer_neural_learning_config_for_learning_run(
    config: &SoccerNeuralLearningConfig,
) -> Result<(), String> {
    for (name, value) in [
        ("learningRate", config.learning_rate),
        ("targetScale", config.target_scale),
        ("targetClip", config.target_clip),
        ("criticBaselineWeight", config.critic_baseline_weight),
        // All convergent multi-agent-RL knobs are validated: MAPPO team-reward SHARE (ours) and the
        // MARL team/intermediate reward weights + the MAPPO clip epsilon (theirs).
        ("mappoTeamRewardShare", config.mappo_team_reward_share),
        ("marlTeamRewardWeight", config.marl_team_reward_weight),
        (
            "marlIntermediateRewardWeight",
            config.marl_intermediate_reward_weight,
        ),
        ("mappoClipEpsilon", config.mappo_clip_epsilon),
    ] {
        if !value.is_finite() {
            return Err(format!("{name} must be finite"));
        }
    }
    if !config.enabled {
        return Ok(());
    }
    if config.learning_rate <= 0.0 {
        return Err("learningRate must be positive when neural learning is enabled".to_string());
    }
    if config.learning_rate > MAX_SOCCER_NEURAL_LEARNING_RATE {
        return Err(format!(
            "learningRate must be <= {MAX_SOCCER_NEURAL_LEARNING_RATE} when neural learning is enabled"
        ));
    }
    if config.batch_size == 0 {
        return Err("batchSize must be at least 1 when neural learning is enabled".to_string());
    }
    if config.train_every_ticks == 0 {
        return Err(
            "trainEveryTicks must be at least 1 when neural learning is enabled".to_string(),
        );
    }
    if config.max_batches_per_tick == 0 {
        return Err(
            "maxBatchesPerTick must be at least 1 when neural learning is enabled".to_string(),
        );
    }
    if config.hidden_units < 2 {
        return Err("hiddenUnits must be at least 2 when neural learning is enabled".to_string());
    }
    if config.target_scale <= 1e-9 {
        return Err("targetScale must be positive when neural learning is enabled".to_string());
    }
    if config.max_pending_batches == 0 {
        return Err(
            "maxPendingBatches must be at least 1 when neural learning is enabled".to_string(),
        );
    }
    if config.replay_capacity == 0 {
        return Err(
            "replayCapacity must be at least 1 when neural learning is enabled".to_string(),
        );
    }
    if config.replay_samples_per_tick == 0 {
        return Err(
            "replaySamplesPerTick must be at least 1 when neural learning is enabled".to_string(),
        );
    }
    if config.target_clip <= 0.0 {
        return Err("targetClip must be positive when neural learning is enabled".to_string());
    }
    if config.snapshot_every_batches == 0 {
        return Err(
            "snapshotEveryBatches must be at least 1 when neural learning is enabled".to_string(),
        );
    }
    if !(0.0..=1.0).contains(&config.critic_baseline_weight) {
        return Err(
            "criticBaselineWeight must be in [0, 1] when neural learning is enabled".to_string(),
        );
    }
    // Keep both convergent MARL/MAPPO validations (ours: team-reward share; theirs: team +
    // intermediate reward weights and the MAPPO clip epsilon).
    if !(0.0..=1.0).contains(&config.mappo_team_reward_share) {
        return Err(
            "mappoTeamRewardShare must be in [0, 1] when neural learning is enabled".to_string(),
        );
    }
    if !(0.0..=1.0).contains(&config.marl_team_reward_weight) {
        return Err(
            "marlTeamRewardWeight must be in [0, 1] when neural learning is enabled".to_string(),
        );
    }
    if !(0.0..=2.0).contains(&config.marl_intermediate_reward_weight) {
        return Err(
            "marlIntermediateRewardWeight must be in [0, 2] when neural learning is enabled"
                .to_string(),
        );
    }
    if !(SOCCER_MAPPO_MIN_CLIP_EPSILON..=1.0).contains(&config.mappo_clip_epsilon) {
        return Err(
            "mappoClipEpsilon must be in [0.01, 1] when neural learning is enabled".to_string(),
        );
    }
    Ok(())
}

pub fn validate_soccer_neural_blend_config_for_learning_run(
    blend: &SoccerNeuralBlendConfig,
) -> Result<(), String> {
    for (name, value) in [
        ("lambda", blend.lambda),
        ("tieEpsilon", blend.tie_epsilon),
        ("mctsExploration", blend.mcts_exploration),
        ("mctsModelWeight", blend.mcts_model_weight),
    ] {
        if !value.is_finite() {
            return Err(format!("{name} must be finite"));
        }
    }
    if !blend.mcts_enabled {
        return Ok(());
    }
    if blend.mcts_simulations == 0 || blend.mcts_simulations > SOCCER_NEURAL_MCTS_MAX_SIMULATIONS {
        return Err(format!(
            "mctsSimulations must be in [1, {SOCCER_NEURAL_MCTS_MAX_SIMULATIONS}] when MCTS is enabled"
        ));
    }
    if !(2..=SOCCER_NEURAL_MCTS_MAX_CANDIDATES).contains(&blend.mcts_candidates) {
        return Err(format!(
            "mctsCandidates must be in [2, {SOCCER_NEURAL_MCTS_MAX_CANDIDATES}] when MCTS is enabled"
        ));
    }
    if blend.mcts_depth == 0 || blend.mcts_depth > SOCCER_NEURAL_MCTS_MAX_DEPTH {
        return Err(format!(
            "mctsDepth must be in [1, {SOCCER_NEURAL_MCTS_MAX_DEPTH}] when MCTS is enabled"
        ));
    }
    if !(0.0..=SOCCER_NEURAL_MCTS_MAX_EXPLORATION).contains(&blend.mcts_exploration) {
        return Err(format!(
            "mctsExploration must be in [0, {SOCCER_NEURAL_MCTS_MAX_EXPLORATION}] when MCTS is enabled"
        ));
    }
    if !(0.0..=1.0).contains(&blend.mcts_model_weight) {
        return Err("mctsModelWeight must be in [0, 1] when MCTS is enabled".to_string());
    }
    Ok(())
}

fn soccer_evolution_population_size_for_learning_run(population_size: usize) -> usize {
    population_size.clamp(1, SOCCER_EVOLUTION_MAX_POPULATION_SIZE)
}

fn parse_soccer_policy_search_max_adapted_population(raw: Option<&str>) -> Result<usize, String> {
    let Some(raw) = raw else {
        return Ok(SOCCER_POLICY_SEARCH_MAX_ADAPTED_POPULATION);
    };
    let trimmed = raw.trim();
    let parsed = trimmed.parse::<usize>().map_err(|_| {
        format!(
            "{SOCCER_POLICY_SEARCH_MAX_ADAPTED_POPULATION_ENV} must be a positive integer, got {raw:?}"
        )
    })?;
    if parsed == 0 {
        return Err(format!(
            "{SOCCER_POLICY_SEARCH_MAX_ADAPTED_POPULATION_ENV} must be a positive integer, got {raw:?}"
        ));
    }
    Ok(parsed.min(SOCCER_EVOLUTION_MAX_POPULATION_SIZE))
}

fn soccer_policy_search_max_adapted_population() -> usize {
    *SOCCER_POLICY_SEARCH_MAX_ADAPTED_POPULATION_OVERRIDE.get_or_init(|| {
        let raw = std::env::var(SOCCER_POLICY_SEARCH_MAX_ADAPTED_POPULATION_ENV).ok();
        match parse_soccer_policy_search_max_adapted_population(raw.as_deref()) {
            Ok(value) => value,
            Err(err) => {
                eprintln!(
                    "soccer-learning: {err}; using default {}",
                    SOCCER_POLICY_SEARCH_MAX_ADAPTED_POPULATION
                );
                SOCCER_POLICY_SEARCH_MAX_ADAPTED_POPULATION
            }
        }
    })
}

fn normalize_soccer_evolution_options_for_learning_search(
    mut options: SoccerEvolutionOptions,
) -> Result<SoccerEvolutionOptions, String> {
    options.population_size =
        soccer_evolution_population_size_for_learning_run(options.population_size);
    validate_soccer_evolution_options_for_learning_run(&options)?;
    Ok(options)
}

#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SoccerEvolutionOptionsPatch {
    #[serde(alias = "mutation_rate")]
    mutation_rate: Option<f64>,
    #[serde(alias = "mutation_scale")]
    mutation_scale: Option<f64>,
    #[serde(alias = "crossover_rate")]
    crossover_rate: Option<f64>,
    #[serde(alias = "exploration_rate")]
    exploration_rate: Option<f64>,
    #[serde(alias = "exploration_scale")]
    exploration_scale: Option<f64>,
    #[serde(alias = "elite_weight_floor")]
    elite_weight_floor: Option<f64>,
    #[serde(alias = "population_size")]
    population_size: Option<usize>,
    seed: Option<u64>,
}

impl SoccerEvolutionOptionsPatch {
    fn has_any_field(self) -> bool {
        self.mutation_rate.is_some()
            || self.mutation_scale.is_some()
            || self.crossover_rate.is_some()
            || self.exploration_rate.is_some()
            || self.exploration_scale.is_some()
            || self.elite_weight_floor.is_some()
            || self.population_size.is_some()
            || self.seed.is_some()
    }
}

fn soccer_evolution_options_patch_from_value(
    search_metadata: &Value,
) -> Option<SoccerEvolutionOptionsPatch> {
    if !search_metadata.is_object() {
        return None;
    }
    if let Some(patch) = search_metadata
        .get("options")
        .filter(|value| value.is_object())
        .and_then(soccer_evolution_options_patch_from_candidate)
    {
        return Some(patch);
    }
    if let Some(patch) = soccer_evolution_options_patch_from_candidate(search_metadata) {
        return Some(patch);
    }
    for key in [
        "tactical",
        "policy",
        "evolution",
        "evolutionaryAlgorithm",
        "evolutionarySearch",
        "geneticAlgorithm",
        "geneticProgramming",
        "adversarialLearning",
        "mdp",
        "pomdp",
        "search",
        "learningProvenance",
        "searchParameters",
    ] {
        if let Some(patch) = search_metadata
            .get(key)
            .and_then(soccer_evolution_options_patch_from_value)
        {
            return Some(patch);
        }
    }
    None
}

fn soccer_evolution_options_patch_from_candidate(
    value: &Value,
) -> Option<SoccerEvolutionOptionsPatch> {
    if !value.is_object() {
        return None;
    }
    let patch = serde_json::from_value::<SoccerEvolutionOptionsPatch>(value.clone()).ok()?;
    patch.has_any_field().then_some(patch)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum SoccerEvolutionSearchPreset {
    GeneticAlgorithm,
    GeneticProgramming,
}

fn soccer_evolution_search_preset_from_value(value: &Value) -> Option<SoccerEvolutionSearchPreset> {
    match value {
        Value::String(text) => soccer_evolution_search_preset_from_text(text),
        Value::Array(items) => items
            .iter()
            .filter_map(soccer_evolution_search_preset_from_value)
            .max(),
        Value::Object(map) => {
            let mut preset = [
                "algorithm",
                "algorithmFamily",
                "searchAlgorithm",
                "searchFamily",
                "searchMode",
                "mode",
                "family",
                "sourceKind",
            ]
            .into_iter()
            .filter_map(|key| map.get(key))
            .filter_map(soccer_evolution_search_preset_from_value)
            .max();

            for (key, nested) in map {
                let key_preset = soccer_evolution_search_preset_from_text(key);
                let nested_preset = soccer_evolution_search_preset_from_value(nested);
                preset = preset.max(key_preset).max(nested_preset);
            }
            preset
        }
        _ => None,
    }
}

fn soccer_evolution_search_preset_from_text(text: &str) -> Option<SoccerEvolutionSearchPreset> {
    let normalized = text
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect::<String>();
    if normalized.is_empty() {
        return None;
    }
    if normalized.contains("geneticprogramming")
        || normalized.contains("symbolicregression")
        || normalized.contains("programtreesearch")
        || normalized == "gp"
    {
        return Some(SoccerEvolutionSearchPreset::GeneticProgramming);
    }
    if normalized.contains("geneticalgorithm")
        || normalized.contains("evolutionary")
        || normalized.contains("evolution")
    {
        return Some(SoccerEvolutionSearchPreset::GeneticAlgorithm);
    }
    None
}

fn apply_soccer_evolution_search_preset(
    options: &mut SoccerEvolutionOptions,
    preset: SoccerEvolutionSearchPreset,
    patch: Option<SoccerEvolutionOptionsPatch>,
) {
    let Some(patch) = patch else {
        return apply_soccer_evolution_search_preset_defaults(options, preset);
    };
    let defaults = soccer_evolution_search_preset_defaults(preset);
    if patch.mutation_rate.is_none() {
        options.mutation_rate = options.mutation_rate.max(defaults.mutation_rate);
    }
    if patch.mutation_scale.is_none() {
        options.mutation_scale = options.mutation_scale.max(defaults.mutation_scale);
    }
    if patch.crossover_rate.is_none() {
        options.crossover_rate = options.crossover_rate.max(defaults.crossover_rate);
    }
    if patch.exploration_rate.is_none() {
        options.exploration_rate = options.exploration_rate.max(defaults.exploration_rate);
    }
    if patch.exploration_scale.is_none() {
        options.exploration_scale = options.exploration_scale.max(defaults.exploration_scale);
    }
    if patch.elite_weight_floor.is_none() {
        options.elite_weight_floor = options.elite_weight_floor.max(defaults.elite_weight_floor);
    }
    if patch.population_size.is_none() {
        options.population_size = options.population_size.max(defaults.population_size);
    }
}

fn apply_soccer_evolution_search_preset_defaults(
    options: &mut SoccerEvolutionOptions,
    preset: SoccerEvolutionSearchPreset,
) {
    let defaults = soccer_evolution_search_preset_defaults(preset);
    options.mutation_rate = options.mutation_rate.max(defaults.mutation_rate);
    options.mutation_scale = options.mutation_scale.max(defaults.mutation_scale);
    options.crossover_rate = options.crossover_rate.max(defaults.crossover_rate);
    options.exploration_rate = options.exploration_rate.max(defaults.exploration_rate);
    options.exploration_scale = options.exploration_scale.max(defaults.exploration_scale);
    options.elite_weight_floor = options.elite_weight_floor.max(defaults.elite_weight_floor);
    options.population_size = options.population_size.max(defaults.population_size);
}

fn soccer_evolution_search_preset_defaults(
    preset: SoccerEvolutionSearchPreset,
) -> SoccerEvolutionOptions {
    match preset {
        SoccerEvolutionSearchPreset::GeneticAlgorithm => SoccerEvolutionOptions {
            mutation_rate: 0.06,
            mutation_scale: 0.26,
            crossover_rate: 0.55,
            exploration_rate: 0.08,
            exploration_scale: 0.65,
            elite_weight_floor: 0.06,
            population_size: 16,
            seed: 0,
        },
        SoccerEvolutionSearchPreset::GeneticProgramming => SoccerEvolutionOptions {
            mutation_rate: 0.08,
            mutation_scale: 0.34,
            crossover_rate: 0.62,
            exploration_rate: 0.14,
            exploration_scale: 0.90,
            elite_weight_floor: 0.08,
            population_size: 24,
            seed: 0,
        },
    }
}

pub fn soccer_evolution_options_from_search_metadata(
    search_metadata: Option<&Value>,
    current: SoccerEvolutionOptions,
) -> Option<SoccerEvolutionOptions> {
    let search_metadata = search_metadata?;
    let patch = soccer_evolution_options_patch_from_value(search_metadata);
    let preset = soccer_evolution_search_preset_from_value(search_metadata);
    if patch.is_none() && preset.is_none() {
        return None;
    }
    let mut options = current;
    if let Some(preset) = preset {
        apply_soccer_evolution_search_preset(&mut options, preset, patch);
    }
    let Some(patch) = patch else {
        options.population_size =
            soccer_evolution_population_size_for_learning_run(options.population_size);
        return Some(options);
    };
    if let Some(value) = patch.mutation_rate.filter(|value| value.is_finite()) {
        options.mutation_rate = value.clamp(0.0, 1.0);
    }
    if let Some(value) = patch.mutation_scale.filter(|value| value.is_finite()) {
        options.mutation_scale = value.max(0.0);
    }
    if let Some(value) = patch.crossover_rate.filter(|value| value.is_finite()) {
        options.crossover_rate = value.clamp(0.0, 1.0);
    }
    if let Some(value) = patch.exploration_rate.filter(|value| value.is_finite()) {
        options.exploration_rate = value.clamp(0.0, 1.0);
    }
    if let Some(value) = patch.exploration_scale.filter(|value| value.is_finite()) {
        options.exploration_scale = value.max(0.0);
    }
    if let Some(value) = patch.elite_weight_floor.filter(|value| value.is_finite()) {
        options.elite_weight_floor = value.max(0.0);
    }
    if let Some(value) = patch.population_size {
        options.population_size = soccer_evolution_population_size_for_learning_run(value);
    }
    if let Some(value) = patch.seed {
        options.seed = value;
    }
    options.population_size =
        soccer_evolution_population_size_for_learning_run(options.population_size);
    Some(options)
}

const SOCCER_LEARNING_FINGERPRINT_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const SOCCER_LEARNING_FINGERPRINT_PRIME: u64 = 0x0000_0100_0000_01b3;

fn soccer_learning_fingerprint_mix(hash: &mut u64, value: u64) {
    *hash ^= value;
    *hash = hash.wrapping_mul(SOCCER_LEARNING_FINGERPRINT_PRIME);
}

fn soccer_learning_fingerprint_usize(hash: &mut u64, value: usize) {
    soccer_learning_fingerprint_mix(hash, value as u64);
}

fn soccer_learning_fingerprint_f64(hash: &mut u64, value: f64) {
    let bits = if value == 0.0 { 0 } else { value.to_bits() };
    soccer_learning_fingerprint_mix(hash, bits);
}

fn soccer_learning_fingerprint_str(hash: &mut u64, value: &str) {
    soccer_learning_fingerprint_usize(hash, value.len());
    for byte in value.as_bytes() {
        soccer_learning_fingerprint_mix(hash, u64::from(*byte));
    }
}

fn soccer_learning_fingerprint_hash<T: Hash>(hash: &mut u64, value: &T) {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    soccer_learning_fingerprint_mix(hash, hasher.finish());
}

fn soccer_q_policy_options_fingerprint(hash: &mut u64, options: &SoccerQPolicyOptions) {
    soccer_learning_fingerprint_f64(hash, options.alpha);
    soccer_learning_fingerprint_f64(hash, options.gamma);
}

fn soccer_q_entry_fingerprint(entry: &SoccerQEntry) -> u64 {
    let mut hash = SOCCER_LEARNING_FINGERPRINT_OFFSET;
    soccer_learning_fingerprint_hash(&mut hash, &entry.state);
    soccer_learning_fingerprint_str(&mut hash, &entry.action);
    soccer_learning_fingerprint_f64(&mut hash, entry.value);
    soccer_learning_fingerprint_mix(&mut hash, u64::from(entry.visits));
    hash
}

fn soccer_q_target_entry_fingerprint(entry: &SoccerQTargetEntry) -> u64 {
    let mut hash = SOCCER_LEARNING_FINGERPRINT_OFFSET;
    soccer_learning_fingerprint_hash(&mut hash, &entry.state);
    soccer_learning_fingerprint_str(&mut hash, &entry.action);
    soccer_learning_fingerprint_usize(&mut hash, entry.target_fine_cell_id);
    soccer_learning_fingerprint_usize(&mut hash, entry.target_tactical_cell_id);
    soccer_learning_fingerprint_usize(&mut hash, entry.target_macro_cell_id);
    soccer_learning_fingerprint_usize(&mut hash, entry.target_root_cell_id);
    soccer_learning_fingerprint_f64(&mut hash, entry.value);
    soccer_learning_fingerprint_mix(&mut hash, u64::from(entry.visits));
    hash
}

fn soccer_q_policy_fingerprint(hash: &mut u64, team: Team, policy: &SoccerQPolicy) {
    soccer_learning_fingerprint_mix(
        hash,
        match team {
            Team::Home => 0,
            Team::Away => 1,
        },
    );
    soccer_q_policy_options_fingerprint(hash, &policy.options);

    let mut action_hashes = policy
        .entries()
        .into_iter()
        .map(|entry| soccer_q_entry_fingerprint(&entry))
        .collect::<Vec<_>>();
    action_hashes.sort_unstable();
    soccer_learning_fingerprint_usize(hash, action_hashes.len());
    for entry_hash in action_hashes {
        soccer_learning_fingerprint_mix(hash, entry_hash);
    }

    let mut target_hashes = policy
        .target_entries()
        .into_iter()
        .map(|entry| soccer_q_target_entry_fingerprint(&entry))
        .collect::<Vec<_>>();
    target_hashes.sort_unstable();
    soccer_learning_fingerprint_usize(hash, target_hashes.len());
    for entry_hash in target_hashes {
        soccer_learning_fingerprint_mix(hash, entry_hash);
    }
}

pub fn soccer_team_q_policies_fingerprint(policies: &SoccerTeamQPolicies) -> u64 {
    let mut hash = SOCCER_LEARNING_FINGERPRINT_OFFSET;
    soccer_q_policy_fingerprint(&mut hash, Team::Home, &policies.home);
    soccer_q_policy_fingerprint(&mut hash, Team::Away, &policies.away);
    hash
}

pub fn soccer_tactical_learning_weights_fingerprint(
    weights: &SoccerTacticalLearningWeights,
) -> u64 {
    let mut hash = SOCCER_LEARNING_FINGERPRINT_OFFSET;
    soccer_learning_fingerprint_f64(&mut hash, weights.attack_spacing_delta_weight);
    soccer_learning_fingerprint_f64(&mut hash, weights.attack_spacing_score_weight);
    soccer_learning_fingerprint_f64(&mut hash, weights.attack_width_delta_weight);
    soccer_learning_fingerprint_f64(&mut hash, weights.attack_width_score_weight);
    soccer_learning_fingerprint_f64(&mut hash, weights.attack_flank_lane_weight);
    soccer_learning_fingerprint_f64(&mut hash, weights.shot_choice_learning_weight);
    soccer_learning_fingerprint_f64(&mut hash, weights.goal_entry_pass_learning_weight);
    soccer_learning_fingerprint_f64(&mut hash, weights.pressure_release_learning_weight);
    soccer_learning_fingerprint_f64(&mut hash, weights.pass_target_ranking_learning_weight);
    soccer_learning_fingerprint_f64(&mut hash, weights.defense_spacing_delta_weight);
    soccer_learning_fingerprint_f64(&mut hash, weights.defense_spacing_score_weight);
    soccer_learning_fingerprint_f64(&mut hash, weights.defense_contract_delta_weight);
    soccer_learning_fingerprint_f64(&mut hash, weights.defense_compactness_score_weight);
    soccer_learning_fingerprint_f64(&mut hash, weights.defense_ball_depth_score_weight);
    soccer_learning_fingerprint_f64(&mut hash, weights.defense_endline_soft_penalty_weight);
    soccer_learning_fingerprint_f64(&mut hash, weights.defense_endline_hard_penalty_weight);
    soccer_learning_fingerprint_f64(&mut hash, weights.defender_midfielder_press_weight);
    soccer_learning_fingerprint_f64(&mut hash, weights.midfielder_press_weight);
    soccer_learning_fingerprint_f64(&mut hash, weights.defensive_line_press_learning_weight);
    soccer_learning_fingerprint_f64(&mut hash, weights.formation_lp_alignment_weight);
    hash
}

pub fn soccer_neural_network_snapshot_fingerprint(snapshot: &SoccerNeuralNetworkSnapshot) -> u64 {
    let mut hash = SOCCER_LEARNING_FINGERPRINT_OFFSET;
    soccer_learning_fingerprint_usize(&mut hash, snapshot.input_dim);
    soccer_learning_fingerprint_usize(&mut hash, snapshot.output_dim);
    soccer_learning_fingerprint_usize(&mut hash, snapshot.parameter_count);
    soccer_learning_fingerprint_f64(&mut hash, snapshot.l2_norm);
    soccer_learning_fingerprint_usize(&mut hash, snapshot.layers.len());
    for layer in &snapshot.layers {
        soccer_learning_fingerprint_str(&mut hash, &layer.activation);
        soccer_learning_fingerprint_usize(&mut hash, layer.weights.len());
        for row in &layer.weights {
            soccer_learning_fingerprint_usize(&mut hash, row.len());
            for value in row {
                soccer_learning_fingerprint_f64(&mut hash, *value);
            }
        }
        soccer_learning_fingerprint_usize(&mut hash, layer.biases.len());
        for value in &layer.biases {
            soccer_learning_fingerprint_f64(&mut hash, *value);
        }
    }
    if let Some(policy_head) = snapshot.policy_head.as_ref() {
        soccer_learning_fingerprint_str(&mut hash, "policyHead");
        soccer_learning_fingerprint_mix(
            &mut hash,
            soccer_neural_network_snapshot_fingerprint(&policy_head.network),
        );
        soccer_learning_fingerprint_usize(&mut hash, policy_head.training_steps);
        if let Some(loss) = policy_head.average_loss {
            soccer_learning_fingerprint_f64(&mut hash, loss);
        }
        soccer_learning_fingerprint_usize(&mut hash, policy_head.specialist_heads.len());
        for specialist in &policy_head.specialist_heads {
            soccer_learning_fingerprint_str(&mut hash, &specialist.kind);
            soccer_learning_fingerprint_mix(
                &mut hash,
                soccer_neural_network_snapshot_fingerprint(&specialist.network),
            );
            soccer_learning_fingerprint_usize(&mut hash, specialist.training_steps);
            if let Some(loss) = specialist.average_loss {
                soccer_learning_fingerprint_f64(&mut hash, loss);
            }
        }
        soccer_learning_fingerprint_usize(&mut hash, policy_head.role_heads.len());
        for role_head in &policy_head.role_heads {
            soccer_learning_fingerprint_str(&mut hash, &role_head.role);
            soccer_learning_fingerprint_mix(
                &mut hash,
                soccer_neural_network_snapshot_fingerprint(&role_head.network),
            );
            soccer_learning_fingerprint_usize(&mut hash, role_head.training_steps);
            if let Some(loss) = role_head.average_loss {
                soccer_learning_fingerprint_f64(&mut hash, loss);
            }
            soccer_learning_fingerprint_usize(&mut hash, role_head.specialist_heads.len());
            for specialist in &role_head.specialist_heads {
                soccer_learning_fingerprint_str(&mut hash, &specialist.kind);
                soccer_learning_fingerprint_mix(
                    &mut hash,
                    soccer_neural_network_snapshot_fingerprint(&specialist.network),
                );
                soccer_learning_fingerprint_usize(&mut hash, specialist.training_steps);
                if let Some(loss) = specialist.average_loss {
                    soccer_learning_fingerprint_f64(&mut hash, loss);
                }
            }
        }
    }
    hash
}

#[derive(Clone, Copy, Debug)]
pub struct SoccerTacticalLearningGenomeParent<'a> {
    pub summary: &'a SoccerTacticalLearningSummary,
    pub weights: &'a SoccerTacticalLearningWeights,
    pub fitness: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct SoccerPostgresPolicyRefreshCheck<'a> {
    pub current_policy_version_id: Option<&'a str>,
    pub current_generation: i32,
    pub current_updated_at_micros: i64,
    pub current_policy_fingerprint: Option<u64>,
    pub latest_policy_fingerprint: Option<u64>,
    pub current_neural_network_present: bool,
    pub current_neural_network_fingerprint: Option<u64>,
    pub latest_policy_version_id: &'a str,
    pub latest_generation: i32,
    pub latest_updated_at_micros: i64,
    pub latest_neural_network_present: bool,
    pub latest_neural_network_fingerprint: Option<u64>,
    pub current_tactical_learning_fingerprint: Option<u64>,
    pub latest_tactical_learning_fingerprint: Option<u64>,
    pub local_tactical_evolved_since_pg_refresh: bool,
    pub postgres_tactical_learning_authoritative: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SoccerPostgresPolicyRefreshDecision {
    pub refresh_policy: bool,
    pub apply_tactical_learning: bool,
    pub same_policy_version: bool,
    pub same_policy_newer_revision: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SoccerPostgresNewSimRefreshPlan {
    pub refresh_from_postgres: bool,
    pub flush_pending_policy_versions: bool,
    pub wait_for_async_policy_versions: bool,
}

pub fn soccer_postgres_policy_refresh_decision(
    check: SoccerPostgresPolicyRefreshCheck<'_>,
) -> SoccerPostgresPolicyRefreshDecision {
    let same_policy_version =
        check.current_policy_version_id == Some(check.latest_policy_version_id);
    let same_policy_newer_revision =
        same_policy_version && check.latest_updated_at_micros > check.current_updated_at_micros;
    let same_policy_neural_weights_changed = same_policy_version
        && check.latest_neural_network_fingerprint.is_some()
        && check.current_neural_network_fingerprint != check.latest_neural_network_fingerprint;
    let same_policy_q_weights_changed = same_policy_version
        && check.latest_policy_fingerprint.is_some()
        && check.current_policy_fingerprint != check.latest_policy_fingerprint;
    let same_policy_tactical_weights_changed = same_policy_version
        && check.latest_tactical_learning_fingerprint.is_some()
        && check.current_tactical_learning_fingerprint
            != check.latest_tactical_learning_fingerprint;
    let newer_generation = check.latest_generation > check.current_generation;
    let different_head_at_same_generation =
        check.latest_generation == check.current_generation && !same_policy_version;

    let refresh_policy = check.current_policy_version_id.is_none()
        || newer_generation
        || different_head_at_same_generation
        || same_policy_newer_revision
        || same_policy_q_weights_changed
        || same_policy_neural_weights_changed
        || (same_policy_version
            && !check.current_neural_network_present
            && check.latest_neural_network_present);
    let apply_tactical_learning = check.current_policy_version_id.is_none()
        || newer_generation
        || different_head_at_same_generation
        || same_policy_newer_revision
        || (same_policy_tactical_weights_changed
            && (check.postgres_tactical_learning_authoritative
                || !check.local_tactical_evolved_since_pg_refresh))
        || (same_policy_version
            && (check.postgres_tactical_learning_authoritative
                || !check.local_tactical_evolved_since_pg_refresh));

    SoccerPostgresPolicyRefreshDecision {
        refresh_policy,
        apply_tactical_learning,
        same_policy_version,
        same_policy_newer_revision,
    }
}

pub fn soccer_should_refresh_postgres_for_new_sim(
    resume_artifact_present: bool,
    refresh_with_resume_artifact: bool,
) -> bool {
    !resume_artifact_present || refresh_with_resume_artifact
}

pub fn soccer_should_flush_postgres_policy_versions_for_new_sim(
    refresh_for_new_sims: bool,
    flush_policy_versions_before_new_sim: bool,
    pending_policy_versions: usize,
) -> bool {
    refresh_for_new_sims && flush_policy_versions_before_new_sim && pending_policy_versions > 0
}

pub fn soccer_postgres_new_sim_refresh_plan(
    resume_artifact_present: bool,
    refresh_with_resume_artifact: bool,
    flush_policy_versions_before_new_sim: bool,
    pending_policy_versions: usize,
    pending_async_policy_version_batches: usize,
) -> SoccerPostgresNewSimRefreshPlan {
    let refresh_from_postgres = soccer_should_refresh_postgres_for_new_sim(
        resume_artifact_present,
        refresh_with_resume_artifact,
    );
    let flush_pending_policy_versions = soccer_should_flush_postgres_policy_versions_for_new_sim(
        refresh_from_postgres,
        flush_policy_versions_before_new_sim,
        pending_policy_versions,
    );
    let wait_for_async_policy_versions =
        refresh_from_postgres && pending_async_policy_version_batches > 0;

    SoccerPostgresNewSimRefreshPlan {
        refresh_from_postgres,
        flush_pending_policy_versions,
        wait_for_async_policy_versions,
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct PolicyEntryKey {
    team: &'static str,
    entry_kind: SoccerLearningPolicyEntryKind,
    state_hash: String,
    state_json: String,
    action: String,
    target_fine_cell_id: i32,
    target_tactical_cell_id: i32,
    target_macro_cell_id: i32,
    target_root_cell_id: i32,
}

#[derive(Clone, Debug)]
struct EntryValue {
    state_key: SoccerQStateKey,
    action: String,
    value: f64,
    visits: u32,
    target_fine_cell_id: i32,
    target_tactical_cell_id: i32,
    target_macro_cell_id: i32,
    target_root_cell_id: i32,
}

#[derive(Clone, Debug)]
struct MergeAccumulator {
    state_key: SoccerQStateKey,
    action: String,
    weighted_value_sum: f64,
    effective_visits: f64,
    display_visits: u32,
    target_fine_cell_id: i32,
    target_tactical_cell_id: i32,
    target_macro_cell_id: i32,
    target_root_cell_id: i32,
}

pub fn soccer_learning_to_micros(value: f64) -> i64 {
    if !value.is_finite() {
        return 0;
    }
    (value * SOCCER_LEARNING_FIXED_SCALE as f64).round() as i64
}

pub fn soccer_learning_from_micros(value: i64) -> f64 {
    value as f64 / SOCCER_LEARNING_FIXED_SCALE as f64
}

pub const SOCCER_POLICY_ACTIVE_MAX_FITNESS_REGRESSION: f64 = 0.25;
const SOCCER_POLICY_ACTIVE_MAX_FITNESS_REGRESSION_ENV: &str =
    "SOCCER_POLICY_ACTIVE_MAX_FITNESS_REGRESSION";
static SOCCER_POLICY_ACTIVE_MAX_FITNESS_REGRESSION_OVERRIDE: OnceLock<f64> = OnceLock::new();

fn parse_soccer_policy_active_max_fitness_regression(raw: Option<&str>) -> Result<f64, String> {
    let Some(raw) = raw else {
        return Ok(SOCCER_POLICY_ACTIVE_MAX_FITNESS_REGRESSION);
    };
    let trimmed = raw.trim();
    let parsed = trimmed.parse::<f64>().map_err(|_| {
        format!("{SOCCER_POLICY_ACTIVE_MAX_FITNESS_REGRESSION_ENV} must be a finite non-negative number, got {raw:?}")
    })?;
    if !parsed.is_finite() || parsed < 0.0 {
        return Err(format!(
            "{SOCCER_POLICY_ACTIVE_MAX_FITNESS_REGRESSION_ENV} must be a finite non-negative number, got {raw:?}"
        ));
    }
    Ok(parsed)
}

pub fn soccer_policy_active_max_fitness_regression() -> f64 {
    *SOCCER_POLICY_ACTIVE_MAX_FITNESS_REGRESSION_OVERRIDE.get_or_init(|| {
        let raw = std::env::var(SOCCER_POLICY_ACTIVE_MAX_FITNESS_REGRESSION_ENV).ok();
        match parse_soccer_policy_active_max_fitness_regression(raw.as_deref()) {
            Ok(value) => value,
            Err(err) => {
                eprintln!(
                    "soccer-learning: {err}; using default {}",
                    SOCCER_POLICY_ACTIVE_MAX_FITNESS_REGRESSION
                );
                SOCCER_POLICY_ACTIVE_MAX_FITNESS_REGRESSION
            }
        }
    })
}

pub fn soccer_policy_version_insert_status_after_active_head(
    requested_status: &'static str,
    parent_policy_version_id: Option<&str>,
    generation: i32,
    fitness: f64,
    latest_active_policy_version_id: Option<&str>,
    latest_active_generation: Option<i32>,
    latest_active_fitness: Option<f64>,
    max_active_fitness_regression: f64,
) -> &'static str {
    if requested_status != SOCCER_POLICY_STATUS_ACTIVE {
        return requested_status;
    }
    let Some(latest_active_policy_version_id) = latest_active_policy_version_id else {
        return SOCCER_POLICY_STATUS_ACTIVE;
    };
    if parent_policy_version_id != Some(latest_active_policy_version_id) {
        return SOCCER_POLICY_STATUS_ARCHIVED;
    }
    if latest_active_generation.is_some_and(|active_generation| active_generation >= generation) {
        return SOCCER_POLICY_STATUS_ARCHIVED;
    }
    if let Some(active_fitness) = latest_active_fitness {
        let allowed_regression = max_active_fitness_regression.max(0.0);
        if !fitness.is_finite()
            || (active_fitness.is_finite() && fitness < active_fitness - allowed_regression)
        {
            return SOCCER_POLICY_STATUS_ARCHIVED;
        }
    }
    SOCCER_POLICY_STATUS_ACTIVE
}

pub fn soccer_team_label(team: Team) -> &'static str {
    match team {
        Team::Home => "home",
        Team::Away => "away",
    }
}

pub fn soccer_learning_run_score(summary: &MatchSummary) -> SoccerLearningRunScore {
    let home = soccer_learning_team_score(Team::Home, summary.score_home, summary.score_away);
    let away = soccer_learning_team_score(Team::Away, summary.score_away, summary.score_home);
    // The evolutionary search selects on `match_fitness`. The OLD definition,
    // `(home.fitness + away.fitness) * 0.5`, was a symmetric average whose goal-difference
    // terms cancel to `0.04 * total_goals` — no skill signal, so fitness sat flat for dozens
    // of generations. Both this branch and origin fixed it convergently; we keep origin's
    // principled structure — the WINNER's (non-cancelling) goal-diff fitness, plus a
    // normalised both-teams play-quality term (now enriched with the proxies for "more goals":
    // assists, completed crosses, consecutive forward passes — see
    // `soccer_learning_team_play_quality`), plus a decisive-margin bonus, all bounded.
    let winner_fitness = home.fitness.max(away.fitness);
    let play_quality = soccer_learning_match_play_quality(summary);
    let decisive_margin = (summary.score_home as i32 - summary.score_away as i32).abs() as f64;
    let match_fitness = (winner_fitness * SOCCER_MATCH_FITNESS_WINNER_WEIGHT
        + play_quality * SOCCER_MATCH_FITNESS_QUALITY_WEIGHT
        + decisive_margin.min(SOCCER_MATCH_FITNESS_MAX_DECISIVE_MARGIN)
            * SOCCER_MATCH_FITNESS_DECISIVE_MARGIN_WEIGHT)
        .clamp(SOCCER_MATCH_FITNESS_MIN, SOCCER_MATCH_FITNESS_MAX);
    SoccerLearningRunScore {
        home,
        away,
        match_fitness,
        match_fitness_micros: soccer_learning_to_micros(match_fitness),
    }
}

pub fn soccer_learning_team_score(
    team: Team,
    goals_for: u32,
    goals_against: u32,
) -> SoccerLearningTeamScore {
    let goal_diff = goals_for as i32 - goals_against as i32;
    let outcome = if goal_diff > 0 {
        SoccerLearningOutcome::Win
    } else if goal_diff == 0 {
        SoccerLearningOutcome::Draw
    } else {
        SoccerLearningOutcome::Loss
    };

    let margin = goal_diff.unsigned_abs() as f64;
    let base_weight = match outcome {
        SoccerLearningOutcome::Win => 1.0 + 0.22 * margin.min(6.0),
        SoccerLearningOutcome::Draw => 0.55,
        SoccerLearningOutcome::Loss => 0.20 / (1.0 + 0.35 * margin.max(1.0)),
    };
    let attacking_signal = (goals_for as f64 * 0.04).min(0.18);
    let defensive_signal = (1.0 / (1.0 + goals_against as f64 * 0.10)).clamp(0.65, 1.0);
    let merge_weight = (base_weight * defensive_signal + attacking_signal).clamp(0.035, 2.5);
    let fitness = goal_diff as f64 + goals_for as f64 * 0.20 - goals_against as f64 * 0.12;

    SoccerLearningTeamScore {
        team,
        goals_for,
        goals_against,
        goal_diff,
        outcome,
        merge_weight,
        merge_weight_micros: soccer_learning_to_micros(merge_weight),
        fitness,
        fitness_micros: soccer_learning_to_micros(fitness),
    }
}

fn soccer_learning_match_play_quality(summary: &MatchSummary) -> f64 {
    (soccer_learning_team_play_quality(Team::Home, summary)
        + soccer_learning_team_play_quality(Team::Away, summary))
        * 0.5
}

fn soccer_learning_team_play_quality(team: Team, summary: &MatchSummary) -> f64 {
    let stats = &summary.stats;
    let (
        pass_attempts,
        pass_completed,
        forward_completed,
        shots,
        shots_on_target,
        dribble_beats,
        loose_recoveries,
        upfield_progress,
        near_ball_progress,
        assists,
        crosses_completed,
        shots_after_pass,
        chain_forward_yards,
        chains_net_loss,
    ) = match team {
        Team::Home => (
            stats.passes_attempted_home,
            stats.passes_completed_home,
            stats.passes_completed_forward_home,
            stats.shots_home,
            stats.shots_on_target_home,
            stats.dribble_beats_home,
            stats.loose_ball_recoveries_home,
            stats.teamwork_upfield_progress_home,
            stats.teamwork_near_ball_progress_home,
            stats.assists_home,
            stats.crosses_completed_home,
            stats.shots_after_pass_home,
            stats.pass_chain_gain_yards_home,
            stats.pass_chains_net_loss_home,
        ),
        Team::Away => (
            stats.passes_attempted_away,
            stats.passes_completed_away,
            stats.passes_completed_forward_away,
            stats.shots_away,
            stats.shots_on_target_away,
            stats.dribble_beats_away,
            stats.loose_ball_recoveries_away,
            stats.teamwork_upfield_progress_away,
            stats.teamwork_near_ball_progress_away,
            stats.assists_away,
            stats.crosses_completed_away,
            stats.shots_after_pass_away,
            stats.pass_chain_gain_yards_away,
            stats.pass_chains_net_loss_away,
        ),
    };
    let pass_volume = soccer_learning_bounded_count(pass_attempts, 60.0);
    let pass_completion = soccer_learning_ratio(pass_completed, pass_attempts);
    let forward_share = soccer_learning_ratio(forward_completed, pass_completed);
    let shot_volume = soccer_learning_bounded_count(shots, 12.0);
    let shot_accuracy = soccer_learning_ratio(shots_on_target, shots);
    let dribble_quality = soccer_learning_bounded_count(dribble_beats, 8.0);
    let recovery_quality = soccer_learning_bounded_count(loose_recoveries, 16.0);
    let upfield_quality = soccer_learning_bounded_metric(upfield_progress, 80.0);
    let near_ball_quality = soccer_learning_bounded_metric(near_ball_progress, 80.0);
    // The user's explicit "good proxies for more goals": assists, completed crosses, worked
    // chances (a shot off >=1 pass) and consecutive FORWARD passing (net-forward chain yards,
    // penalised when chains went backwards).
    let assist_quality = soccer_learning_bounded_count(assists, 4.0);
    let cross_quality = soccer_learning_bounded_count(crosses_completed, 8.0);
    let worked_chance_quality = soccer_learning_bounded_count(shots_after_pass, 8.0);
    let chain_forward = soccer_learning_bounded_metric(chain_forward_yards, 120.0);
    let chain_backward_penalty = soccer_learning_bounded_count(chains_net_loss, 10.0);
    let chain_progress_quality = (chain_forward - chain_backward_penalty * 0.5).clamp(0.0, 1.0);

    (pass_volume * 0.05
        + pass_completion * 0.20
        + forward_share * 0.10
        + shot_volume * 0.08
        + shot_accuracy * 0.13
        + dribble_quality * 0.05
        + recovery_quality * 0.04
        + upfield_quality * 0.08
        + near_ball_quality * 0.03
        + assist_quality * 0.09
        + cross_quality * 0.03
        + worked_chance_quality * 0.04
        + chain_progress_quality * 0.08)
        .clamp(0.0, 1.0)
}

fn soccer_learning_ratio(numerator: u32, denominator: u32) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        (numerator as f64 / denominator as f64).clamp(0.0, 1.0)
    }
}

fn soccer_learning_bounded_count(count: u32, cap: f64) -> f64 {
    if cap <= 0.0 {
        0.0
    } else {
        (count as f64 / cap).clamp(0.0, 1.0)
    }
}

fn soccer_learning_bounded_metric(value: f64, cap: f64) -> f64 {
    if !value.is_finite() || !cap.is_finite() || cap <= 0.0 {
        0.0
    } else {
        (value.max(0.0) / cap).clamp(0.0, 1.0)
    }
}

pub fn soccer_policy_delta_entries(
    before: &SoccerTeamQPolicies,
    after: &SoccerTeamQPolicies,
    score: &SoccerLearningRunScore,
) -> SoccerLearningPolicyDelta {
    let mut entries = Vec::with_capacity(
        after
            .home
            .q_values
            .len()
            .saturating_add(after.home.target_values.len())
            .saturating_add(after.away.q_values.len())
            .saturating_add(after.away.target_values.len()),
    );
    collect_team_policy_delta(
        Team::Home,
        &before.home,
        &after.home,
        score.home.merge_weight,
        &mut entries,
    );
    collect_team_policy_delta(
        Team::Away,
        &before.away,
        &after.away,
        score.away.merge_weight,
        &mut entries,
    );
    SoccerLearningPolicyDelta { entries }
}

/// A frozen opponent in the league: a past policy snapshot plus its head-to-head
/// record against the current (training) policy.
#[derive(Clone, Debug)]
pub struct SoccerLeagueMember {
    pub generation: u32,
    pub policies: SoccerTeamQPolicies,
    /// Games this frozen member has played against the current policy.
    pub games: u32,
    /// Of those, how many the frozen member *won* (i.e. beat the current policy).
    pub wins_vs_current: u32,
}

impl SoccerLeagueMember {
    /// How well this member currently exploits the training policy, in (0, 1),
    /// Laplace-smoothed so a never-played member still gets sampled.
    pub fn exploit_rate(&self) -> f64 {
        (f64::from(self.wins_vs_current) + 1.0) / (f64::from(self.games) + 2.0)
    }
}

/// A bounded population of frozen past policies for **prioritized fictitious
/// self-play** (PFSP). Naive latest-vs-latest self-play overfits to the current
/// mirror image and stays exploitable; training against a *diverse* set of frozen
/// opponents — weighted toward the ones that currently beat us — drives the
/// policy toward a less-exploitable equilibrium. Pure data structure: the
/// orchestrator decides when to snapshot, who to play, and records outcomes.
#[derive(Clone, Debug, Default)]
pub struct SoccerPolicyLeague {
    members: Vec<SoccerLeagueMember>,
    capacity: usize,
}

impl SoccerPolicyLeague {
    pub fn new(capacity: usize) -> Self {
        SoccerPolicyLeague {
            members: Vec::new(),
            capacity: capacity.max(1),
        }
    }

    pub fn len(&self) -> usize {
        self.members.len()
    }

    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    pub fn members(&self) -> &[SoccerLeagueMember] {
        &self.members
    }

    /// Snapshot the current policy into the league. When over capacity, evict the
    /// *least*-exploiting member — keep the opponents that still trouble us.
    ///
    /// The eviction scan deliberately **excludes the member just inserted**: a
    /// fresh snapshot has `games = 0` so its Laplace-smoothed `exploit_rate` is
    /// exactly 0.5, which would make it the global minimum (and self-evict on the
    /// same call) once every incumbent already exploits the current policy at
    /// > 0.5 — the healthy PFSP steady state. Without this guard the league would
    /// silently freeze on its earliest generations and never admit newer policies.
    pub fn insert(&mut self, generation: u32, policies: SoccerTeamQPolicies) {
        self.members.push(SoccerLeagueMember {
            generation,
            policies,
            games: 0,
            wins_vs_current: 0,
        });
        // `insert` adds exactly one member, so len exceeds capacity by at most 1.
        while self.members.len() > self.capacity {
            let last = self.members.len() - 1;
            let evict = self.members[..last]
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.exploit_rate().total_cmp(&b.1.exploit_rate()))
                .map(|(index, _)| index);
            match evict {
                Some(index) => {
                    self.members.remove(index);
                }
                None => break,
            }
        }
    }

    /// PFSP opponent sampling: probability ∝ `exploit_rate`, so the opponents that
    /// beat the current policy are sampled more often and training focuses on its
    /// weaknesses. Deterministic given `rng`. Returns an index into `members()`.
    pub fn sample_opponent_index(&self, rng: &mut SeededRandom) -> Option<usize> {
        if self.members.is_empty() {
            return None;
        }
        let weights: Vec<f64> = self
            .members
            .iter()
            .map(|member| member.exploit_rate().max(1e-3))
            .collect();
        let total: f64 = weights.iter().sum();
        if !(total > 0.0) {
            return Some(0);
        }
        let mut roll = rng.next_float() * total;
        for (index, weight) in weights.iter().enumerate() {
            roll -= weight;
            if roll <= 0.0 {
                return Some(index);
            }
        }
        Some(self.members.len() - 1)
    }

    /// Record a finished game against league `member_index`.
    pub fn record_result(&mut self, member_index: usize, member_beat_current: bool) {
        if let Some(member) = self.members.get_mut(member_index) {
            member.games = member.games.saturating_add(1);
            if member_beat_current {
                member.wins_vs_current = member.wins_vs_current.saturating_add(1);
            }
        }
    }

    /// Exploitability proxy in (0, 1): the worst-case exploit rate across the
    /// league — how badly the best-known past opponent beats the current policy.
    /// Trends down as the policy becomes robust to its own history.
    pub fn exploitability(&self) -> f64 {
        self.members
            .iter()
            .map(SoccerLeagueMember::exploit_rate)
            .fold(0.0, f64::max)
    }
}

/// Build the policy set for a league game: the **current** policy takes one side
/// and the frozen league `member` the other, so the current policy trains against
/// a fixed past opponent rather than its live mirror. `current_plays_home`
/// chooses the side the learner occupies.
pub fn soccer_league_matchup_policies(
    current: &SoccerTeamQPolicies,
    member: &SoccerLeagueMember,
    current_plays_home: bool,
) -> SoccerTeamQPolicies {
    if current_plays_home {
        SoccerTeamQPolicies {
            home: current.home.clone(),
            away: member.policies.away.clone(),
        }
    } else {
        SoccerTeamQPolicies {
            home: member.policies.home.clone(),
            away: current.away.clone(),
        }
    }
}

pub fn merge_soccer_policy_deltas(
    base: &SoccerTeamQPolicies,
    deltas: &[SoccerLearningPolicyDelta],
    prior_weight: f64,
) -> Result<SoccerTeamQPolicies, String> {
    let mut action_accumulators = BTreeMap::<PolicyEntryKey, MergeAccumulator>::new();
    let mut target_accumulators = BTreeMap::<PolicyEntryKey, MergeAccumulator>::new();

    seed_policy_accumulators(
        Team::Home,
        &base.home,
        prior_weight,
        &mut action_accumulators,
        &mut target_accumulators,
    );
    seed_policy_accumulators(
        Team::Away,
        &base.away,
        prior_weight,
        &mut action_accumulators,
        &mut target_accumulators,
    );

    for delta in deltas {
        for entry in &delta.entries {
            let accumulator = match entry.entry_kind {
                SoccerLearningPolicyEntryKind::Action => &mut action_accumulators,
                SoccerLearningPolicyEntryKind::Target => &mut target_accumulators,
            };
            let key = policy_delta_key(entry);
            let effective_visits = entry.effective_visit_weight.max(0.0);
            if effective_visits <= 0.0 {
                continue;
            }
            let item = accumulator.entry(key).or_insert_with(|| MergeAccumulator {
                state_key: entry.state_key.clone(),
                action: entry.action.clone(),
                weighted_value_sum: 0.0,
                effective_visits: 0.0,
                display_visits: 0,
                target_fine_cell_id: entry.target_fine_cell_id,
                target_tactical_cell_id: entry.target_tactical_cell_id,
                target_macro_cell_id: entry.target_macro_cell_id,
                target_root_cell_id: entry.target_root_cell_id,
            });
            item.weighted_value_sum += entry.after_value * effective_visits;
            item.effective_visits += effective_visits;
            item.display_visits = item
                .display_visits
                .saturating_add(effective_visits.round().max(1.0) as u32);
        }
    }

    build_policies_from_accumulators(
        base.home.options.clone(),
        base.away.options.clone(),
        action_accumulators,
        target_accumulators,
    )
}

pub fn evolve_soccer_team_policies(
    parents: &[(&SoccerTeamQPolicies, f64)],
    options: SoccerEvolutionOptions,
) -> Result<SoccerTeamQPolicies, String> {
    let options = normalize_soccer_evolution_options_for_learning_search(options)?;
    let policy_pressure = soccer_policy_search_pressure(parents);
    let options = adapt_soccer_evolution_options_for_policy_search(options, policy_pressure);
    let population_size = options.population_size;
    let mut best = None::<(SoccerTeamQPolicies, f64)>;
    for candidate_index in 0..population_size {
        let mut candidate_options = options;
        candidate_options.population_size = 1;
        candidate_options.seed =
            candidate_seed(options.seed, candidate_index, 0x51f1_7ea9_8d12_0b1d);
        let candidate =
            evolve_soccer_team_policy_candidate(parents, candidate_options, policy_pressure)?;
        let score = soccer_team_policy_search_score(&candidate);
        if best
            .as_ref()
            .map_or(true, |(_, best_score)| score > *best_score)
        {
            best = Some((candidate, score));
        }
    }
    best.map(|(policy, _)| policy)
        .ok_or_else(|| "at least one parent policy is required".to_string())
}

fn evolve_soccer_team_policy_candidate(
    parents: &[(&SoccerTeamQPolicies, f64)],
    options: SoccerEvolutionOptions,
    policy_pressure: f64,
) -> Result<SoccerTeamQPolicies, String> {
    let Some((first_parent, _)) = parents.first() else {
        return Err("at least one parent policy is required".to_string());
    };
    let mut action_accumulators = BTreeMap::<PolicyEntryKey, MergeAccumulator>::new();
    let mut target_accumulators = BTreeMap::<PolicyEntryKey, MergeAccumulator>::new();

    for (policy, fitness) in parents {
        let weight = fitness.max(options.elite_weight_floor).max(0.0);
        seed_policy_accumulators(
            Team::Home,
            &policy.home,
            weight,
            &mut action_accumulators,
            &mut target_accumulators,
        );
        seed_policy_accumulators(
            Team::Away,
            &policy.away,
            weight,
            &mut action_accumulators,
            &mut target_accumulators,
        );
    }

    let mut rng = DeterministicRng::new(options.seed);
    let parent_action_maps = parents
        .iter()
        .map(|(policy, _)| policy_action_entry_map(policy))
        .collect::<Vec<_>>();
    let parent_target_maps = parents
        .iter()
        .map(|(policy, _)| policy_target_entry_map(policy))
        .collect::<Vec<_>>();
    crossover_accumulators(
        &mut action_accumulators,
        &parent_action_maps,
        &mut rng,
        options,
    );
    crossover_accumulators(
        &mut target_accumulators,
        &parent_target_maps,
        &mut rng,
        options,
    );
    inject_policy_plateau_novelty_actions(
        &mut action_accumulators,
        &mut rng,
        options,
        policy_pressure,
    );
    explore_accumulators(&mut action_accumulators, &mut rng, options);
    explore_accumulators(&mut target_accumulators, &mut rng, options);
    mutate_accumulators(&mut action_accumulators, &mut rng, options);
    mutate_accumulators(&mut target_accumulators, &mut rng, options);

    build_policies_from_accumulators(
        first_parent.home.options.clone(),
        first_parent.away.options.clone(),
        action_accumulators,
        target_accumulators,
    )
}

fn soccer_policy_search_pressure(parents: &[(&SoccerTeamQPolicies, f64)]) -> f64 {
    let finite_fitness = parents
        .iter()
        .filter_map(|(_, fitness)| fitness.is_finite().then_some(*fitness))
        .collect::<Vec<_>>();
    let sample_fit = if finite_fitness.is_empty() {
        0.35
    } else {
        (finite_fitness.len() as f64 / 4.0).clamp(0.35, 1.0)
    };
    let (best, worst) = finite_fitness.iter().fold(
        (f64::NEG_INFINITY, f64::INFINITY),
        |(best, worst), fitness| (best.max(*fitness), worst.min(*fitness)),
    );
    let plateau = if finite_fitness.len() >= 2 {
        (1.0 - (best - worst).abs() / SOCCER_POLICY_PLATEAU_FITNESS_SPREAD).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let low_ceiling = if best.is_finite() {
        (1.0 - best.max(0.0) / SOCCER_POLICY_LOW_CEILING_FITNESS).clamp(0.0, 1.0)
    } else {
        0.35
    };
    let coverage_gap = (1.0 - soccer_policy_parent_action_coverage(parents)).clamp(0.0, 1.0);

    ((plateau * 0.52 + low_ceiling * 0.20 + coverage_gap * 0.28) * sample_fit).clamp(0.0, 1.0)
}

fn soccer_policy_parent_action_coverage(parents: &[(&SoccerTeamQPolicies, f64)]) -> f64 {
    let mut families = HashSet::<String>::new();
    for (policy, _) in parents {
        for entry in policy
            .home
            .entries()
            .into_iter()
            .chain(policy.away.entries())
        {
            families.insert(entry.action);
        }
    }
    (families.len() as f64 / soccer_policy_known_action_count() as f64).clamp(0.0, 1.0)
}

fn soccer_policy_known_action_count() -> usize {
    let mut actions = HashSet::<&'static str>::new();
    for family in [
        SOCCER_POLICY_DRIBBLE_NOVELTY_ACTIONS,
        SOCCER_POLICY_PASS_NOVELTY_ACTIONS,
        SOCCER_POLICY_SHOT_NOVELTY_ACTIONS,
        SOCCER_POLICY_DEFENSE_NOVELTY_ACTIONS,
        SOCCER_POLICY_SUPPORT_NOVELTY_ACTIONS,
    ] {
        actions.extend(family.iter().copied());
    }
    actions.len().max(1)
}

pub fn adapt_soccer_evolution_options_for_policy_search(
    mut options: SoccerEvolutionOptions,
    pressure: f64,
) -> SoccerEvolutionOptions {
    options.population_size =
        soccer_evolution_population_size_for_learning_run(options.population_size);
    let pressure = pressure.clamp(0.0, 1.0);
    if pressure <= 1e-12 {
        return options;
    }
    let search_enabled = options.population_size > 1
        || options.mutation_rate > 0.0
        || options.mutation_scale > 0.0
        || options.crossover_rate > 0.0
        || options.exploration_rate > 0.0
        || options.exploration_scale > 0.0;
    if !search_enabled {
        return options;
    }

    if options.crossover_rate > 0.0 {
        options.crossover_rate = (options.crossover_rate + pressure * 0.10).clamp(0.0, 0.90);
    }
    if options.mutation_rate > 0.0 || options.mutation_scale > 0.0 {
        options.mutation_rate = (options.mutation_rate + pressure * 0.05).clamp(0.0, 0.55);
        options.mutation_scale = (options.mutation_scale * (1.0 + pressure * 0.70))
            .max(options.mutation_scale)
            .min(1.20);
    }
    if options.exploration_rate > 0.0 || options.exploration_scale > 0.0 {
        options.exploration_rate = (options.exploration_rate + pressure * 0.12).clamp(0.0, 0.65);
        options.exploration_scale = (options.exploration_scale * (1.0 + pressure * 0.85))
            .max(options.exploration_scale)
            .min(1.60);
    }
    options.elite_weight_floor = options.elite_weight_floor.max(pressure * 0.04).min(0.16);
    if options.population_size > 1 {
        let extra_candidates = ((options.population_size as f64) * pressure * 1.5).ceil() as usize;
        let cap = options
            .population_size
            .saturating_mul(3)
            .min(soccer_policy_search_max_adapted_population());
        options.population_size = options
            .population_size
            .saturating_add(extra_candidates)
            .min(cap)
            .max(2);
    }
    options
}

fn soccer_team_policy_search_score(policy: &SoccerTeamQPolicies) -> f64 {
    let (home_score, home_weight) = soccer_q_policy_search_score(&policy.home);
    let (away_score, away_weight) = soccer_q_policy_search_score(&policy.away);
    let weight = home_weight + away_weight;
    if weight <= 0.0 {
        0.0
    } else {
        (home_score + away_score) / weight
    }
}

#[derive(Default)]
struct SoccerPolicyStateSearchStats {
    total_weight: f64,
    weighted_value_sum: f64,
    best_value: f64,
    best_visits: u32,
    actions: HashSet<String>,
    technical_actions: usize,
}

fn soccer_q_policy_search_score(policy: &SoccerQPolicy) -> (f64, f64) {
    let entries = policy.entries();
    let target_entries = policy.target_entries();
    if entries.is_empty() && target_entries.is_empty() {
        return (0.0, 0.0);
    }

    let mut states = HashMap::<SoccerQStateKey, SoccerPolicyStateSearchStats>::new();
    for entry in entries {
        let value = entry.value.clamp(-120.0, 120.0);
        let visits = entry.visits.max(1);
        let weight = f64::from(visits).sqrt();
        let stats = states.entry(entry.state).or_default();
        stats.total_weight += weight;
        stats.weighted_value_sum += value * weight;
        if stats.actions.is_empty() || value > stats.best_value {
            stats.best_value = value;
            stats.best_visits = visits;
        }
        if soccer_policy_action_is_technical_execution(&entry.action) {
            stats.technical_actions += 1;
        }
        stats.actions.insert(entry.action);
    }

    let mut weighted_score = 0.0;
    let mut total_weight = 0.0;
    for stats in states.values() {
        if stats.total_weight <= 0.0 {
            continue;
        }
        let mean_value = stats.weighted_value_sum / stats.total_weight;
        let dp_residual = (stats.best_value - mean_value).max(0.0);
        let action_coverage =
            ((stats.actions.len().saturating_sub(1)) as f64 / 6.0).clamp(0.0, 1.0);
        let technical_coverage = (stats.technical_actions as f64 / 4.0).clamp(0.0, 1.0);
        let state_score = stats.best_value
            + action_coverage * SOCCER_POLICY_STATE_ACTION_COVERAGE_WEIGHT
            + technical_coverage * SOCCER_POLICY_TECHNICAL_COVERAGE_WEIGHT
            - dp_residual.min(8.0) * SOCCER_POLICY_DP_RESIDUAL_PENALTY_WEIGHT;
        let state_weight = stats.total_weight.max(f64::from(stats.best_visits).sqrt());
        weighted_score += state_score * state_weight;
        total_weight += state_weight;
    }

    let mut target_actions = HashSet::<String>::new();
    let mut target_cells = HashSet::<(usize, usize, usize, usize)>::new();
    let mut target_weighted_score = 0.0;
    let mut target_weight = 0.0;
    for entry in target_entries {
        let weight = f64::from(entry.visits.max(1)).sqrt() * 0.45;
        target_weighted_score += entry.value.clamp(-120.0, 120.0) * weight;
        target_weight += weight;
        target_actions.insert(entry.action);
        target_cells.insert((
            entry.target_fine_cell_id,
            entry.target_tactical_cell_id,
            entry.target_macro_cell_id,
            entry.target_root_cell_id,
        ));
    }
    if target_weight > 0.0 {
        let target_coverage = ((target_actions.len() as f64 / 5.0).clamp(0.0, 1.0) * 0.55)
            + ((target_cells.len() as f64 / 24.0).clamp(0.0, 1.0) * 0.45);
        weighted_score += target_weighted_score
            + target_coverage * SOCCER_POLICY_TARGET_COVERAGE_WEIGHT * target_weight;
        total_weight += target_weight;
    }

    (weighted_score, total_weight)
}

fn soccer_policy_action_is_technical_execution(action: &str) -> bool {
    SOCCER_POLICY_DRIBBLE_NOVELTY_ACTIONS.contains(&action)
        || SOCCER_POLICY_PASS_NOVELTY_ACTIONS.contains(&action)
        || SOCCER_POLICY_SHOT_NOVELTY_ACTIONS.contains(&action)
}

pub fn evolve_soccer_tactical_learning_weights(
    base: &SoccerTacticalLearningWeights,
    parents: &[(&SoccerTacticalLearningSummary, f64)],
    options: SoccerEvolutionOptions,
) -> SoccerTacticalLearningWeights {
    let Ok(options) = normalize_soccer_evolution_options_for_learning_search(options) else {
        return base.clone();
    };
    let Some(weighted_summary) = weighted_tactical_evolution_summary(parents, options) else {
        return base.clone();
    };
    let search_options =
        adapt_soccer_evolution_options_for_tactical_search(options, &weighted_summary);
    let population_size = search_options.population_size.max(1);
    let mut best = base.clone();
    let mut best_score = soccer_tactical_weight_search_score(&best, &weighted_summary);
    search_soccer_tactical_strategy_candidates(base, &weighted_summary, |candidate| {
        keep_best_soccer_tactical_candidate(
            &mut best,
            &mut best_score,
            candidate,
            &weighted_summary,
        );
    });
    for candidate_index in 0..population_size {
        let mut candidate_options = search_options;
        candidate_options.population_size = 1;
        candidate_options.seed =
            candidate_seed(search_options.seed, candidate_index, 0x7ac7_1ca1_5eed_c0de);
        let candidate =
            evolve_soccer_tactical_learning_candidate(base, &weighted_summary, candidate_options);
        let score = soccer_tactical_weight_search_score(&candidate, &weighted_summary);
        if score > best_score {
            best_score = score;
            best = candidate;
        }
    }
    best
}

pub fn evolve_soccer_tactical_learning_weights_from_genomes(
    base: &SoccerTacticalLearningWeights,
    parents: &[SoccerTacticalLearningGenomeParent<'_>],
    options: SoccerEvolutionOptions,
) -> SoccerTacticalLearningWeights {
    let Ok(options) = normalize_soccer_evolution_options_for_learning_search(options) else {
        return base.clone();
    };
    let summary_parents = parents
        .iter()
        .map(|parent| (parent.summary, parent.fitness))
        .collect::<Vec<_>>();
    let Some(weighted_summary) = weighted_tactical_evolution_summary(&summary_parents, options)
    else {
        return base.clone();
    };
    let search_options =
        adapt_soccer_evolution_options_for_tactical_search(options, &weighted_summary);
    let population_size = search_options.population_size.max(1);
    let mut best = base.clone();
    let mut best_score = soccer_tactical_weight_search_score(&best, &weighted_summary);
    search_soccer_tactical_strategy_candidates(base, &weighted_summary, |candidate| {
        keep_best_soccer_tactical_candidate(
            &mut best,
            &mut best_score,
            candidate,
            &weighted_summary,
        );
    });

    for parent in parents {
        if !parent.fitness.is_finite() {
            continue;
        }
        let candidate = clamp_soccer_tactical_learning_weights(parent.weights);
        let score = soccer_tactical_weight_search_score(&candidate, &weighted_summary);
        if score > best_score {
            best_score = score;
            best = candidate;
        }
    }

    search_soccer_tactical_genome_blend_candidates(
        base,
        parents,
        search_options,
        &weighted_summary,
        |candidate| {
            keep_best_soccer_tactical_candidate(
                &mut best,
                &mut best_score,
                candidate,
                &weighted_summary,
            );
        },
    );

    for candidate_index in 0..population_size {
        let mut candidate_options = search_options;
        candidate_options.population_size = 1;
        candidate_options.seed =
            candidate_seed(search_options.seed, candidate_index, 0x7ac7_1ca1_5eed_c0de);
        let mut rng = DeterministicRng::new(candidate_options.seed ^ 0x671e_d1ce_5eed_ba5e);
        let candidate_base =
            crossover_soccer_tactical_learning_weights(base, parents, search_options, &mut rng);
        let candidate = evolve_soccer_tactical_learning_candidate(
            &candidate_base,
            &weighted_summary,
            candidate_options,
        );
        let score = soccer_tactical_weight_search_score(&candidate, &weighted_summary);
        if score > best_score {
            best_score = score;
            best = candidate;
        }
    }
    best
}

fn search_soccer_tactical_genome_blend_candidates<F>(
    base: &SoccerTacticalLearningWeights,
    parents: &[SoccerTacticalLearningGenomeParent<'_>],
    options: SoccerEvolutionOptions,
    weighted_summary: &SoccerTacticalLearningSummary,
    mut visit: F,
) where
    F: FnMut(SoccerTacticalLearningWeights),
{
    let blend_enabled =
        parents.len() >= 2 && (options.population_size > 1 || options.crossover_rate > 0.0);
    if !blend_enabled {
        return;
    }

    let Some(centroid) = weighted_soccer_tactical_genome_centroid(parents, options) else {
        return;
    };
    visit(centroid.clone());

    let pressure = soccer_tactical_search_pressure(weighted_summary);
    if pressure <= 1e-12 {
        return;
    }
    let scale = 1.0 + pressure * 0.75;
    visit(extrapolate_soccer_tactical_learning_weights(
        base, &centroid, scale,
    ));
    search_soccer_tactical_genome_archetype_candidates(
        &centroid,
        weighted_summary,
        pressure,
        &mut visit,
    );
    search_soccer_tactical_gp_program_candidates(
        base,
        &centroid,
        weighted_summary,
        pressure,
        &mut visit,
    );
    search_soccer_tactical_mutated_gp_program_candidates(
        base,
        &centroid,
        weighted_summary,
        pressure,
        options,
        &mut visit,
    );
    search_soccer_tactical_novelty_immigrant_candidates(
        base,
        &centroid,
        weighted_summary,
        pressure,
        options,
        &mut visit,
    );
}

fn search_soccer_tactical_genome_archetype_candidates<F>(
    centroid: &SoccerTacticalLearningWeights,
    weighted_summary: &SoccerTacticalLearningSummary,
    pressure: f64,
    visit: &mut F,
) where
    F: FnMut(SoccerTacticalLearningWeights),
{
    let attack_width_gap = (1.0 - weighted_summary.mean_attack_width_score).clamp(0.0, 1.0);
    let attack_flank_gap = (1.0 - weighted_summary.mean_attack_flank_lane_score).clamp(0.0, 1.0);
    let attack_spacing_gap = (1.0 - weighted_summary.mean_attack_spacing_score).clamp(0.0, 1.0);
    let defense_contract_gap = (1.0 - weighted_summary.mean_defense_contract_score).clamp(0.0, 1.0);
    let defense_spacing_gap = (1.0 - weighted_summary.mean_defense_spacing_score).clamp(0.0, 1.0);
    let defense_ball_gap = (1.0 - weighted_summary.mean_defense_ball_gap_score).clamp(0.0, 1.0);
    let press_gap = (1.0 - weighted_summary.mean_defense_role_press_score).clamp(0.0, 1.0);
    let attack_pressure =
        (attack_width_gap * 0.48 + attack_flank_gap * 0.40 + attack_spacing_gap * 0.12)
            .clamp(0.0, 1.0);
    let defense_pressure = (defense_contract_gap * 0.50
        + defense_spacing_gap * 0.18
        + defense_ball_gap * 0.16
        + press_gap * 0.16)
        .clamp(0.0, 1.0);

    if attack_pressure > 1e-12 {
        let mut flank_switch = centroid.clone();
        flank_switch.attack_width_delta_weight += attack_width_gap * (0.22 + pressure * 0.12);
        flank_switch.attack_width_score_weight += attack_width_gap * 0.10;
        flank_switch.attack_flank_lane_weight += attack_flank_gap * (0.34 + pressure * 0.18);
        flank_switch.attack_spacing_delta_weight +=
            attack_spacing_gap * (0.10 + attack_pressure * 0.08);
        flank_switch.attack_spacing_score_weight += attack_spacing_gap * 0.05;
        flank_switch.goal_entry_pass_learning_weight += attack_pressure * (0.10 + pressure * 0.05);
        flank_switch.pass_target_ranking_learning_weight +=
            attack_width_gap.max(attack_flank_gap) * 0.08;
        visit(clamp_soccer_tactical_learning_weights(&flank_switch));

        let mut support_spread = centroid.clone();
        support_spread.attack_spacing_delta_weight += attack_spacing_gap * (0.22 + pressure * 0.08);
        support_spread.attack_spacing_score_weight += attack_spacing_gap * 0.10;
        support_spread.attack_width_delta_weight += attack_width_gap * 0.16;
        support_spread.attack_width_score_weight += attack_width_gap * 0.06;
        support_spread.attack_flank_lane_weight += attack_flank_gap * 0.20;
        support_spread.pressure_release_learning_weight += attack_spacing_gap * 0.08;
        support_spread.pass_target_ranking_learning_weight += attack_pressure * 0.06;
        visit(clamp_soccer_tactical_learning_weights(&support_spread));
    }

    if defense_pressure > 1e-12 {
        let mut compact_block = centroid.clone();
        compact_block.defense_contract_delta_weight +=
            defense_contract_gap * (0.26 + pressure * 0.14);
        compact_block.defense_compactness_score_weight +=
            defense_contract_gap * (0.18 + defense_pressure * 0.08);
        compact_block.defense_spacing_delta_weight += defense_spacing_gap * 0.08;
        compact_block.defense_ball_depth_score_weight +=
            defense_ball_gap * (0.08 + defense_pressure * 0.04);
        visit(clamp_soccer_tactical_learning_weights(&compact_block));

        let mut press_contract = centroid.clone();
        press_contract.defense_contract_delta_weight +=
            defense_contract_gap * (0.18 + press_gap * 0.10);
        press_contract.defender_midfielder_press_weight +=
            press_gap * (0.09 + defense_pressure * 0.04);
        press_contract.midfielder_press_weight += press_gap * (0.08 + defense_pressure * 0.03);
        press_contract.defense_ball_depth_score_weight += defense_ball_gap * 0.05;
        press_contract.defensive_line_press_learning_weight +=
            press_gap * (0.10 + defense_pressure * 0.06);
        visit(clamp_soccer_tactical_learning_weights(&press_contract));
    }

    if attack_pressure > 1e-12 && defense_pressure > 1e-12 {
        let mut two_phase_shape = centroid.clone();
        two_phase_shape.attack_width_delta_weight += attack_width_gap * 0.18;
        two_phase_shape.attack_width_score_weight += attack_width_gap * 0.06;
        two_phase_shape.attack_flank_lane_weight += attack_flank_gap * (0.26 + pressure * 0.12);
        two_phase_shape.attack_spacing_delta_weight += attack_spacing_gap * 0.08;
        two_phase_shape.goal_entry_pass_learning_weight += attack_pressure * 0.08;
        two_phase_shape.pressure_release_learning_weight += attack_spacing_gap * 0.05;
        two_phase_shape.defense_contract_delta_weight +=
            defense_contract_gap * (0.21 + pressure * 0.11);
        two_phase_shape.defense_compactness_score_weight += defense_contract_gap * 0.14;
        two_phase_shape.defense_spacing_delta_weight += defense_spacing_gap * 0.06;
        two_phase_shape.defensive_line_press_learning_weight += press_gap * 0.09;
        visit(clamp_soccer_tactical_learning_weights(&two_phase_shape));
    }
}

#[derive(Clone, Copy, Debug)]
enum SoccerTacticalGpGene {
    AttackSpacingDelta,
    AttackSpacingScore,
    AttackWidthDelta,
    AttackWidthScore,
    AttackFlankLane,
    ShotChoice,
    GoalEntryPass,
    PressureRelease,
    PassTargetRanking,
    DefenseSpacingDelta,
    DefenseSpacingScore,
    DefenseContractDelta,
    DefenseCompactnessScore,
    DefenseBallDepthScore,
    DefenderMidfielderPress,
    MidfielderPress,
    DefensiveLinePress,
}

#[derive(Clone, Copy, Debug)]
enum SoccerTacticalGpSignal {
    AttackWidthGap,
    AttackFlankGap,
    AttackSpacingGap,
    DefenseContractGap,
    DefenseSpacingGap,
    DefenseBallGap,
    PressGap,
    Pressure,
}

#[derive(Clone, Copy, Debug)]
enum SoccerTacticalGpExpr {
    Signal(SoccerTacticalGpSignal),
    Product(SoccerTacticalGpSignal, SoccerTacticalGpSignal),
    Blend(SoccerTacticalGpSignal, SoccerTacticalGpSignal, f64),
    Max(SoccerTacticalGpSignal, SoccerTacticalGpSignal),
}

#[derive(Clone, Copy, Debug)]
struct SoccerTacticalGpInstruction {
    gene: SoccerTacticalGpGene,
    expr: SoccerTacticalGpExpr,
    scale: f64,
}

const ATTACK_SOCCER_TACTICAL_GP_GENES: [SoccerTacticalGpGene; 9] = [
    SoccerTacticalGpGene::AttackSpacingDelta,
    SoccerTacticalGpGene::AttackSpacingScore,
    SoccerTacticalGpGene::AttackWidthDelta,
    SoccerTacticalGpGene::AttackWidthScore,
    SoccerTacticalGpGene::AttackFlankLane,
    SoccerTacticalGpGene::ShotChoice,
    SoccerTacticalGpGene::GoalEntryPass,
    SoccerTacticalGpGene::PressureRelease,
    SoccerTacticalGpGene::PassTargetRanking,
];

const DEFENSE_SOCCER_TACTICAL_GP_GENES: [SoccerTacticalGpGene; 8] = [
    SoccerTacticalGpGene::DefenseSpacingDelta,
    SoccerTacticalGpGene::DefenseSpacingScore,
    SoccerTacticalGpGene::DefenseContractDelta,
    SoccerTacticalGpGene::DefenseCompactnessScore,
    SoccerTacticalGpGene::DefenseBallDepthScore,
    SoccerTacticalGpGene::DefenderMidfielderPress,
    SoccerTacticalGpGene::MidfielderPress,
    SoccerTacticalGpGene::DefensiveLinePress,
];

const ATTACK_SOCCER_TACTICAL_GP_SIGNALS: [SoccerTacticalGpSignal; 4] = [
    SoccerTacticalGpSignal::AttackWidthGap,
    SoccerTacticalGpSignal::AttackFlankGap,
    SoccerTacticalGpSignal::AttackSpacingGap,
    SoccerTacticalGpSignal::Pressure,
];

const DEFENSE_SOCCER_TACTICAL_GP_SIGNALS: [SoccerTacticalGpSignal; 5] = [
    SoccerTacticalGpSignal::DefenseContractGap,
    SoccerTacticalGpSignal::DefenseSpacingGap,
    SoccerTacticalGpSignal::DefenseBallGap,
    SoccerTacticalGpSignal::PressGap,
    SoccerTacticalGpSignal::Pressure,
];

const WIDE_FLANK_GP_PROGRAM: [SoccerTacticalGpInstruction; 4] = [
    SoccerTacticalGpInstruction {
        gene: SoccerTacticalGpGene::AttackWidthDelta,
        expr: SoccerTacticalGpExpr::Blend(
            SoccerTacticalGpSignal::AttackWidthGap,
            SoccerTacticalGpSignal::AttackFlankGap,
            0.55,
        ),
        scale: 0.30,
    },
    SoccerTacticalGpInstruction {
        gene: SoccerTacticalGpGene::AttackWidthScore,
        expr: SoccerTacticalGpExpr::Signal(SoccerTacticalGpSignal::AttackWidthGap),
        scale: 0.10,
    },
    SoccerTacticalGpInstruction {
        gene: SoccerTacticalGpGene::AttackFlankLane,
        expr: SoccerTacticalGpExpr::Max(
            SoccerTacticalGpSignal::AttackFlankGap,
            SoccerTacticalGpSignal::AttackWidthGap,
        ),
        scale: 0.42,
    },
    SoccerTacticalGpInstruction {
        gene: SoccerTacticalGpGene::AttackSpacingDelta,
        expr: SoccerTacticalGpExpr::Product(
            SoccerTacticalGpSignal::AttackSpacingGap,
            SoccerTacticalGpSignal::Pressure,
        ),
        scale: 0.24,
    },
];

const SUPPORT_SPACING_GP_PROGRAM: [SoccerTacticalGpInstruction; 4] = [
    SoccerTacticalGpInstruction {
        gene: SoccerTacticalGpGene::AttackSpacingDelta,
        expr: SoccerTacticalGpExpr::Blend(
            SoccerTacticalGpSignal::AttackSpacingGap,
            SoccerTacticalGpSignal::AttackWidthGap,
            0.70,
        ),
        scale: 0.32,
    },
    SoccerTacticalGpInstruction {
        gene: SoccerTacticalGpGene::AttackSpacingScore,
        expr: SoccerTacticalGpExpr::Signal(SoccerTacticalGpSignal::AttackSpacingGap),
        scale: 0.14,
    },
    SoccerTacticalGpInstruction {
        gene: SoccerTacticalGpGene::AttackWidthDelta,
        expr: SoccerTacticalGpExpr::Product(
            SoccerTacticalGpSignal::AttackWidthGap,
            SoccerTacticalGpSignal::Pressure,
        ),
        scale: 0.18,
    },
    SoccerTacticalGpInstruction {
        gene: SoccerTacticalGpGene::AttackFlankLane,
        expr: SoccerTacticalGpExpr::Product(
            SoccerTacticalGpSignal::AttackFlankGap,
            SoccerTacticalGpSignal::Pressure,
        ),
        scale: 0.20,
    },
];

const COMPACT_BLOCK_GP_PROGRAM: [SoccerTacticalGpInstruction; 4] = [
    SoccerTacticalGpInstruction {
        gene: SoccerTacticalGpGene::DefenseContractDelta,
        expr: SoccerTacticalGpExpr::Blend(
            SoccerTacticalGpSignal::DefenseContractGap,
            SoccerTacticalGpSignal::DefenseBallGap,
            0.70,
        ),
        scale: 0.44,
    },
    SoccerTacticalGpInstruction {
        gene: SoccerTacticalGpGene::DefenseCompactnessScore,
        expr: SoccerTacticalGpExpr::Signal(SoccerTacticalGpSignal::DefenseContractGap),
        scale: 0.30,
    },
    SoccerTacticalGpInstruction {
        gene: SoccerTacticalGpGene::DefenseSpacingDelta,
        expr: SoccerTacticalGpExpr::Product(
            SoccerTacticalGpSignal::DefenseSpacingGap,
            SoccerTacticalGpSignal::Pressure,
        ),
        scale: 0.14,
    },
    SoccerTacticalGpInstruction {
        gene: SoccerTacticalGpGene::DefenseBallDepthScore,
        expr: SoccerTacticalGpExpr::Product(
            SoccerTacticalGpSignal::DefenseBallGap,
            SoccerTacticalGpSignal::Pressure,
        ),
        scale: 0.20,
    },
];

const PRESS_FUNNEL_GP_PROGRAM: [SoccerTacticalGpInstruction; 5] = [
    SoccerTacticalGpInstruction {
        gene: SoccerTacticalGpGene::DefenseContractDelta,
        expr: SoccerTacticalGpExpr::Max(
            SoccerTacticalGpSignal::DefenseContractGap,
            SoccerTacticalGpSignal::PressGap,
        ),
        scale: 0.28,
    },
    SoccerTacticalGpInstruction {
        gene: SoccerTacticalGpGene::DefenseSpacingScore,
        expr: SoccerTacticalGpExpr::Signal(SoccerTacticalGpSignal::DefenseSpacingGap),
        scale: 0.08,
    },
    SoccerTacticalGpInstruction {
        gene: SoccerTacticalGpGene::DefenderMidfielderPress,
        expr: SoccerTacticalGpExpr::Blend(
            SoccerTacticalGpSignal::PressGap,
            SoccerTacticalGpSignal::DefenseContractGap,
            0.65,
        ),
        scale: 0.16,
    },
    SoccerTacticalGpInstruction {
        gene: SoccerTacticalGpGene::MidfielderPress,
        expr: SoccerTacticalGpExpr::Signal(SoccerTacticalGpSignal::PressGap),
        scale: 0.14,
    },
    SoccerTacticalGpInstruction {
        gene: SoccerTacticalGpGene::DefenseBallDepthScore,
        expr: SoccerTacticalGpExpr::Blend(
            SoccerTacticalGpSignal::DefenseBallGap,
            SoccerTacticalGpSignal::PressGap,
            0.60,
        ),
        scale: 0.12,
    },
];

const TWO_PHASE_GP_PROGRAM: [SoccerTacticalGpInstruction; 5] = [
    SoccerTacticalGpInstruction {
        gene: SoccerTacticalGpGene::AttackWidthDelta,
        expr: SoccerTacticalGpExpr::Blend(
            SoccerTacticalGpSignal::AttackWidthGap,
            SoccerTacticalGpSignal::AttackFlankGap,
            0.50,
        ),
        scale: 0.20,
    },
    SoccerTacticalGpInstruction {
        gene: SoccerTacticalGpGene::AttackFlankLane,
        expr: SoccerTacticalGpExpr::Signal(SoccerTacticalGpSignal::AttackFlankGap),
        scale: 0.30,
    },
    SoccerTacticalGpInstruction {
        gene: SoccerTacticalGpGene::AttackSpacingDelta,
        expr: SoccerTacticalGpExpr::Signal(SoccerTacticalGpSignal::AttackSpacingGap),
        scale: 0.10,
    },
    SoccerTacticalGpInstruction {
        gene: SoccerTacticalGpGene::DefenseContractDelta,
        expr: SoccerTacticalGpExpr::Signal(SoccerTacticalGpSignal::DefenseContractGap),
        scale: 0.34,
    },
    SoccerTacticalGpInstruction {
        gene: SoccerTacticalGpGene::DefenseCompactnessScore,
        expr: SoccerTacticalGpExpr::Product(
            SoccerTacticalGpSignal::DefenseContractGap,
            SoccerTacticalGpSignal::Pressure,
        ),
        scale: 0.22,
    },
];

const FLANK_SWITCH_COMPACT_GP_PROGRAM: [SoccerTacticalGpInstruction; 6] = [
    SoccerTacticalGpInstruction {
        gene: SoccerTacticalGpGene::AttackWidthDelta,
        expr: SoccerTacticalGpExpr::Blend(
            SoccerTacticalGpSignal::AttackWidthGap,
            SoccerTacticalGpSignal::AttackFlankGap,
            0.45,
        ),
        scale: 0.28,
    },
    SoccerTacticalGpInstruction {
        gene: SoccerTacticalGpGene::AttackFlankLane,
        expr: SoccerTacticalGpExpr::Max(
            SoccerTacticalGpSignal::AttackFlankGap,
            SoccerTacticalGpSignal::Pressure,
        ),
        scale: 0.36,
    },
    SoccerTacticalGpInstruction {
        gene: SoccerTacticalGpGene::AttackSpacingDelta,
        expr: SoccerTacticalGpExpr::Blend(
            SoccerTacticalGpSignal::AttackSpacingGap,
            SoccerTacticalGpSignal::AttackFlankGap,
            0.60,
        ),
        scale: 0.16,
    },
    SoccerTacticalGpInstruction {
        gene: SoccerTacticalGpGene::DefenseContractDelta,
        expr: SoccerTacticalGpExpr::Max(
            SoccerTacticalGpSignal::DefenseContractGap,
            SoccerTacticalGpSignal::Pressure,
        ),
        scale: 0.36,
    },
    SoccerTacticalGpInstruction {
        gene: SoccerTacticalGpGene::DefenseCompactnessScore,
        expr: SoccerTacticalGpExpr::Blend(
            SoccerTacticalGpSignal::DefenseContractGap,
            SoccerTacticalGpSignal::DefenseSpacingGap,
            0.75,
        ),
        scale: 0.24,
    },
    SoccerTacticalGpInstruction {
        gene: SoccerTacticalGpGene::DefenseBallDepthScore,
        expr: SoccerTacticalGpExpr::Product(
            SoccerTacticalGpSignal::DefenseBallGap,
            SoccerTacticalGpSignal::Pressure,
        ),
        scale: 0.10,
    },
];

fn search_soccer_tactical_gp_program_candidates<F>(
    base: &SoccerTacticalLearningWeights,
    centroid: &SoccerTacticalLearningWeights,
    weighted_summary: &SoccerTacticalLearningSummary,
    pressure: f64,
    visit: &mut F,
) where
    F: FnMut(SoccerTacticalLearningWeights),
{
    if pressure <= 1e-12 {
        return;
    }

    let programs: [&[SoccerTacticalGpInstruction]; 6] = [
        &WIDE_FLANK_GP_PROGRAM,
        &SUPPORT_SPACING_GP_PROGRAM,
        &COMPACT_BLOCK_GP_PROGRAM,
        &PRESS_FUNNEL_GP_PROGRAM,
        &TWO_PHASE_GP_PROGRAM,
        &FLANK_SWITCH_COMPACT_GP_PROGRAM,
    ];
    for seed in [base, centroid] {
        for program in programs {
            let mut candidate = seed.clone();
            for instruction in program {
                apply_soccer_tactical_gp_instruction(
                    &mut candidate,
                    *instruction,
                    weighted_summary,
                    pressure,
                );
            }
            visit(clamp_soccer_tactical_learning_weights(&candidate));
        }
    }
}

fn search_soccer_tactical_mutated_gp_program_candidates<F>(
    base: &SoccerTacticalLearningWeights,
    centroid: &SoccerTacticalLearningWeights,
    weighted_summary: &SoccerTacticalLearningSummary,
    pressure: f64,
    options: SoccerEvolutionOptions,
    visit: &mut F,
) where
    F: FnMut(SoccerTacticalLearningWeights),
{
    if pressure <= 1e-12 {
        return;
    }
    let search_enabled = options.population_size > 1
        || options.mutation_rate > 0.0
        || options.mutation_scale > 0.0
        || options.crossover_rate > 0.0
        || options.exploration_rate > 0.0
        || options.exploration_scale > 0.0;
    if !search_enabled {
        return;
    }

    let candidate_count = options.population_size.max(2).min(64);
    let crossover_rate = options.crossover_rate.clamp(0.0, 1.0);
    let scale_boost = 1.0
        + options.mutation_scale.max(0.0).min(1.5) * 0.35
        + options.exploration_scale.max(0.0).min(1.5) * 0.25;
    for candidate_index in 0..candidate_count {
        let mut rng = DeterministicRng::new(candidate_seed(
            options.seed,
            candidate_index,
            0x67a5_1c9d_7ac7_1ca1,
        ));
        let mut candidate = if candidate_index % 2 == 0 || rng.next_f64() < crossover_rate {
            centroid.clone()
        } else {
            base.clone()
        };
        seed_soccer_tactical_gp_mutation_bias(
            &mut candidate,
            weighted_summary,
            pressure,
            candidate_index,
            scale_boost,
        );

        let program_len = 3 + rng.next_usize(4);
        for _ in 0..program_len {
            let instruction = random_soccer_tactical_gp_instruction(
                weighted_summary,
                pressure,
                scale_boost,
                &mut rng,
            );
            apply_soccer_tactical_gp_instruction(
                &mut candidate,
                instruction,
                weighted_summary,
                pressure,
            );
        }
        visit(clamp_soccer_tactical_learning_weights(&candidate));
    }
}

fn search_soccer_tactical_novelty_immigrant_candidates<F>(
    base: &SoccerTacticalLearningWeights,
    centroid: &SoccerTacticalLearningWeights,
    weighted_summary: &SoccerTacticalLearningSummary,
    pressure: f64,
    options: SoccerEvolutionOptions,
    visit: &mut F,
) where
    F: FnMut(SoccerTacticalLearningWeights),
{
    if pressure <= 1e-12 {
        return;
    }

    let attack_width_gap = (1.0 - weighted_summary.mean_attack_width_score).clamp(0.0, 1.0);
    let attack_flank_gap = (1.0 - weighted_summary.mean_attack_flank_lane_score).clamp(0.0, 1.0);
    let attack_spacing_gap = (1.0 - weighted_summary.mean_attack_spacing_score).clamp(0.0, 1.0);
    let defense_contract_gap = (1.0 - weighted_summary.mean_defense_contract_score).clamp(0.0, 1.0);
    let defense_spacing_gap = (1.0 - weighted_summary.mean_defense_spacing_score).clamp(0.0, 1.0);
    let defense_ball_gap = (1.0 - weighted_summary.mean_defense_ball_gap_score).clamp(0.0, 1.0);
    let press_gap = (1.0 - weighted_summary.mean_defense_role_press_score).clamp(0.0, 1.0);
    let attack_pressure =
        (attack_width_gap * 0.44 + attack_flank_gap * 0.44 + attack_spacing_gap * 0.12)
            .clamp(0.0, 1.0);
    let defense_pressure = (defense_contract_gap * 0.52
        + defense_spacing_gap * 0.14
        + defense_ball_gap * 0.14
        + press_gap * 0.20)
        .clamp(0.0, 1.0);
    if attack_pressure <= 1e-12 && defense_pressure <= 1e-12 {
        return;
    }

    let candidate_count = options.population_size.max(4).min(24);
    let scale_boost = 1.0
        + options.exploration_scale.max(0.0).min(1.5) * 0.35
        + options.mutation_scale.max(0.0).min(1.5) * 0.20;
    for candidate_index in 0..candidate_count {
        let mut rng = DeterministicRng::new(candidate_seed(
            options.seed,
            candidate_index,
            0x9e37_79b9_7f4a_7c15,
        ));
        let mut candidate = if candidate_index % 4 < 2 {
            centroid.clone()
        } else {
            base.clone()
        };
        let stretch = (0.72 + rng.next_f64() * 0.68) * scale_boost;
        let attack_bias = if candidate_index % 2 == 0 { 1.25 } else { 0.82 };
        let defense_bias = if candidate_index % 3 == 0 { 1.25 } else { 0.86 };

        if attack_pressure > 1e-12 {
            candidate.attack_width_delta_weight +=
                attack_width_gap * (0.26 + pressure * 0.20) * stretch * attack_bias;
            candidate.attack_width_score_weight += attack_width_gap * 0.09 * stretch;
            candidate.attack_flank_lane_weight +=
                attack_flank_gap * (0.42 + pressure * 0.26) * stretch * attack_bias;
            candidate.attack_spacing_delta_weight +=
                attack_spacing_gap * (0.08 + attack_pressure * 0.09) * stretch;
            candidate.attack_spacing_score_weight += attack_spacing_gap * 0.04 * stretch;
        }

        if defense_pressure > 1e-12 {
            candidate.defense_contract_delta_weight +=
                defense_contract_gap * (0.32 + pressure * 0.22) * stretch * defense_bias;
            candidate.defense_compactness_score_weight +=
                defense_contract_gap * (0.22 + defense_pressure * 0.11) * stretch * defense_bias;
            candidate.defense_spacing_delta_weight +=
                defense_spacing_gap * (0.06 + defense_pressure * 0.05) * stretch;
            candidate.defense_ball_depth_score_weight +=
                defense_ball_gap * (0.05 + defense_pressure * 0.06) * stretch;
            candidate.defender_midfielder_press_weight +=
                press_gap * (0.04 + defense_pressure * 0.04) * stretch;
            candidate.midfielder_press_weight +=
                press_gap * (0.04 + defense_pressure * 0.03) * stretch;
        }

        if attack_pressure > 1e-12 && defense_pressure > 1e-12 && candidate_index % 5 == 0 {
            candidate.attack_width_score_weight += attack_width_gap * 0.05 * stretch;
            candidate.attack_flank_lane_weight += attack_flank_gap * 0.20 * stretch;
            candidate.defense_contract_delta_weight += defense_contract_gap * 0.18 * stretch;
        }

        visit(clamp_soccer_tactical_learning_weights(&candidate));
    }
}

fn seed_soccer_tactical_gp_mutation_bias(
    weights: &mut SoccerTacticalLearningWeights,
    summary: &SoccerTacticalLearningSummary,
    pressure: f64,
    candidate_index: usize,
    scale_boost: f64,
) {
    let attack_flank_gap = (1.0 - summary.mean_attack_flank_lane_score).clamp(0.0, 1.0);
    let attack_width_gap = (1.0 - summary.mean_attack_width_score).clamp(0.0, 1.0);
    let defense_contract_gap = (1.0 - summary.mean_defense_contract_score).clamp(0.0, 1.0);
    let press_gap = (1.0 - summary.mean_defense_role_press_score).clamp(0.0, 1.0);

    if candidate_index % 3 == 0 {
        apply_soccer_tactical_gp_instruction(
            weights,
            SoccerTacticalGpInstruction {
                gene: SoccerTacticalGpGene::AttackWidthDelta,
                expr: SoccerTacticalGpExpr::Signal(SoccerTacticalGpSignal::AttackWidthGap),
                scale: (0.10 + attack_width_gap * 0.16) * scale_boost,
            },
            summary,
            pressure,
        );
        apply_soccer_tactical_gp_instruction(
            weights,
            SoccerTacticalGpInstruction {
                gene: SoccerTacticalGpGene::AttackFlankLane,
                expr: SoccerTacticalGpExpr::Blend(
                    SoccerTacticalGpSignal::AttackFlankGap,
                    SoccerTacticalGpSignal::AttackWidthGap,
                    0.70,
                ),
                scale: (0.12 + attack_flank_gap * 0.18) * scale_boost,
            },
            summary,
            pressure,
        );
        apply_soccer_tactical_gp_instruction(
            weights,
            SoccerTacticalGpInstruction {
                gene: SoccerTacticalGpGene::DefenseContractDelta,
                expr: SoccerTacticalGpExpr::Signal(SoccerTacticalGpSignal::DefenseContractGap),
                scale: (0.12 + defense_contract_gap * 0.18) * scale_boost,
            },
            summary,
            pressure,
        );
        apply_soccer_tactical_gp_instruction(
            weights,
            SoccerTacticalGpInstruction {
                gene: SoccerTacticalGpGene::DefenseCompactnessScore,
                expr: SoccerTacticalGpExpr::Product(
                    SoccerTacticalGpSignal::DefenseContractGap,
                    SoccerTacticalGpSignal::Pressure,
                ),
                scale: (0.08 + defense_contract_gap * 0.12) * scale_boost,
            },
            summary,
            pressure,
        );
    } else if candidate_index % 3 == 1 {
        apply_soccer_tactical_gp_instruction(
            weights,
            SoccerTacticalGpInstruction {
                gene: SoccerTacticalGpGene::AttackWidthDelta,
                expr: SoccerTacticalGpExpr::Signal(SoccerTacticalGpSignal::AttackWidthGap),
                scale: (0.10 + attack_width_gap * 0.16) * scale_boost,
            },
            summary,
            pressure,
        );
        apply_soccer_tactical_gp_instruction(
            weights,
            SoccerTacticalGpInstruction {
                gene: SoccerTacticalGpGene::DefenderMidfielderPress,
                expr: SoccerTacticalGpExpr::Signal(SoccerTacticalGpSignal::PressGap),
                scale: (0.06 + press_gap * 0.10) * scale_boost,
            },
            summary,
            pressure,
        );
    }
}

fn random_soccer_tactical_gp_instruction(
    summary: &SoccerTacticalLearningSummary,
    pressure: f64,
    scale_boost: f64,
    rng: &mut DeterministicRng,
) -> SoccerTacticalGpInstruction {
    let gene = random_soccer_tactical_gp_gene(summary, rng);
    let attack_gene = soccer_tactical_gp_gene_is_attack(gene);
    let expr = random_soccer_tactical_gp_expr(attack_gene, rng);
    let scale = (0.05 + rng.next_f64() * 0.34) * (0.65 + pressure * 0.75) * scale_boost;
    SoccerTacticalGpInstruction { gene, expr, scale }
}

fn random_soccer_tactical_gp_gene(
    summary: &SoccerTacticalLearningSummary,
    rng: &mut DeterministicRng,
) -> SoccerTacticalGpGene {
    let attack_pressure = ((1.0 - summary.mean_attack_width_score).clamp(0.0, 1.0) * 0.35
        + (1.0 - summary.mean_attack_flank_lane_score).clamp(0.0, 1.0) * 0.45
        + (1.0 - summary.mean_attack_spacing_score).clamp(0.0, 1.0) * 0.20)
        .max(1e-12);
    let defense_pressure = ((1.0 - summary.mean_defense_contract_score).clamp(0.0, 1.0) * 0.48
        + (1.0 - summary.mean_defense_spacing_score).clamp(0.0, 1.0) * 0.18
        + (1.0 - summary.mean_defense_ball_gap_score).clamp(0.0, 1.0) * 0.17
        + (1.0 - summary.mean_defense_role_press_score).clamp(0.0, 1.0) * 0.17)
        .max(1e-12);
    let attack_probability = attack_pressure / (attack_pressure + defense_pressure);
    if rng.next_f64() < attack_probability {
        ATTACK_SOCCER_TACTICAL_GP_GENES[rng.next_usize(ATTACK_SOCCER_TACTICAL_GP_GENES.len())]
    } else {
        DEFENSE_SOCCER_TACTICAL_GP_GENES[rng.next_usize(DEFENSE_SOCCER_TACTICAL_GP_GENES.len())]
    }
}

fn random_soccer_tactical_gp_expr(
    attack_gene: bool,
    rng: &mut DeterministicRng,
) -> SoccerTacticalGpExpr {
    let first = random_soccer_tactical_gp_signal(attack_gene, rng);
    match rng.next_usize(4) {
        0 => SoccerTacticalGpExpr::Signal(first),
        1 => {
            SoccerTacticalGpExpr::Product(first, random_soccer_tactical_gp_signal(attack_gene, rng))
        }
        2 => SoccerTacticalGpExpr::Blend(
            first,
            random_soccer_tactical_gp_signal(!attack_gene, rng),
            0.25 + rng.next_f64() * 0.50,
        ),
        _ => SoccerTacticalGpExpr::Max(first, random_soccer_tactical_gp_signal(!attack_gene, rng)),
    }
}

fn random_soccer_tactical_gp_signal(
    attack_signal: bool,
    rng: &mut DeterministicRng,
) -> SoccerTacticalGpSignal {
    if attack_signal {
        ATTACK_SOCCER_TACTICAL_GP_SIGNALS[rng.next_usize(ATTACK_SOCCER_TACTICAL_GP_SIGNALS.len())]
    } else {
        DEFENSE_SOCCER_TACTICAL_GP_SIGNALS[rng.next_usize(DEFENSE_SOCCER_TACTICAL_GP_SIGNALS.len())]
    }
}

fn soccer_tactical_gp_gene_is_attack(gene: SoccerTacticalGpGene) -> bool {
    matches!(
        gene,
        SoccerTacticalGpGene::AttackSpacingDelta
            | SoccerTacticalGpGene::AttackSpacingScore
            | SoccerTacticalGpGene::AttackWidthDelta
            | SoccerTacticalGpGene::AttackWidthScore
            | SoccerTacticalGpGene::AttackFlankLane
            | SoccerTacticalGpGene::ShotChoice
            | SoccerTacticalGpGene::GoalEntryPass
            | SoccerTacticalGpGene::PressureRelease
            | SoccerTacticalGpGene::PassTargetRanking
    )
}

fn apply_soccer_tactical_gp_instruction(
    weights: &mut SoccerTacticalLearningWeights,
    instruction: SoccerTacticalGpInstruction,
    summary: &SoccerTacticalLearningSummary,
    pressure: f64,
) {
    let signal = soccer_tactical_gp_expr_value(instruction.expr, summary, pressure);
    if signal <= 0.0 || !signal.is_finite() {
        return;
    }
    adjust_soccer_tactical_gp_gene(weights, instruction.gene, instruction.scale * signal);
}

fn soccer_tactical_gp_expr_value(
    expr: SoccerTacticalGpExpr,
    summary: &SoccerTacticalLearningSummary,
    pressure: f64,
) -> f64 {
    match expr {
        SoccerTacticalGpExpr::Signal(signal) => {
            soccer_tactical_gp_signal_value(signal, summary, pressure)
        }
        SoccerTacticalGpExpr::Product(left, right) => {
            soccer_tactical_gp_signal_value(left, summary, pressure)
                * soccer_tactical_gp_signal_value(right, summary, pressure)
        }
        SoccerTacticalGpExpr::Blend(left, right, mix) => {
            let mix = mix.clamp(0.0, 1.0);
            soccer_tactical_gp_signal_value(left, summary, pressure) * mix
                + soccer_tactical_gp_signal_value(right, summary, pressure) * (1.0 - mix)
        }
        SoccerTacticalGpExpr::Max(left, right) => {
            soccer_tactical_gp_signal_value(left, summary, pressure)
                .max(soccer_tactical_gp_signal_value(right, summary, pressure))
        }
    }
    .clamp(0.0, 1.0)
}

fn soccer_tactical_gp_signal_value(
    signal: SoccerTacticalGpSignal,
    summary: &SoccerTacticalLearningSummary,
    pressure: f64,
) -> f64 {
    match signal {
        SoccerTacticalGpSignal::AttackWidthGap => 1.0 - summary.mean_attack_width_score,
        SoccerTacticalGpSignal::AttackFlankGap => 1.0 - summary.mean_attack_flank_lane_score,
        SoccerTacticalGpSignal::AttackSpacingGap => 1.0 - summary.mean_attack_spacing_score,
        SoccerTacticalGpSignal::DefenseContractGap => 1.0 - summary.mean_defense_contract_score,
        SoccerTacticalGpSignal::DefenseSpacingGap => 1.0 - summary.mean_defense_spacing_score,
        SoccerTacticalGpSignal::DefenseBallGap => 1.0 - summary.mean_defense_ball_gap_score,
        SoccerTacticalGpSignal::PressGap => 1.0 - summary.mean_defense_role_press_score,
        SoccerTacticalGpSignal::Pressure => pressure,
    }
    .clamp(0.0, 1.0)
}

fn adjust_soccer_tactical_gp_gene(
    weights: &mut SoccerTacticalLearningWeights,
    gene: SoccerTacticalGpGene,
    delta: f64,
) {
    match gene {
        SoccerTacticalGpGene::AttackSpacingDelta => {
            weights.attack_spacing_delta_weight += delta;
        }
        SoccerTacticalGpGene::AttackSpacingScore => {
            weights.attack_spacing_score_weight += delta;
        }
        SoccerTacticalGpGene::AttackWidthDelta => {
            weights.attack_width_delta_weight += delta;
        }
        SoccerTacticalGpGene::AttackWidthScore => {
            weights.attack_width_score_weight += delta;
        }
        SoccerTacticalGpGene::AttackFlankLane => {
            weights.attack_flank_lane_weight += delta;
        }
        SoccerTacticalGpGene::ShotChoice => {
            weights.shot_choice_learning_weight += delta;
        }
        SoccerTacticalGpGene::GoalEntryPass => {
            weights.goal_entry_pass_learning_weight += delta;
        }
        SoccerTacticalGpGene::PressureRelease => {
            weights.pressure_release_learning_weight += delta;
        }
        SoccerTacticalGpGene::PassTargetRanking => {
            weights.pass_target_ranking_learning_weight += delta;
        }
        SoccerTacticalGpGene::DefenseSpacingDelta => {
            weights.defense_spacing_delta_weight += delta;
        }
        SoccerTacticalGpGene::DefenseSpacingScore => {
            weights.defense_spacing_score_weight += delta;
        }
        SoccerTacticalGpGene::DefenseContractDelta => {
            weights.defense_contract_delta_weight += delta;
        }
        SoccerTacticalGpGene::DefenseCompactnessScore => {
            weights.defense_compactness_score_weight += delta;
        }
        SoccerTacticalGpGene::DefenseBallDepthScore => {
            weights.defense_ball_depth_score_weight += delta;
        }
        SoccerTacticalGpGene::DefenderMidfielderPress => {
            weights.defender_midfielder_press_weight += delta;
        }
        SoccerTacticalGpGene::MidfielderPress => {
            weights.midfielder_press_weight += delta;
        }
        SoccerTacticalGpGene::DefensiveLinePress => {
            weights.defensive_line_press_learning_weight += delta;
        }
    }
}

fn keep_best_soccer_tactical_candidate(
    best: &mut SoccerTacticalLearningWeights,
    best_score: &mut f64,
    candidate: SoccerTacticalLearningWeights,
    weighted_summary: &SoccerTacticalLearningSummary,
) {
    let candidate = clamp_soccer_tactical_learning_weights(&candidate);
    let score = soccer_tactical_weight_search_score(&candidate, weighted_summary);
    if score > *best_score {
        *best_score = score;
        *best = candidate;
    }
}

fn search_soccer_tactical_strategy_candidates<F>(
    base: &SoccerTacticalLearningWeights,
    weighted_summary: &SoccerTacticalLearningSummary,
    mut visit: F,
) where
    F: FnMut(SoccerTacticalLearningWeights),
{
    let attack_width_gap = (1.0 - weighted_summary.mean_attack_width_score).clamp(0.0, 1.0);
    let attack_flank_gap = (1.0 - weighted_summary.mean_attack_flank_lane_score).clamp(0.0, 1.0);
    let attack_spacing_gap = (1.0 - weighted_summary.mean_attack_spacing_score).clamp(0.0, 1.0);
    let defense_contract_gap = (1.0 - weighted_summary.mean_defense_contract_score).clamp(0.0, 1.0);
    let defense_spacing_gap = (1.0 - weighted_summary.mean_defense_spacing_score).clamp(0.0, 1.0);
    let defense_ball_gap = (1.0 - weighted_summary.mean_defense_ball_gap_score).clamp(0.0, 1.0);
    let press_gap = (1.0 - weighted_summary.mean_defense_role_press_score).clamp(0.0, 1.0);
    let attack_pressure =
        (attack_width_gap * 0.47 + attack_flank_gap * 0.41 + attack_spacing_gap * 0.12)
            .clamp(0.0, 1.0);
    let defense_pressure = (defense_contract_gap * 0.50
        + defense_spacing_gap * 0.18
        + defense_ball_gap * 0.16
        + press_gap * 0.16)
        .clamp(0.0, 1.0);
    let shape_pressure = attack_pressure.max(defense_pressure);

    if attack_pressure > 1e-12 {
        let mut wide_flank = base.clone();
        wide_flank.attack_width_delta_weight += attack_width_gap * 0.32 + attack_pressure * 0.12;
        wide_flank.attack_width_score_weight += attack_width_gap * 0.12;
        wide_flank.attack_flank_lane_weight += attack_flank_gap * 0.36 + attack_pressure * 0.14;
        wide_flank.attack_spacing_delta_weight += attack_spacing_gap * 0.08;
        wide_flank.goal_entry_pass_learning_weight += attack_pressure * 0.10;
        wide_flank.pass_target_ranking_learning_weight +=
            attack_width_gap.max(attack_flank_gap) * 0.08;
        visit(wide_flank);

        let mut flank_overload = base.clone();
        flank_overload.attack_width_delta_weight +=
            attack_width_gap * 0.38 + attack_flank_gap * 0.12;
        flank_overload.attack_width_score_weight += attack_width_gap * 0.14;
        flank_overload.attack_flank_lane_weight += attack_flank_gap * 0.46 + attack_pressure * 0.20;
        flank_overload.attack_spacing_delta_weight +=
            attack_spacing_gap * 0.12 + attack_pressure * 0.04;
        flank_overload.attack_spacing_score_weight += attack_spacing_gap * 0.04;
        flank_overload.shot_choice_learning_weight += attack_width_gap.max(attack_flank_gap) * 0.05;
        flank_overload.goal_entry_pass_learning_weight += attack_pressure * 0.12;
        flank_overload.pass_target_ranking_learning_weight +=
            attack_width_gap.max(attack_flank_gap) * 0.10;
        visit(flank_overload);

        let mut spread_support = base.clone();
        spread_support.attack_spacing_delta_weight +=
            attack_spacing_gap * 0.20 + attack_pressure * 0.06;
        spread_support.attack_spacing_score_weight += attack_spacing_gap * 0.10;
        spread_support.attack_width_delta_weight += attack_width_gap * 0.22;
        spread_support.attack_width_score_weight += attack_width_gap * 0.08;
        spread_support.attack_flank_lane_weight += attack_flank_gap * 0.26;
        spread_support.pressure_release_learning_weight +=
            attack_spacing_gap * 0.10 + attack_pressure * 0.04;
        spread_support.pass_target_ranking_learning_weight += attack_pressure * 0.07;
        visit(spread_support);
    }

    if defense_pressure > 1e-12 {
        let mut compact_defense = base.clone();
        compact_defense.defense_contract_delta_weight +=
            defense_contract_gap * 0.30 + defense_pressure * 0.12;
        compact_defense.defense_compactness_score_weight += defense_contract_gap * 0.18;
        compact_defense.defense_spacing_delta_weight += defense_spacing_gap * 0.08;
        compact_defense.defense_ball_depth_score_weight += defense_ball_gap * 0.06;
        compact_defense.defender_midfielder_press_weight += press_gap * 0.04;
        compact_defense.midfielder_press_weight += press_gap * 0.035;
        compact_defense.defensive_line_press_learning_weight += press_gap * 0.06;
        visit(compact_defense);

        let mut deep_compact_block = base.clone();
        deep_compact_block.defense_contract_delta_weight +=
            defense_contract_gap * 0.34 + defense_pressure * 0.14;
        deep_compact_block.defense_compactness_score_weight += defense_contract_gap * 0.22;
        deep_compact_block.defense_spacing_delta_weight += defense_spacing_gap * 0.07;
        deep_compact_block.defense_spacing_score_weight += defense_spacing_gap * 0.04;
        deep_compact_block.defense_ball_depth_score_weight +=
            defense_ball_gap * 0.12 + defense_pressure * 0.04;
        visit(deep_compact_block);

        let mut press_contract = base.clone();
        press_contract.defense_contract_delta_weight +=
            defense_contract_gap * 0.24 + press_gap * 0.08;
        press_contract.defense_compactness_score_weight += defense_contract_gap * 0.14;
        press_contract.defender_midfielder_press_weight +=
            press_gap * 0.08 + defense_pressure * 0.025;
        press_contract.midfielder_press_weight += press_gap * 0.07 + defense_pressure * 0.020;
        press_contract.defense_ball_depth_score_weight += defense_ball_gap * 0.05;
        press_contract.defensive_line_press_learning_weight +=
            press_gap * 0.10 + defense_pressure * 0.05;
        visit(press_contract);
    }

    if shape_pressure > 1e-12 {
        let mut balanced_shape = base.clone();
        balanced_shape.attack_width_delta_weight +=
            attack_width_gap * 0.28 + attack_pressure * 0.10;
        balanced_shape.attack_width_score_weight += attack_width_gap * 0.08;
        balanced_shape.attack_flank_lane_weight += attack_flank_gap * 0.34 + attack_pressure * 0.12;
        balanced_shape.attack_spacing_delta_weight += attack_spacing_gap * 0.07 * shape_pressure;
        balanced_shape.goal_entry_pass_learning_weight += attack_pressure * 0.08;
        balanced_shape.pressure_release_learning_weight +=
            attack_spacing_gap * 0.05 * shape_pressure;
        balanced_shape.pass_target_ranking_learning_weight +=
            attack_width_gap.max(attack_flank_gap) * 0.07;
        balanced_shape.defense_contract_delta_weight +=
            defense_contract_gap * 0.27 + defense_pressure * 0.10;
        balanced_shape.defense_compactness_score_weight += defense_contract_gap * 0.16;
        balanced_shape.defense_spacing_delta_weight += defense_spacing_gap * 0.06 * shape_pressure;
        balanced_shape.defense_ball_depth_score_weight += defense_ball_gap * 0.05 * shape_pressure;
        balanced_shape.defensive_line_press_learning_weight += press_gap * 0.08 * shape_pressure;
        visit(balanced_shape);
    }
}

pub fn soccer_tactical_search_pressure(summary: &SoccerTacticalLearningSummary) -> f64 {
    let attack_width_gap = (1.0 - summary.mean_attack_width_score).clamp(0.0, 1.0);
    let attack_flank_gap = (1.0 - summary.mean_attack_flank_lane_score).clamp(0.0, 1.0);
    let attack_spacing_gap = (1.0 - summary.mean_attack_spacing_score).clamp(0.0, 1.0);
    let defense_contract_gap = (1.0 - summary.mean_defense_contract_score).clamp(0.0, 1.0);
    let defense_spacing_gap = (1.0 - summary.mean_defense_spacing_score).clamp(0.0, 1.0);
    let defense_ball_gap = (1.0 - summary.mean_defense_ball_gap_score).clamp(0.0, 1.0);
    let press_gap = (1.0 - summary.mean_defense_role_press_score).clamp(0.0, 1.0);

    (attack_width_gap * 0.22
        + attack_flank_gap * 0.28
        + attack_spacing_gap * 0.08
        + defense_contract_gap * 0.26
        + defense_spacing_gap * 0.06
        + defense_ball_gap * 0.05
        + press_gap * 0.05)
        .clamp(0.0, 1.0)
}

pub fn adapt_soccer_evolution_options_for_tactical_search(
    mut options: SoccerEvolutionOptions,
    summary: &SoccerTacticalLearningSummary,
) -> SoccerEvolutionOptions {
    options.population_size =
        soccer_evolution_population_size_for_learning_run(options.population_size);
    let pressure = soccer_tactical_search_pressure(summary);
    if pressure <= 1e-12 {
        return options;
    }
    let search_enabled = options.population_size > 1
        || options.mutation_rate > 0.0
        || options.mutation_scale > 0.0
        || options.crossover_rate > 0.0
        || options.exploration_rate > 0.0
        || options.exploration_scale > 0.0;
    if !search_enabled {
        return options;
    }

    options.mutation_rate = (options.mutation_rate + pressure * 0.015).clamp(0.0, 0.35);
    options.mutation_scale = (options.mutation_scale * (1.0 + pressure * 0.45))
        .max(options.mutation_scale)
        .min(0.85);
    if options.exploration_rate > 0.0 || options.exploration_scale > 0.0 {
        options.exploration_rate = (options.exploration_rate + pressure * 0.08).clamp(0.0, 0.4);
        options.exploration_scale = (options.exploration_scale * (1.0 + pressure * 0.55))
            .max(options.exploration_scale)
            .min(1.2);
    }
    if options.population_size > 1 {
        let extra_candidates = ((options.population_size as f64) * pressure).ceil() as usize;
        let cap = options.population_size.saturating_mul(2).min(64);
        options.population_size = options
            .population_size
            .saturating_add(extra_candidates)
            .min(cap)
            .max(2);
    }
    options
}

fn weighted_tactical_evolution_summary(
    parents: &[(&SoccerTacticalLearningSummary, f64)],
    options: SoccerEvolutionOptions,
) -> Option<SoccerTacticalLearningSummary> {
    if parents.is_empty() {
        return None;
    }
    let mut weighted_summary = SoccerTacticalLearningSummary::default();
    let mut total_weight = 0.0;
    for (summary, fitness) in parents {
        if !tactical_evolution_summary_is_finite(summary) {
            continue;
        }
        let weight = tactical_genome_parent_weight(*fitness, options);
        if !weight.is_finite() || weight <= 0.0 {
            continue;
        }
        total_weight += weight;
        weighted_summary.mean_attack_width_score += summary.mean_attack_width_score * weight;
        weighted_summary.mean_attack_flank_lane_score +=
            summary.mean_attack_flank_lane_score * weight;
        weighted_summary.mean_attack_spacing_score += summary.mean_attack_spacing_score * weight;
        weighted_summary.mean_defense_contract_score +=
            summary.mean_defense_contract_score * weight;
        weighted_summary.mean_defense_spacing_score += summary.mean_defense_spacing_score * weight;
        weighted_summary.mean_defense_ball_gap_score +=
            summary.mean_defense_ball_gap_score * weight;
        weighted_summary.mean_defense_role_press_score +=
            summary.mean_defense_role_press_score * weight;
    }
    if total_weight <= 0.0 {
        return None;
    }
    weighted_summary.mean_attack_width_score /= total_weight;
    weighted_summary.mean_attack_flank_lane_score /= total_weight;
    weighted_summary.mean_attack_spacing_score /= total_weight;
    weighted_summary.mean_defense_contract_score /= total_weight;
    weighted_summary.mean_defense_spacing_score /= total_weight;
    weighted_summary.mean_defense_ball_gap_score /= total_weight;
    weighted_summary.mean_defense_role_press_score /= total_weight;
    Some(weighted_summary)
}

fn tactical_evolution_summary_is_finite(summary: &SoccerTacticalLearningSummary) -> bool {
    summary.mean_attack_width_score.is_finite()
        && summary.mean_attack_flank_lane_score.is_finite()
        && summary.mean_attack_spacing_score.is_finite()
        && summary.mean_defense_contract_score.is_finite()
        && summary.mean_defense_spacing_score.is_finite()
        && summary.mean_defense_ball_gap_score.is_finite()
        && summary.mean_defense_role_press_score.is_finite()
}

fn weighted_soccer_tactical_genome_centroid(
    parents: &[SoccerTacticalLearningGenomeParent<'_>],
    options: SoccerEvolutionOptions,
) -> Option<SoccerTacticalLearningWeights> {
    let mut centroid = SoccerTacticalLearningWeights::default();
    centroid.attack_spacing_delta_weight =
        weighted_soccer_tactical_genome_gene(parents, options, |weights| {
            weights.attack_spacing_delta_weight
        })?;
    centroid.attack_spacing_score_weight =
        weighted_soccer_tactical_genome_gene(parents, options, |weights| {
            weights.attack_spacing_score_weight
        })?;
    centroid.attack_width_delta_weight =
        weighted_soccer_tactical_genome_gene(parents, options, |weights| {
            weights.attack_width_delta_weight
        })?;
    centroid.attack_width_score_weight =
        weighted_soccer_tactical_genome_gene(parents, options, |weights| {
            weights.attack_width_score_weight
        })?;
    centroid.attack_flank_lane_weight =
        weighted_soccer_tactical_genome_gene(parents, options, |weights| {
            weights.attack_flank_lane_weight
        })?;
    centroid.shot_choice_learning_weight =
        weighted_soccer_tactical_genome_gene(parents, options, |weights| {
            weights.shot_choice_learning_weight
        })?;
    centroid.goal_entry_pass_learning_weight =
        weighted_soccer_tactical_genome_gene(parents, options, |weights| {
            weights.goal_entry_pass_learning_weight
        })?;
    centroid.pressure_release_learning_weight =
        weighted_soccer_tactical_genome_gene(parents, options, |weights| {
            weights.pressure_release_learning_weight
        })?;
    centroid.pass_target_ranking_learning_weight =
        weighted_soccer_tactical_genome_gene(parents, options, |weights| {
            weights.pass_target_ranking_learning_weight
        })?;
    centroid.defense_spacing_delta_weight =
        weighted_soccer_tactical_genome_gene(parents, options, |weights| {
            weights.defense_spacing_delta_weight
        })?;
    centroid.defense_spacing_score_weight =
        weighted_soccer_tactical_genome_gene(parents, options, |weights| {
            weights.defense_spacing_score_weight
        })?;
    centroid.defense_contract_delta_weight =
        weighted_soccer_tactical_genome_gene(parents, options, |weights| {
            weights.defense_contract_delta_weight
        })?;
    centroid.defense_compactness_score_weight =
        weighted_soccer_tactical_genome_gene(parents, options, |weights| {
            weights.defense_compactness_score_weight
        })?;
    centroid.defense_ball_depth_score_weight =
        weighted_soccer_tactical_genome_gene(parents, options, |weights| {
            weights.defense_ball_depth_score_weight
        })?;
    centroid.defense_endline_soft_penalty_weight =
        weighted_soccer_tactical_genome_gene(parents, options, |weights| {
            weights.defense_endline_soft_penalty_weight
        })?;
    centroid.defense_endline_hard_penalty_weight =
        weighted_soccer_tactical_genome_gene(parents, options, |weights| {
            weights.defense_endline_hard_penalty_weight
        })?;
    centroid.defender_midfielder_press_weight =
        weighted_soccer_tactical_genome_gene(parents, options, |weights| {
            weights.defender_midfielder_press_weight
        })?;
    centroid.midfielder_press_weight =
        weighted_soccer_tactical_genome_gene(parents, options, |weights| {
            weights.midfielder_press_weight
        })?;
    centroid.defensive_line_press_learning_weight =
        weighted_soccer_tactical_genome_gene(parents, options, |weights| {
            weights.defensive_line_press_learning_weight
        })?;
    centroid.formation_lp_alignment_weight =
        weighted_soccer_tactical_genome_gene(parents, options, |weights| {
            weights.formation_lp_alignment_weight
        })?;

    Some(clamp_soccer_tactical_learning_weights(&centroid))
}

fn weighted_soccer_tactical_genome_gene<F>(
    parents: &[SoccerTacticalLearningGenomeParent<'_>],
    options: SoccerEvolutionOptions,
    getter: F,
) -> Option<f64>
where
    F: Fn(&SoccerTacticalLearningWeights) -> f64,
{
    let mut weighted_gene = 0.0;
    let mut total_weight = 0.0;
    for parent in parents {
        let value = getter(parent.weights);
        if !value.is_finite() {
            continue;
        }
        let weight = tactical_genome_parent_weight(parent.fitness, options);
        if weight <= 0.0 {
            continue;
        }
        weighted_gene += value * weight;
        total_weight += weight;
    }
    if total_weight > 0.0 {
        Some(weighted_gene / total_weight)
    } else {
        None
    }
}

fn extrapolate_soccer_tactical_learning_weights(
    base: &SoccerTacticalLearningWeights,
    target: &SoccerTacticalLearningWeights,
    scale: f64,
) -> SoccerTacticalLearningWeights {
    let scale = scale.max(0.0);
    let mut extrapolated = base.clone();
    extrapolated.attack_spacing_delta_weight = extrapolate_soccer_tactical_weight(
        base.attack_spacing_delta_weight,
        target.attack_spacing_delta_weight,
        scale,
    );
    extrapolated.attack_spacing_score_weight = extrapolate_soccer_tactical_weight(
        base.attack_spacing_score_weight,
        target.attack_spacing_score_weight,
        scale,
    );
    extrapolated.attack_width_delta_weight = extrapolate_soccer_tactical_weight(
        base.attack_width_delta_weight,
        target.attack_width_delta_weight,
        scale,
    );
    extrapolated.attack_width_score_weight = extrapolate_soccer_tactical_weight(
        base.attack_width_score_weight,
        target.attack_width_score_weight,
        scale,
    );
    extrapolated.attack_flank_lane_weight = extrapolate_soccer_tactical_weight(
        base.attack_flank_lane_weight,
        target.attack_flank_lane_weight,
        scale,
    );
    extrapolated.shot_choice_learning_weight = extrapolate_soccer_tactical_weight(
        base.shot_choice_learning_weight,
        target.shot_choice_learning_weight,
        scale,
    );
    extrapolated.goal_entry_pass_learning_weight = extrapolate_soccer_tactical_weight(
        base.goal_entry_pass_learning_weight,
        target.goal_entry_pass_learning_weight,
        scale,
    );
    extrapolated.pressure_release_learning_weight = extrapolate_soccer_tactical_weight(
        base.pressure_release_learning_weight,
        target.pressure_release_learning_weight,
        scale,
    );
    extrapolated.pass_target_ranking_learning_weight = extrapolate_soccer_tactical_weight(
        base.pass_target_ranking_learning_weight,
        target.pass_target_ranking_learning_weight,
        scale,
    );
    extrapolated.defense_spacing_delta_weight = extrapolate_soccer_tactical_weight(
        base.defense_spacing_delta_weight,
        target.defense_spacing_delta_weight,
        scale,
    );
    extrapolated.defense_spacing_score_weight = extrapolate_soccer_tactical_weight(
        base.defense_spacing_score_weight,
        target.defense_spacing_score_weight,
        scale,
    );
    extrapolated.defense_contract_delta_weight = extrapolate_soccer_tactical_weight(
        base.defense_contract_delta_weight,
        target.defense_contract_delta_weight,
        scale,
    );
    extrapolated.defense_compactness_score_weight = extrapolate_soccer_tactical_weight(
        base.defense_compactness_score_weight,
        target.defense_compactness_score_weight,
        scale,
    );
    extrapolated.defense_ball_depth_score_weight = extrapolate_soccer_tactical_weight(
        base.defense_ball_depth_score_weight,
        target.defense_ball_depth_score_weight,
        scale,
    );
    extrapolated.defense_endline_soft_penalty_weight = extrapolate_soccer_tactical_weight(
        base.defense_endline_soft_penalty_weight,
        target.defense_endline_soft_penalty_weight,
        scale,
    );
    extrapolated.defense_endline_hard_penalty_weight = extrapolate_soccer_tactical_weight(
        base.defense_endline_hard_penalty_weight,
        target.defense_endline_hard_penalty_weight,
        scale,
    );
    extrapolated.defender_midfielder_press_weight = extrapolate_soccer_tactical_weight(
        base.defender_midfielder_press_weight,
        target.defender_midfielder_press_weight,
        scale,
    );
    extrapolated.midfielder_press_weight = extrapolate_soccer_tactical_weight(
        base.midfielder_press_weight,
        target.midfielder_press_weight,
        scale,
    );
    extrapolated.defensive_line_press_learning_weight = extrapolate_soccer_tactical_weight(
        base.defensive_line_press_learning_weight,
        target.defensive_line_press_learning_weight,
        scale,
    );
    extrapolated.formation_lp_alignment_weight = extrapolate_soccer_tactical_weight(
        base.formation_lp_alignment_weight,
        target.formation_lp_alignment_weight,
        scale,
    );
    clamp_soccer_tactical_learning_weights(&extrapolated)
}

fn extrapolate_soccer_tactical_weight(base: f64, target: f64, scale: f64) -> f64 {
    if base.is_finite() && target.is_finite() {
        base + (target - base) * scale
    } else {
        base
    }
}

fn evolve_soccer_tactical_learning_candidate(
    base: &SoccerTacticalLearningWeights,
    weighted_summary: &SoccerTacticalLearningSummary,
    options: SoccerEvolutionOptions,
) -> SoccerTacticalLearningWeights {
    let mut rng = DeterministicRng::new(options.seed ^ 0x5eed_f00d_cafe_babe);
    let mut evolved = base.clone();
    let mutation_scale = options.mutation_scale.max(0.0) * 0.22;
    let exploration_rate = options.exploration_rate.clamp(0.0, 1.0);
    let exploration_scale = options.exploration_scale.max(0.0) * 0.18;

    let attack_width_gap = (1.0 - weighted_summary.mean_attack_width_score).clamp(0.0, 1.0);
    let attack_flank_gap = (1.0 - weighted_summary.mean_attack_flank_lane_score).clamp(0.0, 1.0);
    let attack_spacing_gap = (1.0 - weighted_summary.mean_attack_spacing_score).clamp(0.0, 1.0);
    let defense_contract_gap = (1.0 - weighted_summary.mean_defense_contract_score).clamp(0.0, 1.0);
    let defense_spacing_gap = (1.0 - weighted_summary.mean_defense_spacing_score).clamp(0.0, 1.0);
    let defense_ball_gap = (1.0 - weighted_summary.mean_defense_ball_gap_score).clamp(0.0, 1.0);
    let press_gap = (1.0 - weighted_summary.mean_defense_role_press_score).clamp(0.0, 1.0);
    let attack_decision_gap = attack_width_gap
        .max(attack_flank_gap)
        .max(attack_spacing_gap * 0.70);

    evolve_weight(
        &mut evolved.attack_spacing_delta_weight,
        attack_spacing_gap * 0.050,
        mutation_scale,
        exploration_rate,
        exploration_scale,
        &mut rng,
        1.8,
    );
    evolve_weight(
        &mut evolved.attack_width_delta_weight,
        attack_width_gap * 0.090,
        mutation_scale,
        exploration_rate,
        exploration_scale,
        &mut rng,
        2.2,
    );
    evolve_weight(
        &mut evolved.attack_width_score_weight,
        attack_width_gap * 0.045,
        mutation_scale,
        exploration_rate,
        exploration_scale,
        &mut rng,
        1.2,
    );
    evolve_weight(
        &mut evolved.attack_flank_lane_weight,
        attack_flank_gap * 0.105,
        mutation_scale,
        exploration_rate,
        exploration_scale,
        &mut rng,
        2.2,
    );
    evolve_weight(
        &mut evolved.shot_choice_learning_weight,
        attack_decision_gap * 0.050,
        mutation_scale,
        exploration_rate,
        exploration_scale,
        &mut rng,
        2.0,
    );
    evolve_weight(
        &mut evolved.goal_entry_pass_learning_weight,
        attack_width_gap.max(attack_flank_gap) * 0.060 + attack_spacing_gap * 0.020,
        mutation_scale,
        exploration_rate,
        exploration_scale,
        &mut rng,
        2.0,
    );
    evolve_weight(
        &mut evolved.pressure_release_learning_weight,
        attack_spacing_gap * 0.045 + press_gap * 0.030,
        mutation_scale,
        exploration_rate,
        exploration_scale,
        &mut rng,
        2.0,
    );
    evolve_weight(
        &mut evolved.pass_target_ranking_learning_weight,
        attack_width_gap.max(attack_flank_gap) * 0.055 + attack_spacing_gap * 0.025,
        mutation_scale,
        exploration_rate,
        exploration_scale,
        &mut rng,
        2.0,
    );
    evolve_weight(
        &mut evolved.defense_contract_delta_weight,
        defense_contract_gap * 0.080,
        mutation_scale,
        exploration_rate,
        exploration_scale,
        &mut rng,
        2.4,
    );
    evolve_weight(
        &mut evolved.defense_compactness_score_weight,
        defense_contract_gap * 0.052,
        mutation_scale,
        exploration_rate,
        exploration_scale,
        &mut rng,
        2.0,
    );
    evolve_weight(
        &mut evolved.defense_spacing_delta_weight,
        defense_spacing_gap * 0.040,
        mutation_scale,
        exploration_rate,
        exploration_scale,
        &mut rng,
        1.8,
    );
    evolve_weight(
        &mut evolved.defense_ball_depth_score_weight,
        defense_ball_gap * 0.050,
        mutation_scale,
        exploration_rate,
        exploration_scale,
        &mut rng,
        2.0,
    );
    evolve_weight(
        &mut evolved.defender_midfielder_press_weight,
        press_gap * 0.034,
        mutation_scale,
        exploration_rate,
        exploration_scale,
        &mut rng,
        1.6,
    );
    evolve_weight(
        &mut evolved.midfielder_press_weight,
        press_gap * 0.030,
        mutation_scale,
        exploration_rate,
        exploration_scale,
        &mut rng,
        1.6,
    );
    evolve_weight(
        &mut evolved.defensive_line_press_learning_weight,
        press_gap * 0.070 + defense_ball_gap * 0.025,
        mutation_scale,
        exploration_rate,
        exploration_scale,
        &mut rng,
        2.0,
    );

    evolved
}

fn crossover_soccer_tactical_learning_weights(
    base: &SoccerTacticalLearningWeights,
    parents: &[SoccerTacticalLearningGenomeParent<'_>],
    options: SoccerEvolutionOptions,
    rng: &mut DeterministicRng,
) -> SoccerTacticalLearningWeights {
    let crossover_rate = options.crossover_rate.clamp(0.0, 1.0);
    if parents.len() < 2 || crossover_rate <= 0.0 {
        return base.clone();
    }
    let mut child = base.clone();
    crossover_tactical_weight_gene(
        &mut child.attack_spacing_delta_weight,
        parents,
        options,
        rng,
        1.8,
        |weights| weights.attack_spacing_delta_weight,
    );
    crossover_tactical_weight_gene(
        &mut child.attack_spacing_score_weight,
        parents,
        options,
        rng,
        1.2,
        |weights| weights.attack_spacing_score_weight,
    );
    crossover_tactical_weight_gene(
        &mut child.attack_width_delta_weight,
        parents,
        options,
        rng,
        2.2,
        |weights| weights.attack_width_delta_weight,
    );
    crossover_tactical_weight_gene(
        &mut child.attack_width_score_weight,
        parents,
        options,
        rng,
        1.2,
        |weights| weights.attack_width_score_weight,
    );
    crossover_tactical_weight_gene(
        &mut child.attack_flank_lane_weight,
        parents,
        options,
        rng,
        2.2,
        |weights| weights.attack_flank_lane_weight,
    );
    crossover_tactical_weight_gene(
        &mut child.shot_choice_learning_weight,
        parents,
        options,
        rng,
        2.0,
        |weights| weights.shot_choice_learning_weight,
    );
    crossover_tactical_weight_gene(
        &mut child.goal_entry_pass_learning_weight,
        parents,
        options,
        rng,
        2.0,
        |weights| weights.goal_entry_pass_learning_weight,
    );
    crossover_tactical_weight_gene(
        &mut child.pressure_release_learning_weight,
        parents,
        options,
        rng,
        2.0,
        |weights| weights.pressure_release_learning_weight,
    );
    crossover_tactical_weight_gene(
        &mut child.pass_target_ranking_learning_weight,
        parents,
        options,
        rng,
        2.0,
        |weights| weights.pass_target_ranking_learning_weight,
    );
    crossover_tactical_weight_gene(
        &mut child.defense_spacing_delta_weight,
        parents,
        options,
        rng,
        1.8,
        |weights| weights.defense_spacing_delta_weight,
    );
    crossover_tactical_weight_gene(
        &mut child.defense_spacing_score_weight,
        parents,
        options,
        rng,
        1.2,
        |weights| weights.defense_spacing_score_weight,
    );
    crossover_tactical_weight_gene(
        &mut child.defense_contract_delta_weight,
        parents,
        options,
        rng,
        2.4,
        |weights| weights.defense_contract_delta_weight,
    );
    crossover_tactical_weight_gene(
        &mut child.defense_compactness_score_weight,
        parents,
        options,
        rng,
        2.0,
        |weights| weights.defense_compactness_score_weight,
    );
    crossover_tactical_weight_gene(
        &mut child.defense_ball_depth_score_weight,
        parents,
        options,
        rng,
        2.0,
        |weights| weights.defense_ball_depth_score_weight,
    );
    crossover_tactical_weight_gene(
        &mut child.defense_endline_soft_penalty_weight,
        parents,
        options,
        rng,
        5.0,
        |weights| weights.defense_endline_soft_penalty_weight,
    );
    crossover_tactical_weight_gene(
        &mut child.defense_endline_hard_penalty_weight,
        parents,
        options,
        rng,
        5.0,
        |weights| weights.defense_endline_hard_penalty_weight,
    );
    crossover_tactical_weight_gene(
        &mut child.defender_midfielder_press_weight,
        parents,
        options,
        rng,
        1.6,
        |weights| weights.defender_midfielder_press_weight,
    );
    crossover_tactical_weight_gene(
        &mut child.midfielder_press_weight,
        parents,
        options,
        rng,
        1.6,
        |weights| weights.midfielder_press_weight,
    );
    crossover_tactical_weight_gene(
        &mut child.defensive_line_press_learning_weight,
        parents,
        options,
        rng,
        2.0,
        |weights| weights.defensive_line_press_learning_weight,
    );
    child
}

fn crossover_tactical_weight_gene<F>(
    value: &mut f64,
    parents: &[SoccerTacticalLearningGenomeParent<'_>],
    options: SoccerEvolutionOptions,
    rng: &mut DeterministicRng,
    max_value: f64,
    getter: F,
) where
    F: Fn(&SoccerTacticalLearningWeights) -> f64,
{
    if rng.next_f64() > options.crossover_rate.clamp(0.0, 1.0) {
        return;
    }
    let Some(parent_index) = select_tactical_genome_parent(parents, options, rng) else {
        return;
    };
    let parent_value = getter(parents[parent_index].weights);
    if parent_value.is_finite() {
        *value = parent_value.max(0.0).min(max_value.max(0.0));
    }
}

fn select_tactical_genome_parent(
    parents: &[SoccerTacticalLearningGenomeParent<'_>],
    options: SoccerEvolutionOptions,
    rng: &mut DeterministicRng,
) -> Option<usize> {
    let total_weight = parents
        .iter()
        .map(|parent| tactical_genome_parent_weight(parent.fitness, options))
        .sum::<f64>();
    if total_weight <= 0.0 {
        return None;
    }
    let mut cursor = rng.next_f64() * total_weight;
    for (index, parent) in parents.iter().enumerate() {
        let weight = tactical_genome_parent_weight(parent.fitness, options);
        if weight <= 0.0 {
            continue;
        }
        if cursor <= weight {
            return Some(index);
        }
        cursor -= weight;
    }
    parents
        .iter()
        .enumerate()
        .rfind(|(_, parent)| tactical_genome_parent_weight(parent.fitness, options) > 0.0)
        .map(|(index, _)| index)
}

fn tactical_genome_parent_weight(fitness: f64, options: SoccerEvolutionOptions) -> f64 {
    if fitness.is_finite() {
        fitness.max(options.elite_weight_floor).max(0.0)
    } else {
        0.0
    }
}

fn clamp_soccer_tactical_learning_weights(
    weights: &SoccerTacticalLearningWeights,
) -> SoccerTacticalLearningWeights {
    let mut clamped = weights.clone();
    clamped.attack_spacing_delta_weight = clamped.attack_spacing_delta_weight.max(0.0).min(1.8);
    clamped.attack_spacing_score_weight = clamped.attack_spacing_score_weight.max(0.0).min(1.2);
    clamped.attack_width_delta_weight = clamped.attack_width_delta_weight.max(0.0).min(2.2);
    clamped.attack_width_score_weight = clamped.attack_width_score_weight.max(0.0).min(1.2);
    clamped.attack_flank_lane_weight = clamped.attack_flank_lane_weight.max(0.0).min(2.2);
    clamped.shot_choice_learning_weight = clamped.shot_choice_learning_weight.max(0.0).min(2.0);
    clamped.goal_entry_pass_learning_weight =
        clamped.goal_entry_pass_learning_weight.max(0.0).min(2.0);
    clamped.pressure_release_learning_weight =
        clamped.pressure_release_learning_weight.max(0.0).min(2.0);
    clamped.pass_target_ranking_learning_weight = clamped
        .pass_target_ranking_learning_weight
        .max(0.0)
        .min(2.0);
    clamped.defense_spacing_delta_weight = clamped.defense_spacing_delta_weight.max(0.0).min(1.8);
    clamped.defense_spacing_score_weight = clamped.defense_spacing_score_weight.max(0.0).min(1.2);
    clamped.defense_contract_delta_weight = clamped.defense_contract_delta_weight.max(0.0).min(2.4);
    clamped.defense_compactness_score_weight =
        clamped.defense_compactness_score_weight.max(0.0).min(2.0);
    clamped.defense_ball_depth_score_weight =
        clamped.defense_ball_depth_score_weight.max(0.0).min(2.0);
    clamped.defense_endline_soft_penalty_weight = clamped
        .defense_endline_soft_penalty_weight
        .max(0.0)
        .min(5.0);
    clamped.defense_endline_hard_penalty_weight = clamped
        .defense_endline_hard_penalty_weight
        .max(0.0)
        .min(5.0);
    clamped.defender_midfielder_press_weight =
        clamped.defender_midfielder_press_weight.max(0.0).min(1.6);
    clamped.midfielder_press_weight = clamped.midfielder_press_weight.max(0.0).min(1.6);
    clamped.defensive_line_press_learning_weight = clamped
        .defensive_line_press_learning_weight
        .max(0.0)
        .min(2.0);
    clamped.formation_lp_alignment_weight =
        clamped.formation_lp_alignment_weight.max(-5.0).min(5.0);
    clamped
}

fn soccer_tactical_weight_search_score(
    weights: &SoccerTacticalLearningWeights,
    summary: &SoccerTacticalLearningSummary,
) -> f64 {
    let attack_width_gap = (1.0 - summary.mean_attack_width_score).clamp(0.0, 1.0);
    let attack_flank_gap = (1.0 - summary.mean_attack_flank_lane_score).clamp(0.0, 1.0);
    let attack_spacing_gap = (1.0 - summary.mean_attack_spacing_score).clamp(0.0, 1.0);
    let defense_contract_gap = (1.0 - summary.mean_defense_contract_score).clamp(0.0, 1.0);
    let defense_spacing_gap = (1.0 - summary.mean_defense_spacing_score).clamp(0.0, 1.0);
    let defense_ball_gap = (1.0 - summary.mean_defense_ball_gap_score).clamp(0.0, 1.0);
    let press_gap = (1.0 - summary.mean_defense_role_press_score).clamp(0.0, 1.0);
    let attack_decision_gap = attack_width_gap
        .max(attack_flank_gap)
        .max(attack_spacing_gap * 0.70);

    weights.attack_width_delta_weight * attack_width_gap * 1.15
        + weights.attack_width_score_weight * attack_width_gap * 0.45
        + weights.attack_flank_lane_weight * attack_flank_gap * 1.35
        + weights.attack_spacing_delta_weight * attack_spacing_gap * 0.85
        + weights.attack_spacing_score_weight * attack_spacing_gap * 0.35
        + weights.shot_choice_learning_weight * attack_decision_gap * 0.10
        + weights.goal_entry_pass_learning_weight * attack_width_gap.max(attack_flank_gap) * 0.18
        + weights.pressure_release_learning_weight
            * (attack_spacing_gap * 0.50 + press_gap * 0.50)
            * 0.14
        + weights.pass_target_ranking_learning_weight
            * attack_width_gap.max(attack_flank_gap)
            * 0.16
        + weights.defense_contract_delta_weight * defense_contract_gap * 1.35
        + weights.defense_compactness_score_weight * defense_contract_gap * 1.05
        + weights.defense_spacing_delta_weight * defense_spacing_gap * 0.80
        + weights.defense_spacing_score_weight * defense_spacing_gap * 0.35
        + weights.defense_ball_depth_score_weight * defense_ball_gap * 0.70
        + weights.defender_midfielder_press_weight * press_gap * 0.40
        + weights.midfielder_press_weight * press_gap * 0.35
        + weights.defensive_line_press_learning_weight * press_gap * 0.20
}

pub fn run_soccer_learning_game(
    episode: usize,
    config: MatchConfig,
    starting_policies: SoccerTeamQPolicies,
    neural_drain_timeout: Duration,
) -> Result<SoccerLearningCompletedGame, String> {
    run_soccer_learning_game_from_snapshot(
        episode,
        config,
        Arc::new(starting_policies),
        None,
        neural_drain_timeout,
    )
}

fn run_soccer_learning_game_from_snapshot(
    episode: usize,
    mut config: MatchConfig,
    starting_policies: Arc<SoccerTeamQPolicies>,
    initial_neural_network: Option<Arc<SoccerNeuralNetworkSnapshot>>,
    neural_drain_timeout: Duration,
) -> Result<SoccerLearningCompletedGame, String> {
    let started = Instant::now();
    validate_soccer_neural_learning_config_for_learning_run(&config.neural_learning)?;
    validate_soccer_neural_blend_config_for_learning_run(&config.neural_blend)?;
    config.seed = config.seed.wrapping_add(episode as u32);
    // Progressive curriculum (off unless SOCCER_CURRICULUM_ENABLED): reshape this game's
    // pitch / length / team-reward share / formation by how many games have completed, so
    // early episodes are tight individual-skill reps and later ones the full 11v11 contest.
    let _curriculum_stage = maybe_apply_soccer_curriculum_for_episode(episode, &mut config);
    let seed = config.seed as u64;
    let starting_tactical_learning = config.tactical_learning.clone();
    let total_ticks = config.total_ticks();
    let mut sim =
        SoccerMatch::default_11v11(config).with_team_policies((*starting_policies).clone());
    sim.set_uniform_elite_players();
    if let Some(snapshot) = initial_neural_network.as_ref() {
        sim.set_neural_network_snapshot((**snapshot).clone())?;
    }

    for _ in 0..total_ticks {
        sim.run_time_step();
    }

    sim.drain_neural_learning(neural_drain_timeout);
    let policies = sim
        .team_policies()
        .cloned()
        .ok_or_else(|| "soccer learning produced no team policies".to_string())?;
    let mut artifact = sim.team_policy_artifact();
    let neural_network = artifact.learning.neural_network.take();
    let summary = artifact.summary.clone();
    let score = soccer_learning_run_score(&summary);
    let delta = soccer_policy_delta_entries(starting_policies.as_ref(), &policies, &score);
    let config_moments = sim.config_moments();
    let pass_outcome_samples = sim.drain_pass_outcome_samples();
    let episode_summary = SoccerSelfPlayEpisodeSummary {
        episode,
        seed,
        summary,
        transitions: artifact.learning.total_transitions,
        home_policy_entries: artifact.home_entries.len(),
        home_policy_target_entries: artifact.home_target_entries.len(),
        away_policy_entries: artifact.away_entries.len(),
        away_policy_target_entries: artifact.away_target_entries.len(),
    };

    Ok(SoccerLearningCompletedGame {
        episode,
        seed,
        summary: episode_summary.summary.clone(),
        episode_summary,
        tactical_summary: artifact.tactical_summary,
        starting_tactical_learning,
        policies,
        score,
        delta,
        config_moments,
        pass_outcome_samples,
        neural_network,
        elapsed_seconds: started.elapsed().as_secs_f64(),
    })
}

pub fn run_soccer_learning_queue(
    config: SoccerLearningQueueRunnerConfig,
    initial_policies: SoccerTeamQPolicies,
) -> Result<SoccerLearningQueueReport, String> {
    run_soccer_learning_queue_with_observer(config, initial_policies, |_, _| Ok(()))
}

struct SoccerLearningQueueTask {
    episode: usize,
    match_config: MatchConfig,
    starting_policies: Arc<SoccerTeamQPolicies>,
    initial_neural_network: Option<Arc<SoccerNeuralNetworkSnapshot>>,
    neural_drain_timeout: Duration,
}

struct CachedSoccerTeamQPoliciesArc {
    fingerprint: u64,
    policies: Arc<SoccerTeamQPolicies>,
}

fn cached_soccer_team_policies_arc(
    policies: &SoccerTeamQPolicies,
    cache: &mut Option<CachedSoccerTeamQPoliciesArc>,
) -> Arc<SoccerTeamQPolicies> {
    let fingerprint = soccer_team_q_policies_fingerprint(policies);
    let rebuild_cache = match cache.as_ref() {
        Some(cached) => cached.fingerprint != fingerprint,
        None => true,
    };
    if rebuild_cache {
        *cache = Some(CachedSoccerTeamQPoliciesArc {
            fingerprint,
            policies: Arc::new(policies.clone()),
        });
    }
    cache
        .as_ref()
        .map(|cached| Arc::clone(&cached.policies))
        .expect("policy cache must exist after rebuild check")
}

struct CachedSoccerNeuralNetworkSnapshotArc {
    fingerprint: u64,
    snapshot: Arc<SoccerNeuralNetworkSnapshot>,
}

fn cached_soccer_neural_network_snapshot_arc(
    latest_neural_network: &Option<SoccerNeuralNetworkSnapshot>,
    cache: &mut Option<CachedSoccerNeuralNetworkSnapshotArc>,
) -> Option<Arc<SoccerNeuralNetworkSnapshot>> {
    let Some(snapshot) = latest_neural_network.as_ref() else {
        *cache = None;
        return None;
    };
    let fingerprint = soccer_neural_network_snapshot_fingerprint(snapshot);
    let rebuild_cache = match cache.as_ref() {
        Some(cached) => cached.fingerprint != fingerprint,
        None => true,
    };
    if rebuild_cache {
        *cache = Some(CachedSoccerNeuralNetworkSnapshotArc {
            fingerprint,
            snapshot: Arc::new(snapshot.clone()),
        });
    }
    cache.as_ref().map(|cached| Arc::clone(&cached.snapshot))
}

pub fn run_soccer_learning_queue_with_observer<F>(
    config: SoccerLearningQueueRunnerConfig,
    initial_policies: SoccerTeamQPolicies,
    mut on_completed_game: F,
) -> Result<SoccerLearningQueueReport, String>
where
    F: FnMut(&SoccerLearningCompletedGame, &SoccerTeamQPolicies) -> Result<(), String>,
{
    run_soccer_learning_queue_with_events(config, initial_policies, |event| {
        if let SoccerLearningQueueEvent::CompletedGame {
            game,
            merged_policies,
        } = event
        {
            on_completed_game(game, merged_policies)?;
        }
        Ok(())
    })
}

pub fn run_soccer_learning_queue_with_events<F>(
    config: SoccerLearningQueueRunnerConfig,
    initial_policies: SoccerTeamQPolicies,
    mut on_event: F,
) -> Result<SoccerLearningQueueReport, String>
where
    F: for<'a> FnMut(SoccerLearningQueueEvent<'a>) -> Result<(), String>,
{
    let started = Instant::now();
    let mut config = config;
    validate_soccer_q_policy_options_for_learning_run(&config.options)?;
    validate_soccer_neural_learning_config_for_learning_run(&config.match_config.neural_learning)?;
    validate_soccer_neural_blend_config_for_learning_run(&config.match_config.neural_blend)?;
    let parallel_games = config.parallel_games.clamp(1, 100);
    let (task_tx, task_rx) = mpsc::sync_channel::<SoccerLearningQueueTask>(parallel_games);
    let task_rx = Arc::new(Mutex::new(task_rx));
    let (result_tx, result_rx) = mpsc::channel();
    let mut workers = Vec::with_capacity(parallel_games);
    for _ in 0..parallel_games {
        let task_rx = Arc::clone(&task_rx);
        let result_tx = result_tx.clone();
        workers.push(thread::spawn(move || loop {
            let task = {
                let receiver = task_rx
                    .lock()
                    .expect("soccer learning queue task receiver poisoned");
                receiver.recv()
            };
            let Ok(task) = task else {
                break;
            };
            let result = run_soccer_learning_game_from_snapshot(
                task.episode,
                task.match_config,
                task.starting_policies,
                task.initial_neural_network,
                task.neural_drain_timeout,
            );
            let _ = result_tx.send((task.episode, result));
        }));
    }
    drop(result_tx);

    let mut active = 0usize;
    let mut next_episode = 0usize;
    let mut completed_games = 0usize;
    let mut failed_games = 0usize;
    let mut policies = initial_policies;
    let mut episode_summaries = Vec::with_capacity(config.games);
    let mut tactical_summary = SoccerTacticalLearningSummary::default();
    let mut total_home_goals = 0u32;
    let mut total_away_goals = 0u32;
    let mut latest_neural_network = config.initial_neural_network.clone();
    let mut starting_policy_arc_cache = None::<CachedSoccerTeamQPoliciesArc>;
    let mut latest_neural_network_arc_cache = None::<CachedSoccerNeuralNetworkSnapshotArc>;
    let mut first_error = None::<String>;

    while completed_games + failed_games < config.games && first_error.is_none() {
        if active < parallel_games && next_episode < config.games {
            while active < parallel_games && next_episode < config.games {
                if let Err(err) = on_event(SoccerLearningQueueEvent::StartingBatch {
                    next_episode,
                    match_config: &mut config.match_config,
                    policies: &mut policies,
                    neural_network: &mut latest_neural_network,
                }) {
                    first_error = Some(err);
                    break;
                }
                if let Err(err) = validate_soccer_neural_learning_config_for_learning_run(
                    &config.match_config.neural_learning,
                ) {
                    first_error = Some(format!(
                        "soccer learning queue neural config invalid for episode {next_episode}: {err}"
                    ));
                    break;
                }
                if let Err(err) = validate_soccer_neural_blend_config_for_learning_run(
                    &config.match_config.neural_blend,
                ) {
                    first_error = Some(format!(
                        "soccer learning queue neural blend config invalid for episode {next_episode}: {err}"
                    ));
                    break;
                }
                let starting_policies =
                    cached_soccer_team_policies_arc(&policies, &mut starting_policy_arc_cache);
                let starting_neural_network = cached_soccer_neural_network_snapshot_arc(
                    &latest_neural_network,
                    &mut latest_neural_network_arc_cache,
                );
                let episode = next_episode;
                let mut match_config = config.match_config.clone();
                match_config.seed = config.base_seed;
                let neural_drain_timeout = config.neural_drain_timeout;
                if let Err(err) = task_tx.send(SoccerLearningQueueTask {
                    episode,
                    match_config,
                    starting_policies: Arc::clone(&starting_policies),
                    initial_neural_network: starting_neural_network.as_ref().map(Arc::clone),
                    neural_drain_timeout,
                }) {
                    first_error = Some(format!("soccer learning queue task send failed: {err}"));
                    break;
                }
                active += 1;
                next_episode += 1;
            }
        }

        if first_error.is_some() {
            break;
        }

        let game_result = match result_rx.recv() {
            Ok((_, game_result)) => game_result,
            Err(err) => {
                first_error = Some(format!(
                    "soccer learning queue worker channel closed: {err}"
                ));
                break;
            }
        };
        active = active.saturating_sub(1);

        match game_result {
            Ok(mut game) => {
                let merged = match merge_soccer_policy_deltas(
                    &policies,
                    std::slice::from_ref(&game.delta),
                    1.0,
                ) {
                    Ok(merged) => merged,
                    Err(err) => {
                        first_error = Some(err);
                        break;
                    }
                };
                policies = merged;
                policies.prune(
                    config.prune_action_entries_per_team,
                    config.prune_target_entries_per_team,
                    config.min_policy_visits,
                );
                if let Err(err) = on_event(SoccerLearningQueueEvent::CompletedGame {
                    game: &game,
                    merged_policies: &mut policies,
                }) {
                    first_error = Some(err);
                    break;
                }
                if let Some(snapshot) = game.neural_network.take() {
                    latest_neural_network = Some(snapshot);
                }
                total_home_goals = total_home_goals.saturating_add(game.summary.score_home);
                total_away_goals = total_away_goals.saturating_add(game.summary.score_away);
                tactical_summary.merge(&game.tactical_summary);
                episode_summaries.push(game.episode_summary);
                completed_games += 1;
            }
            Err(err) => {
                if failed_games < 5 || failed_games + 1 == config.games {
                    eprintln!(
                        "soccer_learning_queue_game_failed failed_game={} error={err}",
                        failed_games + 1
                    );
                }
                failed_games += 1;
            }
        }
    }

    drop(task_tx);
    for worker in workers {
        if worker.join().is_err() && first_error.is_none() {
            first_error = Some("soccer learning queue worker thread panicked".to_string());
        }
    }
    if let Some(error) = first_error {
        return Err(error);
    }

    episode_summaries.sort_by_key(|summary| summary.episode);
    let final_policy_entries = policies.home.entries().len() + policies.away.entries().len();
    let final_target_entries =
        policies.home.target_entries().len() + policies.away.target_entries().len();

    Ok(SoccerLearningQueueReport {
        completed_games,
        failed_games,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        total_home_goals,
        total_away_goals,
        tactical_summary,
        final_policy_entries,
        final_target_entries,
        episode_summaries,
        final_policies: policies,
        latest_neural_network,
    })
}

pub fn soccer_self_play_artifact_from_queue_report(
    config: MatchConfig,
    options: SoccerQPolicyOptions,
    report: &SoccerLearningQueueReport,
) -> SoccerSelfPlayTrainingArtifact {
    SoccerSelfPlayTrainingArtifact {
        tactical_learning: config.tactical_learning.clone(),
        tactical_summary: report.tactical_summary.clone(),
        config,
        options,
        episodes: report.episode_summaries.clone(),
        home_entries: report.final_policies.home.entries(),
        home_target_entries: report.final_policies.home.target_entries(),
        away_entries: report.final_policies.away.entries(),
        away_target_entries: report.final_policies.away.target_entries(),
    }
}

fn collect_team_policy_delta(
    team: Team,
    before: &SoccerQPolicy,
    after: &SoccerQPolicy,
    merge_weight: f64,
    entries: &mut Vec<SoccerLearningPolicyDeltaEntry>,
) {
    let before_actions = entry_map(
        team,
        SoccerLearningPolicyEntryKind::Action,
        before.entries().into_iter().map(action_entry_value),
    );
    for entry in after.entries() {
        let value = action_entry_value(entry);
        let key = policy_entry_key(team, SoccerLearningPolicyEntryKind::Action, &value);
        let before_value = before_actions.get(&key);
        push_delta_entry(
            team,
            SoccerLearningPolicyEntryKind::Action,
            value,
            before_value,
            merge_weight,
            entries,
        );
    }

    let before_targets = entry_map(
        team,
        SoccerLearningPolicyEntryKind::Target,
        before.target_entries().into_iter().map(target_entry_value),
    );
    for entry in after.target_entries() {
        let value = target_entry_value(entry);
        let key = policy_entry_key(team, SoccerLearningPolicyEntryKind::Target, &value);
        let before_value = before_targets.get(&key);
        push_delta_entry(
            team,
            SoccerLearningPolicyEntryKind::Target,
            value,
            before_value,
            merge_weight,
            entries,
        );
    }
}

fn push_delta_entry(
    team: Team,
    entry_kind: SoccerLearningPolicyEntryKind,
    value: EntryValue,
    before_value: Option<&EntryValue>,
    merge_weight: f64,
    entries: &mut Vec<SoccerLearningPolicyDeltaEntry>,
) {
    let before_visits = before_value.map(|entry| entry.visits).unwrap_or(0);
    if value.visits <= before_visits {
        return;
    }
    let visit_delta = value.visits - before_visits;
    let before_q = before_value.map(|entry| entry.value).unwrap_or(0.0);
    let state_json = serde_json::to_value(&value.state_key)
        .unwrap_or_else(|_| Value::Object(Default::default()));
    let state_hash = state_hash(&state_json);
    let effective_visit_weight = f64::from(visit_delta) * merge_weight.max(0.0);
    entries.push(SoccerLearningPolicyDeltaEntry {
        team,
        entry_kind,
        state_hash,
        state_key: value.state_key,
        state_json,
        action: value.action,
        target_fine_cell_id: value.target_fine_cell_id,
        target_tactical_cell_id: value.target_tactical_cell_id,
        target_macro_cell_id: value.target_macro_cell_id,
        target_root_cell_id: value.target_root_cell_id,
        before_value: before_q,
        after_value: value.value,
        value_delta: value.value - before_q,
        before_value_micros: soccer_learning_to_micros(before_q),
        after_value_micros: soccer_learning_to_micros(value.value),
        value_delta_micros: soccer_learning_to_micros(value.value - before_q),
        visit_delta,
        merge_weight,
        merge_weight_micros: soccer_learning_to_micros(merge_weight),
        effective_visit_weight,
        effective_visit_micros: soccer_learning_to_micros(effective_visit_weight),
    });
}

fn entry_map(
    team: Team,
    entry_kind: SoccerLearningPolicyEntryKind,
    mut entries: impl Iterator<Item = EntryValue>,
) -> HashMap<PolicyEntryKey, EntryValue> {
    let (lower, upper) = entries.size_hint();
    let mut map = HashMap::with_capacity(upper.unwrap_or(lower));
    for entry in entries.by_ref() {
        map.insert(policy_entry_key(team, entry_kind, &entry), entry);
    }
    map
}

fn action_entry_value(entry: SoccerQEntry) -> EntryValue {
    EntryValue {
        state_key: entry.state,
        action: entry.action,
        value: entry.value,
        visits: entry.visits,
        target_fine_cell_id: -1,
        target_tactical_cell_id: -1,
        target_macro_cell_id: -1,
        target_root_cell_id: -1,
    }
}

fn target_entry_value(entry: SoccerQTargetEntry) -> EntryValue {
    EntryValue {
        state_key: entry.state,
        action: entry.action,
        value: entry.value,
        visits: entry.visits,
        target_fine_cell_id: entry.target_fine_cell_id as i32,
        target_tactical_cell_id: entry.target_tactical_cell_id as i32,
        target_macro_cell_id: entry.target_macro_cell_id as i32,
        target_root_cell_id: entry.target_root_cell_id as i32,
    }
}

fn policy_entry_key(
    team: Team,
    entry_kind: SoccerLearningPolicyEntryKind,
    value: &EntryValue,
) -> PolicyEntryKey {
    let state_json = serde_json::to_string(&value.state_key).unwrap_or_default();
    let state_hash = fnv1a_hex(state_json.as_bytes());
    PolicyEntryKey {
        team: soccer_team_label(team),
        entry_kind,
        state_hash,
        state_json,
        action: value.action.clone(),
        target_fine_cell_id: value.target_fine_cell_id,
        target_tactical_cell_id: value.target_tactical_cell_id,
        target_macro_cell_id: value.target_macro_cell_id,
        target_root_cell_id: value.target_root_cell_id,
    }
}

fn policy_delta_key(entry: &SoccerLearningPolicyDeltaEntry) -> PolicyEntryKey {
    let state_json = serde_json::to_string(&entry.state_key).unwrap_or_default();
    PolicyEntryKey {
        team: soccer_team_label(entry.team),
        entry_kind: entry.entry_kind,
        state_hash: entry.state_hash.clone(),
        state_json,
        action: entry.action.clone(),
        target_fine_cell_id: entry.target_fine_cell_id,
        target_tactical_cell_id: entry.target_tactical_cell_id,
        target_macro_cell_id: entry.target_macro_cell_id,
        target_root_cell_id: entry.target_root_cell_id,
    }
}

fn seed_policy_accumulators(
    team: Team,
    policy: &SoccerQPolicy,
    prior_weight: f64,
    action_accumulators: &mut BTreeMap<PolicyEntryKey, MergeAccumulator>,
    target_accumulators: &mut BTreeMap<PolicyEntryKey, MergeAccumulator>,
) {
    for entry in policy.entries().into_iter().map(action_entry_value) {
        seed_entry_accumulator(
            team,
            SoccerLearningPolicyEntryKind::Action,
            entry,
            prior_weight,
            action_accumulators,
        );
    }
    for entry in policy.target_entries().into_iter().map(target_entry_value) {
        seed_entry_accumulator(
            team,
            SoccerLearningPolicyEntryKind::Target,
            entry,
            prior_weight,
            target_accumulators,
        );
    }
}

fn seed_entry_accumulator(
    team: Team,
    entry_kind: SoccerLearningPolicyEntryKind,
    entry: EntryValue,
    prior_weight: f64,
    accumulators: &mut BTreeMap<PolicyEntryKey, MergeAccumulator>,
) {
    let effective_visits = f64::from(entry.visits.max(1)) * prior_weight.max(0.0);
    if effective_visits <= 0.0 {
        return;
    }
    let key = policy_entry_key(team, entry_kind, &entry);
    let item = accumulators.entry(key).or_insert_with(|| MergeAccumulator {
        state_key: entry.state_key.clone(),
        action: entry.action.clone(),
        weighted_value_sum: 0.0,
        effective_visits: 0.0,
        display_visits: 0,
        target_fine_cell_id: entry.target_fine_cell_id,
        target_tactical_cell_id: entry.target_tactical_cell_id,
        target_macro_cell_id: entry.target_macro_cell_id,
        target_root_cell_id: entry.target_root_cell_id,
    });
    item.weighted_value_sum += entry.value * effective_visits;
    item.effective_visits += effective_visits;
    item.display_visits = item.display_visits.saturating_add(entry.visits.max(1));
}

fn policy_action_entry_map(policy: &SoccerTeamQPolicies) -> HashMap<PolicyEntryKey, EntryValue> {
    let mut map = entry_map(
        Team::Home,
        SoccerLearningPolicyEntryKind::Action,
        policy.home.entries().into_iter().map(action_entry_value),
    );
    map.extend(entry_map(
        Team::Away,
        SoccerLearningPolicyEntryKind::Action,
        policy.away.entries().into_iter().map(action_entry_value),
    ));
    map
}

fn policy_target_entry_map(policy: &SoccerTeamQPolicies) -> HashMap<PolicyEntryKey, EntryValue> {
    let mut map = entry_map(
        Team::Home,
        SoccerLearningPolicyEntryKind::Target,
        policy
            .home
            .target_entries()
            .into_iter()
            .map(target_entry_value),
    );
    map.extend(entry_map(
        Team::Away,
        SoccerLearningPolicyEntryKind::Target,
        policy
            .away
            .target_entries()
            .into_iter()
            .map(target_entry_value),
    ));
    map
}

fn crossover_accumulators(
    accumulators: &mut BTreeMap<PolicyEntryKey, MergeAccumulator>,
    parent_maps: &[HashMap<PolicyEntryKey, EntryValue>],
    rng: &mut DeterministicRng,
    options: SoccerEvolutionOptions,
) {
    let crossover_rate = options.crossover_rate.clamp(0.0, 1.0);
    if parent_maps.len() < 2 || crossover_rate <= 0.0 {
        return;
    }
    for (key, accumulator) in accumulators.iter_mut() {
        if rng.next_f64() > crossover_rate || accumulator.effective_visits <= 0.0 {
            continue;
        }
        let start = rng.next_usize(parent_maps.len());
        for offset in 0..parent_maps.len() {
            let index = (start + offset) % parent_maps.len();
            let Some(parent_value) = parent_maps[index].get(key) else {
                continue;
            };
            accumulator.weighted_value_sum =
                parent_value.value.clamp(-120.0, 120.0) * accumulator.effective_visits;
            accumulator.display_visits = accumulator.display_visits.max(parent_value.visits.max(1));
            break;
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct PolicyNoveltyStateKey {
    team: &'static str,
    state_hash: String,
    state_json: String,
}

#[derive(Clone, Debug)]
struct PolicyNoveltyStateInfo {
    state_key: SoccerQStateKey,
    actions: Vec<String>,
    weighted_value_sum: f64,
    effective_visits: f64,
}

fn inject_policy_plateau_novelty_actions(
    accumulators: &mut BTreeMap<PolicyEntryKey, MergeAccumulator>,
    rng: &mut DeterministicRng,
    options: SoccerEvolutionOptions,
    pressure: f64,
) {
    let pressure = pressure.clamp(0.0, 1.0);
    if pressure <= 0.05 {
        return;
    }
    let search_can_create_novelty = options.exploration_rate > 0.0
        || options.exploration_scale > 0.0
        || options.mutation_rate > 0.0
        || options.mutation_scale > 0.0;
    if !search_can_create_novelty || accumulators.is_empty() {
        return;
    }

    let novelty_rate = (options.exploration_rate.max(options.mutation_rate) * 0.56
        + pressure * 0.54
        + options.exploration_scale.min(1.6) * 0.08)
        .clamp(0.0, 1.0);
    if novelty_rate <= 0.0 {
        return;
    }

    let mut states = BTreeMap::<PolicyNoveltyStateKey, PolicyNoveltyStateInfo>::new();
    for (key, accumulator) in accumulators.iter() {
        if key.entry_kind != SoccerLearningPolicyEntryKind::Action
            || accumulator.effective_visits <= 0.0
        {
            continue;
        }
        let state_key = PolicyNoveltyStateKey {
            team: key.team,
            state_hash: key.state_hash.clone(),
            state_json: key.state_json.clone(),
        };
        let info = states
            .entry(state_key)
            .or_insert_with(|| PolicyNoveltyStateInfo {
                state_key: accumulator.state_key.clone(),
                actions: Vec::new(),
                weighted_value_sum: 0.0,
                effective_visits: 0.0,
            });
        info.actions.push(key.action.clone());
        info.weighted_value_sum += accumulator.weighted_value_sum;
        info.effective_visits += accumulator.effective_visits;
    }

    let max_states =
        ((options.population_size.max(1) as f64) * (8.0 + pressure * 24.0)).ceil() as usize;
    for (state_key, info) in states
        .into_iter()
        .take(max_states.min(SOCCER_POLICY_NOVELTY_MAX_STATES))
    {
        if info.effective_visits <= 0.0 || rng.next_f64() > novelty_rate {
            continue;
        }
        let candidates = soccer_policy_novelty_actions_for_existing(&info.actions);
        if candidates.is_empty() {
            continue;
        }
        let current_value = info.weighted_value_sum / info.effective_visits;
        let visit_weight = (0.55 + pressure * 1.10 + options.exploration_scale.max(0.0) * 0.30)
            .clamp(0.5, SOCCER_POLICY_NOVELTY_VISIT_WEIGHT_MAX);
        let jitter_scale =
            (0.04 + options.exploration_scale.max(options.mutation_scale) * 0.06).clamp(0.04, 0.18);
        let start = rng.next_usize(candidates.len());
        let mut added = 0usize;
        for offset in 0..candidates.len() {
            if added >= SOCCER_POLICY_NOVELTY_MAX_ACTIONS_PER_STATE {
                break;
            }
            let action = candidates[(start + offset) % candidates.len()];
            if info.actions.iter().any(|existing| existing == action) {
                continue;
            }
            let novelty_key = PolicyEntryKey {
                team: state_key.team,
                entry_kind: SoccerLearningPolicyEntryKind::Action,
                state_hash: state_key.state_hash.clone(),
                state_json: state_key.state_json.clone(),
                action: action.to_string(),
                target_fine_cell_id: -1,
                target_tactical_cell_id: -1,
                target_macro_cell_id: -1,
                target_root_cell_id: -1,
            };
            if accumulators.contains_key(&novelty_key) {
                continue;
            }
            let centered_jitter = (rng.next_f64() * 2.0 - 1.0) * jitter_scale;
            let value = (current_value - 0.06 + centered_jitter).clamp(-120.0, 120.0);
            accumulators.insert(
                novelty_key,
                MergeAccumulator {
                    state_key: info.state_key.clone(),
                    action: action.to_string(),
                    weighted_value_sum: value * visit_weight,
                    effective_visits: visit_weight,
                    display_visits: 1,
                    target_fine_cell_id: -1,
                    target_tactical_cell_id: -1,
                    target_macro_cell_id: -1,
                    target_root_cell_id: -1,
                },
            );
            added += 1;
        }
    }
}

fn soccer_policy_novelty_actions_for_existing(existing: &[String]) -> Vec<&'static str> {
    let has_any = |families: &[&str]| {
        existing
            .iter()
            .any(|action| families.contains(&action.as_str()))
    };
    let mut candidates = Vec::<&'static str>::new();
    if has_any(SOCCER_POLICY_DRIBBLE_NOVELTY_ACTIONS) {
        candidates.extend_from_slice(SOCCER_POLICY_DRIBBLE_NOVELTY_ACTIONS);
    }
    if has_any(SOCCER_POLICY_PASS_NOVELTY_ACTIONS) {
        candidates.extend_from_slice(SOCCER_POLICY_PASS_NOVELTY_ACTIONS);
    }
    if has_any(SOCCER_POLICY_SHOT_NOVELTY_ACTIONS) {
        candidates.extend_from_slice(SOCCER_POLICY_SHOT_NOVELTY_ACTIONS);
    }
    if has_any(SOCCER_POLICY_DEFENSE_NOVELTY_ACTIONS) {
        candidates.extend_from_slice(SOCCER_POLICY_DEFENSE_NOVELTY_ACTIONS);
    }
    if candidates.is_empty() || has_any(SOCCER_POLICY_SUPPORT_NOVELTY_ACTIONS) {
        candidates.extend_from_slice(SOCCER_POLICY_SUPPORT_NOVELTY_ACTIONS);
    }
    candidates.sort_unstable();
    candidates.dedup();
    candidates
}

fn explore_accumulators(
    accumulators: &mut BTreeMap<PolicyEntryKey, MergeAccumulator>,
    rng: &mut DeterministicRng,
    options: SoccerEvolutionOptions,
) {
    let exploration_rate = options.exploration_rate.clamp(0.0, 1.0);
    let exploration_scale = options.exploration_scale.max(0.0);
    if exploration_rate <= 0.0 || exploration_scale <= 0.0 {
        return;
    }
    for accumulator in accumulators.values_mut() {
        if rng.next_f64() > exploration_rate || accumulator.effective_visits <= 0.0 {
            continue;
        }
        let current = accumulator.weighted_value_sum / accumulator.effective_visits;
        let sign = if rng.next_f64() < 0.5 { -1.0 } else { 1.0 };
        let magnitude = (0.25 + rng.next_f64() * 0.75) * exploration_scale;
        accumulator.weighted_value_sum =
            (current + sign * magnitude).clamp(-120.0, 120.0) * accumulator.effective_visits;
    }
}

fn evolve_weight(
    weight: &mut f64,
    directed_delta: f64,
    mutation_scale: f64,
    exploration_rate: f64,
    exploration_scale: f64,
    rng: &mut DeterministicRng,
    max_value: f64,
) {
    let mutation = if mutation_scale > 0.0 {
        (rng.next_f64() * 2.0 - 1.0) * mutation_scale
    } else {
        0.0
    };
    let exploration = if exploration_scale > 0.0 && rng.next_f64() < exploration_rate {
        (rng.next_f64() * 2.0 - 1.0) * exploration_scale
    } else {
        0.0
    };
    *weight = (*weight + directed_delta + mutation + exploration)
        .max(0.0)
        .min(max_value.max(0.0));
}

fn mutate_accumulators(
    accumulators: &mut BTreeMap<PolicyEntryKey, MergeAccumulator>,
    rng: &mut DeterministicRng,
    options: SoccerEvolutionOptions,
) {
    let mutation_rate = options.mutation_rate.clamp(0.0, 1.0);
    let mutation_scale = options.mutation_scale.max(0.0);
    for accumulator in accumulators.values_mut() {
        if rng.next_f64() > mutation_rate || accumulator.effective_visits <= 0.0 {
            continue;
        }
        let current = accumulator.weighted_value_sum / accumulator.effective_visits;
        let perturbation = (rng.next_f64() * 2.0 - 1.0) * mutation_scale;
        accumulator.weighted_value_sum =
            (current + perturbation).clamp(-120.0, 120.0) * accumulator.effective_visits;
    }
}

fn state_hash(state_json: &Value) -> String {
    let raw = serde_json::to_string(state_json).unwrap_or_default();
    fnv1a_hex(raw.as_bytes())
}

// 128-bit FNV-1a (32 lowercase hex). Must stay byte-for-byte identical to the
// persistence layer's soccer_learning_fnv1a_128_hex so engine-computed delta
// hashes and pg-computed entry hashes agree for the same serialized state.
fn fnv1a_hex(bytes: &[u8]) -> String {
    const OFFSET_BASIS: u128 = 0x6c62272e07bb0142_62b821756295c58d;
    const PRIME: u128 = 0x0000000001000000_000000000000013b;
    let mut hash = OFFSET_BASIS;
    for byte in bytes {
        hash ^= u128::from(*byte);
        hash = hash.wrapping_mul(PRIME);
    }
    format!("{hash:032x}")
}

fn candidate_seed(base_seed: u64, candidate_index: usize, salt: u64) -> u64 {
    if candidate_index == 0 {
        return base_seed;
    }
    let mut mixed = base_seed ^ salt;
    mixed ^= (candidate_index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    mixed = mixed.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    mixed ^ (mixed >> 29)
}

#[derive(Clone, Debug)]
struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9e3779b97f4a7c15,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut z = self.state;
        z ^= z >> 33;
        z = z.wrapping_mul(0xff51afd7ed558ccd);
        z ^= z >> 33;
        z = z.wrapping_mul(0xc4ceb9fe1a85ec53);
        z ^ (z >> 33)
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    fn next_usize(&mut self, upper_bound: usize) -> usize {
        if upper_bound == 0 {
            0
        } else {
            (self.next_u64() as usize) % upper_bound
        }
    }
}

fn build_policies_from_accumulators(
    home_options: SoccerQPolicyOptions,
    away_options: SoccerQPolicyOptions,
    action_accumulators: BTreeMap<PolicyEntryKey, MergeAccumulator>,
    target_accumulators: BTreeMap<PolicyEntryKey, MergeAccumulator>,
) -> Result<SoccerTeamQPolicies, String> {
    let action_capacity = action_accumulators.len();
    let target_capacity = target_accumulators.len();
    let mut home_entries = Vec::with_capacity(action_capacity.saturating_add(1) / 2);
    let mut away_entries = Vec::with_capacity(action_capacity / 2);
    let mut home_targets = Vec::with_capacity(target_capacity.saturating_add(1) / 2);
    let mut away_targets = Vec::with_capacity(target_capacity / 2);

    for (key, accumulator) in action_accumulators {
        if accumulator.effective_visits <= 0.0 {
            continue;
        }
        let entry = SoccerQEntry {
            state: accumulator.state_key,
            action: accumulator.action,
            value: accumulator.weighted_value_sum / accumulator.effective_visits,
            visits: accumulator.display_visits.max(1),
        };
        match key.team {
            "home" => home_entries.push(entry),
            "away" => away_entries.push(entry),
            _ => {}
        }
    }

    for (key, accumulator) in target_accumulators {
        if accumulator.effective_visits <= 0.0 {
            continue;
        }
        let entry = SoccerQTargetEntry {
            state: accumulator.state_key,
            action: accumulator.action,
            target_fine_cell_id: accumulator.target_fine_cell_id.max(0) as usize,
            target_tactical_cell_id: accumulator.target_tactical_cell_id.max(0) as usize,
            target_macro_cell_id: accumulator.target_macro_cell_id.max(0) as usize,
            target_root_cell_id: accumulator.target_root_cell_id.max(0) as usize,
            value: accumulator.weighted_value_sum / accumulator.effective_visits,
            visits: accumulator.display_visits.max(1),
        };
        match key.team {
            "home" => home_targets.push(entry),
            "away" => away_targets.push(entry),
            _ => {}
        }
    }

    Ok(SoccerTeamQPolicies {
        home: SoccerQPolicy::from_entries_with_targets(home_options, &home_entries, &home_targets)?,
        away: SoccerQPolicy::from_entries_with_targets(away_options, &away_entries, &away_targets)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::general::soccer::{
        MatchStats, PlayerRole, SoccerNeuralBlendConfig, SoccerNeuralLayerSnapshot,
        SoccerNeuralLearningBackend, SoccerNeuralLearningConfig, TacticalPhase,
    };

    fn league_policies_with_alpha(alpha: f64) -> SoccerTeamQPolicies {
        let mut options = SoccerQPolicyOptions::default();
        options.alpha = alpha;
        SoccerTeamQPolicies::new(options)
    }

    #[test]
    fn league_insert_evicts_the_least_exploiting_member_over_capacity() {
        let mut league = SoccerPolicyLeague::new(2);
        league.insert(1, league_policies_with_alpha(0.1));
        league.insert(2, league_policies_with_alpha(0.2));
        // gen 1 beats the current policy a lot; gen 2 barely does.
        for _ in 0..10 {
            league.record_result(0, true);
            league.record_result(1, false);
        }
        // Adding a third snapshot evicts the weakest sparring partner (gen 2).
        league.insert(3, league_policies_with_alpha(0.3));
        let generations: Vec<u32> = league.members().iter().map(|m| m.generation).collect();
        assert_eq!(league.len(), 2);
        assert!(generations.contains(&1), "high-exploit member must survive");
        assert!(generations.contains(&3), "fresh member must be retained");
        assert!(
            !generations.contains(&2),
            "weakest sparring partner is evicted"
        );
    }

    #[test]
    fn league_insert_retains_a_fresh_snapshot_even_when_every_incumbent_dominates() {
        // PFSP steady state: a full league where EVERY incumbent beats the current
        // policy (exploit_rate > 0.5). A fresh snapshot (exploit_rate == 0.5) must
        // still be admitted — an *old* member is evicted, not the newcomer.
        let mut league = SoccerPolicyLeague::new(2);
        league.insert(1, league_policies_with_alpha(0.1));
        league.insert(2, league_policies_with_alpha(0.2));
        for _ in 0..10 {
            league.record_result(0, true); // gen 1 → 11/12 ≈ 0.92
            league.record_result(1, true); // gen 2 → 11/12 ≈ 0.92
        }
        league.insert(3, league_policies_with_alpha(0.3));
        let generations: Vec<u32> = league.members().iter().map(|m| m.generation).collect();
        assert_eq!(league.len(), 2);
        assert!(
            generations.contains(&3),
            "fresh snapshot must survive even when all incumbents dominate: {generations:?}"
        );
        assert!(
            generations.iter().filter(|&&g| g == 1 || g == 2).count() == 1,
            "exactly one old generation should have been evicted: {generations:?}"
        );
    }

    #[test]
    fn league_pfsp_sampling_favors_members_that_beat_the_current_policy() {
        let mut league = SoccerPolicyLeague::new(4);
        league.insert(1, league_policies_with_alpha(0.1)); // strong exploiter
        league.insert(2, league_policies_with_alpha(0.2)); // weak exploiter
        for _ in 0..10 {
            league.record_result(0, true);
            league.record_result(1, false);
        }
        let mut rng = SeededRandom::new(99);
        let mut counts = [0u32; 2];
        for _ in 0..400 {
            if let Some(index) = league.sample_opponent_index(&mut rng) {
                counts[index] += 1;
            }
        }
        assert!(
            counts[0] > counts[1] * 2,
            "the stronger exploiter should be sampled far more: {counts:?}"
        );
        // Exploitability is the worst-case exploit rate (gen 1 ≈ 11/12).
        assert!(league.exploitability() > 0.8);
    }

    #[test]
    fn league_matchup_pits_current_against_a_frozen_side() {
        let current = league_policies_with_alpha(0.1);
        let mut league = SoccerPolicyLeague::new(2);
        league.insert(1, league_policies_with_alpha(0.9));
        let member = &league.members()[0];
        // Current plays home: home is current (α 0.1), away is the frozen member (α 0.9).
        let home_side = soccer_league_matchup_policies(&current, member, true);
        assert!((home_side.home.options.alpha - 0.1).abs() < 1e-9);
        assert!((home_side.away.options.alpha - 0.9).abs() < 1e-9);
        // Current plays away: the sides swap.
        let away_side = soccer_league_matchup_policies(&current, member, false);
        assert!((away_side.home.options.alpha - 0.9).abs() < 1e-9);
        assert!((away_side.away.options.alpha - 0.1).abs() < 1e-9);
    }

    fn test_state() -> SoccerQStateKey {
        serde_json::from_value(serde_json::json!({
            "phase": "Kickoff",
            "role": "Midfielder",
            "possessionRelative": 1,
            "ballZoneX": 2,
            "ballZoneY": 3,
            "scoreDiffBucket": 0,
            "hasBall": true,
            "visibleBall": true,
            "shotLaneOpen": false,
            "visiblePassOptionsBin": 2,
            "ballDistanceBin": 1,
            "yardsToGoalBin": 5,
            "pressureBin": 1,
            "openSpaceBin": 3
        }))
        .expect("test state")
    }

    fn policy_with_action(value: f64, visits: u32) -> SoccerTeamQPolicies {
        let state = test_state();
        let entries = vec![SoccerQEntry {
            state,
            action: "pass".to_string(),
            value,
            visits,
        }];
        SoccerTeamQPolicies {
            home: SoccerQPolicy::from_entries(SoccerQPolicyOptions::default(), &entries)
                .expect("home policy"),
            away: SoccerQPolicy::new(SoccerQPolicyOptions::default()),
        }
    }

    fn policy_with_home_actions(actions: &[(&str, f64, u32)]) -> SoccerTeamQPolicies {
        let state = test_state();
        let entries = actions
            .iter()
            .map(|(action, value, visits)| SoccerQEntry {
                state: state.clone(),
                action: (*action).to_string(),
                value: *value,
                visits: *visits,
            })
            .collect::<Vec<_>>();
        SoccerTeamQPolicies {
            home: SoccerQPolicy::from_entries(SoccerQPolicyOptions::default(), &entries)
                .expect("home policy"),
            away: SoccerQPolicy::new(SoccerQPolicyOptions::default()),
        }
    }

    #[test]
    fn losing_team_delta_is_weighted_below_winning_team_delta() {
        let score = soccer_learning_run_score(&MatchSummary {
            score_home: 3,
            score_away: 4,
            ticks: 10,
            simulated_seconds: 1.0,
            stats: Default::default(),
        });

        assert_eq!(score.home.outcome, SoccerLearningOutcome::Loss);
        assert_eq!(score.away.outcome, SoccerLearningOutcome::Win);
        assert!(score.home.merge_weight < score.away.merge_weight);
        assert!(score.home.merge_weight < 0.30);
    }

    #[test]
    fn match_fitness_preserves_goal_difference_for_evolution() {
        let dominant_win = soccer_learning_run_score(&MatchSummary {
            score_home: 5,
            score_away: 0,
            ticks: 10,
            simulated_seconds: 1.0,
            stats: Default::default(),
        });
        let one_goal_win = soccer_learning_run_score(&MatchSummary {
            score_home: 1,
            score_away: 0,
            ticks: 10,
            simulated_seconds: 1.0,
            stats: Default::default(),
        });
        let chaotic_draw = soccer_learning_run_score(&MatchSummary {
            score_home: 3,
            score_away: 3,
            ticks: 10,
            simulated_seconds: 1.0,
            stats: Default::default(),
        });

        assert!(
            dominant_win.match_fitness > chaotic_draw.match_fitness,
            "5-0 must outrank 3-3 for evolution: win={}, draw={}",
            dominant_win.match_fitness,
            chaotic_draw.match_fitness
        );
        assert!(
            one_goal_win.match_fitness > chaotic_draw.match_fitness,
            "1-0 must still preserve goal-diff signal over high-scoring churn: win={}, draw={}",
            one_goal_win.match_fitness,
            chaotic_draw.match_fitness
        );
    }

    #[test]
    fn match_fitness_uses_non_cancelling_play_quality_for_draws() {
        let high_quality = soccer_learning_run_score(&MatchSummary {
            score_home: 1,
            score_away: 1,
            ticks: 10,
            simulated_seconds: 1.0,
            stats: MatchStats {
                passes_attempted_home: 42,
                passes_attempted_away: 40,
                passes_completed_home: 34,
                passes_completed_away: 31,
                passes_completed_forward_home: 18,
                passes_completed_forward_away: 16,
                shots_home: 6,
                shots_away: 5,
                shots_on_target_home: 4,
                shots_on_target_away: 3,
                dribble_beats_home: 5,
                dribble_beats_away: 4,
                loose_ball_recoveries_home: 8,
                loose_ball_recoveries_away: 7,
                teamwork_upfield_progress_home: 48.0,
                teamwork_upfield_progress_away: 44.0,
                teamwork_near_ball_progress_home: 30.0,
                teamwork_near_ball_progress_away: 28.0,
                ..Default::default()
            },
        });
        let low_quality = soccer_learning_run_score(&MatchSummary {
            score_home: 1,
            score_away: 1,
            ticks: 10,
            simulated_seconds: 1.0,
            stats: MatchStats {
                passes_attempted_home: 42,
                passes_attempted_away: 40,
                passes_completed_home: 15,
                passes_completed_away: 14,
                passes_completed_forward_home: 2,
                passes_completed_forward_away: 2,
                shots_home: 1,
                shots_away: 1,
                ..Default::default()
            },
        });

        assert!(
            high_quality.match_fitness > low_quality.match_fitness + 0.05,
            "play quality should shape symmetric draws: high={}, low={}",
            high_quality.match_fitness,
            low_quality.match_fitness
        );
    }

    #[test]
    fn extracts_only_new_visit_deltas() {
        let before = policy_with_action(1.0, 2);
        let after = policy_with_action(3.0, 5);
        let score = soccer_learning_run_score(&MatchSummary {
            score_home: 2,
            score_away: 0,
            ticks: 10,
            simulated_seconds: 1.0,
            stats: Default::default(),
        });
        let delta = soccer_policy_delta_entries(&before, &after, &score);

        assert_eq!(delta.entries.len(), 1);
        assert_eq!(delta.entries[0].visit_delta, 3);
        assert_eq!(delta.entries[0].before_value, 1.0);
        assert_eq!(delta.entries[0].after_value, 3.0);
    }

    #[test]
    fn merge_prefers_outcome_weighted_winner_values() {
        let base = policy_with_action(0.0, 1);
        let mut winning = soccer_policy_delta_entries(
            &base,
            &policy_with_action(4.0, 4),
            &soccer_learning_run_score(&MatchSummary {
                score_home: 2,
                score_away: 0,
                ticks: 10,
                simulated_seconds: 1.0,
                stats: Default::default(),
            }),
        );
        let mut losing = soccer_policy_delta_entries(
            &base,
            &policy_with_action(-4.0, 4),
            &soccer_learning_run_score(&MatchSummary {
                score_home: 0,
                score_away: 2,
                ticks: 10,
                simulated_seconds: 1.0,
                stats: Default::default(),
            }),
        );
        winning.entries.append(&mut losing.entries);

        let merged = merge_soccer_policy_deltas(&base, &[winning], 0.0).expect("merged policy");
        let value = merged.home.entries()[0].value;

        assert!(value > 2.0, "winning signal should dominate, got {value}");
    }

    #[test]
    fn evolution_blends_elite_parent_weights() {
        let low = policy_with_action(0.0, 1);
        let high = policy_with_action(10.0, 1);
        let options = SoccerEvolutionOptions {
            mutation_rate: 0.0,
            mutation_scale: 0.0,
            crossover_rate: 0.0,
            exploration_rate: 0.0,
            exploration_scale: 0.0,
            elite_weight_floor: 0.0,
            population_size: 1,
            seed: 7,
        };

        let evolved =
            evolve_soccer_team_policies(&[(&low, 1.0), (&high, 3.0)], options).expect("evolved");
        let value = evolved.home.entries()[0].value;

        assert!((value - 7.5).abs() < 1e-9, "weighted value was {value}");
    }

    #[test]
    fn evolution_crossover_samples_parent_values() {
        let left = policy_with_action(-4.0, 1);
        let right = policy_with_action(4.0, 1);
        let options = SoccerEvolutionOptions {
            mutation_rate: 0.0,
            mutation_scale: 0.0,
            crossover_rate: 1.0,
            exploration_rate: 0.0,
            exploration_scale: 0.0,
            elite_weight_floor: 0.0,
            population_size: 1,
            seed: 11,
        };

        let evolved =
            evolve_soccer_team_policies(&[(&left, 1.0), (&right, 1.0)], options).expect("evolved");
        let value = evolved.home.entries()[0].value;

        assert!(
            (value - -4.0).abs() < 1e-9 || (value - 4.0).abs() < 1e-9,
            "crossover should inherit a parent value, got {value}"
        );
    }

    #[test]
    fn evolution_population_search_keeps_best_policy_candidate() {
        let low = policy_with_action(0.5, 4);
        let high = policy_with_action(3.0, 4);
        let single_options = SoccerEvolutionOptions {
            mutation_rate: 1.0,
            mutation_scale: 0.35,
            crossover_rate: 1.0,
            exploration_rate: 1.0,
            exploration_scale: 0.45,
            elite_weight_floor: 0.0,
            population_size: 1,
            seed: 41,
        };
        let population_options = SoccerEvolutionOptions {
            population_size: 8,
            ..single_options
        };

        let single = evolve_soccer_team_policies(&[(&low, 1.0), (&high, 2.0)], single_options)
            .expect("single candidate");
        let population =
            evolve_soccer_team_policies(&[(&low, 1.0), (&high, 2.0)], population_options)
                .expect("population candidate");

        assert!(
            soccer_team_policy_search_score(&population)
                >= soccer_team_policy_search_score(&single) - 1e-12
        );
    }

    #[test]
    fn policy_search_adapts_budget_when_parent_fitness_plateaus() {
        let low_coverage = policy_with_home_actions(&[("pass", 0.8, 4)]);
        let pressure =
            soccer_policy_search_pressure(&[(&low_coverage, 0.40), (&low_coverage, 0.41)]);
        assert!(
            pressure > 0.40,
            "flat low-coverage parents should create anti-plateau pressure, got {pressure}"
        );

        let base = SoccerEvolutionOptions::default();
        let adapted = adapt_soccer_evolution_options_for_policy_search(base, pressure);
        assert!(adapted.population_size > base.population_size);
        assert!(adapted.mutation_rate > base.mutation_rate);
        assert!(adapted.mutation_scale > base.mutation_scale);
        assert!(adapted.exploration_rate > base.exploration_rate);
        assert!(adapted.exploration_scale > base.exploration_scale);

        let disabled = SoccerEvolutionOptions {
            mutation_rate: 0.0,
            mutation_scale: 0.0,
            crossover_rate: 0.0,
            exploration_rate: 0.0,
            exploration_scale: 0.0,
            elite_weight_floor: 0.0,
            population_size: 1,
            seed: 77,
        };
        assert_eq!(
            adapt_soccer_evolution_options_for_policy_search(disabled, pressure).population_size,
            disabled.population_size
        );
    }

    #[test]
    fn policy_search_adapted_population_cap_parser_validates_positive_values() {
        assert_eq!(
            parse_soccer_policy_search_max_adapted_population(None).unwrap(),
            SOCCER_POLICY_SEARCH_MAX_ADAPTED_POPULATION
        );
        assert_eq!(
            parse_soccer_policy_search_max_adapted_population(Some("12")).unwrap(),
            12
        );
        assert!(parse_soccer_policy_search_max_adapted_population(Some("0")).is_err());
        assert!(parse_soccer_policy_search_max_adapted_population(Some("nope")).is_err());
    }

    #[test]
    fn policy_search_adapted_population_respects_process_cap() {
        let base = SoccerEvolutionOptions {
            population_size: 16,
            ..SoccerEvolutionOptions::default()
        };
        let pressure = 1.0;
        let adapted = adapt_soccer_evolution_options_for_policy_search(base, pressure);

        assert!(adapted.population_size <= soccer_policy_search_max_adapted_population());
    }

    #[test]
    fn policy_plateau_novelty_injects_missing_sibling_actions() {
        let policy = policy_with_home_actions(&[("pass", 1.0, 6)]);
        let mut action_accumulators = BTreeMap::<PolicyEntryKey, MergeAccumulator>::new();
        let mut target_accumulators = BTreeMap::<PolicyEntryKey, MergeAccumulator>::new();
        seed_policy_accumulators(
            Team::Home,
            &policy.home,
            1.0,
            &mut action_accumulators,
            &mut target_accumulators,
        );
        let mut rng = DeterministicRng::new(90210);
        let options = SoccerEvolutionOptions {
            mutation_rate: 0.25,
            mutation_scale: 0.55,
            exploration_rate: 1.0,
            exploration_scale: 1.0,
            crossover_rate: 0.0,
            elite_weight_floor: 0.0,
            population_size: 8,
            seed: 90210,
        };

        inject_policy_plateau_novelty_actions(&mut action_accumulators, &mut rng, options, 1.0);

        assert!(
            action_accumulators
                .keys()
                .any(|key| key.action != "pass" && SOCCER_POLICY_PASS_NOVELTY_ACTIONS.contains(&key.action.as_str())),
            "plateau novelty should add at least one pass-family sibling action: {action_accumulators:?}"
        );
    }

    #[test]
    fn policy_search_score_rewards_dp_action_coverage_when_values_tie() {
        let one_action = policy_with_home_actions(&[("pass", 1.0, 4)]);
        let covered = policy_with_home_actions(&[
            ("pass", 1.0, 4),
            ("killer-pass", 1.0, 4),
            ("recycle-reset", 1.0, 4),
        ]);

        assert!(
            soccer_team_policy_search_score(&covered)
                > soccer_team_policy_search_score(&one_action),
            "equal-value policies should prefer broader DP action coverage"
        );
    }

    #[test]
    fn default_evolution_options_keep_genetic_search_broad() {
        let options = SoccerEvolutionOptions::default();

        assert!(options.population_size >= 8);
        assert!(options.mutation_rate >= 0.04);
        assert!(options.mutation_scale >= 0.20);
        assert!(options.crossover_rate >= 0.40);
        assert!(options.exploration_rate >= 0.05);
        assert!(options.exploration_scale >= 0.50);
    }

    #[test]
    fn curriculum_stage_advances_monotonically_by_completed_games() {
        let config = SoccerLearningCurriculumConfig {
            ball_skills_after_games: 2,
            duels_after_games: 4,
            small_sided_after_games: 6,
            team_shape_after_games: 8,
            full_match_after_games: 10,
        };

        validate_soccer_learning_curriculum_config_for_learning_run(&config)
            .expect("monotonic curriculum");
        assert_eq!(
            soccer_learning_curriculum_stage_for_completed_games(0, &config),
            SoccerLearningCurriculumStage::Locomotion
        );
        assert_eq!(
            soccer_learning_curriculum_stage_for_completed_games(2, &config),
            SoccerLearningCurriculumStage::BallSkills
        );
        assert_eq!(
            soccer_learning_curriculum_stage_for_completed_games(5, &config),
            SoccerLearningCurriculumStage::Duels
        );
        assert_eq!(
            soccer_learning_curriculum_stage_for_completed_games(10, &config),
            SoccerLearningCurriculumStage::FullMatch
        );

        let invalid = SoccerLearningCurriculumConfig {
            ball_skills_after_games: 5,
            duels_after_games: 4,
            ..config
        };
        assert!(validate_soccer_learning_curriculum_config_for_learning_run(&invalid).is_err());
    }

    fn curriculum_test_base_config() -> MatchConfig {
        let mut config = MatchConfig::default();
        config.field_length_yards = 120.0;
        config.field_width_yards = 80.0;
        config.duration_seconds = 600.0;
        config.half_duration_seconds = 300.0;
        config.formation_lp_enabled = true;
        config.neural_learning.mappo_team_reward_share = 0.8;
        config
    }

    #[test]
    fn curriculum_stage_shaping_ramps_monotonically_and_full_match_is_identity() {
        use SoccerLearningCurriculumStage::*;
        let stages = [
            Locomotion, BallSkills, Duels, SmallSided, TeamShape, FullMatch,
        ];
        let mut previous: Option<SoccerCurriculumStageShaping> = None;
        for stage in stages {
            let shaping = soccer_curriculum_stage_shaping(stage);
            assert!((0.0..=1.0).contains(&shaping.pitch_fraction));
            assert!((0.0..=1.0).contains(&shaping.duration_fraction));
            assert!((0.0..=1.0).contains(&shaping.team_reward_share_fraction));
            if let Some(prev) = previous {
                assert!(shaping.pitch_fraction >= prev.pitch_fraction);
                assert!(shaping.duration_fraction >= prev.duration_fraction);
                assert!(shaping.team_reward_share_fraction >= prev.team_reward_share_fraction);
            }
            previous = Some(shaping);
        }
        // FullMatch is the identity end of the ramp.
        let full = soccer_curriculum_stage_shaping(FullMatch);
        assert_eq!(full.pitch_fraction, 1.0);
        assert_eq!(full.duration_fraction, 1.0);
        assert_eq!(full.team_reward_share_fraction, 1.0);
        assert!(full.formation_lp);
    }

    #[test]
    fn applying_full_match_stage_leaves_config_byte_identical() {
        let base = curriculum_test_base_config();
        let mut config = base.clone();
        apply_soccer_curriculum_stage_to_match_config(
            SoccerLearningCurriculumStage::FullMatch,
            &mut config,
        );
        assert_eq!(config.field_length_yards, base.field_length_yards);
        assert_eq!(config.field_width_yards, base.field_width_yards);
        assert_eq!(config.duration_seconds, base.duration_seconds);
        assert_eq!(config.half_duration_seconds, base.half_duration_seconds);
        assert_eq!(config.formation_lp_enabled, base.formation_lp_enabled);
        assert!(
            (config.neural_learning.mappo_team_reward_share
                - base.neural_learning.mappo_team_reward_share)
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn applying_early_stage_shrinks_pitch_duration_and_individualises_reward() {
        let mut config = curriculum_test_base_config();
        // Locomotion = 0.25 pitch/duration, 0.0 team share, formation off.
        apply_soccer_curriculum_stage_to_match_config(
            SoccerLearningCurriculumStage::Locomotion,
            &mut config,
        );
        // 120*0.25=30 < 40yd floor, so the length floors to the minimum box.
        assert_eq!(config.field_length_yards, 40.0);
        assert_eq!(config.field_width_yards, 30.0);
        // 600*0.25=150 (above the 20s floor), half 300*0.25=75.
        assert_eq!(config.duration_seconds, 150.0);
        assert_eq!(config.half_duration_seconds, 75.0);
        // Individual reward only, and the formation LP disengaged.
        assert_eq!(config.neural_learning.mappo_team_reward_share, 0.0);
        assert!(!config.formation_lp_enabled);

        // Small-sided keeps formation on (AND with true) and partially shares the reward.
        let mut shaped = curriculum_test_base_config();
        apply_soccer_curriculum_stage_to_match_config(
            SoccerLearningCurriculumStage::SmallSided,
            &mut shaped,
        );
        assert!(shaped.formation_lp_enabled);
        assert!((shaped.neural_learning.mappo_team_reward_share - 0.8 * 0.35).abs() < 1e-12);
        assert!(shaped.field_length_yards < 120.0 && shaped.field_length_yards > 40.0);
    }

    #[test]
    fn curriculum_is_inert_when_disabled() {
        // With the curriculum env unset (the default), the per-episode hook is a no-op and
        // leaves the config untouched — the pre-curriculum learner is byte-identical.
        if !soccer_curriculum_enabled_from_env() {
            let base = curriculum_test_base_config();
            let mut config = base.clone();
            assert!(maybe_apply_soccer_curriculum_for_episode(3, &mut config).is_none());
            assert_eq!(config.field_length_yards, base.field_length_yards);
            assert_eq!(config.duration_seconds, base.duration_seconds);
            assert_eq!(
                config.neural_learning.mappo_team_reward_share,
                base.neural_learning.mappo_team_reward_share
            );
        }
    }

    #[test]
    fn curriculum_episode_config_scales_training_rungs_without_compounding() {
        let curriculum = SoccerLearningCurriculumConfig {
            ball_skills_after_games: 2,
            duels_after_games: 4,
            small_sided_after_games: 6,
            team_shape_after_games: 8,
            full_match_after_games: 10,
        };
        let base = MatchConfig {
            duration_seconds: 600.0,
            half_duration_seconds: 300.0,
            field_width_yards: 80.0,
            field_length_yards: 120.0,
            local_mpc_enabled: true,
            local_mpc_max_players_per_team: 11,
            ..MatchConfig::default()
        };

        let (locomotion, locomotion_spec) =
            soccer_learning_curriculum_episode_config(&base, 0, &curriculum);
        assert_eq!(
            locomotion_spec.stage,
            SoccerLearningCurriculumStage::Locomotion
        );
        assert_eq!(locomotion_spec.drill_players_per_team, 2);
        assert!((locomotion.field_width_yards - 36.0).abs() < 1e-9);
        assert!((locomotion.field_length_yards - 54.0).abs() < 1e-9);
        assert!((locomotion.effective_duration_seconds() - 45.0).abs() < 1e-9);
        assert!((locomotion.half_duration_seconds - 45.0).abs() < 1e-9);
        assert_eq!(locomotion.local_mpc_max_players_per_team, 2);

        let (_, ball_skills_spec) =
            soccer_learning_curriculum_episode_config(&base, 2, &curriculum);
        let (_, duels_spec) = soccer_learning_curriculum_episode_config(&base, 4, &curriculum);
        let (_, small_sided_spec) =
            soccer_learning_curriculum_episode_config(&base, 6, &curriculum);
        let (_, team_shape_spec) = soccer_learning_curriculum_episode_config(&base, 8, &curriculum);
        let (full_match, full_match_spec) =
            soccer_learning_curriculum_episode_config(&base, 10, &curriculum);
        assert_eq!(ball_skills_spec.drill_players_per_team, 3);
        assert_eq!(duels_spec.drill_players_per_team, 4);
        assert_eq!(small_sided_spec.drill_players_per_team, 5);
        assert_eq!(team_shape_spec.drill_players_per_team, 8);
        assert_eq!(full_match_spec.drill_players_per_team, 11);
        assert_eq!(
            full_match_spec.stage,
            SoccerLearningCurriculumStage::FullMatch
        );
        assert!((full_match.field_width_yards - base.field_width_yards).abs() < 1e-9);
        assert!((full_match.field_length_yards - base.field_length_yards).abs() < 1e-9);
        assert!(
            (full_match.effective_duration_seconds() - base.effective_duration_seconds()).abs()
                < 1e-9
        );
        assert_eq!(base.local_mpc_max_players_per_team, 11);

        let tiny_base = MatchConfig {
            field_width_yards: 12.0,
            field_length_yards: 18.0,
            ..base.clone()
        };
        let (tiny_full_match, _) =
            soccer_learning_curriculum_episode_config(&tiny_base, 10, &curriculum);
        assert!((tiny_full_match.field_width_yards - 12.0).abs() < 1e-9);
        assert!((tiny_full_match.field_length_yards - 18.0).abs() < 1e-9);
    }

    #[test]
    fn curriculum_episode_config_preserves_short_test_games() {
        let curriculum = SoccerLearningCurriculumConfig::default();
        let base = MatchConfig {
            duration_seconds: 0.2,
            half_duration_seconds: 0.0,
            field_width_yards: 80.0,
            field_length_yards: 120.0,
            local_mpc_enabled: true,
            local_mpc_max_players_per_team: 3,
            ..MatchConfig::default()
        };

        let (episode_config, spec) =
            soccer_learning_curriculum_episode_config(&base, 0, &curriculum);
        assert_eq!(spec.stage, SoccerLearningCurriculumStage::Locomotion);
        assert!((episode_config.effective_duration_seconds() - 0.2).abs() < 1e-9);
        assert_eq!(episode_config.half_duration_seconds, 0.0);
        assert_eq!(episode_config.local_mpc_max_players_per_team, 2);
        assert!((base.field_width_yards - 80.0).abs() < 1e-9);
        assert!((base.field_length_yards - 120.0).abs() < 1e-9);
    }

    #[test]
    fn promotion_gate_uses_quality_and_sample_floor_before_activation() {
        use crate::des::general::soccer::MatchStats;

        let mut strong_stats = MatchStats::default();
        strong_stats.shots_home = 8;
        strong_stats.shots_away = 7;
        strong_stats.shots_on_target_home = 5;
        strong_stats.shots_on_target_away = 4;
        strong_stats.passes_attempted_home = 50;
        strong_stats.passes_attempted_away = 48;
        strong_stats.passes_completed_home = 38;
        strong_stats.passes_completed_away = 36;
        strong_stats.passes_completed_forward_home = 18;
        strong_stats.passes_completed_forward_away = 17;
        strong_stats.assists_home = 2;
        strong_stats.assists_away = 1;
        strong_stats.shots_after_pass_home = 4;
        strong_stats.shots_after_pass_away = 3;
        strong_stats.pass_chain_gain_yards_home = 90.0;
        strong_stats.pass_chain_gain_yards_away = 80.0;
        let strong = MatchSummary {
            score_home: 2,
            score_away: 1,
            ticks: 100,
            simulated_seconds: 90.0,
            stats: strong_stats,
        };

        let strict = SoccerPolicyPromotionGateConfig {
            min_sample_games: 2,
            min_mean_match_fitness: 0.0,
            min_best_match_fitness: 0.5,
            min_mean_play_quality: 0.05,
            max_mean_conceded_goals: 3.0,
            max_mean_goal_margin: 3.0,
            max_mean_chain_net_loss: 2.0,
            ..SoccerPolicyPromotionGateConfig::default()
        };
        let one_sample = evaluate_soccer_policy_promotion_gate([&strong], strict);
        assert!(!one_sample.eligible);
        assert!(one_sample
            .rejection_reasons
            .iter()
            .any(|reason| reason.contains("sample_games")));

        let two_samples = evaluate_soccer_policy_promotion_gate([&strong, &strong], strict);
        assert!(two_samples.eligible, "{two_samples:?}");

        let mut poor_stats = MatchStats::default();
        poor_stats.pass_chains_net_loss_home = 9;
        poor_stats.pass_chains_net_loss_away = 9;
        let poor = MatchSummary {
            score_home: 0,
            score_away: 5,
            ticks: 100,
            simulated_seconds: 90.0,
            stats: poor_stats,
        };
        let poor_eval = evaluate_soccer_policy_promotion_gate([&poor, &poor], strict);
        assert!(!poor_eval.eligible);
        assert!(poor_eval.rejection_reasons.iter().any(|reason| {
            reason.contains("mean_goal_margin") || reason.contains("mean_chain_net_loss")
        }));

        let mut dirty = strong.clone();
        dirty.stats.pass_chain_gain_yards_home = f64::NAN;
        let dirty_eval = evaluate_soccer_policy_promotion_gate([&strong, &dirty], strict);
        assert!(!dirty_eval.eligible);
        assert!(dirty_eval.mean_match_fitness.is_finite());
        assert!(dirty_eval.mean_play_quality.is_finite());
        assert!(dirty_eval.mean_chain_net_loss.is_finite());
        serde_json::to_value(&dirty_eval).expect("dirty gate evaluation remains JSON-safe");
        assert!(dirty_eval
            .rejection_reasons
            .iter()
            .any(|reason| reason.contains("non_finite_learning_metrics")));
    }

    #[test]
    fn learning_run_validators_accept_enabled_neural_and_default_evolution() {
        let neural = SoccerNeuralLearningConfig {
            enabled: true,
            ..SoccerNeuralLearningConfig::default()
        };

        validate_soccer_neural_learning_config_for_learning_run(&neural)
            .expect("default enabled neural learning config should be valid");
        validate_soccer_neural_blend_config_for_learning_run(&SoccerNeuralBlendConfig::default())
            .expect("default neural blend config should be valid");
        validate_soccer_evolution_options_for_learning_run(&SoccerEvolutionOptions::default())
            .expect("default evolution options should be valid");
        validate_soccer_q_policy_options_for_learning_run(&SoccerQPolicyOptions::default())
            .expect("default q-policy options should be valid");
    }

    #[test]
    fn learning_run_validator_rejects_neural_values_that_would_sanitize_silently() {
        let neural = SoccerNeuralLearningConfig {
            enabled: true,
            batch_size: 0,
            ..SoccerNeuralLearningConfig::default()
        };

        let err = validate_soccer_neural_learning_config_for_learning_run(&neural)
            .expect_err("zero neural batch size should fail fast");

        assert!(err.contains("batchSize"), "{err}");

        let neural = SoccerNeuralLearningConfig {
            enabled: true,
            learning_rate: MAX_SOCCER_NEURAL_LEARNING_RATE * 2.0,
            ..SoccerNeuralLearningConfig::default()
        };
        let err = validate_soccer_neural_learning_config_for_learning_run(&neural)
            .expect_err("above-cap neural learning rate should fail fast");

        assert!(err.contains("learningRate"), "{err}");

        let neural = SoccerNeuralLearningConfig {
            enabled: true,
            target_scale: -1.0,
            ..SoccerNeuralLearningConfig::default()
        };
        let err = validate_soccer_neural_learning_config_for_learning_run(&neural)
            .expect_err("negative neural target scale should fail fast");

        assert!(err.contains("targetScale"), "{err}");

        let neural = SoccerNeuralLearningConfig {
            enabled: true,
            marl_team_reward_weight: 1.25,
            ..SoccerNeuralLearningConfig::default()
        };
        let err = validate_soccer_neural_learning_config_for_learning_run(&neural)
            .expect_err("out-of-range MARL team reward weight should fail fast");

        assert!(err.contains("marlTeamRewardWeight"), "{err}");

        let neural = SoccerNeuralLearningConfig {
            enabled: true,
            mappo_clip_epsilon: 0.0,
            ..SoccerNeuralLearningConfig::default()
        };
        let err = validate_soccer_neural_learning_config_for_learning_run(&neural)
            .expect_err("zero MAPPO clip epsilon should fail fast");

        assert!(err.contains("mappoClipEpsilon"), "{err}");

        let neural = SoccerNeuralLearningConfig {
            enabled: true,
            mappo_clip_epsilon: 0.005,
            ..SoccerNeuralLearningConfig::default()
        };
        let err = validate_soccer_neural_learning_config_for_learning_run(&neural)
            .expect_err("sub-minimum MAPPO clip epsilon should fail fast");

        assert!(err.contains("mappoClipEpsilon"), "{err}");
    }

    #[test]
    fn learning_run_validator_rejects_mcts_values_that_would_sanitize_silently() {
        let valid = SoccerNeuralBlendConfig {
            mcts_enabled: true,
            ..SoccerNeuralBlendConfig::default()
        };
        validate_soccer_neural_blend_config_for_learning_run(&valid)
            .expect("default enabled MCTS config should be valid");

        for (name, blend) in [
            (
                "mctsSimulations",
                SoccerNeuralBlendConfig {
                    mcts_enabled: true,
                    mcts_simulations: 0,
                    ..valid
                },
            ),
            (
                "mctsCandidates",
                SoccerNeuralBlendConfig {
                    mcts_enabled: true,
                    mcts_candidates: 1,
                    ..valid
                },
            ),
            (
                "mctsDepth",
                SoccerNeuralBlendConfig {
                    mcts_enabled: true,
                    mcts_depth: 0,
                    ..valid
                },
            ),
            (
                "mctsExploration",
                SoccerNeuralBlendConfig {
                    mcts_enabled: true,
                    mcts_exploration: f64::INFINITY,
                    ..valid
                },
            ),
            (
                "mctsModelWeight",
                SoccerNeuralBlendConfig {
                    mcts_enabled: true,
                    mcts_model_weight: 1.5,
                    ..valid
                },
            ),
        ] {
            let err = validate_soccer_neural_blend_config_for_learning_run(&blend)
                .expect_err("invalid MCTS blend config should fail fast");
            assert!(err.contains(name), "{name}: {err}");
        }
    }

    #[test]
    fn learning_run_validator_bounds_mappo_team_reward_share() {
        // Default (0.0) is the individual-reward objective and must validate.
        let neural = SoccerNeuralLearningConfig {
            enabled: true,
            ..SoccerNeuralLearningConfig::default()
        };
        assert_eq!(neural.mappo_team_reward_share, 0.0);
        validate_soccer_neural_learning_config_for_learning_run(&neural)
            .expect("default mappo team-reward share should be valid");

        // A fully shared team reward (1.0) is the cooperative-MARL extreme and is allowed.
        let neural = SoccerNeuralLearningConfig {
            enabled: true,
            mappo_team_reward_share: 1.0,
            ..SoccerNeuralLearningConfig::default()
        };
        validate_soccer_neural_learning_config_for_learning_run(&neural)
            .expect("fully shared team reward should be valid");

        // Out of [0, 1] must fail fast rather than silently clamp.
        for bad in [-0.1, 1.5] {
            let neural = SoccerNeuralLearningConfig {
                enabled: true,
                mappo_team_reward_share: bad,
                ..SoccerNeuralLearningConfig::default()
            };
            let err = validate_soccer_neural_learning_config_for_learning_run(&neural)
                .expect_err("out-of-range mappo team-reward share should fail fast");
            assert!(err.contains("mappoTeamRewardShare"), "{err}");
        }
    }

    #[test]
    fn learning_run_validator_rejects_q_policy_exploration_epsilon_drift() {
        let options = SoccerQPolicyOptions {
            exploration_epsilon: 1.25,
            ..SoccerQPolicyOptions::default()
        };

        let err = validate_soccer_q_policy_options_for_learning_run(&options)
            .expect_err("exploration epsilon above one should fail fast");

        assert!(err.contains("explorationEpsilon"), "{err}");
    }

    #[test]
    fn learning_run_validator_rejects_out_of_range_evolution_options() {
        let bad_rate = SoccerEvolutionOptions {
            mutation_rate: 1.25,
            ..SoccerEvolutionOptions::default()
        };
        let err = validate_soccer_evolution_options_for_learning_run(&bad_rate)
            .expect_err("mutation rate above one should fail");
        assert!(err.contains("mutationRate"), "{err}");

        let huge_population = SoccerEvolutionOptions {
            population_size: SOCCER_EVOLUTION_MAX_POPULATION_SIZE + 1,
            ..SoccerEvolutionOptions::default()
        };
        let err = validate_soccer_evolution_options_for_learning_run(&huge_population)
            .expect_err("unbounded search population should fail");
        assert!(err.contains("populationSize"), "{err}");
    }

    #[test]
    fn evolution_metadata_caps_population_size_for_learning_run() {
        let metadata = serde_json::json!({
            "algorithm": "evolutionary-genetic-programming",
            "options": {
                "populationSize": SOCCER_EVOLUTION_MAX_POPULATION_SIZE + 99
            }
        });

        let options = soccer_evolution_options_from_search_metadata(
            Some(&metadata),
            SoccerEvolutionOptions::default(),
        )
        .expect("postgres search metadata options");

        assert_eq!(
            options.population_size,
            SOCCER_EVOLUTION_MAX_POPULATION_SIZE
        );
    }

    #[test]
    fn evolution_options_overlay_from_postgres_search_metadata() {
        let base = SoccerEvolutionOptions::default();
        let metadata = serde_json::json!({
            "algorithm": "evolutionary-genetic-programming",
            "options": {
                "mutationRate": 1.25,
                "mutationScale": 0.48,
                "crossoverRate": 0.72,
                "explorationRate": 0.18,
                "explorationScale": 0.95,
                "eliteWeightFloor": 0.08,
                "populationSize": 24,
                "seed": 991
            }
        });

        let options = soccer_evolution_options_from_search_metadata(Some(&metadata), base)
            .expect("postgres search metadata options");

        assert_eq!(options.mutation_rate, 1.0);
        assert_eq!(options.mutation_scale, 0.48);
        assert_eq!(options.crossover_rate, 0.72);
        assert_eq!(options.exploration_rate, 0.18);
        assert_eq!(options.exploration_scale, 0.95);
        assert_eq!(options.elite_weight_floor, 0.08);
        assert_eq!(options.population_size, 24);
        assert_eq!(options.seed, 991);
    }

    #[test]
    fn evolution_options_overlay_accepts_partial_direct_metadata() {
        let base = SoccerEvolutionOptions::default();
        let metadata = serde_json::json!({
            "populationSize": 0,
            "explorationScale": 0.75
        });

        let options = soccer_evolution_options_from_search_metadata(Some(&metadata), base)
            .expect("direct postgres options");

        assert_eq!(options.population_size, 1);
        assert_eq!(options.exploration_scale, 0.75);
        assert_eq!(options.mutation_rate, base.mutation_rate);
        assert_eq!(options.seed, base.seed);
        assert!(soccer_evolution_options_from_search_metadata(
            Some(&serde_json::json!({"algorithm": "merge"})),
            base
        )
        .is_none());
    }

    #[test]
    fn evolution_options_overlay_accepts_nested_tactical_search_metadata() {
        let base = SoccerEvolutionOptions::default();
        let metadata = serde_json::json!({
            "policy": {
                "algorithm": "evolutionary-policy-search"
            },
            "tactical": {
                "algorithm": "evolutionary-genetic-programming-tactical-search",
                "options": {
                    "mutationRate": 0.12,
                    "crossoverRate": 0.66,
                    "populationSize": 32,
                    "seed": 2027
                }
            }
        });

        let options = soccer_evolution_options_from_search_metadata(Some(&metadata), base)
            .expect("nested tactical postgres options");

        assert_eq!(options.mutation_rate, 0.12);
        assert_eq!(options.crossover_rate, 0.66);
        assert_eq!(options.population_size, 32);
        assert_eq!(options.seed, 2027);
        assert!(options.exploration_scale >= 0.90);
    }

    #[test]
    fn evolution_options_overlay_accepts_genetic_programming_snake_case_metadata() {
        let base = SoccerEvolutionOptions::default();
        let metadata = serde_json::json!({
            "geneticProgramming": {
                "algorithm": "evolutionary-genetic-programming-tactical-search",
                "options": {
                    "mutation_rate": 0.16,
                    "mutation_scale": 0.44,
                    "crossover_rate": 0.70,
                    "exploration_rate": 0.24,
                    "exploration_scale": 1.15,
                    "elite_weight_floor": 0.12,
                    "population_size": 48,
                    "seed": 5001
                }
            }
        });

        let options = soccer_evolution_options_from_search_metadata(Some(&metadata), base)
            .expect("genetic programming postgres options");

        assert_eq!(options.mutation_rate, 0.16);
        assert_eq!(options.mutation_scale, 0.44);
        assert_eq!(options.crossover_rate, 0.70);
        assert_eq!(options.exploration_rate, 0.24);
        assert_eq!(options.exploration_scale, 1.15);
        assert_eq!(options.elite_weight_floor, 0.12);
        assert_eq!(options.population_size, 48);
        assert_eq!(options.seed, 5001);
    }

    #[test]
    fn evolution_options_overlay_accepts_learning_provenance_search_parameters() {
        let base = SoccerEvolutionOptions::default();
        let metadata = serde_json::json!({
            "learningProvenance": {
                "sourceKind": "evolution",
                "searchParameters": {
                    "algorithm": "evolutionary-genetic-programming",
                    "options": {
                        "mutationScale": 0.52,
                        "explorationRate": 0.21,
                        "eliteWeightFloor": 0.09
                    }
                }
            }
        });

        let options = soccer_evolution_options_from_search_metadata(Some(&metadata), base)
            .expect("learning provenance postgres options");

        assert_eq!(options.mutation_scale, 0.52);
        assert_eq!(options.exploration_rate, 0.21);
        assert_eq!(options.elite_weight_floor, 0.09);
        assert!(options.population_size >= 24);
    }

    #[test]
    fn evolution_options_overlay_widens_genetic_programming_from_postgres_hint() {
        let base = SoccerEvolutionOptions::default();
        let metadata = serde_json::json!({
            "learningProvenance": {
                "searchParameters": {
                    "algorithm": "geneticProgramming",
                    "options": {
                        "seed": 7701
                    }
                }
            }
        });

        let options = soccer_evolution_options_from_search_metadata(Some(&metadata), base)
            .expect("genetic programming preset from postgres metadata");

        assert_eq!(options.seed, 7701);
        assert!(options.population_size >= 24);
        assert!(options.exploration_rate >= 0.14);
        assert!(options.exploration_scale >= 0.90);
        assert!(options.crossover_rate >= 0.62);
    }

    #[test]
    fn evolution_options_overlay_respects_explicit_genetic_programming_caps() {
        let base = SoccerEvolutionOptions::default();
        let metadata = serde_json::json!({
            "algorithm": "geneticProgramming",
            "options": {
                "populationSize": 6,
                "explorationScale": 0.30,
                "crossoverRate": 0.25
            }
        });

        let options = soccer_evolution_options_from_search_metadata(Some(&metadata), base)
            .expect("explicit genetic programming caps");

        assert_eq!(options.population_size, 6);
        assert_eq!(options.exploration_scale, 0.30);
        assert_eq!(options.crossover_rate, 0.25);
        assert!(options.mutation_scale >= 0.34);
    }

    fn tiny_neural_snapshot() -> SoccerNeuralNetworkSnapshot {
        SoccerNeuralNetworkSnapshot {
            input_dim: 2,
            output_dim: 1,
            parameter_count: 3,
            l2_norm: 0.42,
            layers: vec![SoccerNeuralLayerSnapshot {
                activation: "linear".to_string(),
                weights: vec![vec![0.20, -0.15]],
                biases: vec![0.05],
            }],
            ..Default::default()
        }
    }

    #[test]
    fn queue_policy_arc_cache_reuses_unchanged_policies_and_rebuilds_changes() {
        let policies = policy_with_action(1.0, 2);
        let mut cache = None::<CachedSoccerTeamQPoliciesArc>;

        let first = cached_soccer_team_policies_arc(&policies, &mut cache);
        let repeated = cached_soccer_team_policies_arc(&policies, &mut cache);
        assert!(Arc::ptr_eq(&first, &repeated));

        let updated_policies = policy_with_action(1.25, 2);
        let changed = cached_soccer_team_policies_arc(&updated_policies, &mut cache);
        assert!(!Arc::ptr_eq(&first, &changed));

        let repeated_changed = cached_soccer_team_policies_arc(&updated_policies, &mut cache);
        assert!(Arc::ptr_eq(&changed, &repeated_changed));
    }

    #[test]
    fn queue_neural_snapshot_arc_cache_reuses_unchanged_snapshot_and_rebuilds_changes() {
        let snapshot = tiny_neural_snapshot();
        let mut latest_neural_network = Some(snapshot.clone());
        let mut cache = None::<CachedSoccerNeuralNetworkSnapshotArc>;

        let first = cached_soccer_neural_network_snapshot_arc(&latest_neural_network, &mut cache)
            .expect("first cached snapshot");
        let repeated =
            cached_soccer_neural_network_snapshot_arc(&latest_neural_network, &mut cache)
                .expect("repeated cached snapshot");
        assert!(Arc::ptr_eq(&first, &repeated));

        let mut updated = snapshot;
        updated.layers[0].weights[0][0] += 0.125;
        latest_neural_network = Some(updated);
        let changed = cached_soccer_neural_network_snapshot_arc(&latest_neural_network, &mut cache)
            .expect("changed cached snapshot");
        assert!(!Arc::ptr_eq(&first, &changed));

        latest_neural_network = None;
        assert!(
            cached_soccer_neural_network_snapshot_arc(&latest_neural_network, &mut cache).is_none()
        );
        assert!(cache.is_none());
    }

    #[test]
    fn learning_weight_fingerprints_change_when_weights_change() {
        let weights = SoccerTacticalLearningWeights::default();
        let mut wider_weights = weights.clone();
        wider_weights.attack_flank_lane_weight += 0.25;

        assert_ne!(
            soccer_tactical_learning_weights_fingerprint(&weights),
            soccer_tactical_learning_weights_fingerprint(&wider_weights)
        );

        let snapshot = tiny_neural_snapshot();
        let mut updated_snapshot = snapshot.clone();
        updated_snapshot.layers[0].biases[0] += 0.10;
        updated_snapshot.l2_norm += 0.10;

        assert_ne!(
            soccer_neural_network_snapshot_fingerprint(&snapshot),
            soccer_neural_network_snapshot_fingerprint(&updated_snapshot)
        );

        let policy = policy_with_action(1.0, 2);
        let updated_policy = policy_with_action(1.5, 2);
        assert_ne!(
            soccer_team_q_policies_fingerprint(&policy),
            soccer_team_q_policies_fingerprint(&updated_policy)
        );
    }

    #[test]
    fn active_max_fitness_regression_env_parser_defaults_and_validates() {
        assert_eq!(
            parse_soccer_policy_active_max_fitness_regression(None).unwrap(),
            SOCCER_POLICY_ACTIVE_MAX_FITNESS_REGRESSION
        );
        assert_eq!(
            parse_soccer_policy_active_max_fitness_regression(Some("0")).unwrap(),
            0.0
        );
        assert_eq!(
            parse_soccer_policy_active_max_fitness_regression(Some("0.1")).unwrap(),
            0.1
        );
        assert!(parse_soccer_policy_active_max_fitness_regression(Some("-0.1")).is_err());
        assert!(parse_soccer_policy_active_max_fitness_regression(Some("NaN")).is_err());
        assert!(parse_soccer_policy_active_max_fitness_regression(Some("abc")).is_err());
    }

    #[test]
    fn pending_policy_stays_active_only_when_latest_head_matches_parent() {
        assert_eq!(
            soccer_policy_version_insert_status_after_active_head(
                SOCCER_POLICY_STATUS_ACTIVE,
                Some("parent"),
                12,
                1.10,
                Some("parent"),
                Some(11),
                Some(1.00),
                SOCCER_POLICY_ACTIVE_MAX_FITNESS_REGRESSION,
            ),
            SOCCER_POLICY_STATUS_ACTIVE
        );
        assert_eq!(
            soccer_policy_version_insert_status_after_active_head(
                SOCCER_POLICY_STATUS_ACTIVE,
                Some("parent"),
                12,
                1.10,
                Some("newer-head"),
                Some(12),
                Some(1.00),
                SOCCER_POLICY_ACTIVE_MAX_FITNESS_REGRESSION,
            ),
            SOCCER_POLICY_STATUS_ARCHIVED
        );
        assert_eq!(
            soccer_policy_version_insert_status_after_active_head(
                SOCCER_POLICY_STATUS_ACTIVE,
                Some("parent"),
                12,
                1.10,
                Some("parent"),
                Some(12),
                Some(1.00),
                SOCCER_POLICY_ACTIVE_MAX_FITNESS_REGRESSION,
            ),
            SOCCER_POLICY_STATUS_ARCHIVED
        );
        assert_eq!(
            soccer_policy_version_insert_status_after_active_head(
                SOCCER_POLICY_STATUS_ACTIVE,
                Some("parent"),
                13,
                4.90,
                Some("parent"),
                Some(12),
                Some(5.00),
                SOCCER_POLICY_ACTIVE_MAX_FITNESS_REGRESSION,
            ),
            SOCCER_POLICY_STATUS_ACTIVE
        );
        assert_eq!(
            soccer_policy_version_insert_status_after_active_head(
                SOCCER_POLICY_STATUS_ACTIVE,
                Some("parent"),
                13,
                2.40,
                Some("parent"),
                Some(12),
                Some(5.30),
                SOCCER_POLICY_ACTIVE_MAX_FITNESS_REGRESSION,
            ),
            SOCCER_POLICY_STATUS_ARCHIVED
        );
        assert_eq!(
            soccer_policy_version_insert_status_after_active_head(
                "archived",
                Some("parent"),
                12,
                1.10,
                Some("newer-head"),
                Some(14),
                Some(1.00),
                SOCCER_POLICY_ACTIVE_MAX_FITNESS_REGRESSION,
            ),
            "archived"
        );
    }

    #[test]
    fn postgres_policy_refresh_picks_up_first_or_newer_active_weights() {
        let first = soccer_postgres_policy_refresh_decision(SoccerPostgresPolicyRefreshCheck {
            current_policy_version_id: None,
            current_generation: 0,
            current_updated_at_micros: 0,
            current_policy_fingerprint: None,
            latest_policy_fingerprint: None,
            current_neural_network_present: false,
            current_neural_network_fingerprint: None,
            latest_policy_version_id: "v1",
            latest_generation: 4,
            latest_updated_at_micros: 100,
            latest_neural_network_present: true,
            latest_neural_network_fingerprint: None,
            current_tactical_learning_fingerprint: None,
            latest_tactical_learning_fingerprint: None,
            local_tactical_evolved_since_pg_refresh: false,
            postgres_tactical_learning_authoritative: false,
        });
        assert!(first.refresh_policy);
        assert!(first.apply_tactical_learning);

        let newer_head =
            soccer_postgres_policy_refresh_decision(SoccerPostgresPolicyRefreshCheck {
                current_policy_version_id: Some("v1"),
                current_generation: 4,
                current_updated_at_micros: 100,
                current_policy_fingerprint: None,
                latest_policy_fingerprint: None,
                current_neural_network_present: true,
                current_neural_network_fingerprint: None,
                latest_policy_version_id: "v2",
                latest_generation: 5,
                latest_updated_at_micros: 125,
                latest_neural_network_present: true,
                latest_neural_network_fingerprint: None,
                current_tactical_learning_fingerprint: None,
                latest_tactical_learning_fingerprint: None,
                local_tactical_evolved_since_pg_refresh: true,
                postgres_tactical_learning_authoritative: false,
            });
        assert!(newer_head.refresh_policy);
        assert!(newer_head.apply_tactical_learning);
        assert!(!newer_head.same_policy_version);
    }

    #[test]
    fn postgres_policy_refresh_picks_up_same_head_revision_and_neural_snapshot() {
        let revised_same_head =
            soccer_postgres_policy_refresh_decision(SoccerPostgresPolicyRefreshCheck {
                current_policy_version_id: Some("v1"),
                current_generation: 4,
                current_updated_at_micros: 100,
                current_policy_fingerprint: None,
                latest_policy_fingerprint: None,
                current_neural_network_present: true,
                current_neural_network_fingerprint: None,
                latest_policy_version_id: "v1",
                latest_generation: 4,
                latest_updated_at_micros: 125,
                latest_neural_network_present: true,
                latest_neural_network_fingerprint: None,
                current_tactical_learning_fingerprint: None,
                latest_tactical_learning_fingerprint: None,
                local_tactical_evolved_since_pg_refresh: true,
                postgres_tactical_learning_authoritative: false,
            });
        assert!(revised_same_head.refresh_policy);
        assert!(revised_same_head.apply_tactical_learning);
        assert!(revised_same_head.same_policy_newer_revision);

        let neural_arrived =
            soccer_postgres_policy_refresh_decision(SoccerPostgresPolicyRefreshCheck {
                current_policy_version_id: Some("v1"),
                current_generation: 4,
                current_updated_at_micros: 100,
                current_policy_fingerprint: None,
                latest_policy_fingerprint: None,
                current_neural_network_present: false,
                current_neural_network_fingerprint: None,
                latest_policy_version_id: "v1",
                latest_generation: 4,
                latest_updated_at_micros: 100,
                latest_neural_network_present: true,
                latest_neural_network_fingerprint: None,
                current_tactical_learning_fingerprint: None,
                latest_tactical_learning_fingerprint: None,
                local_tactical_evolved_since_pg_refresh: true,
                postgres_tactical_learning_authoritative: false,
            });
        assert!(neural_arrived.refresh_policy);
        assert!(!neural_arrived.apply_tactical_learning);
    }

    #[test]
    fn postgres_policy_refresh_picks_up_same_head_weight_fingerprint_changes() {
        let policy_weights_changed =
            soccer_postgres_policy_refresh_decision(SoccerPostgresPolicyRefreshCheck {
                current_policy_version_id: Some("v1"),
                current_generation: 4,
                current_updated_at_micros: 100,
                current_policy_fingerprint: Some(7),
                latest_policy_fingerprint: Some(8),
                current_neural_network_present: true,
                current_neural_network_fingerprint: Some(10),
                latest_policy_version_id: "v1",
                latest_generation: 4,
                latest_updated_at_micros: 100,
                latest_neural_network_present: true,
                latest_neural_network_fingerprint: Some(10),
                current_tactical_learning_fingerprint: Some(20),
                latest_tactical_learning_fingerprint: Some(20),
                local_tactical_evolved_since_pg_refresh: true,
                postgres_tactical_learning_authoritative: false,
            });

        assert!(policy_weights_changed.refresh_policy);
        assert!(!policy_weights_changed.apply_tactical_learning);
        assert!(policy_weights_changed.same_policy_version);
        assert!(!policy_weights_changed.same_policy_newer_revision);

        let neural_weights_changed =
            soccer_postgres_policy_refresh_decision(SoccerPostgresPolicyRefreshCheck {
                current_policy_version_id: Some("v1"),
                current_generation: 4,
                current_updated_at_micros: 100,
                current_policy_fingerprint: Some(8),
                latest_policy_fingerprint: Some(8),
                current_neural_network_present: true,
                current_neural_network_fingerprint: Some(10),
                latest_policy_version_id: "v1",
                latest_generation: 4,
                latest_updated_at_micros: 100,
                latest_neural_network_present: true,
                latest_neural_network_fingerprint: Some(11),
                current_tactical_learning_fingerprint: Some(20),
                latest_tactical_learning_fingerprint: Some(20),
                local_tactical_evolved_since_pg_refresh: true,
                postgres_tactical_learning_authoritative: false,
            });

        assert!(neural_weights_changed.refresh_policy);
        assert!(!neural_weights_changed.apply_tactical_learning);
        assert!(neural_weights_changed.same_policy_version);
        assert!(!neural_weights_changed.same_policy_newer_revision);

        let tactical_weights_changed =
            soccer_postgres_policy_refresh_decision(SoccerPostgresPolicyRefreshCheck {
                current_policy_version_id: Some("v1"),
                current_generation: 4,
                current_updated_at_micros: 100,
                current_policy_fingerprint: Some(8),
                latest_policy_fingerprint: Some(8),
                current_neural_network_present: true,
                current_neural_network_fingerprint: Some(10),
                latest_policy_version_id: "v1",
                latest_generation: 4,
                latest_updated_at_micros: 100,
                latest_neural_network_present: true,
                latest_neural_network_fingerprint: Some(10),
                current_tactical_learning_fingerprint: Some(20),
                latest_tactical_learning_fingerprint: Some(21),
                local_tactical_evolved_since_pg_refresh: true,
                postgres_tactical_learning_authoritative: true,
            });

        assert!(!tactical_weights_changed.refresh_policy);
        assert!(tactical_weights_changed.apply_tactical_learning);
        assert!(tactical_weights_changed.same_policy_version);
        assert!(!tactical_weights_changed.same_policy_newer_revision);

        let tactical_weights_changed_without_local_search =
            soccer_postgres_policy_refresh_decision(SoccerPostgresPolicyRefreshCheck {
                current_policy_version_id: Some("v1"),
                current_generation: 4,
                current_updated_at_micros: 100,
                current_policy_fingerprint: Some(8),
                latest_policy_fingerprint: Some(8),
                current_neural_network_present: true,
                current_neural_network_fingerprint: Some(10),
                latest_policy_version_id: "v1",
                latest_generation: 4,
                latest_updated_at_micros: 100,
                latest_neural_network_present: true,
                latest_neural_network_fingerprint: Some(10),
                current_tactical_learning_fingerprint: Some(20),
                latest_tactical_learning_fingerprint: Some(22),
                local_tactical_evolved_since_pg_refresh: false,
                postgres_tactical_learning_authoritative: false,
            });

        assert!(!tactical_weights_changed_without_local_search.refresh_policy);
        assert!(tactical_weights_changed_without_local_search.apply_tactical_learning);
        assert!(tactical_weights_changed_without_local_search.same_policy_version);
    }

    #[test]
    fn postgres_policy_refresh_picks_up_durable_pending_evolved_head() {
        let durable_pending_head =
            soccer_postgres_policy_refresh_decision(SoccerPostgresPolicyRefreshCheck {
                current_policy_version_id: Some("local-mutation-v2"),
                current_generation: 5,
                current_updated_at_micros: 0,
                current_policy_fingerprint: None,
                latest_policy_fingerprint: None,
                current_neural_network_present: true,
                current_neural_network_fingerprint: None,
                latest_policy_version_id: "local-mutation-v2",
                latest_generation: 5,
                latest_updated_at_micros: 250,
                latest_neural_network_present: true,
                latest_neural_network_fingerprint: None,
                current_tactical_learning_fingerprint: None,
                latest_tactical_learning_fingerprint: None,
                local_tactical_evolved_since_pg_refresh: true,
                postgres_tactical_learning_authoritative: true,
            });

        assert!(durable_pending_head.refresh_policy);
        assert!(durable_pending_head.apply_tactical_learning);
        assert!(durable_pending_head.same_policy_version);
        assert!(durable_pending_head.same_policy_newer_revision);
    }

    #[test]
    fn postgres_policy_refresh_preserves_local_tactical_search_without_new_pg_head() {
        let unchanged = soccer_postgres_policy_refresh_decision(SoccerPostgresPolicyRefreshCheck {
            current_policy_version_id: Some("v1"),
            current_generation: 4,
            current_updated_at_micros: 100,
            current_policy_fingerprint: None,
            latest_policy_fingerprint: None,
            current_neural_network_present: true,
            current_neural_network_fingerprint: None,
            latest_policy_version_id: "v1",
            latest_generation: 4,
            latest_updated_at_micros: 100,
            latest_neural_network_present: true,
            latest_neural_network_fingerprint: None,
            current_tactical_learning_fingerprint: None,
            latest_tactical_learning_fingerprint: None,
            local_tactical_evolved_since_pg_refresh: true,
            postgres_tactical_learning_authoritative: false,
        });
        assert!(!unchanged.refresh_policy);
        assert!(!unchanged.apply_tactical_learning);

        let no_local_search =
            soccer_postgres_policy_refresh_decision(SoccerPostgresPolicyRefreshCheck {
                local_tactical_evolved_since_pg_refresh: false,
                ..SoccerPostgresPolicyRefreshCheck {
                    current_policy_version_id: Some("v1"),
                    current_generation: 4,
                    current_updated_at_micros: 100,
                    current_policy_fingerprint: None,
                    latest_policy_fingerprint: None,
                    current_neural_network_present: true,
                    current_neural_network_fingerprint: None,
                    latest_policy_version_id: "v1",
                    latest_generation: 4,
                    latest_updated_at_micros: 100,
                    latest_neural_network_present: true,
                    latest_neural_network_fingerprint: None,
                    current_tactical_learning_fingerprint: None,
                    latest_tactical_learning_fingerprint: None,
                    local_tactical_evolved_since_pg_refresh: true,
                    postgres_tactical_learning_authoritative: false,
                }
            });
        assert!(!no_local_search.refresh_policy);
        assert!(no_local_search.apply_tactical_learning);
    }

    #[test]
    fn postgres_policy_refresh_can_make_postgres_tactical_weights_authoritative() {
        let unchanged_head_with_local_search =
            soccer_postgres_policy_refresh_decision(SoccerPostgresPolicyRefreshCheck {
                current_policy_version_id: Some("v1"),
                current_generation: 4,
                current_updated_at_micros: 100,
                current_policy_fingerprint: None,
                latest_policy_fingerprint: None,
                current_neural_network_present: true,
                current_neural_network_fingerprint: None,
                latest_policy_version_id: "v1",
                latest_generation: 4,
                latest_updated_at_micros: 100,
                latest_neural_network_present: true,
                latest_neural_network_fingerprint: None,
                current_tactical_learning_fingerprint: None,
                latest_tactical_learning_fingerprint: None,
                local_tactical_evolved_since_pg_refresh: true,
                postgres_tactical_learning_authoritative: true,
            });

        assert!(!unchanged_head_with_local_search.refresh_policy);
        assert!(unchanged_head_with_local_search.apply_tactical_learning);
    }

    #[test]
    fn postgres_policy_refresh_does_not_reapply_older_active_tactical_weights() {
        let pending_local_head =
            soccer_postgres_policy_refresh_decision(SoccerPostgresPolicyRefreshCheck {
                current_policy_version_id: Some("pending-v2"),
                current_generation: 5,
                current_updated_at_micros: 0,
                current_policy_fingerprint: None,
                latest_policy_fingerprint: None,
                current_neural_network_present: true,
                current_neural_network_fingerprint: None,
                latest_policy_version_id: "v1",
                latest_generation: 4,
                latest_updated_at_micros: 200,
                latest_neural_network_present: true,
                latest_neural_network_fingerprint: None,
                current_tactical_learning_fingerprint: None,
                latest_tactical_learning_fingerprint: None,
                local_tactical_evolved_since_pg_refresh: false,
                postgres_tactical_learning_authoritative: true,
            });

        assert!(!pending_local_head.refresh_policy);
        assert!(!pending_local_head.apply_tactical_learning);
        assert!(!pending_local_head.same_policy_version);
    }

    #[test]
    fn postgres_refresh_for_new_sim_can_include_resume_artifacts() {
        assert!(soccer_should_refresh_postgres_for_new_sim(false, false));
        assert!(soccer_should_refresh_postgres_for_new_sim(false, true));
        assert!(!soccer_should_refresh_postgres_for_new_sim(true, false));
        assert!(soccer_should_refresh_postgres_for_new_sim(true, true));
    }

    #[test]
    fn postgres_policy_version_flush_for_new_sim_requires_pending_refreshable_head() {
        assert!(!soccer_should_flush_postgres_policy_versions_for_new_sim(
            false, true, 1
        ));
        assert!(!soccer_should_flush_postgres_policy_versions_for_new_sim(
            true, false, 1
        ));
        assert!(!soccer_should_flush_postgres_policy_versions_for_new_sim(
            true, true, 0
        ));
        assert!(soccer_should_flush_postgres_policy_versions_for_new_sim(
            true, true, 2
        ));
    }

    #[test]
    fn postgres_new_sim_refresh_plan_makes_evolved_heads_durable_before_reads() {
        let plan = soccer_postgres_new_sim_refresh_plan(false, false, true, 2, 1);

        assert!(plan.refresh_from_postgres);
        assert!(plan.flush_pending_policy_versions);
        assert!(plan.wait_for_async_policy_versions);
    }

    #[test]
    fn postgres_new_sim_refresh_plan_waits_for_async_heads_without_local_buffer() {
        let plan = soccer_postgres_new_sim_refresh_plan(false, false, true, 0, 2);

        assert!(plan.refresh_from_postgres);
        assert!(!plan.flush_pending_policy_versions);
        assert!(plan.wait_for_async_policy_versions);
    }

    #[test]
    fn postgres_new_sim_refresh_plan_respects_resume_and_flush_opt_outs() {
        let resume_plan = soccer_postgres_new_sim_refresh_plan(true, true, true, 1, 0);
        assert!(resume_plan.refresh_from_postgres);
        assert!(resume_plan.flush_pending_policy_versions);
        assert!(!resume_plan.wait_for_async_policy_versions);

        let stale_resume_plan = soccer_postgres_new_sim_refresh_plan(true, false, true, 1, 1);
        assert!(!stale_resume_plan.refresh_from_postgres);
        assert!(!stale_resume_plan.flush_pending_policy_versions);
        assert!(!stale_resume_plan.wait_for_async_policy_versions);

        let no_flush_plan = soccer_postgres_new_sim_refresh_plan(false, false, false, 1, 1);
        assert!(no_flush_plan.refresh_from_postgres);
        assert!(!no_flush_plan.flush_pending_policy_versions);
        assert!(no_flush_plan.wait_for_async_policy_versions);
    }

    #[test]
    fn tactical_search_pressure_prioritizes_flanks_and_defensive_contraction() {
        let healthy = SoccerTacticalLearningSummary {
            mean_attack_width_score: 0.92,
            mean_attack_flank_lane_score: 0.88,
            mean_attack_spacing_score: 0.86,
            mean_defense_contract_score: 0.90,
            mean_defense_spacing_score: 0.84,
            mean_defense_ball_gap_score: 0.86,
            mean_defense_role_press_score: 0.82,
            ..Default::default()
        };
        let poor_shape = SoccerTacticalLearningSummary {
            mean_attack_width_score: 0.24,
            mean_attack_flank_lane_score: 0.12,
            mean_attack_spacing_score: 0.52,
            mean_defense_contract_score: 0.16,
            mean_defense_spacing_score: 0.46,
            mean_defense_ball_gap_score: 0.56,
            mean_defense_role_press_score: 0.58,
            ..Default::default()
        };

        assert!(
            soccer_tactical_search_pressure(&poor_shape)
                > soccer_tactical_search_pressure(&healthy) + 0.45
        );
    }

    #[test]
    fn tactical_weighted_evolution_summary_skips_corrupt_parents() {
        let good = SoccerTacticalLearningSummary {
            mean_attack_width_score: 0.25,
            mean_attack_flank_lane_score: 0.35,
            mean_attack_spacing_score: 0.45,
            mean_defense_contract_score: 0.55,
            mean_defense_spacing_score: 0.65,
            mean_defense_ball_gap_score: 0.75,
            mean_defense_role_press_score: 0.85,
            ..Default::default()
        };
        let mut corrupt = good.clone();
        corrupt.mean_attack_width_score = f64::NAN;
        let options = SoccerEvolutionOptions {
            elite_weight_floor: 0.0,
            ..SoccerEvolutionOptions::default()
        };

        let weighted =
            weighted_tactical_evolution_summary(&[(&corrupt, 100.0), (&good, 1.0)], options)
                .expect("finite parent should survive corrupt parent");
        assert_eq!(
            weighted.mean_attack_width_score,
            good.mean_attack_width_score
        );
        assert_eq!(
            weighted.mean_defense_role_press_score,
            good.mean_defense_role_press_score
        );
        assert!(tactical_evolution_summary_is_finite(&weighted));
        assert!(weighted_tactical_evolution_summary(
            &[(&corrupt, 100.0), (&good, f64::NAN)],
            options
        )
        .is_none());
    }

    #[test]
    fn tactical_search_adapts_evolution_budget_without_enabling_disabled_search() {
        let summary = SoccerTacticalLearningSummary {
            mean_attack_width_score: 0.15,
            mean_attack_flank_lane_score: 0.10,
            mean_attack_spacing_score: 0.30,
            mean_defense_contract_score: 0.18,
            mean_defense_spacing_score: 0.35,
            mean_defense_ball_gap_score: 0.45,
            mean_defense_role_press_score: 0.38,
            ..Default::default()
        };
        let base = SoccerEvolutionOptions::default();
        let adapted = adapt_soccer_evolution_options_for_tactical_search(base, &summary);

        assert!(adapted.population_size > base.population_size);
        assert!(adapted.mutation_rate > base.mutation_rate);
        assert!(adapted.mutation_scale > base.mutation_scale);
        assert!(adapted.exploration_rate > base.exploration_rate);
        assert!(adapted.exploration_scale > base.exploration_scale);
        assert!(adapted.population_size <= base.population_size * 2);

        let disabled = SoccerEvolutionOptions {
            mutation_rate: 0.0,
            mutation_scale: 0.0,
            crossover_rate: 0.0,
            exploration_rate: 0.0,
            exploration_scale: 0.0,
            elite_weight_floor: 0.0,
            population_size: 1,
            seed: 99,
        };
        assert_eq!(
            adapt_soccer_evolution_options_for_tactical_search(disabled, &summary).population_size,
            disabled.population_size
        );
        assert_eq!(
            adapt_soccer_evolution_options_for_tactical_search(disabled, &summary).mutation_rate,
            disabled.mutation_rate
        );
    }

    #[test]
    fn tactical_weight_evolution_pushes_flank_and_contract_search() {
        let base = SoccerTacticalLearningWeights::default();
        let summary = SoccerTacticalLearningSummary {
            mean_attack_width_score: 0.20,
            mean_attack_flank_lane_score: 0.18,
            mean_attack_spacing_score: 0.35,
            mean_defense_contract_score: 0.22,
            mean_defense_spacing_score: 0.44,
            mean_defense_ball_gap_score: 0.50,
            mean_defense_role_press_score: 0.40,
            ..Default::default()
        };
        let options = SoccerEvolutionOptions {
            mutation_rate: 0.0,
            mutation_scale: 0.0,
            crossover_rate: 0.0,
            exploration_rate: 0.0,
            exploration_scale: 0.0,
            elite_weight_floor: 0.0,
            population_size: 1,
            seed: 29,
        };

        let evolved = evolve_soccer_tactical_learning_weights(&base, &[(&summary, 1.0)], options);

        assert!(evolved.attack_width_delta_weight > base.attack_width_delta_weight);
        assert!(evolved.attack_width_score_weight > base.attack_width_score_weight);
        assert!(evolved.attack_flank_lane_weight > base.attack_flank_lane_weight);
        assert!(evolved.defense_contract_delta_weight > base.defense_contract_delta_weight);
        assert!(evolved.defense_compactness_score_weight > base.defense_compactness_score_weight);
    }

    #[test]
    fn tactical_weight_evolution_mutates_decision_learning_weights() {
        let base = SoccerTacticalLearningWeights::default();
        let summary = SoccerTacticalLearningSummary {
            mean_attack_width_score: 0.16,
            mean_attack_flank_lane_score: 0.14,
            mean_attack_spacing_score: 0.20,
            mean_defense_contract_score: 0.28,
            mean_defense_spacing_score: 0.38,
            mean_defense_ball_gap_score: 0.32,
            mean_defense_role_press_score: 0.18,
            ..Default::default()
        };
        let options = SoccerEvolutionOptions {
            mutation_rate: 0.0,
            mutation_scale: 0.0,
            crossover_rate: 0.0,
            exploration_rate: 0.0,
            exploration_scale: 0.0,
            elite_weight_floor: 0.0,
            population_size: 1,
            seed: 31,
        };

        let evolved = evolve_soccer_tactical_learning_candidate(&base, &summary, options);

        assert!(evolved.shot_choice_learning_weight > base.shot_choice_learning_weight);
        assert!(evolved.goal_entry_pass_learning_weight > base.goal_entry_pass_learning_weight);
        assert!(evolved.pressure_release_learning_weight > base.pressure_release_learning_weight);
        assert!(
            evolved.pass_target_ranking_learning_weight > base.pass_target_ranking_learning_weight
        );
        assert!(
            evolved.defensive_line_press_learning_weight
                > base.defensive_line_press_learning_weight
        );
        assert!(evolved.shot_choice_learning_weight <= 2.0);
        assert!(evolved.goal_entry_pass_learning_weight <= 2.0);
        assert!(evolved.pressure_release_learning_weight <= 2.0);
        assert!(evolved.pass_target_ranking_learning_weight <= 2.0);
        assert!(evolved.defensive_line_press_learning_weight <= 2.0);
    }

    #[test]
    fn tactical_weight_score_prefers_flank_and_contract_when_shape_gaps_are_high() {
        let base = SoccerTacticalLearningWeights::default();
        let summary = SoccerTacticalLearningSummary {
            mean_attack_width_score: 0.75,
            mean_attack_flank_lane_score: 0.10,
            mean_attack_spacing_score: 0.80,
            mean_defense_contract_score: 0.12,
            mean_defense_spacing_score: 0.78,
            mean_defense_ball_gap_score: 0.70,
            mean_defense_role_press_score: 0.76,
            ..Default::default()
        };

        let mut generic_shape = base.clone();
        generic_shape.attack_width_delta_weight += 0.80;
        generic_shape.defense_spacing_delta_weight += 0.80;
        generic_shape.defender_midfielder_press_weight += 0.50;

        let mut flank_and_contract = base.clone();
        flank_and_contract.attack_flank_lane_weight += 0.80;
        flank_and_contract.defense_contract_delta_weight += 0.80;
        flank_and_contract.defense_compactness_score_weight += 0.50;

        let base_score = soccer_tactical_weight_search_score(&base, &summary);
        let generic_score = soccer_tactical_weight_search_score(&generic_shape, &summary);
        let flank_contract_score =
            soccer_tactical_weight_search_score(&flank_and_contract, &summary);

        assert!(generic_score > base_score);
        assert!(flank_contract_score > generic_score);
    }

    #[test]
    fn tactical_strategy_candidates_coordinate_flanks_and_contraction() {
        let base = SoccerTacticalLearningWeights::default();
        let summary = SoccerTacticalLearningSummary {
            mean_attack_width_score: 0.20,
            mean_attack_flank_lane_score: 0.18,
            mean_attack_spacing_score: 0.35,
            mean_defense_contract_score: 0.22,
            mean_defense_spacing_score: 0.44,
            mean_defense_ball_gap_score: 0.50,
            mean_defense_role_press_score: 0.40,
            ..Default::default()
        };
        let options = SoccerEvolutionOptions {
            mutation_rate: 0.0,
            mutation_scale: 0.0,
            crossover_rate: 0.0,
            exploration_rate: 0.0,
            exploration_scale: 0.0,
            elite_weight_floor: 0.0,
            population_size: 1,
            seed: 30,
        };

        let evolved = evolve_soccer_tactical_learning_weights(&base, &[(&summary, 1.0)], options);

        assert!(evolved.attack_width_delta_weight >= base.attack_width_delta_weight + 0.18);
        assert!(evolved.attack_width_score_weight >= base.attack_width_score_weight + 0.06);
        assert!(evolved.attack_flank_lane_weight >= base.attack_flank_lane_weight + 0.24);
        assert!(evolved.defense_contract_delta_weight >= base.defense_contract_delta_weight + 0.22);
        assert!(
            evolved.defense_compactness_score_weight
                >= base.defense_compactness_score_weight + 0.10
        );
    }

    #[test]
    fn tactical_strategy_candidate_families_search_more_shape_space() {
        let base = SoccerTacticalLearningWeights::default();
        let summary = SoccerTacticalLearningSummary {
            mean_attack_width_score: 0.15,
            mean_attack_flank_lane_score: 0.12,
            mean_attack_spacing_score: 0.30,
            mean_defense_contract_score: 0.16,
            mean_defense_spacing_score: 0.35,
            mean_defense_ball_gap_score: 0.45,
            mean_defense_role_press_score: 0.38,
            ..Default::default()
        };
        let mut candidates = Vec::new();
        search_soccer_tactical_strategy_candidates(&base, &summary, |candidate| {
            candidates.push(candidate);
        });

        assert!(candidates.len() >= 7);
        assert!(candidates.iter().any(|candidate| {
            candidate.attack_flank_lane_weight >= base.attack_flank_lane_weight + 0.40
                && candidate.attack_spacing_delta_weight > base.attack_spacing_delta_weight
        }));
        assert!(candidates.iter().any(|candidate| {
            candidate.attack_width_score_weight >= base.attack_width_score_weight + 0.08
        }));
        assert!(candidates.iter().any(|candidate| {
            candidate.defense_contract_delta_weight >= base.defense_contract_delta_weight + 0.35
                && candidate.defense_ball_depth_score_weight
                    >= base.defense_ball_depth_score_weight + 0.09
        }));
        assert!(candidates.iter().any(|candidate| {
            candidate.defender_midfielder_press_weight
                >= base.defender_midfielder_press_weight + 0.06
                && candidate.midfielder_press_weight >= base.midfielder_press_weight + 0.05
        }));
    }

    #[test]
    fn tactical_population_search_keeps_best_weight_candidate() {
        let base = SoccerTacticalLearningWeights::default();
        let summary = SoccerTacticalLearningSummary {
            mean_attack_width_score: 0.15,
            mean_attack_flank_lane_score: 0.12,
            mean_attack_spacing_score: 0.30,
            mean_defense_contract_score: 0.16,
            mean_defense_spacing_score: 0.35,
            mean_defense_ball_gap_score: 0.45,
            mean_defense_role_press_score: 0.38,
            ..Default::default()
        };
        let single_options = SoccerEvolutionOptions {
            mutation_rate: 1.0,
            mutation_scale: 0.40,
            crossover_rate: 0.0,
            exploration_rate: 1.0,
            exploration_scale: 0.55,
            elite_weight_floor: 0.0,
            population_size: 1,
            seed: 73,
        };
        let population_options = SoccerEvolutionOptions {
            population_size: 10,
            ..single_options
        };

        let single =
            evolve_soccer_tactical_learning_weights(&base, &[(&summary, 1.0)], single_options);
        let population =
            evolve_soccer_tactical_learning_weights(&base, &[(&summary, 1.0)], population_options);
        let weighted_summary =
            weighted_tactical_evolution_summary(&[(&summary, 1.0)], single_options)
                .expect("weighted summary");

        assert!(
            soccer_tactical_weight_search_score(&population, &weighted_summary)
                >= soccer_tactical_weight_search_score(&single, &weighted_summary) - 1e-12
        );
    }

    #[test]
    fn tactical_genome_search_crosses_over_elite_weight_sets() {
        let base = SoccerTacticalLearningWeights::default();
        let summary = SoccerTacticalLearningSummary {
            mean_attack_width_score: 0.15,
            mean_attack_flank_lane_score: 0.12,
            mean_attack_spacing_score: 0.30,
            mean_defense_contract_score: 0.16,
            mean_defense_spacing_score: 0.35,
            mean_defense_ball_gap_score: 0.45,
            mean_defense_role_press_score: 0.38,
            ..Default::default()
        };
        let mut elite = base.clone();
        elite.attack_width_delta_weight = 1.15;
        elite.attack_flank_lane_weight = 1.35;
        elite.defense_contract_delta_weight = 1.45;
        elite.defense_compactness_score_weight = 1.05;
        let mut alternate = base.clone();
        alternate.attack_spacing_delta_weight = 1.10;
        alternate.defense_ball_depth_score_weight = 0.95;
        alternate.defender_midfielder_press_weight = 0.72;
        let options = SoccerEvolutionOptions {
            mutation_rate: 0.0,
            mutation_scale: 0.0,
            crossover_rate: 1.0,
            exploration_rate: 0.0,
            exploration_scale: 0.0,
            elite_weight_floor: 0.0,
            population_size: 6,
            seed: 37,
        };

        let summary_only =
            evolve_soccer_tactical_learning_weights(&base, &[(&summary, 10.0)], options);
        let genome = evolve_soccer_tactical_learning_weights_from_genomes(
            &base,
            &[
                SoccerTacticalLearningGenomeParent {
                    summary: &summary,
                    weights: &elite,
                    fitness: 10.0,
                },
                SoccerTacticalLearningGenomeParent {
                    summary: &summary,
                    weights: &alternate,
                    fitness: 1.0,
                },
            ],
            options,
        );
        let weighted_summary =
            weighted_tactical_evolution_summary(&[(&summary, 10.0), (&summary, 1.0)], options)
                .expect("weighted summary");

        assert!(genome.attack_flank_lane_weight >= elite.attack_flank_lane_weight);
        assert!(genome.defense_contract_delta_weight >= elite.defense_contract_delta_weight);
        assert!(
            soccer_tactical_weight_search_score(&genome, &weighted_summary)
                > soccer_tactical_weight_search_score(&summary_only, &weighted_summary)
        );
    }

    #[test]
    fn tactical_genome_blend_candidates_search_centroid_and_extrapolation() {
        let base = SoccerTacticalLearningWeights::default();
        let summary = SoccerTacticalLearningSummary {
            mean_attack_width_score: 0.15,
            mean_attack_flank_lane_score: 0.12,
            mean_attack_spacing_score: 0.30,
            mean_defense_contract_score: 0.16,
            mean_defense_spacing_score: 0.35,
            mean_defense_ball_gap_score: 0.45,
            mean_defense_role_press_score: 0.38,
            ..Default::default()
        };
        let mut left = base.clone();
        left.attack_width_delta_weight = 0.90;
        left.attack_flank_lane_weight = 0.80;
        left.defense_contract_delta_weight = 0.95;
        let mut right = base.clone();
        right.attack_width_delta_weight = 1.70;
        right.attack_flank_lane_weight = 1.60;
        right.defense_contract_delta_weight = 1.75;
        let options = SoccerEvolutionOptions {
            mutation_rate: 0.0,
            mutation_scale: 0.0,
            crossover_rate: 1.0,
            exploration_rate: 0.0,
            exploration_scale: 0.0,
            elite_weight_floor: 0.0,
            population_size: 1,
            seed: 43,
        };
        let parents = [
            SoccerTacticalLearningGenomeParent {
                summary: &summary,
                weights: &left,
                fitness: 1.0,
            },
            SoccerTacticalLearningGenomeParent {
                summary: &summary,
                weights: &right,
                fitness: 3.0,
            },
        ];
        let weighted_summary = weighted_tactical_evolution_summary(
            &parents
                .iter()
                .map(|parent| (parent.summary, parent.fitness))
                .collect::<Vec<_>>(),
            options,
        )
        .expect("weighted summary");
        let mut candidates = Vec::new();

        search_soccer_tactical_genome_blend_candidates(
            &base,
            &parents,
            options,
            &weighted_summary,
            |candidate| candidates.push(candidate),
        );

        assert!(candidates.len() >= 7);
        assert!((candidates[0].attack_width_delta_weight - 1.50).abs() < 1e-12);
        assert!((candidates[0].attack_flank_lane_weight - 1.40).abs() < 1e-12);
        assert!((candidates[0].defense_contract_delta_weight - 1.55).abs() < 1e-12);
        assert!(candidates[1].attack_width_delta_weight > candidates[0].attack_width_delta_weight);
        assert!(candidates[1].attack_flank_lane_weight > candidates[0].attack_flank_lane_weight);
        assert!(
            candidates[1].defense_contract_delta_weight
                > candidates[0].defense_contract_delta_weight
        );
        assert!(candidates[2..].iter().any(|candidate| {
            candidate.attack_flank_lane_weight > candidates[0].attack_flank_lane_weight
                && candidate.attack_spacing_delta_weight > candidates[0].attack_spacing_delta_weight
        }));
        assert!(candidates[2..].iter().any(|candidate| {
            candidate.defense_contract_delta_weight > candidates[0].defense_contract_delta_weight
                && candidate.defense_compactness_score_weight
                    > candidates[0].defense_compactness_score_weight
        }));
        assert!(candidates[2..].iter().any(|candidate| {
            candidate.attack_flank_lane_weight > candidates[0].attack_flank_lane_weight
                && candidate.defense_contract_delta_weight
                    > candidates[0].defense_contract_delta_weight
        }));
    }

    #[test]
    fn tactical_gp_program_candidates_expand_flank_and_compact_search() {
        let base = SoccerTacticalLearningWeights::default();
        let summary = SoccerTacticalLearningSummary {
            mean_attack_width_score: 0.14,
            mean_attack_flank_lane_score: 0.11,
            mean_attack_spacing_score: 0.26,
            mean_defense_contract_score: 0.13,
            mean_defense_spacing_score: 0.34,
            mean_defense_ball_gap_score: 0.40,
            mean_defense_role_press_score: 0.36,
            ..Default::default()
        };
        let pressure = soccer_tactical_search_pressure(&summary);
        let mut candidates = Vec::new();

        search_soccer_tactical_gp_program_candidates(
            &base,
            &base,
            &summary,
            pressure,
            &mut |candidate| candidates.push(candidate),
        );

        assert_eq!(candidates.len(), 12);
        assert!(candidates.iter().all(|candidate| {
            candidate.attack_flank_lane_weight <= 2.2
                && candidate.defense_contract_delta_weight <= 2.4
                && candidate.defender_midfielder_press_weight <= 1.6
        }));
        assert!(candidates.iter().any(|candidate| {
            candidate.attack_width_delta_weight > base.attack_width_delta_weight
                && candidate.attack_flank_lane_weight > base.attack_flank_lane_weight
        }));
        assert!(candidates.iter().any(|candidate| {
            candidate.defense_contract_delta_weight > base.defense_contract_delta_weight
                && candidate.defense_compactness_score_weight
                    > base.defense_compactness_score_weight
        }));
        assert!(candidates.iter().any(|candidate| {
            candidate.attack_width_delta_weight >= base.attack_width_delta_weight + 0.20
                && candidate.attack_flank_lane_weight >= base.attack_flank_lane_weight + 0.30
                && candidate.defense_contract_delta_weight
                    >= base.defense_contract_delta_weight + 0.30
        }));
        assert!(candidates.iter().any(|candidate| {
            soccer_tactical_weight_search_score(candidate, &summary)
                > soccer_tactical_weight_search_score(&base, &summary)
        }));
    }

    #[test]
    fn tactical_mutated_gp_candidates_search_seeded_extra_shape_space() {
        let base = SoccerTacticalLearningWeights::default();
        let summary = SoccerTacticalLearningSummary {
            mean_attack_width_score: 0.12,
            mean_attack_flank_lane_score: 0.10,
            mean_attack_spacing_score: 0.24,
            mean_defense_contract_score: 0.11,
            mean_defense_spacing_score: 0.31,
            mean_defense_ball_gap_score: 0.38,
            mean_defense_role_press_score: 0.34,
            ..Default::default()
        };
        let pressure = soccer_tactical_search_pressure(&summary);
        let options = SoccerEvolutionOptions {
            mutation_rate: 0.10,
            mutation_scale: 0.45,
            crossover_rate: 1.0,
            exploration_rate: 0.20,
            exploration_scale: 0.90,
            elite_weight_floor: 0.0,
            population_size: 12,
            seed: 59,
        };
        let mut candidates = Vec::new();

        search_soccer_tactical_mutated_gp_program_candidates(
            &base,
            &base,
            &summary,
            pressure,
            options,
            &mut |candidate| candidates.push(candidate),
        );

        assert_eq!(candidates.len(), options.population_size);
        assert!(candidates.iter().all(|candidate| {
            candidate.attack_flank_lane_weight <= 2.2
                && candidate.defense_contract_delta_weight <= 2.4
                && candidate.defender_midfielder_press_weight <= 1.6
                && candidate.midfielder_press_weight <= 1.6
        }));
        assert!(candidates.iter().any(|candidate| {
            candidate.attack_width_delta_weight > base.attack_width_delta_weight
                && candidate.attack_flank_lane_weight > base.attack_flank_lane_weight
        }));
        assert!(candidates.iter().any(|candidate| {
            candidate.defense_contract_delta_weight > base.defense_contract_delta_weight
                && candidate.defense_compactness_score_weight
                    > base.defense_compactness_score_weight
        }));
        assert!(candidates.iter().any(|candidate| {
            candidate.attack_flank_lane_weight > base.attack_flank_lane_weight
                && candidate.defense_contract_delta_weight > base.defense_contract_delta_weight
        }));
        assert!(candidates.iter().any(|candidate| {
            candidate.defender_midfielder_press_weight > base.defender_midfielder_press_weight
                || candidate.midfielder_press_weight > base.midfielder_press_weight
        }));
    }

    #[test]
    fn tactical_novelty_immigrants_expand_flank_compact_search_space() {
        let base = SoccerTacticalLearningWeights::default();
        let mut centroid = base.clone();
        centroid.attack_width_delta_weight = 1.10;
        centroid.attack_flank_lane_weight = 1.05;
        centroid.defense_contract_delta_weight = 1.20;
        centroid.defense_compactness_score_weight = 0.70;
        let summary = SoccerTacticalLearningSummary {
            mean_attack_width_score: 0.10,
            mean_attack_flank_lane_score: 0.09,
            mean_attack_spacing_score: 0.22,
            mean_defense_contract_score: 0.10,
            mean_defense_spacing_score: 0.30,
            mean_defense_ball_gap_score: 0.36,
            mean_defense_role_press_score: 0.32,
            ..Default::default()
        };
        let pressure = soccer_tactical_search_pressure(&summary);
        let options = SoccerEvolutionOptions {
            mutation_rate: 0.08,
            mutation_scale: 0.45,
            crossover_rate: 1.0,
            exploration_rate: 0.20,
            exploration_scale: 0.85,
            elite_weight_floor: 0.0,
            population_size: 10,
            seed: 83,
        };
        let mut candidates = Vec::new();

        search_soccer_tactical_novelty_immigrant_candidates(
            &base,
            &centroid,
            &summary,
            pressure,
            options,
            &mut |candidate| candidates.push(candidate),
        );

        assert_eq!(candidates.len(), options.population_size);
        assert!(candidates.iter().all(|candidate| {
            candidate.attack_flank_lane_weight <= 2.2
                && candidate.defense_contract_delta_weight <= 2.4
                && candidate.defense_compactness_score_weight <= 2.0
                && candidate.midfielder_press_weight <= 1.6
        }));
        assert!(candidates.iter().any(|candidate| {
            candidate.attack_flank_lane_weight >= centroid.attack_flank_lane_weight + 0.35
                && candidate.attack_width_delta_weight >= centroid.attack_width_delta_weight + 0.20
        }));
        assert!(candidates.iter().any(|candidate| {
            candidate.defense_contract_delta_weight >= centroid.defense_contract_delta_weight + 0.35
                && candidate.defense_compactness_score_weight
                    >= centroid.defense_compactness_score_weight + 0.25
        }));
        assert!(candidates.iter().any(|candidate| {
            candidate.attack_flank_lane_weight > centroid.attack_flank_lane_weight
                && candidate.defense_contract_delta_weight > centroid.defense_contract_delta_weight
                && soccer_tactical_weight_search_score(candidate, &summary)
                    > soccer_tactical_weight_search_score(&centroid, &summary)
        }));
    }

    #[test]
    fn tactical_genome_search_can_extrapolate_beyond_elite_parents() {
        let base = SoccerTacticalLearningWeights::default();
        let summary = SoccerTacticalLearningSummary {
            mean_attack_width_score: 0.12,
            mean_attack_flank_lane_score: 0.10,
            mean_attack_spacing_score: 0.28,
            mean_defense_contract_score: 0.14,
            mean_defense_spacing_score: 0.32,
            mean_defense_ball_gap_score: 0.42,
            mean_defense_role_press_score: 0.35,
            ..Default::default()
        };
        let mut left = base.clone();
        left.attack_width_delta_weight = 0.90;
        left.attack_flank_lane_weight = 0.85;
        left.defense_contract_delta_weight = 0.95;
        let mut right = base.clone();
        right.attack_width_delta_weight = 1.20;
        right.attack_flank_lane_weight = 1.15;
        right.defense_contract_delta_weight = 1.25;
        let options = SoccerEvolutionOptions {
            mutation_rate: 0.0,
            mutation_scale: 0.0,
            crossover_rate: 1.0,
            exploration_rate: 0.0,
            exploration_scale: 0.0,
            elite_weight_floor: 0.0,
            population_size: 1,
            seed: 47,
        };

        let evolved = evolve_soccer_tactical_learning_weights_from_genomes(
            &base,
            &[
                SoccerTacticalLearningGenomeParent {
                    summary: &summary,
                    weights: &left,
                    fitness: 1.0,
                },
                SoccerTacticalLearningGenomeParent {
                    summary: &summary,
                    weights: &right,
                    fitness: 3.0,
                },
            ],
            options,
        );

        assert!(evolved.attack_width_delta_weight > right.attack_width_delta_weight);
        assert!(evolved.attack_flank_lane_weight > right.attack_flank_lane_weight);
        assert!(evolved.defense_contract_delta_weight > right.defense_contract_delta_weight);
    }

    #[test]
    fn queue_runner_keeps_policy_output_available() {
        let options = SoccerQPolicyOptions::default();
        let report = run_soccer_learning_queue(
            SoccerLearningQueueRunnerConfig {
                games: 2,
                parallel_games: 2,
                base_seed: 1888,
                match_config: MatchConfig {
                    duration_seconds: 0.2,
                    learning_logging_enabled: false,
                    max_human_players: 0,
                    ..Default::default()
                },
                initial_neural_network: None,
                neural_drain_timeout: Duration::from_millis(0),
                options: options.clone(),
                prune_action_entries_per_team: 0,
                prune_target_entries_per_team: 0,
                min_policy_visits: 0,
            },
            SoccerTeamQPolicies::new(options),
        )
        .expect("queue run");

        assert_eq!(report.completed_games, 2);
        assert_eq!(report.failed_games, 0);
        assert_eq!(report.episode_summaries.len(), 2);
        assert_eq!(
            report
                .episode_summaries
                .iter()
                .map(|summary| summary.seed)
                .collect::<Vec<_>>(),
            vec![1888, 1889]
        );
        assert!(report.tactical_summary.total_transitions > 0);
        assert!(report.tactical_summary.shape_transitions > 0);
    }

    #[test]
    fn queue_starting_batch_can_update_match_config() {
        let options = SoccerQPolicyOptions::default();
        let mut saw_starting_batch = false;
        let report = run_soccer_learning_queue_with_events(
            SoccerLearningQueueRunnerConfig {
                games: 1,
                parallel_games: 1,
                base_seed: 1999,
                match_config: MatchConfig {
                    duration_seconds: 0.2,
                    learning_logging_enabled: false,
                    max_human_players: 0,
                    ..Default::default()
                },
                initial_neural_network: None,
                neural_drain_timeout: Duration::from_millis(0),
                options: options.clone(),
                prune_action_entries_per_team: 0,
                prune_target_entries_per_team: 0,
                min_policy_visits: 0,
            },
            SoccerTeamQPolicies::new(options),
            |event| {
                if let SoccerLearningQueueEvent::StartingBatch { match_config, .. } = event {
                    match_config.tactical_learning.attack_flank_lane_weight = 0.91;
                    match_config.tactical_learning.defense_contract_delta_weight = 0.73;
                    saw_starting_batch = true;
                }
                Ok(())
            },
        )
        .expect("queue run");

        assert!(saw_starting_batch);
        assert_eq!(report.completed_games, 1);
        assert_eq!(report.failed_games, 0);
    }

    #[test]
    fn queue_rejects_starting_batch_neural_config_that_would_sanitize_silently() {
        let options = SoccerQPolicyOptions::default();
        let err = run_soccer_learning_queue_with_events(
            SoccerLearningQueueRunnerConfig {
                games: 1,
                parallel_games: 1,
                base_seed: 2001,
                match_config: MatchConfig {
                    duration_seconds: 0.0,
                    half_duration_seconds: 0.0,
                    learning_logging_enabled: false,
                    max_human_players: 0,
                    neural_learning: SoccerNeuralLearningConfig {
                        enabled: true,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                initial_neural_network: None,
                neural_drain_timeout: Duration::from_millis(0),
                options: options.clone(),
                prune_action_entries_per_team: 0,
                prune_target_entries_per_team: 0,
                min_policy_visits: 0,
            },
            SoccerTeamQPolicies::new(options),
            |event| {
                if let SoccerLearningQueueEvent::StartingBatch { match_config, .. } = event {
                    match_config.neural_learning.batch_size = 0;
                }
                Ok(())
            },
        )
        .expect_err("queue should reject invalid per-episode neural config");

        assert!(err.contains("episode 0"), "{err}");
        assert!(err.contains("batchSize"), "{err}");
    }

    #[test]
    fn queue_rejects_invalid_q_policy_options_before_worker_start() {
        let options = SoccerQPolicyOptions {
            exploration_epsilon: f64::NAN,
            ..SoccerQPolicyOptions::default()
        };
        let err = run_soccer_learning_queue_with_events(
            SoccerLearningQueueRunnerConfig {
                games: 1,
                parallel_games: 1,
                base_seed: 2003,
                match_config: MatchConfig {
                    duration_seconds: 0.0,
                    half_duration_seconds: 0.0,
                    learning_logging_enabled: false,
                    max_human_players: 0,
                    ..Default::default()
                },
                initial_neural_network: None,
                neural_drain_timeout: Duration::from_millis(0),
                options,
                prune_action_entries_per_team: 0,
                prune_target_entries_per_team: 0,
                min_policy_visits: 0,
            },
            SoccerTeamQPolicies::new(SoccerQPolicyOptions::default()),
            |_| Ok(()),
        )
        .expect_err("queue should reject invalid q-policy options");

        assert!(err.contains("explorationEpsilon"), "{err}");
    }

    #[test]
    fn queue_completed_game_carries_episode_starting_tactical_weights() {
        let options = SoccerQPolicyOptions::default();
        let mut completed_weights = Vec::new();
        let report = run_soccer_learning_queue_with_events(
            SoccerLearningQueueRunnerConfig {
                games: 2,
                parallel_games: 1,
                base_seed: 2007,
                match_config: MatchConfig {
                    duration_seconds: 0.0,
                    half_duration_seconds: 0.0,
                    learning_logging_enabled: false,
                    max_human_players: 0,
                    ..Default::default()
                },
                initial_neural_network: None,
                neural_drain_timeout: Duration::from_millis(0),
                options: options.clone(),
                prune_action_entries_per_team: 0,
                prune_target_entries_per_team: 0,
                min_policy_visits: 0,
            },
            SoccerTeamQPolicies::new(options),
            |event| {
                match event {
                    SoccerLearningQueueEvent::StartingBatch {
                        next_episode,
                        match_config,
                        ..
                    } => {
                        match_config.tactical_learning.attack_flank_lane_weight =
                            0.90 + next_episode as f64 * 0.05;
                        match_config.tactical_learning.defense_contract_delta_weight =
                            0.70 + next_episode as f64 * 0.03;
                    }
                    SoccerLearningQueueEvent::CompletedGame { game, .. } => {
                        completed_weights.push((
                            game.episode,
                            game.starting_tactical_learning.attack_flank_lane_weight,
                            game.starting_tactical_learning
                                .defense_contract_delta_weight,
                        ));
                    }
                }
                Ok(())
            },
        )
        .expect("queue run");

        assert_eq!(report.completed_games, 2);
        assert_eq!(report.failed_games, 0);
        assert_eq!(completed_weights.len(), 2);
        completed_weights.sort_by_key(|(episode, _, _)| *episode);
        assert_eq!(completed_weights[0].0, 0);
        assert!((completed_weights[0].1 - 0.90).abs() < 1e-12);
        assert!((completed_weights[0].2 - 0.70).abs() < 1e-12);
        assert_eq!(completed_weights[1].0, 1);
        assert!((completed_weights[1].1 - 0.95).abs() < 1e-12);
        assert!((completed_weights[1].2 - 0.73).abs() < 1e-12);
    }

    #[test]
    fn queue_starting_batch_fires_before_each_enqueued_game() {
        let options = SoccerQPolicyOptions::default();
        let mut start_episodes = Vec::new();
        let report = run_soccer_learning_queue_with_events(
            SoccerLearningQueueRunnerConfig {
                games: 3,
                parallel_games: 3,
                base_seed: 2010,
                match_config: MatchConfig {
                    duration_seconds: 0.2,
                    learning_logging_enabled: false,
                    max_human_players: 0,
                    ..Default::default()
                },
                initial_neural_network: None,
                neural_drain_timeout: Duration::from_millis(0),
                options: options.clone(),
                prune_action_entries_per_team: 0,
                prune_target_entries_per_team: 0,
                min_policy_visits: 0,
            },
            SoccerTeamQPolicies::new(options),
            |event| {
                if let SoccerLearningQueueEvent::StartingBatch { next_episode, .. } = event {
                    start_episodes.push(next_episode);
                }
                Ok(())
            },
        )
        .expect("queue run");

        assert_eq!(report.completed_games, 3);
        assert_eq!(report.failed_games, 0);
        assert_eq!(start_episodes, vec![0, 1, 2]);
    }

    #[test]
    fn queue_starting_batch_replaces_policy_and_neural_snapshot_for_next_game() {
        let options = SoccerQPolicyOptions::default();
        let match_config = MatchConfig {
            duration_seconds: 0.0,
            half_duration_seconds: 0.0,
            learning_logging_enabled: false,
            max_human_players: 0,
            neural_learning: SoccerNeuralLearningConfig {
                enabled: true,
                backend: SoccerNeuralLearningBackend::Inline,
                hidden_units: 8,
                ..SoccerNeuralLearningConfig::default()
            },
            ..Default::default()
        };
        let refreshed = policy_with_action(12.0, 7);
        let mut refreshed_neural = SoccerMatch::default_11v11(match_config.clone())
            .learning_snapshot()
            .neural_network
            .expect("initial neural snapshot");
        refreshed_neural.layers[0].weights[0][0] = 0.424_242;
        refreshed_neural.layers[0].biases[0] = -0.313_131;
        let mut refreshed_episodes = Vec::new();
        let report = run_soccer_learning_queue_with_events(
            SoccerLearningQueueRunnerConfig {
                games: 1,
                parallel_games: 1,
                base_seed: 2031,
                match_config,
                initial_neural_network: None,
                neural_drain_timeout: Duration::from_millis(0),
                options: options.clone(),
                prune_action_entries_per_team: 0,
                prune_target_entries_per_team: 0,
                min_policy_visits: 0,
            },
            policy_with_action(-2.0, 1),
            |event| {
                if let SoccerLearningQueueEvent::StartingBatch {
                    next_episode,
                    policies,
                    neural_network,
                    ..
                } = event
                {
                    refreshed_episodes.push(next_episode);
                    *policies = refreshed.clone();
                    *neural_network = Some(refreshed_neural.clone());
                }
                Ok(())
            },
        )
        .expect("queue run");

        assert_eq!(refreshed_episodes, vec![0]);
        assert_eq!(report.completed_games, 1);
        assert_eq!(report.failed_games, 0);
        let entries = report.final_policies.home.entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].action, "pass");
        assert!((entries[0].value - 12.0).abs() < 1e-12);
        assert_eq!(entries[0].visits, 7);
        let latest_neural = report
            .latest_neural_network
            .expect("latest queue neural snapshot");
        assert_eq!(latest_neural.layers[0].weights[0][0], 0.424_242);
        assert_eq!(latest_neural.layers[0].biases[0], -0.313_131);
    }

    #[test]
    fn queue_completed_game_can_replace_merged_policy() {
        let options = SoccerQPolicyOptions::default();
        let replacement = policy_with_action(8.0, 3);
        let mut replaced = false;
        let report = run_soccer_learning_queue_with_events(
            SoccerLearningQueueRunnerConfig {
                games: 1,
                parallel_games: 1,
                base_seed: 2021,
                match_config: MatchConfig {
                    duration_seconds: 0.2,
                    learning_logging_enabled: false,
                    max_human_players: 0,
                    ..Default::default()
                },
                initial_neural_network: None,
                neural_drain_timeout: Duration::from_millis(0),
                options: options.clone(),
                prune_action_entries_per_team: 0,
                prune_target_entries_per_team: 0,
                min_policy_visits: 0,
            },
            SoccerTeamQPolicies::new(options),
            |event| {
                if let SoccerLearningQueueEvent::CompletedGame {
                    merged_policies, ..
                } = event
                {
                    *merged_policies = replacement.clone();
                    replaced = true;
                }
                Ok(())
            },
        )
        .expect("queue run");

        assert!(replaced);
        let entries = report.final_policies.home.entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].action, "pass");
        assert!((entries[0].value - 8.0).abs() < 1e-12);
        assert_eq!(entries[0].visits, 3);
    }

    #[test]
    fn test_state_uses_expected_public_variants() {
        let state = test_state();
        assert_eq!(state.phase, TacticalPhase::Kickoff);
        assert_eq!(state.role, PlayerRole::Midfielder);
    }

    #[test]
    fn quality_fitness_does_not_cancel_and_rewards_skill() {
        use crate::des::general::soccer::MatchStats;
        let make = |score_home: u32, score_away: u32, f: &dyn Fn(&mut MatchStats)| {
            let mut stats = MatchStats::default();
            f(&mut stats);
            MatchSummary {
                score_home,
                score_away,
                ticks: 100,
                simulated_seconds: 90.0,
                stats,
            }
        };

        // A dominant, incisive performance (goals + worked chances + forward progression)
        // must score strictly higher than a sterile sideways-passing game with no end product.
        let incisive = make(3, 1, &|s| {
            s.shots_on_target_home = 7;
            s.assists_home = 3;
            s.crosses_completed_home = 4;
            s.passes_completed_forward_home = 120;
            s.pass_chain_gain_yards_home = 180.0;
            s.shots_after_pass_home = 6;
        });
        let sterile = make(0, 0, &|s| {
            // Lots of safe completions but backwards/sideways: no forward product, chains lose
            // ground.
            s.passes_completed_forward_home = 8;
            s.pass_chains_net_loss_home = 14;
            s.pass_chain_gain_yards_home = -40.0;
        });
        let fitness = |summary: &MatchSummary| soccer_learning_run_score(summary).match_fitness;
        let incisive_fitness = fitness(&incisive);
        let sterile_fitness = fitness(&sterile);
        assert!(
            incisive_fitness > sterile_fitness + 1.0,
            "incisive ({incisive_fitness}) should clearly beat sterile ({sterile_fitness})"
        );

        // The old metric was ONLY total goals, so it was blind to HOW the goals came. With the
        // SAME scoreline (so equal winner-fitness and margin), the game richer in shots on
        // target, assists and forward progression must score higher — the dense gradient the
        // play-quality proxies are meant to provide.
        let worked = make(2, 2, &|s| {
            s.shots_home = 12;
            s.shots_away = 12;
            s.shots_on_target_home = 8;
            s.shots_on_target_away = 8;
            s.assists_home = 2;
            s.assists_away = 2;
            s.passes_completed_forward_home = 90;
            s.passes_completed_forward_away = 90;
            s.crosses_completed_home = 4;
            s.crosses_completed_away = 4;
            s.shots_after_pass_home = 6;
            s.shots_after_pass_away = 6;
            s.pass_chain_gain_yards_home = 140.0;
            s.pass_chain_gain_yards_away = 140.0;
        });
        let scrappy = make(2, 2, &|s| {
            s.shots_home = 3;
            s.shots_away = 3;
            s.shots_on_target_home = 2;
            s.shots_on_target_away = 2;
        });
        assert!(
            fitness(&worked) > fitness(&scrappy),
            "same scoreline: more chance-creation/progression must rank higher"
        );
    }
}
