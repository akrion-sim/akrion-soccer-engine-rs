//! Run accelerated soccer self-play and persist learned MDP/POMDP policies.

use std::collections::{BTreeMap, VecDeque};
use std::error::Error;
use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Error as IoError, ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use soccer_engine::des::general::soccer::{
    soccer_moment_records_from_jsonl, soccer_moment_records_to_learning_dataset,
    train_soccer_pass_completion_head, AttackSpacingHead, BackFourLineHead, CrashBoxHead,
    DefenderLinePolicyHead, GiveAndGoHead, GoalSideRecoveryHead, HeadScanHead, LaneAffinityHead,
    LongPassRunHead, LooseBallCommitHead, MatchConfig, MatchSummary, OnsideSupportHead,
    PassLaneYieldHead, ReceiveApproachHead, RunPredictionHead, SeparationFloorHead,
    ShotTriggerHead, SlipBreakHead, SoccerAuxiliaryHeadSnapshot, SoccerConfigMomentInsert,
    SoccerLearningTransition, SoccerMarlAlgorithm, SoccerMatch, SoccerMomentWindow,
    SoccerNeuralBlendMode, SoccerNeuralLayerSnapshot, SoccerNeuralLearningBackend,
    SoccerNeuralLearningConfig, SoccerNeuralNetworkSnapshot, SoccerPassCompletionHead,
    SoccerPassLearningMetrics, SoccerPassOutcomeSample, SoccerPolicyHeadSnapshot,
    SoccerPolicyRoleHeadSnapshot, SoccerPolicySpecialistHeadSnapshot, SoccerQEntry, SoccerQPolicy,
    SoccerQPolicyOptions, SoccerQTargetEntry, SoccerSelfPlayEpisodeSummary,
    SoccerSelfPlayLearnedParams, SoccerSelfPlayTrainingArtifact, SoccerTacticalLearningSummary,
    SoccerTacticalLearningWeights, SoccerTeamPolicyArtifact, SoccerTeamQPolicies, SoccerWorldModel,
    SupportScorerHead, Team, WingerPinchHead, ATTACK_SPACING_HEAD_MIN_TRAINING_STEPS,
    CRASH_BOX_HEAD_MIN_TRAINING_STEPS, DEFAULT_SOCCER_MAPPO_TEAM_REWARD_SHARE,
    DEFAULT_SOCCER_PASS_COMPLETION_LEARNING_RATE, GIVE_AND_GO_HEAD_MIN_TRAINING_STEPS,
    GOAL_SIDE_RECOVERY_HEAD_MIN_TRAINING_STEPS, HEAD_SCAN_HEAD_MIN_TRAINING_STEPS,
    LANE_AFFINITY_HEAD_MIN_TRAINING_STEPS, LONG_PASS_RUN_HEAD_MIN_TRAINING_STEPS,
    LOOSE_BALL_COMMIT_HEAD_MIN_TRAINING_STEPS, ONSIDE_SUPPORT_HEAD_MIN_TRAINING_STEPS,
    PASS_COMPLETION_HEAD_MIN_TRAINING_STEPS, PASS_LANE_YIELD_HEAD_MIN_TRAINING_STEPS,
    RECEIVE_APPROACH_HEAD_MIN_TRAINING_STEPS, RUN_PREDICTION_HEAD_MIN_TRAINING_STEPS,
    SEPARATION_FLOOR_HEAD_MIN_TRAINING_STEPS, SHOT_TRIGGER_HEAD_MIN_TRAINING_STEPS,
    SLIP_BREAK_HEAD_MIN_TRAINING_STEPS, SUPPORT_SCORER_HEAD_MIN_TRAINING_STEPS,
    WINGER_PINCH_HEAD_MIN_TRAINING_STEPS,
};
use soccer_engine::des::general::soccer_eval_gate::{
    evaluate_promotion, PromotionThresholds, PromotionVerdict,
};
use soccer_engine::des::general::tournament::{
    EngineMatchRunner, EngineMatchRunnerConfig, MatchReport, TeamBrain, TournamentMatchContext,
    TournamentMatchRunner, TournamentStage,
};
use soccer_engine::des::soccer_learning::{
    evaluate_soccer_policy_promotion_gate, evolve_soccer_tactical_learning_weights_from_genomes,
    evolve_soccer_team_policies, merge_soccer_policy_deltas,
    soccer_evolution_options_from_search_metadata, soccer_learning_curriculum_episode_config,
    soccer_learning_curriculum_stage_for_completed_games,
    soccer_learning_directional_objective_fitness, soccer_learning_run_score,
    soccer_neural_network_snapshot_fingerprint, soccer_policy_active_max_fitness_regression,
    soccer_policy_delta_entries, soccer_policy_version_insert_status_after_active_head,
    soccer_postgres_new_sim_refresh_plan, soccer_postgres_policy_refresh_decision,
    soccer_should_flush_postgres_policy_versions_for_new_sim,
    soccer_should_refresh_postgres_for_new_sim, soccer_tactical_learning_weights_fingerprint,
    soccer_team_q_policies_fingerprint, validate_soccer_evolution_options_for_learning_run,
    validate_soccer_learning_curriculum_config_for_learning_run,
    validate_soccer_neural_learning_config_for_learning_run,
    validate_soccer_policy_promotion_gate_config_for_learning_run, SoccerEvolutionOptions,
    SoccerLearningCompletedGame, SoccerLearningCurriculumConfig, SoccerPolicyPromotionGateConfig,
    SoccerPolicyPromotionGateEvaluation, SoccerPostgresPolicyRefreshCheck,
    SoccerTacticalLearningGenomeParent, SOCCER_POLICY_STATUS_ACTIVE, SOCCER_POLICY_STATUS_ARCHIVED,
};
use soccer_engine::des::soccer_learning_pg::{
    soccer_learning_git_commit, SoccerLearningPgCompletedRunInsert,
    SoccerLearningPgPolicyPromotionBaseline, SoccerLearningPgStore,
};
use uuid::Uuid;

const DEFAULT_SOCCER_NEURAL_DRAIN_TIMEOUT_MS: usize = 2;
const DEFAULT_SOCCER_POSTGRES_POLICY_VERSION_INTERVAL_GAMES: usize = 10;
const DEFAULT_SOCCER_POSTGRES_COMPLETED_RUN_BATCH_GAMES: usize = 10;
const DEFAULT_SOCCER_POSTGRES_COMPLETED_RUN_RETENTION_GAMES: usize = 0;
const DEFAULT_SOCCER_POSTGRES_ASYNC_BATCH_QUEUE: usize = 4;
const DEFAULT_SOCCER_POSTGRES_ASYNC_COALESCE_BATCHES: usize = 8;
const DEFAULT_SOCCER_POSTGRES_ASYNC_COALESCE_WAIT_MS: usize = 2;
const DEFAULT_SOCCER_POSTGRES_TACTICAL_LEARNING_AUTHORITATIVE: bool = true;
const DEFAULT_SOCCER_POSTGRES_REFRESH_WITH_RESUME_ARTIFACT: bool = true;
const DEFAULT_SOCCER_POSTGRES_FLUSH_POLICY_VERSIONS_BEFORE_NEW_SIM: bool = true;
const DEFAULT_SOCCER_POSTGRES_TRAINING_REPLAY_DELTA_ROWS: usize = 50_000;
const DEFAULT_SOCCER_POSTGRES_TRAINING_REPLAY_PRIOR_WEIGHT: f64 = 1.0;
const DEFAULT_SOCCER_POLICY_PROMOTION_BASELINE_LOOKBACK_GENERATIONS: usize = 16;
const DEFAULT_SOCCER_CONTINUOUS_PARALLEL_GAMES: usize = 1;
const DEFAULT_SOCCER_WRITE_GAME_ARTIFACTS: bool = false;
const DEFAULT_SOCCER_WRITE_FINAL_POLICY_ARTIFACT: bool = true;
const DEFAULT_SOCCER_WRITE_EPISODE_LOG: bool = true;
const DEFAULT_SOCCER_PRINT_COMPLETED_GAMES: bool = false;
const DEFAULT_SOCCER_EVOLUTION_ENABLED: bool = true;
const DEFAULT_SOCCER_EVOLUTION_INTERVAL_GAMES: usize = 10;
const DEFAULT_SOCCER_EVOLUTION_ELITE_GAMES: usize = 4;
const DEFAULT_SOCCER_NEURAL_POPULATION_SEARCH_ENABLED: bool = false;
const DEFAULT_SOCCER_NEURAL_POPULATION_INTERVAL_GAMES: usize = 6;
const DEFAULT_SOCCER_NEURAL_POPULATION_SIZE: usize = 6;
const MAX_SOCCER_NEURAL_POPULATION_SIZE: usize = 16;
const DEFAULT_SOCCER_NEURAL_POPULATION_EVAL_GAMES: usize = 6;
const DEFAULT_SOCCER_NEURAL_POPULATION_EVAL_MINUTES: f64 = 0.75;
const DEFAULT_SOCCER_NEURAL_POPULATION_MUTATION_RATE: f64 = 0.08;
const DEFAULT_SOCCER_NEURAL_POPULATION_MUTATION_SCALE: f64 = 0.04;
const DEFAULT_SOCCER_NEURAL_POPULATION_CROSSOVER_RATE: f64 = 0.55;
const DEFAULT_SOCCER_NEURAL_POPULATION_MIN_FITNESS_DELTA: f64 = 0.05;
const DEFAULT_SOCCER_NEURAL_POPULATION_MIN_ACCEPTED_FITNESS: f64 = 0.25;
const DEFAULT_SOCCER_NEURAL_POPULATION_MIN_ACCEPTED_GOAL_MARGIN: f64 = 1.0;
const DEFAULT_SOCCER_NEURAL_POPULATION_CONFIRM_GAMES: usize = 6;
const DEFAULT_SOCCER_NEURAL_POPULATION_ACCEPT_FREEZE_GAMES: usize = 0;
const SOCCER_LEARNING_LOCAL_MPC_MAX_PLAYERS_PER_TEAM_LIMIT: usize = 11;
const SOCCER_POLICY_SOURCE_MERGE: &str = "merge";
const SOCCER_POLICY_SOURCE_EVOLUTION: &str = "mutation";
const SOCCER_POLICY_SOURCE_NEURAL_CHECKPOINT: &str = "replay";
const SOCCER_LEARNING_OBJECTIVE_FITNESS_MIN: f64 = -8.0;
const SOCCER_LEARNING_OBJECTIVE_FITNESS_MAX: f64 = 12.0;
const DEFAULT_SOCCER_NEURAL_BATCH_SNAPSHOT_MAX_FITNESS_REGRESSION: f64 = 0.75;
const DEFAULT_SOCCER_NEURAL_BATCH_SNAPSHOT_MIN_FITNESS: f64 = SOCCER_LEARNING_OBJECTIVE_FITNESS_MIN;
const DEFAULT_SOCCER_NEURAL_BATCH_SNAPSHOT_MIN_BATCH_MEAN_FITNESS: f64 =
    SOCCER_LEARNING_OBJECTIVE_FITNESS_MIN;
const DEFAULT_SOCCER_NEURAL_BATCH_SNAPSHOT_RESCUE_MIN_FITNESS: f64 =
    SOCCER_LEARNING_OBJECTIVE_FITNESS_MAX + 1.0;
const DEFAULT_SOCCER_NEURAL_BATCH_SNAPSHOT_VALIDATE_GAMES: usize = 0;
const DEFAULT_SOCCER_NEURAL_BATCH_SNAPSHOT_VALIDATE_MIN_FITNESS: f64 =
    DEFAULT_SOCCER_NEURAL_POPULATION_MIN_ACCEPTED_FITNESS;
const DEFAULT_SOCCER_NEURAL_BATCH_SNAPSHOT_VALIDATE_MIN_GOAL_MARGIN: f64 = 0.0;
const DEFAULT_SOCCER_NEURAL_BATCH_SNAPSHOT_CARRY_MIN_VALIDATION_FITNESS: f64 =
    SOCCER_LEARNING_OBJECTIVE_FITNESS_MIN;
const DEFAULT_SOCCER_NEURAL_BATCH_SNAPSHOT_CARRY_MIN_VALIDATION_GOAL_MARGIN: f64 =
    SOCCER_LEARNING_OBJECTIVE_FITNESS_MIN;

fn env_value(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn env_usize(name: &str, default: usize) -> Result<usize, Box<dyn Error>> {
    let Some(value) = env_value(name) else {
        return Ok(default);
    };
    value.parse::<usize>().map_err(|_| {
        invalid_data(format!("{name} must be an unsigned integer, got {value:?}")).into()
    })
}

/// Minimum visit count an entry must have for the learner to LOAD it on resume.
/// `0` (default) preserves the historical behavior of resuming the full policy exactly.
/// A positive value loads only the high-visit core (like the live server's
/// `SOCCER_LIVE_POLICY_MIN_VISITS`), so the Rust process holds a lean working set instead of
/// dragging the whole multi-million-row tail into RAM. The full policy stays in Postgres (each
/// cycle persists a NEW version; old versions are retained/pruned by the retention policy), so
/// nothing is lost from the source of truth — only the in-process footprint shrinks. Tolerant
/// parse: a malformed value falls back to 0 rather than failing the run.
fn resume_min_visits() -> i32 {
    env_value("SOCCER_RESUME_MIN_VISITS")
        .and_then(|value| value.parse::<i32>().ok())
        .unwrap_or(0)
        .max(0)
}

fn resume_include_unpromoted() -> bool {
    env_value("SOCCER_RESUME_INCLUDE_UNPROMOTED")
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn postgres_persist_pass_outcome_samples() -> bool {
    env_value("SOCCER_PG_PERSIST_PASS_OUTCOMES")
        .map(|value| {
            !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off"
            )
        })
        .unwrap_or(true)
}

fn env_u32(name: &str, default: u32) -> Result<u32, Box<dyn Error>> {
    let Some(value) = env_value(name) else {
        return Ok(default);
    };
    value
        .parse::<u32>()
        .map_err(|_| invalid_data(format!("{name} must be a u32, got {value:?}")).into())
}

fn env_u64(name: &str, default: u64) -> Result<u64, Box<dyn Error>> {
    let Some(value) = env_value(name) else {
        return Ok(default);
    };
    value
        .parse::<u64>()
        .map_err(|_| invalid_data(format!("{name} must be a u64, got {value:?}")).into())
}

fn env_f64(name: &str, default: f64) -> Result<f64, Box<dyn Error>> {
    let Some(value) = env_value(name) else {
        return Ok(default);
    };
    let parsed = value
        .parse::<f64>()
        .map_err(|_| invalid_data(format!("{name} must be a finite number, got {value:?}")))?;
    if !parsed.is_finite() {
        return Err(invalid_data(format!("{name} must be finite, got {value:?}")).into());
    }
    Ok(parsed)
}

fn env_bool(name: &str, default: bool) -> Result<bool, Box<dyn Error>> {
    let Some(value) = env_value(name) else {
        return Ok(default);
    };
    match value.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "y" | "on" => Ok(true),
        "0" | "false" | "no" | "n" | "off" => Ok(false),
        _ => Err(invalid_data(format!("{name} must be a boolean, got {value:?}")).into()),
    }
}

fn env_f64_alias(primary: &str, alias: &str, default: f64) -> Result<f64, Box<dyn Error>> {
    if env_value(primary).is_some() {
        env_f64(primary, default)
    } else {
        env_f64(alias, default)
    }
}

fn env_usize_alias(primary: &str, alias: &str, default: usize) -> Result<usize, Box<dyn Error>> {
    if env_value(primary).is_some() {
        env_usize(primary, default)
    } else {
        env_usize(alias, default)
    }
}

fn env_bool_alias(primary: &str, alias: &str, default: bool) -> Result<bool, Box<dyn Error>> {
    if env_value(primary).is_some() {
        env_bool(primary, default)
    } else {
        env_bool(alias, default)
    }
}

fn neural_backend_label(backend: SoccerNeuralLearningBackend) -> &'static str {
    match backend {
        SoccerNeuralLearningBackend::Inline => "inline",
        SoccerNeuralLearningBackend::Threaded => "threaded",
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BatchNeuralSnapshotSelectionMode {
    Fitness,
    TrainingSteps,
    TrainingStepsGuarded,
    LatestEpisode,
}

impl BatchNeuralSnapshotSelectionMode {
    fn label(self) -> &'static str {
        match self {
            Self::Fitness => "fitness",
            Self::TrainingSteps => "training_steps",
            Self::TrainingStepsGuarded => "training_steps_guarded",
            Self::LatestEpisode => "latest_episode",
        }
    }
}

fn env_batch_neural_snapshot_selection_mode(
) -> Result<BatchNeuralSnapshotSelectionMode, Box<dyn Error>> {
    let Some(value) = env_value("SOCCER_NEURAL_BATCH_SNAPSHOT_SELECTION") else {
        return Ok(BatchNeuralSnapshotSelectionMode::Fitness);
    };
    match value.to_ascii_lowercase().replace('-', "_").as_str() {
        "fitness" | "best_fitness" | "best" => Ok(BatchNeuralSnapshotSelectionMode::Fitness),
        "training_steps" | "steps" | "most_trained" => {
            Ok(BatchNeuralSnapshotSelectionMode::TrainingSteps)
        }
        "training_steps_guarded" | "guarded_training_steps" | "fresh_guarded"
        | "guarded_fresh" => Ok(BatchNeuralSnapshotSelectionMode::TrainingStepsGuarded),
        "latest_episode" | "latest" | "episode" => Ok(BatchNeuralSnapshotSelectionMode::LatestEpisode),
        _ => Err(invalid_data(format!(
            "SOCCER_NEURAL_BATCH_SNAPSHOT_SELECTION must be fitness, training_steps, training_steps_guarded, or latest_episode, got {value:?}"
        ))
        .into()),
    }
}

fn env_batch_neural_snapshot_max_fitness_regression() -> Result<f64, Box<dyn Error>> {
    let regression = env_f64(
        "SOCCER_NEURAL_BATCH_SNAPSHOT_MAX_FITNESS_REGRESSION",
        DEFAULT_SOCCER_NEURAL_BATCH_SNAPSHOT_MAX_FITNESS_REGRESSION,
    )?;
    if regression < 0.0 {
        return Err(invalid_data(
            "SOCCER_NEURAL_BATCH_SNAPSHOT_MAX_FITNESS_REGRESSION must be non-negative",
        )
        .into());
    }
    Ok(regression)
}

fn env_batch_neural_snapshot_min_fitness() -> Result<f64, Box<dyn Error>> {
    env_f64(
        "SOCCER_NEURAL_BATCH_SNAPSHOT_MIN_FITNESS",
        DEFAULT_SOCCER_NEURAL_BATCH_SNAPSHOT_MIN_FITNESS,
    )
}

fn env_batch_neural_snapshot_min_batch_mean_fitness() -> Result<f64, Box<dyn Error>> {
    env_f64(
        "SOCCER_NEURAL_BATCH_SNAPSHOT_MIN_BATCH_MEAN_FITNESS",
        DEFAULT_SOCCER_NEURAL_BATCH_SNAPSHOT_MIN_BATCH_MEAN_FITNESS,
    )
}

fn env_batch_neural_snapshot_rescue_min_fitness() -> Result<f64, Box<dyn Error>> {
    env_f64(
        "SOCCER_NEURAL_BATCH_SNAPSHOT_RESCUE_MIN_FITNESS",
        DEFAULT_SOCCER_NEURAL_BATCH_SNAPSHOT_RESCUE_MIN_FITNESS,
    )
}

fn soccer_learning_objective_match_fitness(
    summary: &MatchSummary,
    analytic_neural_opponent: bool,
) -> f64 {
    let score = soccer_learning_run_score(summary);
    if analytic_neural_opponent {
        soccer_learning_directional_objective_fitness(Team::Home, summary).clamp(
            SOCCER_LEARNING_OBJECTIVE_FITNESS_MIN,
            SOCCER_LEARNING_OBJECTIVE_FITNESS_MAX,
        )
    } else {
        score.match_fitness
    }
}

fn evaluate_soccer_policy_promotion_gate_for_learning_objective(
    summaries: &[&MatchSummary],
    config: SoccerPolicyPromotionGateConfig,
    analytic_neural_opponent: bool,
) -> SoccerPolicyPromotionGateEvaluation {
    let mut evaluation = evaluate_soccer_policy_promotion_gate(summaries.iter().copied(), config);
    if !analytic_neural_opponent {
        return evaluation;
    }
    let mut sum = 0.0;
    let mut best = f64::NEG_INFINITY;
    let mut non_finite = 0usize;
    for summary in summaries {
        let fitness = soccer_learning_objective_match_fitness(summary, true);
        if fitness.is_finite() {
            sum += fitness;
            best = best.max(fitness);
        } else {
            non_finite = non_finite.saturating_add(1);
            sum += SOCCER_LEARNING_OBJECTIVE_FITNESS_MIN;
            best = best.max(SOCCER_LEARNING_OBJECTIVE_FITNESS_MIN);
        }
    }
    let denominator = summaries.len().max(1) as f64;
    evaluation.mean_match_fitness = sum / denominator;
    evaluation.best_match_fitness = if best.is_finite() {
        best
    } else {
        SOCCER_LEARNING_OBJECTIVE_FITNESS_MIN
    };
    evaluation.rejection_reasons.retain(|reason| {
        !reason.starts_with("mean_match_fitness ")
            && !reason.starts_with("best_match_fitness ")
            && !reason.starts_with("non_finite_learning_objective ")
    });
    if config.enabled {
        if non_finite > 0 {
            evaluation.rejection_reasons.push(format!(
                "non_finite_learning_objective {non_finite} samples"
            ));
        }
        if evaluation.mean_match_fitness < config.min_mean_match_fitness {
            evaluation.rejection_reasons.push(format!(
                "mean_match_fitness {:.4} below min_mean_match_fitness {:.4}",
                evaluation.mean_match_fitness, config.min_mean_match_fitness
            ));
        }
        if evaluation.best_match_fitness < config.min_best_match_fitness {
            evaluation.rejection_reasons.push(format!(
                "best_match_fitness {:.4} below min_best_match_fitness {:.4}",
                evaluation.best_match_fitness, config.min_best_match_fitness
            ));
        }
    }
    evaluation.eligible = !config.enabled || evaluation.rejection_reasons.is_empty();
    evaluation
}

fn policy_promotion_evaluation_can_anchor_local_trial(
    evaluation: &SoccerPolicyPromotionGateEvaluation,
    anchor_held_trials: bool,
) -> bool {
    if !evaluation.mean_match_fitness.is_finite() || evaluation.mean_match_fitness < 0.0 {
        return false;
    }
    if evaluation
        .rejection_reasons
        .iter()
        .all(|reason| reason.starts_with("sample_games ") || reason == "local_best_not_improved")
    {
        return true;
    }
    anchor_held_trials
        && evaluation
            .rejection_reasons
            .iter()
            .all(|reason| !reason.starts_with("non_finite_learning_objective "))
}

fn validated_snapshot_can_update_resume_only_training_best(
    kept_validated_snapshot: bool,
    beats_local_best: bool,
    has_live_publish_sample_floor: bool,
    validation_mean_match_fitness: f64,
) -> bool {
    kept_validated_snapshot
        && beats_local_best
        && !has_live_publish_sample_floor
        && validation_mean_match_fitness.is_finite()
        && validation_mean_match_fitness >= 0.0
}

fn default_postgres_policy_version_interval_games(parallel_games: usize) -> usize {
    parallel_games.max(DEFAULT_SOCCER_POSTGRES_POLICY_VERSION_INTERVAL_GAMES)
}

fn default_postgres_completed_run_batch_games(parallel_games: usize) -> usize {
    parallel_games.max(DEFAULT_SOCCER_POSTGRES_COMPLETED_RUN_BATCH_GAMES)
}

fn default_evolution_interval_games(parallel_games: usize) -> usize {
    parallel_games.max(DEFAULT_SOCCER_EVOLUTION_INTERVAL_GAMES)
}

fn default_evolution_window_games(
    evolution_interval_games: usize,
    evolution_elite_games: usize,
) -> usize {
    evolution_interval_games.max(evolution_elite_games).max(1)
}

#[derive(Clone, Copy, Debug)]
struct NeuralPopulationSearchConfig {
    enabled: bool,
    interval_games: usize,
    population_size: usize,
    eval_games: usize,
    eval_minutes: f64,
    mutation_rate: f64,
    mutation_scale: f64,
    crossover_rate: f64,
    min_fitness_delta: f64,
    min_accepted_fitness: f64,
    min_accepted_goal_margin: f64,
    training_min_accepted_fitness: f64,
    training_min_accepted_goal_margin: f64,
    confirm_games: usize,
    confirm_min_fitness_delta: f64,
    confirm_min_accepted_fitness: f64,
    confirm_min_accepted_goal_margin: f64,
    training_confirm_min_accepted_fitness: f64,
    training_confirm_min_accepted_goal_margin: f64,
    seed: u64,
}

fn env_neural_population_search_config(
    seed: u64,
    parallel_games: usize,
) -> Result<NeuralPopulationSearchConfig, Box<dyn Error>> {
    let enabled = env_bool(
        "SOCCER_NEURAL_POPULATION_SEARCH_ENABLED",
        DEFAULT_SOCCER_NEURAL_POPULATION_SEARCH_ENABLED,
    )?;
    let interval_games = env_usize(
        "SOCCER_NEURAL_POPULATION_INTERVAL_GAMES",
        parallel_games.max(DEFAULT_SOCCER_NEURAL_POPULATION_INTERVAL_GAMES),
    )?
    .max(1);
    let population_size = env_usize(
        "SOCCER_NEURAL_POPULATION_SIZE",
        DEFAULT_SOCCER_NEURAL_POPULATION_SIZE,
    )?
    .max(2)
    .min(MAX_SOCCER_NEURAL_POPULATION_SIZE);
    let eval_games = env_usize(
        "SOCCER_NEURAL_POPULATION_EVAL_GAMES",
        DEFAULT_SOCCER_NEURAL_POPULATION_EVAL_GAMES,
    )?
    .max(1);
    let eval_minutes = env_f64(
        "SOCCER_NEURAL_POPULATION_EVAL_MINUTES",
        DEFAULT_SOCCER_NEURAL_POPULATION_EVAL_MINUTES,
    )?;
    if eval_minutes <= 0.0 {
        return Err(invalid_data("SOCCER_NEURAL_POPULATION_EVAL_MINUTES must be positive").into());
    }
    let mutation_rate = env_f64(
        "SOCCER_NEURAL_POPULATION_MUTATION_RATE",
        DEFAULT_SOCCER_NEURAL_POPULATION_MUTATION_RATE,
    )?;
    if !(0.0..=1.0).contains(&mutation_rate) {
        return Err(
            invalid_data("SOCCER_NEURAL_POPULATION_MUTATION_RATE must be in [0, 1]").into(),
        );
    }
    let mutation_scale = env_f64(
        "SOCCER_NEURAL_POPULATION_MUTATION_SCALE",
        DEFAULT_SOCCER_NEURAL_POPULATION_MUTATION_SCALE,
    )?;
    if mutation_scale < 0.0 {
        return Err(
            invalid_data("SOCCER_NEURAL_POPULATION_MUTATION_SCALE must be non-negative").into(),
        );
    }
    let crossover_rate = env_f64(
        "SOCCER_NEURAL_POPULATION_CROSSOVER_RATE",
        DEFAULT_SOCCER_NEURAL_POPULATION_CROSSOVER_RATE,
    )?;
    if !(0.0..=1.0).contains(&crossover_rate) {
        return Err(
            invalid_data("SOCCER_NEURAL_POPULATION_CROSSOVER_RATE must be in [0, 1]").into(),
        );
    }
    let min_fitness_delta = env_f64(
        "SOCCER_NEURAL_POPULATION_MIN_FITNESS_DELTA",
        DEFAULT_SOCCER_NEURAL_POPULATION_MIN_FITNESS_DELTA,
    )?;
    if min_fitness_delta < 0.0 {
        return Err(invalid_data(
            "SOCCER_NEURAL_POPULATION_MIN_FITNESS_DELTA must be non-negative",
        )
        .into());
    }
    let min_accepted_fitness = env_f64(
        "SOCCER_NEURAL_POPULATION_MIN_ACCEPTED_FITNESS",
        DEFAULT_SOCCER_NEURAL_POPULATION_MIN_ACCEPTED_FITNESS,
    )?;
    if !min_accepted_fitness.is_finite() {
        return Err(
            invalid_data("SOCCER_NEURAL_POPULATION_MIN_ACCEPTED_FITNESS must be finite").into(),
        );
    }
    let min_accepted_goal_margin = env_f64(
        "SOCCER_NEURAL_POPULATION_MIN_ACCEPTED_GOAL_MARGIN",
        DEFAULT_SOCCER_NEURAL_POPULATION_MIN_ACCEPTED_GOAL_MARGIN,
    )?;
    if !min_accepted_goal_margin.is_finite() {
        return Err(invalid_data(
            "SOCCER_NEURAL_POPULATION_MIN_ACCEPTED_GOAL_MARGIN must be finite",
        )
        .into());
    }
    let training_min_accepted_fitness = env_f64(
        "SOCCER_NEURAL_POPULATION_TRAINING_MIN_ACCEPTED_FITNESS",
        min_accepted_fitness,
    )?;
    if !training_min_accepted_fitness.is_finite() {
        return Err(invalid_data(
            "SOCCER_NEURAL_POPULATION_TRAINING_MIN_ACCEPTED_FITNESS must be finite",
        )
        .into());
    }
    let training_min_accepted_goal_margin = env_f64(
        "SOCCER_NEURAL_POPULATION_TRAINING_MIN_ACCEPTED_GOAL_MARGIN",
        min_accepted_goal_margin,
    )?;
    if !training_min_accepted_goal_margin.is_finite() {
        return Err(invalid_data(
            "SOCCER_NEURAL_POPULATION_TRAINING_MIN_ACCEPTED_GOAL_MARGIN must be finite",
        )
        .into());
    }
    let confirm_games = env_usize(
        "SOCCER_NEURAL_POPULATION_CONFIRM_GAMES",
        DEFAULT_SOCCER_NEURAL_POPULATION_CONFIRM_GAMES,
    )?;
    let confirm_min_fitness_delta = env_f64(
        "SOCCER_NEURAL_POPULATION_CONFIRM_MIN_FITNESS_DELTA",
        min_fitness_delta,
    )?;
    if confirm_min_fitness_delta < 0.0 {
        return Err(invalid_data(
            "SOCCER_NEURAL_POPULATION_CONFIRM_MIN_FITNESS_DELTA must be non-negative",
        )
        .into());
    }
    let confirm_min_accepted_fitness = env_f64(
        "SOCCER_NEURAL_POPULATION_CONFIRM_MIN_ACCEPTED_FITNESS",
        min_accepted_fitness,
    )?;
    if !confirm_min_accepted_fitness.is_finite() {
        return Err(invalid_data(
            "SOCCER_NEURAL_POPULATION_CONFIRM_MIN_ACCEPTED_FITNESS must be finite",
        )
        .into());
    }
    let confirm_min_accepted_goal_margin = env_f64(
        "SOCCER_NEURAL_POPULATION_CONFIRM_MIN_ACCEPTED_GOAL_MARGIN",
        min_accepted_goal_margin,
    )?;
    if !confirm_min_accepted_goal_margin.is_finite() {
        return Err(invalid_data(
            "SOCCER_NEURAL_POPULATION_CONFIRM_MIN_ACCEPTED_GOAL_MARGIN must be finite",
        )
        .into());
    }
    let training_confirm_min_accepted_fitness = env_f64(
        "SOCCER_NEURAL_POPULATION_TRAINING_CONFIRM_MIN_ACCEPTED_FITNESS",
        confirm_min_accepted_fitness,
    )?;
    if !training_confirm_min_accepted_fitness.is_finite() {
        return Err(invalid_data(
            "SOCCER_NEURAL_POPULATION_TRAINING_CONFIRM_MIN_ACCEPTED_FITNESS must be finite",
        )
        .into());
    }
    let training_confirm_min_accepted_goal_margin = env_f64(
        "SOCCER_NEURAL_POPULATION_TRAINING_CONFIRM_MIN_ACCEPTED_GOAL_MARGIN",
        confirm_min_accepted_goal_margin,
    )?;
    if !training_confirm_min_accepted_goal_margin.is_finite() {
        return Err(invalid_data(
            "SOCCER_NEURAL_POPULATION_TRAINING_CONFIRM_MIN_ACCEPTED_GOAL_MARGIN must be finite",
        )
        .into());
    }
    let search_seed = env_u64(
        "SOCCER_NEURAL_POPULATION_SEED",
        seed ^ 0xA5A5_5A5A_D3C3_B4B4,
    )?;
    Ok(NeuralPopulationSearchConfig {
        enabled,
        interval_games,
        population_size,
        eval_games,
        eval_minutes,
        mutation_rate,
        mutation_scale,
        crossover_rate,
        min_fitness_delta,
        min_accepted_fitness,
        min_accepted_goal_margin,
        training_min_accepted_fitness,
        training_min_accepted_goal_margin,
        confirm_games,
        confirm_min_fitness_delta,
        confirm_min_accepted_fitness,
        confirm_min_accepted_goal_margin,
        training_confirm_min_accepted_fitness,
        training_confirm_min_accepted_goal_margin,
        seed: search_seed,
    })
}

/// Deterministic splitmix64 RNG for local neural population search. The search
/// must be reproducible so a "climb" is not just a lucky one-off process seed.
struct NeuralPopulationRng {
    state: u64,
}

impl NeuralPopulationRng {
    fn new(seed: u64) -> Self {
        NeuralPopulationRng {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    fn gaussian(&mut self) -> f64 {
        let u1 = self.unit().max(1e-12);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    fn coin(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }

    fn index(&mut self, len: usize) -> usize {
        if len == 0 {
            0
        } else {
            (self.next_u64() as usize) % len
        }
    }
}

fn neural_layer_rms(layer: &SoccerNeuralLayerSnapshot) -> f64 {
    let mut sum_sq = 0.0;
    let mut count = 0usize;
    for row in &layer.weights {
        for weight in row {
            if weight.is_finite() {
                sum_sq += *weight * *weight;
                count += 1;
            }
        }
    }
    for bias in &layer.biases {
        if bias.is_finite() {
            sum_sq += *bias * *bias;
            count += 1;
        }
    }
    if count == 0 {
        0.01
    } else {
        (sum_sq / count as f64).sqrt().max(0.01)
    }
}

fn refresh_neural_snapshot_norm(snapshot: &mut SoccerNeuralNetworkSnapshot) {
    let mut sum_sq = 0.0;
    let mut count = 0usize;
    for layer in &mut snapshot.layers {
        for row in &mut layer.weights {
            for weight in row.iter_mut() {
                if !weight.is_finite() {
                    *weight = 0.0;
                }
                sum_sq += *weight * *weight;
                count += 1;
            }
        }
        for bias in layer.biases.iter_mut() {
            if !bias.is_finite() {
                *bias = 0.0;
            }
            sum_sq += *bias * *bias;
            count += 1;
        }
    }
    snapshot.parameter_count = count;
    snapshot.l2_norm = sum_sq.sqrt();
}

fn refresh_neural_snapshot_norm_recursive(
    snapshot: &mut SoccerNeuralNetworkSnapshot,
    depth: usize,
) {
    refresh_neural_snapshot_norm(snapshot);
    if depth >= 8 {
        return;
    }
    if let Some(policy_head) = snapshot.policy_head.as_mut() {
        refresh_policy_head_snapshot_norm(policy_head, depth + 1);
    }
    if let Some(line_depth_head) = snapshot.line_depth_head.as_mut() {
        refresh_auxiliary_head_snapshot_norm(line_depth_head, depth + 1);
    }
}

fn refresh_auxiliary_head_snapshot_norm(head: &mut SoccerAuxiliaryHeadSnapshot, depth: usize) {
    refresh_neural_snapshot_norm_recursive(&mut head.network, depth + 1);
}

fn refresh_specialist_head_snapshot_norm(
    head: &mut SoccerPolicySpecialistHeadSnapshot,
    depth: usize,
) {
    refresh_neural_snapshot_norm_recursive(&mut head.network, depth + 1);
}

fn refresh_role_head_snapshot_norm(head: &mut SoccerPolicyRoleHeadSnapshot, depth: usize) {
    refresh_neural_snapshot_norm_recursive(&mut head.network, depth + 1);
    for specialist in &mut head.specialist_heads {
        refresh_specialist_head_snapshot_norm(specialist, depth + 1);
    }
}

fn refresh_policy_head_snapshot_norm(head: &mut SoccerPolicyHeadSnapshot, depth: usize) {
    refresh_neural_snapshot_norm_recursive(&mut head.network, depth + 1);
    for specialist in &mut head.specialist_heads {
        refresh_specialist_head_snapshot_norm(specialist, depth + 1);
    }
    for role_head in &mut head.role_heads {
        refresh_role_head_snapshot_norm(role_head, depth + 1);
    }
}

fn mutate_neural_layer(
    layer: &mut SoccerNeuralLayerSnapshot,
    rng: &mut NeuralPopulationRng,
    mutation_rate: f64,
    mutation_scale: f64,
) {
    if mutation_rate <= 0.0 || mutation_scale <= 0.0 {
        return;
    }
    let sigma = neural_layer_rms(layer) * mutation_scale;
    for row in &mut layer.weights {
        for weight in row.iter_mut() {
            if rng.unit() <= mutation_rate {
                *weight += rng.gaussian() * sigma;
            }
        }
    }
    for bias in &mut layer.biases {
        if rng.unit() <= mutation_rate {
            *bias += rng.gaussian() * sigma;
        }
    }
}

fn mutate_neural_snapshot_recursive(
    mut snapshot: SoccerNeuralNetworkSnapshot,
    rng: &mut NeuralPopulationRng,
    config: NeuralPopulationSearchConfig,
    depth: usize,
) -> SoccerNeuralNetworkSnapshot {
    for layer in &mut snapshot.layers {
        mutate_neural_layer(layer, rng, config.mutation_rate, config.mutation_scale);
    }
    if depth < 8 {
        if let Some(policy_head) = snapshot.policy_head.as_mut() {
            mutate_policy_head_snapshot(policy_head, rng, config, depth + 1);
        }
        if let Some(line_depth_head) = snapshot.line_depth_head.as_mut() {
            mutate_auxiliary_head_snapshot(line_depth_head, rng, config, depth + 1);
        }
    }
    refresh_neural_snapshot_norm_recursive(&mut snapshot, depth);
    snapshot
}

fn mutate_auxiliary_head_snapshot(
    head: &mut SoccerAuxiliaryHeadSnapshot,
    rng: &mut NeuralPopulationRng,
    config: NeuralPopulationSearchConfig,
    depth: usize,
) {
    let network = std::mem::take(&mut head.network);
    head.network = mutate_neural_snapshot_recursive(network, rng, config, depth + 1);
}

fn mutate_specialist_head_snapshot(
    head: &mut SoccerPolicySpecialistHeadSnapshot,
    rng: &mut NeuralPopulationRng,
    config: NeuralPopulationSearchConfig,
    depth: usize,
) {
    let network = std::mem::take(&mut head.network);
    head.network = mutate_neural_snapshot_recursive(network, rng, config, depth + 1);
}

fn mutate_role_head_snapshot(
    head: &mut SoccerPolicyRoleHeadSnapshot,
    rng: &mut NeuralPopulationRng,
    config: NeuralPopulationSearchConfig,
    depth: usize,
) {
    let network = std::mem::take(&mut head.network);
    head.network = mutate_neural_snapshot_recursive(network, rng, config, depth + 1);
    for specialist in &mut head.specialist_heads {
        mutate_specialist_head_snapshot(specialist, rng, config, depth + 1);
    }
}

fn mutate_policy_head_snapshot(
    head: &mut SoccerPolicyHeadSnapshot,
    rng: &mut NeuralPopulationRng,
    config: NeuralPopulationSearchConfig,
    depth: usize,
) {
    let network = std::mem::take(&mut head.network);
    head.network = mutate_neural_snapshot_recursive(network, rng, config, depth + 1);
    for specialist in &mut head.specialist_heads {
        mutate_specialist_head_snapshot(specialist, rng, config, depth + 1);
    }
    for role_head in &mut head.role_heads {
        mutate_role_head_snapshot(role_head, rng, config, depth + 1);
    }
}

fn crossover_neural_snapshot_layers(
    a: &SoccerNeuralNetworkSnapshot,
    b: &SoccerNeuralNetworkSnapshot,
    rng: &mut NeuralPopulationRng,
) -> SoccerNeuralNetworkSnapshot {
    let mut child = a.clone();
    for (layer_index, layer) in child.layers.iter_mut().enumerate() {
        let Some(b_layer) = b.layers.get(layer_index) else {
            continue;
        };
        for (row_index, row) in layer.weights.iter_mut().enumerate() {
            for (column_index, weight) in row.iter_mut().enumerate() {
                if rng.coin() {
                    if let Some(b_weight) = b_layer
                        .weights
                        .get(row_index)
                        .and_then(|b_row| b_row.get(column_index))
                    {
                        *weight = *b_weight;
                    }
                }
            }
        }
        for (bias_index, bias) in layer.biases.iter_mut().enumerate() {
            if rng.coin() {
                if let Some(b_bias) = b_layer.biases.get(bias_index) {
                    *bias = *b_bias;
                }
            }
        }
    }
    refresh_neural_snapshot_norm(&mut child);
    child
}

fn crossover_neural_snapshot_recursive(
    a: &SoccerNeuralNetworkSnapshot,
    b: &SoccerNeuralNetworkSnapshot,
    rng: &mut NeuralPopulationRng,
    depth: usize,
) -> SoccerNeuralNetworkSnapshot {
    let mut child = crossover_neural_snapshot_layers(a, b, rng);
    if depth < 8 {
        if let (Some(child_head), Some(b_head)) =
            (child.policy_head.as_mut(), b.policy_head.as_ref())
        {
            crossover_policy_head_snapshot(child_head, b_head, rng, depth + 1);
        } else if child.policy_head.is_none() && b.policy_head.is_some() && rng.coin() {
            child.policy_head = b.policy_head.clone();
        }
        if let (Some(child_head), Some(b_head)) =
            (child.line_depth_head.as_mut(), b.line_depth_head.as_ref())
        {
            crossover_auxiliary_head_snapshot(child_head, b_head, rng, depth + 1);
        } else if child.line_depth_head.is_none() && b.line_depth_head.is_some() && rng.coin() {
            child.line_depth_head = b.line_depth_head.clone();
        }
    }
    refresh_neural_snapshot_norm_recursive(&mut child, depth);
    child
}

fn crossover_auxiliary_head_snapshot(
    child: &mut SoccerAuxiliaryHeadSnapshot,
    b: &SoccerAuxiliaryHeadSnapshot,
    rng: &mut NeuralPopulationRng,
    depth: usize,
) {
    let current = std::mem::take(&mut child.network);
    child.network = crossover_neural_snapshot_recursive(&current, &b.network, rng, depth + 1);
}

fn crossover_specialist_head_snapshot(
    child: &mut SoccerPolicySpecialistHeadSnapshot,
    b: &SoccerPolicySpecialistHeadSnapshot,
    rng: &mut NeuralPopulationRng,
    depth: usize,
) {
    if child.kind != b.kind && rng.coin() {
        *child = b.clone();
        refresh_specialist_head_snapshot_norm(child, depth + 1);
        return;
    }
    let current = std::mem::take(&mut child.network);
    child.network = crossover_neural_snapshot_recursive(&current, &b.network, rng, depth + 1);
}

fn crossover_role_head_snapshot(
    child: &mut SoccerPolicyRoleHeadSnapshot,
    b: &SoccerPolicyRoleHeadSnapshot,
    rng: &mut NeuralPopulationRng,
    depth: usize,
) {
    if child.role != b.role && rng.coin() {
        *child = b.clone();
        refresh_role_head_snapshot_norm(child, depth + 1);
        return;
    }
    let current = std::mem::take(&mut child.network);
    child.network = crossover_neural_snapshot_recursive(&current, &b.network, rng, depth + 1);
    for (index, child_specialist) in child.specialist_heads.iter_mut().enumerate() {
        if let Some(b_specialist) = b
            .specialist_heads
            .iter()
            .find(|specialist| specialist.kind == child_specialist.kind)
            .or_else(|| b.specialist_heads.get(index))
        {
            crossover_specialist_head_snapshot(child_specialist, b_specialist, rng, depth + 1);
        }
    }
    if child.specialist_heads.len() < b.specialist_heads.len() {
        for b_specialist in b.specialist_heads.iter().skip(child.specialist_heads.len()) {
            if rng.unit() < 0.25 {
                child.specialist_heads.push(b_specialist.clone());
            }
        }
    }
}

fn crossover_policy_head_snapshot(
    child: &mut SoccerPolicyHeadSnapshot,
    b: &SoccerPolicyHeadSnapshot,
    rng: &mut NeuralPopulationRng,
    depth: usize,
) {
    let current = std::mem::take(&mut child.network);
    child.network = crossover_neural_snapshot_recursive(&current, &b.network, rng, depth + 1);
    for (index, child_specialist) in child.specialist_heads.iter_mut().enumerate() {
        if let Some(b_specialist) = b
            .specialist_heads
            .iter()
            .find(|specialist| specialist.kind == child_specialist.kind)
            .or_else(|| b.specialist_heads.get(index))
        {
            crossover_specialist_head_snapshot(child_specialist, b_specialist, rng, depth + 1);
        }
    }
    if child.specialist_heads.len() < b.specialist_heads.len() {
        for b_specialist in b.specialist_heads.iter().skip(child.specialist_heads.len()) {
            if rng.unit() < 0.25 {
                child.specialist_heads.push(b_specialist.clone());
            }
        }
    }
    for (index, child_role) in child.role_heads.iter_mut().enumerate() {
        if let Some(b_role) = b
            .role_heads
            .iter()
            .find(|role| role.role == child_role.role)
            .or_else(|| b.role_heads.get(index))
        {
            crossover_role_head_snapshot(child_role, b_role, rng, depth + 1);
        }
    }
    if child.role_heads.len() < b.role_heads.len() {
        for b_role in b.role_heads.iter().skip(child.role_heads.len()) {
            if rng.unit() < 0.25 {
                child.role_heads.push(b_role.clone());
            }
        }
    }
}

#[derive(Clone)]
struct NeuralPopulationCandidate {
    index: usize,
    source: String,
    snapshot: SoccerNeuralNetworkSnapshot,
}

#[derive(Clone)]
struct NeuralPopulationCandidateEval {
    index: usize,
    source: String,
    fitness: f64,
    wins: usize,
    draws: usize,
    losses: usize,
    goals_for: u32,
    goals_against: u32,
    snapshot: SoccerNeuralNetworkSnapshot,
}

#[derive(Clone)]
struct NeuralPopulationSearchAcceptance {
    eval: NeuralPopulationCandidateEval,
    preserve_as_local_best: bool,
}

fn neural_population_candidate_eval_is_baseline(eval: &NeuralPopulationCandidateEval) -> bool {
    eval.index == 0 || eval.source.ends_with("_baseline")
}

fn neural_population_candidate_goal_margin(eval: &NeuralPopulationCandidateEval) -> f64 {
    let games = (eval.wins + eval.draws + eval.losses).max(1) as f64;
    (f64::from(eval.goals_for) - f64::from(eval.goals_against)) / games
}

fn neural_population_eval_seed(search_seed: u64, completed_games: usize) -> u64 {
    search_seed ^ (completed_games as u64).wrapping_mul(0x94D0_49BB_1331_11EB)
}

fn neural_population_confirm_seed(search_seed: u64, completed_games: usize) -> u64 {
    neural_population_eval_seed(search_seed, completed_games) ^ 0xD6E8_FD93_51E2_B37D
}

fn neural_population_eval_game_seed(seed: u64, game_index: usize) -> u64 {
    seed ^ (game_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

fn neural_population_improvement_clears_delta(improvement: f64, min_delta: f64) -> bool {
    improvement.is_finite() && improvement + 1e-12 >= min_delta
}

fn neural_population_accept_protection_blocks_local_best_restore(remaining_games: usize) -> bool {
    remaining_games > 0
}

fn neural_population_accept_restore_protection_games(
    acceptance: &NeuralPopulationSearchAcceptance,
    configured_games: usize,
) -> usize {
    if acceptance.eval.fitness.is_finite() {
        configured_games
    } else {
        0
    }
}

fn neural_population_accept_learning_freeze_games(
    acceptance: &NeuralPopulationSearchAcceptance,
    configured_games: usize,
) -> usize {
    if acceptance.preserve_as_local_best {
        configured_games
    } else {
        0
    }
}

fn policy_promotion_evaluation_from_candidate_validation(
    eval: &NeuralPopulationCandidateEval,
    gate: SoccerPolicyPromotionGateConfig,
) -> SoccerPolicyPromotionGateEvaluation {
    let sample_games = eval.wins + eval.draws + eval.losses;
    let denominator = sample_games.max(1) as f64;
    SoccerPolicyPromotionGateEvaluation {
        enabled: gate.enabled,
        eligible: true,
        sample_games,
        min_sample_games: gate.min_sample_games,
        mean_match_fitness: eval.fitness,
        best_match_fitness: eval.fitness,
        mean_play_quality: 0.0,
        mean_conceded_goals: f64::from(eval.goals_against) / denominator,
        mean_goal_margin: neural_population_candidate_goal_margin(eval),
        mean_chain_net_loss: 0.0,
        rejection_reasons: Vec::new(),
    }
}

fn batch_neural_snapshot_validation_passes(
    eval: &NeuralPopulationCandidateEval,
    min_fitness: f64,
    min_goal_margin: f64,
) -> bool {
    eval.fitness + 1e-12 >= min_fitness
        && neural_population_candidate_goal_margin(eval) + 1e-12 >= min_goal_margin
}

fn push_unique_neural_parent(
    parents: &mut Vec<(String, SoccerNeuralNetworkSnapshot)>,
    source: &str,
    snapshot: &SoccerNeuralNetworkSnapshot,
) {
    let fingerprint = soccer_neural_network_snapshot_fingerprint(snapshot);
    if parents
        .iter()
        .any(|(_, existing)| soccer_neural_network_snapshot_fingerprint(existing) == fingerprint)
    {
        return;
    }
    parents.push((source.to_string(), snapshot.clone()));
}

fn neural_population_exploration_config(
    config: NeuralPopulationSearchConfig,
    candidate_index: usize,
) -> (NeuralPopulationSearchConfig, &'static str) {
    let mut child_config = config;
    match candidate_index % 4 {
        0 => {
            child_config.mutation_rate = (config.mutation_rate * 0.5).clamp(0.0, 1.0);
            child_config.mutation_scale = config.mutation_scale * 0.5;
            (child_config, "micro")
        }
        1 => (child_config, "base"),
        2 => {
            child_config.mutation_rate = (config.mutation_rate * 1.5).clamp(0.0, 1.0);
            child_config.mutation_scale = config.mutation_scale * 1.75;
            (child_config, "wide")
        }
        _ => {
            child_config.mutation_rate = (config.mutation_rate * 0.25).clamp(0.0, 1.0);
            child_config.mutation_scale = config.mutation_scale * 1.25;
            (child_config, "pinpoint")
        }
    }
}

fn push_neural_population_mutation_candidate(
    candidates: &mut Vec<NeuralPopulationCandidate>,
    source: &str,
    snapshot: &SoccerNeuralNetworkSnapshot,
    rng: &mut NeuralPopulationRng,
    config: NeuralPopulationSearchConfig,
    exploration_index: usize,
) {
    let child_index = candidates.len();
    let (child_config, exploration_label) =
        neural_population_exploration_config(config, exploration_index);
    let child = mutate_neural_snapshot_recursive(snapshot.clone(), rng, child_config, 0);
    candidates.push(NeuralPopulationCandidate {
        index: child_index,
        source: format!("mutation:{source}:{exploration_label}"),
        snapshot: child,
    });
}

fn build_neural_population_candidates(
    incumbent: &SoccerNeuralNetworkSnapshot,
    local_best: Option<&LocalNeuralPromotionTrialBest>,
    anchor: Option<&SoccerNeuralNetworkSnapshot>,
    config: NeuralPopulationSearchConfig,
    completed_games: usize,
) -> Vec<NeuralPopulationCandidate> {
    let mut parents = Vec::new();
    push_unique_neural_parent(&mut parents, "incumbent", incumbent);
    if let Some(local_best_snapshot) = local_best.and_then(|best| best.neural_network.as_ref()) {
        push_unique_neural_parent(&mut parents, "local_best", local_best_snapshot);
    }
    if let Some(anchor_snapshot) = anchor {
        push_unique_neural_parent(&mut parents, "anchor", anchor_snapshot);
    }

    let mut candidates = Vec::new();
    candidates.push(NeuralPopulationCandidate {
        index: 0,
        source: "incumbent".to_string(),
        snapshot: incumbent.clone(),
    });
    for (source, snapshot) in parents.iter().skip(1) {
        if candidates.len() >= config.population_size {
            break;
        }
        candidates.push(NeuralPopulationCandidate {
            index: candidates.len(),
            source: format!("{source}_baseline"),
            snapshot: snapshot.clone(),
        });
    }

    let mut rng = NeuralPopulationRng::new(
        config.seed
            ^ (completed_games as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)
            ^ (config.population_size as u64).rotate_left(17),
    );
    for (source, snapshot) in parents.iter().skip(1) {
        for exploration_index in 0..4 {
            if candidates.len() >= config.population_size {
                break;
            }
            push_neural_population_mutation_candidate(
                &mut candidates,
                source,
                snapshot,
                &mut rng,
                config,
                exploration_index,
            );
        }
    }
    if parents.len() >= 2 && candidates.len() < config.population_size {
        let (parent_source, parent_snapshot) = &parents[0];
        let (other_source, other_snapshot) = &parents[1];
        let child_index = candidates.len();
        let (child_config, exploration_label) =
            neural_population_exploration_config(config, child_index);
        let child =
            crossover_neural_snapshot_recursive(parent_snapshot, other_snapshot, &mut rng, 0);
        let child = mutate_neural_snapshot_recursive(child, &mut rng, child_config, 0);
        candidates.push(NeuralPopulationCandidate {
            index: child_index,
            source: format!("crossover:{parent_source}+{other_source}:{exploration_label}"),
            snapshot: child,
        });
    }
    while candidates.len() < config.population_size {
        let parent_index = rng.index(parents.len());
        let (parent_source, parent_snapshot) = &parents[parent_index];
        let child_index = candidates.len();
        let (child_config, exploration_label) =
            neural_population_exploration_config(config, child_index);
        let (source, child) = if parents.len() >= 2 && rng.unit() < config.crossover_rate {
            let mut other_index = rng.index(parents.len());
            if other_index == parent_index {
                other_index = (other_index + 1) % parents.len();
            }
            let (other_source, other_snapshot) = &parents[other_index];
            (
                format!("crossover:{parent_source}+{other_source}:{exploration_label}"),
                crossover_neural_snapshot_recursive(parent_snapshot, other_snapshot, &mut rng, 0),
            )
        } else {
            (
                format!("mutation:{parent_source}:{exploration_label}"),
                parent_snapshot.clone(),
            )
        };
        let child = mutate_neural_snapshot_recursive(child, &mut rng, child_config, 0);
        candidates.push(NeuralPopulationCandidate {
            index: child_index,
            source,
            snapshot: child,
        });
    }
    candidates
}

fn neural_population_eval_runner_config(
    base: &MatchConfig,
    config: NeuralPopulationSearchConfig,
) -> EngineMatchRunnerConfig {
    let mut eval_base = base.clone();
    eval_base.max_human_players = 0;
    eval_base.duration_seconds = config.eval_minutes * 60.0;
    eval_base.learning_enabled = false;
    eval_base.learning_logging_enabled = false;
    eval_base.full_game_learning_enabled = false;
    eval_base.neural_learning.enabled = true;
    eval_base.neural_learning.backend = SoccerNeuralLearningBackend::Inline;
    EngineMatchRunnerConfig {
        base: eval_base,
        match_wall_time_limit: Some(Duration::from_secs_f64(
            (config.eval_minutes * 60.0 * 4.0).max(20.0),
        )),
    }
}

fn run_home_neural_against_analytic_learning_eval_match(
    snapshot: &SoccerNeuralNetworkSnapshot,
    base_config: &MatchConfig,
    search_config: NeuralPopulationSearchConfig,
    seed: u64,
    match_index: usize,
) -> Result<MatchSummary, String> {
    let EngineMatchRunnerConfig {
        base: mut config,
        match_wall_time_limit,
    } = neural_population_eval_runner_config(base_config, search_config);
    config.seed = seed as u32;
    let mut sim = SoccerMatch::default_11v11(config);
    sim.set_uniform_elite_players();
    sim.set_team_neural_brain(Team::Home, Some(snapshot.clone()), true)?;
    sim.disable_team_neural_brain(Team::Away);

    let wall_started = Instant::now();
    let mut guard = 0u64;
    let max_ticks = sim
        .config
        .total_ticks()
        .saturating_mul(2)
        .saturating_add(512);
    while !sim.is_done() {
        sim.run_time_step();
        guard = guard.saturating_add(1);
        if match_wall_time_limit.is_some_and(|limit| wall_started.elapsed() >= limit) {
            eprintln!(
                "neural_batch_snapshot_learning_eval_wall_timeout match_index={} wall_secs={:.1} ticks={} max_ticks={} score={}-{}",
                match_index,
                wall_started.elapsed().as_secs_f64(),
                guard,
                max_ticks,
                sim.score_home,
                sim.score_away
            );
            break;
        }
        if guard > max_ticks {
            eprintln!(
                "neural_batch_snapshot_learning_eval_guard_tripped match_index={} max_ticks={} score={}-{}",
                match_index, max_ticks, sim.score_home, sim.score_away
            );
            break;
        }
    }
    sim.drain_neural_learning(Duration::from_millis(500));
    Ok(sim.summary())
}

fn evaluate_neural_population_candidate_against_analytic_home_learning_objective(
    candidate: NeuralPopulationCandidate,
    base_config: MatchConfig,
    search_config: NeuralPopulationSearchConfig,
    seed: u64,
) -> Result<NeuralPopulationCandidateEval, String> {
    let mut total_fitness = 0.0;
    let mut wins = 0usize;
    let mut draws = 0usize;
    let mut losses = 0usize;
    let mut goals_for = 0u32;
    let mut goals_against = 0u32;
    for game_index in 0..search_config.eval_games {
        let game_seed = neural_population_eval_game_seed(seed, game_index);
        let summary = run_home_neural_against_analytic_learning_eval_match(
            &candidate.snapshot,
            &base_config,
            search_config,
            game_seed,
            game_index,
        )?;
        let fitness = soccer_learning_objective_match_fitness(&summary, true);
        total_fitness += fitness;
        goals_for = goals_for.saturating_add(summary.score_home);
        goals_against = goals_against.saturating_add(summary.score_away);
        match summary.score_home.cmp(&summary.score_away) {
            std::cmp::Ordering::Greater => wins = wins.saturating_add(1),
            std::cmp::Ordering::Equal => draws = draws.saturating_add(1),
            std::cmp::Ordering::Less => losses = losses.saturating_add(1),
        }
    }
    Ok(NeuralPopulationCandidateEval {
        index: candidate.index,
        source: candidate.source,
        fitness: total_fitness / search_config.eval_games.max(1) as f64,
        wins,
        draws,
        losses,
        goals_for,
        goals_against,
        snapshot: candidate.snapshot,
    })
}

#[allow(clippy::too_many_arguments)]
fn maybe_run_neural_population_search(
    completed_games: usize,
    config: &MatchConfig,
    search_config: NeuralPopulationSearchConfig,
    local_best: Option<&LocalNeuralPromotionTrialBest>,
    anchor: Option<&SoccerNeuralNetworkSnapshot>,
    latest_neural_network: &mut Option<SoccerNeuralNetworkSnapshot>,
    latest_neural_network_arc_cache: &mut Option<CachedNeuralNetworkSnapshotArc>,
) -> Result<Option<NeuralPopulationSearchAcceptance>, Box<dyn Error>> {
    if !search_config.enabled
        || search_config.interval_games == 0
        || completed_games == 0
        || completed_games % search_config.interval_games != 0
    {
        return Ok(None);
    }
    let Some(incumbent) = latest_neural_network.as_ref() else {
        println!(
            "neural_population_search_held completed_games={} reasons=missing_neural_snapshot",
            completed_games
        );
        return Ok(None);
    };
    let candidates = build_neural_population_candidates(
        incumbent,
        local_best,
        anchor,
        search_config,
        completed_games,
    );
    println!(
        "neural_population_search_start completed_games={} population={} eval_games={} eval_minutes={:.2} confirm_games={} objective=home_directional_learning_vs_analytic mutation_rate={:.4} mutation_scale={:.4} crossover_rate={:.4} min_fitness_delta={:.4} min_accepted_fitness={:.4} min_accepted_goal_margin={:.4} training_min_accepted_fitness={:.4} training_min_accepted_goal_margin={:.4} confirm_min_delta={:.4} confirm_min_accepted_fitness={:.4} confirm_min_accepted_goal_margin={:.4} training_confirm_min_accepted_fitness={:.4} training_confirm_min_accepted_goal_margin={:.4}",
        completed_games,
        candidates.len(),
        search_config.eval_games,
        search_config.eval_minutes,
        search_config.confirm_games,
        search_config.mutation_rate,
        search_config.mutation_scale,
        search_config.crossover_rate,
        search_config.min_fitness_delta,
        search_config.min_accepted_fitness,
        search_config.min_accepted_goal_margin,
        search_config.training_min_accepted_fitness,
        search_config.training_min_accepted_goal_margin,
        search_config.confirm_min_fitness_delta,
        search_config.confirm_min_accepted_fitness,
        search_config.confirm_min_accepted_goal_margin,
        search_config.training_confirm_min_accepted_fitness,
        search_config.training_confirm_min_accepted_goal_margin
    );
    let mut handles = Vec::new();
    let eval_seed = neural_population_eval_seed(search_config.seed, completed_games);
    for candidate in candidates {
        let eval_config = config.clone();
        handles.push(thread::spawn(move || {
            evaluate_neural_population_candidate_against_analytic_home_learning_objective(
                candidate,
                eval_config,
                search_config,
                eval_seed,
            )
        }));
    }

    let mut evals = Vec::new();
    for handle in handles {
        match handle.join() {
            Ok(Ok(candidate_eval)) => {
                println!(
                    "neural_population_candidate_eval index={} source={} fitness={:.4} record={}-{}-{} goals={}-{}",
                    candidate_eval.index,
                    candidate_eval.source,
                    candidate_eval.fitness,
                    candidate_eval.wins,
                    candidate_eval.draws,
                    candidate_eval.losses,
                    candidate_eval.goals_for,
                    candidate_eval.goals_against
                );
                evals.push(candidate_eval);
            }
            Ok(Err(error)) => {
                println!(
                    "neural_population_candidate_failed completed_games={} error={}",
                    completed_games, error
                );
            }
            Err(_) => {
                println!(
                    "neural_population_candidate_failed completed_games={} error=thread_panicked",
                    completed_games
                );
            }
        }
    }
    let Some(incumbent_eval) = evals.iter().find(|candidate| candidate.index == 0) else {
        println!(
            "neural_population_search_held completed_games={} reasons=incumbent_eval_failed",
            completed_games
        );
        return Ok(None);
    };
    let training_reference_eval = evals
        .iter()
        .filter(|candidate| neural_population_candidate_eval_is_baseline(candidate))
        .max_by(|a, b| a.fitness.total_cmp(&b.fitness))
        .cloned()
        .unwrap_or_else(|| incumbent_eval.clone());
    let mut protected_reference_eval = training_reference_eval.clone();
    if let Some(best) = local_best {
        let metadata_fitness = best.evaluation.mean_match_fitness;
        if metadata_fitness.is_finite()
            && metadata_fitness > protected_reference_eval.fitness + 1e-9
        {
            protected_reference_eval.index = usize::MAX;
            protected_reference_eval.source = "local_best_metadata_floor".to_string();
            protected_reference_eval.fitness = metadata_fitness;
            protected_reference_eval.wins = 0;
            protected_reference_eval.draws = best.evaluation.sample_games;
            protected_reference_eval.losses = 0;
            protected_reference_eval.goals_for = 0;
            protected_reference_eval.goals_against = 0;
        }
    }
    let Some(best_eval) = evals
        .iter()
        .max_by(|a, b| a.fitness.total_cmp(&b.fitness))
        .cloned()
    else {
        println!(
            "neural_population_search_held completed_games={} reasons=no_successful_evals",
            completed_games
        );
        return Ok(None);
    };
    let training_improvement = best_eval.fitness - training_reference_eval.fitness;
    let protected_improvement = best_eval.fitness - protected_reference_eval.fitness;
    if neural_population_candidate_eval_is_baseline(&best_eval) {
        if best_eval.index != 0 && best_eval.fitness > incumbent_eval.fitness + 1e-9 {
            *latest_neural_network = Some(best_eval.snapshot.clone());
            *latest_neural_network_arc_cache = None;
            println!(
                "neural_population_search_restored_baseline completed_games={} incumbent_fitness={:.4} restored_index={} restored_source={} restored_fitness={:.4} training_reference_index={} training_reference_source={} training_reference_fitness={:.4} protected_reference_index={} protected_reference_source={} protected_reference_fitness={:.4} record={}-{}-{} goals={}-{} note=baseline_restore_not_plateau_breakthrough",
                completed_games,
                incumbent_eval.fitness,
                best_eval.index,
                best_eval.source,
                best_eval.fitness,
                training_reference_eval.index,
                training_reference_eval.source,
                training_reference_eval.fitness,
                protected_reference_eval.index,
                protected_reference_eval.source,
                protected_reference_eval.fitness,
                best_eval.wins,
                best_eval.draws,
                best_eval.losses,
                best_eval.goals_for,
                best_eval.goals_against
            );
            return Ok(Some(NeuralPopulationSearchAcceptance {
                eval: best_eval,
                preserve_as_local_best: false,
            }));
        }
        println!(
            "neural_population_search_held completed_games={} incumbent_fitness={:.4} best_index={} best_source={} best_fitness={:.4} training_reference_index={} training_reference_source={} training_reference_fitness={:.4} protected_reference_index={} protected_reference_source={} protected_reference_fitness={:.4} training_improvement={:.4} protected_improvement={:.4} min_delta={:.4} reasons=best_candidate_is_baseline",
            completed_games,
            incumbent_eval.fitness,
            best_eval.index,
            best_eval.source,
            best_eval.fitness,
            training_reference_eval.index,
            training_reference_eval.source,
            training_reference_eval.fitness,
            protected_reference_eval.index,
            protected_reference_eval.source,
            protected_reference_eval.fitness,
            training_improvement,
            protected_improvement,
            search_config.min_fitness_delta
        );
        return Ok(None);
    }
    if !neural_population_improvement_clears_delta(
        training_improvement,
        search_config.min_fitness_delta,
    ) {
        println!(
            "neural_population_search_held completed_games={} incumbent_fitness={:.4} training_reference_index={} training_reference_source={} training_reference_fitness={:.4} protected_reference_index={} protected_reference_source={} protected_reference_fitness={:.4} best_index={} best_source={} best_fitness={:.4} training_improvement={:.4} protected_improvement={:.4} min_delta={:.4} reasons=insufficient_heldout_gain",
            completed_games,
            incumbent_eval.fitness,
            training_reference_eval.index,
            training_reference_eval.source,
            training_reference_eval.fitness,
            protected_reference_eval.index,
            protected_reference_eval.source,
            protected_reference_eval.fitness,
            best_eval.index,
            best_eval.source,
            best_eval.fitness,
            training_improvement,
            protected_improvement,
            search_config.min_fitness_delta
        );
        return Ok(None);
    }
    if best_eval.fitness < search_config.training_min_accepted_fitness {
        println!(
            "neural_population_search_held completed_games={} incumbent_fitness={:.4} training_reference_index={} training_reference_source={} training_reference_fitness={:.4} protected_reference_index={} protected_reference_source={} protected_reference_fitness={:.4} best_index={} best_source={} best_fitness={:.4} training_improvement={:.4} protected_improvement={:.4} min_accepted_fitness={:.4} reasons=below_training_min_accepted_fitness",
            completed_games,
            incumbent_eval.fitness,
            training_reference_eval.index,
            training_reference_eval.source,
            training_reference_eval.fitness,
            protected_reference_eval.index,
            protected_reference_eval.source,
            protected_reference_eval.fitness,
            best_eval.index,
            best_eval.source,
            best_eval.fitness,
            training_improvement,
            protected_improvement,
            search_config.training_min_accepted_fitness
        );
        return Ok(None);
    }
    let best_goal_margin = neural_population_candidate_goal_margin(&best_eval);
    if best_goal_margin < search_config.training_min_accepted_goal_margin {
        println!(
            "neural_population_search_held completed_games={} incumbent_fitness={:.4} training_reference_index={} training_reference_source={} training_reference_fitness={:.4} protected_reference_index={} protected_reference_source={} protected_reference_fitness={:.4} best_index={} best_source={} best_fitness={:.4} training_improvement={:.4} protected_improvement={:.4} goals={}-{} goal_margin={:.4} min_accepted_goal_margin={:.4} reasons=below_training_min_accepted_goal_margin",
            completed_games,
            incumbent_eval.fitness,
            training_reference_eval.index,
            training_reference_eval.source,
            training_reference_eval.fitness,
            protected_reference_eval.index,
            protected_reference_eval.source,
            protected_reference_eval.fitness,
            best_eval.index,
            best_eval.source,
            best_eval.fitness,
            training_improvement,
            protected_improvement,
            best_eval.goals_for,
            best_eval.goals_against,
            best_goal_margin,
            search_config.training_min_accepted_goal_margin
        );
        return Ok(None);
    }
    let primary_clears_local_best_floor = best_eval.fitness >= search_config.min_accepted_fitness
        && best_goal_margin >= search_config.min_accepted_goal_margin;
    let mut confirmation_clears_local_best_floor = search_config.confirm_games == 0;
    if search_config.confirm_games > 0 {
        let mut confirm_config = search_config;
        confirm_config.eval_games = search_config.confirm_games;
        let confirm_seed = neural_population_confirm_seed(search_config.seed, completed_games);
        let confirm_candidate =
            evaluate_neural_population_candidate_against_analytic_home_learning_objective(
                NeuralPopulationCandidate {
                    index: best_eval.index,
                    source: format!("confirm:{}", best_eval.source),
                    snapshot: best_eval.snapshot.clone(),
                },
                config.clone(),
                confirm_config,
                confirm_seed,
            )
            .map_err(|error| {
                invalid_data(format!("population confirmation candidate failed: {error}"))
            })?;
        let confirm_reference =
            evaluate_neural_population_candidate_against_analytic_home_learning_objective(
                NeuralPopulationCandidate {
                    index: training_reference_eval.index,
                    source: format!("confirm:{}", training_reference_eval.source),
                    snapshot: training_reference_eval.snapshot.clone(),
                },
                config.clone(),
                confirm_config,
                confirm_seed,
            )
            .map_err(|error| {
                invalid_data(format!("population confirmation reference failed: {error}"))
            })?;
        let confirm_improvement = confirm_candidate.fitness - confirm_reference.fitness;
        let confirm_goal_margin = neural_population_candidate_goal_margin(&confirm_candidate);
        println!(
            "neural_population_search_confirmation completed_games={} candidate_index={} candidate_source={} candidate_fitness={:.4} reference_index={} reference_source={} reference_fitness={:.4} improvement={:.4} min_delta={:.4} candidate_goal_margin={:.4} min_goal_margin={:.4} record={}-{}-{} goals={}-{}",
            completed_games,
            best_eval.index,
            best_eval.source,
            confirm_candidate.fitness,
            training_reference_eval.index,
            training_reference_eval.source,
            confirm_reference.fitness,
            confirm_improvement,
            search_config.confirm_min_fitness_delta,
            confirm_goal_margin,
            search_config.confirm_min_accepted_goal_margin,
            confirm_candidate.wins,
            confirm_candidate.draws,
            confirm_candidate.losses,
            confirm_candidate.goals_for,
            confirm_candidate.goals_against
        );
        if !neural_population_improvement_clears_delta(
            confirm_improvement,
            search_config.confirm_min_fitness_delta,
        ) {
            println!(
                "neural_population_search_held completed_games={} incumbent_fitness={:.4} training_reference_index={} training_reference_source={} training_reference_fitness={:.4} protected_reference_index={} protected_reference_source={} protected_reference_fitness={:.4} best_index={} best_source={} best_fitness={:.4} training_improvement={:.4} protected_improvement={:.4} confirmation_fitness={:.4} confirmation_reference_fitness={:.4} confirmation_improvement={:.4} min_delta={:.4} reasons=insufficient_confirmation_gain",
                completed_games,
                incumbent_eval.fitness,
                training_reference_eval.index,
                training_reference_eval.source,
                training_reference_eval.fitness,
                protected_reference_eval.index,
                protected_reference_eval.source,
                protected_reference_eval.fitness,
                best_eval.index,
                best_eval.source,
                best_eval.fitness,
                training_improvement,
                protected_improvement,
                confirm_candidate.fitness,
                confirm_reference.fitness,
                confirm_improvement,
                search_config.confirm_min_fitness_delta
            );
            return Ok(None);
        }
        confirmation_clears_local_best_floor = confirm_candidate.fitness
            >= search_config.confirm_min_accepted_fitness
            && confirm_goal_margin >= search_config.confirm_min_accepted_goal_margin;
        if confirm_candidate.fitness < search_config.training_confirm_min_accepted_fitness {
            println!(
                "neural_population_search_held completed_games={} incumbent_fitness={:.4} training_reference_index={} training_reference_source={} training_reference_fitness={:.4} protected_reference_index={} protected_reference_source={} protected_reference_fitness={:.4} best_index={} best_source={} best_fitness={:.4} confirmation_fitness={:.4} min_accepted_fitness={:.4} reasons=below_training_confirmation_min_accepted_fitness",
                completed_games,
                incumbent_eval.fitness,
                training_reference_eval.index,
                training_reference_eval.source,
                training_reference_eval.fitness,
                protected_reference_eval.index,
                protected_reference_eval.source,
                protected_reference_eval.fitness,
                best_eval.index,
                best_eval.source,
                best_eval.fitness,
                confirm_candidate.fitness,
                search_config.training_confirm_min_accepted_fitness
            );
            return Ok(None);
        }
        if confirm_goal_margin < search_config.training_confirm_min_accepted_goal_margin {
            println!(
                "neural_population_search_held completed_games={} incumbent_fitness={:.4} training_reference_index={} training_reference_source={} training_reference_fitness={:.4} protected_reference_index={} protected_reference_source={} protected_reference_fitness={:.4} best_index={} best_source={} best_fitness={:.4} confirmation_fitness={:.4} confirmation_goal_margin={:.4} min_accepted_goal_margin={:.4} reasons=below_training_confirmation_min_goal_margin",
                completed_games,
                incumbent_eval.fitness,
                training_reference_eval.index,
                training_reference_eval.source,
                training_reference_eval.fitness,
                protected_reference_eval.index,
                protected_reference_eval.source,
                protected_reference_eval.fitness,
                best_eval.index,
                best_eval.source,
                best_eval.fitness,
                confirm_candidate.fitness,
                confirm_goal_margin,
                search_config.training_confirm_min_accepted_goal_margin
            );
            return Ok(None);
        }
    }

    *latest_neural_network = Some(best_eval.snapshot.clone());
    *latest_neural_network_arc_cache = None;
    let preserve_as_local_best = neural_population_improvement_clears_delta(
        protected_improvement,
        search_config.min_fitness_delta,
    ) && primary_clears_local_best_floor
        && confirmation_clears_local_best_floor;
    println!(
        "neural_population_search_accepted completed_games={} incumbent_fitness={:.4} training_reference_index={} training_reference_source={} training_reference_fitness={:.4} protected_reference_index={} protected_reference_source={} protected_reference_fitness={:.4} accepted_index={} accepted_source={} accepted_fitness={:.4} training_improvement={:.4} protected_improvement={:.4} preserve_as_local_best={} record={}-{}-{} goals={}-{} note=heldout_analytic_candidate_will_train_on_policy_next_batch",
        completed_games,
        incumbent_eval.fitness,
        training_reference_eval.index,
        training_reference_eval.source,
        training_reference_eval.fitness,
        protected_reference_eval.index,
        protected_reference_eval.source,
        protected_reference_eval.fitness,
        best_eval.index,
        best_eval.source,
        best_eval.fitness,
        training_improvement,
        protected_improvement,
        preserve_as_local_best,
        best_eval.wins,
        best_eval.draws,
        best_eval.losses,
        best_eval.goals_for,
        best_eval.goals_against
    );
    Ok(Some(NeuralPopulationSearchAcceptance {
        eval: best_eval,
        preserve_as_local_best,
    }))
}

fn policy_version_status_for_promotion_gate(
    evaluation: &SoccerPolicyPromotionGateEvaluation,
) -> &'static str {
    if evaluation.eligible {
        SOCCER_POLICY_STATUS_ACTIVE
    } else {
        SOCCER_POLICY_STATUS_ARCHIVED
    }
}

fn policy_promotion_search_metadata(
    curriculum_stage: &'static str,
    evaluation: &SoccerPolicyPromotionGateEvaluation,
) -> serde_json::Value {
    serde_json::json!({
        "curriculumStage": curriculum_stage,
        "gate": evaluation,
        "status": policy_version_status_for_promotion_gate(evaluation)
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PolicyPromotionIncumbentBaseline {
    sample_games: usize,
    mean_match_fitness: f64,
    best_match_fitness: f64,
    mean_play_quality: f64,
}

const LOCAL_POLICY_PROMOTION_OBJECTIVE_HOME_ANALYTIC: &str =
    "home_directional_learning_vs_analytic";

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct LocalPolicyPromotionBestMetadata {
    completed_games: usize,
    sample_games: usize,
    mean_match_fitness: f64,
    best_match_fitness: f64,
    mean_play_quality: f64,
    #[serde(default)]
    objective: Option<String>,
}

fn finite_json_f64(value: Option<&serde_json::Value>) -> Option<f64> {
    value
        .and_then(serde_json::Value::as_f64)
        .filter(|value| value.is_finite())
}

fn policy_promotion_incumbent_baseline_from_gate(
    gate: &serde_json::Value,
) -> Option<PolicyPromotionIncumbentBaseline> {
    let sample_games = gate
        .get("sampleGames")
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())?;
    if sample_games == 0 {
        return None;
    }
    Some(PolicyPromotionIncumbentBaseline {
        sample_games,
        mean_match_fitness: finite_json_f64(gate.get("meanMatchFitness"))?,
        best_match_fitness: finite_json_f64(gate.get("bestMatchFitness"))?,
        mean_play_quality: finite_json_f64(gate.get("meanPlayQuality"))?,
    })
}

fn policy_promotion_incumbent_baseline_from_search_metadata(
    search_metadata: Option<&serde_json::Value>,
) -> Option<PolicyPromotionIncumbentBaseline> {
    let search_metadata = search_metadata?;
    search_metadata
        .get("promotion")
        .and_then(|promotion| promotion.get("gate"))
        .or_else(|| search_metadata.get("gate"))
        .and_then(policy_promotion_incumbent_baseline_from_gate)
}

fn policy_promotion_incumbent_baseline_from_pg_baseline(
    baseline: &SoccerLearningPgPolicyPromotionBaseline,
) -> Option<PolicyPromotionIncumbentBaseline> {
    if baseline.sample_games == 0
        || !baseline.mean_match_fitness.is_finite()
        || !baseline.best_match_fitness.is_finite()
        || !baseline.mean_play_quality.is_finite()
    {
        return None;
    }
    Some(PolicyPromotionIncumbentBaseline {
        sample_games: baseline.sample_games,
        mean_match_fitness: baseline.mean_match_fitness,
        best_match_fitness: baseline.best_match_fitness,
        mean_play_quality: baseline.mean_play_quality,
    })
}

fn load_strongest_recent_policy_promotion_baseline_for_run(
    store: &mut SoccerLearningPgStore,
    experiment_id: &str,
    latest_generation: i32,
    lookback_generations: usize,
    min_sample_games: usize,
) -> Result<
    Option<(
        SoccerLearningPgPolicyPromotionBaseline,
        PolicyPromotionIncumbentBaseline,
    )>,
    Box<dyn Error>,
> {
    let lookback_generations = i32::try_from(lookback_generations).unwrap_or(i32::MAX);
    let Some(pg_baseline) = store
        .load_strongest_recent_policy_promotion_baseline(
            experiment_id,
            latest_generation,
            lookback_generations,
            min_sample_games,
        )
        .map_err(invalid_data)?
    else {
        return Ok(None);
    };
    println!(
        "postgres_policy_promotion_strongest_baseline policy_version={} generation={} lookback_generations={} sample_games={} mean_match_fitness={:.4} best_match_fitness={:.4} mean_play_quality={:.4}",
        pg_baseline.policy_version_id,
        pg_baseline.generation,
        lookback_generations,
        pg_baseline.sample_games,
        pg_baseline.mean_match_fitness,
        pg_baseline.best_match_fitness,
        pg_baseline.mean_play_quality
    );
    let Some(baseline) = policy_promotion_incumbent_baseline_from_pg_baseline(&pg_baseline) else {
        return Ok(None);
    };
    Ok(Some((pg_baseline, baseline)))
}

fn policy_promotion_incumbent_baseline_from_evaluation(
    evaluation: &SoccerPolicyPromotionGateEvaluation,
) -> Option<PolicyPromotionIncumbentBaseline> {
    if evaluation.sample_games == 0
        || !evaluation.mean_match_fitness.is_finite()
        || !evaluation.best_match_fitness.is_finite()
        || !evaluation.mean_play_quality.is_finite()
    {
        return None;
    }
    Some(PolicyPromotionIncumbentBaseline {
        sample_games: evaluation.sample_games,
        mean_match_fitness: evaluation.mean_match_fitness,
        best_match_fitness: evaluation.best_match_fitness,
        mean_play_quality: evaluation.mean_play_quality,
    })
}

fn policy_promotion_evaluation_from_incumbent_baseline(
    baseline: PolicyPromotionIncumbentBaseline,
    gate: SoccerPolicyPromotionGateConfig,
) -> SoccerPolicyPromotionGateEvaluation {
    SoccerPolicyPromotionGateEvaluation {
        enabled: gate.enabled,
        eligible: true,
        sample_games: baseline.sample_games,
        min_sample_games: gate.min_sample_games,
        mean_match_fitness: baseline.mean_match_fitness,
        best_match_fitness: baseline.best_match_fitness,
        mean_play_quality: baseline.mean_play_quality,
        mean_conceded_goals: 0.0,
        mean_goal_margin: 0.0,
        mean_chain_net_loss: 0.0,
        rejection_reasons: Vec::new(),
    }
}

fn retain_resume_local_best_seed_evaluation(
    stored_baseline: Option<PolicyPromotionIncumbentBaseline>,
    fresh_evaluation: SoccerPolicyPromotionGateEvaluation,
    _gate: SoccerPolicyPromotionGateConfig,
) -> (SoccerPolicyPromotionGateEvaluation, &'static str) {
    if stored_baseline.is_some() {
        return (fresh_evaluation, "fresh_eval_revalidated");
    }
    (fresh_evaluation, "fresh_eval")
}

fn policy_promotion_incumbent_baseline_from_local_best_metadata(
    metadata: &LocalPolicyPromotionBestMetadata,
) -> Option<PolicyPromotionIncumbentBaseline> {
    if metadata.sample_games == 0
        || !metadata.mean_match_fitness.is_finite()
        || !metadata.best_match_fitness.is_finite()
        || !metadata.mean_play_quality.is_finite()
    {
        return None;
    }
    Some(PolicyPromotionIncumbentBaseline {
        sample_games: metadata.sample_games,
        mean_match_fitness: metadata.mean_match_fitness,
        best_match_fitness: metadata.best_match_fitness,
        mean_play_quality: metadata.mean_play_quality,
    })
}

fn local_policy_promotion_best_metadata_from_evaluation(
    completed_games: usize,
    evaluation: &SoccerPolicyPromotionGateEvaluation,
) -> Option<LocalPolicyPromotionBestMetadata> {
    let baseline = policy_promotion_incumbent_baseline_from_evaluation(evaluation)?;
    Some(LocalPolicyPromotionBestMetadata {
        completed_games,
        sample_games: baseline.sample_games,
        mean_match_fitness: baseline.mean_match_fitness,
        best_match_fitness: baseline.best_match_fitness,
        mean_play_quality: baseline.mean_play_quality,
        objective: Some(LOCAL_POLICY_PROMOTION_OBJECTIVE_HOME_ANALYTIC.to_string()),
    })
}

fn retain_stronger_policy_promotion_baseline(
    current: Option<PolicyPromotionIncumbentBaseline>,
    candidate: Option<PolicyPromotionIncumbentBaseline>,
) -> Option<PolicyPromotionIncumbentBaseline> {
    match (current, candidate) {
        (Some(current), Some(candidate)) => {
            Some(stronger_policy_promotion_baseline(current, candidate))
        }
        (Some(current), None) => Some(current),
        (None, Some(candidate)) => Some(candidate),
        (None, None) => None,
    }
}

fn stronger_policy_promotion_baseline(
    current: PolicyPromotionIncumbentBaseline,
    candidate: PolicyPromotionIncumbentBaseline,
) -> PolicyPromotionIncumbentBaseline {
    const EPSILON: f64 = 1e-12;
    if candidate.mean_match_fitness > current.mean_match_fitness + EPSILON {
        return candidate;
    }
    if current.mean_match_fitness > candidate.mean_match_fitness + EPSILON {
        return current;
    }
    if candidate.mean_play_quality > current.mean_play_quality + EPSILON {
        return candidate;
    }
    if current.mean_play_quality > candidate.mean_play_quality + EPSILON {
        return current;
    }
    if candidate.best_match_fitness > current.best_match_fitness + EPSILON {
        return candidate;
    }
    if current.best_match_fitness > candidate.best_match_fitness + EPSILON {
        return current;
    }
    if candidate.sample_games >= current.sample_games {
        candidate
    } else {
        current
    }
}

fn apply_policy_promotion_incumbent_gate(
    evaluation: &mut SoccerPolicyPromotionGateEvaluation,
    incumbent: Option<PolicyPromotionIncumbentBaseline>,
    compare_incumbent: bool,
    min_mean_fitness_delta: f64,
    max_mean_fitness_regression: f64,
    max_play_quality_regression: f64,
) {
    if !compare_incumbent || !evaluation.enabled || !evaluation.eligible {
        return;
    }
    let Some(incumbent) = incumbent else {
        return;
    };
    let mean_fitness_floor = incumbent.mean_match_fitness + min_mean_fitness_delta.max(0.0)
        - max_mean_fitness_regression.max(0.0);
    let play_quality_floor =
        (incumbent.mean_play_quality - max_play_quality_regression.max(0.0)).max(0.0);
    if evaluation.mean_match_fitness < mean_fitness_floor {
        evaluation.rejection_reasons.push(format!(
            "incumbent_mean_match_fitness {:.4} below incumbent {:.4} + delta {:.4} - regression {:.4}",
            evaluation.mean_match_fitness,
            incumbent.mean_match_fitness,
            min_mean_fitness_delta.max(0.0),
            max_mean_fitness_regression.max(0.0)
        ));
    }
    if evaluation.mean_play_quality < play_quality_floor {
        evaluation.rejection_reasons.push(format!(
            "incumbent_mean_play_quality {:.4} below incumbent {:.4} - regression {:.4}",
            evaluation.mean_play_quality,
            incumbent.mean_play_quality,
            max_play_quality_regression.max(0.0)
        ));
    }
    if !evaluation.rejection_reasons.is_empty() {
        evaluation.eligible = false;
    }
}

fn policy_promotion_rejected_by_incumbent(
    evaluation: &SoccerPolicyPromotionGateEvaluation,
) -> bool {
    !evaluation.eligible
        && evaluation
            .rejection_reasons
            .iter()
            .any(|reason| reason.starts_with("incumbent_"))
}

#[derive(Clone)]
struct LocalNeuralPromotionTrialBest {
    completed_games: usize,
    evaluation: SoccerPolicyPromotionGateEvaluation,
    config: MatchConfig,
    tactical_learning: SoccerTacticalLearningWeights,
    policies: SoccerTeamQPolicies,
    neural_network: Option<SoccerNeuralNetworkSnapshot>,
    carried_world_model: Option<SoccerWorldModel>,
}

fn current_carried_world_model_snapshot() -> Option<SoccerWorldModel> {
    CARRIED_WORLD_MODEL.lock().unwrap().clone()
}

fn install_carried_world_model_snapshot(snapshot: Option<SoccerWorldModel>) -> usize {
    let training_steps = snapshot
        .as_ref()
        .map(SoccerWorldModel::training_steps)
        .unwrap_or(0);
    *CARRIED_WORLD_MODEL.lock().unwrap() = snapshot;
    training_steps
}

fn policy_promotion_evaluation_beats_local_best(
    evaluation: &SoccerPolicyPromotionGateEvaluation,
    best: &LocalNeuralPromotionTrialBest,
) -> bool {
    const EPSILON: f64 = 1e-9;
    if evaluation.mean_match_fitness > best.evaluation.mean_match_fitness + EPSILON {
        return true;
    }
    if best.evaluation.mean_match_fitness > evaluation.mean_match_fitness + EPSILON {
        return false;
    }
    if evaluation.mean_play_quality > best.evaluation.mean_play_quality + EPSILON {
        return true;
    }
    if best.evaluation.mean_play_quality > evaluation.mean_play_quality + EPSILON {
        return false;
    }
    evaluation.best_match_fitness > best.evaluation.best_match_fitness + EPSILON
}

fn policy_promotion_evaluation_beats_incumbent_baseline(
    evaluation: &SoccerPolicyPromotionGateEvaluation,
    baseline: PolicyPromotionIncumbentBaseline,
) -> bool {
    const EPSILON: f64 = 1e-9;
    if evaluation.mean_match_fitness > baseline.mean_match_fitness + EPSILON {
        return true;
    }
    if baseline.mean_match_fitness > evaluation.mean_match_fitness + EPSILON {
        return false;
    }
    if evaluation.mean_play_quality > baseline.mean_play_quality + EPSILON {
        return true;
    }
    if baseline.mean_play_quality > evaluation.mean_play_quality + EPSILON {
        return false;
    }
    evaluation.best_match_fitness > baseline.best_match_fitness + EPSILON
}

fn policy_promotion_evaluation_regresses_from_local_best(
    evaluation: &SoccerPolicyPromotionGateEvaluation,
    best: &LocalNeuralPromotionTrialBest,
    max_mean_match_fitness_regression: f64,
    max_play_quality_regression: f64,
) -> bool {
    const EPSILON: f64 = 1e-9;
    let mean_match_fitness_regression =
        best.evaluation.mean_match_fitness - evaluation.mean_match_fitness;
    if mean_match_fitness_regression > max_mean_match_fitness_regression + EPSILON {
        return true;
    }
    let mean_play_quality_regression =
        best.evaluation.mean_play_quality - evaluation.mean_play_quality;
    mean_match_fitness_regression >= -EPSILON
        && mean_play_quality_regression > max_play_quality_regression + EPSILON
}

fn apply_local_best_neural_backoff(
    config: &mut MatchConfig,
    factor: f64,
    min_learning_rate: f64,
    min_blend_lambda: f64,
) -> (f64, f64, f64, f64) {
    let previous_learning_rate = config.neural_learning.learning_rate;
    let previous_blend_lambda = config.neural_blend.lambda;
    config.neural_learning.learning_rate = (previous_learning_rate * factor)
        .max(min_learning_rate)
        .min(previous_learning_rate);
    config.neural_blend.lambda = (previous_blend_lambda * factor)
        .max(min_blend_lambda)
        .min(previous_blend_lambda);
    (
        previous_learning_rate,
        config.neural_learning.learning_rate,
        previous_blend_lambda,
        config.neural_blend.lambda,
    )
}

#[allow(clippy::too_many_arguments)]
fn restore_local_best_after_held_promotion(
    context_label: &str,
    completed_games: usize,
    evaluation: &SoccerPolicyPromotionGateEvaluation,
    local_neural_promotion_trial_best: &mut Option<LocalNeuralPromotionTrialBest>,
    config: &mut MatchConfig,
    tactical_learning: &mut SoccerTacticalLearningWeights,
    policies: &mut SoccerTeamQPolicies,
    latest_neural_network: &mut Option<SoccerNeuralNetworkSnapshot>,
    latest_policy_arc_cache: &mut Option<CachedTeamQPoliciesArc>,
    latest_neural_network_arc_cache: &mut Option<CachedNeuralNetworkSnapshotArc>,
    local_tactical_evolved_since_pg_refresh: &mut bool,
    backoff_enabled: bool,
    backoff_factor: f64,
    backoff_min_learning_rate: f64,
    backoff_min_blend_lambda: f64,
    min_restore_games: usize,
    max_mean_match_fitness_regression: f64,
    max_play_quality_regression: f64,
) -> bool {
    if latest_neural_network.is_none() {
        return false;
    }
    if completed_games < min_restore_games {
        return false;
    }
    let local_best_restore = local_neural_promotion_trial_best
        .as_ref()
        .filter(|best| {
            policy_promotion_evaluation_regresses_from_local_best(
                evaluation,
                best,
                max_mean_match_fitness_regression,
                max_play_quality_regression,
            )
        })
        .cloned();
    let Some(best) = local_best_restore else {
        return false;
    };
    let regressed_mean_match_fitness = evaluation.mean_match_fitness;
    let regressed_mean_play_quality = evaluation.mean_play_quality;
    *config = best.config.clone();
    *tactical_learning = best.tactical_learning.clone();
    config.tactical_learning = tactical_learning.clone();
    *policies = best.policies.clone();
    *latest_neural_network = best.neural_network.clone();
    *latest_policy_arc_cache = None;
    *latest_neural_network_arc_cache = None;
    *local_tactical_evolved_since_pg_refresh = true;
    let restored_world_model_steps =
        install_carried_world_model_snapshot(best.carried_world_model.clone());
    let backoff = backoff_enabled.then(|| {
        apply_local_best_neural_backoff(
            config,
            backoff_factor,
            backoff_min_learning_rate,
            backoff_min_blend_lambda,
        )
    });
    println!(
        "policy_promotion_local_best_restored context={} completed_games={} restored_from_games={} regressed_mean_match_fitness={:.4} restored_mean_match_fitness={:.4} regressed_mean_play_quality={:.4} restored_mean_play_quality={:.4}",
        context_label,
        completed_games,
        best.completed_games,
        regressed_mean_match_fitness,
        best.evaluation.mean_match_fitness,
        regressed_mean_play_quality,
        best.evaluation.mean_play_quality,
    );
    println!(
        "policy_promotion_local_best_world_model_restored context={} completed_games={} restored_from_games={} training_steps={}",
        context_label,
        completed_games,
        best.completed_games,
        restored_world_model_steps,
    );
    if let Some((
        previous_learning_rate,
        next_learning_rate,
        previous_blend_lambda,
        next_blend_lambda,
    )) = backoff
    {
        if let Some(best) = local_neural_promotion_trial_best.as_mut() {
            best.config = config.clone();
        }
        println!(
            "policy_promotion_local_best_backoff context={} completed_games={} learning_rate={:.5}->{:.5} neural_blend_lambda={:.3}->{:.3}",
            context_label,
            completed_games,
            previous_learning_rate,
            next_learning_rate,
            previous_blend_lambda,
            next_blend_lambda,
        );
    }
    true
}

fn should_write_neural_checkpoint_for_rejected_promotion(
    enabled: bool,
    should_write_policy_version: bool,
    incumbent_rejected: bool,
    latest_neural_network_fingerprint: Option<u64>,
    last_neural_checkpoint_fingerprint: Option<u64>,
) -> bool {
    enabled
        && should_write_policy_version
        && incumbent_rejected
        && latest_neural_network_fingerprint.is_some()
        && latest_neural_network_fingerprint != last_neural_checkpoint_fingerprint
}

fn default_disk_learning_artifacts_enabled(postgres_configured: bool) -> bool {
    !postgres_configured
}

fn soccer_learning_database_url_env_configured() -> bool {
    [
        "SOCCER_DATABASE_URL",
        "AGENT_TASKS_RDS_DATABASE_URL",
        "RDS_DATABASE_URL",
        "DATABASE_URL",
        "PG_DATABASE_URL",
    ]
    .into_iter()
    .any(|name| env_value(name).is_some())
}

fn default_soccer_parallel_games() -> usize {
    // The continuous learner is the deterministic single-game lane; scale-out
    // belongs in main_soccer_learning_queue or an explicit SOCCER_PARALLEL_GAMES.
    DEFAULT_SOCCER_CONTINUOUS_PARALLEL_GAMES
}

fn env_neural_learning_backend(
    default: SoccerNeuralLearningBackend,
) -> Result<SoccerNeuralLearningBackend, Box<dyn Error>> {
    let Some(value) =
        env_value("SOCCER_NEURAL_LEARNING_BACKEND").or_else(|| env_value("SOCCER_NEURAL_BACKEND"))
    else {
        return Ok(default);
    };
    match value.to_ascii_lowercase().as_str() {
        "threaded" | "thread" | "worker" => Ok(SoccerNeuralLearningBackend::Threaded),
        "inline" | "sync" => Ok(SoccerNeuralLearningBackend::Inline),
        _ => Err(invalid_data(format!(
            "SOCCER_NEURAL_LEARNING_BACKEND must be inline or threaded, got {value:?}"
        ))
        .into()),
    }
}

fn env_marl_algorithm(default: SoccerMarlAlgorithm) -> Result<SoccerMarlAlgorithm, Box<dyn Error>> {
    let Some(value) =
        env_value("SOCCER_MARL_ALGORITHM").or_else(|| env_value("SOCCER_NEURAL_MARL_ALGORITHM"))
    else {
        return Ok(default);
    };
    match value.to_ascii_lowercase().as_str() {
        "off" | "disabled" | "none" => Ok(SoccerMarlAlgorithm::Off),
        "independent"
        | "independent-actor-critic"
        | "independent_actor_critic"
        | "independentactorcritic" => Ok(SoccerMarlAlgorithm::IndependentActorCritic),
        "mappo" | "ppo" => Ok(SoccerMarlAlgorithm::Mappo),
        _ => Err(invalid_data(format!(
            "SOCCER_MARL_ALGORITHM must be off, independentActorCritic, or mappo, got {value:?}"
        ))
        .into()),
    }
}

fn env_neural_blend_mode(
    default: SoccerNeuralBlendMode,
) -> Result<SoccerNeuralBlendMode, Box<dyn Error>> {
    let Some(value) = env_value("SOCCER_NEURAL_BLEND_MODE") else {
        return Ok(default);
    };
    match value.to_ascii_lowercase().as_str() {
        "off" | "disabled" | "none" => Ok(SoccerNeuralBlendMode::Off),
        "additive" | "add" => Ok(SoccerNeuralBlendMode::Additive),
        "tiebreak" | "tie" | "tie-break" | "tie_break" => Ok(SoccerNeuralBlendMode::TieBreak),
        "confidence" | "confidencegated" | "confidence-gated" | "gated" => {
            Ok(SoccerNeuralBlendMode::ConfidenceGated)
        }
        "authoritative" | "neural" | "neural-authoritative" | "neural_authoritative" => {
            Ok(SoccerNeuralBlendMode::Authoritative)
        }
        _ => Err(invalid_data(format!(
            "SOCCER_NEURAL_BLEND_MODE must be off, additive, tiebreak, confidence, or authoritative, got {value:?}"
        ))
        .into()),
    }
}

fn env_neural_learning_config() -> Result<SoccerNeuralLearningConfig, Box<dyn Error>> {
    let default = SoccerNeuralLearningConfig {
        enabled: true,
        backend: SoccerNeuralLearningBackend::Threaded,
        max_pending_batches: 128,
        mappo_team_reward_share: DEFAULT_SOCCER_MAPPO_TEAM_REWARD_SHARE,
        ..SoccerNeuralLearningConfig::default()
    };
    Ok(SoccerNeuralLearningConfig {
        enabled: env_bool_alias(
            "SOCCER_NEURAL_LEARNING_ENABLED",
            "SOCCER_NEURAL_LEARNING",
            default.enabled,
        )?,
        backend: env_neural_learning_backend(default.backend)?,
        learning_rate: env_f64_alias(
            "SOCCER_NEURAL_LEARNING_RATE",
            "SOCCER_NEURAL_RATE",
            default.learning_rate,
        )?,
        optimizer_momentum: env_f64_alias(
            "SOCCER_NEURAL_OPTIMIZER_MOMENTUM",
            "SOCCER_NEURAL_MOMENTUM",
            default.optimizer_momentum,
        )?,
        batch_size: env_usize_alias(
            "SOCCER_NEURAL_BATCH_SIZE",
            "SOCCER_NEURAL_LEARNING_BATCH_SIZE",
            default.batch_size,
        )?,
        train_every_ticks: env_usize_alias(
            "SOCCER_NEURAL_TRAIN_EVERY_TICKS",
            "SOCCER_NEURAL_LEARNING_TRAIN_EVERY_TICKS",
            default.train_every_ticks,
        )?,
        max_batches_per_tick: env_usize_alias(
            "SOCCER_NEURAL_MAX_BATCHES_PER_TICK",
            "SOCCER_NEURAL_LEARNING_MAX_BATCHES_PER_TICK",
            default.max_batches_per_tick,
        )?,
        hidden_units: env_usize_alias(
            "SOCCER_NEURAL_HIDDEN_UNITS",
            "SOCCER_NEURAL_LEARNING_HIDDEN_UNITS",
            default.hidden_units,
        )?,
        target_scale: env_f64_alias(
            "SOCCER_NEURAL_TARGET_SCALE",
            "SOCCER_NEURAL_LEARNING_TARGET_SCALE",
            default.target_scale,
        )?,
        max_pending_batches: env_usize_alias(
            "SOCCER_NEURAL_MAX_PENDING_BATCHES",
            "SOCCER_NEURAL_LEARNING_MAX_PENDING_BATCHES",
            default.max_pending_batches,
        )?,
        replay_capacity: env_usize_alias(
            "SOCCER_NEURAL_REPLAY_CAPACITY",
            "SOCCER_NEURAL_LEARNING_REPLAY_CAPACITY",
            default.replay_capacity,
        )?,
        replay_samples_per_tick: env_usize_alias(
            "SOCCER_NEURAL_REPLAY_SAMPLES_PER_TICK",
            "SOCCER_NEURAL_LEARNING_REPLAY_SAMPLES_PER_TICK",
            default.replay_samples_per_tick,
        )?,
        target_clip: env_f64_alias(
            "SOCCER_NEURAL_TARGET_CLIP",
            "SOCCER_NEURAL_LEARNING_TARGET_CLIP",
            default.target_clip,
        )?,
        target_popart_enabled: env_bool(
            "SOCCER_NEURAL_TARGET_POPART",
            default.target_popart_enabled,
        )?,
        snapshot_every_batches: env_usize_alias(
            "SOCCER_NEURAL_SNAPSHOT_EVERY_BATCHES",
            "SOCCER_NEURAL_LEARNING_SNAPSHOT_EVERY_BATCHES",
            default.snapshot_every_batches,
        )?,
        critic_baseline_weight: env_f64(
            "SOCCER_NEURAL_CRITIC_BASELINE_WEIGHT",
            default.critic_baseline_weight,
        )?,
        lp_coupling_enabled: env_bool(
            "SOCCER_NEURAL_LP_COUPLING_ENABLED",
            default.lp_coupling_enabled,
        )?,
        marl_algorithm: env_marl_algorithm(default.marl_algorithm)?,
        marl_team_reward_weight: env_f64(
            "SOCCER_MARL_TEAM_REWARD_WEIGHT",
            default.marl_team_reward_weight,
        )?,
        marl_intermediate_reward_weight: env_f64(
            "SOCCER_MARL_INTERMEDIATE_REWARD_WEIGHT",
            default.marl_intermediate_reward_weight,
        )?,
        mappo_clip_epsilon: env_f64("SOCCER_MAPPO_CLIP_EPSILON", default.mappo_clip_epsilon)?,
        mappo_team_reward_share: env_f64(
            "SOCCER_MAPPO_TEAM_REWARD_SHARE",
            default.mappo_team_reward_share,
        )?,
    })
}

fn apply_env_mpc_config(
    config: &mut MatchConfig,
    defaults: &MatchConfig,
) -> Result<(), Box<dyn Error>> {
    let mpc_override_configured = env_value("SOCCER_LEARNING_MPC_ENABLED").is_some()
        || env_value("SOCCER_MPC_ENABLED").is_some();
    let tier2_player_enabled = env_bool_alias(
        "SOCCER_LEARNING_MPC_ENABLED",
        "SOCCER_MPC_ENABLED",
        defaults.mpc.tier2_player_enabled,
    )?;
    let execution_stack_default = if mpc_override_configured {
        tier2_player_enabled
    } else {
        defaults.mpc.field_aware_enabled
    };

    config.mpc.tier2_player_enabled = tier2_player_enabled;
    config.mpc.field_aware_enabled = env_bool_alias(
        "SOCCER_LEARNING_MPC_FIELD_AWARE_ENABLED",
        "SOCCER_MPC_FIELD_AWARE_ENABLED",
        execution_stack_default,
    )?;
    config.mpc.reconcile_enabled = env_bool_alias(
        "SOCCER_LEARNING_MPC_RECONCILE_ENABLED",
        "SOCCER_MPC_RECONCILE_ENABLED",
        if mpc_override_configured {
            tier2_player_enabled
        } else {
            defaults.mpc.reconcile_enabled
        },
    )?;
    config.mpc.latent_objective_enabled = env_bool_alias(
        "SOCCER_LEARNING_MPC_LATENT_OBJECTIVE_ENABLED",
        "SOCCER_MPC_LATENT_OBJECTIVE_ENABLED",
        if mpc_override_configured {
            tier2_player_enabled
        } else {
            defaults.mpc.latent_objective_enabled
        },
    )?;
    if !config.mpc.tier2_player_enabled
        && (config.mpc.field_aware_enabled
            || config.mpc.reconcile_enabled
            || config.mpc.latent_objective_enabled)
    {
        return Err(invalid_data(
            "SOCCER_MPC_ENABLED must be true before enabling field-aware, reconcile, or latent MPC",
        )
        .into());
    }

    config.mpc.player_horizon = env_usize_alias(
        "SOCCER_LEARNING_MPC_PLAYER_HORIZON",
        "SOCCER_MPC_PLAYER_HORIZON",
        defaults.mpc.player_horizon,
    )?;
    if config.mpc.player_horizon == 0 {
        return Err(invalid_data("SOCCER_MPC_PLAYER_HORIZON must be greater than 0").into());
    }
    config.mpc.active_radius_yards = env_f64_alias(
        "SOCCER_LEARNING_MPC_ACTIVE_RADIUS",
        "SOCCER_MPC_ACTIVE_RADIUS",
        defaults.mpc.active_radius_yards,
    )?;
    if config.mpc.active_radius_yards <= 0.0 {
        return Err(invalid_data("SOCCER_MPC_ACTIVE_RADIUS must be greater than 0").into());
    }

    config.local_mpc_enabled = env_bool_alias(
        "SOCCER_LEARNING_LOCAL_MPC_ENABLED",
        "SOCCER_LOCAL_MPC",
        defaults.local_mpc_enabled,
    )?;
    config.local_mpc_max_players_per_team = env_usize_alias(
        "SOCCER_LEARNING_LOCAL_MPC_MAX_PLAYERS_PER_TEAM",
        "SOCCER_LOCAL_MPC_MAX_PLAYERS_PER_TEAM",
        defaults.local_mpc_max_players_per_team,
    )?
    .min(SOCCER_LEARNING_LOCAL_MPC_MAX_PLAYERS_PER_TEAM_LIMIT);
    if config.local_mpc_max_players_per_team == 0
        || (mpc_override_configured && !tier2_player_enabled)
    {
        config.local_mpc_enabled = false;
    }

    Ok(())
}

fn q_entry_order(a: &SoccerQEntry, b: &SoccerQEntry) -> std::cmp::Ordering {
    b.visits.cmp(&a.visits).then_with(|| {
        b.value
            .abs()
            .partial_cmp(&a.value.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    })
}

fn target_entry_order(a: &SoccerQTargetEntry, b: &SoccerQTargetEntry) -> std::cmp::Ordering {
    b.visits.cmp(&a.visits).then_with(|| {
        b.value
            .abs()
            .partial_cmp(&a.value.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    })
}

fn compact_training_artifact_for_export(
    artifact: &SoccerSelfPlayTrainingArtifact,
    max_entries_per_policy: usize,
) -> SoccerSelfPlayTrainingArtifact {
    let mut export = artifact.clone();
    if max_entries_per_policy == 0 {
        return export;
    }
    if export.home_entries.len() > max_entries_per_policy {
        export.home_entries.sort_by(q_entry_order);
        export.home_entries.truncate(max_entries_per_policy);
    }
    if export.away_entries.len() > max_entries_per_policy {
        export.away_entries.sort_by(q_entry_order);
        export.away_entries.truncate(max_entries_per_policy);
    }
    if export.home_target_entries.len() > max_entries_per_policy {
        export.home_target_entries.sort_by(target_entry_order);
        export.home_target_entries.truncate(max_entries_per_policy);
    }
    if export.away_target_entries.len() > max_entries_per_policy {
        export.away_target_entries.sort_by(target_entry_order);
        export.away_target_entries.truncate(max_entries_per_policy);
    }
    export
}

fn env_tactical_learning_weights() -> Result<SoccerTacticalLearningWeights, Box<dyn Error>> {
    let default = SoccerTacticalLearningWeights::default();
    Ok(SoccerTacticalLearningWeights {
        attack_spacing_delta_weight: env_f64(
            "SOCCER_ATTACK_SPACING_DELTA_WEIGHT",
            default.attack_spacing_delta_weight,
        )?,
        attack_spacing_score_weight: env_f64(
            "SOCCER_ATTACK_SPACING_SCORE_WEIGHT",
            default.attack_spacing_score_weight,
        )?,
        attack_width_delta_weight: env_f64(
            "SOCCER_ATTACK_WIDTH_DELTA_WEIGHT",
            default.attack_width_delta_weight,
        )?,
        attack_width_score_weight: env_f64(
            "SOCCER_ATTACK_WIDTH_SCORE_WEIGHT",
            default.attack_width_score_weight,
        )?,
        attack_flank_lane_weight: env_f64(
            "SOCCER_ATTACK_FLANK_LANE_WEIGHT",
            default.attack_flank_lane_weight,
        )?,
        shot_choice_learning_weight: env_f64(
            "SOCCER_SHOT_CHOICE_LEARNING_WEIGHT",
            default.shot_choice_learning_weight,
        )?,
        goal_entry_pass_learning_weight: env_f64(
            "SOCCER_GOAL_ENTRY_PASS_LEARNING_WEIGHT",
            default.goal_entry_pass_learning_weight,
        )?,
        pressure_release_learning_weight: env_f64(
            "SOCCER_PRESSURE_RELEASE_LEARNING_WEIGHT",
            default.pressure_release_learning_weight,
        )?,
        pass_target_ranking_learning_weight: env_f64(
            "SOCCER_PASS_TARGET_RANKING_LEARNING_WEIGHT",
            default.pass_target_ranking_learning_weight,
        )?,
        defense_spacing_delta_weight: env_f64(
            "SOCCER_DEFENSE_SPACING_DELTA_WEIGHT",
            default.defense_spacing_delta_weight,
        )?,
        defense_spacing_score_weight: env_f64(
            "SOCCER_DEFENSE_SPACING_SCORE_WEIGHT",
            default.defense_spacing_score_weight,
        )?,
        defense_contract_delta_weight: env_f64(
            "SOCCER_DEFENSE_CONTRACT_DELTA_WEIGHT",
            default.defense_contract_delta_weight,
        )?,
        defense_compactness_score_weight: env_f64(
            "SOCCER_DEFENSE_COMPACTNESS_SCORE_WEIGHT",
            default.defense_compactness_score_weight,
        )?,
        defense_ball_depth_score_weight: env_f64(
            "SOCCER_DEFENSE_BALL_DEPTH_SCORE_WEIGHT",
            default.defense_ball_depth_score_weight,
        )?,
        defense_endline_soft_penalty_weight: env_f64(
            "SOCCER_DEFENSE_ENDLINE_SOFT_PENALTY_WEIGHT",
            default.defense_endline_soft_penalty_weight,
        )?,
        defense_endline_hard_penalty_weight: env_f64(
            "SOCCER_DEFENSE_ENDLINE_HARD_PENALTY_WEIGHT",
            default.defense_endline_hard_penalty_weight,
        )?,
        defender_midfielder_press_weight: env_f64(
            "SOCCER_DEFENDER_MIDFIELDER_PRESS_WEIGHT",
            default.defender_midfielder_press_weight,
        )?,
        midfielder_press_weight: env_f64(
            "SOCCER_MIDFIELDER_PRESS_WEIGHT",
            default.midfielder_press_weight,
        )?,
        defensive_line_press_learning_weight: env_f64(
            "SOCCER_DEFENSIVE_LINE_PRESS_LEARNING_WEIGHT",
            default.defensive_line_press_learning_weight,
        )?,
        formation_lp_alignment_weight: env_f64(
            "SOCCER_FORMATION_LP_ALIGNMENT_WEIGHT",
            default.formation_lp_alignment_weight,
        )?,
    })
}

fn artifact_minutes_label(minutes: f64) -> String {
    if (minutes - minutes.round()).abs() < 1e-9 {
        format!("{:.0}", minutes)
    } else {
        format!("{:.2}", minutes).replace('.', "p")
    }
}

fn default_artifact_path_in_run_dir(
    run_dir: &Path,
    games: usize,
    minutes: f64,
    shard_index: usize,
    shard_count: usize,
) -> PathBuf {
    let minutes = artifact_minutes_label(minutes);
    let file_name = if shard_count > 1 {
        format!(
            "final-policy-shard-{}-of-{}-{}x{}.json",
            shard_index, shard_count, games, minutes
        )
    } else {
        "final-policy.json".to_string()
    };
    run_dir.join(file_name)
}

fn validate_run_settings(
    games: usize,
    halves: usize,
    half_minutes: f64,
    minutes: f64,
    period_break_recovery_seconds: f64,
    dt_seconds: f64,
    learning_interval_ticks: usize,
    parallel_games: usize,
    shard_seed_stride: u32,
    options: &SoccerQPolicyOptions,
    tactical_learning: &SoccerTacticalLearningWeights,
) -> Result<(), Box<dyn Error>> {
    if games == 0 {
        return Err(invalid_data("SOCCER_GAMES must be at least 1").into());
    }
    if !(1..=8).contains(&halves) {
        return Err(invalid_data("SOCCER_HALVES must be between 1 and 8").into());
    }
    if !half_minutes.is_finite() || half_minutes <= 0.0 || half_minutes > 120.0 {
        return Err(invalid_data("SOCCER_HALF_MINUTES must be finite and in (0, 120]").into());
    }
    if !minutes.is_finite() || minutes <= 0.0 || minutes > 24.0 * 60.0 {
        return Err(invalid_data("SOCCER_MINUTES must be finite and in (0, 1440]").into());
    }
    if env_value("SOCCER_MINUTES").is_some() {
        let expected = half_minutes * halves as f64;
        if (minutes - expected).abs() > 1e-6 {
            return Err(invalid_data(format!(
                "SOCCER_MINUTES ({minutes}) must equal SOCCER_HALF_MINUTES * SOCCER_HALVES ({expected})"
            ))
            .into());
        }
    }
    if !period_break_recovery_seconds.is_finite()
        || !(0.0..=60.0 * 60.0).contains(&period_break_recovery_seconds)
    {
        return Err(invalid_data(
            "SOCCER_PERIOD_BREAK_RECOVERY_SECONDS must be finite and in [0, 3600]",
        )
        .into());
    }
    if !dt_seconds.is_finite() || !(0.01..=4.0).contains(&dt_seconds) {
        return Err(invalid_data("SOCCER_DT_SECONDS must be finite and in [0.01, 4.0]").into());
    }
    if learning_interval_ticks == 0 {
        return Err(invalid_data("SOCCER_LEARNING_INTERVAL_TICKS must be at least 1").into());
    }
    if !(1..=100).contains(&parallel_games) {
        return Err(invalid_data("SOCCER_PARALLEL_GAMES must be between 1 and 100").into());
    }
    if shard_seed_stride == 0 {
        return Err(invalid_data("SOCCER_SHARD_SEED_STRIDE must be at least 1").into());
    }
    validate_soccer_q_policy_options_for_runner(options)?;
    validate_tactical_learning_weights(tactical_learning)?;
    Ok(())
}

fn validate_soccer_q_policy_options_for_runner(
    options: &SoccerQPolicyOptions,
) -> Result<(), Box<dyn Error>> {
    if !options.alpha.is_finite() || !(0.0..=1.0).contains(&options.alpha) {
        return Err(invalid_data("SOCCER_ALPHA must be finite and in [0, 1]").into());
    }
    if !options.gamma.is_finite() || !(0.0..=1.0).contains(&options.gamma) {
        return Err(invalid_data("SOCCER_GAMMA must be finite and in [0, 1]").into());
    }
    if !options.exploration_epsilon.is_finite()
        || !(0.0..=1.0).contains(&options.exploration_epsilon)
    {
        return Err(invalid_data("SOCCER_EXPLORATION_EPSILON must be finite and in [0, 1]").into());
    }
    Ok(())
}

fn validate_tactical_learning_weights(
    tactical_learning: &SoccerTacticalLearningWeights,
) -> Result<(), Box<dyn Error>> {
    let weights = [
        (
            "SOCCER_ATTACK_SPACING_DELTA_WEIGHT",
            tactical_learning.attack_spacing_delta_weight,
        ),
        (
            "SOCCER_ATTACK_SPACING_SCORE_WEIGHT",
            tactical_learning.attack_spacing_score_weight,
        ),
        (
            "SOCCER_ATTACK_WIDTH_DELTA_WEIGHT",
            tactical_learning.attack_width_delta_weight,
        ),
        (
            "SOCCER_ATTACK_WIDTH_SCORE_WEIGHT",
            tactical_learning.attack_width_score_weight,
        ),
        (
            "SOCCER_ATTACK_FLANK_LANE_WEIGHT",
            tactical_learning.attack_flank_lane_weight,
        ),
        (
            "SOCCER_SHOT_CHOICE_LEARNING_WEIGHT",
            tactical_learning.shot_choice_learning_weight,
        ),
        (
            "SOCCER_GOAL_ENTRY_PASS_LEARNING_WEIGHT",
            tactical_learning.goal_entry_pass_learning_weight,
        ),
        (
            "SOCCER_PRESSURE_RELEASE_LEARNING_WEIGHT",
            tactical_learning.pressure_release_learning_weight,
        ),
        (
            "SOCCER_PASS_TARGET_RANKING_LEARNING_WEIGHT",
            tactical_learning.pass_target_ranking_learning_weight,
        ),
        (
            "SOCCER_DEFENSE_SPACING_DELTA_WEIGHT",
            tactical_learning.defense_spacing_delta_weight,
        ),
        (
            "SOCCER_DEFENSE_SPACING_SCORE_WEIGHT",
            tactical_learning.defense_spacing_score_weight,
        ),
        (
            "SOCCER_DEFENSE_CONTRACT_DELTA_WEIGHT",
            tactical_learning.defense_contract_delta_weight,
        ),
        (
            "SOCCER_DEFENSE_COMPACTNESS_SCORE_WEIGHT",
            tactical_learning.defense_compactness_score_weight,
        ),
        (
            "SOCCER_DEFENSE_BALL_DEPTH_SCORE_WEIGHT",
            tactical_learning.defense_ball_depth_score_weight,
        ),
        (
            "SOCCER_DEFENSE_ENDLINE_SOFT_PENALTY_WEIGHT",
            tactical_learning.defense_endline_soft_penalty_weight,
        ),
        (
            "SOCCER_DEFENSE_ENDLINE_HARD_PENALTY_WEIGHT",
            tactical_learning.defense_endline_hard_penalty_weight,
        ),
        (
            "SOCCER_DEFENDER_MIDFIELDER_PRESS_WEIGHT",
            tactical_learning.defender_midfielder_press_weight,
        ),
        (
            "SOCCER_MIDFIELDER_PRESS_WEIGHT",
            tactical_learning.midfielder_press_weight,
        ),
        (
            "SOCCER_DEFENSIVE_LINE_PRESS_LEARNING_WEIGHT",
            tactical_learning.defensive_line_press_learning_weight,
        ),
        (
            "SOCCER_FORMATION_LP_ALIGNMENT_WEIGHT",
            tactical_learning.formation_lp_alignment_weight,
        ),
    ];
    for (name, value) in weights {
        if !value.is_finite() {
            return Err(invalid_data(format!("{name} must be finite")).into());
        }
    }
    Ok(())
}

fn tactical_learning_weight_values(weights: &SoccerTacticalLearningWeights) -> [f64; 20] {
    [
        weights.attack_spacing_delta_weight,
        weights.attack_spacing_score_weight,
        weights.attack_width_delta_weight,
        weights.attack_width_score_weight,
        weights.attack_flank_lane_weight,
        weights.shot_choice_learning_weight,
        weights.goal_entry_pass_learning_weight,
        weights.pressure_release_learning_weight,
        weights.pass_target_ranking_learning_weight,
        weights.defense_spacing_delta_weight,
        weights.defense_spacing_score_weight,
        weights.defense_contract_delta_weight,
        weights.defense_compactness_score_weight,
        weights.defense_ball_depth_score_weight,
        weights.defense_endline_soft_penalty_weight,
        weights.defense_endline_hard_penalty_weight,
        weights.defender_midfielder_press_weight,
        weights.midfielder_press_weight,
        weights.defensive_line_press_learning_weight,
        weights.formation_lp_alignment_weight,
    ]
}

fn tactical_learning_weights_match(
    left: &SoccerTacticalLearningWeights,
    right: &SoccerTacticalLearningWeights,
) -> bool {
    tactical_learning_weight_values(left)
        .into_iter()
        .zip(tactical_learning_weight_values(right))
        .all(|(left, right)| (left - right).abs() <= 1e-12)
}

fn maybe_apply_postgres_tactical_learning(
    event_label: &str,
    next_episode: usize,
    policy_version_id: &str,
    generation: i32,
    config: &mut MatchConfig,
    active_weights: &mut SoccerTacticalLearningWeights,
    postgres_weights: Option<SoccerTacticalLearningWeights>,
) -> Result<bool, Box<dyn Error>> {
    let Some(postgres_weights) = postgres_weights else {
        return Ok(false);
    };
    validate_tactical_learning_weights(&postgres_weights)?;
    if tactical_learning_weights_match(active_weights, &postgres_weights)
        && tactical_learning_weights_match(&config.tactical_learning, &postgres_weights)
    {
        return Ok(false);
    }
    println!(
        "{} next_episode={} policy_version={} generation={} attack_flank_lane={:.3} defense_contract_delta={:.3}",
        event_label,
        next_episode,
        policy_version_id,
        generation,
        postgres_weights.attack_flank_lane_weight,
        postgres_weights.defense_contract_delta_weight
    );
    config.tactical_learning = postgres_weights.clone();
    *active_weights = postgres_weights;
    Ok(true)
}

fn maybe_apply_postgres_evolution_options(
    event_label: &str,
    next_episode: usize,
    policy_version_id: &str,
    generation: i32,
    evolution_options: &mut SoccerEvolutionOptions,
    search_metadata: Option<&serde_json::Value>,
) -> bool {
    let Some(postgres_options) =
        soccer_evolution_options_from_search_metadata(search_metadata, *evolution_options)
    else {
        return false;
    };
    if postgres_options == *evolution_options {
        return false;
    }
    println!(
        "{} next_episode={} policy_version={} generation={} mutation_rate={:.4}->{:.4} crossover_rate={:.4}->{:.4} exploration_scale={:.4}->{:.4} population_size={}->{}",
        event_label,
        next_episode,
        policy_version_id,
        generation,
        evolution_options.mutation_rate,
        postgres_options.mutation_rate,
        evolution_options.crossover_rate,
        postgres_options.crossover_rate,
        evolution_options.exploration_scale,
        postgres_options.exploration_scale,
        evolution_options.population_size,
        postgres_options.population_size
    );
    *evolution_options = postgres_options;
    true
}

fn maybe_apply_latest_postgres_neural_snapshot(
    event_label: &str,
    next_episode: usize,
    store: &mut SoccerLearningPgStore,
    experiment_id: &str,
    latest_neural_network: &mut Option<SoccerNeuralNetworkSnapshot>,
    active_neural_network_fingerprint: &mut Option<u64>,
) -> Result<bool, Box<dyn Error>> {
    let Some(metadata) = store
        .load_latest_neural_policy_metadata(experiment_id)
        .map_err(invalid_data)?
    else {
        return Ok(false);
    };
    let Some(snapshot) = metadata.neural_network else {
        return Ok(false);
    };
    let fingerprint = soccer_neural_network_snapshot_fingerprint(&snapshot);
    if Some(fingerprint) == *active_neural_network_fingerprint {
        return Ok(false);
    }
    let previous_fingerprint = (*active_neural_network_fingerprint)
        .map(|value| value.to_string())
        .unwrap_or_else(|| "none".to_string());
    let reason = metadata
        .search_metadata
        .as_ref()
        .and_then(|search_metadata| search_metadata.get("neuralCheckpoint"))
        .and_then(|checkpoint| checkpoint.get("reason"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("policy-version");
    println!(
        "{} next_episode={} policy_version={} generation={} neural_fingerprint={} previous_neural_fingerprint={} reason={}",
        event_label,
        next_episode,
        metadata.id,
        metadata.generation,
        fingerprint,
        previous_fingerprint,
        reason
    );
    *latest_neural_network = Some(snapshot);
    *active_neural_network_fingerprint = Some(fingerprint);
    Ok(true)
}

fn refresh_postgres_policy_for_next_sim(
    store: &mut SoccerLearningPgStore,
    experiment_id: &str,
    next_episode: usize,
    options: &SoccerQPolicyOptions,
    evolution_options: &mut SoccerEvolutionOptions,
    config: &mut MatchConfig,
    tactical_learning: &mut SoccerTacticalLearningWeights,
    policies: &mut SoccerTeamQPolicies,
    latest_neural_network: &mut Option<SoccerNeuralNetworkSnapshot>,
    pg_base_policy_version_id: &mut Option<String>,
    pg_last_policy_version_id: &mut Option<String>,
    pg_generation: &mut i32,
    pg_base_policy_version_updated_at_micros: &mut i64,
    pg_base_policy_fingerprint: &mut Option<u64>,
    pg_base_neural_network_fingerprint: &mut Option<u64>,
    pg_base_tactical_learning_fingerprint: &mut Option<u64>,
    pg_base_policy_promotion_baseline: &mut Option<PolicyPromotionIncumbentBaseline>,
    local_tactical_evolved_since_pg_refresh: &mut bool,
    postgres_tactical_learning_authoritative: bool,
    pg_resume_include_unpromoted: bool,
    policy_promotion_gate_min_sample_games: usize,
    policy_promotion_baseline_lookback_generations: usize,
    pg_policy_version_buffer_len: usize,
    pending_async_pg_batches: usize,
    pending_async_policy_version_batches: usize,
) -> Result<bool, Box<dyn Error>> {
    let Some(metadata) = store
        .load_latest_policy_metadata(experiment_id, pg_resume_include_unpromoted)
        .map_err(invalid_data)?
    else {
        return Ok(false);
    };
    let latest_neural_network_fingerprint = metadata
        .neural_network
        .as_ref()
        .map(soccer_neural_network_snapshot_fingerprint);
    let latest_policy_fingerprint = metadata.policy_fingerprint;
    let latest_tactical_learning_fingerprint = metadata
        .tactical_learning
        .as_ref()
        .map(soccer_tactical_learning_weights_fingerprint);
    let refresh_decision =
        soccer_postgres_policy_refresh_decision(SoccerPostgresPolicyRefreshCheck {
            current_policy_version_id: pg_base_policy_version_id.as_deref(),
            current_generation: *pg_generation,
            current_updated_at_micros: *pg_base_policy_version_updated_at_micros,
            current_policy_fingerprint: *pg_base_policy_fingerprint,
            latest_policy_fingerprint,
            current_neural_network_present: latest_neural_network.is_some(),
            current_neural_network_fingerprint: *pg_base_neural_network_fingerprint,
            latest_policy_version_id: &metadata.id,
            latest_generation: metadata.generation,
            latest_updated_at_micros: metadata.updated_at_micros,
            latest_neural_network_present: metadata.neural_network.is_some(),
            latest_neural_network_fingerprint,
            current_tactical_learning_fingerprint: *pg_base_tactical_learning_fingerprint,
            latest_tactical_learning_fingerprint,
            local_tactical_evolved_since_pg_refresh: *local_tactical_evolved_since_pg_refresh,
            postgres_tactical_learning_authoritative,
        });
    let mut changed = false;
    if maybe_apply_postgres_evolution_options(
        "postgres_refresh_evolution_options_for_game",
        next_episode,
        &metadata.id,
        metadata.generation,
        evolution_options,
        metadata.search_metadata.as_ref(),
    ) {
        changed = true;
    }
    if refresh_decision.apply_tactical_learning {
        if maybe_apply_postgres_tactical_learning(
            "postgres_refresh_tactical_learning_for_game",
            next_episode,
            &metadata.id,
            metadata.generation,
            config,
            tactical_learning,
            metadata.tactical_learning.clone(),
        )? {
            *local_tactical_evolved_since_pg_refresh = false;
            changed = true;
        }
        if metadata.tactical_learning.is_some() {
            *pg_base_tactical_learning_fingerprint = latest_tactical_learning_fingerprint;
        }
    }
    if refresh_decision.refresh_policy {
        if let Some(version) = store
            .load_latest_policy_with_min_visits(
                experiment_id,
                options.clone(),
                options.clone(),
                resume_min_visits(),
                pg_resume_include_unpromoted,
            )
            .map_err(invalid_data)?
        {
            let version_neural_network_fingerprint = version
                .neural_network
                .as_ref()
                .map(soccer_neural_network_snapshot_fingerprint);
            let version_policy_fingerprint = version
                .policy_fingerprint
                .or_else(|| Some(soccer_team_q_policies_fingerprint(&version.policies)));
            let version_tactical_learning_fingerprint = version
                .tactical_learning
                .as_ref()
                .map(soccer_tactical_learning_weights_fingerprint);
            let version_policy_promotion_baseline =
                policy_promotion_incumbent_baseline_from_search_metadata(
                    version.search_metadata.as_ref(),
                );
            let strongest_policy_promotion_baseline =
                load_strongest_recent_policy_promotion_baseline_for_run(
                    store,
                    experiment_id,
                    version.generation,
                    policy_promotion_baseline_lookback_generations,
                    policy_promotion_gate_min_sample_games,
                )?;
            let version_refresh_decision =
                soccer_postgres_policy_refresh_decision(SoccerPostgresPolicyRefreshCheck {
                    current_policy_version_id: pg_base_policy_version_id.as_deref(),
                    current_generation: *pg_generation,
                    current_updated_at_micros: *pg_base_policy_version_updated_at_micros,
                    current_policy_fingerprint: *pg_base_policy_fingerprint,
                    latest_policy_fingerprint: version_policy_fingerprint,
                    current_neural_network_present: latest_neural_network.is_some(),
                    current_neural_network_fingerprint: *pg_base_neural_network_fingerprint,
                    latest_policy_version_id: &version.id,
                    latest_generation: version.generation,
                    latest_updated_at_micros: version.updated_at_micros,
                    latest_neural_network_present: version.neural_network.is_some(),
                    latest_neural_network_fingerprint: version_neural_network_fingerprint,
                    current_tactical_learning_fingerprint: *pg_base_tactical_learning_fingerprint,
                    latest_tactical_learning_fingerprint: version_tactical_learning_fingerprint,
                    local_tactical_evolved_since_pg_refresh:
                        *local_tactical_evolved_since_pg_refresh,
                    postgres_tactical_learning_authoritative,
                });
            maybe_apply_postgres_evolution_options(
                "postgres_refresh_evolution_options_for_loaded_policy",
                next_episode,
                &version.id,
                version.generation,
                evolution_options,
                version.search_metadata.as_ref(),
            );
            if version_refresh_decision.apply_tactical_learning {
                if maybe_apply_postgres_tactical_learning(
                    "postgres_refresh_tactical_learning_for_loaded_policy",
                    next_episode,
                    &version.id,
                    version.generation,
                    config,
                    tactical_learning,
                    version.tactical_learning.clone(),
                )? {
                    *local_tactical_evolved_since_pg_refresh = false;
                }
                if version.tactical_learning.is_some() {
                    *pg_base_tactical_learning_fingerprint = version_tactical_learning_fingerprint;
                }
            }
            println!(
                "postgres_refresh_policy_for_game next_episode={} policy_version={} previous_policy_version={} generation={} neural_network={} include_unpromoted={} pending_policy_versions={} pending_async_batches={} pending_async_policy_version_batches={}",
                next_episode,
                version.id,
                pg_base_policy_version_id.as_deref().unwrap_or("none"),
                version.generation,
                version.neural_network.is_some(),
                pg_resume_include_unpromoted,
                pg_policy_version_buffer_len,
                pending_async_pg_batches,
                pending_async_policy_version_batches
            );
            *policies = version.policies;
            *latest_neural_network = version.neural_network;
            *pg_base_policy_version_id = Some(version.id.clone());
            *pg_last_policy_version_id = Some(version.id);
            *pg_generation = version.generation;
            *pg_base_policy_version_updated_at_micros = version.updated_at_micros;
            *pg_base_policy_fingerprint = version_policy_fingerprint;
            *pg_base_neural_network_fingerprint = version_neural_network_fingerprint;
            *pg_base_tactical_learning_fingerprint = version_tactical_learning_fingerprint;
            *pg_base_policy_promotion_baseline = retain_stronger_policy_promotion_baseline(
                retain_stronger_policy_promotion_baseline(
                    *pg_base_policy_promotion_baseline,
                    strongest_policy_promotion_baseline
                        .as_ref()
                        .map(|(_, baseline)| *baseline),
                ),
                version_policy_promotion_baseline,
            );
            *local_tactical_evolved_since_pg_refresh = false;
            changed = true;
        }
    }
    if maybe_apply_latest_postgres_neural_snapshot(
        "postgres_refresh_neural_checkpoint_for_game",
        next_episode,
        store,
        experiment_id,
        latest_neural_network,
        pg_base_neural_network_fingerprint,
    )? {
        changed = true;
    }
    Ok(changed)
}

fn reset_policy_to_strongest_promotion_baseline(
    store: &mut SoccerLearningPgStore,
    experiment_id: &str,
    completed_games: usize,
    options: &SoccerQPolicyOptions,
    config: &mut MatchConfig,
    tactical_learning: &mut SoccerTacticalLearningWeights,
    policies: &mut SoccerTeamQPolicies,
    latest_neural_network: &mut Option<SoccerNeuralNetworkSnapshot>,
    pg_base_policy_version_id: &mut Option<String>,
    pg_last_policy_version_id: &mut Option<String>,
    pg_generation: &mut i32,
    pg_base_policy_version_updated_at_micros: &mut i64,
    pg_base_policy_fingerprint: &mut Option<u64>,
    pg_base_neural_network_fingerprint: &mut Option<u64>,
    pg_base_tactical_learning_fingerprint: &mut Option<u64>,
    pg_base_policy_promotion_baseline: &mut Option<PolicyPromotionIncumbentBaseline>,
    local_tactical_evolved_since_pg_refresh: &mut bool,
    policy_promotion_gate_min_sample_games: usize,
    policy_promotion_baseline_lookback_generations: usize,
) -> Result<bool, Box<dyn Error>> {
    let Some((pg_baseline, baseline)) = load_strongest_recent_policy_promotion_baseline_for_run(
        store,
        experiment_id,
        *pg_generation,
        policy_promotion_baseline_lookback_generations,
        policy_promotion_gate_min_sample_games,
    )?
    else {
        return Ok(false);
    };
    let previous_policy_version = pg_base_policy_version_id
        .as_deref()
        .unwrap_or("none")
        .to_string();
    let Some(version) = store
        .load_policy_version_with_min_visits(
            &pg_baseline.policy_version_id,
            options.clone(),
            options.clone(),
            resume_min_visits(),
        )
        .map_err(invalid_data)?
    else {
        println!(
            "policy_local_trial_reset_to_incumbent_missing completed_games={} policy_version={} generation={}",
            completed_games, pg_baseline.policy_version_id, pg_baseline.generation
        );
        return Ok(false);
    };
    let version_neural_network_fingerprint = version
        .neural_network
        .as_ref()
        .map(soccer_neural_network_snapshot_fingerprint);
    let version_policy_fingerprint = version
        .policy_fingerprint
        .or_else(|| Some(soccer_team_q_policies_fingerprint(&version.policies)));
    let version_tactical_learning_fingerprint = version
        .tactical_learning
        .as_ref()
        .map(soccer_tactical_learning_weights_fingerprint);
    maybe_apply_postgres_tactical_learning(
        "policy_local_trial_reset_tactical_learning",
        completed_games,
        &version.id,
        version.generation,
        config,
        tactical_learning,
        version.tactical_learning.clone(),
    )?;
    println!(
        "policy_local_trial_reset_to_incumbent completed_games={} policy_version={} previous_policy_version={} generation={} sample_games={} mean_match_fitness={:.4} mean_play_quality={:.4}",
        completed_games,
        version.id,
        previous_policy_version,
        version.generation,
        pg_baseline.sample_games,
        pg_baseline.mean_match_fitness,
        pg_baseline.mean_play_quality
    );
    *policies = version.policies;
    *latest_neural_network = version.neural_network;
    *pg_base_policy_version_id = Some(version.id.clone());
    *pg_last_policy_version_id = Some(version.id);
    *pg_generation = version.generation;
    *pg_base_policy_version_updated_at_micros = version.updated_at_micros;
    *pg_base_policy_fingerprint = version_policy_fingerprint;
    *pg_base_neural_network_fingerprint = version_neural_network_fingerprint;
    *pg_base_tactical_learning_fingerprint = version_tactical_learning_fingerprint;
    *pg_base_policy_promotion_baseline = Some(baseline);
    *local_tactical_evolved_since_pg_refresh = false;
    let _ = maybe_apply_latest_postgres_neural_snapshot(
        "policy_local_trial_reset_neural_checkpoint",
        completed_games,
        store,
        experiment_id,
        latest_neural_network,
        pg_base_neural_network_fingerprint,
    )?;
    Ok(true)
}

fn write_episode_log<W: Write>(
    file: &mut W,
    episode: &SoccerSelfPlayEpisodeSummary,
) -> Result<(), Box<dyn Error>> {
    serde_json::to_writer(&mut *file, episode)?;
    writeln!(file)?;
    Ok(())
}
fn action_summary(entries: &[SoccerQEntry]) -> Vec<(String, u64, f64)> {
    let mut by_action: BTreeMap<String, (u64, f64)> = BTreeMap::new();
    for entry in entries {
        let visits = u64::from(entry.visits.max(1));
        let item = by_action.entry(entry.action.clone()).or_insert((0, 0.0));
        item.0 += visits;
        item.1 += entry.value * visits as f64;
    }
    let mut summary = by_action
        .into_iter()
        .map(|(action, (visits, weighted_value))| {
            let mean_value = if visits == 0 {
                0.0
            } else {
                weighted_value / visits as f64
            };
            (action, visits, mean_value)
        })
        .collect::<Vec<_>>();
    summary.sort_by(|a, b| {
        b.1.cmp(&a.1)
            .then_with(|| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal))
    });
    summary
}

#[derive(Clone, Debug, Default)]
struct LearningActionOutcomeStats {
    count: usize,
    reward_sum: f64,
    positive_rewards: usize,
    negative_rewards: usize,
    neural_mcts: usize,
    replanned: usize,
    discretized_kick: usize,
    mpc_feasibility_sum: f64,
    pass_receipt_probability_sum: f64,
    pass_receipt_qp_fit_sum: f64,
    pass_receipt_race_advantage_seconds_sum: f64,
    chosen_probability_sum: f64,
    score_margin_sum: f64,
    target_forward_yards_sum: f64,
    realized_forward_yards_sum: f64,
}

impl LearningActionOutcomeStats {
    fn record(&mut self, transition: &SoccerLearningTransition) {
        self.count = self.count.saturating_add(1);
        self.reward_sum += finite_log_metric(transition.reward);
        if transition.reward > 1e-9 {
            self.positive_rewards = self.positive_rewards.saturating_add(1);
        } else if transition.reward < -1e-9 {
            self.negative_rewards = self.negative_rewards.saturating_add(1);
        }
        if transition.decision_context.neural_mcts_selected {
            self.neural_mcts = self.neural_mcts.saturating_add(1);
        }
        if transition.decision_context.learned_mpc_replanned {
            self.replanned = self.replanned.saturating_add(1);
        }
        if learning_action_has_discretized_kick_for_log(&transition.action) {
            self.discretized_kick = self.discretized_kick.saturating_add(1);
        }
        self.mpc_feasibility_sum +=
            finite_log_metric(transition.decision_context.chosen_action_mpc_feasibility);
        self.pass_receipt_probability_sum +=
            finite_log_metric(transition.decision_context.pass_mpc_receipt_probability);
        self.pass_receipt_qp_fit_sum +=
            finite_log_metric(transition.decision_context.pass_receipt_qp_accel_fit);
        self.pass_receipt_race_advantage_seconds_sum += finite_log_metric(
            transition
                .decision_context
                .pass_receipt_race_advantage_seconds,
        );
        self.chosen_probability_sum +=
            finite_log_metric(transition.decision_context.chosen_action_probability);
        self.score_margin_sum += finite_log_metric(transition.decision_context.action_score_margin);
        self.target_forward_yards_sum +=
            finite_log_metric(transition.decision_context.target_forward_yards);
        self.realized_forward_yards_sum +=
            finite_log_metric(transition.decision_context.realized_ball_forward_yards).max(
                finite_log_metric(transition.decision_context.realized_player_forward_yards),
            );
    }

    fn mean_reward(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.reward_sum / self.count as f64
        }
    }

    fn mean_mpc_feasibility(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.mpc_feasibility_sum / self.count as f64
        }
    }

    fn mean_pass_receipt_probability(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.pass_receipt_probability_sum / self.count as f64
        }
    }

    fn mean_pass_receipt_qp_fit(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.pass_receipt_qp_fit_sum / self.count as f64
        }
    }

    fn mean_pass_receipt_race_advantage_seconds(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.pass_receipt_race_advantage_seconds_sum / self.count as f64
        }
    }

    fn mean_chosen_probability(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.chosen_probability_sum / self.count as f64
        }
    }

    fn mean_score_margin(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.score_margin_sum / self.count as f64
        }
    }

    fn mean_target_forward_yards(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.target_forward_yards_sum / self.count as f64
        }
    }

    fn mean_realized_forward_yards(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.realized_forward_yards_sum / self.count as f64
        }
    }
}

fn finite_log_metric(value: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

fn learning_action_has_discretized_kick_for_log(action: &str) -> bool {
    action
        .trim()
        .to_ascii_lowercase()
        .rsplit_once("-kp")
        .is_some_and(|(_, suffix)| {
            !suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_digit())
        })
}

fn learning_ranked_pass_family_for_log(action: &str) -> Option<&'static str> {
    ["aerial-pass", "pass"].iter().copied().find(|prefix| {
        action
            .strip_prefix(prefix)
            .is_some_and(|suffix| !suffix.is_empty() && suffix.bytes().all(|b| b.is_ascii_digit()))
    })
}

fn learning_action_label_for_log(action: &str) -> String {
    let mut label = action.trim().to_ascii_lowercase();
    if label.is_empty() {
        label.push_str("unknown");
    }
    if let Some((base, suffix)) = label.rsplit_once("-kp") {
        if !suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_digit()) {
            label = base.to_string();
        }
    }
    if let Some(family) = learning_ranked_pass_family_for_log(&label) {
        label = family.to_string();
    }
    label
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn team_label_for_log(team: Team) -> &'static str {
    match team {
        Team::Home => "home",
        Team::Away => "away",
    }
}

fn learning_action_outcome_top(buckets: &BTreeMap<String, LearningActionOutcomeStats>) -> String {
    let mut ranked = buckets.iter().collect::<Vec<_>>();
    ranked.sort_by(|(left_action, left), (right_action, right)| {
        right
            .count
            .cmp(&left.count)
            .then_with(|| {
                right
                    .reward_sum
                    .partial_cmp(&left.reward_sum)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| left_action.cmp(right_action))
    });
    let entries = ranked
        .into_iter()
        .take(10)
        .map(|(action, stats)| {
            format!(
                "{}:n={},rmean={:.4},rsum={:.2},pos={},neg={},mcts={},replan={},dk={},mpc={:.3},receipt={:.3},qp={:.3},race={:.2},prob={:.3},margin={:.3},tfwd={:.2},rfwd={:.2}",
                action,
                stats.count,
                stats.mean_reward(),
                stats.reward_sum,
                stats.positive_rewards,
                stats.negative_rewards,
                stats.neural_mcts,
                stats.replanned,
                stats.discretized_kick,
                stats.mean_mpc_feasibility(),
                stats.mean_pass_receipt_probability(),
                stats.mean_pass_receipt_qp_fit(),
                stats.mean_pass_receipt_race_advantage_seconds(),
                stats.mean_chosen_probability(),
                stats.mean_score_margin(),
                stats.mean_target_forward_yards(),
                stats.mean_realized_forward_yards()
            )
        })
        .collect::<Vec<_>>();
    if entries.is_empty() {
        "none".to_string()
    } else {
        entries.join(";")
    }
}

fn learning_action_outcome_entry(action: &str, stats: &LearningActionOutcomeStats) -> String {
    format!(
        "{}:n={},rmean={:.4},rsum={:.2},pos={},neg={},mcts={},replan={},dk={},mpc={:.3},receipt={:.3},qp={:.3},race={:.2},prob={:.3},margin={:.3},tfwd={:.2},rfwd={:.2}",
        action,
        stats.count,
        stats.mean_reward(),
        stats.reward_sum,
        stats.positive_rewards,
        stats.negative_rewards,
        stats.neural_mcts,
        stats.replanned,
        stats.discretized_kick,
        stats.mean_mpc_feasibility(),
        stats.mean_pass_receipt_probability(),
        stats.mean_pass_receipt_qp_fit(),
        stats.mean_pass_receipt_race_advantage_seconds(),
        stats.mean_chosen_probability(),
        stats.mean_score_margin(),
        stats.mean_target_forward_yards(),
        stats.mean_realized_forward_yards()
    )
}

fn learning_action_outcome_key(buckets: &BTreeMap<String, LearningActionOutcomeStats>) -> String {
    const KEY_ACTIONS: [&str; 12] = [
        "pass",
        "aerial-pass",
        "shoot",
        "first-time-shot",
        "dribble",
        "runaround-dribble",
        "nutmeg",
        "vertical-attack",
        "overlap-run",
        "exploit-space-run",
        "hold",
        "recover",
    ];
    KEY_ACTIONS
        .iter()
        .map(|action| {
            let empty = LearningActionOutcomeStats::default();
            learning_action_outcome_entry(action, buckets.get(*action).unwrap_or(&empty))
        })
        .collect::<Vec<_>>()
        .join(";")
}

fn log_learning_action_outcomes(episode: usize, transitions: &[SoccerLearningTransition]) {
    for team in [Team::Home, Team::Away] {
        let mut total = LearningActionOutcomeStats::default();
        let mut buckets: BTreeMap<String, LearningActionOutcomeStats> = BTreeMap::new();
        for transition in transitions
            .iter()
            .filter(|transition| transition.team == team)
        {
            total.record(transition);
            buckets
                .entry(learning_action_label_for_log(&transition.action))
                .or_default()
                .record(transition);
        }
        eprintln!(
            "learning_action_outcomes episode={} team={} transitions={} reward_sum={:.4} reward_mean={:.4} positive={} negative={} neural_mcts={} replanned={} discretized_kick={} mpc_mean={:.4} prob_mean={:.4} margin_mean={:.4} target_forward_mean={:.4} realized_forward_mean={:.4} key={} top={}",
            episode + 1,
            team_label_for_log(team),
            total.count,
            total.reward_sum,
            total.mean_reward(),
            total.positive_rewards,
            total.negative_rewards,
            total.neural_mcts,
            total.replanned,
            total.discretized_kick,
            total.mean_mpc_feasibility(),
            total.mean_chosen_probability(),
            total.mean_score_margin(),
            total.mean_target_forward_yards(),
            total.mean_realized_forward_yards(),
            learning_action_outcome_key(&buckets),
            learning_action_outcome_top(&buckets)
        );
    }
}

#[derive(Debug)]
struct CompletedGame {
    episode_summary: SoccerSelfPlayEpisodeSummary,
    artifact: SoccerTeamPolicyArtifact,
    starting_policies: Arc<SoccerTeamQPolicies>,
    starting_policy_version_id: Option<String>,
    starting_policy_generation: i32,
    starting_tactical_learning: SoccerTacticalLearningWeights,
    policies: SoccerTeamQPolicies,
    config_moments: Vec<SoccerConfigMomentInsert>,
    pass_outcome_samples: Vec<SoccerPassOutcomeSample>,
    neural_network: Option<SoccerNeuralNetworkSnapshot>,
    world_model: Option<SoccerWorldModel>,
    elapsed_seconds: f64,
}

struct BatchNeuralSnapshotSelection {
    episode: usize,
    match_fitness: f64,
    training_steps: usize,
    snapshot_count: usize,
    snapshot: SoccerNeuralNetworkSnapshot,
    world_model: Option<SoccerWorldModel>,
}

fn batch_neural_snapshot_candidate_better(
    mode: BatchNeuralSnapshotSelectionMode,
    candidate_fitness: f64,
    candidate_episode: usize,
    candidate_training_steps: usize,
    best_fitness: f64,
    best_episode: usize,
    best_training_steps: usize,
) -> bool {
    const EPSILON: f64 = 1e-12;
    match mode {
        BatchNeuralSnapshotSelectionMode::Fitness => {
            if candidate_fitness > best_fitness + EPSILON {
                return true;
            }
            if best_fitness > candidate_fitness + EPSILON {
                return false;
            }
            candidate_episode > best_episode
        }
        BatchNeuralSnapshotSelectionMode::TrainingSteps
        | BatchNeuralSnapshotSelectionMode::TrainingStepsGuarded => {
            if candidate_training_steps != best_training_steps {
                return candidate_training_steps > best_training_steps;
            }
            if candidate_fitness > best_fitness + EPSILON {
                return true;
            }
            if best_fitness > candidate_fitness + EPSILON {
                return false;
            }
            candidate_episode > best_episode
        }
        BatchNeuralSnapshotSelectionMode::LatestEpisode => {
            if candidate_episode != best_episode {
                return candidate_episode > best_episode;
            }
            if candidate_training_steps != best_training_steps {
                return candidate_training_steps > best_training_steps;
            }
            candidate_fitness > best_fitness + EPSILON
        }
    }
}

fn select_batch_neural_snapshot(
    completed_games: &mut [CompletedGame],
    mode: BatchNeuralSnapshotSelectionMode,
    analytic_neural_opponent: bool,
    max_fitness_regression: f64,
    min_fitness: f64,
    min_batch_mean_fitness: f64,
    rescue_min_fitness: f64,
) -> Option<BatchNeuralSnapshotSelection> {
    let snapshot_count = completed_games
        .iter()
        .filter(|game| game.neural_network.is_some())
        .count();
    if snapshot_count == 0 {
        return None;
    }
    let batch_fitness_sum: f64 = completed_games
        .iter()
        .filter(|game| game.neural_network.is_some())
        .map(|game| {
            soccer_learning_objective_match_fitness(
                &game.episode_summary.summary,
                analytic_neural_opponent,
            )
        })
        .sum();
    let batch_mean_fitness = batch_fitness_sum / snapshot_count as f64;
    let batch_best_fitness = completed_games
        .iter()
        .filter(|game| game.neural_network.is_some())
        .map(|game| {
            soccer_learning_objective_match_fitness(
                &game.episode_summary.summary,
                analytic_neural_opponent,
            )
        })
        .fold(f64::NEG_INFINITY, f64::max);
    if batch_mean_fitness + 1e-12 < min_batch_mean_fitness
        && batch_best_fitness + 1e-12 < rescue_min_fitness
    {
        return None;
    }
    let fitness_floor = batch_best_fitness
        - max_fitness_regression
            .max(0.0)
            .min(SOCCER_LEARNING_OBJECTIVE_FITNESS_MAX - SOCCER_LEARNING_OBJECTIVE_FITNESS_MIN);
    let mut best = None::<(usize, usize, f64, usize)>;
    for (index, game) in completed_games.iter().enumerate() {
        let Some(snapshot) = game.neural_network.as_ref() else {
            continue;
        };
        let episode = game.episode_summary.episode + 1;
        let fitness = soccer_learning_objective_match_fitness(
            &game.episode_summary.summary,
            analytic_neural_opponent,
        );
        if fitness + 1e-12 < min_fitness {
            continue;
        }
        if mode == BatchNeuralSnapshotSelectionMode::TrainingStepsGuarded
            && fitness + 1e-12 < fitness_floor
        {
            continue;
        }
        let training_steps = snapshot.training_steps;
        let replace = best
            .map(|(_, best_episode, best_fitness, best_training_steps)| {
                batch_neural_snapshot_candidate_better(
                    mode,
                    fitness,
                    episode,
                    training_steps,
                    best_fitness,
                    best_episode,
                    best_training_steps,
                )
            })
            .unwrap_or(true);
        if replace {
            best = Some((index, episode, fitness, training_steps));
        }
    }
    let (index, episode, match_fitness, _) = best?;
    let snapshot = completed_games[index].neural_network.take()?;
    let world_model = completed_games[index].world_model.take();
    let training_steps = snapshot.training_steps;
    Some(BatchNeuralSnapshotSelection {
        episode,
        match_fitness,
        training_steps,
        snapshot_count,
        snapshot,
        world_model,
    })
}

fn select_batch_neural_snapshot_for_training_carry(
    completed_games: &mut [CompletedGame],
    analytic_neural_opponent: bool,
) -> Option<BatchNeuralSnapshotSelection> {
    select_batch_neural_snapshot(
        completed_games,
        BatchNeuralSnapshotSelectionMode::TrainingSteps,
        analytic_neural_opponent,
        SOCCER_LEARNING_OBJECTIVE_FITNESS_MAX - SOCCER_LEARNING_OBJECTIVE_FITNESS_MIN,
        SOCCER_LEARNING_OBJECTIVE_FITNESS_MIN,
        SOCCER_LEARNING_OBJECTIVE_FITNESS_MIN,
        SOCCER_LEARNING_OBJECTIVE_FITNESS_MIN,
    )
}

fn should_merge_batch_policy_delta(
    selected_only: bool,
    selected_episode: Option<usize>,
    completed_episode: usize,
) -> bool {
    if !selected_only {
        return true;
    }
    selected_episode == Some(completed_episode)
}

fn should_carry_no_publishable_neural_snapshot(
    carry_unvalidated: bool,
    carry_no_publishable: bool,
    selected_episode: Option<usize>,
    carried_unvalidated_episode: Option<usize>,
) -> bool {
    carry_unvalidated
        && carry_no_publishable
        && selected_episode.is_none()
        && carried_unvalidated_episode.is_none()
}

fn batch_policy_delta_frontier_episode(
    selected_episode: Option<usize>,
    carried_unvalidated_episode: Option<usize>,
) -> Option<usize> {
    selected_episode.or(carried_unvalidated_episode)
}

#[derive(Clone, Debug)]
struct EvolutionSearchSample {
    summary: MatchSummary,
    policies: SoccerTeamQPolicies,
    tactical_summary: SoccerTacticalLearningSummary,
    starting_tactical_learning: SoccerTacticalLearningWeights,
    fitness: f64,
}

impl EvolutionSearchSample {
    fn from_completed_game(game: &CompletedGame) -> Self {
        Self {
            summary: game.episode_summary.summary.clone(),
            policies: game.policies.clone(),
            tactical_summary: game.artifact.tactical_summary.clone(),
            starting_tactical_learning: game.starting_tactical_learning.clone(),
            fitness: soccer_learning_run_score(&game.episode_summary.summary).match_fitness,
        }
    }
}

/// Resident set size in MB (Linux `/proc/self/statm`), 0 if unavailable. Feeds the
/// per-game `soccer_mem_diag` line so we can locate unbounded growth in the learner.
fn process_rss_mb() -> u64 {
    std::fs::read_to_string("/proc/self/statm")
        .ok()
        .and_then(|s| {
            s.split_whitespace()
                .nth(1)
                .and_then(|pages| pages.parse::<u64>().ok())
        })
        .map(|pages| pages.saturating_mul(4096) / (1024 * 1024))
        .unwrap_or(0)
}

fn push_evolution_search_samples(
    samples: &mut VecDeque<EvolutionSearchSample>,
    completed_games: &[CompletedGame],
    window_games: usize,
) {
    let window_games = window_games.max(1);
    for game in completed_games {
        samples.push_back(EvolutionSearchSample::from_completed_game(game));
        while samples.len() > window_games {
            samples.pop_front();
        }
    }
}

fn push_policy_promotion_summary(
    summaries: &mut VecDeque<MatchSummary>,
    summary: &MatchSummary,
    window_games: usize,
) {
    let window_games = window_games.max(1);
    summaries.push_back(summary.clone());
    while summaries.len() > window_games {
        summaries.pop_front();
    }
}

fn clear_evolution_search_samples_after_postgres_refresh(
    event_label: &str,
    next_episode: usize,
    policy_version_id: Option<&str>,
    samples: &mut VecDeque<EvolutionSearchSample>,
) -> usize {
    let samples_cleared = samples.len();
    samples.clear();
    if samples_cleared > 0 {
        println!(
            "{event_label} next_episode={next_episode} policy_version={} samples_cleared={samples_cleared}",
            policy_version_id.unwrap_or("none")
        );
    }
    samples_cleared
}

fn run_evolution_search_metadata(
    completed_games: usize,
    elite_games: usize,
    window_games: usize,
    sample_count: usize,
    best_fitness: f64,
    options: SoccerEvolutionOptions,
    previous_tactical_learning: &SoccerTacticalLearningWeights,
    evolved_tactical_learning: &SoccerTacticalLearningWeights,
    curriculum_stage: &'static str,
    promotion_evaluation: &SoccerPolicyPromotionGateEvaluation,
    candidate_promoted: bool,
) -> serde_json::Value {
    let tactical = serde_json::json!({
        "algorithm": "evolutionary-genetic-programming-tactical-search",
        "completedGames": completed_games,
        "eliteGames": elite_games,
        "windowGames": window_games,
        "sampleCount": sample_count,
        "bestFitness": best_fitness,
        "options": options,
        "previousTacticalLearning": previous_tactical_learning,
        "evolvedTacticalLearning": evolved_tactical_learning,
        "previousTacticalLearningFingerprint": soccer_tactical_learning_weights_fingerprint(
            previous_tactical_learning,
        ),
        "evolvedTacticalLearningFingerprint": soccer_tactical_learning_weights_fingerprint(
            evolved_tactical_learning,
        )
    });
    serde_json::json!({
        "algorithm": "evolutionary-genetic-programming",
        "completedGames": completed_games,
        "eliteGames": elite_games,
        "windowGames": window_games,
        "sampleCount": sample_count,
        "bestFitness": best_fitness,
        "curriculumStage": curriculum_stage,
        "candidatePromoted": candidate_promoted,
        "options": options,
        "promotion": policy_promotion_search_metadata(curriculum_stage, promotion_evaluation),
        "tactical": tactical
    })
}

struct SoccerLearningWorkerTask {
    episode: usize,
    config: MatchConfig,
    starting_policies: Arc<SoccerTeamQPolicies>,
    starting_policy_version_id: Option<String>,
    starting_policy_generation: i32,
    initial_neural_network: Option<Arc<SoccerNeuralNetworkSnapshot>>,
    analytic_neural_opponent: bool,
    adversarial_moment_windows: Arc<Vec<SoccerMomentWindow>>,
    print_progress: bool,
    neural_drain_timeout: Duration,
    retain_neural_network_in_game_artifact: bool,
}

struct CachedTeamQPoliciesArc {
    fingerprint: u64,
    policies: Arc<SoccerTeamQPolicies>,
}

fn cached_team_policies_arc(
    policies: &SoccerTeamQPolicies,
    cache: &mut Option<CachedTeamQPoliciesArc>,
) -> Arc<SoccerTeamQPolicies> {
    let fingerprint = soccer_team_q_policies_fingerprint(policies);
    let rebuild_cache = match cache.as_ref() {
        Some(cached) => cached.fingerprint != fingerprint,
        None => true,
    };
    if rebuild_cache {
        *cache = Some(CachedTeamQPoliciesArc {
            fingerprint,
            policies: Arc::new(policies.clone()),
        });
    }
    cache
        .as_ref()
        .map(|cached| Arc::clone(&cached.policies))
        .expect("policy cache must exist after rebuild check")
}

struct CachedNeuralNetworkSnapshotArc {
    fingerprint: u64,
    snapshot: Arc<SoccerNeuralNetworkSnapshot>,
}

fn cached_neural_network_snapshot_arc(
    latest_neural_network: &Option<SoccerNeuralNetworkSnapshot>,
    cache: &mut Option<CachedNeuralNetworkSnapshotArc>,
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
        *cache = Some(CachedNeuralNetworkSnapshotArc {
            fingerprint,
            snapshot: Arc::new(snapshot.clone()),
        });
    }
    cache.as_ref().map(|cached| Arc::clone(&cached.snapshot))
}

struct SoccerLearningWorkerPool {
    sender: Option<mpsc::SyncSender<SoccerLearningWorkerTask>>,
    receiver: mpsc::Receiver<(usize, Result<CompletedGame, String>)>,
    handles: Vec<thread::JoinHandle<()>>,
}

impl SoccerLearningWorkerPool {
    fn start(worker_count: usize) -> Self {
        let worker_count = worker_count.clamp(1, 100);
        let (task_sender, task_receiver) =
            mpsc::sync_channel::<SoccerLearningWorkerTask>(worker_count);
        let task_receiver = Arc::new(Mutex::new(task_receiver));
        let (result_sender, result_receiver) =
            mpsc::channel::<(usize, Result<CompletedGame, String>)>();
        let mut handles = Vec::with_capacity(worker_count);

        for _ in 0..worker_count {
            let task_receiver = Arc::clone(&task_receiver);
            let result_sender = result_sender.clone();
            handles.push(thread::spawn(move || loop {
                let task = {
                    let receiver = task_receiver
                        .lock()
                        .expect("soccer learning run task receiver poisoned");
                    receiver.recv()
                };
                let Ok(task) = task else {
                    break;
                };
                let episode = task.episode;
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    run_game(
                        task.episode,
                        task.config,
                        task.starting_policies,
                        task.starting_policy_version_id,
                        task.starting_policy_generation,
                        task.initial_neural_network,
                        task.analytic_neural_opponent,
                        task.adversarial_moment_windows,
                        task.print_progress,
                        task.neural_drain_timeout,
                        task.retain_neural_network_in_game_artifact,
                    )
                }))
                .unwrap_or_else(|_| Err("soccer learning worker thread panicked".to_string()));
                let _ = result_sender.send((episode, result));
            }));
        }
        drop(result_sender);

        Self {
            sender: Some(task_sender),
            receiver: result_receiver,
            handles,
        }
    }

    fn submit(&self, task: SoccerLearningWorkerTask) -> Result<(), Box<dyn Error>> {
        let Some(sender) = &self.sender else {
            return Err(io_error("soccer learning worker pool is closed").into());
        };
        sender
            .send(task)
            .map_err(|err| io_error(format!("soccer learning worker task send failed: {err}")))?;
        Ok(())
    }

    fn recv_completed_game(&self) -> Result<CompletedGame, Box<dyn Error>> {
        let (_, result) = self.receiver.recv().map_err(|err| {
            invalid_data(format!(
                "soccer learning worker result channel closed: {err}"
            ))
        })?;
        result.map_err(|err| invalid_data(err).into())
    }

    fn shutdown(mut self) -> Result<(), Box<dyn Error>> {
        self.sender.take();
        let mut panicked = false;
        for handle in self.handles.drain(..) {
            if handle.join().is_err() {
                panicked = true;
            }
        }
        if panicked {
            Err(io_error("soccer learning worker thread panicked").into())
        } else {
            Ok(())
        }
    }
}

impl Drop for SoccerLearningWorkerPool {
    fn drop(&mut self) {
        self.sender.take();
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

#[derive(Debug)]
struct PendingPostgresCompletedRun {
    completed_episode: usize,
    game: SoccerLearningCompletedGame,
    base_policy_version_id: Option<String>,
    output_policy_version_id: Option<String>,
    generation: i32,
}

#[derive(Debug)]
struct PendingPostgresPolicyVersion {
    id: String,
    parent_policy_version_id: Option<String>,
    generation: i32,
    version_label: String,
    source_kind: &'static str,
    status: &'static str,
    comparative_promotion_approved: bool,
    config: MatchConfig,
    home_options: SoccerQPolicyOptions,
    away_options: SoccerQPolicyOptions,
    policies: SoccerTeamQPolicies,
    fitness: f64,
    neural_network: Option<SoccerNeuralNetworkSnapshot>,
    search_metadata: Option<serde_json::Value>,
}

struct PostgresCompletedRunBatch {
    experiment_id: String,
    runner_id: String,
    completed_run_retention_games: usize,
    policy_version_batches: usize,
    pending_policy_versions: Vec<PendingPostgresPolicyVersion>,
    pending_runs: Vec<PendingPostgresCompletedRun>,
    shard_index: usize,
    shard_count: usize,
}

impl PostgresCompletedRunBatch {
    fn can_absorb(&self, other: &Self) -> bool {
        self.experiment_id == other.experiment_id
            && self.runner_id == other.runner_id
            && self.completed_run_retention_games == other.completed_run_retention_games
            && self.shard_index == other.shard_index
            && self.shard_count == other.shard_count
    }

    fn absorb(&mut self, mut other: Self) {
        self.policy_version_batches = self
            .policy_version_batches
            .saturating_add(other.policy_version_batches);
        self.pending_policy_versions
            .append(&mut other.pending_policy_versions);
        self.pending_runs.append(&mut other.pending_runs);
    }
}

struct PostgresCompletedRunWriteResult {
    queue_batches: usize,
    policy_version_batches: usize,
    result: Result<usize, String>,
}

struct AsyncPostgresCompletedRunWriter {
    sender: Option<mpsc::SyncSender<PostgresCompletedRunBatch>>,
    receiver: mpsc::Receiver<PostgresCompletedRunWriteResult>,
    handle: Option<thread::JoinHandle<()>>,
    pending_batches: usize,
    pending_policy_version_batches: usize,
}

impl AsyncPostgresCompletedRunWriter {
    fn start(queue_batches: usize, coalesce_batches: usize, coalesce_wait: Duration) -> Self {
        let (sender, receiver) =
            mpsc::sync_channel::<PostgresCompletedRunBatch>(queue_batches.max(1));
        let (result_sender, result_receiver) = mpsc::channel::<PostgresCompletedRunWriteResult>();
        let coalesce_batches = coalesce_batches.max(1);
        let handle = thread::spawn(move || {
            let mut store = match SoccerLearningPgStore::connect_from_env() {
                Ok(Some(store)) => store,
                Ok(None) => {
                    while let Ok(batch) = receiver.recv() {
                        let _ = result_sender.send(PostgresCompletedRunWriteResult {
                            queue_batches: 1,
                            policy_version_batches: batch.policy_version_batches,
                            result: Err(
                                "postgres completed-run writer could not find a database URL"
                                    .to_string(),
                            ),
                        });
                    }
                    return;
                }
                Err(error) => {
                    while let Ok(batch) = receiver.recv() {
                        let _ = result_sender.send(PostgresCompletedRunWriteResult {
                            queue_batches: 1,
                            policy_version_batches: batch.policy_version_batches,
                            result: Err(format!(
                                "postgres completed-run writer connect failed: {error}"
                            )),
                        });
                    }
                    return;
                }
            };

            let mut deferred_batch = None::<PostgresCompletedRunBatch>;
            loop {
                let mut batch = match deferred_batch.take() {
                    Some(batch) => batch,
                    None => match receiver.recv() {
                        Ok(batch) => batch,
                        Err(_) => break,
                    },
                };
                let mut queue_batches = 1usize;
                let coalesce_started = Instant::now();
                while queue_batches < coalesce_batches {
                    match receiver.try_recv() {
                        Ok(next_batch) => {
                            if batch.can_absorb(&next_batch) {
                                batch.absorb(next_batch);
                                queue_batches = queue_batches.saturating_add(1);
                            } else {
                                deferred_batch = Some(next_batch);
                                break;
                            }
                        }
                        Err(mpsc::TryRecvError::Empty) => {
                            if coalesce_wait.is_zero() {
                                break;
                            }
                            let elapsed = coalesce_started.elapsed();
                            if elapsed >= coalesce_wait {
                                break;
                            }
                            match receiver.recv_timeout(coalesce_wait.saturating_sub(elapsed)) {
                                Ok(next_batch) => {
                                    if batch.can_absorb(&next_batch) {
                                        batch.absorb(next_batch);
                                        queue_batches = queue_batches.saturating_add(1);
                                    } else {
                                        deferred_batch = Some(next_batch);
                                        break;
                                    }
                                }
                                Err(mpsc::RecvTimeoutError::Timeout)
                                | Err(mpsc::RecvTimeoutError::Disconnected) => break,
                            }
                        }
                        Err(mpsc::TryRecvError::Disconnected) => break,
                    }
                }
                let result = flush_postgres_completed_runs(
                    &mut store,
                    &batch.experiment_id,
                    &batch.runner_id,
                    &mut batch.pending_policy_versions,
                    &mut batch.pending_runs,
                    batch.completed_run_retention_games,
                    batch.shard_index,
                    batch.shard_count,
                )
                .map_err(|err| err.to_string());
                let _ = result_sender.send(PostgresCompletedRunWriteResult {
                    queue_batches,
                    policy_version_batches: batch.policy_version_batches,
                    result,
                });
            }
        });

        AsyncPostgresCompletedRunWriter {
            sender: Some(sender),
            receiver: result_receiver,
            handle: Some(handle),
            pending_batches: 0,
            pending_policy_version_batches: 0,
        }
    }

    fn record_write_result(
        &mut self,
        write_result: PostgresCompletedRunWriteResult,
    ) -> Result<usize, String> {
        self.pending_batches = self
            .pending_batches
            .saturating_sub(write_result.queue_batches);
        self.pending_policy_version_batches = self
            .pending_policy_version_batches
            .saturating_sub(write_result.policy_version_batches);
        write_result.result
    }

    fn drain_finished(&mut self) -> Result<usize, String> {
        let mut persisted = 0usize;
        loop {
            match self.receiver.try_recv() {
                Ok(write_result) => {
                    persisted = persisted.saturating_add(self.record_write_result(write_result)?);
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    if self.pending_batches > 0 {
                        return Err(
                            "postgres completed-run writer stopped before all batches finished"
                                .to_string(),
                        );
                    }
                    break;
                }
            }
        }
        Ok(persisted)
    }

    fn drain_policy_versions_finished(&mut self) -> Result<usize, String> {
        let mut persisted = self.drain_finished()?;
        while self.pending_policy_version_batches > 0 {
            match self.receiver.recv() {
                Ok(write_result) => {
                    persisted = persisted.saturating_add(self.record_write_result(write_result)?);
                }
                Err(error) => {
                    return Err(format!(
                        "postgres completed-run writer channel closed before policy versions were durable: {error}"
                    ));
                }
            }
        }
        Ok(persisted)
    }

    fn enqueue(
        &mut self,
        experiment_id: &str,
        runner_id: &str,
        pending_policy_versions: &mut Vec<PendingPostgresPolicyVersion>,
        pending_runs: &mut Vec<PendingPostgresCompletedRun>,
        completed_run_retention_games: usize,
        shard_index: usize,
        shard_count: usize,
    ) -> Result<usize, String> {
        let persisted = self.drain_finished()?;
        if pending_policy_versions.is_empty() && pending_runs.is_empty() {
            return Ok(persisted);
        }
        let Some(sender) = &self.sender else {
            return Err("postgres completed-run writer is closed".to_string());
        };
        let policy_version_batches = usize::from(!pending_policy_versions.is_empty());
        let batch = PostgresCompletedRunBatch {
            experiment_id: experiment_id.to_string(),
            runner_id: runner_id.to_string(),
            completed_run_retention_games,
            policy_version_batches,
            pending_policy_versions: std::mem::take(pending_policy_versions),
            pending_runs: std::mem::take(pending_runs),
            shard_index,
            shard_count,
        };
        sender
            .send(batch)
            .map_err(|err| format!("queue postgres completed-run batch: {err}"))?;
        self.pending_batches = self.pending_batches.saturating_add(1);
        self.pending_policy_version_batches = self
            .pending_policy_version_batches
            .saturating_add(policy_version_batches);
        Ok(persisted)
    }

    fn finish(mut self) -> Result<usize, String> {
        let mut persisted = self.drain_finished()?;
        let mut first_error = None::<String>;
        self.sender.take();

        while self.pending_batches > 0 {
            match self.receiver.recv() {
                Ok(write_result) => match self.record_write_result(write_result) {
                    Ok(count) => persisted = persisted.saturating_add(count),
                    Err(error) => {
                        if first_error.is_none() {
                            first_error = Some(error);
                        }
                    }
                },
                Err(error) => {
                    if first_error.is_none() {
                        first_error = Some(format!(
                            "postgres completed-run writer channel closed early: {error}"
                        ));
                    }
                    break;
                }
            }
        }

        if let Some(handle) = self.handle.take() {
            if handle.join().is_err() && first_error.is_none() {
                first_error = Some("postgres completed-run writer thread panicked".to_string());
            }
        }

        if let Some(error) = first_error {
            Err(error)
        } else {
            Ok(persisted)
        }
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GameManifestEntry {
    episode: usize,
    seed: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    artifact_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    artifact_kind: Option<String>,
    score_home: u32,
    score_away: u32,
    ticks: u64,
    simulated_seconds: f64,
    transitions: usize,
    home_policy_entries: usize,
    away_policy_entries: usize,
    home_policy_target_entries: usize,
    away_policy_target_entries: usize,
    elapsed_seconds: f64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct RunManifest {
    run_id: String,
    run_dir: String,
    game_dir: String,
    games: usize,
    parallel_games: usize,
    shard_index: usize,
    shard_count: usize,
    base_seed: u32,
    effective_seed: u32,
    config: MatchConfig,
    options: SoccerQPolicyOptions,
    final_artifact_path: String,
    learned_params_path: String,
    checkpoint_artifact_path: String,
    episode_log_path: String,
    checkpoint_interval_games: usize,
    artifact_max_entries_per_policy: usize,
    max_policy_entries_per_team: usize,
    max_policy_target_entries_per_team: usize,
    min_policy_visits: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    moment_replay_path: Option<String>,
    moment_replay_records: usize,
    moment_replay_transitions: usize,
    moment_replay_passes: usize,
    moment_replay_reward_scale: f64,
    postgres_enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    postgres_experiment_slug: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    postgres_experiment_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    postgres_last_policy_version_id: Option<String>,
    postgres_persisted_games: usize,
    postgres_completed_run_batch_games: usize,
    write_final_policy_artifact: bool,
    write_game_artifacts: bool,
    game_artifact_mode: String,
    game_artifacts: Vec<GameManifestEntry>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ActionSummaryEntry {
    action: String,
    visits: u64,
    mean_q: f64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CompactGameArtifact {
    artifact_kind: String,
    episode: usize,
    seed: u64,
    config: MatchConfig,
    summary: soccer_engine::des::general::soccer::MatchSummary,
    tactical_summary: SoccerTacticalLearningSummary,
    transitions: usize,
    home_policy_entries: usize,
    away_policy_entries: usize,
    home_policy_target_entries: usize,
    away_policy_target_entries: usize,
    elapsed_seconds: f64,
    home_action_summary: Vec<ActionSummaryEntry>,
    away_action_summary: Vec<ActionSummaryEntry>,
}

fn io_error(message: impl Into<String>) -> IoError {
    IoError::new(ErrorKind::Other, message.into())
}

fn invalid_data(message: impl Into<String>) -> IoError {
    IoError::new(ErrorKind::InvalidData, message.into())
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut tmp_name = path.as_os_str().to_os_string();
    tmp_name.push(format!(".tmp-{}", std::process::id()));
    let tmp_path = PathBuf::from(tmp_name);
    let result = (|| -> Result<(), Box<dyn Error>> {
        let file = fs::File::create(&tmp_path)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer(&mut writer, value)?;
        writer.flush()?;
        let file = writer.into_inner()?;
        file.sync_all()?;
        Ok(())
    })();
    if let Err(err) = result {
        let _ = fs::remove_file(&tmp_path);
        return Err(err);
    }
    fs::rename(&tmp_path, path)?;
    if let Some(parent) = path.parent() {
        let _ = fs::File::open(parent).and_then(|dir| dir.sync_all());
    }
    Ok(())
}

fn neural_sidecar_path_for_policy_artifact(policy_path: &Path) -> PathBuf {
    let Some(parent) = policy_path.parent() else {
        return PathBuf::from("out/soccer-live/team-policy.neural.json");
    };
    let stem = policy_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.is_empty())
        .unwrap_or("team-policy");
    parent.join(format!("{stem}.neural.json"))
}

fn local_policy_promotion_best_path_for_policy_artifact(policy_path: &Path) -> PathBuf {
    let Some(parent) = policy_path.parent() else {
        return PathBuf::from("out/soccer-live/team-policy.local-best.json");
    };
    let stem = policy_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.is_empty())
        .unwrap_or("team-policy");
    parent.join(format!("{stem}.local-best.json"))
}

fn candidate_checkpoint_path(path: &Path) -> PathBuf {
    let candidate_name = match (
        path.file_stem().and_then(|stem| stem.to_str()),
        path.extension().and_then(|extension| extension.to_str()),
        path.file_name().and_then(|file_name| file_name.to_str()),
    ) {
        (Some(stem), Some(extension), _) if !stem.is_empty() && !extension.is_empty() => {
            format!("{stem}.candidate.{extension}")
        }
        (Some(stem), _, _) if !stem.is_empty() => format!("{stem}.candidate"),
        (_, _, Some(file_name)) if !file_name.is_empty() => format!("{file_name}.candidate"),
        _ => "checkpoint.candidate.json".to_string(),
    };
    path.parent()
        .map(|parent| parent.join(&candidate_name))
        .unwrap_or_else(|| PathBuf::from(candidate_name))
}

fn training_best_checkpoint_path(path: &Path) -> PathBuf {
    let training_best_name = match (
        path.file_stem().and_then(|stem| stem.to_str()),
        path.extension().and_then(|extension| extension.to_str()),
        path.file_name().and_then(|file_name| file_name.to_str()),
    ) {
        (Some(stem), Some(extension), _) if !stem.is_empty() && !extension.is_empty() => {
            format!("{stem}.training-best.{extension}")
        }
        (Some(stem), _, _) if !stem.is_empty() => format!("{stem}.training-best"),
        (_, _, Some(file_name)) if !file_name.is_empty() => format!("{file_name}.training-best"),
        _ => "checkpoint.training-best.json".to_string(),
    };
    path.parent()
        .map(|parent| parent.join(&training_best_name))
        .unwrap_or_else(|| PathBuf::from(training_best_name))
}

fn should_publish_live_checkpoint(
    best_only: bool,
    evaluation: &SoccerPolicyPromotionGateEvaluation,
    protected_baseline: Option<PolicyPromotionIncumbentBaseline>,
    local_best: Option<&LocalNeuralPromotionTrialBest>,
    has_neural_snapshot: bool,
) -> bool {
    if !best_only {
        return true;
    }
    if !has_neural_snapshot {
        return false;
    }
    if evaluation.enabled && !evaluation.eligible {
        return false;
    }
    if protected_baseline
        .map(|baseline| !policy_promotion_evaluation_beats_incumbent_baseline(evaluation, baseline))
        .unwrap_or(false)
    {
        return false;
    }
    local_best
        .map(|best| policy_promotion_evaluation_beats_local_best(evaluation, best))
        .unwrap_or(true)
}

fn write_neural_sidecar_for_policy_artifact(
    policy_path: &Path,
    snapshot: Option<&SoccerNeuralNetworkSnapshot>,
) -> Result<Option<PathBuf>, Box<dyn Error>> {
    let Some(snapshot) = snapshot else {
        return Ok(None);
    };
    let sidecar_path = neural_sidecar_path_for_policy_artifact(policy_path);
    write_json(&sidecar_path, snapshot)?;
    Ok(Some(sidecar_path))
}

fn load_neural_sidecar_for_policy_artifact(
    policy_path: &Path,
) -> Result<Option<(PathBuf, SoccerNeuralNetworkSnapshot)>, Box<dyn Error>> {
    let sidecar_path = neural_sidecar_path_for_policy_artifact(policy_path);
    if !sidecar_path.exists() {
        return Ok(None);
    }
    let raw = fs::read_to_string(&sidecar_path)?;
    let snapshot = serde_json::from_str::<SoccerNeuralNetworkSnapshot>(&raw)?;
    Ok(Some((sidecar_path, snapshot)))
}

fn load_local_policy_promotion_best(
    policy_path: &Path,
) -> Result<Option<(PathBuf, LocalPolicyPromotionBestMetadata)>, Box<dyn Error>> {
    let metadata_path = local_policy_promotion_best_path_for_policy_artifact(policy_path);
    if !metadata_path.exists() {
        return Ok(None);
    }
    let raw = fs::read_to_string(&metadata_path)?;
    let metadata = serde_json::from_str::<LocalPolicyPromotionBestMetadata>(&raw)?;
    Ok(Some((metadata_path, metadata)))
}

fn write_local_policy_promotion_best(
    policy_path: &Path,
    completed_games: usize,
    evaluation: &SoccerPolicyPromotionGateEvaluation,
) -> Result<Option<PathBuf>, Box<dyn Error>> {
    let Some(metadata) =
        local_policy_promotion_best_metadata_from_evaluation(completed_games, evaluation)
    else {
        return Ok(None);
    };
    let metadata_path = local_policy_promotion_best_path_for_policy_artifact(policy_path);
    write_json(&metadata_path, &metadata)?;
    Ok(Some(metadata_path))
}

#[allow(clippy::too_many_arguments)]
fn write_training_best_resume_only_checkpoint(
    reason: &str,
    checkpoint_completed_games: usize,
    best_completed_games: usize,
    artifact: &SoccerSelfPlayTrainingArtifact,
    neural_network: Option<&SoccerNeuralNetworkSnapshot>,
    evaluation: &SoccerPolicyPromotionGateEvaluation,
    training_best_checkpoint_artifact_path: &Path,
    training_best_learned_params_path: &Path,
    write_learned_params_artifact: bool,
    artifact_max_entries_per_policy: usize,
) -> Result<(), Box<dyn Error>> {
    let training_best_export =
        compact_training_artifact_for_export(artifact, artifact_max_entries_per_policy);
    write_json(
        training_best_checkpoint_artifact_path,
        &training_best_export,
    )?;
    if let Some(metadata_path) = write_local_policy_promotion_best(
        training_best_checkpoint_artifact_path,
        best_completed_games,
        evaluation,
    )? {
        println!(
            "checkpoint_training_best_resume_only_metadata_written checkpoint_games={} best_games={} sample_games={} mean_match_fitness={:.4} reason={} artifact={}",
            checkpoint_completed_games,
            best_completed_games,
            evaluation.sample_games,
            evaluation.mean_match_fitness,
            reason,
            metadata_path.display()
        );
    }
    if write_learned_params_artifact {
        let training_best_params =
            SoccerSelfPlayLearnedParams::from_training_artifact_with_neural_network(
                artifact,
                neural_network.cloned(),
            );
        write_json(training_best_learned_params_path, &training_best_params)?;
    }
    if let Some(sidecar_path) = write_neural_sidecar_for_policy_artifact(
        training_best_checkpoint_artifact_path,
        neural_network,
    )? {
        println!(
            "checkpoint_training_best_resume_only_neural_sidecar checkpoint_games={} best_games={} reason={} artifact={}",
            checkpoint_completed_games,
            best_completed_games,
            reason,
            sidecar_path.display()
        );
    }
    println!(
        "checkpoint_training_best_resume_only_synced checkpoint_games={} best_games={} sample_games={} mean_match_fitness={:.4} best_match_fitness={:.4} mean_play_quality={:.4} reason={} artifact={}",
        checkpoint_completed_games,
        best_completed_games,
        evaluation.sample_games,
        evaluation.mean_match_fitness,
        evaluation.best_match_fitness,
        evaluation.mean_play_quality,
        reason,
        training_best_checkpoint_artifact_path.display()
    );
    Ok(())
}

fn default_run_id() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    format!("soccer-learning-{seconds}")
}

fn policies_from_self_play_artifact(
    artifact: &SoccerSelfPlayTrainingArtifact,
) -> Result<SoccerTeamQPolicies, String> {
    Ok(SoccerTeamQPolicies {
        home: SoccerQPolicy::from_entries_with_targets(
            artifact.options.clone(),
            &artifact.home_entries,
            &artifact.home_target_entries,
        )?,
        away: SoccerQPolicy::from_entries_with_targets(
            artifact.options.clone(),
            &artifact.away_entries,
            &artifact.away_target_entries,
        )?,
    })
}

fn load_initial_policies(
    path: Option<&str>,
    options: SoccerQPolicyOptions,
) -> Result<SoccerTeamQPolicies, Box<dyn Error>> {
    let Some(path) = path else {
        return Ok(SoccerTeamQPolicies::new(options));
    };
    let raw = fs::read_to_string(path)?;
    if let Ok(artifact) = serde_json::from_str::<SoccerSelfPlayTrainingArtifact>(&raw) {
        return policies_from_self_play_artifact(&artifact).map_err(|err| invalid_data(err).into());
    }
    if let Ok(params) = serde_json::from_str::<SoccerSelfPlayLearnedParams>(&raw) {
        return SoccerTeamQPolicies::from_learned_params(&params)
            .map_err(|err| invalid_data(err).into());
    }
    if let Ok(artifact) = serde_json::from_str::<SoccerTeamPolicyArtifact>(&raw) {
        return SoccerTeamQPolicies::from_artifact(&artifact)
            .map_err(|err| invalid_data(err).into());
    }
    Err(invalid_data(format!(
        "resume artifact {path} is neither a self-play, learned-params, nor team-policy artifact"
    ))
    .into())
}

#[allow(clippy::too_many_arguments)]
fn fallback_rejected_resume_policy_to_checkpoint_artifact(
    reason: &str,
    resume_artifact: Option<&str>,
    rejected_policy_path: &Path,
    checkpoint_artifact_path: &Path,
    options: &SoccerQPolicyOptions,
    policies: &mut SoccerTeamQPolicies,
    pg_base_policy_fingerprint: &mut Option<u64>,
) -> Result<bool, Box<dyn Error>> {
    if !resume_artifact
        .map(|path| Path::new(path) == rejected_policy_path)
        .unwrap_or(false)
    {
        return Ok(false);
    }
    if rejected_policy_path == checkpoint_artifact_path {
        return Ok(false);
    }
    if !checkpoint_artifact_path.exists() {
        println!(
            "resume_policy_artifact_fallback_skipped rejected_artifact={} fallback_artifact={} reason={} fallback_missing=true",
            rejected_policy_path.display(),
            checkpoint_artifact_path.display(),
            reason
        );
        return Ok(false);
    }
    let fallback_path = checkpoint_artifact_path.to_string_lossy().to_string();
    *policies = load_initial_policies(Some(fallback_path.as_str()), options.clone())?;
    *pg_base_policy_fingerprint = Some(soccer_team_q_policies_fingerprint(policies));
    println!(
        "resume_policy_artifact_fallback_to_checkpoint rejected_artifact={} fallback_artifact={} reason={}",
        rejected_policy_path.display(),
        checkpoint_artifact_path.display(),
        reason
    );
    Ok(true)
}

/// Max fitness *regression* (relative to the incumbent active head) tolerated before
/// a newer generation still goes `active`, resolved once from env at learner startup
/// (`SOCCER_POLICY_ACTIVE_MAX_FITNESS_REGRESSION`, or the `SOCCER_BATCH_…` alias) so
/// the anti-regression ratchet is tunable from the configmap without a rebuild. Falls
/// back to the compile-time [`SOCCER_POLICY_ACTIVE_MAX_FITNESS_REGRESSION`] default
/// when never set (the queue binary and unit tests keep the constant). Lower ⇒
/// stricter: a newer generation must land closer to — or above — the incumbent's
/// fitness to promote, which directly counters the observed tip regression; too low
/// risks the promotion stall, so it is operator-tuned against the live climb/stall
/// tradeoff rather than baked in.
static ACTIVE_MAX_FITNESS_REGRESSION: std::sync::OnceLock<f64> = std::sync::OnceLock::new();

/// Resolve the configured activation regression tolerance, defaulting to the
/// compile-time constant when unset.
fn active_max_fitness_regression() -> f64 {
    ACTIVE_MAX_FITNESS_REGRESSION
        .get()
        .copied()
        .unwrap_or_else(soccer_policy_active_max_fitness_regression)
}

/// Frozen-anchor promotion gate (DEFAULT-OFF). The live analogue of
/// [`soccer_engine::des::general::soccer_eval_gate::evaluate_promotion`]: a
/// newly-learned candidate may only be written `active` (and advance the anchor) if
/// it beats the current *frozen anchor* brain head-to-head over a held-out slate with
/// statistical confidence. Wired into the learner so "newer generation" means "beat
/// the one it replaces" rather than "scored a lucky self-play `match_fitness`", which
/// is the mechanism that let the active tip regress below earlier peaks. Brains are
/// compared by their neural snapshots — the live on-ball driver — mirroring the
/// standalone `soccer_eval_gate_run` gate. Off unless
/// `SOCCER_ANCHOR_PROMOTION_GATE_ENABLED=1`.
#[derive(Clone, Copy, Debug)]
struct AnchorPromotionGateConfig {
    enabled: bool,
    /// Held-out games per verdict (candidate vs anchor, home/away alternated).
    games: usize,
    /// Sim minutes per held-out fixture (short: this runs inline in the learner).
    minutes: f64,
    /// Run the gate only every Nth `active` write (1 = every promotion), to bound the
    /// extra sim cost when policy versions are written frequently.
    interval_writes: usize,
    /// Held-out promote/reject thresholds (Wilson floor, worst-case floor, min games).
    thresholds: PromotionThresholds,
    /// Base seed for held-out fixtures — kept disjoint from the training seed space so
    /// the verdict is on matches the candidate never trained on.
    seed_base: u32,
}

#[derive(Clone, Debug)]
struct AnchorPromotionGateDecision {
    status: &'static str,
    verdict: Option<PromotionVerdict>,
}

fn anchor_promotion_verdict_search_metadata(verdict: &PromotionVerdict) -> serde_json::Value {
    let worst_case = verdict.worst_case.map(|(opponent_id, payoff)| {
        serde_json::json!({
            "opponentId": opponent_id,
            "payoff": payoff
        })
    });
    serde_json::json!({
        "promote": verdict.promote,
        "candidateElo": verdict.candidate_elo,
        "baselineElo": verdict.baseline_elo,
        "eloDelta": verdict.elo_delta,
        "meanPayoffVsField": verdict.mean_payoff_vs_field,
        "payoffVsBaseline": verdict.payoff_vs_baseline,
        "worstCase": worst_case,
        "wilsonLowerBound": verdict.wilson_lower_bound,
        "record": {
            "wins": verdict.record.wins,
            "draws": verdict.record.draws,
            "losses": verdict.record.losses,
            "goalsFor": verdict.record.goals_for,
            "goalsAgainst": verdict.record.goals_against,
            "goalDifference": verdict.record.goal_difference(),
            "games": verdict.record.games(),
            "meanScore": verdict.record.mean_score()
        },
        "reasons": verdict.reasons
    })
}

fn anchor_promotion_gate_search_metadata(
    decision: &AnchorPromotionGateDecision,
    due: bool,
    had_baseline: bool,
) -> serde_json::Value {
    serde_json::json!({
        "due": due,
        "hadBaseline": had_baseline,
        "evaluated": decision.verdict.is_some(),
        "status": decision.status,
        "verdict": decision
            .verdict
            .as_ref()
            .map(anchor_promotion_verdict_search_metadata)
    })
}

/// Play `cfg.games` held-out FROZEN fixtures (neither side learns) between the
/// candidate (id 0) and the anchor (id 1), alternating home/away, and fold them into
/// a promotion verdict. Returns `None` when no comparison is possible — either brain
/// lacks a neural snapshot, or every fixture errored — in which case the caller must
/// NOT block promotion (absence of evidence is not evidence of regression).
fn anchor_promotion_gate_verdict(
    runner: &mut EngineMatchRunner,
    candidate_neural: Option<&SoccerNeuralNetworkSnapshot>,
    anchor_neural: Option<&SoccerNeuralNetworkSnapshot>,
    cfg: &AnchorPromotionGateConfig,
    seed_salt: u32,
) -> Option<PromotionVerdict> {
    let (candidate_neural, anchor_neural) = match (candidate_neural, anchor_neural) {
        (Some(candidate), Some(anchor)) => (candidate, anchor),
        _ => return None,
    };
    let candidate = TeamBrain::from_snapshot(candidate_neural.clone());
    let anchor = TeamBrain::from_snapshot(anchor_neural.clone());
    let candidate_id = 0usize;
    let anchor_id = 1usize;
    let games = cfg.games.max(2);
    let mut reports = Vec::with_capacity(games);
    for g in 0..games {
        let seed = cfg
            .seed_base
            .wrapping_add(seed_salt.wrapping_mul(2_246_822_519))
            .wrapping_add(g as u32);
        // Alternate orientation so the candidate's edge isn't a home-field artifact.
        let (home_id, away_id, home, away) = if g % 2 == 0 {
            (candidate_id, anchor_id, &candidate, &anchor)
        } else {
            (anchor_id, candidate_id, &anchor, &candidate)
        };
        let ctx = TournamentMatchContext {
            stage: TournamentStage::Group,
            round_index: 0,
            match_index: g,
            seed,
            home_id,
            away_id,
            home_name: format!("team{home_id}"),
            away_name: format!("team{away_id}"),
            home_learns: false,
            away_learns: false,
        };
        match runner.play(&ctx, home, away) {
            Ok(outcome) => reports.push(MatchReport {
                stage: ctx.stage,
                home_id,
                away_id,
                home_name: ctx.home_name,
                away_name: ctx.away_name,
                home_goals: outcome.home_goals,
                away_goals: outcome.away_goals,
                shootout_winner: None,
                home_training_steps: outcome.home_training_steps,
                away_training_steps: outcome.away_training_steps,
            }),
            Err(e) => eprintln!("anchor_gate fixture error (g {g}): {e}"),
        }
    }
    if reports.is_empty() {
        return None;
    }
    Some(evaluate_promotion(
        &reports,
        candidate_id,
        anchor_id,
        cfg.thresholds,
    ))
}

/// Apply the frozen-anchor gate to a candidate about to be written with `base_status`.
/// Returns the status to actually persist: unchanged when the gate is disabled, not a
/// promotion, not due this write, or cannot compare; downgraded to `archived` when the
/// candidate fails to beat the anchor. On a passing verdict (or the bootstrap write
/// that sets the first anchor) the anchor advances to the candidate — so the anchor
/// can only move forward by being beaten, giving a real monotonic ratchet.
#[allow(clippy::too_many_arguments)]
fn apply_anchor_promotion_gate_with_decision(
    cfg: &AnchorPromotionGateConfig,
    base_status: &'static str,
    should_write: bool,
    candidate_neural: Option<&SoccerNeuralNetworkSnapshot>,
    anchor_neural: &mut Option<SoccerNeuralNetworkSnapshot>,
    runner: &mut Option<EngineMatchRunner>,
    write_index: &mut usize,
    seed_salt: u32,
    context_label: &str,
) -> AnchorPromotionGateDecision {
    if !cfg.enabled || !should_write || base_status != SOCCER_POLICY_STATUS_ACTIVE {
        return AnchorPromotionGateDecision {
            status: base_status,
            verdict: None,
        };
    }
    *write_index = write_index.wrapping_add(1);
    if *write_index % cfg.interval_writes != 0 {
        // Not due this write; keep the base gate's decision (bounded gate cost).
        return AnchorPromotionGateDecision {
            status: base_status,
            verdict: None,
        };
    }
    if anchor_neural.is_none() {
        // Bootstrap: nothing to beat yet — accept and set the first frozen anchor.
        *anchor_neural = candidate_neural.cloned();
        if anchor_neural.is_some() {
            println!("anchor_gate_bootstrap {context_label}");
        }
        return AnchorPromotionGateDecision {
            status: base_status,
            verdict: None,
        };
    }
    let runner = runner.get_or_insert_with(|| {
        let mut runner_config = EngineMatchRunnerConfig::default();
        runner_config.base.duration_seconds = cfg.minutes * 60.0;
        EngineMatchRunner::new(runner_config)
    });
    match anchor_promotion_gate_verdict(
        runner,
        candidate_neural,
        anchor_neural.as_ref(),
        cfg,
        seed_salt,
    ) {
        Some(verdict) if verdict.promote => {
            *anchor_neural = candidate_neural.cloned();
            println!(
                "anchor_gate_promote {context_label} record={}W-{}D-{}L wilson={:.3} elo_delta={:+.1} mean_payoff={:.3}",
                verdict.record.wins,
                verdict.record.draws,
                verdict.record.losses,
                verdict.wilson_lower_bound,
                verdict.elo_delta,
                verdict.mean_payoff_vs_field.unwrap_or(0.0),
            );
            AnchorPromotionGateDecision {
                status: base_status,
                verdict: Some(verdict),
            }
        }
        Some(verdict) => {
            println!(
                "anchor_gate_blocked {context_label} record={}W-{}D-{}L wilson={:.3} elo_delta={:+.1} reasons={}",
                verdict.record.wins,
                verdict.record.draws,
                verdict.record.losses,
                verdict.wilson_lower_bound,
                verdict.elo_delta,
                verdict.reasons.join("|"),
            );
            AnchorPromotionGateDecision {
                status: SOCCER_POLICY_STATUS_ARCHIVED,
                verdict: Some(verdict),
            }
        }
        // No verdict (e.g. neural disabled, so nothing to compare) — never block.
        None => AnchorPromotionGateDecision {
            status: base_status,
            verdict: None,
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_anchor_promotion_gate(
    cfg: &AnchorPromotionGateConfig,
    base_status: &'static str,
    should_write: bool,
    candidate_neural: Option<&SoccerNeuralNetworkSnapshot>,
    anchor_neural: &mut Option<SoccerNeuralNetworkSnapshot>,
    runner: &mut Option<EngineMatchRunner>,
    write_index: &mut usize,
    seed_salt: u32,
    context_label: &str,
) -> &'static str {
    apply_anchor_promotion_gate_with_decision(
        cfg,
        base_status,
        should_write,
        candidate_neural,
        anchor_neural,
        runner,
        write_index,
        seed_salt,
        context_label,
    )
    .status
}

/// In-memory back-four line-depth head, carried + trained across games WITHIN a
/// learner process (parallel_games=1), so it accumulates and the back-four line
/// consumes it live once trained. Resets on pod restart — cross-restart Postgres
/// persistence (round-trip via SoccerNeuralNetworkSnapshot) is the remaining
/// durability step. Untouched unless a line-depth model is enabled.
static CARRIED_LINE_DEPTH_HEAD: std::sync::Mutex<Option<BackFourLineHead>> =
    std::sync::Mutex::new(None);

/// In-memory per-defender individual line head (the MAPPO layer on top of the group
/// line centre), carried + trained across games WITHIN a learner process, mirroring
/// `CARRIED_LINE_DEPTH_HEAD`. Resets on pod restart; untouched unless the individual
/// model is enabled (DD_SOCCER_ENABLE_BACK_FOUR_INDIVIDUAL_MODEL).
static CARRIED_DEFENDER_LINE_HEAD: std::sync::Mutex<Option<DefenderLinePolicyHead>> =
    std::sync::Mutex::new(None);

/// In-memory loose-ball commit head (which player attacks a loose ball), carried +
/// trained across games WITHIN a learner process, mirroring `CARRIED_LINE_DEPTH_HEAD`.
/// Resets on pod restart; untouched unless the commit model is enabled
/// (DD_SOCCER_ENABLE_LOOSE_BALL_COMMIT_MODEL).
static CARRIED_LOOSE_BALL_COMMIT_HEAD: std::sync::Mutex<Option<LooseBallCommitHead>> =
    std::sync::Mutex::new(None);

/// In-memory long-pass run head (which attacker should break forward so a deep carrier can
/// pick them out), carried + trained across games WITHIN a learner process, mirroring
/// `CARRIED_LOOSE_BALL_COMMIT_HEAD`. Resets on pod restart; untouched unless the model is
/// enabled (DD_SOCCER_ENABLE_LEARNED_LONG_PASS_RUN).
static CARRIED_LONG_PASS_RUN_HEAD: std::sync::Mutex<Option<LongPassRunHead>> =
    std::sync::Mutex::new(None);

/// In-memory give-and-go / wall-pass appetite head (which one-two to commit to and when),
/// carried + trained across games WITHIN a learner process, mirroring
/// `CARRIED_LONG_PASS_RUN_HEAD`. Resets on pod restart; untouched unless the model is enabled
/// (DD_SOCCER_ENABLE_LEARNED_GIVE_AND_GO).
static CARRIED_GIVE_AND_GO_HEAD: std::sync::Mutex<Option<GiveAndGoHead>> =
    std::sync::Mutex::new(None);

/// In-memory receive-approach head (how far a receiver steps toward/away from an incoming
/// ball), carried + trained across games WITHIN a learner process, mirroring
/// `CARRIED_LOOSE_BALL_COMMIT_HEAD`. Resets on pod restart; untouched unless the model is
/// enabled (DD_SOCCER_ENABLE_RECEIVE_APPROACH_MODEL).
static CARRIED_RECEIVE_APPROACH_HEAD: std::sync::Mutex<Option<ReceiveApproachHead>> =
    std::sync::Mutex::new(None);

/// In-memory lane-affinity head (whether an out-of-lane player breaks out or is held
/// back into its channel), carried + trained across games WITHIN a learner process,
/// mirroring `CARRIED_RECEIVE_APPROACH_HEAD`. Resets on pod restart; the seam is on by
/// default in prod (seeded by the analytic prior) so the head is consumed live once it
/// crosses `LANE_AFFINITY_HEAD_MIN_TRAINING_STEPS`.
static CARRIED_LANE_AFFINITY_HEAD: std::sync::Mutex<Option<LaneAffinityHead>> =
    std::sync::Mutex::new(None);

/// In-memory goal-side recovery head (how hard a recovering defender collapses onto the
/// ball→goal line vs. holds width), carried + trained across games WITHIN a learner process.
/// Resets on pod restart; consumed live once it crosses
/// `GOAL_SIDE_RECOVERY_HEAD_MIN_TRAINING_STEPS` (seam on by default in prod).
static CARRIED_GOAL_SIDE_RECOVERY_HEAD: std::sync::Mutex<Option<GoalSideRecoveryHead>> =
    std::sync::Mutex::new(None);

/// In-memory winger pinch-appetite head (stay wide vs pinch infield), carried + trained across
/// games WITHIN a learner process. Consumed live once it crosses
/// `WINGER_PINCH_HEAD_MIN_TRAINING_STEPS` (seam on by default in prod).
static CARRIED_WINGER_PINCH_HEAD: std::sync::Mutex<Option<WingerPinchHead>> =
    std::sync::Mutex::new(None);

/// In-memory same-team separation head (desired keep-out radius: spread vs combine), carried +
/// trained across games WITHIN a learner process. Consumed live once it crosses
/// `SEPARATION_FLOOR_HEAD_MIN_TRAINING_STEPS` (seam on by default in prod).
static CARRIED_SEPARATION_FLOOR_HEAD: std::sync::Mutex<Option<SeparationFloorHead>> =
    std::sync::Mutex::new(None);

/// In-memory pass-lane yield head (step out of the lane vs hold), carried + trained across games
/// WITHIN a learner process. Consumed live once it crosses
/// `PASS_LANE_YIELD_HEAD_MIN_TRAINING_STEPS` (only when pass-lane-yield is enabled).
static CARRIED_PASS_LANE_YIELD_HEAD: std::sync::Mutex<Option<PassLaneYieldHead>> =
    std::sync::Mutex::new(None);

/// In-memory head-scan effort head, carried + trained across games WITHIN a learner process.
static CARRIED_HEAD_SCAN_HEAD: std::sync::Mutex<Option<HeadScanHead>> = std::sync::Mutex::new(None);

/// In-memory crash-the-box commit head, carried + trained across games WITHIN a learner process.
static CARRIED_CRASH_BOX_HEAD: std::sync::Mutex<Option<CrashBoxHead>> = std::sync::Mutex::new(None);

/// In-memory off-ball run-selection head, carried + trained across games WITHIN a learner process.
static CARRIED_RUN_PREDICTION_HEAD: std::sync::Mutex<Option<RunPredictionHead>> =
    std::sync::Mutex::new(None);

/// In-memory slip-break commit head, carried + trained across games WITHIN a learner process.
static CARRIED_SLIP_BREAK_HEAD: std::sync::Mutex<Option<SlipBreakHead>> =
    std::sync::Mutex::new(None);

/// In-memory onside-support push head, carried + trained across games WITHIN a learner process.
static CARRIED_ONSIDE_SUPPORT_HEAD: std::sync::Mutex<Option<OnsideSupportHead>> =
    std::sync::Mutex::new(None);

/// In-memory learned pass-completion head, carried + trained across games WITHIN a learner
/// process (seeded once from the Postgres corpus at startup), mirroring
/// `CARRIED_LINE_DEPTH_HEAD`. Installed on each game so the pass-quality assessor consumes it
/// live once trained. Resets on pod restart; only consumed when
/// DD_SOCCER_ENABLE_LEARNED_PASS_COMPLETION is on, but trained whenever pass-outcome samples
/// accrue so the head is warm by the time the gate is flipped.
static CARRIED_PASS_COMPLETION_HEAD: std::sync::Mutex<Option<SoccerPassCompletionHead>> =
    std::sync::Mutex::new(None);

/// In-memory attacking-spacing target head, carried + trained across games WITHIN a
/// learner process, then installed into the next game so off-ball support and the LP
/// spacing floor consume what was learned.
static CARRIED_ATTACK_SPACING_HEAD: std::sync::Mutex<Option<AttackSpacingHead>> =
    std::sync::Mutex::new(None);

/// In-memory off-ball support scorer head, carried + trained across games WITHIN a
/// learner process, then installed into the next game so `open_space_for` can turn
/// support movement predilections into learned candidate choices once warm.
static CARRIED_SUPPORT_SCORER_HEAD: std::sync::Mutex<Option<SupportScorerHead>> =
    std::sync::Mutex::new(None);

/// In-memory shot-trigger value head (when to pull the trigger on a shot), carried +
/// trained across games WITHIN a learner process, then installed into the next game so
/// the shot decision consumes what was learned. Mirrors `CARRIED_LINE_DEPTH_HEAD`.
static CARRIED_SHOT_TRIGGER_HEAD: std::sync::Mutex<Option<ShotTriggerHead>> =
    std::sync::Mutex::new(None);

/// In-memory learned world model carried across local games so neural MCTS/lookahead
/// can plan with accumulated dynamics instead of a fresh per-match model.
static CARRIED_WORLD_MODEL: std::sync::Mutex<Option<SoccerWorldModel>> =
    std::sync::Mutex::new(None);

fn run_game(
    episode: usize,
    config: MatchConfig,
    starting_policies: Arc<SoccerTeamQPolicies>,
    starting_policy_version_id: Option<String>,
    starting_policy_generation: i32,
    initial_neural_network: Option<Arc<SoccerNeuralNetworkSnapshot>>,
    analytic_neural_opponent: bool,
    adversarial_moment_windows: Arc<Vec<SoccerMomentWindow>>,
    print_progress: bool,
    neural_drain_timeout: Duration,
    retain_neural_network_in_game_artifact: bool,
) -> Result<CompletedGame, String> {
    let started = Instant::now();
    let total_ticks = config.total_ticks();
    let episode_seed = config.seed as u64;
    let starting_tactical_learning = config.tactical_learning.clone();
    let progress_interval = (total_ticks / 9).max(1);
    let mut sim =
        SoccerMatch::default_11v11(config).with_team_policies((*starting_policies).clone());
    sim.set_uniform_elite_players();
    if let Some(snapshot) = initial_neural_network.as_ref() {
        // Warm-starting the neural net from a persisted snapshot is best-effort: a
        // snapshot whose input_dim no longer matches the current feature schema
        // (append-only feature additions not covered by SOCCER_NEURAL_LEGACY_FEATURE_DIMS)
        // must NOT abort the whole run — that would freeze the learner indefinitely.
        // Drop the stale snapshot and start the net cold at the current dim. The game
        // still warm-starts the tabular policy + tactical weights, trains a fresh net,
        // and persists a forward-compatible snapshot, so subsequent games warm-start
        // normally (self-healing). Any other snapshot error still aborts.
        let install_result = if analytic_neural_opponent {
            sim.set_team_neural_brain(Team::Home, Some((**snapshot).clone()), false)
                .map(|()| sim.disable_team_neural_brain(Team::Away))
        } else {
            sim.set_neural_network_snapshot((**snapshot).clone())
        };
        match install_result {
            Ok(()) => {
                if let Some(line_depth_snapshot) = snapshot.line_depth_head.as_deref() {
                    match BackFourLineHead::from_snapshot(line_depth_snapshot) {
                        Ok(restored_head) => {
                            let restored_steps = restored_head.training_steps();
                            let mut guard = CARRIED_LINE_DEPTH_HEAD.lock().unwrap();
                            let should_replace = guard
                                .as_ref()
                                .map(|head| head.training_steps() < restored_steps)
                                .unwrap_or(true);
                            if should_replace {
                                *guard = Some(restored_head);
                                eprintln!(
                                    "soccer warm-start: carried line-depth head restored training_steps={restored_steps}"
                                );
                            }
                        }
                        Err(err) => {
                            eprintln!(
                                "soccer warm-start: dropping incompatible line-depth snapshot ({err})"
                            );
                        }
                    }
                }
            }
            Err(err) => {
                if err.contains("input_dim") && err.contains("does not match expected") {
                    eprintln!(
                        "soccer warm-start: dropping incompatible neural snapshot, starting net cold ({err})"
                    );
                } else {
                    return Err(err);
                }
            }
        }
    } else if analytic_neural_opponent {
        sim.set_team_neural_brain(Team::Home, None, false)?;
        sim.disable_team_neural_brain(Team::Away);
    }
    for window in adversarial_moment_windows.iter() {
        sim.remember_adversarial_moment_window(window.clone());
    }
    if sim.config.neural_blend.world_model {
        if let Some(model) = CARRIED_WORLD_MODEL.lock().unwrap().as_ref() {
            let training_steps = model.training_steps();
            sim.set_world_model(model.clone());
            eprintln!(
                "world_model_installed episode={} training_steps={} last_loss={:?} validation_loss={:?}",
                episode + 1,
                training_steps,
                model.last_loss(),
                model.last_validation_loss()
            );
        }
    }
    // Gap 5 step 3b: install the carried line-depth head so the back-four line
    // consumes it live once trained. No-op unless a line-depth model is enabled (the
    // head only accumulates while collection is on).
    if let Some(head) = CARRIED_LINE_DEPTH_HEAD.lock().unwrap().as_ref() {
        sim.set_line_depth_head(head.clone());
    }
    // Install the carried per-defender individual line head so the per-defender push/drop
    // seam consumes it live once trained. No-op unless the individual model is enabled.
    if let Some(head) = CARRIED_DEFENDER_LINE_HEAD.lock().unwrap().as_ref() {
        sim.set_defender_line_head(head.clone());
    }
    // Install the carried loose-ball commit head so the retriever election consumes it
    // live once trained. No-op unless the commit model is enabled.
    if let Some(head) = CARRIED_LOOSE_BALL_COMMIT_HEAD.lock().unwrap().as_ref() {
        sim.set_loose_ball_commit_head(head.clone());
    }
    // Install the carried receive-approach head so `receive_approach_adjusted_target`
    // consumes it live once trained. No-op unless DD_SOCCER_ENABLE_RECEIVE_APPROACH_MODEL
    // is set.
    if let Some(head) = CARRIED_RECEIVE_APPROACH_HEAD.lock().unwrap().as_ref() {
        sim.set_receive_approach_head(head.clone());
    }
    // Install the carried lane-affinity head so the lane clamp seam consumes it live once
    // trained. No-op unless the lane-affinity model is enabled (on by default in prod).
    if let Some(head) = CARRIED_LANE_AFFINITY_HEAD.lock().unwrap().as_ref() {
        sim.set_lane_affinity_head(head.clone());
    }
    // Install the carried goal-side recovery head so the goal-side lateral-pull seam consumes it
    // live once trained. No-op unless the model is enabled (on by default in prod).
    if let Some(head) = CARRIED_GOAL_SIDE_RECOVERY_HEAD.lock().unwrap().as_ref() {
        sim.set_goal_side_recovery_head(head.clone());
    }
    // Install the carried winger pinch-appetite head so the bucket scoring consumes it live once
    // trained. No-op unless the model is enabled (on by default in prod).
    if let Some(head) = CARRIED_WINGER_PINCH_HEAD.lock().unwrap().as_ref() {
        sim.set_winger_pinch_head(head.clone());
    }
    // Install the carried same-team separation head so the MPC keep-out consumes it live once
    // trained. No-op unless the model is enabled (on by default in prod).
    if let Some(head) = CARRIED_SEPARATION_FLOOR_HEAD.lock().unwrap().as_ref() {
        sim.set_separation_floor_head(head.clone());
    }
    // Install the carried pass-lane yield head so the yield seam consumes it live once trained.
    if let Some(head) = CARRIED_PASS_LANE_YIELD_HEAD.lock().unwrap().as_ref() {
        sim.set_pass_lane_yield_head(head.clone());
    }
    // Install the carried head-scan effort head so the head-scan visibility seam consumes it live.
    if let Some(head) = CARRIED_HEAD_SCAN_HEAD.lock().unwrap().as_ref() {
        sim.set_head_scan_head(head.clone());
    }
    // Install the carried crash-the-box commit head so the box-flood seam consumes it live.
    if let Some(head) = CARRIED_CRASH_BOX_HEAD.lock().unwrap().as_ref() {
        sim.set_crash_box_head(head.clone());
    }
    // Install the carried off-ball run-selection head so the open-space run seam consumes it live.
    if let Some(head) = CARRIED_RUN_PREDICTION_HEAD.lock().unwrap().as_ref() {
        sim.set_run_prediction_head(head.clone());
    }
    // Install the carried slip-break commit head so the slip-break opportunity seam consumes it live.
    if let Some(head) = CARRIED_SLIP_BREAK_HEAD.lock().unwrap().as_ref() {
        sim.set_slip_break_head(head.clone());
    }
    // Install the carried onside-support push head so the onside-support clamp consumes it live.
    if let Some(head) = CARRIED_ONSIDE_SUPPORT_HEAD.lock().unwrap().as_ref() {
        sim.set_onside_support_head(head.clone());
    }
    // Install the carried long-pass run head so `backfield_long_pass_run_invite_for` consumes
    // it live once trained. No-op unless DD_SOCCER_ENABLE_LEARNED_LONG_PASS_RUN is set.
    if let Some(head) = CARRIED_LONG_PASS_RUN_HEAD.lock().unwrap().as_ref() {
        sim.set_long_pass_run_head(head.clone());
    }
    // Install the carried give-and-go head so the carrier's wall-pass appetite consumes it live
    // once trained. No-op unless DD_SOCCER_ENABLE_LEARNED_GIVE_AND_GO is set.
    if let Some(head) = CARRIED_GIVE_AND_GO_HEAD.lock().unwrap().as_ref() {
        sim.set_give_and_go_head(head.clone());
    }
    // Install the carried pass-completion head so the pass-quality assessor consumes it live
    // once trained. No-op on completion scoring unless DD_SOCCER_ENABLE_LEARNED_PASS_COMPLETION
    // is on; the head still trains regardless so it is warm when the gate is flipped.
    if let Some(head) = CARRIED_PASS_COMPLETION_HEAD.lock().unwrap().as_ref() {
        sim.set_pass_completion_head(head.clone());
    }
    // Install the carried attacking-spacing target head so off-ball support and the
    // formation LP consume the learned spacing band once trained.
    if let Some(head) = CARRIED_ATTACK_SPACING_HEAD.lock().unwrap().as_ref() {
        sim.set_attack_spacing_head(head.clone());
    }
    // Install the carried support scorer so open-space support destination ranking can
    // consume learned candidate values once warm.
    if let Some(head) = CARRIED_SUPPORT_SCORER_HEAD.lock().unwrap().as_ref() {
        sim.set_support_scorer_head(head.clone());
    }
    // Install the carried shot-trigger head so the shot decision consumes it live once
    // trained. Default-ON model; no-op unless DD_SOCCER_DISABLE_SHOT_TRIGGER_MDP is unset
    // (the head still trains regardless so it stays warm).
    if let Some(head) = CARRIED_SHOT_TRIGGER_HEAD.lock().unwrap().as_ref() {
        sim.set_shot_trigger_head(head.clone());
    }

    for tick_idx in 0..total_ticks {
        sim.run_time_step();
        let completed_ticks = tick_idx + 1;
        if print_progress
            && (completed_ticks == total_ticks || completed_ticks % progress_interval == 0)
        {
            println!(
                "progress_game={} seed={} ticks={}/{}",
                episode + 1,
                episode_seed,
                completed_ticks,
                total_ticks
            );
            let _ = std::io::stdout().flush();
        }
    }

    sim.drain_neural_learning(neural_drain_timeout);
    let policies = sim
        .team_policies()
        .cloned()
        .ok_or_else(|| "soccer learning produced no team policies".to_string())?;
    let config_moments = sim.config_moments();
    let pass_outcome_samples = sim.drain_pass_outcome_samples();
    // Gap 5: train the CARRIED line-depth head on this game's reward-weighted RL
    // corpus so it accumulates across games and the back-four line consumes it live
    // once trained. Empty + skipped unless a line-depth model is enabled
    // (DD_SOCCER_ENABLE_BACK_FOUR_LINE_MODEL / _MIDFIELD_LINE_MODEL). Cross-restart
    // Postgres persistence is the remaining durability step.
    let line_depth_samples = sim.drain_line_depth_samples();
    if !line_depth_samples.is_empty() {
        let mut guard = CARRIED_LINE_DEPTH_HEAD.lock().unwrap();
        let head = guard.get_or_insert_with(|| BackFourLineHead::new(episode_seed as u32));
        let mut final_loss = 0.0;
        for _ in 0..4 {
            final_loss = head.train_reward_weighted(&line_depth_samples, 0.02);
        }
        sim.set_line_depth_head(head.clone());
        eprintln!(
            "line_depth_training samples={} training_steps={} consumed={} final_loss={:.5}",
            line_depth_samples.len(),
            head.training_steps(),
            head.training_steps()
                >= soccer_engine::des::general::soccer::LINE_DEPTH_HEAD_MIN_TRAINING_STEPS,
            final_loss
        );
    }
    // Train the CARRIED per-defender individual line head on this game's reward-weighted RL
    // corpus (the MAPPO shared policy; reward = the defending team's windowed territorial
    // advantage). Empty + skipped unless the individual model is enabled
    // (DD_SOCCER_ENABLE_BACK_FOUR_INDIVIDUAL_MODEL). Cross-restart Postgres persistence is the
    // remaining durability step, mirroring the group line-depth head.
    let defender_line_samples = sim.drain_defender_line_samples();
    if !defender_line_samples.is_empty() {
        let mut guard = CARRIED_DEFENDER_LINE_HEAD.lock().unwrap();
        let head = guard.get_or_insert_with(|| DefenderLinePolicyHead::new(episode_seed as u32));
        let mut final_loss = 0.0;
        for _ in 0..4 {
            final_loss = head.train_reward_weighted(&defender_line_samples, 0.02);
        }
        eprintln!(
            "defender_line_training samples={} training_steps={} consumed={} final_loss={:.5}",
            defender_line_samples.len(),
            head.training_steps(),
            head.training_steps()
                >= soccer_engine::des::general::soccer::DEFENDER_LINE_HEAD_MIN_TRAINING_STEPS,
            final_loss
        );
    }
    // Train the CARRIED loose-ball commit head on this game's reward-weighted RL
    // corpus (which player attacking the loose ball won/kept possession without
    // breaking shape). Empty + skipped unless the commit model is enabled.
    let loose_ball_commit_samples = sim.drain_loose_ball_commit_samples();
    if !loose_ball_commit_samples.is_empty() {
        let mut guard = CARRIED_LOOSE_BALL_COMMIT_HEAD.lock().unwrap();
        let head = guard.get_or_insert_with(|| LooseBallCommitHead::new(episode_seed as u32));
        let mut final_loss = 0.0;
        for _ in 0..4 {
            final_loss = head.train_reward_weighted(&loose_ball_commit_samples, 0.02);
        }
        eprintln!(
            "loose_ball_commit_training samples={} training_steps={} consumed={} final_loss={:.5}",
            loose_ball_commit_samples.len(),
            head.training_steps(),
            head.training_steps() >= LOOSE_BALL_COMMIT_HEAD_MIN_TRAINING_STEPS,
            final_loss
        );
    }
    // Train the CARRIED receive-approach head on this game's reward-weighted RL corpus (how
    // far a receiver stepping toward/away from an incoming ball led to securing possession).
    // Empty + skipped unless DD_SOCCER_ENABLE_RECEIVE_APPROACH_MODEL is set.
    let receive_approach_samples = sim.drain_receive_approach_samples();
    if !receive_approach_samples.is_empty() {
        let mut guard = CARRIED_RECEIVE_APPROACH_HEAD.lock().unwrap();
        let head = guard.get_or_insert_with(|| ReceiveApproachHead::new(episode_seed as u32));
        let mut final_loss = 0.0;
        for _ in 0..4 {
            final_loss = head.train_reward_weighted(&receive_approach_samples, 0.02);
        }
        eprintln!(
            "receive_approach_training samples={} training_steps={} consumed={} final_loss={:.5}",
            receive_approach_samples.len(),
            head.training_steps(),
            head.training_steps() >= RECEIVE_APPROACH_HEAD_MIN_TRAINING_STEPS,
            final_loss
        );
    }
    // Train the CARRIED lane-affinity head on this game's reward-weighted RL corpus
    // (whether breaking out of / holding the lane improved territorial control). Empty +
    // skipped unless the lane-affinity model is enabled (on by default in prod).
    let lane_affinity_samples = sim.drain_lane_affinity_samples();
    if !lane_affinity_samples.is_empty() {
        let mut guard = CARRIED_LANE_AFFINITY_HEAD.lock().unwrap();
        let head = guard.get_or_insert_with(|| LaneAffinityHead::new(episode_seed as u32));
        let mut final_loss = 0.0;
        for _ in 0..4 {
            final_loss = head.train_reward_weighted(&lane_affinity_samples, 0.02);
        }
        eprintln!(
            "lane_affinity_training samples={} training_steps={} consumed={} final_loss={:.5}",
            lane_affinity_samples.len(),
            head.training_steps(),
            head.training_steps() >= LANE_AFFINITY_HEAD_MIN_TRAINING_STEPS,
            final_loss
        );
    }
    // Train the CARRIED goal-side recovery head on this game's reward-weighted RL corpus
    // (whether collapsing onto the line / holding width best protected the goal). Empty +
    // skipped unless the model is enabled (on by default in prod).
    let goal_side_recovery_samples = sim.drain_goal_side_recovery_samples();
    if !goal_side_recovery_samples.is_empty() {
        let mut guard = CARRIED_GOAL_SIDE_RECOVERY_HEAD.lock().unwrap();
        let head = guard.get_or_insert_with(|| GoalSideRecoveryHead::new(episode_seed as u32));
        let mut final_loss = 0.0;
        for _ in 0..4 {
            final_loss = head.train_reward_weighted(&goal_side_recovery_samples, 0.02);
        }
        eprintln!(
            "goal_side_recovery_training samples={} training_steps={} consumed={} final_loss={:.5}",
            goal_side_recovery_samples.len(),
            head.training_steps(),
            head.training_steps() >= GOAL_SIDE_RECOVERY_HEAD_MIN_TRAINING_STEPS,
            final_loss
        );
    }
    // Train the CARRIED winger pinch-appetite head on this game's reward-weighted RL corpus
    // (whether pinching in / holding width led to a rewarded box arrival). Empty + skipped unless
    // the model is enabled (on by default in prod).
    let winger_pinch_samples = sim.drain_winger_pinch_samples();
    if !winger_pinch_samples.is_empty() {
        let mut guard = CARRIED_WINGER_PINCH_HEAD.lock().unwrap();
        let head = guard.get_or_insert_with(|| WingerPinchHead::new(episode_seed as u32));
        let mut final_loss = 0.0;
        for _ in 0..4 {
            final_loss = head.train_reward_weighted(&winger_pinch_samples, 0.02);
        }
        eprintln!(
            "winger_pinch_training samples={} training_steps={} consumed={} final_loss={:.5}",
            winger_pinch_samples.len(),
            head.training_steps(),
            head.training_steps() >= WINGER_PINCH_HEAD_MIN_TRAINING_STEPS,
            final_loss
        );
    }
    // Train the CARRIED same-team separation head on this game's reward-weighted RL corpus
    // (whether spreading / combining improved outcomes). Empty + skipped unless the model is
    // enabled (on by default in prod).
    let separation_floor_samples = sim.drain_separation_floor_samples();
    if !separation_floor_samples.is_empty() {
        let mut guard = CARRIED_SEPARATION_FLOOR_HEAD.lock().unwrap();
        let head = guard.get_or_insert_with(|| SeparationFloorHead::new(episode_seed as u32));
        let mut final_loss = 0.0;
        for _ in 0..4 {
            final_loss = head.train_reward_weighted(&separation_floor_samples, 0.02);
        }
        eprintln!(
            "separation_floor_training samples={} training_steps={} consumed={} final_loss={:.5}",
            separation_floor_samples.len(),
            head.training_steps(),
            head.training_steps() >= SEPARATION_FLOOR_HEAD_MIN_TRAINING_STEPS,
            final_loss
        );
    }
    // Train the CARRIED pass-lane yield head on this game's reward-weighted RL corpus.
    let pass_lane_yield_samples = sim.drain_pass_lane_yield_samples();
    if !pass_lane_yield_samples.is_empty() {
        let mut guard = CARRIED_PASS_LANE_YIELD_HEAD.lock().unwrap();
        let head = guard.get_or_insert_with(|| PassLaneYieldHead::new(episode_seed as u32));
        let mut final_loss = 0.0;
        for _ in 0..4 {
            final_loss = head.train_reward_weighted(&pass_lane_yield_samples, 0.02);
        }
        eprintln!(
            "pass_lane_yield_training samples={} training_steps={} consumed={} final_loss={:.5}",
            pass_lane_yield_samples.len(),
            head.training_steps(),
            head.training_steps() >= PASS_LANE_YIELD_HEAD_MIN_TRAINING_STEPS,
            final_loss
        );
    }
    // Train the CARRIED head-scan effort head on this game's reward-weighted RL corpus.
    let head_scan_samples = sim.drain_head_scan_samples();
    if !head_scan_samples.is_empty() {
        let mut guard = CARRIED_HEAD_SCAN_HEAD.lock().unwrap();
        let head = guard.get_or_insert_with(|| HeadScanHead::new(episode_seed as u32));
        let mut final_loss = 0.0;
        for _ in 0..4 {
            final_loss = head.train_reward_weighted(&head_scan_samples, 0.02);
        }
        eprintln!(
            "head_scan_training samples={} training_steps={} consumed={} final_loss={:.5}",
            head_scan_samples.len(),
            head.training_steps(),
            head.training_steps() >= HEAD_SCAN_HEAD_MIN_TRAINING_STEPS,
            final_loss
        );
    }
    // Train the CARRIED crash-the-box commit head on this game's reward-weighted RL corpus.
    let crash_box_samples = sim.drain_crash_box_samples();
    if !crash_box_samples.is_empty() {
        let mut guard = CARRIED_CRASH_BOX_HEAD.lock().unwrap();
        let head = guard.get_or_insert_with(|| CrashBoxHead::new(episode_seed as u32));
        let mut final_loss = 0.0;
        for _ in 0..4 {
            final_loss = head.train_reward_weighted(&crash_box_samples, 0.02);
        }
        eprintln!(
            "crash_box_training samples={} training_steps={} consumed={} final_loss={:.5}",
            crash_box_samples.len(),
            head.training_steps(),
            head.training_steps() >= CRASH_BOX_HEAD_MIN_TRAINING_STEPS,
            final_loss
        );
    }
    // Train the CARRIED off-ball run-selection head on this game's reward-weighted RL corpus.
    let run_prediction_samples = sim.drain_run_prediction_samples();
    if !run_prediction_samples.is_empty() {
        let mut guard = CARRIED_RUN_PREDICTION_HEAD.lock().unwrap();
        let head = guard.get_or_insert_with(|| RunPredictionHead::new(episode_seed as u32));
        let mut final_loss = 0.0;
        for _ in 0..4 {
            final_loss = head.train_reward_weighted(&run_prediction_samples, 0.02);
        }
        eprintln!(
            "run_prediction_training samples={} training_steps={} consumed={} final_loss={:.5}",
            run_prediction_samples.len(),
            head.training_steps(),
            head.training_steps() >= RUN_PREDICTION_HEAD_MIN_TRAINING_STEPS,
            final_loss
        );
    }
    // Train the CARRIED slip-break commit head on this game's reward-weighted RL corpus.
    let slip_break_samples = sim.drain_slip_break_samples();
    if !slip_break_samples.is_empty() {
        let mut guard = CARRIED_SLIP_BREAK_HEAD.lock().unwrap();
        let head = guard.get_or_insert_with(|| SlipBreakHead::new(episode_seed as u32));
        let mut final_loss = 0.0;
        for _ in 0..4 {
            final_loss = head.train_reward_weighted(&slip_break_samples, 0.02);
        }
        eprintln!(
            "slip_break_training samples={} training_steps={} consumed={} final_loss={:.5}",
            slip_break_samples.len(),
            head.training_steps(),
            head.training_steps() >= SLIP_BREAK_HEAD_MIN_TRAINING_STEPS,
            final_loss
        );
    }
    // Train the CARRIED onside-support push head on this game's reward-weighted RL corpus.
    let onside_support_samples = sim.drain_onside_support_samples();
    if !onside_support_samples.is_empty() {
        let mut guard = CARRIED_ONSIDE_SUPPORT_HEAD.lock().unwrap();
        let head = guard.get_or_insert_with(|| OnsideSupportHead::new(episode_seed as u32));
        let mut final_loss = 0.0;
        for _ in 0..4 {
            final_loss = head.train_reward_weighted(&onside_support_samples, 0.02);
        }
        eprintln!(
            "onside_support_training samples={} training_steps={} consumed={} final_loss={:.5}",
            onside_support_samples.len(),
            head.training_steps(),
            head.training_steps() >= ONSIDE_SUPPORT_HEAD_MIN_TRAINING_STEPS,
            final_loss
        );
    }
    // Train the CARRIED long-pass run head on this game's reward-weighted RL corpus (which
    // attacker breaking forward led to the team advancing the ball). Empty + skipped unless
    // DD_SOCCER_ENABLE_LEARNED_LONG_PASS_RUN is set.
    let long_pass_run_samples = sim.drain_long_pass_run_samples();
    if !long_pass_run_samples.is_empty() {
        let mut guard = CARRIED_LONG_PASS_RUN_HEAD.lock().unwrap();
        let head = guard.get_or_insert_with(|| LongPassRunHead::new(episode_seed as u32));
        let mut final_loss = 0.0;
        for _ in 0..4 {
            final_loss = head.train_reward_weighted(&long_pass_run_samples, 0.02);
        }
        eprintln!(
            "long_pass_run_training samples={} training_steps={} consumed={} final_loss={:.5}",
            long_pass_run_samples.len(),
            head.training_steps(),
            head.training_steps() >= LONG_PASS_RUN_HEAD_MIN_TRAINING_STEPS,
            final_loss
        );
    }
    // Train the CARRIED give-and-go head on this game's reward-weighted RL corpus (reward = the
    // attacking team's windowed territorial gain when the carrier committed the one-two) and
    // install it into the next game. Empty + skipped unless DD_SOCCER_ENABLE_LEARNED_GIVE_AND_GO
    // is set.
    let give_and_go_samples = sim.drain_give_and_go_samples();
    if !give_and_go_samples.is_empty() {
        let mut guard = CARRIED_GIVE_AND_GO_HEAD.lock().unwrap();
        let head = guard.get_or_insert_with(|| GiveAndGoHead::new(episode_seed as u32));
        let mut final_loss = 0.0;
        for _ in 0..4 {
            final_loss = head.train_reward_weighted(&give_and_go_samples, 0.02);
        }
        eprintln!(
            "give_and_go_training samples={} training_steps={} consumed={} final_loss={:.5}",
            give_and_go_samples.len(),
            head.training_steps(),
            head.training_steps() >= GIVE_AND_GO_HEAD_MIN_TRAINING_STEPS,
            final_loss
        );
    }
    // Train the CARRIED attacking-spacing head on this game's reward-weighted RL
    // corpus and install it into the next game, so the learner actually changes the
    // served off-ball/LP spacing band once warm. Empty + skipped unless
    // DD_SOCCER_ENABLE_LEARNED_SPACING_TARGET is set.
    let attack_spacing_samples = sim.drain_attack_spacing_samples();
    if !attack_spacing_samples.is_empty() {
        let mut guard = CARRIED_ATTACK_SPACING_HEAD.lock().unwrap();
        let head = guard.get_or_insert_with(|| AttackSpacingHead::new(episode_seed as u32));
        let mut final_loss = 0.0;
        for _ in 0..4 {
            final_loss = head.train(&attack_spacing_samples, 0.02);
        }
        eprintln!(
            "attack_spacing_training samples={} training_steps={} consumed={} final_loss={:.5}",
            attack_spacing_samples.len(),
            head.training_steps(),
            head.training_steps() >= ATTACK_SPACING_HEAD_MIN_TRAINING_STEPS,
            final_loss
        );
    }
    // Train the CARRIED support scorer on this game's off-ball support destination
    // choices and install it into the next game, so lane/spacing/open-space
    // predilections remain biases but the candidate choice becomes learnable once warm.
    let support_move_samples = sim.drain_support_move_samples();
    if !support_move_samples.is_empty() {
        let mut guard = CARRIED_SUPPORT_SCORER_HEAD.lock().unwrap();
        let head = guard.get_or_insert_with(|| SupportScorerHead::new(episode_seed as u32));
        let mut final_loss = 0.0;
        for _ in 0..4 {
            final_loss = head.train(&support_move_samples, 0.02);
        }
        eprintln!(
            "support_scorer_training samples={} training_steps={} consumed={} final_loss={:.5}",
            support_move_samples.len(),
            head.training_steps(),
            head.training_steps() >= SUPPORT_SCORER_HEAD_MIN_TRAINING_STEPS,
            final_loss
        );
    }
    // Train the CARRIED shot-trigger head on this game's reward-weighted RL corpus (the
    // shot-decision state + shoot/held action + windowed reward) and install it into the
    // next game so the shot decision consumes what was learned once warm. Empty + skipped
    // unless the model is enabled (default-ON; off under DD_SOCCER_DISABLE_SHOT_TRIGGER_MDP).
    let shot_trigger_samples = sim.drain_shot_trigger_samples();
    if !shot_trigger_samples.is_empty() {
        let mut guard = CARRIED_SHOT_TRIGGER_HEAD.lock().unwrap();
        let head = guard.get_or_insert_with(|| ShotTriggerHead::new(episode_seed as u32));
        let mut final_loss = 0.0;
        for _ in 0..4 {
            final_loss = head.train_reward_weighted(&shot_trigger_samples, 0.02);
        }
        eprintln!(
            "shot_trigger_training samples={} training_steps={} consumed={} final_loss={:.5}",
            shot_trigger_samples.len(),
            head.training_steps(),
            head.training_steps() >= SHOT_TRIGGER_HEAD_MIN_TRAINING_STEPS,
            final_loss
        );
    }
    // Train the CARRIED pass-completion head on this game's pass-outcome corpus (whole-field
    // config + pass features → completed?), so it accumulates across games and the assessor
    // consumes it live once trained. Always trains when samples accrue (so the head warms
    // independent of the consumption gate); consumption is gated separately.
    if !pass_outcome_samples.is_empty() {
        let mut guard = CARRIED_PASS_COMPLETION_HEAD.lock().unwrap();
        let head = guard.get_or_insert_with(|| SoccerPassCompletionHead::new(episode_seed as u32));
        let mut final_loss = 0.0;
        for _ in 0..4 {
            final_loss = head.train(&pass_outcome_samples, 0.05);
        }
        eprintln!(
            "pass_completion_training samples={} training_steps={} consumed={} final_loss={:.5}",
            pass_outcome_samples.len(),
            head.training_steps(),
            head.training_steps() >= PASS_COMPLETION_HEAD_MIN_TRAINING_STEPS,
            final_loss
        );
    }
    let completed_world_model = sim
        .config
        .neural_blend
        .world_model
        .then(|| sim.world_model_snapshot())
        .flatten();
    if sim.config.neural_blend.world_model {
        let world_model_stats = sim.world_model_stats();
        let planning_stats = sim.planning_validation_stats();
        if let Some(model) = completed_world_model.clone() {
            let model_steps = model.training_steps();
            let mut guard = CARRIED_WORLD_MODEL.lock().unwrap();
            let should_replace = guard
                .as_ref()
                .map(|current| current.training_steps() <= model_steps)
                .unwrap_or(true);
            if should_replace {
                *guard = Some(model);
            }
        }
        eprintln!(
            "world_model_training episode={} enabled={} training_steps={} loss={:?} validation_loss={:?} planning_decisions={} neural_mcts_selection_rate={:.4} neural_mcts_discretized_kick_rate={:.4} neural_mcts_technical_action_rate={:.4} neural_mcts_discretized_kick_candidate_share={:.4} neural_mcts_root_discretized_kick_candidate_share={:.4} neural_mcts_discretized_kick_candidate_set_rate={:.4} neural_mcts_root_discretized_kick_candidate_set_rate={:.4} mpc_replan_rate={:.4} policy_priority_samples={} policy_priority_rate={:.4} policy_priority_mean_weight={:.3} neural_mcts_distillation_samples={} neural_mcts_distillation_rate={:.4} neural_mcts_distillation_mean_weight={:.3} policy_entropy={:.4} learning_transitions_captured={} learning_transitions_trained={} learning_nonzero_reward_transitions={} learning_reward_events={} learning_deferred_reward_drained={} learning_deferred_reward_backlog={}",
            episode + 1,
            world_model_stats.enabled,
            world_model_stats.training_steps,
            world_model_stats.last_loss,
            world_model_stats.last_validation_loss,
            planning_stats.decisions,
            planning_stats.neural_mcts_selection_rate,
            planning_stats.neural_mcts_discretized_kick_selection_rate,
            planning_stats.neural_mcts_technical_action_selection_rate,
            planning_stats.neural_mcts_discretized_kick_candidate_share,
            planning_stats.neural_mcts_root_discretized_kick_candidate_share,
            planning_stats.neural_mcts_discretized_kick_candidate_set_rate,
            planning_stats.neural_mcts_root_discretized_kick_candidate_set_rate,
            planning_stats.learned_mpc_replan_rate,
            planning_stats.policy_priority_samples,
            planning_stats.policy_priority_sample_rate,
            planning_stats.mean_policy_priority_weight,
            planning_stats.neural_mcts_distillation_samples,
            planning_stats.neural_mcts_distillation_rate,
            planning_stats.mean_neural_mcts_distillation_weight,
            planning_stats.mean_policy_entropy,
            sim.stats.learning_transitions_captured,
            sim.stats.learning_transitions_trained,
            sim.stats.learning_nonzero_reward_transitions,
            sim.stats.learning_reward_events,
            sim.stats.learning_deferred_reward_transitions_drained,
            sim.stats.learning_deferred_reward_backlog
        );
    }
    log_learning_action_outcomes(episode, sim.episode_learning_transitions());
    let mut artifact = sim.team_policy_artifact();
    let neural_network = if retain_neural_network_in_game_artifact {
        artifact.learning.neural_network.clone()
    } else {
        artifact.learning.neural_network.take()
    };
    let episode_summary = SoccerSelfPlayEpisodeSummary {
        episode,
        seed: episode_seed,
        summary: artifact.summary.clone(),
        transitions: artifact.learning.total_transitions,
        home_policy_entries: artifact.home_entries.len(),
        home_policy_target_entries: artifact.home_target_entries.len(),
        away_policy_entries: artifact.away_entries.len(),
        away_policy_target_entries: artifact.away_target_entries.len(),
    };

    Ok(CompletedGame {
        episode_summary,
        artifact,
        starting_policies,
        starting_policy_version_id,
        starting_policy_generation,
        starting_tactical_learning,
        policies,
        config_moments,
        pass_outcome_samples,
        neural_network,
        world_model: completed_world_model,
        elapsed_seconds: started.elapsed().as_secs_f64(),
    })
}

#[cfg(test)]
fn tactical_genome_parents_from_completed_games<'a>(
    completed_games: &'a [CompletedGame],
    ranked_parents: &[(usize, f64)],
    elite_count: usize,
) -> Vec<SoccerTacticalLearningGenomeParent<'a>> {
    ranked_parents
        .iter()
        .take(elite_count)
        .map(|(game_index, fitness)| {
            let game = &completed_games[*game_index];
            SoccerTacticalLearningGenomeParent {
                summary: &game.artifact.tactical_summary,
                weights: &game.starting_tactical_learning,
                fitness: *fitness,
            }
        })
        .collect()
}

fn tactical_genome_parents_from_evolution_samples<'a>(
    samples: &'a VecDeque<EvolutionSearchSample>,
    ranked_samples: &[(usize, f64)],
    elite_count: usize,
) -> Vec<SoccerTacticalLearningGenomeParent<'a>> {
    ranked_samples
        .iter()
        .take(elite_count)
        .map(|(sample_index, fitness)| {
            let sample = &samples[*sample_index];
            SoccerTacticalLearningGenomeParent {
                summary: &sample.tactical_summary,
                weights: &sample.starting_tactical_learning,
                fitness: *fitness,
            }
        })
        .collect()
}

fn merge_policy_delta(dst: &mut SoccerQPolicy, before: &SoccerQPolicy, after: &SoccerQPolicy) {
    for (key, after_visits) in &after.visits {
        let before_visits = before.visits.get(key).copied().unwrap_or(0);
        if *after_visits <= before_visits {
            continue;
        }
        let delta_visits = *after_visits - before_visits;
        let after_value = after.q_values.get(key).copied().unwrap_or(0.0);
        let dst_visits = dst.visits.get(key).copied().unwrap_or(0);
        let dst_value = dst.q_values.get(key).copied().unwrap_or(0.0);
        let merged_visits = dst_visits.saturating_add(delta_visits);
        let merged_value = if merged_visits == 0 {
            after_value
        } else {
            (dst_value * f64::from(dst_visits) + after_value * f64::from(delta_visits))
                / f64::from(merged_visits)
        };
        dst.q_values.insert(key.clone(), merged_value);
        dst.visits.insert(key.clone(), merged_visits);
    }

    for (key, after_visits) in &after.target_visits {
        let before_visits = before.target_visits.get(key).copied().unwrap_or(0);
        if *after_visits <= before_visits {
            continue;
        }
        let delta_visits = *after_visits - before_visits;
        let after_value = after.target_values.get(key).copied().unwrap_or(0.0);
        let dst_visits = dst.target_visits.get(key).copied().unwrap_or(0);
        let dst_value = dst.target_values.get(key).copied().unwrap_or(0.0);
        let merged_visits = dst_visits.saturating_add(delta_visits);
        let merged_value = if merged_visits == 0 {
            after_value
        } else {
            (dst_value * f64::from(dst_visits) + after_value * f64::from(delta_visits))
                / f64::from(merged_visits)
        };
        dst.target_values.insert(key.clone(), merged_value);
        dst.target_visits.insert(key.clone(), merged_visits);
    }
}

fn merge_team_policy_delta(
    dst: &mut SoccerTeamQPolicies,
    before: &SoccerTeamQPolicies,
    after: &SoccerTeamQPolicies,
) {
    merge_policy_delta(&mut dst.home, &before.home, &after.home);
    merge_policy_delta(&mut dst.away, &before.away, &after.away);
}

fn soccer_learning_completed_game_from_completed(
    game: &CompletedGame,
) -> SoccerLearningCompletedGame {
    let summary = game.episode_summary.summary.clone();
    let score = soccer_learning_run_score(&summary);
    let delta =
        soccer_policy_delta_entries(game.starting_policies.as_ref(), &game.policies, &score);
    SoccerLearningCompletedGame {
        episode: game.episode_summary.episode,
        seed: game.episode_summary.seed,
        summary,
        episode_summary: game.episode_summary.clone(),
        tactical_summary: game.artifact.tactical_summary.clone(),
        starting_tactical_learning: game.starting_tactical_learning.clone(),
        policies: game.policies.clone(),
        score,
        delta,
        config_moments: game.config_moments.clone(),
        pass_outcome_samples: if postgres_persist_pass_outcome_samples() {
            game.pass_outcome_samples.clone()
        } else {
            Vec::new()
        },
        neural_network: None,
        elapsed_seconds: game.elapsed_seconds,
    }
}

fn flush_postgres_completed_runs(
    store: &mut SoccerLearningPgStore,
    experiment_id: &str,
    runner_id: &str,
    pending_policy_versions: &mut Vec<PendingPostgresPolicyVersion>,
    pending_runs: &mut Vec<PendingPostgresCompletedRun>,
    completed_run_retention_games: usize,
    shard_index: usize,
    shard_count: usize,
) -> Result<usize, Box<dyn Error>> {
    if pending_policy_versions.is_empty() && pending_runs.is_empty() {
        return Ok(0);
    }
    let policy_versions_written = pending_policy_versions.len();
    for policy_version in pending_policy_versions.iter() {
        let latest_active_metadata = if policy_version.status == SOCCER_POLICY_STATUS_ACTIVE {
            store
                .load_latest_active_policy_metadata(experiment_id)
                .map_err(invalid_data)?
        } else {
            None
        };
        let latest_active_fitness_for_guard = if policy_version.comparative_promotion_approved {
            None
        } else {
            latest_active_metadata.as_ref().map(|metadata| {
                soccer_engine::des::soccer_learning::soccer_learning_from_micros(
                    metadata.fitness_micros,
                )
            })
        };
        let insert_status = soccer_policy_version_insert_status_after_active_head(
            policy_version.status,
            policy_version.parent_policy_version_id.as_deref(),
            policy_version.generation,
            policy_version.fitness,
            latest_active_metadata
                .as_ref()
                .map(|metadata| metadata.id.as_str()),
            latest_active_metadata
                .as_ref()
                .map(|metadata| metadata.generation),
            latest_active_fitness_for_guard,
            active_max_fitness_regression(),
        );
        if insert_status != policy_version.status {
            println!(
                "postgres_policy_version_marked_stale policy_version={} parent_policy_version={} generation={} latest_active={} latest_generation={}",
                policy_version.id,
                policy_version
                    .parent_policy_version_id
                    .as_deref()
                    .unwrap_or("none"),
                policy_version.generation,
                latest_active_metadata
                    .as_ref()
                    .map(|metadata| metadata.id.as_str())
                    .unwrap_or("none"),
                latest_active_metadata
                    .as_ref()
                    .map(|metadata| metadata.generation.to_string())
                    .unwrap_or_else(|| "none".to_string())
            );
        }
        store
            .insert_policy_version_with_id_and_neural_network_and_search_metadata(
                &policy_version.id,
                experiment_id,
                policy_version.parent_policy_version_id.as_deref(),
                policy_version.generation,
                &policy_version.version_label,
                policy_version.source_kind,
                insert_status,
                &policy_version.config,
                policy_version.home_options.clone(),
                policy_version.away_options.clone(),
                &policy_version.policies,
                policy_version.fitness,
                policy_version.neural_network.as_ref(),
                policy_version.search_metadata.as_ref(),
            )
            .map_err(invalid_data)?;
    }
    if pending_runs.is_empty() {
        pending_policy_versions.clear();
        return Ok(0);
    }
    let inserts = pending_runs
        .iter()
        .map(|pending| SoccerLearningPgCompletedRunInsert {
            base_policy_version_id: pending.base_policy_version_id.as_deref(),
            output_policy_version_id: pending.output_policy_version_id.as_deref(),
            game: &pending.game,
        })
        .collect::<Vec<_>>();
    let run_row_ids = store
        .insert_completed_runs(experiment_id, runner_id, &inserts)
        .map_err(invalid_data)?;
    let persisted = run_row_ids.len();
    // Roll this batch's pass-quality metrics into the per-commit learning-progress series.
    // Non-fatal: a metrics write must never fail the run persistence.
    {
        let mut pass_metrics = SoccerPassLearningMetrics::default();
        for pending in pending_runs.iter() {
            pass_metrics.accumulate(&pending.game.summary.stats.pass_learning_metrics());
        }
        if let Err(err) = store.upsert_pass_learning_metrics(
            &soccer_learning_git_commit(),
            persisted as u64,
            &pass_metrics,
        ) {
            eprintln!("soccer-learning: pass-metrics upsert failed (non-fatal): {err}");
        }
    }
    if completed_run_retention_games > 0 {
        let prune = store
            .prune_completed_runs_for_experiment(experiment_id, completed_run_retention_games)
            .map_err(invalid_data)?;
        if prune.deleted_runs > 0 || prune.deleted_delta_rows > 0 {
            println!(
                "postgres_completed_run_retention keep_latest_runs={} deleted_runs={} deleted_delta_rows={}",
                completed_run_retention_games, prune.deleted_runs, prune.deleted_delta_rows
            );
        }
    }
    if let (Some(first), Some(last), Some(first_run_id), Some(last_run_id)) = (
        pending_runs.first(),
        pending_runs.last(),
        run_row_ids.first(),
        run_row_ids.last(),
    ) {
        println!(
            "postgres_persisted_batch episodes={}..{} shard={}/{} first_run_id={} last_run_id={} first_policy_version={} last_policy_version={} first_generation={} last_generation={} policy_versions_written={} batch_size={}",
            first.completed_episode,
            last.completed_episode,
            shard_index,
            shard_count,
            first_run_id,
            last_run_id,
            first
                .output_policy_version_id
                .as_deref()
                .unwrap_or("none"),
            last.output_policy_version_id.as_deref().unwrap_or("none"),
            first.generation,
            last.generation,
            policy_versions_written,
            persisted
        );
    }
    pending_policy_versions.clear();
    pending_runs.clear();
    Ok(persisted)
}

fn flush_postgres_policy_versions_for_new_sims(
    store: &mut SoccerLearningPgStore,
    experiment_id: &str,
    runner_id: &str,
    pending_policy_versions: &mut Vec<PendingPostgresPolicyVersion>,
    shard_index: usize,
    shard_count: usize,
) -> Result<(), Box<dyn Error>> {
    if pending_policy_versions.is_empty() {
        return Ok(());
    }
    let pending_count = pending_policy_versions.len();
    let mut pending_runs = Vec::new();
    flush_postgres_completed_runs(
        store,
        experiment_id,
        runner_id,
        pending_policy_versions,
        &mut pending_runs,
        0,
        shard_index,
        shard_count,
    )?;
    println!(
        "postgres_policy_versions_flushed_for_new_sims shard={}/{} policy_versions_written={pending_count}",
        shard_index,
        shard_count
    );
    Ok(())
}

const SOCCER_PG_VERSION_LABEL_MAX_LEN: usize = 160;

fn soccer_learning_pg_version_label(run_id: &str, shard_index: usize, episode: usize) -> String {
    soccer_learning_pg_version_label_with_suffix(run_id, shard_index, episode, "")
}

fn soccer_learning_pg_version_label_with_suffix(
    run_id: &str,
    shard_index: usize,
    episode: usize,
    extra_suffix: &str,
) -> String {
    let suffix = format!("-s{:03}-e{:06}", shard_index, episode + 1);
    let mut extra_suffix = extra_suffix
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | ':' | '/' | '-'))
        .collect::<String>();
    if !extra_suffix.is_empty() && !extra_suffix.starts_with('-') {
        extra_suffix.insert(0, '-');
    }
    let max_extra_suffix_len = SOCCER_PG_VERSION_LABEL_MAX_LEN
        .saturating_sub(suffix.len())
        .saturating_sub("run".len());
    if extra_suffix.len() > max_extra_suffix_len {
        extra_suffix.truncate(max_extra_suffix_len);
    }
    let max_prefix_len =
        SOCCER_PG_VERSION_LABEL_MAX_LEN.saturating_sub(suffix.len() + extra_suffix.len());
    let mut prefix = run_id
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | ':' | '/' | '-'))
        .take(max_prefix_len)
        .collect::<String>();
    if prefix.is_empty() {
        prefix.push_str(&"run"[.."run".len().min(max_prefix_len)]);
    }
    format!("{prefix}{suffix}{extra_suffix}")
}

fn soccer_learning_pg_runner_id(run_id: &str, shard_index: usize, shard_count: usize) -> String {
    let suffix = format!("-shard-{}-of-{}", shard_index, shard_count);
    let max_prefix_len = 200usize.saturating_sub(suffix.len());
    let mut prefix = run_id.chars().take(max_prefix_len).collect::<String>();
    if prefix.is_empty() {
        prefix.push_str("soccer-self-play");
    }
    format!("{prefix}{suffix}")
}

fn print_completed_game(game: &CompletedGame) {
    let stats = &game.episode_summary.summary.stats;
    println!(
        "completed_game={} seed={} score={}-{} shots={} on_target={} blocks={} pass_completion={}/{} interceptions={} elapsed={:.2}s",
        game.episode_summary.episode + 1,
        game.episode_summary.seed,
        game.episode_summary.summary.score_home,
        game.episode_summary.summary.score_away,
        stats.shots_home + stats.shots_away,
        stats.shots_on_target_home + stats.shots_on_target_away,
        stats.shot_blocks_home + stats.shot_blocks_away,
        stats.passes_completed_home + stats.passes_completed_away,
        stats.passes_attempted_home + stats.passes_attempted_away,
        stats.interceptions_home + stats.interceptions_away,
        game.elapsed_seconds,
    );
    let _ = std::io::stdout().flush();
}

fn game_manifest_entry(game: &CompletedGame, artifact_path: Option<String>) -> GameManifestEntry {
    GameManifestEntry {
        episode: game.episode_summary.episode,
        seed: game.episode_summary.seed,
        artifact_path,
        artifact_kind: None,
        score_home: game.episode_summary.summary.score_home,
        score_away: game.episode_summary.summary.score_away,
        ticks: game.episode_summary.summary.ticks,
        simulated_seconds: game.episode_summary.summary.simulated_seconds,
        transitions: game.episode_summary.transitions,
        home_policy_entries: game.episode_summary.home_policy_entries,
        away_policy_entries: game.episode_summary.away_policy_entries,
        home_policy_target_entries: game.episode_summary.home_policy_target_entries,
        away_policy_target_entries: game.episode_summary.away_policy_target_entries,
        elapsed_seconds: game.elapsed_seconds,
    }
}

fn action_summary_entries(entries: &[SoccerQEntry]) -> Vec<ActionSummaryEntry> {
    action_summary(entries)
        .into_iter()
        .take(12)
        .map(|(action, visits, mean_q)| ActionSummaryEntry {
            action,
            visits,
            mean_q,
        })
        .collect()
}

fn compact_game_artifact(game: &CompletedGame) -> CompactGameArtifact {
    CompactGameArtifact {
        artifact_kind: "soccer-compact-game-summary".to_string(),
        episode: game.episode_summary.episode,
        seed: game.episode_summary.seed,
        config: game.artifact.config.clone(),
        summary: game.episode_summary.summary.clone(),
        tactical_summary: game.artifact.tactical_summary.clone(),
        transitions: game.episode_summary.transitions,
        home_policy_entries: game.episode_summary.home_policy_entries,
        away_policy_entries: game.episode_summary.away_policy_entries,
        home_policy_target_entries: game.episode_summary.home_policy_target_entries,
        away_policy_target_entries: game.episode_summary.away_policy_target_entries,
        elapsed_seconds: game.elapsed_seconds,
        home_action_summary: action_summary_entries(&game.artifact.home_entries),
        away_action_summary: action_summary_entries(&game.artifact.away_entries),
    }
}

fn self_play_artifact_from_policies(
    config: MatchConfig,
    options: SoccerQPolicyOptions,
    tactical_summary: SoccerTacticalLearningSummary,
    episode_summaries: Vec<SoccerSelfPlayEpisodeSummary>,
    policies: &SoccerTeamQPolicies,
) -> SoccerSelfPlayTrainingArtifact {
    SoccerSelfPlayTrainingArtifact {
        tactical_learning: config.tactical_learning.clone(),
        tactical_summary,
        config,
        options,
        episodes: episode_summaries,
        home_entries: policies.home.entries(),
        home_target_entries: policies.home.target_entries(),
        away_entries: policies.away.entries(),
        away_target_entries: policies.away.target_entries(),
    }
}

fn run_manifest(
    run_id: &str,
    run_dir: &Path,
    game_dir: &Path,
    games: usize,
    parallel_games: usize,
    shard_index: usize,
    shard_count: usize,
    base_seed: u32,
    effective_seed: u32,
    config: MatchConfig,
    options: SoccerQPolicyOptions,
    final_artifact_path: &Path,
    learned_params_path: &Path,
    checkpoint_artifact_path: &Path,
    episode_log_path: &Path,
    checkpoint_interval_games: usize,
    artifact_max_entries_per_policy: usize,
    max_policy_entries_per_team: usize,
    max_policy_target_entries_per_team: usize,
    min_policy_visits: u32,
    moment_replay_path: Option<String>,
    moment_replay_records: usize,
    moment_replay_transitions: usize,
    moment_replay_passes: usize,
    moment_replay_reward_scale: f64,
    postgres_enabled: bool,
    postgres_experiment_slug: Option<String>,
    postgres_experiment_id: Option<String>,
    postgres_last_policy_version_id: Option<String>,
    postgres_persisted_games: usize,
    postgres_completed_run_batch_games: usize,
    write_final_policy_artifact: bool,
    write_game_artifacts: bool,
    game_artifact_mode: &str,
    game_artifacts: Vec<GameManifestEntry>,
) -> RunManifest {
    RunManifest {
        run_id: run_id.to_string(),
        run_dir: run_dir.display().to_string(),
        game_dir: game_dir.display().to_string(),
        games,
        parallel_games,
        shard_index,
        shard_count,
        base_seed,
        effective_seed,
        config,
        options,
        final_artifact_path: final_artifact_path.display().to_string(),
        learned_params_path: learned_params_path.display().to_string(),
        checkpoint_artifact_path: checkpoint_artifact_path.display().to_string(),
        episode_log_path: episode_log_path.display().to_string(),
        checkpoint_interval_games,
        artifact_max_entries_per_policy,
        max_policy_entries_per_team,
        max_policy_target_entries_per_team,
        min_policy_visits,
        moment_replay_path,
        moment_replay_records,
        moment_replay_transitions,
        moment_replay_passes,
        moment_replay_reward_scale,
        postgres_enabled,
        postgres_experiment_slug,
        postgres_experiment_id,
        postgres_last_policy_version_id,
        postgres_persisted_games,
        postgres_completed_run_batch_games,
        write_final_policy_artifact,
        write_game_artifacts,
        game_artifact_mode: game_artifact_mode.to_string(),
        game_artifacts,
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let games = env_usize("SOCCER_GAMES", 100)?;
    let halves = env_usize("SOCCER_HALVES", 2)?;
    let half_minutes = env_f64("SOCCER_HALF_MINUTES", 45.0)?;
    let minutes = env_f64("SOCCER_MINUTES", half_minutes * halves as f64)?;
    let effective_minutes = minutes;
    let period_break_recovery_seconds = env_f64_alias(
        "SOCCER_PERIOD_BREAK_RECOVERY_SECONDS",
        "SOCCER_HALFTIME_RECOVERY_SECONDS",
        900.0,
    )?;
    let dt_seconds = env_f64("SOCCER_DT_SECONDS", 0.2)?;
    let learning_interval_ticks = env_usize("SOCCER_LEARNING_INTERVAL_TICKS", 1)?;
    let parallel_games = env_usize("SOCCER_PARALLEL_GAMES", default_soccer_parallel_games())?;
    let checkpoint_interval_games = env_usize("SOCCER_CHECKPOINT_INTERVAL_GAMES", 10)?;
    let artifact_max_entries_per_policy =
        env_usize("SOCCER_ARTIFACT_MAX_ENTRIES_PER_POLICY", 10_000)?;
    let max_policy_entries_per_team = env_usize("SOCCER_MAX_POLICY_ENTRIES_PER_TEAM", 0)?;
    let max_policy_target_entries_per_team = env_usize(
        "SOCCER_MAX_POLICY_TARGET_ENTRIES_PER_TEAM",
        max_policy_entries_per_team,
    )?;
    let min_policy_visits = env_u32("SOCCER_MIN_POLICY_VISITS", 0)?;
    let moment_replay_path = env_value("SOCCER_MOMENT_REPLAY_PATH");
    let moment_replay_limit = env_usize("SOCCER_MOMENT_REPLAY_LIMIT", 0)?;
    let moment_replay_passes = env_usize("SOCCER_MOMENT_REPLAY_PASSES", 1)?;
    let moment_replay_reward_scale = env_f64("SOCCER_MOMENT_REPLAY_REWARD_SCALE", 1.0)?;
    let postgres_store_configured = soccer_learning_database_url_env_configured();
    let postgres_required = env_bool("SOCCER_REQUIRE_POSTGRES", false)?;
    if postgres_required && !postgres_store_configured {
        return Err(
            invalid_data("SOCCER_REQUIRE_POSTGRES=true requires SOCCER_DATABASE_URL").into(),
        );
    }
    let default_disk_learning_artifacts =
        default_disk_learning_artifacts_enabled(postgres_store_configured);
    let write_game_artifacts = env_bool(
        "SOCCER_WRITE_GAME_ARTIFACTS",
        DEFAULT_SOCCER_WRITE_GAME_ARTIFACTS,
    )?;
    let write_final_artifacts = env_bool(
        "SOCCER_WRITE_FINAL_ARTIFACTS",
        default_disk_learning_artifacts,
    )?;
    let write_final_policy_artifact = env_bool(
        "SOCCER_WRITE_FINAL_POLICY_ARTIFACT",
        write_final_artifacts && DEFAULT_SOCCER_WRITE_FINAL_POLICY_ARTIFACT,
    )?;
    let write_checkpoint_artifacts =
        env_bool("SOCCER_WRITE_CHECKPOINT_ARTIFACTS", write_final_artifacts)?;
    let write_learned_params_artifact = env_bool_alias(
        "SOCCER_WRITE_LEARNED_PARAMS_ARTIFACT",
        "SOCCER_WRITE_LEARNED_PARAMS_ARTIFACTS",
        write_final_artifacts,
    )?;
    let checkpoint_artifact_best_only = env_bool("SOCCER_CHECKPOINT_ARTIFACT_BEST_ONLY", false)?;
    let write_episode_log_file = env_bool(
        "SOCCER_WRITE_EPISODE_LOG",
        default_disk_learning_artifacts && DEFAULT_SOCCER_WRITE_EPISODE_LOG,
    )?;
    let print_progress = env_bool("SOCCER_PRINT_PROGRESS", false)?;
    let print_completed_games = env_bool(
        "SOCCER_PRINT_COMPLETED_GAMES",
        DEFAULT_SOCCER_PRINT_COMPLETED_GAMES,
    )?;
    let episode_log_flush_interval_games = env_usize(
        "SOCCER_EPISODE_LOG_FLUSH_INTERVAL_GAMES",
        parallel_games.max(1),
    )?;
    let pg_policy_version_interval_games = env_usize(
        "SOCCER_POSTGRES_POLICY_VERSION_INTERVAL_GAMES",
        default_postgres_policy_version_interval_games(parallel_games),
    )?;
    let pg_completed_run_batch_games = env_usize(
        "SOCCER_POSTGRES_COMPLETED_RUN_BATCH_GAMES",
        default_postgres_completed_run_batch_games(parallel_games),
    )?
    .max(1);
    let pg_completed_run_retention_games = env_usize(
        "SOCCER_POSTGRES_COMPLETED_RUN_RETENTION_GAMES",
        DEFAULT_SOCCER_POSTGRES_COMPLETED_RUN_RETENTION_GAMES,
    )?;
    let pg_completed_run_async_queue_batches = env_usize(
        "SOCCER_POSTGRES_ASYNC_BATCH_QUEUE",
        DEFAULT_SOCCER_POSTGRES_ASYNC_BATCH_QUEUE,
    )?
    .max(1);
    let pg_completed_run_async_coalesce_batches = env_usize(
        "SOCCER_POSTGRES_ASYNC_COALESCE_BATCHES",
        DEFAULT_SOCCER_POSTGRES_ASYNC_COALESCE_BATCHES,
    )?
    .max(1);
    let pg_completed_run_async_coalesce_wait_ms = env_usize(
        "SOCCER_POSTGRES_ASYNC_COALESCE_WAIT_MS",
        DEFAULT_SOCCER_POSTGRES_ASYNC_COALESCE_WAIT_MS,
    )?;
    let pg_tactical_learning_authoritative = env_bool(
        "SOCCER_POSTGRES_TACTICAL_LEARNING_AUTHORITATIVE",
        DEFAULT_SOCCER_POSTGRES_TACTICAL_LEARNING_AUTHORITATIVE,
    )?;
    let pg_refresh_with_resume_artifact = env_bool(
        "SOCCER_POSTGRES_REFRESH_WITH_RESUME_ARTIFACT",
        DEFAULT_SOCCER_POSTGRES_REFRESH_WITH_RESUME_ARTIFACT,
    )?;
    let pg_flush_policy_versions_before_new_sim = env_bool(
        "SOCCER_POSTGRES_FLUSH_POLICY_VERSIONS_BEFORE_NEW_SIM",
        DEFAULT_SOCCER_POSTGRES_FLUSH_POLICY_VERSIONS_BEFORE_NEW_SIM,
    )?;
    let pg_training_replay_delta_rows = env_usize(
        "SOCCER_POSTGRES_TRAINING_REPLAY_DELTA_ROWS",
        DEFAULT_SOCCER_POSTGRES_TRAINING_REPLAY_DELTA_ROWS,
    )?;
    let pg_resume_include_unpromoted = resume_include_unpromoted();
    let pg_training_replay_prior_weight = env_f64(
        "SOCCER_POSTGRES_TRAINING_REPLAY_PRIOR_WEIGHT",
        DEFAULT_SOCCER_POSTGRES_TRAINING_REPLAY_PRIOR_WEIGHT,
    )?;
    if pg_training_replay_prior_weight < 0.0 {
        return Err(invalid_data(
            "SOCCER_POSTGRES_TRAINING_REPLAY_PRIOR_WEIGHT must be non-negative",
        )
        .into());
    }
    let pg_completed_run_async_coalesce_wait = Duration::from_millis(
        pg_completed_run_async_coalesce_wait_ms.min(u64::MAX as usize) as u64,
    );
    let evolution_enabled = env_bool("SOCCER_EVOLUTION_ENABLED", DEFAULT_SOCCER_EVOLUTION_ENABLED)?;
    let evolution_interval_games = env_usize(
        "SOCCER_EVOLUTION_INTERVAL_GAMES",
        default_evolution_interval_games(parallel_games),
    )?
    .max(1);
    let evolution_elite_games = env_usize(
        "SOCCER_EVOLUTION_ELITE_GAMES",
        DEFAULT_SOCCER_EVOLUTION_ELITE_GAMES,
    )?
    .max(1);
    let evolution_window_games = env_usize(
        "SOCCER_EVOLUTION_WINDOW_GAMES",
        default_evolution_window_games(evolution_interval_games, evolution_elite_games),
    )?
    .max(evolution_elite_games)
    .max(1);
    let default_evolution_options = SoccerEvolutionOptions::default();
    let mut evolution_options = SoccerEvolutionOptions {
        mutation_rate: env_f64(
            "SOCCER_EVOLUTION_MUTATION_RATE",
            default_evolution_options.mutation_rate,
        )?,
        mutation_scale: env_f64(
            "SOCCER_EVOLUTION_MUTATION_SCALE",
            default_evolution_options.mutation_scale,
        )?,
        crossover_rate: env_f64(
            "SOCCER_EVOLUTION_CROSSOVER_RATE",
            default_evolution_options.crossover_rate,
        )?,
        exploration_rate: env_f64(
            "SOCCER_EVOLUTION_EXPLORATION_RATE",
            default_evolution_options.exploration_rate,
        )?,
        exploration_scale: env_f64(
            "SOCCER_EVOLUTION_EXPLORATION_SCALE",
            default_evolution_options.exploration_scale,
        )?,
        elite_weight_floor: env_f64(
            "SOCCER_EVOLUTION_ELITE_WEIGHT_FLOOR",
            default_evolution_options.elite_weight_floor,
        )?,
        population_size: env_usize(
            "SOCCER_EVOLUTION_POPULATION_SIZE",
            default_evolution_options.population_size,
        )?
        .max(1),
        seed: env_u32(
            "SOCCER_EVOLUTION_SEED",
            default_evolution_options.seed as u32,
        )? as u64,
    };
    validate_soccer_evolution_options_for_learning_run(&evolution_options).map_err(invalid_data)?;
    let default_curriculum = SoccerLearningCurriculumConfig::default();
    let curriculum_config = SoccerLearningCurriculumConfig {
        ball_skills_after_games: env_usize_alias(
            "SOCCER_BATCH_CURRICULUM_BALL_SKILLS_AFTER_GAMES",
            "SOCCER_CURRICULUM_BALL_SKILLS_AFTER_GAMES",
            default_curriculum.ball_skills_after_games,
        )?,
        duels_after_games: env_usize_alias(
            "SOCCER_BATCH_CURRICULUM_DUELS_AFTER_GAMES",
            "SOCCER_CURRICULUM_DUELS_AFTER_GAMES",
            default_curriculum.duels_after_games,
        )?,
        small_sided_after_games: env_usize_alias(
            "SOCCER_BATCH_CURRICULUM_SMALL_SIDED_AFTER_GAMES",
            "SOCCER_CURRICULUM_SMALL_SIDED_AFTER_GAMES",
            default_curriculum.small_sided_after_games,
        )?,
        team_shape_after_games: env_usize_alias(
            "SOCCER_BATCH_CURRICULUM_TEAM_SHAPE_AFTER_GAMES",
            "SOCCER_CURRICULUM_TEAM_SHAPE_AFTER_GAMES",
            default_curriculum.team_shape_after_games,
        )?,
        full_match_after_games: env_usize_alias(
            "SOCCER_BATCH_CURRICULUM_FULL_MATCH_AFTER_GAMES",
            "SOCCER_CURRICULUM_FULL_MATCH_AFTER_GAMES",
            default_curriculum.full_match_after_games,
        )?,
    };
    validate_soccer_learning_curriculum_config_for_learning_run(&curriculum_config)
        .map_err(invalid_data)?;
    let default_promotion_gate = SoccerPolicyPromotionGateConfig::default();
    let policy_promotion_gate = SoccerPolicyPromotionGateConfig {
        enabled: env_bool_alias(
            "SOCCER_BATCH_POLICY_PROMOTION_GATE_ENABLED",
            "SOCCER_POLICY_PROMOTION_GATE_ENABLED",
            default_promotion_gate.enabled,
        )?,
        min_sample_games: env_usize_alias(
            "SOCCER_BATCH_POLICY_PROMOTION_MIN_SAMPLE_GAMES",
            "SOCCER_POLICY_PROMOTION_MIN_SAMPLE_GAMES",
            default_promotion_gate.min_sample_games,
        )?
        .max(1),
        min_mean_match_fitness: env_f64_alias(
            "SOCCER_BATCH_POLICY_PROMOTION_MIN_MEAN_MATCH_FITNESS",
            "SOCCER_POLICY_PROMOTION_MIN_MEAN_MATCH_FITNESS",
            default_promotion_gate.min_mean_match_fitness,
        )?,
        min_best_match_fitness: env_f64_alias(
            "SOCCER_BATCH_POLICY_PROMOTION_MIN_BEST_MATCH_FITNESS",
            "SOCCER_POLICY_PROMOTION_MIN_BEST_MATCH_FITNESS",
            default_promotion_gate.min_best_match_fitness,
        )?,
        min_mean_play_quality: env_f64_alias(
            "SOCCER_BATCH_POLICY_PROMOTION_MIN_MEAN_PLAY_QUALITY",
            "SOCCER_POLICY_PROMOTION_MIN_MEAN_PLAY_QUALITY",
            default_promotion_gate.min_mean_play_quality,
        )?,
        max_mean_conceded_goals: env_f64_alias(
            "SOCCER_BATCH_POLICY_PROMOTION_MAX_MEAN_CONCEDED_GOALS",
            "SOCCER_POLICY_PROMOTION_MAX_MEAN_CONCEDED_GOALS",
            default_promotion_gate.max_mean_conceded_goals,
        )?,
        max_mean_goal_margin: env_f64_alias(
            "SOCCER_BATCH_POLICY_PROMOTION_MAX_MEAN_GOAL_MARGIN",
            "SOCCER_POLICY_PROMOTION_MAX_MEAN_GOAL_MARGIN",
            default_promotion_gate.max_mean_goal_margin,
        )?,
        max_mean_chain_net_loss: env_f64_alias(
            "SOCCER_BATCH_POLICY_PROMOTION_MAX_MEAN_CHAIN_NET_LOSS",
            "SOCCER_POLICY_PROMOTION_MAX_MEAN_CHAIN_NET_LOSS",
            default_promotion_gate.max_mean_chain_net_loss,
        )?,
    };
    validate_soccer_policy_promotion_gate_config_for_learning_run(&policy_promotion_gate)
        .map_err(invalid_data)?;
    // Anti-regression ratchet for activation (see `active_max_fitness_regression`):
    // how much fitness a newer generation may lose vs the incumbent and still go
    // `active`. Configmap-tunable so the operator can tighten it (→ 0.0 = "no
    // regression") to stop the tip drifting below earlier peaks, without a rebuild.
    let active_max_fitness_regression_setting = env_f64_alias(
        "SOCCER_BATCH_POLICY_ACTIVE_MAX_FITNESS_REGRESSION",
        "SOCCER_POLICY_ACTIVE_MAX_FITNESS_REGRESSION",
        soccer_policy_active_max_fitness_regression(),
    )?;
    if !active_max_fitness_regression_setting.is_finite()
        || active_max_fitness_regression_setting < 0.0
    {
        return Err(invalid_data(
            "SOCCER_POLICY_ACTIVE_MAX_FITNESS_REGRESSION must be finite and non-negative",
        )
        .into());
    }
    let _ = ACTIVE_MAX_FITNESS_REGRESSION.set(active_max_fitness_regression_setting);
    println!(
        "policy_active_max_fitness_regression value={active_max_fitness_regression_setting:.4} (lower = stricter anti-regression ratchet; 0.0 = newer generation must not regress)"
    );
    // Frozen-anchor promotion gate (DEFAULT-OFF): a candidate must beat the frozen
    // anchor brain over a held-out slate before it is written `active`. See
    // `AnchorPromotionGateConfig` / `anchor_promotion_gate_verdict`.
    let anchor_promotion_gate = {
        let enabled = env_bool_alias(
            "SOCCER_BATCH_ANCHOR_PROMOTION_GATE_ENABLED",
            "SOCCER_ANCHOR_PROMOTION_GATE_ENABLED",
            false,
        )?;
        let games = env_usize("SOCCER_ANCHOR_PROMOTION_GATE_GAMES", 8)?.max(2);
        let minutes = env_f64("SOCCER_ANCHOR_PROMOTION_GATE_MINUTES", 2.0)?;
        if !minutes.is_finite() || minutes <= 0.0 {
            return Err(invalid_data(
                "SOCCER_ANCHOR_PROMOTION_GATE_MINUTES must be finite and positive",
            )
            .into());
        }
        let interval_writes = env_usize("SOCCER_ANCHOR_PROMOTION_GATE_INTERVAL_WRITES", 1)?.max(1);
        let mut thresholds = PromotionThresholds::default();
        // We play exactly `games` fixtures, so require the full slate as evidence.
        thresholds.min_games = games as u32;
        thresholds.wilson_floor = env_f64(
            "SOCCER_ANCHOR_PROMOTION_GATE_WILSON_FLOOR",
            thresholds.wilson_floor,
        )?;
        thresholds.worst_case_floor = env_f64(
            "SOCCER_ANCHOR_PROMOTION_GATE_WORST_CASE_FLOOR",
            thresholds.worst_case_floor,
        )?;
        if !(0.0..=1.0).contains(&thresholds.wilson_floor)
            || !(0.0..=1.0).contains(&thresholds.worst_case_floor)
        {
            return Err(invalid_data(
                "SOCCER_ANCHOR_PROMOTION_GATE_WILSON_FLOOR / _WORST_CASE_FLOOR must be in [0, 1]",
            )
            .into());
        }
        AnchorPromotionGateConfig {
            enabled,
            games,
            minutes,
            interval_writes,
            thresholds,
            // Disjoint from the training seed space (see `effective_seed`).
            seed_base: 0xE7A1_0000,
        }
    };
    println!(
        "anchor_promotion_gate enabled={} games={} minutes={:.2} interval_writes={} wilson_floor={:.3} worst_case_floor={:.3}",
        anchor_promotion_gate.enabled,
        anchor_promotion_gate.games,
        anchor_promotion_gate.minutes,
        anchor_promotion_gate.interval_writes,
        anchor_promotion_gate.thresholds.wilson_floor,
        anchor_promotion_gate.thresholds.worst_case_floor,
    );
    let policy_promotion_baseline_lookback_generations = env_usize(
        "SOCCER_POLICY_PROMOTION_BASELINE_LOOKBACK_GENERATIONS",
        DEFAULT_SOCCER_POLICY_PROMOTION_BASELINE_LOOKBACK_GENERATIONS,
    )?;
    let policy_promotion_compare_incumbent = env_bool_alias(
        "SOCCER_BATCH_POLICY_PROMOTION_COMPARE_INCUMBENT",
        "SOCCER_POLICY_PROMOTION_COMPARE_INCUMBENT",
        false,
    )?;
    let policy_promotion_min_mean_fitness_delta = env_f64_alias(
        "SOCCER_BATCH_POLICY_PROMOTION_MIN_MEAN_FITNESS_DELTA",
        "SOCCER_POLICY_PROMOTION_MIN_MEAN_FITNESS_DELTA",
        0.0,
    )?;
    if policy_promotion_min_mean_fitness_delta < 0.0 {
        return Err(invalid_data(
            "SOCCER_POLICY_PROMOTION_MIN_MEAN_FITNESS_DELTA must be non-negative",
        )
        .into());
    }
    let policy_promotion_max_mean_fitness_regression = env_f64_alias(
        "SOCCER_BATCH_POLICY_PROMOTION_MAX_MEAN_FITNESS_REGRESSION",
        "SOCCER_POLICY_PROMOTION_MAX_MEAN_FITNESS_REGRESSION",
        0.0,
    )?;
    if policy_promotion_max_mean_fitness_regression < 0.0 {
        return Err(invalid_data(
            "SOCCER_POLICY_PROMOTION_MAX_MEAN_FITNESS_REGRESSION must be non-negative",
        )
        .into());
    }
    let policy_promotion_max_play_quality_regression = env_f64_alias(
        "SOCCER_BATCH_POLICY_PROMOTION_MAX_PLAY_QUALITY_REGRESSION",
        "SOCCER_POLICY_PROMOTION_MAX_PLAY_QUALITY_REGRESSION",
        0.02,
    )?;
    if policy_promotion_max_play_quality_regression < 0.0 {
        return Err(invalid_data(
            "SOCCER_POLICY_PROMOTION_MAX_PLAY_QUALITY_REGRESSION must be non-negative",
        )
        .into());
    }
    let policy_evolution_apply_when_promotion_held = env_bool_alias(
        "SOCCER_BATCH_POLICY_EVOLUTION_APPLY_WHEN_PROMOTION_HELD",
        "SOCCER_POLICY_EVOLUTION_APPLY_WHEN_PROMOTION_HELD",
        false,
    )?;
    let policy_evolution_reset_to_incumbent_after_held_trial = env_bool_alias(
        "SOCCER_BATCH_POLICY_EVOLUTION_RESET_TO_INCUMBENT_AFTER_HELD_TRIAL",
        "SOCCER_POLICY_EVOLUTION_RESET_TO_INCUMBENT_AFTER_HELD_TRIAL",
        policy_evolution_apply_when_promotion_held,
    )?;
    let policy_evolution_require_trial_before_write = env_bool_alias(
        "SOCCER_BATCH_POLICY_EVOLUTION_REQUIRE_TRIAL_BEFORE_WRITE",
        "SOCCER_POLICY_EVOLUTION_REQUIRE_TRIAL_BEFORE_WRITE",
        true,
    )?;
    let policy_write_neural_checkpoint_when_promotion_held = env_bool_alias(
        "SOCCER_BATCH_POLICY_WRITE_NEURAL_CHECKPOINT_WHEN_PROMOTION_HELD",
        "SOCCER_POLICY_WRITE_NEURAL_CHECKPOINT_WHEN_PROMOTION_HELD",
        true,
    )?;
    let policy_resume_latest_neural_checkpoint = env_bool_alias(
        "SOCCER_BATCH_RESUME_LATEST_NEURAL_CHECKPOINT",
        "SOCCER_RESUME_LATEST_NEURAL_CHECKPOINT",
        false,
    )?;
    let policy_resume_strongest_promotion_baseline = env_bool_alias(
        "SOCCER_BATCH_RESUME_STRONGEST_PROMOTION_BASELINE",
        "SOCCER_RESUME_STRONGEST_PROMOTION_BASELINE",
        true,
    )?;
    let policy_promotion_recalibrate_incumbent_on_resume = env_bool_alias(
        "SOCCER_BATCH_POLICY_PROMOTION_RECALIBRATE_INCUMBENT_ON_RESUME",
        "SOCCER_POLICY_PROMOTION_RECALIBRATE_INCUMBENT_ON_RESUME",
        false,
    )?;
    let policy_promotion_seed_local_best_from_resume = env_bool(
        "SOCCER_POLICY_PROMOTION_SEED_LOCAL_BEST_ROLLBACK_FROM_RESUME",
        true,
    )?;
    let policy_promotion_local_best_rollback =
        env_bool("SOCCER_POLICY_PROMOTION_LOCAL_BEST_ROLLBACK", false)?;
    let policy_promotion_local_best_backoff =
        env_bool("SOCCER_POLICY_PROMOTION_LOCAL_BEST_BACKOFF", false)?;
    let policy_promotion_local_best_backoff_factor =
        env_f64("SOCCER_POLICY_PROMOTION_LOCAL_BEST_BACKOFF_FACTOR", 0.7)?;
    if !(0.0..=1.0).contains(&policy_promotion_local_best_backoff_factor)
        || policy_promotion_local_best_backoff_factor == 0.0
    {
        return Err(invalid_data(
            "SOCCER_POLICY_PROMOTION_LOCAL_BEST_BACKOFF_FACTOR must be in (0, 1]",
        )
        .into());
    }
    let policy_promotion_local_best_backoff_min_learning_rate = env_f64(
        "SOCCER_POLICY_PROMOTION_LOCAL_BEST_BACKOFF_MIN_LEARNING_RATE",
        0.0006,
    )?;
    if policy_promotion_local_best_backoff_min_learning_rate < 0.0 {
        return Err(invalid_data(
            "SOCCER_POLICY_PROMOTION_LOCAL_BEST_BACKOFF_MIN_LEARNING_RATE must be non-negative",
        )
        .into());
    }
    let policy_promotion_local_best_backoff_min_blend_lambda = env_f64(
        "SOCCER_POLICY_PROMOTION_LOCAL_BEST_BACKOFF_MIN_BLEND_LAMBDA",
        0.12,
    )?;
    if policy_promotion_local_best_backoff_min_blend_lambda < 0.0 {
        return Err(invalid_data(
            "SOCCER_POLICY_PROMOTION_LOCAL_BEST_BACKOFF_MIN_BLEND_LAMBDA must be non-negative",
        )
        .into());
    }
    let policy_promotion_local_best_anchor_held =
        env_bool("SOCCER_POLICY_PROMOTION_LOCAL_BEST_ANCHOR_HELD", false)?;
    let policy_promotion_local_best_restore_min_games =
        env_usize("SOCCER_POLICY_PROMOTION_LOCAL_BEST_RESTORE_MIN_GAMES", 0)?;
    let policy_promotion_local_best_restore_max_mean_fitness_regression = env_f64(
        "SOCCER_POLICY_PROMOTION_LOCAL_BEST_RESTORE_MAX_MEAN_FITNESS_REGRESSION",
        0.0,
    )?;
    if !policy_promotion_local_best_restore_max_mean_fitness_regression.is_finite()
        || policy_promotion_local_best_restore_max_mean_fitness_regression < 0.0
    {
        return Err(invalid_data(
            "SOCCER_POLICY_PROMOTION_LOCAL_BEST_RESTORE_MAX_MEAN_FITNESS_REGRESSION must be finite and non-negative",
        )
        .into());
    }
    let policy_promotion_local_best_restore_max_play_quality_regression = env_f64(
        "SOCCER_POLICY_PROMOTION_LOCAL_BEST_RESTORE_MAX_PLAY_QUALITY_REGRESSION",
        0.0,
    )?;
    if !policy_promotion_local_best_restore_max_play_quality_regression.is_finite()
        || policy_promotion_local_best_restore_max_play_quality_regression < 0.0
    {
        return Err(invalid_data(
            "SOCCER_POLICY_PROMOTION_LOCAL_BEST_RESTORE_MAX_PLAY_QUALITY_REGRESSION must be finite and non-negative",
        )
        .into());
    }
    let neural_batch_snapshot_max_fitness_regression =
        env_batch_neural_snapshot_max_fitness_regression()?;
    let neural_batch_snapshot_min_fitness = env_batch_neural_snapshot_min_fitness()?;
    let neural_batch_snapshot_min_batch_mean_fitness =
        env_batch_neural_snapshot_min_batch_mean_fitness()?;
    let neural_batch_snapshot_rescue_min_fitness = env_batch_neural_snapshot_rescue_min_fitness()?;
    let neural_batch_policy_merge_selected_only =
        env_bool("SOCCER_NEURAL_BATCH_POLICY_MERGE_SELECTED_ONLY", false)?;
    let neural_batch_snapshot_carry_unvalidated =
        env_bool("SOCCER_NEURAL_BATCH_SNAPSHOT_CARRY_UNVALIDATED", false)?;
    let neural_batch_snapshot_carry_no_publishable = env_bool(
        "SOCCER_NEURAL_BATCH_SNAPSHOT_CARRY_NO_PUBLISHABLE",
        neural_batch_snapshot_carry_unvalidated,
    )?;
    let neural_drain_timeout_ms = env_usize(
        "SOCCER_NEURAL_DRAIN_TIMEOUT_MS",
        DEFAULT_SOCCER_NEURAL_DRAIN_TIMEOUT_MS,
    )?;
    let neural_drain_timeout =
        Duration::from_millis(neural_drain_timeout_ms.min(u64::MAX as usize) as u64);
    let neural_batch_snapshot_selection_mode = env_batch_neural_snapshot_selection_mode()?;
    let game_artifact_mode = env_value("SOCCER_GAME_ARTIFACT_MODE")
        .unwrap_or_else(|| "summary".to_string())
        .to_ascii_lowercase();
    if !matches!(game_artifact_mode.as_str(), "summary" | "full") {
        return Err(invalid_data("SOCCER_GAME_ARTIFACT_MODE must be summary or full").into());
    }
    if moment_replay_passes == 0 {
        return Err(invalid_data("SOCCER_MOMENT_REPLAY_PASSES must be at least 1").into());
    }
    let learning_logging_enabled = env_bool("SOCCER_LEARNING_LOGGING", false)?;
    let shard_index = env_usize("SOCCER_SHARD_INDEX", 0)?;
    let shard_count = env_usize("SOCCER_SHARD_COUNT", 1)?;
    if shard_count == 0 {
        return Err(invalid_data("SOCCER_SHARD_COUNT must be at least 1").into());
    }
    if shard_index >= shard_count {
        return Err(invalid_data("SOCCER_SHARD_INDEX must be less than SOCCER_SHARD_COUNT").into());
    }
    let seed = env_u32("SOCCER_SEED", 2026)?;
    let shard_seed_stride = env_u32("SOCCER_SHARD_SEED_STRIDE", 1_000_000)?;
    let effective_seed = seed.wrapping_add((shard_index as u32).wrapping_mul(shard_seed_stride));
    let neural_population_search_config =
        env_neural_population_search_config(u64::from(effective_seed), parallel_games)?;
    let neural_population_accept_freeze_games = env_usize(
        "SOCCER_NEURAL_POPULATION_ACCEPT_FREEZE_GAMES",
        DEFAULT_SOCCER_NEURAL_POPULATION_ACCEPT_FREEZE_GAMES,
    )?;
    let neural_batch_snapshot_validate_games = env_usize(
        "SOCCER_NEURAL_BATCH_SNAPSHOT_VALIDATE_GAMES",
        DEFAULT_SOCCER_NEURAL_BATCH_SNAPSHOT_VALIDATE_GAMES,
    )?;
    let neural_batch_snapshot_validate_candidates =
        env_usize("SOCCER_NEURAL_BATCH_SNAPSHOT_VALIDATE_CANDIDATES", 1)?.max(1);
    let neural_batch_snapshot_validate_minutes = env_f64(
        "SOCCER_NEURAL_BATCH_SNAPSHOT_VALIDATE_MINUTES",
        neural_population_search_config.eval_minutes,
    )?;
    if neural_batch_snapshot_validate_minutes <= 0.0 {
        return Err(
            invalid_data("SOCCER_NEURAL_BATCH_SNAPSHOT_VALIDATE_MINUTES must be positive").into(),
        );
    }
    let policy_promotion_seed_local_best_eval_games = env_usize(
        "SOCCER_POLICY_PROMOTION_SEED_LOCAL_BEST_EVAL_GAMES",
        policy_promotion_gate.min_sample_games.max(1),
    )?
    .max(1);
    let policy_promotion_seed_local_best_eval_minutes = env_f64(
        "SOCCER_POLICY_PROMOTION_SEED_LOCAL_BEST_EVAL_MINUTES",
        neural_batch_snapshot_validate_minutes,
    )?;
    if policy_promotion_seed_local_best_eval_minutes <= 0.0 {
        return Err(invalid_data(
            "SOCCER_POLICY_PROMOTION_SEED_LOCAL_BEST_EVAL_MINUTES must be positive",
        )
        .into());
    }
    let neural_batch_snapshot_validate_min_fitness = env_f64(
        "SOCCER_NEURAL_BATCH_SNAPSHOT_VALIDATE_MIN_FITNESS",
        DEFAULT_SOCCER_NEURAL_BATCH_SNAPSHOT_VALIDATE_MIN_FITNESS,
    )?;
    if !neural_batch_snapshot_validate_min_fitness.is_finite() {
        return Err(invalid_data(
            "SOCCER_NEURAL_BATCH_SNAPSHOT_VALIDATE_MIN_FITNESS must be finite",
        )
        .into());
    }
    let neural_batch_snapshot_validate_min_goal_margin = env_f64(
        "SOCCER_NEURAL_BATCH_SNAPSHOT_VALIDATE_MIN_GOAL_MARGIN",
        DEFAULT_SOCCER_NEURAL_BATCH_SNAPSHOT_VALIDATE_MIN_GOAL_MARGIN,
    )?;
    if !neural_batch_snapshot_validate_min_goal_margin.is_finite() {
        return Err(invalid_data(
            "SOCCER_NEURAL_BATCH_SNAPSHOT_VALIDATE_MIN_GOAL_MARGIN must be finite",
        )
        .into());
    }
    let neural_batch_snapshot_carry_min_validation_fitness = env_f64(
        "SOCCER_NEURAL_BATCH_SNAPSHOT_CARRY_MIN_VALIDATION_FITNESS",
        DEFAULT_SOCCER_NEURAL_BATCH_SNAPSHOT_CARRY_MIN_VALIDATION_FITNESS,
    )?;
    if !neural_batch_snapshot_carry_min_validation_fitness.is_finite() {
        return Err(invalid_data(
            "SOCCER_NEURAL_BATCH_SNAPSHOT_CARRY_MIN_VALIDATION_FITNESS must be finite",
        )
        .into());
    }
    let neural_batch_snapshot_carry_min_validation_goal_margin = env_f64(
        "SOCCER_NEURAL_BATCH_SNAPSHOT_CARRY_MIN_VALIDATION_GOAL_MARGIN",
        DEFAULT_SOCCER_NEURAL_BATCH_SNAPSHOT_CARRY_MIN_VALIDATION_GOAL_MARGIN,
    )?;
    if !neural_batch_snapshot_carry_min_validation_goal_margin.is_finite() {
        return Err(invalid_data(
            "SOCCER_NEURAL_BATCH_SNAPSHOT_CARRY_MIN_VALIDATION_GOAL_MARGIN must be finite",
        )
        .into());
    }
    let options = SoccerQPolicyOptions {
        alpha: env_f64("SOCCER_ALPHA", 0.20)?,
        gamma: env_f64("SOCCER_GAMMA", 0.96)?,
        exploration_epsilon: env_f64("SOCCER_EXPLORATION_EPSILON", 0.0)?,
    };
    let mut tactical_learning = env_tactical_learning_weights()?;
    validate_run_settings(
        games,
        halves,
        half_minutes,
        minutes,
        period_break_recovery_seconds,
        dt_seconds,
        learning_interval_ticks,
        parallel_games,
        shard_seed_stride,
        &options,
        &tactical_learning,
    )?;
    let half_duration_seconds = half_minutes * 60.0;
    let halftime_fatigue_recovery = if half_duration_seconds > 0.0 {
        (period_break_recovery_seconds / half_duration_seconds).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let neural_learning = env_neural_learning_config()?;
    validate_soccer_neural_learning_config_for_learning_run(&neural_learning)
        .map_err(invalid_data)?;
    let default_config = MatchConfig::default();
    let adversarial_embedding_exploitation_enabled = env_bool_alias(
        "SOCCER_ADVERSARIAL_EMBEDDING_EXPLOITATION_ENABLED",
        "SOCCER_ADVERSARIAL_EMBEDDINGS",
        default_config.adversarial_embedding_exploitation_enabled,
    )?;
    let adversarial_embedding_memory_limit = env_usize_alias(
        "SOCCER_ADVERSARIAL_EMBEDDING_MEMORY_LIMIT",
        "SOCCER_ADVERSARIAL_MOMENT_MEMORY_LIMIT",
        default_config.adversarial_embedding_memory_limit,
    )?;
    let formation_lp_enabled = env_bool(
        "SOCCER_FORMATION_LP_ENABLED",
        default_config.formation_lp_enabled,
    )?;
    let opponent_belief_enabled = env_bool_alias(
        "SOCCER_OPPONENT_BELIEF_ENABLED",
        "SOCCER_OPPONENT_BELIEF",
        true,
    )?;
    let analytic_neural_opponent = env_bool_alias(
        "SOCCER_LEARNING_ANALYTIC_OPPONENT",
        "SOCCER_ANALYTIC_OPPONENT",
        false,
    )?;
    let policy_promotion_seed_local_best_reevaluate_resume = env_bool(
        "SOCCER_POLICY_PROMOTION_SEED_LOCAL_BEST_REEVALUATE_RESUME",
        analytic_neural_opponent,
    )?;
    let mut config = MatchConfig {
        dt_seconds,
        duration_seconds: minutes * 60.0,
        halves: halves as u8,
        half_duration_seconds,
        halftime_fatigue_recovery,
        period_count: halves,
        period_break_recovery_seconds,
        learning_enabled: true,
        learning_logging_enabled,
        learning_interval_ticks,
        formation_lp_enabled,
        tactical_learning: tactical_learning.clone(),
        neural_learning: neural_learning.clone(),
        adversarial_embedding_exploitation_enabled,
        adversarial_embedding_memory_limit,
        opponent_belief_enabled,
        max_human_players: 0,
        seed: effective_seed,
        ..default_config.clone()
    };
    config.neural_blend.actor_critic =
        env_bool_alias("SOCCER_NEURAL_ACTOR_CRITIC", "SOCCER_ACTOR_CRITIC", true)?;
    config.policy_train_max_transitions_per_tick = env_usize(
        "SOCCER_POLICY_TRAIN_MAX_TRANSITIONS_PER_TICK",
        config.policy_train_max_transitions_per_tick,
    )?;
    config.neural_blend.mode = env_neural_blend_mode(config.neural_blend.mode)?;
    config.neural_blend.lambda = env_f64("SOCCER_NEURAL_BLEND_LAMBDA", config.neural_blend.lambda)?;
    config.neural_blend.tie_epsilon = env_f64(
        "SOCCER_NEURAL_BLEND_TIE_EPSILON",
        config.neural_blend.tie_epsilon,
    )?;
    config.neural_blend.min_confidence_visits = env_u32(
        "SOCCER_NEURAL_BLEND_MIN_CONFIDENCE_VISITS",
        config.neural_blend.min_confidence_visits,
    )?;
    config.neural_blend.candidates = env_usize(
        "SOCCER_NEURAL_BLEND_CANDIDATES",
        config.neural_blend.candidates,
    )?;
    config.neural_blend.warmup_steps = env_usize(
        "SOCCER_NEURAL_BLEND_WARMUP_STEPS",
        config.neural_blend.warmup_steps,
    )?;
    config.neural_blend.mcts_enabled = env_bool(
        "SOCCER_NEURAL_MCTS_ENABLED",
        config.neural_blend.mcts_enabled,
    )?;
    config.neural_blend.mcts_simulations = env_usize(
        "SOCCER_NEURAL_MCTS_SIMULATIONS",
        config.neural_blend.mcts_simulations,
    )?;
    config.neural_blend.mcts_candidates = env_usize(
        "SOCCER_NEURAL_MCTS_CANDIDATES",
        config.neural_blend.mcts_candidates,
    )?;
    config.neural_blend.mcts_depth =
        env_usize("SOCCER_NEURAL_MCTS_DEPTH", config.neural_blend.mcts_depth)?;
    config.neural_blend.mcts_model_weight = env_f64(
        "SOCCER_NEURAL_MCTS_MODEL_WEIGHT",
        config.neural_blend.mcts_model_weight,
    )?;
    apply_env_mpc_config(&mut config, &default_config)?;
    // The neural ACTOR (and MAPPO, which is itself gated on `actor_critic`) was previously left
    // disabled in standard learning runs: `neural_blend` came straight from `MatchConfig::default()`
    // where `actor_critic = false`, so a run that asked for Mappo/IndependentActorCritic still only
    // trained the value head. Tie the actor to the requested MARL algorithm — on whenever an actor
    // is wanted (anything but `Off`) — with an explicit `SOCCER_ENABLE_ACTOR_CRITIC` override either way.
    let actor_critic_default = config.neural_learning.marl_algorithm != SoccerMarlAlgorithm::Off;
    config.neural_blend.actor_critic =
        env_bool("SOCCER_ENABLE_ACTOR_CRITIC", actor_critic_default)?;
    // Train the learned dynamics model P̂(s'|s,a) on each episode's replay so MCTS look-ahead can
    // plan inside the model (model-based / "imagined-rollout" learning — more updates per real
    // game). Mirrors the set-play learner's `SOCCER_NEURAL_WORLD_MODEL` hook; default preserves
    // `MatchConfig::default()` (off), so an unset env is byte-identical to before.
    config.neural_blend.world_model =
        env_bool("SOCCER_NEURAL_WORLD_MODEL", config.neural_blend.world_model)?;
    let run_id = env_value("SOCCER_RUN_ID").unwrap_or_else(default_run_id);
    let run_dir = PathBuf::from(
        env_value("SOCCER_RUN_DIR").unwrap_or_else(|| format!("out/soccer-learning-runs/{run_id}")),
    );
    let game_dir = run_dir.join("games");
    let final_artifact_path = env_value("SOCCER_ARTIFACT_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            default_artifact_path_in_run_dir(&run_dir, games, minutes, shard_index, shard_count)
        });
    let learned_params_path = env_value("SOCCER_LEARNED_PARAMS_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| run_dir.join("learned-params.json"));
    let checkpoint_artifact_path = env_value("SOCCER_CHECKPOINT_ARTIFACT_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| run_dir.join("checkpoint-policy.json"));
    let local_policy_promotion_best_path =
        local_policy_promotion_best_path_for_policy_artifact(&checkpoint_artifact_path);
    let training_best_checkpoint_artifact_path = env_value("SOCCER_TRAINING_BEST_ARTIFACT_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| training_best_checkpoint_path(&checkpoint_artifact_path));
    let training_best_learned_params_path = env_value("SOCCER_TRAINING_BEST_LEARNED_PARAMS_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| training_best_checkpoint_path(&learned_params_path));
    let episode_log_path = env_value("SOCCER_EPISODE_LOG_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| run_dir.join("episodes.jsonl"));
    let mut episode_log = if write_episode_log_file {
        if let Some(parent) = episode_log_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let episode_log_file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&episode_log_path)?;
        Some(BufWriter::new(episode_log_file))
    } else {
        None
    };
    let manifest_path = run_dir.join("manifest.json");
    let resume_artifact =
        env_value("SOCCER_RESUME_ARTIFACT").or_else(|| env_value("SOCCER_RESUME_ARTIFACT_PATH"));
    let pg_load_initial_policy_from_postgres = soccer_should_refresh_postgres_for_new_sim(
        resume_artifact.is_some(),
        pg_refresh_with_resume_artifact,
    );
    let pg_refresh_for_new_sims = env_bool(
        "SOCCER_POSTGRES_REFRESH_FOR_NEW_SIMS",
        pg_load_initial_policy_from_postgres,
    )?;
    let mut pg_store = SoccerLearningPgStore::connect_from_env().map_err(invalid_data)?;
    let retrieval_capture_enabled = env_bool_alias(
        "SOCCER_RETRIEVAL_CAPTURE_ENABLED",
        "SOCCER_RETRIEVAL_CAPTURE",
        pg_store.is_some(),
    )?;
    config.retrieval.capture_enabled = retrieval_capture_enabled;
    let mut pg_experiment_slug = None::<String>;
    let mut pg_experiment_id = None::<String>;
    let mut pg_base_policy_version_id = None::<String>;
    let mut pg_last_policy_version_id = None::<String>;
    let mut pg_generation = 0i32;
    let mut pg_base_policy_version_updated_at_micros = 0i64;
    let mut pg_base_policy_fingerprint = None::<u64>;
    let mut pg_base_neural_network_fingerprint = None::<u64>;
    let mut pg_base_tactical_learning_fingerprint = None::<u64>;
    let mut pg_base_policy_promotion_baseline = None::<PolicyPromotionIncumbentBaseline>;
    let mut local_policy_promotion_best_seed_neural_network = None::<SoccerNeuralNetworkSnapshot>;
    let mut pg_persisted_games = 0usize;
    let mut policies = load_initial_policies(resume_artifact.as_deref(), options.clone())?;
    let mut initial_neural_network = None::<SoccerNeuralNetworkSnapshot>;
    if let Some(resume_path) = resume_artifact.as_deref() {
        if let Some((sidecar_path, snapshot)) =
            load_neural_sidecar_for_policy_artifact(Path::new(resume_path))?
        {
            println!(
                "resume_neural_sidecar={} line_depth_head={}",
                sidecar_path.display(),
                snapshot.line_depth_head.is_some()
            );
            pg_base_neural_network_fingerprint =
                Some(soccer_neural_network_snapshot_fingerprint(&snapshot));
            initial_neural_network = Some(snapshot);
        }
    }
    if let Some(store) = pg_store.as_mut() {
        let experiment_slug =
            env_value("SOCCER_EXPERIMENT_SLUG").unwrap_or_else(|| "soccer-self-play".to_string());
        let experiment_name =
            env_value("SOCCER_EXPERIMENT_NAME").unwrap_or_else(|| "Soccer self-play".to_string());
        let experiment_id = store
            .ensure_experiment(&experiment_slug, &experiment_name, &config)
            .map_err(invalid_data)?;
        if pg_load_initial_policy_from_postgres {
            if let Some(mut version) = store
                .load_latest_policy_with_min_visits(
                    &experiment_id,
                    options.clone(),
                    options.clone(),
                    resume_min_visits(),
                    pg_resume_include_unpromoted,
                )
                .map_err(invalid_data)?
            {
                let mut version_neural_network_fingerprint = version
                    .neural_network
                    .as_ref()
                    .map(soccer_neural_network_snapshot_fingerprint);
                let mut version_policy_fingerprint = version
                    .policy_fingerprint
                    .or_else(|| Some(soccer_team_q_policies_fingerprint(&version.policies)));
                let mut version_tactical_learning_fingerprint = version
                    .tactical_learning
                    .as_ref()
                    .map(soccer_tactical_learning_weights_fingerprint);
                let mut version_search_metadata = version.search_metadata.clone();
                let mut version_policy_promotion_baseline =
                    policy_promotion_incumbent_baseline_from_search_metadata(
                        version_search_metadata.as_ref(),
                    );
                let strongest_policy_promotion_baseline =
                    if policy_resume_strongest_promotion_baseline {
                        load_strongest_recent_policy_promotion_baseline_for_run(
                            store,
                            &experiment_id,
                            version.generation,
                            policy_promotion_baseline_lookback_generations,
                            policy_promotion_gate.min_sample_games,
                        )?
                    } else {
                        println!(
                            "postgres_resume_policy_strongest_baseline_skipped policy_version={} generation={} reason=resume-strongest-baseline-disabled",
                            version.id,
                            version.generation
                        );
                        None
                    };
                if let Some((pg_baseline, _)) = strongest_policy_promotion_baseline.as_ref() {
                    if pg_baseline.policy_version_id != version.id {
                        if let Some(strongest_version) = store
                            .load_policy_version_with_min_visits(
                                &pg_baseline.policy_version_id,
                                options.clone(),
                                options.clone(),
                                resume_min_visits(),
                            )
                            .map_err(invalid_data)?
                        {
                            println!(
                                "postgres_resume_policy_switched_to_strongest_baseline latest_policy_version={} latest_generation={} strongest_policy_version={} strongest_generation={} sample_games={} mean_match_fitness={:.4} mean_play_quality={:.4}",
                                version.id,
                                version.generation,
                                pg_baseline.policy_version_id,
                                pg_baseline.generation,
                                pg_baseline.sample_games,
                                pg_baseline.mean_match_fitness,
                                pg_baseline.mean_play_quality
                            );
                            version = strongest_version;
                            version_neural_network_fingerprint = version
                                .neural_network
                                .as_ref()
                                .map(soccer_neural_network_snapshot_fingerprint);
                            version_policy_fingerprint = version.policy_fingerprint.or_else(|| {
                                Some(soccer_team_q_policies_fingerprint(&version.policies))
                            });
                            version_tactical_learning_fingerprint = version
                                .tactical_learning
                                .as_ref()
                                .map(soccer_tactical_learning_weights_fingerprint);
                            version_search_metadata = version.search_metadata.clone();
                            version_policy_promotion_baseline =
                                policy_promotion_incumbent_baseline_from_search_metadata(
                                    version_search_metadata.as_ref(),
                                );
                        }
                    }
                }
                println!(
                    "postgres_resume_policy experiment={} policy_version={} generation={} neural_network={} include_unpromoted={}",
                    experiment_slug,
                    version.id,
                    version.generation,
                    version.neural_network.is_some(),
                    pg_resume_include_unpromoted
                );
                policies = version.policies;
                initial_neural_network = version.neural_network;
                maybe_apply_postgres_tactical_learning(
                    "postgres_resume_tactical_learning",
                    1,
                    &version.id,
                    version.generation,
                    &mut config,
                    &mut tactical_learning,
                    version.tactical_learning,
                )?;
                maybe_apply_postgres_evolution_options(
                    "postgres_resume_evolution_options",
                    1,
                    &version.id,
                    version.generation,
                    &mut evolution_options,
                    version_search_metadata.as_ref(),
                );
                let resumed_policy_version_id = version.id.clone();
                let resumed_generation = version.generation;
                pg_base_policy_version_id = Some(resumed_policy_version_id.clone());
                pg_last_policy_version_id = Some(resumed_policy_version_id.clone());
                pg_generation = resumed_generation;
                pg_base_policy_version_updated_at_micros = version.updated_at_micros;
                pg_base_policy_fingerprint = version_policy_fingerprint;
                pg_base_neural_network_fingerprint = version_neural_network_fingerprint;
                pg_base_tactical_learning_fingerprint = version_tactical_learning_fingerprint;
                pg_base_policy_promotion_baseline = retain_stronger_policy_promotion_baseline(
                    strongest_policy_promotion_baseline
                        .as_ref()
                        .map(|(_, baseline)| *baseline),
                    version_policy_promotion_baseline,
                );
                if policy_resume_latest_neural_checkpoint {
                    let _ = maybe_apply_latest_postgres_neural_snapshot(
                        "postgres_resume_neural_checkpoint",
                        0,
                        store,
                        &experiment_id,
                        &mut initial_neural_network,
                        &mut pg_base_neural_network_fingerprint,
                    )?;
                } else {
                    println!(
                        "postgres_resume_neural_checkpoint_skipped next_episode=0 policy_version={} generation={} reason=active-policy-neural-authoritative",
                        resumed_policy_version_id,
                        resumed_generation
                    );
                }
            }
        }
        if pg_training_replay_delta_rows > 0 {
            let replay_created_after_micros = if pg_base_policy_version_updated_at_micros > 0 {
                Some(pg_base_policy_version_updated_at_micros)
            } else {
                None
            };
            let replay_delta = store
                .load_recent_completed_run_policy_delta(
                    &experiment_id,
                    pg_training_replay_delta_rows,
                    replay_created_after_micros,
                )
                .map_err(invalid_data)?;
            let replay_entries = replay_delta.entries.len();
            if replay_entries > 0 {
                policies = merge_soccer_policy_deltas(
                    &policies,
                    std::slice::from_ref(&replay_delta),
                    pg_training_replay_prior_weight,
                )
                .map_err(invalid_data)?;
                pg_base_policy_fingerprint = Some(soccer_team_q_policies_fingerprint(&policies));
            }
            println!(
                "postgres_training_replay experiment={} entries={} max_delta_rows={} prior_weight={:.3} created_after_micros={}",
                experiment_slug,
                replay_entries,
                pg_training_replay_delta_rows,
                pg_training_replay_prior_weight,
                replay_created_after_micros
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "none".to_string())
            );
        } else {
            println!("postgres_training_replay disabled max_delta_rows=0");
        }
        // Load the persisted pass-completion training corpus (`des_soccer_pass_outcome_samples`)
        // back from Postgres and train the learned pass-completion head on it — the accumulated
        // cross-game experience, not just the current run's samples. Opt-in via sample limit > 0
        // (0 = skip); never fatal — a DB hiccup must not sink a learning run.
        let pass_corpus_limit = env_usize("SOCCER_PG_PASS_COMPLETION_SAMPLE_LIMIT", 0)?;
        if pass_corpus_limit > 0 {
            let epochs = env_usize("SOCCER_PG_PASS_COMPLETION_EPOCHS", 8)?;
            let learning_rate = env_f64(
                "SOCCER_PG_PASS_COMPLETION_LR",
                DEFAULT_SOCCER_PASS_COMPLETION_LEARNING_RATE,
            )?;
            match store.load_pass_outcome_samples(&experiment_id, pass_corpus_limit as i64) {
                Ok(samples) => {
                    // Seed the carried pass-completion head from the accumulated cross-game
                    // corpus (not just this run's samples), and KEEP it so `run_game` installs
                    // it live. Previously the trained head was reported then discarded — the
                    // "trained-but-unconsumed" gap this wiring closes.
                    match train_soccer_pass_completion_head(&samples, seed, epochs, learning_rate) {
                        Some((head, report)) => {
                            println!(
                                "postgres_pass_completion_training experiment={} loaded={} trained_steps={} epochs={} final_loss={:.5} accuracy={:.3}",
                                experiment_slug,
                                report.samples,
                                report.training_steps,
                                report.epochs,
                                report.final_loss,
                                report.accuracy
                            );
                            *CARRIED_PASS_COMPLETION_HEAD.lock().unwrap() = Some(head);
                        }
                        None => println!(
                            "postgres_pass_completion_training experiment={} loaded={} (no usable samples to train)",
                            experiment_slug,
                            samples.len()
                        ),
                    }
                }
                Err(err) => println!(
                    "postgres_pass_completion_training experiment={} load_error={}",
                    experiment_slug, err
                ),
            }
        } else {
            println!("postgres_pass_completion_training disabled sample_limit=0");
        }
        println!(
            "postgres_enabled experiment={} experiment_id={} base_policy_version={} generation={} retrieval_capture={}",
            experiment_slug,
            experiment_id,
            pg_base_policy_version_id.as_deref().unwrap_or("none"),
            pg_generation,
            config.retrieval.capture_enabled
        );
        pg_experiment_slug = Some(experiment_slug);
        pg_experiment_id = Some(experiment_id);
    } else {
        println!("postgres_enabled=false");
    }
    let mut pg_completed_writer = if pg_store.is_some() {
        Some(AsyncPostgresCompletedRunWriter::start(
            pg_completed_run_async_queue_batches,
            pg_completed_run_async_coalesce_batches,
            pg_completed_run_async_coalesce_wait,
        ))
    } else {
        None
    };
    let mut moment_replay_records = 0usize;
    let mut moment_replay_transitions = 0usize;
    let mut adversarial_moment_windows = Vec::new();
    if let Some(path) = moment_replay_path.as_deref() {
        let raw = fs::read_to_string(path)?;
        let mut records = soccer_moment_records_from_jsonl(&raw).map_err(invalid_data)?;
        if moment_replay_limit > 0 && records.len() > moment_replay_limit {
            records = records.split_off(records.len() - moment_replay_limit);
        }
        let replay_dataset = soccer_moment_records_to_learning_dataset(
            &records,
            config.clone(),
            moment_replay_reward_scale,
        )
        .map_err(invalid_data)?;
        moment_replay_records = records.len();
        moment_replay_transitions = replay_dataset.transitions.len();
        adversarial_moment_windows = records
            .iter()
            .map(|record| record.window.clone())
            .collect::<Vec<_>>();
        for _ in 0..moment_replay_passes {
            policies.train_adversarial(&replay_dataset.transitions);
        }
    }
    if checkpoint_artifact_best_only {
        for policy_path in [
            &checkpoint_artifact_path,
            &training_best_checkpoint_artifact_path,
        ] {
            match load_local_policy_promotion_best(policy_path) {
                Ok(Some((metadata_path, metadata))) => {
                    let metadata_objective = metadata.objective.as_deref().unwrap_or("legacy");
                    if analytic_neural_opponent
                        && metadata.objective.as_deref()
                            != Some(LOCAL_POLICY_PROMOTION_OBJECTIVE_HOME_ANALYTIC)
                    {
                        println!(
                            "local_policy_promotion_best_ignored path={} completed_games={} sample_games={} mean_match_fitness={:.4} objective={} expected_objective={} reason=objective-mismatch",
                            metadata_path.display(),
                            metadata.completed_games,
                            metadata.sample_games,
                            metadata.mean_match_fitness,
                            metadata_objective,
                            LOCAL_POLICY_PROMOTION_OBJECTIVE_HOME_ANALYTIC,
                        );
                        fallback_rejected_resume_policy_to_checkpoint_artifact(
                            "metadata_objective_mismatch",
                            resume_artifact.as_deref(),
                            policy_path,
                            &checkpoint_artifact_path,
                            &options,
                            &mut policies,
                            &mut pg_base_policy_fingerprint,
                        )?;
                        continue;
                    }
                    if analytic_neural_opponent
                        && metadata.sample_games < policy_promotion_seed_local_best_eval_games
                    {
                        println!(
                            "local_policy_promotion_best_ignored path={} completed_games={} sample_games={} min_sample_games={} mean_match_fitness={:.4} objective={} reason=sample-games-below-current-local-best-floor",
                            metadata_path.display(),
                            metadata.completed_games,
                            metadata.sample_games,
                            policy_promotion_seed_local_best_eval_games,
                            metadata.mean_match_fitness,
                            metadata_objective,
                        );
                        fallback_rejected_resume_policy_to_checkpoint_artifact(
                            "metadata_sample_games_below_floor",
                            resume_artifact.as_deref(),
                            policy_path,
                            &checkpoint_artifact_path,
                            &options,
                            &mut policies,
                            &mut pg_base_policy_fingerprint,
                        )?;
                        continue;
                    }
                    if analytic_neural_opponent
                        && (!metadata.mean_match_fitness.is_finite()
                            || metadata.mean_match_fitness < 0.0)
                    {
                        println!(
                            "local_policy_promotion_best_ignored path={} completed_games={} sample_games={} mean_match_fitness={:.4} objective={} reason=negative-or-nonfinite-mean-fitness",
                            metadata_path.display(),
                            metadata.completed_games,
                            metadata.sample_games,
                            metadata.mean_match_fitness,
                            metadata_objective,
                        );
                        fallback_rejected_resume_policy_to_checkpoint_artifact(
                            "metadata_negative_or_nonfinite_mean_fitness",
                            resume_artifact.as_deref(),
                            policy_path,
                            &checkpoint_artifact_path,
                            &options,
                            &mut policies,
                            &mut pg_base_policy_fingerprint,
                        )?;
                        continue;
                    }
                    let loaded_baseline =
                        policy_promotion_incumbent_baseline_from_local_best_metadata(&metadata);
                    if let Some(baseline) = loaded_baseline {
                        let retained_baseline = retain_stronger_policy_promotion_baseline(
                            pg_base_policy_promotion_baseline,
                            Some(baseline),
                        );
                        if retained_baseline == Some(baseline) {
                            match load_neural_sidecar_for_policy_artifact(policy_path) {
                                Ok(Some((sidecar_path, snapshot))) => {
                                    println!(
                                        "local_policy_promotion_best_neural_loaded path={} sidecar={} fingerprint={}",
                                        metadata_path.display(),
                                        sidecar_path.display(),
                                        soccer_neural_network_snapshot_fingerprint(&snapshot),
                                    );
                                    local_policy_promotion_best_seed_neural_network =
                                        Some(snapshot);
                                }
                                Ok(None) => {
                                    println!(
                                        "local_policy_promotion_best_neural_missing path={} reason=no-sidecar",
                                        metadata_path.display()
                                    );
                                    local_policy_promotion_best_seed_neural_network = None;
                                }
                                Err(err) => {
                                    eprintln!(
                                        "local_policy_promotion_best_neural_load_error path={} error={}",
                                        neural_sidecar_path_for_policy_artifact(policy_path).display(),
                                        err
                                    );
                                    local_policy_promotion_best_seed_neural_network = None;
                                }
                            }
                        }
                        pg_base_policy_promotion_baseline = retained_baseline;
                        println!(
                            "local_policy_promotion_best_loaded path={} completed_games={} sample_games={} mean_match_fitness={:.4} best_match_fitness={:.4} mean_play_quality={:.4} objective={}",
                            metadata_path.display(),
                            metadata.completed_games,
                            metadata.sample_games,
                            metadata.mean_match_fitness,
                            metadata.best_match_fitness,
                            metadata.mean_play_quality,
                            metadata_objective,
                        );
                    } else {
                        eprintln!(
                            "local_policy_promotion_best_ignored path={} reason=invalid-metadata",
                            metadata_path.display()
                        );
                        fallback_rejected_resume_policy_to_checkpoint_artifact(
                            "metadata_invalid",
                            resume_artifact.as_deref(),
                            policy_path,
                            &checkpoint_artifact_path,
                            &options,
                            &mut policies,
                            &mut pg_base_policy_fingerprint,
                        )?;
                    }
                }
                Ok(None) => {}
                Err(err) => {
                    eprintln!(
                        "local_policy_promotion_best_load_error path={} error={}",
                        local_policy_promotion_best_path_for_policy_artifact(policy_path).display(),
                        err
                    );
                }
            }
        }
    }

    println!(
        "soccer_self_play_start run_id={} games={} parallel_games={} minutes={:.1} halves={} half_minutes={:.1} period_break_recovery_seconds={:.1} dt={:.3}s learning_interval_ticks={} ticks_per_game={} shard={}/{} base_seed={} effective_seed={} logging_transitions={} print_progress={} print_completed_games={} episode_log_flush_interval_games={} pg_policy_version_interval_games={} pg_completed_run_batch_games={} pg_completed_run_retention_games={} pg_completed_async={} pg_completed_async_queue_batches={} pg_completed_async_coalesce_batches={} pg_completed_async_coalesce_wait_ms={} neural_drain_timeout_ms={} game_artifact_mode={} checkpoint_interval_games={} artifact_max_entries_per_policy={} max_policy_entries_per_team={} max_policy_target_entries_per_team={} min_policy_visits={} moment_replay_records={} moment_replay_transitions={} moment_replay_passes={} moment_replay_reward_scale={:.3}",
        run_id,
        games,
        parallel_games,
        effective_minutes,
        halves,
        half_minutes,
        period_break_recovery_seconds,
        dt_seconds,
        learning_interval_ticks,
        config.total_ticks(),
        shard_index,
        shard_count,
        seed,
        effective_seed,
        learning_logging_enabled,
        print_progress,
        print_completed_games,
        episode_log_flush_interval_games,
        pg_policy_version_interval_games,
        pg_completed_run_batch_games,
        pg_completed_run_retention_games,
        pg_completed_writer.is_some(),
        pg_completed_run_async_queue_batches,
        pg_completed_run_async_coalesce_batches,
        pg_completed_run_async_coalesce_wait_ms,
        neural_drain_timeout_ms,
        game_artifact_mode,
        checkpoint_interval_games,
        artifact_max_entries_per_policy,
        max_policy_entries_per_team,
        max_policy_target_entries_per_team,
        min_policy_visits,
        moment_replay_records,
        moment_replay_transitions,
        if moment_replay_path.is_some() { moment_replay_passes } else { 0 },
        moment_replay_reward_scale
    );
    println!("artifact={}", final_artifact_path.display());
    println!("learned_params={}", learned_params_path.display());
    println!("manifest={}", manifest_path.display());
    println!("postgres_store_configured={postgres_store_configured}");
    println!("postgres_required={postgres_required}");
    println!("postgres_tactical_learning_authoritative={pg_tactical_learning_authoritative}");
    println!("postgres_refresh_with_resume_artifact={pg_refresh_with_resume_artifact}");
    println!("postgres_refresh_for_new_sims={pg_refresh_for_new_sims}");
    println!("postgres_resume_min_visits={}", resume_min_visits());
    println!(
        "postgres_flush_policy_versions_before_new_sim={pg_flush_policy_versions_before_new_sim}"
    );
    println!("default_disk_learning_artifacts={default_disk_learning_artifacts}");
    println!(
        "evolution enabled={} interval_games={} window_games={} elite_games={} mutation_rate={:.4} mutation_scale={:.4} crossover_rate={:.4} exploration_rate={:.4} exploration_scale={:.4} elite_weight_floor={:.4} population_size={} seed={}",
        evolution_enabled,
        evolution_interval_games,
        evolution_window_games,
        evolution_elite_games,
        evolution_options.mutation_rate,
        evolution_options.mutation_scale,
        evolution_options.crossover_rate,
        evolution_options.exploration_rate,
        evolution_options.exploration_scale,
        evolution_options.elite_weight_floor,
        evolution_options.population_size,
        evolution_options.seed
    );
    println!(
        "neural_population_search enabled={} interval_games={} population_size={} eval_games={} eval_minutes={:.2} confirm_games={} confirm_min_delta={:.4} confirm_min_accepted_fitness={:.4} confirm_min_accepted_goal_margin={:.4} training_confirm_min_accepted_fitness={:.4} training_confirm_min_accepted_goal_margin={:.4} mutation_rate={:.4} mutation_scale={:.4} crossover_rate={:.4} min_fitness_delta={:.4} min_accepted_fitness={:.4} min_accepted_goal_margin={:.4} training_min_accepted_fitness={:.4} training_min_accepted_goal_margin={:.4} accept_freeze_games={} seed={}",
        neural_population_search_config.enabled,
        neural_population_search_config.interval_games,
        neural_population_search_config.population_size,
        neural_population_search_config.eval_games,
        neural_population_search_config.eval_minutes,
        neural_population_search_config.confirm_games,
        neural_population_search_config.confirm_min_fitness_delta,
        neural_population_search_config.confirm_min_accepted_fitness,
        neural_population_search_config.confirm_min_accepted_goal_margin,
        neural_population_search_config.training_confirm_min_accepted_fitness,
        neural_population_search_config.training_confirm_min_accepted_goal_margin,
        neural_population_search_config.mutation_rate,
        neural_population_search_config.mutation_scale,
        neural_population_search_config.crossover_rate,
        neural_population_search_config.min_fitness_delta,
        neural_population_search_config.min_accepted_fitness,
        neural_population_search_config.min_accepted_goal_margin,
        neural_population_search_config.training_min_accepted_fitness,
        neural_population_search_config.training_min_accepted_goal_margin,
        neural_population_accept_freeze_games,
        neural_population_search_config.seed
    );
    println!(
        "curriculum ball_skills_after_games={} duels_after_games={} small_sided_after_games={} team_shape_after_games={} full_match_after_games={}",
        curriculum_config.ball_skills_after_games,
        curriculum_config.duels_after_games,
        curriculum_config.small_sided_after_games,
        curriculum_config.team_shape_after_games,
        curriculum_config.full_match_after_games
    );
    println!(
        "policy_promotion_gate enabled={} min_sample_games={} min_mean_match_fitness={:.4} min_best_match_fitness={:.4} min_mean_play_quality={:.4} max_mean_conceded_goals={:.4} max_mean_goal_margin={:.4} max_mean_chain_net_loss={:.4} active_max_fitness_regression={:.4} compare_incumbent={} min_mean_fitness_delta={:.4} max_mean_fitness_regression={:.4} max_play_quality_regression={:.4} baseline_lookback_generations={} apply_evolution_when_held={} reset_to_incumbent_after_held_trial={} require_trial_before_write={} neural_checkpoint_when_held={} recalibrate_incumbent_on_resume={} local_best_rollback={} local_best_anchor_held={} local_best_backoff={} local_best_backoff_factor={:.3} local_best_backoff_min_lr={:.5} local_best_backoff_min_blend={:.3} local_best_restore_min_games={} local_best_restore_max_mean_fitness_regression={:.4} local_best_restore_max_play_quality_regression={:.4}",
        policy_promotion_gate.enabled,
        policy_promotion_gate.min_sample_games,
        policy_promotion_gate.min_mean_match_fitness,
        policy_promotion_gate.min_best_match_fitness,
        policy_promotion_gate.min_mean_play_quality,
        policy_promotion_gate.max_mean_conceded_goals,
        policy_promotion_gate.max_mean_goal_margin,
        policy_promotion_gate.max_mean_chain_net_loss,
        soccer_policy_active_max_fitness_regression(),
        policy_promotion_compare_incumbent,
        policy_promotion_min_mean_fitness_delta,
        policy_promotion_max_mean_fitness_regression,
        policy_promotion_max_play_quality_regression,
        policy_promotion_baseline_lookback_generations,
        policy_evolution_apply_when_promotion_held,
        policy_evolution_reset_to_incumbent_after_held_trial,
        policy_evolution_require_trial_before_write,
        policy_write_neural_checkpoint_when_promotion_held,
        policy_promotion_recalibrate_incumbent_on_resume,
        policy_promotion_local_best_rollback,
        policy_promotion_local_best_anchor_held,
        policy_promotion_local_best_backoff,
        policy_promotion_local_best_backoff_factor,
        policy_promotion_local_best_backoff_min_learning_rate,
        policy_promotion_local_best_backoff_min_blend_lambda,
        policy_promotion_local_best_restore_min_games,
        policy_promotion_local_best_restore_max_mean_fitness_regression,
        policy_promotion_local_best_restore_max_play_quality_regression
    );
    println!(
        "learning_opponent analytic_neural_opponent={} (true = home neural trains against away net-less analytic stack)",
        analytic_neural_opponent
    );
    println!(
        "postgres_resumed_neural_network={}",
        initial_neural_network.is_some()
    );
    if write_episode_log_file {
        println!("episode_log={}", episode_log_path.display());
    } else {
        println!("episode_log=disabled");
    }
    println!("write_final_artifacts={write_final_artifacts}");
    println!("write_final_policy_artifact={write_final_policy_artifact}");
    println!("write_checkpoint_artifacts={write_checkpoint_artifacts}");
    println!("write_learned_params_artifact={write_learned_params_artifact}");
    println!("checkpoint_artifact_best_only={checkpoint_artifact_best_only}");
    println!("write_episode_log={write_episode_log_file}");
    if checkpoint_interval_games == 0 || !write_checkpoint_artifacts {
        println!("checkpoint_artifact=disabled");
    } else {
        println!("checkpoint_artifact={}", checkpoint_artifact_path.display());
        println!(
            "checkpoint_training_best_artifact={}",
            training_best_checkpoint_artifact_path.display()
        );
        println!("checkpoint_interval_games={}", checkpoint_interval_games);
    }
    if checkpoint_artifact_best_only {
        println!(
            "checkpoint_local_best_metadata={}",
            local_policy_promotion_best_path.display()
        );
    }
    println!(
        "policy_promotion_local_best_resume_seed enabled={} reevaluate_resume={} eval_games={} eval_minutes={:.2} (false keeps resume metadata as publish baseline only)",
        policy_promotion_seed_local_best_from_resume,
        policy_promotion_seed_local_best_reevaluate_resume,
        policy_promotion_seed_local_best_eval_games,
        policy_promotion_seed_local_best_eval_minutes
    );
    if let Some(path) = &resume_artifact {
        println!("resume_artifact={path}");
    }
    if let Some(path) = &moment_replay_path {
        println!(
            "moment_replay path={} records={} transitions={} passes={} reward_scale={:.3}",
            path,
            moment_replay_records,
            moment_replay_transitions,
            moment_replay_passes,
            moment_replay_reward_scale
        );
    }
    println!(
        "tactical_learning attack_spacing_delta={:.3} attack_spacing_score={:.3} attack_width_delta={:.3} attack_width_score={:.3} attack_flank_lane={:.3} defense_spacing_delta={:.3} defense_spacing_score={:.3} defense_contract_delta={:.3} defense_compactness_score={:.3}",
        tactical_learning.attack_spacing_delta_weight,
        tactical_learning.attack_spacing_score_weight,
        tactical_learning.attack_width_delta_weight,
        tactical_learning.attack_width_score_weight,
        tactical_learning.attack_flank_lane_weight,
        tactical_learning.defense_spacing_delta_weight,
        tactical_learning.defense_spacing_score_weight,
        tactical_learning.defense_contract_delta_weight,
        tactical_learning.defense_compactness_score_weight,
    );
    println!(
        "neural_learning enabled={} backend={} learning_rate={:.5} optimizer_momentum={:.3} batch_size={} train_every_ticks={} max_batches_per_tick={} hidden_units={} target_scale={:.3} max_pending_batches={} replay_capacity={} replay_samples_per_tick={} target_clip={:.3} target_popart={} snapshot_every_batches={} batch_snapshot_selection={} batch_snapshot_max_fitness_regression={:.3} batch_snapshot_min_fitness={:.3} batch_snapshot_min_batch_mean_fitness={:.3} batch_snapshot_rescue_min_fitness={:.3} batch_policy_merge_selected_only={} batch_snapshot_carry_unvalidated={} batch_snapshot_carry_no_publishable={}",
        neural_learning.enabled,
        neural_backend_label(neural_learning.backend),
        neural_learning.learning_rate,
        neural_learning.optimizer_momentum,
        neural_learning.batch_size,
        neural_learning.train_every_ticks,
        neural_learning.max_batches_per_tick,
        neural_learning.hidden_units,
        neural_learning.target_scale,
        neural_learning.max_pending_batches,
        neural_learning.replay_capacity,
        neural_learning.replay_samples_per_tick,
        neural_learning.target_clip,
        neural_learning.target_popart_enabled,
        neural_learning.snapshot_every_batches,
        neural_batch_snapshot_selection_mode.label(),
        neural_batch_snapshot_max_fitness_regression,
        neural_batch_snapshot_min_fitness,
        neural_batch_snapshot_min_batch_mean_fitness,
        neural_batch_snapshot_rescue_min_fitness,
        neural_batch_policy_merge_selected_only,
        neural_batch_snapshot_carry_unvalidated,
        neural_batch_snapshot_carry_no_publishable,
    );
    println!(
        "neural_batch_snapshot_validation games={} candidates={} eval_minutes={:.2} objective=home_directional_learning_vs_analytic min_fitness={:.4} min_goal_margin={:.4} carry_min_fitness={:.4} carry_min_goal_margin={:.4}",
        neural_batch_snapshot_validate_games,
        neural_batch_snapshot_validate_candidates,
        neural_batch_snapshot_validate_minutes,
        neural_batch_snapshot_validate_min_fitness,
        neural_batch_snapshot_validate_min_goal_margin,
        neural_batch_snapshot_carry_min_validation_fitness,
        neural_batch_snapshot_carry_min_validation_goal_margin,
    );
    println!(
        "adversarial_embedding enabled={} memory_limit={} preloaded_windows={}",
        adversarial_embedding_exploitation_enabled,
        adversarial_embedding_memory_limit,
        adversarial_moment_windows.len(),
    );
    if print_completed_games {
        println!("game seed score shots on_target passes_completed/pass_attempted interceptions");
    }

    let started = Instant::now();
    let mut episode_summaries = Vec::new();
    let mut manifest_games = Vec::new();
    let mut tactical_summary = SoccerTacticalLearningSummary::default();
    let mut total_home_goals = 0u32;
    let mut total_away_goals = 0u32;
    let mut total_shots = 0u32;
    let mut total_on_target = 0u32;
    let mut total_pass_attempts = 0u32;
    let mut total_pass_completions = 0u32;
    let mut total_interceptions = 0u32;
    let mut next_episode = 0usize;
    let mut last_checkpoint_episode = 0usize;
    let mut latest_neural_network = initial_neural_network.clone();
    let mut latest_policy_arc_cache = None::<CachedTeamQPoliciesArc>;
    let mut latest_neural_network_arc_cache = None::<CachedNeuralNetworkSnapshotArc>;
    // Frozen-anchor promotion-gate state (used only when the gate is enabled): the
    // current anchor brain a candidate must beat to advance, a reusable held-out
    // match runner, and a counter to apply the gate every `interval_writes` writes.
    let mut anchor_neural_network = latest_neural_network.clone();
    let mut anchor_gate_runner: Option<EngineMatchRunner> = None;
    let mut anchor_gate_write_index: usize = 0;
    let policy_promotion_window_games =
        pg_policy_version_interval_games.max(policy_promotion_gate.min_sample_games);
    let mut policy_promotion_summaries =
        VecDeque::<MatchSummary>::with_capacity(policy_promotion_window_games.max(1));
    let mut evolution_search_samples =
        VecDeque::<EvolutionSearchSample>::with_capacity(evolution_window_games);
    let mut local_tactical_evolved_since_pg_refresh = false;
    let mut local_evolution_trial_active = false;
    let mut local_neural_promotion_trial_best = None::<LocalNeuralPromotionTrialBest>;
    if policy_promotion_seed_local_best_from_resume {
        let mut seeded_from_fresh_eval = false;
        if policy_promotion_seed_local_best_reevaluate_resume {
            let seed_snapshot = local_policy_promotion_best_seed_neural_network
                .clone()
                .or_else(|| latest_neural_network.clone());
            if let Some(snapshot) = seed_snapshot {
                let seed_neural_source =
                    if local_policy_promotion_best_seed_neural_network.is_some() {
                        "stored_sidecar"
                    } else {
                        "resume_sidecar"
                    };
                let mut seed_eval_config = neural_population_search_config;
                seed_eval_config.eval_games = policy_promotion_seed_local_best_eval_games;
                seed_eval_config.eval_minutes = policy_promotion_seed_local_best_eval_minutes;
                let candidate = NeuralPopulationCandidate {
                    index: 0,
                    source: format!("resume-local-best:{seed_neural_source}"),
                    snapshot: snapshot.clone(),
                };
                match evaluate_neural_population_candidate_against_analytic_home_learning_objective(
                    candidate,
                    config.clone(),
                    seed_eval_config,
                    neural_population_search_config.seed ^ 0xA77A_C01D_5EED_BA5E,
                ) {
                    Ok(eval) => {
                        let sample_games = eval.wins + eval.draws + eval.losses;
                        let mean_goal_margin = neural_population_candidate_goal_margin(&eval);
                        println!(
                            "policy_promotion_local_best_seeded_from_resume_eval eval_games={} mean_match_fitness={:.4} mean_goal_margin={:.4} record={}-{}-{} goals={}-{}",
                            sample_games,
                            eval.fitness,
                            mean_goal_margin,
                            eval.wins,
                            eval.draws,
                            eval.losses,
                            eval.goals_for,
                            eval.goals_against,
                        );
                        let fresh_evaluation =
                            policy_promotion_evaluation_from_candidate_validation(
                                &eval,
                                policy_promotion_gate,
                            );
                        let (seed_evaluation, seed_source) =
                            retain_resume_local_best_seed_evaluation(
                                pg_base_policy_promotion_baseline,
                                fresh_evaluation.clone(),
                                policy_promotion_gate,
                            );
                        let seed_neural_network = Some(snapshot.clone());
                        println!(
                            "policy_promotion_local_best_seeded_from_resume_neural_source source={} fingerprint={}",
                            seed_neural_source,
                            soccer_neural_network_snapshot_fingerprint(&snapshot)
                        );
                        latest_neural_network = Some(snapshot);
                        latest_neural_network_arc_cache = None;
                        anchor_neural_network = latest_neural_network.clone();
                        if let Some(stored_baseline) = pg_base_policy_promotion_baseline {
                            let stored_evaluation =
                                policy_promotion_evaluation_from_incumbent_baseline(
                                    stored_baseline,
                                    policy_promotion_gate,
                                );
                            if stored_evaluation.mean_match_fitness
                                > fresh_evaluation.mean_match_fitness + 1e-9
                            {
                                println!(
                                    "policy_promotion_local_best_resume_seed_stored_metadata_publish_floor_only sample_games={} mean_match_fitness={:.4} fresh_mean_match_fitness={:.4} reason=fresh_eval_controls_restore",
                                    stored_evaluation.sample_games,
                                    stored_evaluation.mean_match_fitness,
                                    fresh_evaluation.mean_match_fitness,
                                );
                            } else if fresh_evaluation.mean_match_fitness
                                > stored_evaluation.mean_match_fitness + 1e-9
                            {
                                println!(
                                    "policy_promotion_local_best_resume_seed_fresh_eval_retained sample_games={} fresh_mean_match_fitness={:.4} stored_mean_match_fitness={:.4} reason=fresh_eval_stronger_than_stored_floor",
                                    fresh_evaluation.sample_games,
                                    fresh_evaluation.mean_match_fitness,
                                    stored_evaluation.mean_match_fitness,
                                );
                            }
                        }
                        pg_base_policy_promotion_baseline =
                            policy_promotion_incumbent_baseline_from_evaluation(&seed_evaluation);
                        if let Some(baseline) = pg_base_policy_promotion_baseline {
                            println!(
                                "policy_promotion_incumbent_recalibrated_from_resume_eval sample_games={} mean_match_fitness={:.4} best_match_fitness={:.4} mean_play_quality={:.4}",
                                baseline.sample_games,
                                baseline.mean_match_fitness,
                                baseline.best_match_fitness,
                                baseline.mean_play_quality,
                            );
                        }
                        println!(
                            "policy_promotion_local_best_seeded_from_resume_effective source={} sample_games={} mean_match_fitness={:.4} best_match_fitness={:.4} mean_play_quality={:.4}",
                            seed_source,
                            seed_evaluation.sample_games,
                            seed_evaluation.mean_match_fitness,
                            seed_evaluation.best_match_fitness,
                            seed_evaluation.mean_play_quality,
                        );
                        local_neural_promotion_trial_best = Some(LocalNeuralPromotionTrialBest {
                            completed_games: 0,
                            evaluation: seed_evaluation,
                            config: config.clone(),
                            tactical_learning: tactical_learning.clone(),
                            policies: policies.clone(),
                            neural_network: seed_neural_network,
                            carried_world_model: current_carried_world_model_snapshot(),
                        });
                        seeded_from_fresh_eval = true;
                    }
                    Err(error) => {
                        println!(
                            "policy_promotion_local_best_resume_seed_reevaluate_failed error={}",
                            error.replace(char::is_whitespace, "_")
                        );
                    }
                }
            } else {
                println!("policy_promotion_local_best_resume_seed_reevaluate_skipped reason=no_neural_network");
            }
        }
        if !seeded_from_fresh_eval {
            if let Some(baseline) = pg_base_policy_promotion_baseline {
                if policy_promotion_seed_local_best_reevaluate_resume {
                    println!(
                        "policy_promotion_local_best_resume_seed_stored_metadata_ignored sample_games={} mean_match_fitness={:.4} best_match_fitness={:.4} mean_play_quality={:.4} reason=reevaluate_required",
                        baseline.sample_games,
                        baseline.mean_match_fitness,
                        baseline.best_match_fitness,
                        baseline.mean_play_quality,
                    );
                } else {
                    let seed_neural_network = local_policy_promotion_best_seed_neural_network
                        .clone()
                        .or_else(|| latest_neural_network.clone());
                    if let Some(snapshot) = local_policy_promotion_best_seed_neural_network.clone()
                    {
                        println!(
                            "policy_promotion_local_best_seeded_from_resume_neural_source source=stored_sidecar fingerprint={}",
                            soccer_neural_network_snapshot_fingerprint(&snapshot)
                        );
                        latest_neural_network = Some(snapshot);
                        latest_neural_network_arc_cache = None;
                        anchor_neural_network = latest_neural_network.clone();
                    }
                    local_neural_promotion_trial_best = Some(LocalNeuralPromotionTrialBest {
                        completed_games: 0,
                        evaluation: policy_promotion_evaluation_from_incumbent_baseline(
                            baseline,
                            policy_promotion_gate,
                        ),
                        config: config.clone(),
                        tactical_learning: tactical_learning.clone(),
                        policies: policies.clone(),
                        neural_network: seed_neural_network,
                        carried_world_model: current_carried_world_model_snapshot(),
                    });
                    println!(
                "policy_promotion_local_best_seeded_from_resume sample_games={} mean_match_fitness={:.4} best_match_fitness={:.4} mean_play_quality={:.4}",
                baseline.sample_games,
                baseline.mean_match_fitness,
                baseline.best_match_fitness,
                baseline.mean_play_quality,
            );
                }
            }
        }
    } else if let Some(baseline) = pg_base_policy_promotion_baseline {
        println!(
            "policy_promotion_local_best_resume_seed_skipped sample_games={} mean_match_fitness={:.4} best_match_fitness={:.4} mean_play_quality={:.4} reason=publish-baseline-only",
            baseline.sample_games,
            baseline.mean_match_fitness,
            baseline.best_match_fitness,
            baseline.mean_play_quality,
        );
    }
    let mut pg_last_neural_checkpoint_fingerprint = pg_base_neural_network_fingerprint;
    let mut policy_promotion_recalibration_pending =
        policy_promotion_recalibrate_incumbent_on_resume && pg_load_initial_policy_from_postgres;
    let mut pg_policy_version_buffer = Vec::new();
    let mut pg_completed_buffer = Vec::new();
    let pg_runner_id = soccer_learning_pg_runner_id(&run_id, shard_index, shard_count);
    let worker_pool = SoccerLearningWorkerPool::start(parallel_games);
    let retain_neural_network_in_game_artifact =
        write_game_artifacts && game_artifact_mode.as_str() == "full";
    let adversarial_moment_windows = Arc::new(adversarial_moment_windows);
    let mut neural_population_accept_restore_protection_games_remaining = 0usize;
    let mut neural_population_accept_learning_freeze_games_remaining = 0usize;

    while next_episode < games {
        let batch_size = parallel_games.min(games - next_episode);
        let batch_start_episode = next_episode;
        let (_, batch_start_curriculum) = soccer_learning_curriculum_episode_config(
            &config,
            batch_start_episode,
            &curriculum_config,
        );
        let (_, batch_end_curriculum) = soccer_learning_curriculum_episode_config(
            &config,
            batch_start_episode + batch_size.saturating_sub(1),
            &curriculum_config,
        );
        println!(
            "starting_batch episodes={}..{} parallel_games={} curriculum_stage={}..{} curriculum_drill_players_per_team={}..{} curriculum_duration_seconds={:.1}..{:.1}",
            batch_start_episode + 1,
            batch_start_episode + batch_size,
            batch_size,
            batch_start_curriculum.stage.as_str(),
            batch_end_curriculum.stage.as_str(),
            batch_start_curriculum.drill_players_per_team,
            batch_end_curriculum.drill_players_per_team,
            batch_start_curriculum.duration_seconds,
            batch_end_curriculum.duration_seconds
        );

        for offset in 0..batch_size {
            let episode = batch_start_episode + offset;
            if pg_refresh_for_new_sims {
                let mut pending_async_pg_batches = 0usize;
                let mut pending_async_policy_version_batches = 0usize;
                if let Some(writer) = pg_completed_writer.as_mut() {
                    pg_persisted_games += writer.drain_finished().map_err(invalid_data)?;
                    pending_async_pg_batches = writer.pending_batches;
                    pending_async_policy_version_batches = writer.pending_policy_version_batches;
                }
                let new_sim_refresh_plan = soccer_postgres_new_sim_refresh_plan(
                    resume_artifact.is_some(),
                    pg_refresh_with_resume_artifact,
                    pg_flush_policy_versions_before_new_sim,
                    pg_policy_version_buffer.len(),
                    pending_async_policy_version_batches,
                );
                if let (Some(experiment_id), Some(store)) =
                    (pg_experiment_id.as_deref(), pg_store.as_mut())
                {
                    if new_sim_refresh_plan.flush_pending_policy_versions {
                        flush_postgres_policy_versions_for_new_sims(
                            store,
                            experiment_id,
                            &pg_runner_id,
                            &mut pg_policy_version_buffer,
                            shard_index,
                            shard_count,
                        )?;
                    }
                    if new_sim_refresh_plan.wait_for_async_policy_versions {
                        if let Some(writer) = pg_completed_writer.as_mut() {
                            pg_persisted_games += writer
                                .drain_policy_versions_finished()
                                .map_err(invalid_data)?;
                            pending_async_pg_batches = writer.pending_batches;
                            pending_async_policy_version_batches =
                                writer.pending_policy_version_batches;
                        }
                    }
                    if new_sim_refresh_plan.refresh_from_postgres
                        && refresh_postgres_policy_for_next_sim(
                            store,
                            experiment_id,
                            episode + 1,
                            &options,
                            &mut evolution_options,
                            &mut config,
                            &mut tactical_learning,
                            &mut policies,
                            &mut latest_neural_network,
                            &mut pg_base_policy_version_id,
                            &mut pg_last_policy_version_id,
                            &mut pg_generation,
                            &mut pg_base_policy_version_updated_at_micros,
                            &mut pg_base_policy_fingerprint,
                            &mut pg_base_neural_network_fingerprint,
                            &mut pg_base_tactical_learning_fingerprint,
                            &mut pg_base_policy_promotion_baseline,
                            &mut local_tactical_evolved_since_pg_refresh,
                            pg_tactical_learning_authoritative,
                            pg_resume_include_unpromoted,
                            policy_promotion_gate.min_sample_games,
                            policy_promotion_baseline_lookback_generations,
                            pg_policy_version_buffer.len(),
                            pending_async_pg_batches,
                            pending_async_policy_version_batches,
                        )?
                    {
                        clear_evolution_search_samples_after_postgres_refresh(
                            "postgres_refresh_evolution_samples_reset_for_game",
                            episode + 1,
                            pg_base_policy_version_id.as_deref(),
                            &mut evolution_search_samples,
                        );
                        // Re-anchor the promotion gate to the incumbent just pulled
                        // from Postgres: that refreshed brain is the new baseline a
                        // candidate must beat, so the ratchet tracks it.
                        if anchor_promotion_gate.enabled {
                            anchor_neural_network = latest_neural_network.clone();
                        }
                        local_evolution_trial_active = false;
                    }
                }
            }
            let episode_starting_policies =
                cached_team_policies_arc(&policies, &mut latest_policy_arc_cache);
            let episode_starting_neural_network = cached_neural_network_snapshot_arc(
                &latest_neural_network,
                &mut latest_neural_network_arc_cache,
            );
            let episode_starting_policy_version_id = pg_base_policy_version_id.clone();
            let episode_starting_policy_generation = pg_generation;
            let (mut episode_config, _) =
                soccer_learning_curriculum_episode_config(&config, episode, &curriculum_config);
            episode_config.seed = effective_seed.wrapping_add(episode as u32);
            if neural_population_accept_restore_protection_games_remaining > 0 {
                neural_population_accept_restore_protection_games_remaining =
                    neural_population_accept_restore_protection_games_remaining.saturating_sub(1);
                println!(
                    "neural_population_accept_restore_protection episode={} remaining_after={} learning_rate={:.5} optimizer_momentum={:.3} reason=protect_accepted_population_candidate",
                    episode + 1,
                    neural_population_accept_restore_protection_games_remaining,
                    episode_config.neural_learning.learning_rate,
                    episode_config.neural_learning.optimizer_momentum,
                );
            }
            if neural_population_accept_learning_freeze_games_remaining > 0 {
                let previous_learning_rate = episode_config.neural_learning.learning_rate;
                let previous_optimizer_momentum = episode_config.neural_learning.optimizer_momentum;
                episode_config.neural_learning.learning_rate = 0.0;
                episode_config.neural_learning.optimizer_momentum = 0.0;
                neural_population_accept_learning_freeze_games_remaining =
                    neural_population_accept_learning_freeze_games_remaining.saturating_sub(1);
                println!(
                    "neural_population_accept_learning_freeze episode={} remaining_after={} learning_rate={:.5}->0.00000 optimizer_momentum={:.3}->0.000 reason=protect_accepted_population_local_best_candidate",
                    episode + 1,
                    neural_population_accept_learning_freeze_games_remaining,
                    previous_learning_rate,
                    previous_optimizer_momentum,
                );
            }
            worker_pool.submit(SoccerLearningWorkerTask {
                episode,
                config: episode_config,
                starting_policies: episode_starting_policies,
                starting_policy_version_id: episode_starting_policy_version_id,
                starting_policy_generation: episode_starting_policy_generation,
                initial_neural_network: episode_starting_neural_network.as_ref().map(Arc::clone),
                analytic_neural_opponent,
                adversarial_moment_windows: Arc::clone(&adversarial_moment_windows),
                print_progress,
                neural_drain_timeout,
                retain_neural_network_in_game_artifact,
            })?;
        }

        let mut completed_games = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            completed_games.push(worker_pool.recv_completed_game()?);
        }
        completed_games.sort_by_key(|game| game.episode_summary.episode);

        let merge_deltas = completed_games.len() > 1;
        let neural_batch_snapshot_count = completed_games
            .iter()
            .filter(|game| game.neural_network.is_some())
            .count();
        let neural_batch_best_snapshot_fitness = completed_games
            .iter()
            .filter(|game| game.neural_network.is_some())
            .map(|game| {
                soccer_learning_objective_match_fitness(
                    &game.episode_summary.summary,
                    analytic_neural_opponent,
                )
            })
            .fold(f64::NEG_INFINITY, f64::max);
        let neural_batch_mean_snapshot_fitness = if neural_batch_snapshot_count == 0 {
            f64::NEG_INFINITY
        } else {
            completed_games
                .iter()
                .filter(|game| game.neural_network.is_some())
                .map(|game| {
                    soccer_learning_objective_match_fitness(
                        &game.episode_summary.summary,
                        analytic_neural_opponent,
                    )
                })
                .sum::<f64>()
                / neural_batch_snapshot_count as f64
        };
        let mut selected_neural_snapshot_episode = None::<usize>;
        let mut carried_unvalidated_neural_snapshot_episode = None::<usize>;
        let mut batch_kept_publish_validated_training_snapshot = false;
        let mut batch_publish_validation_evaluation = None::<SoccerPolicyPromotionGateEvaluation>;
        let mut neural_batch_snapshot_attempted = false;
        let mut neural_batch_snapshot_attempts = 0usize;
        while selected_neural_snapshot_episode.is_none()
            && carried_unvalidated_neural_snapshot_episode.is_none()
            && neural_batch_snapshot_attempts < neural_batch_snapshot_validate_candidates
        {
            let Some(selection) = select_batch_neural_snapshot(
                &mut completed_games,
                neural_batch_snapshot_selection_mode,
                analytic_neural_opponent,
                neural_batch_snapshot_max_fitness_regression,
                neural_batch_snapshot_min_fitness,
                neural_batch_snapshot_min_batch_mean_fitness,
                neural_batch_snapshot_rescue_min_fitness,
            ) else {
                break;
            };
            neural_batch_snapshot_attempted = true;
            neural_batch_snapshot_attempts += 1;
            let BatchNeuralSnapshotSelection {
                episode: selected_episode,
                match_fitness: selected_match_fitness,
                training_steps: selected_training_steps,
                snapshot_count: selected_snapshot_count,
                snapshot: selected_snapshot,
                world_model: selected_world_model,
            } = selection;
            let mut validation_eval = None::<NeuralPopulationCandidateEval>;
            let mut validation_failed = false;
            let mut validation_carry_allowed = true;
            let mut validation_carry_fitness = None::<f64>;
            let mut validation_carry_goal_margin = None::<f64>;
            let mut validation_promotion_evaluation = None::<SoccerPolicyPromotionGateEvaluation>;
            if neural_batch_snapshot_validate_games > 0 {
                let mut validation_config = neural_population_search_config;
                validation_config.eval_games = neural_batch_snapshot_validate_games.max(1);
                validation_config.eval_minutes = neural_batch_snapshot_validate_minutes;
                let validation_seed = u64::from(effective_seed)
                    ^ (selected_episode as u64).wrapping_mul(0xB492_B66F_BE98_F273)
                    ^ ((batch_start_episode + batch_size) as u64)
                        .wrapping_mul(0x9AE1_6A3B_2F90_404F);
                let validation_candidate = NeuralPopulationCandidate {
                    index: 0,
                    source: format!("batch_snapshot_episode_{selected_episode}"),
                    snapshot: selected_snapshot.clone(),
                };
                match evaluate_neural_population_candidate_against_analytic_home_learning_objective(
                    validation_candidate,
                    config.clone(),
                    validation_config,
                    validation_seed,
                ) {
                    Ok(eval) => {
                        let goal_margin = neural_population_candidate_goal_margin(&eval);
                        if batch_neural_snapshot_validation_passes(
                            &eval,
                            neural_batch_snapshot_validate_min_fitness,
                            neural_batch_snapshot_validate_min_goal_margin,
                        ) {
                            println!(
                                "neural_batch_snapshot_validation_passed episodes={}..{} selected_episode={} attempt={} eval_games={} eval_minutes={:.2} fitness={:.4} goal_margin={:.4} record={}-{}-{} goals={}-{}",
                                batch_start_episode + 1,
                                batch_start_episode + batch_size,
                                selected_episode,
                                neural_batch_snapshot_attempts,
                                neural_batch_snapshot_validate_games,
                                neural_batch_snapshot_validate_minutes,
                                eval.fitness,
                                goal_margin,
                                eval.wins,
                                eval.draws,
                                eval.losses,
                                eval.goals_for,
                                eval.goals_against,
                            );
                            validation_promotion_evaluation =
                                Some(policy_promotion_evaluation_from_candidate_validation(
                                    &eval,
                                    policy_promotion_gate,
                                ));
                            validation_eval = Some(eval);
                        } else {
                            validation_carry_fitness = Some(eval.fitness);
                            validation_carry_goal_margin = Some(goal_margin);
                            validation_carry_allowed = batch_neural_snapshot_validation_passes(
                                &eval,
                                neural_batch_snapshot_carry_min_validation_fitness,
                                neural_batch_snapshot_carry_min_validation_goal_margin,
                            );
                            println!(
                                "neural_batch_snapshot_validation_held episodes={}..{} selected_episode={} selected_match_fitness={:.4} attempt={} eval_games={} eval_minutes={:.2} fitness={:.4} goal_margin={:.4} min_fitness={:.4} min_goal_margin={:.4} record={}-{}-{} goals={}-{}",
                                batch_start_episode + 1,
                                batch_start_episode + batch_size,
                                selected_episode,
                                selected_match_fitness,
                                neural_batch_snapshot_attempts,
                                neural_batch_snapshot_validate_games,
                                neural_batch_snapshot_validate_minutes,
                                eval.fitness,
                                goal_margin,
                                neural_batch_snapshot_validate_min_fitness,
                                neural_batch_snapshot_validate_min_goal_margin,
                                eval.wins,
                                eval.draws,
                                eval.losses,
                                eval.goals_for,
                                eval.goals_against,
                            );
                            validation_failed = true;
                        }
                    }
                    Err(error) => {
                        validation_carry_allowed = false;
                        println!(
                            "neural_batch_snapshot_validation_error episodes={}..{} selected_episode={} attempt={} error={}",
                            batch_start_episode + 1,
                            batch_start_episode + batch_size,
                            selected_episode,
                            neural_batch_snapshot_attempts,
                            error.replace(char::is_whitespace, "_"),
                        );
                        validation_failed = true;
                    }
                }
            }
            if !validation_failed {
                selected_neural_snapshot_episode = Some(selected_episode);
                batch_kept_publish_validated_training_snapshot =
                    neural_batch_snapshot_validate_games > 0;
                batch_publish_validation_evaluation = validation_promotion_evaluation;
                let selected_world_model_steps =
                    install_carried_world_model_snapshot(selected_world_model);
                latest_neural_network = Some(selected_snapshot);
                latest_neural_network_arc_cache = None;
                println!(
                    "neural_batch_snapshot_selected episodes={}..{} snapshots={} selection_mode={} selected_episode={} selected_match_fitness={:.4} selected_training_steps={} selected_world_model_steps={} validation_fitness={} validation_goal_margin={}",
                    batch_start_episode + 1,
                    batch_start_episode + batch_size,
                    selected_snapshot_count,
                    neural_batch_snapshot_selection_mode.label(),
                    selected_episode,
                    selected_match_fitness,
                    selected_training_steps,
                    selected_world_model_steps,
                    validation_eval
                        .as_ref()
                        .map(|eval| format!("{:.4}", eval.fitness))
                        .unwrap_or_else(|| "disabled".to_string()),
                    validation_eval
                        .as_ref()
                        .map(|eval| format!(
                            "{:.4}",
                            neural_population_candidate_goal_margin(eval)
                        ))
                        .unwrap_or_else(|| "disabled".to_string()),
                );
            } else if neural_batch_snapshot_carry_unvalidated && validation_carry_allowed {
                let selected_world_model_steps =
                    install_carried_world_model_snapshot(selected_world_model);
                latest_neural_network = Some(selected_snapshot);
                latest_neural_network_arc_cache = None;
                carried_unvalidated_neural_snapshot_episode = Some(selected_episode);
                println!(
                    "neural_batch_snapshot_carried_unvalidated episodes={}..{} snapshots={} selection_mode={} carried_episode={} carried_match_fitness={:.4} carried_training_steps={} carried_world_model_steps={} reason=validation_failed publish_selected=false",
                    batch_start_episode + 1,
                    batch_start_episode + batch_size,
                    selected_snapshot_count,
                    neural_batch_snapshot_selection_mode.label(),
                    selected_episode,
                    selected_match_fitness,
                    selected_training_steps,
                    selected_world_model_steps,
                );
            } else if neural_batch_snapshot_carry_unvalidated {
                println!(
                    "neural_batch_snapshot_carry_unvalidated_held episodes={}..{} snapshots={} selection_mode={} held_episode={} held_match_fitness={:.4} attempt={} validation_fitness={} validation_goal_margin={} carry_min_fitness={:.4} carry_min_goal_margin={:.4} reason=validation_floor publish_selected=false",
                    batch_start_episode + 1,
                    batch_start_episode + batch_size,
                    selected_snapshot_count,
                    neural_batch_snapshot_selection_mode.label(),
                    selected_episode,
                    selected_match_fitness,
                    neural_batch_snapshot_attempts,
                    validation_carry_fitness
                        .map(|fitness| format!("{fitness:.4}"))
                        .unwrap_or_else(|| "unavailable".to_string()),
                    validation_carry_goal_margin
                        .map(|goal_margin| format!("{goal_margin:.4}"))
                        .unwrap_or_else(|| "unavailable".to_string()),
                    neural_batch_snapshot_carry_min_validation_fitness,
                    neural_batch_snapshot_carry_min_validation_goal_margin,
                );
            }
        }
        if !neural_batch_snapshot_attempted && neural_batch_snapshot_count > 0 {
            println!(
                "neural_batch_snapshot_held episodes={}..{} snapshots={} selection_mode={} mean_match_fitness={:.4} best_match_fitness={:.4} min_snapshot_fitness={:.4} min_batch_mean_fitness={:.4} rescue_min_fitness={:.4}",
                batch_start_episode + 1,
                batch_start_episode + batch_size,
                neural_batch_snapshot_count,
                neural_batch_snapshot_selection_mode.label(),
                neural_batch_mean_snapshot_fitness,
                neural_batch_best_snapshot_fitness,
                neural_batch_snapshot_min_fitness,
                neural_batch_snapshot_min_batch_mean_fitness,
                neural_batch_snapshot_rescue_min_fitness,
            );
        } else if neural_batch_snapshot_attempted
            && selected_neural_snapshot_episode.is_none()
            && carried_unvalidated_neural_snapshot_episode.is_none()
        {
            println!(
                "neural_batch_snapshot_validation_candidates_exhausted episodes={}..{} attempts={} max_candidates={} publish_selected=false carry_selected=false",
                batch_start_episode + 1,
                batch_start_episode + batch_size,
                neural_batch_snapshot_attempts,
                neural_batch_snapshot_validate_candidates,
            );
        }
        if should_carry_no_publishable_neural_snapshot(
            neural_batch_snapshot_carry_unvalidated,
            neural_batch_snapshot_carry_no_publishable,
            selected_neural_snapshot_episode,
            carried_unvalidated_neural_snapshot_episode,
        ) {
            if let Some(carry_selection) = select_batch_neural_snapshot_for_training_carry(
                &mut completed_games,
                analytic_neural_opponent,
            ) {
                let BatchNeuralSnapshotSelection {
                    episode: carried_episode,
                    match_fitness: carried_match_fitness,
                    training_steps: carried_training_steps,
                    snapshot_count: carried_snapshot_count,
                    snapshot: carried_snapshot,
                    world_model: carried_world_model,
                } = carry_selection;
                let carried_world_model_steps =
                    install_carried_world_model_snapshot(carried_world_model);
                latest_neural_network = Some(carried_snapshot);
                latest_neural_network_arc_cache = None;
                carried_unvalidated_neural_snapshot_episode = Some(carried_episode);
                println!(
                    "neural_batch_snapshot_carried_unvalidated episodes={}..{} snapshots={} selection_mode=training_steps carried_episode={} carried_match_fitness={:.4} carried_training_steps={} carried_world_model_steps={} reason=no_publishable_snapshot publish_selected=false",
                    batch_start_episode + 1,
                    batch_start_episode + batch_size,
                    carried_snapshot_count,
                    carried_episode,
                    carried_match_fitness,
                    carried_training_steps,
                    carried_world_model_steps,
                );
            } else if neural_batch_snapshot_count > 0 {
                println!(
                    "neural_batch_snapshot_carry_unvalidated_held episodes={}..{} snapshots={} reason=no_available_training_snapshot",
                    batch_start_episode + 1,
                    batch_start_episode + batch_size,
                    neural_batch_snapshot_count,
                );
            }
        }
        if neural_batch_policy_merge_selected_only {
            println!(
                "neural_batch_policy_merge_selected_only episodes={}..{} selected_episode={} carried_unvalidated_episode={}",
                batch_start_episode + 1,
                batch_start_episode + batch_size,
                selected_neural_snapshot_episode
                    .map(|episode| episode.to_string())
                    .unwrap_or_else(|| "none".to_string()),
                carried_unvalidated_neural_snapshot_episode
                    .map(|episode| episode.to_string())
                    .unwrap_or_else(|| "none".to_string()),
            );
        }
        let policy_delta_frontier_episode = batch_policy_delta_frontier_episode(
            selected_neural_snapshot_episode,
            carried_unvalidated_neural_snapshot_episode,
        );
        for game in completed_games.iter_mut() {
            let completed_learning_game = soccer_learning_completed_game_from_completed(game);
            let completed_episode_for_policy_merge = game.episode_summary.episode + 1;
            let should_merge_policy_delta = should_merge_batch_policy_delta(
                neural_batch_policy_merge_selected_only,
                policy_delta_frontier_episode,
                completed_episode_for_policy_merge,
            );
            if merge_deltas {
                if should_merge_policy_delta {
                    merge_team_policy_delta(
                        &mut policies,
                        game.starting_policies.as_ref(),
                        &game.policies,
                    );
                }
            } else if should_merge_policy_delta {
                policies = game.policies.clone();
            }
            push_policy_promotion_summary(
                &mut policy_promotion_summaries,
                &game.episode_summary.summary,
                policy_promotion_window_games,
            );
            if pg_experiment_id.is_some() {
                let pg_game_base_policy_version_id = game.starting_policy_version_id.clone();
                let completed_episode = game.episode_summary.episode + 1;
                let should_write_policy_version = pg_policy_version_interval_games <= 1
                    || completed_episode >= games
                    || completed_episode % pg_policy_version_interval_games == 0;
                let curriculum_stage = soccer_learning_curriculum_stage_for_completed_games(
                    completed_episode,
                    &curriculum_config,
                );
                let curriculum_stage_label = curriculum_stage.as_str();
                let promotion_summary_refs = policy_promotion_summaries.iter().collect::<Vec<_>>();
                let mut policy_promotion_evaluation =
                    evaluate_soccer_policy_promotion_gate_for_learning_objective(
                        &promotion_summary_refs,
                        policy_promotion_gate,
                        analytic_neural_opponent,
                    );
                let mut policy_promotion_recalibrated_this_window = false;
                // Incumbent recalibration (origin): when a resume trial has settled, freeze the
                // current evaluation as the new incumbent baseline before comparing against it.
                if policy_promotion_recalibration_pending
                    && !local_evolution_trial_active
                    && policy_promotion_evaluation.enabled
                    && policy_promotion_evaluation.sample_games
                        >= policy_promotion_gate.min_sample_games
                    && policy_promotion_evaluation.rejection_reasons.is_empty()
                {
                    pg_base_policy_promotion_baseline =
                        policy_promotion_incumbent_baseline_from_evaluation(
                            &policy_promotion_evaluation,
                        );
                    policy_promotion_recalibration_pending = false;
                    policy_promotion_recalibrated_this_window = true;
                    println!(
                        "policy_promotion_incumbent_recalibrated completed_games={} curriculum_stage={} sample_games={} mean_match_fitness={:.4} best_match_fitness={:.4} mean_play_quality={:.4}",
                        completed_episode,
                        curriculum_stage_label,
                        policy_promotion_evaluation.sample_games,
                        policy_promotion_evaluation.mean_match_fitness,
                        policy_promotion_evaluation.best_match_fitness,
                        policy_promotion_evaluation.mean_play_quality
                    );
                }
                // Incumbent-comparison gate (origin): may downgrade the evaluation in place.
                if !policy_promotion_recalibrated_this_window {
                    apply_policy_promotion_incumbent_gate(
                        &mut policy_promotion_evaluation,
                        pg_base_policy_promotion_baseline,
                        policy_promotion_compare_incumbent,
                        policy_promotion_min_mean_fitness_delta,
                        policy_promotion_max_mean_fitness_regression,
                        policy_promotion_max_play_quality_regression,
                    );
                }
                // Frozen-anchor gate (HEAD): the incumbent-adjusted status is then run through the
                // held-out anchor gate, so a candidate must clear BOTH gates to be written active.
                let base_policy_version_status =
                    policy_version_status_for_promotion_gate(&policy_promotion_evaluation);
                let anchor_gate_had_baseline = anchor_neural_network.is_some();
                let anchor_gate_due = anchor_promotion_gate.enabled
                    && should_write_policy_version
                    && base_policy_version_status == SOCCER_POLICY_STATUS_ACTIVE
                    && anchor_promotion_gate.interval_writes > 0
                    && anchor_gate_write_index
                        .wrapping_add(1)
                        .is_multiple_of(anchor_promotion_gate.interval_writes);
                let anchor_gate_decision = apply_anchor_promotion_gate_with_decision(
                    &anchor_promotion_gate,
                    base_policy_version_status,
                    should_write_policy_version,
                    latest_neural_network.as_ref(),
                    &mut anchor_neural_network,
                    &mut anchor_gate_runner,
                    &mut anchor_gate_write_index,
                    completed_episode as u32,
                    &format!("completed_games={completed_episode}"),
                );
                let policy_version_status = anchor_gate_decision.status;
                let comparative_promotion_approved = anchor_gate_due
                    && anchor_gate_had_baseline
                    && latest_neural_network.is_some()
                    && policy_version_status == SOCCER_POLICY_STATUS_ACTIVE;
                let skip_incumbent_rejected_policy_version = should_write_policy_version
                    && policy_promotion_rejected_by_incumbent(&policy_promotion_evaluation);
                let latest_neural_network_fingerprint = latest_neural_network
                    .as_ref()
                    .map(soccer_neural_network_snapshot_fingerprint);
                let write_neural_checkpoint_for_rejected_promotion =
                    should_write_neural_checkpoint_for_rejected_promotion(
                        policy_write_neural_checkpoint_when_promotion_held,
                        should_write_policy_version,
                        skip_incumbent_rejected_policy_version,
                        latest_neural_network_fingerprint,
                        pg_last_neural_checkpoint_fingerprint,
                    );
                if should_write_policy_version
                    && policy_version_status != SOCCER_POLICY_STATUS_ACTIVE
                {
                    println!(
                        "policy_promotion_blocked completed_games={} curriculum_stage={} status={} sample_games={} mean_match_fitness={:.4} best_match_fitness={:.4} mean_play_quality={:.4} mean_goal_margin={:.4} reasons={}",
                        completed_episode,
                        curriculum_stage_label,
                        policy_version_status,
                        policy_promotion_evaluation.sample_games,
                        policy_promotion_evaluation.mean_match_fitness,
                        policy_promotion_evaluation.best_match_fitness,
                        policy_promotion_evaluation.mean_play_quality,
                        policy_promotion_evaluation.mean_goal_margin,
                        policy_promotion_evaluation.rejection_reasons.join("|")
                    );
                }
                let output_policy_version_id = if should_write_policy_version
                    && !skip_incumbent_rejected_policy_version
                {
                    let next_generation = pg_generation
                        .max(game.starting_policy_generation)
                        .saturating_add(1);
                    let effective_policy_version_status =
                        soccer_policy_version_insert_status_after_active_head(
                            policy_version_status,
                            pg_game_base_policy_version_id.as_deref(),
                            next_generation,
                            policy_promotion_evaluation.mean_match_fitness,
                            pg_base_policy_version_id.as_deref(),
                            Some(pg_generation),
                            if comparative_promotion_approved {
                                None
                            } else {
                                pg_base_policy_promotion_baseline
                                    .map(|baseline| baseline.mean_match_fitness)
                            },
                            active_max_fitness_regression(),
                        );
                    if effective_policy_version_status != policy_version_status {
                        println!(
                            "policy_promotion_local_marked_stale completed_games={} curriculum_stage={} status={} effective_status={} generation={} active_generation={} mean_match_fitness={:.4} active_mean_match_fitness={}",
                            completed_episode,
                            curriculum_stage_label,
                            policy_version_status,
                            effective_policy_version_status,
                            next_generation,
                            pg_generation,
                            policy_promotion_evaluation.mean_match_fitness,
                            pg_base_policy_promotion_baseline
                                .map(|baseline| format!("{:.4}", baseline.mean_match_fitness))
                                .unwrap_or_else(|| "none".to_string())
                        );
                    }
                    let version_label = soccer_learning_pg_version_label(
                        &run_id,
                        shard_index,
                        game.episode_summary.episode,
                    );
                    let output_policy_version_id = Uuid::new_v4().to_string();
                    pg_policy_version_buffer.push(PendingPostgresPolicyVersion {
                        id: output_policy_version_id.clone(),
                        parent_policy_version_id: pg_game_base_policy_version_id.clone(),
                        generation: next_generation,
                        version_label,
                        source_kind: SOCCER_POLICY_SOURCE_MERGE,
                        status: effective_policy_version_status,
                        comparative_promotion_approved,
                        config: config.clone(),
                        home_options: options.clone(),
                        away_options: options.clone(),
                        policies: policies.clone(),
                        fitness: policy_promotion_evaluation.mean_match_fitness,
                        neural_network: latest_neural_network.clone(),
                        search_metadata: policy_promotion_gate.enabled.then(|| {
                            serde_json::json!({
                                "comparativePromotionApproved": comparative_promotion_approved,
                                "anchorPromotion": anchor_promotion_gate_search_metadata(
                                    &anchor_gate_decision,
                                    anchor_gate_due,
                                    anchor_gate_had_baseline,
                                ),
                                "promotion": policy_promotion_search_metadata(
                                    curriculum_stage_label,
                                    &policy_promotion_evaluation,
                                )
                            })
                        }),
                    });
                    if effective_policy_version_status == SOCCER_POLICY_STATUS_ACTIVE {
                        local_evolution_trial_active = false;
                        local_neural_promotion_trial_best = Some(LocalNeuralPromotionTrialBest {
                            completed_games: completed_episode,
                            evaluation: policy_promotion_evaluation.clone(),
                            config: config.clone(),
                            tactical_learning: tactical_learning.clone(),
                            policies: policies.clone(),
                            neural_network: latest_neural_network.clone(),
                            carried_world_model: current_carried_world_model_snapshot(),
                        });
                        pg_base_policy_version_id = Some(output_policy_version_id.clone());
                        pg_last_policy_version_id = Some(output_policy_version_id.clone());
                        pg_generation = next_generation;
                        pg_base_policy_version_updated_at_micros = 0;
                        pg_base_policy_fingerprint =
                            Some(soccer_team_q_policies_fingerprint(&policies));
                        pg_base_neural_network_fingerprint = latest_neural_network
                            .as_ref()
                            .map(soccer_neural_network_snapshot_fingerprint);
                        pg_base_tactical_learning_fingerprint = Some(
                            soccer_tactical_learning_weights_fingerprint(&config.tactical_learning),
                        );
                        pg_base_policy_promotion_baseline =
                            retain_stronger_policy_promotion_baseline(
                                pg_base_policy_promotion_baseline,
                                policy_promotion_incumbent_baseline_from_evaluation(
                                    &policy_promotion_evaluation,
                                ),
                            );
                    } else if policy_promotion_local_best_rollback {
                        restore_local_best_after_held_promotion(
                            "local-stale",
                            completed_episode,
                            &policy_promotion_evaluation,
                            &mut local_neural_promotion_trial_best,
                            &mut config,
                            &mut tactical_learning,
                            &mut policies,
                            &mut latest_neural_network,
                            &mut latest_policy_arc_cache,
                            &mut latest_neural_network_arc_cache,
                            &mut local_tactical_evolved_since_pg_refresh,
                            policy_promotion_local_best_backoff,
                            policy_promotion_local_best_backoff_factor,
                            policy_promotion_local_best_backoff_min_learning_rate,
                            policy_promotion_local_best_backoff_min_blend_lambda,
                            policy_promotion_local_best_restore_min_games,
                            policy_promotion_local_best_restore_max_mean_fitness_regression,
                            policy_promotion_local_best_restore_max_play_quality_regression,
                        );
                    } else if policy_promotion_local_best_rollback {
                        restore_local_best_after_held_promotion(
                            "written-non-active",
                            completed_episode,
                            &policy_promotion_evaluation,
                            &mut local_neural_promotion_trial_best,
                            &mut config,
                            &mut tactical_learning,
                            &mut policies,
                            &mut latest_neural_network,
                            &mut latest_policy_arc_cache,
                            &mut latest_neural_network_arc_cache,
                            &mut local_tactical_evolved_since_pg_refresh,
                            policy_promotion_local_best_backoff,
                            policy_promotion_local_best_backoff_factor,
                            policy_promotion_local_best_backoff_min_learning_rate,
                            policy_promotion_local_best_backoff_min_blend_lambda,
                            policy_promotion_local_best_restore_min_games,
                            policy_promotion_local_best_restore_max_mean_fitness_regression,
                            policy_promotion_local_best_restore_max_play_quality_regression,
                        );
                    }
                    policy_promotion_summaries.clear();
                    Some(output_policy_version_id)
                } else {
                    if skip_incumbent_rejected_policy_version {
                        println!(
                            "policy_promotion_rejected_not_written completed_games={} curriculum_stage={} sample_games={} mean_match_fitness={:.4} mean_play_quality={:.4} reasons={}",
                            completed_episode,
                            curriculum_stage_label,
                            policy_promotion_evaluation.sample_games,
                            policy_promotion_evaluation.mean_match_fitness,
                            policy_promotion_evaluation.mean_play_quality,
                            policy_promotion_evaluation.rejection_reasons.join("|")
                        );
                        policy_promotion_summaries.clear();
                        if write_neural_checkpoint_for_rejected_promotion {
                            let output_policy_version_id = Uuid::new_v4().to_string();
                            let checkpoint_generation =
                                pg_generation.max(game.starting_policy_generation);
                            let version_label = soccer_learning_pg_version_label_with_suffix(
                                &run_id,
                                shard_index,
                                game.episode_summary.episode,
                                "neural-checkpoint",
                            );
                            pg_policy_version_buffer.push(PendingPostgresPolicyVersion {
                                id: output_policy_version_id.clone(),
                                parent_policy_version_id: pg_game_base_policy_version_id.clone(),
                                generation: checkpoint_generation,
                                version_label,
                                source_kind: SOCCER_POLICY_SOURCE_NEURAL_CHECKPOINT,
                                status: SOCCER_POLICY_STATUS_ARCHIVED,
                                comparative_promotion_approved: false,
                                config: config.clone(),
                                home_options: options.clone(),
                                away_options: options.clone(),
                                policies: policies.clone(),
                                fitness: policy_promotion_evaluation.mean_match_fitness,
                                neural_network: latest_neural_network.clone(),
                                search_metadata: Some(serde_json::json!({
                                    "neuralCheckpoint": {
                                        "reason": "incumbent-promotion-held",
                                        "completedGames": completed_episode,
                                        "policyGeneration": pg_generation,
                                        "neuralFingerprint": latest_neural_network_fingerprint,
                                    },
                                    "promotion": policy_promotion_search_metadata(
                                        curriculum_stage_label,
                                        &policy_promotion_evaluation,
                                    )
                                })),
                            });
                            pg_last_neural_checkpoint_fingerprint =
                                latest_neural_network_fingerprint;
                            println!(
                                "policy_promotion_rejected_neural_checkpoint_written completed_games={} curriculum_stage={} policy_version={} generation={} neural_fingerprint={}",
                                completed_episode,
                                curriculum_stage_label,
                                output_policy_version_id,
                                checkpoint_generation,
                                latest_neural_network_fingerprint
                                    .map(|fingerprint| fingerprint.to_string())
                                    .unwrap_or_else(|| "none".to_string())
                            );
                        }
                        if policy_promotion_local_best_rollback
                            && latest_neural_network.is_some()
                            && completed_episode >= policy_promotion_local_best_restore_min_games
                        {
                            if neural_population_accept_protection_blocks_local_best_restore(
                                neural_population_accept_restore_protection_games_remaining,
                            ) {
                                println!(
                                    "policy_promotion_local_best_restore_skipped completed_games={} reason=population_accept_restore_protection remaining={}",
                                    completed_episode,
                                    neural_population_accept_restore_protection_games_remaining
                                );
                            } else {
                                let local_best_restore = local_neural_promotion_trial_best
                                .as_ref()
                                .filter(|best| {
                                    policy_promotion_evaluation_regresses_from_local_best(
                                        &policy_promotion_evaluation,
                                        best,
                                        policy_promotion_local_best_restore_max_mean_fitness_regression,
                                        policy_promotion_local_best_restore_max_play_quality_regression,
                                    )
                                })
                                .cloned();
                                if let Some(best) = local_best_restore {
                                    let regressed_mean_match_fitness =
                                        policy_promotion_evaluation.mean_match_fitness;
                                    let regressed_mean_play_quality =
                                        policy_promotion_evaluation.mean_play_quality;
                                    config = best.config.clone();
                                    tactical_learning = best.tactical_learning.clone();
                                    config.tactical_learning = tactical_learning.clone();
                                    policies = best.policies.clone();
                                    latest_neural_network = best.neural_network.clone();
                                    latest_policy_arc_cache = None;
                                    latest_neural_network_arc_cache = None;
                                    local_tactical_evolved_since_pg_refresh = true;
                                    let restored_world_model_steps =
                                        install_carried_world_model_snapshot(
                                            best.carried_world_model.clone(),
                                        );
                                    let backoff = policy_promotion_local_best_backoff.then(|| {
                                        apply_local_best_neural_backoff(
                                            &mut config,
                                            policy_promotion_local_best_backoff_factor,
                                            policy_promotion_local_best_backoff_min_learning_rate,
                                            policy_promotion_local_best_backoff_min_blend_lambda,
                                        )
                                    });
                                    println!(
                                    "policy_promotion_local_best_restored completed_games={} restored_from_games={} regressed_mean_match_fitness={:.4} restored_mean_match_fitness={:.4} regressed_mean_play_quality={:.4} restored_mean_play_quality={:.4}",
                                    completed_episode,
                                    best.completed_games,
                                    regressed_mean_match_fitness,
                                    best.evaluation.mean_match_fitness,
                                    regressed_mean_play_quality,
                                    best.evaluation.mean_play_quality,
                                );
                                    println!(
                                    "policy_promotion_local_best_world_model_restored completed_games={} restored_from_games={} training_steps={}",
                                    completed_episode,
                                    best.completed_games,
                                    restored_world_model_steps,
                                );
                                    if let Some((
                                        previous_learning_rate,
                                        next_learning_rate,
                                        previous_blend_lambda,
                                        next_blend_lambda,
                                    )) = backoff
                                    {
                                        if let Some(best) =
                                            local_neural_promotion_trial_best.as_mut()
                                        {
                                            best.config = config.clone();
                                        }
                                        println!(
                                        "policy_promotion_local_best_backoff completed_games={} learning_rate={:.5}->{:.5} neural_blend_lambda={:.3}->{:.3}",
                                        completed_episode,
                                        previous_learning_rate,
                                        next_learning_rate,
                                        previous_blend_lambda,
                                        next_blend_lambda,
                                    );
                                    }
                                } else if local_neural_promotion_trial_best
                                    .as_ref()
                                    .map(|best| {
                                        policy_promotion_evaluation_beats_local_best(
                                            &policy_promotion_evaluation,
                                            best,
                                        )
                                    })
                                    .unwrap_or(true)
                                {
                                    local_neural_promotion_trial_best =
                                        Some(LocalNeuralPromotionTrialBest {
                                            completed_games: completed_episode,
                                            evaluation: policy_promotion_evaluation.clone(),
                                            config: config.clone(),
                                            tactical_learning: tactical_learning.clone(),
                                            policies: policies.clone(),
                                            neural_network: latest_neural_network.clone(),
                                            carried_world_model:
                                                current_carried_world_model_snapshot(),
                                        });
                                    println!(
                                    "policy_promotion_local_best_updated completed_games={} sample_games={} mean_match_fitness={:.4} best_match_fitness={:.4} mean_play_quality={:.4}",
                                    completed_episode,
                                    policy_promotion_evaluation.sample_games,
                                    policy_promotion_evaluation.mean_match_fitness,
                                    policy_promotion_evaluation.best_match_fitness,
                                    policy_promotion_evaluation.mean_play_quality,
                                );
                                }
                            }
                        }
                        if policy_evolution_reset_to_incumbent_after_held_trial {
                            if let (Some(experiment_id), Some(store)) =
                                (pg_experiment_id.as_deref(), pg_store.as_mut())
                            {
                                match reset_policy_to_strongest_promotion_baseline(
                                    store,
                                    experiment_id,
                                    completed_episode,
                                    &options,
                                    &mut config,
                                    &mut tactical_learning,
                                    &mut policies,
                                    &mut latest_neural_network,
                                    &mut pg_base_policy_version_id,
                                    &mut pg_last_policy_version_id,
                                    &mut pg_generation,
                                    &mut pg_base_policy_version_updated_at_micros,
                                    &mut pg_base_policy_fingerprint,
                                    &mut pg_base_neural_network_fingerprint,
                                    &mut pg_base_tactical_learning_fingerprint,
                                    &mut pg_base_policy_promotion_baseline,
                                    &mut local_tactical_evolved_since_pg_refresh,
                                    policy_promotion_gate.min_sample_games,
                                    policy_promotion_baseline_lookback_generations,
                                ) {
                                    Ok(true) => {
                                        evolution_search_samples.clear();
                                        policy_promotion_summaries.clear();
                                        local_evolution_trial_active = false;
                                        local_neural_promotion_trial_best = None;
                                        policy_promotion_recalibration_pending =
                                            policy_promotion_recalibrate_incumbent_on_resume;
                                    }
                                    Ok(false) => {}
                                    Err(err) => {
                                        eprintln!(
                                            "policy_promotion_reset_to_incumbent_error completed_games={} error={}",
                                            completed_episode, err
                                        );
                                    }
                                }
                            }
                        }
                    }
                    pg_last_policy_version_id.clone()
                };
                pg_completed_buffer.push(PendingPostgresCompletedRun {
                    completed_episode,
                    game: completed_learning_game,
                    base_policy_version_id: pg_game_base_policy_version_id,
                    output_policy_version_id,
                    generation: pg_generation,
                });
            }
            if print_completed_games {
                print_completed_game(&game);
            }

            let (game_artifact_path, game_artifact_kind) = if write_game_artifacts {
                let path = game_dir.join(format!(
                    "game-{:06}-seed-{}.json",
                    game.episode_summary.episode + 1,
                    game.episode_summary.seed
                ));
                if game_artifact_mode == "full" {
                    write_json(&path, &game.artifact)?;
                    (
                        Some(path.display().to_string()),
                        Some("soccer-full-team-policy-artifact".to_string()),
                    )
                } else {
                    let compact = compact_game_artifact(&game);
                    write_json(&path, &compact)?;
                    (
                        Some(path.display().to_string()),
                        Some(compact.artifact_kind.clone()),
                    )
                }
            } else {
                (None, None)
            };

            let stats = &game.episode_summary.summary.stats;
            total_home_goals += game.episode_summary.summary.score_home;
            total_away_goals += game.episode_summary.summary.score_away;
            total_shots += stats.shots_home + stats.shots_away;
            total_on_target += stats.shots_on_target_home + stats.shots_on_target_away;
            total_pass_attempts += stats.passes_attempted_home + stats.passes_attempted_away;
            total_pass_completions += stats.passes_completed_home + stats.passes_completed_away;
            total_interceptions += stats.interceptions_home + stats.interceptions_away;
            tactical_summary.merge(&game.artifact.tactical_summary);

            let mut manifest_entry = game_manifest_entry(&game, game_artifact_path);
            manifest_entry.artifact_kind = game_artifact_kind;
            manifest_games.push(manifest_entry);
            if let Some(episode_log) = episode_log.as_mut() {
                write_episode_log(episode_log, &game.episode_summary)?;
            }
            episode_summaries.push(game.episode_summary.clone());
            // Per-game memory diagnostic: locate the unbounded structure behind the
            // 122GB OOM (every other accumulator audited as bounded). Grep
            // `soccer_mem_diag` in the pod logs and watch which column climbs.
            eprintln!(
                "soccer_mem_diag episode={} rss_mb={} home_q={} home_t={} away_q={} away_t={} pg_completed_buf={} evo_samples={} episode_summaries={} manifest_games={}",
                game.episode_summary.episode,
                process_rss_mb(),
                policies.home.q_values.len(),
                policies.home.target_values.len(),
                policies.away.q_values.len(),
                policies.away.target_values.len(),
                pg_completed_buffer.len(),
                evolution_search_samples.len(),
                episode_summaries.len(),
                manifest_games.len()
            );
            eprintln!(
                "soccer_learning_signal_capture episode={} pass_outcome_samples={} defender_line_depth_rows={} midfield_line_depth_rows={} back_four_gate={} midfield_gate={} pitch_value_gate={} loose_ball_commit_gate={} attack_spacing_gate={}",
                game.episode_summary.episode,
                game.pass_outcome_samples.len(),
                game.artifact
                    .tactical_summary
                    .defender_line_depth_training_rows,
                game.artifact
                    .tactical_summary
                    .midfield_line_depth_training_rows,
                env_value("DD_SOCCER_ENABLE_BACK_FOUR_LINE_MODEL").unwrap_or_else(|| "0".to_string()),
                env_value("DD_SOCCER_ENABLE_MIDFIELD_LINE_MODEL").unwrap_or_else(|| "0".to_string()),
                env_value("DD_SOCCER_ENABLE_PITCH_VALUE_REWARD").unwrap_or_else(|| "0".to_string()),
                env_value("DD_SOCCER_ENABLE_LOOSE_BALL_COMMIT_MODEL").unwrap_or_else(|| "0".to_string()),
                env_value("DD_SOCCER_ENABLE_LEARNED_SPACING_TARGET").unwrap_or_else(|| "0".to_string())
            );
            if let Some(episode_log) = episode_log.as_mut() {
                if episode_log_flush_interval_games > 0
                    && episode_summaries.len() % episode_log_flush_interval_games == 0
                {
                    episode_log.flush()?;
                }
            }
        }

        push_evolution_search_samples(
            &mut evolution_search_samples,
            &completed_games,
            evolution_window_games,
        );
        let completed_after_batch = episode_summaries.len();
        let should_evolve = evolution_enabled
            && !completed_games.is_empty()
            && !evolution_search_samples.is_empty()
            && (completed_after_batch >= games
                || completed_after_batch % evolution_interval_games == 0);
        if should_evolve {
            let mut ranked_samples = evolution_search_samples
                .iter()
                .enumerate()
                .filter(|(_, sample)| sample.fitness.is_finite())
                .map(|(idx, sample)| (idx, sample.fitness))
                .collect::<Vec<_>>();
            ranked_samples.sort_by(|left, right| {
                right
                    .1
                    .partial_cmp(&left.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let elite_count = evolution_elite_games.min(ranked_samples.len()).max(1);
            let best_fitness = ranked_samples
                .first()
                .map(|(_, fitness)| *fitness)
                .unwrap_or(0.0);
            let mut parents = Vec::with_capacity(elite_count + 1);
            parents.push((&policies, best_fitness));
            for (sample_index, fitness) in ranked_samples.iter().take(elite_count) {
                parents.push((&evolution_search_samples[*sample_index].policies, *fitness));
            }
            let tactical_parents = tactical_genome_parents_from_evolution_samples(
                &evolution_search_samples,
                &ranked_samples,
                elite_count,
            );
            let mut batch_evolution_options = evolution_options;
            batch_evolution_options.seed = batch_evolution_options
                .seed
                .wrapping_add(completed_after_batch as u64)
                .wrapping_add((shard_index as u64) << 32);
            let evolution_promotion_summary_refs = evolution_search_samples
                .iter()
                .map(|sample| &sample.summary)
                .collect::<Vec<_>>();
            let mut policy_promotion_evaluation =
                evaluate_soccer_policy_promotion_gate_for_learning_objective(
                    &evolution_promotion_summary_refs,
                    policy_promotion_gate,
                    analytic_neural_opponent,
                );
            apply_policy_promotion_incumbent_gate(
                &mut policy_promotion_evaluation,
                pg_base_policy_promotion_baseline,
                policy_promotion_compare_incumbent,
                policy_promotion_min_mean_fitness_delta,
                policy_promotion_max_mean_fitness_regression,
                policy_promotion_max_play_quality_regression,
            );
            let evaluated_local_trial = local_evolution_trial_active;
            let candidate_promoted = policy_promotion_evaluation.eligible;
            let promoted_played_local_trial = evaluated_local_trial
                && candidate_promoted
                && policy_evolution_require_trial_before_write;
            let curriculum_stage = soccer_learning_curriculum_stage_for_completed_games(
                completed_after_batch,
                &curriculum_config,
            );
            let curriculum_stage_label = curriculum_stage.as_str();
            let promoted_trial_policies = promoted_played_local_trial.then(|| policies.clone());
            let promoted_trial_config = promoted_played_local_trial.then(|| config.clone());
            let promoted_trial_neural_network =
                promoted_played_local_trial.then(|| latest_neural_network.clone());
            let evolved_policies = evolve_soccer_team_policies(&parents, batch_evolution_options)
                .map_err(invalid_data)?;
            let previous_tactical_learning = tactical_learning.clone();
            let evolved_tactical_learning = evolve_soccer_tactical_learning_weights_from_genomes(
                &tactical_learning,
                &tactical_parents,
                batch_evolution_options,
            );
            validate_tactical_learning_weights(&evolved_tactical_learning)?;
            let mut evolved_config = config.clone();
            evolved_config.tactical_learning = evolved_tactical_learning.clone();
            let apply_local_evolution = candidate_promoted
                || (policy_evolution_apply_when_promotion_held && !evaluated_local_trial);
            if apply_local_evolution {
                policies = evolved_policies.clone();
                tactical_learning = evolved_tactical_learning.clone();
                config.tactical_learning = tactical_learning.clone();
                local_tactical_evolved_since_pg_refresh = true;
                local_evolution_trial_active =
                    !candidate_promoted || policy_evolution_require_trial_before_write;
                if candidate_promoted {
                    println!(
                        "policy_evolved games_completed={} curriculum_stage={} sample_window_games={} samples={} elite_games={} best_fitness={:.4} mutation_rate={:.4} mutation_scale={:.4} crossover_rate={:.4} exploration_rate={:.4} exploration_scale={:.4} population_size={} require_trial_before_write={}",
                        completed_after_batch,
                        curriculum_stage_label,
                        evolution_window_games,
                        evolution_search_samples.len(),
                        elite_count,
                        best_fitness,
                        batch_evolution_options.mutation_rate,
                        batch_evolution_options.mutation_scale,
                        batch_evolution_options.crossover_rate,
                        batch_evolution_options.exploration_rate,
                        batch_evolution_options.exploration_scale,
                        batch_evolution_options.population_size,
                        policy_evolution_require_trial_before_write
                    );
                } else {
                    println!(
                        "policy_evolved_local_trial games_completed={} curriculum_stage={} sample_window_games={} samples={} elite_games={} best_fitness={:.4} reasons={}",
                        completed_after_batch,
                        curriculum_stage_label,
                        evolution_window_games,
                        evolution_search_samples.len(),
                        elite_count,
                        best_fitness,
                        policy_promotion_evaluation.rejection_reasons.join("|")
                    );
                }
                println!(
                    "tactical_weights_evolved games_completed={} sample_window_games={} samples={} population_size={} attack_width_delta={:.3}->{:.3} attack_flank_lane={:.3}->{:.3} defense_contract_delta={:.3}->{:.3}",
                    completed_after_batch,
                    evolution_window_games,
                    evolution_search_samples.len(),
                    batch_evolution_options.population_size,
                    previous_tactical_learning.attack_width_delta_weight,
                    tactical_learning.attack_width_delta_weight,
                    previous_tactical_learning.attack_flank_lane_weight,
                    tactical_learning.attack_flank_lane_weight,
                    previous_tactical_learning.defense_contract_delta_weight,
                    tactical_learning.defense_contract_delta_weight
                );
            } else {
                println!(
                    "policy_evolution_held games_completed={} curriculum_stage={} sample_window_games={} samples={} elite_games={} best_fitness={:.4} reasons={}",
                    completed_after_batch,
                    curriculum_stage_label,
                    evolution_window_games,
                    evolution_search_samples.len(),
                    elite_count,
                    best_fitness,
                    policy_promotion_evaluation.rejection_reasons.join("|")
                );
            }
            let _ = std::io::stdout().flush();

            if pg_experiment_id.is_some() {
                let skip_incumbent_rejected_policy_version =
                    policy_promotion_rejected_by_incumbent(&policy_promotion_evaluation);
                if skip_incumbent_rejected_policy_version {
                    println!(
                        "policy_evolution_rejected_not_written games_completed={} curriculum_stage={} sample_games={} mean_match_fitness={:.4} mean_play_quality={:.4} reasons={}",
                        completed_after_batch,
                        curriculum_stage_label,
                        policy_promotion_evaluation.sample_games,
                        policy_promotion_evaluation.mean_match_fitness,
                        policy_promotion_evaluation.mean_play_quality,
                        policy_promotion_evaluation.rejection_reasons.join("|")
                    );
                    if policy_evolution_reset_to_incumbent_after_held_trial && evaluated_local_trial
                    {
                        if let (Some(experiment_id), Some(store)) =
                            (pg_experiment_id.as_deref(), pg_store.as_mut())
                        {
                            match reset_policy_to_strongest_promotion_baseline(
                                store,
                                experiment_id,
                                completed_after_batch,
                                &options,
                                &mut config,
                                &mut tactical_learning,
                                &mut policies,
                                &mut latest_neural_network,
                                &mut pg_base_policy_version_id,
                                &mut pg_last_policy_version_id,
                                &mut pg_generation,
                                &mut pg_base_policy_version_updated_at_micros,
                                &mut pg_base_policy_fingerprint,
                                &mut pg_base_neural_network_fingerprint,
                                &mut pg_base_tactical_learning_fingerprint,
                                &mut pg_base_policy_promotion_baseline,
                                &mut local_tactical_evolved_since_pg_refresh,
                                policy_promotion_gate.min_sample_games,
                                policy_promotion_baseline_lookback_generations,
                            ) {
                                Ok(true) => {
                                    evolution_search_samples.clear();
                                    policy_promotion_summaries.clear();
                                    local_evolution_trial_active = false;
                                    policy_promotion_recalibration_pending =
                                        policy_promotion_recalibrate_incumbent_on_resume;
                                }
                                Ok(false) => {}
                                Err(err) => {
                                    eprintln!(
                                        "policy_local_trial_reset_error games_completed={} error={}",
                                        completed_after_batch, err
                                    );
                                }
                            }
                        }
                    }
                } else if promoted_played_local_trial {
                    let next_generation = pg_generation.saturating_add(1);
                    let output_policy_version_id = Uuid::new_v4().to_string();
                    let version_label = soccer_learning_pg_version_label_with_suffix(
                        &run_id,
                        shard_index,
                        completed_after_batch.saturating_sub(1),
                        "evolution-trial",
                    );
                    let policy_version_status =
                        policy_version_status_for_promotion_gate(&policy_promotion_evaluation);
                    let promoted_trial_policies =
                        promoted_trial_policies.expect("promoted trial policies");
                    let promoted_trial_config =
                        promoted_trial_config.expect("promoted trial config");
                    let promoted_trial_neural_network = promoted_trial_neural_network.flatten();
                    let promoted_trial_policy_fingerprint =
                        soccer_team_q_policies_fingerprint(&promoted_trial_policies);
                    let promoted_trial_neural_network_fingerprint = promoted_trial_neural_network
                        .as_ref()
                        .map(soccer_neural_network_snapshot_fingerprint);
                    let promoted_trial_tactical_learning_fingerprint =
                        Some(soccer_tactical_learning_weights_fingerprint(
                            &promoted_trial_config.tactical_learning,
                        ));
                    pg_policy_version_buffer.push(PendingPostgresPolicyVersion {
                        id: output_policy_version_id.clone(),
                        parent_policy_version_id: pg_base_policy_version_id.clone(),
                        generation: next_generation,
                        version_label,
                        source_kind: SOCCER_POLICY_SOURCE_EVOLUTION,
                        status: policy_version_status,
                        comparative_promotion_approved: false,
                        config: promoted_trial_config,
                        home_options: options.clone(),
                        away_options: options.clone(),
                        policies: promoted_trial_policies,
                        fitness: policy_promotion_evaluation.mean_match_fitness,
                        neural_network: promoted_trial_neural_network,
                        search_metadata: Some(serde_json::json!({
                            "algorithm": "evolutionary-policy-trial",
                            "completedGames": completed_after_batch,
                            "eliteGames": elite_count,
                            "windowGames": evolution_window_games,
                            "sampleCount": evolution_search_samples.len(),
                            "bestFitness": best_fitness,
                            "curriculumStage": curriculum_stage_label,
                            "candidatePromoted": candidate_promoted,
                            "trialPromoted": true,
                            "promotion": policy_promotion_search_metadata(
                                curriculum_stage_label,
                                &policy_promotion_evaluation,
                            ),
                            "options": batch_evolution_options
                        })),
                    });
                    if policy_version_status == SOCCER_POLICY_STATUS_ACTIVE {
                        pg_base_policy_version_id = Some(output_policy_version_id.clone());
                        pg_last_policy_version_id = Some(output_policy_version_id.clone());
                        pg_generation = next_generation;
                        pg_base_policy_version_updated_at_micros = 0;
                        pg_base_policy_fingerprint = Some(promoted_trial_policy_fingerprint);
                        pg_base_neural_network_fingerprint =
                            promoted_trial_neural_network_fingerprint;
                        pg_base_tactical_learning_fingerprint =
                            promoted_trial_tactical_learning_fingerprint;
                        pg_base_policy_promotion_baseline =
                            retain_stronger_policy_promotion_baseline(
                                pg_base_policy_promotion_baseline,
                                policy_promotion_incumbent_baseline_from_evaluation(
                                    &policy_promotion_evaluation,
                                ),
                            );
                    }
                    println!(
                        "policy_evolution_trial_promoted_written games_completed={} curriculum_stage={} policy_version={} generation={} sample_games={} mean_match_fitness={:.4} best_match_fitness={:.4} mean_play_quality={:.4} new_local_trial_active={}",
                        completed_after_batch,
                        curriculum_stage_label,
                        output_policy_version_id,
                        next_generation,
                        policy_promotion_evaluation.sample_games,
                        policy_promotion_evaluation.mean_match_fitness,
                        policy_promotion_evaluation.best_match_fitness,
                        policy_promotion_evaluation.mean_play_quality,
                        local_evolution_trial_active
                    );
                } else if policy_evolution_require_trial_before_write {
                    println!(
                        "policy_evolution_trial_not_written games_completed={} curriculum_stage={} sample_games={} mean_match_fitness={:.4} mean_play_quality={:.4} evaluated_local_trial={} candidate_promoted={}",
                        completed_after_batch,
                        curriculum_stage_label,
                        policy_promotion_evaluation.sample_games,
                        policy_promotion_evaluation.mean_match_fitness,
                        policy_promotion_evaluation.mean_play_quality,
                        evaluated_local_trial,
                        candidate_promoted
                    );
                } else {
                    let next_generation = pg_generation.saturating_add(1);
                    let output_policy_version_id = Uuid::new_v4().to_string();
                    let version_label = soccer_learning_pg_version_label_with_suffix(
                        &run_id,
                        shard_index,
                        completed_after_batch.saturating_sub(1),
                        "evolution",
                    );
                    // Frozen-anchor gate (HEAD): gate the evolution-sourced candidate through the
                    // held-out anchor before it can be written active (no-op when the gate is off).
                    let policy_version_status = apply_anchor_promotion_gate(
                        &anchor_promotion_gate,
                        policy_version_status_for_promotion_gate(&policy_promotion_evaluation),
                        true,
                        latest_neural_network.as_ref(),
                        &mut anchor_neural_network,
                        &mut anchor_gate_runner,
                        &mut anchor_gate_write_index,
                        completed_after_batch as u32,
                        &format!("evolution completed_games={completed_after_batch}"),
                    );
                    pg_policy_version_buffer.push(PendingPostgresPolicyVersion {
                        id: output_policy_version_id.clone(),
                        parent_policy_version_id: pg_base_policy_version_id.clone(),
                        generation: next_generation,
                        version_label,
                        source_kind: SOCCER_POLICY_SOURCE_EVOLUTION,
                        status: policy_version_status,
                        comparative_promotion_approved: false,
                        config: evolved_config,
                        home_options: options.clone(),
                        away_options: options.clone(),
                        policies: evolved_policies.clone(),
                        fitness: policy_promotion_evaluation.mean_match_fitness,
                        neural_network: latest_neural_network.clone(),
                        search_metadata: Some(run_evolution_search_metadata(
                            completed_after_batch,
                            elite_count,
                            evolution_window_games,
                            evolution_search_samples.len(),
                            best_fitness,
                            batch_evolution_options,
                            &previous_tactical_learning,
                            &evolved_tactical_learning,
                            curriculum_stage_label,
                            &policy_promotion_evaluation,
                            candidate_promoted,
                        )),
                    });
                    if policy_version_status == SOCCER_POLICY_STATUS_ACTIVE {
                        local_evolution_trial_active = false;
                        pg_base_policy_version_id = Some(output_policy_version_id.clone());
                        pg_last_policy_version_id = Some(output_policy_version_id);
                        pg_generation = next_generation;
                        pg_base_policy_version_updated_at_micros = 0;
                        pg_base_policy_fingerprint =
                            Some(soccer_team_q_policies_fingerprint(&policies));
                        pg_base_neural_network_fingerprint = latest_neural_network
                            .as_ref()
                            .map(soccer_neural_network_snapshot_fingerprint);
                        pg_base_tactical_learning_fingerprint = Some(
                            soccer_tactical_learning_weights_fingerprint(&config.tactical_learning),
                        );
                        pg_base_policy_promotion_baseline =
                            retain_stronger_policy_promotion_baseline(
                                pg_base_policy_promotion_baseline,
                                policy_promotion_incumbent_baseline_from_evaluation(
                                    &policy_promotion_evaluation,
                                ),
                            );
                    }
                }
            }
        }

        let prune_summary = policies.prune(
            max_policy_entries_per_team,
            max_policy_target_entries_per_team,
            min_policy_visits,
        );
        if prune_summary.removed_entries() > 0 {
            println!(
                "policy_pruned games_completed={} removed={} home_actions={}->{} away_actions={}->{} home_targets={}->{} away_targets={}->{}",
                episode_summaries.len(),
                prune_summary.removed_entries(),
                prune_summary.home.action_entries_before,
                prune_summary.home.action_entries_after,
                prune_summary.away.action_entries_before,
                prune_summary.away.action_entries_after,
                prune_summary.home.target_entries_before,
                prune_summary.home.target_entries_after,
                prune_summary.away.target_entries_before,
                prune_summary.away.target_entries_after,
            );
            let _ = std::io::stdout().flush();
        }

        next_episode += batch_size;

        let should_checkpoint = write_checkpoint_artifacts
            && checkpoint_interval_games > 0
            && (next_episode >= games
                || next_episode.saturating_sub(last_checkpoint_episode)
                    >= checkpoint_interval_games);
        if let Some(experiment_id) = pg_experiment_id.as_deref() {
            let should_flush_completed_runs = !pg_completed_buffer.is_empty()
                && (next_episode >= games
                    || should_checkpoint
                    || pg_completed_buffer.len() >= pg_completed_run_batch_games);
            if should_flush_completed_runs {
                if soccer_should_flush_postgres_policy_versions_for_new_sim(
                    pg_refresh_for_new_sims,
                    pg_flush_policy_versions_before_new_sim,
                    pg_policy_version_buffer.len(),
                ) {
                    if let Some(store) = pg_store.as_mut() {
                        flush_postgres_policy_versions_for_new_sims(
                            store,
                            experiment_id,
                            &pg_runner_id,
                            &mut pg_policy_version_buffer,
                            shard_index,
                            shard_count,
                        )?;
                    }
                }
                pg_persisted_games += if let Some(writer) = pg_completed_writer.as_mut() {
                    writer
                        .enqueue(
                            experiment_id,
                            &pg_runner_id,
                            &mut pg_policy_version_buffer,
                            &mut pg_completed_buffer,
                            pg_completed_run_retention_games,
                            shard_index,
                            shard_count,
                        )
                        .map_err(invalid_data)?
                } else if let Some(store) = pg_store.as_mut() {
                    flush_postgres_completed_runs(
                        store,
                        experiment_id,
                        &pg_runner_id,
                        &mut pg_policy_version_buffer,
                        &mut pg_completed_buffer,
                        pg_completed_run_retention_games,
                        shard_index,
                        shard_count,
                    )?
                } else {
                    0
                };
            }
        }
        if should_checkpoint {
            let checkpoint_summary_refs = policy_promotion_summaries.iter().collect::<Vec<_>>();
            let checkpoint_promotion_evaluation =
                evaluate_soccer_policy_promotion_gate_for_learning_objective(
                    &checkpoint_summary_refs,
                    policy_promotion_gate,
                    analytic_neural_opponent,
                );
            let publish_live_checkpoint = should_publish_live_checkpoint(
                checkpoint_artifact_best_only,
                &checkpoint_promotion_evaluation,
                pg_base_policy_promotion_baseline,
                local_neural_promotion_trial_best.as_ref(),
                latest_neural_network.is_some(),
            );
            let checkpoint_write_path = if publish_live_checkpoint {
                checkpoint_artifact_path.clone()
            } else {
                candidate_checkpoint_path(&checkpoint_artifact_path)
            };
            let checkpoint_learned_params_path = if publish_live_checkpoint {
                learned_params_path.clone()
            } else {
                candidate_checkpoint_path(&learned_params_path)
            };
            let checkpoint_artifact = self_play_artifact_from_policies(
                config.clone(),
                options.clone(),
                tactical_summary.clone(),
                episode_summaries.clone(),
                &policies,
            );
            let checkpoint_export = compact_training_artifact_for_export(
                &checkpoint_artifact,
                artifact_max_entries_per_policy,
            );
            write_json(&checkpoint_write_path, &checkpoint_export)?;
            if write_learned_params_artifact {
                let checkpoint_params =
                    SoccerSelfPlayLearnedParams::from_training_artifact_with_neural_network(
                        &checkpoint_artifact,
                        latest_neural_network.clone(),
                    );
                write_json(&checkpoint_learned_params_path, &checkpoint_params)?;
            } else {
                println!(
                    "checkpoint_learned_params=disabled games_completed={} artifact={}",
                    episode_summaries.len(),
                    checkpoint_learned_params_path.display()
                );
            }
            if let Some(sidecar_path) = write_neural_sidecar_for_policy_artifact(
                &checkpoint_write_path,
                latest_neural_network.as_ref(),
            )? {
                println!(
                    "checkpoint_neural_sidecar games_completed={} artifact={}",
                    episode_summaries.len(),
                    sidecar_path.display()
                );
            }
            if checkpoint_artifact_best_only {
                if publish_live_checkpoint {
                    if let Some(metadata_path) = write_local_policy_promotion_best(
                        &checkpoint_artifact_path,
                        episode_summaries.len(),
                        &checkpoint_promotion_evaluation,
                    )? {
                        println!(
                            "checkpoint_local_best_metadata_written games_completed={} artifact={}",
                            episode_summaries.len(),
                            metadata_path.display()
                        );
                    }
                    local_neural_promotion_trial_best = Some(LocalNeuralPromotionTrialBest {
                        completed_games: episode_summaries.len(),
                        evaluation: checkpoint_promotion_evaluation.clone(),
                        config: config.clone(),
                        tactical_learning: tactical_learning.clone(),
                        policies: policies.clone(),
                        neural_network: latest_neural_network.clone(),
                        carried_world_model: current_carried_world_model_snapshot(),
                    });
                    println!(
                        "checkpoint_local_best_published games_completed={} sample_games={} mean_match_fitness={:.4} best_match_fitness={:.4} mean_play_quality={:.4} artifact={}",
                        episode_summaries.len(),
                        checkpoint_promotion_evaluation.sample_games,
                        checkpoint_promotion_evaluation.mean_match_fitness,
                        checkpoint_promotion_evaluation.best_match_fitness,
                        checkpoint_promotion_evaluation.mean_play_quality,
                        checkpoint_write_path.display()
                    );
                    write_json(&training_best_checkpoint_artifact_path, &checkpoint_export)?;
                    if let Some(metadata_path) = write_local_policy_promotion_best(
                        &training_best_checkpoint_artifact_path,
                        episode_summaries.len(),
                        &checkpoint_promotion_evaluation,
                    )? {
                        println!(
                            "checkpoint_training_best_metadata_written games_completed={} artifact={}",
                            episode_summaries.len(),
                            metadata_path.display()
                        );
                    }
                    if write_learned_params_artifact {
                        let training_best_params =
                            SoccerSelfPlayLearnedParams::from_training_artifact_with_neural_network(
                                &checkpoint_artifact,
                                latest_neural_network.clone(),
                            );
                        write_json(&training_best_learned_params_path, &training_best_params)?;
                    }
                    if let Some(sidecar_path) = write_neural_sidecar_for_policy_artifact(
                        &training_best_checkpoint_artifact_path,
                        latest_neural_network.as_ref(),
                    )? {
                        println!(
                            "checkpoint_training_best_neural_sidecar games_completed={} artifact={}",
                            episode_summaries.len(),
                            sidecar_path.display()
                        );
                    }
                    println!(
                        "checkpoint_training_best_synced_from_publish games_completed={} artifact={}",
                        episode_summaries.len(),
                        training_best_checkpoint_artifact_path.display()
                    );
                } else {
                    let hold_reasons = if latest_neural_network.is_none() {
                        "missing_neural_snapshot".to_string()
                    } else if checkpoint_promotion_evaluation.rejection_reasons.is_empty() {
                        "local_best_not_improved".to_string()
                    } else {
                        checkpoint_promotion_evaluation.rejection_reasons.join("|")
                    };
                    println!(
                        "checkpoint_local_best_held games_completed={} sample_games={} mean_match_fitness={:.4} best_match_fitness={:.4} mean_play_quality={:.4} reasons={} candidate_artifact={} live_artifact={}",
                        episode_summaries.len(),
                        checkpoint_promotion_evaluation.sample_games,
                        checkpoint_promotion_evaluation.mean_match_fitness,
                        checkpoint_promotion_evaluation.best_match_fitness,
                        checkpoint_promotion_evaluation.mean_play_quality,
                        hold_reasons,
                        checkpoint_write_path.display(),
                        checkpoint_artifact_path.display()
                    );
                    let batch_publish_validation_beats_local_best =
                        batch_publish_validation_evaluation
                            .as_ref()
                            .map(|validation_evaluation| {
                                local_neural_promotion_trial_best
                                    .as_ref()
                                    .map(|best| {
                                        policy_promotion_evaluation_beats_local_best(
                                            validation_evaluation,
                                            best,
                                        )
                                    })
                                    .unwrap_or(true)
                            })
                            .unwrap_or(false);
                    let batch_publish_validation_has_local_best_samples =
                        batch_publish_validation_evaluation
                            .as_ref()
                            .map(|validation_evaluation| {
                                validation_evaluation.sample_games
                                    >= policy_promotion_gate.min_sample_games
                            })
                            .unwrap_or(false);
                    let batch_publish_validation_can_update_local_best =
                        batch_kept_publish_validated_training_snapshot
                            && batch_publish_validation_beats_local_best
                            && batch_publish_validation_has_local_best_samples;
                    let batch_publish_validation_can_update_resume_best =
                        validated_snapshot_can_update_resume_only_training_best(
                            batch_kept_publish_validated_training_snapshot,
                            batch_publish_validation_beats_local_best,
                            batch_publish_validation_has_local_best_samples,
                            batch_publish_validation_evaluation
                                .as_ref()
                                .map(|evaluation| evaluation.mean_match_fitness)
                                .unwrap_or(f64::NEG_INFINITY),
                        );
                    if batch_publish_validation_can_update_local_best {
                        let validation_evaluation = batch_publish_validation_evaluation
                            .as_ref()
                            .expect("validated snapshot improvement should carry evaluation");
                        local_neural_promotion_trial_best = Some(LocalNeuralPromotionTrialBest {
                            completed_games: episode_summaries.len(),
                            evaluation: validation_evaluation.clone(),
                            config: config.clone(),
                            tactical_learning: tactical_learning.clone(),
                            policies: policies.clone(),
                            neural_network: latest_neural_network.clone(),
                            carried_world_model: current_carried_world_model_snapshot(),
                        });
                        println!(
                            "checkpoint_training_best_validation_local_best_updated games_completed={} sample_games={} mean_match_fitness={:.4} best_match_fitness={:.4} mean_play_quality={:.4}",
                            episode_summaries.len(),
                            validation_evaluation.sample_games,
                            validation_evaluation.mean_match_fitness,
                            validation_evaluation.best_match_fitness,
                            validation_evaluation.mean_play_quality
                        );
                        write_json(&training_best_checkpoint_artifact_path, &checkpoint_export)?;
                        if let Some(metadata_path) = write_local_policy_promotion_best(
                            &training_best_checkpoint_artifact_path,
                            episode_summaries.len(),
                            validation_evaluation,
                        )? {
                            println!(
                                "checkpoint_training_best_validation_metadata_written games_completed={} artifact={}",
                                episode_summaries.len(),
                                metadata_path.display()
                            );
                        }
                        if write_learned_params_artifact {
                            let training_best_params =
                                SoccerSelfPlayLearnedParams::from_training_artifact_with_neural_network(
                                    &checkpoint_artifact,
                                    latest_neural_network.clone(),
                                );
                            write_json(&training_best_learned_params_path, &training_best_params)?;
                        }
                        if let Some(sidecar_path) = write_neural_sidecar_for_policy_artifact(
                            &training_best_checkpoint_artifact_path,
                            latest_neural_network.as_ref(),
                        )? {
                            println!(
                                "checkpoint_training_best_synced_from_snapshot_validation_neural_sidecar games_completed={} artifact={}",
                                episode_summaries.len(),
                                sidecar_path.display()
                            );
                        }
                        println!(
                            "checkpoint_training_best_synced_from_snapshot_validation games_completed={} artifact={}",
                            episode_summaries.len(),
                            training_best_checkpoint_artifact_path.display()
                        );
                    } else if batch_publish_validation_can_update_resume_best {
                        let validation_evaluation = batch_publish_validation_evaluation
                            .as_ref()
                            .expect("validated snapshot improvement should carry evaluation");
                        local_neural_promotion_trial_best = Some(LocalNeuralPromotionTrialBest {
                            completed_games: episode_summaries.len(),
                            evaluation: validation_evaluation.clone(),
                            config: config.clone(),
                            tactical_learning: tactical_learning.clone(),
                            policies: policies.clone(),
                            neural_network: latest_neural_network.clone(),
                            carried_world_model: current_carried_world_model_snapshot(),
                        });
                        println!(
                            "checkpoint_training_best_validation_resume_only_local_best_updated games_completed={} validation_sample_games={} live_min_sample_games={} mean_match_fitness={:.4} best_match_fitness={:.4} mean_play_quality={:.4}",
                            episode_summaries.len(),
                            validation_evaluation.sample_games,
                            policy_promotion_gate.min_sample_games,
                            validation_evaluation.mean_match_fitness,
                            validation_evaluation.best_match_fitness,
                            validation_evaluation.mean_play_quality
                        );
                        write_training_best_resume_only_checkpoint(
                            "snapshot_validation_below_live_sample_floor",
                            episode_summaries.len(),
                            episode_summaries.len(),
                            &checkpoint_artifact,
                            latest_neural_network.as_ref(),
                            validation_evaluation,
                            &training_best_checkpoint_artifact_path,
                            &training_best_learned_params_path,
                            write_learned_params_artifact,
                            artifact_max_entries_per_policy,
                        )?;
                    } else if batch_kept_publish_validated_training_snapshot {
                        let validation_fitness = batch_publish_validation_evaluation
                            .as_ref()
                            .map(|evaluation| format!("{:.4}", evaluation.mean_match_fitness))
                            .unwrap_or_else(|| "unavailable".to_string());
                        let incumbent_fitness = local_neural_promotion_trial_best
                            .as_ref()
                            .map(|best| format!("{:.4}", best.evaluation.mean_match_fitness))
                            .unwrap_or_else(|| "none".to_string());
                        let validation_sample_games = batch_publish_validation_evaluation
                            .as_ref()
                            .map(|evaluation| evaluation.sample_games.to_string())
                            .unwrap_or_else(|| "unavailable".to_string());
                        let sync_skip_reason = if batch_publish_validation_beats_local_best
                            && !batch_publish_validation_has_local_best_samples
                        {
                            "validation_sample_games_below_local_best_min"
                        } else {
                            "validation_not_improved"
                        };
                        println!(
                            "checkpoint_training_best_snapshot_validation_not_synced games_completed={} validation_sample_games={} min_sample_games={} validation_mean_match_fitness={} incumbent_mean_match_fitness={} reason={}",
                            episode_summaries.len(),
                            validation_sample_games,
                            policy_promotion_gate.min_sample_games,
                            validation_fitness,
                            incumbent_fitness,
                            sync_skip_reason
                        );
                    }
                    if policy_promotion_local_best_rollback {
                        let restored = if batch_publish_validation_can_update_local_best {
                            println!(
                                "policy_promotion_local_best_restore_skipped context=checkpoint-local-best-held completed_games={} reason=heldout_training_snapshot_kept",
                                episode_summaries.len()
                            );
                            false
                        } else if neural_population_accept_protection_blocks_local_best_restore(
                            neural_population_accept_restore_protection_games_remaining,
                        ) {
                            println!(
                                "policy_promotion_local_best_restore_skipped context=checkpoint-local-best-held completed_games={} reason=population_accept_restore_protection remaining={}",
                                episode_summaries.len(),
                                neural_population_accept_restore_protection_games_remaining
                            );
                            false
                        } else {
                            restore_local_best_after_held_promotion(
                                "checkpoint-local-best-held",
                                episode_summaries.len(),
                                &checkpoint_promotion_evaluation,
                                &mut local_neural_promotion_trial_best,
                                &mut config,
                                &mut tactical_learning,
                                &mut policies,
                                &mut latest_neural_network,
                                &mut latest_policy_arc_cache,
                                &mut latest_neural_network_arc_cache,
                                &mut local_tactical_evolved_since_pg_refresh,
                                policy_promotion_local_best_backoff,
                                policy_promotion_local_best_backoff_factor,
                                policy_promotion_local_best_backoff_min_learning_rate,
                                policy_promotion_local_best_backoff_min_blend_lambda,
                                policy_promotion_local_best_restore_min_games,
                                policy_promotion_local_best_restore_max_mean_fitness_regression,
                                policy_promotion_local_best_restore_max_play_quality_regression,
                            )
                        };
                        if !restored
                            && latest_neural_network.is_some()
                            && policy_promotion_evaluation_can_anchor_local_trial(
                                &checkpoint_promotion_evaluation,
                                policy_promotion_local_best_anchor_held,
                            )
                            && local_neural_promotion_trial_best
                                .as_ref()
                                .map(|best| {
                                    policy_promotion_evaluation_beats_local_best(
                                        &checkpoint_promotion_evaluation,
                                        best,
                                    )
                                })
                                .unwrap_or(true)
                        {
                            local_neural_promotion_trial_best =
                                Some(LocalNeuralPromotionTrialBest {
                                    completed_games: episode_summaries.len(),
                                    evaluation: checkpoint_promotion_evaluation.clone(),
                                    config: config.clone(),
                                    tactical_learning: tactical_learning.clone(),
                                    policies: policies.clone(),
                                    neural_network: latest_neural_network.clone(),
                                    carried_world_model: current_carried_world_model_snapshot(),
                                });
                            println!(
                                "checkpoint_local_trial_best_updated games_completed={} sample_games={} mean_match_fitness={:.4} best_match_fitness={:.4} mean_play_quality={:.4}",
                                episode_summaries.len(),
                                checkpoint_promotion_evaluation.sample_games,
                                checkpoint_promotion_evaluation.mean_match_fitness,
                                checkpoint_promotion_evaluation.best_match_fitness,
                                checkpoint_promotion_evaluation.mean_play_quality,
                            );
                            println!(
                                "checkpoint_local_trial_best_persisting_to_training_best_resume_only games_completed={} reason=held_checkpoint_anchor_not_live_publish",
                                episode_summaries.len()
                            );
                            write_training_best_resume_only_checkpoint(
                                "held_checkpoint_anchor",
                                episode_summaries.len(),
                                episode_summaries.len(),
                                &checkpoint_artifact,
                                latest_neural_network.as_ref(),
                                &checkpoint_promotion_evaluation,
                                &training_best_checkpoint_artifact_path,
                                &training_best_learned_params_path,
                                write_learned_params_artifact,
                                artifact_max_entries_per_policy,
                            )?;
                        } else if restored {
                            let restored_checkpoint_artifact = self_play_artifact_from_policies(
                                config.clone(),
                                options.clone(),
                                tactical_summary.clone(),
                                episode_summaries.clone(),
                                &policies,
                            );
                            let restored_checkpoint_export = compact_training_artifact_for_export(
                                &restored_checkpoint_artifact,
                                artifact_max_entries_per_policy,
                            );
                            write_json(&checkpoint_write_path, &restored_checkpoint_export)?;
                            if write_learned_params_artifact {
                                let restored_checkpoint_params =
                                    SoccerSelfPlayLearnedParams::from_training_artifact_with_neural_network(
                                        &restored_checkpoint_artifact,
                                        latest_neural_network.clone(),
                                    );
                                write_json(
                                    &checkpoint_learned_params_path,
                                    &restored_checkpoint_params,
                                )?;
                            }
                            if let Some(sidecar_path) = write_neural_sidecar_for_policy_artifact(
                                &checkpoint_write_path,
                                latest_neural_network.as_ref(),
                            )? {
                                println!(
                                    "checkpoint_local_best_restored_candidate_neural_sidecar games_completed={} artifact={}",
                                    episode_summaries.len(),
                                    sidecar_path.display()
                                );
                            }
                            println!(
                                "checkpoint_local_best_restored_candidate_rewritten games_completed={} artifact={}",
                                episode_summaries.len(),
                                checkpoint_write_path.display()
                            );
                            if let Some(restored_best) = local_neural_promotion_trial_best.as_ref()
                            {
                                write_training_best_resume_only_checkpoint(
                                    "local_best_restore",
                                    episode_summaries.len(),
                                    restored_best.completed_games,
                                    &restored_checkpoint_artifact,
                                    latest_neural_network.as_ref(),
                                    &restored_best.evaluation,
                                    &training_best_checkpoint_artifact_path,
                                    &training_best_learned_params_path,
                                    write_learned_params_artifact,
                                    artifact_max_entries_per_policy,
                                )?;
                            }
                        }
                    }
                }
            }
            let checkpoint_manifest = run_manifest(
                &run_id,
                &run_dir,
                &game_dir,
                games,
                parallel_games,
                shard_index,
                shard_count,
                seed,
                effective_seed,
                config.clone(),
                options.clone(),
                &final_artifact_path,
                &checkpoint_learned_params_path,
                &checkpoint_write_path,
                &episode_log_path,
                checkpoint_interval_games,
                artifact_max_entries_per_policy,
                max_policy_entries_per_team,
                max_policy_target_entries_per_team,
                min_policy_visits,
                moment_replay_path.clone(),
                moment_replay_records,
                moment_replay_transitions,
                if moment_replay_path.is_some() {
                    moment_replay_passes
                } else {
                    0
                },
                moment_replay_reward_scale,
                pg_store.is_some(),
                pg_experiment_slug.clone(),
                pg_experiment_id.clone(),
                pg_last_policy_version_id.clone(),
                pg_persisted_games,
                pg_completed_run_batch_games,
                write_final_policy_artifact,
                write_game_artifacts,
                &game_artifact_mode,
                manifest_games.clone(),
            );
            write_json(&manifest_path, &checkpoint_manifest)?;
            last_checkpoint_episode = next_episode;
            println!(
                "checkpoint games_completed={} artifact={} manifest={}",
                episode_summaries.len(),
                checkpoint_write_path.display(),
                manifest_path.display()
            );
            let _ = std::io::stdout().flush();
        }
        let neural_population_acceptance = maybe_run_neural_population_search(
            completed_after_batch,
            &config,
            neural_population_search_config,
            local_neural_promotion_trial_best.as_ref(),
            anchor_neural_network.as_ref(),
            &mut latest_neural_network,
            &mut latest_neural_network_arc_cache,
        )?;
        if let Some(acceptance) = neural_population_acceptance {
            let _ = std::io::stdout().flush();
            if !acceptance.preserve_as_local_best {
                neural_population_accept_restore_protection_games_remaining =
                    neural_population_accept_restore_protection_games(
                        &acceptance,
                        neural_population_accept_freeze_games,
                    );
                neural_population_accept_learning_freeze_games_remaining =
                    neural_population_accept_learning_freeze_games(
                        &acceptance,
                        neural_population_accept_freeze_games,
                    );
                if neural_population_accept_restore_protection_games_remaining > 0 {
                    println!(
                        "neural_population_accept_restore_protection_armed completed_games={} protected_games={} reason=heldout_population_candidate_accepted_for_training",
                        completed_after_batch,
                        neural_population_accept_restore_protection_games_remaining
                    );
                }
                if neural_population_accept_learning_freeze_games_remaining > 0 {
                    println!(
                        "neural_population_accept_learning_freeze_armed completed_games={} freeze_games={} reason=heldout_population_candidate_accepted_for_training",
                        completed_after_batch,
                        neural_population_accept_learning_freeze_games_remaining
                    );
                }
                if policy_promotion_local_best_backoff {
                    let (
                        previous_learning_rate,
                        next_learning_rate,
                        previous_blend_lambda,
                        next_blend_lambda,
                    ) = apply_local_best_neural_backoff(
                        &mut config,
                        policy_promotion_local_best_backoff_factor,
                        policy_promotion_local_best_backoff_min_learning_rate,
                        policy_promotion_local_best_backoff_min_blend_lambda,
                    );
                    println!(
                        "neural_population_search_training_candidate_backoff completed_games={} learning_rate={:.5}->{:.5} neural_blend_lambda={:.3}->{:.3} reason=protect_accepted_population_training_candidate",
                        completed_after_batch,
                        previous_learning_rate,
                        next_learning_rate,
                        previous_blend_lambda,
                        next_blend_lambda,
                    );
                }
            }
            if acceptance.preserve_as_local_best {
                let accepted_snapshot = acceptance.eval.snapshot.clone();
                let accepted_evaluation = policy_promotion_evaluation_from_candidate_validation(
                    &acceptance.eval,
                    policy_promotion_gate,
                );
                let beats_local_best = local_neural_promotion_trial_best
                    .as_ref()
                    .map(|best| {
                        policy_promotion_evaluation_beats_local_best(&accepted_evaluation, best)
                    })
                    .unwrap_or(true);
                if beats_local_best {
                    latest_neural_network = Some(accepted_snapshot.clone());
                    latest_neural_network_arc_cache = None;
                    local_neural_promotion_trial_best = Some(LocalNeuralPromotionTrialBest {
                        completed_games: completed_after_batch,
                        evaluation: accepted_evaluation.clone(),
                        config: config.clone(),
                        tactical_learning: tactical_learning.clone(),
                        policies: policies.clone(),
                        neural_network: Some(accepted_snapshot.clone()),
                        carried_world_model: current_carried_world_model_snapshot(),
                    });
                    pg_base_policy_promotion_baseline = retain_stronger_policy_promotion_baseline(
                        pg_base_policy_promotion_baseline,
                        policy_promotion_incumbent_baseline_from_evaluation(&accepted_evaluation),
                    );
                    let population_best_artifact = self_play_artifact_from_policies(
                        config.clone(),
                        options.clone(),
                        tactical_summary.clone(),
                        episode_summaries.clone(),
                        &policies,
                    );
                    let population_best_export = compact_training_artifact_for_export(
                        &population_best_artifact,
                        artifact_max_entries_per_policy,
                    );
                    write_json(
                        &training_best_checkpoint_artifact_path,
                        &population_best_export,
                    )?;
                    if let Some(metadata_path) = write_local_policy_promotion_best(
                        &training_best_checkpoint_artifact_path,
                        completed_after_batch,
                        &accepted_evaluation,
                    )? {
                        println!(
                            "neural_population_search_training_best_metadata_written completed_games={} artifact={}",
                            completed_after_batch,
                            metadata_path.display()
                        );
                    }
                    if write_learned_params_artifact {
                        let training_best_params =
                            SoccerSelfPlayLearnedParams::from_training_artifact_with_neural_network(
                                &population_best_artifact,
                                Some(accepted_snapshot.clone()),
                            );
                        write_json(&training_best_learned_params_path, &training_best_params)?;
                    }
                    if let Some(sidecar_path) = write_neural_sidecar_for_policy_artifact(
                        &training_best_checkpoint_artifact_path,
                        Some(&accepted_snapshot),
                    )? {
                        println!(
                            "neural_population_search_training_best_neural_sidecar completed_games={} artifact={}",
                            completed_after_batch,
                            sidecar_path.display()
                        );
                    }
                    println!(
                        "neural_population_search_local_best_updated completed_games={} accepted_index={} accepted_source={} sample_games={} mean_match_fitness={:.4} best_match_fitness={:.4} mean_play_quality={:.4} record={}-{}-{} goals={}-{} artifact={}",
                        completed_after_batch,
                        acceptance.eval.index,
                        acceptance.eval.source,
                        accepted_evaluation.sample_games,
                        accepted_evaluation.mean_match_fitness,
                        accepted_evaluation.best_match_fitness,
                        accepted_evaluation.mean_play_quality,
                        acceptance.eval.wins,
                        acceptance.eval.draws,
                        acceptance.eval.losses,
                        acceptance.eval.goals_for,
                        acceptance.eval.goals_against,
                        training_best_checkpoint_artifact_path.display()
                    );
                    neural_population_accept_restore_protection_games_remaining =
                        neural_population_accept_restore_protection_games(
                            &acceptance,
                            neural_population_accept_freeze_games,
                        );
                    neural_population_accept_learning_freeze_games_remaining =
                        neural_population_accept_learning_freeze_games(
                            &acceptance,
                            neural_population_accept_freeze_games,
                        );
                    if neural_population_accept_restore_protection_games_remaining > 0 {
                        println!(
                            "neural_population_accept_restore_protection_armed completed_games={} protected_games={} reason=heldout_population_candidate_accepted",
                            completed_after_batch,
                            neural_population_accept_restore_protection_games_remaining
                        );
                    }
                    if neural_population_accept_learning_freeze_games_remaining > 0 {
                        println!(
                            "neural_population_accept_learning_freeze_armed completed_games={} freeze_games={} reason=heldout_population_candidate_accepted",
                            completed_after_batch,
                            neural_population_accept_learning_freeze_games_remaining
                        );
                    }
                    if policy_promotion_local_best_backoff {
                        let (
                            previous_learning_rate,
                            next_learning_rate,
                            previous_blend_lambda,
                            next_blend_lambda,
                        ) = apply_local_best_neural_backoff(
                            &mut config,
                            policy_promotion_local_best_backoff_factor,
                            policy_promotion_local_best_backoff_min_learning_rate,
                            policy_promotion_local_best_backoff_min_blend_lambda,
                        );
                        if let Some(best) = local_neural_promotion_trial_best.as_mut() {
                            best.config = config.clone();
                        }
                        println!(
                            "neural_population_search_local_best_backoff completed_games={} learning_rate={:.5}->{:.5} neural_blend_lambda={:.3}->{:.3} reason=protect_accepted_population_candidate",
                            completed_after_batch,
                            previous_learning_rate,
                            next_learning_rate,
                            previous_blend_lambda,
                            next_blend_lambda,
                        );
                    }
                } else {
                    let incumbent_fitness = local_neural_promotion_trial_best
                        .as_ref()
                        .map(|best| format!("{:.4}", best.evaluation.mean_match_fitness))
                        .unwrap_or_else(|| "none".to_string());
                    println!(
                        "neural_population_search_local_best_held completed_games={} accepted_index={} accepted_source={} accepted_fitness={:.4} incumbent_mean_match_fitness={} reason=accepted_not_stronger_than_local_best",
                        completed_after_batch,
                        acceptance.eval.index,
                        acceptance.eval.source,
                        accepted_evaluation.mean_match_fitness,
                        incumbent_fitness
                    );
                }
            }
        }
    }
    worker_pool.shutdown()?;
    if let Some(episode_log) = episode_log.as_mut() {
        episode_log.flush()?;
    }
    if let Some(experiment_id) = pg_experiment_id.as_deref() {
        if soccer_should_flush_postgres_policy_versions_for_new_sim(
            pg_refresh_for_new_sims,
            pg_flush_policy_versions_before_new_sim,
            pg_policy_version_buffer.len(),
        ) {
            if let Some(store) = pg_store.as_mut() {
                flush_postgres_policy_versions_for_new_sims(
                    store,
                    experiment_id,
                    &pg_runner_id,
                    &mut pg_policy_version_buffer,
                    shard_index,
                    shard_count,
                )?;
            }
        }
    }
    if let Some(writer) = pg_completed_writer {
        pg_persisted_games += writer.finish().map_err(invalid_data)?;
    }

    let artifact = self_play_artifact_from_policies(
        config.clone(),
        options.clone(),
        tactical_summary,
        episode_summaries,
        &policies,
    );
    if write_final_artifacts {
        if write_final_policy_artifact {
            let final_export =
                compact_training_artifact_for_export(&artifact, artifact_max_entries_per_policy);
            write_json(&final_artifact_path, &final_export)?;
            if let Some(sidecar_path) = write_neural_sidecar_for_policy_artifact(
                &final_artifact_path,
                latest_neural_network.as_ref(),
            )? {
                println!("final_neural_sidecar={}", sidecar_path.display());
            }
        } else {
            println!("final_policy_artifact=disabled");
        }
        if write_learned_params_artifact {
            let learned_params =
                SoccerSelfPlayLearnedParams::from_training_artifact_with_neural_network(
                    &artifact,
                    latest_neural_network.clone(),
                );
            write_json(&learned_params_path, &learned_params)?;
        } else {
            println!(
                "learned_params_artifact=disabled path={}",
                learned_params_path.display()
            );
        }

        let manifest = run_manifest(
            &run_id,
            &run_dir,
            &game_dir,
            games,
            parallel_games,
            shard_index,
            shard_count,
            seed,
            effective_seed,
            config.clone(),
            options,
            &final_artifact_path,
            &learned_params_path,
            &checkpoint_artifact_path,
            &episode_log_path,
            checkpoint_interval_games,
            artifact_max_entries_per_policy,
            max_policy_entries_per_team,
            max_policy_target_entries_per_team,
            min_policy_visits,
            moment_replay_path.clone(),
            moment_replay_records,
            moment_replay_transitions,
            if moment_replay_path.is_some() {
                moment_replay_passes
            } else {
                0
            },
            moment_replay_reward_scale,
            pg_store.is_some(),
            pg_experiment_slug.clone(),
            pg_experiment_id.clone(),
            pg_last_policy_version_id.clone(),
            pg_persisted_games,
            pg_completed_run_batch_games,
            write_final_policy_artifact,
            write_game_artifacts,
            &game_artifact_mode,
            manifest_games,
        );
        write_json(&manifest_path, &manifest)?;
    } else {
        println!("final_artifacts=disabled");
    }

    let elapsed = started.elapsed();
    let game_count = games.max(1) as f64;
    println!(
        "soccer_self_play_done games={} minutes={:.1} halves={} half_minutes={:.1} dt={:.3}s ticks_per_game={} elapsed={:.2?}",
        games,
        effective_minutes,
        halves,
        half_minutes,
        dt_seconds,
        config.total_ticks(),
        elapsed
    );
    println!("artifact={}", final_artifact_path.display());
    println!("learned_params={}", learned_params_path.display());
    println!("manifest={}", manifest_path.display());
    if write_episode_log_file {
        println!("episode_log={}", episode_log_path.display());
    } else {
        println!("episode_log=disabled");
    }
    println!(
        "aggregate goals_per_game={:.2} home_goals={} away_goals={} shots_per_game={:.2} on_target_rate={:.2} pass_completion={:.2} interceptions_per_game={:.2}",
        (total_home_goals + total_away_goals) as f64 / game_count,
        total_home_goals,
        total_away_goals,
        total_shots as f64 / game_count,
        if total_shots == 0 { 0.0 } else { total_on_target as f64 / total_shots as f64 },
        if total_pass_attempts == 0 { 0.0 } else { total_pass_completions as f64 / total_pass_attempts as f64 },
        total_interceptions as f64 / game_count,
    );
    println!(
        "tactical_summary shape_transitions={} attack={} defense={} attack_width_score={:.3} attack_flank_lane={:.3} defense_contract_score={:.3} defense_contract_delta_yards={:.3} mean_tactical_reward={:.3}",
        artifact.tactical_summary.shape_transitions,
        artifact.tactical_summary.attack_transitions,
        artifact.tactical_summary.defense_transitions,
        artifact.tactical_summary.mean_attack_width_score,
        artifact.tactical_summary.mean_attack_flank_lane_score,
        artifact.tactical_summary.mean_defense_contract_score,
        artifact.tactical_summary.mean_defense_contract_delta_yards,
        artifact.tactical_summary.mean_tactical_reward,
    );

    println!("home learned actions action visits mean_q");
    for (action, visits, mean_q) in action_summary(&artifact.home_entries).into_iter().take(8) {
        println!("{action} {visits} {mean_q:.3}");
    }
    println!("away learned actions action visits mean_q");
    for (action, visits, mean_q) in action_summary(&artifact.away_entries).into_iter().take(8) {
        println!("{action} {visits} {mean_q:.3}");
    }
    println!(
        "policy_entries home={} away={} target_entries home={} away={}",
        artifact.home_entries.len(),
        artifact.away_entries.len(),
        artifact.home_target_entries.len(),
        artifact.away_target_entries.len()
    );

    Ok(())
}

fn main() {
    let service_name = "main_soccer_learning_run";
    let _telemetry = soccer_engine::telemetry::init_soccer_telemetry(service_name);
    soccer_engine::telemetry::emit_process_start(service_name);
    if let Err(error) = run() {
        soccer_engine::telemetry::emit_process_error(service_name, &error.to_string());
        eprintln!("soccer learning run failed: {error}");
        std::process::exit(1);
    }
    soccer_engine::telemetry::emit_process_complete(service_name);
}

#[cfg(test)]
mod tests {
    use super::*;
    use soccer_engine::des::general::soccer::{
        PlayerRole, Team, CONFIG_FEATURE_DIM, SOCCER_MOMENT_EMBEDDING_DIM,
    };
    use std::sync::Mutex;

    static SOCCER_RUN_PG_ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn neural_sidecar_path_matches_live_policy_autoload_contract() {
        assert_eq!(
            neural_sidecar_path_for_policy_artifact(Path::new(
                "out/local-learning/run/checkpoint-policy.json"
            )),
            PathBuf::from("out/local-learning/run/checkpoint-policy.neural.json")
        );
        assert_eq!(
            neural_sidecar_path_for_policy_artifact(Path::new("team-policy.json")),
            PathBuf::from("team-policy.neural.json")
        );
    }

    #[test]
    fn local_policy_promotion_best_path_matches_live_policy_contract() {
        assert_eq!(
            local_policy_promotion_best_path_for_policy_artifact(Path::new(
                "/tmp/soccer-live/team-policy.json"
            )),
            PathBuf::from("/tmp/soccer-live/team-policy.local-best.json")
        );
        assert_eq!(
            local_policy_promotion_best_path_for_policy_artifact(Path::new("team-policy.json")),
            PathBuf::from("team-policy.local-best.json")
        );
        assert_eq!(
            local_policy_promotion_best_path_for_policy_artifact(Path::new(
                "/tmp/soccer-live/team-policy.training-best.json"
            )),
            PathBuf::from("/tmp/soccer-live/team-policy.training-best.local-best.json")
        );
    }

    #[test]
    fn candidate_checkpoint_path_preserves_live_artifact_name() {
        assert_eq!(
            candidate_checkpoint_path(Path::new("/tmp/soccer-live/team-policy.json")),
            PathBuf::from("/tmp/soccer-live/team-policy.candidate.json")
        );
        assert_eq!(
            candidate_checkpoint_path(Path::new("learned-params.json")),
            PathBuf::from("learned-params.candidate.json")
        );
    }

    #[test]
    fn validated_snapshot_below_live_sample_floor_can_update_resume_best_only() {
        assert!(validated_snapshot_can_update_resume_only_training_best(
            true, true, false, 0.15
        ));
        assert!(!validated_snapshot_can_update_resume_only_training_best(
            false, true, false, 0.15
        ));
        assert!(!validated_snapshot_can_update_resume_only_training_best(
            true, false, false, 0.15
        ));
        assert!(!validated_snapshot_can_update_resume_only_training_best(
            true, true, true, 0.15
        ));
        assert!(!validated_snapshot_can_update_resume_only_training_best(
            true, true, false, -0.01
        ));
    }

    #[test]
    fn negative_held_checkpoint_cannot_anchor_resume_trial_best() {
        let mut evaluation = promotion_eval(-0.3354, 0.19);
        evaluation.rejection_reasons = vec![
            "sample_games 6 below min_sample_games 18".to_string(),
            "mean_match_fitness -0.3354 below min_mean_match_fitness -0.2500".to_string(),
        ];
        assert!(!policy_promotion_evaluation_can_anchor_local_trial(
            &evaluation,
            true
        ));
    }

    #[test]
    fn training_best_checkpoint_path_preserves_live_artifact_name() {
        assert_eq!(
            training_best_checkpoint_path(Path::new("/tmp/soccer-live/team-policy.json")),
            PathBuf::from("/tmp/soccer-live/team-policy.training-best.json")
        );
        assert_eq!(
            training_best_checkpoint_path(Path::new("learned-params.json")),
            PathBuf::from("learned-params.training-best.json")
        );
    }

    #[test]
    fn learning_action_outcome_log_groups_ranked_and_bucketed_passes() {
        assert_eq!(learning_action_label_for_log("pass1"), "pass");
        assert_eq!(learning_action_label_for_log("pass1-kp7"), "pass");
        assert_eq!(
            learning_action_label_for_log("aerial-pass2-kp4"),
            "aerial-pass"
        );
        assert!(learning_action_has_discretized_kick_for_log("pass1-kp7"));
        assert!(learning_action_has_discretized_kick_for_log(
            "aerial-pass2-kp4"
        ));
        assert!(!learning_action_has_discretized_kick_for_log("pass1"));
    }

    fn neural_population_test_config() -> NeuralPopulationSearchConfig {
        NeuralPopulationSearchConfig {
            enabled: true,
            interval_games: 1,
            population_size: 4,
            eval_games: 2,
            eval_minutes: 0.1,
            mutation_rate: 1.0,
            mutation_scale: 0.35,
            crossover_rate: 1.0,
            min_fitness_delta: 0.0,
            min_accepted_fitness: -8.0,
            min_accepted_goal_margin: -99.0,
            training_min_accepted_fitness: -8.0,
            training_min_accepted_goal_margin: -99.0,
            confirm_games: 0,
            confirm_min_fitness_delta: 0.0,
            confirm_min_accepted_fitness: -8.0,
            confirm_min_accepted_goal_margin: -99.0,
            training_confirm_min_accepted_fitness: -8.0,
            training_confirm_min_accepted_goal_margin: -99.0,
            seed: 99,
        }
    }

    fn neural_population_layer(base: f64) -> SoccerNeuralLayerSnapshot {
        SoccerNeuralLayerSnapshot {
            activation: "linear".to_string(),
            weights: vec![vec![base, base + 0.25], vec![base + 0.5, base + 0.75]],
            biases: vec![base + 1.0, base + 1.25],
        }
    }

    fn neural_population_network(base: f64) -> SoccerNeuralNetworkSnapshot {
        let mut snapshot = SoccerNeuralNetworkSnapshot {
            input_dim: 2,
            output_dim: 2,
            layers: vec![neural_population_layer(base)],
            ..SoccerNeuralNetworkSnapshot::default()
        };
        refresh_neural_snapshot_norm_recursive(&mut snapshot, 0);
        snapshot
    }

    fn neural_population_full_snapshot(base: f64) -> SoccerNeuralNetworkSnapshot {
        let mut snapshot = neural_population_network(base);
        snapshot.policy_head = Some(Box::new(SoccerPolicyHeadSnapshot {
            network: neural_population_network(base + 10.0),
            training_steps: 17,
            average_loss: Some(0.125),
            specialist_heads: vec![SoccerPolicySpecialistHeadSnapshot {
                kind: "shoot".to_string(),
                network: neural_population_network(base + 20.0),
                training_steps: 7,
                average_loss: Some(0.25),
            }],
            role_heads: vec![SoccerPolicyRoleHeadSnapshot {
                role: "winger".to_string(),
                network: neural_population_network(base + 30.0),
                training_steps: 11,
                average_loss: Some(0.5),
                specialist_heads: vec![SoccerPolicySpecialistHeadSnapshot {
                    kind: "pass".to_string(),
                    network: neural_population_network(base + 40.0),
                    training_steps: 5,
                    average_loss: Some(0.75),
                }],
            }],
        }));
        snapshot.line_depth_head = Some(Box::new(SoccerAuxiliaryHeadSnapshot {
            network: neural_population_network(base + 50.0),
            training_steps: 13,
            average_loss: Some(0.0625),
        }));
        refresh_neural_snapshot_norm_recursive(&mut snapshot, 0);
        snapshot
    }

    fn neural_population_first_weight(snapshot: &SoccerNeuralNetworkSnapshot) -> f64 {
        snapshot.layers[0].weights[0][0]
    }

    fn collect_neural_population_values(
        snapshot: &SoccerNeuralNetworkSnapshot,
        values: &mut Vec<f64>,
    ) {
        for layer in &snapshot.layers {
            for row in &layer.weights {
                values.extend(row.iter().copied());
            }
            values.extend(layer.biases.iter().copied());
        }
        if let Some(policy_head) = snapshot.policy_head.as_ref() {
            collect_neural_population_values(&policy_head.network, values);
            for specialist in &policy_head.specialist_heads {
                collect_neural_population_values(&specialist.network, values);
            }
            for role_head in &policy_head.role_heads {
                collect_neural_population_values(&role_head.network, values);
                for specialist in &role_head.specialist_heads {
                    collect_neural_population_values(&specialist.network, values);
                }
            }
        }
        if let Some(line_depth_head) = snapshot.line_depth_head.as_ref() {
            collect_neural_population_values(&line_depth_head.network, values);
        }
    }

    #[test]
    fn neural_population_mutation_reaches_actor_role_specialist_and_line_heads() {
        let original = neural_population_full_snapshot(1.0);
        let mut rng = NeuralPopulationRng::new(1234);
        let mutated = mutate_neural_snapshot_recursive(
            original.clone(),
            &mut rng,
            neural_population_test_config(),
            0,
        );

        assert_ne!(
            neural_population_first_weight(&original),
            neural_population_first_weight(&mutated)
        );
        assert_ne!(
            neural_population_first_weight(&original.policy_head.as_ref().unwrap().network),
            neural_population_first_weight(&mutated.policy_head.as_ref().unwrap().network)
        );
        assert_ne!(
            neural_population_first_weight(
                &original.policy_head.as_ref().unwrap().role_heads[0].network
            ),
            neural_population_first_weight(
                &mutated.policy_head.as_ref().unwrap().role_heads[0].network
            )
        );
        assert_ne!(
            neural_population_first_weight(
                &original.policy_head.as_ref().unwrap().role_heads[0].specialist_heads[0].network
            ),
            neural_population_first_weight(
                &mutated.policy_head.as_ref().unwrap().role_heads[0].specialist_heads[0].network
            )
        );
        assert_ne!(
            neural_population_first_weight(&original.line_depth_head.as_ref().unwrap().network),
            neural_population_first_weight(&mutated.line_depth_head.as_ref().unwrap().network)
        );
        assert!(mutated.l2_norm.is_finite());
        assert_eq!(mutated.parameter_count, 6);
    }

    #[test]
    fn neural_population_crossover_preserves_recursive_heads() {
        let a = neural_population_full_snapshot(1.0);
        let b = neural_population_full_snapshot(100.0);
        let mut saw_b_value = false;
        for seed in 0..16 {
            let mut rng = NeuralPopulationRng::new(seed);
            let child = crossover_neural_snapshot_recursive(&a, &b, &mut rng, 0);
            assert_eq!(child.layers.len(), a.layers.len());
            assert_eq!(
                child.policy_head.as_ref().unwrap().role_heads.len(),
                a.policy_head.as_ref().unwrap().role_heads.len()
            );
            assert!(child.line_depth_head.is_some());
            let mut child_values = Vec::new();
            let mut b_values = Vec::new();
            collect_neural_population_values(&child, &mut child_values);
            collect_neural_population_values(&b, &mut b_values);
            assert!(child_values.iter().all(|value| value.is_finite()));
            saw_b_value |= child_values
                .iter()
                .any(|value| b_values.iter().any(|b_value| value == b_value));
        }
        assert!(saw_b_value);
    }

    #[test]
    fn neural_population_candidates_keep_incumbent_as_baseline() {
        let incumbent = neural_population_full_snapshot(1.0);
        let anchor = neural_population_full_snapshot(3.0);
        let mut config = neural_population_test_config();
        config.population_size = 8;
        let candidates =
            build_neural_population_candidates(&incumbent, None, Some(&anchor), config, 6);

        assert_eq!(candidates.len(), config.population_size);
        assert_eq!(candidates[0].index, 0);
        assert_eq!(candidates[0].source, "incumbent");
        assert_eq!(
            soccer_neural_network_snapshot_fingerprint(&candidates[0].snapshot),
            soccer_neural_network_snapshot_fingerprint(&incumbent)
        );
        assert!(candidates.iter().skip(1).any(
            |candidate| soccer_neural_network_snapshot_fingerprint(&candidate.snapshot)
                != soccer_neural_network_snapshot_fingerprint(&incumbent)
        ));
        assert!(candidates
            .iter()
            .any(|candidate| candidate.source.starts_with("crossover:incumbent+anchor:")));
    }

    #[test]
    fn neural_population_candidates_mutate_protected_local_best_before_fill() {
        let incumbent = neural_population_full_snapshot(1.0);
        let local_best_snapshot = neural_population_full_snapshot(2.0);
        let mut local_best = local_neural_trial_best_for_test(12, 0.90, 0.0);
        local_best.neural_network = Some(local_best_snapshot.clone());
        let anchor = neural_population_full_snapshot(3.0);
        let mut config = neural_population_test_config();
        config.population_size = 12;

        let candidates = build_neural_population_candidates(
            &incumbent,
            Some(&local_best),
            Some(&anchor),
            config,
            24,
        );

        assert_eq!(candidates.len(), config.population_size);
        assert!(candidates
            .iter()
            .any(|candidate| candidate.source == "local_best_baseline"));
        for label in ["micro", "base", "wide", "pinpoint"] {
            assert!(
                candidates
                    .iter()
                    .any(|candidate| candidate.source == format!("mutation:local_best:{label}")),
                "missing direct local_best mutation {label}"
            );
        }
        assert!(
            candidates
                .iter()
                .position(|candidate| candidate.source == "mutation:local_best:micro")
                < candidates
                    .iter()
                    .position(|candidate| candidate.source.starts_with("crossover:"))
        );
        assert!(candidates
            .iter()
            .any(
                |candidate| soccer_neural_network_snapshot_fingerprint(&candidate.snapshot)
                    != soccer_neural_network_snapshot_fingerprint(&local_best_snapshot)
            ));
    }

    #[test]
    fn neural_population_goal_margin_gate_rejects_non_positive_margin() {
        let eval = NeuralPopulationCandidateEval {
            index: 1,
            source: "mutation:test".to_string(),
            fitness: 0.3333,
            wins: 3,
            draws: 2,
            losses: 1,
            goals_for: 5,
            goals_against: 5,
            snapshot: neural_population_full_snapshot(1.0),
        };
        let config = NeuralPopulationSearchConfig {
            min_accepted_goal_margin: 1.0,
            ..neural_population_test_config()
        };

        assert!(
            neural_population_candidate_goal_margin(&eval) < config.min_accepted_goal_margin,
            "a short eval with draw-level goal margin should stay a scout, not become active"
        );
    }

    #[test]
    fn neural_population_baseline_candidates_are_not_plateau_breakthroughs() {
        let baseline = NeuralPopulationCandidateEval {
            index: 1,
            source: "local_best_baseline".to_string(),
            fitness: 0.5,
            wins: 3,
            draws: 2,
            losses: 1,
            goals_for: 4,
            goals_against: 2,
            snapshot: neural_population_full_snapshot(1.0),
        };
        let mutation = NeuralPopulationCandidateEval {
            index: 2,
            source: "mutation:local_best:wide".to_string(),
            snapshot: neural_population_full_snapshot(2.0),
            ..baseline.clone()
        };

        assert!(neural_population_candidate_eval_is_baseline(&baseline));
        assert!(!neural_population_candidate_eval_is_baseline(&mutation));
    }

    #[test]
    fn neural_population_can_train_from_candidate_below_protected_local_best() {
        let current_training_reference = 0.6324;
        let protected_local_best_floor = 1.0707;
        let candidate_fitness = 0.9148;
        let min_delta = 0.1500;
        let training_improvement = candidate_fitness - current_training_reference;
        let protected_improvement = candidate_fitness - protected_local_best_floor;

        assert!(
            neural_population_improvement_clears_delta(training_improvement, min_delta),
            "candidate should be allowed to carry training forward when it beats the evaluated frontier"
        );
        assert!(
            !neural_population_improvement_clears_delta(protected_improvement, min_delta),
            "candidate should not rewrite protected local-best metadata until it beats that floor"
        );
    }

    #[test]
    fn neural_population_training_candidate_protects_restore_without_freezing_learning() {
        let eval = NeuralPopulationCandidateEval {
            index: 3,
            source: "mutation:incumbent:base".to_string(),
            fitness: 0.6564,
            wins: 5,
            draws: 6,
            losses: 1,
            goals_for: 9,
            goals_against: 3,
            snapshot: neural_population_full_snapshot(2.0),
        };
        let training_candidate = NeuralPopulationSearchAcceptance {
            eval: eval.clone(),
            preserve_as_local_best: false,
        };
        let local_best_candidate = NeuralPopulationSearchAcceptance {
            eval,
            preserve_as_local_best: true,
        };

        assert_eq!(
            neural_population_accept_restore_protection_games(&training_candidate, 12),
            12
        );
        assert_eq!(
            neural_population_accept_learning_freeze_games(&training_candidate, 12),
            0,
            "candidate accepted for training must continue on-policy updates"
        );
        assert_eq!(
            neural_population_accept_restore_protection_games(&local_best_candidate, 12),
            12
        );
        assert_eq!(
            neural_population_accept_learning_freeze_games(&local_best_candidate, 12),
            12,
            "protected local-best accepts can keep the conservative evaluation freeze"
        );
    }

    #[test]
    fn neural_population_eval_seed_schedule_is_candidate_invariant() {
        let primary_seed = neural_population_eval_seed(99, 12);
        let confirm_seed = neural_population_confirm_seed(99, 12);
        let candidate_one: Vec<u64> = (0..6)
            .map(|game_index| neural_population_eval_game_seed(primary_seed, game_index))
            .collect();
        let candidate_nine: Vec<u64> = (0..6)
            .map(|game_index| neural_population_eval_game_seed(primary_seed, game_index))
            .collect();
        let confirm_block: Vec<u64> = (0..6)
            .map(|game_index| neural_population_eval_game_seed(confirm_seed, game_index))
            .collect();

        assert_eq!(
            candidate_one, candidate_nine,
            "population candidates must compare on the same fixture seed block"
        );
        assert_ne!(
            candidate_one, confirm_block,
            "confirmation must use a disjoint fixture seed block"
        );
    }

    #[test]
    fn neural_population_accept_protection_blocks_local_best_restore_helper() {
        assert!(neural_population_accept_protection_blocks_local_best_restore(12));
        assert!(neural_population_accept_protection_blocks_local_best_restore(1));
        assert!(!neural_population_accept_protection_blocks_local_best_restore(0));
    }

    #[test]
    fn neural_population_goal_margin_gate_uses_mean_margin_per_eval_game() {
        let eval = NeuralPopulationCandidateEval {
            index: 2,
            source: "mutation:wide".to_string(),
            fitness: -0.15,
            wins: 5,
            draws: 2,
            losses: 5,
            goals_for: 9,
            goals_against: 15,
            snapshot: neural_population_full_snapshot(2.0),
        };
        let config = NeuralPopulationSearchConfig {
            min_accepted_goal_margin: -1.0,
            ..neural_population_test_config()
        };

        assert_eq!(neural_population_candidate_goal_margin(&eval), -0.5);
        assert!(
            neural_population_candidate_goal_margin(&eval) >= config.min_accepted_goal_margin,
            "population search should compare goal margin per eval game, not total margin"
        );
    }

    #[test]
    fn batch_snapshot_validation_requires_fitness_and_goal_margin() {
        let losing_eval = NeuralPopulationCandidateEval {
            index: 3,
            source: "batch_snapshot_episode_42".to_string(),
            fitness: 0.35,
            wins: 3,
            draws: 1,
            losses: 2,
            goals_for: 7,
            goals_against: 9,
            snapshot: neural_population_full_snapshot(3.0),
        };
        assert!(!batch_neural_snapshot_validation_passes(
            &losing_eval,
            0.25,
            0.0
        ));

        let winning_eval = NeuralPopulationCandidateEval {
            goals_for: 10,
            goals_against: 8,
            ..losing_eval
        };
        assert!(batch_neural_snapshot_validation_passes(
            &winning_eval,
            0.25,
            0.0
        ));
    }

    #[test]
    fn candidate_validation_promotion_eval_uses_validation_objective() {
        let eval = NeuralPopulationCandidateEval {
            index: 0,
            source: "batch_snapshot_episode_2".to_string(),
            fitness: 0.4117,
            wins: 3,
            draws: 1,
            losses: 2,
            goals_for: 6,
            goals_against: 4,
            snapshot: neural_population_network(0.0),
        };

        let promotion = policy_promotion_evaluation_from_candidate_validation(
            &eval,
            SoccerPolicyPromotionGateConfig::default(),
        );

        assert_eq!(promotion.sample_games, 6);
        assert!((promotion.mean_match_fitness - 0.4117).abs() < 1e-12);
        assert!((promotion.best_match_fitness - 0.4117).abs() < 1e-12);
        assert!((promotion.mean_goal_margin - (2.0 / 6.0)).abs() < 1e-12);
        assert!((promotion.mean_conceded_goals - (4.0 / 6.0)).abs() < 1e-12);
    }

    #[test]
    fn incumbent_baseline_promotion_eval_preserves_restart_floor() {
        let baseline = PolicyPromotionIncumbentBaseline {
            sample_games: 6,
            mean_match_fitness: 1.5217,
            best_match_fitness: 1.5217,
            mean_play_quality: 0.0,
        };
        let promotion = policy_promotion_evaluation_from_incumbent_baseline(
            baseline,
            SoccerPolicyPromotionGateConfig::default(),
        );

        assert_eq!(promotion.sample_games, 6);
        assert!((promotion.mean_match_fitness - 1.5217).abs() < 1e-12);
        assert!((promotion.best_match_fitness - 1.5217).abs() < 1e-12);
        assert_eq!(promotion.mean_play_quality, 0.0);
        assert!(promotion.eligible);
    }

    fn promotion_eval(
        mean_match_fitness: f64,
        mean_play_quality: f64,
    ) -> SoccerPolicyPromotionGateEvaluation {
        SoccerPolicyPromotionGateEvaluation {
            enabled: true,
            eligible: false,
            sample_games: 8,
            min_sample_games: 8,
            mean_match_fitness,
            best_match_fitness: mean_match_fitness,
            mean_play_quality,
            mean_conceded_goals: 0.0,
            mean_goal_margin: 0.0,
            mean_chain_net_loss: 0.0,
            rejection_reasons: vec!["incumbent_mean_match_fitness below".to_string()],
        }
    }

    fn eligible_promotion_eval(
        mean_match_fitness: f64,
        mean_play_quality: f64,
    ) -> SoccerPolicyPromotionGateEvaluation {
        let mut evaluation = promotion_eval(mean_match_fitness, mean_play_quality);
        evaluation.eligible = true;
        evaluation.rejection_reasons.clear();
        evaluation
    }

    fn local_neural_trial_best_for_test(
        completed_games: usize,
        mean_match_fitness: f64,
        mean_play_quality: f64,
    ) -> LocalNeuralPromotionTrialBest {
        LocalNeuralPromotionTrialBest {
            completed_games,
            evaluation: promotion_eval(mean_match_fitness, mean_play_quality),
            config: MatchConfig::default(),
            tactical_learning: SoccerTacticalLearningWeights::default(),
            policies: SoccerTeamQPolicies::new(SoccerQPolicyOptions::default()),
            neural_network: None,
            carried_world_model: None,
        }
    }

    #[test]
    fn local_neural_trial_best_compares_mean_fitness_before_play_quality() {
        let best = local_neural_trial_best_for_test(8, 0.70, 0.42);

        assert!(policy_promotion_evaluation_beats_local_best(
            &promotion_eval(0.72, 0.30),
            &best
        ));
        assert!(policy_promotion_evaluation_regresses_from_local_best(
            &promotion_eval(0.68, 0.90),
            &best,
            0.0,
            0.0
        ));
        assert!(!policy_promotion_evaluation_regresses_from_local_best(
            &promotion_eval(0.68, 0.90),
            &best,
            0.03,
            0.0
        ));
        assert!(policy_promotion_evaluation_beats_local_best(
            &promotion_eval(0.70, 0.43),
            &best
        ));
        assert!(policy_promotion_evaluation_regresses_from_local_best(
            &promotion_eval(0.70, 0.41),
            &best,
            0.0,
            0.0
        ));
        assert!(!policy_promotion_evaluation_regresses_from_local_best(
            &promotion_eval(0.70, 0.41),
            &best,
            0.0,
            0.02
        ));
    }

    #[test]
    fn local_trial_anchor_can_opt_into_held_fitness_for_rollback() {
        let mut too_few_samples = promotion_eval(0.40, 0.30);
        too_few_samples.rejection_reasons =
            vec!["sample_games 6 below min_sample_games 18".to_string()];
        assert!(policy_promotion_evaluation_can_anchor_local_trial(
            &too_few_samples,
            false
        ));

        let mut below_mean = promotion_eval(-0.80, 0.30);
        below_mean.rejection_reasons = vec![
            "sample_games 6 below min_sample_games 18".to_string(),
            "mean_match_fitness -0.8000 below min_mean_match_fitness -0.2500".to_string(),
        ];
        assert!(!policy_promotion_evaluation_can_anchor_local_trial(
            &below_mean,
            false
        ));
        assert!(policy_promotion_evaluation_can_anchor_local_trial(
            &below_mean,
            true
        ));
    }

    #[test]
    fn best_only_checkpoint_publish_requires_eligible_local_improvement() {
        let best = local_neural_trial_best_for_test(8, 0.70, 0.42);

        assert!(!should_publish_live_checkpoint(
            true,
            &promotion_eval(0.90, 0.70),
            None,
            Some(&best),
            true
        ));
        assert!(!should_publish_live_checkpoint(
            true,
            &eligible_promotion_eval(0.72, 0.30),
            None,
            Some(&best),
            false
        ));
        assert!(!should_publish_live_checkpoint(
            true,
            &eligible_promotion_eval(0.68, 0.90),
            None,
            Some(&best),
            true
        ));
        assert!(!should_publish_live_checkpoint(
            true,
            &eligible_promotion_eval(0.70, 0.41),
            None,
            Some(&best),
            true
        ));
        assert!(should_publish_live_checkpoint(
            true,
            &eligible_promotion_eval(0.72, 0.30),
            None,
            Some(&best),
            true
        ));
        assert!(should_publish_live_checkpoint(
            false,
            &promotion_eval(0.10, 0.10),
            None,
            Some(&best),
            false
        ));
    }

    #[test]
    fn local_policy_promotion_best_metadata_blocks_restart_regression() {
        let metadata = LocalPolicyPromotionBestMetadata {
            completed_games: 6,
            sample_games: 6,
            mean_match_fitness: 1.3124,
            best_match_fitness: 2.8707,
            mean_play_quality: 0.3702,
            objective: Some(LOCAL_POLICY_PROMOTION_OBJECTIVE_HOME_ANALYTIC.to_string()),
        };
        let baseline = policy_promotion_incumbent_baseline_from_local_best_metadata(&metadata)
            .expect("valid local-best metadata");
        assert!(!should_publish_live_checkpoint(
            true,
            &eligible_promotion_eval(0.8307, 0.3118),
            Some(baseline),
            None,
            true
        ));
        assert!(should_publish_live_checkpoint(
            true,
            &eligible_promotion_eval(1.3300, 0.3000),
            Some(baseline),
            None,
            true
        ));
    }

    #[test]
    fn resume_seed_uses_fresh_eval_for_restore_even_when_stored_floor_is_stronger() {
        let metadata = LocalPolicyPromotionBestMetadata {
            completed_games: 12,
            sample_games: 12,
            mean_match_fitness: 0.6488,
            best_match_fitness: 0.6488,
            mean_play_quality: 0.0,
            objective: Some(LOCAL_POLICY_PROMOTION_OBJECTIVE_HOME_ANALYTIC.to_string()),
        };
        let stored_baseline =
            policy_promotion_incumbent_baseline_from_local_best_metadata(&metadata)
                .expect("valid local-best metadata");
        let gate = SoccerPolicyPromotionGateConfig::default();

        let (retained, source) = retain_resume_local_best_seed_evaluation(
            Some(stored_baseline),
            eligible_promotion_eval(-0.0581, 0.0),
            gate,
        );
        assert_eq!(source, "fresh_eval_revalidated");
        assert!((retained.mean_match_fitness - -0.0581).abs() < 1e-12);
        assert_eq!(retained.sample_games, 8);

        let (retained, source) = retain_resume_local_best_seed_evaluation(
            Some(stored_baseline),
            eligible_promotion_eval(0.9000, 0.0),
            gate,
        );
        assert_eq!(source, "fresh_eval_revalidated");
        assert!((retained.mean_match_fitness - 0.9000).abs() < 1e-12);
    }

    #[test]
    fn batch_neural_snapshot_selection_prefers_fitness_then_later_episode() {
        assert!(batch_neural_snapshot_candidate_better(
            BatchNeuralSnapshotSelectionMode::Fitness,
            1.2,
            8,
            100,
            1.1,
            9,
            200,
        ));
        assert!(!batch_neural_snapshot_candidate_better(
            BatchNeuralSnapshotSelectionMode::Fitness,
            1.0,
            12,
            400,
            1.1,
            9,
            100,
        ));
        assert!(batch_neural_snapshot_candidate_better(
            BatchNeuralSnapshotSelectionMode::Fitness,
            1.1,
            12,
            100,
            1.1,
            9,
            200,
        ));
        assert!(!batch_neural_snapshot_candidate_better(
            BatchNeuralSnapshotSelectionMode::Fitness,
            1.1,
            8,
            400,
            1.1,
            9,
            100,
        ));
        assert!(batch_neural_snapshot_candidate_better(
            BatchNeuralSnapshotSelectionMode::TrainingSteps,
            1.0,
            8,
            400,
            1.5,
            9,
            300,
        ));
        assert!(!batch_neural_snapshot_candidate_better(
            BatchNeuralSnapshotSelectionMode::TrainingSteps,
            1.5,
            12,
            300,
            1.0,
            9,
            400,
        ));
    }

    #[test]
    fn guarded_training_step_snapshot_selection_rejects_bad_fresh_snapshot() {
        let mut games = vec![
            completed_game_with_score_and_neural_steps(0, 3, 0, 100),
            completed_game_with_score_and_neural_steps(1, 0, 1, 500),
            completed_game_with_score_and_neural_steps(2, 0, 8, 900),
        ];

        let selection = select_batch_neural_snapshot(
            &mut games,
            BatchNeuralSnapshotSelectionMode::TrainingStepsGuarded,
            true,
            0.75,
            0.0,
            SOCCER_LEARNING_OBJECTIVE_FITNESS_MIN,
            DEFAULT_SOCCER_NEURAL_BATCH_SNAPSHOT_RESCUE_MIN_FITNESS,
        )
        .expect("guarded selector should keep an eligible snapshot");

        assert_eq!(selection.episode, 1);
        assert_eq!(selection.training_steps, 100);
        assert!(
            selection.match_fitness > 0.0,
            "guard should prefer the useful snapshot over fresh regressions, got {}",
            selection.match_fitness
        );
    }

    #[test]
    fn unguarded_training_step_snapshot_selection_can_pick_freshest_snapshot() {
        let mut games = vec![
            completed_game_with_score_and_neural_steps(0, 3, 0, 100),
            completed_game_with_score_and_neural_steps(1, 0, 1, 500),
            completed_game_with_score_and_neural_steps(2, 0, 8, 900),
        ];

        let selection = select_batch_neural_snapshot(
            &mut games,
            BatchNeuralSnapshotSelectionMode::TrainingSteps,
            true,
            0.75,
            SOCCER_LEARNING_OBJECTIVE_FITNESS_MIN,
            SOCCER_LEARNING_OBJECTIVE_FITNESS_MIN,
            DEFAULT_SOCCER_NEURAL_BATCH_SNAPSHOT_RESCUE_MIN_FITNESS,
        )
        .expect("unguarded selector should pick a snapshot");

        assert_eq!(selection.episode, 3);
        assert_eq!(selection.training_steps, 900);
    }

    #[test]
    fn batch_neural_snapshot_selection_rejects_batch_below_min_fitness_floor() {
        let mut games = vec![
            completed_game_with_score_and_neural_steps(0, 0, 1, 100),
            completed_game_with_score_and_neural_steps(1, 0, 2, 500),
            completed_game_with_score_and_neural_steps(2, 1, 3, 900),
        ];

        let selection = select_batch_neural_snapshot(
            &mut games,
            BatchNeuralSnapshotSelectionMode::TrainingStepsGuarded,
            true,
            0.0,
            0.0,
            SOCCER_LEARNING_OBJECTIVE_FITNESS_MIN,
            DEFAULT_SOCCER_NEURAL_BATCH_SNAPSHOT_RESCUE_MIN_FITNESS,
        );

        assert!(
            selection.is_none(),
            "negative HOME-objective batches must not install the least-bad network"
        );
    }

    #[test]
    fn batch_neural_snapshot_selection_rejects_positive_outlier_in_negative_batch() {
        let mut games = vec![
            completed_game_with_score_and_neural_steps(0, 1, 0, 100),
            completed_game_with_score_and_neural_steps(1, 0, 3, 500),
            completed_game_with_score_and_neural_steps(2, 0, 3, 900),
        ];
        let mean = games
            .iter()
            .map(|game| {
                soccer_learning_objective_match_fitness(&game.episode_summary.summary, true)
            })
            .sum::<f64>()
            / games.len() as f64;
        assert!(
            mean < 0.0,
            "test setup should model a lucky positive outlier inside a bad batch"
        );

        let selection = select_batch_neural_snapshot(
            &mut games,
            BatchNeuralSnapshotSelectionMode::TrainingStepsGuarded,
            true,
            0.0,
            0.0,
            0.0,
            DEFAULT_SOCCER_NEURAL_BATCH_SNAPSHOT_RESCUE_MIN_FITNESS,
        );

        assert!(
            selection.is_none(),
            "negative batches must not install a lucky positive outlier snapshot"
        );
    }

    #[test]
    fn batch_neural_snapshot_selection_rescues_strong_snapshot_from_noisy_batch() {
        let mut games = vec![
            completed_game_with_score_and_neural_steps(0, 3, 0, 100),
            completed_game_with_score_and_neural_steps(1, 0, 4, 500),
            completed_game_with_score_and_neural_steps(2, 0, 4, 900),
        ];
        let mean = games
            .iter()
            .map(|game| {
                soccer_learning_objective_match_fitness(&game.episode_summary.summary, true)
            })
            .sum::<f64>()
            / games.len() as f64;
        assert!(
            mean < 0.0,
            "test setup should model a noisy negative batch with one breakthrough"
        );

        let selection = select_batch_neural_snapshot(
            &mut games,
            BatchNeuralSnapshotSelectionMode::Fitness,
            true,
            0.0,
            0.0,
            0.0,
            1.0,
        )
        .expect("strong breakthrough snapshot should bypass the noisy-batch mean gate");

        assert_eq!(selection.episode, 1);
        assert!(
            selection.match_fitness >= 1.0,
            "rescued snapshot must be clearly above the noise floor, got {}",
            selection.match_fitness
        );
    }

    #[test]
    fn selected_only_policy_merge_skips_all_deltas_without_selected_snapshot() {
        assert!(should_merge_batch_policy_delta(false, None, 7));
        assert!(!should_merge_batch_policy_delta(true, None, 7));
        assert!(should_merge_batch_policy_delta(true, Some(7), 7));
        assert!(!should_merge_batch_policy_delta(true, Some(8), 7));
    }

    #[test]
    fn no_publishable_neural_snapshot_carry_has_separate_gate() {
        assert!(should_carry_no_publishable_neural_snapshot(
            true, true, None, None
        ));
        assert!(!should_carry_no_publishable_neural_snapshot(
            true, false, None, None
        ));
        assert!(!should_carry_no_publishable_neural_snapshot(
            false, true, None, None
        ));
        assert!(!should_carry_no_publishable_neural_snapshot(
            true,
            true,
            Some(7),
            None
        ));
        assert!(!should_carry_no_publishable_neural_snapshot(
            true,
            true,
            None,
            Some(7)
        ));
    }

    #[test]
    fn carried_frontier_episode_drives_selected_only_policy_delta_merge() {
        assert_eq!(batch_policy_delta_frontier_episode(Some(7), None), Some(7));
        assert_eq!(batch_policy_delta_frontier_episode(None, Some(7)), Some(7));
        assert_eq!(
            batch_policy_delta_frontier_episode(Some(7), Some(8)),
            Some(7)
        );
        let frontier = batch_policy_delta_frontier_episode(None, Some(7));
        assert!(should_merge_batch_policy_delta(true, frontier, 7));
        assert!(!should_merge_batch_policy_delta(true, frontier, 8));
    }

    #[test]
    fn trainer_carry_selector_preserves_warm_snapshot_when_publish_gate_rejects_batch() {
        let mut guarded_games = vec![
            completed_game_with_score_and_neural_steps(0, 0, 1, 100),
            completed_game_with_score_and_neural_steps(1, 0, 2, 500),
            completed_game_with_score_and_neural_steps(2, 1, 3, 900),
        ];
        let guarded = select_batch_neural_snapshot(
            &mut guarded_games,
            BatchNeuralSnapshotSelectionMode::TrainingStepsGuarded,
            true,
            0.0,
            0.0,
            0.0,
            DEFAULT_SOCCER_NEURAL_BATCH_SNAPSHOT_RESCUE_MIN_FITNESS,
        );
        assert!(
            guarded.is_none(),
            "publish/activation selector should still reject the weak batch"
        );

        let mut carry_games = vec![
            completed_game_with_score_and_neural_steps(0, 0, 1, 100),
            completed_game_with_score_and_neural_steps(1, 0, 2, 500),
            completed_game_with_score_and_neural_steps(2, 1, 3, 900),
        ];
        let carry = select_batch_neural_snapshot_for_training_carry(&mut carry_games, true)
            .expect("trainer carry should preserve learning warmth");
        assert_eq!(carry.episode, 3);
        assert_eq!(carry.training_steps, 900);
        assert!(
            carry.match_fitness < 0.0,
            "carry may keep an unpublishable snapshot, but only for trainer memory"
        );
    }

    #[test]
    fn analytic_opponent_objective_scores_home_neural_not_match_winner() {
        let summary = soccer_engine::des::general::soccer::MatchSummary {
            score_home: 3,
            score_away: 12,
            ticks: 10,
            simulated_seconds: 1.0,
            stats: Default::default(),
        };

        assert!(soccer_learning_objective_match_fitness(&summary, false) > 1.0);
        assert!(soccer_learning_objective_match_fitness(&summary, true) < 0.0);
    }

    #[test]
    fn analytic_opponent_objective_uses_dense_home_shot_pressure() {
        let mut active_stats = soccer_engine::des::general::soccer::MatchStats::default();
        active_stats.shots_home = 10;
        active_stats.shots_on_target_home = 6;
        active_stats.shots_after_pass_home = 4;
        let active = soccer_engine::des::general::soccer::MatchSummary {
            score_home: 0,
            score_away: 0,
            ticks: 10,
            simulated_seconds: 1.0,
            stats: active_stats,
        };

        let mut exposed = active.clone();
        exposed.stats.shots_away = 10;
        exposed.stats.shots_on_target_away = 7;
        exposed.stats.shots_after_pass_away = 4;

        assert!(
            soccer_learning_objective_match_fitness(&active, true)
                > soccer_learning_objective_match_fitness(&exposed, true) + 0.40
        );
    }

    #[test]
    fn local_best_neural_backoff_reduces_learning_rate_and_blend_to_floors() {
        let mut config = MatchConfig::default();
        config.neural_learning.learning_rate = 0.0025;
        config.neural_blend.lambda = 0.35;

        let (previous_lr, next_lr, previous_blend, next_blend) =
            apply_local_best_neural_backoff(&mut config, 0.5, 0.0015, 0.20);

        assert_eq!(previous_lr, 0.0025);
        assert_eq!(next_lr, 0.0015);
        assert_eq!(previous_blend, 0.35);
        assert_eq!(next_blend, 0.20);
        assert_eq!(config.neural_learning.learning_rate, 0.0015);
        assert_eq!(config.neural_blend.lambda, 0.20);
    }

    struct EnvVarGuard {
        key: &'static str,
        previous: Option<String>,
    }

    impl EnvVarGuard {
        fn clear(key: &'static str) -> Self {
            let previous = std::env::var(key).ok();
            std::env::remove_var(key);
            Self { key, previous }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.previous {
                Some(previous) => std::env::set_var(self.key, previous),
                None => std::env::remove_var(self.key),
            }
        }
    }

    fn clear_postgres_env() -> Vec<EnvVarGuard> {
        [
            "SOCCER_DATABASE_URL",
            "AGENT_TASKS_RDS_DATABASE_URL",
            "RDS_DATABASE_URL",
            "DATABASE_URL",
            "PG_DATABASE_URL",
        ]
        .into_iter()
        .map(EnvVarGuard::clear)
        .collect()
    }

    fn clear_mpc_env() -> Vec<EnvVarGuard> {
        [
            "SOCCER_LEARNING_MPC_ENABLED",
            "SOCCER_MPC_ENABLED",
            "SOCCER_LEARNING_MPC_FIELD_AWARE_ENABLED",
            "SOCCER_MPC_FIELD_AWARE_ENABLED",
            "SOCCER_LEARNING_MPC_RECONCILE_ENABLED",
            "SOCCER_MPC_RECONCILE_ENABLED",
            "SOCCER_LEARNING_MPC_LATENT_OBJECTIVE_ENABLED",
            "SOCCER_MPC_LATENT_OBJECTIVE_ENABLED",
            "SOCCER_LEARNING_MPC_PLAYER_HORIZON",
            "SOCCER_MPC_PLAYER_HORIZON",
            "SOCCER_LEARNING_MPC_ACTIVE_RADIUS",
            "SOCCER_MPC_ACTIVE_RADIUS",
            "SOCCER_LEARNING_LOCAL_MPC_ENABLED",
            "SOCCER_LOCAL_MPC",
            "SOCCER_LEARNING_LOCAL_MPC_MAX_PLAYERS_PER_TEAM",
            "SOCCER_LOCAL_MPC_MAX_PLAYERS_PER_TEAM",
        ]
        .into_iter()
        .map(EnvVarGuard::clear)
        .collect()
    }

    const CONTINUOUS_DEPLOYMENT_YAML: &str =
        include_str!("../../k8s/soccer-learning-rds-continuous.deployment.yaml");

    fn continuous_manifest_env_value(name: &str) -> Option<&'static str> {
        let marker = format!("- name: {name}");
        let mut lines = CONTINUOUS_DEPLOYMENT_YAML.lines();
        while let Some(line) = lines.next() {
            if line.trim() == marker {
                let value_line = lines.next()?.trim();
                let raw_value = value_line.strip_prefix("value: ")?;
                return Some(raw_value.trim_matches('"'));
            }
        }
        None
    }

    fn assert_continuous_manifest_contains(snippet: &str) {
        assert!(
            CONTINUOUS_DEPLOYMENT_YAML.contains(snippet),
            "continuous deployment manifest missing snippet: {snippet}"
        );
    }

    #[test]
    fn learning_run_mpc_env_enables_execution_stack() {
        let _lock = SOCCER_RUN_PG_ENV_LOCK.lock().expect("env lock poisoned");
        let _guards = clear_mpc_env();
        std::env::set_var("SOCCER_MPC_ENABLED", "1");
        std::env::set_var("SOCCER_MPC_PLAYER_HORIZON", "12");
        std::env::set_var("SOCCER_MPC_ACTIVE_RADIUS", "18");
        std::env::set_var("SOCCER_LOCAL_MPC", "1");
        std::env::set_var("SOCCER_LOCAL_MPC_MAX_PLAYERS_PER_TEAM", "99");

        let defaults = MatchConfig::default();
        let mut config = defaults.clone();
        apply_env_mpc_config(&mut config, &defaults).expect("valid mpc env config");

        assert!(config.mpc.tier2_player_enabled);
        assert!(config.mpc.field_aware_enabled);
        assert!(config.mpc.reconcile_enabled);
        assert!(config.mpc.latent_objective_enabled);
        assert_eq!(config.mpc.player_horizon, 12);
        assert_eq!(config.mpc.active_radius_yards, 18.0);
        assert!(config.local_mpc_enabled);
        assert_eq!(config.local_mpc_max_players_per_team, 11);
    }

    #[test]
    fn learning_run_mpc_env_rejects_dependent_stack_without_tier2() {
        let _lock = SOCCER_RUN_PG_ENV_LOCK.lock().expect("env lock poisoned");
        let _guards = clear_mpc_env();
        std::env::set_var("SOCCER_MPC_FIELD_AWARE_ENABLED", "1");

        let defaults = MatchConfig::default();
        let mut config = defaults.clone();
        let err = apply_env_mpc_config(&mut config, &defaults)
            .expect_err("dependent MPC features require tier2 MPC");

        assert!(err.to_string().contains("SOCCER_MPC_ENABLED"));
    }

    #[test]
    fn learning_run_mpc_master_off_disables_local_mpc() {
        let _lock = SOCCER_RUN_PG_ENV_LOCK.lock().expect("env lock poisoned");
        let _guards = clear_mpc_env();
        std::env::set_var("SOCCER_MPC_ENABLED", "0");
        std::env::set_var("SOCCER_LOCAL_MPC", "1");
        std::env::set_var("SOCCER_LOCAL_MPC_MAX_PLAYERS_PER_TEAM", "3");

        let defaults = MatchConfig::default();
        let mut config = defaults.clone();
        apply_env_mpc_config(&mut config, &defaults).expect("valid mpc env config");

        assert!(!config.mpc.tier2_player_enabled);
        assert!(!config.mpc.field_aware_enabled);
        assert!(!config.mpc.reconcile_enabled);
        assert!(!config.mpc.latent_objective_enabled);
        assert!(!config.local_mpc_enabled);
        assert_eq!(config.local_mpc_max_players_per_team, 3);
    }

    #[test]
    fn continuous_manifest_keeps_postgres_neural_and_refresh_required() {
        assert_eq!(
            continuous_manifest_env_value("SOCCER_REQUIRE_POSTGRES"),
            Some("true")
        );
        assert_eq!(
            continuous_manifest_env_value("SOCCER_NEURAL_LEARNING_ENABLED"),
            Some("true")
        );
        assert_eq!(
            continuous_manifest_env_value("SOCCER_NEURAL_LEARNING_BACKEND"),
            Some("threaded")
        );
        assert_eq!(
            continuous_manifest_env_value("SOCCER_POSTGRES_POLICY_VERSION_INTERVAL_GAMES"),
            Some("1")
        );
        assert_eq!(
            continuous_manifest_env_value("SOCCER_POSTGRES_COMPLETED_RUN_BATCH_GAMES"),
            Some("1")
        );
        assert_eq!(
            continuous_manifest_env_value("SOCCER_POSTGRES_ASYNC_BATCH_QUEUE"),
            Some("1")
        );
        assert_eq!(
            continuous_manifest_env_value("SOCCER_POSTGRES_ASYNC_COALESCE_BATCHES"),
            Some("1")
        );
        assert_eq!(
            continuous_manifest_env_value("SOCCER_POSTGRES_ASYNC_COALESCE_WAIT_MS"),
            Some("50")
        );
        assert_eq!(
            continuous_manifest_env_value("SOCCER_POSTGRES_TACTICAL_LEARNING_AUTHORITATIVE"),
            Some("true")
        );
        assert_eq!(
            continuous_manifest_env_value("SOCCER_POSTGRES_REFRESH_WITH_RESUME_ARTIFACT"),
            Some("true")
        );
        assert_eq!(
            continuous_manifest_env_value("SOCCER_POSTGRES_FLUSH_POLICY_VERSIONS_BEFORE_NEW_SIM"),
            Some("true")
        );
        assert_eq!(
            continuous_manifest_env_value("SOCCER_EVOLUTION_ENABLED"),
            Some("true")
        );
        assert_eq!(
            continuous_manifest_env_value("SOCCER_EVOLUTION_INTERVAL_GAMES"),
            Some("5")
        );
        assert_eq!(
            continuous_manifest_env_value("DD_SOCCER_ENABLE_BACK_FOUR_LINE_MODEL"),
            Some("true")
        );
        assert_eq!(
            continuous_manifest_env_value("DD_SOCCER_ENABLE_MIDFIELD_LINE_MODEL"),
            Some("true")
        );
        assert_eq!(
            continuous_manifest_env_value("DD_SOCCER_ENABLE_PITCH_VALUE_REWARD"),
            Some("true")
        );
        assert_eq!(
            continuous_manifest_env_value("DD_SOCCER_ENABLE_LOOSE_BALL_COMMIT_MODEL"),
            Some("true")
        );
        assert_eq!(
            continuous_manifest_env_value("DD_SOCCER_ENABLE_LEARNED_SPACING_TARGET"),
            Some("true")
        );
        assert_continuous_manifest_contains(
            "require_value DD_SOCCER_ENABLE_LOOSE_BALL_COMMIT_MODEL true",
        );
        assert_continuous_manifest_contains(
            "require_value DD_SOCCER_ENABLE_LEARNED_SPACING_TARGET true",
        );
        assert_eq!(
            continuous_manifest_env_value("SOCCER_GAME_ARTIFACT_MODE"),
            Some("summary")
        );
        assert_eq!(
            continuous_manifest_env_value("SOCCER_WRITE_GAME_ARTIFACTS"),
            Some("false")
        );
        assert_continuous_manifest_contains("key: RDS_DATABASE_URL");
        assert_continuous_manifest_contains("memory: 16Gi");
        assert_continuous_manifest_contains("memory: 64Gi");
        assert_continuous_manifest_contains("cpu: \"2\"");
        assert_continuous_manifest_contains("cpu: \"8\"");
    }

    #[test]
    fn continuous_manifest_batches_postgres_writes_by_parallel_wave() {
        assert_eq!(continuous_manifest_env_value("SOCCER_GAMES"), Some("100"));
        assert_eq!(
            continuous_manifest_env_value("SOCCER_PARALLEL_GAMES"),
            Some("1")
        );
        assert_eq!(
            continuous_manifest_env_value("SOCCER_DT_SECONDS"),
            Some("0.2")
        );
        assert_eq!(
            continuous_manifest_env_value("SOCCER_POSTGRES_POLICY_VERSION_INTERVAL_GAMES"),
            continuous_manifest_env_value("SOCCER_PARALLEL_GAMES")
        );
        assert_eq!(
            continuous_manifest_env_value("SOCCER_POSTGRES_COMPLETED_RUN_BATCH_GAMES"),
            continuous_manifest_env_value("SOCCER_PARALLEL_GAMES")
        );
        // Evolution interval is intentionally decoupled from the parallel wave
        // (the hardened manifest runs parallel=1 but evolves every 5 games — see
        // the `require_value SOCCER_EVOLUTION_INTERVAL_GAMES 5` startup preflight).
        assert_eq!(
            continuous_manifest_env_value("SOCCER_EVOLUTION_INTERVAL_GAMES"),
            Some("5")
        );
        assert_eq!(
            continuous_manifest_env_value("SOCCER_MAX_POLICY_ENTRIES_PER_TEAM"),
            Some("250000")
        );
        assert_eq!(
            continuous_manifest_env_value("SOCCER_MAX_POLICY_TARGET_ENTRIES_PER_TEAM"),
            Some("250000")
        );
    }

    #[test]
    fn continuous_manifest_uses_restart_safe_source_checkout() {
        assert_eq!(
            continuous_manifest_env_value("SOCCER_SOURCE_REPO"),
            Some("https://github.com/ORESoftware/soccer-sim-game-engine.rs.git")
        );
        assert_eq!(
            continuous_manifest_env_value("SOCCER_SOURCE_REF"),
            Some("learning")
        );
        assert_eq!(
            continuous_manifest_env_value("SOCCER_ENGINE_SOURCE_REPO"),
            Some("https://github.com/ORESoftware/discrete-event-system.rs.git")
        );
        assert_eq!(
            continuous_manifest_env_value("SOCCER_ENGINE_SOURCE_REF"),
            Some("main")
        );
        assert_continuous_manifest_contains("source_ref=\"${SOCCER_SOURCE_REF:-learning}\"");
        assert_continuous_manifest_contains("engine_ref=\"${SOCCER_ENGINE_SOURCE_REF:-main}\"");
        assert_continuous_manifest_contains("sync_ref() {");
        assert_continuous_manifest_contains("if [ -d \"${dir}/.git\" ]; then");
        assert_continuous_manifest_contains("git -C \"${dir}\" fetch --depth 1 origin \"${ref}\"");
        assert_continuous_manifest_contains("git -C \"${dir}\" switch --detach FETCH_HEAD");
        assert_continuous_manifest_contains("source_checkout_mode=fetch dir=${dir} ref=${ref}");
        assert_continuous_manifest_contains("source_checkout_mode=clone dir=${dir} ref=${ref}");
        assert_continuous_manifest_contains(
            "git clone --depth 1 --filter=blob:none --no-checkout \"${repo}\" \"${dir}\"",
        );
        assert_continuous_manifest_contains(
            "sync_ref \"${engine_repo}\" \"${engine_ref}\" \"${engine}\"",
        );
        assert_continuous_manifest_contains(
            "sync_ref \"${source_repo}\" \"${source_ref}\" \"${soccer}\"",
        );
        assert_continuous_manifest_contains(
            "/usr/local/cargo/bin/cargo build --release --features postgres-persistence --bin main_soccer_learning_run",
        );
    }

    #[test]
    fn continuous_manifest_varies_cycle_run_ids_and_seeds() {
        assert_eq!(
            continuous_manifest_env_value("SOCCER_RUN_ID_PREFIX"),
            Some("codex-soccer-learning-overnight")
        );
        assert_eq!(
            continuous_manifest_env_value("SOCCER_SEED_BASE"),
            Some("2026")
        );
        assert_eq!(
            continuous_manifest_env_value("SOCCER_SEED_STRIDE"),
            Some("1000")
        );
        assert_eq!(
            continuous_manifest_env_value("SOCCER_EVOLUTION_SEED_BASE"),
            Some("20260608")
        );
        assert_continuous_manifest_contains("stamp=\"$(date -u +%Y%m%dT%H%M%SZ)\"");
        assert_continuous_manifest_contains(
            "export SOCCER_SEED=\"$((seed_base + cycle * seed_stride))\"",
        );
        assert_continuous_manifest_contains(
            "export SOCCER_EVOLUTION_SEED=\"$((evolution_seed_base + cycle))\"",
        );
        assert_continuous_manifest_contains(
            "export SOCCER_RUN_ID=\"${run_prefix}-${stamp}-c${cycle}\"",
        );
        assert_continuous_manifest_contains("continuous_cycle_start cycle=${cycle} run_id=${SOCCER_RUN_ID} seed=${SOCCER_SEED} evolution_seed=${SOCCER_EVOLUTION_SEED}");
    }

    #[test]
    fn continuous_manifest_marks_ready_only_after_release_build() {
        assert_continuous_manifest_contains(
            "/usr/local/cargo/bin/cargo build --release --features postgres-persistence --bin main_soccer_learning_run",
        );
        assert_continuous_manifest_contains(
            "ready_file=\"/tmp/${run_prefix}-ready-${HOSTNAME:-pod}\"",
        );
        assert_continuous_manifest_contains("touch \"${ready_file}\"");
        assert_continuous_manifest_contains("readinessProbe:");
        assert_continuous_manifest_contains(
            "test -f \"/tmp/codex-soccer-learning-overnight-ready-${HOSTNAME:-pod}\"",
        );
        assert_continuous_manifest_contains("progressDeadlineSeconds: 1200");
        assert_continuous_manifest_contains("revisionHistoryLimit: 2");
    }

    #[test]
    fn run_settings_reject_dt_above_soccer_runtime_limit() {
        let options = SoccerQPolicyOptions::default();
        let tactical_learning = SoccerTacticalLearningWeights::default();

        assert!(validate_run_settings(
            1,
            2,
            45.0,
            90.0,
            900.0,
            4.0,
            4,
            1,
            1,
            &options,
            &tactical_learning,
        )
        .is_ok());
        let err = validate_run_settings(
            1,
            2,
            45.0,
            90.0,
            900.0,
            4.01,
            4,
            1,
            1,
            &options,
            &tactical_learning,
        )
        .expect_err("dt above runtime-safe bound should fail");
        assert!(err.to_string().contains("[0.01, 4.0]"));
    }

    fn empty_pg_batch(
        experiment_id: &str,
        runner_id: &str,
        shard_index: usize,
        shard_count: usize,
    ) -> PostgresCompletedRunBatch {
        PostgresCompletedRunBatch {
            experiment_id: experiment_id.to_string(),
            runner_id: runner_id.to_string(),
            completed_run_retention_games: 0,
            policy_version_batches: 0,
            pending_policy_versions: Vec::new(),
            pending_runs: Vec::new(),
            shard_index,
            shard_count,
        }
    }

    fn test_policy_with_action(value: f64, visits: u32) -> SoccerTeamQPolicies {
        let state = serde_json::from_value(serde_json::json!({
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
        .expect("test state");
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

    fn completed_game_with_starting_tactical_learning(
        episode: usize,
        starting_tactical_learning: SoccerTacticalLearningWeights,
        tactical_summary: SoccerTacticalLearningSummary,
    ) -> CompletedGame {
        let policies = test_policy_with_action(1.0 + episode as f64, 2 + episode as u32);
        let starting_policies = Arc::new(policies.clone());
        let summary = soccer_engine::des::general::soccer::MatchSummary {
            score_home: 1,
            score_away: 0,
            ticks: 10,
            simulated_seconds: 1.0,
            stats: Default::default(),
        };
        let artifact = SoccerTeamPolicyArtifact {
            config: MatchConfig::default(),
            summary: summary.clone(),
            learning: soccer_engine::des::general::soccer::SoccerLearningSnapshot::default(),
            tactical_summary,
            adversarial: false,
            home_options: Some(SoccerQPolicyOptions::default()),
            away_options: Some(SoccerQPolicyOptions::default()),
            home_entries: policies.home.entries(),
            home_target_entries: policies.home.target_entries(),
            home_action_probabilities: Vec::new(),
            away_entries: policies.away.entries(),
            away_target_entries: policies.away.target_entries(),
            away_action_probabilities: Vec::new(),
            events: Vec::new(),
        };

        CompletedGame {
            episode_summary: SoccerSelfPlayEpisodeSummary {
                episode,
                seed: 1 + episode as u64,
                summary,
                transitions: 0,
                home_policy_entries: 1,
                home_policy_target_entries: 0,
                away_policy_entries: 0,
                away_policy_target_entries: 0,
            },
            artifact,
            starting_policies,
            starting_policy_version_id: None,
            starting_policy_generation: 0,
            starting_tactical_learning,
            policies,
            config_moments: Vec::new(),
            pass_outcome_samples: Vec::new(),
            neural_network: None,
            world_model: None,
            elapsed_seconds: 0.0,
        }
    }

    fn test_neural_snapshot_with_training_steps(
        training_steps: usize,
    ) -> SoccerNeuralNetworkSnapshot {
        SoccerNeuralNetworkSnapshot {
            input_dim: 3,
            output_dim: 1,
            parameter_count: 0,
            l2_norm: 0.0,
            layers: Vec::new(),
            training_steps,
            average_loss: Some(0.1),
            target_popart: None,
            policy_head: None,
            line_depth_head: None,
        }
    }

    fn completed_game_with_score_and_neural_steps(
        episode: usize,
        score_home: u32,
        score_away: u32,
        training_steps: usize,
    ) -> CompletedGame {
        let mut game = completed_game_with_starting_tactical_learning(
            episode,
            SoccerTacticalLearningWeights::default(),
            SoccerTacticalLearningSummary::default(),
        );
        game.episode_summary.summary.score_home = score_home;
        game.episode_summary.summary.score_away = score_away;
        game.artifact.summary = game.episode_summary.summary.clone();
        game.neural_network = Some(test_neural_snapshot_with_training_steps(training_steps));
        game
    }

    #[test]
    fn completed_game_delta_uses_its_own_starting_policy_snapshot() {
        let starting_policies = Arc::new(test_policy_with_action(1.0, 2));
        let policies = test_policy_with_action(3.0, 5);
        let summary = soccer_engine::des::general::soccer::MatchSummary {
            score_home: 2,
            score_away: 0,
            ticks: 10,
            simulated_seconds: 1.0,
            stats: Default::default(),
        };
        let artifact = SoccerTeamPolicyArtifact {
            config: MatchConfig::default(),
            summary: summary.clone(),
            learning: soccer_engine::des::general::soccer::SoccerLearningSnapshot::default(),
            tactical_summary: Default::default(),
            adversarial: false,
            home_options: Some(SoccerQPolicyOptions::default()),
            away_options: Some(SoccerQPolicyOptions::default()),
            home_entries: policies.home.entries(),
            home_target_entries: policies.home.target_entries(),
            home_action_probabilities: Vec::new(),
            away_entries: policies.away.entries(),
            away_target_entries: policies.away.target_entries(),
            away_action_probabilities: Vec::new(),
            events: Vec::new(),
        };
        let game = CompletedGame {
            episode_summary: SoccerSelfPlayEpisodeSummary {
                episode: 0,
                seed: 1,
                summary,
                transitions: 0,
                home_policy_entries: 1,
                home_policy_target_entries: 0,
                away_policy_entries: 0,
                away_policy_target_entries: 0,
            },
            artifact,
            starting_policies,
            starting_policy_version_id: Some("pg-v3".to_string()),
            starting_policy_generation: 3,
            starting_tactical_learning: SoccerTacticalLearningWeights::default(),
            policies,
            config_moments: Vec::new(),
            pass_outcome_samples: Vec::new(),
            neural_network: None,
            world_model: None,
            elapsed_seconds: 0.0,
        };

        let completed = soccer_learning_completed_game_from_completed(&game);

        assert_eq!(completed.delta.entries.len(), 1);
        assert_eq!(completed.delta.entries[0].visit_delta, 3);
        assert_eq!(completed.delta.entries[0].before_value, 1.0);
        assert_eq!(completed.delta.entries[0].after_value, 3.0);
    }

    #[test]
    fn completed_game_conversion_preserves_config_moments() {
        let mut game = completed_game_with_starting_tactical_learning(
            0,
            SoccerTacticalLearningWeights::default(),
            SoccerTacticalLearningSummary::default(),
        );
        game.config_moments.push(SoccerConfigMomentInsert {
            team: Team::Home,
            tick: 42,
            role: PlayerRole::Midfielder,
            action: "pass-forward".to_string(),
            reward: -0.25,
            nstep_return: -0.75,
            value: Some(-0.5),
            embedding: vec![0.0; SOCCER_MOMENT_EMBEDDING_DIM],
            features: vec![0.0; CONFIG_FEATURE_DIM],
            embedding_assigned: vec![0.0; SOCCER_MOMENT_EMBEDDING_DIM],
            features_assigned: vec![0.0; CONFIG_FEATURE_DIM],
        });

        let completed = soccer_learning_completed_game_from_completed(&game);

        assert_eq!(completed.config_moments.len(), 1);
        assert_eq!(completed.config_moments[0].tick, 42);
        assert_eq!(completed.config_moments[0].action, "pass-forward");
        assert_eq!(
            completed.config_moments[0].features.len(),
            CONFIG_FEATURE_DIM
        );
        assert_eq!(
            completed.config_moments[0].embedding.len(),
            SOCCER_MOMENT_EMBEDDING_DIM
        );
    }

    #[test]
    fn tactical_genome_parents_use_each_games_starting_weights() {
        let mut flank_weights = SoccerTacticalLearningWeights::default();
        flank_weights.attack_flank_lane_weight = 1.75;
        let mut contract_weights = SoccerTacticalLearningWeights::default();
        contract_weights.defense_contract_delta_weight = 0.55;
        let games = vec![
            completed_game_with_starting_tactical_learning(
                0,
                flank_weights.clone(),
                SoccerTacticalLearningSummary::default(),
            ),
            completed_game_with_starting_tactical_learning(
                1,
                contract_weights.clone(),
                SoccerTacticalLearningSummary::default(),
            ),
        ];
        let ranked = vec![(1, 4.0), (0, 3.0)];

        let parents = tactical_genome_parents_from_completed_games(&games, &ranked, 2);

        assert_eq!(parents.len(), 2);
        assert_eq!(
            parents[0].weights.defense_contract_delta_weight,
            contract_weights.defense_contract_delta_weight
        );
        assert_eq!(
            parents[1].weights.attack_flank_lane_weight,
            flank_weights.attack_flank_lane_weight
        );
        assert_eq!(parents[0].fitness, 4.0);
        assert_eq!(parents[1].fitness, 3.0);
    }

    #[test]
    fn run_evolution_metadata_persists_tactical_genome_weights() {
        let options = SoccerEvolutionOptions {
            mutation_rate: 0.12,
            crossover_rate: 0.62,
            exploration_scale: 1.05,
            population_size: 24,
            seed: 90210,
            ..Default::default()
        };
        let mut previous = SoccerTacticalLearningWeights::default();
        previous.attack_flank_lane_weight = 1.25;
        previous.defense_contract_delta_weight = 0.35;
        let mut evolved = previous.clone();
        evolved.attack_flank_lane_weight = 1.85;
        evolved.defense_contract_delta_weight = 0.70;

        let promotion_evaluation = SoccerPolicyPromotionGateEvaluation {
            enabled: true,
            eligible: true,
            sample_games: 12,
            min_sample_games: 8,
            mean_match_fitness: 0.65,
            best_match_fitness: 2.75,
            mean_play_quality: 1.4,
            mean_conceded_goals: 1.2,
            mean_goal_margin: 2.0,
            mean_chain_net_loss: 3.5,
            rejection_reasons: Vec::new(),
        };
        let metadata = run_evolution_search_metadata(
            40,
            4,
            16,
            12,
            2.75,
            options,
            &previous,
            &evolved,
            "duels",
            &promotion_evaluation,
            true,
        );

        assert_eq!(metadata["algorithm"], "evolutionary-genetic-programming");
        assert_eq!(
            metadata["tactical"]["algorithm"],
            "evolutionary-genetic-programming-tactical-search"
        );
        assert_eq!(metadata["completedGames"].as_u64(), Some(40));
        assert_eq!(metadata["curriculumStage"], "duels");
        assert_eq!(metadata["candidatePromoted"], true);
        assert_eq!(metadata["promotion"]["status"], "active");
        assert_eq!(
            metadata["promotion"]["gate"]["sampleGames"].as_u64(),
            Some(12)
        );
        assert_eq!(metadata["tactical"]["sampleCount"].as_u64(), Some(12));
        assert_eq!(
            metadata["tactical"]["previousTacticalLearning"]["attackFlankLaneWeight"].as_f64(),
            Some(previous.attack_flank_lane_weight)
        );
        assert_eq!(
            metadata["tactical"]["evolvedTacticalLearning"]["defenseContractDeltaWeight"].as_f64(),
            Some(evolved.defense_contract_delta_weight)
        );
        assert_eq!(
            metadata["tactical"]["previousTacticalLearningFingerprint"].as_u64(),
            Some(soccer_tactical_learning_weights_fingerprint(&previous))
        );
        assert_eq!(
            metadata["tactical"]["evolvedTacticalLearningFingerprint"].as_u64(),
            Some(soccer_tactical_learning_weights_fingerprint(&evolved))
        );
        let refreshed_options =
            soccer_evolution_options_from_search_metadata(Some(&metadata), Default::default())
                .expect("evolution options from run metadata");
        assert_eq!(refreshed_options.population_size, 24);
        assert_eq!(refreshed_options.seed, 90210);
        assert!((refreshed_options.crossover_rate - 0.62).abs() < 1e-12);
    }

    #[test]
    fn batch_evolution_window_keeps_prior_batch_genomes_for_search() {
        let mut flank_weights = SoccerTacticalLearningWeights::default();
        flank_weights.attack_flank_lane_weight = 1.75;
        let mut contract_weights = SoccerTacticalLearningWeights::default();
        contract_weights.defense_contract_delta_weight = 0.55;
        let mut samples = std::collections::VecDeque::new();

        push_evolution_search_samples(
            &mut samples,
            &[completed_game_with_starting_tactical_learning(
                0,
                flank_weights.clone(),
                SoccerTacticalLearningSummary::default(),
            )],
            2,
        );
        push_evolution_search_samples(
            &mut samples,
            &[completed_game_with_starting_tactical_learning(
                1,
                contract_weights.clone(),
                SoccerTacticalLearningSummary::default(),
            )],
            2,
        );

        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].policies.home.entries()[0].visits, 2);
        assert_eq!(samples[1].policies.home.entries()[0].visits, 3);
        let parents =
            tactical_genome_parents_from_evolution_samples(&samples, &[(0, 3.0), (1, 4.0)], 2);
        assert_eq!(
            parents[0].weights.attack_flank_lane_weight,
            flank_weights.attack_flank_lane_weight
        );
        assert_eq!(
            parents[1].weights.defense_contract_delta_weight,
            contract_weights.defense_contract_delta_weight
        );

        push_evolution_search_samples(
            &mut samples,
            &[completed_game_with_starting_tactical_learning(
                2,
                SoccerTacticalLearningWeights::default(),
                SoccerTacticalLearningSummary::default(),
            )],
            2,
        );

        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].policies.home.entries()[0].visits, 3);
        assert_eq!(samples[1].policies.home.entries()[0].visits, 4);
    }

    #[test]
    fn policy_promotion_summary_window_keeps_recent_games() {
        let mut summaries = std::collections::VecDeque::new();
        for score in 1..=3 {
            let summary = MatchSummary {
                score_home: score,
                score_away: 0,
                ticks: 10,
                simulated_seconds: 1.0,
                stats: Default::default(),
            };
            push_policy_promotion_summary(&mut summaries, &summary, 2);
        }

        assert_eq!(summaries.len(), 2);
        assert_eq!(summaries[0].score_home, 2);
        assert_eq!(summaries[1].score_home, 3);
        let evaluation = evaluate_soccer_policy_promotion_gate(
            summaries.iter().collect::<Vec<_>>(),
            SoccerPolicyPromotionGateConfig {
                min_sample_games: 2,
                ..Default::default()
            },
        );
        assert_eq!(evaluation.sample_games, 2);
    }

    #[test]
    fn policy_promotion_incumbent_baseline_decodes_metadata() {
        let metadata = serde_json::json!({
            "promotion": {
                "gate": {
                    "sampleGames": 8,
                    "meanMatchFitness": 1.25,
                    "bestMatchFitness": 2.50,
                    "meanPlayQuality": 0.42
                }
            }
        });

        let baseline = policy_promotion_incumbent_baseline_from_search_metadata(Some(&metadata))
            .expect("baseline");

        assert_eq!(baseline.sample_games, 8);
        assert!((baseline.mean_match_fitness - 1.25).abs() < 1e-12);
        assert!((baseline.best_match_fitness - 2.50).abs() < 1e-12);
        assert!((baseline.mean_play_quality - 0.42).abs() < 1e-12);
    }

    #[test]
    fn policy_promotion_incumbent_baseline_retains_stronger_actual_policy() {
        let current = PolicyPromotionIncumbentBaseline {
            sample_games: 8,
            mean_match_fitness: 1.20,
            best_match_fitness: 2.40,
            mean_play_quality: 0.44,
        };
        let candidate = PolicyPromotionIncumbentBaseline {
            sample_games: 16,
            mean_match_fitness: 0.95,
            best_match_fitness: 2.80,
            mean_play_quality: 0.39,
        };

        let retained = retain_stronger_policy_promotion_baseline(Some(current), Some(candidate))
            .expect("retained baseline");

        assert_eq!(retained.sample_games, 8);
        assert!((retained.mean_match_fitness - 1.20).abs() < 1e-12);
        assert!((retained.best_match_fitness - 2.40).abs() < 1e-12);
        assert!((retained.mean_play_quality - 0.44).abs() < 1e-12);
    }

    #[test]
    fn rejected_promotion_neural_checkpoint_requires_new_snapshot() {
        assert!(should_write_neural_checkpoint_for_rejected_promotion(
            true,
            true,
            true,
            Some(101),
            Some(100)
        ));
        assert!(!should_write_neural_checkpoint_for_rejected_promotion(
            false,
            true,
            true,
            Some(101),
            Some(100)
        ));
        assert!(!should_write_neural_checkpoint_for_rejected_promotion(
            true,
            true,
            false,
            Some(101),
            Some(100)
        ));
        assert!(!should_write_neural_checkpoint_for_rejected_promotion(
            true,
            true,
            true,
            Some(101),
            Some(101)
        ));
        assert!(!should_write_neural_checkpoint_for_rejected_promotion(
            true,
            true,
            true,
            None,
            Some(101)
        ));
    }

    #[test]
    fn policy_promotion_incumbent_gate_blocks_regression() {
        let mut evaluation = SoccerPolicyPromotionGateEvaluation {
            enabled: true,
            eligible: true,
            sample_games: 8,
            min_sample_games: 8,
            mean_match_fitness: 0.90,
            best_match_fitness: 1.80,
            mean_play_quality: 0.37,
            mean_conceded_goals: 1.0,
            mean_goal_margin: 1.0,
            mean_chain_net_loss: 0.0,
            rejection_reasons: Vec::new(),
        };
        let incumbent = PolicyPromotionIncumbentBaseline {
            sample_games: 8,
            mean_match_fitness: 1.00,
            best_match_fitness: 2.00,
            mean_play_quality: 0.40,
        };

        apply_policy_promotion_incumbent_gate(
            &mut evaluation,
            Some(incumbent),
            true,
            0.0,
            0.0,
            0.02,
        );

        assert!(!evaluation.eligible);
        assert!(evaluation
            .rejection_reasons
            .iter()
            .any(|reason| reason.contains("mean_match_fitness")));
        assert!(evaluation
            .rejection_reasons
            .iter()
            .any(|reason| reason.contains("mean_play_quality")));
    }

    #[test]
    fn postgres_refresh_clears_batch_evolution_sample_window() {
        let mut flank_weights = SoccerTacticalLearningWeights::default();
        flank_weights.attack_flank_lane_weight = 1.75;
        let mut contract_weights = SoccerTacticalLearningWeights::default();
        contract_weights.defense_contract_delta_weight = 0.55;
        let mut samples = std::collections::VecDeque::new();
        push_evolution_search_samples(
            &mut samples,
            &[
                completed_game_with_starting_tactical_learning(
                    0,
                    flank_weights.clone(),
                    SoccerTacticalLearningSummary::default(),
                ),
                completed_game_with_starting_tactical_learning(
                    1,
                    contract_weights.clone(),
                    SoccerTacticalLearningSummary::default(),
                ),
            ],
            8,
        );

        let cleared = clear_evolution_search_samples_after_postgres_refresh(
            "test_postgres_refresh_evolution_samples_reset",
            3,
            Some("pg-v3"),
            &mut samples,
        );

        assert_eq!(cleared, 2);
        assert!(samples.is_empty());
        push_evolution_search_samples(
            &mut samples,
            &[completed_game_with_starting_tactical_learning(
                2,
                SoccerTacticalLearningWeights::default(),
                SoccerTacticalLearningSummary::default(),
            )],
            8,
        );
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].policies.home.entries()[0].visits, 4);
    }

    #[test]
    fn postgres_tactical_learning_is_cloned_into_next_worker_config() {
        let mut active_weights = SoccerTacticalLearningWeights::default();
        let mut config = MatchConfig {
            tactical_learning: active_weights.clone(),
            ..Default::default()
        };
        let mut postgres_weights = SoccerTacticalLearningWeights::default();
        postgres_weights.attack_flank_lane_weight = 1.42;
        postgres_weights.defense_contract_delta_weight = 1.17;
        postgres_weights.formation_lp_alignment_weight = 0.37;

        let applied = maybe_apply_postgres_tactical_learning(
            "test_postgres_tactical_learning",
            17,
            "pg-v17",
            17,
            &mut config,
            &mut active_weights,
            Some(postgres_weights.clone()),
        )
        .expect("valid postgres tactical learning weights");
        let mut episode_config = config.clone();
        episode_config.seed = 9090;

        assert!(applied);
        assert_eq!(
            active_weights.attack_flank_lane_weight,
            postgres_weights.attack_flank_lane_weight
        );
        assert_eq!(
            episode_config.tactical_learning.attack_flank_lane_weight,
            postgres_weights.attack_flank_lane_weight
        );
        assert_eq!(
            episode_config
                .tactical_learning
                .defense_contract_delta_weight,
            postgres_weights.defense_contract_delta_weight
        );
        assert_eq!(
            episode_config
                .tactical_learning
                .formation_lp_alignment_weight,
            postgres_weights.formation_lp_alignment_weight
        );
    }

    #[test]
    fn batch_new_sims_apply_postgres_weights_before_each_worker_task() {
        let mut active_weights = SoccerTacticalLearningWeights::default();
        let mut config = MatchConfig {
            tactical_learning: active_weights.clone(),
            ..Default::default()
        };
        let mut submitted_weights = Vec::new();

        for episode in 0..3 {
            let mut postgres_weights = SoccerTacticalLearningWeights::default();
            postgres_weights.attack_flank_lane_weight = 1.15 + episode as f64 * 0.09;
            postgres_weights.defense_contract_delta_weight = 0.88 + episode as f64 * 0.06;
            postgres_weights.formation_lp_alignment_weight = 0.24 + episode as f64 * 0.04;
            maybe_apply_postgres_tactical_learning(
                "test_postgres_tactical_learning_for_worker",
                episode + 1,
                &format!("pg-v-{episode}"),
                episode as i32,
                &mut config,
                &mut active_weights,
                Some(postgres_weights),
            )
            .expect("valid postgres tactical learning weights");

            let mut episode_config = config.clone();
            episode_config.seed = 4400 + episode as u32;
            submitted_weights.push((
                episode,
                episode_config.tactical_learning.attack_flank_lane_weight,
                episode_config
                    .tactical_learning
                    .defense_contract_delta_weight,
                episode_config
                    .tactical_learning
                    .formation_lp_alignment_weight,
            ));
        }

        assert_eq!(submitted_weights.len(), 3);
        for (episode, attack_flank, defense_contract, lp_alignment) in submitted_weights {
            assert!((attack_flank - (1.15 + episode as f64 * 0.09)).abs() < 1e-12);
            assert!((defense_contract - (0.88 + episode as f64 * 0.06)).abs() < 1e-12);
            assert!((lp_alignment - (0.24 + episode as f64 * 0.04)).abs() < 1e-12);
        }
    }

    #[test]
    fn batch_new_sims_apply_postgres_evolution_options_before_search() {
        let mut evolution_options = SoccerEvolutionOptions::default();
        let mut submitted_options = Vec::new();

        for episode in 0..3 {
            let metadata = serde_json::json!({
                "algorithm": "evolutionary-genetic-programming",
                "options": {
                    "mutationRate": 0.06 + episode as f64 * 0.02,
                    "crossoverRate": 0.40 + episode as f64 * 0.05,
                    "explorationScale": 0.55 + episode as f64 * 0.10,
                    "populationSize": 8 + episode * 4,
                    "seed": 900 + episode as u64
                }
            });

            assert!(maybe_apply_postgres_evolution_options(
                "test_postgres_evolution_options_for_worker",
                episode + 1,
                &format!("pg-v-{episode}"),
                episode as i32,
                &mut evolution_options,
                Some(&metadata),
            ));
            submitted_options.push((episode, evolution_options));
        }

        assert_eq!(submitted_options.len(), 3);
        for (episode, options) in submitted_options {
            assert!((options.mutation_rate - (0.06 + episode as f64 * 0.02)).abs() < 1e-12);
            assert!((options.crossover_rate - (0.40 + episode as f64 * 0.05)).abs() < 1e-12);
            assert!((options.exploration_scale - (0.55 + episode as f64 * 0.10)).abs() < 1e-12);
            assert_eq!(options.population_size, 8 + episode * 4);
            assert_eq!(options.seed, 900 + episode as u64);
        }
    }

    #[test]
    fn batch_loaded_policy_evolution_option_refresh_clears_stale_search_window() {
        let mut evolution_options = SoccerEvolutionOptions::default();
        let mut samples = std::collections::VecDeque::new();
        push_evolution_search_samples(
            &mut samples,
            &[completed_game_with_starting_tactical_learning(
                0,
                SoccerTacticalLearningWeights::default(),
                SoccerTacticalLearningSummary::default(),
            )],
            8,
        );
        let metadata = serde_json::json!({
            "algorithm": "geneticProgramming",
            "options": {
                "mutationRate": 0.11,
                "crossoverRate": 0.64,
                "explorationScale": 1.05,
                "populationSize": 24,
                "seed": 4217
            }
        });

        let changed = maybe_apply_postgres_evolution_options(
            "test_loaded_policy_evolution_options_for_worker",
            5,
            "pg-v-gp",
            12,
            &mut evolution_options,
            Some(&metadata),
        );
        if changed {
            clear_evolution_search_samples_after_postgres_refresh(
                "test_loaded_policy_evolution_samples_reset",
                5,
                Some("pg-v-gp"),
                &mut samples,
            );
        }

        assert!(changed);
        assert!(samples.is_empty());
        assert_eq!(evolution_options.population_size, 24);
        assert_eq!(evolution_options.seed, 4217);
        assert!((evolution_options.crossover_rate - 0.64).abs() < 1e-12);
        assert!((evolution_options.exploration_scale - 1.05).abs() < 1e-12);
    }

    #[test]
    fn batch_new_sims_cache_refreshed_policy_and_neural_snapshot_into_worker_task() {
        let mut policies = test_policy_with_action(-1.0, 1);
        let mut latest_neural_network = None::<SoccerNeuralNetworkSnapshot>;
        let refreshed_policies = test_policy_with_action(7.5, 11);
        let refreshed_neural_network = SoccerNeuralNetworkSnapshot {
            input_dim: 3,
            output_dim: 2,
            parameter_count: 8,
            l2_norm: 0.625,
            layers: Vec::new(),
            training_steps: 0,
            average_loss: None,
            target_popart: None,
            policy_head: None,
            line_depth_head: None,
        };

        assert_eq!(policies.home.entries()[0].visits, 1);
        assert!(latest_neural_network.is_none());
        policies = refreshed_policies;
        latest_neural_network = Some(refreshed_neural_network.clone());
        let mut policy_cache = None::<CachedTeamQPoliciesArc>;
        let mut neural_cache = None::<CachedNeuralNetworkSnapshotArc>;
        let episode_starting_policies = cached_team_policies_arc(&policies, &mut policy_cache);
        let repeated_starting_policies = cached_team_policies_arc(&policies, &mut policy_cache);
        let episode_starting_neural_network =
            cached_neural_network_snapshot_arc(&latest_neural_network, &mut neural_cache);
        let repeated_starting_neural_network =
            cached_neural_network_snapshot_arc(&latest_neural_network, &mut neural_cache);
        let task = SoccerLearningWorkerTask {
            episode: 0,
            config: MatchConfig::default(),
            starting_policies: episode_starting_policies,
            starting_policy_version_id: Some("pg-v11".to_string()),
            starting_policy_generation: 11,
            initial_neural_network: episode_starting_neural_network.as_ref().map(Arc::clone),
            analytic_neural_opponent: true,
            adversarial_moment_windows: Arc::new(Vec::new()),
            print_progress: false,
            neural_drain_timeout: Duration::from_millis(0),
            retain_neural_network_in_game_artifact: false,
        };

        let entries = task.starting_policies.home.entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].action, "pass");
        assert!((entries[0].value - 7.5).abs() < 1e-12);
        assert_eq!(entries[0].visits, 11);
        assert!(Arc::ptr_eq(
            &task.starting_policies,
            &repeated_starting_policies
        ));
        policies = test_policy_with_action(8.25, 11);
        let updated_starting_policies = cached_team_policies_arc(&policies, &mut policy_cache);
        assert!(!Arc::ptr_eq(
            &task.starting_policies,
            &updated_starting_policies
        ));
        let neural = task
            .initial_neural_network
            .as_ref()
            .expect("worker neural snapshot");
        assert!(Arc::ptr_eq(
            neural,
            repeated_starting_neural_network
                .as_ref()
                .expect("cached worker neural snapshot")
        ));
        assert_eq!(
            neural.parameter_count,
            refreshed_neural_network.parameter_count
        );
        assert_eq!(neural.l2_norm, refreshed_neural_network.l2_norm);
        let mut updated_neural_network = refreshed_neural_network.clone();
        updated_neural_network.l2_norm += 0.125;
        latest_neural_network = Some(updated_neural_network);
        let updated_starting_neural_network =
            cached_neural_network_snapshot_arc(&latest_neural_network, &mut neural_cache)
                .expect("updated cached worker neural snapshot");
        assert!(!Arc::ptr_eq(neural, &updated_starting_neural_network));
        assert_eq!(task.starting_policy_version_id.as_deref(), Some("pg-v11"));
        assert_eq!(task.starting_policy_generation, 11);
    }

    #[test]
    fn tactical_learning_match_detects_formation_lp_alignment_weight() {
        let left = SoccerTacticalLearningWeights::default();
        let mut right = left.clone();
        right.formation_lp_alignment_weight += 0.05;

        assert!(!tactical_learning_weights_match(&left, &right));
    }

    #[test]
    fn default_postgres_policy_versions_are_batched_for_single_game_workers() {
        assert_eq!(default_postgres_policy_version_interval_games(0), 10);
        assert_eq!(default_postgres_policy_version_interval_games(1), 10);
        assert_eq!(default_postgres_policy_version_interval_games(4), 10);
    }

    #[test]
    fn default_postgres_policy_versions_respect_large_parallel_batches() {
        assert_eq!(default_postgres_policy_version_interval_games(16), 16);
        assert_eq!(default_postgres_policy_version_interval_games(64), 64);
    }

    #[test]
    fn default_postgres_completed_runs_are_batched_for_single_game_workers() {
        assert_eq!(default_postgres_completed_run_batch_games(0), 10);
        assert_eq!(default_postgres_completed_run_batch_games(1), 10);
        assert_eq!(default_postgres_completed_run_batch_games(4), 10);
    }

    #[test]
    fn default_postgres_completed_runs_respect_large_parallel_batches() {
        assert_eq!(default_postgres_completed_run_batch_games(16), 16);
        assert_eq!(default_postgres_completed_run_batch_games(64), 64);
    }

    #[test]
    fn default_postgres_completed_run_retention_is_disabled() {
        assert_eq!(DEFAULT_SOCCER_POSTGRES_COMPLETED_RUN_RETENTION_GAMES, 0);
    }

    #[test]
    fn default_neural_drain_timeout_keeps_worker_boundary_wait_bounded() {
        assert_eq!(DEFAULT_SOCCER_NEURAL_DRAIN_TIMEOUT_MS, 2);
    }

    #[test]
    fn default_postgres_async_writer_stays_bounded() {
        assert_eq!(DEFAULT_SOCCER_POSTGRES_ASYNC_BATCH_QUEUE, 4);
    }

    #[test]
    fn default_postgres_async_writer_coalesces_io_batches() {
        assert_eq!(DEFAULT_SOCCER_POSTGRES_ASYNC_COALESCE_BATCHES, 8);
    }

    #[test]
    fn default_postgres_async_writer_waits_briefly_to_coalesce_io() {
        assert_eq!(DEFAULT_SOCCER_POSTGRES_ASYNC_COALESCE_WAIT_MS, 2);
    }

    #[test]
    fn async_postgres_writer_missing_db_drains_policy_version_wait_counter() {
        let _lock = SOCCER_RUN_PG_ENV_LOCK.lock().expect("lock pg env");
        let _guards = clear_postgres_env();
        let mut writer = AsyncPostgresCompletedRunWriter::start(1, 1, Duration::from_millis(0));
        let sender = writer.sender.as_ref().expect("writer sender").clone();
        sender
            .send(PostgresCompletedRunBatch {
                experiment_id: "experiment-a".to_string(),
                runner_id: "runner-a".to_string(),
                completed_run_retention_games: 0,
                policy_version_batches: 1,
                pending_policy_versions: Vec::new(),
                pending_runs: Vec::new(),
                shard_index: 0,
                shard_count: 1,
            })
            .expect("send synthetic batch");
        writer.pending_batches = 1;
        writer.pending_policy_version_batches = 1;

        let result = writer.drain_policy_versions_finished();

        assert!(result
            .expect_err("missing DB should fail")
            .contains("could not find a database URL"));
        assert_eq!(writer.pending_batches, 0);
        assert_eq!(writer.pending_policy_version_batches, 0);
        drop(sender);
        writer.sender.take();
        if let Some(handle) = writer.handle.take() {
            handle.join().expect("join async postgres writer");
        }
    }

    #[test]
    fn default_postgres_tactical_learning_is_authoritative() {
        assert!(DEFAULT_SOCCER_POSTGRES_TACTICAL_LEARNING_AUTHORITATIVE);
    }

    #[test]
    fn default_postgres_policy_heads_flush_before_new_sims() {
        assert!(DEFAULT_SOCCER_POSTGRES_FLUSH_POLICY_VERSIONS_BEFORE_NEW_SIM);
    }

    #[test]
    fn default_postgres_refreshes_before_new_sims_even_with_resume_artifacts() {
        assert!(DEFAULT_SOCCER_POSTGRES_REFRESH_WITH_RESUME_ARTIFACT);
        assert!(soccer_should_refresh_postgres_for_new_sim(
            true,
            DEFAULT_SOCCER_POSTGRES_REFRESH_WITH_RESUME_ARTIFACT
        ));
    }

    #[test]
    fn default_postgres_new_sim_plan_publishes_evolved_heads_before_reads() {
        let plan = soccer_postgres_new_sim_refresh_plan(
            true,
            DEFAULT_SOCCER_POSTGRES_REFRESH_WITH_RESUME_ARTIFACT,
            DEFAULT_SOCCER_POSTGRES_FLUSH_POLICY_VERSIONS_BEFORE_NEW_SIM,
            1,
            1,
        );

        assert!(plan.refresh_from_postgres);
        assert!(plan.flush_pending_policy_versions);
        assert!(plan.wait_for_async_policy_versions);
    }

    #[test]
    fn default_evolutionary_search_stays_enabled_for_new_sims() {
        let options = SoccerEvolutionOptions::default();

        assert!(DEFAULT_SOCCER_EVOLUTION_ENABLED);
        assert!(DEFAULT_SOCCER_EVOLUTION_ELITE_GAMES >= 1);
        assert_eq!(
            default_evolution_window_games(10, DEFAULT_SOCCER_EVOLUTION_ELITE_GAMES),
            10
        );
        assert_eq!(default_evolution_window_games(4, 10), 10);
        assert_eq!(default_evolution_window_games(0, 0), 1);
        assert!(options.population_size > 1);
        assert!(options.mutation_rate > 0.0);
        assert!(options.mutation_scale > 0.0);
        assert!(options.crossover_rate > 0.0);
    }

    #[test]
    fn default_evolution_interval_respects_parallelism() {
        assert_eq!(default_evolution_interval_games(0), 10);
        assert_eq!(default_evolution_interval_games(4), 10);
        assert_eq!(default_evolution_interval_games(16), 16);
    }

    #[test]
    fn postgres_policy_sources_name_evolutionary_search() {
        let label = soccer_learning_pg_version_label_with_suffix("run-a", 2, 4, "evolution");
        let long_checkpoint_label = soccer_learning_pg_version_label_with_suffix(
            &"long-run-prefix-".repeat(20),
            0,
            123,
            "neural-checkpoint",
        );

        assert_eq!(SOCCER_POLICY_SOURCE_MERGE, "merge");
        assert_eq!(SOCCER_POLICY_SOURCE_EVOLUTION, "mutation");
        assert_eq!(SOCCER_POLICY_SOURCE_NEURAL_CHECKPOINT, "replay");
        assert!(label.ends_with("-evolution"));
        assert!(!label.ends_with("-mutation"));
        assert!(long_checkpoint_label.len() <= SOCCER_PG_VERSION_LABEL_MAX_LEN);
        assert!(long_checkpoint_label.ends_with("-neural-checkpoint"));
    }

    #[test]
    fn postgres_batches_only_coalesce_for_same_run_and_shard() {
        let mut batch = empty_pg_batch("experiment-a", "runner-a", 0, 2);
        let mut different_retention = empty_pg_batch("experiment-a", "runner-a", 0, 2);
        different_retention.completed_run_retention_games = 10;

        assert!(batch.can_absorb(&empty_pg_batch("experiment-a", "runner-a", 0, 2)));
        assert!(!batch.can_absorb(&empty_pg_batch("experiment-b", "runner-a", 0, 2)));
        assert!(!batch.can_absorb(&empty_pg_batch("experiment-a", "runner-b", 0, 2)));
        assert!(!batch.can_absorb(&empty_pg_batch("experiment-a", "runner-a", 1, 2)));
        assert!(!batch.can_absorb(&empty_pg_batch("experiment-a", "runner-a", 0, 3)));
        assert!(!batch.can_absorb(&different_retention));

        batch.absorb(empty_pg_batch("experiment-a", "runner-a", 0, 2));
        assert!(batch.pending_policy_versions.is_empty());
        assert!(batch.pending_runs.is_empty());
    }

    #[test]
    fn postgres_batches_track_policy_version_sources() {
        let mut batch = empty_pg_batch("experiment-a", "runner-a", 0, 2);
        let mut other = empty_pg_batch("experiment-a", "runner-a", 0, 2);
        batch.policy_version_batches = 1;
        other.policy_version_batches = 1;

        batch.absorb(other);

        assert_eq!(batch.policy_version_batches, 2);
    }

    #[test]
    fn default_parallel_games_keeps_continuous_runner_single_game() {
        assert_eq!(default_soccer_parallel_games(), 1);
    }

    #[test]
    fn default_batch_runner_outputs_keep_per_game_io_off_hot_path() {
        assert!(!DEFAULT_SOCCER_WRITE_GAME_ARTIFACTS);
        assert!(DEFAULT_SOCCER_WRITE_FINAL_POLICY_ARTIFACT);
        assert!(DEFAULT_SOCCER_WRITE_EPISODE_LOG);
        assert!(!DEFAULT_SOCCER_PRINT_COMPLETED_GAMES);
    }

    #[test]
    fn default_disk_learning_artifacts_are_disabled_when_postgres_is_configured() {
        assert!(default_disk_learning_artifacts_enabled(false));
        assert!(!default_disk_learning_artifacts_enabled(true));
    }

    fn anchor_gate_config(enabled: bool) -> AnchorPromotionGateConfig {
        let mut thresholds = PromotionThresholds::default();
        thresholds.min_games = 8;
        AnchorPromotionGateConfig {
            enabled,
            games: 8,
            minutes: 2.0,
            interval_writes: 1,
            thresholds,
            seed_base: 0xE7A1_0000,
        }
    }

    #[test]
    fn anchor_gate_disabled_is_a_no_op() {
        let cfg = anchor_gate_config(false);
        let mut anchor = None;
        let mut runner = None;
        let mut index = 0usize;
        let status = apply_anchor_promotion_gate(
            &cfg,
            SOCCER_POLICY_STATUS_ACTIVE,
            true,
            None,
            &mut anchor,
            &mut runner,
            &mut index,
            0,
            "test",
        );
        assert_eq!(status, SOCCER_POLICY_STATUS_ACTIVE);
        assert!(anchor.is_none(), "disabled gate must not touch the anchor");
        assert_eq!(index, 0, "disabled gate must not consume the write cadence");
    }

    #[test]
    fn anchor_gate_only_applies_to_active_promotions() {
        let cfg = anchor_gate_config(true);
        let mut anchor = None;
        let mut runner = None;
        let mut index = 0usize;
        // An already-archived candidate is never resurrected or gate-evaluated.
        let status = apply_anchor_promotion_gate(
            &cfg,
            SOCCER_POLICY_STATUS_ARCHIVED,
            true,
            None,
            &mut anchor,
            &mut runner,
            &mut index,
            0,
            "test",
        );
        assert_eq!(status, SOCCER_POLICY_STATUS_ARCHIVED);
        assert!(anchor.is_none());
    }

    #[test]
    fn anchor_gate_bootstraps_first_anchor_without_playing() {
        let cfg = anchor_gate_config(true);
        let candidate = SoccerNeuralNetworkSnapshot::default();
        let mut anchor = None;
        // `runner` stays None: bootstrap must not build a match runner or play games.
        let mut runner = None;
        let mut index = 0usize;
        let status = apply_anchor_promotion_gate(
            &cfg,
            SOCCER_POLICY_STATUS_ACTIVE,
            true,
            Some(&candidate),
            &mut anchor,
            &mut runner,
            &mut index,
            0,
            "test",
        );
        assert_eq!(status, SOCCER_POLICY_STATUS_ACTIVE);
        assert!(anchor.is_some(), "bootstrap sets the first frozen anchor");
        assert!(runner.is_none(), "bootstrap must not play any fixtures");
    }

    #[test]
    fn anchor_gate_verdict_is_none_without_neural_brains() {
        let cfg = anchor_gate_config(true);
        let mut runner = EngineMatchRunner::with_defaults();
        // No neural snapshot on either side ⇒ nothing to compare ⇒ never block.
        assert!(anchor_promotion_gate_verdict(&mut runner, None, None, &cfg, 0).is_none());
    }
}
