//! Queue-style soccer self-play runner.
//!
//! Unlike the batch runner, this keeps a fixed number of simulation slots full:
//! when one game finishes, its deltas are merged and the next game starts from
//! the newest available policy.

use std::collections::{HashMap, VecDeque};
use std::error::Error;
use std::fs;
use std::io::{BufWriter, Error as IoError, ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;
use soccer_engine::des::general::soccer::{
    MatchConfig, MatchSummary, SoccerMarlAlgorithm, SoccerNeuralLearningBackend,
    SoccerNeuralLearningConfig, SoccerNeuralNetworkSnapshot, SoccerPassLearningMetrics,
    SoccerQPolicyOptions, SoccerSelfPlayLearnedParams, SoccerSelfPlayTrainingArtifact,
    SoccerTacticalLearningSummary, SoccerTacticalLearningWeights, SoccerTeamPolicyArtifact,
    SoccerTeamQPolicies, DEFAULT_SOCCER_MAPPO_TEAM_REWARD_SHARE,
};
use soccer_engine::des::soccer_learning::{
    evaluate_soccer_policy_promotion_gate, evolve_soccer_tactical_learning_weights_from_genomes,
    evolve_soccer_team_policies, merge_soccer_policy_deltas, run_soccer_learning_queue_with_events,
    soccer_evolution_options_from_search_metadata, soccer_learning_curriculum_episode_config,
    soccer_learning_curriculum_stage_for_completed_games,
    soccer_neural_network_snapshot_fingerprint,
    soccer_policy_version_insert_status_after_active_head, soccer_postgres_new_sim_refresh_plan,
    soccer_postgres_policy_refresh_decision, soccer_self_play_artifact_from_queue_report,
    soccer_should_flush_postgres_policy_versions_for_new_sim,
    soccer_should_refresh_postgres_for_new_sim, soccer_tactical_learning_weights_fingerprint,
    soccer_team_q_policies_fingerprint, validate_soccer_evolution_options_for_learning_run,
    validate_soccer_learning_curriculum_config_for_learning_run,
    validate_soccer_neural_learning_config_for_learning_run,
    validate_soccer_policy_promotion_gate_config_for_learning_run, SoccerEvolutionOptions,
    SoccerLearningCompletedGame, SoccerLearningCurriculumConfig, SoccerLearningQueueEvent,
    SoccerLearningQueueRunnerConfig, SoccerPolicyPromotionGateConfig,
    SoccerPolicyPromotionGateEvaluation, SoccerPostgresPolicyRefreshCheck,
    SoccerTacticalLearningGenomeParent, SOCCER_POLICY_STATUS_ACTIVE, SOCCER_POLICY_STATUS_ARCHIVED,
};
use soccer_engine::des::soccer_learning_pg::{
    soccer_learning_git_commit, SoccerLearningPgCompletedRunInsert, SoccerLearningPgStore,
};
use uuid::Uuid;

const DEFAULT_SOCCER_QUEUE_POSTGRES_POLICY_VERSION_INTERVAL_GAMES: usize = 10;
const DEFAULT_SOCCER_QUEUE_POSTGRES_COMPLETED_RUN_BATCH_GAMES: usize = 10;
const DEFAULT_SOCCER_QUEUE_POSTGRES_COMPLETED_RUN_RETENTION_GAMES: usize = 0;
const DEFAULT_SOCCER_QUEUE_POSTGRES_ASYNC_BATCH_QUEUE: usize = 16;
const DEFAULT_SOCCER_QUEUE_POSTGRES_ASYNC_COALESCE_BATCHES: usize = 16;
const DEFAULT_SOCCER_QUEUE_POSTGRES_ASYNC_COALESCE_WAIT_MS: usize = 2;
const DEFAULT_SOCCER_QUEUE_POSTGRES_TACTICAL_LEARNING_AUTHORITATIVE: bool = true;
const DEFAULT_SOCCER_QUEUE_POSTGRES_REFRESH_WITH_RESUME_ARTIFACT: bool = true;
const DEFAULT_SOCCER_QUEUE_POSTGRES_FLUSH_POLICY_VERSIONS_BEFORE_NEW_SIM: bool = true;
const DEFAULT_SOCCER_QUEUE_POSTGRES_TRAINING_REPLAY_DELTA_ROWS: usize = 50_000;
const DEFAULT_SOCCER_QUEUE_POSTGRES_TRAINING_REPLAY_PRIOR_WEIGHT: f64 = 1.0;
const DEFAULT_SOCCER_QUEUE_NEURAL_DRAIN_TIMEOUT_MS: usize = 2;
const DEFAULT_SOCCER_QUEUE_EVOLUTION_ENABLED: bool = true;
const DEFAULT_SOCCER_QUEUE_EVOLUTION_ELITE_GAMES: usize = 4;
const SOCCER_QUEUE_LOCAL_MPC_MAX_PLAYERS_PER_TEAM_LIMIT: usize = 11;
const SOCCER_QUEUE_POLICY_SOURCE_MERGE: &str = "merge";
const SOCCER_QUEUE_POLICY_SOURCE_EVOLUTION: &str = "evolution";

#[derive(Clone, Debug)]
struct TacticalEvolutionSample {
    summary: SoccerTacticalLearningSummary,
    weights: SoccerTacticalLearningWeights,
    fitness: f64,
}

#[derive(Clone, Debug)]
struct PolicyEvolutionSample {
    summary: MatchSummary,
    policies: SoccerTeamQPolicies,
    fitness: f64,
}

#[derive(Clone, Debug)]
struct PendingPostgresCompletedRun {
    completed_game: SoccerLearningCompletedGame,
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
}

impl PostgresCompletedRunBatch {
    fn can_absorb(&self, other: &Self) -> bool {
        self.experiment_id == other.experiment_id
            && self.runner_id == other.runner_id
            && self.completed_run_retention_games == other.completed_run_retention_games
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

fn env_usize_alias(primary: &str, alias: &str, default: usize) -> Result<usize, Box<dyn Error>> {
    let Some(value) = env_value(primary).or_else(|| env_value(alias)) else {
        return Ok(default);
    };
    value.parse::<usize>().map_err(|_| {
        invalid_data(format!(
            "{primary}/{alias} must be an unsigned integer, got {value:?}"
        ))
        .into()
    })
}

fn env_u32(name: &str, default: u32) -> Result<u32, Box<dyn Error>> {
    let Some(value) = env_value(name) else {
        return Ok(default);
    };
    value
        .parse::<u32>()
        .map_err(|_| invalid_data(format!("{name} must be a u32, got {value:?}")).into())
}

fn env_u32_alias(primary: &str, alias: &str, default: u32) -> Result<u32, Box<dyn Error>> {
    let Some(value) = env_value(primary).or_else(|| env_value(alias)) else {
        return Ok(default);
    };
    value
        .parse::<u32>()
        .map_err(|_| invalid_data(format!("{primary}/{alias} must be a u32, got {value:?}")).into())
}

fn env_f64(name: &str, default: f64) -> Result<f64, Box<dyn Error>> {
    let Some(value) = env_value(name) else {
        return Ok(default);
    };
    let parsed = value
        .parse::<f64>()
        .map_err(|_| invalid_data(format!("{name} must be a finite number, got {value:?}")))?;
    if parsed.is_finite() {
        Ok(parsed)
    } else {
        Err(invalid_data(format!("{name} must be finite, got {value:?}")).into())
    }
}

fn env_f64_alias(primary: &str, alias: &str, default: f64) -> Result<f64, Box<dyn Error>> {
    let Some(value) = env_value(primary).or_else(|| env_value(alias)) else {
        return Ok(default);
    };
    let parsed = value.parse::<f64>().map_err(|_| {
        invalid_data(format!(
            "{primary}/{alias} must be a finite number, got {value:?}"
        ))
    })?;
    if parsed.is_finite() {
        Ok(parsed)
    } else {
        Err(invalid_data(format!("{primary}/{alias} must be finite, got {value:?}")).into())
    }
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

fn env_bool_alias(primary: &str, alias: &str, default: bool) -> Result<bool, Box<dyn Error>> {
    let Some(value) = env_value(primary).or_else(|| env_value(alias)) else {
        return Ok(default);
    };
    match value.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "y" | "on" => Ok(true),
        "0" | "false" | "no" | "n" | "off" => Ok(false),
        _ => Err(invalid_data(format!(
            "{primary}/{alias} must be a boolean, got {value:?}"
        ))
        .into()),
    }
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
            "SOCCER_NEURAL_LEARNING_BACKEND/SOCCER_NEURAL_BACKEND must be inline or threaded, got {value:?}"
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
        "independent" | "independent-actor-critic" | "independent_actor_critic"
        | "independentactorcritic" => Ok(SoccerMarlAlgorithm::IndependentActorCritic),
        "mappo" | "ppo" => Ok(SoccerMarlAlgorithm::Mappo),
        _ => Err(invalid_data(format!(
            "SOCCER_MARL_ALGORITHM must be off, independentActorCritic, or mappo, got {value:?}"
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
    .min(SOCCER_QUEUE_LOCAL_MPC_MAX_PLAYERS_PER_TEAM_LIMIT);
    if config.local_mpc_max_players_per_team == 0
        || (mpc_override_configured && !tier2_player_enabled)
    {
        config.local_mpc_enabled = false;
    }

    Ok(())
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
        formation_lp_alignment_weight: env_f64(
            "SOCCER_FORMATION_LP_ALIGNMENT_WEIGHT",
            default.formation_lp_alignment_weight,
        )?,
    })
}

fn validate_tactical_learning_weights(
    weights: &SoccerTacticalLearningWeights,
) -> Result<(), Box<dyn Error>> {
    for (name, value) in [
        (
            "SOCCER_ATTACK_SPACING_DELTA_WEIGHT",
            weights.attack_spacing_delta_weight,
        ),
        (
            "SOCCER_ATTACK_SPACING_SCORE_WEIGHT",
            weights.attack_spacing_score_weight,
        ),
        (
            "SOCCER_ATTACK_WIDTH_DELTA_WEIGHT",
            weights.attack_width_delta_weight,
        ),
        (
            "SOCCER_ATTACK_WIDTH_SCORE_WEIGHT",
            weights.attack_width_score_weight,
        ),
        (
            "SOCCER_ATTACK_FLANK_LANE_WEIGHT",
            weights.attack_flank_lane_weight,
        ),
        (
            "SOCCER_DEFENSE_SPACING_DELTA_WEIGHT",
            weights.defense_spacing_delta_weight,
        ),
        (
            "SOCCER_DEFENSE_SPACING_SCORE_WEIGHT",
            weights.defense_spacing_score_weight,
        ),
        (
            "SOCCER_DEFENSE_CONTRACT_DELTA_WEIGHT",
            weights.defense_contract_delta_weight,
        ),
        (
            "SOCCER_DEFENSE_COMPACTNESS_SCORE_WEIGHT",
            weights.defense_compactness_score_weight,
        ),
        (
            "SOCCER_DEFENSE_BALL_DEPTH_SCORE_WEIGHT",
            weights.defense_ball_depth_score_weight,
        ),
        (
            "SOCCER_DEFENSE_ENDLINE_SOFT_PENALTY_WEIGHT",
            weights.defense_endline_soft_penalty_weight,
        ),
        (
            "SOCCER_DEFENSE_ENDLINE_HARD_PENALTY_WEIGHT",
            weights.defense_endline_hard_penalty_weight,
        ),
        (
            "SOCCER_DEFENDER_MIDFIELDER_PRESS_WEIGHT",
            weights.defender_midfielder_press_weight,
        ),
        (
            "SOCCER_MIDFIELDER_PRESS_WEIGHT",
            weights.midfielder_press_weight,
        ),
        (
            "SOCCER_FORMATION_LP_ALIGNMENT_WEIGHT",
            weights.formation_lp_alignment_weight,
        ),
    ] {
        if !value.is_finite() {
            return Err(invalid_data(format!("{name} must be finite")).into());
        }
    }
    Ok(())
}

fn tactical_learning_weight_values(weights: &SoccerTacticalLearningWeights) -> [f64; 15] {
    [
        weights.attack_spacing_delta_weight,
        weights.attack_spacing_score_weight,
        weights.attack_width_delta_weight,
        weights.attack_width_score_weight,
        weights.attack_flank_lane_weight,
        weights.defense_spacing_delta_weight,
        weights.defense_spacing_score_weight,
        weights.defense_contract_delta_weight,
        weights.defense_compactness_score_weight,
        weights.defense_ball_depth_score_weight,
        weights.defense_endline_soft_penalty_weight,
        weights.defense_endline_hard_penalty_weight,
        weights.defender_midfielder_press_weight,
        weights.midfielder_press_weight,
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
    match_config: &mut MatchConfig,
    active_weights: &mut SoccerTacticalLearningWeights,
    postgres_weights: Option<SoccerTacticalLearningWeights>,
) -> Result<bool, String> {
    let Some(postgres_weights) = postgres_weights else {
        return Ok(false);
    };
    validate_tactical_learning_weights(&postgres_weights).map_err(|err| err.to_string())?;
    if tactical_learning_weights_match(active_weights, &postgres_weights)
        && tactical_learning_weights_match(&match_config.tactical_learning, &postgres_weights)
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
    match_config.tactical_learning = postgres_weights.clone();
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

fn clear_queue_evolution_samples_after_postgres_refresh(
    event_label: &str,
    next_episode: usize,
    policy_version_id: &str,
    tactical_evolution_samples: &mut VecDeque<TacticalEvolutionSample>,
    policy_evolution_samples: &mut VecDeque<PolicyEvolutionSample>,
    clear_tactical_samples: bool,
    clear_policy_samples: bool,
) -> (usize, usize) {
    let tactical_samples_cleared = if clear_tactical_samples {
        let samples = tactical_evolution_samples.len();
        tactical_evolution_samples.clear();
        samples
    } else {
        0
    };
    let policy_samples_cleared = if clear_policy_samples {
        let samples = policy_evolution_samples.len();
        policy_evolution_samples.clear();
        samples
    } else {
        0
    };
    if tactical_samples_cleared > 0 || policy_samples_cleared > 0 {
        println!(
            "{event_label} next_episode={next_episode} policy_version={policy_version_id} tactical_samples_cleared={tactical_samples_cleared} policy_samples_cleared={policy_samples_cleared}"
        );
    }
    (tactical_samples_cleared, policy_samples_cleared)
}

fn invalid_data(message: impl Into<String>) -> IoError {
    IoError::new(ErrorKind::InvalidData, message.into())
}

fn default_run_id() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    format!("soccer-learning-queue-{seconds}")
}

fn default_postgres_policy_version_interval_games(parallel_games: usize) -> usize {
    parallel_games.max(DEFAULT_SOCCER_QUEUE_POSTGRES_POLICY_VERSION_INTERVAL_GAMES)
}

fn default_postgres_completed_run_batch_games(parallel_games: usize) -> usize {
    parallel_games.max(DEFAULT_SOCCER_QUEUE_POSTGRES_COMPLETED_RUN_BATCH_GAMES)
}

fn default_queue_evolution_interval_games(parallel_games: usize) -> usize {
    parallel_games.max(10)
}

fn default_queue_evolution_window_games(
    evolution_interval_games: usize,
    evolution_elite_games: usize,
) -> usize {
    evolution_interval_games.max(evolution_elite_games).max(1)
}

fn queue_policy_version_status_for_promotion_gate(
    evaluation: &SoccerPolicyPromotionGateEvaluation,
) -> &'static str {
    if evaluation.eligible {
        SOCCER_POLICY_STATUS_ACTIVE
    } else {
        SOCCER_POLICY_STATUS_ARCHIVED
    }
}

fn queue_policy_version_source_kind(policy_evolved: bool, tactical_evolved: bool) -> &'static str {
    if policy_evolved || tactical_evolved {
        SOCCER_QUEUE_POLICY_SOURCE_EVOLUTION
    } else {
        SOCCER_QUEUE_POLICY_SOURCE_MERGE
    }
}

fn queue_tactical_evolution_search_metadata(
    completed_games: usize,
    elite_games: usize,
    window_games: usize,
    sample_count: usize,
    best_fitness: f64,
    options: SoccerEvolutionOptions,
    previous_tactical_learning: &SoccerTacticalLearningWeights,
    evolved_tactical_learning: &SoccerTacticalLearningWeights,
) -> serde_json::Value {
    serde_json::json!({
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
    })
}

fn queue_policy_promotion_search_metadata(
    curriculum_stage: &'static str,
    evaluation: &SoccerPolicyPromotionGateEvaluation,
) -> serde_json::Value {
    serde_json::json!({
        "curriculumStage": curriculum_stage,
        "gate": evaluation,
        "status": queue_policy_version_status_for_promotion_gate(evaluation)
    })
}

fn take_episode_starting_policy_version(
    episode_starting_policy_versions: &mut HashMap<usize, (Option<String>, i32)>,
    episode: usize,
    current_policy_version_id: &Option<String>,
    current_generation: i32,
) -> (Option<String>, i32) {
    episode_starting_policy_versions
        .remove(&episode)
        .unwrap_or_else(|| (current_policy_version_id.clone(), current_generation))
}

fn flush_postgres_completed_runs(
    store: &mut SoccerLearningPgStore,
    experiment_id: &str,
    runner_id: &str,
    pending_policy_versions: &mut Vec<PendingPostgresPolicyVersion>,
    pending_runs: &mut Vec<PendingPostgresCompletedRun>,
    completed_run_retention_games: usize,
) -> Result<usize, String> {
    if pending_policy_versions.is_empty() && pending_runs.is_empty() {
        return Ok(0);
    }
    let policy_versions_written = pending_policy_versions.len();
    for policy_version in pending_policy_versions.iter() {
        let latest_active_metadata = if policy_version.status == SOCCER_POLICY_STATUS_ACTIVE {
            store.load_latest_active_policy_metadata(experiment_id)?
        } else {
            None
        };
        let insert_status = soccer_policy_version_insert_status_after_active_head(
            policy_version.status,
            policy_version.parent_policy_version_id.as_deref(),
            policy_version.generation,
            latest_active_metadata
                .as_ref()
                .map(|metadata| metadata.id.as_str()),
            latest_active_metadata
                .as_ref()
                .map(|metadata| metadata.generation),
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
        store.insert_policy_version_with_id_and_neural_network_and_search_metadata(
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
        )?;
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
            game: &pending.completed_game,
        })
        .collect::<Vec<_>>();
    let batch_size = inserts.len();
    let run_ids = store.insert_completed_runs(experiment_id, runner_id, &inserts)?;
    drop(inserts);
    // Roll this batch's pass-quality metrics into the per-commit learning-progress series.
    // Non-fatal: a metrics write must never fail the run persistence.
    {
        let mut pass_metrics = SoccerPassLearningMetrics::default();
        for pending in pending_runs.iter() {
            pass_metrics.accumulate(&pending.completed_game.summary.stats.pass_learning_metrics());
        }
        if let Err(err) = store.upsert_pass_learning_metrics(
            &soccer_learning_git_commit(),
            batch_size as u64,
            &pass_metrics,
        ) {
            eprintln!("soccer-learning: pass-metrics upsert failed (non-fatal): {err}");
        }
    }
    if completed_run_retention_games > 0 {
        let prune = store
            .prune_completed_runs_for_experiment(experiment_id, completed_run_retention_games)?;
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
        run_ids.first(),
        run_ids.last(),
    ) {
        println!(
            "postgres_persisted_batch episodes={}..{} first_run_id={} last_run_id={} first_policy_version={} last_policy_version={} first_generation={} last_generation={} policy_versions_written={} batch_size={}",
            first.completed_game.episode + 1,
            last.completed_game.episode + 1,
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
            batch_size,
        );
    }
    let flushed = pending_runs.len();
    pending_policy_versions.clear();
    pending_runs.clear();
    Ok(flushed)
}

fn flush_postgres_policy_versions_for_new_sims(
    store: &mut SoccerLearningPgStore,
    experiment_id: &str,
    runner_id: &str,
    pending_policy_versions: &mut Vec<PendingPostgresPolicyVersion>,
) -> Result<(), String> {
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
    )?;
    println!(
        "postgres_policy_versions_flushed_for_new_sims policy_versions_written={pending_count}"
    );
    Ok(())
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
                                Err(mpsc::RecvTimeoutError::Timeout) => break,
                                Err(mpsc::RecvTimeoutError::Disconnected) => break,
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
                );
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
        // Best-effort: fsync the directory so the rename is durable across a
        // crash. Some filesystems reject directory fsync, so don't fail the
        // write — but surface the error instead of silently dropping it.
        if let Err(err) = fs::File::open(parent).and_then(|dir| dir.sync_all()) {
            eprintln!(
                "soccer_learning_queue: directory fsync after writing {} failed (artifact may not be crash-durable): {err}",
                path.display()
            );
        }
    }
    Ok(())
}

fn validate_queue_settings(
    games: usize,
    halves: usize,
    half_minutes: f64,
    minutes: f64,
    period_break_recovery_seconds: f64,
    dt_seconds: f64,
    learning_interval_ticks: usize,
    parallel_games: usize,
    options: &SoccerQPolicyOptions,
) -> Result<(), Box<dyn Error>> {
    if games == 0 {
        return Err(invalid_data("SOCCER_QUEUE_GAMES/SOCCER_GAMES must be at least 1").into());
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
        return Err(invalid_data(
            "SOCCER_QUEUE_PARALLEL_GAMES/SOCCER_PARALLEL_GAMES must be between 1 and 100",
        )
        .into());
    }
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

fn load_initial_policies(
    path: Option<&str>,
    options: SoccerQPolicyOptions,
) -> Result<SoccerTeamQPolicies, Box<dyn Error>> {
    let Some(path) = path else {
        return Ok(SoccerTeamQPolicies::new(options));
    };
    let raw = fs::read_to_string(path)?;
    if let Ok(artifact) = serde_json::from_str::<SoccerSelfPlayTrainingArtifact>(&raw) {
        return SoccerTeamQPolicies::from_self_play_artifact(&artifact)
            .map_err(|err| invalid_data(err).into());
    }
    if let Ok(artifact) = serde_json::from_str::<SoccerTeamPolicyArtifact>(&raw) {
        return SoccerTeamQPolicies::from_artifact(&artifact)
            .map_err(|err| invalid_data(err).into());
    }
    Err(invalid_data(format!(
        "resume artifact {path} is neither a self-play nor team-policy artifact"
    ))
    .into())
}

fn run() -> Result<(), Box<dyn Error>> {
    let default_games = env_usize("SOCCER_GAMES", 100)?;
    let games = env_usize("SOCCER_QUEUE_GAMES", default_games)?;
    let default_parallel_games = env_usize("SOCCER_PARALLEL_GAMES", 10)?;
    let parallel_games = env_usize("SOCCER_QUEUE_PARALLEL_GAMES", default_parallel_games)?;
    let halves = env_usize("SOCCER_HALVES", 2)?;
    let half_minutes = env_f64("SOCCER_HALF_MINUTES", 45.0)?;
    let minutes = env_f64("SOCCER_MINUTES", half_minutes * halves as f64)?;
    let dt_seconds = env_f64("SOCCER_DT_SECONDS", 0.2)?;
    let learning_interval_ticks = env_usize("SOCCER_LEARNING_INTERVAL_TICKS", 4)?;
    let seed = env_u32("SOCCER_SEED", 2026)?;
    let neural_learning = env_neural_learning_config()?;
    validate_soccer_neural_learning_config_for_learning_run(&neural_learning)
        .map_err(invalid_data)?;
    let neural_drain_timeout_ms = env_usize_alias(
        "SOCCER_QUEUE_NEURAL_DRAIN_TIMEOUT_MS",
        "SOCCER_NEURAL_DRAIN_TIMEOUT_MS",
        DEFAULT_SOCCER_QUEUE_NEURAL_DRAIN_TIMEOUT_MS,
    )?;
    let neural_drain_timeout =
        Duration::from_millis(neural_drain_timeout_ms.min(u64::MAX as usize) as u64);
    let pg_policy_version_interval_games = env_usize_alias(
        "SOCCER_QUEUE_POSTGRES_POLICY_VERSION_INTERVAL_GAMES",
        "SOCCER_POSTGRES_POLICY_VERSION_INTERVAL_GAMES",
        default_postgres_policy_version_interval_games(parallel_games),
    )?;
    let pg_completed_run_batch_games = env_usize_alias(
        "SOCCER_QUEUE_POSTGRES_COMPLETED_RUN_BATCH_GAMES",
        "SOCCER_POSTGRES_COMPLETED_RUN_BATCH_GAMES",
        default_postgres_completed_run_batch_games(parallel_games),
    )?;
    let pg_completed_run_retention_games = env_usize_alias(
        "SOCCER_QUEUE_POSTGRES_COMPLETED_RUN_RETENTION_GAMES",
        "SOCCER_POSTGRES_COMPLETED_RUN_RETENTION_GAMES",
        DEFAULT_SOCCER_QUEUE_POSTGRES_COMPLETED_RUN_RETENTION_GAMES,
    )?;
    let pg_completed_run_async_queue_batches = env_usize_alias(
        "SOCCER_QUEUE_POSTGRES_ASYNC_BATCH_QUEUE",
        "SOCCER_POSTGRES_ASYNC_BATCH_QUEUE",
        DEFAULT_SOCCER_QUEUE_POSTGRES_ASYNC_BATCH_QUEUE,
    )?;
    let pg_completed_run_async_coalesce_batches = env_usize_alias(
        "SOCCER_QUEUE_POSTGRES_ASYNC_COALESCE_BATCHES",
        "SOCCER_POSTGRES_ASYNC_COALESCE_BATCHES",
        DEFAULT_SOCCER_QUEUE_POSTGRES_ASYNC_COALESCE_BATCHES,
    )?;
    let pg_completed_run_async_coalesce_wait_ms = env_usize_alias(
        "SOCCER_QUEUE_POSTGRES_ASYNC_COALESCE_WAIT_MS",
        "SOCCER_POSTGRES_ASYNC_COALESCE_WAIT_MS",
        DEFAULT_SOCCER_QUEUE_POSTGRES_ASYNC_COALESCE_WAIT_MS,
    )?;
    let pg_tactical_learning_authoritative = env_bool_alias(
        "SOCCER_QUEUE_POSTGRES_TACTICAL_LEARNING_AUTHORITATIVE",
        "SOCCER_POSTGRES_TACTICAL_LEARNING_AUTHORITATIVE",
        DEFAULT_SOCCER_QUEUE_POSTGRES_TACTICAL_LEARNING_AUTHORITATIVE,
    )?;
    let pg_refresh_with_resume_artifact = env_bool_alias(
        "SOCCER_QUEUE_POSTGRES_REFRESH_WITH_RESUME_ARTIFACT",
        "SOCCER_POSTGRES_REFRESH_WITH_RESUME_ARTIFACT",
        DEFAULT_SOCCER_QUEUE_POSTGRES_REFRESH_WITH_RESUME_ARTIFACT,
    )?;
    let pg_flush_policy_versions_before_new_sim = env_bool_alias(
        "SOCCER_QUEUE_POSTGRES_FLUSH_POLICY_VERSIONS_BEFORE_NEW_SIM",
        "SOCCER_POSTGRES_FLUSH_POLICY_VERSIONS_BEFORE_NEW_SIM",
        DEFAULT_SOCCER_QUEUE_POSTGRES_FLUSH_POLICY_VERSIONS_BEFORE_NEW_SIM,
    )?;
    let pg_training_replay_delta_rows = env_usize_alias(
        "SOCCER_QUEUE_POSTGRES_TRAINING_REPLAY_DELTA_ROWS",
        "SOCCER_POSTGRES_TRAINING_REPLAY_DELTA_ROWS",
        DEFAULT_SOCCER_QUEUE_POSTGRES_TRAINING_REPLAY_DELTA_ROWS,
    )?;
    let pg_training_replay_prior_weight = env_f64_alias(
        "SOCCER_QUEUE_POSTGRES_TRAINING_REPLAY_PRIOR_WEIGHT",
        "SOCCER_POSTGRES_TRAINING_REPLAY_PRIOR_WEIGHT",
        DEFAULT_SOCCER_QUEUE_POSTGRES_TRAINING_REPLAY_PRIOR_WEIGHT,
    )?;
    if pg_training_replay_prior_weight < 0.0 {
        return Err(invalid_data(
            "SOCCER_QUEUE_POSTGRES_TRAINING_REPLAY_PRIOR_WEIGHT must be non-negative",
        )
        .into());
    }
    let pg_policy_version_interval_games = pg_policy_version_interval_games.max(1);
    let pg_completed_run_batch_games = pg_completed_run_batch_games.max(1);
    let pg_completed_run_async_queue_batches = pg_completed_run_async_queue_batches.max(1);
    let pg_completed_run_async_coalesce_batches = pg_completed_run_async_coalesce_batches.max(1);
    let pg_completed_run_async_coalesce_wait = Duration::from_millis(
        pg_completed_run_async_coalesce_wait_ms.min(u64::MAX as usize) as u64,
    );
    let evolution_enabled = env_bool_alias(
        "SOCCER_QUEUE_EVOLUTION_ENABLED",
        "SOCCER_EVOLUTION_ENABLED",
        DEFAULT_SOCCER_QUEUE_EVOLUTION_ENABLED,
    )?;
    let evolution_interval_games = env_usize_alias(
        "SOCCER_QUEUE_EVOLUTION_INTERVAL_GAMES",
        "SOCCER_EVOLUTION_INTERVAL_GAMES",
        default_queue_evolution_interval_games(parallel_games),
    )?
    .max(1);
    let evolution_elite_games = env_usize_alias(
        "SOCCER_QUEUE_EVOLUTION_ELITE_GAMES",
        "SOCCER_EVOLUTION_ELITE_GAMES",
        DEFAULT_SOCCER_QUEUE_EVOLUTION_ELITE_GAMES,
    )?
    .max(1);
    let evolution_window_games = env_usize_alias(
        "SOCCER_QUEUE_EVOLUTION_WINDOW_GAMES",
        "SOCCER_EVOLUTION_WINDOW_GAMES",
        default_queue_evolution_window_games(evolution_interval_games, evolution_elite_games),
    )?
    .max(evolution_elite_games)
    .max(1);
    let default_evolution_options = SoccerEvolutionOptions::default();
    let mut evolution_options = SoccerEvolutionOptions {
        mutation_rate: env_f64_alias(
            "SOCCER_QUEUE_EVOLUTION_MUTATION_RATE",
            "SOCCER_EVOLUTION_MUTATION_RATE",
            default_evolution_options.mutation_rate,
        )?,
        mutation_scale: env_f64_alias(
            "SOCCER_QUEUE_EVOLUTION_MUTATION_SCALE",
            "SOCCER_EVOLUTION_MUTATION_SCALE",
            default_evolution_options.mutation_scale,
        )?,
        crossover_rate: env_f64_alias(
            "SOCCER_QUEUE_EVOLUTION_CROSSOVER_RATE",
            "SOCCER_EVOLUTION_CROSSOVER_RATE",
            default_evolution_options.crossover_rate,
        )?,
        exploration_rate: env_f64_alias(
            "SOCCER_QUEUE_EVOLUTION_EXPLORATION_RATE",
            "SOCCER_EVOLUTION_EXPLORATION_RATE",
            default_evolution_options.exploration_rate,
        )?,
        exploration_scale: env_f64_alias(
            "SOCCER_QUEUE_EVOLUTION_EXPLORATION_SCALE",
            "SOCCER_EVOLUTION_EXPLORATION_SCALE",
            default_evolution_options.exploration_scale,
        )?,
        elite_weight_floor: env_f64_alias(
            "SOCCER_QUEUE_EVOLUTION_ELITE_WEIGHT_FLOOR",
            "SOCCER_EVOLUTION_ELITE_WEIGHT_FLOOR",
            default_evolution_options.elite_weight_floor,
        )?,
        population_size: env_usize_alias(
            "SOCCER_QUEUE_EVOLUTION_POPULATION_SIZE",
            "SOCCER_EVOLUTION_POPULATION_SIZE",
            default_evolution_options.population_size,
        )?
        .max(1),
        seed: env_u32_alias(
            "SOCCER_QUEUE_EVOLUTION_SEED",
            "SOCCER_EVOLUTION_SEED",
            default_evolution_options.seed as u32,
        )? as u64,
    };
    validate_soccer_evolution_options_for_learning_run(&evolution_options).map_err(invalid_data)?;
    let default_curriculum = SoccerLearningCurriculumConfig::default();
    let curriculum_config = SoccerLearningCurriculumConfig {
        ball_skills_after_games: env_usize_alias(
            "SOCCER_QUEUE_CURRICULUM_BALL_SKILLS_AFTER_GAMES",
            "SOCCER_CURRICULUM_BALL_SKILLS_AFTER_GAMES",
            default_curriculum.ball_skills_after_games,
        )?,
        duels_after_games: env_usize_alias(
            "SOCCER_QUEUE_CURRICULUM_DUELS_AFTER_GAMES",
            "SOCCER_CURRICULUM_DUELS_AFTER_GAMES",
            default_curriculum.duels_after_games,
        )?,
        small_sided_after_games: env_usize_alias(
            "SOCCER_QUEUE_CURRICULUM_SMALL_SIDED_AFTER_GAMES",
            "SOCCER_CURRICULUM_SMALL_SIDED_AFTER_GAMES",
            default_curriculum.small_sided_after_games,
        )?,
        team_shape_after_games: env_usize_alias(
            "SOCCER_QUEUE_CURRICULUM_TEAM_SHAPE_AFTER_GAMES",
            "SOCCER_CURRICULUM_TEAM_SHAPE_AFTER_GAMES",
            default_curriculum.team_shape_after_games,
        )?,
        full_match_after_games: env_usize_alias(
            "SOCCER_QUEUE_CURRICULUM_FULL_MATCH_AFTER_GAMES",
            "SOCCER_CURRICULUM_FULL_MATCH_AFTER_GAMES",
            default_curriculum.full_match_after_games,
        )?,
    };
    validate_soccer_learning_curriculum_config_for_learning_run(&curriculum_config)
        .map_err(invalid_data)?;
    let default_promotion_gate = SoccerPolicyPromotionGateConfig::default();
    let policy_promotion_gate = SoccerPolicyPromotionGateConfig {
        enabled: env_bool_alias(
            "SOCCER_QUEUE_POLICY_PROMOTION_GATE_ENABLED",
            "SOCCER_POLICY_PROMOTION_GATE_ENABLED",
            default_promotion_gate.enabled,
        )?,
        min_sample_games: env_usize_alias(
            "SOCCER_QUEUE_POLICY_PROMOTION_MIN_SAMPLE_GAMES",
            "SOCCER_POLICY_PROMOTION_MIN_SAMPLE_GAMES",
            default_promotion_gate.min_sample_games,
        )?
        .max(1),
        min_mean_match_fitness: env_f64_alias(
            "SOCCER_QUEUE_POLICY_PROMOTION_MIN_MEAN_MATCH_FITNESS",
            "SOCCER_POLICY_PROMOTION_MIN_MEAN_MATCH_FITNESS",
            default_promotion_gate.min_mean_match_fitness,
        )?,
        min_best_match_fitness: env_f64_alias(
            "SOCCER_QUEUE_POLICY_PROMOTION_MIN_BEST_MATCH_FITNESS",
            "SOCCER_POLICY_PROMOTION_MIN_BEST_MATCH_FITNESS",
            default_promotion_gate.min_best_match_fitness,
        )?,
        min_mean_play_quality: env_f64_alias(
            "SOCCER_QUEUE_POLICY_PROMOTION_MIN_MEAN_PLAY_QUALITY",
            "SOCCER_POLICY_PROMOTION_MIN_MEAN_PLAY_QUALITY",
            default_promotion_gate.min_mean_play_quality,
        )?,
        max_mean_conceded_goals: env_f64_alias(
            "SOCCER_QUEUE_POLICY_PROMOTION_MAX_MEAN_CONCEDED_GOALS",
            "SOCCER_POLICY_PROMOTION_MAX_MEAN_CONCEDED_GOALS",
            default_promotion_gate.max_mean_conceded_goals,
        )?,
        max_mean_goal_margin: env_f64_alias(
            "SOCCER_QUEUE_POLICY_PROMOTION_MAX_MEAN_GOAL_MARGIN",
            "SOCCER_POLICY_PROMOTION_MAX_MEAN_GOAL_MARGIN",
            default_promotion_gate.max_mean_goal_margin,
        )?,
        max_mean_chain_net_loss: env_f64_alias(
            "SOCCER_QUEUE_POLICY_PROMOTION_MAX_MEAN_CHAIN_NET_LOSS",
            "SOCCER_POLICY_PROMOTION_MAX_MEAN_CHAIN_NET_LOSS",
            default_promotion_gate.max_mean_chain_net_loss,
        )?,
    };
    validate_soccer_policy_promotion_gate_config_for_learning_run(&policy_promotion_gate)
        .map_err(invalid_data)?;
    let options = SoccerQPolicyOptions {
        alpha: env_f64("SOCCER_ALPHA", 0.20)?,
        gamma: env_f64("SOCCER_GAMMA", 0.96)?,
        exploration_epsilon: env_f64("SOCCER_EXPLORATION_EPSILON", 0.0)?,
    };
    let period_break_recovery_seconds = env_f64(
        "SOCCER_PERIOD_BREAK_RECOVERY_SECONDS",
        env_f64("SOCCER_HALFTIME_RECOVERY_SECONDS", 900.0)?,
    )?;
    validate_queue_settings(
        games,
        halves,
        half_minutes,
        minutes,
        period_break_recovery_seconds,
        dt_seconds,
        learning_interval_ticks,
        parallel_games,
        &options,
    )?;
    let mut tactical_learning = env_tactical_learning_weights()?;
    validate_tactical_learning_weights(&tactical_learning)?;
    let default_config = MatchConfig::default();
    let formation_lp_enabled = env_bool(
        "SOCCER_FORMATION_LP_ENABLED",
        default_config.formation_lp_enabled,
    )?;
    let opponent_belief_enabled = env_bool_alias(
        "SOCCER_OPPONENT_BELIEF_ENABLED",
        "SOCCER_OPPONENT_BELIEF",
        true,
    )?;
    let half_duration_seconds = half_minutes * 60.0;
    let halftime_fatigue_recovery = if half_duration_seconds > 0.0 {
        (period_break_recovery_seconds / half_duration_seconds).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let mut config = MatchConfig {
        dt_seconds,
        duration_seconds: minutes * 60.0,
        halves: halves as u8,
        half_duration_seconds,
        halftime_fatigue_recovery,
        period_count: halves,
        period_break_recovery_seconds,
        learning_enabled: true,
        learning_logging_enabled: env_bool("SOCCER_LEARNING_LOGGING", false)?,
        learning_interval_ticks,
        formation_lp_enabled,
        tactical_learning: tactical_learning.clone(),
        neural_learning: neural_learning.clone(),
        opponent_belief_enabled,
        max_human_players: 0,
        seed,
        ..default_config.clone()
    };
    apply_env_mpc_config(&mut config, &default_config)?;
    let resume_artifact =
        env_value("SOCCER_RESUME_ARTIFACT").or_else(|| env_value("SOCCER_RESUME_ARTIFACT_PATH"));
    let run_id = env_value("SOCCER_RUN_ID").unwrap_or_else(default_run_id);
    let run_dir = PathBuf::from(
        env_value("SOCCER_RUN_DIR")
            .unwrap_or_else(|| format!("out/soccer-learning-queue-runs/{run_id}")),
    );
    let artifact_path = env_value("SOCCER_ARTIFACT_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| run_dir.join("final-policy.json"));
    let learned_params_path = env_value("SOCCER_LEARNED_PARAMS_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| run_dir.join("learned-params.json"));
    let mut pg_store = SoccerLearningPgStore::connect_from_env().map_err(invalid_data)?;
    let retrieval_capture_enabled = env_bool_alias(
        "SOCCER_RETRIEVAL_CAPTURE_ENABLED",
        "SOCCER_RETRIEVAL_CAPTURE",
        pg_store.is_some(),
    )?;
    config.retrieval.capture_enabled = retrieval_capture_enabled;
    let postgres_required = env_bool("SOCCER_REQUIRE_POSTGRES", false)?;
    if postgres_required && pg_store.is_none() {
        return Err(
            invalid_data("SOCCER_REQUIRE_POSTGRES=true requires SOCCER_DATABASE_URL").into(),
        );
    }
    let mut pg_experiment_id = None::<String>;
    let mut pg_base_policy_version_id = None::<String>;
    let mut pg_last_policy_version_id = None::<String>;
    let mut pg_generation = 0i32;
    let mut pg_base_policy_version_updated_at_micros = 0i64;
    let mut pg_base_policy_fingerprint = None::<u64>;
    let mut pg_base_neural_network_fingerprint = None::<u64>;
    let mut pg_base_tactical_learning_fingerprint = None::<u64>;
    let mut pg_completed_games_seen = 0usize;
    let mut pg_policy_version_buffer = Vec::<PendingPostgresPolicyVersion>::new();
    let mut pg_completed_buffer = Vec::<PendingPostgresCompletedRun>::new();
    let mut pg_episode_starting_policy_versions = HashMap::<usize, (Option<String>, i32)>::new();
    let mut pg_persisted_games = 0usize;

    let mut initial_policies = load_initial_policies(resume_artifact.as_deref(), options.clone())?;
    let pg_refresh_for_new_sims = soccer_should_refresh_postgres_for_new_sim(
        resume_artifact.is_some(),
        pg_refresh_with_resume_artifact,
    );
    let mut initial_neural_network = None::<SoccerNeuralNetworkSnapshot>;
    if let Some(store) = pg_store.as_mut() {
        let experiment_slug =
            env_value("SOCCER_EXPERIMENT_SLUG").unwrap_or_else(|| "soccer-self-play".to_string());
        let experiment_name =
            env_value("SOCCER_EXPERIMENT_NAME").unwrap_or_else(|| "Soccer self-play".to_string());
        let experiment_id = store
            .ensure_experiment(&experiment_slug, &experiment_name, &config)
            .map_err(invalid_data)?;
        if pg_refresh_for_new_sims {
            if let Some(version) = store
                .load_latest_active_policy(&experiment_id, options.clone(), options.clone())
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
                let version_search_metadata = version.search_metadata.clone();
                println!(
                    "postgres_resume_policy experiment={} policy_version={} generation={} neural_network={}",
                    experiment_slug,
                    version.id,
                    version.generation,
                    version.neural_network.is_some()
                );
                initial_policies = version.policies;
                initial_neural_network = version.neural_network;
                maybe_apply_postgres_tactical_learning(
                    "postgres_resume_tactical_learning",
                    1,
                    &version.id,
                    version.generation,
                    &mut config,
                    &mut tactical_learning,
                    version.tactical_learning,
                )
                .map_err(invalid_data)?;
                maybe_apply_postgres_evolution_options(
                    "postgres_resume_evolution_options",
                    1,
                    &version.id,
                    version.generation,
                    &mut evolution_options,
                    version_search_metadata.as_ref(),
                );
                pg_base_policy_version_id = Some(version.id.clone());
                pg_last_policy_version_id = Some(version.id);
                pg_generation = version.generation;
                pg_base_policy_version_updated_at_micros = version.updated_at_micros;
                pg_base_policy_fingerprint = version_policy_fingerprint;
                pg_base_neural_network_fingerprint = version_neural_network_fingerprint;
                pg_base_tactical_learning_fingerprint = version_tactical_learning_fingerprint;
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
                initial_policies = merge_soccer_policy_deltas(
                    &initial_policies,
                    std::slice::from_ref(&replay_delta),
                    pg_training_replay_prior_weight,
                )
                .map_err(invalid_data)?;
                pg_base_policy_fingerprint =
                    Some(soccer_team_q_policies_fingerprint(&initial_policies));
            }
            println!(
                "postgres_training_replay_queue experiment={} entries={} max_delta_rows={} prior_weight={:.3} created_after_micros={}",
                experiment_slug,
                replay_entries,
                pg_training_replay_delta_rows,
                pg_training_replay_prior_weight,
                replay_created_after_micros
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "none".to_string())
            );
        } else {
            println!("postgres_training_replay_queue disabled max_delta_rows=0");
        }
        pg_experiment_id = Some(experiment_id);
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

    println!(
        "soccer_learning_queue_start run_id={} games={} parallel_games={} minutes={:.1} dt={:.3}s ticks_per_game={} seed={} neural_enabled={} neural_backend={:?} neural_snapshot_every_batches={} neural_drain_timeout_ms={} postgres_required={} pg_policy_version_interval_games={} pg_completed_run_batch_games={} pg_completed_run_retention_games={} pg_completed_async={} pg_completed_async_queue_batches={} pg_completed_async_coalesce_batches={} pg_completed_async_coalesce_wait_ms={} pg_tactical_learning_authoritative={} pg_refresh_with_resume_artifact={} pg_flush_policy_versions_before_new_sim={} pg_training_replay_delta_rows={} pg_training_replay_prior_weight={:.3}",
        run_id,
        games,
        parallel_games,
        minutes,
        dt_seconds,
        config.total_ticks(),
        seed,
        neural_learning.enabled,
        neural_learning.backend,
        neural_learning.snapshot_every_batches,
        neural_drain_timeout_ms,
        postgres_required,
        pg_policy_version_interval_games,
        pg_completed_run_batch_games,
        pg_completed_run_retention_games,
        pg_completed_writer.is_some(),
        pg_completed_run_async_queue_batches,
        pg_completed_run_async_coalesce_batches,
        pg_completed_run_async_coalesce_wait_ms,
        pg_tactical_learning_authoritative,
        pg_refresh_with_resume_artifact,
        pg_flush_policy_versions_before_new_sim,
        pg_training_replay_delta_rows,
        pg_training_replay_prior_weight,
    );
    println!(
        "queue_evolution enabled={} interval_games={} window_games={} elite_games={} mutation_rate={:.4} mutation_scale={:.4} crossover_rate={:.4} exploration_rate={:.4} exploration_scale={:.4} elite_weight_floor={:.4} population_size={} seed={}",
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
        "queue_curriculum ball_skills_after_games={} duels_after_games={} small_sided_after_games={} team_shape_after_games={} full_match_after_games={}",
        curriculum_config.ball_skills_after_games,
        curriculum_config.duels_after_games,
        curriculum_config.small_sided_after_games,
        curriculum_config.team_shape_after_games,
        curriculum_config.full_match_after_games
    );
    println!(
        "queue_policy_promotion_gate enabled={} min_sample_games={} min_mean_match_fitness={:.4} min_best_match_fitness={:.4} min_mean_play_quality={:.4} max_mean_conceded_goals={:.4} max_mean_goal_margin={:.4} max_mean_chain_net_loss={:.4}",
        policy_promotion_gate.enabled,
        policy_promotion_gate.min_sample_games,
        policy_promotion_gate.min_mean_match_fitness,
        policy_promotion_gate.min_best_match_fitness,
        policy_promotion_gate.min_mean_play_quality,
        policy_promotion_gate.max_mean_conceded_goals,
        policy_promotion_gate.max_mean_goal_margin,
        policy_promotion_gate.max_mean_chain_net_loss
    );
    if let Some(path) = &resume_artifact {
        println!("resume_artifact={path}");
    }

    let mut active_config = config.clone();
    let mut last_queue_curriculum_stage_label = None::<String>;
    let mut queue_completed_games_seen = 0usize;
    let mut local_tactical_evolved_since_pg_refresh = false;
    let tactical_evolution_window_games = evolution_window_games;
    let mut tactical_evolution_samples =
        VecDeque::<TacticalEvolutionSample>::with_capacity(tactical_evolution_window_games);
    let mut policy_evolution_samples =
        VecDeque::<PolicyEvolutionSample>::with_capacity(tactical_evolution_window_games);
    let report = run_soccer_learning_queue_with_events(
        SoccerLearningQueueRunnerConfig {
            games,
            parallel_games,
            base_seed: seed,
            match_config: config.clone(),
            initial_neural_network: initial_neural_network.clone(),
            neural_drain_timeout,
            options: options.clone(),
            prune_action_entries_per_team: env_usize("SOCCER_MAX_POLICY_ENTRIES_PER_TEAM", 0)?,
            prune_target_entries_per_team: env_usize(
                "SOCCER_MAX_POLICY_TARGET_ENTRIES_PER_TEAM",
                env_usize("SOCCER_MAX_POLICY_ENTRIES_PER_TEAM", 0)?,
            )?,
            min_policy_visits: env_u32("SOCCER_MIN_POLICY_VISITS", 0)?,
        },
        initial_policies,
        |event| {
            match event {
                SoccerLearningQueueEvent::StartingBatch {
                    next_episode,
                    match_config,
                    policies: starting_policies,
                    neural_network,
                } => {
                    *match_config = active_config.clone();
                    if pg_refresh_for_new_sims {
                        let mut pending_async_pg_batches = 0usize;
                        let mut pending_async_policy_version_batches = 0usize;
                        if let Some(writer) = pg_completed_writer.as_mut() {
                            pg_persisted_games += writer.drain_finished()?;
                            pending_async_pg_batches = writer.pending_batches;
                            pending_async_policy_version_batches =
                                writer.pending_policy_version_batches;
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
                                    &run_id,
                                    &mut pg_policy_version_buffer,
                                )?;
                            }
                            if new_sim_refresh_plan.wait_for_async_policy_versions {
                                if let Some(writer) = pg_completed_writer.as_mut() {
                                    pg_persisted_games += writer.drain_policy_versions_finished()?;
                                    pending_async_pg_batches = writer.pending_batches;
                                    pending_async_policy_version_batches =
                                        writer.pending_policy_version_batches;
                                }
                            }
                            if let Some(metadata) =
                                store.load_latest_active_policy_metadata(experiment_id)?
                            {
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
                                    soccer_postgres_policy_refresh_decision(
                                        SoccerPostgresPolicyRefreshCheck {
                                            current_policy_version_id: pg_base_policy_version_id
                                                .as_deref(),
                                            current_generation: pg_generation,
                                            current_updated_at_micros:
                                                pg_base_policy_version_updated_at_micros,
                                            current_policy_fingerprint: pg_base_policy_fingerprint,
                                            latest_policy_fingerprint,
                                            current_neural_network_present: neural_network.is_some(),
                                            current_neural_network_fingerprint:
                                                pg_base_neural_network_fingerprint,
                                            latest_policy_version_id: &metadata.id,
                                            latest_generation: metadata.generation,
                                            latest_updated_at_micros: metadata.updated_at_micros,
                                            latest_neural_network_present: metadata
                                                .neural_network
                                                .is_some(),
                                            latest_neural_network_fingerprint,
                                            current_tactical_learning_fingerprint:
                                                pg_base_tactical_learning_fingerprint,
                                            latest_tactical_learning_fingerprint,
                                            local_tactical_evolved_since_pg_refresh,
                                            postgres_tactical_learning_authoritative:
                                                pg_tactical_learning_authoritative,
                                        },
                                );
                                let postgres_evolution_options_changed =
                                    maybe_apply_postgres_evolution_options(
                                    "postgres_refresh_evolution_options_for_queue",
                                    next_episode + 1,
                                    &metadata.id,
                                    metadata.generation,
                                    &mut evolution_options,
                                    metadata.search_metadata.as_ref(),
                                );
                                if postgres_evolution_options_changed {
                                    clear_queue_evolution_samples_after_postgres_refresh(
                                        "postgres_refresh_evolution_option_samples_reset_for_queue",
                                        next_episode + 1,
                                        &metadata.id,
                                        &mut tactical_evolution_samples,
                                        &mut policy_evolution_samples,
                                        true,
                                        true,
                                    );
                                }
                                if refresh_decision.apply_tactical_learning {
                                    let applied_postgres_tactical =
                                        maybe_apply_postgres_tactical_learning(
                                        "postgres_refresh_tactical_learning_for_queue",
                                        next_episode + 1,
                                        &metadata.id,
                                        metadata.generation,
                                        match_config,
                                        &mut tactical_learning,
                                        metadata.tactical_learning.clone(),
                                    )?;
                                    if applied_postgres_tactical {
                                        local_tactical_evolved_since_pg_refresh = false;
                                        clear_queue_evolution_samples_after_postgres_refresh(
                                            "postgres_refresh_tactical_samples_reset_for_queue",
                                            next_episode + 1,
                                            &metadata.id,
                                            &mut tactical_evolution_samples,
                                            &mut policy_evolution_samples,
                                            true,
                                            false,
                                        );
                                    }
                                    if metadata.tactical_learning.is_some() {
                                        pg_base_tactical_learning_fingerprint =
                                            latest_tactical_learning_fingerprint;
                                    }
                                    active_config = match_config.clone();
                                }
                                if refresh_decision.refresh_policy {
                                    if let Some(version) = store.load_latest_active_policy(
                                        experiment_id,
                                        options.clone(),
                                        options.clone(),
                                    )? {
                                        let version_neural_network_fingerprint = version
                                            .neural_network
                                            .as_ref()
                                            .map(soccer_neural_network_snapshot_fingerprint);
                                        let version_policy_fingerprint = version
                                            .policy_fingerprint
                                            .or_else(|| {
                                                Some(soccer_team_q_policies_fingerprint(
                                                    &version.policies,
                                                ))
                                            });
                                        let version_tactical_learning_fingerprint = version
                                            .tactical_learning
                                            .as_ref()
                                            .map(soccer_tactical_learning_weights_fingerprint);
                                        let version_refresh_decision =
                                            soccer_postgres_policy_refresh_decision(
                                                SoccerPostgresPolicyRefreshCheck {
                                                    current_policy_version_id:
                                                        pg_base_policy_version_id.as_deref(),
                                                    current_generation: pg_generation,
                                                    current_updated_at_micros:
                                                        pg_base_policy_version_updated_at_micros,
                                                    current_policy_fingerprint:
                                                        pg_base_policy_fingerprint,
                                                    latest_policy_fingerprint:
                                                        version_policy_fingerprint,
                                                    current_neural_network_present:
                                                        neural_network.is_some(),
                                                    current_neural_network_fingerprint:
                                                        pg_base_neural_network_fingerprint,
                                                    latest_policy_version_id: &version.id,
                                                    latest_generation: version.generation,
                                                    latest_updated_at_micros:
                                                        version.updated_at_micros,
                                                    latest_neural_network_present:
                                                        version.neural_network.is_some(),
                                                    latest_neural_network_fingerprint:
                                                        version_neural_network_fingerprint,
                                                    current_tactical_learning_fingerprint:
                                                        pg_base_tactical_learning_fingerprint,
                                                    latest_tactical_learning_fingerprint:
                                                        version_tactical_learning_fingerprint,
                                                    local_tactical_evolved_since_pg_refresh,
                                                    postgres_tactical_learning_authoritative:
                                                        pg_tactical_learning_authoritative,
                                                },
                                            );
                                        maybe_apply_postgres_evolution_options(
                                            "postgres_refresh_evolution_options_for_loaded_queue_policy",
                                            next_episode + 1,
                                            &version.id,
                                            version.generation,
                                            &mut evolution_options,
                                            version.search_metadata.as_ref(),
                                        );
                                        println!(
                                            "postgres_refresh_policy_for_queue next_episode={} policy_version={} previous_policy_version={} generation={} neural_network={} pending_policy_versions={} pending_async_batches={} pending_async_policy_version_batches={}",
                                            next_episode + 1,
                                            version.id,
                                            pg_base_policy_version_id
                                                .as_deref()
                                                .unwrap_or("none"),
                                            version.generation,
                                            version.neural_network.is_some(),
                                            pg_policy_version_buffer.len(),
                                            pending_async_pg_batches,
                                            pending_async_policy_version_batches
                                        );
                                        *starting_policies = version.policies;
                                        *neural_network = version.neural_network;
                                        let applied_loaded_tactical = if version_refresh_decision
                                            .apply_tactical_learning
                                        {
                                            maybe_apply_postgres_tactical_learning(
                                                "postgres_refresh_loaded_tactical_learning_for_queue",
                                                next_episode + 1,
                                                &version.id,
                                                version.generation,
                                                match_config,
                                                &mut tactical_learning,
                                                version.tactical_learning.clone(),
                                            )?
                                        } else {
                                            false
                                        };
                                        if applied_loaded_tactical {
                                            local_tactical_evolved_since_pg_refresh = false;
                                            active_config = match_config.clone();
                                        }
                                        clear_queue_evolution_samples_after_postgres_refresh(
                                            "postgres_refresh_policy_samples_reset_for_queue",
                                            next_episode + 1,
                                            &version.id,
                                            &mut tactical_evolution_samples,
                                            &mut policy_evolution_samples,
                                            true,
                                            true,
                                        );
                                        pg_base_policy_version_id = Some(version.id.clone());
                                        pg_last_policy_version_id = Some(version.id);
                                        pg_generation = version.generation;
                                        pg_base_policy_version_updated_at_micros =
                                            version.updated_at_micros;
                                        pg_base_policy_fingerprint = version_policy_fingerprint;
                                        pg_base_neural_network_fingerprint =
                                            version_neural_network_fingerprint;
                                        pg_base_tactical_learning_fingerprint =
                                            version_tactical_learning_fingerprint;
                                        local_tactical_evolved_since_pg_refresh = false;
                                    }
                                }
                            }
                        }
                    }
                    if !tactical_learning_weights_match(
                        &match_config.tactical_learning,
                        &tactical_learning,
                    ) {
                        match_config.tactical_learning = tactical_learning.clone();
                        active_config = match_config.clone();
                    }
                    if pg_experiment_id.is_some() {
                        pg_episode_starting_policy_versions.insert(
                            next_episode,
                            (pg_base_policy_version_id.clone(), pg_generation),
                        );
                    }
                    let (curriculum_match_config, curriculum_episode) =
                        soccer_learning_curriculum_episode_config(
                            match_config,
                            next_episode,
                            &curriculum_config,
                        );
                    let curriculum_stage_label = curriculum_episode.stage.as_str();
                    if last_queue_curriculum_stage_label.as_deref()
                        != Some(curriculum_stage_label)
                    {
                        println!(
                            "queue_curriculum_stage next_episode={} stage={} drill_players_per_team={} field_yards={:.1}x{:.1} duration_seconds={:.1} local_mpc_max_players_per_team={}",
                            next_episode + 1,
                            curriculum_stage_label,
                            curriculum_episode.drill_players_per_team,
                            curriculum_episode.field_width_yards,
                            curriculum_episode.field_length_yards,
                            curriculum_episode.duration_seconds,
                            curriculum_episode.local_mpc_max_players_per_team
                        );
                        last_queue_curriculum_stage_label =
                            Some(curriculum_stage_label.to_string());
                    }
                    *match_config = curriculum_match_config;
                    Ok(())
                }
                SoccerLearningQueueEvent::CompletedGame {
                    game,
                    merged_policies,
                } => {
                    queue_completed_games_seen = queue_completed_games_seen.saturating_add(1);
                    let game_fitness = game.score.match_fitness;
                    let game_tactical_weights = game.starting_tactical_learning.clone();
                    tactical_evolution_samples.push_back(TacticalEvolutionSample {
                        summary: game.tactical_summary.clone(),
                        weights: game_tactical_weights,
                        fitness: game_fitness,
                    });
                    policy_evolution_samples.push_back(PolicyEvolutionSample {
                        summary: game.summary.clone(),
                        policies: game.policies.clone(),
                        fitness: game_fitness,
                    });
                    while tactical_evolution_samples.len() > tactical_evolution_window_games {
                        tactical_evolution_samples.pop_front();
                    }
                    while policy_evolution_samples.len() > tactical_evolution_window_games {
                        policy_evolution_samples.pop_front();
                    }
                    let should_evolve_tactical = evolution_enabled
                        && (evolution_interval_games <= 1
                            || queue_completed_games_seen >= games
                            || queue_completed_games_seen % evolution_interval_games == 0);
                    let curriculum_stage = soccer_learning_curriculum_stage_for_completed_games(
                        queue_completed_games_seen,
                        &curriculum_config,
                    );
                    let curriculum_stage_label = curriculum_stage.as_str();
                    let policy_promotion_evaluation = evaluate_soccer_policy_promotion_gate(
                        policy_evolution_samples
                            .iter()
                            .map(|sample| &sample.summary),
                        policy_promotion_gate,
                    );
                    let mut policy_evolved_fitness = None::<f64>;
                    let mut tactical_evolved_fitness = None::<f64>;
                    let mut policy_search_metadata = None::<serde_json::Value>;
                    let mut tactical_search_metadata = None::<serde_json::Value>;
                    if should_evolve_tactical && !policy_evolution_samples.is_empty() {
                        let mut ranked_policy_samples = policy_evolution_samples
                            .iter()
                            .enumerate()
                            .filter(|(_, sample)| sample.fitness.is_finite())
                            .map(|(sample_index, sample)| (sample_index, sample.fitness))
                            .collect::<Vec<_>>();
                        ranked_policy_samples.sort_by(|left, right| {
                            right
                                .1
                                .partial_cmp(&left.1)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });
                        if !ranked_policy_samples.is_empty() {
                            let elite_count = evolution_elite_games
                                .min(ranked_policy_samples.len())
                                .max(1);
                            let best_fitness = ranked_policy_samples
                                .first()
                                .map(|(_, fitness)| *fitness)
                                .unwrap_or(0.0);
                            let mut queue_policy_evolution_options = evolution_options;
                            queue_policy_evolution_options.seed = queue_policy_evolution_options
                                .seed
                                .wrapping_add(queue_completed_games_seen as u64)
                                .wrapping_add((game.episode as u64) << 32)
                                .wrapping_add(0x9e37_79b9_7f4a_7c15);
                            let evolved_policies = {
                                let mut parents = Vec::with_capacity(elite_count + 1);
                                parents.push((&*merged_policies, best_fitness));
                                for (sample_index, fitness) in
                                    ranked_policy_samples.iter().take(elite_count)
                                {
                                    parents
                                        .push((&policy_evolution_samples[*sample_index].policies, *fitness));
                                }
                                evolve_soccer_team_policies(&parents, queue_policy_evolution_options)
                                    .map_err(|err| err.to_string())?
                            };
                            let candidate_promoted = policy_promotion_evaluation.eligible;
                            if candidate_promoted {
                                *merged_policies = evolved_policies;
                                policy_evolved_fitness = Some(best_fitness);
                            }
                            policy_search_metadata = Some(serde_json::json!({
                                "algorithm": "evolutionary-policy-search",
                                "completedGames": queue_completed_games_seen,
                                "eliteGames": elite_count,
                                "windowGames": tactical_evolution_window_games,
                                "curriculumStage": curriculum_stage_label,
                                "bestFitness": best_fitness,
                                "candidatePromoted": candidate_promoted,
                                "promotion": queue_policy_promotion_search_metadata(
                                    curriculum_stage_label,
                                    &policy_promotion_evaluation,
                                ),
                                "options": queue_policy_evolution_options
                            }));
                            if candidate_promoted {
                                println!(
                                    "queue_policy_evolved completed_games={} curriculum_stage={} window_games={} elite_games={} best_fitness={:.4} mutation_rate={:.4} mutation_scale={:.4} crossover_rate={:.4} exploration_rate={:.4} exploration_scale={:.4} population_size={}",
                                    queue_completed_games_seen,
                                    curriculum_stage_label,
                                    tactical_evolution_window_games,
                                    elite_count,
                                    best_fitness,
                                    queue_policy_evolution_options.mutation_rate,
                                    queue_policy_evolution_options.mutation_scale,
                                    queue_policy_evolution_options.crossover_rate,
                                    queue_policy_evolution_options.exploration_rate,
                                    queue_policy_evolution_options.exploration_scale,
                                    queue_policy_evolution_options.population_size
                                );
                            } else {
                                println!(
                                    "queue_policy_evolution_held completed_games={} curriculum_stage={} window_games={} elite_games={} best_fitness={:.4} reasons={}",
                                    queue_completed_games_seen,
                                    curriculum_stage_label,
                                    tactical_evolution_window_games,
                                    elite_count,
                                    best_fitness,
                                    policy_promotion_evaluation.rejection_reasons.join("|")
                                );
                            }
                        }
                    }
                    if should_evolve_tactical && !tactical_evolution_samples.is_empty() {
                        let mut ranked_samples = tactical_evolution_samples
                            .iter()
                            .enumerate()
                            .filter(|(_, sample)| sample.fitness.is_finite())
                            .map(|(sample_index, sample)| (sample_index, sample.fitness))
                            .collect::<Vec<_>>();
                        ranked_samples.sort_by(|left, right| {
                            right
                                .1
                                .partial_cmp(&left.1)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });
                        if !ranked_samples.is_empty() {
                            let elite_count =
                                evolution_elite_games.min(ranked_samples.len()).max(1);
                            let best_fitness = ranked_samples
                                .first()
                                .map(|(_, fitness)| *fitness)
                                .unwrap_or(0.0);
                            let tactical_parents = ranked_samples
                                .iter()
                                .take(elite_count)
                                .map(|(sample_index, fitness)| {
                                    let sample = &tactical_evolution_samples[*sample_index];
                                    SoccerTacticalLearningGenomeParent {
                                        summary: &sample.summary,
                                        weights: &sample.weights,
                                        fitness: *fitness,
                                    }
                                })
                                .collect::<Vec<_>>();
                            let mut queue_evolution_options = evolution_options;
                            queue_evolution_options.seed = queue_evolution_options
                                .seed
                                .wrapping_add(queue_completed_games_seen as u64)
                                .wrapping_add((game.episode as u64) << 32);
                            let previous_tactical_learning = tactical_learning.clone();
                            tactical_learning = evolve_soccer_tactical_learning_weights_from_genomes(
                                &tactical_learning,
                                &tactical_parents,
                                queue_evolution_options,
                            );
                            validate_tactical_learning_weights(&tactical_learning)
                                .map_err(|err| err.to_string())?;
                            active_config.tactical_learning = tactical_learning.clone();
                            local_tactical_evolved_since_pg_refresh = true;
                            tactical_evolved_fitness = Some(best_fitness);
                            tactical_search_metadata =
                                Some(queue_tactical_evolution_search_metadata(
                                    queue_completed_games_seen,
                                    elite_count,
                                    tactical_evolution_window_games,
                                    tactical_evolution_samples.len(),
                                    best_fitness,
                                    queue_evolution_options,
                                    &previous_tactical_learning,
                                    &tactical_learning,
                                ));
                            println!(
                                "queue_tactical_weights_evolved completed_games={} window_games={} elite_games={} best_fitness={:.4} population_size={} attack_width_delta={:.3}->{:.3} attack_flank_lane={:.3}->{:.3} defense_contract_delta={:.3}->{:.3}",
                                queue_completed_games_seen,
                                tactical_evolution_window_games,
                                elite_count,
                                best_fitness,
                                queue_evolution_options.population_size,
                                previous_tactical_learning.attack_width_delta_weight,
                                tactical_learning.attack_width_delta_weight,
                                previous_tactical_learning.attack_flank_lane_weight,
                                tactical_learning.attack_flank_lane_weight,
                                previous_tactical_learning.defense_contract_delta_weight,
                                tactical_learning.defense_contract_delta_weight
                            );
                        }
                    }
                    let Some(experiment_id) = pg_experiment_id.as_deref() else {
                        return Ok(());
                    };
                    pg_completed_games_seen = pg_completed_games_seen.saturating_add(1);
                    let (pg_batch_base_policy_version_id, pg_batch_base_policy_generation) =
                        take_episode_starting_policy_version(
                            &mut pg_episode_starting_policy_versions,
                            game.episode,
                            &pg_base_policy_version_id,
                            pg_generation,
                        );
                    let policy_version_status =
                        queue_policy_version_status_for_promotion_gate(&policy_promotion_evaluation);
                    let should_write_policy_version = policy_evolved_fitness.is_some()
                        || pg_policy_version_interval_games <= 1
                        || pg_completed_games_seen >= games
                        || pg_completed_games_seen % pg_policy_version_interval_games == 0;
                    if should_write_policy_version
                        && policy_version_status != SOCCER_POLICY_STATUS_ACTIVE
                    {
                        println!(
                            "queue_policy_promotion_blocked completed_games={} curriculum_stage={} status={} sample_games={} mean_match_fitness={:.4} best_match_fitness={:.4} mean_play_quality={:.4} mean_goal_margin={:.4} reasons={}",
                            queue_completed_games_seen,
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
                    let output_policy_version_id = if should_write_policy_version {
                        let next_generation = pg_generation
                            .max(pg_batch_base_policy_generation)
                            .saturating_add(1);
                        let version_label = format!("{}-episode-{:06}", run_id, game.episode + 1);
                        let output_policy_version_id = Uuid::new_v4().to_string();
                        let promotion_search_metadata = queue_policy_promotion_search_metadata(
                            curriculum_stage_label,
                            &policy_promotion_evaluation,
                        );
                        let search_metadata = if policy_search_metadata.is_some()
                            || tactical_search_metadata.is_some()
                            || policy_promotion_gate.enabled
                        {
                            Some(serde_json::json!({
                                "policy": policy_search_metadata,
                                "tactical": tactical_search_metadata,
                                "promotion": promotion_search_metadata
                            }))
                        } else {
                            None
                        };
                        pg_policy_version_buffer.push(PendingPostgresPolicyVersion {
                            id: output_policy_version_id.clone(),
                            parent_policy_version_id: pg_batch_base_policy_version_id.clone(),
                            generation: next_generation,
                            version_label,
                            source_kind: queue_policy_version_source_kind(
                                policy_evolved_fitness.is_some(),
                                tactical_evolved_fitness.is_some(),
                            ),
                            status: policy_version_status,
                            config: active_config.clone(),
                            home_options: options.clone(),
                            away_options: options.clone(),
                            policies: (*merged_policies).clone(),
                            fitness: policy_evolved_fitness.unwrap_or(game.score.match_fitness),
                            neural_network: game.neural_network.clone(),
                            search_metadata,
                        });
                        if policy_version_status == SOCCER_POLICY_STATUS_ACTIVE {
                            pg_base_policy_version_id = Some(output_policy_version_id.clone());
                            pg_last_policy_version_id = Some(output_policy_version_id.clone());
                            pg_generation = next_generation;
                            pg_base_policy_version_updated_at_micros = 0;
                            pg_base_policy_fingerprint =
                                Some(soccer_team_q_policies_fingerprint(&*merged_policies));
                            pg_base_neural_network_fingerprint = game
                                .neural_network
                                .as_ref()
                                .map(soccer_neural_network_snapshot_fingerprint);
                            pg_base_tactical_learning_fingerprint = Some(
                                soccer_tactical_learning_weights_fingerprint(
                                    &active_config.tactical_learning,
                                ),
                            );
                        }
                        Some(output_policy_version_id)
                    } else {
                        pg_last_policy_version_id.clone()
                    };
                    pg_completed_buffer.push(PendingPostgresCompletedRun {
                        completed_game: game.clone(),
                        base_policy_version_id: pg_batch_base_policy_version_id,
                        output_policy_version_id,
                        generation: pg_generation,
                    });
                    if pg_completed_buffer.len() >= pg_completed_run_batch_games {
                        if soccer_should_flush_postgres_policy_versions_for_new_sim(
                            pg_refresh_for_new_sims,
                            pg_flush_policy_versions_before_new_sim,
                            pg_policy_version_buffer.len(),
                        ) {
                            if let Some(store) = pg_store.as_mut() {
                                flush_postgres_policy_versions_for_new_sims(
                                    store,
                                    experiment_id,
                                    &run_id,
                                    &mut pg_policy_version_buffer,
                                )?;
                            }
                        }
                        pg_persisted_games += if let Some(writer) = pg_completed_writer.as_mut() {
                            writer.enqueue(
                                experiment_id,
                                &run_id,
                                &mut pg_policy_version_buffer,
                                &mut pg_completed_buffer,
                                pg_completed_run_retention_games,
                            )?
                        } else if let Some(store) = pg_store.as_mut() {
                            flush_postgres_completed_runs(
                                store,
                                experiment_id,
                                &run_id,
                                &mut pg_policy_version_buffer,
                                &mut pg_completed_buffer,
                                pg_completed_run_retention_games,
                            )?
                        } else {
                            0
                        };
                    }
                    Ok(())
                }
            }
        },
    )
    .map_err(invalid_data)?;

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
                    &run_id,
                    &mut pg_policy_version_buffer,
                )
                .map_err(invalid_data)?;
            }
        }
        pg_persisted_games += if let Some(writer) = pg_completed_writer.as_mut() {
            writer
                .enqueue(
                    experiment_id,
                    &run_id,
                    &mut pg_policy_version_buffer,
                    &mut pg_completed_buffer,
                    pg_completed_run_retention_games,
                )
                .map_err(invalid_data)?
        } else if let Some(store) = pg_store.as_mut() {
            flush_postgres_completed_runs(
                store,
                experiment_id,
                &run_id,
                &mut pg_policy_version_buffer,
                &mut pg_completed_buffer,
                pg_completed_run_retention_games,
            )
            .map_err(invalid_data)?
        } else {
            0
        };
    }
    if let Some(writer) = pg_completed_writer {
        pg_persisted_games += writer.finish().map_err(invalid_data)?;
    }

    let artifact = soccer_self_play_artifact_from_queue_report(active_config, options, &report);
    write_json(&artifact_path, &artifact)?;
    let learned_params = SoccerSelfPlayLearnedParams::from_training_artifact_with_neural_network(
        &artifact,
        report.latest_neural_network.clone(),
    );
    write_json(&learned_params_path, &learned_params)?;

    println!(
        "soccer_learning_queue_done completed={} failed={} elapsed={:.2}s home_goals={} away_goals={} policy_entries={} target_entries={} postgres_persisted_games={}",
        report.completed_games,
        report.failed_games,
        report.elapsed_seconds,
        report.total_home_goals,
        report.total_away_goals,
        report.final_policy_entries,
        report.final_target_entries,
        pg_persisted_games,
    );
    println!("run_dir={}", run_dir.display());
    println!("artifact={}", artifact_path.display());
    println!("learned_params={}", learned_params_path.display());
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("soccer learning queue failed: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use soccer_engine::des::general::soccer::{
        SoccerMatch, SoccerQEntry, SoccerQPolicy, SoccerQStateKey,
    };
    use std::sync::Mutex;

    static SOCCER_QUEUE_PG_ENV_LOCK: Mutex<()> = Mutex::new(());

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

    #[test]
    fn learning_queue_mpc_env_enables_execution_stack() {
        let _lock = SOCCER_QUEUE_PG_ENV_LOCK.lock().expect("env lock poisoned");
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
    fn learning_queue_mpc_env_rejects_dependent_stack_without_tier2() {
        let _lock = SOCCER_QUEUE_PG_ENV_LOCK.lock().expect("env lock poisoned");
        let _guards = clear_mpc_env();
        std::env::set_var("SOCCER_MPC_FIELD_AWARE_ENABLED", "1");

        let defaults = MatchConfig::default();
        let mut config = defaults.clone();
        let err = apply_env_mpc_config(&mut config, &defaults)
            .expect_err("dependent MPC features require tier2 MPC");

        assert!(err.to_string().contains("SOCCER_MPC_ENABLED"));
    }

    #[test]
    fn learning_queue_mpc_master_off_disables_local_mpc() {
        let _lock = SOCCER_QUEUE_PG_ENV_LOCK.lock().expect("env lock poisoned");
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
    fn queue_settings_reject_dt_above_soccer_runtime_limit() {
        let options = SoccerQPolicyOptions::default();

        assert!(validate_queue_settings(1, 2, 45.0, 90.0, 900.0, 4.0, 4, 1, &options).is_ok());
        let err = validate_queue_settings(1, 2, 45.0, 90.0, 900.0, 4.01, 4, 1, &options)
            .expect_err("dt above runtime-safe bound should fail");
        assert!(err.to_string().contains("[0.01, 4.0]"));
    }

    fn empty_pg_batch(experiment_id: &str, runner_id: &str) -> PostgresCompletedRunBatch {
        PostgresCompletedRunBatch {
            experiment_id: experiment_id.to_string(),
            runner_id: runner_id.to_string(),
            completed_run_retention_games: 0,
            policy_version_batches: 0,
            pending_policy_versions: Vec::new(),
            pending_runs: Vec::new(),
        }
    }

    fn queue_test_state() -> SoccerQStateKey {
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
        .expect("queue test state")
    }

    fn queue_test_policy_with_action(value: f64, visits: u32) -> SoccerTeamQPolicies {
        let entries = vec![SoccerQEntry {
            state: queue_test_state(),
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

    #[test]
    fn default_queue_postgres_policy_versions_are_batched() {
        assert_eq!(default_postgres_policy_version_interval_games(0), 10);
        assert_eq!(default_postgres_policy_version_interval_games(1), 10);
        assert_eq!(default_postgres_policy_version_interval_games(4), 10);
    }

    #[test]
    fn default_queue_postgres_batches_respect_parallelism() {
        assert_eq!(default_postgres_completed_run_batch_games(4), 10);
        assert_eq!(default_postgres_completed_run_batch_games(16), 16);
        assert_eq!(default_postgres_completed_run_batch_games(64), 64);
    }

    #[test]
    fn queue_postgres_persistence_is_completed_game_batched_not_timestep_or_clock_based() {
        assert_eq!(
            default_postgres_policy_version_interval_games(1),
            DEFAULT_SOCCER_QUEUE_POSTGRES_POLICY_VERSION_INTERVAL_GAMES
        );
        assert_eq!(
            default_postgres_completed_run_batch_games(1),
            DEFAULT_SOCCER_QUEUE_POSTGRES_COMPLETED_RUN_BATCH_GAMES
        );
        assert!(DEFAULT_SOCCER_QUEUE_POSTGRES_POLICY_VERSION_INTERVAL_GAMES > 1);
        assert!(DEFAULT_SOCCER_QUEUE_POSTGRES_COMPLETED_RUN_BATCH_GAMES > 1);
        assert!(DEFAULT_SOCCER_QUEUE_POSTGRES_ASYNC_BATCH_QUEUE > 1);
        assert!(DEFAULT_SOCCER_QUEUE_POSTGRES_ASYNC_COALESCE_BATCHES > 1);
        assert!(
            DEFAULT_SOCCER_QUEUE_POSTGRES_ASYNC_COALESCE_WAIT_MS
                < DEFAULT_SOCCER_QUEUE_NEURAL_DRAIN_TIMEOUT_MS.saturating_mul(10)
        );
    }

    #[test]
    fn default_queue_postgres_completed_run_retention_is_disabled() {
        assert_eq!(
            DEFAULT_SOCCER_QUEUE_POSTGRES_COMPLETED_RUN_RETENTION_GAMES,
            0
        );
    }

    #[test]
    fn default_queue_neural_drain_timeout_keeps_worker_wait_bounded() {
        assert_eq!(DEFAULT_SOCCER_QUEUE_NEURAL_DRAIN_TIMEOUT_MS, 2);
    }

    #[test]
    fn default_queue_evolution_interval_respects_parallelism() {
        assert_eq!(default_queue_evolution_interval_games(0), 10);
        assert_eq!(default_queue_evolution_interval_games(4), 10);
        assert_eq!(default_queue_evolution_interval_games(16), 16);
    }

    #[test]
    fn default_queue_evolution_window_preserves_elite_search_history() {
        assert_eq!(default_queue_evolution_window_games(10, 4), 10);
        assert_eq!(default_queue_evolution_window_games(4, 10), 10);
        assert_eq!(default_queue_evolution_window_games(0, 0), 1);
    }

    #[test]
    fn default_queue_postgres_async_writer_stays_bounded() {
        assert_eq!(DEFAULT_SOCCER_QUEUE_POSTGRES_ASYNC_BATCH_QUEUE, 16);
    }

    #[test]
    fn default_queue_postgres_async_writer_coalesces_io_batches() {
        assert_eq!(DEFAULT_SOCCER_QUEUE_POSTGRES_ASYNC_COALESCE_BATCHES, 16);
    }

    #[test]
    fn default_queue_postgres_async_writer_waits_briefly_to_coalesce_io() {
        assert_eq!(DEFAULT_SOCCER_QUEUE_POSTGRES_ASYNC_COALESCE_WAIT_MS, 2);
    }

    #[test]
    fn queue_async_postgres_writer_missing_db_drains_policy_version_wait_counter() {
        let _lock = SOCCER_QUEUE_PG_ENV_LOCK.lock().expect("lock pg env");
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
    fn default_queue_postgres_tactical_learning_is_authoritative() {
        assert!(DEFAULT_SOCCER_QUEUE_POSTGRES_TACTICAL_LEARNING_AUTHORITATIVE);
    }

    #[test]
    fn default_queue_postgres_policy_heads_flush_before_new_sims() {
        assert!(DEFAULT_SOCCER_QUEUE_POSTGRES_FLUSH_POLICY_VERSIONS_BEFORE_NEW_SIM);
    }

    #[test]
    fn default_queue_postgres_refreshes_before_new_sims_even_with_resume_artifacts() {
        assert!(DEFAULT_SOCCER_QUEUE_POSTGRES_REFRESH_WITH_RESUME_ARTIFACT);
        assert!(soccer_should_refresh_postgres_for_new_sim(
            true,
            DEFAULT_SOCCER_QUEUE_POSTGRES_REFRESH_WITH_RESUME_ARTIFACT
        ));
    }

    #[test]
    fn default_queue_postgres_new_sim_plan_publishes_evolved_heads_before_reads() {
        let plan = soccer_postgres_new_sim_refresh_plan(
            true,
            DEFAULT_SOCCER_QUEUE_POSTGRES_REFRESH_WITH_RESUME_ARTIFACT,
            DEFAULT_SOCCER_QUEUE_POSTGRES_FLUSH_POLICY_VERSIONS_BEFORE_NEW_SIM,
            1,
            1,
        );

        assert!(plan.refresh_from_postgres);
        assert!(plan.flush_pending_policy_versions);
        assert!(plan.wait_for_async_policy_versions);
    }

    #[test]
    fn default_queue_evolutionary_search_stays_enabled_for_new_sims() {
        let options = SoccerEvolutionOptions::default();

        assert!(DEFAULT_SOCCER_QUEUE_EVOLUTION_ENABLED);
        assert!(DEFAULT_SOCCER_QUEUE_EVOLUTION_ELITE_GAMES >= 1);
        assert!(
            default_queue_evolution_window_games(
                default_queue_evolution_interval_games(4),
                DEFAULT_SOCCER_QUEUE_EVOLUTION_ELITE_GAMES,
            ) >= DEFAULT_SOCCER_QUEUE_EVOLUTION_ELITE_GAMES
        );
        assert!(options.population_size > 1);
        assert!(options.mutation_rate > 0.0);
        assert!(options.mutation_scale > 0.0);
        assert!(options.crossover_rate > 0.0);
    }

    #[test]
    fn postgres_refresh_can_clear_stale_evolution_sample_windows() {
        let options = SoccerQPolicyOptions::default();
        let mut tactical_samples = VecDeque::from([
            TacticalEvolutionSample {
                summary: SoccerTacticalLearningSummary::default(),
                weights: SoccerTacticalLearningWeights::default(),
                fitness: 0.5,
            },
            TacticalEvolutionSample {
                summary: SoccerTacticalLearningSummary::default(),
                weights: SoccerTacticalLearningWeights::default(),
                fitness: 0.8,
            },
        ]);
        let mut policy_samples = VecDeque::from([PolicyEvolutionSample {
            summary: MatchSummary {
                score_home: 0,
                score_away: 0,
                ticks: 0,
                simulated_seconds: 0.0,
                stats: Default::default(),
            },
            policies: SoccerTeamQPolicies::new(options),
            fitness: 0.7,
        }]);

        let tactical_only = clear_queue_evolution_samples_after_postgres_refresh(
            "test_postgres_refresh_tactical_samples_reset",
            4,
            "policy-v1",
            &mut tactical_samples,
            &mut policy_samples,
            true,
            false,
        );

        assert_eq!(tactical_only, (2, 0));
        assert!(tactical_samples.is_empty());
        assert_eq!(policy_samples.len(), 1);

        let policy_refresh = clear_queue_evolution_samples_after_postgres_refresh(
            "test_postgres_refresh_policy_samples_reset",
            5,
            "policy-v2",
            &mut tactical_samples,
            &mut policy_samples,
            true,
            true,
        );

        assert_eq!(policy_refresh, (0, 1));
        assert!(tactical_samples.is_empty());
        assert!(policy_samples.is_empty());
    }

    #[test]
    fn queue_policy_version_source_kind_tracks_tactical_and_policy_evolution() {
        assert_eq!(
            queue_policy_version_source_kind(false, false),
            SOCCER_QUEUE_POLICY_SOURCE_MERGE
        );
        assert_eq!(
            queue_policy_version_source_kind(true, false),
            SOCCER_QUEUE_POLICY_SOURCE_EVOLUTION
        );
        assert_eq!(
            queue_policy_version_source_kind(false, true),
            SOCCER_QUEUE_POLICY_SOURCE_EVOLUTION
        );
        assert_eq!(
            queue_policy_version_source_kind(true, true),
            SOCCER_QUEUE_POLICY_SOURCE_EVOLUTION
        );
    }

    #[test]
    fn queue_policy_promotion_gate_maps_failed_eval_to_archived_status() {
        let mut evaluation = SoccerPolicyPromotionGateEvaluation {
            enabled: true,
            eligible: true,
            sample_games: 2,
            min_sample_games: 1,
            mean_match_fitness: 0.5,
            best_match_fitness: 0.8,
            mean_play_quality: 0.1,
            mean_conceded_goals: 1.0,
            mean_goal_margin: 1.0,
            mean_chain_net_loss: 0.0,
            rejection_reasons: Vec::new(),
        };

        assert_eq!(
            queue_policy_version_status_for_promotion_gate(&evaluation),
            SOCCER_POLICY_STATUS_ACTIVE
        );

        evaluation.eligible = false;
        evaluation
            .rejection_reasons
            .push("mean_goal_margin 5.0000 above max_mean_goal_margin 3.0000".to_string());

        assert_eq!(
            queue_policy_version_status_for_promotion_gate(&evaluation),
            SOCCER_POLICY_STATUS_ARCHIVED
        );
        let metadata = queue_policy_promotion_search_metadata("duels", &evaluation);
        assert_eq!(metadata["curriculumStage"], serde_json::json!("duels"));
        assert_eq!(
            metadata["status"],
            serde_json::json!(SOCCER_POLICY_STATUS_ARCHIVED)
        );
    }

    #[test]
    fn queue_tactical_evolution_metadata_stores_weight_json_and_fingerprints() {
        let mut previous = SoccerTacticalLearningWeights::default();
        previous.attack_flank_lane_weight = 1.10;
        previous.defense_contract_delta_weight = 0.84;
        previous.formation_lp_alignment_weight = 0.22;
        let mut evolved = previous.clone();
        evolved.attack_flank_lane_weight = 1.38;
        evolved.defense_contract_delta_weight = 1.17;
        evolved.formation_lp_alignment_weight = 0.31;
        let options = SoccerEvolutionOptions {
            mutation_rate: 0.08,
            mutation_scale: 0.30,
            crossover_rate: 0.55,
            exploration_rate: 0.12,
            exploration_scale: 0.70,
            elite_weight_floor: 0.02,
            population_size: 18,
            seed: 4242,
        };

        let metadata = queue_tactical_evolution_search_metadata(
            21, 4, 12, 9, 1.75, options, &previous, &evolved,
        );

        assert_eq!(
            metadata["algorithm"],
            serde_json::json!("evolutionary-genetic-programming-tactical-search")
        );
        assert_eq!(metadata["completedGames"], serde_json::json!(21));
        assert_eq!(metadata["eliteGames"], serde_json::json!(4));
        assert_eq!(metadata["windowGames"], serde_json::json!(12));
        assert_eq!(metadata["sampleCount"], serde_json::json!(9));
        assert_eq!(metadata["bestFitness"], serde_json::json!(1.75));
        assert_eq!(metadata["options"]["populationSize"], serde_json::json!(18));
        assert_eq!(metadata["options"]["seed"], serde_json::json!(4242));
        assert_eq!(
            metadata["previousTacticalLearning"]["attackFlankLaneWeight"],
            serde_json::json!(1.10)
        );
        assert_eq!(
            metadata["evolvedTacticalLearning"]["attackFlankLaneWeight"],
            serde_json::json!(1.38)
        );
        assert_eq!(
            metadata["previousTacticalLearning"]["defenseContractDeltaWeight"],
            serde_json::json!(0.84)
        );
        assert_eq!(
            metadata["evolvedTacticalLearning"]["defenseContractDeltaWeight"],
            serde_json::json!(1.17)
        );
        assert_eq!(
            metadata["previousTacticalLearningFingerprint"],
            serde_json::json!(soccer_tactical_learning_weights_fingerprint(&previous))
        );
        assert_eq!(
            metadata["evolvedTacticalLearningFingerprint"],
            serde_json::json!(soccer_tactical_learning_weights_fingerprint(&evolved))
        );
        assert_ne!(
            metadata["previousTacticalLearningFingerprint"],
            metadata["evolvedTacticalLearningFingerprint"]
        );
    }

    #[test]
    fn queue_completed_game_uses_episode_starting_policy_snapshot() {
        let mut episode_versions = HashMap::new();
        episode_versions.insert(7, (Some("episode-v7".to_string()), 4));
        let current_policy_version_id = Some("current-v9".to_string());

        let recorded = take_episode_starting_policy_version(
            &mut episode_versions,
            7,
            &current_policy_version_id,
            9,
        );
        let fallback = take_episode_starting_policy_version(
            &mut episode_versions,
            8,
            &current_policy_version_id,
            9,
        );

        assert_eq!(recorded, (Some("episode-v7".to_string()), 4));
        assert_eq!(fallback, (Some("current-v9".to_string()), 9));
    }

    #[test]
    fn queue_new_sims_sample_latest_postgres_tactical_weights() {
        let mut active_weights = SoccerTacticalLearningWeights::default();
        let mut active_config = MatchConfig {
            tactical_learning: active_weights.clone(),
            ..Default::default()
        };
        let mut samples = Vec::new();

        for episode in 0..3 {
            let mut postgres_weights = SoccerTacticalLearningWeights::default();
            postgres_weights.attack_flank_lane_weight = 1.10 + episode as f64 * 0.07;
            postgres_weights.defense_contract_delta_weight = 0.90 + episode as f64 * 0.05;
            postgres_weights.formation_lp_alignment_weight = 0.20 + episode as f64 * 0.03;

            maybe_apply_postgres_tactical_learning(
                "test_postgres_refresh_tactical_learning_for_queue",
                episode + 1,
                &format!("pg-v-{episode}"),
                episode as i32,
                &mut active_config,
                &mut active_weights,
                Some(postgres_weights),
            )
            .expect("valid postgres tactical learning weights");
            samples.push(TacticalEvolutionSample {
                summary: SoccerTacticalLearningSummary::default(),
                weights: active_config.tactical_learning.clone(),
                fitness: 1.0 + episode as f64,
            });
        }

        assert_eq!(samples.len(), 3);
        for (episode, sample) in samples.iter().enumerate() {
            assert!(
                (sample.weights.attack_flank_lane_weight - (1.10 + episode as f64 * 0.07)).abs()
                    < 1e-12
            );
            assert!(
                (sample.weights.defense_contract_delta_weight - (0.90 + episode as f64 * 0.05))
                    .abs()
                    < 1e-12
            );
            assert!(
                (sample.weights.formation_lp_alignment_weight - (0.20 + episode as f64 * 0.03))
                    .abs()
                    < 1e-12
            );
        }
    }

    #[test]
    fn queue_new_sims_sample_latest_postgres_evolution_options() {
        let mut evolution_options = SoccerEvolutionOptions::default();
        let mut samples = Vec::new();

        for episode in 0..3 {
            let metadata = serde_json::json!({
                "algorithm": "evolutionary-genetic-programming-tactical-search",
                "options": {
                    "mutationRate": 0.05 + episode as f64 * 0.03,
                    "crossoverRate": 0.35 + episode as f64 * 0.06,
                    "explorationScale": 0.60 + episode as f64 * 0.08,
                    "populationSize": 10 + episode * 3,
                    "seed": 1200 + episode as u64
                }
            });

            assert!(maybe_apply_postgres_evolution_options(
                "test_postgres_refresh_evolution_options_for_queue",
                episode + 1,
                &format!("pg-v-{episode}"),
                episode as i32,
                &mut evolution_options,
                Some(&metadata),
            ));
            samples.push((episode, evolution_options));
        }

        assert_eq!(samples.len(), 3);
        for (episode, options) in samples {
            assert!((options.mutation_rate - (0.05 + episode as f64 * 0.03)).abs() < 1e-12);
            assert!((options.crossover_rate - (0.35 + episode as f64 * 0.06)).abs() < 1e-12);
            assert!((options.exploration_scale - (0.60 + episode as f64 * 0.08)).abs() < 1e-12);
            assert_eq!(options.population_size, 10 + episode * 3);
            assert_eq!(options.seed, 1200 + episode as u64);
        }
    }

    #[test]
    fn queue_loaded_policy_refresh_uses_full_row_metadata_before_next_sim() {
        let stale_metadata_decision =
            soccer_postgres_policy_refresh_decision(SoccerPostgresPolicyRefreshCheck {
                current_policy_version_id: Some("pg-v1"),
                current_generation: 4,
                current_updated_at_micros: 100,
                current_policy_fingerprint: Some(10),
                latest_policy_fingerprint: Some(10),
                current_neural_network_present: true,
                current_neural_network_fingerprint: Some(20),
                latest_policy_version_id: "pg-v1",
                latest_generation: 4,
                latest_updated_at_micros: 100,
                latest_neural_network_present: true,
                latest_neural_network_fingerprint: Some(20),
                current_tactical_learning_fingerprint: Some(30),
                latest_tactical_learning_fingerprint: Some(30),
                local_tactical_evolved_since_pg_refresh: true,
                postgres_tactical_learning_authoritative: false,
            });
        assert!(!stale_metadata_decision.refresh_policy);
        assert!(!stale_metadata_decision.apply_tactical_learning);

        let mut loaded_weights = SoccerTacticalLearningWeights::default();
        loaded_weights.attack_flank_lane_weight = 1.48;
        loaded_weights.defense_contract_delta_weight = 1.31;
        loaded_weights.formation_lp_alignment_weight = 0.42;
        let loaded_tactical_fingerprint =
            soccer_tactical_learning_weights_fingerprint(&loaded_weights);
        let loaded_decision =
            soccer_postgres_policy_refresh_decision(SoccerPostgresPolicyRefreshCheck {
                current_policy_version_id: Some("pg-v1"),
                current_generation: 4,
                current_updated_at_micros: 100,
                current_policy_fingerprint: Some(10),
                latest_policy_fingerprint: Some(11),
                current_neural_network_present: true,
                current_neural_network_fingerprint: Some(20),
                latest_policy_version_id: "pg-v2",
                latest_generation: 5,
                latest_updated_at_micros: 150,
                latest_neural_network_present: true,
                latest_neural_network_fingerprint: Some(21),
                current_tactical_learning_fingerprint: Some(30),
                latest_tactical_learning_fingerprint: Some(loaded_tactical_fingerprint),
                local_tactical_evolved_since_pg_refresh: true,
                postgres_tactical_learning_authoritative: false,
            });
        assert!(loaded_decision.refresh_policy);
        assert!(loaded_decision.apply_tactical_learning);

        let mut active_weights = SoccerTacticalLearningWeights::default();
        let mut active_config = MatchConfig {
            tactical_learning: active_weights.clone(),
            ..Default::default()
        };
        if loaded_decision.apply_tactical_learning {
            assert!(maybe_apply_postgres_tactical_learning(
                "test_loaded_queue_policy_tactical_learning",
                8,
                "pg-v2",
                5,
                &mut active_config,
                &mut active_weights,
                Some(loaded_weights.clone()),
            )
            .expect("valid loaded tactical weights"));
        }
        assert_eq!(
            active_config.tactical_learning.attack_flank_lane_weight,
            loaded_weights.attack_flank_lane_weight
        );
        assert_eq!(
            active_config
                .tactical_learning
                .defense_contract_delta_weight,
            loaded_weights.defense_contract_delta_weight
        );

        let mut evolution_options = SoccerEvolutionOptions::default();
        let loaded_search_metadata = serde_json::json!({
            "algorithm": "evolutionary-genetic-programming-tactical-search",
            "options": {
                "mutationRate": 0.12,
                "crossoverRate": 0.62,
                "explorationScale": 1.05,
                "populationSize": 24,
                "seed": 90210
            }
        });
        assert!(maybe_apply_postgres_evolution_options(
            "test_loaded_queue_policy_evolution_options",
            8,
            "pg-v2",
            5,
            &mut evolution_options,
            Some(&loaded_search_metadata),
        ));
        assert_eq!(evolution_options.population_size, 24);
        assert_eq!(evolution_options.seed, 90210);
        assert!((evolution_options.crossover_rate - 0.62).abs() < 1e-12);
    }

    #[test]
    fn queue_new_sim_uses_postgres_policy_neural_weights_and_gp_options() {
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
        let refreshed_policies = queue_test_policy_with_action(14.0, 9);
        let refreshed_policy_fingerprint = soccer_team_q_policies_fingerprint(&refreshed_policies);
        let mut refreshed_weights = SoccerTacticalLearningWeights::default();
        refreshed_weights.attack_flank_lane_weight = 1.52;
        refreshed_weights.defense_contract_delta_weight = 1.24;
        refreshed_weights.formation_lp_alignment_weight = 0.47;
        let refreshed_tactical_fingerprint =
            soccer_tactical_learning_weights_fingerprint(&refreshed_weights);
        let mut refreshed_neural = SoccerMatch::default_11v11(match_config.clone())
            .learning_snapshot()
            .neural_network
            .expect("queue neural snapshot");
        refreshed_neural.layers[0].weights[0][0] = 0.515_151;
        refreshed_neural.layers[0].biases[0] = -0.212_121;
        let refreshed_neural_fingerprint =
            soccer_neural_network_snapshot_fingerprint(&refreshed_neural);
        let postgres_search_metadata = serde_json::json!({
            "algorithm": "geneticProgramming",
            "options": {
                "mutationRate": 0.13,
                "crossoverRate": 0.67,
                "explorationScale": 1.20,
                "populationSize": 32,
                "seed": 5511
            }
        });
        let mut active_weights = SoccerTacticalLearningWeights::default();
        let mut evolution_options = SoccerEvolutionOptions::default();
        let mut pg_base_policy_version_id = Some("pg-v1".to_string());
        let mut pg_generation = 4;
        let mut submitted_versions = Vec::new();
        let mut completed_weights = Vec::new();

        let report = run_soccer_learning_queue_with_events(
            SoccerLearningQueueRunnerConfig {
                games: 1,
                parallel_games: 1,
                base_seed: 2097,
                match_config,
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
                        policies,
                        neural_network,
                    } => {
                        let decision = soccer_postgres_policy_refresh_decision(
                            SoccerPostgresPolicyRefreshCheck {
                                current_policy_version_id: pg_base_policy_version_id.as_deref(),
                                current_generation: pg_generation,
                                current_updated_at_micros: 100,
                                current_policy_fingerprint: Some(1),
                                latest_policy_fingerprint: Some(refreshed_policy_fingerprint),
                                current_neural_network_present: neural_network.is_some(),
                                current_neural_network_fingerprint: None,
                                latest_policy_version_id: "pg-v2",
                                latest_generation: 5,
                                latest_updated_at_micros: 200,
                                latest_neural_network_present: true,
                                latest_neural_network_fingerprint: Some(
                                    refreshed_neural_fingerprint,
                                ),
                                current_tactical_learning_fingerprint: Some(
                                    soccer_tactical_learning_weights_fingerprint(&active_weights),
                                ),
                                latest_tactical_learning_fingerprint: Some(
                                    refreshed_tactical_fingerprint,
                                ),
                                local_tactical_evolved_since_pg_refresh: true,
                                postgres_tactical_learning_authoritative: true,
                            },
                        );
                        assert!(decision.refresh_policy);
                        assert!(decision.apply_tactical_learning);
                        *policies = refreshed_policies.clone();
                        *neural_network = Some(refreshed_neural.clone());
                        maybe_apply_postgres_tactical_learning(
                            "test_postgres_refresh_tactical_learning_for_queue",
                            next_episode + 1,
                            "pg-v2",
                            5,
                            match_config,
                            &mut active_weights,
                            Some(refreshed_weights.clone()),
                        )?;
                        assert!(maybe_apply_postgres_evolution_options(
                            "test_postgres_refresh_evolution_options_for_queue",
                            next_episode + 1,
                            "pg-v2",
                            5,
                            &mut evolution_options,
                            Some(&postgres_search_metadata),
                        ));
                        pg_base_policy_version_id = Some("pg-v2".to_string());
                        pg_generation = 5;
                        submitted_versions.push((
                            next_episode,
                            pg_base_policy_version_id.clone(),
                            pg_generation,
                            evolution_options,
                        ));
                    }
                    SoccerLearningQueueEvent::CompletedGame { game, .. } => {
                        completed_weights.push((
                            game.episode,
                            game.starting_tactical_learning.attack_flank_lane_weight,
                            game.starting_tactical_learning
                                .defense_contract_delta_weight,
                            game.starting_tactical_learning
                                .formation_lp_alignment_weight,
                        ));
                    }
                }
                Ok(())
            },
        )
        .expect("queue run");

        assert_eq!(submitted_versions.len(), 1);
        assert_eq!(submitted_versions[0].0, 0);
        assert_eq!(submitted_versions[0].1.as_deref(), Some("pg-v2"));
        assert_eq!(submitted_versions[0].2, 5);
        assert_eq!(submitted_versions[0].3.population_size, 32);
        assert_eq!(submitted_versions[0].3.seed, 5511);
        assert!((submitted_versions[0].3.exploration_scale - 1.20).abs() < 1e-12);
        assert_eq!(report.completed_games, 1);
        assert_eq!(report.failed_games, 0);
        let entries = report.final_policies.home.entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].action, "pass");
        assert!((entries[0].value - 14.0).abs() < 1e-12);
        assert_eq!(entries[0].visits, 9);
        let latest_neural = report
            .latest_neural_network
            .expect("latest queue neural snapshot");
        assert_eq!(latest_neural.layers[0].weights[0][0], 0.515_151);
        assert_eq!(latest_neural.layers[0].biases[0], -0.212_121);
        assert_eq!(completed_weights, vec![(0, 1.52, 1.24, 0.47)]);
    }

    #[test]
    fn queue_postgres_evolution_option_refresh_clears_stale_search_windows() {
        let options = SoccerQPolicyOptions::default();
        let mut evolution_options = SoccerEvolutionOptions::default();
        let mut tactical_samples = VecDeque::from([TacticalEvolutionSample {
            summary: SoccerTacticalLearningSummary::default(),
            weights: SoccerTacticalLearningWeights::default(),
            fitness: 0.75,
        }]);
        let mut policy_samples = VecDeque::from([PolicyEvolutionSample {
            summary: MatchSummary {
                score_home: 0,
                score_away: 0,
                ticks: 0,
                simulated_seconds: 0.0,
                stats: Default::default(),
            },
            policies: SoccerTeamQPolicies::new(options),
            fitness: 0.80,
        }]);
        let metadata = serde_json::json!({
            "algorithm": "evolutionary-genetic-programming-tactical-search",
            "options": {
                "mutationRate": 0.09,
                "crossoverRate": 0.58,
                "explorationScale": 0.95,
                "populationSize": 18,
                "seed": 3201
            }
        });

        let changed = maybe_apply_postgres_evolution_options(
            "test_postgres_refresh_evolution_options_for_queue",
            4,
            "pg-v-options",
            9,
            &mut evolution_options,
            Some(&metadata),
        );
        if changed {
            clear_queue_evolution_samples_after_postgres_refresh(
                "test_postgres_refresh_evolution_option_samples_reset_for_queue",
                4,
                "pg-v-options",
                &mut tactical_samples,
                &mut policy_samples,
                true,
                true,
            );
        }

        assert!(changed);
        assert!(tactical_samples.is_empty());
        assert!(policy_samples.is_empty());
        assert_eq!(evolution_options.population_size, 18);
        assert_eq!(evolution_options.seed, 3201);
    }

    #[test]
    fn queue_new_sims_apply_postgres_weights_before_each_enqueued_game() {
        let options = SoccerQPolicyOptions::default();
        let mut active_weights = SoccerTacticalLearningWeights::default();
        let mut completed_weights = Vec::new();
        let report = run_soccer_learning_queue_with_events(
            SoccerLearningQueueRunnerConfig {
                games: 3,
                parallel_games: 3,
                base_seed: 2089,
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
                        let mut postgres_weights = SoccerTacticalLearningWeights::default();
                        postgres_weights.attack_flank_lane_weight =
                            1.05 + next_episode as f64 * 0.11;
                        postgres_weights.defense_contract_delta_weight =
                            0.82 + next_episode as f64 * 0.07;
                        postgres_weights.formation_lp_alignment_weight =
                            0.18 + next_episode as f64 * 0.05;
                        maybe_apply_postgres_tactical_learning(
                            "test_postgres_refresh_tactical_learning_for_queue",
                            next_episode + 1,
                            &format!("pg-v-{next_episode}"),
                            next_episode as i32,
                            match_config,
                            &mut active_weights,
                            Some(postgres_weights),
                        )?;
                    }
                    SoccerLearningQueueEvent::CompletedGame { game, .. } => {
                        completed_weights.push((
                            game.episode,
                            game.starting_tactical_learning.attack_flank_lane_weight,
                            game.starting_tactical_learning
                                .defense_contract_delta_weight,
                            game.starting_tactical_learning
                                .formation_lp_alignment_weight,
                        ));
                    }
                }
                Ok(())
            },
        )
        .expect("queue run");

        assert_eq!(report.completed_games, 3);
        assert_eq!(report.failed_games, 0);
        completed_weights.sort_by_key(|(episode, _, _, _)| *episode);
        assert_eq!(completed_weights.len(), 3);
        for (episode, attack_flank, defense_contract, lp_alignment) in completed_weights {
            assert!((attack_flank - (1.05 + episode as f64 * 0.11)).abs() < 1e-12);
            assert!((defense_contract - (0.82 + episode as f64 * 0.07)).abs() < 1e-12);
            assert!((lp_alignment - (0.18 + episode as f64 * 0.05)).abs() < 1e-12);
        }
    }

    #[test]
    fn queue_tactical_samples_carry_episode_starting_weight_genomes() {
        let base = SoccerTacticalLearningWeights::default();
        let mut pg_weights = base.clone();
        pg_weights.attack_flank_lane_weight = 1.30;
        pg_weights.attack_width_delta_weight = 1.05;
        pg_weights.defense_contract_delta_weight = 1.40;
        pg_weights.defense_compactness_score_weight = 0.95;
        pg_weights.formation_lp_alignment_weight = 0.33;
        let mut episode_weights = HashMap::new();
        episode_weights.insert(3, pg_weights.clone());
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
        let sample = TacticalEvolutionSample {
            summary,
            weights: episode_weights.remove(&3).unwrap_or_else(|| base.clone()),
            fitness: 9.0,
        };
        let options = SoccerEvolutionOptions {
            mutation_rate: 0.0,
            mutation_scale: 0.0,
            crossover_rate: 1.0,
            exploration_rate: 0.0,
            exploration_scale: 0.0,
            elite_weight_floor: 0.0,
            population_size: 4,
            seed: 41,
        };

        let evolved = evolve_soccer_tactical_learning_weights_from_genomes(
            &base,
            &[SoccerTacticalLearningGenomeParent {
                summary: &sample.summary,
                weights: &sample.weights,
                fitness: sample.fitness,
            }],
            options,
        );

        assert!(evolved.attack_flank_lane_weight >= pg_weights.attack_flank_lane_weight);
        assert!(evolved.defense_contract_delta_weight >= pg_weights.defense_contract_delta_weight);
    }

    #[test]
    fn queue_tactical_learning_match_detects_formation_lp_alignment_weight() {
        let left = SoccerTacticalLearningWeights::default();
        let mut right = left.clone();
        right.formation_lp_alignment_weight += 0.05;

        assert!(!tactical_learning_weights_match(&left, &right));
    }

    #[test]
    fn queue_postgres_batches_only_coalesce_for_same_run() {
        let mut batch = empty_pg_batch("experiment-a", "runner-a");
        let mut different_retention = empty_pg_batch("experiment-a", "runner-a");
        different_retention.completed_run_retention_games = 10;

        assert!(batch.can_absorb(&empty_pg_batch("experiment-a", "runner-a")));
        assert!(!batch.can_absorb(&empty_pg_batch("experiment-b", "runner-a")));
        assert!(!batch.can_absorb(&empty_pg_batch("experiment-a", "runner-b")));
        assert!(!batch.can_absorb(&different_retention));

        batch.absorb(empty_pg_batch("experiment-a", "runner-a"));
        assert!(batch.pending_policy_versions.is_empty());
        assert!(batch.pending_runs.is_empty());
    }

    #[test]
    fn queue_postgres_batches_track_policy_version_sources() {
        let mut batch = empty_pg_batch("experiment-a", "runner-a");
        let mut other = empty_pg_batch("experiment-a", "runner-a");
        batch.policy_version_batches = 1;
        other.policy_version_batches = 1;

        batch.absorb(other);

        assert_eq!(batch.policy_version_batches, 2);
    }
}
