//! Run accelerated soccer self-play and persist learned MDP/POMDP policies.

use std::collections::{BTreeMap, VecDeque};
use std::error::Error;
use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Error as IoError, ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;
use soccer_engine::des::general::soccer::{
    soccer_moment_records_from_jsonl, soccer_moment_records_to_learning_dataset,
    train_soccer_pass_completion_head, AttackSpacingHead, BackFourLineHead, GiveAndGoHead,
    CrashBoxHead, GoalSideRecoveryHead, HeadScanHead, LaneAffinityHead, LongPassRunHead,
    LooseBallCommitHead, MatchConfig, MatchSummary, PassLaneYieldHead, RunPredictionHead,
    SeparationFloorHead, WingerPinchHead,
    ReceiveApproachHead, ShotTriggerHead, SoccerConfigMomentInsert, SoccerMarlAlgorithm,
    SoccerMatch, SoccerMomentWindow, SoccerNeuralLearningBackend, SoccerNeuralLearningConfig,
    SoccerNeuralNetworkSnapshot, SoccerPassCompletionHead, SoccerPassLearningMetrics,
    SoccerPassOutcomeSample, SoccerQEntry, SoccerQPolicy, SoccerQPolicyOptions, SoccerQTargetEntry,
    SoccerSelfPlayEpisodeSummary, SoccerSelfPlayLearnedParams, SoccerSelfPlayTrainingArtifact,
    SoccerTacticalLearningSummary, SoccerTacticalLearningWeights, SoccerTeamPolicyArtifact,
    SoccerTeamQPolicies, ATTACK_SPACING_HEAD_MIN_TRAINING_STEPS,
    DEFAULT_SOCCER_MAPPO_TEAM_REWARD_SHARE, DEFAULT_SOCCER_PASS_COMPLETION_LEARNING_RATE,
    CRASH_BOX_HEAD_MIN_TRAINING_STEPS, GIVE_AND_GO_HEAD_MIN_TRAINING_STEPS,
    GOAL_SIDE_RECOVERY_HEAD_MIN_TRAINING_STEPS, HEAD_SCAN_HEAD_MIN_TRAINING_STEPS,
    LANE_AFFINITY_HEAD_MIN_TRAINING_STEPS,
    PASS_LANE_YIELD_HEAD_MIN_TRAINING_STEPS, RUN_PREDICTION_HEAD_MIN_TRAINING_STEPS,
    SEPARATION_FLOOR_HEAD_MIN_TRAINING_STEPS, WINGER_PINCH_HEAD_MIN_TRAINING_STEPS,
    LONG_PASS_RUN_HEAD_MIN_TRAINING_STEPS, LOOSE_BALL_COMMIT_HEAD_MIN_TRAINING_STEPS,
    PASS_COMPLETION_HEAD_MIN_TRAINING_STEPS, RECEIVE_APPROACH_HEAD_MIN_TRAINING_STEPS,
    SHOT_TRIGGER_HEAD_MIN_TRAINING_STEPS,
};
use soccer_engine::des::soccer_learning::{
    evaluate_soccer_policy_promotion_gate, evolve_soccer_tactical_learning_weights_from_genomes,
    evolve_soccer_team_policies, merge_soccer_policy_deltas,
    soccer_evolution_options_from_search_metadata, soccer_learning_curriculum_episode_config,
    soccer_learning_curriculum_stage_for_completed_games, soccer_learning_run_score,
    soccer_neural_network_snapshot_fingerprint, soccer_policy_delta_entries,
    soccer_policy_version_insert_status_after_active_head, soccer_postgres_new_sim_refresh_plan,
    soccer_postgres_policy_refresh_decision,
    soccer_should_flush_postgres_policy_versions_for_new_sim,
    soccer_should_refresh_postgres_for_new_sim, soccer_tactical_learning_weights_fingerprint,
    soccer_team_q_policies_fingerprint, validate_soccer_evolution_options_for_learning_run,
    validate_soccer_learning_curriculum_config_for_learning_run,
    validate_soccer_neural_learning_config_for_learning_run,
    validate_soccer_policy_promotion_gate_config_for_learning_run, SoccerEvolutionOptions,
    SoccerLearningCompletedGame, SoccerLearningCurriculumConfig, SoccerPolicyPromotionGateConfig,
    SoccerPolicyPromotionGateEvaluation, SoccerPostgresPolicyRefreshCheck,
    SoccerTacticalLearningGenomeParent, SOCCER_POLICY_ACTIVE_MAX_FITNESS_REGRESSION,
    SOCCER_POLICY_STATUS_ACTIVE, SOCCER_POLICY_STATUS_ARCHIVED,
};
use soccer_engine::des::soccer_learning_pg::{
    soccer_learning_git_commit, SoccerLearningPgCompletedRunInsert, SoccerLearningPgStore,
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
const DEFAULT_SOCCER_CONTINUOUS_PARALLEL_GAMES: usize = 1;
const DEFAULT_SOCCER_WRITE_GAME_ARTIFACTS: bool = false;
const DEFAULT_SOCCER_WRITE_FINAL_POLICY_ARTIFACT: bool = true;
const DEFAULT_SOCCER_WRITE_EPISODE_LOG: bool = true;
const DEFAULT_SOCCER_PRINT_COMPLETED_GAMES: bool = false;
const DEFAULT_SOCCER_EVOLUTION_ENABLED: bool = true;
const DEFAULT_SOCCER_EVOLUTION_INTERVAL_GAMES: usize = 10;
const DEFAULT_SOCCER_EVOLUTION_ELITE_GAMES: usize = 4;
const SOCCER_LEARNING_LOCAL_MPC_MAX_PLAYERS_PER_TEAM_LIMIT: usize = 11;
const SOCCER_POLICY_SOURCE_MERGE: &str = "merge";
const SOCCER_POLICY_SOURCE_EVOLUTION: &str = "mutation";

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

fn env_u32(name: &str, default: u32) -> Result<u32, Box<dyn Error>> {
    let Some(value) = env_value(name) else {
        return Ok(default);
    };
    value
        .parse::<u32>()
        .map_err(|_| invalid_data(format!("{name} must be a u32, got {value:?}")).into())
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
    local_tactical_evolved_since_pg_refresh: &mut bool,
    postgres_tactical_learning_authoritative: bool,
    pg_policy_version_buffer_len: usize,
    pending_async_pg_batches: usize,
    pending_async_policy_version_batches: usize,
) -> Result<bool, Box<dyn Error>> {
    let Some(metadata) = store
        .load_latest_active_policy_metadata(experiment_id)
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
            .load_latest_active_policy_with_min_visits(
                experiment_id,
                options.clone(),
                options.clone(),
                resume_min_visits(),
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
                "postgres_refresh_policy_for_game next_episode={} policy_version={} previous_policy_version={} generation={} neural_network={} pending_policy_versions={} pending_async_batches={} pending_async_policy_version_batches={}",
                next_episode,
                version.id,
                pg_base_policy_version_id.as_deref().unwrap_or("none"),
                version.generation,
                version.neural_network.is_some(),
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
            *local_tactical_evolved_since_pg_refresh = false;
            changed = true;
        }
    }
    Ok(changed)
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
    elapsed_seconds: f64,
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

/// In-memory back-four line-depth head, carried + trained across games WITHIN a
/// learner process (parallel_games=1), so it accumulates and the back-four line
/// consumes it live once trained. Resets on pod restart — cross-restart Postgres
/// persistence (round-trip via SoccerNeuralNetworkSnapshot) is the remaining
/// durability step. Untouched unless a line-depth model is enabled.
static CARRIED_LINE_DEPTH_HEAD: std::sync::Mutex<Option<BackFourLineHead>> =
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

/// In-memory shot-trigger value head (when to pull the trigger on a shot), carried +
/// trained across games WITHIN a learner process, then installed into the next game so
/// the shot decision consumes what was learned. Mirrors `CARRIED_LINE_DEPTH_HEAD`.
static CARRIED_SHOT_TRIGGER_HEAD: std::sync::Mutex<Option<ShotTriggerHead>> =
    std::sync::Mutex::new(None);

fn run_game(
    episode: usize,
    config: MatchConfig,
    starting_policies: Arc<SoccerTeamQPolicies>,
    starting_policy_version_id: Option<String>,
    starting_policy_generation: i32,
    initial_neural_network: Option<Arc<SoccerNeuralNetworkSnapshot>>,
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
        if let Err(err) = sim.set_neural_network_snapshot((**snapshot).clone()) {
            if err.contains("input_dim") && err.contains("does not match expected") {
                eprintln!(
                    "soccer warm-start: dropping incompatible neural snapshot, starting net cold ({err})"
                );
            } else {
                return Err(err);
            }
        }
    }
    for window in adversarial_moment_windows.iter() {
        sim.remember_adversarial_moment_window(window.clone());
    }
    // Gap 5 step 3b: install the carried line-depth head so the back-four line
    // consumes it live once trained. No-op unless a line-depth model is enabled (the
    // head only accumulates while collection is on).
    if let Some(head) = CARRIED_LINE_DEPTH_HEAD.lock().unwrap().as_ref() {
        sim.set_line_depth_head(head.clone());
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
        eprintln!(
            "line_depth_training samples={} training_steps={} consumed={} final_loss={:.5}",
            line_depth_samples.len(),
            head.training_steps(),
            head.training_steps()
                >= soccer_engine::des::general::soccer::LINE_DEPTH_HEAD_MIN_TRAINING_STEPS,
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
        pass_outcome_samples: game.pass_outcome_samples.clone(),
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
            latest_active_metadata.as_ref().map(|metadata| {
                soccer_engine::des::soccer_learning::soccer_learning_from_micros(
                    metadata.fitness_micros,
                )
            }),
            SOCCER_POLICY_ACTIVE_MAX_FITNESS_REGRESSION,
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

fn soccer_learning_pg_version_label(run_id: &str, shard_index: usize, episode: usize) -> String {
    let suffix = format!("-s{:03}-e{:06}", shard_index, episode + 1);
    let max_prefix_len = 160usize.saturating_sub(suffix.len());
    let mut prefix = run_id
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | ':' | '/' | '-'))
        .take(max_prefix_len)
        .collect::<String>();
    if prefix.is_empty() {
        prefix.push_str("run");
    }
    format!("{prefix}{suffix}")
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
    let learning_interval_ticks = env_usize("SOCCER_LEARNING_INTERVAL_TICKS", 4)?;
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
    let neural_drain_timeout_ms = env_usize(
        "SOCCER_NEURAL_DRAIN_TIMEOUT_MS",
        DEFAULT_SOCCER_NEURAL_DRAIN_TIMEOUT_MS,
    )?;
    let neural_drain_timeout =
        Duration::from_millis(neural_drain_timeout_ms.min(u64::MAX as usize) as u64);
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
    apply_env_mpc_config(&mut config, &default_config)?;
    // The neural ACTOR (and MAPPO, which is itself gated on `actor_critic`) was previously left
    // disabled in standard learning runs: `neural_blend` came straight from `MatchConfig::default()`
    // where `actor_critic = false`, so a run that asked for Mappo/IndependentActorCritic still only
    // trained the value head. Tie the actor to the requested MARL algorithm — on whenever an actor
    // is wanted (anything but `Off`) — with an explicit `SOCCER_ENABLE_ACTOR_CRITIC` override either way.
    let actor_critic_default = config.neural_learning.marl_algorithm != SoccerMarlAlgorithm::Off;
    config.neural_blend.actor_critic =
        env_bool("SOCCER_ENABLE_ACTOR_CRITIC", actor_critic_default)?;
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
    let pg_refresh_for_new_sims = soccer_should_refresh_postgres_for_new_sim(
        resume_artifact.is_some(),
        pg_refresh_with_resume_artifact,
    );
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
    let mut pg_persisted_games = 0usize;
    let mut policies = load_initial_policies(resume_artifact.as_deref(), options.clone())?;
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
                .load_latest_active_policy_with_min_visits(
                    &experiment_id,
                    options.clone(),
                    options.clone(),
                    resume_min_visits(),
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
                let version_search_metadata = version.search_metadata.clone();
                println!(
                    "postgres_resume_policy experiment={} policy_version={} generation={} neural_network={}",
                    experiment_slug,
                    version.id,
                    version.generation,
                    version.neural_network.is_some()
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
        "curriculum ball_skills_after_games={} duels_after_games={} small_sided_after_games={} team_shape_after_games={} full_match_after_games={}",
        curriculum_config.ball_skills_after_games,
        curriculum_config.duels_after_games,
        curriculum_config.small_sided_after_games,
        curriculum_config.team_shape_after_games,
        curriculum_config.full_match_after_games
    );
    println!(
        "policy_promotion_gate enabled={} min_sample_games={} min_mean_match_fitness={:.4} min_best_match_fitness={:.4} min_mean_play_quality={:.4} max_mean_conceded_goals={:.4} max_mean_goal_margin={:.4} max_mean_chain_net_loss={:.4}",
        policy_promotion_gate.enabled,
        policy_promotion_gate.min_sample_games,
        policy_promotion_gate.min_mean_match_fitness,
        policy_promotion_gate.min_best_match_fitness,
        policy_promotion_gate.min_mean_play_quality,
        policy_promotion_gate.max_mean_conceded_goals,
        policy_promotion_gate.max_mean_goal_margin,
        policy_promotion_gate.max_mean_chain_net_loss
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
    println!("write_episode_log={write_episode_log_file}");
    if checkpoint_interval_games == 0 || !write_checkpoint_artifacts {
        println!("checkpoint_artifact=disabled");
    } else {
        println!("checkpoint_artifact={}", checkpoint_artifact_path.display());
        println!("checkpoint_interval_games={}", checkpoint_interval_games);
    }
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
        "neural_learning enabled={} backend={} learning_rate={:.5} batch_size={} train_every_ticks={} max_batches_per_tick={} hidden_units={} target_scale={:.3} max_pending_batches={} replay_capacity={} replay_samples_per_tick={} target_clip={:.3} snapshot_every_batches={}",
        neural_learning.enabled,
        neural_backend_label(neural_learning.backend),
        neural_learning.learning_rate,
        neural_learning.batch_size,
        neural_learning.train_every_ticks,
        neural_learning.max_batches_per_tick,
        neural_learning.hidden_units,
        neural_learning.target_scale,
        neural_learning.max_pending_batches,
        neural_learning.replay_capacity,
        neural_learning.replay_samples_per_tick,
        neural_learning.target_clip,
        neural_learning.snapshot_every_batches,
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
    let mut evolution_search_samples =
        VecDeque::<EvolutionSearchSample>::with_capacity(evolution_window_games);
    let mut local_tactical_evolved_since_pg_refresh = false;
    let mut pg_policy_version_buffer = Vec::new();
    let mut pg_completed_buffer = Vec::new();
    let pg_runner_id = soccer_learning_pg_runner_id(&run_id, shard_index, shard_count);
    let worker_pool = SoccerLearningWorkerPool::start(parallel_games);
    let retain_neural_network_in_game_artifact =
        write_game_artifacts && game_artifact_mode.as_str() == "full";
    let adversarial_moment_windows = Arc::new(adversarial_moment_windows);

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
                            &mut local_tactical_evolved_since_pg_refresh,
                            pg_tactical_learning_authoritative,
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
            worker_pool.submit(SoccerLearningWorkerTask {
                episode,
                config: episode_config,
                starting_policies: episode_starting_policies,
                starting_policy_version_id: episode_starting_policy_version_id,
                starting_policy_generation: episode_starting_policy_generation,
                initial_neural_network: episode_starting_neural_network.as_ref().map(Arc::clone),
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
        for game in completed_games.iter_mut() {
            if let Some(snapshot) = game.neural_network.take() {
                latest_neural_network = Some(snapshot);
            }
            let completed_learning_game = soccer_learning_completed_game_from_completed(game);
            if merge_deltas {
                merge_team_policy_delta(
                    &mut policies,
                    game.starting_policies.as_ref(),
                    &game.policies,
                );
            } else {
                policies = game.policies.clone();
            }
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
                let policy_promotion_evaluation = evaluate_soccer_policy_promotion_gate(
                    [&game.episode_summary.summary],
                    policy_promotion_gate,
                );
                let policy_version_status =
                    policy_version_status_for_promotion_gate(&policy_promotion_evaluation);
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
                let output_policy_version_id = if should_write_policy_version {
                    let next_generation = pg_generation
                        .max(game.starting_policy_generation)
                        .saturating_add(1);
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
                        status: policy_version_status,
                        config: config.clone(),
                        home_options: options.clone(),
                        away_options: options.clone(),
                        policies: policies.clone(),
                        fitness: completed_learning_game.score.match_fitness,
                        neural_network: latest_neural_network.clone(),
                        search_metadata: policy_promotion_gate.enabled.then(|| {
                            serde_json::json!({
                                "promotion": policy_promotion_search_metadata(
                                    curriculum_stage_label,
                                    &policy_promotion_evaluation,
                                )
                            })
                        }),
                    });
                    if policy_version_status == SOCCER_POLICY_STATUS_ACTIVE {
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
                    }
                    Some(output_policy_version_id)
                } else {
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
            let policy_promotion_evaluation = evaluate_soccer_policy_promotion_gate(
                evolution_search_samples
                    .iter()
                    .map(|sample| &sample.summary),
                policy_promotion_gate,
            );
            let candidate_promoted = policy_promotion_evaluation.eligible;
            let curriculum_stage = soccer_learning_curriculum_stage_for_completed_games(
                completed_after_batch,
                &curriculum_config,
            );
            let curriculum_stage_label = curriculum_stage.as_str();
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
            if candidate_promoted {
                policies = evolved_policies.clone();
                tactical_learning = evolved_tactical_learning.clone();
                config.tactical_learning = tactical_learning.clone();
                local_tactical_evolved_since_pg_refresh = true;
                println!(
                    "policy_evolved games_completed={} curriculum_stage={} sample_window_games={} samples={} elite_games={} best_fitness={:.4} mutation_rate={:.4} mutation_scale={:.4} crossover_rate={:.4} exploration_rate={:.4} exploration_scale={:.4} population_size={}",
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
                    batch_evolution_options.population_size
                );
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
                let next_generation = pg_generation.saturating_add(1);
                let output_policy_version_id = Uuid::new_v4().to_string();
                let version_label = format!(
                    "{}-evolution",
                    soccer_learning_pg_version_label(
                        &run_id,
                        shard_index,
                        completed_after_batch.saturating_sub(1),
                    )
                );
                let policy_version_status =
                    policy_version_status_for_promotion_gate(&policy_promotion_evaluation);
                pg_policy_version_buffer.push(PendingPostgresPolicyVersion {
                    id: output_policy_version_id.clone(),
                    parent_policy_version_id: pg_base_policy_version_id.clone(),
                    generation: next_generation,
                    version_label,
                    source_kind: SOCCER_POLICY_SOURCE_EVOLUTION,
                    status: policy_version_status,
                    config: evolved_config,
                    home_options: options.clone(),
                    away_options: options.clone(),
                    policies: evolved_policies.clone(),
                    fitness: best_fitness,
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
            write_json(&checkpoint_artifact_path, &checkpoint_export)?;
            let checkpoint_params =
                SoccerSelfPlayLearnedParams::from_training_artifact_with_neural_network(
                    &checkpoint_artifact,
                    latest_neural_network.clone(),
                );
            write_json(&learned_params_path, &checkpoint_params)?;
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
                manifest_games.clone(),
            );
            write_json(&manifest_path, &checkpoint_manifest)?;
            last_checkpoint_episode = next_episode;
            println!(
                "checkpoint games_completed={} artifact={} manifest={}",
                episode_summaries.len(),
                checkpoint_artifact_path.display(),
                manifest_path.display()
            );
            let _ = std::io::stdout().flush();
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
        } else {
            println!("final_policy_artifact=disabled");
        }
        let learned_params =
            SoccerSelfPlayLearnedParams::from_training_artifact_with_neural_network(
                &artifact,
                latest_neural_network.clone(),
            );
        write_json(&learned_params_path, &learned_params)?;

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
            elapsed_seconds: 0.0,
        }
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
            policy_head: None,
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
        let label = format!(
            "{}-evolution",
            soccer_learning_pg_version_label("run-a", 2, 4)
        );

        assert_eq!(SOCCER_POLICY_SOURCE_MERGE, "merge");
        assert_eq!(SOCCER_POLICY_SOURCE_EVOLUTION, "mutation");
        assert!(label.ends_with("-evolution"));
        assert!(!label.ends_with("-mutation"));
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
}
