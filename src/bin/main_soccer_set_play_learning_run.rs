use std::env;
use std::error::Error;
use std::fs::{self, File};
use std::io::{Error as IoError, ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;
use soccer_engine::des::general::soccer::{
    train_soccer_set_play_restarts_with_events, MatchConfig, SoccerMarlAlgorithm,
    SoccerNeuralBlendConfig, SoccerNeuralBlendMode, SoccerNeuralLearningBackend,
    SoccerNeuralLearningConfig, SoccerNeuralNetworkSnapshot, SoccerQPolicyOptions,
    SoccerSetPlayRestartKind, SoccerSetPlayTrainingEvent, SoccerSetPlayTrainingRequest,
    SoccerTeamQPolicies, Team, Vec2, DEFAULT_SOCCER_MAPPO_TEAM_REWARD_SHARE,
};
use soccer_engine::des::soccer_learning::{
    soccer_neural_network_snapshot_fingerprint,
    soccer_policy_version_insert_status_after_active_head, soccer_postgres_policy_refresh_decision,
    soccer_tactical_learning_weights_fingerprint, soccer_team_q_policies_fingerprint,
    validate_soccer_neural_learning_config_for_learning_run,
    validate_soccer_q_policy_options_for_learning_run, SoccerPostgresPolicyRefreshCheck,
    SOCCER_POLICY_STATUS_ACTIVE,
};
use soccer_engine::des::soccer_learning_pg::SoccerLearningPgStore;

const DEFAULT_SOCCER_SET_PLAY_POSTGRES_TACTICAL_LEARNING_AUTHORITATIVE: bool = true;

fn env_value(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn env_bool(name: &str, default: bool) -> Result<bool, Box<dyn Error>> {
    let Some(value) = env_value(name) else {
        return Ok(default);
    };
    match value.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "y" | "on" => Ok(true),
        "0" | "false" | "no" | "n" | "off" => Ok(false),
        _ => Err(format!("{name} must be boolean, got {value:?}").into()),
    }
}

fn env_parse<T>(name: &str, default: T) -> Result<T, Box<dyn Error>>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let Some(value) = env_value(name) else {
        return Ok(default);
    };
    value
        .parse::<T>()
        .map_err(|err| format!("{name}={value:?} is invalid: {err}").into())
}

fn env_neural_backend() -> Result<SoccerNeuralLearningBackend, Box<dyn Error>> {
    let Some(value) = env_value("SOCCER_NEURAL_LEARNING_BACKEND") else {
        return Ok(SoccerNeuralLearningBackend::Threaded);
    };
    match value.to_ascii_lowercase().as_str() {
        "threaded" | "thread" | "worker" => Ok(SoccerNeuralLearningBackend::Threaded),
        "inline" | "sync" => Ok(SoccerNeuralLearningBackend::Inline),
        _ => Err(format!("SOCCER_NEURAL_LEARNING_BACKEND={value:?} is invalid").into()),
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
        _ => Err(format!(
            "SOCCER_MARL_ALGORITHM must be off, independentActorCritic, or mappo, got {value:?}"
        )
        .into()),
    }
}

/// Decision-time coupling of the trained value head, from the environment. The
/// set-piece curriculum is where it pays off: short restarts give many dense reps
/// to warm the head before its value drives play. The specialist actor is on by
/// default so passing/dribbling/shooting/GK heads also get dense set-play reps.
///   SOCCER_NEURAL_BLEND_MODE = off | additive | tiebreak | confidence
///   SOCCER_NEURAL_BLEND_LAMBDA, SOCCER_NEURAL_BLEND_WARMUP_STEPS (optional)
fn env_neural_blend() -> Result<SoccerNeuralBlendConfig, Box<dyn Error>> {
    let mut blend = SoccerNeuralBlendConfig::default();
    blend.actor_critic = true;
    if let Some(value) = env_value("SOCCER_NEURAL_BLEND_MODE") {
        blend.mode = match value.to_ascii_lowercase().as_str() {
            "off" | "none" => SoccerNeuralBlendMode::Off,
            "additive" | "add" => SoccerNeuralBlendMode::Additive,
            "tiebreak" | "tie" => SoccerNeuralBlendMode::TieBreak,
            "confidence" | "confidencegated" | "gated" => SoccerNeuralBlendMode::ConfidenceGated,
            _ => return Err(format!("SOCCER_NEURAL_BLEND_MODE={value:?} is invalid").into()),
        };
    }
    blend.lambda = env_parse("SOCCER_NEURAL_BLEND_LAMBDA", blend.lambda)?;
    blend.warmup_steps = env_parse("SOCCER_NEURAL_BLEND_WARMUP_STEPS", blend.warmup_steps)?;
    // Train + consult the neural actor π(family|s) by advantage policy-gradient.
    blend.actor_critic = env_bool("SOCCER_NEURAL_ACTOR_CRITIC", blend.actor_critic)?;
    // Train the learned world model P̂(s'|s,a) on the episode replay.
    blend.world_model = env_bool("SOCCER_NEURAL_WORLD_MODEL", blend.world_model)?;
    Ok(blend)
}

fn env_team() -> Result<Team, Box<dyn Error>> {
    let Some(value) = env_value("SOCCER_SET_PLAY_TEAM") else {
        return Ok(Team::Home);
    };
    match value.to_ascii_lowercase().as_str() {
        "home" => Ok(Team::Home),
        "away" => Ok(Team::Away),
        _ => Err(format!("SOCCER_SET_PLAY_TEAM={value:?} is invalid").into()),
    }
}

fn parse_restart_token(value: &str) -> Result<SoccerSetPlayRestartKind, Box<dyn Error>> {
    match value.to_ascii_lowercase().replace('_', "-").as_str() {
        "direct-free-kick" | "direct-freekick" | "dfk" | "free-kick" | "freekick" | "fk" => {
            Ok(SoccerSetPlayRestartKind::DirectFreeKick)
        }
        "indirect-free-kick" | "indirect-freekick" | "ifk" => {
            Ok(SoccerSetPlayRestartKind::IndirectFreeKick)
        }
        _ => Err(format!("restart {value:?} is invalid").into()),
    }
}

fn env_restarts() -> Result<Vec<SoccerSetPlayRestartKind>, Box<dyn Error>> {
    let raw = env_value("SOCCER_SET_PLAY_RESTARTS")
        .or_else(|| env_value("SOCCER_SET_PLAY_RESTART"))
        .unwrap_or_else(|| "indirect-free-kick,direct-free-kick".to_string());
    let normalized = raw.to_ascii_lowercase().replace('_', "-");
    if matches!(
        normalized.as_str(),
        "both" | "all" | "free-kicks" | "free-kick"
    ) {
        return Ok(vec![
            SoccerSetPlayRestartKind::IndirectFreeKick,
            SoccerSetPlayRestartKind::DirectFreeKick,
        ]);
    }
    let mut restarts = Vec::new();
    for token in raw
        .split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
    {
        let restart = parse_restart_token(token)?;
        if !restarts.contains(&restart) {
            restarts.push(restart);
        }
    }
    if restarts.is_empty() {
        return Err("SOCCER_SET_PLAY_RESTARTS must name at least one restart".into());
    }
    Ok(restarts)
}

fn run_id_default() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("free-kick-{now}")
}

fn version_label_from_run_id(run_id: &str) -> String {
    let mut label = run_id
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | ':' | '/' | '-') {
                ch
            } else {
                '-'
            }
        })
        .collect::<String>();
    if label.is_empty() {
        label = run_id_default();
    }
    label.truncate(120);
    format!("{label}-set-play")
}

fn free_kick_spot(config: &MatchConfig, team: Team, distance_yards: f64) -> Vec2 {
    let width = config.field_width_yards.max(1.0);
    let length = config.field_length_yards.max(1.0);
    let distance_yards = if distance_yards.is_finite() && distance_yards > 0.0 {
        distance_yards
    } else {
        25.0
    };
    Vec2::new(
        width * 0.5,
        team.goal_y(length) - team.attack_dir() * distance_yards,
    )
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let temp_path = path.with_extension(format!(
        "{}.tmp",
        path.extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("json")
    ));
    {
        let mut file = File::create(&temp_path)?;
        serde_json::to_writer_pretty(&mut file, value)?;
        file.write_all(b"\n")?;
        file.flush()?;
        file.sync_all()?;
    }
    fs::rename(&temp_path, path)?;
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        let _ = File::open(parent).and_then(|dir| dir.sync_all());
    }
    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("main_soccer_set_play_learning_run: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let started = Instant::now();
    let run_id = env_value("SOCCER_SET_PLAY_RUN_ID")
        .or_else(|| env_value("SOCCER_RUN_ID"))
        .unwrap_or_else(run_id_default);
    let run_dir = env_value("SOCCER_SET_PLAY_RUN_DIR")
        .or_else(|| env_value("SOCCER_RUN_DIR"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("out/soccer-set-play-learning").join(&run_id));
    let artifact_path = env_value("SOCCER_SET_PLAY_ARTIFACT_PATH")
        .or_else(|| env_value("SOCCER_ARTIFACT_PATH"))
        .map(PathBuf::from)
        .unwrap_or_else(|| run_dir.join("artifact.json"));

    let episodes = env_parse("SOCCER_SET_PLAY_EPISODES", 100usize)?;
    if episodes == 0 {
        return Err("SOCCER_SET_PLAY_EPISODES must be at least 1".into());
    }
    let duration_seconds = env_parse("SOCCER_SET_PLAY_DURATION_SECONDS", 10.0_f64)?;
    if !duration_seconds.is_finite() || duration_seconds <= 0.0 {
        return Err("SOCCER_SET_PLAY_DURATION_SECONDS must be positive and finite".into());
    }
    let distance_yards = env_parse("SOCCER_FREE_KICK_DISTANCE_YARDS", 25.0_f64)?;
    let team = env_team()?;
    let restarts = env_restarts()?;
    let restart = restarts[0];
    let pg_tactical_learning_authoritative = env_bool(
        "SOCCER_SET_PLAY_POSTGRES_TACTICAL_LEARNING_AUTHORITATIVE",
        env_bool(
            "SOCCER_POSTGRES_TACTICAL_LEARNING_AUTHORITATIVE",
            DEFAULT_SOCCER_SET_PLAY_POSTGRES_TACTICAL_LEARNING_AUTHORITATIVE,
        )?,
    )?;
    let mut config = MatchConfig::default();
    config.duration_seconds = duration_seconds;
    config.dt_seconds = env_parse("SOCCER_DT_SECONDS", config.dt_seconds)?;
    if !config.dt_seconds.is_finite() || config.dt_seconds <= 0.0 {
        return Err("SOCCER_DT_SECONDS must be positive and finite".into());
    }
    config.seed = env_parse("SOCCER_SEED", config.seed)?;
    config.learning_interval_ticks = env_parse(
        "SOCCER_LEARNING_INTERVAL_TICKS",
        config.learning_interval_ticks,
    )?
    .max(1);
    config.learning_enabled = true;
    config.learning_logging_enabled = false;
    config.max_human_players = 0;
    config.opponent_belief_enabled = env_bool(
        "SOCCER_OPPONENT_BELIEF_ENABLED",
        env_bool("SOCCER_OPPONENT_BELIEF", true)?,
    )?;
    config.neural_learning = SoccerNeuralLearningConfig {
        enabled: env_bool("SOCCER_NEURAL_LEARNING_ENABLED", true)?,
        backend: env_neural_backend()?,
        learning_rate: env_parse(
            "SOCCER_NEURAL_LEARNING_RATE",
            SoccerNeuralLearningConfig::default().learning_rate,
        )?,
        batch_size: env_parse("SOCCER_NEURAL_BATCH_SIZE", 32usize)?,
        train_every_ticks: env_parse("SOCCER_NEURAL_TRAIN_EVERY_TICKS", 1usize)?,
        max_batches_per_tick: env_parse("SOCCER_NEURAL_MAX_BATCHES_PER_TICK", 1usize)?,
        hidden_units: env_parse("SOCCER_NEURAL_HIDDEN_UNITS", 24usize)?,
        target_scale: env_parse(
            "SOCCER_NEURAL_TARGET_SCALE",
            SoccerNeuralLearningConfig::default().target_scale,
        )?,
        max_pending_batches: env_parse("SOCCER_NEURAL_MAX_PENDING_BATCHES", 32usize)?,
        replay_capacity: env_parse("SOCCER_NEURAL_REPLAY_CAPACITY", 1024usize)?,
        replay_samples_per_tick: env_parse("SOCCER_NEURAL_REPLAY_SAMPLES_PER_TICK", 16usize)?,
        target_clip: env_parse(
            "SOCCER_NEURAL_TARGET_CLIP",
            SoccerNeuralLearningConfig::default().target_clip,
        )?,
        snapshot_every_batches: env_parse(
            "SOCCER_NEURAL_SNAPSHOT_EVERY_BATCHES",
            SoccerNeuralLearningConfig::default().snapshot_every_batches,
        )?,
        critic_baseline_weight: env_parse(
            "SOCCER_NEURAL_CRITIC_BASELINE_WEIGHT",
            SoccerNeuralLearningConfig::default().critic_baseline_weight,
        )?,
        lp_coupling_enabled: env_bool(
            "SOCCER_NEURAL_LP_COUPLING_ENABLED",
            SoccerNeuralLearningConfig::default().lp_coupling_enabled,
        )?,
        marl_algorithm: env_marl_algorithm(SoccerNeuralLearningConfig::default().marl_algorithm)?,
        marl_team_reward_weight: env_parse(
            "SOCCER_MARL_TEAM_REWARD_WEIGHT",
            SoccerNeuralLearningConfig::default().marl_team_reward_weight,
        )?,
        marl_intermediate_reward_weight: env_parse(
            "SOCCER_MARL_INTERMEDIATE_REWARD_WEIGHT",
            SoccerNeuralLearningConfig::default().marl_intermediate_reward_weight,
        )?,
        mappo_clip_epsilon: env_parse(
            "SOCCER_MAPPO_CLIP_EPSILON",
            SoccerNeuralLearningConfig::default().mappo_clip_epsilon,
        )?,
        mappo_team_reward_share: env_parse(
            "SOCCER_MAPPO_TEAM_REWARD_SHARE",
            DEFAULT_SOCCER_MAPPO_TEAM_REWARD_SHARE,
        )?,
    };
    validate_soccer_neural_learning_config_for_learning_run(&config.neural_learning)
        .map_err(|err| IoError::new(ErrorKind::InvalidData, err))?;
    let options = SoccerQPolicyOptions {
        alpha: env_parse("SOCCER_Q_ALPHA", SoccerQPolicyOptions::default().alpha)?,
        gamma: env_parse("SOCCER_Q_GAMMA", SoccerQPolicyOptions::default().gamma)?,
        exploration_epsilon: env_parse(
            "SOCCER_EXPLORATION_EPSILON",
            SoccerQPolicyOptions::default().exploration_epsilon,
        )?,
    };
    validate_soccer_q_policy_options_for_learning_run(&options)
        .map_err(|err| IoError::new(ErrorKind::InvalidData, err))?;

    let mut pg_store = SoccerLearningPgStore::connect_from_env()?;
    let mut pg_experiment_id = None::<String>;
    let mut pg_base_policy_version_id = None::<String>;
    let mut pg_base_generation = None::<i32>;
    let mut pg_base_policy_version_updated_at_micros = 0i64;
    let mut pg_base_policy_fingerprint = None::<u64>;
    let mut pg_base_neural_network_fingerprint = None::<u64>;
    let mut pg_base_tactical_learning_fingerprint = None::<u64>;
    let mut postgres_resume_policy = false;
    let mut initial_neural_network = None::<SoccerNeuralNetworkSnapshot>;
    let initial_policies = if let Some(store) = pg_store.as_mut() {
        let slug = env_value("SOCCER_EXPERIMENT_SLUG")
            .unwrap_or_else(|| "soccer-free-kick-restarts".to_string());
        let display_name = env_value("SOCCER_EXPERIMENT_NAME")
            .unwrap_or_else(|| "Soccer free-kick restart learning".to_string());
        let experiment_id = store.ensure_experiment(&slug, &display_name, &config)?;
        let resume = env_bool("SOCCER_RESUME_POSTGRES_POLICY", true)?;
        postgres_resume_policy = resume;
        let latest = if resume {
            store.load_latest_active_policy(&experiment_id, options.clone(), options.clone())?
        } else {
            None
        };
        if let Some(version) = latest {
            let policy_version_id = version.id.clone();
            let generation = version.generation;
            if let Some(tactical_learning) = version.tactical_learning.clone() {
                config.tactical_learning = tactical_learning;
                println!(
                    "postgres_resume_tactical_learning experiment={} policy_version={} generation={} attack_flank_lane={:.3} defense_contract_delta={:.3}",
                    experiment_id,
                    policy_version_id,
                    generation,
                    config.tactical_learning.attack_flank_lane_weight,
                    config.tactical_learning.defense_contract_delta_weight
                );
            }
            println!(
                "postgres_resume_policy experiment={} policy_version={} generation={} neural_network={}",
                experiment_id,
                policy_version_id,
                generation,
                version.neural_network.is_some()
            );
            pg_base_neural_network_fingerprint = version
                .neural_network
                .as_ref()
                .map(soccer_neural_network_snapshot_fingerprint);
            pg_base_policy_fingerprint = version
                .policy_fingerprint
                .or_else(|| Some(soccer_team_q_policies_fingerprint(&version.policies)));
            pg_base_tactical_learning_fingerprint = version
                .tactical_learning
                .as_ref()
                .map(soccer_tactical_learning_weights_fingerprint);
            initial_neural_network = version.neural_network;
            pg_base_policy_version_id = Some(policy_version_id);
            pg_base_generation = Some(generation);
            pg_base_policy_version_updated_at_micros = version.updated_at_micros;
            pg_experiment_id = Some(experiment_id);
            Some(version.policies)
        } else {
            pg_experiment_id = Some(experiment_id);
            None
        }
    } else {
        None
    };

    let spot = free_kick_spot(&config, team, distance_yards);
    println!(
        "soccer_set_play_learning_start run_id={} episodes={} restarts={:?} team={:?} duration={:.3}s distance={:.2}yd dt={:.3}s neural_enabled={} neural_backend={:?} pg_tactical_learning_authoritative={}",
        run_id,
        episodes,
        restarts,
        team,
        duration_seconds,
        distance_yards,
        config.dt_seconds,
        config.neural_learning.enabled,
        config.neural_learning.backend,
        pg_tactical_learning_authoritative,
    );

    let request = SoccerSetPlayTrainingRequest {
        config,
        episodes,
        restart,
        restarts,
        team,
        spot: Some(spot),
        duration_seconds,
        options: Some(options.clone()),
        initial_neural_network,
        vector_hint: None,
        neural_blend: env_neural_blend()?,
    };
    let starting_policies =
        initial_policies.unwrap_or_else(|| SoccerTeamQPolicies::new(options.clone()));
    let artifact = train_soccer_set_play_restarts_with_events(
        request,
        starting_policies,
        |event| -> Result<(), String> {
            let SoccerSetPlayTrainingEvent::StartingEpisode {
                episode,
                config,
                policies,
                neural_network,
                reset_neural_learner,
            } = event;
            if !postgres_resume_policy {
                return Ok(());
            }
            let Some(experiment_id) = pg_experiment_id.as_deref() else {
                return Ok(());
            };
            let Some(store) = pg_store.as_mut() else {
                return Ok(());
            };
            let Some(metadata) = store.load_latest_active_policy_metadata(experiment_id)? else {
                return Ok(());
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
                    current_generation: pg_base_generation.unwrap_or(-1),
                    current_updated_at_micros: pg_base_policy_version_updated_at_micros,
                    current_policy_fingerprint: pg_base_policy_fingerprint,
                    latest_policy_fingerprint,
                    current_neural_network_present: neural_network.is_some(),
                    current_neural_network_fingerprint: pg_base_neural_network_fingerprint,
                    latest_policy_version_id: &metadata.id,
                    latest_generation: metadata.generation,
                    latest_updated_at_micros: metadata.updated_at_micros,
                    latest_neural_network_present: metadata.neural_network.is_some(),
                    latest_neural_network_fingerprint,
                    current_tactical_learning_fingerprint: pg_base_tactical_learning_fingerprint,
                    latest_tactical_learning_fingerprint,
                    local_tactical_evolved_since_pg_refresh: false,
                    postgres_tactical_learning_authoritative: pg_tactical_learning_authoritative,
                });
            if refresh_decision.refresh_policy {
                if let Some(version) = store.load_latest_active_policy(
                    experiment_id,
                    options.clone(),
                    options.clone(),
                )? {
                    println!(
                        "postgres_refresh_policy_for_set_play episode={} policy_version={} previous_policy_version={} generation={} neural_network={}",
                        episode + 1,
                        version.id,
                        pg_base_policy_version_id.as_deref().unwrap_or("none"),
                        version.generation,
                        version.neural_network.is_some()
                    );
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
                            current_generation: pg_base_generation.unwrap_or(-1),
                            current_updated_at_micros: pg_base_policy_version_updated_at_micros,
                            current_policy_fingerprint: pg_base_policy_fingerprint,
                            latest_policy_fingerprint: version_policy_fingerprint,
                            current_neural_network_present: neural_network.is_some(),
                            current_neural_network_fingerprint: pg_base_neural_network_fingerprint,
                            latest_policy_version_id: &version.id,
                            latest_generation: version.generation,
                            latest_updated_at_micros: version.updated_at_micros,
                            latest_neural_network_present: version.neural_network.is_some(),
                            latest_neural_network_fingerprint: version_neural_network_fingerprint,
                            current_tactical_learning_fingerprint:
                                pg_base_tactical_learning_fingerprint,
                            latest_tactical_learning_fingerprint:
                                version_tactical_learning_fingerprint,
                            local_tactical_evolved_since_pg_refresh: false,
                            postgres_tactical_learning_authoritative:
                                pg_tactical_learning_authoritative,
                        });
                    *policies = version.policies;
                    *neural_network = version.neural_network;
                    *reset_neural_learner = true;
                    if version_refresh_decision.apply_tactical_learning {
                        if let Some(tactical_learning) = version.tactical_learning {
                            config.tactical_learning = tactical_learning;
                        }
                        pg_base_tactical_learning_fingerprint =
                            version_tactical_learning_fingerprint;
                    }
                    pg_base_policy_version_id = Some(version.id);
                    pg_base_generation = Some(version.generation);
                    pg_base_policy_version_updated_at_micros = version.updated_at_micros;
                    pg_base_policy_fingerprint = version_policy_fingerprint;
                    pg_base_neural_network_fingerprint = version_neural_network_fingerprint;
                }
            } else if refresh_decision.apply_tactical_learning {
                if let Some(tactical_learning) = metadata.tactical_learning {
                    config.tactical_learning = tactical_learning;
                    pg_base_tactical_learning_fingerprint = latest_tactical_learning_fingerprint;
                }
            }
            Ok(())
        },
    )?;
    let elapsed = started.elapsed().as_secs_f64();
    write_json(&artifact_path, &artifact)?;

    let mut pg_policy_version_id = None::<String>;
    let mut pg_run_id = None::<String>;
    if let (Some(store), Some(experiment_id)) = (pg_store.as_mut(), pg_experiment_id.as_deref()) {
        let generation = pg_base_generation.unwrap_or(-1).saturating_add(1);
        let version_label = env_value("SOCCER_POLICY_VERSION_LABEL")
            .unwrap_or_else(|| version_label_from_run_id(&run_id));
        let latest_active_metadata = store.load_latest_active_policy_metadata(experiment_id)?;
        let insert_status = soccer_policy_version_insert_status_after_active_head(
            SOCCER_POLICY_STATUS_ACTIVE,
            pg_base_policy_version_id.as_deref(),
            generation,
            latest_active_metadata
                .as_ref()
                .map(|metadata| metadata.id.as_str()),
            latest_active_metadata
                .as_ref()
                .map(|metadata| metadata.generation),
        );
        if insert_status != SOCCER_POLICY_STATUS_ACTIVE {
            println!(
                "postgres_set_play_policy_marked_stale run_id={} parent_policy_version={} generation={} latest_active={} latest_generation={}",
                run_id,
                pg_base_policy_version_id.as_deref().unwrap_or("none"),
                generation,
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
        let (policy_version_id, run_pg_id) = store.insert_set_play_training_artifact(
            experiment_id,
            &run_id,
            pg_base_policy_version_id.as_deref(),
            generation,
            &version_label,
            insert_status,
            &artifact,
            elapsed,
        )?;
        println!(
            "postgres_persisted_set_play run_id={} policy_version={} pg_run={}",
            run_id, policy_version_id, run_pg_id
        );
        pg_policy_version_id = Some(policy_version_id);
        pg_run_id = Some(run_pg_id);
    }

    println!(
        "soccer_set_play_learning_done episodes={} goals={} goal_rate={:.4} first_window={:.4} last_window={:.4} delta={:.4} elapsed={:.2}s neural_steps={} neural_samples={}",
        artifact.episodes.len(),
        artifact.goals,
        artifact.goal_rate,
        artifact.first_window_goal_rate,
        artifact.last_window_goal_rate,
        artifact.goal_rate_delta,
        elapsed,
        artifact.learning.neural_learning_training_steps,
        artifact.learning.neural_learning_samples,
    );
    println!("artifact={}", artifact_path.display());
    if let Some(policy_version_id) = pg_policy_version_id {
        println!("postgres_policy_version={policy_version_id}");
    }
    if let Some(run_id) = pg_run_id {
        println!("postgres_run={run_id}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_play_new_episodes_apply_refreshed_tactical_weights_before_sim() {
        let options = SoccerQPolicyOptions::default();
        let mut config = MatchConfig {
            duration_seconds: 0.1,
            dt_seconds: 0.1,
            learning_enabled: true,
            learning_logging_enabled: false,
            max_human_players: 0,
            neural_learning: SoccerNeuralLearningConfig {
                enabled: false,
                ..SoccerNeuralLearningConfig::default()
            },
            ..MatchConfig::default()
        };
        config.tactical_learning.attack_flank_lane_weight = 1.0;
        config.tactical_learning.defense_contract_delta_weight = 1.0;

        let mut submitted_weights = Vec::new();
        let artifact = train_soccer_set_play_restarts_with_events(
            SoccerSetPlayTrainingRequest {
                config,
                episodes: 3,
                restart: SoccerSetPlayRestartKind::DirectFreeKick,
                restarts: vec![SoccerSetPlayRestartKind::DirectFreeKick],
                team: Team::Home,
                spot: Some(Vec2 { x: 45.0, y: 72.0 }),
                duration_seconds: 0.1,
                options: Some(options.clone()),
                initial_neural_network: None,
                vector_hint: None,
                neural_blend: SoccerNeuralBlendConfig::default(),
            },
            SoccerTeamQPolicies::new(options),
            |event| {
                let SoccerSetPlayTrainingEvent::StartingEpisode {
                    episode, config, ..
                } = event;
                config.tactical_learning.attack_flank_lane_weight = 1.20 + episode as f64 * 0.07;
                config.tactical_learning.defense_contract_delta_weight =
                    0.90 + episode as f64 * 0.05;
                config.tactical_learning.formation_lp_alignment_weight =
                    0.30 + episode as f64 * 0.03;
                submitted_weights.push((
                    episode,
                    config.tactical_learning.attack_flank_lane_weight,
                    config.tactical_learning.defense_contract_delta_weight,
                    config.tactical_learning.formation_lp_alignment_weight,
                ));
                Ok(())
            },
        )
        .expect("set-play training");

        assert_eq!(artifact.episodes.len(), 3);
        assert_eq!(submitted_weights.len(), 3);
        for (episode, attack_flank, defense_contract, lp_alignment) in submitted_weights {
            assert!((attack_flank - (1.20 + episode as f64 * 0.07)).abs() < 1e-12);
            assert!((defense_contract - (0.90 + episode as f64 * 0.05)).abs() < 1e-12);
            assert!((lp_alignment - (0.30 + episode as f64 * 0.03)).abs() < 1e-12);
        }
        assert!((artifact.config.tactical_learning.attack_flank_lane_weight - 1.34).abs() < 1e-12);
        assert!(
            (artifact
                .config
                .tactical_learning
                .defense_contract_delta_weight
                - 1.00)
                .abs()
                < 1e-12
        );
        assert!(
            (artifact
                .config
                .tactical_learning
                .formation_lp_alignment_weight
                - 0.36)
                .abs()
                < 1e-12
        );
    }
}
