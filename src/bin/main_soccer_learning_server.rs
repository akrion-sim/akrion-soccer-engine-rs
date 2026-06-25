//! Run soccer self-play through the des-rs HTTP training endpoint.
//!
//! This is the Rust equivalent of the JSON-building/parsing glue in
//! `scripts/soccer_self_play_server.sh`; the script keeps environment defaults
//! and run metadata, while this binary owns payload construction, curl dispatch,
//! and response artifact validation.

use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{self, BufWriter, ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde_json::{json, Value};
use soccer_engine::des::general::soccer::{
    MatchConfig, SoccerQPolicyOptions, SoccerSelfPlayLearnedParams, SoccerSelfPlayTrainingArtifact,
    SoccerTacticalLearningSummary, SoccerTacticalLearningWeights, SoccerTeamQPolicies,
    SOCCER_SELF_PLAY_LEARNED_PARAMS_VERSION,
};
use soccer_engine::des::soccer_learning::{
    soccer_learning_run_score, soccer_neural_network_snapshot_fingerprint,
    soccer_policy_version_insert_status_after_active_head,
    soccer_tactical_learning_weights_fingerprint, soccer_team_q_policies_fingerprint,
    SOCCER_POLICY_STATUS_ACTIVE,
};
use soccer_engine::des::soccer_learning_pg::SoccerLearningPgStore;
use uuid::Uuid;

#[derive(Clone, Debug)]
struct Args {
    endpoint: String,
    payload_path: PathBuf,
    response_path: PathBuf,
    artifact_path: PathBuf,
    learned_params_path: PathBuf,
    episode_log_path: PathBuf,
    server_artifact_path: String,
    server_learned_params_path: String,
    auth_header_name: String,
    auth_env_name: Option<String>,
    auth_value: Option<String>,
}

#[derive(Debug)]
struct ExtractedServerResponse {
    episodes: usize,
    home_entries: usize,
    away_entries: usize,
    artifact: SoccerSelfPlayTrainingArtifact,
    learned_params: SoccerSelfPlayLearnedParams,
}

#[derive(Debug, PartialEq, Eq)]
struct PostgresResumeMetadata {
    experiment_id: String,
    policy_version_id: String,
    generation: i32,
}

#[derive(Debug)]
struct PostgresImportSummary {
    experiment_id: String,
    policy_version_id: String,
    parent_policy_version_id: Option<String>,
    generation: i32,
    status: &'static str,
    fitness: f64,
}

#[derive(Debug)]
struct PostgresInitialLearnedParams {
    experiment_id: String,
    policy_version_id: String,
    generation: i32,
    policy_fingerprint: Option<u64>,
    tactical_learning_fingerprint: u64,
    neural_network_fingerprint: Option<u64>,
    search_metadata: Option<Value>,
    params: SoccerSelfPlayLearnedParams,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ServerRequestChunk {
    index: usize,
    total_chunks: usize,
    episode_offset: usize,
    episodes: usize,
    seed: u32,
}

const ALLOW_POSTGRES_MULTI_GAME_SERVER_CHUNKS_ENV: &str =
    "SOCCER_SERVER_ALLOW_POSTGRES_MULTI_GAME_CHUNKS";
const REQUIRE_FRESH_POSTGRES_WEIGHTS_ENV: &str = "SOCCER_SERVER_REQUIRE_FRESH_POSTGRES_WEIGHTS";

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidInput, message.into())
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, message.into())
}

fn next_arg<I>(args: &mut I, flag: &str) -> Result<String, Box<dyn Error>>
where
    I: Iterator<Item = OsString>,
{
    let value = args
        .next()
        .ok_or_else(|| invalid_input(format!("{flag} requires a value")))?;
    value
        .into_string()
        .map_err(|_| invalid_input(format!("{flag} must be valid UTF-8")).into())
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut endpoint = None::<String>;
    let mut payload_path = None::<PathBuf>;
    let mut response_path = None::<PathBuf>;
    let mut artifact_path = None::<PathBuf>;
    let mut learned_params_path = None::<PathBuf>;
    let mut episode_log_path = None::<PathBuf>;
    let mut server_artifact_path = None::<String>;
    let mut server_learned_params_path = None::<String>;
    let mut auth_header_name = "Auth".to_string();
    let mut auth_env_name = Some("DES_RS_AUTH".to_string());
    let mut auth_value = None::<String>;

    let mut args = env::args_os().skip(1);
    while let Some(arg) = args.next() {
        let flag = arg
            .into_string()
            .map_err(|_| invalid_input("argument flag must be valid UTF-8"))?;
        match flag.as_str() {
            "--endpoint" => endpoint = Some(next_arg(&mut args, "--endpoint")?),
            "--payload" => payload_path = Some(PathBuf::from(next_arg(&mut args, "--payload")?)),
            "--response" => response_path = Some(PathBuf::from(next_arg(&mut args, "--response")?)),
            "--artifact" => artifact_path = Some(PathBuf::from(next_arg(&mut args, "--artifact")?)),
            "--learned-params" => {
                learned_params_path = Some(PathBuf::from(next_arg(&mut args, "--learned-params")?))
            }
            "--episode-log" => {
                episode_log_path = Some(PathBuf::from(next_arg(&mut args, "--episode-log")?))
            }
            "--server-artifact-path" => {
                server_artifact_path = Some(next_arg(&mut args, "--server-artifact-path")?)
            }
            "--server-learned-params-path" => {
                server_learned_params_path =
                    Some(next_arg(&mut args, "--server-learned-params-path")?)
            }
            "--auth-header-name" => auth_header_name = next_arg(&mut args, "--auth-header-name")?,
            "--auth-env-name" => {
                let value = next_arg(&mut args, "--auth-env-name")?;
                auth_env_name = if value.trim().is_empty() {
                    None
                } else {
                    Some(value)
                };
            }
            "--auth-value" => {
                let value = next_arg(&mut args, "--auth-value")?;
                if !value.trim().is_empty() {
                    auth_value = Some(value);
                }
            }
            "--help" | "-h" => {
                println!(
                    "usage: main_soccer_learning_server --endpoint URL --payload PATH --response PATH --artifact PATH --learned-params PATH --episode-log PATH --server-artifact-path PATH --server-learned-params-path PATH [--auth-header-name NAME] [--auth-env-name NAME]"
                );
                std::process::exit(0);
            }
            _ => return Err(invalid_input(format!("unknown argument {flag}")).into()),
        }
    }

    Ok(Args {
        endpoint: endpoint.ok_or_else(|| invalid_input("--endpoint is required"))?,
        payload_path: payload_path.ok_or_else(|| invalid_input("--payload is required"))?,
        response_path: response_path.ok_or_else(|| invalid_input("--response is required"))?,
        artifact_path: artifact_path.ok_or_else(|| invalid_input("--artifact is required"))?,
        learned_params_path: learned_params_path
            .ok_or_else(|| invalid_input("--learned-params is required"))?,
        episode_log_path: episode_log_path
            .ok_or_else(|| invalid_input("--episode-log is required"))?,
        server_artifact_path: server_artifact_path
            .ok_or_else(|| invalid_input("--server-artifact-path is required"))?,
        server_learned_params_path: server_learned_params_path
            .ok_or_else(|| invalid_input("--server-learned-params-path is required"))?,
        auth_header_name,
        auth_env_name,
        auth_value,
    })
}

fn env_string(name: &str, default: &str) -> String {
    env::var(name).unwrap_or_else(|_| default.to_string())
}

fn env_value(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn env_f64(name: &str, default: f64) -> Result<f64, Box<dyn Error>> {
    match env_value(name) {
        Some(raw) => {
            let parsed = raw.parse::<f64>().map_err(|err| {
                invalid_input(format!("{name}={raw:?} is not a finite number: {err}"))
            })?;
            if !parsed.is_finite() {
                return Err(invalid_input(format!("{name}={raw:?} is not finite")).into());
            }
            Ok(parsed)
        }
        None => Ok(default),
    }
}

fn env_usize(name: &str, default: usize) -> Result<usize, Box<dyn Error>> {
    match env_value(name) {
        Some(raw) => raw.parse::<usize>().map_err(|err| {
            invalid_input(format!(
                "{name}={raw:?} is not a non-negative integer: {err}"
            ))
            .into()
        }),
        None => Ok(default),
    }
}

fn env_usize_alias(primary: &str, alias: &str, default: usize) -> Result<usize, Box<dyn Error>> {
    match env_value(primary).or_else(|| env_value(alias)) {
        Some(raw) => raw.parse::<usize>().map_err(|err| {
            invalid_input(format!(
                "{primary}/{alias}={raw:?} is not a non-negative integer: {err}"
            ))
            .into()
        }),
        None => Ok(default),
    }
}

fn env_u32(name: &str, default: u32) -> Result<u32, Box<dyn Error>> {
    match env_value(name) {
        Some(raw) => raw
            .parse::<u32>()
            .map_err(|err| invalid_input(format!("{name}={raw:?} is not a u32: {err}")).into()),
        None => Ok(default),
    }
}

fn env_bool(name: &str, default: bool) -> Result<bool, Box<dyn Error>> {
    match env_value(name) {
        Some(raw) => match raw.to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "y" | "on" => Ok(true),
            "0" | "false" | "no" | "n" | "off" => Ok(false),
            _ => Err(invalid_input(format!("{name}={raw:?} is not a boolean")).into()),
        },
        None => Ok(default),
    }
}

fn validate_payload_settings(
    episodes: usize,
    minutes: f64,
    period_count: usize,
    period_break_recovery_seconds: f64,
    dt_seconds: f64,
    learning_interval_ticks: usize,
    alpha: f64,
    gamma: f64,
    tactical_weights: &[(&str, f64)],
) -> Result<(), Box<dyn Error>> {
    if episodes == 0 {
        return Err(invalid_input("SOCCER_GAMES must be at least 1").into());
    }
    if !minutes.is_finite() || minutes <= 0.0 || minutes > 24.0 * 60.0 {
        return Err(invalid_input("SOCCER_MINUTES must be finite and in (0, 1440]").into());
    }
    if !(1..=8).contains(&period_count) {
        return Err(invalid_input("SOCCER_HALVES must be between 1 and 8").into());
    }
    if !period_break_recovery_seconds.is_finite()
        || !(0.0..=60.0 * 60.0).contains(&period_break_recovery_seconds)
    {
        return Err(invalid_input(
            "SOCCER_PERIOD_BREAK_RECOVERY_SECONDS must be finite and in [0, 3600]",
        )
        .into());
    }
    if !dt_seconds.is_finite() || !(0.01..=4.0).contains(&dt_seconds) {
        return Err(invalid_input("SOCCER_DT_SECONDS must be finite and in [0.01, 4.0]").into());
    }
    if learning_interval_ticks == 0 {
        return Err(invalid_input("SOCCER_LEARNING_INTERVAL_TICKS must be at least 1").into());
    }
    if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
        return Err(invalid_input("SOCCER_ALPHA must be finite and in [0, 1]").into());
    }
    if !gamma.is_finite() || !(0.0..=1.0).contains(&gamma) {
        return Err(invalid_input("SOCCER_GAMMA must be finite and in [0, 1]").into());
    }
    for (name, value) in tactical_weights {
        if !value.is_finite() {
            return Err(invalid_input(format!("{name} must be finite")).into());
        }
    }
    Ok(())
}

fn postgres_resume_enabled() -> Result<bool, Box<dyn Error>> {
    if env_value("SOCCER_SERVER_RESUME_POSTGRES_POLICY").is_some() {
        env_bool("SOCCER_SERVER_RESUME_POSTGRES_POLICY", true)
    } else {
        env_bool("SOCCER_RESUME_POSTGRES_POLICY", true)
    }
}

fn postgres_import_enabled() -> Result<bool, Box<dyn Error>> {
    if env_value("SOCCER_SERVER_IMPORT_POSTGRES_POLICY").is_some() {
        env_bool("SOCCER_SERVER_IMPORT_POSTGRES_POLICY", true)
    } else if env_value("SOCCER_IMPORT_POSTGRES_POLICY").is_some() {
        env_bool("SOCCER_IMPORT_POSTGRES_POLICY", true)
    } else {
        postgres_resume_enabled()
    }
}

fn postgres_database_url_present() -> bool {
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

fn default_server_request_games(
    total_episodes: usize,
    postgres_database_url_present: bool,
    postgres_resume_enabled: bool,
) -> usize {
    if postgres_database_url_present && postgres_resume_enabled {
        1
    } else {
        total_episodes
    }
}

fn effective_server_request_games(
    requested_episodes_per_request: usize,
    postgres_database_url_present: bool,
    postgres_resume_enabled: bool,
    allow_postgres_multi_game_chunks: bool,
    require_fresh_postgres_weights: bool,
) -> usize {
    if postgres_database_url_present
        && postgres_resume_enabled
        && (require_fresh_postgres_weights || !allow_postgres_multi_game_chunks)
    {
        requested_episodes_per_request.min(1)
    } else {
        requested_episodes_per_request
    }
}

fn postgres_resume_metadata_from_payload(payload: &Value) -> Option<PostgresResumeMetadata> {
    let resume = payload.get("postgresResume")?;
    let experiment_id = resume.get("experimentId")?.as_str()?.trim();
    let policy_version_id = resume.get("policyVersionId")?.as_str()?.trim();
    if experiment_id.is_empty() || policy_version_id.is_empty() {
        return None;
    }
    let generation = resume
        .get("generation")
        .and_then(Value::as_i64)
        .and_then(|value| i32::try_from(value).ok())
        .unwrap_or(0);
    Some(PostgresResumeMetadata {
        experiment_id: experiment_id.to_string(),
        policy_version_id: policy_version_id.to_string(),
        generation,
    })
}

fn postgres_import_version_label(run_id: &str, episodes: usize) -> String {
    let suffix = format!("-server-e{episodes:06}");
    let max_prefix_len = 160usize.saturating_sub(suffix.len());
    let mut prefix = run_id
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | ':' | '/' | '-'))
        .take(max_prefix_len)
        .collect::<String>();
    if prefix.is_empty() {
        prefix.push_str("server");
    }
    format!("{prefix}{suffix}")
}

fn server_response_fitness(artifact: &SoccerSelfPlayTrainingArtifact) -> f64 {
    if artifact.episodes.is_empty() {
        return 0.0;
    }
    let total = artifact
        .episodes
        .iter()
        .map(|episode| soccer_learning_run_score(&episode.summary).match_fitness)
        .sum::<f64>();
    let fitness = total / artifact.episodes.len() as f64;
    if fitness.is_finite() {
        fitness
    } else {
        0.0
    }
}

fn server_import_search_metadata(
    payload: &Value,
    response: &ExtractedServerResponse,
    parent_policy_version_id: Option<&str>,
    parent_generation: i32,
    generation: i32,
    fitness: f64,
) -> Value {
    let resume = payload.get("postgresResume").cloned().unwrap_or_else(|| {
        json!({
            "policyVersionId": parent_policy_version_id,
            "generation": parent_generation,
        })
    });
    json!({
        "algorithm": "server-mdp-pomdp-neural-self-play",
        "source": "des-rs-http-training-endpoint",
        "request": {
            "episodes": payload.get("episodes").and_then(Value::as_u64),
            "seed": payload.get("seed").and_then(Value::as_u64),
            "minutes": payload.get("minutes").and_then(Value::as_f64),
            "dtSeconds": payload.get("dtSeconds").and_then(Value::as_f64),
            "learningIntervalTicks": payload
                .get("learningIntervalTicks")
                .and_then(Value::as_u64),
            "formationLpEnabled": payload
                .get("formationLpEnabled")
                .and_then(Value::as_bool),
            "importIntoSession": payload
                .get("importIntoSession")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            "initialLearnedParamsIncluded": payload.get("initialLearnedParams").is_some(),
            "serverArtifactPath": payload.get("artifactPath").and_then(Value::as_str),
            "serverLearnedParamsPath": payload.get("learnedParamsPath").and_then(Value::as_str),
        },
        "postgresResume": resume,
        "response": {
            "episodes": response.episodes,
            "homeEntries": response.home_entries,
            "awayEntries": response.away_entries,
            "fitness": fitness,
            "generation": generation,
        },
        "learningComponents": {
            "mdpPolicyEntries": response.home_entries + response.away_entries > 0,
            "pomdpStateKeyFeatures": true,
            "neuralNetworkSnapshot": response.learned_params.neural_network.is_some(),
            "tacticalLearningWeights": true,
        }
    })
}

fn payload_match_config(
    minutes: f64,
    period_count: usize,
    period_break_recovery_seconds: f64,
    dt_seconds: f64,
    learning_interval_ticks: usize,
    formation_lp_enabled: bool,
    seed: u32,
    tactical_learning: SoccerTacticalLearningWeights,
) -> MatchConfig {
    MatchConfig {
        duration_seconds: minutes * 60.0,
        period_count,
        period_break_recovery_seconds,
        dt_seconds,
        learning_interval_ticks,
        formation_lp_enabled,
        seed,
        learning_enabled: true,
        learning_logging_enabled: false,
        max_human_players: 0,
        tactical_learning,
        ..MatchConfig::default()
    }
}

fn postgres_initial_learned_params(
    config: &MatchConfig,
    options: &SoccerQPolicyOptions,
) -> Result<Option<PostgresInitialLearnedParams>, Box<dyn Error>> {
    if !postgres_resume_enabled()? {
        return Ok(None);
    }
    let Some(mut store) = SoccerLearningPgStore::connect_from_env()? else {
        return Ok(None);
    };
    let slug =
        env_value("SOCCER_EXPERIMENT_SLUG").unwrap_or_else(|| "soccer-self-play".to_string());
    let display_name =
        env_value("SOCCER_EXPERIMENT_NAME").unwrap_or_else(|| "Soccer self-play".to_string());
    let experiment_id = store.ensure_experiment(&slug, &display_name, config)?;
    let Some(version) =
        store.load_latest_active_policy(&experiment_id, options.clone(), options.clone())?
    else {
        return Ok(None);
    };
    let tactical_learning = version
        .tactical_learning
        .clone()
        .unwrap_or_else(|| config.tactical_learning.clone());
    let tactical_learning_fingerprint =
        soccer_tactical_learning_weights_fingerprint(&tactical_learning);
    let neural_network_fingerprint = version
        .neural_network
        .as_ref()
        .map(soccer_neural_network_snapshot_fingerprint);
    let policy_fingerprint = version
        .policy_fingerprint
        .or_else(|| Some(soccer_team_q_policies_fingerprint(&version.policies)));
    let params = SoccerSelfPlayLearnedParams {
        version: SOCCER_SELF_PLAY_LEARNED_PARAMS_VERSION,
        config: MatchConfig {
            tactical_learning: tactical_learning.clone(),
            ..config.clone()
        },
        options: options.clone(),
        tactical_learning,
        tactical_summary: SoccerTacticalLearningSummary::default(),
        episodes: 0,
        home_entries: version.policies.home.entries(),
        home_target_entries: version.policies.home.target_entries(),
        away_entries: version.policies.away.entries(),
        away_target_entries: version.policies.away.target_entries(),
        neural_network: version.neural_network,
    };
    Ok(Some(PostgresInitialLearnedParams {
        experiment_id,
        policy_version_id: version.id,
        generation: version.generation,
        policy_fingerprint,
        tactical_learning_fingerprint,
        neural_network_fingerprint,
        search_metadata: version.search_metadata,
        params,
    }))
}

fn server_request_chunks(
    total_episodes: usize,
    episodes_per_request: usize,
    base_seed: u32,
) -> Result<Vec<ServerRequestChunk>, Box<dyn Error>> {
    if total_episodes == 0 {
        return Err(invalid_input("SOCCER_GAMES must be at least 1").into());
    }
    if episodes_per_request == 0 {
        return Err(invalid_input(
            "SOCCER_SERVER_REQUEST_GAMES/SOCCER_SERVER_CHUNK_GAMES must be at least 1",
        )
        .into());
    }
    let total_chunks = total_episodes.div_ceil(episodes_per_request);
    let mut chunks = Vec::with_capacity(total_chunks);
    let mut episode_offset = 0usize;
    while episode_offset < total_episodes {
        let remaining = total_episodes - episode_offset;
        let episodes = remaining.min(episodes_per_request);
        chunks.push(ServerRequestChunk {
            index: chunks.len(),
            total_chunks,
            episode_offset,
            episodes,
            seed: base_seed.wrapping_add(episode_offset as u32),
        });
        episode_offset += episodes;
    }
    Ok(chunks)
}

fn chunk_suffix(chunk: ServerRequestChunk) -> String {
    format!("chunk-{:04}-of-{:04}", chunk.index + 1, chunk.total_chunks)
}

fn chunked_path(path: &Path, chunk: ServerRequestChunk) -> PathBuf {
    if chunk.total_chunks <= 1 || chunk.index + 1 == chunk.total_chunks {
        return path.to_path_buf();
    }
    let parent = path.parent();
    let stem = path
        .file_stem()
        .map(|value| value.to_string_lossy())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "request".into());
    let mut file_name = format!("{}.{}", stem, chunk_suffix(chunk));
    if let Some(extension) = path.extension().map(|value| value.to_string_lossy()) {
        if !extension.is_empty() {
            file_name.push('.');
            file_name.push_str(&extension);
        }
    }
    if let Some(dir) = parent {
        dir.join(file_name)
    } else {
        PathBuf::from(file_name)
    }
}

fn chunked_server_path(path: &str, chunk: ServerRequestChunk) -> String {
    chunked_path(Path::new(path), chunk)
        .to_string_lossy()
        .to_string()
}

fn args_for_server_chunk(args: &Args, chunk: ServerRequestChunk) -> Args {
    if chunk.total_chunks <= 1 || chunk.index + 1 == chunk.total_chunks {
        return args.clone();
    }
    Args {
        endpoint: args.endpoint.clone(),
        payload_path: chunked_path(&args.payload_path, chunk),
        response_path: chunked_path(&args.response_path, chunk),
        artifact_path: chunked_path(&args.artifact_path, chunk),
        learned_params_path: chunked_path(&args.learned_params_path, chunk),
        episode_log_path: chunked_path(&args.episode_log_path, chunk),
        server_artifact_path: chunked_server_path(&args.server_artifact_path, chunk),
        server_learned_params_path: chunked_server_path(&args.server_learned_params_path, chunk),
        auth_header_name: args.auth_header_name.clone(),
        auth_env_name: args.auth_env_name.clone(),
        auth_value: args.auth_value.clone(),
    }
}

fn build_payload(
    args: &Args,
    episodes_override: Option<usize>,
    seed_override: Option<u32>,
) -> Result<Value, Box<dyn Error>> {
    let episodes = match episodes_override {
        Some(episodes) => episodes,
        None => env_usize("SOCCER_GAMES", 100)?,
    };
    let minutes = env_f64("SOCCER_MINUTES", 90.0)?;
    let period_count = env_usize("SOCCER_HALVES", 2)?;
    let period_break_recovery_seconds = env_f64("SOCCER_PERIOD_BREAK_RECOVERY_SECONDS", 900.0)?;
    let dt_seconds = env_f64("SOCCER_DT_SECONDS", 0.2)?;
    let learning_interval_ticks = env_usize("SOCCER_LEARNING_INTERVAL_TICKS", 4)?;
    let default_config = MatchConfig::default();
    let formation_lp_enabled = env_bool(
        "SOCCER_FORMATION_LP_ENABLED",
        default_config.formation_lp_enabled,
    )?;
    let seed = match seed_override {
        Some(seed) => seed,
        None => env_u32("SOCCER_SEED", 2026)?,
    };
    let alpha = env_f64("SOCCER_ALPHA", 0.20)?;
    let gamma = env_f64("SOCCER_GAMMA", 0.96)?;
    let attack_spacing_delta_weight = env_f64("SOCCER_ATTACK_SPACING_DELTA_WEIGHT", 0.22)?;
    let attack_spacing_score_weight = env_f64("SOCCER_ATTACK_SPACING_SCORE_WEIGHT", 0.06)?;
    let attack_width_delta_weight = env_f64("SOCCER_ATTACK_WIDTH_DELTA_WEIGHT", 0.68)?;
    let attack_width_score_weight = env_f64("SOCCER_ATTACK_WIDTH_SCORE_WEIGHT", 0.32)?;
    let attack_flank_lane_weight = env_f64("SOCCER_ATTACK_FLANK_LANE_WEIGHT", 0.50)?;
    let shot_choice_learning_weight = env_f64("SOCCER_SHOT_CHOICE_LEARNING_WEIGHT", 1.0)?;
    let goal_entry_pass_learning_weight = env_f64("SOCCER_GOAL_ENTRY_PASS_LEARNING_WEIGHT", 1.0)?;
    let pressure_release_learning_weight = env_f64("SOCCER_PRESSURE_RELEASE_LEARNING_WEIGHT", 1.0)?;
    let pass_target_ranking_learning_weight =
        env_f64("SOCCER_PASS_TARGET_RANKING_LEARNING_WEIGHT", 1.0)?;
    let defense_spacing_delta_weight = env_f64("SOCCER_DEFENSE_SPACING_DELTA_WEIGHT", 0.08)?;
    let defense_spacing_score_weight = env_f64("SOCCER_DEFENSE_SPACING_SCORE_WEIGHT", 0.04)?;
    let defense_contract_delta_weight = env_f64("SOCCER_DEFENSE_CONTRACT_DELTA_WEIGHT", 0.42)?;
    let defense_compactness_score_weight =
        env_f64("SOCCER_DEFENSE_COMPACTNESS_SCORE_WEIGHT", 0.14)?;
    let defense_ball_depth_score_weight = env_f64("SOCCER_DEFENSE_BALL_DEPTH_SCORE_WEIGHT", 0.22)?;
    let defense_endline_soft_penalty_weight =
        env_f64("SOCCER_DEFENSE_ENDLINE_SOFT_PENALTY_WEIGHT", 0.18)?;
    let defense_endline_hard_penalty_weight =
        env_f64("SOCCER_DEFENSE_ENDLINE_HARD_PENALTY_WEIGHT", 0.90)?;
    let defender_midfielder_press_weight =
        env_f64("SOCCER_DEFENDER_MIDFIELDER_PRESS_WEIGHT", 0.18)?;
    let midfielder_press_weight = env_f64("SOCCER_MIDFIELDER_PRESS_WEIGHT", 0.20)?;
    let defensive_line_press_learning_weight =
        env_f64("SOCCER_DEFENSIVE_LINE_PRESS_LEARNING_WEIGHT", 1.0)?;
    let formation_lp_alignment_weight = env_f64("SOCCER_FORMATION_LP_ALIGNMENT_WEIGHT", 0.16)?;
    let tactical_weights = [
        (
            "SOCCER_ATTACK_SPACING_DELTA_WEIGHT",
            attack_spacing_delta_weight,
        ),
        (
            "SOCCER_ATTACK_SPACING_SCORE_WEIGHT",
            attack_spacing_score_weight,
        ),
        (
            "SOCCER_ATTACK_WIDTH_DELTA_WEIGHT",
            attack_width_delta_weight,
        ),
        (
            "SOCCER_ATTACK_WIDTH_SCORE_WEIGHT",
            attack_width_score_weight,
        ),
        ("SOCCER_ATTACK_FLANK_LANE_WEIGHT", attack_flank_lane_weight),
        (
            "SOCCER_SHOT_CHOICE_LEARNING_WEIGHT",
            shot_choice_learning_weight,
        ),
        (
            "SOCCER_GOAL_ENTRY_PASS_LEARNING_WEIGHT",
            goal_entry_pass_learning_weight,
        ),
        (
            "SOCCER_PRESSURE_RELEASE_LEARNING_WEIGHT",
            pressure_release_learning_weight,
        ),
        (
            "SOCCER_PASS_TARGET_RANKING_LEARNING_WEIGHT",
            pass_target_ranking_learning_weight,
        ),
        (
            "SOCCER_DEFENSE_SPACING_DELTA_WEIGHT",
            defense_spacing_delta_weight,
        ),
        (
            "SOCCER_DEFENSE_SPACING_SCORE_WEIGHT",
            defense_spacing_score_weight,
        ),
        (
            "SOCCER_DEFENSE_CONTRACT_DELTA_WEIGHT",
            defense_contract_delta_weight,
        ),
        (
            "SOCCER_DEFENSE_COMPACTNESS_SCORE_WEIGHT",
            defense_compactness_score_weight,
        ),
        (
            "SOCCER_DEFENSE_BALL_DEPTH_SCORE_WEIGHT",
            defense_ball_depth_score_weight,
        ),
        (
            "SOCCER_DEFENSE_ENDLINE_SOFT_PENALTY_WEIGHT",
            defense_endline_soft_penalty_weight,
        ),
        (
            "SOCCER_DEFENSE_ENDLINE_HARD_PENALTY_WEIGHT",
            defense_endline_hard_penalty_weight,
        ),
        (
            "SOCCER_DEFENDER_MIDFIELDER_PRESS_WEIGHT",
            defender_midfielder_press_weight,
        ),
        ("SOCCER_MIDFIELDER_PRESS_WEIGHT", midfielder_press_weight),
        (
            "SOCCER_DEFENSIVE_LINE_PRESS_LEARNING_WEIGHT",
            defensive_line_press_learning_weight,
        ),
        (
            "SOCCER_FORMATION_LP_ALIGNMENT_WEIGHT",
            formation_lp_alignment_weight,
        ),
    ];
    validate_payload_settings(
        episodes,
        minutes,
        period_count,
        period_break_recovery_seconds,
        dt_seconds,
        learning_interval_ticks,
        alpha,
        gamma,
        &tactical_weights,
    )?;
    let options = SoccerQPolicyOptions {
        alpha,
        gamma,
        exploration_epsilon: env_f64("SOCCER_EXPLORATION_EPSILON", 0.0)?,
    };
    let tactical_learning = SoccerTacticalLearningWeights {
        attack_spacing_delta_weight,
        attack_spacing_score_weight,
        attack_width_delta_weight,
        attack_width_score_weight,
        attack_flank_lane_weight,
        shot_choice_learning_weight,
        goal_entry_pass_learning_weight,
        pressure_release_learning_weight,
        pass_target_ranking_learning_weight,
        defense_spacing_delta_weight,
        defense_spacing_score_weight,
        defense_contract_delta_weight,
        defense_compactness_score_weight,
        defense_ball_depth_score_weight,
        defense_endline_soft_penalty_weight,
        defense_endline_hard_penalty_weight,
        defender_midfielder_press_weight,
        midfielder_press_weight,
        defensive_line_press_learning_weight,
        formation_lp_alignment_weight,
    };
    let config = payload_match_config(
        minutes,
        period_count,
        period_break_recovery_seconds,
        dt_seconds,
        learning_interval_ticks,
        formation_lp_enabled,
        seed,
        tactical_learning.clone(),
    );
    let import_into_session = env_bool("SOCCER_IMPORT_INTO_SESSION", true)?;
    let mut payload = json!({
        "episodes": episodes,
        "minutes": minutes,
        "periodCount": period_count,
        "periodBreakRecoverySeconds": period_break_recovery_seconds,
        "dtSeconds": dt_seconds,
        "learningIntervalTicks": learning_interval_ticks,
        "formationLpEnabled": formation_lp_enabled,
        "seed": seed,
        "options": {
            "alpha": alpha,
            "gamma": gamma,
        },
        "tacticalLearning": {
            "attackSpacingDeltaWeight": attack_spacing_delta_weight,
            "attackSpacingScoreWeight": attack_spacing_score_weight,
            "attackWidthDeltaWeight": attack_width_delta_weight,
            "attackWidthScoreWeight": attack_width_score_weight,
            "attackFlankLaneWeight": attack_flank_lane_weight,
            "shotChoiceLearningWeight": shot_choice_learning_weight,
            "goalEntryPassLearningWeight": goal_entry_pass_learning_weight,
            "pressureReleaseLearningWeight": pressure_release_learning_weight,
            "passTargetRankingLearningWeight": pass_target_ranking_learning_weight,
            "defenseSpacingDeltaWeight": defense_spacing_delta_weight,
            "defenseSpacingScoreWeight": defense_spacing_score_weight,
            "defenseContractDeltaWeight": defense_contract_delta_weight,
            "defenseCompactnessScoreWeight": defense_compactness_score_weight,
            "defenseBallDepthScoreWeight": defense_ball_depth_score_weight,
            "defenseEndlineSoftPenaltyWeight": defense_endline_soft_penalty_weight,
            "defenseEndlineHardPenaltyWeight": defense_endline_hard_penalty_weight,
            "defenderMidfielderPressWeight": defender_midfielder_press_weight,
            "midfielderPressWeight": midfielder_press_weight,
            "defensiveLinePressLearningWeight": defensive_line_press_learning_weight,
            "formationLpAlignmentWeight": formation_lp_alignment_weight,
        },
        "artifactPath": &args.server_artifact_path,
        "learnedParamsPath": &args.server_learned_params_path,
        "importIntoSession": import_into_session,
    });
    if let Some(postgres_initial) = postgres_initial_learned_params(&config, &options)? {
        payload["tacticalLearning"] =
            serde_json::to_value(&postgres_initial.params.tactical_learning)?;
        payload["initialLearnedParams"] = serde_json::to_value(&postgres_initial.params)?;
        payload["postgresResume"] = json!({
            "experimentId": postgres_initial.experiment_id,
            "policyVersionId": postgres_initial.policy_version_id,
            "generation": postgres_initial.generation,
            "policyFingerprint": postgres_initial.policy_fingerprint,
            "tacticalLearningFingerprint": postgres_initial.tactical_learning_fingerprint,
            "neuralNetworkFingerprint": postgres_initial.neural_network_fingerprint,
            "searchParameters": postgres_initial.search_metadata,
        });
    }
    Ok(payload)
}

fn ensure_parent(path: &Path) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn sync_parent_best_effort(path: &Path) {
    if let Some(parent) = path.parent() {
        let _ = File::open(parent).and_then(|dir| dir.sync_all());
    }
}

fn write_json_pretty(path: &Path, value: &Value) -> Result<(), Box<dyn Error>> {
    ensure_parent(path)?;
    let mut tmp_name = path.as_os_str().to_os_string();
    tmp_name.push(format!(".tmp-{}", std::process::id()));
    let tmp_path = PathBuf::from(tmp_name);
    let result = (|| -> Result<(), Box<dyn Error>> {
        let file = File::create(&tmp_path)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, value)?;
        writeln!(writer)?;
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
    sync_parent_best_effort(path);
    Ok(())
}

fn write_payload(path: &Path, payload: &Value) -> Result<(), Box<dyn Error>> {
    write_json_pretty(path, payload)
}

fn run_curl(args: &Args) -> Result<(), Box<dyn Error>> {
    ensure_parent(&args.response_path)?;
    let curl_bin = env_string("CURL_BIN", "curl");
    let auth_value = resolved_auth_value(args);
    if args.endpoint.starts_with("https://54.91.17.58") && auth_value.is_none() {
        return Err(invalid_input(
            "DES_RS_AUTH is required for the protected https://54.91.17.58 des-rs endpoint",
        )
        .into());
    }
    let curl_config = curl_config(args, auth_value.as_deref())?;
    let mut command = Command::new(curl_bin);
    command.arg("-fsS").arg("-K").arg("-").stdin(Stdio::piped());
    let mut child = command.spawn()?;
    {
        let stdin = child
            .stdin
            .as_mut()
            .ok_or_else(|| invalid_data("failed to open curl stdin"))?;
        stdin.write_all(curl_config.as_bytes())?;
    }
    let status = child.wait()?;
    if !status.success() {
        return Err(invalid_data(format!("curl exited with status {status}")).into());
    }
    Ok(())
}

fn resolved_auth_value(args: &Args) -> Option<String> {
    args.auth_value
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| args.auth_env_name.as_deref().and_then(env_value))
}

fn curl_config_quote(value: &str) -> Result<String, Box<dyn Error>> {
    if value.contains('\n') || value.contains('\r') {
        return Err(invalid_input("curl config values must not contain newlines").into());
    }
    let escaped = value.replace('\\', "\\\\").replace('"', "\\\"");
    Ok(format!("\"{escaped}\""))
}

fn curl_config(args: &Args, auth_value: Option<&str>) -> Result<String, Box<dyn Error>> {
    let payload_arg = format!("@{}", args.payload_path.display());
    let mut lines = vec![
        format!("url = {}", curl_config_quote(&args.endpoint)?),
        "request = \"POST\"".to_string(),
        "header = \"Content-Type: application/json\"".to_string(),
        format!("data-binary = {}", curl_config_quote(&payload_arg)?),
        format!(
            "output = {}",
            curl_config_quote(&args.response_path.display().to_string())?
        ),
    ];
    if let Some(auth_value) = auth_value {
        lines.push(format!(
            "header = {}",
            curl_config_quote(&format!("{}: {}", args.auth_header_name, auth_value))?
        ));
    }
    lines.push(String::new());
    Ok(lines.join("\n"))
}

fn write_episode_log(path: &Path, episodes: &[Value]) -> Result<(), Box<dyn Error>> {
    ensure_parent(path)?;
    let mut tmp_name = path.as_os_str().to_os_string();
    tmp_name.push(format!(".tmp-{}", std::process::id()));
    let tmp_path = PathBuf::from(tmp_name);
    let result = (|| -> Result<(), Box<dyn Error>> {
        let file = File::create(&tmp_path)?;
        let mut writer = BufWriter::new(file);
        for episode in episodes {
            serde_json::to_writer(&mut writer, episode)?;
            writeln!(writer)?;
        }
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
    sync_parent_best_effort(path);
    Ok(())
}

fn extract_response(args: &Args) -> Result<ExtractedServerResponse, Box<dyn Error>> {
    let raw = fs::read_to_string(&args.response_path)?;
    let response: Value = serde_json::from_str(&raw)?;
    if response.get("ok").and_then(Value::as_bool) == Some(false) {
        return Err(invalid_data(format!(
            "server returned error response: {}",
            response.get("error").unwrap_or(&response)
        ))
        .into());
    }
    let artifact = response
        .get("artifact")
        .filter(|value| value.is_object())
        .ok_or_else(|| invalid_data("server response did not include an artifact object"))?;
    let learned_params = response
        .get("learnedParams")
        .filter(|value| value.is_object())
        .ok_or_else(|| invalid_data("server response did not include learnedParams"))?;
    let artifact_model = serde_json::from_value::<SoccerSelfPlayTrainingArtifact>(artifact.clone())
        .map_err(|err| invalid_data(format!("decode server artifact: {err}")))?;
    let learned_params_model =
        serde_json::from_value::<SoccerSelfPlayLearnedParams>(learned_params.clone())
            .map_err(|err| invalid_data(format!("decode server learnedParams: {err}")))?;

    write_json_pretty(&args.artifact_path, artifact)?;
    write_json_pretty(&args.learned_params_path, learned_params)?;

    let episode_values = artifact
        .get("episodes")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    write_episode_log(&args.episode_log_path, episode_values)?;

    Ok(ExtractedServerResponse {
        episodes: artifact_model.episodes.len(),
        home_entries: artifact_model.home_entries.len(),
        away_entries: artifact_model.away_entries.len(),
        artifact: artifact_model,
        learned_params: learned_params_model,
    })
}

fn import_server_response_to_postgres(
    payload: &Value,
    response: &ExtractedServerResponse,
) -> Result<Option<PostgresImportSummary>, Box<dyn Error>> {
    if !postgres_import_enabled()? {
        return Ok(None);
    }
    let Some(mut store) = SoccerLearningPgStore::connect_from_env()? else {
        return Ok(None);
    };
    let resume = postgres_resume_metadata_from_payload(payload);
    let (experiment_id, parent_policy_version_id, parent_generation) = if let Some(resume) = resume
    {
        (
            resume.experiment_id,
            Some(resume.policy_version_id),
            resume.generation,
        )
    } else {
        let slug =
            env_value("SOCCER_EXPERIMENT_SLUG").unwrap_or_else(|| "soccer-self-play".to_string());
        let display_name =
            env_value("SOCCER_EXPERIMENT_NAME").unwrap_or_else(|| "Soccer self-play".to_string());
        let experiment_id =
            store.ensure_experiment(&slug, &display_name, &response.learned_params.config)?;
        let latest = store.load_latest_active_policy_metadata(&experiment_id)?;
        let parent_policy_version_id = latest.as_ref().map(|metadata| metadata.id.clone());
        let parent_generation = latest.as_ref().map_or(0, |metadata| metadata.generation);
        (experiment_id, parent_policy_version_id, parent_generation)
    };

    let policies = SoccerTeamQPolicies::from_learned_params(&response.learned_params)
        .map_err(|err| invalid_data(format!("restore server learned params policy: {err}")))?;
    let generation = parent_generation.saturating_add(1);
    let latest_active_metadata = store.load_latest_active_policy_metadata(&experiment_id)?;
    let insert_status = soccer_policy_version_insert_status_after_active_head(
        SOCCER_POLICY_STATUS_ACTIVE,
        parent_policy_version_id.as_deref(),
        generation,
        latest_active_metadata
            .as_ref()
            .map(|metadata| metadata.id.as_str()),
        latest_active_metadata
            .as_ref()
            .map(|metadata| metadata.generation),
    );
    let mut config = response.learned_params.config.clone();
    config.tactical_learning = response.learned_params.tactical_learning.clone();
    let run_id = env_value("SOCCER_RUN_ID").unwrap_or_else(|| "server-response".to_string());
    let version_label = postgres_import_version_label(&run_id, response.episodes);
    let policy_version_id = Uuid::new_v4().to_string();
    let fitness = server_response_fitness(&response.artifact);
    let search_metadata = server_import_search_metadata(
        payload,
        response,
        parent_policy_version_id.as_deref(),
        parent_generation,
        generation,
        fitness,
    );

    if insert_status != SOCCER_POLICY_STATUS_ACTIVE {
        println!(
            "postgres_server_policy_marked_stale policy_version={} parent_policy_version={} generation={} latest_active={} latest_generation={}",
            policy_version_id,
            parent_policy_version_id.as_deref().unwrap_or("none"),
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
    store.insert_policy_version_with_id_and_neural_network_and_search_metadata(
        &policy_version_id,
        &experiment_id,
        parent_policy_version_id.as_deref(),
        generation,
        &version_label,
        "server-import",
        insert_status,
        &config,
        response.learned_params.options.clone(),
        response.learned_params.options.clone(),
        &policies,
        fitness,
        response.learned_params.neural_network.as_ref(),
        Some(&search_metadata),
    )?;

    Ok(Some(PostgresImportSummary {
        experiment_id,
        policy_version_id,
        parent_policy_version_id,
        generation,
        status: insert_status,
        fitness,
    }))
}

fn run_server_request(
    args: &Args,
    chunk: ServerRequestChunk,
) -> Result<(ExtractedServerResponse, Option<PostgresImportSummary>), Box<dyn Error>> {
    let payload = build_payload(args, Some(chunk.episodes), Some(chunk.seed))?;
    write_payload(&args.payload_path, &payload)?;
    run_curl(args)?;
    let response = extract_response(args)?;
    let postgres_import = import_server_response_to_postgres(&payload, &response)?;
    Ok((response, postgres_import))
}

fn print_server_response_summary(
    args: &Args,
    chunk: ServerRequestChunk,
    response: &ExtractedServerResponse,
    postgres_import: Option<&PostgresImportSummary>,
) {
    if chunk.total_chunks > 1 {
        println!(
            "server_request_chunk={}/{} episode_offset={} requested_episodes={} seed={}",
            chunk.index + 1,
            chunk.total_chunks,
            chunk.episode_offset,
            chunk.episodes,
            chunk.seed
        );
    }
    println!("server_response={}", args.response_path.display());
    println!("artifact={}", args.artifact_path.display());
    println!("learned_params={}", args.learned_params_path.display());
    println!("episode_log={}", args.episode_log_path.display());
    println!("episodes={}", response.episodes);
    println!("home_entries={}", response.home_entries);
    println!("away_entries={}", response.away_entries);
    if let Some(import) = postgres_import {
        println!(
            "postgres_imported_server_policy experiment={} policy_version={} parent_policy_version={} generation={} status={} fitness={:.4}",
            import.experiment_id,
            import.policy_version_id,
            import.parent_policy_version_id.as_deref().unwrap_or("none"),
            import.generation,
            import.status,
            import.fitness
        );
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let total_episodes = env_usize("SOCCER_GAMES", 100)?;
    let base_seed = env_u32("SOCCER_SEED", 2026)?;
    let postgres_url_present = postgres_database_url_present();
    let postgres_resume_enabled = postgres_resume_enabled()?;
    let default_request_games = default_server_request_games(
        total_episodes,
        postgres_url_present,
        postgres_resume_enabled,
    );
    let requested_episodes_per_request = env_usize_alias(
        "SOCCER_SERVER_REQUEST_GAMES",
        "SOCCER_SERVER_CHUNK_GAMES",
        default_request_games,
    )?;
    let allow_postgres_multi_game_chunks =
        env_bool(ALLOW_POSTGRES_MULTI_GAME_SERVER_CHUNKS_ENV, false)?;
    let require_fresh_postgres_weights = env_bool(REQUIRE_FRESH_POSTGRES_WEIGHTS_ENV, true)?;
    let episodes_per_request = effective_server_request_games(
        requested_episodes_per_request,
        postgres_url_present,
        postgres_resume_enabled,
        allow_postgres_multi_game_chunks,
        require_fresh_postgres_weights,
    );
    if episodes_per_request != requested_episodes_per_request {
        println!(
            "server_request_games_clamped_for_postgres_refresh requested={} effective={} set {}=0 and {}=1 to allow multi-game chunks",
            requested_episodes_per_request,
            episodes_per_request,
            REQUIRE_FRESH_POSTGRES_WEIGHTS_ENV,
            ALLOW_POSTGRES_MULTI_GAME_SERVER_CHUNKS_ENV
        );
    }
    let chunks = server_request_chunks(total_episodes, episodes_per_request, base_seed)?;
    if chunks.len() > 1 {
        println!(
            "server_request_chunking total_episodes={} request_episodes={} chunks={}",
            total_episodes,
            episodes_per_request,
            chunks.len()
        );
    }

    let mut total_response_episodes = 0usize;
    for chunk in chunks.iter().copied() {
        let chunk_args = args_for_server_chunk(&args, chunk);
        let (response, postgres_import) = run_server_request(&chunk_args, chunk)?;
        total_response_episodes = total_response_episodes.saturating_add(response.episodes);
        print_server_response_summary(&chunk_args, chunk, &response, postgres_import.as_ref());
    }
    if chunks.len() > 1 {
        println!("server_request_total_episodes={total_response_episodes}");
    }
    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("main_soccer_learning_server: {err}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use soccer_engine::des::general::soccer::{MatchSummary, SoccerSelfPlayEpisodeSummary};

    fn episode_summary(
        episode: usize,
        score_home: u32,
        score_away: u32,
    ) -> SoccerSelfPlayEpisodeSummary {
        SoccerSelfPlayEpisodeSummary {
            episode,
            seed: 9000 + episode as u64,
            summary: MatchSummary {
                score_home,
                score_away,
                ticks: 10,
                simulated_seconds: 1.0,
                stats: Default::default(),
            },
            transitions: 0,
            home_policy_entries: 0,
            home_policy_target_entries: 0,
            away_policy_entries: 0,
            away_policy_target_entries: 0,
        }
    }

    #[test]
    fn payload_settings_reject_dt_above_soccer_runtime_limit() {
        assert!(validate_payload_settings(1, 90.0, 2, 900.0, 4.0, 4, 0.12, 0.96, &[]).is_ok());
        let err = validate_payload_settings(1, 90.0, 2, 900.0, 4.01, 4, 0.12, 0.96, &[])
            .expect_err("dt above runtime-safe bound should fail");
        assert!(err.to_string().contains("[0.01, 4.0]"));
    }

    #[test]
    fn postgres_resume_metadata_reads_payload() {
        let payload = json!({
            "postgresResume": {
                "experimentId": "exp-1",
                "policyVersionId": "policy-7",
                "generation": 12
            }
        });

        assert_eq!(
            postgres_resume_metadata_from_payload(&payload),
            Some(PostgresResumeMetadata {
                experiment_id: "exp-1".to_string(),
                policy_version_id: "policy-7".to_string(),
                generation: 12,
            })
        );
    }

    #[test]
    fn postgres_import_version_label_is_bounded_and_sanitized() {
        let label = postgres_import_version_label(&"bad run !".repeat(40), 123);

        assert!(label.len() <= 160);
        assert!(label.ends_with("-server-e000123"));
        assert!(!label.contains(' '));
        assert!(!label.contains('!'));
    }

    #[test]
    fn default_server_request_games_refreshes_each_episode_when_postgres_resume_is_live() {
        assert_eq!(default_server_request_games(100, true, true), 1);
        assert_eq!(default_server_request_games(1, true, true), 1);
        assert_eq!(default_server_request_games(100, false, true), 100);
        assert_eq!(default_server_request_games(100, true, false), 100);
    }

    #[test]
    fn effective_server_request_games_clamps_manual_override_for_postgres_resume() {
        assert_eq!(
            effective_server_request_games(25, true, true, false, true),
            1
        );
        assert_eq!(
            effective_server_request_games(25, true, true, true, true),
            1
        );
        assert_eq!(
            effective_server_request_games(25, true, true, false, false),
            1
        );
        assert_eq!(
            effective_server_request_games(25, true, true, true, false),
            25
        );
        assert_eq!(
            effective_server_request_games(25, false, true, false, true),
            25
        );
        assert_eq!(
            effective_server_request_games(25, true, false, false, true),
            25
        );
    }

    #[test]
    fn postgres_resume_request_planner_builds_one_episode_chunks_for_fresh_weights() {
        let episodes_per_request = effective_server_request_games(25, true, true, true, true);
        let chunks = server_request_chunks(100, episodes_per_request, 2026).expect("chunks");

        assert_eq!(episodes_per_request, 1);
        assert_eq!(chunks.len(), 100);
        assert!(chunks.iter().all(|chunk| chunk.episodes == 1));
        assert_eq!(chunks.first().expect("first chunk").seed, 2026);
        assert_eq!(chunks.last().expect("last chunk").seed, 2125);
    }

    #[test]
    fn server_request_chunks_split_episodes_and_advance_seed() {
        let chunks = server_request_chunks(100, 25, 2026).expect("chunks");

        assert_eq!(chunks.len(), 4);
        assert_eq!(
            chunks,
            vec![
                ServerRequestChunk {
                    index: 0,
                    total_chunks: 4,
                    episode_offset: 0,
                    episodes: 25,
                    seed: 2026,
                },
                ServerRequestChunk {
                    index: 1,
                    total_chunks: 4,
                    episode_offset: 25,
                    episodes: 25,
                    seed: 2051,
                },
                ServerRequestChunk {
                    index: 2,
                    total_chunks: 4,
                    episode_offset: 50,
                    episodes: 25,
                    seed: 2076,
                },
                ServerRequestChunk {
                    index: 3,
                    total_chunks: 4,
                    episode_offset: 75,
                    episodes: 25,
                    seed: 2101,
                },
            ]
        );
    }

    #[test]
    fn server_request_chunks_keep_tail_size() {
        let chunks = server_request_chunks(103, 25, 7).expect("chunks");

        assert_eq!(chunks.len(), 5);
        assert_eq!(chunks.last().expect("tail").episode_offset, 100);
        assert_eq!(chunks.last().expect("tail").episodes, 3);
        assert_eq!(chunks.last().expect("tail").seed, 107);
    }

    #[test]
    fn server_request_chunks_reject_zero_size() {
        assert!(server_request_chunks(100, 0, 2026).is_err());
        assert!(server_request_chunks(0, 10, 2026).is_err());
    }

    #[test]
    fn server_chunk_args_suffix_intermediate_files_but_keep_final_outputs() {
        let args = Args {
            endpoint: "https://example.invalid/des-rs/api/train-self-play".to_string(),
            payload_path: PathBuf::from("out/run/payload.json"),
            response_path: PathBuf::from("out/run/response.json"),
            artifact_path: PathBuf::from("out/run/artifact.json"),
            learned_params_path: PathBuf::from("out/run/learned-params.json"),
            episode_log_path: PathBuf::from("out/run/episodes.jsonl"),
            server_artifact_path: "out/server/artifact.json".to_string(),
            server_learned_params_path: "out/server/learned-params.json".to_string(),
            auth_header_name: "Auth".to_string(),
            auth_env_name: Some("DES_RS_AUTH".to_string()),
            auth_value: None,
        };
        let first = ServerRequestChunk {
            index: 0,
            total_chunks: 3,
            episode_offset: 0,
            episodes: 10,
            seed: 11,
        };
        let final_chunk = ServerRequestChunk {
            index: 2,
            total_chunks: 3,
            episode_offset: 20,
            episodes: 10,
            seed: 31,
        };

        let first_args = args_for_server_chunk(&args, first);
        let final_args = args_for_server_chunk(&args, final_chunk);

        assert_eq!(
            first_args.payload_path,
            PathBuf::from("out/run/payload.chunk-0001-of-0003.json")
        );
        assert_eq!(
            first_args.server_artifact_path,
            "out/server/artifact.chunk-0001-of-0003.json"
        );
        assert_eq!(final_args.payload_path, args.payload_path);
        assert_eq!(final_args.artifact_path, args.artifact_path);
        assert_eq!(final_args.server_artifact_path, args.server_artifact_path);
    }

    #[test]
    fn server_response_fitness_averages_episode_scores() {
        let artifact = SoccerSelfPlayTrainingArtifact {
            config: MatchConfig::default(),
            options: SoccerQPolicyOptions::default(),
            tactical_learning: SoccerTacticalLearningWeights::default(),
            tactical_summary: SoccerTacticalLearningSummary::default(),
            episodes: vec![episode_summary(0, 2, 0), episode_summary(1, 0, 1)],
            home_entries: Vec::new(),
            home_target_entries: Vec::new(),
            away_entries: Vec::new(),
            away_target_entries: Vec::new(),
        };
        let expected = artifact
            .episodes
            .iter()
            .map(|episode| soccer_learning_run_score(&episode.summary).match_fitness)
            .sum::<f64>()
            / artifact.episodes.len() as f64;

        assert!((server_response_fitness(&artifact) - expected).abs() < 1e-12);
    }

    #[test]
    fn server_import_search_metadata_records_resume_and_learning_shape() {
        let artifact = SoccerSelfPlayTrainingArtifact {
            config: MatchConfig::default(),
            options: SoccerQPolicyOptions::default(),
            tactical_learning: SoccerTacticalLearningWeights::default(),
            tactical_summary: SoccerTacticalLearningSummary::default(),
            episodes: vec![episode_summary(0, 1, 0)],
            home_entries: Vec::new(),
            home_target_entries: Vec::new(),
            away_entries: Vec::new(),
            away_target_entries: Vec::new(),
        };
        let learned_params = SoccerSelfPlayLearnedParams::from_training_artifact(&artifact);
        let response = ExtractedServerResponse {
            episodes: 1,
            home_entries: 4,
            away_entries: 3,
            artifact,
            learned_params,
        };
        let payload = json!({
            "episodes": 1,
            "seed": 2026,
            "minutes": 0.5,
            "dtSeconds": 0.2,
            "learningIntervalTicks": 4,
            "formationLpEnabled": true,
            "artifactPath": "out/server/artifact.json",
            "learnedParamsPath": "out/server/learned-params.json",
            "initialLearnedParams": {},
            "postgresResume": {
                "experimentId": "exp-1",
                "policyVersionId": "policy-7",
                "generation": 12,
                "policyFingerprint": 1234567_u64,
                "tacticalLearningFingerprint": 7654321_u64,
                "neuralNetworkFingerprint": null,
                "searchParameters": {
                    "algorithm": "evolutionary-genetic-programming-tactical-search",
                    "options": {
                        "mutationRate": 0.09,
                        "populationSize": 18
                    }
                }
            }
        });
        let metadata =
            server_import_search_metadata(&payload, &response, Some("policy-7"), 12, 13, 0.75);

        assert_eq!(
            metadata["algorithm"],
            json!("server-mdp-pomdp-neural-self-play")
        );
        assert_eq!(metadata["request"]["episodes"], json!(1));
        assert_eq!(metadata["request"]["seed"], json!(2026));
        assert_eq!(
            metadata["request"]["initialLearnedParamsIncluded"],
            json!(true)
        );
        assert_eq!(
            metadata["postgresResume"]["policyVersionId"],
            json!("policy-7")
        );
        assert_eq!(
            metadata["postgresResume"]["policyFingerprint"],
            json!(1234567_u64)
        );
        assert_eq!(
            metadata["postgresResume"]["tacticalLearningFingerprint"],
            json!(7654321_u64)
        );
        assert_eq!(
            metadata["postgresResume"]["searchParameters"]["algorithm"],
            json!("evolutionary-genetic-programming-tactical-search")
        );
        assert_eq!(
            metadata["postgresResume"]["searchParameters"]["options"]["populationSize"],
            json!(18)
        );
        assert_eq!(metadata["response"]["generation"], json!(13));
        assert_eq!(metadata["response"]["fitness"], json!(0.75));
        assert_eq!(
            metadata["learningComponents"]["mdpPolicyEntries"],
            json!(true)
        );
        assert_eq!(
            metadata["learningComponents"]["tacticalLearningWeights"],
            json!(true)
        );
    }
}
