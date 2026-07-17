//! Blocking live server for the 2D soccer simulation.

use crate::des::general::soccer::{
    run_live_soccer_server, SoccerLiveServerConfig, SoccerNeuralBlendMode,
    SoccerNeuralLearningBackend,
};

pub const SOCCER_LIVE_REVIVAL_PORT: u16 = 5056;
const LIVE_MPC_DEFAULT_PLAYER_HORIZON: usize = 12;
const LIVE_MPC_MAX_PLAYER_HORIZON: usize = 45;
const LIVE_MPC_DEFAULT_ACTIVE_RADIUS_YARDS: f64 = 5.0;
const LIVE_LOCAL_MPC_DEFAULT_ENABLED: bool = false;
const LIVE_LOCAL_MPC_DEFAULT_MAX_PLAYERS_PER_TEAM: usize = 1;
const LIVE_LOCAL_MPC_MAX_PLAYERS_PER_TEAM_LIMIT: usize = 11;

fn env_value<F>(lookup: &F, primary: &str, fallback: &str) -> Option<String>
where
    F: Fn(&str) -> Option<String>,
{
    lookup(primary).or_else(|| lookup(fallback))
}

fn env_text<F>(lookup: &F, primary: &str, fallback: &str) -> Option<String>
where
    F: Fn(&str) -> Option<String>,
{
    env_value(lookup, primary, fallback).filter(|raw| !raw.trim().is_empty())
}

fn env_positive_f64<F>(lookup: &F, primary: &str, fallback: &str) -> Option<f64>
where
    F: Fn(&str) -> Option<String>,
{
    env_value(lookup, primary, fallback)
        .and_then(|raw| raw.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
}

fn env_positive_u64<F>(lookup: &F, primary: &str, fallback: &str) -> Option<u64>
where
    F: Fn(&str) -> Option<String>,
{
    env_value(lookup, primary, fallback)
        .and_then(|raw| raw.parse::<u64>().ok())
        .filter(|value| *value > 0)
}

fn env_nonnegative_u64<F>(lookup: &F, primary: &str, fallback: &str) -> Option<u64>
where
    F: Fn(&str) -> Option<String>,
{
    env_value(lookup, primary, fallback).and_then(|raw| raw.parse::<u64>().ok())
}

fn env_positive_usize<F>(lookup: &F, primary: &str, fallback: &str) -> Option<usize>
where
    F: Fn(&str) -> Option<String>,
{
    env_value(lookup, primary, fallback)
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|value| *value > 0)
}

fn env_nonnegative_usize<F>(lookup: &F, primary: &str, fallback: &str) -> Option<usize>
where
    F: Fn(&str) -> Option<String>,
{
    env_value(lookup, primary, fallback).and_then(|raw| raw.parse::<usize>().ok())
}

fn env_nonnegative_f64<F>(lookup: &F, primary: &str, fallback: &str) -> Option<f64>
where
    F: Fn(&str) -> Option<String>,
{
    env_value(lookup, primary, fallback)
        .and_then(|raw| raw.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0)
}

fn env_bool<F>(lookup: &F, primary: &str, fallback: &str) -> Option<bool>
where
    F: Fn(&str) -> Option<String>,
{
    env_value(lookup, primary, fallback).and_then(|raw| {
        match raw.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        }
    })
}

fn env_neural_backend<F>(
    lookup: &F,
    primary: &str,
    fallback: &str,
) -> Option<SoccerNeuralLearningBackend>
where
    F: Fn(&str) -> Option<String>,
{
    env_value(lookup, primary, fallback).and_then(|raw| {
        match raw.trim().to_ascii_lowercase().as_str() {
            "inline" => Some(SoccerNeuralLearningBackend::Inline),
            "threaded" | "thread" | "background" => Some(SoccerNeuralLearningBackend::Threaded),
            _ => None,
        }
    })
}

fn env_neural_blend_mode<F>(
    lookup: &F,
    primary: &str,
    fallback: &str,
) -> Option<SoccerNeuralBlendMode>
where
    F: Fn(&str) -> Option<String>,
{
    env_value(lookup, primary, fallback).and_then(|raw| {
        match raw.trim().to_ascii_lowercase().as_str() {
            "off" | "disabled" | "none" => Some(SoccerNeuralBlendMode::Off),
            "additive" | "add" => Some(SoccerNeuralBlendMode::Additive),
            "tiebreak" | "tie" | "tie-break" | "tie_break" => Some(SoccerNeuralBlendMode::TieBreak),
            "confidence" | "confidencegated" | "confidence-gated" | "gated" => {
                Some(SoccerNeuralBlendMode::ConfidenceGated)
            }
            "authoritative" | "neural" | "neural-authoritative" | "neural_authoritative" => {
                Some(SoccerNeuralBlendMode::Authoritative)
            }
            _ => None,
        }
    })
}

fn live_server_config_from_lookup<F>(lookup: F) -> SoccerLiveServerConfig
where
    F: Fn(&str) -> Option<String>,
{
    live_server_config_from_lookup_with_default_port(lookup, None)
}

fn live_server_config_from_lookup_with_default_port<F>(
    lookup: F,
    default_port: Option<u16>,
) -> SoccerLiveServerConfig
where
    F: Fn(&str) -> Option<String>,
{
    let mut cfg = SoccerLiveServerConfig::default();
    cfg.match_config.mpc.player_horizon = LIVE_MPC_DEFAULT_PLAYER_HORIZON;
    cfg.match_config.mpc.active_radius_yards = LIVE_MPC_DEFAULT_ACTIVE_RADIUS_YARDS;
    cfg.match_config.local_mpc_enabled = LIVE_LOCAL_MPC_DEFAULT_ENABLED;
    cfg.match_config.local_mpc_max_players_per_team = LIVE_LOCAL_MPC_DEFAULT_MAX_PLAYERS_PER_TEAM;
    if let Some(host) = env_text(&lookup, "SOCCER_LIVE_HOST", "SOCCER_HOST") {
        cfg.host = host;
    }
    match env_value(&lookup, "SOCCER_LIVE_PORT", "SOCCER_PORT")
        .and_then(|raw| raw.parse::<u16>().ok())
    {
        Some(port) => cfg.port = port,
        None => {
            if let Some(port) = default_port {
                cfg.port = port;
            }
        }
    }
    if let Some(workers) =
        env_positive_usize(&lookup, "SOCCER_LIVE_HTTP_WORKERS", "SOCCER_HTTP_WORKERS")
    {
        cfg.http_worker_threads = workers;
    }
    if let Some(duration) = env_positive_f64(
        &lookup,
        "SOCCER_MATCH_DURATION_SECONDS",
        "SOCCER_DURATION_SECONDS",
    ) {
        cfg.match_config.duration_seconds = duration;
    }
    if let Some(dt) = env_positive_f64(&lookup, "SOCCER_MATCH_DT_SECONDS", "SOCCER_DT_SECONDS") {
        cfg.match_config.dt_seconds = dt;
    }
    if let Some(period_count) = env_positive_usize(&lookup, "SOCCER_MATCH_HALVES", "SOCCER_HALVES")
    {
        cfg.match_config.period_count = period_count;
    }
    if let Some(period_break_recovery_seconds) = env_nonnegative_f64(
        &lookup,
        "SOCCER_MATCH_PERIOD_BREAK_RECOVERY_SECONDS",
        "SOCCER_PERIOD_BREAK_RECOVERY_SECONDS",
    ) {
        cfg.match_config.period_break_recovery_seconds = period_break_recovery_seconds;
    }
    if let Some(learning_enabled) = env_bool(
        &lookup,
        "SOCCER_LIVE_LEARNING_ENABLED",
        "SOCCER_LEARNING_ENABLED",
    ) {
        cfg.match_config.learning_enabled = learning_enabled;
    }
    if let Some(logging_enabled) = env_bool(
        &lookup,
        "SOCCER_LIVE_LEARNING_LOGGING_ENABLED",
        "SOCCER_LEARNING_LOGGING_ENABLED",
    ) {
        cfg.match_config.learning_logging_enabled = logging_enabled;
    }
    if let Some(interval) = env_positive_usize(
        &lookup,
        "SOCCER_LIVE_LEARNING_INTERVAL_TICKS",
        "SOCCER_LEARNING_INTERVAL_TICKS",
    ) {
        cfg.match_config.learning_interval_ticks = interval;
    }
    if let Some(full_game_enabled) = env_bool(
        &lookup,
        "SOCCER_LIVE_FULL_GAME_LEARNING_ENABLED",
        "SOCCER_FULL_GAME_LEARNING_ENABLED",
    ) {
        cfg.match_config.full_game_learning_enabled = full_game_enabled;
    }
    if let Some(neural_enabled) = env_bool(
        &lookup,
        "SOCCER_LIVE_NEURAL_LEARNING_ENABLED",
        "SOCCER_NEURAL_LEARNING_ENABLED",
    ) {
        cfg.match_config.neural_learning.enabled = neural_enabled;
    }
    if let Some(backend) = env_neural_backend(
        &lookup,
        "SOCCER_LIVE_NEURAL_LEARNING_BACKEND",
        "SOCCER_NEURAL_LEARNING_BACKEND",
    ) {
        cfg.match_config.neural_learning.backend = backend;
    }
    if let Some(learning_rate) = env_positive_f64(
        &lookup,
        "SOCCER_LIVE_NEURAL_LEARNING_RATE",
        "SOCCER_NEURAL_LEARNING_RATE",
    ) {
        cfg.match_config.neural_learning.learning_rate = learning_rate;
    }
    if let Some(batch_size) = env_positive_usize(
        &lookup,
        "SOCCER_LIVE_NEURAL_BATCH_SIZE",
        "SOCCER_NEURAL_BATCH_SIZE",
    ) {
        cfg.match_config.neural_learning.batch_size = batch_size;
    }
    if let Some(train_every_ticks) = env_positive_usize(
        &lookup,
        "SOCCER_LIVE_NEURAL_TRAIN_EVERY_TICKS",
        "SOCCER_NEURAL_TRAIN_EVERY_TICKS",
    ) {
        cfg.match_config.neural_learning.train_every_ticks = train_every_ticks;
    }
    if let Some(max_batches_per_tick) = env_positive_usize(
        &lookup,
        "SOCCER_LIVE_NEURAL_MAX_BATCHES_PER_TICK",
        "SOCCER_NEURAL_MAX_BATCHES_PER_TICK",
    ) {
        cfg.match_config.neural_learning.max_batches_per_tick = max_batches_per_tick;
    }
    if let Some(hidden_units) = env_positive_usize(
        &lookup,
        "SOCCER_LIVE_NEURAL_HIDDEN_UNITS",
        "SOCCER_NEURAL_HIDDEN_UNITS",
    ) {
        cfg.match_config.neural_learning.hidden_units = hidden_units;
    }
    if let Some(target_scale) = env_positive_f64(
        &lookup,
        "SOCCER_LIVE_NEURAL_TARGET_SCALE",
        "SOCCER_NEURAL_TARGET_SCALE",
    ) {
        cfg.match_config.neural_learning.target_scale = target_scale;
    }
    if let Some(target_clip) = env_positive_f64(
        &lookup,
        "SOCCER_LIVE_NEURAL_TARGET_CLIP",
        "SOCCER_NEURAL_TARGET_CLIP",
    ) {
        cfg.match_config.neural_learning.target_clip = target_clip;
    }
    if let Some(max_pending_batches) = env_positive_usize(
        &lookup,
        "SOCCER_LIVE_NEURAL_MAX_PENDING_BATCHES",
        "SOCCER_NEURAL_MAX_PENDING_BATCHES",
    ) {
        cfg.match_config.neural_learning.max_pending_batches = max_pending_batches;
    }
    if let Some(replay_capacity) = env_positive_usize(
        &lookup,
        "SOCCER_LIVE_NEURAL_REPLAY_CAPACITY",
        "SOCCER_NEURAL_REPLAY_CAPACITY",
    ) {
        cfg.match_config.neural_learning.replay_capacity = replay_capacity;
    }
    if let Some(replay_samples_per_tick) = env_positive_usize(
        &lookup,
        "SOCCER_LIVE_NEURAL_REPLAY_SAMPLES_PER_TICK",
        "SOCCER_NEURAL_REPLAY_SAMPLES_PER_TICK",
    ) {
        cfg.match_config.neural_learning.replay_samples_per_tick = replay_samples_per_tick;
    }
    if let Some(snapshot_every_batches) = env_positive_usize(
        &lookup,
        "SOCCER_LIVE_NEURAL_SNAPSHOT_EVERY_BATCHES",
        "SOCCER_NEURAL_SNAPSHOT_EVERY_BATCHES",
    ) {
        cfg.match_config.neural_learning.snapshot_every_batches = snapshot_every_batches;
    }
    if let Some(target_popart) = env_bool(
        &lookup,
        "SOCCER_LIVE_NEURAL_TARGET_POPART",
        "SOCCER_NEURAL_TARGET_POPART",
    ) {
        cfg.match_config.neural_learning.target_popart_enabled = target_popart;
    }
    if let Some(lp_coupling) = env_bool(
        &lookup,
        "SOCCER_LIVE_NEURAL_LP_COUPLING_ENABLED",
        "SOCCER_NEURAL_LP_COUPLING_ENABLED",
    ) {
        cfg.match_config.neural_learning.lp_coupling_enabled = lp_coupling;
    }
    if let Some(mappo_team_reward_share) = env_nonnegative_f64(
        &lookup,
        "SOCCER_LIVE_MAPPO_TEAM_REWARD_SHARE",
        "SOCCER_MAPPO_TEAM_REWARD_SHARE",
    ) {
        cfg.match_config.neural_learning.mappo_team_reward_share =
            mappo_team_reward_share.clamp(0.0, 1.0);
    }
    if let Some(marl_team_reward_weight) = env_nonnegative_f64(
        &lookup,
        "SOCCER_LIVE_MARL_TEAM_REWARD_WEIGHT",
        "SOCCER_MARL_TEAM_REWARD_WEIGHT",
    ) {
        cfg.match_config.neural_learning.marl_team_reward_weight = marl_team_reward_weight;
    }
    if let Some(marl_intermediate_reward_weight) = env_nonnegative_f64(
        &lookup,
        "SOCCER_LIVE_MARL_INTERMEDIATE_REWARD_WEIGHT",
        "SOCCER_MARL_INTERMEDIATE_REWARD_WEIGHT",
    ) {
        cfg.match_config
            .neural_learning
            .marl_intermediate_reward_weight = marl_intermediate_reward_weight;
    }
    if let Some(mappo_clip_epsilon) = env_nonnegative_f64(
        &lookup,
        "SOCCER_LIVE_MAPPO_CLIP_EPSILON",
        "SOCCER_MAPPO_CLIP_EPSILON",
    ) {
        cfg.match_config.neural_learning.mappo_clip_epsilon = mappo_clip_epsilon;
    }
    if let Some(mode) = env_neural_blend_mode(
        &lookup,
        "SOCCER_LIVE_NEURAL_BLEND_MODE",
        "SOCCER_NEURAL_BLEND_MODE",
    ) {
        cfg.match_config.neural_blend.mode = mode;
    }
    if let Some(lambda) = env_positive_f64(
        &lookup,
        "SOCCER_LIVE_NEURAL_BLEND_LAMBDA",
        "SOCCER_NEURAL_BLEND_LAMBDA",
    ) {
        cfg.match_config.neural_blend.lambda = lambda;
    }
    if let Some(warmup_steps) = env_nonnegative_usize(
        &lookup,
        "SOCCER_LIVE_NEURAL_BLEND_WARMUP_STEPS",
        "SOCCER_NEURAL_BLEND_WARMUP_STEPS",
    ) {
        cfg.match_config.neural_blend.warmup_steps = warmup_steps;
    }
    if let Some(candidates) = env_positive_usize(
        &lookup,
        "SOCCER_LIVE_NEURAL_BLEND_CANDIDATES",
        "SOCCER_NEURAL_BLEND_CANDIDATES",
    ) {
        cfg.match_config.neural_blend.candidates = candidates;
    }
    if let Some(actor_critic) = env_bool(
        &lookup,
        "SOCCER_LIVE_NEURAL_ACTOR_CRITIC",
        "SOCCER_NEURAL_ACTOR_CRITIC",
    ) {
        cfg.match_config.neural_blend.actor_critic = actor_critic;
    }
    if let Some(world_model) = env_bool(
        &lookup,
        "SOCCER_LIVE_NEURAL_WORLD_MODEL",
        "SOCCER_NEURAL_WORLD_MODEL",
    ) {
        cfg.match_config.neural_blend.world_model = world_model;
    }
    if let Some(mcts_enabled) = env_bool(
        &lookup,
        "SOCCER_LIVE_NEURAL_MCTS_ENABLED",
        "SOCCER_NEURAL_MCTS_ENABLED",
    ) {
        cfg.match_config.neural_blend.mcts_enabled = mcts_enabled;
    }
    if let Some(mcts_simulations) = env_positive_usize(
        &lookup,
        "SOCCER_LIVE_NEURAL_MCTS_SIMULATIONS",
        "SOCCER_NEURAL_MCTS_SIMULATIONS",
    ) {
        cfg.match_config.neural_blend.mcts_simulations = mcts_simulations;
    }
    if let Some(mcts_candidates) = env_positive_usize(
        &lookup,
        "SOCCER_LIVE_NEURAL_MCTS_CANDIDATES",
        "SOCCER_NEURAL_MCTS_CANDIDATES",
    ) {
        cfg.match_config.neural_blend.mcts_candidates = mcts_candidates;
    }
    if let Some(mcts_depth) = env_positive_usize(
        &lookup,
        "SOCCER_LIVE_NEURAL_MCTS_DEPTH",
        "SOCCER_NEURAL_MCTS_DEPTH",
    ) {
        cfg.match_config.neural_blend.mcts_depth = mcts_depth;
    }
    if let Some(adversarial_enabled) = env_bool(
        &lookup,
        "SOCCER_LIVE_ADVERSARIAL_EMBEDDING_ENABLED",
        "SOCCER_ADVERSARIAL_EMBEDDING_ENABLED",
    ) {
        cfg.match_config.adversarial_embedding_exploitation_enabled = adversarial_enabled;
    }
    if let Some(path) = env_text(&lookup, "SOCCER_LIVE_POLICY_PATH", "SOCCER_POLICY_PATH") {
        cfg.policy_disk_path = path;
    }
    if let Some(path) = env_text(&lookup, "SOCCER_LIVE_MOMENT_PATH", "SOCCER_MOMENT_PATH") {
        cfg.moment_disk_path = path;
    }
    if let Some(autoload) = env_bool(
        &lookup,
        "SOCCER_LIVE_AUTOLOAD_POLICY",
        "SOCCER_AUTOLOAD_POLICY",
    ) {
        cfg.autoload_team_policy = autoload;
    }
    // MPC execution layer: when enabled, the per-player projected-gradient QP
    // (`PlanarPointMassMpc`) plans each active player's next-tick velocity, is made
    // field-aware (other players + ball as moving keep-outs), and is reconciled/blended
    // with the neural-policy velocity. The separate formation-local MPC is much more
    // expensive in the live HTTP loop, so it stays opt-in and capped below.
    let live_mpc = env_bool(&lookup, "SOCCER_LIVE_MPC", "SOCCER_MPC_ENABLED");
    if let Some(mpc) = live_mpc {
        cfg.match_config.mpc.tier2_player_enabled = mpc;
        cfg.match_config.mpc.field_aware_enabled = mpc;
        cfg.match_config.mpc.reconcile_enabled = mpc;
        if mpc {
            println!(
                "# soccer-live: MPC execution ENABLED — projected-gradient QP plans each active player's velocity (field-aware, reconciled with the neural policy)"
            );
        }
    }
    if let Some(local_mpc) = env_bool(&lookup, "SOCCER_LIVE_LOCAL_MPC", "SOCCER_LOCAL_MPC") {
        cfg.match_config.local_mpc_enabled = local_mpc;
    }
    if let Some(max_players) = env_nonnegative_usize(
        &lookup,
        "SOCCER_LIVE_LOCAL_MPC_MAX_PLAYERS_PER_TEAM",
        "SOCCER_LOCAL_MPC_MAX_PLAYERS_PER_TEAM",
    ) {
        cfg.match_config.local_mpc_max_players_per_team =
            max_players.min(LIVE_LOCAL_MPC_MAX_PLAYERS_PER_TEAM_LIMIT);
    }
    if cfg.match_config.local_mpc_max_players_per_team == 0 || live_mpc == Some(false) {
        cfg.match_config.local_mpc_enabled = false;
    }
    if cfg.match_config.local_mpc_enabled {
        println!(
            "# soccer-live: formation-local MPC enabled for up to {} player(s)/team",
            cfg.match_config.local_mpc_max_players_per_team
        );
    }
    if let Some(horizon) = env_positive_usize(
        &lookup,
        "SOCCER_LIVE_MPC_PLAYER_HORIZON",
        "SOCCER_MPC_PLAYER_HORIZON",
    ) {
        cfg.match_config.mpc.player_horizon = horizon.min(LIVE_MPC_MAX_PLAYER_HORIZON);
    }
    // How close to the ball a player must be to run the per-player MPC (the "active
    // subset"). The live default is tighter than the base match config because this
    // path runs inside the HTTP frame loop; raise it only for profiling/heavier runs.
    if let Some(radius) = env_positive_f64(
        &lookup,
        "SOCCER_LIVE_MPC_ACTIVE_RADIUS",
        "SOCCER_MPC_ACTIVE_RADIUS",
    ) {
        cfg.match_config.mpc.active_radius_yards = radius;
        println!(
            "# soccer-live: MPC active radius set to {radius:.0}yd (all players within run the QP)"
        );
    }
    if let Some(max_bytes) = env_nonnegative_u64(
        &lookup,
        "SOCCER_LIVE_POLICY_AUTOLOAD_MAX_BYTES",
        "SOCCER_POLICY_AUTOLOAD_MAX_BYTES",
    ) {
        cfg.policy_autoload_max_bytes = max_bytes;
    }
    if let Some(autosave) = env_bool(
        &lookup,
        "SOCCER_LIVE_AUTOSAVE_POLICY",
        "SOCCER_AUTOSAVE_POLICY",
    ) {
        cfg.autosave_team_policy = autosave;
    }
    if let Some(interval) = env_positive_u64(
        &lookup,
        "SOCCER_LIVE_POLICY_AUTOSAVE_INTERVAL_TICKS",
        "SOCCER_POLICY_AUTOSAVE_INTERVAL_TICKS",
    ) {
        cfg.policy_autosave_interval_ticks = interval;
    }
    if let Some(keep_best) = env_bool(
        &lookup,
        "SOCCER_LIVE_POLICY_KEEP_BEST",
        "SOCCER_POLICY_KEEP_BEST",
    ) {
        cfg.policy_keep_best = keep_best;
    }
    cfg
}

fn live_server_config_from_env() -> SoccerLiveServerConfig {
    live_server_config_from_lookup(|name| std::env::var(name).ok())
}

fn live_server_config_from_env_with_default_port(default_port: u16) -> SoccerLiveServerConfig {
    live_server_config_from_lookup_with_default_port(
        |name| std::env::var(name).ok(),
        Some(default_port),
    )
}

pub fn run() {
    run_live_soccer_server(live_server_config_from_env()).expect("run live soccer server");
}

pub fn run_with_default_port(default_port: u16) {
    run_live_soccer_server(live_server_config_from_env_with_default_port(default_port))
        .expect("run live soccer server");
}

pub fn run_5056() {
    run_with_default_port(SOCCER_LIVE_REVIVAL_PORT);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::general::soccer::{DEFAULT_DT_SECONDS, DEFAULT_DURATION_SECONDS};
    use std::collections::BTreeMap;

    #[test]
    fn live_server_env_defaults_preserve_gameplay_learning_mode() {
        let cfg = live_server_config_from_lookup(|_| None);

        assert_eq!(cfg.match_config.human_slots(), 4);
        assert_eq!(cfg.http_worker_threads, 4);
        assert_eq!(cfg.match_config.dt_seconds, DEFAULT_DT_SECONDS);
        assert_eq!(cfg.match_config.duration_seconds, DEFAULT_DURATION_SECONDS);
        // Derive the expected tick count from the default duration/dt (single period) rather
        // than pinning a magic number, so a future duration/dt tweak doesn't falsely fail here.
        assert_eq!(
            cfg.match_config.total_ticks(),
            (DEFAULT_DURATION_SECONDS / DEFAULT_DT_SECONDS).round() as u64,
        );
        // The live demo runs a FROZEN trained policy: the neural snapshot is loaded and
        // used for decisions, but there is no online training (that happens in headless
        // self-play / tournaments). So learning + its logging are off; the net stays on.
        assert!(!cfg.match_config.learning_enabled);
        assert!(!cfg.match_config.learning_logging_enabled);
        assert!(cfg.match_config.neural_learning.enabled);
        assert_eq!(cfg.policy_autoload_max_bytes, 768 * 1024 * 1024);
        assert_eq!(
            cfg.match_config.neural_learning.backend,
            SoccerNeuralLearningBackend::Threaded
        );
        assert!(cfg.match_config.mpc.tier2_player_enabled);
        assert!(cfg.match_config.mpc.reconcile_enabled);
        assert!(cfg.match_config.mpc.field_aware_enabled);
        assert!(!cfg.match_config.neural_blend.mcts_enabled);
        assert!(cfg.match_config.adversarial_embedding_exploitation_enabled);
        assert!(!cfg.match_config.full_game_learning_enabled);
    }

    #[test]
    fn live_server_env_can_opt_into_neural_mcts() {
        let cfg = live_server_config_from_lookup(|name| {
            (name == "SOCCER_LIVE_NEURAL_MCTS_ENABLED").then(|| "1".to_string())
        });

        assert!(cfg.match_config.neural_blend.mcts_enabled);
    }

    #[test]
    fn live_server_env_can_enable_autonomous_learning_modes() {
        let vars = BTreeMap::from([
            ("SOCCER_LIVE_LEARNING_ENABLED", "true"),
            ("SOCCER_LIVE_LEARNING_LOGGING_ENABLED", "on"),
            ("SOCCER_LIVE_LEARNING_INTERVAL_TICKS", "7"),
            ("SOCCER_LIVE_FULL_GAME_LEARNING_ENABLED", "true"),
            ("SOCCER_LIVE_HTTP_WORKERS", "6"),
            ("SOCCER_LIVE_NEURAL_LEARNING_ENABLED", "1"),
            ("SOCCER_LIVE_NEURAL_LEARNING_BACKEND", "threaded"),
            ("SOCCER_LIVE_NEURAL_LEARNING_RATE", "0.035"),
            ("SOCCER_LIVE_NEURAL_BATCH_SIZE", "64"),
            ("SOCCER_LIVE_NEURAL_TRAIN_EVERY_TICKS", "1"),
            ("SOCCER_LIVE_NEURAL_MAX_BATCHES_PER_TICK", "6"),
            ("SOCCER_LIVE_NEURAL_HIDDEN_UNITS", "128"),
            ("SOCCER_LIVE_NEURAL_TARGET_SCALE", "8"),
            ("SOCCER_LIVE_NEURAL_TARGET_CLIP", "8"),
            ("SOCCER_LIVE_NEURAL_MAX_PENDING_BATCHES", "128"),
            ("SOCCER_LIVE_NEURAL_REPLAY_CAPACITY", "8192"),
            ("SOCCER_LIVE_NEURAL_REPLAY_SAMPLES_PER_TICK", "128"),
            ("SOCCER_LIVE_NEURAL_SNAPSHOT_EVERY_BATCHES", "8"),
            ("SOCCER_LIVE_NEURAL_TARGET_POPART", "1"),
            ("SOCCER_LIVE_NEURAL_LP_COUPLING_ENABLED", "1"),
            ("SOCCER_LIVE_NEURAL_BLEND_MODE", "authoritative"),
            ("SOCCER_LIVE_NEURAL_BLEND_LAMBDA", "1.25"),
            ("SOCCER_LIVE_NEURAL_BLEND_WARMUP_STEPS", "0"),
            ("SOCCER_LIVE_NEURAL_BLEND_CANDIDATES", "12"),
            ("SOCCER_LIVE_NEURAL_ACTOR_CRITIC", "1"),
            ("SOCCER_LIVE_NEURAL_WORLD_MODEL", "1"),
            ("SOCCER_LIVE_NEURAL_MCTS_ENABLED", "0"),
            ("SOCCER_LIVE_MAPPO_TEAM_REWARD_SHARE", "0.6"),
            ("SOCCER_LIVE_MARL_TEAM_REWARD_WEIGHT", "0.55"),
            ("SOCCER_LIVE_MARL_INTERMEDIATE_REWARD_WEIGHT", "1.0"),
            ("SOCCER_LIVE_MAPPO_CLIP_EPSILON", "0.15"),
            ("SOCCER_LIVE_ADVERSARIAL_EMBEDDING_ENABLED", "yes"),
            ("SOCCER_LIVE_POLICY_AUTOLOAD_MAX_BYTES", "123456"),
        ]);

        let cfg =
            live_server_config_from_lookup(|name| vars.get(name).map(|value| value.to_string()));

        assert!(cfg.match_config.learning_enabled);
        assert!(cfg.match_config.learning_logging_enabled);
        assert_eq!(cfg.http_worker_threads, 6);
        assert_eq!(cfg.match_config.learning_interval_ticks, 7);
        assert!(cfg.match_config.full_game_learning_enabled);
        assert_eq!(cfg.policy_autoload_max_bytes, 123456);
        assert!(cfg.match_config.neural_learning.enabled);
        assert_eq!(
            cfg.match_config.neural_learning.backend,
            SoccerNeuralLearningBackend::Threaded
        );
        assert_eq!(cfg.match_config.neural_learning.learning_rate, 0.035);
        assert_eq!(cfg.match_config.neural_learning.batch_size, 64);
        assert_eq!(cfg.match_config.neural_learning.train_every_ticks, 1);
        assert_eq!(cfg.match_config.neural_learning.max_batches_per_tick, 6);
        assert_eq!(cfg.match_config.neural_learning.hidden_units, 128);
        assert_eq!(cfg.match_config.neural_learning.target_scale, 8.0);
        assert_eq!(cfg.match_config.neural_learning.target_clip, 8.0);
        assert_eq!(cfg.match_config.neural_learning.max_pending_batches, 128);
        assert_eq!(cfg.match_config.neural_learning.replay_capacity, 8192);
        assert_eq!(
            cfg.match_config.neural_learning.replay_samples_per_tick,
            128
        );
        assert_eq!(cfg.match_config.neural_learning.snapshot_every_batches, 8);
        assert!(cfg.match_config.neural_learning.target_popart_enabled);
        assert!(cfg.match_config.neural_learning.lp_coupling_enabled);
        assert_eq!(
            cfg.match_config.neural_learning.mappo_team_reward_share,
            0.6
        );
        assert_eq!(
            cfg.match_config.neural_learning.marl_team_reward_weight,
            0.55
        );
        assert_eq!(
            cfg.match_config
                .neural_learning
                .marl_intermediate_reward_weight,
            1.0
        );
        assert_eq!(cfg.match_config.neural_learning.mappo_clip_epsilon, 0.15);
        assert_eq!(
            cfg.match_config.neural_blend.mode,
            SoccerNeuralBlendMode::Authoritative
        );
        assert_eq!(cfg.match_config.neural_blend.lambda, 1.25);
        assert_eq!(cfg.match_config.neural_blend.warmup_steps, 0);
        assert_eq!(cfg.match_config.neural_blend.candidates, 12);
        assert!(cfg.match_config.neural_blend.actor_critic);
        assert!(cfg.match_config.neural_blend.world_model);
        assert!(!cfg.match_config.neural_blend.mcts_enabled);
        assert!(cfg.match_config.adversarial_embedding_exploitation_enabled);
    }

    #[test]
    fn live_server_env_toggles_mpc_execution_stack() {
        // The live server runs the MPC execution stack (QP + field-aware + neural reconcile)
        // by default; `SOCCER_LIVE_MPC` is the master toggle that can turn it OFF or back ON.
        let default_on = live_server_config_from_lookup(|_| None);
        assert!(default_on.match_config.mpc.tier2_player_enabled);
        assert!(default_on.match_config.mpc.field_aware_enabled);
        assert!(default_on.match_config.mpc.reconcile_enabled);
        assert_eq!(
            default_on.match_config.mpc.player_horizon,
            LIVE_MPC_DEFAULT_PLAYER_HORIZON
        );
        assert_eq!(
            default_on.match_config.mpc.active_radius_yards,
            LIVE_MPC_DEFAULT_ACTIVE_RADIUS_YARDS
        );
        assert!(!default_on.match_config.local_mpc_enabled);
        assert_eq!(
            default_on.match_config.local_mpc_max_players_per_team,
            LIVE_LOCAL_MPC_DEFAULT_MAX_PLAYERS_PER_TEAM
        );

        // SOCCER_LIVE_MPC=0 turns the whole stack off (analytic path).
        let off_vars = BTreeMap::from([("SOCCER_LIVE_MPC", "0"), ("SOCCER_LIVE_LOCAL_MPC", "1")]);
        let off = live_server_config_from_lookup(|name| {
            off_vars.get(name).map(|value| value.to_string())
        });
        assert!(!off.match_config.mpc.tier2_player_enabled);
        assert!(!off.match_config.mpc.field_aware_enabled);
        assert!(!off.match_config.mpc.reconcile_enabled);
        assert!(!off.match_config.local_mpc_enabled);

        // SOCCER_LIVE_MPC=1 keeps the active-player QP + field-aware + neural-reconcile
        // stack on; formation-local MPC stays separate because it is heavier in live HTTP.
        let on_vars = BTreeMap::from([("SOCCER_LIVE_MPC", "1")]);
        let on =
            live_server_config_from_lookup(|name| on_vars.get(name).map(|value| value.to_string()));
        assert!(on.match_config.mpc.tier2_player_enabled);
        assert!(on.match_config.mpc.field_aware_enabled);
        assert!(on.match_config.mpc.reconcile_enabled);
        assert!(!on.match_config.local_mpc_enabled);
    }

    #[test]
    fn live_server_mpc_profile_is_lightweight_with_heavier_env_override() {
        let vars = BTreeMap::from([
            ("SOCCER_LIVE_MPC_PLAYER_HORIZON", "99"),
            ("SOCCER_LIVE_MPC_ACTIVE_RADIUS", "18"),
        ]);
        let cfg =
            live_server_config_from_lookup(|name| vars.get(name).map(|value| value.to_string()));

        assert_eq!(
            cfg.match_config.mpc.player_horizon,
            LIVE_MPC_MAX_PLAYER_HORIZON
        );
        assert_eq!(cfg.match_config.mpc.active_radius_yards, 18.0);
    }

    #[test]
    fn live_server_local_mpc_is_opt_in_and_capped() {
        let vars = BTreeMap::from([
            ("SOCCER_LIVE_LOCAL_MPC", "1"),
            ("SOCCER_LIVE_LOCAL_MPC_MAX_PLAYERS_PER_TEAM", "4"),
        ]);
        let cfg =
            live_server_config_from_lookup(|name| vars.get(name).map(|value| value.to_string()));
        assert!(cfg.match_config.local_mpc_enabled);
        assert_eq!(cfg.match_config.local_mpc_max_players_per_team, 4);

        let clamped_vars = BTreeMap::from([
            ("SOCCER_LIVE_LOCAL_MPC", "1"),
            ("SOCCER_LIVE_LOCAL_MPC_MAX_PLAYERS_PER_TEAM", "99"),
        ]);
        let clamped = live_server_config_from_lookup(|name| {
            clamped_vars.get(name).map(|value| value.to_string())
        });
        assert!(clamped.match_config.local_mpc_enabled);
        assert_eq!(
            clamped.match_config.local_mpc_max_players_per_team,
            LIVE_LOCAL_MPC_MAX_PLAYERS_PER_TEAM_LIMIT
        );

        let zero_vars = BTreeMap::from([
            ("SOCCER_LIVE_LOCAL_MPC", "1"),
            ("SOCCER_LIVE_LOCAL_MPC_MAX_PLAYERS_PER_TEAM", "0"),
        ]);
        let zero = live_server_config_from_lookup(|name| {
            zero_vars.get(name).map(|value| value.to_string())
        });
        assert!(!zero.match_config.local_mpc_enabled);
        assert_eq!(zero.match_config.local_mpc_max_players_per_team, 0);
    }

    #[test]
    fn live_server_env_allows_zero_policy_autoload_cap() {
        let vars = BTreeMap::from([("SOCCER_LIVE_POLICY_AUTOLOAD_MAX_BYTES", "0")]);

        let cfg =
            live_server_config_from_lookup(|name| vars.get(name).map(|value| value.to_string()));

        assert_eq!(cfg.policy_autoload_max_bytes, 0);
    }

    #[test]
    fn live_server_5056_feature_preserves_default_and_env_override() {
        let default_cfg = live_server_config_from_lookup(|_| None);
        assert_eq!(default_cfg.port, 5055);

        let revival_cfg = live_server_config_from_lookup_with_default_port(
            |_| None,
            Some(SOCCER_LIVE_REVIVAL_PORT),
        );
        assert_eq!(revival_cfg.port, SOCCER_LIVE_REVIVAL_PORT);

        let env_cfg = live_server_config_from_lookup_with_default_port(
            |name| {
                if name == "SOCCER_LIVE_PORT" {
                    Some("6060".to_string())
                } else {
                    None
                }
            },
            Some(SOCCER_LIVE_REVIVAL_PORT),
        );
        assert_eq!(env_cfg.port, 6060);
    }

    #[test]
    fn live_server_env_allows_ephemeral_port_zero() {
        let cfg = live_server_config_from_lookup(|name| {
            if name == "SOCCER_LIVE_PORT" {
                Some("0".to_string())
            } else {
                None
            }
        });

        assert_eq!(cfg.port, 0);
    }
}
