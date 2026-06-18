//! Blocking live server for the 2D soccer simulation.

use crate::des::general::soccer::{
    run_live_soccer_server, SoccerLiveServerConfig, SoccerNeuralLearningBackend,
};

pub const SOCCER_LIVE_REVIVAL_PORT: u16 = 5056;

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
    // with the neural-policy velocity. Off by default (byte-identical); this master
    // toggle turns the whole stack on so the QP + neural net actually drive execution.
    if let Some(mpc) = env_bool(&lookup, "SOCCER_LIVE_MPC", "SOCCER_MPC_ENABLED") {
        cfg.match_config.mpc.tier2_player_enabled = mpc;
        cfg.match_config.mpc.field_aware_enabled = mpc;
        cfg.match_config.mpc.reconcile_enabled = mpc;
        if mpc {
            println!(
                "# soccer-live: MPC execution ENABLED — projected-gradient QP plans each active player's velocity (field-aware, reconciled with the neural policy)"
            );
        }
    }
    // How close to the ball a player must be to run the per-player MPC (the "active
    // subset"). Default ~14yd keeps it near the ball; raise it past the pitch diagonal to
    // have ALL 22 plan their execution through the field-aware QP. There is ample per-tick
    // budget for it (the field loop is a small fraction of the step), so this is safe.
    if let Some(radius) = env_positive_f64(
        &lookup,
        "SOCCER_LIVE_MPC_ACTIVE_RADIUS",
        "SOCCER_MPC_ACTIVE_RADIUS",
    ) {
        cfg.match_config.mpc.active_radius_yards = radius;
        println!("# soccer-live: MPC active radius set to {radius:.0}yd (all players within run the QP)");
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
        assert!(cfg.match_config.adversarial_embedding_exploitation_enabled);
        assert!(!cfg.match_config.full_game_learning_enabled);
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

        // SOCCER_LIVE_MPC=0 turns the whole stack off (analytic path).
        let off_vars = BTreeMap::from([("SOCCER_LIVE_MPC", "0")]);
        let off =
            live_server_config_from_lookup(|name| off_vars.get(name).map(|value| value.to_string()));
        assert!(!off.match_config.mpc.tier2_player_enabled);
        assert!(!off.match_config.mpc.field_aware_enabled);
        assert!(!off.match_config.mpc.reconcile_enabled);

        // SOCCER_LIVE_MPC=1 keeps the whole QP + field-aware + neural-reconcile stack on.
        let on_vars = BTreeMap::from([("SOCCER_LIVE_MPC", "1")]);
        let on =
            live_server_config_from_lookup(|name| on_vars.get(name).map(|value| value.to_string()));
        assert!(on.match_config.mpc.tier2_player_enabled);
        assert!(on.match_config.mpc.field_aware_enabled);
        assert!(on.match_config.mpc.reconcile_enabled);
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
}
