//! CPU profiling harness for the 2D soccer simulation.
//!
//! Examples:
//!   SOCCER_PROFILE_TICKS=2000 cargo run --release --bin main_soccer_profile
//!   cargo flamegraph --release --bin main_soccer_profile

use std::env;
use std::time::Instant;

use soccer_engine::des::general::soccer::{
    MatchConfig, SoccerMatch, SoccerNeuralLearningBackend, SoccerRealtimeSession,
    SoccerStepRequest, SoccerStepTimingStats,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProfileMode {
    MatchLoop,
    LiveHttp,
    FullStep,
}

impl ProfileMode {
    fn from_env() -> Self {
        match env_text("SOCCER_PROFILE_MODE")
            .unwrap_or_else(|| "live-http".to_string())
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "match" | "match-loop" | "loop" => ProfileMode::MatchLoop,
            "full" | "full-step" | "step" => ProfileMode::FullStep,
            _ => ProfileMode::LiveHttp,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            ProfileMode::MatchLoop => "match-loop",
            ProfileMode::LiveHttp => "live-http",
            ProfileMode::FullStep => "full-step",
        }
    }
}

#[derive(Clone, Debug)]
struct ProfileConfig {
    mode: ProfileMode,
    ticks: u64,
    ticks_per_batch: u64,
    seed: u32,
    learning_enabled: bool,
    learning_logging_enabled: bool,
    full_game_learning_enabled: bool,
    neural_learning_enabled: bool,
    neural_backend: SoccerNeuralLearningBackend,
    serialize_json: bool,
}

impl ProfileConfig {
    fn from_env() -> Self {
        ProfileConfig {
            mode: ProfileMode::from_env(),
            ticks: env_positive_u64("SOCCER_PROFILE_TICKS").unwrap_or(1000),
            ticks_per_batch: env_positive_u64("SOCCER_PROFILE_TICKS_PER_BATCH").unwrap_or(10),
            seed: env_positive_u32("SOCCER_PROFILE_SEED").unwrap_or(2026),
            learning_enabled: env_bool("SOCCER_PROFILE_LEARNING").unwrap_or(true),
            learning_logging_enabled: env_bool("SOCCER_PROFILE_LEARNING_LOGGING").unwrap_or(false),
            full_game_learning_enabled: env_bool("SOCCER_PROFILE_FULL_GAME_LEARNING")
                .unwrap_or(false),
            neural_learning_enabled: env_bool("SOCCER_PROFILE_NEURAL").unwrap_or(false),
            neural_backend: env_neural_backend("SOCCER_PROFILE_NEURAL_BACKEND")
                .unwrap_or(SoccerNeuralLearningBackend::Inline),
            serialize_json: env_bool("SOCCER_PROFILE_SERIALIZE_JSON").unwrap_or(false),
        }
    }

    fn match_config(&self) -> MatchConfig {
        let mut config = MatchConfig::default();
        config.duration_seconds = self.ticks as f64 * config.dt_seconds;
        config.learning_enabled = self.learning_enabled;
        config.learning_logging_enabled = self.learning_logging_enabled;
        config.full_game_learning_enabled = self.full_game_learning_enabled;
        config.neural_learning.enabled = self.neural_learning_enabled;
        config.neural_learning.backend = self.neural_backend;
        config.max_human_players = 0;
        config.seed = self.seed;
        config
    }
}

#[derive(Clone, Debug)]
struct ProfileResult {
    ticks: u64,
    elapsed_ms: f64,
    serialized_bytes: usize,
    score_home: u32,
    score_away: u32,
    timing: SoccerStepTimingStats,
}

fn main() {
    let cfg = ProfileConfig::from_env();
    let result = match cfg.mode {
        ProfileMode::MatchLoop => run_match_loop_profile(&cfg),
        ProfileMode::LiveHttp => run_live_http_profile(&cfg),
        ProfileMode::FullStep => run_full_step_profile(&cfg),
    };
    let ticks_per_second = if result.elapsed_ms > 0.0 {
        result.ticks as f64 / (result.elapsed_ms / 1000.0)
    } else {
        0.0
    };
    println!(
        concat!(
            "soccer_profile mode={} ticks={} batch={} learning={} logging={} neural={} ",
            "serializeJson={} elapsedMs={:.3} ticksPerSecond={:.2} serializedBytes={} ",
            "score={}-{} phaseTotalMs={:.3} phaseFieldLoopMs={:.3} ",
            "phaseFieldLoopPct={:.2} fieldScheduleMs={:.3} fieldSnapshotMs={:.3} ",
            "fieldPlayerDecisionMs={:.3} fieldIntentMs={:.3} fieldOfficialMs={:.3} ",
            "fieldBallMs={:.3} playerPossessionMs={:.3} playerSupportMs={:.3} ",
            "playerDefenseMs={:.3} playerLooseMs={:.3} phaseLearningMs={:.3} ",
            "phaseLearningPct={:.2} learningDefenseMs={:.3} learningTransitionsMs={:.3} ",
            "recentLearningMs={:.3} neuralSamplesMs={:.3} policyTrainMs={:.3} ",
            "learningLogMs={:.3} fullGameLearningMs={:.3} ",
            "maxStepMs={:.3}"
        ),
        cfg.mode.as_str(),
        result.ticks,
        cfg.ticks_per_batch,
        cfg.learning_enabled,
        cfg.learning_logging_enabled,
        cfg.neural_learning_enabled,
        cfg.serialize_json,
        result.elapsed_ms,
        ticks_per_second,
        result.serialized_bytes,
        result.score_home,
        result.score_away,
        result.timing.total_ms,
        result.timing.field_loop_ms,
        result.timing.field_loop_pct(),
        result.timing.field_schedule_ms,
        result.timing.field_snapshot_ms,
        result.timing.field_player_decision_ms,
        result.timing.field_intent_ms,
        result.timing.field_official_ms,
        result.timing.field_ball_ms,
        result.timing.field_player_possession_decision_ms,
        result.timing.field_player_support_decision_ms,
        result.timing.field_player_defensive_decision_ms,
        result.timing.field_player_loose_decision_ms,
        result.timing.learning_ms(),
        result.timing.learning_pct(),
        result.timing.learning_defense_ms,
        result.timing.learning_transitions_ms,
        result.timing.recent_learning_ms,
        result.timing.neural_samples_ms,
        result.timing.policy_train_ms,
        result.timing.learning_log_ms,
        result.timing.full_game_learning_ms,
        result.timing.max_step_ms,
    );
}

fn run_match_loop_profile(cfg: &ProfileConfig) -> ProfileResult {
    let mut sim = SoccerMatch::default_11v11(cfg.match_config());
    let start = Instant::now();
    let mut ticks = 0;
    while ticks < cfg.ticks && !sim.is_done() {
        sim.run_time_step();
        ticks += 1;
    }
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let summary = sim.summary();
    let serialized_bytes = if cfg.serialize_json {
        serde_json::to_vec(&sim.to_frame())
            .map(|body| body.len())
            .unwrap_or(0)
    } else {
        0
    };
    ProfileResult {
        ticks,
        elapsed_ms,
        serialized_bytes,
        score_home: summary.score_home,
        score_away: summary.score_away,
        timing: sim.step_timing_stats(),
    }
}

fn run_live_http_profile(cfg: &ProfileConfig) -> ProfileResult {
    let mut session = SoccerRealtimeSession::new_without_controller_threads(cfg.match_config());
    let start = Instant::now();
    let mut ticks = 0;
    let mut serialized_bytes = 0;
    let mut last_score = (0, 0);
    while ticks < cfg.ticks {
        let batch = cfg.ticks_per_batch.min(cfg.ticks - ticks).max(1);
        let response = session.step_for_live_http(SoccerStepRequest {
            ticks: batch,
            record_every_ticks: Some(batch),
            ..SoccerStepRequest::default()
        });
        ticks = response.summary.ticks;
        last_score = (response.summary.score_home, response.summary.score_away);
        if cfg.serialize_json {
            serialized_bytes += serde_json::to_vec(&response)
                .map(|body| body.len())
                .unwrap_or(0);
        }
        if response.done {
            break;
        }
    }
    ProfileResult {
        ticks,
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        serialized_bytes,
        score_home: last_score.0,
        score_away: last_score.1,
        timing: session.match_ref().step_timing_stats(),
    }
}

fn run_full_step_profile(cfg: &ProfileConfig) -> ProfileResult {
    let mut session = SoccerRealtimeSession::new_without_controller_threads(cfg.match_config());
    let start = Instant::now();
    let mut ticks = 0;
    let mut serialized_bytes = 0;
    let mut last_score = (0, 0);
    while ticks < cfg.ticks {
        let batch = cfg.ticks_per_batch.min(cfg.ticks - ticks).max(1);
        let response = session.step(SoccerStepRequest {
            ticks: batch,
            record_every_ticks: Some(batch),
            ..SoccerStepRequest::default()
        });
        ticks = response.summary.ticks;
        last_score = (response.summary.score_home, response.summary.score_away);
        if cfg.serialize_json {
            serialized_bytes += serde_json::to_vec(&response)
                .map(|body| body.len())
                .unwrap_or(0);
        }
        if response.done {
            break;
        }
    }
    ProfileResult {
        ticks,
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        serialized_bytes,
        score_home: last_score.0,
        score_away: last_score.1,
        timing: session.match_ref().step_timing_stats(),
    }
}

fn env_text(name: &str) -> Option<String> {
    env::var(name).ok().filter(|value| !value.trim().is_empty())
}

fn env_bool(name: &str) -> Option<bool> {
    env_text(name).and_then(|value| match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    })
}

fn env_positive_u64(name: &str) -> Option<u64> {
    env_text(name)
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
}

fn env_positive_u32(name: &str) -> Option<u32> {
    env_text(name)
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|value| *value > 0)
}

fn env_neural_backend(name: &str) -> Option<SoccerNeuralLearningBackend> {
    env_text(name).and_then(|value| match value.trim().to_ascii_lowercase().as_str() {
        "inline" => Some(SoccerNeuralLearningBackend::Inline),
        "threaded" | "thread" | "background" => Some(SoccerNeuralLearningBackend::Threaded),
        _ => None,
    })
}
