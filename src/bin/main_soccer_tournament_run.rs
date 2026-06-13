//! Nightly **learning tournament** runner.
//!
//! Plays a 128-team World-Cup-style learning tournament (32 groups of 4 → a
//! 64-team knockout) with many matches running in parallel, warm-starting each
//! team's brain from the previous night's result in Postgres and persisting the
//! evolved brains + full bracket back. Built to run as a CronJob from 2am→7am
//! (see `k8s/soccer-learning-tournament.cronjob.yaml`): the bracket's critical
//! path is only ~10 "waves", so the work is compute-bound, not deadline-bound.
//!
//! All knobs are environment variables (so the same image is reusable):
//!
//! - `SOCCER_TOURNAMENT_TEAMS` (128), `_GROUP_SIZE` (4), `_ADVANCERS` (2),
//!   `_DOUBLE_ROUND_ROBIN` (false), `_THIRD_PLACE` (true)
//! - `SOCCER_TOURNAMENT_MODE` = `bilearning` (default) | `frozen` | `frozenopponent`
//! - `SOCCER_TOURNAMENT_PARALLELISM` (default: CPU count)
//! - `SOCCER_TOURNAMENT_MATCH_SECONDS` (simulated minutes per match)
//! - `SOCCER_TOURNAMENT_SEED`, `SOCCER_TOURNAMENT_DATE` (`YYYY-MM-DD`, default: today UTC)
//! - `SOCCER_TOURNAMENT_EXPERIMENT` (Postgres experiment slug)
//! - `SOCCER_TOURNAMENT_DEADLINE_SECONDS` (graceful stop before the k8s kill)
//! - Postgres via `SOCCER_DATABASE_URL` / `RDS_DATABASE_URL` (optional — runs
//!   without persistence if unset).

use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use soccer_engine::des::general::tournament::{
    EngineMatchRunner, EngineMatchRunnerConfig, TeamBrain, Tournament, TournamentFormat,
    TournamentLearningMode, TournamentProgress, TournamentTeam, TOURNAMENT_DEFAULT_MATCH_SECONDS,
};
use soccer_engine::des::soccer_learning_pg::SoccerLearningPgStore;

fn env_var(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.trim().is_empty())
}

fn env_usize(name: &str, default: usize) -> usize {
    env_var(name).and_then(|value| value.trim().parse().ok()).unwrap_or(default)
}

fn env_u32(name: &str, default: u32) -> u32 {
    env_var(name).and_then(|value| value.trim().parse().ok()).unwrap_or(default)
}

fn env_f64(name: &str, default: f64) -> f64 {
    env_var(name).and_then(|value| value.trim().parse().ok()).unwrap_or(default)
}

fn env_bool(name: &str, default: bool) -> bool {
    env_var(name)
        .map(|value| matches!(value.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(default)
}

fn parse_mode(raw: &str) -> TournamentLearningMode {
    match raw.to_ascii_lowercase().replace(['-', '_'], "").as_str() {
        "frozen" => TournamentLearningMode::Frozen,
        "frozenopponent" => TournamentLearningMode::FrozenOpponent,
        _ => TournamentLearningMode::BiLearning,
    }
}

/// Epoch seconds → `"YYYY-MM-DD"` (UTC), via Howard Hinnant's civil-from-days.
/// Avoids pulling in a date crate just to stamp the nightly run.
fn utc_date_string(epoch_secs: u64) -> String {
    let days = (epoch_secs / 86_400) as i64;
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let day = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let month = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let year = if month <= 2 { year + 1 } else { year };
    format!("{year:04}-{month:02}-{day:02}")
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{{\"event\":\"tournament_error\",\"error\":{err:?}}}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let started = Instant::now();

    let format = TournamentFormat {
        team_count: env_usize("SOCCER_TOURNAMENT_TEAMS", 128),
        group_size: env_usize("SOCCER_TOURNAMENT_GROUP_SIZE", 4),
        advancers_per_group: env_usize("SOCCER_TOURNAMENT_ADVANCERS", 2),
        double_round_robin: env_bool("SOCCER_TOURNAMENT_DOUBLE_ROUND_ROBIN", false),
        third_place_match: env_bool("SOCCER_TOURNAMENT_THIRD_PLACE", true),
    };
    let mode = parse_mode(&env_var("SOCCER_TOURNAMENT_MODE").unwrap_or_default());
    let now_epoch = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0);
    let date = env_var("SOCCER_TOURNAMENT_DATE").unwrap_or_else(|| utc_date_string(now_epoch));
    let seed = env_u32("SOCCER_TOURNAMENT_SEED", (now_epoch as u32) ^ 0x5005_cccc);
    let parallelism = env_usize(
        "SOCCER_TOURNAMENT_PARALLELISM",
        std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1),
    )
    .max(1);
    let match_seconds = env_f64("SOCCER_TOURNAMENT_MATCH_SECONDS", TOURNAMENT_DEFAULT_MATCH_SECONDS);
    let experiment_slug = env_var("SOCCER_TOURNAMENT_EXPERIMENT")
        .unwrap_or_else(|| "nightly-soccer-tournament".to_string());
    // Stop a bit before the CronJob's activeDeadlineSeconds so we flush Postgres
    // and exit cleanly rather than getting SIGKILLed mid-write.
    let deadline = env_var("SOCCER_TOURNAMENT_DEADLINE_SECONDS")
        .and_then(|value| value.trim().parse::<u64>().ok())
        .map(|secs| Instant::now() + Duration::from_secs(secs));

    let mut runner_config = EngineMatchRunnerConfig::default();
    runner_config.base.duration_seconds = match_seconds;
    let runner = EngineMatchRunner::new(runner_config.clone());

    // Postgres (optional): ensure the experiment + warm-start brains from the
    // most recent completed tournament so learning compounds night over night.
    let mut store = SoccerLearningPgStore::connect_from_env()?;
    let mut experiment_id: Option<String> = None;
    let mut seed_brains = HashMap::new();
    if let Some(store) = store.as_mut() {
        let id = store.ensure_experiment(
            &experiment_slug,
            "Nightly soccer learning tournament",
            &runner_config.base,
        )?;
        seed_brains = store.load_latest_tournament_team_brains(&id)?;
        eprintln!(
            "{{\"event\":\"warm_start\",\"experiment\":{experiment_slug:?},\"seed_brains\":{}}}",
            seed_brains.len()
        );
        experiment_id = Some(id);
    } else {
        eprintln!("{{\"event\":\"no_postgres\",\"note\":\"running without persistence\"}}");
    }

    let teams: Vec<TournamentTeam> = (0..format.team_count)
        .map(|id| {
            let brain = seed_brains
                .remove(&id)
                .map(TeamBrain::from_snapshot)
                .unwrap_or_default();
            TournamentTeam::new(
                id,
                format!("Team {:03}", id + 1),
                seed.wrapping_add(id as u32).wrapping_mul(2_654_435_761),
                brain,
            )
        })
        .collect();

    let tournament = Tournament::new(teams, format, mode, seed)?;
    eprintln!(
        "{{\"event\":\"tournament_start\",\"date\":{date:?},\"teams\":{},\"mode\":{:?},\"parallelism\":{parallelism},\"match_seconds\":{match_seconds}}}",
        format.team_count,
        format!("{mode:?}")
    );

    let report = tournament.run_parallel(&runner, parallelism, deadline, |progress: &TournamentProgress| {
        eprintln!(
            "{{\"event\":\"progress\",\"stage\":{:?},\"matches\":{},\"total\":{},\"elapsed_s\":{:.1},\"avg_match_s\":{:.2},\"projected_total_s\":{:.1}}}",
            progress.stage_label,
            progress.matches_played,
            progress.matches_total,
            progress.elapsed.as_secs_f64(),
            progress.avg_match_seconds(),
            progress.projected_total_seconds(),
        );
    })?;

    let wall = started.elapsed().as_secs_f64();

    if let (Some(store), Some(experiment_id)) = (store.as_mut(), experiment_id.as_ref()) {
        let tournament_id = store.persist_tournament(experiment_id, &date, seed, wall, &report)?;
        eprintln!("{{\"event\":\"persisted\",\"tournament_id\":{tournament_id}}}");
    }

    let champion = report.champion();
    println!(
        "{{\"event\":\"champion\",\"date\":{date:?},\"champion_id\":{},\"champion_name\":{:?},\"runner_up_id\":{},\"third_place_id\":{},\"matches\":{},\"wall_s\":{:.1},\"champion_matches_learned\":{},\"champion_training_steps\":{}}}",
        report.champion_id,
        champion.name,
        report.runner_up_id,
        report.third_place_id.map(|id| id as i64).unwrap_or(-1),
        report.match_count(),
        wall,
        champion.brain.matches_learned,
        champion.brain.training_steps,
    );
    Ok(())
}
