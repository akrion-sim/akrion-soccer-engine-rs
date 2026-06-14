//! Nightly learning **tournament** runner.
//!
//! Wires the existing (previously unwired) 128-team tournament engine
//! [`soccer_engine::des::general::tournament`] to a runnable binary so it can be
//! launched by a k8s CronJob each night (2am → hard 7am deadline). It:
//!
//!   1. seeds the entrant field from the latest *active* Postgres policy (its
//!      neural value/critic snapshot), with the rest of the field fresh so the
//!      bracket both exploits the current best brain and explores new ones;
//!   2. plays the full World-Cup-style bracket with the real `SoccerMatch`
//!      engine (neural-authoritative — `EngineMatchRunner` uses `live_gameplay`),
//!      learning as it goes; and
//!   3. promotes the champion's evolved neural snapshot back to Postgres as the
//!      new active policy version, so each night builds on the last.
//!
//! Postgres is optional: with no `SOCCER_DATABASE_URL` it runs an all-fresh
//! evaluation and skips promotion. The hard 2am–7am budget is enforced by the
//! CronJob's `activeDeadlineSeconds`; this binary logs elapsed time so the
//! schedule can be calibrated against measured per-match runtime.
//!
//! Env knobs:
//!   SOCCER_TOURNAMENT_SEED            (u32, default 20260613)
//!   SOCCER_TOURNAMENT_LEARNING_MODE  (frozen|bilearning|frozen_opponent; default bilearning)
//!   SOCCER_TOURNAMENT_MATCH_SECONDS  (f64, default TOURNAMENT_DEFAULT_MATCH_SECONDS)
//!   SOCCER_TOURNAMENT_SEED_FRACTION  (0..=1, default 0.5 — share of teams seeded from PG)
//!   SOCCER_TOURNAMENT_PROMOTE        (bool, default true)
//!   SOCCER_EXPERIMENT_SLUG           (default "soccer-nightly-tournament")

use std::error::Error;
use std::time::Instant;

use soccer_engine::des::general::soccer::{
    MatchConfig, SoccerNeuralNetworkSnapshot, SoccerQPolicyOptions, SoccerTeamQPolicies,
};
use soccer_engine::des::general::tournament::{
    EngineMatchRunner, EngineMatchRunnerConfig, TeamBrain, Tournament, TournamentFormat,
    TournamentLearningMode, TournamentReport, TournamentTeam, TOURNAMENT_DEFAULT_MATCH_SECONDS,
};
use soccer_engine::des::soccer_learning::SOCCER_POLICY_STATUS_ACTIVE;
use soccer_engine::des::soccer_learning_pg::SoccerLearningPgStore;
use uuid::Uuid;

const DEFAULT_EXPERIMENT_SLUG: &str = "soccer-nightly-tournament";
/// Upper bound on parallel match workers — each holds a live SoccerMatch sim, so this caps
/// peak memory/threads regardless of the env value or the host core count.
const MAX_TOURNAMENT_THREADS: usize = 256;

fn env_string(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn env_u32(key: &str, default: u32) -> u32 {
    env_string(key)
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_f64(key: &str, default: f64) -> f64 {
    env_string(key)
        .and_then(|value| value.parse().ok())
        .filter(|value: &f64| value.is_finite())
        .unwrap_or(default)
}

fn env_bool(key: &str, default: bool) -> bool {
    match env_string(key) {
        Some(value) => matches!(value.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"),
        None => default,
    }
}

/// Parse the learning mode; defaults to whole-field learning (BiLearning).
fn parse_learning_mode(raw: Option<String>) -> TournamentLearningMode {
    match raw.map(|value| value.to_ascii_lowercase()).as_deref() {
        Some("frozen") => TournamentLearningMode::Frozen,
        Some("frozen_opponent") | Some("frozen-opponent") => {
            TournamentLearningMode::FrozenOpponent
        }
        _ => TournamentLearningMode::BiLearning,
    }
}

/// Build the entrant field. The first `round(team_count * seed_fraction)` teams
/// inherit the seed neural snapshot (exploit the current best); the rest start
/// fresh (explore). With no seed snapshot the whole field is fresh.
fn build_entrants(
    format: &TournamentFormat,
    seed: u32,
    seed_snapshot: Option<SoccerNeuralNetworkSnapshot>,
    seed_fraction: f64,
) -> Vec<TournamentTeam> {
    let seeded_count = ((format.team_count as f64) * seed_fraction.clamp(0.0, 1.0)).round() as usize;
    let seeded_count = seeded_count.min(format.team_count);
    (0..format.team_count)
        .map(|id| {
            let brain = match (id < seeded_count).then(|| seed_snapshot.clone()).flatten() {
                Some(snapshot) => TeamBrain::from_snapshot(snapshot),
                None => TeamBrain::fresh(),
            };
            TournamentTeam::new(
                id,
                format!("Team {:03}", id + 1),
                seed.wrapping_add(id as u32).wrapping_mul(2_654_435_761),
                brain,
            )
        })
        .collect()
}

/// Champion fitness for the promoted policy version: league points dominate,
/// goal difference breaks ties.
fn champion_fitness(report: &TournamentReport) -> f64 {
    let record = report.champion().record;
    record.points() as f64 + record.goal_difference() as f64 * 0.1
}

/// Persist the champion's evolved neural snapshot to Postgres as the new active
/// policy version. Tabular Q is intentionally blank: the tournament carries only
/// the neural value/critic (the real cross-match learner). Promoted as `merge`
/// (the schema's source-kind enum has no "tournament" member) with tournament
/// provenance recorded in the search metadata.
fn promote_champion(
    store: &mut SoccerLearningPgStore,
    experiment_id: &str,
    parent_policy_version_id: Option<&str>,
    generation: i32,
    report: &TournamentReport,
    config: &MatchConfig,
    options: &SoccerQPolicyOptions,
) -> Result<String, String> {
    let policy_version_id = Uuid::new_v4().to_string();
    let label_suffix = &policy_version_id[..8];
    let version_label = format!("tournament/gen{generation}/{label_suffix}");
    let policies = SoccerTeamQPolicies::new(options.clone());
    let fitness = champion_fitness(report);
    let champion = report.champion();
    let search_metadata = serde_json::json!({
        "tournament": {
            "championId": report.champion_id,
            "runnerUpId": report.runner_up_id,
            "thirdPlaceId": report.third_place_id,
            "matchCount": report.match_count(),
            "learningMode": format!("{:?}", report.learning_mode),
            "format": {
                "teamCount": report.format.team_count,
                "groupSize": report.format.group_size,
                "advancersPerGroup": report.format.advancers_per_group,
            },
            "championRecord": {
                "played": champion.record.played,
                "wins": champion.record.wins,
                "draws": champion.record.draws,
                "losses": champion.record.losses,
                "goalsFor": champion.record.goals_for,
                "goalsAgainst": champion.record.goals_against,
            },
            "championMatchesLearned": champion.brain.matches_learned,
            "championTrainingSteps": champion.brain.training_steps,
        }
    });

    store.insert_policy_version_with_id_and_neural_network_and_search_metadata(
        &policy_version_id,
        experiment_id,
        parent_policy_version_id,
        generation,
        &version_label,
        "merge",
        SOCCER_POLICY_STATUS_ACTIVE,
        config,
        options.clone(),
        options.clone(),
        &policies,
        fitness,
        champion.brain.neural.as_ref(),
        Some(&search_metadata),
    )?;
    Ok(policy_version_id)
}

fn main() -> Result<(), Box<dyn Error>> {
    let started = Instant::now();

    // Fixed 128-team default format (32 groups of 4, top-2 → 64-team knockout).
    let format = TournamentFormat::default();
    format.validate()?;

    let seed = env_u32("SOCCER_TOURNAMENT_SEED", 20_260_613);
    let mode = parse_learning_mode(env_string("SOCCER_TOURNAMENT_LEARNING_MODE"));
    // Clamp to a sane positive range: 0/negative would make `total_ticks()` zero (a no-op
    // match silently recorded as 0-0), and the engine already caps the top at 24h anyway.
    let match_seconds =
        env_f64("SOCCER_TOURNAMENT_MATCH_SECONDS", TOURNAMENT_DEFAULT_MATCH_SECONDS)
            .clamp(1.0, 86_400.0);
    let seed_fraction = env_f64("SOCCER_TOURNAMENT_SEED_FRACTION", 0.5);
    let promote = env_bool("SOCCER_TOURNAMENT_PROMOTE", true);
    let slug = env_string("SOCCER_EXPERIMENT_SLUG").unwrap_or_else(|| DEFAULT_EXPERIMENT_SLUG.to_string());
    // Parallelism: independent fixtures (groups, and matches within a knockout round)
    // play concurrently. Default to the machine's parallelism; results are identical
    // to a serial run regardless of thread count.
    let threads = env_string("SOCCER_TOURNAMENT_THREADS")
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value >= 1)
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        })
        // Hard cap: each worker runs a full SoccerMatch sim, so an absurd value (typo'd
        // env, or a many-hundred-core box) must not spawn a thread/OOM storm.
        .clamp(1, MAX_TOURNAMENT_THREADS);

    let mut runner_config = EngineMatchRunnerConfig::default();
    runner_config.base.duration_seconds = match_seconds;
    let promote_config = runner_config.base.clone();
    let options = SoccerQPolicyOptions::default();

    println!(
        "tournament_start teams={} groups={} knockout={} mode={:?} seed={} match_seconds={:.1} seed_fraction={:.2} threads={} promote={}",
        format.team_count,
        format.group_count(),
        format.knockout_team_count(),
        mode,
        seed,
        match_seconds,
        seed_fraction,
        threads,
        promote,
    );

    // Optional Postgres: seed from the latest active policy and promote the champion.
    let mut store = SoccerLearningPgStore::connect_from_env()?;
    let mut experiment_id: Option<String> = None;
    let mut parent_policy_version_id: Option<String> = None;
    let mut parent_generation: i32 = -1;
    let mut seed_snapshot: Option<SoccerNeuralNetworkSnapshot> = None;

    if let Some(store) = store.as_mut() {
        let eid = store.ensure_experiment(&slug, "Soccer nightly learning tournament", &promote_config)?;
        match store.load_latest_active_policy_metadata(&eid)? {
            Some(metadata) => {
                parent_policy_version_id = Some(metadata.id.clone());
                parent_generation = metadata.generation;
                seed_snapshot = metadata.neural_network.clone();
                println!(
                    "tournament_seed experiment={} parent_policy_version={} parent_generation={} has_neural={}",
                    eid,
                    metadata.id,
                    metadata.generation,
                    seed_snapshot.is_some(),
                );
            }
            None => {
                println!("tournament_seed experiment={eid} no_active_policy=true (cold start)");
            }
        }
        experiment_id = Some(eid);
    } else {
        println!("tournament_seed postgres=absent (all-fresh field, promotion skipped)");
    }

    let teams = build_entrants(&format, seed, seed_snapshot, seed_fraction);
    let tournament = Tournament::new(teams, format, mode, seed)?;
    let runner = EngineMatchRunner::new(runner_config);
    // Engine API (HEAD line): run_parallel borrows the runner and takes an
    // optional wall-clock deadline + a per-wave progress callback. The CronJob
    // enforces the hard 2am–7am stop via activeDeadlineSeconds, so no app-level
    // deadline is passed here; progress is logged for schedule calibration.
    let report = tournament.run_parallel(&runner, threads, None, |progress| {
        println!(
            "tournament_progress stage={:?} played={} total={} elapsed_secs={:.1}",
            progress.stage_label,
            progress.matches_played,
            progress.matches_total,
            progress.elapsed.as_secs_f64(),
        );
    })?;

    let champion = report.champion();
    println!(
        "tournament_complete champion_id={} champion_name={:?} runner_up_id={} matches={} champion_points={} champion_gd={} elapsed_secs={:.1}",
        report.champion_id,
        champion.name,
        report.runner_up_id,
        report.match_count(),
        champion.record.points(),
        champion.record.goal_difference(),
        started.elapsed().as_secs_f64(),
    );

    if promote {
        if let (Some(store), Some(eid)) = (store.as_mut(), experiment_id.as_ref()) {
            let generation = parent_generation.saturating_add(1).max(0);
            let version_id = promote_champion(
                store,
                eid,
                parent_policy_version_id.as_deref(),
                generation,
                &report,
                &promote_config,
                &options,
            )?;
            println!(
                "tournament_promote policy_version={version_id} generation={generation} parent={} fitness={:.3}",
                parent_policy_version_id.as_deref().unwrap_or("none"),
                champion_fitness(&report),
            );
        } else {
            println!("tournament_promote skipped=true reason=no_postgres");
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_format_is_128_team_world_cup_shape() {
        let format = TournamentFormat::default();
        assert!(format.validate().is_ok());
        assert_eq!(format.team_count, 128);
        assert_eq!(format.group_count(), 32);
        assert_eq!(format.knockout_team_count(), 64);
    }

    #[test]
    fn learning_mode_parses_with_bilearning_default() {
        assert_eq!(parse_learning_mode(None), TournamentLearningMode::BiLearning);
        assert_eq!(
            parse_learning_mode(Some("frozen".to_string())),
            TournamentLearningMode::Frozen
        );
        assert_eq!(
            parse_learning_mode(Some("FROZEN_OPPONENT".to_string())),
            TournamentLearningMode::FrozenOpponent
        );
        assert_eq!(
            parse_learning_mode(Some("nonsense".to_string())),
            TournamentLearningMode::BiLearning
        );
    }

    #[test]
    fn entrant_field_matches_format_and_honors_seed_fraction() {
        let format = TournamentFormat::default();
        // No snapshot ⇒ all fresh regardless of fraction.
        let fresh = build_entrants(&format, 7, None, 0.75);
        assert_eq!(fresh.len(), format.team_count);
        assert!(fresh.iter().all(|team| team.brain.neural.is_none()));

        // With a snapshot, exactly round(team_count * fraction) teams are seeded.
        let snapshot = SoccerNeuralNetworkSnapshot::default();
        let seeded = build_entrants(&format, 7, Some(snapshot), 0.25);
        let seeded_count = seeded.iter().filter(|team| team.brain.neural.is_some()).count();
        assert_eq!(seeded_count, 32); // 128 * 0.25
        // Ids are unique (Tournament::new requires it).
        let mut ids: Vec<usize> = seeded.iter().map(|team| team.id).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), format.team_count);
    }
}
