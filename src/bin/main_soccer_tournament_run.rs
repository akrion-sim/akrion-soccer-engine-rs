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
//!   SOCCER_TOURNAMENT_LOCK_KEY       (default SOCCER_EXPERIMENT_SLUG)
//!   SOCCER_TOURNAMENT_SOFT_DEADLINE_SECONDS (f64, default 0/off)
//!   SOCCER_TOURNAMENT_MATCH_WATCHDOG_SECONDS (f64, default 900; 0/off)
//!   SOCCER_EXPERIMENT_SLUG           (default "soccer-nightly-tournament")

use std::error::Error;
use std::time::{Duration, Instant};

use soccer_engine::des::general::soccer::{
    MatchConfig, SoccerNeuralNetworkSnapshot, SoccerQPolicyOptions, SoccerTeamQPolicies,
};
use soccer_engine::des::general::soccer_elo::{CrossPlayMatrix, EloRatings, ELO_DEFAULT_K};
use soccer_engine::des::general::tournament::{
    EngineMatchRunner, EngineMatchRunnerConfig, GenomeRng, SoccerTeamGenome, TeamBrain, Tournament,
    TournamentFormat, TournamentLearningMode, TournamentReport, TournamentTeam,
    TOURNAMENT_DEFAULT_MATCH_SECONDS,
};
use soccer_engine::des::soccer_learning::SOCCER_POLICY_STATUS_ACTIVE;
use soccer_engine::des::soccer_learning_pg::{SoccerLearningPgStore, TournamentElite};
use uuid::Uuid;

const DEFAULT_EXPERIMENT_SLUG: &str = "soccer-nightly-tournament";
/// Upper bound on parallel match workers — each holds a live SoccerMatch sim, so this caps
/// peak memory/threads regardless of the env value or the host core count.
const MAX_TOURNAMENT_THREADS: usize = 256;
const DEFAULT_TOURNAMENT_MATCH_WATCHDOG_SECONDS: f64 = 15.0 * 60.0;
const MAX_TOURNAMENT_MATCH_WATCHDOG_SECONDS: f64 = 86_400.0;

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
        Some(value) => matches!(
            value.to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        None => default,
    }
}

fn tournament_run_lock_key(raw: Option<String>, slug: &str) -> String {
    raw.map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| slug.to_string())
}

fn release_tournament_run_lock_if_held(
    store: &mut Option<SoccerLearningPgStore>,
    held: bool,
    key: &str,
) {
    if !held {
        return;
    }
    if let Some(store) = store.as_mut() {
        match store.release_tournament_run_lock(key) {
            Ok(true) => println!("tournament_lock_released key={key}"),
            Ok(false) => eprintln!("tournament_lock_release_not_held key={key}"),
            Err(err) => eprintln!("tournament_lock_release_failed key={key} error={err}"),
        }
    }
}

/// Parse the learning mode; defaults to whole-field learning (BiLearning).
fn parse_learning_mode(raw: Option<String>) -> TournamentLearningMode {
    match raw.map(|value| value.to_ascii_lowercase()).as_deref() {
        Some("frozen") => TournamentLearningMode::Frozen,
        Some("frozen_opponent") | Some("frozen-opponent") => TournamentLearningMode::FrozenOpponent,
        _ => TournamentLearningMode::BiLearning,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TournamentProgressGranularity {
    group_round_count: usize,
    first_checkpoint_matches: usize,
    largest_wave_matches: usize,
    progress_event_count: usize,
    matches_total: usize,
}

fn tournament_progress_granularity(format: &TournamentFormat) -> TournamentProgressGranularity {
    let group_count = format.group_count();
    let legs = if format.double_round_robin { 2 } else { 1 };
    let padded_group_size = if format.group_size % 2 == 0 {
        format.group_size
    } else {
        format.group_size + 1
    };
    let group_round_count = padded_group_size.saturating_sub(1) * legs;
    let group_wave_matches = group_count * (format.group_size / 2).max(1);
    let group_match_total =
        group_count * format.group_size * format.group_size.saturating_sub(1) / 2 * legs;
    let knockout_team_count = format.knockout_team_count();
    let mut knockout_wave_count = 0usize;
    let mut remaining = knockout_team_count;
    while remaining > 1 {
        knockout_wave_count += 1;
        remaining /= 2;
    }
    let knockout_match_total =
        knockout_team_count.saturating_sub(1) + usize::from(format.third_place_match);
    let largest_wave_matches = group_wave_matches.max(knockout_team_count / 2);

    TournamentProgressGranularity {
        group_round_count,
        first_checkpoint_matches: group_wave_matches,
        largest_wave_matches,
        progress_event_count: group_round_count
            + knockout_wave_count
            + usize::from(format.third_place_match),
        matches_total: group_match_total + knockout_match_total,
    }
}

/// Deterministic per-team RNG (splitmix64) so a given tournament seed reproduces
/// the exact same diversified field — diversity must not break reproducibility.
struct DiversityRng {
    state: u64,
}

impl DiversityRng {
    fn new(seed: u64) -> Self {
        DiversityRng {
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
    /// Uniform in [0, 1).
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
    /// Standard normal via Box–Muller.
    fn gaussian(&mut self) -> f64 {
        let u1 = self.unit().max(1e-12);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
    fn coin(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }
}

/// Recompute the cached parameter count + L2 norm after editing weights, AND
/// scrub any non-finite value to 0. Mutation/crossover must never emit a NaN/inf
/// net: the persistence validator rejects non-finite weights (which would abort a
/// whole checkpoint), and a NaN parent would otherwise poison every descendant.
fn refresh_snapshot_norm(snapshot: &mut SoccerNeuralNetworkSnapshot) {
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

/// Add per-weight Gaussian noise so a descendant explores a basin near its parent.
fn mutate_neural(
    mut snapshot: SoccerNeuralNetworkSnapshot,
    rng: &mut DiversityRng,
    scale: f64,
) -> SoccerNeuralNetworkSnapshot {
    if !scale.is_finite() || scale <= 0.0 {
        refresh_snapshot_norm(&mut snapshot);
        return snapshot;
    }
    for layer in &mut snapshot.layers {
        for row in &mut layer.weights {
            for weight in row.iter_mut() {
                *weight += rng.gaussian() * scale;
            }
        }
        for bias in &mut layer.biases {
            *bias += rng.gaussian() * scale;
        }
    }
    refresh_snapshot_norm(&mut snapshot);
    snapshot
}

/// Uniform crossover: each weight is taken from one parent or the other. Parents
/// must share architecture; on any shape mismatch parent `a` wins that slot (so a
/// heterogeneous pool degrades to cloning rather than panicking).
fn crossover_neural(
    a: &SoccerNeuralNetworkSnapshot,
    b: &SoccerNeuralNetworkSnapshot,
    rng: &mut DiversityRng,
) -> SoccerNeuralNetworkSnapshot {
    let mut child = a.clone();
    for (li, layer) in child.layers.iter_mut().enumerate() {
        let Some(bl) = b.layers.get(li) else { continue };
        for (ri, row) in layer.weights.iter_mut().enumerate() {
            for (ci, weight) in row.iter_mut().enumerate() {
                if rng.coin() {
                    if let Some(bw) = bl.weights.get(ri).and_then(|r| r.get(ci)) {
                        *weight = *bw;
                    }
                }
            }
        }
        for (bi, bias) in layer.biases.iter_mut().enumerate() {
            if rng.coin() {
                if let Some(bb) = bl.biases.get(bi) {
                    *bias = *bb;
                }
            }
        }
    }
    refresh_snapshot_norm(&mut child);
    child
}

/// Sample per-team RL hyperparameters around the defaults so teams learn with
/// different horizons / step sizes / exploration appetites — cheap behavioural
/// diversity that holds even before the nets diverge.
fn sample_team_options(
    base: &SoccerQPolicyOptions,
    rng: &mut DiversityRng,
) -> SoccerQPolicyOptions {
    let mut options = base.clone();
    options.alpha = (base.alpha * rng.range(0.6, 1.6)).clamp(1.0e-4, 1.0);
    options.gamma = rng.range(0.95, 0.995);
    options.exploration_epsilon = rng.range(0.02, 0.15);
    options
}

#[derive(Clone, Copy)]
struct EntrantDiversitySettings {
    enabled: bool,
    mutation_scale: f64,
    genome_mutation_rate: f64,
}

impl EntrantDiversitySettings {
    fn from_env() -> Self {
        EntrantDiversitySettings {
            enabled: env_bool("SOCCER_TOURNAMENT_DIVERSITY", true),
            mutation_scale: env_f64("SOCCER_TOURNAMENT_MUTATION_SCALE", 0.03).max(0.0),
            genome_mutation_rate: env_f64("SOCCER_TOURNAMENT_GENOME_MUTATION", 0.1).clamp(0.0, 1.0),
        }
    }
}

/// Build the entrant field. A tournament of clones learns nothing, so every team
/// is seeded with its OWN starting brain (deterministically, off the tournament
/// seed + team id):
///
/// * `parents` empty            → all teams start from a fresh net (cold start),
///   each with its own sampled hyperparameters.
/// * `parents.len() == 1`       → the first `round(team_count * seed_fraction)`
///   teams are independent MUTATED descendants of that parent (exploit the
///   current best along different directions); the rest are fresh explorers.
/// * `parents.len() >= 2`       → the generational case (last tournament's
///   top-N elite pool): each seeded team is a CROSSOVER of two random parents
///   plus mutation — a random hybrid of the previous generation's best finishers.
///
/// With diversity enabled, the back-four line band is also cycled through the
/// discrete GA search grid by team id, so every 128-team generation deliberately
/// covers the 1/2/3 yard min choices crossed with 20/23/25/27/29 yard max choices.
///
/// `SOCCER_TOURNAMENT_DIVERSITY=0` restores the old behaviour (seeded teams clone
/// the first parent; all share default options). `SOCCER_TOURNAMENT_MUTATION_SCALE`
/// (default 0.03) sets the per-weight noise.
fn build_entrants(
    format: &TournamentFormat,
    seed: u32,
    net_parents: &[SoccerNeuralNetworkSnapshot],
    genome_parents: &[SoccerTeamGenome],
    seed_fraction: f64,
) -> Vec<TournamentTeam> {
    build_entrants_with_settings(
        format,
        seed,
        net_parents,
        genome_parents,
        seed_fraction,
        EntrantDiversitySettings::from_env(),
    )
}

fn build_entrants_with_settings(
    format: &TournamentFormat,
    seed: u32,
    net_parents: &[SoccerNeuralNetworkSnapshot],
    genome_parents: &[SoccerTeamGenome],
    seed_fraction: f64,
    settings: EntrantDiversitySettings,
) -> Vec<TournamentTeam> {
    let base_options = SoccerQPolicyOptions::default();
    let seeded_count = (((format.team_count as f64) * seed_fraction.clamp(0.0, 1.0)).round()
        as usize)
        .min(format.team_count);

    (0..format.team_count)
        .map(|id| {
            let mut rng =
                DiversityRng::new((seed as u64) ^ (id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));

            let neural = if id < seeded_count && !net_parents.is_empty() {
                let net = if !settings.enabled {
                    net_parents[0].clone()
                } else if net_parents.len() == 1 {
                    mutate_neural(net_parents[0].clone(), &mut rng, settings.mutation_scale)
                } else {
                    let a = &net_parents[(rng.next_u64() as usize) % net_parents.len()];
                    let b = &net_parents[(rng.next_u64() as usize) % net_parents.len()];
                    mutate_neural(
                        crossover_neural(a, b, &mut rng),
                        &mut rng,
                        settings.mutation_scale,
                    )
                };
                Some(net)
            } else {
                None
            };

            let options = if settings.enabled {
                sample_team_options(&base_options, &mut rng)
            } else {
                base_options.clone()
            };

            // Tactical genome: hybridize the elite genomes when we have a pool,
            // mutate a lone parent, or roll a fresh random style on a cold start.
            let mut genome = if !settings.enabled {
                SoccerTeamGenome::default()
            } else {
                let mut grng = GenomeRng::new(
                    (seed as u64).rotate_left(32) ^ (id as u64).wrapping_mul(0x2545_F491_4F6C_DD1D),
                );
                match genome_parents.len() {
                    0 => SoccerTeamGenome::random(&mut grng),
                    1 => {
                        let mut g = genome_parents[0].clone();
                        g.mutate(&mut grng, settings.genome_mutation_rate);
                        g
                    }
                    n => {
                        let a = &genome_parents[grng.index(n)];
                        let b = &genome_parents[grng.index(n)];
                        let mut g = SoccerTeamGenome::crossover(a, b, &mut grng);
                        g.mutate(&mut grng, settings.genome_mutation_rate);
                        g
                    }
                }
            };
            if settings.enabled {
                genome.apply_defensive_line_band_permutation(id);
            }

            let brain = TeamBrain {
                neural,
                options,
                genome,
                ..TeamBrain::default()
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

/// Fitness of a single competitor's evolving brain (same league-points-then-GD basis as the
/// champion), used to pick the strongest brain to salvage if the run fails before a champion.
fn team_brain_fitness(team: &TournamentTeam) -> f64 {
    team.record.points() as f64 + team.record.goal_difference() as f64 * 0.1
}

/// Promote a salvaged neural snapshot (the strongest brain seen before a failure) as the new
/// active policy version, so a crashed/aborted tournament still banks the night's learning.
fn promote_salvaged_brain(
    store: &mut SoccerLearningPgStore,
    experiment_id: &str,
    parent_policy_version_id: Option<&str>,
    generation: i32,
    neural: &SoccerNeuralNetworkSnapshot,
    fitness: f64,
    config: &MatchConfig,
) -> Result<String, String> {
    let policy_version_id = Uuid::new_v4().to_string();
    let label = format!(
        "tournament-salvage/gen{generation}/{}",
        &policy_version_id[..8]
    );
    let options = SoccerQPolicyOptions::default();
    let policies = SoccerTeamQPolicies::new(options.clone());
    let search_metadata =
        serde_json::json!({ "tournament": { "salvage": true, "fitness": fitness } });
    store.insert_policy_version_with_id_and_neural_network_and_search_metadata(
        &policy_version_id,
        experiment_id,
        parent_policy_version_id,
        generation,
        &label,
        "merge",
        SOCCER_POLICY_STATUS_ACTIVE,
        config,
        options.clone(),
        options.clone(),
        &policies,
        fitness,
        Some(neural),
        Some(&search_metadata),
    )?;
    Ok(policy_version_id)
}

fn run() -> Result<(), Box<dyn Error>> {
    let started = Instant::now();

    // Tournament shape: defaults to the 128-team format (32 groups of 4, top-2 →
    // 64-team knockout) but each field is env-overridable so a smaller bracket can
    // be run on demand (e.g. SOCCER_TOURNAMENT_TEAMS=24 SOCCER_TOURNAMENT_GROUP_SIZE=3
    // SOCCER_TOURNAMENT_ADVANCERS=1 → 8 groups of 3, top-1 → 8-team knockout).
    // validate() still enforces a power-of-two knockout field, so a bad combination
    // fails fast with a clear message instead of producing a lopsided bracket.
    let default_format = TournamentFormat::default();
    let format = TournamentFormat {
        team_count: env_string("SOCCER_TOURNAMENT_TEAMS")
            .and_then(|v| v.parse().ok())
            .unwrap_or(default_format.team_count),
        group_size: env_string("SOCCER_TOURNAMENT_GROUP_SIZE")
            .and_then(|v| v.parse().ok())
            .unwrap_or(default_format.group_size),
        advancers_per_group: env_string("SOCCER_TOURNAMENT_ADVANCERS")
            .and_then(|v| v.parse().ok())
            .unwrap_or(default_format.advancers_per_group),
        double_round_robin: env_bool(
            "SOCCER_TOURNAMENT_DOUBLE_RR",
            default_format.double_round_robin,
        ),
        third_place_match: env_bool(
            "SOCCER_TOURNAMENT_THIRD_PLACE",
            default_format.third_place_match,
        ),
    };
    format.validate()?;

    let seed = env_u32("SOCCER_TOURNAMENT_SEED", 20_260_613);
    let mode = parse_learning_mode(env_string("SOCCER_TOURNAMENT_LEARNING_MODE"));
    // Clamp to a sane positive range: 0/negative would make `total_ticks()` zero (a no-op
    // match silently recorded as 0-0), and the engine already caps the top at 24h anyway.
    let match_seconds = env_f64(
        "SOCCER_TOURNAMENT_MATCH_SECONDS",
        TOURNAMENT_DEFAULT_MATCH_SECONDS,
    )
    .clamp(1.0, 86_400.0);
    let seed_fraction = env_f64("SOCCER_TOURNAMENT_SEED_FRACTION", 0.5);
    let promote = env_bool("SOCCER_TOURNAMENT_PROMOTE", true);
    let soft_deadline_seconds =
        env_f64("SOCCER_TOURNAMENT_SOFT_DEADLINE_SECONDS", 0.0).clamp(0.0, 86_400.0);
    let match_watchdog_seconds = env_f64(
        "SOCCER_TOURNAMENT_MATCH_WATCHDOG_SECONDS",
        DEFAULT_TOURNAMENT_MATCH_WATCHDOG_SECONDS,
    )
    .clamp(0.0, MAX_TOURNAMENT_MATCH_WATCHDOG_SECONDS);
    let slug =
        env_string("SOCCER_EXPERIMENT_SLUG").unwrap_or_else(|| DEFAULT_EXPERIMENT_SLUG.to_string());
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
    runner_config.match_wall_time_limit =
        (match_watchdog_seconds > 0.0).then(|| Duration::from_secs_f64(match_watchdog_seconds));
    let promote_config = runner_config.base.clone();
    let options = SoccerQPolicyOptions::default();
    let progress_granularity = tournament_progress_granularity(&format);

    println!(
        "tournament_start teams={} groups={} knockout={} mode={:?} seed={} match_seconds={:.1} seed_fraction={:.2} threads={} promote={} soft_deadline_seconds={:.1} match_watchdog_seconds={:.1}",
        format.team_count,
        format.group_count(),
        format.knockout_team_count(),
        mode,
        seed,
        match_seconds,
        seed_fraction,
        threads,
        promote,
        soft_deadline_seconds,
        match_watchdog_seconds,
    );
    println!(
        "tournament_progress_plan progress_unit=wave first_checkpoint_matches={} largest_wave_matches={} group_rounds={} progress_events={} matches_total={} threads={}",
        progress_granularity.first_checkpoint_matches,
        progress_granularity.largest_wave_matches,
        progress_granularity.group_round_count,
        progress_granularity.progress_event_count,
        progress_granularity.matches_total,
        threads,
    );

    // Optional Postgres: seed from the latest active policy and promote the champion.
    let mut store = SoccerLearningPgStore::connect_from_env()?;
    let tournament_lock_key =
        tournament_run_lock_key(env_string("SOCCER_TOURNAMENT_LOCK_KEY"), &slug);
    if tournament_lock_key != slug && !env_bool("SOCCER_TOURNAMENT_ALLOW_NONCANONICAL_LOCK", false)
    {
        println!(
            "tournament_lock_noncanonical key={tournament_lock_key} slug={slug} skipped=true reason=noncanonical_lock_key"
        );
        return Ok(());
    }
    let mut tournament_lock_held = false;
    if let Some(store) = store.as_mut() {
        if store.try_acquire_tournament_run_lock(&tournament_lock_key)? {
            tournament_lock_held = true;
            println!("tournament_lock_acquired key={tournament_lock_key}");
        } else {
            println!(
                "tournament_lock_busy key={tournament_lock_key} skipped=true reason=another_cluster_running"
            );
            return Ok(());
        }
    }
    let mut experiment_id: Option<String> = None;
    let mut parent_policy_version_id: Option<String> = None;
    let mut parent_generation: i32 = -1;
    let mut seed_snapshot: Option<SoccerNeuralNetworkSnapshot> = None;
    // The generational breeding pool: the previous tournament's top-N finishers
    // (their neural snapshots + tactical genomes).
    let mut elite_pool: Vec<TournamentElite> = Vec::new();
    let elite_pool_size = env_string("SOCCER_TOURNAMENT_ELITE_POOL")
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value >= 1)
        .unwrap_or(16);

    if let Some(store) = store.as_mut() {
        let eid = store.ensure_experiment(
            &slug,
            "Soccer nightly learning tournament",
            &promote_config,
        )?;
        // Reclaim any prior run left `running` by a hard kill (we hold the run lock,
        // so those are provably zombies): flip them to `partial` BEFORE loading the
        // elite pool so last night's stranded brains breed into this generation.
        match store.reconcile_orphaned_running_tournaments(&eid) {
            Ok(n) if n > 0 => {
                println!("tournament_reconciled_orphans experiment={eid} count={n} status=partial")
            }
            Ok(_) => {}
            Err(err) => eprintln!(
                "tournament_reconcile_orphans_failed experiment={eid} error={err} (continuing)"
            ),
        }
        match store.load_latest_active_policy_metadata(&eid) {
            Ok(Some(metadata)) => {
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
            Ok(None) => {
                println!("tournament_seed experiment={eid} no_active_policy=true (cold start)");
            }
            Err(err) => {
                eprintln!(
                    "tournament_seed_active_policy_load_failed experiment={eid} error={err} (continuing with elite pool or cold field)"
                );
            }
        }
        // Prefer the previous tournament's elite finishers as the breeding pool so
        // each generation hybridizes the best of the last (a true GA), falling back
        // to the single active policy / a cold field when none exists yet.
        match store.load_latest_tournament_elite_neural_pool(&eid, elite_pool_size) {
            Ok(pool) if !pool.is_empty() => {
                println!(
                    "tournament_elite_pool experiment={eid} size={} requested={elite_pool_size} source=last_completed_tournament",
                    pool.len(),
                );
                elite_pool = pool;
            }
            Ok(_) => {
                println!(
                    "tournament_elite_pool experiment={eid} size=0 (no completed tournament yet)"
                );
            }
            Err(err) => {
                eprintln!("tournament_elite_pool_load_failed experiment={eid} error={err} (continuing without it)");
            }
        }
        experiment_id = Some(eid);
    } else {
        println!("tournament_seed postgres=absent (all-fresh field, promotion skipped)");
    }

    // Total fixtures (group round-robins + knockout + third place) — matches the engine's
    // own count, so the header records progress as `matches_played / match_count`.
    let matches_total = progress_granularity.matches_total;
    let tournament_date = env_string("SOCCER_RUN_ID").unwrap_or_else(|| format!("seed-{seed}"));
    let learning_mode_label = format!("{mode:?}");

    // Create the tournament header NOW (status='running') so the bracket persists as it
    // plays — a worker panic or a killed pod leaves a durable record + standings + brains.
    let mut tournament_db_id: Option<i64> = None;
    if let (Some(store), Some(eid)) = (store.as_mut(), experiment_id.as_ref()) {
        match store.start_tournament(
            eid,
            &tournament_date,
            seed,
            &learning_mode_label,
            &format,
            matches_total,
        ) {
            Ok(id) => {
                tournament_db_id = Some(id);
                println!("tournament_db_started db_id={id} matches_total={matches_total}");
            }
            Err(err) => {
                eprintln!("tournament_db_start_failed (continuing without persistence): {err}")
            }
        }
    }

    // Parent pool for seeding: the previous tournament's top-N elites (true
    // generational crossover of nets AND genomes) when available, else the single
    // active policy (mutated descendants), else empty (a cold, fresh field).
    let (net_parents, genome_parents, field_fraction) = if !elite_pool.is_empty() {
        // The whole 128-team field is hybridized from the elite pool by default
        // (SOCCER_TOURNAMENT_HYBRID_FRACTION, 1.0); lower it to keep fresh explorers.
        let hybrid_fraction = env_f64("SOCCER_TOURNAMENT_HYBRID_FRACTION", 1.0);
        let nets: Vec<SoccerNeuralNetworkSnapshot> =
            elite_pool.iter().map(|e| e.neural.clone()).collect();
        let genomes: Vec<SoccerTeamGenome> =
            elite_pool.iter().filter_map(|e| e.genome.clone()).collect();
        (nets, genomes, hybrid_fraction)
    } else {
        (
            seed_snapshot.into_iter().collect(),
            Vec::new(),
            seed_fraction,
        )
    };
    let teams = build_entrants(&format, seed, &net_parents, &genome_parents, field_fraction);
    println!(
        "tournament_field net_parents={} genome_parents={} field_fraction={:.2} seeding={}",
        net_parents.len(),
        genome_parents.len(),
        field_fraction,
        if net_parents.len() >= 2 {
            "elite_crossover_hybridize"
        } else if net_parents.len() == 1 {
            "single_parent_mutate"
        } else {
            "cold_fresh"
        },
    );
    let tournament = Tournament::new(teams, format, mode, seed)?;
    let runner = EngineMatchRunner::new(runner_config);
    // Engine API (HEAD line): run_parallel borrows the runner and takes an
    // optional wall-clock deadline + a per-wave progress callback. A zero soft
    // deadline lets continuous tournament lanes run each generation to completion.
    // Persist each wave AS IT COMMITS (matches + per-team brain checkpoints) and track the
    // strongest brain in memory, so a failure mid-bracket salvages the night's learning.
    let mut persisted_matches: usize = 0;
    let mut best_salvage: Option<(f64, SoccerNeuralNetworkSnapshot)> = None;
    let run_deadline = (soft_deadline_seconds > 0.0)
        .then(|| started + Duration::from_secs_f64(soft_deadline_seconds));
    let run_result = tournament.run_parallel(
        &runner,
        threads,
        run_deadline,
        |progress, reports, teams| {
            println!(
                "tournament_progress stage={:?} played={} total={} elapsed_secs={:.1}",
                progress.stage_label,
                progress.matches_played,
                progress.matches_total,
                progress.elapsed.as_secs_f64(),
            );
            if let (Some(store), Some(db_id)) = (store.as_mut(), tournament_db_id) {
                if reports.len() > persisted_matches {
                    match store.record_tournament_matches(
                        db_id,
                        &reports[persisted_matches..],
                        persisted_matches,
                    ) {
                        Ok(()) => persisted_matches = reports.len(),
                        Err(err) => eprintln!("tournament_persist_matches_failed: {err}"),
                    }
                }
                if let Err(err) = store.checkpoint_tournament_team_brains(db_id, teams) {
                    eprintln!("tournament_persist_brains_failed: {err}");
                }
            }
            if let Some(team) = teams
                .iter()
                .filter(|team| team.brain.neural.is_some())
                .max_by(|a, b| {
                    team_brain_fitness(a)
                        .partial_cmp(&team_brain_fitness(b))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
            {
                let fitness = team_brain_fitness(team);
                if best_salvage
                    .as_ref()
                    .map_or(true, |(best, _)| fitness > *best)
                {
                    if let Some(neural) = &team.brain.neural {
                        best_salvage = Some((fitness, neural.clone()));
                    }
                }
            }
        },
    );

    let report = match run_result {
        Ok(report) => report,
        Err(err) => {
            // A soft-deadline stop (`check_deadline`) is a graceful partial finish, not a
            // crash: the bracket simply ran out of wall-clock between waves. Mark it `partial`
            // (NOT `failed`) so its checkpointed top finishers still feed the next generation's
            // elite pool — the night's learning is banked, not discarded.
            let deadline_stop = err.contains("deadline reached");
            let finish_status = if deadline_stop { "partial" } else { "failed" };
            eprintln!(
                "tournament_{} error={err} persisted_matches={persisted_matches}",
                if deadline_stop { "partial" } else { "failed" }
            );
            if let (Some(store), Some(db_id)) = (store.as_mut(), tournament_db_id) {
                let _ = store.finish_tournament(
                    db_id,
                    finish_status,
                    None,
                    None,
                    None,
                    started.elapsed().as_secs_f64(),
                );
            }
            // Salvage: promote the strongest brain seen so the run's learning isn't discarded.
            if promote {
                if let (Some(store), Some(eid), Some((fitness, neural))) = (
                    store.as_mut(),
                    experiment_id.as_ref(),
                    best_salvage.as_ref(),
                ) {
                    let generation = parent_generation.saturating_add(1).max(0);
                    match promote_salvaged_brain(
                        store,
                        eid,
                        parent_policy_version_id.as_deref(),
                        generation,
                        neural,
                        *fitness,
                        &promote_config,
                    ) {
                        Ok(id) => println!(
                            "tournament_salvage_promoted policy_version={id} generation={generation} fitness={fitness:.3}"
                        ),
                        Err(e) => eprintln!("tournament_salvage_promote_failed: {e}"),
                    }
                }
            }
            release_tournament_run_lock_if_held(
                &mut store,
                tournament_lock_held,
                &tournament_lock_key,
            );
            // A graceful soft-deadline stop is a successful partial run (learning banked
            // + salvage promoted), so exit 0 — only a genuine crash propagates an error.
            if deadline_stop {
                return Ok(());
            }
            return Err(err.into());
        }
    };

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

    // Relative-strength read (#7): fold the whole fixture list into Elo + a
    // cross-play payoff matrix. Unlike the bracket result, this is robust to who
    // a team happened to draw and surfaces EXPLOITABILITY — a champion that wins
    // the bracket but has a hard counter in the field is fragile, not dominant.
    let elo = EloRatings::from_match_reports(&report.matches, ELO_DEFAULT_K, 16);
    let cross_play = CrossPlayMatrix::from_match_reports(&report.matches);
    let champion_elo = elo.rating(report.champion_id);
    let elo_leader = elo.leader();
    let champion_field_mean = cross_play
        .mean_payoff_vs_field(report.champion_id)
        .unwrap_or(f64::NAN);
    let (worst_opponent, worst_payoff) = cross_play
        .worst_case(report.champion_id)
        .map(|(o, p)| (o as i64, p))
        .unwrap_or((-1, f64::NAN));
    println!(
        "tournament_elo champion_elo={:.0} elo_leader_id={} elo_leader_is_champion={} champion_field_mean={:.3} champion_worst_opponent={} champion_worst_payoff={:.3}",
        champion_elo,
        elo_leader.map(|id| id as i64).unwrap_or(-1),
        elo_leader == Some(report.champion_id),
        champion_field_mean,
        worst_opponent,
        worst_payoff,
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

    // Finalize the header as completed (champion = winning entrant slot).
    if let (Some(store), Some(db_id)) = (store.as_mut(), tournament_db_id) {
        if let Err(err) = store.finish_tournament(
            db_id,
            "completed",
            Some(report.champion_id),
            Some(report.runner_up_id),
            report.third_place_id,
            started.elapsed().as_secs_f64(),
        ) {
            eprintln!("tournament_finish_failed: {err}");
        }
    }

    release_tournament_run_lock_if_held(&mut store, tournament_lock_held, &tournament_lock_key);

    Ok(())
}

fn main() {
    let service_name = "main_soccer_tournament_run";
    let _telemetry = soccer_engine::telemetry::init_soccer_telemetry(service_name);
    soccer_engine::telemetry::emit_process_start(service_name);
    if let Err(error) = run() {
        soccer_engine::telemetry::emit_process_error(service_name, &error.to_string());
        eprintln!("main_soccer_tournament_run: {error}");
        std::process::exit(1);
    }
    soccer_engine::telemetry::emit_process_complete(service_name);
}

#[cfg(test)]
mod tests {
    use super::*;
    use soccer_engine::des::general::soccer::SoccerNeuralLayerSnapshot;
    use soccer_engine::des::general::tournament::DEF_LINE_BAND_PERMUTATION_COUNT;

    #[test]
    fn default_format_is_128_team_world_cup_shape() {
        let format = TournamentFormat::default();
        assert!(format.validate().is_ok());
        assert_eq!(format.team_count, 128);
        assert_eq!(format.group_count(), 32);
        assert_eq!(format.knockout_team_count(), 64);
    }

    #[test]
    fn default_progress_granularity_explains_first_persisted_checkpoint() {
        let format = TournamentFormat::default();
        let granularity = tournament_progress_granularity(&format);
        assert_eq!(granularity.matches_total, 256);
        assert_eq!(granularity.first_checkpoint_matches, 64);
        assert_eq!(granularity.largest_wave_matches, 64);
        assert_eq!(granularity.group_round_count, 3);
        assert_eq!(granularity.progress_event_count, 10);
    }

    #[test]
    fn progress_granularity_handles_odd_group_sizes() {
        let format = TournamentFormat {
            team_count: 24,
            group_size: 3,
            advancers_per_group: 1,
            double_round_robin: false,
            third_place_match: false,
        };
        assert!(format.validate().is_ok());
        let granularity = tournament_progress_granularity(&format);
        assert_eq!(granularity.matches_total, 31);
        assert_eq!(granularity.first_checkpoint_matches, 8);
        assert_eq!(granularity.largest_wave_matches, 8);
        assert_eq!(granularity.group_round_count, 3);
        assert_eq!(granularity.progress_event_count, 6);
    }

    #[test]
    fn learning_mode_parses_with_bilearning_default() {
        assert_eq!(
            parse_learning_mode(None),
            TournamentLearningMode::BiLearning
        );
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
    fn tournament_lock_key_defaults_to_slug_and_ignores_blank_override() {
        assert_eq!(
            tournament_run_lock_key(None, "soccer-nightly-tournament"),
            "soccer-nightly-tournament"
        );
        assert_eq!(
            tournament_run_lock_key(Some("   ".to_string()), "soccer-nightly-tournament"),
            "soccer-nightly-tournament"
        );
        assert_eq!(
            tournament_run_lock_key(
                Some("global-nightly".to_string()),
                "soccer-nightly-tournament"
            ),
            "global-nightly"
        );
    }

    #[test]
    fn entrant_field_matches_format_and_honors_seed_fraction() {
        let format = TournamentFormat::default();
        // No parents ⇒ all fresh regardless of fraction.
        let fresh = build_entrants(&format, 7, &[], &[], 0.75);
        assert_eq!(fresh.len(), format.team_count);
        assert!(fresh.iter().all(|team| team.brain.neural.is_none()));

        // With one parent, exactly round(team_count * fraction) teams are seeded.
        let snapshot = SoccerNeuralNetworkSnapshot::default();
        let seeded = build_entrants(&format, 7, std::slice::from_ref(&snapshot), &[], 0.25);
        let seeded_count = seeded
            .iter()
            .filter(|team| team.brain.neural.is_some())
            .count();
        assert_eq!(seeded_count, 32); // 128 * 0.25
                                      // Ids are unique (Tournament::new requires it).
        let mut ids: Vec<usize> = seeded.iter().map(|team| team.id).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), format.team_count);
    }

    #[test]
    fn entrant_field_cycles_defensive_line_band_permutations() {
        let format = TournamentFormat::default();
        let field =
            build_entrants_with_settings(&format, 7, &[], &[], 1.0, test_diversity_settings());
        let mut seen = std::collections::HashSet::new();

        for (index, team) in field.iter().enumerate() {
            let band = team.brain.genome.defensive_line_band_yards();
            assert_eq!(
                band,
                SoccerTeamGenome::defensive_line_band_for_permutation(index)
            );
            seen.insert((band.0 as i32, band.1 as i32));
        }

        assert_eq!(seen.len(), DEF_LINE_BAND_PERMUTATION_COUNT);
    }

    fn sample_snapshot(fill: f64) -> SoccerNeuralNetworkSnapshot {
        let layer = SoccerNeuralLayerSnapshot {
            activation: "relu".to_string(),
            weights: vec![vec![fill, fill, fill], vec![fill, fill, fill]],
            biases: vec![fill, fill],
        };
        let mut snapshot = SoccerNeuralNetworkSnapshot {
            input_dim: 3,
            output_dim: 2,
            parameter_count: 0,
            l2_norm: 0.0,
            layers: vec![layer],
            training_steps: 0,
            average_loss: None,
            target_popart: None,
            policy_head: None,
            skill_policy_heads: None,
            keeper_policy_head: None,
            mpc_objective_head: None,
            line_depth_head: None,
        };
        refresh_snapshot_norm(&mut snapshot);
        snapshot
    }

    fn test_diversity_settings() -> EntrantDiversitySettings {
        EntrantDiversitySettings {
            enabled: true,
            mutation_scale: 0.03,
            genome_mutation_rate: 0.1,
        }
    }

    #[test]
    fn mutation_is_deterministic_and_perturbs_weights() {
        let base = sample_snapshot(1.0);
        let a = mutate_neural(base.clone(), &mut DiversityRng::new(42), 0.1);
        let b = mutate_neural(base.clone(), &mut DiversityRng::new(42), 0.1);
        // Same seed ⇒ identical mutation (the field stays reproducible).
        assert_eq!(a.layers[0].weights, b.layers[0].weights);
        // ...but the weights actually moved off the parent.
        assert_ne!(a.layers[0].weights, base.layers[0].weights);
        // Zero scale is a no-op.
        let unchanged = mutate_neural(base.clone(), &mut DiversityRng::new(7), 0.0);
        assert_eq!(unchanged.layers[0].weights, base.layers[0].weights);
    }

    #[test]
    fn mutation_refreshes_snapshot_metadata_without_noise() {
        let mut stale = sample_snapshot(1.0);
        stale.parameter_count = 0;
        stale.l2_norm = f64::NAN;

        for scale in [0.0, f64::NAN] {
            let refreshed = mutate_neural(stale.clone(), &mut DiversityRng::new(7), scale);
            assert_eq!(refreshed.layers[0].weights, stale.layers[0].weights);
            assert_eq!(refreshed.parameter_count, 8);
            assert!(refreshed.l2_norm.is_finite());
            assert!(refreshed.l2_norm > 0.0);
        }
    }

    #[test]
    fn net_ops_scrub_non_finite_weights() {
        // A NaN/inf must never survive into a persisted net (the validator rejects
        // them and a NaN parent would poison descendants).
        let mut base = sample_snapshot(1.0);
        base.layers[0].weights[0][0] = f64::NAN;
        base.layers[0].biases[0] = f64::INFINITY;
        let mutated = mutate_neural(base.clone(), &mut DiversityRng::new(3), 0.01);
        let crossed = crossover_neural(&base, &base, &mut DiversityRng::new(4));
        for net in [&mutated, &crossed] {
            assert!(net.l2_norm.is_finite());
            assert!(net
                .layers
                .iter()
                .all(|l| l.weights.iter().flatten().all(|w| w.is_finite())
                    && l.biases.iter().all(|b| b.is_finite())));
        }
    }

    #[test]
    fn crossover_takes_each_weight_from_one_parent() {
        let a = sample_snapshot(0.0);
        let b = sample_snapshot(1.0);
        let child = crossover_neural(&a, &b, &mut DiversityRng::new(99));
        for row in &child.layers[0].weights {
            for weight in row {
                assert!(
                    *weight == 0.0 || *weight == 1.0,
                    "weight {weight} not from a parent"
                );
            }
        }
        // With opposite parents the child differs from both.
        assert_ne!(child.layers[0].weights, a.layers[0].weights);
        assert_ne!(child.layers[0].weights, b.layers[0].weights);
    }

    #[test]
    fn diverse_seeding_gives_distinct_descendants() {
        // Two seeded teams descended from the same single parent must differ — in
        // both their nets and their sampled hyperparameters.
        let format = TournamentFormat::default();
        let parent = sample_snapshot(0.5);
        let field = build_entrants_with_settings(
            &format,
            123,
            std::slice::from_ref(&parent),
            &[],
            1.0,
            test_diversity_settings(),
        );
        let n0 = field[0].brain.neural.as_ref().unwrap();
        let n1 = field[1].brain.neural.as_ref().unwrap();
        assert_ne!(n0.layers[0].weights, n1.layers[0].weights);
        assert_ne!(field[0].brain.options.gamma, field[1].brain.options.gamma);
        // Genomes are diversified per team too (cold pool ⇒ random styles).
        assert!(
            field
                .iter()
                .any(|t| t.brain.genome.formation != field[0].brain.genome.formation)
                || field
                    .iter()
                    .map(|t| t.brain.genome.press_urgency)
                    .collect::<Vec<_>>()
                    .windows(2)
                    .any(|w| w[0] != w[1])
        );
    }

    #[test]
    fn crossover_field_hybridizes_an_elite_pool() {
        // The generational case: a 16-strong elite pool ⇒ each seeded net is a
        // crossover+mutation hybrid; teams are distinct and weights stay finite.
        let format = TournamentFormat::default();
        let pool: Vec<SoccerNeuralNetworkSnapshot> =
            (0..16).map(|i| sample_snapshot(i as f64)).collect();
        let field =
            build_entrants_with_settings(&format, 55, &pool, &[], 1.0, test_diversity_settings());
        let child = field[3].brain.neural.as_ref().unwrap();
        assert!(child
            .layers
            .iter()
            .all(|l| l.weights.iter().flatten().all(|w| w.is_finite())));
        let other = field[4].brain.neural.as_ref().unwrap();
        assert_ne!(child.layers[0].weights, other.layers[0].weights);
    }

    #[test]
    fn elite_pool_contract_builds_full_next_generation() {
        let format = TournamentFormat::default();
        let net_pool: Vec<SoccerNeuralNetworkSnapshot> =
            (0..16).map(|i| sample_snapshot(i as f64)).collect();
        let mut genome_parent = SoccerTeamGenome::default();
        genome_parent.press_urgency = 0.9;
        genome_parent.pass_willingness_under_pressure = 0.8;
        let genome_pool = vec![genome_parent; 16];
        let field = build_entrants_with_settings(
            &format,
            2026,
            &net_pool,
            &genome_pool,
            1.0,
            EntrantDiversitySettings {
                genome_mutation_rate: 0.0,
                ..test_diversity_settings()
            },
        );

        assert_eq!(field.len(), 128);
        assert_eq!(net_pool.len(), 16);
        assert!(field.iter().all(|team| team.brain.neural.is_some()));
        assert!(field
            .iter()
            .all(|team| team
                .brain
                .neural
                .as_ref()
                .is_some_and(|net| net.l2_norm.is_finite()
                    && net.layers.iter().all(|layer| layer
                        .weights
                        .iter()
                        .flatten()
                        .all(|weight| weight.is_finite())))));
        assert!(field
            .iter()
            .all(|team| (team.brain.genome.press_urgency - 0.9).abs() < 1e-12));
        assert!(field.iter().any(|team| {
            let net = team.brain.neural.as_ref().expect("seeded neural net");
            net.layers[0].weights != net_pool[0].layers[0].weights
        }));
    }
}
