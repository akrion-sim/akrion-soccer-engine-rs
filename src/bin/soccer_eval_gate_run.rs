//! Promotion eval gate (operational runner).
//!
//! Plays a **candidate** brain against a frozen league pool on a **held-out** seed
//! range (disjoint from any training seeds), then folds the fixtures through
//! `soccer_eval_gate::evaluate_promotion` and prints the promote/reject verdict.
//! This is the gate the learner/operator runs *before* a candidate replaces the
//! incumbent — raw training reward going up is not enough; the candidate must beat
//! a diverse frozen field on matches it never trained on, with statistical
//! confidence and no hard counter.
//!
//! Every fixture is played FROZEN (neither side learns) by the deterministic
//! `EngineMatchRunner`, so the verdict is reproducible for a given seed base.
//!
//! Run:
//!   cargo run --release --bin soccer_eval_gate_run -- [eval_games_per_opp] [minutes] [pool_size] [train_games]
//!   # default: 6 games/opp, 3 min, 4-team pool, fresh (untrained) candidate
//!   cargo run --release --bin soccer_eval_gate_run
//!   # train the candidate for 16 self-play games first, then gate it:
//!   cargo run --release --bin soccer_eval_gate_run -- 6 3 4 16
//!
//! The candidate is team id 0; the baseline (incumbent) is team id 1; ids 1..pool
//! are the frozen field. Held-out eval seeds come from a seed space disjoint from
//! the training seed space.

use std::sync::Arc;
use std::time::{Duration, Instant};

use soccer_engine::des::general::soccer::SoccerMatch;
use soccer_engine::des::general::soccer::{
    enable_deterministic_formation_lp, MatchConfig, SoccerMarlAlgorithm,
    SoccerNeuralLearningBackend, SoccerNeuralLearningConfig, SoccerNeuralNetworkSnapshot,
    SoccerQPolicyOptions, SoccerQTargetEntry, SoccerTeamQPolicies,
    DEFAULT_SOCCER_MAPPO_TEAM_REWARD_SHARE,
};
use soccer_engine::des::general::soccer_eval_gate::{evaluate_promotion, PromotionThresholds};
use soccer_engine::des::general::tournament::{
    EngineMatchRunner, EngineMatchRunnerConfig, MatchReport, TeamBrain, TournamentMatchContext,
    TournamentMatchRunner, TournamentStage,
};

/// Train a candidate neural brain by inline self-play carry-forward (the real
/// learner's per-process pattern, mirroring `measure_learning_ab`). Returns the
/// trained brain, including target-Q, or `None` for `games == 0` (fresh candidate).
fn train_candidate_brain(games: usize, minutes: f64, seed_base: u32) -> Option<TeamBrain> {
    if games == 0 {
        return None;
    }
    let neural = SoccerNeuralLearningConfig {
        enabled: true,
        backend: SoccerNeuralLearningBackend::Inline,
        marl_algorithm: SoccerMarlAlgorithm::Mappo,
        mappo_team_reward_share: DEFAULT_SOCCER_MAPPO_TEAM_REWARD_SHARE,
        ..SoccerNeuralLearningConfig::default()
    };
    let mut policies = Arc::new(SoccerTeamQPolicies::new(SoccerQPolicyOptions::default()));
    let mut snapshot: Option<SoccerNeuralNetworkSnapshot> = None;
    for g in 0..games {
        let mut config = MatchConfig {
            duration_seconds: minutes * 60.0,
            learning_enabled: true,
            neural_learning: neural.clone(),
            seed: seed_base.wrapping_add(g as u32),
            ..MatchConfig::default()
        };
        // Drive play with the trained actor so the learning signal actually shows.
        config.neural_blend.actor_critic = true;
        let total_ticks = config.total_ticks();
        let mut sim = SoccerMatch::default_11v11(config).with_team_policies((*policies).clone());
        sim.set_uniform_elite_players();
        if let Some(s) = snapshot.as_ref() {
            if let Err(e) = sim.set_neural_network_snapshot(s.clone()) {
                eprintln!("train game {g}: snapshot install failed: {e}");
            }
        }
        for _ in 0..total_ticks {
            sim.run_time_step();
        }
        sim.drain_neural_learning(Duration::from_millis(100));
        if let Some(p) = sim.team_policies() {
            policies = Arc::new(p.clone());
        }
        if let Some(s) = sim.neural_network_snapshot() {
            snapshot = Some(s);
        }
        eprintln!("  trained candidate: game {}/{games}", g + 1);
    }
    snapshot.map(|neural| {
        let mut brain = TeamBrain::from_snapshot_with_targets(
            neural,
            policies.home.target_entries(),
            policies.away.target_entries(),
        );
        brain.matches_learned = games as u32;
        brain
    })
}

fn target_entries_from_value(value: &serde_json::Value, key: &str) -> Vec<SoccerQTargetEntry> {
    value
        .get(key)
        .cloned()
        .and_then(|entries| serde_json::from_value(entries).ok())
        .unwrap_or_default()
}

fn neural_snapshot_from_value(value: &serde_json::Value) -> Option<SoccerNeuralNetworkSnapshot> {
    value
        .get("neuralNetwork")
        .cloned()
        .or_else(|| value.get("learning")?.get("neuralNetwork").cloned())
        .and_then(|neural| serde_json::from_value(neural).ok())
}

fn neural_sidecar_path(path: &str) -> Option<String> {
    path.strip_suffix(".json")
        .map(|stem| format!("{stem}.neural.json"))
}

fn load_neural_snapshot(
    path: &str,
    value: &serde_json::Value,
) -> Option<SoccerNeuralNetworkSnapshot> {
    if let Some(neural) = neural_snapshot_from_value(value) {
        return Some(neural);
    }
    let sidecar = neural_sidecar_path(path)?;
    let raw = std::fs::read_to_string(&sidecar).ok()?;
    let sidecar_value: serde_json::Value = serde_json::from_str(&raw).ok()?;
    neural_snapshot_from_value(&sidecar_value)
        .or_else(|| serde_json::from_value::<SoccerNeuralNetworkSnapshot>(sidecar_value).ok())
}

/// Load a candidate/champion brain directly from a local `learned-params.json`
/// artifact (the fully-local learner's durable policy file), via an env var pointing at the path.
/// Lets the gate score the REAL accumulated local policy against the frozen field instead of only
/// an inline-trained-from-fresh one. The artifact embeds the snapshot under the `neuralNetwork` key
/// and may also carry side-specific target-Q.
fn brain_from_env_file(var: &str) -> Option<TeamBrain> {
    let path = std::env::var(var).ok()?;
    let raw = match std::fs::read_to_string(&path) {
        Ok(raw) => raw,
        Err(e) => {
            eprintln!("eval_snapshot_read_failed var={var} path={path}: {e}");
            return None;
        }
    };
    let value: serde_json::Value = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("eval_snapshot_json_failed var={var} path={path}: {e}");
            return None;
        }
    };
    match load_neural_snapshot(&path, &value) {
        Some(s) => {
            let home_targets = target_entries_from_value(&value, "homeTargetEntries");
            let away_targets = target_entries_from_value(&value, "awayTargetEntries");
            eprintln!(
                "eval_brain_loaded_from_file var={var} path={path} home_targets={} away_targets={}",
                home_targets.len(),
                away_targets.len()
            );
            Some(TeamBrain::from_snapshot_with_targets(
                s,
                home_targets,
                away_targets,
            ))
        }
        None => {
            eprintln!("eval_snapshot_parse_failed var={var} path={path}: missing neural snapshot");
            None
        }
    }
}

/// Load a candidate neural snapshot directly from a local `learned-params.json`
/// artifact (the fully-local learner's durable policy file), via `SOCCER_EVAL_CANDIDATE_PATH`.
/// This lets the gate score the REAL accumulated local policy against the frozen field
/// (a held-out climb number over time) instead of only an inline-trained-from-fresh one.
/// The learned-params artifact embeds the snapshot under the `neuralNetwork` key.
fn snapshot_from_env_file(var: &str) -> Option<SoccerNeuralNetworkSnapshot> {
    let path = std::env::var(var).ok()?;
    let raw = match std::fs::read_to_string(&path) {
        Ok(raw) => raw,
        Err(e) => {
            eprintln!("eval_snapshot_read_failed var={var} path={path}: {e}");
            return None;
        }
    };
    let value: serde_json::Value = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("eval_snapshot_json_failed var={var} path={path}: {e}");
            return None;
        }
    };
    let nn = value.get("neuralNetwork")?.clone();
    match serde_json::from_value::<SoccerNeuralNetworkSnapshot>(nn) {
        Ok(s) => {
            eprintln!("eval_snapshot_loaded_from_file var={var} path={path}");
            Some(s)
        }
        Err(e) => {
            eprintln!("eval_snapshot_parse_failed var={var} path={path}: {e}");
            None
        }
    }
}

/// A held-out frozen-vs-frozen fixture between two brains. Builds the context with
/// learning OFF for both sides and a held-out seed, plays it, and returns the
/// `MatchReport`.
fn play_holdout_fixture(
    runner: &mut EngineMatchRunner,
    home_id: usize,
    away_id: usize,
    seed: u32,
    home: &TeamBrain,
    away: &TeamBrain,
) -> Result<MatchReport, String> {
    let ctx = TournamentMatchContext {
        stage: TournamentStage::Group,
        round_index: 0,
        match_index: 0,
        seed,
        home_id,
        away_id,
        home_name: format!("team{home_id}"),
        away_name: format!("team{away_id}"),
        home_learns: false,
        away_learns: false,
    };
    let outcome = runner.play(&ctx, home, away)?;
    Ok(MatchReport {
        stage: ctx.stage,
        home_id,
        away_id,
        home_name: ctx.home_name,
        away_name: ctx.away_name,
        home_goals: outcome.home_goals,
        away_goals: outcome.away_goals,
        shootout_winner: None,
        home_training_steps: outcome.home_training_steps,
        away_training_steps: outcome.away_training_steps,
    })
}

fn main() {
    enable_deterministic_formation_lp();
    let args: Vec<String> = std::env::args().collect();
    let eval_games_per_opp: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(6);
    let minutes: f64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3.0);
    let pool_size: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(4).max(2);
    let train_games: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);

    // Disjoint seed spaces: training never sees an eval seed.
    let train_seed_base: u32 = 0x71A1_0000;
    let holdout_seed_base: u32 = 0xE7A1_0000;
    let candidate_id = 0usize;
    let baseline_id = 1usize;

    println!("===== PROMOTION EVAL GATE =====");
    println!(
        "pool={pool_size} eval_games/opp={eval_games_per_opp} minutes={minutes} \
         train_games={train_games} holdout_seed_base=0x{holdout_seed_base:08X}"
    );

    // Candidate brain (id 0): prefer a local learned-params file (the fully-local learner's
    // accumulated policy) when SOCCER_EVAL_CANDIDATE_PATH is set; otherwise inline self-play train.
    let candidate_brain = brain_from_env_file("SOCCER_EVAL_CANDIDATE_PATH")
        .or_else(|| train_candidate_brain(train_games, minutes, train_seed_base))
        .unwrap_or_else(|| TeamBrain::fresh_with_seed(0xCA11_D1DA, candidate_id));

    // Frozen field (ids 1..pool): distinct-genome fresh brains, the incumbent +
    // diverse opponents the candidate must beat without being countered.
    let mut pool: Vec<TeamBrain> = (1..pool_size)
        .map(|id| {
            TeamBrain::fresh_with_seed(0xF0_0000u32.wrapping_add(id as u32 * 2_654_435_761), id)
        })
        .collect();
    // SOCCER_EVAL_BASELINE_PATH: replace the incumbent (id 1 == pool[0]) with a champion loaded
    // from a local learned-params file, making the fixtures a direct candidate-vs-champion gate.
    if let Some(brain) = brain_from_env_file("SOCCER_EVAL_BASELINE_PATH") {
        if let Some(first) = pool.first_mut() {
            *first = brain;
            eprintln!("eval_baseline_replaced_with_champion id={baseline_id}");
        }
    }
    // SOCCER_EVAL_ANALYTIC_FIELD=1: strip the net from every pool opponent so they play PURE
    // ANALYTIC (no net -> authoritative branch skipped -> the hand-built engine decides). The
    // candidate keeps its net, so this is the honest "authoritative neural vs analytic" meter
    // (the field's distinct genomes still give diverse analytic styles).
    if std::env::var("SOCCER_EVAL_ANALYTIC_FIELD")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        for brain in pool.iter_mut() {
            brain.neural = None;
        }
        eprintln!("eval_analytic_field=1 (pool plays pure analytic; candidate authoritative)");
    }

    let mut runner_config = EngineMatchRunnerConfig::default();
    runner_config.base.duration_seconds = minutes * 60.0;
    let runner = EngineMatchRunner::new(runner_config);

    let started = Instant::now();

    // Pre-build every fixture with the SAME sequential seeds/order as the old serial loop, so
    // parallel results are byte-identical (each match is independent + deterministic).
    struct Fixture {
        index: usize,
        opponent_idx: usize,
        opponent_id: usize,
        g: usize,
        seed: u32,
        candidate_home: bool,
    }
    let mut fixtures: Vec<Fixture> = Vec::new();
    let mut fixture_ctr = 0u32;
    for offset in 0..pool.len() {
        let opponent_id = offset + 1;
        for g in 0..eval_games_per_opp {
            let seed = holdout_seed_base
                .wrapping_add(fixture_ctr.wrapping_mul(2_246_822_519))
                .wrapping_add(g as u32);
            fixtures.push(Fixture {
                index: fixtures.len(),
                opponent_idx: offset,
                opponent_id,
                g,
                seed,
                candidate_home: g % 2 == 0,
            });
            fixture_ctr += 1;
        }
    }

    // Run fixtures on a bounded thread pool — each worker clones the runner and pulls the next
    // fixture off a shared atomic cursor. Capped low (default 3, override SOCCER_EVAL_PARALLELISM)
    // so we don't starve a co-running trainer.
    let workers = std::env::var("SOCCER_EVAL_PARALLELISM")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(3)
        .clamp(1, fixtures.len().max(1));
    eprintln!("eval_parallelism={workers} fixtures={}", fixtures.len());
    let next = std::sync::atomic::AtomicUsize::new(0);
    let collected = std::sync::Mutex::new(Vec::<(usize, MatchReport)>::new());
    std::thread::scope(|scope| {
        for _ in 0..workers {
            let next = &next;
            let collected = &collected;
            let fixtures = &fixtures;
            let pool = &pool;
            let candidate_brain = &candidate_brain;
            let mut runner = runner.clone();
            scope.spawn(move || loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some(fx) = fixtures.get(i) else { break };
                let opponent = &pool[fx.opponent_idx];
                let result = if fx.candidate_home {
                    play_holdout_fixture(
                        &mut runner,
                        candidate_id,
                        fx.opponent_id,
                        fx.seed,
                        candidate_brain,
                        opponent,
                    )
                } else {
                    play_holdout_fixture(
                        &mut runner,
                        fx.opponent_id,
                        candidate_id,
                        fx.seed,
                        opponent,
                        candidate_brain,
                    )
                };
                match result {
                    Ok(report) => {
                        eprintln!(
                            "  holdout {}v{} seed=0x{:08X} -> {}-{}",
                            report.home_id,
                            report.away_id,
                            fx.seed,
                            report.home_goals,
                            report.away_goals
                        );
                        collected.lock().unwrap().push((fx.index, report));
                    }
                    Err(e) => {
                        eprintln!("  fixture error (opp {}, g {}): {e}", fx.opponent_id, fx.g)
                    }
                }
            });
        }
    });
    // Sort back into the original fixture order so scoring is identical to the serial path.
    let mut collected = collected.into_inner().unwrap();
    collected.sort_by_key(|(i, _)| *i);
    let reports: Vec<MatchReport> = collected.into_iter().map(|(_, r)| r).collect();

    let verdict = evaluate_promotion(
        &reports,
        candidate_id,
        baseline_id,
        PromotionThresholds::default(),
    );

    println!(
        "\n----- verdict ({} held-out games, {:.1}s) -----",
        verdict.record.games(),
        started.elapsed().as_secs_f64()
    );
    println!(
        "record: {}W-{}D-{}L  GF/GA {}/{} (GD {:+})",
        verdict.record.wins,
        verdict.record.draws,
        verdict.record.losses,
        verdict.record.goals_for,
        verdict.record.goals_against,
        verdict.record.goal_difference(),
    );
    println!(
        "held-out Elo: candidate {:.1}  baseline {:.1}  Δ {:+.1}",
        verdict.candidate_elo, verdict.baseline_elo, verdict.elo_delta
    );
    println!(
        "cross-play: mean payoff vs field {:?}  vs baseline {:?}  worst-case {:?}",
        verdict
            .mean_payoff_vs_field
            .map(|p| (p * 1000.0).round() / 1000.0),
        verdict
            .payoff_vs_baseline
            .map(|p| (p * 1000.0).round() / 1000.0),
        verdict
            .worst_case
            .map(|(o, p)| (o, (p * 1000.0).round() / 1000.0)),
    );
    println!("Wilson lower bound: {:.3}", verdict.wilson_lower_bound);
    println!(
        "\nDECISION: {}",
        if verdict.promote { "PROMOTE" } else { "REJECT" }
    );
    for reason in &verdict.reasons {
        println!("  - {reason}");
    }
    std::process::exit(i32::from(!verdict.promote));
}
