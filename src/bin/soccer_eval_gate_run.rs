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

use soccer_engine::des::general::soccer::{
    enable_deterministic_formation_lp, MatchConfig, SoccerMarlAlgorithm,
    SoccerNeuralLearningBackend, SoccerNeuralLearningConfig, SoccerNeuralNetworkSnapshot,
    SoccerQPolicyOptions, SoccerTeamQPolicies, DEFAULT_SOCCER_MAPPO_TEAM_REWARD_SHARE,
};
use soccer_engine::des::general::soccer::SoccerMatch;
use soccer_engine::des::general::soccer_eval_gate::{evaluate_promotion, PromotionThresholds};
use soccer_engine::des::general::tournament::{
    EngineMatchRunner, EngineMatchRunnerConfig, MatchReport, TeamBrain, TournamentMatchContext,
    TournamentMatchRunner, TournamentStage,
};

/// Train a candidate neural brain by inline self-play carry-forward (the real
/// learner's per-process pattern, mirroring `measure_learning_ab`). Returns the
/// trained value/critic snapshot, or `None` for `games == 0` (fresh candidate).
fn train_candidate_snapshot(
    games: usize,
    minutes: f64,
    seed_base: u32,
) -> Option<SoccerNeuralNetworkSnapshot> {
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
    snapshot
}

/// Load a candidate neural snapshot directly from a local `learned-params.json`
/// artifact (the fully-local learner's durable policy file), via `SOCCER_EVAL_CANDIDATE_PATH`.
/// This lets the gate score the REAL accumulated local policy against the frozen field
/// (a held-out climb number over time) instead of only an inline-trained-from-fresh one.
/// The learned-params artifact embeds the snapshot under the `neuralNetwork` key.
fn candidate_snapshot_from_env_file() -> Option<SoccerNeuralNetworkSnapshot> {
    let path = std::env::var("SOCCER_EVAL_CANDIDATE_PATH").ok()?;
    let raw = match std::fs::read_to_string(&path) {
        Ok(raw) => raw,
        Err(e) => {
            eprintln!("eval_candidate_read_failed path={path}: {e}");
            return None;
        }
    };
    let value: serde_json::Value = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("eval_candidate_json_failed path={path}: {e}");
            return None;
        }
    };
    let nn = value.get("neuralNetwork")?.clone();
    match serde_json::from_value::<SoccerNeuralNetworkSnapshot>(nn) {
        Ok(s) => {
            eprintln!(
                "eval_candidate_loaded_from_file path={path} params={} training_steps={}",
                s.parameter_count, s.training_steps
            );
            Some(s)
        }
        Err(e) => {
            eprintln!("eval_candidate_snapshot_parse_failed path={path}: {e}");
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

    // Candidate brain (id 0): optionally trained, then FROZEN for the gate.
    let candidate_snapshot = train_candidate_snapshot(train_games, minutes, train_seed_base);
    let candidate_brain = match candidate_snapshot {
        Some(s) => TeamBrain::from_snapshot(s),
        None => TeamBrain::fresh_with_seed(0xCA11_D1DA, candidate_id),
    };

    // Frozen field (ids 1..pool): distinct-genome fresh brains, the incumbent +
    // diverse opponents the candidate must beat without being countered.
    let pool: Vec<TeamBrain> = (1..pool_size)
        .map(|id| TeamBrain::fresh_with_seed(0xF0_0000u32.wrapping_add(id as u32 * 2_654_435_761), id))
        .collect();

    let mut runner_config = EngineMatchRunnerConfig::default();
    runner_config.base.duration_seconds = minutes * 60.0;
    let mut runner = EngineMatchRunner::new(runner_config);

    let started = Instant::now();
    let mut reports: Vec<MatchReport> = Vec::new();
    let mut fixture = 0u32;
    for (offset, opponent) in pool.iter().enumerate() {
        let opponent_id = offset + 1;
        for g in 0..eval_games_per_opp {
            // Alternate home/away so the candidate's edge isn't a home-field artifact.
            let seed = holdout_seed_base
                .wrapping_add(fixture.wrapping_mul(2_246_822_519))
                .wrapping_add(g as u32);
            fixture += 1;
            let result = if g % 2 == 0 {
                play_holdout_fixture(
                    &mut runner,
                    candidate_id,
                    opponent_id,
                    seed,
                    &candidate_brain,
                    opponent,
                )
            } else {
                play_holdout_fixture(
                    &mut runner,
                    opponent_id,
                    candidate_id,
                    seed,
                    opponent,
                    &candidate_brain,
                )
            };
            match result {
                Ok(report) => {
                    eprintln!(
                        "  holdout {}v{} seed=0x{:08X} -> {}-{}",
                        report.home_id,
                        report.away_id,
                        seed,
                        report.home_goals,
                        report.away_goals
                    );
                    reports.push(report);
                }
                Err(e) => eprintln!("  fixture error (opp {opponent_id}, g {g}): {e}"),
            }
        }
    }

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
        verdict.mean_payoff_vs_field.map(|p| (p * 1000.0).round() / 1000.0),
        verdict.payoff_vs_baseline.map(|p| (p * 1000.0).round() / 1000.0),
        verdict
            .worst_case
            .map(|(o, p)| (o, (p * 1000.0).round() / 1000.0)),
    );
    println!("Wilson lower bound: {:.3}", verdict.wilson_lower_bound);
    println!("\nDECISION: {}", if verdict.promote { "PROMOTE" } else { "REJECT" });
    for reason in &verdict.reasons {
        println!("  - {reason}");
    }
    std::process::exit(i32::from(!verdict.promote));
}
