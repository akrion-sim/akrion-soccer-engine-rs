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
    enable_deterministic_formation_lp, MatchConfig, MatchSummary, SoccerMarlAlgorithm,
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
    if let Ok(neural) = serde_json::from_value::<SoccerNeuralNetworkSnapshot>(value.clone()) {
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

#[derive(Clone, Debug)]
struct HoldoutReport {
    report: MatchReport,
    advancement: AdvancementFixture,
}

#[derive(Clone, Copy, Debug, Default)]
struct AdvancementFixture {
    candidate_forward_passes: u32,
    opponent_forward_passes: u32,
    candidate_completed_passes: u32,
    opponent_completed_passes: u32,
    candidate_attempted_passes: u32,
    opponent_attempted_passes: u32,
    candidate_turnovers: u32,
    opponent_turnovers: u32,
    candidate_pass_gain_yards: f64,
    opponent_pass_gain_yards: f64,
}

impl AdvancementFixture {
    fn from_summary(summary: &MatchSummary, candidate_home: bool) -> Self {
        let stats = &summary.stats;
        if candidate_home {
            AdvancementFixture {
                candidate_forward_passes: stats.passes_completed_forward_home,
                opponent_forward_passes: stats.passes_completed_forward_away,
                candidate_completed_passes: stats.passes_completed_home,
                opponent_completed_passes: stats.passes_completed_away,
                candidate_attempted_passes: stats.passes_attempted_home,
                opponent_attempted_passes: stats.passes_attempted_away,
                candidate_turnovers: stats.interceptions_away,
                opponent_turnovers: stats.interceptions_home,
                candidate_pass_gain_yards: finite_yards(stats.completed_pass_gain_yards_home),
                opponent_pass_gain_yards: finite_yards(stats.completed_pass_gain_yards_away),
            }
        } else {
            AdvancementFixture {
                candidate_forward_passes: stats.passes_completed_forward_away,
                opponent_forward_passes: stats.passes_completed_forward_home,
                candidate_completed_passes: stats.passes_completed_away,
                opponent_completed_passes: stats.passes_completed_home,
                candidate_attempted_passes: stats.passes_attempted_away,
                opponent_attempted_passes: stats.passes_attempted_home,
                candidate_turnovers: stats.interceptions_home,
                opponent_turnovers: stats.interceptions_away,
                candidate_pass_gain_yards: finite_yards(stats.completed_pass_gain_yards_away),
                opponent_pass_gain_yards: finite_yards(stats.completed_pass_gain_yards_home),
            }
        }
    }

    fn forward_pass_margin(&self) -> i32 {
        self.candidate_forward_passes as i32 - self.opponent_forward_passes as i32
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct AdvancementRecord {
    wins: u32,
    draws: u32,
    losses: u32,
    candidate_forward_passes: u32,
    opponent_forward_passes: u32,
    candidate_completed_passes: u32,
    opponent_completed_passes: u32,
    candidate_attempted_passes: u32,
    opponent_attempted_passes: u32,
    candidate_turnovers: u32,
    opponent_turnovers: u32,
    candidate_pass_gain_yards: f64,
    opponent_pass_gain_yards: f64,
}

impl AdvancementRecord {
    fn from_fixtures(fixtures: &[HoldoutReport]) -> Self {
        let mut record = AdvancementRecord::default();
        for fixture in fixtures {
            record.add(fixture.advancement);
        }
        record
    }

    fn add(&mut self, fixture: AdvancementFixture) {
        match fixture
            .candidate_forward_passes
            .cmp(&fixture.opponent_forward_passes)
        {
            std::cmp::Ordering::Greater => self.wins += 1,
            std::cmp::Ordering::Equal => self.draws += 1,
            std::cmp::Ordering::Less => self.losses += 1,
        }
        self.candidate_forward_passes = self
            .candidate_forward_passes
            .saturating_add(fixture.candidate_forward_passes);
        self.opponent_forward_passes = self
            .opponent_forward_passes
            .saturating_add(fixture.opponent_forward_passes);
        self.candidate_completed_passes = self
            .candidate_completed_passes
            .saturating_add(fixture.candidate_completed_passes);
        self.opponent_completed_passes = self
            .opponent_completed_passes
            .saturating_add(fixture.opponent_completed_passes);
        self.candidate_attempted_passes = self
            .candidate_attempted_passes
            .saturating_add(fixture.candidate_attempted_passes);
        self.opponent_attempted_passes = self
            .opponent_attempted_passes
            .saturating_add(fixture.opponent_attempted_passes);
        self.candidate_turnovers = self
            .candidate_turnovers
            .saturating_add(fixture.candidate_turnovers);
        self.opponent_turnovers = self
            .opponent_turnovers
            .saturating_add(fixture.opponent_turnovers);
        self.candidate_pass_gain_yards += fixture.candidate_pass_gain_yards;
        self.opponent_pass_gain_yards += fixture.opponent_pass_gain_yards;
    }

    fn games(&self) -> u32 {
        self.wins + self.draws + self.losses
    }

    fn mean_score(&self) -> f64 {
        let games = self.games();
        if games == 0 {
            0.0
        } else {
            (self.wins as f64 + 0.5 * self.draws as f64) / games as f64
        }
    }

    fn forward_pass_margin(&self) -> i32 {
        self.candidate_forward_passes as i32 - self.opponent_forward_passes as i32
    }

    fn candidate_forward_pass_rate(&self) -> f64 {
        ratio(
            self.candidate_forward_passes,
            self.candidate_completed_passes,
        )
    }

    fn opponent_forward_pass_rate(&self) -> f64 {
        ratio(self.opponent_forward_passes, self.opponent_completed_passes)
    }

    fn forward_pass_rate_margin(&self) -> f64 {
        self.candidate_forward_pass_rate() - self.opponent_forward_pass_rate()
    }

    fn candidate_completion_rate(&self) -> f64 {
        ratio(self.candidate_completed_passes, self.candidate_attempted_passes)
    }

    fn opponent_completion_rate(&self) -> f64 {
        ratio(self.opponent_completed_passes, self.opponent_attempted_passes)
    }

    fn completion_rate_margin(&self) -> f64 {
        self.candidate_completion_rate() - self.opponent_completion_rate()
    }

    fn candidate_net_forward_passes(&self) -> i32 {
        self.candidate_forward_passes as i32 - self.candidate_turnovers as i32
    }

    fn opponent_net_forward_passes(&self) -> i32 {
        self.opponent_forward_passes as i32 - self.opponent_turnovers as i32
    }

    fn net_forward_pass_margin(&self) -> i32 {
        self.candidate_net_forward_passes() - self.opponent_net_forward_passes()
    }

    fn pass_gain_yards_margin(&self) -> f64 {
        self.candidate_pass_gain_yards - self.opponent_pass_gain_yards
    }
}

fn ratio(numerator: u32, denominator: u32) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn forward_pass_gate_reasons(
    advancement: &AdvancementRecord,
    min_forward_pass_margin: i32,
    min_net_forward_pass_margin: i32,
    min_forward_pass_rate_margin: f64,
) -> Vec<String> {
    let mut reasons = Vec::new();
    if advancement.forward_pass_margin() <= min_forward_pass_margin {
        reasons.push(format!(
            "forward-pass margin {:+} <= required {:+}",
            advancement.forward_pass_margin(),
            min_forward_pass_margin,
        ));
    }
    if advancement.net_forward_pass_margin() <= min_net_forward_pass_margin {
        reasons.push(format!(
            "net forward-pass margin {:+} <= required {:+}",
            advancement.net_forward_pass_margin(),
            min_net_forward_pass_margin,
        ));
    }
    if advancement.forward_pass_rate_margin() <= min_forward_pass_rate_margin {
        reasons.push(format!(
            "forward-pass rate margin {:+.1}pp <= required {:+.1}pp",
            advancement.forward_pass_rate_margin() * 100.0,
            min_forward_pass_rate_margin * 100.0,
        ));
    }
    reasons
}

fn forward_pass_sample_reasons(advancement: &AdvancementRecord, min_games: usize) -> Vec<String> {
    let mut reasons = Vec::new();
    if (advancement.games() as usize) < min_games {
        reasons.push(format!(
            "insufficient forward-pass evidence: {} held-out games < min {}",
            advancement.games(),
            min_games,
        ));
    }
    reasons
}

fn goal_diff_gate_reasons(goal_difference: i32, min_goal_diff_margin: i32) -> Vec<String> {
    let mut reasons = Vec::new();
    if goal_difference <= min_goal_diff_margin {
        reasons.push(format!(
            "goal difference {goal_difference:+} <= required {min_goal_diff_margin:+}"
        ));
    }
    reasons
}

fn eval_gate_promotes(
    require_forward_pass_climb: bool,
    scoreline_promote: bool,
    advancement_reasons: &[String],
) -> bool {
    if require_forward_pass_climb {
        advancement_reasons.is_empty()
    } else {
        scoreline_promote
    }
}

fn finite_yards(value: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

fn env_bool(key: &str, default: bool) -> bool {
    std::env::var(key)
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(default)
}

fn env_i32(key: &str, default: i32) -> i32 {
    std::env::var(key)
        .ok()
        .and_then(|value| value.trim().parse::<i32>().ok())
        .unwrap_or(default)
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|value| value.trim().parse::<f64>().ok())
        .unwrap_or(default)
}

/// A held-out frozen-vs-frozen fixture between two brains. Builds the context with
/// learning OFF for both sides and a held-out seed, plays it, and returns the
/// `HoldoutReport`.
fn play_holdout_fixture(
    runner: &mut EngineMatchRunner,
    home_id: usize,
    away_id: usize,
    seed: u32,
    candidate_home: bool,
    home: &TeamBrain,
    away: &TeamBrain,
) -> Result<HoldoutReport, String> {
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
    let advancement = AdvancementFixture::from_summary(&outcome.summary, candidate_home);
    Ok(HoldoutReport {
        report: MatchReport {
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
        },
        advancement,
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
    let candidate_brain = if std::env::var_os("SOCCER_EVAL_CANDIDATE_PATH").is_some() {
        brain_from_env_file("SOCCER_EVAL_CANDIDATE_PATH").unwrap_or_else(|| {
            eprintln!(
                "eval_candidate_load_required_failed var=SOCCER_EVAL_CANDIDATE_PATH; aborting"
            );
            std::process::exit(2);
        })
    } else {
        train_candidate_brain(train_games, minutes, train_seed_base)
            .unwrap_or_else(|| TeamBrain::fresh_with_seed(0xCA11_D1DA, candidate_id))
    };

    // Frozen field (ids 1..pool): distinct-genome fresh brains, the incumbent +
    // diverse opponents the candidate must beat without being countered.
    let mut pool: Vec<TeamBrain> = (1..pool_size)
        .map(|id| {
            TeamBrain::fresh_with_seed(0xF0_0000u32.wrapping_add(id as u32 * 2_654_435_761), id)
        })
        .collect();
    // SOCCER_EVAL_BASELINE_PATH: replace the incumbent (id 1 == pool[0]) with a champion loaded
    // from a local learned-params file, making the fixtures a direct candidate-vs-champion gate.
    if std::env::var_os("SOCCER_EVAL_BASELINE_PATH").is_some() {
        let brain = brain_from_env_file("SOCCER_EVAL_BASELINE_PATH").unwrap_or_else(|| {
            eprintln!("eval_baseline_load_required_failed var=SOCCER_EVAL_BASELINE_PATH; aborting");
            std::process::exit(2);
        });
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
    let collected = std::sync::Mutex::new(Vec::<(usize, HoldoutReport)>::new());
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
                        fx.candidate_home,
                        candidate_brain,
                        opponent,
                    )
                } else {
                    play_holdout_fixture(
                        &mut runner,
                        fx.opponent_id,
                        candidate_id,
                        fx.seed,
                        fx.candidate_home,
                        opponent,
                        candidate_brain,
                    )
                };
                match result {
                    Ok(report) => {
                        eprintln!(
                            "  holdout {}v{} seed=0x{:08X} -> {}-{} forward_passes={}-{} margin={:+}",
                            report.report.home_id,
                            report.report.away_id,
                            fx.seed,
                            report.report.home_goals,
                            report.report.away_goals,
                            report.advancement.candidate_forward_passes,
                            report.advancement.opponent_forward_passes,
                            report.advancement.forward_pass_margin(),
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
    let holdout_reports: Vec<HoldoutReport> = collected.into_iter().map(|(_, r)| r).collect();
    let advancement = AdvancementRecord::from_fixtures(&holdout_reports);
    let reports: Vec<MatchReport> = holdout_reports
        .iter()
        .map(|holdout| holdout.report.clone())
        .collect();

    let verdict = evaluate_promotion(
        &reports,
        candidate_id,
        baseline_id,
        PromotionThresholds::default(),
    );
    let require_forward_pass_climb = env_bool("SOCCER_EVAL_REQUIRE_FORWARD_PASS_CLIMB", false);
    let min_forward_pass_margin = env_i32("SOCCER_EVAL_MIN_FORWARD_PASS_MARGIN", 0);
    let min_net_forward_pass_margin = env_i32("SOCCER_EVAL_MIN_NET_FORWARD_PASS_MARGIN", 0);
    let min_forward_pass_rate_margin = env_f64("SOCCER_EVAL_MIN_FORWARD_PASS_RATE_MARGIN", 0.0);
    let min_forward_pass_games = env_usize("SOCCER_EVAL_MIN_FORWARD_PASS_GAMES", 8);
    let require_goal_diff_for_forward_pass =
        env_bool("SOCCER_EVAL_REQUIRE_GOAL_DIFF_FOR_FORWARD_PASS", false);
    let min_goal_diff_margin = env_i32("SOCCER_EVAL_MIN_GOAL_DIFF_MARGIN", 0);
    let advancement_reasons = if require_forward_pass_climb {
        let mut reasons = forward_pass_gate_reasons(
            &advancement,
            min_forward_pass_margin,
            min_net_forward_pass_margin,
            min_forward_pass_rate_margin,
        );
        reasons.extend(forward_pass_sample_reasons(
            &advancement,
            min_forward_pass_games,
        ));
        if require_goal_diff_for_forward_pass {
            reasons.extend(goal_diff_gate_reasons(
                verdict.record.goal_difference(),
                min_goal_diff_margin,
            ));
        }
        reasons
    } else {
        Vec::new()
    };
    let advancement_promote = !require_forward_pass_climb || advancement_reasons.is_empty();
    let promote = eval_gate_promotes(
        require_forward_pass_climb,
        verdict.promote,
        &advancement_reasons,
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
        "advancement: {}W-{}D-{}L by completed_forward_passes  FP {}/{} (margin {:+}, fixture_score {:.3})  net_forward {}/{} (margin {:+})  forward_share {:.1}%/{:.1}% (margin {:+.1}pp)  completed_passes {}/{}  turnovers {}/{}  pass_gain_yards_margin {:+.1}",
        advancement.wins,
        advancement.draws,
        advancement.losses,
        advancement.candidate_forward_passes,
        advancement.opponent_forward_passes,
        advancement.forward_pass_margin(),
        advancement.mean_score(),
        advancement.candidate_net_forward_passes(),
        advancement.opponent_net_forward_passes(),
        advancement.net_forward_pass_margin(),
        advancement.candidate_forward_pass_rate() * 100.0,
        advancement.opponent_forward_pass_rate() * 100.0,
        advancement.forward_pass_rate_margin() * 100.0,
        advancement.candidate_completed_passes,
        advancement.opponent_completed_passes,
        advancement.candidate_turnovers,
        advancement.opponent_turnovers,
        advancement.pass_gain_yards_margin(),
    );
    println!(
        "completion_rate cand={:.4} opp={:.4} margin={:+.4} (cand {}/{} opp {}/{})",
        advancement.candidate_completion_rate(),
        advancement.opponent_completion_rate(),
        advancement.completion_rate_margin(),
        advancement.candidate_completed_passes,
        advancement.candidate_attempted_passes,
        advancement.opponent_completed_passes,
        advancement.opponent_attempted_passes,
    );
    if require_forward_pass_climb {
        println!("scoreline gate: diagnostic only with forward-pass climb");
        println!(
            "forward-pass gate: margin>{:+} net_margin>{:+} rate_margin>{:+.1}pp min_games={} goal_diff_floor={} -> {}",
            min_forward_pass_margin,
            min_net_forward_pass_margin,
            min_forward_pass_rate_margin * 100.0,
            min_forward_pass_games,
            if require_goal_diff_for_forward_pass {
                format!(">{min_goal_diff_margin:+}")
            } else {
                "off".to_string()
            },
            if advancement_promote {
                "PASS"
            } else {
                "REJECT"
            },
        );
    }
    println!("\nDECISION: {}", if promote { "PROMOTE" } else { "REJECT" });
    for reason in &verdict.reasons {
        if require_forward_pass_climb {
            println!("  - scoreline diagnostic: {reason}");
        } else {
            println!("  - {reason}");
        }
    }
    for reason in &advancement_reasons {
        println!("  - {reason}");
    }
    std::process::exit(i32::from(!promote));
}

#[cfg(test)]
mod tests {
    use super::*;
    use soccer_engine::des::general::soccer::MatchStats;

    #[test]
    fn advancement_record_uses_completed_forward_passes_not_score_or_shots() {
        let mut stats = MatchStats::default();
        stats.shots_home = 1;
        stats.shots_away = 18;
        stats.shots_on_target_home = 0;
        stats.shots_on_target_away = 9;
        stats.passes_completed_home = 20;
        stats.passes_completed_away = 40;
        stats.passes_completed_forward_home = 12;
        stats.passes_completed_forward_away = 4;
        stats.interceptions_home = 3;
        stats.interceptions_away = 1;
        stats.completed_pass_gain_yards_home = 72.0;
        stats.completed_pass_gain_yards_away = 24.0;
        let summary = MatchSummary {
            score_home: 0,
            score_away: 5,
            ticks: 100,
            simulated_seconds: 90.0,
            stats,
        };
        let fixture = AdvancementFixture::from_summary(&summary, true);
        let report = HoldoutReport {
            report: MatchReport {
                stage: TournamentStage::Group,
                home_id: 0,
                away_id: 1,
                home_name: "candidate".to_string(),
                away_name: "baseline".to_string(),
                home_goals: summary.score_home,
                away_goals: summary.score_away,
                shootout_winner: None,
                home_training_steps: 0,
                away_training_steps: 0,
            },
            advancement: fixture,
        };

        let record = AdvancementRecord::from_fixtures(&[report]);
        assert_eq!(record.wins, 1);
        assert_eq!(record.losses, 0);
        assert_eq!(record.forward_pass_margin(), 8);
        assert_eq!(record.net_forward_pass_margin(), 10);
        assert!((record.forward_pass_rate_margin() - 0.5).abs() < 1e-12);
        assert!((record.mean_score() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn forward_pass_gate_uses_totals_not_fixture_score() {
        let mut record = AdvancementRecord::default();
        record.add(AdvancementFixture {
            candidate_forward_passes: 0,
            opponent_forward_passes: 1,
            candidate_completed_passes: 10,
            opponent_completed_passes: 1,
            candidate_turnovers: 0,
            opponent_turnovers: 0,
            candidate_pass_gain_yards: 0.0,
            opponent_pass_gain_yards: 4.0,
        });
        record.add(AdvancementFixture {
            candidate_forward_passes: 20,
            opponent_forward_passes: 2,
            candidate_completed_passes: 20,
            opponent_completed_passes: 20,
            candidate_turnovers: 0,
            opponent_turnovers: 0,
            candidate_pass_gain_yards: 120.0,
            opponent_pass_gain_yards: 12.0,
        });
        record.add(AdvancementFixture {
            candidate_forward_passes: 0,
            opponent_forward_passes: 1,
            candidate_completed_passes: 10,
            opponent_completed_passes: 1,
            candidate_turnovers: 0,
            opponent_turnovers: 0,
            candidate_pass_gain_yards: 0.0,
            opponent_pass_gain_yards: 4.0,
        });

        assert!(
            record.mean_score() < 0.5,
            "fixture score is intentionally below parity"
        );
        assert!(record.forward_pass_margin() > 0);
        assert!(record.net_forward_pass_margin() > 0);
        assert!(record.forward_pass_rate_margin() > 0.0);
        assert!(
            forward_pass_gate_reasons(&record, 0, 0, 0.0).is_empty(),
            "direct forward-pass totals should pass without a fixture-WDL veto"
        );
    }

    #[test]
    fn forward_pass_mode_decision_uses_advancement_not_scoreline() {
        let advancement_reasons = Vec::new();

        assert!(
            eval_gate_promotes(true, false, &advancement_reasons),
            "forward-pass mode should use the dense advancement scoreboard, not scoreline"
        );
        assert!(
            eval_gate_promotes(true, true, &advancement_reasons),
            "forward-pass mode should pass when advancement gates pass"
        );
        assert!(
            !eval_gate_promotes(false, false, &advancement_reasons),
            "scoreline verdict remains authoritative when forward-pass climb is not required"
        );
        assert!(
            !eval_gate_promotes(true, true, &["net regression".to_string()]),
            "forward-pass mode must still reject advancement failures"
        );
    }

    #[test]
    fn goal_diff_gate_reasons_can_enforce_optional_floor() {
        let reasons = goal_diff_gate_reasons(0, 0);
        assert!(
            reasons
                .iter()
                .any(|reason| reason.contains("goal difference")),
            "optional goal-difference floor should still be available when explicitly enabled: {reasons:?}"
        );
        assert!(goal_diff_gate_reasons(1, 0).is_empty());
    }

    #[test]
    fn forward_pass_gate_rejects_turnover_net_regression() {
        let mut record = AdvancementRecord::default();
        record.add(AdvancementFixture {
            candidate_forward_passes: 12,
            opponent_forward_passes: 8,
            candidate_completed_passes: 24,
            opponent_completed_passes: 24,
            candidate_turnovers: 8,
            opponent_turnovers: 0,
            candidate_pass_gain_yards: 72.0,
            opponent_pass_gain_yards: 48.0,
        });

        assert!(record.forward_pass_margin() > 0);
        assert!(record.net_forward_pass_margin() < 0);
        let reasons = forward_pass_gate_reasons(&record, 0, 0, 0.0);
        assert!(
            reasons
                .iter()
                .any(|reason| reason.contains("net forward-pass margin")),
            "turnovers must count against the forward-pass climb gate: {reasons:?}"
        );
    }
}
