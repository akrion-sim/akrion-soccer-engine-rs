//! Isolated A/B for the terminal won-game reward, judged on held-out Elo.
//!
//! The reward gate (`match_outcome_reward_enabled`) and the advantage-normalization
//! gate are read once per process (OnceLock), so the two arms CANNOT differ inside a
//! single process — each arm is its own process, and its trained brain is persisted to
//! JSON. A third `eval` process then plays the two frozen brains head-to-head over a
//! HELD-OUT seed range (disjoint from training) and folds the fixtures through
//! `soccer_eval_gate::evaluate_promotion` for a Wilson-bounded held-out verdict.
//!
//! To isolate the won-game label as the *only* difference, run BOTH arms with
//! advantage normalization on (the treatment forces it on anyway via the hardening
//! coupling), so the arms differ only by `DD_SOCCER_ENABLE_MATCH_OUTCOME_REWARD`.
//!
//! Usage:
//!   soccer_outcome_ab_run train <out.json> [games=40] [minutes=3] [seed_base_hex=71A10000]
//!   soccer_outcome_ab_run eval  <candidate.json> <baseline.json> [games=30] [minutes=3] [holdout_hex=E7A10000]
//!
//! Orchestrated by `scripts/run_outcome_ab.sh`.

use std::sync::Arc;
use std::time::{Duration, Instant};

use soccer_engine::des::general::soccer::{
    enable_deterministic_formation_lp, learned_mpc_objective_enabled, match_outcome_reward_enabled,
    match_outcome_win_reward_points, MatchConfig, SoccerMarlAlgorithm, SoccerMatch,
    SoccerMpcObjectiveHead, SoccerNeuralBlendConfig, SoccerNeuralBlendMode,
    SoccerNeuralLearningBackend, SoccerNeuralLearningConfig, SoccerNeuralNetworkSnapshot,
    SoccerQPolicyOptions, SoccerTeamQPolicies, Team, DEFAULT_SOCCER_MAPPO_TEAM_REWARD_SHARE,
};
use soccer_engine::des::general::soccer_eval_gate::{evaluate_promotion, PromotionThresholds};
use soccer_engine::des::general::tournament::{
    EngineMatchRunner, EngineMatchRunnerConfig, MatchReport, TeamBrain, TournamentMatchContext,
    TournamentMatchRunner, TournamentStage,
};

fn parse_hex(s: Option<&String>, default: u32) -> u32 {
    s.and_then(|raw| {
        let raw = raw.trim_start_matches("0x");
        u32::from_str_radix(raw, 16).ok()
    })
    .unwrap_or(default)
}

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(default)
}

fn env_bool(name: &str, default: bool) -> bool {
    env_bool_opt(name).unwrap_or(default)
}

fn env_bool_opt(name: &str) -> Option<bool> {
    std::env::var(name).ok().map(|v| {
        matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(default)
}

fn env_neural_blend(mut blend: SoccerNeuralBlendConfig) -> SoccerNeuralBlendConfig {
    if let Ok(value) = std::env::var("SOCCER_NEURAL_BLEND_MODE") {
        blend.mode = match value.trim().to_ascii_lowercase().as_str() {
            "off" => SoccerNeuralBlendMode::Off,
            "additive" => SoccerNeuralBlendMode::Additive,
            "tiebreak" | "tie-break" | "tie_break" => SoccerNeuralBlendMode::TieBreak,
            "confidence" | "confidencegated" | "confidence-gated" | "confidence_gated" => {
                SoccerNeuralBlendMode::ConfidenceGated
            }
            "authoritative" => SoccerNeuralBlendMode::Authoritative,
            other => panic!("SOCCER_NEURAL_BLEND_MODE={other:?} is invalid"),
        };
    }
    blend.lambda = env_f64("SOCCER_NEURAL_BLEND_LAMBDA", blend.lambda);
    blend.warmup_steps = env_usize("SOCCER_NEURAL_BLEND_WARMUP_STEPS", blend.warmup_steps);
    blend.candidates = env_usize("SOCCER_NEURAL_BLEND_CANDIDATES", blend.candidates);
    blend.world_model = env_bool("SOCCER_NEURAL_WORLD_MODEL", blend.world_model);
    blend.mcts_enabled = env_bool("SOCCER_NEURAL_MCTS_ENABLED", blend.mcts_enabled);
    blend.mcts_simulations = env_usize("SOCCER_NEURAL_MCTS_SIMULATIONS", blend.mcts_simulations);
    blend.mcts_candidates = env_usize("SOCCER_NEURAL_MCTS_CANDIDATES", blend.mcts_candidates);
    blend.mcts_depth = env_usize("SOCCER_NEURAL_MCTS_DEPTH", blend.mcts_depth);
    blend.mcts_model_weight = env_f64("SOCCER_NEURAL_MCTS_MODEL_WEIGHT", blend.mcts_model_weight);
    if let Some(actor_critic) = env_bool_opt("SOCCER_NEURAL_ACTOR_CRITIC")
        .or_else(|| env_bool_opt("SOCCER_ENABLE_ACTOR_CRITIC"))
    {
        blend.actor_critic = actor_critic;
    }
    blend
}

/// Train and held-out evaluation exercise the learned team's complete isolated actor plus the same
/// MCTS decision seam; otherwise actor/planner-teacher changes are trained but silently absent at
/// the gate. Explicit env overrides remain available for deliberate ablations.
fn analytic_experiment_neural_blend(mut blend: SoccerNeuralBlendConfig) -> SoccerNeuralBlendConfig {
    blend.actor_critic = true;
    blend.mcts_enabled = true;
    env_neural_blend(blend)
}

fn print_neural_blend(label: &str, blend: SoccerNeuralBlendConfig) {
    println!(
        "[{label}] neural_blend: mode={:?} lambda={} warmup_steps={} candidates={} \
         actor_critic={} world_model={} mcts_enabled={} mcts_simulations={} \
         mcts_candidates={} mcts_depth={} mcts_model_weight={}",
        blend.mode,
        blend.lambda,
        blend.warmup_steps,
        blend.candidates,
        blend.actor_critic,
        blend.world_model,
        blend.mcts_enabled,
        blend.mcts_simulations,
        blend.mcts_candidates,
        blend.mcts_depth,
        blend.mcts_model_weight,
    );
}

/// Train a candidate by inline self-play carry-forward, honoring whatever env gates
/// this process was launched with, and return the final value/actor snapshot.
fn train(out_path: &str, games: usize, minutes: f64, seed_base: u32) {
    enable_deterministic_formation_lp();
    println!(
        "[train] out={out_path} games={games} minutes={minutes} seed_base=0x{seed_base:08X} \
         match_outcome_reward={} effective_match_win_reward={}",
        match_outcome_reward_enabled(),
        match_outcome_win_reward_points()
    );
    let mut neural = SoccerNeuralLearningConfig {
        enabled: true,
        backend: SoccerNeuralLearningBackend::Inline,
        marl_algorithm: SoccerMarlAlgorithm::Mappo,
        mappo_team_reward_share: DEFAULT_SOCCER_MAPPO_TEAM_REWARD_SHARE,
        ..SoccerNeuralLearningConfig::default()
    };
    // Value-target window / capacity overrides for the climb factorial. Default = unchanged, so
    // an arm that sets none of these is byte-identical to before.
    neural.target_scale = env_f64("SOCCER_NEURAL_TARGET_SCALE", neural.target_scale);
    neural.target_clip = env_f64("SOCCER_NEURAL_TARGET_CLIP", neural.target_clip);
    neural.target_popart_enabled =
        env_bool("SOCCER_NEURAL_TARGET_POPART", neural.target_popart_enabled);
    neural.hidden_units = env_usize("SOCCER_NEURAL_HIDDEN_UNITS", neural.hidden_units);
    println!(
        "[train] neural window: target_scale={} target_clip={} popart={} hidden_units={}",
        neural.target_scale, neural.target_clip, neural.target_popart_enabled, neural.hidden_units
    );
    let mut policies = Arc::new(SoccerTeamQPolicies::new(SoccerQPolicyOptions::default()));
    let mut snapshot: Option<SoccerNeuralNetworkSnapshot> = None;
    // Carried MPC-objective (executor) head: install per game so the learned aim/lead residual is
    // APPLIED live, drain + RWR-train it after each game, and carry the warm head forward — mirrors
    // main_soccer_learning_run. Gated on DD_SOCCER_ENABLE_LEARNED_MPC_OBJECTIVE, so when the flag is
    // off no head is installed and the arm is byte-identical to before (all prior A/Bs unchanged).
    let mut mpc_head: Option<SoccerMpcObjectiveHead> = None;
    let started = Instant::now();
    for g in 0..games {
        let mut config = MatchConfig {
            duration_seconds: minutes * 60.0,
            learning_enabled: true,
            neural_learning: neural.clone(),
            seed: seed_base.wrapping_add(g as u32),
            ..MatchConfig::default()
        };
        config.neural_blend.actor_critic = true;
        config.neural_blend = env_neural_blend(config.neural_blend);
        if g == 0 {
            print_neural_blend("train", config.neural_blend);
        }
        let total_ticks = config.total_ticks();
        let mut sim = SoccerMatch::default_11v11(config).with_team_policies((*policies).clone());
        sim.set_uniform_elite_players();
        if let Some(s) = snapshot.as_ref() {
            if let Err(e) = sim.set_neural_network_snapshot(s.clone()) {
                eprintln!("[train] game {g}: snapshot install failed: {e}");
            }
        }
        // Install the carried executor head so this game applies the learned aim/lead residual and
        // records fresh samples. Seeded on first install; no-op when the gate is off.
        if learned_mpc_objective_enabled() {
            let head = mpc_head.get_or_insert_with(|| SoccerMpcObjectiveHead::new(seed_base));
            sim.set_mpc_objective_head(head.clone());
        }
        for _ in 0..total_ticks {
            sim.run_time_step();
        }
        sim.drain_neural_learning(Duration::from_millis(100));
        // Drain this game's executor-head samples and RWR-train the carried head (warm for next game).
        if learned_mpc_objective_enabled() {
            let samples = sim.drain_mpc_objective_samples();
            if !samples.is_empty() {
                if let Some(head) = mpc_head.as_mut() {
                    head.train_rwr(&samples, 0.05);
                }
            }
        }
        if let Some(p) = sim.team_policies() {
            policies = Arc::new(p.clone());
        }
        if let Some(s) = sim.neural_network_snapshot() {
            snapshot = Some(s);
        }
        let s = sim.summary();
        eprintln!(
            "[train] game {:>2}/{games} score {}-{} ({:.0}s elapsed)",
            g + 1,
            s.score_home,
            s.score_away,
            started.elapsed().as_secs_f64()
        );
    }
    let snapshot = snapshot.expect("training produced a snapshot");
    let json = serde_json::to_string(&snapshot).expect("serialize snapshot");
    std::fs::write(out_path, json).expect("write snapshot json");
    println!(
        "[train] wrote {out_path} after {games} games ({:.0}s)",
        started.elapsed().as_secs_f64()
    );
}

fn train_analytic(out_path: &str, games: usize, minutes: f64, seed_base: u32) {
    enable_deterministic_formation_lp();
    let analytic_pool_size = env_usize("SOCCER_TRAIN_ANALYTIC_POOL_SIZE", 1).max(1);
    let alternate_sides = env_bool("SOCCER_TRAIN_ANALYTIC_ALTERNATE_SIDES", false);
    println!(
        "[train-analytic] out={out_path} games={games} minutes={minutes} seed_base=0x{seed_base:08X} \
         analytic_pool_size={analytic_pool_size} alternate_sides={alternate_sides} \
         match_outcome_reward={} effective_match_win_reward={}",
        match_outcome_reward_enabled(),
        match_outcome_win_reward_points()
    );
    let mut neural = SoccerNeuralLearningConfig {
        enabled: true,
        backend: SoccerNeuralLearningBackend::Inline,
        marl_algorithm: SoccerMarlAlgorithm::Mappo,
        mappo_team_reward_share: DEFAULT_SOCCER_MAPPO_TEAM_REWARD_SHARE,
        ..SoccerNeuralLearningConfig::default()
    };
    neural.target_scale = env_f64("SOCCER_NEURAL_TARGET_SCALE", neural.target_scale);
    neural.target_clip = env_f64("SOCCER_NEURAL_TARGET_CLIP", neural.target_clip);
    neural.target_popart_enabled =
        env_bool("SOCCER_NEURAL_TARGET_POPART", neural.target_popart_enabled);
    neural.hidden_units = env_usize("SOCCER_NEURAL_HIDDEN_UNITS", neural.hidden_units);
    println!(
        "[train-analytic] neural window: target_scale={} target_clip={} popart={} hidden_units={}",
        neural.target_scale, neural.target_clip, neural.target_popart_enabled, neural.hidden_units
    );
    let mut policies = Arc::new(SoccerTeamQPolicies::new(SoccerQPolicyOptions::default()));
    let mut snapshot: Option<SoccerNeuralNetworkSnapshot> = None;
    let mut mpc_head: Option<SoccerMpcObjectiveHead> = None;
    let started = Instant::now();
    for g in 0..games {
        let mut config = MatchConfig {
            duration_seconds: minutes * 60.0,
            learning_enabled: true,
            neural_learning: neural.clone(),
            seed: seed_base.wrapping_add(g as u32),
            ..MatchConfig::default()
        };
        config.neural_blend = analytic_experiment_neural_blend(config.neural_blend);
        if g == 0 {
            print_neural_blend("train-analytic", config.neural_blend);
        }
        let total_ticks = config.total_ticks();
        let candidate_team = if alternate_sides && g % 2 == 1 {
            Team::Away
        } else {
            Team::Home
        };
        let opponent_team = candidate_team.other();
        let opponent_id = 1 + (g % analytic_pool_size);
        let opponent = TeamBrain::fresh_with_seed(
            0xF0_0000u32.wrapping_add(opponent_id as u32 * 2_654_435_761),
            opponent_id,
        );
        let mut sim = SoccerMatch::default_11v11(config).with_team_policies((*policies).clone());
        sim.set_uniform_elite_players();
        sim.set_team_neural_brain(candidate_team, snapshot.clone(), false)
            .expect("install neural learner");
        sim.disable_team_neural_brain(opponent_team);
        sim.set_team_tactical_genome(opponent_team, opponent.genome.clone());
        if learned_mpc_objective_enabled() {
            let head = mpc_head.get_or_insert_with(|| SoccerMpcObjectiveHead::new(seed_base));
            sim.set_mpc_objective_head(head.clone());
        }
        for _ in 0..total_ticks {
            sim.run_time_step();
        }
        sim.drain_neural_learning(Duration::from_millis(100));
        if learned_mpc_objective_enabled() {
            let samples = sim.drain_mpc_objective_samples();
            if !samples.is_empty() {
                if let Some(head) = mpc_head.as_mut() {
                    head.train_rwr(&samples, 0.05);
                }
            }
        }
        if let Some(p) = sim.team_policies() {
            policies = Arc::new(p.clone());
        }
        if let Some(s) = sim.neural_network_snapshot_for(candidate_team) {
            snapshot = Some(s);
        }
        let s = sim.summary();
        let (candidate_goals, opponent_goals) = match candidate_team {
            Team::Home => (s.score_home, s.score_away),
            Team::Away => (s.score_away, s.score_home),
        };
        eprintln!(
            "[train-analytic] game {:>2}/{games} candidate={:?} opp_style={} score {}-{} ({:.0}s elapsed)",
            g + 1,
            candidate_team,
            opponent_id,
            candidate_goals,
            opponent_goals,
            started.elapsed().as_secs_f64()
        );
    }
    let snapshot = snapshot.expect("analytic-opponent training produced a snapshot");
    let json = serde_json::to_string(&snapshot).expect("serialize snapshot");
    std::fs::write(out_path, json).expect("write snapshot json");
    println!(
        "[train-analytic] wrote {out_path} after {games} games ({:.0}s)",
        started.elapsed().as_secs_f64()
    );
}

fn load_snapshot(path: &str) -> SoccerNeuralNetworkSnapshot {
    let json = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    serde_json::from_str(&json).unwrap_or_else(|e| panic!("parse {path}: {e}"))
}

/// Per-team pass breakdown for one match, so the eval can separate WHO/WHEN (the POMDP chooses to
/// pass forward) from HOW (MPC executes it): completion rate = completed/attempted (low ⇒ execution
/// failing), forward share = forward/completed (low ⇒ the policy recycles instead of progressing).
#[derive(Clone, Copy, Default)]
struct PassBreak {
    attempted: u32,
    completed: u32,
    forward: u32,
    backward: u32,
}

fn play_holdout_fixture(
    runner: &mut EngineMatchRunner,
    home_id: usize,
    away_id: usize,
    seed: u32,
    home: &TeamBrain,
    away: &TeamBrain,
) -> Option<(MatchReport, PassBreak, PassBreak)> {
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
    match runner.play(&ctx, home, away) {
        Ok(o) => {
            // Dense advancement signal: completed FORWARD passes per team this match (already
            // counted in MatchStats, surfaced via the outcome summary). Returned alongside the
            // score so the caller can measure progression, not just the rare goal outcome.
            let st = &o.summary.stats;
            let home_pass = PassBreak {
                attempted: st.passes_attempted_home,
                completed: st.passes_completed_home,
                forward: st.passes_completed_forward_home,
                backward: st.passes_completed_backward_home,
            };
            let away_pass = PassBreak {
                attempted: st.passes_attempted_away,
                completed: st.passes_completed_away,
                forward: st.passes_completed_forward_away,
                backward: st.passes_completed_backward_away,
            };
            let report = MatchReport {
                stage: ctx.stage,
                home_id,
                away_id,
                home_name: ctx.home_name,
                away_name: ctx.away_name,
                home_goals: o.home_goals,
                away_goals: o.away_goals,
                shootout_winner: None,
                home_training_steps: o.home_training_steps,
                away_training_steps: o.away_training_steps,
            };
            Some((report, home_pass, away_pass))
        }
        Err(e) => {
            eprintln!("[eval] fixture error: {e}");
            None
        }
    }
}

/// Play the candidate (A) vs the baseline (B) head-to-head over a held-out seed range
/// (both frozen), then print the held-out promotion verdict for A relative to B.
fn eval(candidate_path: &str, baseline_path: &str, games: usize, minutes: f64, holdout_base: u32) {
    enable_deterministic_formation_lp();
    let candidate = TeamBrain::from_snapshot(load_snapshot(candidate_path));
    let baseline = TeamBrain::from_snapshot(load_snapshot(baseline_path));
    let (candidate_id, baseline_id) = (0usize, 1usize);
    println!(
        "[eval] candidate={candidate_path} (id {candidate_id}) vs baseline={baseline_path} \
         (id {baseline_id}); {games} held-out games, minutes={minutes}, base=0x{holdout_base:08X}"
    );

    let mut runner_config = EngineMatchRunnerConfig::default();
    runner_config.base.duration_seconds = minutes * 60.0;
    runner_config.base.neural_blend = env_neural_blend(runner_config.base.neural_blend);
    print_neural_blend("eval", runner_config.base.neural_blend);
    let mut runner = EngineMatchRunner::new(runner_config);

    let started = Instant::now();
    let mut reports: Vec<MatchReport> = Vec::new();
    // Dense advancement metric: the per-game (candidate − baseline) completed-forward-pass
    // differential. Because both sides play the same seeded fixture head-to-head, this is a PAIRED
    // sample — far more statistically powerful than the sparse, draw-heavy goal margin.
    let mut forward_pass_diffs: Vec<f64> = Vec::new();
    let mut candidate_forward_total: u64 = 0;
    let mut baseline_forward_total: u64 = 0;
    // Full pass breakdown so we can diagnose WHO/WHEN (POMDP selection) vs HOW (MPC execution).
    let mut cand_att: u64 = 0;
    let mut cand_comp: u64 = 0;
    let mut cand_back: u64 = 0;
    let mut base_att: u64 = 0;
    let mut base_comp: u64 = 0;
    let mut base_back: u64 = 0;
    for g in 0..games {
        let seed = holdout_base.wrapping_add((g as u32).wrapping_mul(2_246_822_519));
        // Alternate home/away so the verdict isn't a home-field artifact.
        let report = if g % 2 == 0 {
            play_holdout_fixture(
                &mut runner,
                candidate_id,
                baseline_id,
                seed,
                &candidate,
                &baseline,
            )
        } else {
            play_holdout_fixture(
                &mut runner,
                baseline_id,
                candidate_id,
                seed,
                &baseline,
                &candidate,
            )
        };
        if let Some((r, home_pass, away_pass)) = report {
            // Map the fixture's home/away breakdown back to candidate/baseline regardless of
            // this game's orientation.
            let (cand_pass, base_pass) = if r.home_id == candidate_id {
                (home_pass, away_pass)
            } else {
                (away_pass, home_pass)
            };
            let (candidate_forward, baseline_forward) = (cand_pass.forward, base_pass.forward);
            candidate_forward_total += u64::from(candidate_forward);
            baseline_forward_total += u64::from(baseline_forward);
            cand_att += u64::from(cand_pass.attempted);
            cand_comp += u64::from(cand_pass.completed);
            cand_back += u64::from(cand_pass.backward);
            base_att += u64::from(base_pass.attempted);
            base_comp += u64::from(base_pass.completed);
            base_back += u64::from(base_pass.backward);
            forward_pass_diffs.push(f64::from(candidate_forward) - f64::from(baseline_forward));
            eprintln!(
                "[eval] game {:>2}/{games} {}v{} -> {}-{}  fwd-passes cand/base {}/{} ({:.0}s)",
                g + 1,
                r.home_id,
                r.away_id,
                r.home_goals,
                r.away_goals,
                candidate_forward,
                baseline_forward,
                started.elapsed().as_secs_f64()
            );
            reports.push(r);
        }
    }

    let verdict = evaluate_promotion(
        &reports,
        candidate_id,
        baseline_id,
        PromotionThresholds::default(),
    );
    println!("\n===== HELD-OUT A/B VERDICT (candidate=won-game-reward arm) =====");
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
        "payoff vs baseline {:?}  Wilson lower bound {:.3}",
        verdict
            .payoff_vs_baseline
            .map(|p| (p * 1000.0).round() / 1000.0),
        verdict.wilson_lower_bound,
    );
    println!(
        "DECISION: {}",
        if verdict.promote { "PROMOTE" } else { "REJECT" }
    );
    for reason in &verdict.reasons {
        println!("  - {reason}");
    }

    // ---- Forward-pass advancement: the dense progressive-play climb metric ----
    // Completed forward passes are frequent (dozens/game), directly measure ball progression, and
    // can't be farmed the way shots can. On a PAIRED head-to-head sample this gives a tight
    // confidence bound that detects a real climb the sparse goal margin would call a plateau.
    let n = forward_pass_diffs.len();
    if n > 0 {
        let n_f = n as f64;
        let mean_diff = forward_pass_diffs.iter().sum::<f64>() / n_f;
        let variance = if n > 1 {
            forward_pass_diffs
                .iter()
                .map(|d| (d - mean_diff).powi(2))
                .sum::<f64>()
                / (n_f - 1.0)
        } else {
            0.0
        };
        let std_error = (variance / n_f).sqrt();
        // 95% normal lower confidence bound on the mean paired differential.
        let lower_bound = mean_diff - 1.96 * std_error;
        let candidate_per_game = candidate_forward_total as f64 / n_f;
        let baseline_per_game = baseline_forward_total as f64 / n_f;
        println!("\n===== FORWARD-PASS ADVANCEMENT (dense progressive-play climb metric) =====");
        println!(
            "completed forward passes/game: candidate {candidate_per_game:.1}  baseline {baseline_per_game:.1}  Δ {:+.2}/game",
            candidate_per_game - baseline_per_game
        );
        println!(
            "paired per-game Δ: mean {mean_diff:+.2}, 95% lower bound {lower_bound:+.2} over {n} games"
        );
        let advancement = if lower_bound > 0.0 {
            "CLIMB — candidate significantly out-progresses the baseline (lower bound > 0)"
        } else if mean_diff > 0.0 {
            "directional climb — more forward passes on average, not yet significant (need more games)"
        } else {
            "no advancement on forward passes"
        };
        println!("ADVANCEMENT: {advancement}");

        // ---- Pass diagnosis: separate WHO/WHEN (POMDP selection) from HOW (MPC execution) ----
        // completion rate = completed/attempted (low ⇒ passes fail in flight ⇒ execution/aim = MPC).
        // forward share  = forward/completed (low ⇒ policy recycles laterally/back ⇒ selection = POMDP).
        let pct = |num: u64, den: u64| {
            if den > 0 {
                100.0 * num as f64 / den as f64
            } else {
                0.0
            }
        };
        let fwd_share = |fwd: u64, comp: u64| pct(fwd, comp);
        println!("\n===== PASS DIAGNOSIS (WHO/WHEN vs HOW) =====");
        println!(
            "candidate: {:.1} att/g, {:.0}% completed, forward {:.0}% / lateral {:.0}% / back {:.0}% of completions",
            cand_att as f64 / n_f,
            pct(cand_comp, cand_att),
            fwd_share(candidate_forward_total, cand_comp),
            pct(cand_comp.saturating_sub(candidate_forward_total + cand_back), cand_comp),
            pct(cand_back, cand_comp),
        );
        println!(
            "baseline:  {:.1} att/g, {:.0}% completed, forward {:.0}% / lateral {:.0}% / back {:.0}% of completions",
            base_att as f64 / n_f,
            pct(base_comp, base_att),
            fwd_share(baseline_forward_total, base_comp),
            pct(base_comp.saturating_sub(baseline_forward_total + base_back), base_comp),
            pct(base_back, base_comp),
        );
        let comp_rate = pct(cand_comp, cand_att);
        let fshare = fwd_share(candidate_forward_total, cand_comp);
        println!(
            "READ: {}",
            if comp_rate < 65.0 {
                "LOW completion ⇒ passes fail in flight — bottleneck is EXECUTION (MPC aim/speed/curve)"
            } else if fshare < 20.0 {
                "HIGH completion but LOW forward share ⇒ the POMDP is CHOOSING safe (lateral/back) — bottleneck is SELECTION (WHO/WHEN)"
            } else {
                "completion healthy and forward share non-trivial — passing is being expressed; look elsewhere"
            }
        );
    } else {
        println!("\n[advancement] no completed fixtures to measure forward-pass progression");
    }
}

fn eval_analytic(
    candidate_path: &str,
    games_per_opp: usize,
    minutes: f64,
    pool_size: usize,
    holdout_base: u32,
) {
    enable_deterministic_formation_lp();
    let candidate = TeamBrain::from_snapshot(load_snapshot(candidate_path));
    let candidate_id = 0usize;
    let baseline_id = 1usize;
    let pool_size = pool_size.max(2);
    println!(
        "[eval-analytic] candidate={candidate_path} (id {candidate_id}) vs analytic field \
         ids 1..{}; {games_per_opp} held-out games/opponent, minutes={minutes}, \
         base=0x{holdout_base:08X}",
        pool_size - 1
    );

    let mut runner_config = EngineMatchRunnerConfig::default();
    runner_config.base.duration_seconds = minutes * 60.0;
    runner_config.base.neural_blend =
        analytic_experiment_neural_blend(runner_config.base.neural_blend);
    print_neural_blend("eval-analytic", runner_config.base.neural_blend);
    let mut runner = EngineMatchRunner::new(runner_config);

    let started = Instant::now();
    let mut reports: Vec<MatchReport> = Vec::new();
    let mut forward_pass_diffs: Vec<f64> = Vec::new();
    let mut candidate_forward_total: u64 = 0;
    let mut analytic_forward_total: u64 = 0;
    let mut cand_att: u64 = 0;
    let mut cand_comp: u64 = 0;
    let mut cand_back: u64 = 0;
    let mut ana_att: u64 = 0;
    let mut ana_comp: u64 = 0;
    let mut ana_back: u64 = 0;
    let mut fixture_index = 0u32;

    for opponent_id in 1..pool_size {
        let opponent = TeamBrain::fresh_with_seed(
            0xF0_0000u32.wrapping_add(opponent_id as u32 * 2_654_435_761),
            opponent_id,
        );
        for g in 0..games_per_opp {
            let seed = holdout_base
                .wrapping_add(fixture_index.wrapping_mul(2_246_822_519))
                .wrapping_add(g as u32);
            fixture_index = fixture_index.wrapping_add(1);
            let report = if g % 2 == 0 {
                play_holdout_fixture(
                    &mut runner,
                    candidate_id,
                    opponent_id,
                    seed,
                    &candidate,
                    &opponent,
                )
            } else {
                play_holdout_fixture(
                    &mut runner,
                    opponent_id,
                    candidate_id,
                    seed,
                    &opponent,
                    &candidate,
                )
            };
            if let Some((r, home_pass, away_pass)) = report {
                let (cand_pass, analytic_pass) = if r.home_id == candidate_id {
                    (home_pass, away_pass)
                } else {
                    (away_pass, home_pass)
                };
                candidate_forward_total += u64::from(cand_pass.forward);
                analytic_forward_total += u64::from(analytic_pass.forward);
                cand_att += u64::from(cand_pass.attempted);
                cand_comp += u64::from(cand_pass.completed);
                cand_back += u64::from(cand_pass.backward);
                ana_att += u64::from(analytic_pass.attempted);
                ana_comp += u64::from(analytic_pass.completed);
                ana_back += u64::from(analytic_pass.backward);
                forward_pass_diffs
                    .push(f64::from(cand_pass.forward) - f64::from(analytic_pass.forward));
                eprintln!(
                    "[eval-analytic] game {:>2} opp={} {}v{} -> {}-{}  fwd-passes cand/analytic {}/{} ({:.0}s)",
                    reports.len() + 1,
                    opponent_id,
                    r.home_id,
                    r.away_id,
                    r.home_goals,
                    r.away_goals,
                    cand_pass.forward,
                    analytic_pass.forward,
                    started.elapsed().as_secs_f64()
                );
                reports.push(r);
            }
        }
    }

    let verdict = evaluate_promotion(
        &reports,
        candidate_id,
        baseline_id,
        PromotionThresholds::default(),
    );
    println!("\n===== HELD-OUT ANALYTIC-FIELD VERDICT =====");
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
        "held-out Elo: candidate {:.1}  analytic-baseline {:.1}  Δ {:+.1}",
        verdict.candidate_elo, verdict.baseline_elo, verdict.elo_delta
    );
    println!(
        "payoff vs analytic field {:?}  payoff vs baseline {:?}  worst-case {:?}  Wilson lower bound {:.3}",
        verdict
            .mean_payoff_vs_field
            .map(|p| (p * 1000.0).round() / 1000.0),
        verdict
            .payoff_vs_baseline
            .map(|p| (p * 1000.0).round() / 1000.0),
        verdict
            .worst_case
            .map(|(id, p)| (id, (p * 1000.0).round() / 1000.0)),
        verdict.wilson_lower_bound,
    );
    println!(
        "DECISION: {}",
        if verdict.promote { "PROMOTE" } else { "REJECT" }
    );
    for reason in &verdict.reasons {
        println!("  - {reason}");
    }

    let n = forward_pass_diffs.len();
    if n > 0 {
        let n_f = n as f64;
        let mean_diff = forward_pass_diffs.iter().sum::<f64>() / n_f;
        let variance = if n > 1 {
            forward_pass_diffs
                .iter()
                .map(|d| (d - mean_diff).powi(2))
                .sum::<f64>()
                / (n_f - 1.0)
        } else {
            0.0
        };
        let std_error = (variance / n_f).sqrt();
        let lower_bound = mean_diff - 1.96 * std_error;
        let candidate_per_game = candidate_forward_total as f64 / n_f;
        let analytic_per_game = analytic_forward_total as f64 / n_f;
        println!("\n===== ANALYTIC-FIELD FORWARD-PASS ADVANCEMENT =====");
        println!(
            "completed forward passes/game: candidate {candidate_per_game:.1}  analytic {analytic_per_game:.1}  Δ {:+.2}/game",
            candidate_per_game - analytic_per_game
        );
        println!(
            "paired per-game Δ: mean {mean_diff:+.2}, 95% lower bound {lower_bound:+.2} over {n} games"
        );

        let pct = |num: u64, den: u64| {
            if den > 0 {
                100.0 * num as f64 / den as f64
            } else {
                0.0
            }
        };
        let fwd_share = |fwd: u64, comp: u64| pct(fwd, comp);
        println!("\n===== ANALYTIC-FIELD PASS DIAGNOSIS =====");
        println!(
            "candidate: {:.1} att/g, {:.0}% completed, forward {:.0}% / lateral {:.0}% / back {:.0}% of completions",
            cand_att as f64 / n_f,
            pct(cand_comp, cand_att),
            fwd_share(candidate_forward_total, cand_comp),
            pct(cand_comp.saturating_sub(candidate_forward_total + cand_back), cand_comp),
            pct(cand_back, cand_comp),
        );
        println!(
            "analytic:  {:.1} att/g, {:.0}% completed, forward {:.0}% / lateral {:.0}% / back {:.0}% of completions",
            ana_att as f64 / n_f,
            pct(ana_comp, ana_att),
            fwd_share(analytic_forward_total, ana_comp),
            pct(ana_comp.saturating_sub(analytic_forward_total + ana_back), ana_comp),
            pct(ana_back, ana_comp),
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("effective-win-reward") => {
            println!("{}", match_outcome_win_reward_points());
        }
        Some("train") => {
            let out = args
                .get(2)
                .expect("usage: train <out.json> [games] [minutes] [seed_hex]");
            let games = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(40);
            let minutes = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(3.0);
            let seed = parse_hex(args.get(5), 0x71A1_0000);
            train(out, games, minutes, seed);
        }
        Some("train-analytic") => {
            let out = args
                .get(2)
                .expect("usage: train-analytic <out.json> [games] [minutes] [seed_hex]");
            let games = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(40);
            let minutes = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(3.0);
            let seed = parse_hex(args.get(5), 0x71A1_0000);
            train_analytic(out, games, minutes, seed);
        }
        Some("eval") => {
            let candidate = args
                .get(2)
                .expect("usage: eval <candidate.json> <baseline.json> ...");
            let baseline = args
                .get(3)
                .expect("usage: eval <candidate.json> <baseline.json> ...");
            let games = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(30);
            let minutes = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(3.0);
            let holdout = parse_hex(args.get(6), 0xE7A1_0000);
            eval(candidate, baseline, games, minutes, holdout);
        }
        Some("eval-analytic") => {
            let candidate = args
                .get(2)
                .expect("usage: eval-analytic <candidate.json> [games_per_opp] [minutes] [pool_size] [holdout_hex]");
            let games = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(6);
            let minutes = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(3.0);
            let pool_size = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(4);
            let holdout = parse_hex(args.get(6), 0xE7A1_0000);
            eval_analytic(candidate, games, minutes, pool_size, holdout);
        }
        _ => {
            eprintln!(
                "usage:\n  soccer_outcome_ab_run effective-win-reward\n  \
                 soccer_outcome_ab_run train <out.json> [games=40] [minutes=3] [seed_hex]\n  \
                 soccer_outcome_ab_run train-analytic <out.json> [games=40] [minutes=3] [seed_hex]\n  \
                 soccer_outcome_ab_run eval <candidate.json> <baseline.json> [games=30] [minutes=3] [holdout_hex]\n  \
                 soccer_outcome_ab_run eval-analytic <candidate.json> [games_per_opp=6] [minutes=3] [pool_size=4] [holdout_hex]"
            );
            std::process::exit(2);
        }
    }
}
