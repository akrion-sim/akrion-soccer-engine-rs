//! Deterministic learning A/B harness.
//!
//! Runs a short self-play CURRICULUM in-process: a neural value+actor (MAPPO) and the tabular team
//! Q-policies are carried + trained across N games (no Postgres), exactly like the real learner's
//! per-process carry-forward, then the resulting LEARNED play is measured. Unlike `measure_offense`
//! (which runs the heuristic policy and so is blind to reward-shaping changes), this drives play
//! with the trained actor, so a change to the LEARNING SIGNAL — e.g. the overload-weighted
//! progression reward or the balanced MAPPO team component — actually shows up in the KPIs.
//!
//! The feature gates (`DD_SOCCER_ENABLE_OVERLOAD_WEIGHTED_PROGRESSION`,
//! `DD_SOCCER_ENABLE_MARL_BALANCED_TEAM_COMPONENT`, …) are read once per process, so an A/B is TWO
//! runs of this binary: once with the env unset (baseline) and once with it set (treatment), same
//! seeds. The formation LP is forced through the deterministic simplex so each arm is reproducible.
//!
//! Run:
//!   cargo run --release --bin measure_learning_ab -- [games] [minutes] [seed_base]
//!   # baseline
//!   cargo run --release --bin measure_learning_ab -- 24 3
//!   # treatment
//!   DD_SOCCER_ENABLE_OVERLOAD_WEIGHTED_PROGRESSION=1 \
//!   DD_SOCCER_ENABLE_MARL_BALANCED_TEAM_COMPONENT=1 \
//!   cargo run --release --bin measure_learning_ab -- 24 3

use std::sync::Arc;
use std::time::{Duration, Instant};

use soccer_engine::des::general::soccer::{
    enable_deterministic_formation_lp, learned_mpc_objective_enabled, MatchConfig,
    SoccerMarlAlgorithm, SoccerMatch, SoccerMpcObjectiveHead, SoccerNeuralLearningBackend,
    SoccerNeuralLearningConfig, SoccerNeuralNetworkSnapshot, SoccerPassCompletionHead,
    SoccerQPolicyOptions, SoccerTeamQPolicies, DEFAULT_SOCCER_MAPPO_TEAM_REWARD_SHARE,
    PASS_COMPLETION_HEAD_MIN_TRAINING_STEPS,
};

/// One game's measured offense KPIs (both teams summed — self-play is symmetric).
#[derive(Default, Clone, Copy)]
struct GameKpis {
    goals: f64,
    shots: f64,
    shots_on_target: f64,
    shots_after_pass: f64,
    passes_attempted: f64,
    passes_completed: f64,
    passes_forward: f64,
    pass_turnovers: f64,
    pass_chains: f64,
    pass_chain_gain_yards: f64,
    pass_chains_net_loss: f64,
    crosses_completed: f64,
    dribble_beats: f64,
    dribble_decisions: f64,
}

impl GameKpis {
    fn accumulate(&mut self, other: &GameKpis) {
        self.goals += other.goals;
        self.shots += other.shots;
        self.shots_on_target += other.shots_on_target;
        self.shots_after_pass += other.shots_after_pass;
        self.passes_attempted += other.passes_attempted;
        self.passes_completed += other.passes_completed;
        self.passes_forward += other.passes_forward;
        self.pass_turnovers += other.pass_turnovers;
        self.pass_chains += other.pass_chains;
        self.pass_chain_gain_yards += other.pass_chain_gain_yards;
        self.pass_chains_net_loss += other.pass_chains_net_loss;
        self.crosses_completed += other.crosses_completed;
        self.dribble_beats += other.dribble_beats;
        self.dribble_decisions += other.dribble_decisions;
    }

    fn scaled(&self, inv: f64) -> GameKpis {
        GameKpis {
            goals: self.goals * inv,
            shots: self.shots * inv,
            shots_on_target: self.shots_on_target * inv,
            shots_after_pass: self.shots_after_pass * inv,
            passes_attempted: self.passes_attempted * inv,
            passes_completed: self.passes_completed * inv,
            passes_forward: self.passes_forward * inv,
            pass_turnovers: self.pass_turnovers * inv,
            pass_chains: self.pass_chains * inv,
            pass_chain_gain_yards: self.pass_chain_gain_yards * inv,
            pass_chains_net_loss: self.pass_chains_net_loss * inv,
            crosses_completed: self.crosses_completed * inv,
            dribble_beats: self.dribble_beats * inv,
            dribble_decisions: self.dribble_decisions * inv,
        }
    }

    fn print_row(&self, label: &str) {
        let pass_completion = ratio(self.passes_completed, self.passes_attempted);
        let shot_accuracy = ratio(self.shots_on_target, self.shots);
        let net_forward = self.passes_forward - self.pass_turnovers;
        let dribble_beat_per_100 = ratio(self.dribble_beats * 100.0, self.dribble_decisions);
        println!(
            "{label:<14} pass_completion={:.3} passes={:.0}/{:.0} fwd_passes={:.0} \
             pass_turnovers={:.1} net_forward={:.1} goals={:.2} shots={:.1} \
             on_target={:.1} shot_acc={:.3} shot_after_pass={:.1} dribble_beats={:.1} \
             dribble_decisions={:.0} dribble_beats_per_100_decisions={:.2} chains={:.1} \
             chain_gain_yds={:.1} chain_net_loss={:.2} crosses={:.1}",
            pass_completion,
            self.passes_completed,
            self.passes_attempted,
            self.passes_forward,
            self.pass_turnovers,
            net_forward,
            self.goals,
            self.shots,
            self.shots_on_target,
            shot_accuracy,
            self.shots_after_pass,
            self.dribble_beats,
            self.dribble_decisions,
            dribble_beat_per_100,
            self.pass_chains,
            self.pass_chain_gain_yards,
            self.pass_chains_net_loss,
            self.crosses_completed,
        );
    }
}

fn env_on(name: &str) -> bool {
    std::env::var(name)
        .map(|v| {
            let v = v.trim();
            v == "1" || v.eq_ignore_ascii_case("true")
        })
        .unwrap_or(false)
}

fn env_f64(name: &str) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.trim().parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(0.0)
}

fn ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator.abs() > 1e-9 {
        numerator / denominator
    } else {
        0.0
    }
}

fn is_dribble_decision(action: &str) -> bool {
    let compact = action.trim().to_ascii_lowercase();
    compact == "dribble"
        || compact.starts_with("dribble-")
        || compact == "carry-forward"
        || compact == "carry-out-left"
        || compact == "carry-out-right"
        || compact == "open-pass-lane"
}

fn main() {
    // Reproducible A/B: deterministic formation LP so the same seed yields the same match.
    enable_deterministic_formation_lp();

    let args: Vec<String> = std::env::args().collect();
    let games: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(24);
    let minutes: f64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3.0);
    let seed_base: u32 = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0x5EED_0000);

    // Inline backend keeps neural training synchronous (deterministic); MAPPO actor on so the
    // trained policy actually drives play.
    let neural = SoccerNeuralLearningConfig {
        enabled: true,
        backend: SoccerNeuralLearningBackend::Inline,
        marl_algorithm: SoccerMarlAlgorithm::Mappo,
        mappo_team_reward_share: DEFAULT_SOCCER_MAPPO_TEAM_REWARD_SHARE,
        ..SoccerNeuralLearningConfig::default()
    };

    println!("===== LEARNING A/B HARNESS =====");
    println!(
        "games={games} minutes={minutes} seed_base=0x{seed_base:08X} \
         backend=Inline marl=Mappo team_reward_share={:.3}",
        neural.mappo_team_reward_share
    );
    println!(
        "gates: overload_weighted_progression={} marl_balanced_team_component={} \
         advantage_normalization={}",
        env_on("DD_SOCCER_ENABLE_OVERLOAD_WEIGHTED_PROGRESSION"),
        env_on("DD_SOCCER_ENABLE_MARL_BALANCED_TEAM_COMPONENT"),
        env_on("DD_SOCCER_ENABLE_ADVANTAGE_NORMALIZATION"),
    );

    println!(
        "gates: learned_mpc_objective={}",
        learned_mpc_objective_enabled()
    );
    println!(
        "gates: dp_critic_target={} dp_bootstrap={} learned_epv={} learned_pass_completion={} \
         deferred_pass_credit={} neural_mcts={} league_neural_mcts={} xt_terminal_cost={} \
         mpc_pass={} pass_completion_score_weight={:.2} pass_turnover_penalty_scale={:.2} \
         pass_target_completion_primary_scale={:.2}",
        env_on("DD_SOCCER_ENABLE_DP_CRITIC_TARGET"),
        env_on("DD_SOCCER_ENABLE_DP_BOOTSTRAP"),
        env_on("DD_SOCCER_ENABLE_LEARNED_EPV"),
        env_on("DD_SOCCER_ENABLE_LEARNED_PASS_COMPLETION"),
        env_on("DD_SOCCER_ENABLE_DEFERRED_PASS_CREDIT"),
        env_on("SOCCER_NEURAL_MCTS_ENABLED"),
        env_on("SOCCER_LEAGUE_NEURAL_MCTS_ENABLED"),
        env_on("DD_SOCCER_ENABLE_XT_TERMINAL_COST"),
        env_on("DD_SOCCER_ENABLE_MPC_PASS"),
        env_f64("DD_SOCCER_PASS_COMPLETION_SCORE_WEIGHT"),
        env_f64("DD_SOCCER_PASS_TURNOVER_PENALTY_SCALE"),
        env_f64("DD_SOCCER_PASS_TARGET_COMPLETION_PRIMARY_SCALE"),
    );

    // Carried across games within this process (the real learner's per-process pattern).
    let mut policies = Arc::new(SoccerTeamQPolicies::new(SoccerQPolicyOptions::default()));
    let mut snapshot: Option<SoccerNeuralNetworkSnapshot> = None;
    let mut pass_completion_head: Option<SoccerPassCompletionHead> = None;
    // Learned MPC execution-objective head, carried + RWR-trained across games (mirrors the
    // pass-completion head's per-process carry). Seeded up-front when the gate is on so game 1
    // already captures cold-exploration samples (the head must exist for the residual to apply);
    // stays `None` only when the production-default-on gate is disabled with a falsey env value.
    let mut mpc_objective_head: Option<SoccerMpcObjectiveHead> = if learned_mpc_objective_enabled()
    {
        Some(SoccerMpcObjectiveHead::new(seed_base))
    } else {
        None
    };

    let mut per_game: Vec<GameKpis> = Vec::with_capacity(games);
    let started = Instant::now();

    for g in 0..games {
        let mut config = MatchConfig {
            duration_seconds: minutes * 60.0,
            learning_enabled: true,
            neural_learning: neural.clone(),
            seed: seed_base.wrapping_add(g as u32),
            ..MatchConfig::default()
        };
        // Drive play with the trained actor (otherwise the reward shaping is invisible).
        config.neural_blend.actor_critic = true;
        let total_ticks = config.total_ticks();

        let mut sim = SoccerMatch::default_11v11(config).with_team_policies((*policies).clone());
        sim.set_uniform_elite_players();
        if let Some(s) = snapshot.as_ref() {
            if let Err(e) = sim.set_neural_network_snapshot(s.clone()) {
                eprintln!("game {g}: failed to install snapshot: {e}");
            }
        }
        if let Some(head) = pass_completion_head.as_ref() {
            sim.set_pass_completion_head(head.clone());
        }
        // Install the carried executor head so this game's passes get the learned aim/lead residual
        // (gated; a no-op only when DD_SOCCER_ENABLE_LEARNED_MPC_OBJECTIVE is falsey).
        if let Some(head) = mpc_objective_head.as_ref() {
            sim.set_mpc_objective_head(head.clone());
        }

        let mut game_dribble_decision_ticks = 0_u64;
        for _ in 0..total_ticks {
            sim.run_time_step();
            for player in &sim.players {
                if let Some(decision) = player.last_decision.as_ref() {
                    if is_dribble_decision(&decision.action) {
                        game_dribble_decision_ticks = game_dribble_decision_ticks.saturating_add(1);
                    }
                }
            }
        }
        sim.drain_neural_learning(Duration::from_millis(100));

        // Mirror the real learner: train the carried pass-completion head on this game's resolved
        // pass outcomes, then install it into subsequent games once it clears the live blend floor.
        let pass_outcome_samples = sim.drain_pass_outcome_samples();
        if !pass_outcome_samples.is_empty() {
            let head = pass_completion_head.get_or_insert_with(|| {
                SoccerPassCompletionHead::new(seed_base.wrapping_add(g as u32))
            });
            let mut final_loss = 0.0;
            for _ in 0..4 {
                final_loss = head.train(&pass_outcome_samples, 0.05);
            }
            eprintln!(
                "pass_completion_training game={} samples={} training_steps={} consumed={} final_loss={:.5}",
                g + 1,
                pass_outcome_samples.len(),
                head.training_steps(),
                head.training_steps() >= PASS_COMPLETION_HEAD_MIN_TRAINING_STEPS,
                final_loss
            );
        }

        // Train the executor head on this game's captured (features, applied_residual, advantage)
        // samples (RWR, 4 epochs), carried into the next game — the executor-head analog of the
        // MAPPO/neural carry above.
        let mpc_samples = sim.drain_mpc_objective_samples();
        if !mpc_samples.is_empty() {
            let head = mpc_objective_head.get_or_insert_with(|| {
                SoccerMpcObjectiveHead::new(seed_base.wrapping_add(g as u32))
            });
            for _ in 0..4 {
                head.train_rwr(&mpc_samples, 0.05);
            }
        }

        let summary = sim.summary();
        let st = &summary.stats;
        let kpis = GameKpis {
            goals: f64::from(summary.score_home + summary.score_away),
            shots: f64::from(st.shots_home + st.shots_away),
            shots_on_target: f64::from(st.shots_on_target_home + st.shots_on_target_away),
            shots_after_pass: f64::from(st.shots_after_pass_home + st.shots_after_pass_away),
            passes_attempted: f64::from(st.passes_attempted_home + st.passes_attempted_away),
            passes_completed: f64::from(st.passes_completed_home + st.passes_completed_away),
            passes_forward: f64::from(
                st.passes_completed_forward_home + st.passes_completed_forward_away,
            ),
            pass_turnovers: f64::from(
                st.pass_interceptions_own_half + st.pass_interceptions_opp_half,
            ),
            pass_chains: f64::from(st.pass_chains_home + st.pass_chains_away),
            pass_chain_gain_yards: st.pass_chain_gain_yards_home + st.pass_chain_gain_yards_away,
            pass_chains_net_loss: f64::from(
                st.pass_chains_net_loss_home + st.pass_chains_net_loss_away,
            ),
            crosses_completed: f64::from(st.crosses_completed_home + st.crosses_completed_away),
            dribble_beats: f64::from(st.dribble_beats_home + st.dribble_beats_away),
            dribble_decisions: game_dribble_decision_ticks as f64,
        };
        per_game.push(kpis);

        // Carry the trained policy + neural net into the next game.
        if let Some(p) = sim.team_policies() {
            policies = Arc::new(p.clone());
        }
        if let Some(s) = sim.neural_network_snapshot() {
            snapshot = Some(s);
        }

        eprintln!(
            "game {:>2}/{games} seed=0x{:08X} score {}-{} chains={} chain_gain_yds={:.1}",
            g + 1,
            seed_base.wrapping_add(g as u32),
            summary.score_home,
            summary.score_away,
            st.pass_chains_home + st.pass_chains_away,
            st.pass_chain_gain_yards_home + st.pass_chain_gain_yards_away,
        );
    }

    // Aggregate: overall + first-third vs last-third so a learning TREND is visible (does the
    // trained policy progress the ball more as training accrues?).
    let mut overall = GameKpis::default();
    for k in &per_game {
        overall.accumulate(k);
    }
    let n = per_game.len().max(1) as f64;

    let third = (per_game.len() / 3).max(1);
    let mut early = GameKpis::default();
    for k in &per_game[..third] {
        early.accumulate(k);
    }
    let mut late = GameKpis::default();
    for k in &per_game[per_game.len() - third..] {
        late.accumulate(k);
    }

    println!(
        "\n----- per-game averages ({games} games, {:.1}s elapsed) -----",
        started.elapsed().as_secs_f64()
    );
    overall.scaled(1.0 / n).print_row("OVERALL");
    early.scaled(1.0 / third as f64).print_row("EARLY(1st⅓)");
    late.scaled(1.0 / third as f64).print_row("LATE(last⅓)");

    let early_avg = early.scaled(1.0 / third as f64);
    let late_avg = late.scaled(1.0 / third as f64);
    let pct = |a: f64, b: f64| {
        if a.abs() > 1e-9 {
            (b - a) / a * 100.0
        } else {
            0.0
        }
    };
    println!(
        "\nTREND late-vs-early: fwd_passes {:+.1}%  chains {:+.1}%  chain_gain_yds {:+.1}%  \
         shot_after_pass {:+.1}%  goals {:+.1}%",
        pct(early_avg.passes_forward, late_avg.passes_forward),
        pct(early_avg.pass_chains, late_avg.pass_chains),
        pct(
            early_avg.pass_chain_gain_yards,
            late_avg.pass_chain_gain_yards
        ),
        pct(early_avg.shots_after_pass, late_avg.shots_after_pass),
        pct(early_avg.goals, late_avg.goals),
    );
}
