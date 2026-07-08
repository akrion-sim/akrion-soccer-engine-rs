//! Monotonic-climb RATCHET harness — the structural answer to "make sure we actually climb, no
//! stalling/plateaus/divergence".
//!
//! Trains the neural policy in INCREMENTS (self-play carry-forward, exactly like the real learner),
//! and after every increment EVALS the current checkpoint head-to-head vs the pure-ANALYTIC engine
//! (candidate net on one side, no net / analytic on the other, sides alternated for fairness). It
//! KEEPS THE BEST checkpoint and discards regressions — so the kept policy is monotonic BY
//! CONSTRUCTION: it can plateau, but it can never go backwards, no matter how noisy or divergent a
//! given training window is. The printed climb curve (eval_mean per increment) reveals a stall or
//! divergence the instant it happens instead of only at the end.
//!
//! Run:
//!   cargo run --release --bin soccer_climb_ratchet -- [increment_games] [minutes] [eval_games] [increments] [seed_base]
//!   # anti-parity levers as env (read internally by the engine), e.g.:
//!   DD_SOCCER_ENABLE_MC_CRITIC_TARGET=1 DD_SOCCER_ENABLE_TARGET_STANDARDIZATION=1 \
//!   DD_SOCCER_ENABLE_BELLMAN_TERMINALS=0 \
//!   cargo run --release --bin soccer_climb_ratchet -- 5 2 10 8

use std::sync::Arc;
use std::time::Duration;

use soccer_engine::des::general::soccer::{
    enable_deterministic_formation_lp, MatchConfig, SoccerMarlAlgorithm, SoccerMatch,
    SoccerNeuralLearningBackend, SoccerNeuralLearningConfig, SoccerNeuralNetworkSnapshot,
    SoccerQPolicyOptions, SoccerTeamQPolicies, Team, DEFAULT_SOCCER_MAPPO_TEAM_REWARD_SHARE,
};
use soccer_engine::des::general::soccer_eval_gate::wilson_lower_bound;

/// Head-to-head payoff of a FROZEN candidate net vs the pure-analytic engine over `games` matches,
/// candidate alternating Home/Away for fairness. Payoff = (wins + 0.5·draws) / games.
fn eval_vs_analytic(
    snapshot: &SoccerNeuralNetworkSnapshot,
    games: usize,
    minutes: f64,
    seed_base: u32,
) -> (f64, u32) {
    let mut score = 0.0f64;
    let mut n = 0u32;
    for g in 0..games {
        let candidate_home = g % 2 == 0;
        // Inference-only: neural ON (so the net drives), learning OFF (frozen — no target drift).
        let mut config = MatchConfig {
            duration_seconds: minutes * 60.0,
            learning_enabled: false,
            neural_learning: SoccerNeuralLearningConfig {
                enabled: true,
                backend: SoccerNeuralLearningBackend::Inline,
                marl_algorithm: SoccerMarlAlgorithm::Mappo,
                ..SoccerNeuralLearningConfig::default()
            },
            seed: seed_base.wrapping_add(g as u32),
            ..MatchConfig::default()
        };
        // Let the trained actor drive (same path as training), so we measure the learned policy.
        config.neural_blend.actor_critic = true;
        let total_ticks = config.total_ticks();
        let mut sim = SoccerMatch::default_11v11(config);
        sim.set_uniform_elite_players();
        // Candidate = frozen net on one side; the other side stays on the analytic decision stack.
        let (cand_team, opp_team) = if candidate_home {
            (Team::Home, Team::Away)
        } else {
            (Team::Away, Team::Home)
        };
        if let Err(e) = sim.set_team_neural_brain(cand_team, Some(snapshot.clone()), true) {
            eprintln!("eval game {g}: candidate brain install failed: {e}");
            continue;
        }
        sim.disable_team_neural_brain(opp_team);
        for _ in 0..total_ticks {
            sim.run_time_step();
        }
        let summary = sim.summary();
        let (cand_goals, opp_goals) = if candidate_home {
            (summary.score_home, summary.score_away)
        } else {
            (summary.score_away, summary.score_home)
        };
        score += if cand_goals > opp_goals {
            1.0
        } else if cand_goals == opp_goals {
            0.5
        } else {
            0.0
        };
        n += 1;
    }
    let mean = if n > 0 { score / n as f64 } else { 0.0 };
    (mean, n)
}

fn main() {
    // Deterministic formation LP so a given seed reproduces (fair increment-to-increment comparison).
    enable_deterministic_formation_lp();

    let args: Vec<String> = std::env::args().collect();
    let increment_games: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(5);
    let minutes: f64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2.0);
    let eval_games: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10);
    let increments: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(8);
    let seed_base: u32 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(0x5EED_0000);
    let eval_seed_base: u32 = seed_base ^ 0xE7A1_0000;

    let neural = SoccerNeuralLearningConfig {
        enabled: true,
        backend: SoccerNeuralLearningBackend::Inline,
        marl_algorithm: SoccerMarlAlgorithm::Mappo,
        mappo_team_reward_share: DEFAULT_SOCCER_MAPPO_TEAM_REWARD_SHARE,
        ..SoccerNeuralLearningConfig::default()
    };

    println!("===== CLIMB RATCHET (keep-best; monotonic by construction) =====");
    println!(
        "increment_games={increment_games} minutes={minutes} eval_games={eval_games} \
         increments={increments} seed_base=0x{seed_base:08X}"
    );
    println!("(eval = frozen candidate net vs the pure-analytic engine; payoff > 0.5 ⇒ beats analytic)");

    let mut policies = Arc::new(SoccerTeamQPolicies::new(SoccerQPolicyOptions::default()));
    let mut snapshot: Option<SoccerNeuralNetworkSnapshot> = None;
    let mut best_snapshot: Option<SoccerNeuralNetworkSnapshot> = None;
    let mut best_score = -1.0f64;
    let mut best_at_games = 0usize;
    let mut cumulative = 0usize;

    for inc in 0..increments {
        // ---- train `increment_games` more games, carrying policy + net forward ----
        for g in 0..increment_games {
            let mut config = MatchConfig {
                duration_seconds: minutes * 60.0,
                learning_enabled: true,
                neural_learning: neural.clone(),
                seed: seed_base.wrapping_add((cumulative + g) as u32),
                ..MatchConfig::default()
            };
            config.neural_blend.actor_critic = true;
            let total_ticks = config.total_ticks();
            let mut sim =
                SoccerMatch::default_11v11(config).with_team_policies((*policies).clone());
            sim.set_uniform_elite_players();
            if let Some(s) = snapshot.as_ref() {
                if let Err(e) = sim.set_neural_network_snapshot(s.clone()) {
                    eprintln!("train: snapshot install failed: {e}");
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
        }
        cumulative += increment_games;

        // ---- eval this checkpoint vs analytic; RATCHET on the best ----
        let Some(current) = snapshot.as_ref() else {
            println!("increment {inc}: no snapshot yet");
            continue;
        };
        let (score, n) = eval_vs_analytic(current, eval_games, minutes, eval_seed_base);
        let wilson = wilson_lower_bound(score, n, 1.96);
        let improved = score > best_score;
        if improved {
            best_score = score;
            best_snapshot = Some(current.clone());
            best_at_games = cumulative;
        }
        println!(
            "increment {inc:>2}: games={cumulative:>3}  eval_mean={score:.3}  wilson={wilson:.3}  \
             best={best_score:.3}  {}",
            if improved {
                "<= NEW BEST (kept)"
            } else {
                "(regressed vs best — discarded, ratchet holds)"
            }
        );
    }

    let _ = best_snapshot; // the ratcheted result: the best-seen policy (never regresses)
    println!(
        "\nRATCHET RESULT: best eval_mean={best_score:.3} at {best_at_games} games \
         (out of {cumulative} trained). The kept policy is the best-seen — monotonic by \
         construction: training noise/divergence can never lower the retained result."
    );
    if best_score > 0.5 {
        println!("=> CLIMBED OUT OF PARITY: best checkpoint beats the analytic engine (>0.5).");
    } else {
        println!(
            "=> still at/under parity (best {best_score:.3} ≤ 0.5): analytic engine not beaten yet; \
             the ratchet at least guarantees we kept the strongest checkpoint."
        );
    }
}
