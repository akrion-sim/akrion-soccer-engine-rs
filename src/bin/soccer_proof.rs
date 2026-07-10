#![recursion_limit = "256"]

//! `soccer_proof` — a rigorous, self-contained learnability proof for the neural + DP +
//! POMDP + MPC passing stack.
//!
//! It answers one question with statistics: does training with the neural/DP learning stack
//! *cause* passing to improve — measured primarily by PASS COMPLETION RATE and
//! completed forward passes, with yards/completed-pass, route-one, pass-turnovers,
//! shots-on-target, and dribble success as guardrails — against a FROZEN opponent so there is no
//! self-play moving-target confound.
//!
//! Subcommands:
//!   fresh <out.json> <seed_hex>
//!       Writes a zero-gradient neural snapshot with the same env-controlled architecture as
//!       `train-ckpt`. Use this as the gen0/untrained anchor.
//!   train-ckpt <out_prefix> <games> <minutes> <seed_hex> <ckpt_every>
//!       Inline self-play carry-forward honoring env gates. Writes `<out_prefix>.gen{K}.json`
//!       every `ckpt_every` games and a final `<out_prefix>.genFINAL.json`. The learning-curve
//!       proof evals each checkpoint against the same frozen anchor.
//!   eval <candidate> <baseline> <games> <minutes> <holdout_hex>
//!       Plays two FROZEN brains head-to-head over a HELD-OUT disjoint seed range, alternating
//!       home/away. `candidate`/`baseline` may be a snapshot path or the literal `analytic`
//!       (a net-less pure-analytic brain). Emits one JSON object per game to $PROOF_JSONL (if
//!       set) and prints a paired-difference summary with 95% AND 99.99% lower bounds.
//!
//! Parallelism: run several `eval` processes over DISJOINT holdout ranges and concatenate their
//! JSONL; `scripts/proof_aggregate.py` folds them into one paired verdict. DP gates
//! (DP_CRITIC_TARGET / DP_BOOTSTRAP / DEFERRED_PASS_CREDIT) are TRAINING-ONLY, so eval of frozen
//! brains is identical regardless of them — toggling them across two `train-ckpt` runs is a
//! clean causal test on the trained policy.
//!
//! Compatibility: set `SOCCER_PROOF_STRIP_AUX_HEADS=1` only when a legacy snapshot's carried
//! side actors (policy/specialist/keeper/line-depth/MPC heads) have stale feature dimensions and
//! you need a value/critic-only proof run. Leave it unset for full-stack proof.

use std::io::Write;
use std::sync::Arc;
use std::time::{Duration, Instant};

use soccer_engine::des::general::soccer::{
    enable_deterministic_formation_lp, learned_mpc_objective_enabled, AttackSpacingHead,
    MatchConfig, MatchStats, SoccerMarlAlgorithm, SoccerMatch, SoccerMpcObjectiveHead,
    SoccerNeuralLearningBackend, SoccerNeuralLearningConfig, SoccerNeuralNetworkSnapshot,
    SoccerPassCompletionHead, SoccerQPolicyOptions, SoccerTeamQPolicies, SupportScorerHead, Team,
    ATTACK_SPACING_HEAD_MIN_TRAINING_STEPS, DEFAULT_SOCCER_MAPPO_TEAM_REWARD_SHARE,
    DEFAULT_SOCCER_PASS_COMPLETION_LEARNING_RATE, PASS_COMPLETION_HEAD_MIN_TRAINING_STEPS,
    SUPPORT_SCORER_HEAD_MIN_TRAINING_STEPS,
};
use soccer_engine::des::general::tournament::{
    EngineMatchRunner, EngineMatchRunnerConfig, TeamBrain, TournamentMatchContext,
    TournamentMatchRunner, TournamentStage,
};

fn parse_hex(s: Option<&String>, default: u32) -> u32 {
    s.and_then(|raw| u32::from_str_radix(raw.trim_start_matches("0x"), 16).ok())
        .unwrap_or(default)
}
fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(default)
}
fn env_bool(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(default)
}
fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(default)
}

// Stride used to spread held-out seeds; coprime-ish with small factors so consecutive games are
// far apart in seed space (same constant the outcome_ab harness uses).
const SEED_STRIDE: u32 = 2_246_822_519;

/// Train a candidate by inline self-play carry-forward, honoring whatever env gates this process
/// was launched with, writing a checkpoint snapshot every `ckpt_every` games.
fn proof_neural_config() -> SoccerNeuralLearningConfig {
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
    neural
}

fn write_snapshot(path: &str, snap: &SoccerNeuralNetworkSnapshot) {
    let json = serde_json::to_string(snap).unwrap_or_else(|e| panic!("serialize {path}: {e}"));
    std::fs::write(path, json).unwrap_or_else(|e| panic!("write {path}: {e}"));
}

fn fresh_snapshot(out_path: &str, seed_base: u32) {
    enable_deterministic_formation_lp();
    let neural = proof_neural_config();
    let mut config = MatchConfig {
        duration_seconds: 0.0,
        learning_enabled: true,
        neural_learning: neural.clone(),
        seed: seed_base,
        ..MatchConfig::default()
    };
    config.neural_blend.actor_critic = true;
    let mut sim = SoccerMatch::default_11v11(config);
    sim.set_uniform_elite_players();
    if env_bool("DD_SOCCER_ENABLE_LEARNED_PASS_COMPLETION", false) {
        sim.set_pass_completion_head(SoccerPassCompletionHead::new(seed_base));
    }
    if learned_mpc_objective_enabled() {
        sim.set_mpc_objective_head(SoccerMpcObjectiveHead::new(seed_base));
    }
    if std::env::var("DD_SOCCER_ENABLE_LEARNED_SPACING_TARGET").is_ok() {
        sim.set_attack_spacing_head(AttackSpacingHead::new(seed_base));
    }
    if !env_bool("DD_SOCCER_DISABLE_LEARNED_SUPPORT_SCORER", false) {
        sim.set_support_scorer_head(SupportScorerHead::new(seed_base));
    }
    let snapshot = sim
        .neural_network_snapshot()
        .unwrap_or_else(|| panic!("fresh neural snapshot was not initialized for {out_path}"));
    write_snapshot(out_path, &snapshot);
    println!(
        "[fresh] wrote {out_path} seed=0x{seed_base:08X} target_scale={} target_clip={} popart={} hidden={} training_steps={}",
        neural.target_scale,
        neural.target_clip,
        neural.target_popart_enabled,
        neural.hidden_units,
        snapshot.training_steps
    );
}

/// Train a candidate by inline self-play carry-forward, honoring whatever env gates this process
/// was launched with, writing a checkpoint snapshot every `ckpt_every` games.
fn train_ckpt(out_prefix: &str, games: usize, minutes: f64, seed_base: u32, ckpt_every: usize) {
    enable_deterministic_formation_lp();
    let neural = proof_neural_config();
    println!(
        "[train-ckpt] prefix={out_prefix} games={games} minutes={minutes} seed=0x{seed_base:08X} \
         ckpt_every={ckpt_every} target_scale={} target_clip={} popart={} hidden={}",
        neural.target_scale, neural.target_clip, neural.target_popart_enabled, neural.hidden_units
    );

    let mut policies = Arc::new(SoccerTeamQPolicies::new(SoccerQPolicyOptions::default()));
    let mut snapshot: Option<SoccerNeuralNetworkSnapshot> = None;
    let mut pass_completion_head: Option<SoccerPassCompletionHead> = None;
    let mut mpc_head: Option<SoccerMpcObjectiveHead> = None;
    let mut attack_spacing_head: Option<AttackSpacingHead> = None;
    let mut support_scorer_head: Option<SupportScorerHead> = None;
    let started = Instant::now();

    let write_ckpt = |snap: &SoccerNeuralNetworkSnapshot, tag: &str| {
        let path = format!("{out_prefix}.gen{tag}.json");
        match serde_json::to_string(snap) {
            Ok(json) => {
                if let Err(e) = std::fs::write(&path, json) {
                    eprintln!("[train-ckpt] write {path} failed: {e}");
                } else {
                    println!("[train-ckpt] wrote checkpoint {path}");
                }
            }
            Err(e) => eprintln!("[train-ckpt] serialize failed: {e}"),
        }
    };

    for g in 0..games {
        let mut config = MatchConfig {
            duration_seconds: minutes * 60.0,
            learning_enabled: true,
            neural_learning: neural.clone(),
            seed: seed_base.wrapping_add(g as u32),
            ..MatchConfig::default()
        };
        config.neural_blend.actor_critic = true;
        let total_ticks = config.total_ticks();
        let mut sim = SoccerMatch::default_11v11(config).with_team_policies((*policies).clone());
        sim.set_uniform_elite_players();
        // ESCAPE SELF-PLAY PARITY (raise the ceiling): for a fraction of games, install the net
        // only on HOME (which learns) and leave AWAY as PURE ANALYTIC — so the net trains to BEAT
        // the analytic engine (an external "better") instead of only tying itself. Set via
        // SOCCER_PROOF_ANALYTIC_OPPONENT_FRAC (0=pure self-play, 1=always vs analytic).
        let analytic_frac = env_f64("SOCCER_PROOF_ANALYTIC_OPPONENT_FRAC", 0.0);
        let vs_analytic = analytic_frac > 0.0
            && (analytic_frac >= 1.0 || g % ((1.0 / analytic_frac).round() as usize).max(1) == 0);
        if let Some(s) = snapshot.as_ref() {
            let installed = if vs_analytic {
                sim.set_team_neural_brain(Team::Home, Some(s.clone()), false)
                    .and_then(|_| sim.set_team_neural_brain(Team::Away, None, true))
            } else {
                sim.set_neural_network_snapshot(s.clone())
            };
            if let Err(e) = installed {
                eprintln!("[train-ckpt] game {g}: brain install failed: {e}");
            }
        } else if vs_analytic {
            // Cold start vs analytic: fresh net on Home, analytic Away.
            let _ = sim.set_team_neural_brain(Team::Away, None, true);
        }
        if let Some(head) = pass_completion_head.as_ref() {
            sim.set_pass_completion_head(head.clone());
        }
        if learned_mpc_objective_enabled() {
            let head = mpc_head.get_or_insert_with(|| SoccerMpcObjectiveHead::new(seed_base));
            sim.set_mpc_objective_head(head.clone());
        }
        if let Some(head) = attack_spacing_head.as_ref() {
            sim.set_attack_spacing_head(head.clone());
        }
        if let Some(head) = support_scorer_head.as_ref() {
            sim.set_support_scorer_head(head.clone());
        }
        for _ in 0..total_ticks {
            sim.run_time_step();
        }
        sim.drain_neural_learning(Duration::from_millis(100));
        let pass_samples = sim.drain_pass_outcome_samples();
        if !pass_samples.is_empty() {
            let head = pass_completion_head.get_or_insert_with(|| {
                SoccerPassCompletionHead::new(seed_base.wrapping_add(g as u32))
            });
            let mut final_loss = 0.0;
            for _ in 0..4 {
                final_loss =
                    head.train(&pass_samples, DEFAULT_SOCCER_PASS_COMPLETION_LEARNING_RATE);
            }
            sim.set_pass_completion_head(head.clone());
            eprintln!(
                "[train-ckpt] pass-completion game={} samples={} training_steps={} consumed={} final_loss={:.5}",
                g + 1,
                pass_samples.len(),
                head.training_steps(),
                head.training_steps() >= PASS_COMPLETION_HEAD_MIN_TRAINING_STEPS,
                final_loss
            );
        }
        if learned_mpc_objective_enabled() {
            let samples = sim.drain_mpc_objective_samples();
            if !samples.is_empty() {
                if let Some(head) = mpc_head.as_mut() {
                    head.train_rwr(&samples, 0.05);
                    sim.set_mpc_objective_head(head.clone());
                }
            }
        }
        let attack_spacing_samples = sim.drain_attack_spacing_samples();
        if !attack_spacing_samples.is_empty() {
            let head = attack_spacing_head
                .get_or_insert_with(|| AttackSpacingHead::new(seed_base.wrapping_add(g as u32)));
            let mut final_loss = 0.0;
            for _ in 0..4 {
                final_loss = head.train(&attack_spacing_samples, 0.02);
            }
            sim.set_attack_spacing_head(head.clone());
            eprintln!(
                "[train-ckpt] attack-spacing game={} samples={} training_steps={} consumed={} final_loss={:.5}",
                g + 1,
                attack_spacing_samples.len(),
                head.training_steps(),
                head.training_steps() >= ATTACK_SPACING_HEAD_MIN_TRAINING_STEPS,
                final_loss
            );
        }
        let support_move_samples = sim.drain_support_move_samples();
        if !support_move_samples.is_empty() {
            let head = support_scorer_head
                .get_or_insert_with(|| SupportScorerHead::new(seed_base.wrapping_add(g as u32)));
            let mut final_loss = 0.0;
            for _ in 0..4 {
                final_loss = head.train(&support_move_samples, 0.02);
            }
            sim.set_support_scorer_head(head.clone());
            eprintln!(
                "[train-ckpt] support-scorer game={} samples={} training_steps={} consumed={} final_loss={:.5}",
                g + 1,
                support_move_samples.len(),
                head.training_steps(),
                head.training_steps() >= SUPPORT_SCORER_HEAD_MIN_TRAINING_STEPS,
                final_loss
            );
        }
        if let Some(p) = sim.team_policies() {
            policies = Arc::new(p.clone());
        }
        if let Some(s) = sim.neural_network_snapshot() {
            snapshot = Some(s);
        }
        let s = sim.summary();
        eprintln!(
            "[train-ckpt] game {:>3}/{games} score {}-{} ({:.0}s)",
            g + 1,
            s.score_home,
            s.score_away,
            started.elapsed().as_secs_f64()
        );
        let done = g + 1;
        if ckpt_every > 0 && done % ckpt_every == 0 {
            if let Some(s) = snapshot.as_ref() {
                write_ckpt(s, &done.to_string());
            }
        }
    }
    if let Some(s) = snapshot.as_ref() {
        write_ckpt(s, "FINAL");
    } else {
        eprintln!("[train-ckpt] no snapshot produced");
    }
    println!(
        "[train-ckpt] done {games} games ({:.0}s)",
        started.elapsed().as_secs_f64()
    );
}

/// Per-team metrics extracted from one match, for the passing-progression proof + guardrails.
#[derive(Clone, Copy, Default)]
struct TeamMetrics {
    goals: f64,
    shots: f64,
    sot: f64,
    att: f64,
    comp: f64,
    fwd: f64,
    back: f64,
    gain_yards: f64, // sum of forward yards over completed passes (can be negative)
    strict_att: f64,
    strict_comp: f64,
    strict_fwd: f64,
    strict_back: f64,
    strict_gain_yards: f64,
    strict_pass_turnovers: f64,
    chains: f64,
    shots_after_pass: f64,
    assists: f64,
    dribble_beats: f64,
    dribble_turnovers: f64,
    route_one: f64,
    interceptions_won: f64,
    pass_turnovers: f64,
    tackles: f64,
    loose_ball_recoveries: f64,
    teamwork_upfield: f64,
    teamwork_near_ball: f64,
    possession_chase_advantage: f64,
    defensive_chase_load: f64,
}

fn home_metrics(st: &MatchStats) -> TeamMetrics {
    TeamMetrics {
        goals: 0.0, // set from o.home_goals in play_fixture
        shots: f64::from(st.shots_home),
        sot: f64::from(st.shots_on_target_home),
        att: f64::from(st.passes_attempted_home),
        comp: f64::from(st.passes_completed_home),
        fwd: f64::from(st.passes_completed_forward_home),
        back: f64::from(st.passes_completed_backward_home),
        gain_yards: st.completed_pass_gain_yards_home,
        strict_att: f64::from(st.intentional_passes_attempted_home),
        strict_comp: f64::from(st.intentional_passes_completed_home),
        strict_fwd: f64::from(st.intentional_passes_completed_forward_home),
        strict_back: f64::from(st.intentional_passes_completed_backward_home),
        strict_gain_yards: st.intentional_completed_pass_gain_yards_home,
        strict_pass_turnovers: f64::from(st.intentional_pass_interceptions_home),
        chains: f64::from(st.pass_chains_home),
        shots_after_pass: f64::from(st.shots_after_pass_home),
        assists: f64::from(st.assists_home),
        dribble_beats: f64::from(st.dribble_beats_home),
        dribble_turnovers: f64::from(st.dribble_turnovers_home),
        route_one: f64::from(st.route_one_balls_home),
        interceptions_won: f64::from(st.interceptions_home),
        pass_turnovers: f64::from(st.interceptions_away),
        tackles: f64::from(st.tackles_home),
        loose_ball_recoveries: f64::from(st.loose_ball_recoveries_home),
        teamwork_upfield: st.teamwork_upfield_progress_home,
        teamwork_near_ball: st.teamwork_near_ball_progress_home,
        possession_chase_advantage: st.possession_chase_advantage_home,
        defensive_chase_load: st.defensive_chase_load_home,
    }
}
fn away_metrics(st: &MatchStats) -> TeamMetrics {
    TeamMetrics {
        goals: 0.0,
        shots: f64::from(st.shots_away),
        sot: f64::from(st.shots_on_target_away),
        att: f64::from(st.passes_attempted_away),
        comp: f64::from(st.passes_completed_away),
        fwd: f64::from(st.passes_completed_forward_away),
        back: f64::from(st.passes_completed_backward_away),
        gain_yards: st.completed_pass_gain_yards_away,
        strict_att: f64::from(st.intentional_passes_attempted_away),
        strict_comp: f64::from(st.intentional_passes_completed_away),
        strict_fwd: f64::from(st.intentional_passes_completed_forward_away),
        strict_back: f64::from(st.intentional_passes_completed_backward_away),
        strict_gain_yards: st.intentional_completed_pass_gain_yards_away,
        strict_pass_turnovers: f64::from(st.intentional_pass_interceptions_away),
        chains: f64::from(st.pass_chains_away),
        shots_after_pass: f64::from(st.shots_after_pass_away),
        assists: f64::from(st.assists_away),
        dribble_beats: f64::from(st.dribble_beats_away),
        dribble_turnovers: f64::from(st.dribble_turnovers_away),
        route_one: f64::from(st.route_one_balls_away),
        interceptions_won: f64::from(st.interceptions_away),
        pass_turnovers: f64::from(st.interceptions_home),
        tackles: f64::from(st.tackles_away),
        loose_ball_recoveries: f64::from(st.loose_ball_recoveries_away),
        teamwork_upfield: st.teamwork_upfield_progress_away,
        teamwork_near_ball: st.teamwork_near_ball_progress_away,
        possession_chase_advantage: st.possession_chase_advantage_away,
        defensive_chase_load: st.defensive_chase_load_away,
    }
}

fn load_brain(spec: &str) -> TeamBrain {
    if spec.eq_ignore_ascii_case("analytic") {
        // Net-less brain => the authoritative-neural branch is skipped => the hand-built analytic
        // engine decides. This is the honest frozen "target" the learner must climb toward.
        TeamBrain::fresh()
    } else {
        let json = std::fs::read_to_string(spec).unwrap_or_else(|e| panic!("read {spec}: {e}"));
        let mut snap: SoccerNeuralNetworkSnapshot =
            serde_json::from_str(&json).unwrap_or_else(|e| panic!("parse {spec}: {e}"));
        if env_bool("SOCCER_PROOF_STRIP_AUX_HEADS", false) {
            eprintln!(
                "[load] SOCCER_PROOF_STRIP_AUX_HEADS=1: evaluating value/critic net only for {spec}"
            );
            snap.policy_head = None;
            snap.skill_policy_heads = None;
            snap.keeper_policy_head = None;
            snap.line_depth_head = None;
            snap.pass_completion_head = None;
            snap.mpc_objective_head = None;
            snap.attack_spacing_head = None;
            snap.support_scorer_head = None;
        }
        TeamBrain::from_snapshot(snap)
    }
}

/// One held-out fixture; returns (candidate_metrics, baseline_metrics) regardless of orientation.
fn play_fixture(
    runner: &mut EngineMatchRunner,
    cand: &TeamBrain,
    base: &TeamBrain,
    seed: u32,
    cand_home: bool,
) -> Option<(TeamMetrics, TeamMetrics)> {
    let (home_id, away_id) = if cand_home {
        (0usize, 1usize)
    } else {
        (1usize, 0usize)
    };
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
    let (home, away) = if cand_home {
        (cand, base)
    } else {
        (base, cand)
    };
    match runner.play(&ctx, home, away) {
        Ok(o) => {
            let st = &o.summary.stats;
            let mut hm = home_metrics(st);
            let mut am = away_metrics(st);
            hm.goals = f64::from(o.home_goals);
            am.goals = f64::from(o.away_goals);
            if cand_home {
                Some((hm, am))
            } else {
                Some((am, hm))
            }
        }
        Err(e) => {
            eprintln!("[eval] fixture error: {e}");
            None
        }
    }
}

struct Stat {
    mean: f64,
    lb95: f64,
    lb9999: f64,
    n: usize,
}
/// Paired one-sided lower confidence bounds on the mean of `diffs` (candidate − baseline).
fn paired_stat(diffs: &[f64]) -> Stat {
    let n = diffs.len();
    if n == 0 {
        return Stat {
            mean: 0.0,
            lb95: 0.0,
            lb9999: 0.0,
            n,
        };
    }
    let nf = n as f64;
    let mean = diffs.iter().sum::<f64>() / nf;
    let var = if n > 1 {
        diffs.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (nf - 1.0)
    } else {
        0.0
    };
    let se = (var / nf).sqrt();
    Stat {
        mean,
        lb95: mean - 1.959_964 * se,   // 97.5% one-sided
        lb9999: mean - 3.719_016 * se, // 99.99% one-sided (z for p=1e-4)
        n,
    }
}

fn safe_ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator.abs() > 1e-9 {
        numerator / denominator
    } else {
        0.0
    }
}

fn eval(cand_spec: &str, base_spec: &str, games: usize, minutes: f64, holdout: u32) {
    enable_deterministic_formation_lp();
    let cand = load_brain(cand_spec);
    let base = load_brain(base_spec);
    let mut cfg = EngineMatchRunnerConfig::default();
    cfg.base.duration_seconds = minutes * 60.0;
    // The runner defaults to critic-only (actor_critic=false) so a SHARED actor can't blur a
    // neural-vs-neural A/B. But when exactly one side is a net (neural-vs-analytic), the shared
    // actor IS that net's own actor, so enabling it captures the FULL trained policy (actor +
    // critic) — where pass-SELECTION learning lives. Opt in via SOCCER_PROOF_EVAL_ACTOR_CRITIC=1.
    if env_bool("SOCCER_PROOF_EVAL_ACTOR_CRITIC", false) {
        cfg.base.neural_blend.actor_critic = true;
        println!("[eval] actor_critic=ON (full policy; valid only for neural-vs-analytic)");
    }
    let mut runner = EngineMatchRunner::new(cfg);

    let jsonl_path = std::env::var("PROOF_JSONL").ok();
    let mut jsonl = jsonl_path.as_ref().map(|p| {
        let f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(p)
            .expect("open jsonl");
        std::io::BufWriter::new(f)
    });

    println!(
        "[eval] candidate={cand_spec} vs baseline={base_spec}; {games} held-out games, minutes={minutes}, base=0x{holdout:08X}"
    );
    let started = Instant::now();

    // Accumulators for the headline passing metrics and anti-gaming guards.
    let mut d_completed: Vec<f64> = Vec::with_capacity(games);
    let mut d_completion_rate: Vec<f64> = Vec::with_capacity(games);
    let mut d_fwd: Vec<f64> = Vec::with_capacity(games);
    let mut d_yards: Vec<f64> = Vec::with_capacity(games);
    let mut d_pass_turnover_rate_improvement: Vec<f64> = Vec::with_capacity(games);
    let mut d_strict_fwd: Vec<f64> = Vec::with_capacity(games);
    let mut d_strict_net_fwd: Vec<f64> = Vec::with_capacity(games);
    let mut d_strict_pass_turnover_rate_improvement: Vec<f64> = Vec::with_capacity(games);
    // Totals for rate/guardrail reporting.
    let mut c = TeamMetrics::default();
    let mut b = TeamMetrics::default();
    macro_rules! acc {
        ($dst:expr, $src:expr, $($f:ident),+) => { $( $dst.$f += $src.$f; )+ };
    }
    let mut n_done = 0usize;
    for g in 0..games {
        let seed = holdout.wrapping_add((g as u32).wrapping_mul(SEED_STRIDE));
        let cand_home = g % 2 == 0;
        let Some((cm, bm)) = play_fixture(&mut runner, &cand, &base, seed, cand_home) else {
            continue;
        };
        n_done += 1;
        d_completed.push(cm.comp - bm.comp);
        d_completion_rate.push(safe_ratio(cm.comp, cm.att) - safe_ratio(bm.comp, bm.att));
        d_fwd.push(cm.fwd - bm.fwd);
        d_yards.push(cm.gain_yards - bm.gain_yards);
        d_pass_turnover_rate_improvement
            .push(safe_ratio(bm.pass_turnovers, bm.att) - safe_ratio(cm.pass_turnovers, cm.att));
        d_strict_fwd.push(cm.strict_fwd - bm.strict_fwd);
        d_strict_net_fwd.push(
            (cm.strict_fwd - cm.strict_pass_turnovers) - (bm.strict_fwd - bm.strict_pass_turnovers),
        );
        d_strict_pass_turnover_rate_improvement.push(
            safe_ratio(bm.strict_pass_turnovers, bm.strict_att)
                - safe_ratio(cm.strict_pass_turnovers, cm.strict_att),
        );
        acc!(
            c,
            cm,
            goals,
            shots,
            sot,
            att,
            comp,
            fwd,
            back,
            gain_yards,
            strict_att,
            strict_comp,
            strict_fwd,
            strict_back,
            strict_gain_yards,
            strict_pass_turnovers,
            chains,
            shots_after_pass,
            assists,
            dribble_beats,
            dribble_turnovers,
            route_one,
            interceptions_won,
            pass_turnovers,
            tackles,
            loose_ball_recoveries,
            teamwork_upfield,
            teamwork_near_ball,
            possession_chase_advantage,
            defensive_chase_load
        );
        acc!(
            b,
            bm,
            goals,
            shots,
            sot,
            att,
            comp,
            fwd,
            back,
            gain_yards,
            strict_att,
            strict_comp,
            strict_fwd,
            strict_back,
            strict_gain_yards,
            strict_pass_turnovers,
            chains,
            shots_after_pass,
            assists,
            dribble_beats,
            dribble_turnovers,
            route_one,
            interceptions_won,
            pass_turnovers,
            tackles,
            loose_ball_recoveries,
            teamwork_upfield,
            teamwork_near_ball,
            possession_chase_advantage,
            defensive_chase_load
        );
        if let Some(w) = jsonl.as_mut() {
            let _ = writeln!(
                w,
                "{}",
                serde_json::json!({
                    "seed": seed,
                    "cand_home": cand_home,
                    "c_goals": cm.goals,
                    "b_goals": bm.goals,
                    "c_shots": cm.shots,
                    "b_shots": bm.shots,
                    "c_sot": cm.sot,
                    "b_sot": bm.sot,
                    "c_att": cm.att,
                    "b_att": bm.att,
                    "c_comp": cm.comp,
                    "b_comp": bm.comp,
                    "c_fwd": cm.fwd,
                    "b_fwd": bm.fwd,
                    "c_back": cm.back,
                    "b_back": bm.back,
                    "c_yards": cm.gain_yards,
                    "b_yards": bm.gain_yards,
                    "c_ipatt": cm.strict_att,
                    "b_ipatt": bm.strict_att,
                    "c_ipcomp": cm.strict_comp,
                    "b_ipcomp": bm.strict_comp,
                    "c_ipfwd": cm.strict_fwd,
                    "b_ipfwd": bm.strict_fwd,
                    "c_ipback": cm.strict_back,
                    "b_ipback": bm.strict_back,
                    "c_ipyards": cm.strict_gain_yards,
                    "b_ipyards": bm.strict_gain_yards,
                    "c_ipto": cm.strict_pass_turnovers,
                    "b_ipto": bm.strict_pass_turnovers,
                    "c_chains": cm.chains,
                    "b_chains": bm.chains,
                    "c_sap": cm.shots_after_pass,
                    "b_sap": bm.shots_after_pass,
                    "c_assist": cm.assists,
                    "b_assist": bm.assists,
                    "c_drib": cm.dribble_beats,
                    "b_drib": bm.dribble_beats,
                    "c_drib_to": cm.dribble_turnovers,
                    "b_drib_to": bm.dribble_turnovers,
                    "c_route1": cm.route_one,
                    "b_route1": bm.route_one,
                    "c_int": cm.interceptions_won,
                    "b_int": bm.interceptions_won,
                    "c_int_won": cm.interceptions_won,
                    "b_int_won": bm.interceptions_won,
                    "c_pto": cm.pass_turnovers,
                    "b_pto": bm.pass_turnovers,
                    "c_tackles": cm.tackles,
                    "b_tackles": bm.tackles,
                    "c_lbr": cm.loose_ball_recoveries,
                    "b_lbr": bm.loose_ball_recoveries,
                    "c_team_up": cm.teamwork_upfield,
                    "b_team_up": bm.teamwork_upfield,
                    "c_team_near": cm.teamwork_near_ball,
                    "b_team_near": bm.teamwork_near_ball,
                    "c_chase_adv": cm.possession_chase_advantage,
                    "b_chase_adv": bm.possession_chase_advantage,
                    "c_def_chase": cm.defensive_chase_load,
                    "b_def_chase": bm.defensive_chase_load
                })
            );
            let _ = w.flush(); // durable per-game so killed workers keep data & progress is monitorable
        }
        if (g + 1) % 10 == 0 || g + 1 == games {
            eprintln!(
                "[eval] {}/{games} played ({:.0}s)",
                g + 1,
                started.elapsed().as_secs_f64()
            );
        }
    }
    if let Some(w) = jsonl.as_mut() {
        let _ = w.flush();
    }
    if n_done == 0 {
        println!("[eval] no completed fixtures");
        return;
    }
    let nf = n_done as f64;
    let pct = |num: f64, den: f64| 100.0 * safe_ratio(num, den);
    let completed_stat = paired_stat(&d_completed);
    let completion_rate_stat = paired_stat(&d_completion_rate);
    let fwd_stat = paired_stat(&d_fwd);
    let yards_stat = paired_stat(&d_yards);
    let pass_turnover_rate_improvement_stat = paired_stat(&d_pass_turnover_rate_improvement);
    let strict_fwd_stat = paired_stat(&d_strict_fwd);
    let strict_net_fwd_stat = paired_stat(&d_strict_net_fwd);
    let strict_pass_turnover_rate_improvement_stat =
        paired_stat(&d_strict_pass_turnover_rate_improvement);
    let cand_yards_per_pass = safe_ratio(c.gain_yards, c.comp);
    let base_yards_per_pass = safe_ratio(b.gain_yards, b.comp);
    let cand_strict_yards_per_pass = safe_ratio(c.strict_gain_yards, c.strict_comp);
    let base_strict_yards_per_pass = safe_ratio(b.strict_gain_yards, b.strict_comp);
    let yards_per_pass_delta = cand_yards_per_pass - base_yards_per_pass;
    let route_one_delta_pg = (c.route_one - b.route_one) / nf;
    let cand_pass_turnover_rate = safe_ratio(c.pass_turnovers, c.att);
    let base_pass_turnover_rate = safe_ratio(b.pass_turnovers, b.att);
    let cand_strict_pass_turnover_rate = safe_ratio(c.strict_pass_turnovers, c.strict_att);
    let base_strict_pass_turnover_rate = safe_ratio(b.strict_pass_turnovers, b.strict_att);
    let pass_turnover_rate_delta = cand_pass_turnover_rate - base_pass_turnover_rate;
    let strict_pass_turnover_rate_delta =
        cand_strict_pass_turnover_rate - base_strict_pass_turnover_rate;
    let yards_guard_floor = env_f64("SOCCER_PROOF_MIN_YARDS_PER_PASS_DELTA", -0.25);
    let route_one_guard_max = env_f64("SOCCER_PROOF_MAX_ROUTE_ONE_DELTA_PER_GAME", 0.10);
    let pass_turnover_rate_guard_max = env_f64("SOCCER_PROOF_MAX_PASS_TURNOVER_RATE_DELTA", 0.0);
    let yards_guard_ok = yards_per_pass_delta >= yards_guard_floor;
    let route_one_guard_ok = route_one_delta_pg <= route_one_guard_max;
    let strict_pass_turnover_guard_ok =
        strict_pass_turnover_rate_delta <= pass_turnover_rate_guard_max;

    println!("\n===== PASSING PROGRESSION PROOF (candidate − baseline, paired over {n_done} held-out games) =====");
    println!(
        "completed passes/game:         cand {:.2}  base {:.2}  Δ {:+.2}  [95% LB {:+.2}]  [99.99% LB {:+.2}]  n={}",
        c.comp / nf, b.comp / nf, completed_stat.mean, completed_stat.lb95, completed_stat.lb9999, completed_stat.n
    );
    println!(
        "pass completion rate:          cand {:.2}%  base {:.2}%  Δ {:+.2}pp  [95% LB {:+.2}pp]  [99.99% LB {:+.2}pp]  n={}",
        pct(c.comp, c.att),
        pct(b.comp, b.att),
        completion_rate_stat.mean * 100.0,
        completion_rate_stat.lb95 * 100.0,
        completion_rate_stat.lb9999 * 100.0,
        completion_rate_stat.n
    );
    println!(
        "completed FORWARD passes/game: cand {:.2}  base {:.2}  Δ {:+.2}  [95% LB {:+.2}]  [99.99% LB {:+.2}]  n={}",
        c.fwd / nf, b.fwd / nf, fwd_stat.mean, fwd_stat.lb95, fwd_stat.lb9999, fwd_stat.n
    );
    println!(
        "forward YARDS gained/game:     cand {:.1}  base {:.1}  Δ {:+.1}  [95% LB {:+.2}]  [99.99% LB {:+.2}]  n={}",
        c.gain_yards / nf, b.gain_yards / nf, yards_stat.mean, yards_stat.lb95, yards_stat.lb9999, yards_stat.n
    );
    println!(
        "pass turnover rate:            cand {:.2}%  base {:.2}%  improvement {:+.2}pp  [95% LB {:+.2}pp]  [99.99% LB {:+.2}pp]  n={}",
        cand_pass_turnover_rate * 100.0,
        base_pass_turnover_rate * 100.0,
        pass_turnover_rate_improvement_stat.mean * 100.0,
        pass_turnover_rate_improvement_stat.lb95 * 100.0,
        pass_turnover_rate_improvement_stat.lb9999 * 100.0,
        pass_turnover_rate_improvement_stat.n
    );
    println!(
        "strict FORWARD passes/game:    cand {:.2}  base {:.2}  Δ {:+.2}  [95% LB {:+.2}]  [99.99% LB {:+.2}]  n={}",
        c.strict_fwd / nf,
        b.strict_fwd / nf,
        strict_fwd_stat.mean,
        strict_fwd_stat.lb95,
        strict_fwd_stat.lb9999,
        strict_fwd_stat.n
    );
    println!(
        "strict NET forward/game:       cand {:.2}  base {:.2}  Δ {:+.2}  [95% LB {:+.2}]  [99.99% LB {:+.2}]  n={}",
        (c.strict_fwd - c.strict_pass_turnovers) / nf,
        (b.strict_fwd - b.strict_pass_turnovers) / nf,
        strict_net_fwd_stat.mean,
        strict_net_fwd_stat.lb95,
        strict_net_fwd_stat.lb9999,
        strict_net_fwd_stat.n
    );
    println!(
        "strict pass turnover rate:     cand {:.2}%  base {:.2}%  improvement {:+.2}pp  [95% LB {:+.2}pp]  [99.99% LB {:+.2}pp]  n={}",
        cand_strict_pass_turnover_rate * 100.0,
        base_strict_pass_turnover_rate * 100.0,
        strict_pass_turnover_rate_improvement_stat.mean * 100.0,
        strict_pass_turnover_rate_improvement_stat.lb95 * 100.0,
        strict_pass_turnover_rate_improvement_stat.lb9999 * 100.0,
        strict_pass_turnover_rate_improvement_stat.n
    );
    let headline = if completion_rate_stat.lb9999 > 0.0
        && strict_net_fwd_stat.lb9999 > 0.0
        && yards_guard_ok
        && route_one_guard_ok
        && strict_pass_turnover_guard_ok
    {
        "PROVEN@99.99% — pass completion rate and strict net-forward intentional passes have 99.99% lower bounds above zero, with yards/pass, route-one, and strict pass-turnover guards intact"
    } else if completion_rate_stat.lb95 > 0.0
        && strict_net_fwd_stat.lb95 > 0.0
        && yards_guard_ok
        && route_one_guard_ok
        && strict_pass_turnover_guard_ok
    {
        "CLIMB@95% — completion rate and strict net-forward intentional passes are significant at 95%, with strict pass-turnover guard intact; need more held-out games for 99.99%"
    } else if completion_rate_stat.mean > 0.0
        && strict_net_fwd_stat.mean > 0.0
        && yards_guard_ok
        && strict_pass_turnover_guard_ok
    {
        "directional — completion rate and strict net-forward intentional passing improved on average, not yet significant"
    } else if strict_net_fwd_stat.lb95 > 0.0 && yards_guard_ok && strict_pass_turnover_guard_ok {
        "progression-only climb — strict net-forward intentional progression is significant, but pass-completion-rate proof is incomplete"
    } else if !strict_pass_turnover_guard_ok {
        "anti-gaming guard failed — candidate increased strict intentional pass-turnover rate"
    } else if strict_net_fwd_stat.mean > 0.0 || completion_rate_stat.mean > 0.0 {
        "partial directional movement — more held-out games or a better candidate are needed"
    } else {
        "no passing advancement"
    };
    println!("VERDICT: {headline}");

    println!("\n----- rate & progression detail -----");
    println!(
        "cand: {:.1} att/g  {:.1}% comp  fwd {:.1}%  back {:.1}%  yards/pass {:.2}  chains {:.1}/g",
        c.att / nf,
        pct(c.comp, c.att),
        pct(c.fwd, c.comp),
        pct(c.back, c.comp),
        cand_yards_per_pass,
        c.chains / nf
    );
    println!(
        "base: {:.1} att/g  {:.1}% comp  fwd {:.1}%  back {:.1}%  yards/pass {:.2}  chains {:.1}/g",
        b.att / nf,
        pct(b.comp, b.att),
        pct(b.fwd, b.comp),
        pct(b.back, b.comp),
        base_yards_per_pass,
        b.chains / nf
    );
    println!(
        "strict cand: {:.1} att/g  {:.1}% comp  fwd {:.1}%  back {:.1}%  yards/pass {:.2}",
        c.strict_att / nf,
        pct(c.strict_comp, c.strict_att),
        pct(c.strict_fwd, c.strict_comp),
        pct(c.strict_back, c.strict_comp),
        cand_strict_yards_per_pass
    );
    println!(
        "strict base: {:.1} att/g  {:.1}% comp  fwd {:.1}%  back {:.1}%  yards/pass {:.2}",
        b.strict_att / nf,
        pct(b.strict_comp, b.strict_att),
        pct(b.strict_fwd, b.strict_comp),
        pct(b.strict_back, b.strict_comp),
        base_strict_yards_per_pass
    );
    println!(
        "guards: yards/pass delta {:+.2} (floor {:+.2}) route-one delta/g {:+.2} (max {:+.2}) pass-turnover rate delta {:+.2}pp strict {:+.2}pp (max {:+.2}pp)",
        yards_per_pass_delta,
        yards_guard_floor,
        route_one_delta_pg,
        route_one_guard_max,
        pass_turnover_rate_delta * 100.0,
        strict_pass_turnover_rate_delta * 100.0,
        pass_turnover_rate_guard_max * 100.0
    );

    println!("\n----- guardrails (candidate vs baseline per game) -----");
    println!(
        "goals:        {:.2} vs {:.2}   (GD {:+.2})",
        c.goals / nf,
        b.goals / nf,
        (c.goals - b.goals) / nf
    );
    println!("shots:        {:.2} vs {:.2}", c.shots / nf, b.shots / nf);
    println!("shots on tgt: {:.2} vs {:.2}", c.sot / nf, b.sot / nf);
    println!(
        "SOT rate:     {:.2}% vs {:.2}%   goals/shot {:.2}% vs {:.2}%",
        safe_ratio(c.sot, c.shots) * 100.0,
        safe_ratio(b.sot, b.shots) * 100.0,
        safe_ratio(c.goals, c.shots) * 100.0,
        safe_ratio(b.goals, b.shots) * 100.0
    );
    println!(
        "shots-after-pass: {:.2} vs {:.2}",
        c.shots_after_pass / nf,
        b.shots_after_pass / nf
    );
    println!(
        "worked SOT share:{:.2}% vs {:.2}%",
        safe_ratio(c.shots_after_pass, c.sot) * 100.0,
        safe_ratio(b.shots_after_pass, b.sot) * 100.0
    );
    println!(
        "assists:      {:.2} vs {:.2}",
        c.assists / nf,
        b.assists / nf
    );
    println!(
        "dribble beats:{:.2} vs {:.2}",
        c.dribble_beats / nf,
        b.dribble_beats / nf
    );
    println!(
        "dribble contest success:{:.2}% vs {:.2}%   turnovers {:.2} vs {:.2}",
        safe_ratio(c.dribble_beats, c.dribble_beats + c.dribble_turnovers) * 100.0,
        safe_ratio(b.dribble_beats, b.dribble_beats + b.dribble_turnovers) * 100.0,
        c.dribble_turnovers / nf,
        b.dribble_turnovers / nf
    );
    println!(
        "route-one:    {:.2} vs {:.2}   (lower=better; long-ball regression guard)",
        c.route_one / nf,
        b.route_one / nf
    );
    println!(
        "pass turnovers:{:.2} vs {:.2}   (lower=better; opponent interceptions)",
        c.pass_turnovers / nf,
        b.pass_turnovers / nf
    );
    println!(
        "strict pass turnovers:{:.2} vs {:.2}   (lower=better; intentional pass interceptions)",
        c.strict_pass_turnovers / nf,
        b.strict_pass_turnovers / nf
    );
    println!(
        "interceptions won:{:.2} vs {:.2}",
        c.interceptions_won / nf,
        b.interceptions_won / nf
    );
    println!(
        "recoveries/tackles:{:.2}/{:.2} vs {:.2}/{:.2}",
        c.loose_ball_recoveries / nf,
        c.tackles / nf,
        b.loose_ball_recoveries / nf,
        b.tackles / nf
    );
    println!(
        "teamwork upfield/near-ball:{:.2}/{:.2} vs {:.2}/{:.2}",
        c.teamwork_upfield / nf,
        c.teamwork_near_ball / nf,
        b.teamwork_upfield / nf,
        b.teamwork_near_ball / nf
    );
    println!(
        "chase advantage/defensive load:{:.2}/{:.2} vs {:.2}/{:.2}",
        c.possession_chase_advantage / nf,
        c.defensive_chase_load / nf,
        b.possession_chase_advantage / nf,
        b.defensive_chase_load / nf
    );
    println!("\n[eval] done in {:.0}s", started.elapsed().as_secs_f64());
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("fresh") => {
            let out = args.get(2).expect("usage: fresh <out.json> <seed_hex>");
            let seed = parse_hex(args.get(3), 0x71A1_0000);
            fresh_snapshot(out, seed);
        }
        Some("train-ckpt") => {
            let prefix = args
                .get(2)
                .expect("usage: train-ckpt <out_prefix> <games> <minutes> <seed_hex> <ckpt_every>");
            let games = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(40);
            let minutes = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(2.0);
            let seed = parse_hex(args.get(5), 0x71A1_0000);
            let ckpt = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(5);
            train_ckpt(prefix, games, minutes, seed, ckpt);
        }
        Some("eval") => {
            let cand = args.get(2).expect("usage: eval <candidate|analytic> <baseline|analytic> <games> <minutes> <holdout_hex>");
            let base = args
                .get(3)
                .expect("usage: eval <candidate|analytic> <baseline|analytic> ...");
            let games = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(60);
            let minutes = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(2.0);
            let holdout = parse_hex(args.get(6), 0xE7A1_0000);
            eval(cand, base, games, minutes, holdout);
        }
        _ => {
            eprintln!(
                "usage:\n  soccer_proof fresh <out.json> <seed_hex>\n  \
                 soccer_proof train-ckpt <out_prefix> <games> <minutes> <seed_hex> <ckpt_every>\n  \
                 soccer_proof eval <candidate|analytic> <baseline|analytic> <games> <minutes> <holdout_hex>\n  \
                 (set $PROOF_JSONL to append per-game rows; run several evals over disjoint holdouts in parallel)"
            );
            std::process::exit(2);
        }
    }
}
