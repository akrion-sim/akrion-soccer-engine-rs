//! `soccer_proof` — a rigorous, self-contained learnability proof for the neural + DP +
//! POMDP + MPC passing stack.
//!
//! It answers one question with statistics: does training with the neural/DP learning stack
//! *cause* passing to improve — measured primarily by COMPLETED FORWARD PASSES and YARDS
//! GAINED PER PASS (progression that cannot be farmed by short safe pokes), with completion
//! rate, shots-on-target, and dribble success as guardrails — against a FROZEN opponent so
//! there is no self-play moving-target confound.
//!
//! Subcommands:
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

use std::sync::Arc;
use std::time::{Duration, Instant};

use soccer_engine::des::general::soccer::{
    enable_deterministic_formation_lp, learned_mpc_objective_enabled, MatchConfig, MatchStats,
    SoccerMarlAlgorithm, SoccerMatch, SoccerMpcObjectiveHead, SoccerNeuralLearningBackend,
    SoccerNeuralLearningConfig, SoccerNeuralNetworkSnapshot, SoccerQPolicyOptions,
    SoccerTeamQPolicies, DEFAULT_SOCCER_MAPPO_TEAM_REWARD_SHARE,
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
fn train_ckpt(out_prefix: &str, games: usize, minutes: f64, seed_base: u32, ckpt_every: usize) {
    enable_deterministic_formation_lp();
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
        "[train-ckpt] prefix={out_prefix} games={games} minutes={minutes} seed=0x{seed_base:08X} \
         ckpt_every={ckpt_every} target_scale={} target_clip={} popart={} hidden={}",
        neural.target_scale, neural.target_clip, neural.target_popart_enabled, neural.hidden_units
    );

    let mut policies = Arc::new(SoccerTeamQPolicies::new(SoccerQPolicyOptions::default()));
    let mut snapshot: Option<SoccerNeuralNetworkSnapshot> = None;
    let mut mpc_head: Option<SoccerMpcObjectiveHead> = None;
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
        if let Some(s) = snapshot.as_ref() {
            if let Err(e) = sim.set_neural_network_snapshot(s.clone()) {
                eprintln!("[train-ckpt] game {g}: snapshot install failed: {e}");
            }
        }
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
    chains: f64,
    shots_after_pass: f64,
    assists: f64,
    dribble_beats: f64,
    route_one: f64,
    interceptions: f64,
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
        chains: f64::from(st.pass_chains_home),
        shots_after_pass: f64::from(st.shots_after_pass_home),
        assists: f64::from(st.assists_home),
        dribble_beats: f64::from(st.dribble_beats_home),
        route_one: f64::from(st.route_one_balls_home),
        interceptions: f64::from(st.interceptions_home),
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
        chains: f64::from(st.pass_chains_away),
        shots_after_pass: f64::from(st.shots_after_pass_away),
        assists: f64::from(st.assists_away),
        dribble_beats: f64::from(st.dribble_beats_away),
        route_one: f64::from(st.route_one_balls_away),
        interceptions: f64::from(st.interceptions_away),
    }
}

/// Produce a FRESH, untrained neural snapshot (random init, zero gradient steps) for the
/// "learning is real" baseline: the net exists and re-ranks the analytic candidates but has
/// learned nothing. We build the sim exactly as training does and play the match to force lazy
/// net creation, but NEVER call `drain_neural_learning` — so no gradient is ever applied and the
/// extracted weights are the random initialization.
fn fresh_snapshot(out_path: &str, seed: u32) {
    enable_deterministic_formation_lp();
    let neural = SoccerNeuralLearningConfig {
        enabled: true,
        backend: SoccerNeuralLearningBackend::Inline,
        marl_algorithm: SoccerMarlAlgorithm::Mappo,
        mappo_team_reward_share: DEFAULT_SOCCER_MAPPO_TEAM_REWARD_SHARE,
        ..SoccerNeuralLearningConfig::default()
    };
    let policies = Arc::new(SoccerTeamQPolicies::new(SoccerQPolicyOptions::default()));
    let mut config = MatchConfig {
        duration_seconds: 120.0,
        learning_enabled: true,
        neural_learning: neural,
        seed,
        ..MatchConfig::default()
    };
    config.neural_blend.actor_critic = true;
    let total_ticks = config.total_ticks();
    let mut sim = SoccerMatch::default_11v11(config).with_team_policies((*policies).clone());
    sim.set_uniform_elite_players();
    // Play so the net is created and exercised, but do NOT drain -> zero gradient updates.
    for _ in 0..total_ticks {
        sim.run_time_step();
    }
    let snap = sim
        .neural_network_snapshot()
        .expect("fresh net snapshot (net should exist after play)");
    let json = serde_json::to_string(&snap).expect("serialize fresh snapshot");
    std::fs::write(out_path, json).expect("write fresh snapshot");
    println!("[fresh] wrote untrained snapshot {out_path} (0 gradient steps, seed 0x{seed:08X})");
}

fn load_brain(spec: &str) -> TeamBrain {
    if spec.eq_ignore_ascii_case("analytic") {
        // Net-less brain => the authoritative-neural branch is skipped => the hand-built analytic
        // engine decides. This is the honest frozen "target" the learner must climb toward.
        TeamBrain::fresh()
    } else {
        let json = std::fs::read_to_string(spec).unwrap_or_else(|e| panic!("read {spec}: {e}"));
        let snap: SoccerNeuralNetworkSnapshot =
            serde_json::from_str(&json).unwrap_or_else(|e| panic!("parse {spec}: {e}"));
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

fn eval(cand_spec: &str, base_spec: &str, games: usize, minutes: f64, holdout: u32) {
    use std::io::Write;
    enable_deterministic_formation_lp();
    let cand = load_brain(cand_spec);
    let base = load_brain(base_spec);
    let mut cfg = EngineMatchRunnerConfig::default();
    cfg.base.duration_seconds = minutes * 60.0;
    let mut runner = EngineMatchRunner::new(cfg);

    let jsonl_path = std::env::var("PROOF_JSONL").ok();
    let mut jsonl = jsonl_path.as_ref().map(|p| {
        use std::io::Write;
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

    // Accumulators for the two headline PROGRESSION metrics (paired per game).
    let mut d_fwd: Vec<f64> = Vec::with_capacity(games);
    let mut d_yards: Vec<f64> = Vec::with_capacity(games);
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
        d_fwd.push(cm.fwd - bm.fwd);
        d_yards.push(cm.gain_yards - bm.gain_yards);
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
            chains,
            shots_after_pass,
            assists,
            dribble_beats,
            route_one,
            interceptions
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
            chains,
            shots_after_pass,
            assists,
            dribble_beats,
            route_one,
            interceptions
        );
        if let Some(w) = jsonl.as_mut() {
            let _ = writeln!(
                w,
                "{{\"seed\":{},\"cand_home\":{},\"c_goals\":{},\"b_goals\":{},\"c_shots\":{},\"b_shots\":{},\"c_sot\":{},\"b_sot\":{},\"c_att\":{},\"b_att\":{},\"c_comp\":{},\"b_comp\":{},\"c_fwd\":{},\"b_fwd\":{},\"c_back\":{},\"b_back\":{},\"c_yards\":{:.3},\"b_yards\":{:.3},\"c_chains\":{},\"b_chains\":{},\"c_sap\":{},\"b_sap\":{},\"c_assist\":{},\"b_assist\":{},\"c_drib\":{},\"b_drib\":{},\"c_route1\":{},\"b_route1\":{},\"c_int\":{},\"b_int\":{}}}",
                seed, cand_home,
                cm.goals, bm.goals, cm.shots, bm.shots, cm.sot, bm.sot, cm.att, bm.att,
                cm.comp, bm.comp, cm.fwd, bm.fwd, cm.back, bm.back, cm.gain_yards, bm.gain_yards,
                cm.chains, bm.chains, cm.shots_after_pass, bm.shots_after_pass, cm.assists, bm.assists,
                cm.dribble_beats, bm.dribble_beats, cm.route_one, bm.route_one, cm.interceptions, bm.interceptions
            );
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
    let pct = |num: f64, den: f64| if den > 0.0 { 100.0 * num / den } else { 0.0 };
    let fwd_stat = paired_stat(&d_fwd);
    let yards_stat = paired_stat(&d_yards);

    println!("\n===== PASSING PROGRESSION PROOF (candidate − baseline, paired over {n_done} held-out games) =====");
    println!(
        "completed FORWARD passes/game: cand {:.2}  base {:.2}  Δ {:+.2}  [95% LB {:+.2}]  [99.99% LB {:+.2}]",
        c.fwd / nf, b.fwd / nf, fwd_stat.mean, fwd_stat.lb95, fwd_stat.lb9999
    );
    println!(
        "forward YARDS gained/game:     cand {:.1}  base {:.1}  Δ {:+.1}  [95% LB {:+.2}]  [99.99% LB {:+.2}]",
        c.gain_yards / nf, b.gain_yards / nf, yards_stat.mean, yards_stat.lb95, yards_stat.lb9999
    );
    let headline = if fwd_stat.lb9999 > 0.0 && yards_stat.lb9999 > 0.0 {
        "PROVEN@99.99% — candidate out-progresses baseline on forward passes AND yards (both 99.99% LB > 0)"
    } else if fwd_stat.lb95 > 0.0 {
        "CLIMB@95% — significant at 95% on forward passes; need more games for 99.99%"
    } else if fwd_stat.mean > 0.0 {
        "directional — more forward passes on average, not yet significant"
    } else {
        "no forward-pass advancement"
    };
    println!("VERDICT: {headline}");

    println!("\n----- rate & progression detail -----");
    println!(
        "cand: {:.1} att/g  {:.1}% comp  fwd {:.1}%  back {:.1}%  yards/pass {:.2}  chains {:.1}/g",
        c.att / nf,
        pct(c.comp, c.att),
        pct(c.fwd, c.comp),
        pct(c.back, c.comp),
        if c.comp > 0.0 {
            c.gain_yards / c.comp
        } else {
            0.0
        },
        c.chains / nf
    );
    println!(
        "base: {:.1} att/g  {:.1}% comp  fwd {:.1}%  back {:.1}%  yards/pass {:.2}  chains {:.1}/g",
        b.att / nf,
        pct(b.comp, b.att),
        pct(b.fwd, b.comp),
        pct(b.back, b.comp),
        if b.comp > 0.0 {
            b.gain_yards / b.comp
        } else {
            0.0
        },
        b.chains / nf
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
        "shots-after-pass: {:.2} vs {:.2}",
        c.shots_after_pass / nf,
        b.shots_after_pass / nf
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
        "route-one:    {:.2} vs {:.2}   (lower=better; long-ball regression guard)",
        c.route_one / nf,
        b.route_one / nf
    );
    println!(
        "interceptions:{:.2} vs {:.2}",
        c.interceptions / nf,
        b.interceptions / nf
    );
    println!("\n[eval] done in {:.0}s", started.elapsed().as_secs_f64());
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
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
                "usage:\n  soccer_proof train-ckpt <out_prefix> <games> <minutes> <seed_hex> <ckpt_every>\n  \
                 soccer_proof eval <candidate|analytic> <baseline|analytic> <games> <minutes> <holdout_hex>\n  \
                 (set $PROOF_JSONL to append per-game rows; run several evals over disjoint holdouts in parallel)"
            );
            std::process::exit(2);
        }
    }
}
