//! Standalone 5-a-side reinforcement-learning demo.
//!
//! Fully independent & isolated: no dependency on des_engine / soccer_engine or
//! any surrounding crate. Own PRNG, own MLP + PPO. The point is a system small
//! enough that you can watch a policy learn to actually play — pass, dribble,
//! shoot — and climb from losing to the scripted baseline to beating it.
//!
//! Usage:
//!   cargo run --release -- train [iters]   # train + write out/learning_curve.csv, out/match.json
//!   cargo run --release -- sanity          # scripted-vs-scripted sanity check
//!   cargo run --release -- eval            # (re)train briefly and print eval

#![allow(clippy::needless_range_loop)]

mod game;
mod nn;
mod rng;
mod train;

use game::*;
use rng::Rng;
use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

type AppResult<T> = Result<T, Box<dyn Error>>;
const SELFPLAY_MAX_GENERATIONS: usize = 200;
const SELFPLAY_MAX_SPEED_WARMUP: usize = 10_000;

struct SelfplayCleanup;

impl Drop for SelfplayCleanup {
    fn drop(&mut self) {
        train::clear_selfplay_champion();
        train::set_speed_frozen(false);
    }
}

#[derive(Clone)]
struct RunConfig {
    iters: usize,
    seed: u64,
    games_per_iter: usize,
    bc_games: usize,
    bc_epochs: usize,
    eval_every: usize,
    eval_games: usize,
    final_games: usize,
    display_seed_max: u64,
    out_dir: PathBuf,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            iters: 100,
            seed: 20260710,
            games_per_iter: 12,
            bc_games: 12,
            bc_epochs: 2,
            eval_every: 5,
            eval_games: 60,
            final_games: 300,
            display_seed_max: 120,
            out_dir: PathBuf::from("out"),
        }
    }
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> AppResult<()> {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("train");
    match cmd {
        "train" => {
            let cfg = parse_run_config(&args)?;
            run_training(&cfg)?;
        }
        "selfplay" => {
            let cfg = parse_run_config(&args)?;
            run_selfplay(&cfg)?;
        }
        "sanity" => sanity(),
        "inspect" => {
            let seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(7);
            let out_dir = parse_out_dir(&args, 3)?;
            inspect(seed, &out_dir)?;
        }
        "play" => {
            let seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(7);
            let mut out_dir = PathBuf::from("out");
            let mut out_path: Option<PathBuf> = None;
            let mut i = 3;
            while i < args.len() {
                let flag = args[i].as_str();
                let val = args
                    .get(i + 1)
                    .ok_or_else(|| format!("missing value for {flag}"))?;
                match flag {
                    "--out-dir" => out_dir = PathBuf::from(val),
                    "--out" => out_path = Some(PathBuf::from(val)),
                    other => return Err(format!("unknown option for play: {other}").into()),
                }
                i += 2;
            }
            let out_path = out_path.unwrap_or_else(|| out_dir.join("match_live.json"));
            play(seed, &out_dir, &out_path)?;
        }
        "help" | "--help" | "-h" => print_usage(),
        other => return Err(format!("unknown command: {other}; run with --help for usage").into()),
    }
    Ok(())
}

fn print_usage() {
    println!("usage:");
    println!("  cargo run --release -- train [iters] [--seed N] [--games-per-iter N] [--bc-games N] [--bc-epochs N] [--eval-every N] [--eval-games N] [--final-games N] [--display-seed-max N] [--out-dir DIR]");
    println!("  cargo run --release -- selfplay [iters-per-gen] [--seed N] [--games-per-iter N] [--bc-games N] [--bc-epochs N] [--eval-games N] [--out-dir DIR]");
    println!("      # adversarial self-play ladder — both teams learned; new winner must beat the frozen");
    println!("      # champion by PROMOTE_MARGIN (goal-diff) to advance. env: GENERATIONS, PROMOTE_MARGIN, SPEED_WARMUP");
    println!("  cargo run --release -- inspect [seed] [--out-dir DIR]");
    println!("  cargo run --release -- play [seed] [--out-dir DIR] [--out PATH]  # record one match (live New Game)");
    println!("  cargo run --release -- sanity");
}

fn parse_run_config(args: &[String]) -> AppResult<RunConfig> {
    let mut cfg = RunConfig::default();
    let mut i = 2usize;
    if let Some(raw) = args.get(i) {
        if !raw.starts_with("--") {
            cfg.iters = raw
                .parse()
                .map_err(|_| format!("invalid iteration count: {raw}"))?;
            i += 1;
        }
    }
    while i < args.len() {
        let flag = args[i].as_str();
        let value = args
            .get(i + 1)
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag {
            "--seed" => {
                cfg.seed = value
                    .parse()
                    .map_err(|_| format!("invalid seed: {value}"))?
            }
            "--games-per-iter" => {
                cfg.games_per_iter = value
                    .parse()
                    .map_err(|_| format!("invalid games-per-iter: {value}"))?
            }
            "--bc-games" => {
                cfg.bc_games = value
                    .parse()
                    .map_err(|_| format!("invalid bc-games: {value}"))?
            }
            "--bc-epochs" => {
                cfg.bc_epochs = value
                    .parse()
                    .map_err(|_| format!("invalid bc-epochs: {value}"))?
            }
            "--eval-every" => {
                cfg.eval_every = value
                    .parse()
                    .map_err(|_| format!("invalid eval-every: {value}"))?
            }
            "--eval-games" => {
                cfg.eval_games = value
                    .parse()
                    .map_err(|_| format!("invalid eval-games: {value}"))?
            }
            "--final-games" => {
                cfg.final_games = value
                    .parse()
                    .map_err(|_| format!("invalid final-games: {value}"))?
            }
            "--display-seed-max" => {
                cfg.display_seed_max = value
                    .parse()
                    .map_err(|_| format!("invalid display-seed-max: {value}"))?
            }
            "--out-dir" => cfg.out_dir = PathBuf::from(value),
            other => return Err(format!("unknown option: {other}").into()),
        }
        i += 2;
    }
    if cfg.iters == 0 {
        return Err("iters must be > 0".into());
    }
    if cfg.games_per_iter == 0 || cfg.eval_every == 0 || cfg.eval_games == 0 || cfg.final_games == 0
    {
        return Err("games/eval counts must be > 0".into());
    }
    if cfg.bc_games > 0 && cfg.bc_epochs == 0 {
        return Err("bc-epochs must be > 0 when bc-games is enabled".into());
    }
    if cfg.display_seed_max == 0 {
        return Err("display-seed-max must be > 0".into());
    }
    Ok(cfg)
}

fn env_usize_clamped(name: &str, default: usize, min: usize, max: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .map(|value| value.clamp(min, max))
        .unwrap_or(default.clamp(min, max))
}

fn env_f32_clamped(name: &str, default: f32, min: f32, max: f32) -> f32 {
    let lo = min.min(max);
    let hi = min.max(max);
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .filter(|value| value.is_finite())
        .map(|value| value.clamp(lo, hi))
        .unwrap_or(default.clamp(lo, hi))
}

/// Load the trained policy from `out_dir` and record ONE fresh match (viz JSON)
/// with the given seed to `out_path` — powers the live "New Game" button, which
/// the live server (viz/serve_live.py) invokes per click with a random seed.
///   cargo run --release -- play [seed] [--out-dir DIR] [--out PATH]
fn play(seed: u64, out_dir: &Path, out_path: &Path) -> AppResult<()> {
    let actor = nn::Mlp::load(&out_dir.join("actor.txt"))
        .map_err(|e| format!("no trained policy in {}: {e}", out_dir.display()))?;
    let critic = nn::Mlp::load(&out_dir.join("critic.txt"))
        .map_err(|e| format!("failed to load critic: {e}"))?;
    let speedor = nn::Mlp::load(&out_dir.join("speedor.txt"))
        .map_err(|e| format!("failed to load speedor: {e}"))?;
    if actor.in_dim() != OBS_DIM || actor.out_dim() != NA {
        return Err("actor shape mismatch; retrain the policy".into());
    }
    let policy = train::Policy {
        actor,
        speedor,
        critic,
    };
    record_match(&policy, &mut Rng::new(seed), out_path)?;
    println!("wrote {}", out_path.display());
    Ok(())
}

fn parse_out_dir(args: &[String], start: usize) -> AppResult<PathBuf> {
    let mut out_dir = PathBuf::from("out");
    let mut i = start;
    while i < args.len() {
        let flag = args[i].as_str();
        let value = args
            .get(i + 1)
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag {
            "--out-dir" => out_dir = PathBuf::from(value),
            other => return Err(format!("unknown option for inspect: {other}").into()),
        }
        i += 2;
    }
    Ok(out_dir)
}

fn action_name(a: usize) -> &'static str {
    match a {
        A_SHOOT => "shoot",
        A_PASS_A => "pass_a",
        A_PASS_B => "pass_b",
        A_PASS_C => "pass_c",
        A_DRIB_FWD => "dribble_fwd",
        A_DRIB_LEFT => "dribble_left",
        A_DRIB_RIGHT => "dribble_right",
        A_CLEAR => "clear",
        A_HOLD => "hold",
        A_CHASE => "chase",
        A_SUPPORT => "support",
        A_SPREAD => "spread",
        A_MARK => "mark",
        _ => "stay",
    }
}

/// Introspection API: load the trained policy and play one game, emitting the
/// FULL per-tick state as JSON Lines to out/trace.jsonl (positions, velocities,
/// possession, each player's chosen action, per-player nearest-teammate /
/// nearest-opponent distances, closest-pair, bunch flag, events, score) plus a
/// live per-second console trace and an end-of-game bunching summary.
///
///   cargo run --release -- inspect [seed]
///   # then: jq -c '{t,cp:.closest_pair_a,bunch}' out/trace.jsonl | head
fn inspect(seed: u64, out_dir: &Path) -> AppResult<()> {
    let actor_path = out_dir.join("actor.txt");
    let critic_path = out_dir.join("critic.txt");
    let actor = match nn::Mlp::load(&actor_path) {
        Ok(m) => m,
        Err(_) => {
            return Err(format!("no trained policy found at {}", actor_path.display()).into());
        }
    };
    let critic = nn::Mlp::load(&critic_path)
        .map_err(|err| format!("failed to load critic at {}: {err}", critic_path.display()))?;
    let speedor_path = out_dir.join("speedor.txt");
    let speedor = nn::Mlp::load(&speedor_path).map_err(|err| {
        format!(
            "failed to load speedor at {}: {err}",
            speedor_path.display()
        )
    })?;
    if actor.in_dim() != OBS_DIM || actor.out_dim() != NA {
        return Err(format!(
            "actor shape mismatch in {}: expected input {} / actions {}, got input {} / actions {}; retrain the policy",
            actor_path.display(),
            OBS_DIM,
            NA,
            actor.in_dim(),
            actor.out_dim()
        )
        .into());
    }
    if critic.in_dim() != GLOBAL_DIM || critic.out_dim() != 1 {
        return Err(format!(
            "critic shape mismatch in {}: expected input {} / output 1, got input {} / output {}; retrain the policy",
            critic_path.display(),
            GLOBAL_DIM,
            critic.in_dim(),
            critic.out_dim()
        )
        .into());
    }
    let policy = train::Policy {
        actor,
        speedor,
        critic,
    };
    let mut rng = Rng::new(seed);
    let mut w = World::new();
    fs::create_dir_all(out_dir)?;

    let mut jsonl = String::new();
    let mut bunch_ticks = 0u32;
    let mut overlap_ticks = 0u32;
    let mut sum_closest = 0.0f32;
    let mut sum_avg = 0.0f32;
    const BUNCH_THRESH: f32 = 2.5;
    const OVERLAP_THRESH: f32 = 1.5;

    println!("== inspect: greedy game, seed {} ==", seed);
    println!(
        "{:>4} | {:>7} | own | closest_pair | avg_near | bunch | score",
        "t", "ball_x"
    );

    for t in 0..STEPS {
        // choose actions (record them for the trace)
        let mut act_a = [A_STAY; N];
        for i in 1..N {
            act_a[i] = policy.act_greedy_world(&w, Team::A, i);
        }
        let act_b = w.scripted_actions(Team::B);
        w.step(&act_a, &act_b, &mut rng);

        let closest = w.closest_pair_a();
        let avg = w.avg_nearest_teammate_a();
        let bunched = closest < BUNCH_THRESH;
        if bunched {
            bunch_ticks += 1;
        }
        if closest < OVERLAP_THRESH {
            overlap_ticks += 1;
        }
        sum_closest += closest;
        sum_avg += avg;

        jsonl.push_str(&frame_json(&w, t, &act_a, &act_b, closest, avg, bunched));
        jsonl.push('\n');

        if t % 10 == 0 {
            let own = match w.owner {
                Some(o) if matches!(o.team, Team::A) => "A ",
                Some(_) => "B ",
                None => "- ",
            };
            println!(
                "{:>4} | {:>7.1} | {}  | {:>12.1} | {:>8.1} | {:>5} | {}-{}",
                t,
                w.ball.x,
                own,
                closest,
                avg,
                if bunched { "YES" } else { "no" },
                w.goals_a,
                w.goals_b
            );
        }
    }

    let trace_path = out_dir.join("trace.jsonl");
    write_atomic(&trace_path, &jsonl)?;
    let n = STEPS as f32;
    println!("\n-- bunching summary (Team A outfielders) --");
    println!("  closest-pair distance: mean {:.1}", sum_closest / n);
    println!("  avg nearest-teammate:  mean {:.1}", sum_avg / n);
    println!(
        "  frames with a pair < {:.1} (bunched): {:.0}%  ({}/{})",
        BUNCH_THRESH,
        100.0 * bunch_ticks as f32 / n,
        bunch_ticks,
        STEPS
    );
    println!(
        "  frames with a pair < {:.1} (OVERLAP/major): {:.0}%  ({}/{})",
        OVERLAP_THRESH,
        100.0 * overlap_ticks as f32 / n,
        overlap_ticks,
        STEPS
    );
    println!(
        "  full per-tick trace -> {} ({} frames)",
        trace_path.display(),
        STEPS
    );
    Ok(())
}

/// One frame of the introspection trace as a JSON object (hand-rolled).
fn frame_json(
    w: &World,
    t: usize,
    act_a: &[usize; N],
    act_b: &[usize; N],
    closest: f32,
    avg: f32,
    bunched: bool,
) -> String {
    let (own, oi) = match w.owner {
        Some(o) if matches!(o.team, Team::A) => (0i32, o.idx as i32),
        Some(o) => (1i32, o.idx as i32),
        None => (-1i32, -1i32),
    };
    let mut s = String::new();
    let _ = write!(
        s,
        "{{\"t\":{},\"ball\":[{:.2},{:.2}],\"ball_vel\":[{:.2},{:.2}],\"own\":{},\"oi\":{},\"ga\":{},\"gb\":{},\"closest_pair_a\":{:.2},\"avg_near_a\":{:.2},\"bunch\":{},\"players\":[",
        t, w.ball.x, w.ball.y, w.ball_vel.x, w.ball_vel.y, own, oi, w.goals_a, w.goals_b, closest, avg, bunched
    );
    let mut first = true;
    for (team, code, acts) in [(Team::A, 0i32, act_a), (Team::B, 1i32, act_b)] {
        for i in 0..N {
            if !first {
                s.push(',');
            }
            first = false;
            let p = match team {
                Team::A => w.a[i],
                Team::B => w.b[i],
            };
            // nearest teammate + nearest opponent (outfielders only for teammate)
            let team_arr = if code == 0 { &w.a } else { &w.b };
            let opp_arr = if code == 0 { &w.b } else { &w.a };
            let mut nt = f32::INFINITY;
            for j in 0..N {
                if j != i {
                    let d = team_arr[j].pos.sub(p.pos).len();
                    if d < nt {
                        nt = d;
                    }
                }
            }
            let mut no = f32::INFINITY;
            for j in 0..N {
                let d = opp_arr[j].pos.sub(p.pos).len();
                if d < no {
                    no = d;
                }
            }
            let _ = write!(
                s,
                "{{\"team\":{},\"idx\":{},\"gk\":{},\"x\":{:.2},\"y\":{:.2},\"vx\":{:.2},\"vy\":{:.2},\"action\":\"{}\",\"near_team\":{:.2},\"near_opp\":{:.2}}}",
                code, i, i == GK, p.pos.x, p.pos.y, p.vel.x, p.vel.y, action_name(acts[i]), nt, no
            );
        }
    }
    s.push_str("]}");
    s
}

fn sanity() {
    let mut rng = Rng::new(1234);
    let (diff, ga, gb) = train::evaluate_scripted_vs_scripted(200, &mut rng);
    println!("scripted vs scripted over 200 games:");
    println!(
        "  avg goal diff = {:+.3}  (should be ~0.0 — symmetric baseline)",
        diff
    );
    println!("  avg goals A = {:.3}, avg goals B = {:.3}", ga, gb);
}

/// Persist a policy's three networks (actor/critic/speedor) into `dir` in the same
/// layout `run_training` uses, so `inspect`/`play`/the viz can load it.
fn save_policy(policy: &train::Policy, dir: &Path) -> AppResult<()> {
    fs::create_dir_all(dir)?;
    policy.actor.save(&dir.join("actor.txt"))?;
    policy.speedor.save(&dir.join("speedor.txt"))?;
    policy.critic.save(&dir.join("critic.txt"))?;
    Ok(())
}

/// Adversarial SELF-PLAY ladder. Both teams are learned policies: a frozen champion
/// plays Team B while a challenger trains (Team A) against it. Each generation the
/// challenger is scored head-to-head vs the champion; if it wins by `PROMOTE_MARGIN`
/// (goal-diff) it becomes the new champion ("new winner beats old winner to advance").
/// All moves execute through `World::step`, so play stays within the physics bounds.
fn run_selfplay(cfg: &RunConfig) -> AppResult<()> {
    let _selfplay_cleanup = SelfplayCleanup;
    let mut rng = Rng::new(cfg.seed);
    fs::create_dir_all(&cfg.out_dir)?;
    let champ_dir = cfg.out_dir.join("champions");
    fs::create_dir_all(&champ_dir)?;

    let generations = env_usize_clamped("GENERATIONS", 12, 1, SELFPLAY_MAX_GENERATIONS);
    let promote_margin = env_f32_clamped("PROMOTE_MARGIN", 0.25, -20.0, 20.0);
    let speed_warmup = env_usize_clamped(
        "SPEED_WARMUP",
        (cfg.iters / 2).max(1),
        0,
        SELFPLAY_MAX_SPEED_WARMUP,
    );

    // Challenger: warm-start by behavior-cloning the scripted baseline so gen-0 play
    // is coherent (not random flailing); it then improves purely via self-play.
    let mut challenger = train::Policy::new(&mut rng);
    if cfg.bc_games > 0 {
        let bc =
            train::behavior_clone_scripted(&mut challenger, cfg.bc_games, cfg.bc_epochs, &mut rng);
        println!(
            "warm start (behavior-clone scripted): {} samples, teacher-match {:.0}%",
            bc.samples,
            bc.accuracy * 100.0
        );
    }
    // Generation-0 champion = the scripted baseline itself (Team B scripted). The
    // first learned policy to beat it by the margin becomes gen-1 champion; the
    // ladder climbs from there ("new winner beats old winner to advance").
    let mut champion: Option<train::Policy> = None;
    let mut champion_gen = 0usize;

    let (svs, _, _) = train::evaluate_scripted_vs_scripted(60, &mut rng);
    println!("=== 5-a-side ADVERSARIAL SELF-PLAY ladder — both teams learned, physics-bounded ===");
    println!("scripted-vs-scripted goal_diff={svs:+.2} (sanity ~0)");
    println!(
        "generations={generations}  iters/gen={}  games/iter={}  promote_margin={promote_margin:+.2}  eval_games={}  speed_warmup={speed_warmup}",
        cfg.iters, cfg.games_per_iter, cfg.eval_games
    );
    println!("{}", "-".repeat(92));
    println!(
        "{:>4} | {:>18} | {:>18} | {:>9} | champion",
        "gen", "vs champion", "vs scripted", "result"
    );

    let mut ladder = String::from(
        "generation,iters_per_gen,cand_vs_champ_gd,cand_vs_champ_wr,cand_vs_scripted_gd,cand_vs_scripted_wr,promoted,champion_gen\n",
    );

    for round in 1..=generations {
        // Install the current champion (None => scripted baseline) as Team B, then
        // train the challenger against it.
        train::set_selfplay_champion(champion.clone());
        for it in 1..=cfg.iters {
            train::set_speed_frozen(it <= speed_warmup);
            let beta = train::ent_beta_at(it, cfg.iters);
            let _ = train::train_iter(&mut challenger, cfg.games_per_iter, beta, &mut rng);
        }
        train::set_selfplay_champion(None); // detach so evals are clean

        // Champion-ladder gate: beat the current champion (scripted at gen 0) by the margin.
        let (gd_champ, wr_champ) = match &champion {
            Some(c) => train::evaluate_vs_policy(&challenger, c, cfg.eval_games, &mut rng),
            None => {
                let s = train::evaluate(&challenger, cfg.eval_games, &mut rng);
                (s.goal_diff, s.winrate)
            }
        };
        // Absolute progress reference: challenger vs the scripted baseline.
        let s = train::evaluate(&challenger, cfg.eval_games, &mut rng);
        let promoted = gd_champ >= promote_margin;
        if promoted {
            champion = Some(challenger.clone());
            champion_gen += 1;
            save_policy(&challenger, &champ_dir.join(format!("gen{champion_gen}")))?;
        }
        let champ_label = if champion_gen == 0 {
            "scripted".to_string()
        } else {
            format!("gen{champion_gen}")
        };
        println!(
            "{round:>4} | gd {gd_champ:>+6.2} wr {wr_champ:>4.2} | gd {:>+6.2} wr {:>4.2} | {:>9} | {champ_label}",
            s.goal_diff,
            s.winrate,
            if promoted { "PROMOTED" } else { "hold" }
        );
        ladder.push_str(&format!(
            "{round},{},{gd_champ:.4},{wr_champ:.4},{:.4},{:.4},{},{champion_gen}\n",
            cfg.iters, s.goal_diff, s.winrate, promoted
        ));
    }

    // Persist the final champion (or the challenger if none was ever promoted) in the
    // standard layout so inspect/play/viz can load it.
    let final_policy = champion.as_ref().unwrap_or(&challenger);
    save_policy(final_policy, &cfg.out_dir)?;
    write_atomic(&cfg.out_dir.join("selfplay_ladder.csv"), &ladder)?;
    let sfinal = train::evaluate(final_policy, cfg.final_games.max(cfg.eval_games), &mut rng);
    println!("{}", "-".repeat(92));
    println!(
        "final champion = gen{champion_gen} after {generations} generations | vs scripted: goal_diff={:+.2} winrate={:.2} goals {:.2}-{:.2}",
        sfinal.goal_diff, sfinal.winrate, sfinal.ga, sfinal.gb
    );
    println!(
        "saved champion -> {} (+ champions/gen*/ snapshots, selfplay_ladder.csv)",
        cfg.out_dir.display()
    );
    Ok(())
}

fn run_training(cfg: &RunConfig) -> AppResult<()> {
    let mut rng = Rng::new(cfg.seed);
    let mut policy = train::Policy::new(&mut rng);

    fs::create_dir_all(&cfg.out_dir)?;

    // Baseline reference: how does the *scripted* team fare vs itself, and how
    // does an UNTRAINED (random) policy fare vs scripted?
    let (svs, sga, sgb) = train::evaluate_scripted_vs_scripted(100, &mut rng);
    let s0 = train::evaluate(&policy, 100, &mut rng);
    println!("=== 5-a-side standalone learning demo ===");
    println!(
        "scripted-vs-scripted:   goal_diff={:+.2}  (A {:.2} / B {:.2})",
        svs, sga, sgb
    );
    println!(
        "untrained-vs-scripted:  goal_diff={:+.2}  winrate={:.2}  (A {:.2} / B {:.2})  spacing={:.1}",
        s0.goal_diff, s0.winrate, s0.ga, s0.gb, s0.spacing
    );

    // Keep the untrained policy so we can later pick a display game where the
    // before/after contrast is representative (before loses, after wins) under
    // the SAME seed — the contrast is then the policy, not the dice.
    let untrained = policy.clone();

    if cfg.bc_games > 0 {
        let bc = train::behavior_clone_scripted(&mut policy, cfg.bc_games, cfg.bc_epochs, &mut rng);
        println!(
            "behavior-clone warm start: {} samples, loss {:.3}, teacher-match {:.0}%",
            bc.samples,
            bc.loss,
            bc.accuracy * 100.0
        );
    }

    println!(
        "training for {} iterations ({} games/iter, {} rollout workers, seed {})...\n",
        cfg.iters,
        cfg.games_per_iter,
        train::rollout_worker_count(cfg.games_per_iter),
        cfg.seed
    );

    let mut csv = String::new();
    csv.push_str("iter,avg_goal_diff,winrate,goals_a,goals_b,spacing,bunch,possession,pass_att,pass_cmp,pass_completion,pass_fwd,pass_lat,pass_back,shots,shots_scored,conversion,turnovers,balls_won,avg_reward,entropy,value_loss\n");
    let csv_row = |iter: usize,
                   s: &train::Stats,
                   avg_reward: f32,
                   ent: f32,
                   vloss: f32|
     -> String {
        format!(
            "{},{:.4},{:.4},{:.4},{:.4},{:.3},{:.4},{:.4},{:.3},{:.3},{:.4},{:.3},{:.3},{:.3},{:.3},{:.3},{:.4},{:.3},{:.3},{:.4},{:.4},{:.4}\n",
            iter, s.goal_diff, s.winrate, s.ga, s.gb, s.spacing, s.bunch, s.possession,
            s.pass_att, s.pass_cmp, s.pass_completion(), s.pass_fwd, s.pass_lat, s.pass_back,
            s.shots, s.shots_scored, s.conversion(), s.turnovers, s.wins_won, avg_reward, ent, vloss
        )
    };
    csv.push_str(&csv_row(0, &s0, 0.0, 0.0, 0.0));

    println!(
        "{:>5} | {:>10} | {:>8} | {:>6} {:>6} | {:>6} | {:>6} | {:>7} | {:>9}",
        "iter", "goal_diff", "winrate", "gA", "gB", "space", "gate", "entropy", "val_loss"
    );
    println!("{}", "-".repeat(80));

    // Keep the best behaviorally-valid checkpoint, not the last. If no checkpoint
    // clears the gates we still persist the best observed model, but the manifest
    // makes that fact explicit so stale demos cannot masquerade as proof.
    let mut best_any_policy = policy.clone();
    let mut best_any_quality = f32::NEG_INFINITY;
    let mut best_any_iter = 0usize;
    let mut best_gated: Option<(train::Policy, usize, f32)> = None;

    // Speed-warmup curriculum: fix the gear (v3 behavior) for the first stretch so
    // the action policy learns to attack before the speed policy adds variability.
    let speed_warmup = std::env::var("SPEED_WARMUP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| (cfg.iters / 2).clamp(1, 300));
    for it in 1..=cfg.iters {
        train::set_speed_frozen(it <= speed_warmup);
        let beta = train::ent_beta_at(it, cfg.iters);
        let stats = train::train_iter(&mut policy, cfg.games_per_iter, beta, &mut rng);

        if it % cfg.eval_every == 0 || it == cfg.iters {
            let st = train::evaluate(&policy, cfg.eval_games, &mut rng);
            let quality = checkpoint_quality(&st);
            let gated = checkpoint_clears_behavior_gates(&st);
            if quality > best_any_quality {
                best_any_quality = quality;
                best_any_policy = policy.clone();
                best_any_iter = it;
            }
            if gated && best_gated.as_ref().is_none_or(|(_, _, q)| quality > *q) {
                best_gated = Some((policy.clone(), it, quality));
            }
            println!(
                "{:>5} | {:>+10.3} | {:>8.3} | {:>6.2} {:>6.2} | sp {:>4.1} | {:>6} | pass {:>4.1} ({:.0}%) | shot {:>3.1}",
                it,
                st.goal_diff,
                st.winrate,
                st.ga,
                st.gb,
                st.spacing,
                if gated { "ok" } else { "watch" },
                st.pass_att,
                st.pass_completion() * 100.0,
                st.shots
            );
            csv.push_str(&csv_row(
                it,
                &st,
                stats.avg_reward,
                stats.entropy,
                stats.value_loss,
            ));
        }
    }

    let curve_path = cfg.out_dir.join("learning_curve.csv");
    write_atomic(&curve_path, &csv)?;
    println!("\nwrote {}", curve_path.display());

    let (policy, best_iter, best_quality, best_cleared_gates) = if let Some((
        policy,
        iter,
        quality,
    )) = best_gated
    {
        (policy, iter, quality, true)
    } else {
        eprintln!(
                "warning: no checkpoint cleared the behavior gates; persisting best observed checkpoint with manifest flag"
            );
        (best_any_policy, best_any_iter, best_any_quality, false)
    };
    println!(
        "best policy at iter {} (quality {:+.3}, gates={})",
        best_iter, best_quality, best_cleared_gates
    );

    // Everything below showcases the selected snapshot, not the collapsed final.
    // Persist the trained policy so `inspect` can load and trace it.
    let actor_path = cfg.out_dir.join("actor.txt");
    let critic_path = cfg.out_dir.join("critic.txt");
    let speedor_path = cfg.out_dir.join("speedor.txt");
    policy.actor.save(&actor_path)?;
    policy.speedor.save(&speedor_path)?;
    policy.critic.save(&critic_path)?;
    println!(
        "saved policy -> {}, {}",
        actor_path.display(),
        critic_path.display()
    );

    // Final richer evaluation.
    let f = train::evaluate(&policy, cfg.final_games, &mut rng);
    // append the FINAL row so the viz has the trained model's full stat line
    csv.push_str(&csv_row(best_iter, &f, 0.0, 0.0, 0.0));
    write_atomic(&curve_path, &csv)?;
    println!(
        "\nFINAL ({} games): goal_diff={:+.3}  winrate={:.3}  goals {:.2}-{:.2}  passes/game {:.1}  spacing={:.1}  bunch={:.0}%",
        cfg.final_games, f.goal_diff, f.winrate, f.ga, f.gb, f.pass_cmp, f.spacing, f.bunch * 100.0
    );
    println!(
        "  pass: {:.1} att, {:.0}% complete ({:.0}% fwd / {:.0}% lat / {:.0}% back) | shots {:.1}, {:.0}% converted | poss {:.0}% | turnovers {:.1} | balls won {:.1}",
        f.pass_att, f.pass_completion() * 100.0,
        if f.pass_att>0.0 {f.pass_fwd/f.pass_att*100.0} else {0.0},
        if f.pass_att>0.0 {f.pass_lat/f.pass_att*100.0} else {0.0},
        if f.pass_att>0.0 {f.pass_back/f.pass_att*100.0} else {0.0},
        f.shots, f.conversion()*100.0, f.possession*100.0, f.turnovers, f.wins_won
    );
    if f.goal_diff > 0.0 {
        println!(">>> The learned policy BEATS the scripted analytic baseline. <<<");
    }

    // Pick a display seed that showcases GOOD PLAY, not a blowout: the trained
    // side should win by a sensible margin while dominating possession and
    // stringing passes; the untrained side (same seed) should be clearly worse.
    let mut best_seed = None;
    let mut best_score = f32::NEG_INFINITY;
    for s in 1..=cfg.display_seed_max {
        let (bga, bgb, ba_poss, _) = game_stats(&untrained, s);
        let (aga, agb, aa_poss, a_pass) = game_stats(&policy, s);
        let before_m = bga as f32 - bgb as f32;
        let after_m = aga as f32 - agb as f32;
        // hard filter: trained wins by 1..=4, untrained doesn't win
        if before_m > 0.0 || !(1.0..=4.0).contains(&after_m) {
            continue;
        }
        // reward possession dominance + completed passes + possession swing
        let score =
            aa_poss as f32 * 0.02 + a_pass as f32 * 0.4 + (aa_poss as f32 - ba_poss as f32) * 0.01;
        if score > best_score {
            best_score = score;
            best_seed = Some(s);
        }
    }
    let display_seed_matched_filter = best_seed.is_some();
    let best_seed = best_seed.unwrap_or_else(|| {
        eprintln!(
            "warning: no display seed matched the showcase filter; using seed 1 as a fallback"
        );
        1
    });
    let (bga, bgb, bposs, _) = game_stats(&untrained, best_seed);
    let (aga, agb, aposs, apass) = game_stats(&policy, best_seed);
    println!(
        "display seed {}{}: before {}-{} (poss {})  ->  after {}-{} (poss {}, passes {})",
        best_seed,
        if display_seed_matched_filter {
            ""
        } else {
            " (fallback)"
        },
        bga,
        bgb,
        bposs,
        aga,
        agb,
        aposs,
        apass
    );
    let before_path = cfg.out_dir.join("match_before.json");
    let after_path = cfg.out_dir.join("match_after.json");
    record_match(&untrained, &mut Rng::new(best_seed), &before_path)?;
    record_match(&policy, &mut Rng::new(best_seed), &after_path)?;
    println!(
        "wrote {} + {} (visual demo traces)",
        before_path.display(),
        after_path.display()
    );

    let manifest_path = cfg.out_dir.join("run_manifest.json");
    write_run_manifest(
        &manifest_path,
        cfg,
        best_iter,
        best_quality,
        best_cleared_gates,
        best_seed,
        display_seed_matched_filter,
        (svs, sga, sgb),
        &s0,
        &f,
        &[
            &actor_path,
            &critic_path,
            &curve_path,
            &before_path,
            &after_path,
        ],
    )?;
    println!("wrote {}", manifest_path.display());
    Ok(())
}

fn checkpoint_quality(st: &train::Stats) -> f32 {
    let mut q = st.goal_diff.min(3.0) + (st.pass_cmp * 0.03).min(0.6) - 1.6 * st.bunch;
    if st.winrate < 0.55 {
        q -= 0.8;
    }
    if st.shots < 1.0 {
        q -= 1.2;
    }
    if st.pass_cmp < 2.0 {
        q -= 0.7;
    }
    if st.conversion() <= 0.0 {
        q -= 0.6;
    }
    if st.possession < 0.03 {
        q -= 0.4;
    }
    q
}

fn checkpoint_clears_behavior_gates(st: &train::Stats) -> bool {
    st.goal_diff > 0.0
        && st.winrate >= 0.55
        && st.pass_cmp >= 2.0
        && st.shots >= 1.0
        && st.conversion() > 0.0
        && st.possession >= 0.03
        && st.bunch <= 0.50
}

/// Play a greedy game vs scripted at `seed`; return
/// (goals_a, goals_b, frames_A_in_possession, pass_completions_A).
fn game_stats(policy: &train::Policy, seed: u64) -> (u32, u32, u32, u32) {
    let mut rng = Rng::new(seed);
    let mut w = World::new();
    let mut aposs = 0u32;
    let mut passes = 0u32;
    for _ in 0..STEPS {
        let mut act_a = [A_STAY; N];
        for i in 1..N {
            act_a[i] = policy.act_greedy_world(&w, Team::A, i);
        }
        let act_b = w.scripted_actions(Team::B);
        w.step(&act_a, &act_b, &mut rng);
        if matches!(w.owner, Some(o) if matches!(o.team, Team::A)) {
            aposs += 1;
        }
        if w.ev_pass_completed_a {
            passes += 1;
        }
    }
    (w.goals_a, w.goals_b, aposs, passes)
}

/// Play one greedy game and dump per-tick positions to a compact JSON for the
/// HTML viewer. Hand-rolled JSON — no serde, keeping the crate dependency-free.
fn record_match(policy: &train::Policy, rng: &mut Rng, path: &Path) -> AppResult<()> {
    let mut w = World::new();
    let mut frames = String::new();
    frames.push('[');
    for t in 0..RECORD_STEPS {
        if t > 0 {
            frames.push(',');
        }
        let mut act_a = [A_STAY; N];
        for i in 1..N {
            act_a[i] = policy.act_greedy_world(&w, Team::A, i);
        }
        let act_b = w.scripted_actions(Team::B);
        w.step(&act_a, &act_b, rng);

        let (owner_code, owner_team, owner_idx) = match w.owner {
            Some(o) if matches!(o.team, Team::A) => (0i32, 0i32, o.idx as i32),
            Some(o) => (1i32, 1i32, o.idx as i32),
            None => (-1i32, -1i32, -1i32),
        };
        let _ = owner_team;
        frames.push_str("{\"a\":[");
        for i in 0..N {
            if i > 0 {
                frames.push(',');
            }
            let _ = write!(frames, "[{:.1},{:.1}]", w.a[i].pos.x, w.a[i].pos.y);
        }
        frames.push_str("],\"b\":[");
        for i in 0..N {
            if i > 0 {
                frames.push(',');
            }
            let _ = write!(frames, "[{:.1},{:.1}]", w.b[i].pos.x, w.b[i].pos.y);
        }
        let _ = write!(
            frames,
            "],\"ball\":[{:.1},{:.1}],\"own\":{},\"oi\":{},\"ga\":{},\"gb\":{}}}",
            w.ball.x, w.ball.y, owner_code, owner_idx, w.goals_a, w.goals_b
        );
    }
    frames.push(']');

    let meta = format!(
        "{{\"field\":[{},{}],\"goal_half\":{},\"n\":{},\"hz\":{},\"frames\":{}}}",
        FIELD_L, FIELD_W, GOAL_HALF, N, HZ, frames
    );
    write_atomic(path, &meta)?;
    Ok(())
}

fn write_atomic(path: &Path, contents: &str) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file_name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("artifact");
    let tmp = path.with_file_name(format!(".{file_name}.tmp.{}", std::process::id()));
    fs::write(&tmp, contents)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_run_manifest(
    path: &Path,
    cfg: &RunConfig,
    best_iter: usize,
    best_quality: f32,
    best_cleared_gates: bool,
    display_seed: u64,
    display_seed_matched_filter: bool,
    scripted: (f32, f32, f32),
    untrained: &train::Stats,
    final_stats: &train::Stats,
    artifacts: &[&PathBuf],
) -> AppResult<()> {
    let created = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let mut artifact_json = String::new();
    artifact_json.push('[');
    for (i, artifact) in artifacts.iter().enumerate() {
        if i > 0 {
            artifact_json.push(',');
        }
        let checksum = checksum_file(artifact)?;
        let bytes = fs::metadata(artifact)?.len();
        let _ = write!(
            artifact_json,
            "{{\"path\":{},\"bytes\":{},\"fnv1a64\":\"{}\"}}",
            json_string(&artifact.display().to_string()),
            bytes,
            checksum
        );
    }
    artifact_json.push(']');

    let spacing_w = std::env::var("SPACING_W").ok();
    let manifest = format!(
        concat!(
            "{{\n",
            "  \"created_unix_seconds\": {},\n",
            "  \"git_commit\": {},\n",
            "  \"config\": {{\"iters\":{},\"seed\":{},\"games_per_iter\":{},\"bc_games\":{},\"bc_epochs\":{},\"eval_every\":{},\"eval_games\":{},\"final_games\":{},\"display_seed_max\":{},\"out_dir\":{}}},\n",
            "  \"env\": {{\"SPACING_W\": {}}},\n",
            "  \"selection\": {{\"best_iter\":{},\"best_quality\":{:.6},\"best_cleared_hardening_gates\":{},\"display_seed\":{},\"display_seed_matched_filter\":{}}},\n",
            "  \"scripted_vs_scripted\": {{\"goal_diff\":{:.6},\"goals_a\":{:.6},\"goals_b\":{:.6}}},\n",
            "  \"untrained\": {},\n",
            "  \"final\": {},\n",
            "  \"artifacts\": {}\n",
            "}}\n"
        ),
        created,
        json_string(&git_commit()),
        cfg.iters,
        cfg.seed,
        cfg.games_per_iter,
        cfg.bc_games,
        cfg.bc_epochs,
        cfg.eval_every,
        cfg.eval_games,
        cfg.final_games,
        cfg.display_seed_max,
        json_string(&cfg.out_dir.display().to_string()),
        spacing_w.as_deref().map(json_string).unwrap_or_else(|| "null".to_string()),
        best_iter,
        best_quality,
        best_cleared_gates,
        display_seed,
        display_seed_matched_filter,
        scripted.0,
        scripted.1,
        scripted.2,
        stats_json(untrained),
        stats_json(final_stats),
        artifact_json
    );
    write_atomic(path, &manifest)?;
    Ok(())
}

fn stats_json(s: &train::Stats) -> String {
    format!(
        "{{\"goal_diff\":{:.6},\"winrate\":{:.6},\"goals_a\":{:.6},\"goals_b\":{:.6},\"spacing\":{:.6},\"bunch\":{:.6},\"possession\":{:.6},\"pass_att\":{:.6},\"pass_cmp\":{:.6},\"pass_completion\":{:.6},\"pass_fwd\":{:.6},\"pass_lat\":{:.6},\"pass_back\":{:.6},\"shots\":{:.6},\"shots_scored\":{:.6},\"conversion\":{:.6},\"turnovers\":{:.6},\"balls_won\":{:.6}}}",
        s.goal_diff,
        s.winrate,
        s.ga,
        s.gb,
        s.spacing,
        s.bunch,
        s.possession,
        s.pass_att,
        s.pass_cmp,
        s.pass_completion(),
        s.pass_fwd,
        s.pass_lat,
        s.pass_back,
        s.shots,
        s.shots_scored,
        s.conversion(),
        s.turnovers,
        s.wins_won
    )
}

fn checksum_file(path: &Path) -> AppResult<String> {
    let bytes = fs::read(path)?;
    let mut hash = 0xcbf29ce484222325u64;
    for b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    Ok(format!("{hash:016x}"))
}

fn git_commit() -> String {
    Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()
        .and_then(|out| {
            if out.status.success() {
                Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
            } else {
                None
            }
        })
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

fn json_string(s: &str) -> String {
    let mut out = String::from("\"");
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
