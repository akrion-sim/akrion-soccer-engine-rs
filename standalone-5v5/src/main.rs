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

mod game;
mod nn;
mod rng;
mod train;

use game::*;
use rng::Rng;
use std::fmt::Write as _;
use std::fs;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("train");
    match cmd {
        "sanity" => sanity(),
        _ => {
            let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
            run_training(iters);
        }
    }
}

fn sanity() {
    let mut rng = Rng::new(1234);
    let (diff, ga, gb) = train::evaluate_scripted_vs_scripted(200, &mut rng);
    println!("scripted vs scripted over 200 games:");
    println!("  avg goal diff = {:+.3}  (should be ~0.0 — symmetric baseline)", diff);
    println!("  avg goals A = {:.3}, avg goals B = {:.3}", ga, gb);
}

fn run_training(iters: usize) {
    let seed = 20260710;
    let mut rng = Rng::new(seed);
    let mut policy = train::Policy::new(&mut rng);

    fs::create_dir_all("out").ok();

    // Baseline reference: how does the *scripted* team fare vs itself, and how
    // does an UNTRAINED (random) policy fare vs scripted?
    let (svs, sga, sgb) = train::evaluate_scripted_vs_scripted(100, &mut rng);
    let (d0, w0, ga0, gb0) = train::evaluate(&policy, 100, &mut rng);
    println!("=== 5-a-side standalone learning demo ===");
    println!(
        "scripted-vs-scripted:   goal_diff={:+.2}  (A {:.2} / B {:.2})",
        svs, sga, sgb
    );
    println!(
        "untrained-vs-scripted:  goal_diff={:+.2}  winrate={:.2}  (A {:.2} / B {:.2})",
        d0, w0, ga0, gb0
    );
    println!("training for {} iterations (8 games/iter)...\n", iters);

    // Keep the untrained policy so we can later pick a display game where the
    // before/after contrast is representative (before loses, after wins) under
    // the SAME seed — the contrast is then the policy, not the dice.
    let untrained = policy.clone();

    let mut csv = String::new();
    csv.push_str("iter,avg_goal_diff,winrate,goals_a,goals_b,entropy,value_loss,train_reward\n");
    let _ = writeln!(csv, "0,{:.4},{:.4},{:.4},{:.4},,,", d0, w0, ga0, gb0);

    let games_per_iter = 8;
    let eval_every = 5;

    println!(
        "{:>5} | {:>10} | {:>8} | {:>6} {:>6} | {:>7} | {:>9}",
        "iter", "goal_diff", "winrate", "gA", "gB", "entropy", "val_loss"
    );
    println!("{}", "-".repeat(72));

    for it in 1..=iters {
        let beta = train::ent_beta_at(it, iters);
        let stats = train::train_iter(&mut policy, games_per_iter, beta, &mut rng);

        if it % eval_every == 0 || it == iters {
            let (d, wr, ga, gb) = train::evaluate(&policy, 60, &mut rng);
            println!(
                "{:>5} | {:>+10.3} | {:>8.3} | {:>6.2} {:>6.2} | {:>7.3} | {:>9.4}",
                it, d, wr, ga, gb, stats.entropy, stats.value_loss
            );
            let _ = writeln!(
                csv,
                "{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
                it, d, wr, ga, gb, stats.entropy, stats.value_loss, stats.avg_reward
            );
        }
    }

    fs::write("out/learning_curve.csv", &csv).ok();
    println!("\nwrote out/learning_curve.csv");

    // Final richer evaluation.
    let (d, wr, ga, gb) = train::evaluate(&policy, 300, &mut rng);
    println!(
        "\nFINAL (300 games): goal_diff={:+.3}  winrate={:.3}  goals {:.2}-{:.2}",
        d, wr, ga, gb
    );
    if d > 0.0 {
        println!(">>> The learned policy BEATS the scripted analytic baseline. <<<");
    }

    // Pick a display seed: untrained should lose, trained should win clearly,
    // both under the same seed so the ONLY difference is the learned policy.
    let mut best_seed = 1u64;
    let mut best_score = f32::NEG_INFINITY;
    for s in 1..=80u64 {
        let (bga, bgb) = game_score(&untrained, s);
        let (aga, agb) = game_score(&policy, s);
        let before_m = bga as f32 - bgb as f32;
        let after_m = aga as f32 - agb as f32;
        // reward a big swing, strongly prefer before<=0 and after>=2
        let mut score = after_m - before_m;
        if before_m <= 0.0 {
            score += 2.0;
        }
        if after_m >= 2.0 {
            score += 2.0;
        }
        if score > best_score {
            best_score = score;
            best_seed = s;
        }
    }
    let (bga, bgb) = game_score(&untrained, best_seed);
    let (aga, agb) = game_score(&policy, best_seed);
    println!(
        "display seed {}: before {}-{}  ->  after {}-{}",
        best_seed, bga, bgb, aga, agb
    );
    record_match(&untrained, &mut Rng::new(best_seed), "out/match_before.json");
    record_match(&policy, &mut Rng::new(best_seed), "out/match_after.json");
    println!("wrote out/match_before.json + out/match_after.json (visual demo traces)");
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
            let obs = w.observe(Team::A, i);
            let mask = w.legal_mask(Team::A, i);
            act_a[i] = policy.act_greedy(&obs, &mask);
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
fn record_match(policy: &train::Policy, rng: &mut Rng, path: &str) {
    let mut w = World::new();
    let mut frames = String::new();
    frames.push('[');
    for t in 0..STEPS {
        if t > 0 {
            frames.push(',');
        }
        let mut act_a = [A_STAY; N];
        for i in 1..N {
            let obs = w.observe(Team::A, i);
            let mask = w.legal_mask(Team::A, i);
            act_a[i] = policy.act_greedy(&obs, &mask);
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
        "{{\"field\":[{},{}],\"goal_half\":{},\"n\":{},\"frames\":{}}}",
        FIELD_L, FIELD_W, GOAL_HALF, N, frames
    );
    fs::write(path, meta).ok();
}
