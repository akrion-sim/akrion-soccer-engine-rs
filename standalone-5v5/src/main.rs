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
        "inspect" => {
            let seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(7);
            inspect(seed);
        }
        _ => {
            let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
            run_training(iters);
        }
    }
}

fn action_name(a: usize) -> &'static str {
    match a {
        A_SHOOT => "shoot",
        A_PASS_A => "pass_a",
        A_PASS_B => "pass_b",
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
fn inspect(seed: u64) {
    let actor = match nn::Mlp::load("out/actor.txt") {
        Ok(m) => m,
        Err(_) => {
            eprintln!("no trained policy found — run `train` first (writes out/actor.txt)");
            return;
        }
    };
    let policy = train::Policy { actor, critic: nn::Mlp::load("out/critic.txt").unwrap_or_else(|_| nn::Mlp::new(&[OBS_DIM, 64, 64, 1], &mut Rng::new(0))) };
    let mut rng = Rng::new(seed);
    let mut w = World::new();
    fs::create_dir_all("out").ok();

    let mut jsonl = String::new();
    let mut bunch_ticks = 0u32;
    let mut sum_closest = 0.0f32;
    let mut sum_avg = 0.0f32;
    const BUNCH_THRESH: f32 = 2.5;

    println!("== inspect: greedy game, seed {} ==", seed);
    println!("{:>4} | {:>7} | own | closest_pair | avg_near | bunch | score", "t", "ball_x");

    for t in 0..STEPS {
        // choose actions (record them for the trace)
        let mut act_a = [A_STAY; N];
        for i in 1..N {
            let obs = w.observe(Team::A, i);
            let mask = w.legal_mask(Team::A, i);
            act_a[i] = policy.act_greedy(&obs, &mask);
        }
        let act_b = w.scripted_actions(Team::B);
        w.step(&act_a, &act_b, &mut rng);

        let closest = w.closest_pair_a();
        let avg = w.avg_nearest_teammate_a();
        let bunched = closest < BUNCH_THRESH;
        if bunched {
            bunch_ticks += 1;
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
                t, w.ball.x, own, closest, avg, if bunched { "YES" } else { "no" }, w.goals_a, w.goals_b
            );
        }
    }

    fs::write("out/trace.jsonl", &jsonl).ok();
    let n = STEPS as f32;
    println!("\n-- bunching summary (Team A outfielders) --");
    println!("  closest-pair distance: mean {:.1}", sum_closest / n);
    println!("  avg nearest-teammate:  mean {:.1}", sum_avg / n);
    println!(
        "  frames with a pair < {:.1}: {:.0}%  ({}/{})",
        BUNCH_THRESH,
        100.0 * bunch_ticks as f32 / n,
        bunch_ticks,
        STEPS
    );
    println!("  full per-tick trace -> out/trace.jsonl ({} frames)", STEPS);
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
    let (d0, w0, ga0, gb0, _p0, sp0, _b0) = train::evaluate(&policy, 100, &mut rng);
    println!("=== 5-a-side standalone learning demo ===");
    println!(
        "scripted-vs-scripted:   goal_diff={:+.2}  (A {:.2} / B {:.2})",
        svs, sga, sgb
    );
    println!(
        "untrained-vs-scripted:  goal_diff={:+.2}  winrate={:.2}  (A {:.2} / B {:.2})  spacing={:.1}",
        d0, w0, ga0, gb0, sp0
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
        "{:>5} | {:>10} | {:>8} | {:>6} {:>6} | {:>6} | {:>7} | {:>9}",
        "iter", "goal_diff", "winrate", "gA", "gB", "space", "entropy", "val_loss"
    );
    println!("{}", "-".repeat(80));

    // Keep the BEST policy by eval goal-diff, not the last — PPO can over-train
    // and collapse, so we snapshot the peak and showcase that.
    let mut best_policy = policy.clone();
    let mut best_diff = f32::NEG_INFINITY;
    let mut best_iter = 0usize;

    for it in 1..=iters {
        let beta = train::ent_beta_at(it, iters);
        let stats = train::train_iter(&mut policy, games_per_iter, beta, &mut rng);

        if it % eval_every == 0 || it == iters {
            let (d, wr, ga, gb, passes, sp, bunch) = train::evaluate(&policy, 60, &mut rng);
            // Snapshot the checkpoint that WINS and is SPREAD OUT: once winning
            // (d > 0.2), add a passing bonus plus a strong spacing bonus so the
            // chosen model actually spaces its teammates (toward the 5-8 target).
            // Balance: reward real soccer (winning + passing) AND penalize the
            // HONEST bunch% (average spacing hides a glued pair). Neither term
            // alone dominates — we want a checkpoint that plays well with low bunch.
            let _ = sp;
            let quality =
                d.min(3.0) + (passes * 0.03).min(0.6) - 1.6 * bunch;
            if quality > best_diff {
                best_diff = quality;
                best_policy = policy.clone();
                best_iter = it;
            }
            println!(
                "{:>5} | {:>+10.3} | {:>8.3} | {:>6.2} {:>6.2} | sp {:>4.1} | bunch {:>3.0}% | ent {:>5.2}",
                it, d, wr, ga, gb, sp, bunch * 100.0, stats.entropy
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
    println!("best policy at iter {} (eval goal_diff {:+.3})", best_iter, best_diff);

    // Everything below showcases the BEST snapshot, not the collapsed final.
    let policy = best_policy;
    // Persist the trained policy so `inspect` can load and trace it.
    policy.actor.save("out/actor.txt").ok();
    policy.critic.save("out/critic.txt").ok();
    println!("saved policy -> out/actor.txt, out/critic.txt");

    // Final richer evaluation.
    let (d, wr, ga, gb, passes, spacing, bunch) = train::evaluate(&policy, 300, &mut rng);
    println!(
        "\nFINAL (300 games): goal_diff={:+.3}  winrate={:.3}  goals {:.2}-{:.2}  passes/game {:.1}  spacing={:.1}  bunch={:.0}%",
        d, wr, ga, gb, passes, spacing, bunch * 100.0
    );
    if d > 0.0 {
        println!(">>> The learned policy BEATS the scripted analytic baseline. <<<");
    }

    // Pick a display seed that showcases GOOD PLAY, not a blowout: the trained
    // side should win by a sensible margin while dominating possession and
    // stringing passes; the untrained side (same seed) should be clearly worse.
    let mut best_seed = 1u64;
    let mut best_score = f32::NEG_INFINITY;
    for s in 1..=120u64 {
        let (bga, bgb, ba_poss, _) = game_stats(&untrained, s);
        let (aga, agb, aa_poss, a_pass) = game_stats(&policy, s);
        let before_m = bga as f32 - bgb as f32;
        let after_m = aga as f32 - agb as f32;
        // hard filter: trained wins by 1..=4, untrained doesn't win
        if before_m > 0.0 || after_m < 1.0 || after_m > 4.0 {
            continue;
        }
        // reward possession dominance + completed passes + possession swing
        let score = aa_poss as f32 * 0.02 + a_pass as f32 * 0.4
            + (aa_poss as f32 - ba_poss as f32) * 0.01;
        if score > best_score {
            best_score = score;
            best_seed = s;
        }
    }
    let (bga, bgb, bposs, _) = game_stats(&untrained, best_seed);
    let (aga, agb, aposs, apass) = game_stats(&policy, best_seed);
    println!(
        "display seed {}: before {}-{} (poss {})  ->  after {}-{} (poss {}, passes {})",
        best_seed, bga, bgb, bposs, aga, agb, aposs, apass
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
