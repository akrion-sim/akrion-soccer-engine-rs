//! Ad-hoc measurement harness: run full matches and tally how often each
//! offensive mechanic is actually CHOSEN by players, to verify the five
//! offensive strategies are utilizable (not just present-but-unreachable).
//!
//! Run: cargo run --release --bin measure_offense [ticks] [seeds]

use std::collections::BTreeMap;

use soccer_engine::des::general::soccer::{MatchConfig, SoccerMatch};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let ticks: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(30_000);
    let seeds: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3);

    // The mechanics we care about (chosen-action labels) + a few neighbors.
    let watched = [
        "overlap-run",
        "wall-pass",
        "scoop-pass",
        "killer-pass",
        "run-in-behind",
        "wide-outlet",
        "check-to-ball",
        "flank-cross",
        "support-shape",
        "support-roam",
        "dribble",
        "shoot",
        "pass",
        "clear",
    ];

    let mut totals: BTreeMap<String, u64> = BTreeMap::new();
    let mut one_two_engagements: u64 = 0;
    let mut decision_ticks: u64 = 0; // player-ticks with a recorded decision
    let mut total_player_ticks: u64 = 0;
    let mut on_ball_ticks: u64 = 0; // has_ball decisions
    let mut threaded_available_ticks: u64 = 0; // has_ball AND threaded_goal_pass_available
    let mut threaded_avail_chose_killer: u64 = 0; // available AND chose killer-pass

    for s in 0..seeds {
        let config = MatchConfig {
            seed: 0x5EED_0000u32.wrapping_add(s as u32),
            ..MatchConfig::default()
        };
        let mut sim = SoccerMatch::default_11v11(config);
        // Construction fingerprint: if two processes print different values here, the
        // nondeterminism is in match construction; if identical but play diverges, it's
        // in run_time_step.
        let ctor_fp: f64 = sim
            .players
            .iter()
            .map(|p| p.position.x * 7.0 + p.position.y * 13.0)
            .sum();
        eprintln!("seed {s}: ctor_fp={ctor_fp:.6}");
        let mut prev_one_two = vec![false; sim.players.len()];

        let bisect = std::env::var("MO_BISECT").is_ok();
        for t in 0..ticks {
            sim.run_time_step();
            if bisect && s == 0 {
                // Per-tick state + action fingerprint, to diff two runs and find the
                // FIRST diverging tick (and whether it's positions or chosen actions).
                let mut pos_h: u64 = 1469598103934665603;
                let mut act = String::new();
                for p in sim.players.iter() {
                    let bits = (p.position.x.to_bits()) ^ (p.position.y.to_bits().rotate_left(17));
                    pos_h ^= bits;
                    pos_h = pos_h.wrapping_mul(1099511628211);
                    act.push('|');
                    if let Some(d) = p.last_decision.as_ref() {
                        act.push_str(&d.action);
                    }
                }
                println!("T{} pos={:016x} act={}", t + 1, pos_h, act);
            }
            if let Ok(dt) = std::env::var("MO_DUMP_TICK") {
                if s == 0 && (t + 1) == dt.parse::<u64>().unwrap_or(0) {
                    for (i, p) in sim.players.iter().enumerate() {
                        println!(
                            "P{i} x={:.17e} y={:.17e} vx={:.17e} vy={:.17e}",
                            p.position.x, p.position.y, p.velocity.x, p.velocity.y
                        );
                    }
                }
            }
            for (i, p) in sim.players.iter().enumerate() {
                total_player_ticks += 1;
                if let Some(d) = p.last_decision.as_ref() {
                    decision_ticks += 1;
                    *totals.entry(d.action.clone()).or_insert(0) += 1;
                    if d.observation.has_ball {
                        on_ball_ticks += 1;
                        if d.observation.threaded_goal_pass_available {
                            threaded_available_ticks += 1;
                            if d.action == "killer-pass" {
                                threaded_avail_chose_killer += 1;
                            }
                        }
                    }
                }
                let now = p.one_two.is_some();
                if now && !prev_one_two[i] {
                    one_two_engagements += 1;
                }
                prev_one_two[i] = now;
            }
        }
        eprintln!(
            "seed {s}: final {}-{}",
            sim.score_home, sim.score_away
        );
    }

    println!("\n===== OFFENSE UTILIZATION ({seeds} matches x {ticks} ticks) =====");
    println!("total player-ticks: {total_player_ticks}, with-decision: {decision_ticks}");
    println!("one-two engagements (None->Some rising edges): {one_two_engagements}");
    println!("on-ball decisions: {on_ball_ticks}");
    println!(
        "  threaded-pass AVAILABLE: {threaded_available_ticks} ({:.3}% of on-ball)",
        100.0 * threaded_available_ticks as f64 / on_ball_ticks.max(1) as f64
    );
    println!(
        "  of those, CHOSE killer-pass: {threaded_avail_chose_killer} ({:.2}% of available)\n",
        100.0 * threaded_avail_chose_killer as f64 / threaded_available_ticks.max(1) as f64
    );

    println!("--- WATCHED MECHANICS (chosen-action player-ticks) ---");
    for w in watched {
        let count: u64 = totals
            .iter()
            .filter(|(k, _)| k.as_str() == w || k.starts_with(w))
            .map(|(_, v)| *v)
            .sum();
        let pct = if decision_ticks > 0 {
            100.0 * count as f64 / decision_ticks as f64
        } else {
            0.0
        };
        println!("  {w:<16} {count:>10}  ({pct:>6.3}% of decisions)");
    }

    println!("\n--- FULL ACTION HISTOGRAM (all labels, desc) ---");
    let mut all: Vec<(&String, &u64)> = totals.iter().collect();
    all.sort_by(|a, b| b.1.cmp(a.1));
    for (label, count) in all {
        let pct = 100.0 * **&count as f64 / decision_ticks.max(1) as f64;
        println!("  {label:<32} {count:>10}  ({pct:>6.3}%)");
    }
}
