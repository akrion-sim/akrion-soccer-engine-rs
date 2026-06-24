//! Ad-hoc measurement harness: run full matches and tally how often each
//! offensive mechanic is actually CHOSEN by players, to verify the five
//! offensive strategies are utilizable (not just present-but-unreachable).
//!
//! Run: cargo run --release --bin measure_offense [ticks] [seeds]

use std::collections::BTreeMap;

use soccer_engine::des::general::soccer::{
    enable_deterministic_formation_lp, MatchConfig, SoccerMatch,
};

fn main() {
    // Reproducible A/B: route the formation LP through the deterministic simplex so the same
    // seed yields byte-identical matches (the default Clarabel solver is per-process
    // nondeterministic). This is a headless harness, so the deterministic solve's latency
    // is irrelevant.
    enable_deterministic_formation_lp();
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
        let mut prev_one_two = vec![false; sim.players.len()];

        for _ in 0..ticks {
            sim.run_time_step();
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
