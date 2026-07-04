//! Ad-hoc A/B harness for **formation lane discipline**. Runs full matches and
//! measures how tightly outfield players hold their assigned vertical channel —
//! the thing that "messy formation" is the absence of.
//!
//! Compare the legacy vs. centralized path on the SAME seeds:
//!   cargo run --release --bin measure_lane_discipline                      # gate off (legacy)
//!   DD_SOCCER_ENABLE_LANE_DISCIPLINE_V2=1 \
//!     cargo run --release --bin measure_lane_discipline                    # gate on  (v2)
//!
//! Metrics (outfield players only; lower drift / bunching = tidier shape, while
//! team width must NOT collapse):
//!   * lateral drift   — mean |x - home_x| over player-ticks
//!   * out-of-band     — % of player-ticks more than 15yd off the home channel
//!   * lane bunching   — mean over ticks of the most-occupied of the 12 lanes
//!   * team width      — mean per-tick (max_x - min_x) of the outfield (guard rail)

use soccer_engine::des::general::soccer::{
    enable_deterministic_formation_lp, MatchConfig, PlayerRole, SoccerMatch,
};

const LANES: usize = 12;

fn main() {
    // Reproducible A/B: deterministic formation LP so the same seed is byte-identical
    // between runs (the default Clarabel solve is per-process nondeterministic).
    enable_deterministic_formation_lp();
    let args: Vec<String> = std::env::args().collect();
    let ticks: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let seeds: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);

    let gate = std::env::var("DD_SOCCER_ENABLE_LANE_DISCIPLINE_V2")
        .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false);

    let mut drift_sum = 0.0_f64; // sum |x - home_x| over outfield player-ticks
    let mut player_ticks = 0_u64;
    let mut out_of_band = 0_u64; // player-ticks > 15yd off home channel
    let mut bunch_sum = 0.0_f64; // sum of (max lane occupancy) over team-ticks
    let mut team_ticks = 0_u64;
    let mut width_sum = 0.0_f64; // sum of per-team-tick (max_x - min_x)

    for s in 0..seeds {
        let config = MatchConfig {
            seed: 0x5EED_0000u32.wrapping_add(s as u32),
            ..MatchConfig::default()
        };
        let field_width = config.field_width_yards.max(1.0);
        let lane_width = field_width / LANES as f64;
        let mut sim = SoccerMatch::default_11v11(config);

        for _ in 0..ticks {
            sim.run_time_step();
            for team in [0u8, 1u8] {
                let mut occ = [0u32; LANES];
                let mut min_x = f64::INFINITY;
                let mut max_x = f64::NEG_INFINITY;
                let mut any = false;
                for p in sim.players.iter() {
                    if p.role == PlayerRole::Goalkeeper || p.team as u8 != team {
                        continue;
                    }
                    any = true;
                    let drift = (p.position.x - p.home_position.x).abs();
                    drift_sum += drift;
                    player_ticks += 1;
                    if drift > 15.0 {
                        out_of_band += 1;
                    }
                    let lane = ((p.position.x.max(0.0) / lane_width).floor() as usize)
                        .min(LANES - 1);
                    occ[lane] += 1;
                    min_x = min_x.min(p.position.x);
                    max_x = max_x.max(p.position.x);
                }
                if any {
                    bunch_sum += *occ.iter().max().unwrap() as f64;
                    width_sum += max_x - min_x;
                    team_ticks += 1;
                }
            }
        }
        eprintln!("seed {s}: final {}-{}", sim.score_home, sim.score_away);
    }

    let pt = player_ticks.max(1) as f64;
    let tt = team_ticks.max(1) as f64;
    println!(
        "\n===== LANE DISCIPLINE ({seeds} matches x {ticks} ticks) — v2 gate {} =====",
        if gate { "ON" } else { "off" }
    );
    println!("outfield player-ticks: {player_ticks}");
    println!("lateral drift   |x-home_x|  : {:.3} yd  (lower = tighter)", drift_sum / pt);
    println!(
        "out-of-band     >15yd off home: {:.2}%      (lower = tighter)",
        100.0 * out_of_band as f64 / pt
    );
    println!("lane bunching   max lane occ : {:.3} / 11 (lower = less bunched)", bunch_sum / tt);
    println!("team width      max_x-min_x  : {:.3} yd  (must NOT collapse)", width_sum / tt);
}
