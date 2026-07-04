//! Headless A/B harness for the back-four SHAPE rules made default-ON in this change:
//!   1. one wing-back released forward at a time (the other three held),
//!   2. the back-four AVERAGE never sits more than ~5yd into the opponent half,
//!   3. wide defenders (wing-backs) stick to their wide lane (low lateral drift).
//!
//! The gates are read once per process (OnceLock), so the A/B is run as two PROCESSES on
//! the SAME seeds:
//!   cargo run --release --bin measure_wingback_shape                         # features ON (default)
//!   DD_SOCCER_DISABLE_SINGLE_WINGBACK_OVERLAP=1 \
//!   DD_SOCCER_DISABLE_WINGBACK_LANE_STICKINESS=1 \
//!   DD_SOCCER_DISABLE_BACK_FOUR_RESYNC_GRACE=1 \
//!     cargo run --release --bin measure_wingback_shape                       # features OFF
//!
//! Metrics (per defending/attacking team, over full matches):
//!   * both-WB-forward %  — % of IN-POSSESSION team-ticks where BOTH wide defenders are
//!     > 4yd (the run margin) ahead of the central-defender mean. Rule 1 should drive ~0.
//!   * back-four avg into opp half — mean / max yards the four-defender average sits past
//!     halfway, and the % of team-ticks past 5yd. Rule 2 should keep max/▒% near the cap.
//!   * wide-defender lateral drift |x-home_x| — mean over wide-defender-ticks. Rule 3
//!     (stickiness) should lower it without collapsing team width.

use soccer_engine::des::general::soccer::{
    enable_deterministic_formation_lp, MatchConfig, PlayerRole, SoccerMatch, Team,
};

/// Margin (yd) a wide defender must be ahead of the central-defender mean to count as
/// "released forward" — matches the engine's `BACK_LINE_WINGBACK_RUN_MARGIN_YARDS`.
const RUN_MARGIN_YARDS: f64 = 4.0;
/// The into-opponent-half allowance the back-four average should respect.
const OPP_HALF_CAP_YARDS: f64 = 5.0;

fn env_on(name: &str) -> bool {
    std::env::var(name)
        .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}

fn main() {
    enable_deterministic_formation_lp();
    let args: Vec<String> = std::env::args().collect();
    let ticks: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(8_000);
    let seeds: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);

    // The default is ON; a feature is OFF only when its DISABLE gate is set.
    let single = !env_on("DD_SOCCER_DISABLE_SINGLE_WINGBACK_OVERLAP");
    let stick = !env_on("DD_SOCCER_DISABLE_WINGBACK_LANE_STICKINESS");
    let grace = !env_on("DD_SOCCER_DISABLE_BACK_FOUR_RESYNC_GRACE");

    // Rule 1: both wing-backs forward, among in-possession team-ticks.
    let mut poss_team_ticks = 0_u64;
    let mut both_wb_forward = 0_u64;
    let mut one_wb_forward = 0_u64;
    // Rule 2: back-four average into the opponent half.
    let mut into_opp_sum = 0.0_f64;
    let mut into_opp_max = f64::NEG_INFINITY;
    let mut line_team_ticks = 0_u64;
    let mut into_opp_over_cap = 0_u64;
    // Rule 3: wide-defender lateral drift + team width guard rail, split by phase. Defending
    // is where "hold your channel" matters and the in-possession width-opening confound is
    // absent — the cleanest read on lane stickiness.
    let mut wb_drift_sum = 0.0_f64;
    let mut wb_ticks = 0_u64;
    let mut wb_drift_def_sum = 0.0_f64; // wide defender, team NOT in possession
    let mut wb_ticks_def = 0_u64;
    let mut wb_drift_poss_sum = 0.0_f64; // wide defender, team in possession
    let mut wb_ticks_poss = 0_u64;
    let mut width_sum = 0.0_f64;
    let mut width_ticks = 0_u64;

    for s in 0..seeds {
        let config = MatchConfig {
            seed: 0x5EED_0000u32.wrapping_add(s as u32),
            ..MatchConfig::default()
        };
        let field_width = config.field_width_yards.max(1.0);
        let half_length = config.field_length_yards * 0.5;
        let mut sim = SoccerMatch::default_11v11(config);
        let wide_lo = field_width * 0.28;
        let wide_hi = field_width * 0.72;

        for _ in 0..ticks {
            sim.run_time_step();
            let holder_team = sim
                .ball
                .holder
                .and_then(|id| sim.players.iter().find(|p| p.id == id))
                .map(|p| p.team);

            for team in [Team::Home, Team::Away] {
                let attack = team.attack_dir();
                let defenders: Vec<&_> = sim
                    .players
                    .iter()
                    .filter(|p| p.team == team && p.role == PlayerRole::Defender)
                    .collect();
                if defenders.len() < 3 {
                    continue;
                }
                let mut central_y = Vec::new();
                let mut wide = Vec::new();
                let mut min_x = f64::INFINITY;
                let mut max_x = f64::NEG_INFINITY;
                for p in &defenders {
                    if p.home_position.x < wide_lo || p.home_position.x > wide_hi {
                        wide.push(*p);
                    } else {
                        central_y.push(p.position.y);
                    }
                    min_x = min_x.min(p.position.x);
                    max_x = max_x.max(p.position.x);
                }

                // Rule 2: four-defender average depth past halfway (in attack frame).
                let avg_y = defenders.iter().map(|p| p.position.y).sum::<f64>()
                    / defenders.len() as f64;
                let into_opp = (avg_y - half_length) * attack;
                into_opp_sum += into_opp;
                into_opp_max = into_opp_max.max(into_opp);
                if into_opp > OPP_HALF_CAP_YARDS + 0.5 {
                    into_opp_over_cap += 1;
                }
                line_team_ticks += 1;
                width_sum += (max_x - min_x).max(0.0);
                width_ticks += 1;

                // Rule 3: wide-defender lateral drift, overall + split by possession phase.
                let team_in_possession = holder_team == Some(team);
                for p in &wide {
                    let drift = (p.position.x - p.home_position.x).abs();
                    wb_drift_sum += drift;
                    wb_ticks += 1;
                    if team_in_possession {
                        wb_drift_poss_sum += drift;
                        wb_ticks_poss += 1;
                    } else {
                        wb_drift_def_sum += drift;
                        wb_ticks_def += 1;
                    }
                }

                // Rule 1: count wide defenders released ahead of the central mean, but only
                // while this team is in possession (the overlap phase).
                if holder_team == Some(team) && !central_y.is_empty() && wide.len() >= 2 {
                    let central_mean = central_y.iter().sum::<f64>() / central_y.len() as f64;
                    let advanced = wide
                        .iter()
                        .filter(|p| (p.position.y - central_mean) * attack > RUN_MARGIN_YARDS)
                        .count();
                    poss_team_ticks += 1;
                    if advanced >= 2 {
                        both_wb_forward += 1;
                    } else if advanced == 1 {
                        one_wb_forward += 1;
                    }
                }
            }
        }
        eprintln!("seed {s}: final {}-{}", sim.score_home, sim.score_away);
    }

    let label = |on: bool| if on { "ON " } else { "off" };
    let pt = poss_team_ticks.max(1) as f64;
    let lt = line_team_ticks.max(1) as f64;
    let wt = wb_ticks.max(1) as f64;
    let wd = width_ticks.max(1) as f64;
    println!(
        "\n===== WING-BACK / BACK-FOUR SHAPE ({seeds} matches x {ticks} ticks) ====="
    );
    println!(
        "gates: single-wingback={}  lane-stickiness={}  resync-grace={}",
        label(single),
        label(stick),
        label(grace)
    );
    println!("-- Rule 1: one wing-back forward at a time (in-possession team-ticks: {poss_team_ticks}) --");
    println!(
        "  BOTH wing-backs forward : {:.3}%   (should be ~0 with single-wingback ON)",
        100.0 * both_wb_forward as f64 / pt
    );
    println!(
        "  exactly ONE forward     : {:.3}%",
        100.0 * one_wb_forward as f64 / pt
    );
    println!("-- Rule 2: back-four average into opponent half (team-ticks: {line_team_ticks}) --");
    println!("  mean into opp half      : {:.3} yd", into_opp_sum / lt);
    println!(
        "  max  into opp half      : {:.3} yd   (cap ~{:.0})",
        into_opp_max, OPP_HALF_CAP_YARDS
    );
    println!(
        "  ticks past {:.0}yd cap     : {:.3}%",
        OPP_HALF_CAP_YARDS,
        100.0 * into_opp_over_cap as f64 / lt
    );
    println!("-- Rule 3: wide-defender lane stickiness (wing-back-ticks: {wb_ticks}) --");
    println!(
        "  lateral drift |x-home_x|: {:.3} yd   (lower = stickier)",
        wb_drift_sum / wt
    );
    println!(
        "    while DEFENDING       : {:.3} yd   (ticks {wb_ticks_def})",
        wb_drift_def_sum / wb_ticks_def.max(1) as f64
    );
    println!(
        "    while IN POSSESSION   : {:.3} yd   (ticks {wb_ticks_poss}; opening wide is OK here)",
        wb_drift_poss_sum / wb_ticks_poss.max(1) as f64
    );
    println!(
        "  back-four width max-min : {:.3} yd   (guard rail: must NOT collapse)",
        width_sum / wd
    );
}
