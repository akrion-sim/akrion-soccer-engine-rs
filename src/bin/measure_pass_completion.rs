//! Pass-completion measurement harness for the NN+DP+MPC pass-completion proof.
//! Runs N self-play matches and reports pass completion RATE (completed/attempted),
//! forward/backward breakdown, forward-share, and per-match mean+CI.
//!
//! Run: measure_pass_completion [ticks] [seeds]
//! Env: SOCCER_MEASURE_MPC=1 -> enable tier-2 per-player MPC (execution model).
//!
//! Baseline (analytic, no learned heads, mpc off) is the number the NN+DP+MPC
//! treatment must beat. Learned-head treatments need a trained frontier loaded
//! (separate harness); this bin measures the analytic base + MPC-on execution arm.

use soccer_engine::des::general::soccer::{
    enable_deterministic_formation_lp, MatchConfig, SoccerMatch,
};

fn main() {
    // Deterministic formation LP so a given seed yields byte-identical matches
    // (default Clarabel is per-process nondeterministic).
    enable_deterministic_formation_lp();
    let args: Vec<String> = std::env::args().collect();
    let ticks: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(9_000);
    let seeds: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(30);
    let mpc_on = std::env::var("SOCCER_MEASURE_MPC").ok().as_deref() == Some("1");

    let (mut att, mut comp) = (0u64, 0u64);
    let (mut fwd_att, mut fwd_comp) = (0u64, 0u64);
    let (mut lat_att, mut lat_comp) = (0u64, 0u64);
    let (mut back_att, mut back_comp) = (0u64, 0u64);
    let (mut intentional_att, mut intentional_comp) = (0u64, 0u64);
    let (mut intentional_fwd_att, mut intentional_fwd_comp) = (0u64, 0u64);
    let (mut intentional_lat_att, mut intentional_lat_comp) = (0u64, 0u64);
    let (mut intentional_back_att, mut intentional_back_comp) = (0u64, 0u64);
    let mut per_match_rate: Vec<f64> = Vec::new();

    for s in 0..seeds {
        let mut config = MatchConfig {
            seed: 0x9A55_0000u32.wrapping_add(s as u32),
            ..MatchConfig::default()
        };
        if mpc_on {
            config.mpc.tier2_player_enabled = true;
        }
        let mut sim = SoccerMatch::default_11v11(config);
        for _ in 0..ticks {
            sim.run_time_step();
        }
        let a = (sim.stats.passes_attempted_home + sim.stats.passes_attempted_away) as u64;
        let c = (sim.stats.passes_completed_home + sim.stats.passes_completed_away) as u64;
        att += a;
        comp += c;
        fwd_att += (sim.stats.passes_attempted_forward_home
            + sim.stats.passes_attempted_forward_away) as u64;
        fwd_comp += (sim.stats.passes_completed_forward_home
            + sim.stats.passes_completed_forward_away) as u64;
        lat_att += (sim.stats.passes_attempted_lateral_home
            + sim.stats.passes_attempted_lateral_away) as u64;
        lat_comp += (sim.stats.passes_completed_lateral_home
            + sim.stats.passes_completed_lateral_away) as u64;
        back_att += (sim.stats.passes_attempted_backward_home
            + sim.stats.passes_attempted_backward_away) as u64;
        back_comp += (sim.stats.passes_completed_backward_home
            + sim.stats.passes_completed_backward_away) as u64;

        intentional_att += (sim.stats.intentional_passes_attempted_home
            + sim.stats.intentional_passes_attempted_away) as u64;
        intentional_comp += (sim.stats.intentional_passes_completed_home
            + sim.stats.intentional_passes_completed_away) as u64;
        intentional_fwd_att += (sim.stats.intentional_passes_attempted_forward_home
            + sim.stats.intentional_passes_attempted_forward_away)
            as u64;
        intentional_fwd_comp += (sim.stats.intentional_passes_completed_forward_home
            + sim.stats.intentional_passes_completed_forward_away)
            as u64;
        intentional_lat_att += (sim.stats.intentional_passes_attempted_lateral_home
            + sim.stats.intentional_passes_attempted_lateral_away)
            as u64;
        intentional_lat_comp += (sim.stats.intentional_passes_completed_lateral_home
            + sim.stats.intentional_passes_completed_lateral_away)
            as u64;
        intentional_back_att += (sim.stats.intentional_passes_attempted_backward_home
            + sim.stats.intentional_passes_attempted_backward_away)
            as u64;
        intentional_back_comp += (sim.stats.intentional_passes_completed_backward_home
            + sim.stats.intentional_passes_completed_backward_away)
            as u64;
        if a > 0 {
            per_match_rate.push(c as f64 / a as f64);
        }
        eprintln!(
            "seed {s}: att={a} comp={c} rate={:.4} score {}-{}",
            c as f64 / a.max(1) as f64,
            sim.score_home,
            sim.score_away
        );
    }

    let n = per_match_rate.len().max(1) as f64;
    let rate = comp as f64 / att.max(1) as f64;
    let mean_pm = per_match_rate.iter().sum::<f64>() / n;
    let var = per_match_rate
        .iter()
        .map(|r| (r - mean_pm).powi(2))
        .sum::<f64>()
        / n;
    let sd = var.sqrt();
    let se = sd / n.sqrt();
    println!(
        "\n===== PASS COMPLETION ({seeds} matches x {ticks} ticks, mpc={}) =====",
        mpc_on
    );
    println!("attempted={att} completed={comp}");
    println!("OVERALL completion rate = {:.4}", rate);
    println!(
        "per-match mean rate = {:.4}  sd={:.4}  se={:.4}  95%CI=[{:.4},{:.4}]",
        mean_pm,
        sd,
        se,
        mean_pm - 1.96 * se,
        mean_pm + 1.96 * se
    );
    println!(
        "all direction rates: forward={}/{} ({:.1}%, target 85%) lateral={}/{} ({:.1}%) backward={}/{} ({:.1}%, target 95%)",
        fwd_comp,
        fwd_att,
        completion_percent(fwd_comp, fwd_att),
        lat_comp,
        lat_att,
        completion_percent(lat_comp, lat_att),
        back_comp,
        back_att,
        completion_percent(back_comp, back_att),
    );
    println!(
        "intentional: completed={}/{} ({:.1}%, target 90%); forward={}/{} ({:.1}%, target 85%) lateral={}/{} ({:.1}%) backward={}/{} ({:.1}%, target 95%)",
        intentional_comp,
        intentional_att,
        completion_percent(intentional_comp, intentional_att),
        intentional_fwd_comp,
        intentional_fwd_att,
        completion_percent(intentional_fwd_comp, intentional_fwd_att),
        intentional_lat_comp,
        intentional_lat_att,
        completion_percent(intentional_lat_comp, intentional_lat_att),
        intentional_back_comp,
        intentional_back_att,
        completion_percent(intentional_back_comp, intentional_back_att),
    );
    println!("END");
}

fn completion_percent(completed: u64, attempted: u64) -> f64 {
    100.0 * completed as f64 / attempted.max(1) as f64
}
