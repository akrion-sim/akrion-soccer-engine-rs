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

    let (mut att, mut comp, mut fwd, mut back) = (0u64, 0u64, 0u64, 0u64);
    let (mut att_fwd, mut att_back) = (0u64, 0u64);
    let (mut comp_fwd_intent, mut comp_back_intent) = (0u64, 0u64);
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
        fwd += (sim.stats.passes_completed_forward_home + sim.stats.passes_completed_forward_away)
            as u64;
        back += (sim.stats.passes_completed_backward_home
            + sim.stats.passes_completed_backward_away) as u64;
        att_fwd += (sim.stats.passes_attempted_forward_home
            + sim.stats.passes_attempted_forward_away) as u64;
        att_back += (sim.stats.passes_attempted_backward_home
            + sim.stats.passes_attempted_backward_away) as u64;
        comp_fwd_intent += (sim.stats.passes_completed_forward_intent_home
            + sim.stats.passes_completed_forward_intent_away) as u64;
        comp_back_intent += (sim.stats.passes_completed_backward_intent_home
            + sim.stats.passes_completed_backward_intent_away) as u64;
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
        "forward completed = {fwd}  backward completed = {back}  forward-share = {:.3}",
        fwd as f64 / comp.max(1) as f64
    );
    // Per-direction completion RATES against the training targets
    // (~85% forward / ~95% backward / ~90% overall). BOTH sides of each rate
    // use the LAUNCH-intent classifier (intended target vs origin), so the
    // ratio is a true rate. The realized-direction line below re-buckets the
    // same completions by actual reception — the gap between "forward intent"
    // and "realized forward gain" is the under-hit-forward-ball diagnostic
    // (receivers checking back to collect shrink realized gain to lateral).
    println!(
        "DIRECTION RATES (launch-intent): forward {}/{} = {:.4} (target 0.85)  backward {}/{} = {:.4} (target 0.95)  overall {:.4} (target 0.90)",
        comp_fwd_intent,
        att_fwd,
        comp_fwd_intent as f64 / att_fwd.max(1) as f64,
        comp_back_intent,
        att_back,
        comp_back_intent as f64 / att_back.max(1) as f64,
        rate
    );
    println!(
        "REALIZED direction of completions: forward {} backward {} lateral {} (forward-gain share {:.3})",
        fwd,
        back,
        comp.saturating_sub(fwd + back),
        fwd as f64 / comp.max(1) as f64
    );
    println!("END");
}
