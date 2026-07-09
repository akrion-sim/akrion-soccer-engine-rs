//! Rollout PoC driver — go/no-go on the premise that the analytic base wrongly
//! suppresses useful forward passes. See `docs/rollout-poc-harness-spec.md`.
//!
//! Thin driver over the `pub` PoC methods on `SoccerMatch` (the rollout / pending
//! internals are `pub(crate)`, so all rollout/leaf/event logic lives in the lib).
//! Walks ONE long analytic self-play match (restarting if it ends before we have
//! enough states), evaluates arms inline at sampled on-ball states, and aggregates.
//!
//! Run (release is mandatory — debug is ~10-50x too slow):
//!   POC_STATES=100 POC_SEEDS=4 POC_HOLDOUT=4 POC_MAX_H=45 POC_K=3 \
//!     CARGO_TARGET_DIR=.../target ./target/release/rollout_poc

use soccer_engine::des::general::soccer::{
    enable_deterministic_formation_lp, MatchConfig, SoccerMatch,
};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

/// Deterministic SplitMix64 for reproducible bootstrap resampling.
struct SplitMix64(u64);
impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Percentile bootstrap CI for the mean of `xs`. Returns (lo, hi) at the given
/// two-sided percentiles (e.g. 2.5 / 97.5 for a 95% CI).
fn bootstrap_ci(xs: &[f64], iters: usize, lo_pct: f64, hi_pct: f64, seed: u64) -> (f64, f64) {
    if xs.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let mut rng = SplitMix64(seed.wrapping_mul(0x2545_F491_4F6C_DD1D).wrapping_add(1));
    let mut means: Vec<f64> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let mut acc = 0.0;
        for _ in 0..xs.len() {
            acc += xs[rng.below(xs.len())];
        }
        means.push(acc / xs.len() as f64);
    }
    means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pick = |p: f64| -> f64 {
        let idx = ((p / 100.0) * (means.len() as f64 - 1.0)).round() as usize;
        means[idx.min(means.len() - 1)]
    };
    (pick(lo_pct), pick(hi_pct))
}

/// Running per-arm diagnostic accumulator.
#[derive(Default)]
struct ArmDiag {
    n: u64,
    fwd_delta_sum: f64,
    turnover_sum: f64,
    retained_sum: f64,
    ticks_sum: f64,
    leaf_sum: f64,
}
impl ArmDiag {
    fn push(&mut self, r: &soccer_engine::des::general::soccer::PocArmResult) {
        self.n += 1;
        self.fwd_delta_sum += r.completed_fwd_delta as f64;
        self.turnover_sum += r.turnover as i32 as f64;
        self.retained_sum += r.possession_retained as i32 as f64;
        self.ticks_sum += r.ticks_run as f64;
        self.leaf_sum += r.leaf;
    }
    fn report(&self, name: &str) {
        let n = self.n.max(1) as f64;
        println!(
            "  {name:<10} leaf={:.4}  fwd_delta={:.3}  turnover={:.3}  retained={:.3}  ticks={:.1}  (n={})",
            self.leaf_sum / n,
            self.fwd_delta_sum / n,
            self.turnover_sum / n,
            self.retained_sum / n,
            self.ticks_sum / n,
            self.n
        );
    }
}

fn build_match(seed: u32) -> SoccerMatch {
    let config = MatchConfig {
        seed,
        learning_enabled: false,
        ..MatchConfig::default()
    };
    SoccerMatch::default_11v11(config)
}

fn main() {
    // Route the formation LP through the deterministic simplex so forks from the
    // same pre-state + seed are byte-reproducible (Common Random Numbers). This is
    // MANDATORY for the parity gate to be meaningful — the default Clarabel solver
    // is per-process nondeterministic and would inject fork noise.
    enable_deterministic_formation_lp();

    let n_states = env_usize("POC_STATES", 200);
    let n_seeds = env_u64("POC_SEEDS", 4);
    let n_holdout = env_u64("POC_HOLDOUT", 4);
    let max_h = env_u32("POC_MAX_H", 45);
    let k = env_usize("POC_K", 3);
    let sample_every = env_u64("POC_SAMPLE_EVERY", 10);
    let base_seed = env_u64("POC_BASE_SEED", 0x0000_1234);
    let match_seed0 = env_u32("POC_MATCH_SEED", 0xA5EED);
    const OPEN_YDS: f64 = 3.0;
    const DRIBBLE_YDS: f64 = 6.0;
    const PARITY_TOL: f64 = 0.05; // leaf-scale tolerance for "≈0"

    // Selection seeds (CRN for arm selection) and DISJOINT held-out eval seeds.
    let selection_seeds: Vec<u64> = (0..n_seeds).map(|i| base_seed + i).collect();
    let holdout_seeds: Vec<u64> = (0..n_holdout).map(|i| base_seed + 1_000 + i).collect();

    println!("=== ROLLOUT PoC ===");
    println!(
        "config: STATES={n_states} SEEDS={n_seeds} HOLDOUT={n_holdout} MAX_H={max_h} K={k} SAMPLE_EVERY={sample_every} OPEN_YDS={OPEN_YDS} DRIBBLE_YDS={DRIBBLE_YDS}"
    );
    println!("leaf = expected_threat(end) + 0.30*retained - 0.60*turnover");
    println!("selection_seeds={selection_seeds:?} holdout_seeds={holdout_seeds:?}");

    // Denominator + outcome accumulators.
    let mut total_on_ball: u64 = 0;
    let mut with_forward_option: u64 = 0;
    let mut parity_deltas: Vec<f64> = Vec::new();
    let mut parity_states: u64 = 0;
    let mut held_deltas: Vec<f64> = Vec::new(); // best_fwd_leaf - natural_leaf (held-out, per state)
    let mut best_fwd_gt_natural: u64 = 0;
    let mut fwd_is_global_argmax: u64 = 0;

    // Per-arm held-out diagnostics.
    let mut diag_nat = ArmDiag::default();
    let mut diag_fwd = ArmDiag::default();
    let mut diag_drb = ArmDiag::default();

    // forced_action_fired probe over pass arms (guards silent no-ops).
    let mut pass_arm_evals: u64 = 0;
    let mut pass_arm_fired: u64 = 0;

    // Stratification: held-out delta by zone (defensive/middle/attacking third)
    // and by whether the natural arm already plays forward.
    let mut strat_zone: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let zone_name = ["def-third", "mid-third", "att-third"];
    let mut strat_natfwd: [Vec<f64>; 2] = [Vec::new(), Vec::new()]; // [not-already-fwd, already-fwd]

    let mut states_evaluated: usize = 0;
    let mut match_seed = match_seed0;
    let mut sim = build_match(match_seed);
    let mut matches_played: u64 = 1;
    let mut live_ticks: u64 = 0;
    let safety_cap: u64 = 20_000_000;
    let smoke = std::env::var("POC_SMOKE_TRACE").ok().as_deref() == Some("1");

    while states_evaluated < n_states {
        if sim.is_done() {
            match_seed = match_seed.wrapping_add(1);
            sim = build_match(match_seed);
            matches_played += 1;
            continue;
        }
        sim.run_time_step();
        live_ticks += 1;
        if live_ticks > safety_cap {
            eprintln!("[warn] hit live-tick safety cap; stopping sampling early");
            break;
        }
        if live_ticks % sample_every != 0 {
            continue;
        }
        let carrier = match sim.poc_on_ball_field_carrier() {
            Some(c) => c,
            None => continue,
        };
        total_on_ball += 1;

        let targets = sim.poc_forward_pass_targets(carrier, k, OPEN_YDS);
        if targets.is_empty() {
            // stratum "no-forward-option" — still counts in the denominator.
            continue;
        }
        with_forward_option += 1;
        states_evaluated += 1;

        let epv_start = sim.poc_epv_now(carrier).unwrap_or(0.0);
        let ysign = sim.poc_carrier_forward_ysign(carrier).unwrap_or(1.0);
        let ball_y = sim.ball.position.y;
        let fl = sim.config.field_length_yards;
        let progress = if ysign >= 0.0 {
            ball_y / fl
        } else {
            (fl - ball_y) / fl
        };
        let zone = if progress < 1.0 / 3.0 {
            0
        } else if progress < 2.0 / 3.0 {
            1
        } else {
            2
        };

        // --- SELECTION (CRN seeds): Q_hat(arm) = mean leaf over selection seeds ---
        let nat_q = mean(
            &selection_seeds
                .iter()
                .map(|&s| sim.poc_rollout_arm(carrier, None, s, max_h).leaf)
                .collect::<Vec<_>>(),
        );
        let mut best_fwd_q = f64::NEG_INFINITY;
        let mut best_fwd_target = targets[0];
        for &t in &targets {
            let q = mean(
                &selection_seeds
                    .iter()
                    .map(|&s| {
                        sim.poc_rollout_arm(carrier, Some(("pass".to_string(), Some(t))), s, max_h)
                            .leaf
                    })
                    .collect::<Vec<_>>(),
            );
            if q > best_fwd_q {
                best_fwd_q = q;
                best_fwd_target = t;
            }
        }
        let dribble_target = sim.poc_dribble_forward_target(carrier, DRIBBLE_YDS);
        let drb_q = mean(
            &selection_seeds
                .iter()
                .map(|&s| {
                    sim.poc_rollout_arm(
                        carrier,
                        Some(("dribble".to_string(), dribble_target)),
                        s,
                        max_h,
                    )
                    .leaf
                })
                .collect::<Vec<_>>(),
        );

        // --- HELD-OUT (disjoint seeds): re-roll natural + best_fwd + dribble ---
        let mut nat_h: Vec<f64> = Vec::with_capacity(holdout_seeds.len());
        let mut fwd_h: Vec<f64> = Vec::with_capacity(holdout_seeds.len());
        let mut drb_h: Vec<f64> = Vec::with_capacity(holdout_seeds.len());
        let mut nat_retained = 0.0;
        let mut nat_epv_end = 0.0;
        for &s in &holdout_seeds {
            let nr = sim.poc_rollout_arm(carrier, None, s, max_h);
            nat_retained += nr.possession_retained as i32 as f64;
            nat_epv_end += nr.epv_end;
            nat_h.push(nr.leaf);
            diag_nat.push(&nr);

            let fr =
                sim.poc_rollout_arm(carrier, Some(("pass".to_string(), Some(best_fwd_target))), s, max_h);
            pass_arm_evals += 1;
            if fr.forced_action_fired {
                pass_arm_fired += 1;
            }
            fwd_h.push(fr.leaf);
            diag_fwd.push(&fr);

            let dr = sim.poc_rollout_arm(
                carrier,
                Some(("dribble".to_string(), dribble_target)),
                s,
                max_h,
            );
            drb_h.push(dr.leaf);
            diag_drb.push(&dr);
        }
        let nat_hm = mean(&nat_h);
        let fwd_hm = mean(&fwd_h);
        let drb_hm = mean(&drb_h);
        let held_delta = fwd_hm - nat_hm;
        held_deltas.push(held_delta);
        if fwd_hm > nat_hm {
            best_fwd_gt_natural += 1;
        }
        if fwd_hm >= nat_hm && fwd_hm >= drb_hm {
            fwd_is_global_argmax += 1;
        }

        let nat_retained_frac = nat_retained / holdout_seeds.len().max(1) as f64;
        let nat_epv_end_mean = nat_epv_end / holdout_seeds.len().max(1) as f64;
        let nat_already_forward = nat_retained_frac >= 0.5 && nat_epv_end_mean > epv_start;

        strat_zone[zone].push(held_delta);
        strat_natfwd[nat_already_forward as usize].push(held_delta);

        // --- PARITY gate: natural's own pass, re-forced, must ≈ natural ---
        let s0 = selection_seeds[0];
        if let Some(t_nat) = sim.poc_natural_pass_target(carrier, s0) {
            parity_states += 1;
            for (i, &s) in holdout_seeds.iter().enumerate() {
                let forced_nat_leaf = sim
                    .poc_rollout_arm(carrier, Some(("pass".to_string(), Some(t_nat))), s, max_h)
                    .leaf;
                parity_deltas.push((forced_nat_leaf - nat_h[i]).abs());
            }
        }

        if smoke {
            eprintln!(
                "[state {states_evaluated}] carrier={carrier} zone={} targets={} nat_q={nat_q:.4} best_fwd_q={best_fwd_q:.4} drb_q={drb_q:.4} | held: nat={nat_hm:.4} fwd={fwd_hm:.4} drb={drb_hm:.4} delta={held_delta:.4} epv0={epv_start:.4} nat_fwd={nat_already_forward}",
                zone_name[zone],
                targets.len()
            );
        }
        // The LIVE match is never forced — sampling walks real analytic self-play.
    }

    // ================= REPORT =================
    println!("\n========== ROLLOUT PoC RESULTS ==========");
    println!(
        "matches_played={matches_played} live_ticks={live_ticks} states_evaluated={states_evaluated}"
    );

    // --- PARITY GATE (validated FIRST) ---
    println!("\n--- PARITY GATE (natural pass re-forced must ≈ natural) ---");
    let parity_mean = mean(&parity_deltas);
    let (p_lo, p_hi) = bootstrap_ci(&parity_deltas, 4000, 5.0, 95.0, 0xPAR1TY ^ 0);
    let parity_has_samples = !parity_deltas.is_empty();
    println!(
        "parity_states={parity_states} parity_seed_pairs={} mean|forced_nat - natural| = {parity_mean:.5}  90% CI [{p_lo:.5}, {p_hi:.5}]  (tol={PARITY_TOL})",
        parity_deltas.len()
    );
    let parity_pass = parity_has_samples && parity_mean <= PARITY_TOL;
    if !parity_has_samples {
        println!("  PARITY: N/A — no natural-pass states sampled (cannot validate fork fidelity)");
    } else if parity_pass {
        println!("  PARITY: PASS — forced injection reproduces natural within CRN noise");
    } else {
        println!("  PARITY: FAIL — injection/fork is BIASED; do NOT over-interpret the arm comparison");
    }

    // --- Denominator ---
    println!("\n--- DENOMINATOR ---");
    let fwd_pct = 100.0 * with_forward_option as f64 / total_on_ball.max(1) as f64;
    println!(
        "total_on_ball={total_on_ball}  with_forward_option={with_forward_option} ({fwd_pct:.1}%)"
    );

    // --- Held-out paired delta ---
    println!("\n--- HELD-OUT PAIRED DELTA (best_fwd_leaf - natural_leaf) ---");
    let delta_mean = mean(&held_deltas);
    let (d_lo, d_hi) = bootstrap_ci(&held_deltas, 4000, 2.5, 97.5, 0xDE17A5);
    println!(
        "n={}  mean delta = {delta_mean:.5}  95% CI [{d_lo:.5}, {d_hi:.5}]",
        held_deltas.len()
    );
    let frac_gt = 100.0 * best_fwd_gt_natural as f64 / held_deltas.len().max(1) as f64;
    let frac_argmax = 100.0 * fwd_is_global_argmax as f64 / held_deltas.len().max(1) as f64;
    println!("frac(best_fwd > natural) = {frac_gt:.1}%   frac(fwd is global argmax incl nat/dribble) = {frac_argmax:.1}%");

    // --- Diagnostics ---
    println!("\n--- PER-ARM DIAGNOSTICS (held-out) ---");
    diag_nat.report("natural");
    diag_fwd.report("best_fwd");
    diag_drb.report("dribble");
    let fire_pct = 100.0 * pass_arm_fired as f64 / pass_arm_evals.max(1) as f64;
    println!(
        "forced_action_fired on pass arms: {pass_arm_fired}/{pass_arm_evals} ({fire_pct:.1}%)"
    );

    // --- Stratified ---
    println!("\n--- STRATIFIED HELD-OUT DELTA ---");
    println!("by zone (thirds by ball y vs attack dir):");
    for z in 0..3 {
        let m = mean(&strat_zone[z]);
        println!("  {:<10} n={:<4} mean_delta={m:.5}", zone_name[z], strat_zone[z].len());
    }
    println!("by natural-already-forward (epv gain>0 & retained):");
    for (idx, label) in ["nat-NOT-already-fwd", "nat-already-fwd"].iter().enumerate() {
        let m = mean(&strat_natfwd[idx]);
        println!("  {:<20} n={:<4} mean_delta={m:.5}", label, strat_natfwd[idx].len());
    }

    // --- VERDICT ---
    println!("\n--- VERDICT ---");
    let arm_positive = delta_mean > 0.0 && d_lo > 0.0;
    let verdict = if !parity_has_samples {
        "INCONCLUSIVE (no parity samples — fork fidelity unverified)"
    } else if !parity_pass {
        "INCONCLUSIVE (PARITY GATE FAILED — fork/injection biased; a single-arm loss here does NOT falsify the premise)"
    } else if arm_positive {
        "POSITIVE (parity passed AND held-out best_fwd - natural > 0 with 95% CI excluding 0)"
    } else {
        "NEGATIVE/INCONCLUSIVE (parity passed but held-out best_fwd - natural not > 0 with CI excluding 0)"
    };
    println!("VERDICT: {verdict}");
}
