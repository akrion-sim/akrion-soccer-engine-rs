//! Port of `src/des/main-soccer-rotation.ts`.
//!
//! 7v7 youth-soccer player rotation solved every which way — random,
//! per-period Hungarian, multi-period LP relaxation, time-boxed IP/MIP
//! branch-and-cut, and exact MDP backward induction — with the match outcome
//! simulated by the DES engine and (optionally) animated as a pitch + bench
//! scene plus an IP/MIP solver-entity scene.
//!
//! Conversion notes:
//!   - `process.argv` / `process.env` → `std::env`; `Date.now()` timing →
//!     `std::time::Instant`.
//!   - the layered solvers, evaluation, and match DES all delegate to
//!     `general::soccer_rotation`; the LP-relaxation backend selector maps to
//!     `general::ip_mip_des::LpRelaxationAlgorithm`.
//!   - both animation scenes (`soccer_scene`, `soccer_ipmip_solver_scene`) are
//!     ported, so the render path is wired (not stubbed).

use std::time::Instant;

use crate::des::animation::frame_recorder::{FrameRecorder, FrameRecorderOpts};
use crate::des::animation::scenes::soccer_ipmip_solver_scene as solver_scene;
use crate::des::animation::scenes::soccer_scene as scene;
use crate::des::general::ip_mip_des::{ConcreteLpRelaxationAlgorithm, LpRelaxationAlgorithm};
use crate::des::general::soccer_rotation::{
    build_sample_soccer_problem, default_outfield_formation, evaluate_schedule,
    evaluate_soccer_pomdp_features, formation_position_names, formation_with_gk,
    parse_outfield_formation, policy_greedy_hungarian, policy_ipmip_feasible, policy_lp_relaxed,
    policy_mdp_vi, policy_mdp_vi_memoryless, policy_random_schedule, run_many_matches,
    simulate_match_des, validate_schedule_structure, welch_t, AffinityBuilderOptions, GoalSide,
    GreedyHungarianOptions, MatchAggregate, MatchSimOptions, Schedule, SoccerIPMIPPolicyOptions,
    SoccerIPMIPPolicyResult, SoccerPOMDPFeatureOptions, SoccerProblem,
};

// -----------------------------------------------------------------------------
// Small formatting helpers.
// -----------------------------------------------------------------------------

/// `Number.prototype.toExponential(digits)` (signed exponent, no leading zeros).
fn to_exponential(x: f64, digits: usize) -> String {
    if x.is_nan() {
        return "NaN".to_string();
    }
    if x.is_infinite() {
        return if x < 0.0 { "-Infinity" } else { "Infinity" }.to_string();
    }
    let s = format!("{:.*e}", digits, x);
    let (mant, exp) = s.split_once('e').unwrap_or((s.as_str(), "0"));
    let exp_num: i32 = exp.parse().unwrap_or(0);
    let sign = if exp_num < 0 { '-' } else { '+' };
    format!("{}e{}{}", mant, sign, exp_num.abs())
}

/// Convert a `general::ip_mip_des::IPMIPSolution` to the subset mirror the
/// solver scene reads (`soccer_ipmip_solver_scene::IPMIPSolution`).
fn to_scene_solution(
    mip: &crate::des::general::ip_mip_des::IPMIPSolution,
) -> solver_scene::IPMIPSolution {
    use crate::des::general::ip_mip_des::TraceAction;
    let action_str = |a: TraceAction| -> String {
        match a {
            TraceAction::Branch => "branch",
            TraceAction::Cut => "cut",
            TraceAction::Prune => "prune",
            TraceAction::Incumbent => "incumbent",
            TraceAction::Unbounded => "unbounded",
        }
        .to_string()
    };
    let trace = mip
        .trace
        .iter()
        .map(|e| solver_scene::IPMIPTraceEvent {
            node_id: e.node_id.to_string(),
            depth: e.depth as f64,
            action: action_str(e.action),
            lp_z: e.lp_z,
            fractional: e.fractional.iter().map(|&v| v as f64).collect(),
            reason: e.reason.clone(),
        })
        .collect();
    let mut usage: Vec<(String, f64)> = mip
        .lp_algorithm_usage
        .iter()
        .map(|(k, v)| (k.as_str().to_string(), *v as f64))
        .collect();
    usage.sort_by(|a, b| a.0.cmp(&b.0));
    solver_scene::IPMIPSolution {
        trace,
        status: mip.status.as_str().to_string(),
        z: mip.z,
        gap: mip.gap,
        best_bound: mip.best_bound,
        lp_algorithm: mip.lp_algorithm.as_str().to_string(),
        lp_solves: mip.lp_solves as f64,
        elapsed_ms: mip.elapsed_ms,
        nodes_explored: mip.nodes_explored as f64,
        cuts_added: mip.cuts_added as f64,
        candidates_tried: mip.candidates_tried as f64,
        lp_algorithm_usage: usage,
        incumbent_source: mip.incumbent_source.clone(),
    }
}

/// Integer-valued floats without a trailing `.0` (mirrors JS `${number}`).
fn jn(x: f64) -> String {
    if x.is_finite() && x.fract() == 0.0 {
        format!("{}", x as i64)
    } else {
        format!("{}", x)
    }
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_bool(key: &str, default: bool) -> bool {
    std::env::var(key)
        .ok()
        .map(|value| {
            matches!(
                value.trim(),
                "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"
            )
        })
        .unwrap_or(default)
}

/// `process.env.MIP_LP_ALGO` (a `LPRelaxationAlgorithm` string) → enum.
fn parse_lp_algorithm(s: &str) -> LpRelaxationAlgorithm {
    use ConcreteLpRelaxationAlgorithm as C;
    match s {
        "auto" => LpRelaxationAlgorithm::Auto,
        "incremental-primal-dual" => LpRelaxationAlgorithm::Concrete(C::IncrementalPrimalDual),
        "des-simplex-dantzig" => LpRelaxationAlgorithm::Concrete(C::DesSimplexDantzig),
        "des-simplex-bland" => LpRelaxationAlgorithm::Concrete(C::DesSimplexBland),
        "internal-ipm" | "internal-interior-point" => {
            LpRelaxationAlgorithm::Concrete(C::InternalInteriorPoint)
        }
        "external-highs" => LpRelaxationAlgorithm::Concrete(C::ExternalHighs),
        "external-highs-ds" => LpRelaxationAlgorithm::Concrete(C::ExternalHighsDs),
        "external-highs-ipm" => LpRelaxationAlgorithm::Concrete(C::ExternalHighsIpm),
        // "internal-simplex" and any unknown spelling default to the internal simplex.
        _ => LpRelaxationAlgorithm::Concrete(C::InternalSimplex),
    }
}

/// `name.replace(/[^a-z0-9._-]+/gi, '-').replace(/^-+|-+$/g, '')`.
fn safe_policy_name(name: &str) -> String {
    let mut out = String::new();
    let mut prev_dash = false;
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() || ch == '.' || ch == '_' || ch == '-' {
            out.push(ch);
            prev_dash = false;
        } else if !prev_dash {
            out.push('-');
            prev_dash = true;
        }
    }
    out.trim_matches('-').to_string()
}

/// Longest run (in periods) any single player stays on the field, across the
/// whole schedule. Used for the stamina audit print-out.
fn longest_on_field_run(problem: &SoccerProblem, schedule: &Schedule) -> usize {
    let mut longest = 0usize;
    for p in 0..problem.num_players {
        let mut run = 0usize;
        for t in 0..problem.num_periods {
            let on_field = schedule.assignment[t].contains(&(p as i64));
            if on_field {
                run += 1;
                longest = longest.max(run);
            } else {
                run = 0;
            }
        }
    }
    longest
}

struct PolicyDesc {
    name: &'static str,
    note: &'static str,
}

/// Build a schedule for `name` (mirrors the per-policy `build` closures). The
/// IP/MIP build records its full result in `mip_latest`.
fn build_schedule(
    name: &str,
    problem: &SoccerProblem,
    seed: u32,
    mip_opts: &SoccerIPMIPPolicyOptions,
    mip_latest: &mut Option<SoccerIPMIPPolicyResult>,
) -> Schedule {
    match name {
        "random" => policy_random_schedule(problem, seed + 1),
        "MDP-memoryless" => policy_mdp_vi_memoryless(problem).schedule,
        "greedy-Hungarian" => policy_greedy_hungarian(
            problem,
            &GreedyHungarianOptions {
                fairness_aware: Some(true),
            },
        ),
        "LP-relaxation" => policy_lp_relaxed(problem).schedule,
        "IP/MIP-feasible" => {
            let r = policy_ipmip_feasible(problem, mip_opts);
            let schedule = r.schedule.clone();
            *mip_latest = Some(r);
            schedule
        }
        "MDP-VI-exact" => policy_mdp_vi(problem).to_schedule(),
        _ => unreachable!("unknown policy {name}"),
    }
}

/// Pre-env defaults for a soccer run. `run()` keeps the historical
/// 12-player / 20-minute shape with animation gated by `ANIMATE=1`; `run_anim()`
/// uses a 13-player / 10-minute showcase with the animation forced on. Every
/// field is still overridable through the same env vars.
struct SoccerRunDefaults {
    force_animate: bool,
    num_players: usize,
    num_positions: usize,
    num_periods: usize,
    minutes_per_period: usize,
    max_consec_on_field: usize,
    policy: &'static str,
    /// Default outfield formation (GK excluded), e.g. `"3-2-1"`. Empty → derive
    /// a sensible default from `num_positions`. Overridden by `FORMATION` env.
    formation: &'static str,
}

impl Default for SoccerRunDefaults {
    fn default() -> Self {
        SoccerRunDefaults {
            force_animate: false,
            num_players: 12,
            num_positions: 7,
            num_periods: 4,
            minutes_per_period: 20,
            max_consec_on_field: 3,
            policy: "",
            formation: "",
        }
    }
}

/// Entry point (`main()` in the TS source). Roster/match shape is read from env
/// (`NUM_PLAYERS`, `NUM_POSITIONS`, `NUM_PERIODS`, `MINUTES_PER_PERIOD`,
/// `MAX_CONSEC_ON_FIELD`, `POLICY`); animation is gated by `ANIMATE=1`.
pub fn run() {
    run_with_defaults(&SoccerRunDefaults::default());
}

/// Always-animating showcase used by the `main_soccer_rotation_anim` catalogue
/// entry (mirrors the `*_anim` convention): a 13-player squad, still 7-a-side,
/// four 10-minute periods, a stamina cap of 3 consecutive periods on the field,
/// focused on the IP/MIP solver. Every value is still overridable via env.
pub fn run_anim() {
    run_with_defaults(&SoccerRunDefaults {
        force_animate: true,
        num_players: 13,
        num_periods: 4,
        minutes_per_period: 10,
        max_consec_on_field: 3,
        policy: "IP/MIP",
        // 7-a-side with 3 at the back (GK + 3-2-1). Override with FORMATION=...
        // (e.g. FORMATION=3-4-1 renders a 9-a-side shape).
        formation: "3-2-1",
        ..SoccerRunDefaults::default()
    });
}

fn run_with_defaults(defaults: &SoccerRunDefaults) {
    let seed = env_usize("SEED", 4242) as u32;
    let num_matches = env_usize("N_MATCHES", 100);
    let animate = defaults.force_animate || std::env::var("ANIMATE").as_deref() == Ok("1");
    let policy_filter = std::env::var("POLICY")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| defaults.policy.to_string())
        .trim()
        .to_lowercase();
    let mip_time_limit_ms = env_f64("MIP_TIME_LIMIT_MS", 30_000.0);
    let mip_max_nodes = env_usize("MIP_MAX_NODES", 5_000);
    let mip_lp_algorithm =
        parse_lp_algorithm(&std::env::var("MIP_LP_ALGO").unwrap_or_else(|_| "auto".to_string()));
    let mip_allow_external_solvers = env_bool("MIP_ALLOW_EXTERNAL_SOLVERS", true);

    // Roster / match shape (all overridable so the same model serves 11-, 12-,
    // or 13-player squads, different match lengths, formations, and a tunable
    // stamina cap).
    //
    // Formation drives the on-field size: a `FORMATION` env (e.g. "3-4-1") or
    // the per-run default is the *outfield* line-up (GK excluded); we prepend
    // the keeper, and `num_positions` is the row-size sum. With neither set we
    // derive a default formation for `NUM_POSITIONS` on-field players.
    let formation_str = std::env::var("FORMATION")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| defaults.formation.to_string());
    let outfield = if formation_str.trim().is_empty() {
        let on_field = env_usize("NUM_POSITIONS", defaults.num_positions).max(2);
        default_outfield_formation(on_field)
    } else {
        parse_outfield_formation(&formation_str).unwrap_or_else(|| {
            default_outfield_formation(env_usize("NUM_POSITIONS", defaults.num_positions).max(2))
        })
    };
    let formation = formation_with_gk(&outfield);
    let num_positions = formation.iter().sum::<usize>();
    let position_names = formation_position_names(&formation);

    let num_players = env_usize("NUM_PLAYERS", defaults.num_players).max(num_positions + 1);
    let num_periods = env_usize("NUM_PERIODS", defaults.num_periods).max(1);
    let minutes_per_period = env_usize("MINUTES_PER_PERIOD", defaults.minutes_per_period).max(1);
    // 0 (or unset-to-0) disables the stamina cap; default caps on-field runs at 3.
    let max_consecutive_on_field =
        match env_usize("MAX_CONSEC_ON_FIELD", defaults.max_consec_on_field) {
            0 => None,
            m => Some(m),
        };

    let mut problem = build_sample_soccer_problem(&AffinityBuilderOptions {
        num_players: Some(num_players),
        num_positions: Some(num_positions),
        num_periods: Some(num_periods),
        max_consecutive_on_field,
        seed: Some(seed),
    });
    // Relabel positions with formation roles (GK / LB / CM / ST …) so the
    // banner, audit and pitch all agree on what each slot is.
    problem.position_names = Some(position_names);
    let match_opts = MatchSimOptions {
        minutes_per_period: Some(minutes_per_period),
        ..Default::default()
    };
    let mip_opts = SoccerIPMIPPolicyOptions {
        time_limit_ms: Some(mip_time_limit_ms),
        max_nodes: Some(mip_max_nodes),
        max_ticks: Some(100.max(mip_max_nodes * 8)),
        lp_algorithm: Some(mip_lp_algorithm),
        allow_external_solvers: Some(mip_allow_external_solvers),
        ..Default::default()
    };
    let mut mip_latest: Option<SoccerIPMIPPolicyResult> = None;

    // ─── Banner ───────────────────────────────────────────────────────────
    println!(
        "# {}v{} youth soccer player rotation as combinatorial optimisation",
        problem.num_positions, problem.num_positions
    );
    println!(
        "# {} players, {} positions ({} on the bench), {} periods of {} min each",
        problem.num_players,
        problem.num_positions,
        problem.bench_size,
        problem.num_periods,
        minutes_per_period
    );
    println!(
        "# formation: GK + {} ({})",
        outfield
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join("-"),
        problem
            .position_names
            .as_ref()
            .map(|n| n.join(" "))
            .unwrap_or_default()
    );
    println!("# fairness constraint: no player benched two consecutive periods");
    match problem.max_consecutive_on_field {
        Some(m) => println!(
            "# stamina constraint: no player on the field for more than {} consecutive periods ({} min)",
            m,
            m * minutes_per_period
        ),
        None => println!("# stamina constraint: disabled (MAX_CONSEC_ON_FIELD=0)"),
    }
    println!("# affinity tensor seed = {}", seed);
    println!();
    println!("# Per-player best position and period peak (sample of affinity tensor):");
    for p in 0..problem.num_players {
        let mut best_pos = 0usize;
        let mut best_val = f64::NEG_INFINITY;
        let mut best_t = 0usize;
        for pos in 0..problem.num_positions {
            for t in 0..problem.num_periods {
                if problem.affinity[p][pos][t] > best_val {
                    best_val = problem.affinity[p][pos][t];
                    best_pos = pos;
                    best_t = t;
                }
            }
        }
        let pos_name = problem
            .position_names
            .as_ref()
            .and_then(|n| n.get(best_pos))
            .cloned()
            .unwrap_or_else(|| best_pos.to_string());
        let player_name = problem
            .player_names
            .as_ref()
            .and_then(|n| n.get(p))
            .cloned()
            .unwrap_or_else(|| format!("P{p}"));
        println!(
            "#   {}  best @ pos {}, period {}, affinity {:.2}",
            player_name,
            pos_name,
            best_t + 1,
            best_val
        );
    }
    println!();

    // ─── Policies ───────────────────────────────────────────────────────────
    let policies = [PolicyDesc { name: "random", note: "L3: trivial baseline, ignores fairness" },
        PolicyDesc {
            name: "MDP-memoryless",
            note: "L2 with state=(t,) — Markov but NO history → cannot express fairness",
        },
        PolicyDesc {
            name: "greedy-Hungarian",
            note: "L3: per-period bipartite assignment with fairness pre-fill",
        },
        PolicyDesc {
            name: "LP-relaxation",
            note: "L3: simplex / interior-point on the LP relaxation, Hungarian-rounded",
        },
        PolicyDesc {
            name: "IP/MIP-feasible",
            note: "L3: exact 0/1 IP/MIP branch-and-cut DES; time-boxed, returns best feasible incumbent",
        },
        PolicyDesc {
            name: "MDP-VI-exact",
            note: "L2 with state=(t, prev-bench) — Markov + 1-period memory ⇒ fairness",
        }];
    let filtered: Vec<&PolicyDesc> = if policy_filter.is_empty() {
        policies.iter().collect()
    } else {
        policies
            .iter()
            .filter(|p| p.name.to_lowercase().contains(&policy_filter))
            .collect()
    };

    let should_solve_lp_context = std::env::var("SOCCER_SHOW_LP_BOUND").as_deref() == Ok("1")
        || filtered.iter().any(|p| p.name == "LP-relaxation");
    if should_solve_lp_context {
        // LP upper bound for context.
        let lp = policy_lp_relaxed(&problem);
        println!(
            "# LP relaxation upper bound on total deterministic affinity = {:.4}",
            lp.lp_value
        );
        println!("# (solved via {}, {} iterations)", lp.solver, lp.iters);
        println!();
    } else {
        println!("# LP relaxation upper bound skipped (set SOCCER_SHOW_LP_BOUND=1 to include it)");
        println!();
    }

    let mut aggs: Vec<MatchAggregate> = Vec::new();
    for p in &filtered {
        print!("# Building schedule '{}' ... ", p.name);
        let t0 = Instant::now();
        let schedule = build_schedule(p.name, &problem, seed, &mip_opts, &mut mip_latest);
        let build_ms = t0.elapsed().as_millis();
        if let Some(err) = validate_schedule_structure(&problem, &schedule) {
            println!("FAIL: {}", err);
            continue;
        }
        let eval_res = evaluate_schedule(&problem, &schedule);
        let belief = evaluate_soccer_pomdp_features(
            &problem,
            &schedule,
            &SoccerPOMDPFeatureOptions::default(),
        );
        let stamina_str = match problem.max_consecutive_on_field {
            None => "n/a".to_string(),
            Some(_) if eval_res.stamina_ok => "OK".to_string(),
            Some(_) => format!(
                "VIOLATED({})",
                eval_res.consecutive_on_field_violations.len()
            ),
        };
        println!(
            "affinity={:.2}, fairness={}, stamina={}, beliefFresh={:.3}, build={}ms",
            eval_res.affinity_sum,
            if eval_res.fairness_ok {
                "OK"
            } else {
                "VIOLATED"
            },
            stamina_str,
            belief.mean_expected_fresh_on_field,
            build_ms
        );
        if p.name == "IP/MIP-feasible" {
            if let Some(latest) = &mip_latest {
                // PORT NOTE: `lpAlgorithmUsage` is a `HashMap` in the Rust model
                // (no insertion order), so the comma-joined pairs are emitted in a
                // stable key-sorted order rather than `Object.entries` order.
                let usage_str = if latest.mip.lp_algorithm_usage.is_empty() {
                    "none".to_string()
                } else {
                    let mut entries: Vec<(String, u64)> = latest
                        .mip
                        .lp_algorithm_usage
                        .iter()
                        .map(|(k, v)| (k.as_str().to_string(), *v))
                        .collect();
                    entries.sort_by(|a, b| a.0.cmp(&b.0));
                    entries
                        .iter()
                        .map(|(k, v)| format!("{}={}", k, v))
                        .collect::<Vec<_>>()
                        .join(",")
                };
                let fallback = if latest.used_fallback {
                    format!(
                        ", fallback={}",
                        latest.fallback_reason.as_deref().unwrap_or("")
                    )
                } else {
                    String::new()
                };
                println!(
                    "#   IP/MIP status={}, gap={}, nodes={}, lpSolves={}, lpUsage={}, avgLp={}ms, elapsed={}ms, incumbent={}{}",
                    latest.mip.status.as_str(),
                    to_exponential(latest.mip.gap, 2),
                    latest.mip.nodes_explored,
                    latest.mip.lp_solves,
                    usage_str,
                    format!("{:.2}", latest.mip.performance.avg_lp_solver_ms),
                    jn(latest.mip.elapsed_ms),
                    latest.mip.incumbent_source.as_deref().unwrap_or("none"),
                    fallback
                );
            }
        }
        print!("#   simulating {} matches ... ", num_matches);
        let t_sim = Instant::now();
        let agg = run_many_matches(
            &problem,
            &schedule,
            p.name,
            num_matches,
            seed + 1000,
            &match_opts,
        );
        println!("done in {}ms", t_sim.elapsed().as_millis());
        aggs.push(agg);
    }
    println!();

    // ─── Comparison table ────────────────────────────────────────────────
    let rule = "─".repeat(94);
    println!("# {}", rule);
    println!(
        "# Policy comparison: deterministic affinity (offline) + simulated match outcome (DES)"
    );
    println!("# {}", rule);
    println!(
        "#   {}{}{}{}{}{}{}{}",
        format!("{:<20}", "policy"),
        format!("{:>11}", "affinity"),
        format!("{:>11}", "goal diff"),
        format!("{:>8}", "sd"),
        format!("{:>7}", "gF"),
        format!("{:>7}", "gA"),
        format!("{:>11}", "fairness"),
        format!("{:>13}", "t vs random"),
    );
    let random = aggs.iter().find(|a| a.policy_name == "random");
    for a in &aggs {
        let t = match random {
            Some(r) => welch_t(&a.raw_goal_diffs, &r.raw_goal_diffs),
            None => f64::NAN,
        };
        let tstr = if t.is_nan() {
            "   —".to_string()
        } else {
            format!("{:.2}", t)
        };
        println!(
            "#   {}{}{}{}{}{}{}{}",
            format!("{:<20}", a.policy_name),
            format!("{:>11}", format!("{:.3}", a.affinity_sum_deterministic)),
            format!("{:>11}", format!("{:.3}", a.mean_goal_diff)),
            format!("{:>8}", format!("{:.3}", a.sd_goal_diff)),
            format!("{:>7}", format!("{:.2}", a.mean_goals_for)),
            format!("{:>7}", format!("{:.2}", a.mean_goals_against)),
            format!("{:>11}", if a.fairness_ok { "OK" } else { "VIOLATED" }),
            format!("{:>13}", tstr),
        );
    }
    println!();

    // ─── Player-fairness audit ──────────────────────────────────────────
    println!(
        "# Per-player periods on bench (out of {}):",
        problem.num_periods
    );
    for a in &aggs {
        let counts = a
            .bench_counts
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let name = problem
                    .player_names
                    .as_ref()
                    .and_then(|n| n.get(i))
                    .cloned()
                    .unwrap_or_else(|| format!("P{i}"));
                format!("{}={}", name, c)
            })
            .collect::<Vec<_>>()
            .join(" ");
        let flag = if a.fairness_ok {
            ""
        } else {
            "  ← VIOLATES fairness"
        };
        println!(
            "#   {}  {}{}",
            format!("{:<20}", a.policy_name),
            counts,
            flag
        );
    }
    println!();

    // ─── Stamina audit (only when the cap is active) ────────────────────
    if let Some(m) = problem.max_consecutive_on_field {
        println!(
            "# Stamina audit: longest consecutive on-field run per policy (cap = {} periods):",
            m
        );
        for a in &aggs {
            let eval = evaluate_schedule(&problem, &a.schedule);
            let longest = longest_on_field_run(&problem, &a.schedule);
            let flag = if eval.stamina_ok {
                ""
            } else {
                "  ← EXCEEDS stamina cap"
            };
            println!(
                "#   {}  longest run = {} period(s){}",
                format!("{:<20}", a.policy_name),
                longest,
                flag
            );
        }
        println!("#   (only the LP-relaxation and IP/MIP encode this rolling-window constraint;");
        println!(
            "#    the memoryless/greedy/1-step-MDP baselines cannot express it and may exceed it.)"
        );
        println!();
    }

    // ─── Architectural recap ────────────────────────────────────────────
    println!("# Architectural recap:");
    for p in &filtered {
        println!("#   {} → {}", format!("{:<20}", p.name), p.note);
    }
    println!("#");
    println!(
        "#   Layer 1 (DES): simulateMatchDES runs {} game-minutes per match,",
        problem.num_periods * minutes_per_period
    );
    println!("#                  samples Poisson goal events from on-field affinity,");
    println!("#                  triggers a substitution event at every period boundary.");
    println!(
        "#   Layer 2 (MDP): exact backward induction over {} periods × C({},{}) bench-sets,",
        problem.num_periods, problem.num_players, problem.bench_size
    );
    println!("#                  reward at each (s, a) is the Hungarian-optimal assignment.");
    if let Some(m) = problem.max_consecutive_on_field {
        println!(
            "#   Stamina:       only the LP/IP-MIP encode the rolling \"≤ {m} consecutive periods",
        );
        println!(
            "#                  on field\" constraint; the 1-step-memory MDP and greedy cannot."
        );
    }
    println!(
        "#   POMDP feature: hidden fatigue belief is carried across periods for audit metrics."
    );
    println!("#   Layer 3:       random / greedy-Hungarian / LP-relaxation / IP-MIP / MDP-VI.");
    println!();

    // ─── Optional animation of the best policy ──────────────────────────
    if animate {
        let best = aggs.iter().fold(aggs[0].clone(), |acc, x| {
            if x.mean_goal_diff > acc.mean_goal_diff {
                x.clone()
            } else {
                acc
            }
        });
        println!(
            "# Animating policy '{}' (best mean goal diff = {:.3})",
            best.policy_name, best.mean_goal_diff
        );
        let out_dir = std::path::Path::new("out");
        let _ = std::fs::create_dir_all(out_dir);
        let safe = safe_policy_name(&best.policy_name);
        let frames_path = out_dir.join(format!("soccer-{}.frames.jsonl", safe));
        let html_path = out_dir.join(format!("soccer-{}.html", safe));
        let mut rec = FrameRecorder::new(FrameRecorderOpts {
            frames_path: frames_path.to_string_lossy().into_owned(),
            html_path: Some(html_path.to_string_lossy().into_owned()),
            width: scene::STAGE_W,
            height: scene::STAGE_H,
            fps: Some(6.0),
            title: Some(format!(
                "{}-a-side rotation — {}",
                problem.num_positions, best.policy_name
            )),
            subtitle: Some(format!(
                "GK + {} · affinity {:.2}, goal diff {:.2}",
                outfield
                    .iter()
                    .map(|n| n.to_string())
                    .collect::<Vec<_>>()
                    .join("-"),
                best.affinity_sum_deterministic,
                best.mean_goal_diff
            )),
            background: Some("#0b1220".to_string()),
            ..Default::default()
        })
        .expect("create frame recorder");

        let scene_problem = scene::SoccerProblem {
            num_positions: problem.num_positions,
            num_periods: problem.num_periods as f64,
            formation: formation.clone(),
            player_names: problem.player_names.clone(),
            position_names: problem.position_names.clone(),
        };
        let m = simulate_match_des(
            &problem,
            &best.schedule,
            &MatchSimOptions {
                seed: Some(seed + 1000),
                minutes_per_period: Some(minutes_per_period),
                ..Default::default()
            },
        );
        // Per-period arrangements (who is on the field, who is on the bench).
        let n_periods = problem.num_periods;
        let mpp = minutes_per_period;
        let total_minutes = (n_periods * mpp) as f64;
        let arrangements: Vec<(Vec<usize>, Vec<usize>)> = (0..n_periods)
            .map(|t| {
                let pos: Vec<usize> = best.schedule.assignment[t]
                    .iter()
                    .map(|&x| x as usize)
                    .collect();
                (pos, best.schedule.bench[t].clone())
            })
            .collect();

        // Per-period substitution deltas (vs the previous period). Period 0 is
        // the kick-off lineup (everyone "enters" from the bench).
        let mut entered_p: Vec<Vec<usize>> = vec![Vec::new(); n_periods];
        let mut left_p: Vec<Vec<usize>> = vec![Vec::new(); n_periods];
        let mut subs_p: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n_periods];
        if n_periods > 0 {
            entered_p[0] = arrangements[0].0.clone();
        }
        for t in 1..n_periods {
            let prev_on: std::collections::HashSet<usize> =
                arrangements[t - 1].0.iter().copied().collect();
            let cur_on: std::collections::HashSet<usize> =
                arrangements[t].0.iter().copied().collect();
            let entered: Vec<usize> = arrangements[t]
                .0
                .iter()
                .copied()
                .filter(|p| !prev_on.contains(p))
                .collect();
            let left: Vec<usize> = arrangements[t - 1]
                .0
                .iter()
                .copied()
                .filter(|p| !cur_on.contains(p))
                .collect();
            subs_p[t] = left
                .iter()
                .zip(entered.iter())
                .map(|(&o, &i)| (o, i))
                .collect();
            entered_p[t] = entered;
            left_p[t] = left;
        }

        let all_on_bench: Vec<usize> = (0..problem.num_players).collect();
        let trans_steps = 5usize; // walk-on/off interpolation frames per boundary

        let mut ts: Vec<f64> = Vec::new();
        let mut affs: Vec<f64> = Vec::new();
        let mut g_fs: Vec<f64> = Vec::new();
        let mut g_as: Vec<f64> = Vec::new();
        let mut tick = 0.0_f64;
        let sp = &scene_problem;

        for tr in m.trace.iter() {
            let minute = tr.t;
            let period = tr.period;
            let (cur_pos, cur_bench) = &arrangements[period];

            // Substitution / kick-off walk-on frames at each period boundary.
            if minute % mpp == 0 {
                let (prev_pos, prev_bench) = if period == 0 {
                    (Vec::new(), all_on_bench.clone())
                } else {
                    arrangements[period - 1].clone()
                };
                for k in 0..trans_steps {
                    let alpha = (k as f64 + 1.0) / (trans_steps as f64 + 1.0);
                    let i_f = tick;
                    tick += 1.0;
                    let cur_pos_c = cur_pos.clone();
                    let cur_bench_c = cur_bench.clone();
                    let prev_pos_c = prev_pos.clone();
                    let prev_bench_c = prev_bench.clone();
                    let entered_c = entered_p[period].clone();
                    let left_c = left_p[period].clone();
                    let subs_c = subs_p[period].clone();
                    let gf = tr.goals_for_cum as f64;
                    let ga = tr.goals_against_cum as f64;
                    let aff = tr.affinity_now;
                    let t_f = minute as f64;
                    let period_f = period as f64;
                    rec.frame(t_f, i_f, || {
                        scene::build_soccer_frame(
                            t_f,
                            i_f,
                            &scene::SoccerFrameInput {
                                t: t_f,
                                period: period_f,
                                total_minutes,
                                positions: cur_pos_c,
                                bench: cur_bench_c,
                                prev_positions: prev_pos_c,
                                prev_bench: prev_bench_c,
                                transition: alpha,
                                entered: entered_c,
                                left: left_c,
                                recent_subs: subs_c,
                                goals_for: gf,
                                goals_against: ga,
                                affinity_now: aff,
                                goal_this_tick: None,
                                problem: sp,
                            },
                        )
                    });
                }
            }

            // Settled minute frame.
            let mut goal_this_tick: Option<scene::GoalSide> = None;
            for ev in &m.goal_events {
                if ev.t == minute {
                    goal_this_tick = Some(match ev.side {
                        GoalSide::Us => scene::GoalSide::Us,
                        GoalSide::Them => scene::GoalSide::Them,
                    });
                    break;
                }
            }
            ts.push(minute as f64);
            affs.push(tr.affinity_now);
            g_fs.push(tr.goals_for_cum as f64);
            g_as.push(tr.goals_against_cum as f64);

            let i_f = tick;
            tick += 1.0;
            let cur_pos_c = cur_pos.clone();
            let cur_bench_c = cur_bench.clone();
            let entered_c = entered_p[period].clone();
            let left_c = left_p[period].clone();
            let subs_c = subs_p[period].clone();
            let gf = tr.goals_for_cum as f64;
            let ga = tr.goals_against_cum as f64;
            let aff = tr.affinity_now;
            let t_f = minute as f64;
            let period_f = period as f64;
            rec.frame(t_f, i_f, || {
                scene::build_soccer_frame(
                    t_f,
                    i_f,
                    &scene::SoccerFrameInput {
                        t: t_f,
                        period: period_f,
                        total_minutes,
                        positions: cur_pos_c.clone(),
                        bench: cur_bench_c.clone(),
                        prev_positions: cur_pos_c,
                        prev_bench: cur_bench_c,
                        transition: 1.0,
                        entered: entered_c,
                        left: left_c,
                        recent_subs: subs_c,
                        goals_for: gf,
                        goals_against: ga,
                        affinity_now: aff,
                        goal_this_tick,
                        problem: sp,
                    },
                )
            });
        }
        rec.set_charts(scene::build_soccer_charts(&ts, &affs, &g_fs, &g_as));
        rec.finish().expect("finish recorder");
        println!("# Animation written to {}", html_path.display());

        if let Some(latest) = &mip_latest {
            let solver_solution = to_scene_solution(&latest.mip);
            let solver_frames_path = out_dir.join("soccer-IP-MIP-feasible-solver.frames.jsonl");
            let solver_html_path = out_dir.join("soccer-IP-MIP-feasible-solver.html");
            let mut solver_rec = FrameRecorder::new(FrameRecorderOpts {
                frames_path: solver_frames_path.to_string_lossy().into_owned(),
                html_path: Some(solver_html_path.to_string_lossy().into_owned()),
                width: solver_scene::SOCCER_IPMIP_SOLVER_W,
                height: solver_scene::SOCCER_IPMIP_SOLVER_H,
                fps: Some(5.0),
                title: Some("7v7 IP/MIP Solver Entities".to_string()),
                subtitle: Some(format!(
                    "status {}, LP {}, nodes {}",
                    latest.mip.status.as_str(),
                    latest.mip.lp_algorithm.as_str(),
                    latest.mip.nodes_explored
                )),
                background: Some("#f8fafc".to_string()),
                ..Default::default()
            })
            .expect("create solver frame recorder");
            let total = solver_scene::soccer_ipmip_solver_frame_count(&solver_solution);
            for i in 0..total {
                let i_f = i as f64;
                solver_rec.frame(i_f, i_f, || {
                    solver_scene::build_soccer_ipmip_solver_frame(&solver_solution, i)
                });
            }
            solver_rec.set_charts(solver_scene::build_soccer_ipmip_solver_charts(
                &solver_solution,
            ));
            solver_rec.finish().expect("finish solver recorder");
            println!(
                "# Solver entity animation written to {}",
                solver_html_path.display()
            );
        }
    }
}
