//! Port of `src/des/runners/validate-soccer.ts`.
//!
//! Validates the 7v7 rotation problem: LP relaxation tightness vs MDP-VI, state
//! augmentation for fairness, cross-solver LP agreement, Hungarian dominance, and
//! a stochastic-match Welch-t study. Driver → [`run`].
//!
//! PORT NOTES:
//!   * Uses the real Rust soccer-rotation model, LP relaxation, exact MDP-VI,
//!     greedy/random schedules, DES match simulator, and LP solver adapters.

#![allow(dead_code)]

use crate::des::general::lp::{
    solve_lp_internal as solve_lp_internal_model, ExternalSolver, ExternalSolverOptions,
    InternalSimplexOptions, LPProblem,
};
use crate::des::general::lp_des::{solve_lp_via_des as solve_lp_via_des_model, DESSimplexOptions};
use crate::des::general::soccer_rotation::{
    build_sample_soccer_problem as build_sample_soccer_problem_model,
    build_soccer_lp as build_soccer_lp_model, evaluate_schedule as evaluate_schedule_model,
    policy_greedy_hungarian as policy_greedy_hungarian_model,
    policy_lp_relaxed as policy_lp_relaxed_model, policy_mdp_vi as policy_mdp_vi_model,
    policy_mdp_vi_memoryless as policy_mdp_vi_memoryless_model,
    policy_random_schedule as policy_random_schedule_model,
    run_many_matches as run_many_matches_model,
    validate_schedule_structure as validate_schedule_structure_model, welch_t as welch_t_model,
    AffinityBuilderOptions, GreedyHungarianOptions, LPRelaxedScheduleResult, MatchAggregate,
    MatchSimOptions, Schedule, ScheduleEvaluation, SoccerProblem,
};
use crate::des::shared::transform::Transform;

type ScheduleEval = ScheduleEvaluation;
type LpRelaxedResult = LPRelaxedScheduleResult;
type ManyMatchesResult = MatchAggregate;
type LpProblem = LPProblem;

#[derive(Clone, Debug)]
struct MdpViResult {
    optimal_value: f64,
    schedule: Schedule,
}

#[derive(Clone, Debug)]
struct MemorylessResult {
    schedule: Schedule,
}

#[derive(Clone, Debug, Default)]
struct LpSolution {
    status: String,
    objective: f64,
}

fn build_sample_soccer_problem(seed: u64) -> SoccerProblem {
    build_sample_soccer_problem_model(&AffinityBuilderOptions {
        seed: Some(seed as u32),
        ..Default::default()
    })
}
fn policy_lp_relaxed(p: &SoccerProblem) -> LpRelaxedResult {
    policy_lp_relaxed_model(p)
}
fn policy_mdp_vi(p: &SoccerProblem) -> MdpViResult {
    let result = policy_mdp_vi_model(p);
    MdpViResult {
        optimal_value: result.optimal_value,
        schedule: result.to_schedule(),
    }
}
fn policy_mdp_vi_memoryless(p: &SoccerProblem) -> MemorylessResult {
    let result = policy_mdp_vi_memoryless_model(p);
    MemorylessResult {
        schedule: result.schedule,
    }
}
fn policy_random_schedule(p: &SoccerProblem, seed: usize) -> Schedule {
    policy_random_schedule_model(p, seed as u32)
}
fn policy_greedy_hungarian(p: &SoccerProblem, fairness_aware: bool) -> Schedule {
    policy_greedy_hungarian_model(
        p,
        &GreedyHungarianOptions {
            fairness_aware: Some(fairness_aware),
        },
    )
}
fn evaluate_schedule(p: &SoccerProblem, s: &Schedule) -> ScheduleEval {
    evaluate_schedule_model(p, s)
}
fn validate_schedule_structure(p: &SoccerProblem, s: &Schedule) -> Option<String> {
    validate_schedule_structure_model(p, s)
}
fn build_soccer_lp(p: &SoccerProblem) -> LpProblem {
    build_soccer_lp_model(p)
}
fn run_many_matches(
    p: &SoccerProblem,
    s: &Schedule,
    name: &str,
    num_matches: usize,
    seed: u64,
) -> ManyMatchesResult {
    run_many_matches_model(
        p,
        s,
        name,
        num_matches,
        seed as u32,
        &MatchSimOptions::default(),
    )
}
fn welch_t(a: &[f64], b: &[f64]) -> f64 {
    welch_t_model(a, b)
}

fn lp_solution(status: crate::des::general::lp::LPStatus, objective: f64) -> LpSolution {
    LpSolution {
        status: status.as_str().to_string(),
        objective,
    }
}
fn solve_lp_internal(lp: &LpProblem) -> LpSolution {
    let sol = solve_lp_internal_model(lp, &InternalSimplexOptions::default());
    lp_solution(sol.status, sol.objective)
}
fn solve_lp_via_des(lp: &LpProblem) -> LpSolution {
    let sol = solve_lp_via_des_model(lp, &DESSimplexOptions::default());
    lp_solution(sol.status, sol.objective)
}
fn solve_lp_external(lp: &LpProblem, method: &str) -> LpSolution {
    let sol = ExternalSolver::new(ExternalSolverOptions {
        method: Some(method.to_string()),
        ..Default::default()
    })
    .transform(lp.clone());
    lp_solution(sol.status, sol.objective)
}

// =============================================================================
// Driver.
// =============================================================================

struct Checker {
    pass: u32,
    fail: u32,
}

impl Checker {
    fn new() -> Self {
        Checker { pass: 0, fail: 0 }
    }
    fn check(&mut self, label: &str, ok: bool, detail: &str) {
        let tail = if detail.is_empty() {
            String::new()
        } else {
            format!("  — {}", detail)
        };
        println!(
            "{}  {}{}",
            if ok { "  PASS" } else { "  FAIL" },
            label,
            tail
        );
        if ok {
            self.pass += 1;
        } else {
            self.fail += 1;
        }
    }
}

/// `validate-soccer.ts` top-level driver.
pub fn run() {
    let mut c = Checker::new();
    let problem = build_sample_soccer_problem(4242);

    println!("\nStudy 1 — LP relaxation upper bound = MDP-VI exact optimum");
    {
        let lp = policy_lp_relaxed(&problem);
        let mdp = policy_mdp_vi(&problem);
        let mdp_eval = evaluate_schedule(&problem, &mdp.schedule);
        println!("    LP upper bound       = {:.6}", lp.lp_value);
        println!("    MDP-VI optimum       = {:.6}", mdp.optimal_value);
        println!(
            "    LP-rounded affinity  = {:.6}",
            evaluate_schedule(&problem, &lp.schedule).affinity_sum
        );
        println!("    MDP-VI affinity      = {:.6}", mdp_eval.affinity_sum);
        c.check(
            "LP upper bound ≥ MDP optimum (relaxation property)",
            lp.lp_value + 1e-9 >= mdp.optimal_value,
            &format!("Δ = {:.6}", lp.lp_value - mdp.optimal_value),
        );
        c.check(
            "MDP optimum = LP upper bound to 1e-6 (LP is tight ⇒ no integrality gap)",
            (lp.lp_value - mdp.optimal_value).abs() < 1e-6,
            &format!("|Δ| = {:.2e}", (lp.lp_value - mdp.optimal_value).abs()),
        );
        c.check(
            "MDP-VI affinity = MDP optimum (consistent)",
            (mdp_eval.affinity_sum - mdp.optimal_value).abs() < 1e-9,
            "",
        );
        c.check(
            "MDP-VI is structurally valid",
            validate_schedule_structure(&problem, &mdp.schedule).is_none(),
            "",
        );
        c.check(
            "MDP-VI satisfies fairness constraint",
            mdp_eval.fairness_ok,
            &format!("{} violations", mdp_eval.fairness_violations.len()),
        );
    }

    println!("\nStudy 2 — State augmentation: memoryless MDP violates, augmented MDP enforces");
    {
        let memoryless = policy_mdp_vi_memoryless(&problem);
        let mem_eval = evaluate_schedule(&problem, &memoryless.schedule);
        let augmented = policy_mdp_vi(&problem);
        let aug_eval = evaluate_schedule(&problem, &augmented.schedule);
        println!(
            "    state = (t,)              affinity = {:.4}, fairness = {}",
            mem_eval.affinity_sum, mem_eval.fairness_ok
        );
        println!(
            "    state = (t, prev_bench)   affinity = {:.4}, fairness = {}",
            aug_eval.affinity_sum, aug_eval.fairness_ok
        );
        println!(
            "    cost of fairness          = {:.4}",
            mem_eval.affinity_sum - aug_eval.affinity_sum
        );
        c.check(
            "memoryless MDP affinity > augmented MDP affinity (relaxation is unconstrained)",
            mem_eval.affinity_sum > aug_eval.affinity_sum,
            &format!("Δ = {:.4}", mem_eval.affinity_sum - aug_eval.affinity_sum),
        );
        c.check(
            "memoryless MDP VIOLATES fairness (no history in state ⇒ constraint inexpressible)",
            !mem_eval.fairness_ok,
            &format!("{} violations", mem_eval.fairness_violations.len()),
        );
        c.check(
            "augmented MDP RESPECTS fairness (history in state ⇒ constraint enforced)",
            aug_eval.fairness_ok,
            &format!(
                "0 violations expected, got {}",
                aug_eval.fairness_violations.len()
            ),
        );
        c.check(
            "augmented MDP optimum within the LP upper bound",
            aug_eval.affinity_sum <= policy_lp_relaxed(&problem).lp_value + 1e-9,
            "",
        );
    }

    println!("\nStudy 3 — Cross-solver consistency on the soccer LP");
    {
        let lp = build_soccer_lp(&problem);
        println!(
            "    LP shape: {} variables, {} equality + {} inequality rows",
            lp.c.len(),
            lp.a_eq.as_ref().map(|m| m.len()).unwrap_or(0),
            lp.a_ub.as_ref().map(|m| m.len()).unwrap_or(0)
        );
        let s_int = solve_lp_internal(&lp);
        let s_des = solve_lp_via_des(&lp);
        let s_ds = solve_lp_external(&lp, "highs-ds");
        let s_ipm = solve_lp_external(&lp, "highs-ipm");
        println!(
            "    internal simplex     status={} obj={:.8}",
            s_int.status, s_int.objective
        );
        println!(
            "    DES-engine simplex   status={} obj={:.8}",
            s_des.status, s_des.objective
        );
        println!(
            "    external:highs-ds    status={} obj={:.8}",
            s_ds.status, s_ds.objective
        );
        println!(
            "    external:highs-ipm   status={} obj={:.8}",
            s_ipm.status, s_ipm.objective
        );
        let objs: Vec<f64> = [&s_int, &s_des, &s_ds, &s_ipm]
            .iter()
            .filter(|s| s.status == "optimal")
            .map(|s| s.objective)
            .collect();
        if objs.len() < 2 {
            c.check(
                "At least two LP solvers available",
                false,
                "cannot cross-validate",
            );
        } else {
            let ref_obj = objs[0];
            let max_diff = objs
                .iter()
                .map(|o| (o - ref_obj).abs())
                .fold(0.0_f64, f64::max);
            c.check(
                "all available solvers agree on the LP objective to 1e-5",
                max_diff < 1e-5,
                &format!("max |Δ| = {:.2e}", max_diff),
            );
        }
    }

    println!("\nStudy 4 — Per-period Hungarian dominates random");
    {
        let random = evaluate_schedule(&problem, &policy_random_schedule(&problem, 7));
        let greedy = evaluate_schedule(&problem, &policy_greedy_hungarian(&problem, true));
        let mdp = evaluate_schedule(&problem, &policy_mdp_vi(&problem).schedule);
        println!("    random           affinity = {:.3}", random.affinity_sum);
        println!("    greedy-Hungarian affinity = {:.3}", greedy.affinity_sum);
        println!("    MDP-VI exact     affinity = {:.3}", mdp.affinity_sum);
        c.check(
            "greedy-Hungarian > random (per-period assignment beats permutation)",
            greedy.affinity_sum > random.affinity_sum + 1.0,
            &format!("Δ = {:.3}", greedy.affinity_sum - random.affinity_sum),
        );
        c.check(
            "MDP-VI ≥ greedy-Hungarian (multi-period optimum ≥ greedy)",
            mdp.affinity_sum + 1e-9 >= greedy.affinity_sum,
            &format!("Δ = {:.3}", mdp.affinity_sum - greedy.affinity_sum),
        );
    }

    println!("\nStudy 5 — DES match: MDP-VI vs random goal differential (Welch t-test)");
    {
        let num_matches = 50;
        let random = run_many_matches(
            &problem,
            &policy_random_schedule(&problem, 11),
            "random",
            num_matches,
            999,
        );
        let mdp = run_many_matches(
            &problem,
            &policy_mdp_vi(&problem).schedule,
            "mdp",
            num_matches,
            999,
        );
        let greedy = run_many_matches(
            &problem,
            &policy_greedy_hungarian(&problem, true),
            "greedy",
            num_matches,
            999,
        );
        println!(
            "    random:  goal diff = {:.2} ± {:.2}",
            random.mean_goal_diff, random.sd_goal_diff
        );
        println!(
            "    greedy:  goal diff = {:.2} ± {:.2}",
            greedy.mean_goal_diff, greedy.sd_goal_diff
        );
        println!(
            "    MDP-VI:  goal diff = {:.2} ± {:.2}",
            mdp.mean_goal_diff, mdp.sd_goal_diff
        );
        let t_mdp_vs_random = welch_t(&mdp.raw_goal_diffs, &random.raw_goal_diffs);
        let t_greedy_vs_random = welch_t(&greedy.raw_goal_diffs, &random.raw_goal_diffs);
        println!("    Welch t  random→MDP    = {:.2}", t_mdp_vs_random);
        println!("    Welch t  random→greedy = {:.2}", t_greedy_vs_random);
        c.check(
            "MDP-VI > random in goal differential (Welch-t > 3)",
            t_mdp_vs_random > 3.0,
            &format!("t = {:.2}", t_mdp_vs_random),
        );
        c.check(
            "greedy > random in goal differential (Welch-t > 3)",
            t_greedy_vs_random > 3.0,
            &format!("t = {:.2}", t_greedy_vs_random),
        );
        c.check("MDP-VI fairness OK in DES match", mdp.fairness_ok, "");
        c.check(
            "greedy-Hungarian fairness OK in DES match",
            greedy.fairness_ok,
            "",
        );
    }

    println!(
        "\n{} checks: {} passed, {} failed",
        c.pass + c.fail,
        c.pass,
        c.fail
    );
    if c.fail > 0 {
        std::process::exit(1);
    }
}
