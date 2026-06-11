//! Port of src/des/test/soccer-test.ts
//!
//! Unit tests for the 7v7 rotation problem. Spans soccer-rotation, hungarian,
//! and lp. The original used an ad-hoc expect()/close() tally; here each TS
//! group becomes one or more `#[test]` functions using `assert!`/`assert_eq!`.
#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use crate::des::general::hungarian::{hungarian, AssignmentDirection};
    use crate::des::general::ip_mip_des::{ConcreteLpRelaxationAlgorithm, LpRelaxationAlgorithm};
    use crate::des::general::lp::{solve_lp_internal, InternalSimplexOptions, LPStatus};
    use crate::des::general::soccer_rotation::{
        build_sample_soccer_problem, build_soccer_ipmip, build_soccer_lp, evaluate_schedule,
        evaluate_soccer_pomdp_features, policy_greedy_hungarian, policy_ipmip_feasible,
        policy_lp_relaxed, policy_mdp_vi, policy_mdp_vi_memoryless, policy_random_schedule,
        run_many_matches, schedule_from_soccer_ipmip_vector, simulate_match_des,
        validate_schedule_structure, AffinityBuilderOptions, GreedyHungarianOptions,
        MatchSimOptions, SoccerIPMIPPolicyOptions, SoccerPOMDPFeatureOptions,
    };

    const TOL: f64 = 1e-9;

    fn problem(seed: u32) -> crate::des::general::soccer_rotation::SoccerProblem {
        build_sample_soccer_problem(&AffinityBuilderOptions {
            seed: Some(seed),
            ..Default::default()
        })
    }

    // -------------------------------------------------------------------------
    // Group 1 — Hungarian algorithm correctness
    // -------------------------------------------------------------------------
    #[test]
    fn hungarian_textbook_min() {
        let cost = vec![
            vec![4.0, 1.0, 3.0],
            vec![2.0, 0.0, 5.0],
            vec![3.0, 2.0, 2.0],
        ];
        let r = hungarian(&cost, AssignmentDirection::Min);
        assert!((r.total - 5.0).abs() < TOL);
        assert_eq!(r.rows[0], 1);
        assert_eq!(r.rows[1], 0);
        assert_eq!(r.rows[2], 2);
    }

    #[test]
    fn hungarian_textbook_max() {
        let w = vec![
            vec![4.0, 1.0, 3.0],
            vec![2.0, 0.0, 5.0],
            vec![3.0, 2.0, 2.0],
        ];
        let r = hungarian(&w, AssignmentDirection::Max);
        assert!((r.total - 11.0).abs() < TOL);
        assert_eq!(r.rows[1], 2);
    }

    #[test]
    fn hungarian_rectangular_max() {
        let w = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![5.0, 4.0, 3.0, 2.0, 1.0],
            vec![3.0, 3.0, 3.0, 3.0, 3.0],
        ];
        let r = hungarian(&w, AssignmentDirection::Max);
        assert!((r.total - 13.0).abs() < TOL);
        assert!(r.rows.iter().all(|&j| j >= 0));
        let mut seen = r.rows.clone();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), r.rows.len());
    }

    #[test]
    fn hungarian_empty_input() {
        let empty: Vec<Vec<f64>> = Vec::new();
        let r = hungarian(&empty, AssignmentDirection::Min);
        assert!(r.rows.is_empty() && r.total == 0.0);
    }

    // -------------------------------------------------------------------------
    // Group 2 — Problem builder produces well-formed instance
    // -------------------------------------------------------------------------
    #[test]
    fn problem_builder_well_formed() {
        let p = problem(42);
        assert_eq!(p.num_players, 12);
        assert_eq!(p.num_positions, 7);
        assert_eq!(p.num_periods, 4);
        assert_eq!(p.bench_size, 5);
        assert_eq!(p.affinity.len(), 12);
        assert_eq!(p.affinity[0].len(), 7);
        assert_eq!(p.affinity[0][0].len(), 4);
        assert!(p.affinity.iter().all(|p1| p1
            .iter()
            .all(|p2| p2.iter().all(|&v| (0.0..=1.0).contains(&v)))));
        assert_eq!(p.player_names.as_ref().map(|v| v.len()), Some(12));
        assert_eq!(p.position_names.as_ref().map(|v| v.len()), Some(7));
    }

    // -------------------------------------------------------------------------
    // Group 3 — Schedule structure validation
    // -------------------------------------------------------------------------
    #[test]
    fn schedule_structure_validation() {
        let p = problem(1);
        let random = policy_random_schedule(&p, 7);
        assert!(validate_schedule_structure(&p, &random).is_none());

        let greedy = policy_greedy_hungarian(
            &p,
            &GreedyHungarianOptions {
                fairness_aware: Some(true),
            },
        );
        assert!(validate_schedule_structure(&p, &greedy).is_none());

        let mdp = policy_mdp_vi(&p);
        assert!(validate_schedule_structure(&p, &mdp.to_schedule()).is_none());
        for t in 0..p.num_periods {
            let on_field = mdp.assignment[t].len();
            let bench = mdp.bench[t].len();
            assert_eq!(on_field + bench, p.num_players);
        }
    }

    // -------------------------------------------------------------------------
    // Group 4 — Fairness behaviour matches state-augmentation theory
    // -------------------------------------------------------------------------
    #[test]
    fn fairness_state_augmentation() {
        let p = problem(2026);
        let memoryless = policy_mdp_vi_memoryless(&p);
        let augmented = policy_mdp_vi(&p);
        let mem_eval = evaluate_schedule(&p, &memoryless.schedule);
        let aug_eval = evaluate_schedule(&p, &augmented.to_schedule());

        // Memoryless objective is an upper bound on the augmented objective.
        assert!(memoryless.value + 1e-9 >= aug_eval.affinity_sum);
        // The augmented MDP respects fairness.
        assert!(aug_eval.fairness_ok);
        // The memoryless MDP violates fairness for seed 2026.
        assert!(!mem_eval.fairness_ok);
    }

    // -------------------------------------------------------------------------
    // Group 5 — LP relaxation upper-bounds the MDP optimum
    // -------------------------------------------------------------------------
    #[test]
    fn lp_relaxation_upper_bounds_mdp() {
        let p = problem(1234);
        let lp = policy_lp_relaxed(&p);
        let mdp = policy_mdp_vi(&p);
        assert!(lp.lp_value + 1e-9 >= mdp.optimal_value);
        assert!((lp.lp_value - mdp.optimal_value).abs() < 1e-6);
        assert!(!lp.solver.is_empty());
    }

    // -------------------------------------------------------------------------
    // Group 6 — DES match simulator basic invariants
    // -------------------------------------------------------------------------
    #[test]
    fn des_match_simulator_invariants() {
        let p = problem(3);
        let schedule = policy_mdp_vi(&p).to_schedule();
        let r = simulate_match_des(
            &p,
            &schedule,
            &MatchSimOptions {
                seed: Some(11),
                ..Default::default()
            },
        );
        assert!(r.goals_for >= 0);
        assert!(r.goals_against >= 0);
        assert_eq!(r.goal_differential, r.goals_for - r.goals_against);
        assert_eq!(r.trace.len(), p.num_periods * 20);
        assert_eq!(r.sub_events.len(), p.num_periods);
        assert_eq!(r.sub_events[0].t, 0);
        let direct = evaluate_schedule(&p, &schedule).affinity_sum;
        assert!((r.affinity_sum_deterministic - direct).abs() < TOL);
    }

    // -------------------------------------------------------------------------
    // Group 7 — Reproducibility
    // -------------------------------------------------------------------------
    #[test]
    fn reproducibility_same_seed() {
        let p = problem(99);
        let sched = policy_mdp_vi(&p).to_schedule();
        let r1 = simulate_match_des(
            &p,
            &sched,
            &MatchSimOptions {
                seed: Some(7),
                ..Default::default()
            },
        );
        let r2 = simulate_match_des(
            &p,
            &sched,
            &MatchSimOptions {
                seed: Some(7),
                ..Default::default()
            },
        );
        assert_eq!(r1.goals_for, r2.goals_for);
        assert_eq!(r1.goals_against, r2.goals_against);
        assert_eq!(r1.goal_events.len(), r2.goal_events.len());
    }

    // -------------------------------------------------------------------------
    // Group 8 — LP shape is correct
    // -------------------------------------------------------------------------
    #[test]
    fn lp_shape_is_correct() {
        let p = problem(5);
        let lp = build_soccer_lp(&p);
        assert_eq!(lp.c.len(), 336);
        assert_eq!(lp.a_eq.as_ref().map(|m| m.len()).unwrap_or(0), 28);
        assert_eq!(lp.a_ub.as_ref().map(|m| m.len()).unwrap_or(0), 84);
        let sol = solve_lp_internal(&lp, &InternalSimplexOptions::default());
        assert_eq!(sol.status, LPStatus::Optimal);
    }

    // -------------------------------------------------------------------------
    // Group 9 — IP/MIP program finds a feasible 7v7 schedule
    // -------------------------------------------------------------------------
    #[test]
    fn ipmip_finds_feasible_schedule() {
        let p = problem(1234);
        let model = build_soccer_ipmip(&p);
        assert_eq!(model.ip.c.len(), 336);
        assert_eq!(model.ip.a.len(), 140);
        assert!(model.ip.b.iter().any(|&v| v < 0.0));

        let opts = SoccerIPMIPPolicyOptions {
            time_limit_ms: Some(10_000.0),
            max_nodes: Some(100),
            max_ticks: Some(1_000),
            lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                ConcreteLpRelaxationAlgorithm::InternalSimplex,
            )),
            max_cut_rounds: Some(0),
            fallback_to_mdp: Some(false),
            ..Default::default()
        };
        let mip = policy_ipmip_feasible(&p, &opts);
        assert!(
            !mip.used_fallback,
            "fallback_reason = {:?}",
            mip.fallback_reason
        );
        assert!(validate_schedule_structure(&p, &mip.schedule).is_none());
        let eval_res = evaluate_schedule(&p, &mip.schedule);
        assert!(eval_res.fairness_ok);
        let decoded = schedule_from_soccer_ipmip_vector(&p, &model, &mip.mip.x, 0.5);
        assert!(decoded.is_some());
        let mdp = policy_mdp_vi(&p);
        assert!((eval_res.affinity_sum - mdp.optimal_value).abs() < 1e-6);
    }

    // -------------------------------------------------------------------------
    // Group 10 — POMDP-style hidden-fatigue feature trace
    // -------------------------------------------------------------------------
    #[test]
    fn pomdp_feature_trace() {
        let p = problem(21);
        let sched = policy_mdp_vi(&p).to_schedule();
        let belief =
            evaluate_soccer_pomdp_features(&p, &sched, &SoccerPOMDPFeatureOptions::default());
        assert_eq!(belief.per_period.len(), p.num_periods);
        assert!(belief
            .final_fresh_probability
            .iter()
            .all(|&v| (0.0..=1.0).contains(&v)));
        assert!(
            belief.mean_expected_fresh_on_field.is_finite()
                && belief.mean_expected_fresh_on_field > 0.0
        );
        assert!(
            belief.mean_expected_lineup_reliability.is_finite()
                && belief.mean_expected_lineup_reliability > 0.0
        );
    }

    // -------------------------------------------------------------------------
    // Group 11 — runManyMatches aggregates correctly
    // -------------------------------------------------------------------------
    #[test]
    fn run_many_matches_aggregates() {
        let p = problem(17);
        let sched = policy_mdp_vi(&p).to_schedule();
        let agg = run_many_matches(&p, &sched, "mdp", 5, 100, &MatchSimOptions::default());
        assert_eq!(agg.policy_name, "mdp");
        assert_eq!(agg.raw_goal_diffs.len(), 5);
        assert!(agg.fairness_ok);
        let computed = agg.raw_goal_diffs.iter().sum::<f64>() / 5.0;
        assert!((agg.mean_goal_diff - computed).abs() < TOL);
    }
}
