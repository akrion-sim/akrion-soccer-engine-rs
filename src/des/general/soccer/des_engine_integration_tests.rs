//! Cross-crate contract tests for the generic DES primitives used by soccer.
//!
//! Keeping these tests under the soccer supermodule lets them exercise the
//! private production wiring, while every solver/network call still crosses the
//! real `des_engine` path dependency used by the crate.

use super::*;

fn policy_features(role: PlayerRole) -> [f64; SOCCER_POLICY_FEATURE_DIM] {
    soccer_policy_features_for_role(&[0.0; SOCCER_NEURAL_FEATURE_DIM], role, None)
        .try_into()
        .expect("soccer policy feature dimension")
}

#[test]
fn des_lp_reexport_solves_a_bounded_soccer_lane_choice() {
    // Pick exactly one of three lanes. The left lane has the highest value.
    let problem = LPProblem {
        sense: Sense::Max,
        c: vec![9.0, 7.0, 5.0],
        a_ub: None,
        b_ub: None,
        a_eq: Some(vec![vec![1.0, 1.0, 1.0]]),
        b_eq: Some(vec![1.0]),
        lb: Some(vec![Some(0.0); 3]),
        ub: Some(vec![Some(1.0); 3]),
        var_names: Some(vec![
            "left_lane".to_string(),
            "central_lane".to_string(),
            "right_lane".to_string(),
        ]),
        con_names: Some(vec!["one_lane".to_string()]),
    };

    let solution = solve_lp_clarabel(&problem);

    assert_eq!(solution.status, LPStatus::Optimal, "{solution:?}");
    assert_eq!(solution.x.len(), 3);
    assert!((solution.x.iter().sum::<f64>() - 1.0).abs() < 1e-6);
    assert!(solution.x[0] > 1.0 - 1e-6, "{solution:?}");
    assert!((solution.objective - 9.0).abs() < 1e-5);
}

#[test]
fn des_qp_reexport_enforces_soccer_acceleration_bounds() {
    // The unconstrained optimum is [3, -2], but a soccer agent's per-axis
    // acceleration box restricts it to [1, -1].
    let problem = QuadraticProgram {
        q: vec![vec![2.0, 0.0], vec![0.0, 2.0]],
        c: vec![-6.0, 4.0],
        a_ub: None,
        b_ub: None,
        a_eq: None,
        b_eq: None,
        lb: Some(vec![Some(-1.0), Some(-1.0)]),
        ub: Some(vec![Some(1.0), Some(1.0)]),
        var_names: Some(vec!["ax".to_string(), "ay".to_string()]),
    };

    let solution = solve_qp_active_set(&problem, QPOptions::default());

    assert_eq!(solution.status, QPStatus::Optimal, "{solution:?}");
    assert!((solution.x[0] - 1.0).abs() < 1e-8, "{solution:?}");
    assert!((solution.x[1] + 1.0).abs() < 1e-8, "{solution:?}");
    assert_eq!(solution.active_upper_bounds, vec![0]);
    assert_eq!(solution.active_lower_bounds, vec![1]);
}

#[test]
fn des_qp_reexport_reports_non_finite_soccer_inputs_honestly() {
    let problem = QuadraticProgram {
        q: vec![vec![1.0]],
        c: vec![f64::NAN],
        ..QuadraticProgram::default()
    };

    let solution = solve_qp_active_set(&problem, QPOptions::default());

    assert_eq!(solution.status, QPStatus::NumericalError);
    assert!(solution.x.is_empty());
    assert!(solution
        .message
        .as_deref()
        .is_some_and(|message| message.contains("non-finite problem data")));
}

#[test]
fn des_planar_mpc_reexport_returns_finite_bounded_soccer_control() {
    let config = PlanarMpcConfig {
        horizon: 16,
        dt: 0.1,
        q_pos: 10.0,
        q_vel: 1.0,
        qf_pos: 40.0,
        qf_vel: 4.0,
        r: 0.1,
        a_max: 4.0,
        iters: 80,
        obstacle_decay_per_step: 0.9,
    };
    let state = PlanarState {
        pos: [0.0, 0.0],
        vel: [0.0, 0.0],
    };
    let target = PlanarReference::through([12.0, 0.0], [6.0, 0.0]);
    let mut controller = PlanarPointMassMpc::new(config).expect("valid soccer MPC config");

    let acceleration = controller.control(state, &[target]);
    let magnitude = acceleration[0].hypot(acceleration[1]);

    assert!(acceleration.iter().all(|value| value.is_finite()));
    assert!(acceleration[0] > 0.0, "{acceleration:?}");
    assert!(acceleration[1].abs() < 1e-9, "{acceleration:?}");
    assert!(magnitude <= config.a_max + 1e-9, "{acceleration:?}");
}

#[test]
fn soccer_role_actor_keeps_illegal_des_logits_frozen() {
    let role = PlayerRole::Midfielder;
    let features = policy_features(role);
    let pass_index = soccer_policy_action_index("pass").expect("pass action");
    let shoot_index = soccer_policy_action_index("shoot").expect("shoot action");
    let mut legal = vec![false; SOCCER_POLICY_ACTIONS.len()];
    legal[pass_index] = true;
    legal[shoot_index] = true;

    let mut actor = SoccerPolicyHead::new(91_337);
    let before_distribution = actor
        .action_distribution_for_role_with_mask(&features, role, &legal)
        .expect("masked midfielder distribution");
    assert!((before_distribution.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    assert!(before_distribution
        .iter()
        .enumerate()
        .all(|(index, probability)| legal[index] || *probability == 0.0));

    let role_head = actor
        .role_head_for_role_mut(role)
        .expect("midfielder role head");
    let output_layer = role_head
        .network
        .layers
        .last()
        .expect("policy output layer");
    let frozen_illegal_rows = output_layer
        .weights
        .iter()
        .zip(&output_layer.biases)
        .enumerate()
        .filter(|(index, _)| !legal[*index])
        .map(|(index, (weights, bias))| (index, weights.clone(), *bias))
        .collect::<Vec<_>>();

    for _ in 0..32 {
        let result = role_head
            .network
            .train_policy_gradient_sample_masked(&features, pass_index, &legal, 1.0, 0.0, 0.1, 5.0);
        assert!(result.applied);
    }

    let after_distribution = role_head
        .action_distribution_with_mask(&features, Some(&legal))
        .expect("trained masked distribution");
    assert!(after_distribution[pass_index] > before_distribution[pass_index]);
    assert!(after_distribution
        .iter()
        .enumerate()
        .all(|(index, probability)| legal[index] || *probability == 0.0));

    let output_layer = role_head
        .network
        .layers
        .last()
        .expect("policy output layer after training");
    for (index, weights, bias) in frozen_illegal_rows {
        assert_eq!(output_layer.weights[index], weights);
        assert_eq!(output_layer.biases[index], bias);
    }
}

#[test]
fn soccer_role_actor_rejects_malformed_des_action_masks_without_panicking() {
    let actor = SoccerPolicyHead::new(44_021);
    let role = PlayerRole::Forward;
    let features = policy_features(role);
    let wrong_length = vec![true; SOCCER_POLICY_ACTIONS.len() - 1];
    let no_legal_actions = vec![false; SOCCER_POLICY_ACTIONS.len()];

    assert!(actor
        .action_distribution_for_role_with_mask(&features, role, &wrong_length)
        .is_none());
    assert!(actor
        .action_distribution_for_role_with_mask(&features, role, &no_legal_actions)
        .is_none());
}
