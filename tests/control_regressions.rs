use akrion_soccer_engine::{
    collision::{local_planes, swept_contact, Neighbor},
    locomotion::{plan, safe_braking, solve_reference, Config, Decision, Halfspace, MotionGoal, QpStep, SolverStatus, State},
    math::Vec2,
    policy::{observe, select_motion, ObservationDelay, Player, PolicyFeedback, TacticalChoice, Vision, World}
};

fn config() -> Config {
    Config {
        dt: 0.1, horizon: 5, max_speed: 7.0, max_acceleration: 4.0,
        contact_radius: 0.35, collision_margin: 0.1,
    }
}
fn state(position: Vec2, velocity: Vec2) -> State { State { position, velocity } }
fn goal(position: Vec2) -> MotionGoal { MotionGoal { position, velocity: Vec2::ZERO } }

#[test]
fn translational_motion_is_finite_and_obeys_euclidean_accel_and_speed_limits() {
    let a = plan(state(Vec2::ZERO, Vec2::ZERO), goal(Vec2::new(10.0, 3.0)), config(), &[]).unwrap();
    assert_eq!(a.decision, Decision::Projected);
    assert_eq!(a.solver_status, SolverStatus::Feasible);
    assert_eq!(a.predicted.len(), config().horizon);
    assert!(a.first_acceleration.norm() <= config().max_acceleration + 1e-8);
    assert!(a.predicted.iter().all(|s| s.velocity.norm() <= config().max_speed + 1e-8));
    assert!(a.predicted[0].position.norm() > 0.0);
    // The controller does NOT teleport to the destination.
    assert!(a.predicted[0].position.x < 10.0);
}

#[test]
fn invalid_and_nonfinite_state_is_rejected_without_silent_clamping() {
    assert!(plan(state(Vec2::new(f64::NAN, 0.0), Vec2::ZERO), goal(Vec2::ZERO), config(), &[]).is_err());
    assert!(plan(state(Vec2::ZERO, Vec2::ZERO), goal(Vec2::new(f64::INFINITY, 0.0)), config(), &[]).is_err());
    let mut bad = config(); bad.dt = 0.0;
    assert!(plan(state(Vec2::ZERO, Vec2::ZERO), goal(Vec2::ZERO), bad, &[]).is_err());
    bad = config(); bad.horizon = 0;
    assert!(plan(state(Vec2::ZERO, Vec2::ZERO), goal(Vec2::ZERO), bad, &[]).is_err());
}

#[test]
fn infeasible_and_malformed_solver_paths_fail_closed_with_bounded_braking() {
    let bad = QpStep { hessian_scalar: f64::NAN, linear: Vec2::ZERO, inequalities: vec![] };
    assert_eq!(solve_reference(&bad).0, SolverStatus::NonFinite);
    let impossible = QpStep {
        hessian_scalar: 1.0, linear: Vec2::ZERO,
        inequalities: vec![
            Halfspace { normal: Vec2::new(1.0, 0.0), upper: -2.0 },
            Halfspace { normal: Vec2::new(-1.0, 0.0), upper: -2.0 },
        ],
    };
    assert_eq!(solve_reference(&impossible).0, SolverStatus::Infeasible);
    let s = state(Vec2::ZERO, Vec2::new(5.0, 0.0));
    let braking = safe_braking(s, config());
    assert!(braking.x < 0.0 && braking.norm() <= config().max_acceleration + 1e-9);
    let overlapped = [Neighbor {
        id: 1, position: Vec2::ZERO, velocity: Vec2::ZERO,
        radius: 0.35, intentional_contact: false,
    }];
    let p = plan(s, goal(Vec2::new(4.0, 0.0)), config(), &overlapped).unwrap();
    assert_eq!(p.decision, Decision::SafeBraking);
    assert_eq!(p.solver_status, SolverStatus::Infeasible);
    assert_eq!(p.predicted.len(), 1);
    assert!(p.predicted[0].position.finite());
}

#[test]
fn local_contact_plane_restricts_approach_without_global_collision_claim() {
    let c = config();
    let neighbor = Neighbor {
        id: 9, position: Vec2::ZERO, velocity: Vec2::ZERO,
        radius: 0.35, intentional_contact: false,
    };
    let p = plan(
        state(Vec2::new(-0.81, 0.0), Vec2::ZERO),
        goal(Vec2::new(5.0, 0.0)), c, &[neighbor]
    ).unwrap();
    assert!(p.first_acceleration.norm() <= c.max_acceleration + 1e-9);
    if p.decision == Decision::Projected {
        let relative = p.predicted[0].position - neighbor.position;
        assert!(relative.norm() >= 0.35 + 0.35 + c.collision_margin - 1e-7);
    } else {
        // Explicit failure and braking, NOT a claimed safe separation.
        assert_eq!(p.solver_status, SolverStatus::Infeasible);
    }
}
#[test]
fn broad_phase_is_order_independent_and_omits_distant_players() {
    let c = config();
    let near = Neighbor {
        id: 3, position: Vec2::new(1.0, 0.0), velocity: Vec2::ZERO,
        radius: 0.35, intentional_contact: false,
    };
    let far = Neighbor { id: 42, position: Vec2::new(300.0, 0.0), ..near };
    let first = local_planes(Vec2::ZERO, 1, c, &[far, near]).unwrap();
    let second = local_planes(Vec2::ZERO, 1, c, &[near, far]).unwrap();
    assert_eq!(first.len(), 1);
    assert_eq!(second.len(), 1);
    assert!((first[0].upper - second[0].upper).abs() < 1e-12);
    assert_eq!(first[0].normal, second[0].normal);
}
#[test]
fn tunneling_head_on_crossing_chase_and_intentional_contact_are_reported_to_physics() {
    let head_on = swept_contact(
        Vec2::new(-2.0, 0.0), Vec2::new(10.0, 0.0), 0.3,
        Vec2::new(2.0, 0.0), Vec2::new(-10.0, 0.0), 0.3, 0.3, false
    ).unwrap();
    assert!((head_on.time - 0.2).abs() < 1e-9);
    assert!(head_on.minimum_distance < 1e-9);
    let crossing = swept_contact(
        Vec2::new(-1.0, 0.0), Vec2::new(4.0, 0.0), 0.25,
        Vec2::new(0.0, -1.0), Vec2::new(0.0, 4.0), 0.25, 0.5, false
    ).unwrap();
    assert!(crossing.minimum_distance < 1e-9);
    let chase = swept_contact(
        Vec2::new(-2.0, 0.0), Vec2::new(5.0, 0.0), 0.3,
        Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), 0.3, 0.5, false
    ).unwrap();
    assert!(chase.minimum_distance <= 0.6);
    let legal = swept_contact(
        Vec2::ZERO, Vec2::ZERO, 0.35,
        Vec2::new(0.2, 0.0), Vec2::ZERO, 0.35, 0.1, true
    ).unwrap();
    assert!(legal.intentional);
    assert!(swept_contact(
        Vec2::ZERO, Vec2::ZERO, 0.35,
        Vec2::new(10.0, 0.0), Vec2::ZERO, 0.35, 0.1, false
    ).is_none());
}
#[test]
fn crowding_and_initial_overlap_have_explicit_non_optimistic_outcome() {
    let c = config();
    let n = |id, x, y| Neighbor {
        id, position: Vec2::new(x,y), velocity: Vec2::ZERO,
        radius: 0.35, intentional_contact: false,
    };
    let crowd = [n(1, 1.0, 0.0), n(2, -1.0, 0.0), n(3, 0.0, 1.0), n(4, 0.0, -1.0)];
    let planes = local_planes(Vec2::ZERO, 1, c, &crowd).unwrap();
    assert_eq!(planes.len(), 4);
    assert!(local_planes(Vec2::ZERO, 1, c, &[n(5, 0.0, 0.0)]).is_none());
    let deliberate = [Neighbor { intentional_contact: true, ..n(5, 0.0, 0.0) }];
    assert!(local_planes(Vec2::ZERO, 1, c, &deliberate).unwrap().is_empty());
}
fn vision() -> Vision {
    Vision { maximum_distance: 12.0, half_angle: core::f64::consts::FRAC_PI_3 }
}
fn world() -> World {
    World {
        ball: Vec2::new(6.0, 0.0),
        players: vec![
            Player { id: 1, position: Vec2::ZERO, radius: 0.3 },
            Player { id: 2, position: Vec2::new(2.0, 2.0), radius: 0.35 },
        ],
    }
}
#[test]
fn policy_cannot_observe_ball_behind_head_or_occluding_player() {
    let w = world();
    let forward = Vec2::new(1.0, 0.0);
    let visible = observe(&w, 1, forward, vision(), 10).unwrap();
    assert_eq!(visible.ball_position, Some(Vec2::new(6.0,0.0)));
    let facing_away = observe(&w, 1, Vec2::new(-1.0,0.0), vision(), 10).unwrap();
    assert_eq!(facing_away.ball_position, None);
    assert_eq!(select_motion(Some(&facing_away), TacticalChoice::ChaseVisibleBall).0,
        PolicyFeedback::OccludedTarget);
    let mut behind = world();
    behind.players[1].position = Vec2::new(3.0, 0.0);
    assert!(observe(&behind,1,forward,vision(),10).unwrap().ball_position.is_none());
}
#[test]
fn reaction_latency_never_leaks_latest_world_state() {
    let mut q = ObservationDelay::new(2).unwrap();
    let w = world();
    let o0 = observe(&w, 1, Vec2::new(1.0,0.0), vision(), 0).unwrap();
    assert!(q.push(o0.clone()).is_none());
    let mut moved = world();
    moved.ball = Vec2::new(-6.0, 0.0);
    let o1 = observe(&moved, 1, Vec2::new(1.0,0.0), vision(), 1).unwrap();
    assert!(q.push(o1).is_none());
    let o2 = observe(&moved, 1, Vec2::new(1.0,0.0), vision(), 2).unwrap();
    let delayed = q.push(o2).unwrap();
    assert_eq!(delayed.frame, 0);
    assert_eq!(delayed.ball_position, Some(Vec2::new(6.0,0.0)));
    assert_eq!(select_motion(None, TacticalChoice::ChaseVisibleBall).0,
        PolicyFeedback::NoObservation);
    assert_eq!(select_motion(Some(&delayed), TacticalChoice::MoveTo(Vec2::new(f64::NAN,0.0))).0,
        PolicyFeedback::NonFiniteTarget);
}
#[test]
fn scripted_tactical_switch_replans_without_policy_world_access_and_replays_identically() {
    fn episode() -> Vec<(u64,u64)> {
        let obs = observe(&world(),1,Vec2::new(1.0,0.0),vision(),3).unwrap();
        let initial = state(Vec2::ZERO,Vec2::ZERO);
        let (_, first_goal) = select_motion(Some(&obs),TacticalChoice::MoveTo(Vec2::new(0.0,5.0)));
        let p1 = plan(initial,first_goal.unwrap(),config(),&[]).unwrap();
        let (_, second_goal) = select_motion(Some(&obs),TacticalChoice::ChaseVisibleBall);
        let p2 = plan(p1.predicted[0],second_goal.unwrap(),config(),&[]).unwrap();
        assert!(p1.first_acceleration.y > p1.first_acceleration.x);
        assert!(p2.first_acceleration.x > p2.first_acceleration.y);
        [p1.first_acceleration,p2.first_acceleration].iter()
            .map(|v|(v.x.to_bits(),v.y.to_bits())).collect()
    }
    assert_eq!(episode(),episode());
}
