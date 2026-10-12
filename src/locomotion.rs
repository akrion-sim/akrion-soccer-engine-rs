//! One-step convex QP-like locomotion with bounded deterministic projections.
//! This is a dependency-free reference solver, not a globally optimal MPC/QP.
//! Only the first control of a repeatedly rebuilt finite-horizon plan is applied.
use crate::collision::{self, Neighbor};
use crate::math::Vec2;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct State {
    /// Metres.
    pub position: Vec2,
    /// Metres / second.
    pub velocity: Vec2,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotionGoal {
    pub position: Vec2,
    pub velocity: Vec2,
}
#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// Seconds per control tick.
    pub dt: f64,
    /// Number of planned ticks, currently 1..=20.
    pub horizon: usize,
    pub max_speed: f64,        // m/s
    pub max_acceleration: f64, // m/s^2
    pub contact_radius: f64,   // m
    pub collision_margin: f64, // m
}
impl Config {
    pub fn validate(self) -> Result<Self, ControlError> {
        if !self.dt.is_finite()
            || self.dt <= 0.0
            || self.dt > 1.0
            || !(1..=20).contains(&self.horizon)
            || !self.max_speed.is_finite()
            || self.max_speed <= 0.0
            || !self.max_acceleration.is_finite()
            || self.max_acceleration <= 0.0
            || !self.contact_radius.is_finite()
            || self.contact_radius <= 0.0
            || !self.collision_margin.is_finite()
            || self.collision_margin < 0.0
        {
            return Err(ControlError::InvalidInput);
        }
        Ok(self)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlError {
    InvalidInput,
}

#[derive(Clone, Copy, Debug)]
pub struct Halfspace {
    /// Linear acceleration inequality normal dot u <= upper.
    pub normal: Vec2,
    pub upper: f64,
}
impl Halfspace {
    pub fn valid(self) -> bool {
        self.normal.finite() && self.upper.is_finite() && self.normal.norm_sq() > 1e-15
    }
    pub fn slack(self, u: Vec2) -> f64 {
        self.upper - self.normal.dot(u)
    }
}

/// A 2-variable strictly convex quadratic objective and linear constraints.
/// The finite-horizon planner solves a sequence of these local approximations.
/// For a production QP, pass these coefficients to a reviewed solver.
#[derive(Clone, Debug)]
pub struct QpStep {
    /// Objective 0.5*h*(ux^2+uy^2) - linear dot u, h > 0.
    pub hessian_scalar: f64,
    pub linear: Vec2,
    pub inequalities: Vec<Halfspace>,
}
impl QpStep {
    pub fn valid(&self) -> bool {
        self.hessian_scalar.is_finite()
            && self.hessian_scalar > 0.0
            && self.linear.finite()
            && self.inequalities.iter().all(|c| c.valid())
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolverStatus {
    Feasible,
    Infeasible,
    NonFinite,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decision {
    Projected,
    SafeBraking,
}
#[derive(Clone, Debug)]
pub struct Plan {
    pub first_acceleration: Vec2,
    pub predicted: Vec<State>,
    pub decision: Decision,
    pub constraint_count: usize,
    pub solver_status: SolverStatus,
}
pub fn step_dynamics(s: State, a: Vec2, dt: f64) -> State {
    State {
        position: s.position + s.velocity * dt + a * (0.5 * dt * dt),
        velocity: s.velocity + a * dt,
    }
}
/// Minimize positive quadratic tracking + effort on the exact translational
/// discrete-time model. All nonconvex body/contact geometry lives elsewhere.
pub fn formulate(
    s: State,
    goal: MotionGoal,
    config: Config,
    collision_planes: &[Halfspace],
) -> QpStep {
    let dt = config.dt;
    let pos_gain = 0.5 * dt * dt;
    let position_error = goal.position - (s.position + s.velocity * dt);
    let speed_error = goal.velocity - s.velocity;
    // Cost = ||p_next-target||^2 + ||v_next-v_target||^2 + 0.2||a||^2.
    let h = 2.0 * (pos_gain * pos_gain + dt * dt + 0.2);
    let linear = (position_error * pos_gain + speed_error * dt) * 2.0;
    let a = config.max_acceleration / core::f64::consts::SQRT_2;
    let v = config.max_speed / core::f64::consts::SQRT_2;
    let mut inequalities = vec![
        Halfspace {
            normal: Vec2::new(1.0, 0.0),
            upper: a,
        },
        Halfspace {
            normal: Vec2::new(-1.0, 0.0),
            upper: a,
        },
        Halfspace {
            normal: Vec2::new(0.0, 1.0),
            upper: a,
        },
        Halfspace {
            normal: Vec2::new(0.0, -1.0),
            upper: a,
        },
        Halfspace {
            normal: Vec2::new(dt, 0.0),
            upper: v - s.velocity.x,
        },
        Halfspace {
            normal: Vec2::new(-dt, 0.0),
            upper: v + s.velocity.x,
        },
        Halfspace {
            normal: Vec2::new(0.0, dt),
            upper: v - s.velocity.y,
        },
        Halfspace {
            normal: Vec2::new(0.0, -dt),
            upper: v + s.velocity.y,
        },
    ];
    inequalities.extend_from_slice(collision_planes);
    QpStep {
        hessian_scalar: h,
        linear,
        inequalities,
    }
}
/// Projected halfspace reference feasibility solver. Cyclic projection is NOT
/// a general QP optimizer. Status is checked after a fixed iteration budget.
/// Infeasible/nonfinite => deterministic braking, not a fabricated solution.
pub fn solve_reference(qp: &QpStep) -> (SolverStatus, Vec2) {
    if !qp.valid() {
        return (SolverStatus::NonFinite, Vec2::ZERO);
    }
    let mut u = qp.linear * (1.0 / qp.hessian_scalar);
    if !u.finite() {
        return (SolverStatus::NonFinite, Vec2::ZERO);
    }
    for _ in 0..96 {
        for c in &qp.inequalities {
            let violation = c.normal.dot(u) - c.upper;
            if violation > 0.0 {
                u = u - c.normal * (violation / c.normal.norm_sq());
            }
        }
    }
    if !u.finite() {
        return (SolverStatus::NonFinite, Vec2::ZERO);
    }
    if qp.inequalities.iter().any(|c| c.slack(u) < -1e-7) {
        return (SolverStatus::Infeasible, Vec2::ZERO);
    }
    (SolverStatus::Feasible, u)
}
/// Braking magnitude is physically capped; no jump to zero velocity.
pub fn safe_braking(s: State, config: Config) -> Vec2 {
    let speed = s.velocity.norm();
    if speed <= 1e-12 || !speed.is_finite() {
        return Vec2::ZERO;
    }
    s.velocity * (-config.max_acceleration.min(speed / config.dt) / speed)
}
pub fn plan(
    initial: State,
    goal: MotionGoal,
    config: Config,
    neighbors: &[Neighbor],
) -> Result<Plan, ControlError> {
    config.validate()?;
    if !initial.position.finite()
        || !initial.velocity.finite()
        || !goal.position.finite()
        || !goal.velocity.finite()
        || neighbors.iter().any(|n| !n.valid())
    {
        return Err(ControlError::InvalidInput);
    }
    let mut predicted = Vec::with_capacity(config.horizon);
    let mut current = initial;
    let mut first = Vec2::ZERO;
    let mut decision = Decision::Projected;
    let mut status = SolverStatus::Feasible;
    let mut constraint_count = 0;
    for tick in 0..config.horizon {
        let nominal = step_dynamics(current, Vec2::ZERO, config.dt);
        let planes = collision::local_planes(nominal.position, tick + 1, config, neighbors);
        let qp = if let Some(planes) = planes {
            formulate(current, goal, config, &planes)
        } else {
            status = SolverStatus::Infeasible; // overlapped/undefined normal
            formulate(current, goal, config, &[])
        };
        constraint_count += qp.inequalities.len();
        let (step_status, candidate) = if status == SolverStatus::Infeasible {
            (SolverStatus::Infeasible, Vec2::ZERO)
        } else {
            solve_reference(&qp)
        };
        let chosen = if step_status == SolverStatus::Feasible {
            candidate
        } else {
            decision = Decision::SafeBraking;
            status = step_status;
            safe_braking(current, config)
        };
        if tick == 0 {
            first = chosen;
        }
        current = step_dynamics(current, chosen, config.dt);
        if !current.position.finite() || !current.velocity.finite() {
            return Err(ControlError::InvalidInput);
        }
        predicted.push(current);
        // Freeze after a fallback; do not call a "planned" future safe.
        if decision == Decision::SafeBraking {
            break;
        }
    }
    Ok(Plan {
        first_acceleration: first,
        predicted,
        decision,
        constraint_count,
        solver_status: status,
    })
}
