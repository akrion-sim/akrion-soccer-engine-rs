//! Local collision avoidance uses conservative tangent halfspaces.
//! Disk-exterior constraints are nonconvex; this does not replace physics contact.
use crate::locomotion::{Config, Halfspace};
use crate::math::Vec2;

#[derive(Clone, Copy, Debug)]
pub struct Neighbor {
    pub id: u32,
    pub position: Vec2,
    pub velocity: Vec2,
    pub radius: f64,
    /// Intentional contact is assigned to physics, not forbidden by MPC.
    pub intentional_contact: bool,
}
impl Neighbor {
    pub fn valid(&self) -> bool {
        self.position.finite()
            && self.velocity.finite()
            && self.radius.is_finite()
            && self.radius > 0.0
    }
}
/// Conservative linearized separation at one sampled horizon tick.
/// Returns None when a safe outward normal cannot be selected (already
/// overlapping/undefined). The controller must brake and let physics resolve.
/// A deliberately configured collision remains the physics engine's concern.
pub fn local_planes(
    nominal: Vec2,
    tick: usize,
    config: Config,
    neighbors: &[Neighbor],
) -> Option<Vec<Halfspace>> {
    if !nominal.finite() {
        return None;
    }
    let time = config.dt * tick as f64;
    let mut sorted = neighbors.to_vec();
    sorted.sort_by_key(|n| n.id);
    let mut planes = Vec::new();
    for n in sorted {
        if n.intentional_contact {
            continue;
        }
        let predicted = n.position + n.velocity * time;
        let delta = nominal - predicted;
        let separation = config.contact_radius + n.radius + config.collision_margin;
        let distance = delta.norm();
        // Distant players do not consume 22*21/2 pair constraints.
        if distance > separation + config.max_speed * config.dt * 2.0 {
            continue;
        }
        if distance < separation || distance < 1e-9 {
            return None;
        }
        let normal = delta * (1.0 / distance);
        // n . (nominal + 0.5*dt^2*a - predicted) >= separation
        // therefore (-n) . a <= (n . (nominal-predicted) - separation)/(0.5*dt^2).
        let upper = (distance - separation) / (0.5 * config.dt * config.dt);
        planes.push(Halfspace {
            normal: normal * -1.0,
            upper,
        });
    }
    Some(planes)
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SweptContact {
    /// Closest-approach time (seconds after beginning of physics tick).
    pub time: f64,
    pub minimum_distance: f64,
    pub intentional: bool,
}
/// Analytical swept-circle contact detection across an entire physics tick:
/// catches tunneling between discrete MPC samples without mutating positions.
/// A separate deterministic physics solver owns impulses, friction and tackles.
pub fn swept_contact(
    left_position: Vec2,
    left_velocity: Vec2,
    left_radius: f64,
    right_position: Vec2,
    right_velocity: Vec2,
    right_radius: f64,
    dt: f64,
    intentional: bool,
) -> Option<SweptContact> {
    if !left_position.finite()
        || !left_velocity.finite()
        || !right_position.finite()
        || !right_velocity.finite()
        || !dt.is_finite()
        || dt <= 0.0
        || !left_radius.is_finite()
        || !right_radius.is_finite()
        || left_radius <= 0.0
        || right_radius <= 0.0
    {
        return None;
    }
    let relative = left_position - right_position;
    let velocity = left_velocity - right_velocity;
    let v2 = velocity.norm_sq();
    let time = if v2 > 1e-12 {
        (-relative.dot(velocity) / v2).clamp(0.0, dt)
    } else {
        0.0
    };
    let minimum_distance = (relative + velocity * time).norm();
    if minimum_distance <= left_radius + right_radius {
        Some(SweptContact {
            time,
            minimum_distance,
            intentional,
        })
    } else {
        None
    }
}
