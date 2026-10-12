//! Tactical policy is intentionally isolated from omniscient world state.
//! MDP/POMDP belief and learned networks can be plugged in above this boundary.
use std::collections::VecDeque;
use crate::locomotion::MotionGoal;
use crate::math::Vec2;

#[derive(Clone, Copy, Debug)]
pub struct Player {
    pub id: u32,
    pub position: Vec2,
    pub radius: f64,
}
#[derive(Clone, Debug)]
pub struct World {
    pub ball: Vec2,
    pub players: Vec<Player>,
}
#[derive(Clone, Copy, Debug)]
pub struct Vision {
    pub maximum_distance: f64,
    /// Radians in (0, pi], e.g. pi/3 permits a 120-degree field of view.
    pub half_angle: f64,
}
impl Vision {
    pub fn valid(self) -> bool {
        self.maximum_distance.is_finite() && self.maximum_distance > 0.0
            && self.half_angle.is_finite() && self.half_angle > 0.0
            && self.half_angle <= core::f64::consts::PI
    }
}
#[derive(Clone, Debug, PartialEq)]
pub struct Observation {
    pub own_position: Vec2,
    pub ball_position: Option<Vec2>,
    pub visible_players: Vec<(u32, Vec2)>,
    pub frame: u64,
}
fn seen(origin: Vec2, forward: Vec2, target: Vec2, vision: Vision) -> bool {
    let relative = target - origin;
    let distance = relative.norm();
    if distance > vision.maximum_distance { return false; }
    if distance <= 1e-9 { return true; }
    let Some(direction) = forward.unit() else { return false; };
    let cosine = (direction.dot(relative) / distance).clamp(-1.0, 1.0);
    cosine >= vision.half_angle.cos()
}
fn occluded(origin: Vec2, target: Vec2, blockers: &[Player], own_id: u32) -> bool {
    let segment = target - origin;
    let length_sq = segment.norm_sq();
    if length_sq <= 1e-12 { return false; }
    blockers.iter().filter(|p| p.id != own_id).any(|p| {
        let projection = (p.position - origin).dot(segment) / length_sq;
        if projection <= 0.0 || projection >= 1.0 { return false; }
        let nearest = origin + segment * projection;
        (p.position - nearest).norm_sq() <= p.radius * p.radius
    })
}
/// Only this sensing function can inspect World. Policies receive Observation.
pub fn observe(
    world: &World,
    own_id: u32,
    facing: Vec2,
    vision: Vision,
    frame: u64,
) -> Option<Observation> {
    if !vision.valid() || !facing.finite() || !world.ball.finite()
        || world.players.iter().any(|p| !p.position.finite() || !p.radius.is_finite() || p.radius <= 0.0)
    {
        return None;
    }
    let me = world.players.iter().find(|p| p.id == own_id)?;
    let ball_position = if seen(me.position, facing, world.ball, vision)
        && !occluded(me.position, world.ball, &world.players, own_id)
    {
        Some(world.ball)
    } else { None };
    let mut visible_players: Vec<_> = world.players.iter()
        .filter(|p| p.id != own_id && seen(me.position, facing, p.position, vision))
        .map(|p| (p.id, p.position)).collect();
    visible_players.sort_by_key(|(id, _)| *id);
    Some(Observation {
        own_position: me.position, ball_position, visible_players, frame
    })
}
/// Reaction latency is expressed in whole **physics frames**. Before enough
/// observations have accumulated, policy input is None, never current truth.
pub struct ObservationDelay {
    lag_frames: usize,
    history: VecDeque<Observation>,
}
impl ObservationDelay {
    pub fn new(lag_frames: usize) -> Option<Self> {
        if lag_frames <= 1000 { Some(Self { lag_frames, history: VecDeque::new() }) }
        else { None }
    }
    pub fn push(&mut self, observation: Observation) -> Option<Observation> {
        self.history.push_back(observation);
        if self.history.len() > self.lag_frames {
            self.history.pop_front()
        } else { None }
    }
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TacticalChoice {
    Hold,
    MoveTo(Vec2),
    ChaseVisibleBall,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyFeedback {
    Accepted,
    NoObservation,
    OccludedTarget,
    NonFiniteTarget,
}
/// Pure tactical-to-control handoff: never accepts the omniscient World.
pub fn select_motion(
    observation: Option<&Observation>,
    choice: TacticalChoice,
) -> (PolicyFeedback, Option<MotionGoal>) {
    let Some(observation) = observation else {
        return (PolicyFeedback::NoObservation, None);
    };
    let position = match choice {
        TacticalChoice::Hold => observation.own_position,
        TacticalChoice::MoveTo(target) => {
            if !target.finite() { return (PolicyFeedback::NonFiniteTarget, None); }
            target
        }
        TacticalChoice::ChaseVisibleBall => {
            let Some(ball) = observation.ball_position else {
                return (PolicyFeedback::OccludedTarget, None);
            };
            ball
        }
    };
    (PolicyFeedback::Accepted, Some(MotionGoal { position, velocity: Vec2::ZERO }))
}
