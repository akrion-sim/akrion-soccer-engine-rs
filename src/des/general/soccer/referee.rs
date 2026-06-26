//! Referee / match officials: the officiating agents and their decision/offside
//! belief types. Extracted from the soccer god-module; see `super` for shared
//! primitives (Vec2, Team, WorldSnapshot, constants).

use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OfficialKind {
    CenterReferee,
    AssistantRefereeNear,
    AssistantRefereeFar,
}

impl OfficialKind {
    pub(crate) fn label(self) -> &'static str {
        match self {
            OfficialKind::CenterReferee => "Center ref",
            OfficialKind::AssistantRefereeNear => "Near assistant",
            OfficialKind::AssistantRefereeFar => "Far assistant",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssistantFlank {
    Near,
    Far,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OfficialPositionSample {
    pub official_id: usize,
    pub kind: OfficialKind,
    pub tick: u64,
    pub clock_seconds: f64,
    pub position: Vec2,
    pub velocity: Vec2,
    pub acceleration: Vec2,
    pub jerk: Vec2,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OfficialDecisionTrace {
    pub tick: u64,
    pub action: String,
    pub kind: OfficialKind,
    pub position: Vec2,
    pub target: Vec2,
    #[serde(default)]
    pub scheduled_index: Option<usize>,
    #[serde(default)]
    pub offside_line_y: Option<f64>,
    #[serde(default)]
    pub local_mpc_guidance: bool,
    #[serde(default)]
    pub local_mpc_status: String,
    #[serde(default)]
    pub local_mpc_objective: f64,
    #[serde(default)]
    pub local_mpc_target_delta_yards: f64,
    #[serde(default)]
    pub operation_order: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OfficialAgent {
    pub id: usize,
    pub kind: OfficialKind,
    pub position: Vec2,
    pub velocity: Vec2,
    pub acceleration: Vec2,
    pub jerk: Vec2,
    pub position_history: VecDeque<Vec2>,
    #[serde(default)]
    pub last_decision: Option<OfficialDecisionTrace>,
}

impl OfficialAgent {
    pub(crate) fn new(id: usize, kind: OfficialKind, position: Vec2) -> Self {
        OfficialAgent {
            id,
            kind,
            position,
            velocity: Vec2::zero(),
            acceleration: Vec2::zero(),
            jerk: Vec2::zero(),
            position_history: VecDeque::from([position]),
            last_decision: None,
        }
    }

    pub(crate) fn record_position_history(&mut self) {
        self.position_history.push_back(self.position);
        while self.position_history.len() > OFFICIAL_POSITION_HISTORY_LIMIT {
            self.position_history.pop_front();
        }
    }

    pub fn history_velocity_estimate(&self, dt_seconds: f64) -> Vec2 {
        if dt_seconds <= 0.0 || self.position_history.len() < 2 {
            return self.velocity;
        }
        let last = self.position_history.len() - 1;
        (self.position_history[last] - self.position_history[last - 1]) / dt_seconds
    }

    pub fn history_acceleration_estimate(&self, dt_seconds: f64) -> Vec2 {
        if dt_seconds <= 0.0 || self.position_history.len() < 3 {
            return self.acceleration;
        }
        let last = self.position_history.len() - 1;
        let v0 = (self.position_history[last - 1] - self.position_history[last - 2]) / dt_seconds;
        let v1 = (self.position_history[last] - self.position_history[last - 1]) / dt_seconds;
        (v1 - v0) / dt_seconds
    }

    pub fn history_jerk_estimate(&self, dt_seconds: f64) -> Vec2 {
        if dt_seconds <= 0.0 || self.position_history.len() < 4 {
            return self.jerk;
        }
        let last = self.position_history.len() - 1;
        let v0 = (self.position_history[last - 2] - self.position_history[last - 3]) / dt_seconds;
        let v1 = (self.position_history[last - 1] - self.position_history[last - 2]) / dt_seconds;
        let v2 = (self.position_history[last] - self.position_history[last - 1]) / dt_seconds;
        let a0 = (v1 - v0) / dt_seconds;
        let a1 = (v2 - v1) / dt_seconds;
        (a1 - a0) / dt_seconds
    }

    pub(crate) fn run_time_step(&mut self, snapshot: &WorldSnapshot, _rng: &mut SeededRandom) {
        let previous_velocity = self.velocity;
        let previous_acceleration = self.acceleration;
        let offside_line = assistant_offside_line_snapshot(snapshot, self.kind);
        let base_target = match self.kind {
            OfficialKind::CenterReferee => center_referee_play_centroid_target(snapshot),
            OfficialKind::AssistantRefereeNear | OfficialKind::AssistantRefereeFar => Vec2::new(
                assistant_ref_touchline_x(self.kind, snapshot.field_width),
                assistant_ref_half_y(
                    self.kind,
                    offside_line
                        .as_ref()
                        .map(|line| line.effective_line_y)
                        // No live offside line (e.g. a loose ball with no settled
                        // possession): hold the current line rather than sprinting
                        // off to chase the ball — assistants stay level with the
                        // second-last defender and only react, they don't roam.
                        .unwrap_or(self.position.y),
                    snapshot.field_length,
                ),
            ),
        };
        let ball_lane = ((snapshot.ball.position.x - snapshot.field_width * 0.5)
            / (snapshot.field_width * 0.5).max(1.0))
        .clamp(-1.0, 1.0);
        let ball_depth = ((snapshot.ball.position.y - snapshot.field_length * 0.5)
            / (snapshot.field_length * 0.5).max(1.0))
        .clamp(-1.0, 1.0);
        let ball_motion = if snapshot.ball.velocity.len() > 1e-6 {
            snapshot.ball.velocity.normalized()
        } else {
            Vec2::zero()
        };
        let jitter = match self.kind {
            OfficialKind::CenterReferee => {
                Vec2::new(ball_lane * 0.16, ball_depth * 0.10) + ball_motion * 0.10
            }
            OfficialKind::AssistantRefereeNear => Vec2::new(0.0, ball_motion.y * 0.08),
            OfficialKind::AssistantRefereeFar => Vec2::new(0.0, -ball_motion.y * 0.08),
        };
        let target = official_position_bounds(
            self.kind,
            official_clearance_target(self.kind, base_target + jitter, self.position, snapshot),
            snapshot.field_width,
            snapshot.field_length,
        );
        let local_mpc_step = if snapshot.local_mpc_enabled {
            official_local_mpc_step(self, snapshot, target)
        } else {
            None
        };
        let local_mpc_guidance = local_mpc_step.is_some();
        let local_mpc_objective = local_mpc_step
            .as_ref()
            .map(|step| step.objective)
            .unwrap_or(0.0);
        let local_mpc_target_delta_yards = local_mpc_step
            .as_ref()
            .map(|step| step.position.distance(target))
            .unwrap_or(0.0);
        if let Some(step) = local_mpc_step {
            self.position = step.position;
            self.velocity = step.velocity;
            self.acceleration = step.acceleration;
            self.jerk = if snapshot.dt_seconds > 0.0 {
                (self.acceleration - previous_acceleration) / snapshot.dt_seconds
            } else {
                Vec2::zero()
            };
        } else {
            let desired = (target - self.position).normalized() * 6.1;
            self.velocity = approach_velocity(self.velocity, desired, 5.2, snapshot.dt_seconds);
            self.acceleration = if snapshot.dt_seconds > 0.0 {
                (self.velocity - previous_velocity) / snapshot.dt_seconds
            } else {
                Vec2::zero()
            };
            self.jerk = if snapshot.dt_seconds > 0.0 {
                (self.acceleration - previous_acceleration) / snapshot.dt_seconds
            } else {
                Vec2::zero()
            };
            self.position += self.velocity * snapshot.dt_seconds;
            self.position = official_position_bounds(
                self.kind,
                self.position,
                snapshot.field_width,
                snapshot.field_length,
            );
        }
        let scheduled_index = snapshot.scheduled_official_index(self.id);
        let action = match self.kind {
            OfficialKind::CenterReferee => "track-center-of-play",
            OfficialKind::AssistantRefereeNear | OfficialKind::AssistantRefereeFar => {
                "track-offside-line"
            }
        };
        let mut operation_order =
            official_agent_operation_order(snapshot.tick, self.id, self.kind, scheduled_index);
        let local_mpc_status = if local_mpc_guidance {
            operation_order.push("solve-local-mpc".to_string());
            "optimal"
        } else if snapshot.local_mpc_enabled {
            operation_order.push("fallback-local-mpc".to_string());
            "fallback"
        } else {
            "disabled"
        };
        self.last_decision = Some(OfficialDecisionTrace {
            tick: snapshot.tick,
            action: action.to_string(),
            kind: self.kind,
            position: self.position,
            target,
            scheduled_index,
            offside_line_y: offside_line.as_ref().map(|line| line.effective_line_y),
            local_mpc_guidance,
            local_mpc_status: local_mpc_status.to_string(),
            local_mpc_objective,
            local_mpc_target_delta_yards,
            operation_order,
        });
        self.record_position_history();
    }
}

#[derive(Clone, Copy, Debug)]
struct OfficialLocalMpcStep {
    position: Vec2,
    velocity: Vec2,
    acceleration: Vec2,
    objective: f64,
}

fn official_local_mpc_step(
    official: &OfficialAgent,
    snapshot: &WorldSnapshot,
    target: Vec2,
) -> Option<OfficialLocalMpcStep> {
    let (width, length) = sane_pitch_dimensions(snapshot.field_width, snapshot.field_length);
    let dt = sane_dt_seconds(snapshot.dt_seconds, DEFAULT_DT_SECONDS).max(1e-6);
    let current = official_position_bounds(official.kind, official.position, width, length);
    let current_velocity = finite_vec2(official.velocity, Vec2::zero());
    let target = official_position_bounds(official.kind, target, width, length);
    let speed_cap = match official.kind {
        OfficialKind::CenterReferee => 6.8,
        OfficialKind::AssistantRefereeNear | OfficialKind::AssistantRefereeFar => 6.2,
    };
    let accel_cap = match official.kind {
        OfficialKind::CenterReferee => 5.4,
        OfficialKind::AssistantRefereeNear | OfficialKind::AssistantRefereeFar => 4.8,
    };
    let (min_x, max_x, min_y, max_y) =
        official_local_mpc_position_bounds(official.kind, width, length);
    let (lb_x, ub_x) = official_local_mpc_axis_bounds(
        current.x,
        current_velocity.x,
        dt,
        min_x,
        max_x,
        accel_cap,
        speed_cap,
    )?;
    let (lb_y, ub_y) = official_local_mpc_axis_bounds(
        current.y,
        current_velocity.y,
        dt,
        min_y,
        max_y,
        accel_cap,
        speed_cap,
    )?;
    if lb_x > ub_x || lb_y > ub_y {
        return None;
    }

    let half_dt2 = 0.5 * dt * dt;
    let pos_base = current + current_velocity * dt;
    let desired_velocity = limit_vec2_len((target - current) / dt, speed_cap);
    let pos_weight = match official.kind {
        OfficialKind::CenterReferee => 8.0,
        OfficialKind::AssistantRefereeNear | OfficialKind::AssistantRefereeFar => 10.0,
    };
    let vel_weight = 1.35;
    let accel_ref_weight = 0.18;
    let effort_weight = 0.16;
    let previous_acceleration = limit_vec2_len(official.acceleration, accel_cap);
    let q_diag = 2.0
        * (pos_weight * half_dt2 * half_dt2
            + vel_weight * dt * dt
            + accel_ref_weight
            + effort_weight);
    if !q_diag.is_finite() || q_diag <= 1e-9 {
        return None;
    }
    let c_x = 2.0
        * (pos_weight * half_dt2 * (pos_base.x - target.x)
            + vel_weight * dt * (current_velocity.x - desired_velocity.x)
            - accel_ref_weight * previous_acceleration.x);
    let c_y = 2.0
        * (pos_weight * half_dt2 * (pos_base.y - target.y)
            + vel_weight * dt * (current_velocity.y - desired_velocity.y)
            - accel_ref_weight * previous_acceleration.y);
    let qp = QuadraticProgram {
        q: vec![vec![q_diag, 0.0], vec![0.0, q_diag]],
        c: vec![c_x, c_y],
        a_ub: None,
        b_ub: None,
        a_eq: None,
        b_eq: None,
        lb: Some(vec![Some(lb_x), Some(lb_y)]),
        ub: Some(vec![Some(ub_x), Some(ub_y)]),
        var_names: Some(vec![
            format!("official_{}_ax", official.id),
            format!("official_{}_ay", official.id),
        ]),
    };
    let solution = solve_qp_active_set(
        &qp,
        QPOptions {
            tol: 1e-7,
            max_active_sets: 64,
        },
    );
    if solution.status != QPStatus::Optimal || solution.x.len() < 2 {
        return None;
    }
    let mut acceleration = limit_vec2_len(Vec2::new(solution.x[0], solution.x[1]), accel_cap);
    let mut velocity = limit_vec2_len(current_velocity + acceleration * dt, speed_cap);
    if matches!(
        official.kind,
        OfficialKind::AssistantRefereeNear | OfficialKind::AssistantRefereeFar
    ) {
        acceleration.x = 0.0;
        velocity.x = 0.0;
    }
    let position = official_position_bounds(
        official.kind,
        current + current_velocity * dt + acceleration * half_dt2,
        width,
        length,
    );
    if !position.x.is_finite()
        || !position.y.is_finite()
        || !velocity.x.is_finite()
        || !velocity.y.is_finite()
        || !acceleration.x.is_finite()
        || !acceleration.y.is_finite()
    {
        return None;
    }

    Some(OfficialLocalMpcStep {
        position,
        velocity,
        acceleration,
        objective: finite_metric(solution.objective),
    })
}

fn official_local_mpc_position_bounds(
    kind: OfficialKind,
    field_width: f64,
    field_length: f64,
) -> (f64, f64, f64, f64) {
    match kind {
        OfficialKind::CenterReferee => (0.0, field_width, 0.0, field_length),
        OfficialKind::AssistantRefereeNear => {
            let x = assistant_ref_touchline_x(kind, field_width);
            (x, x, 0.0, field_length * 0.5)
        }
        OfficialKind::AssistantRefereeFar => {
            let x = assistant_ref_touchline_x(kind, field_width);
            (x, x, field_length * 0.5, field_length)
        }
    }
}

fn official_local_mpc_axis_bounds(
    position: f64,
    velocity: f64,
    dt: f64,
    min_position: f64,
    max_position: f64,
    accel_cap: f64,
    speed_cap: f64,
) -> Option<(f64, f64)> {
    let mut lower = -accel_cap;
    let mut upper = accel_cap;
    if dt > 1e-9 {
        lower = lower.max((-speed_cap - velocity) / dt);
        upper = upper.min((speed_cap - velocity) / dt);
    }
    let half_dt2 = 0.5 * dt * dt;
    if half_dt2 > 1e-12 {
        let base = position + velocity * dt;
        lower = lower.max((min_position - base) / half_dt2);
        upper = upper.min((max_position - base) / half_dt2);
    }
    if lower.is_finite() && upper.is_finite() && lower <= upper {
        Some((lower, upper))
    } else {
        None
    }
}

pub(crate) fn official_agent_operation_order(
    tick: u64,
    id: usize,
    kind: OfficialKind,
    scheduled_index: Option<usize>,
) -> Vec<String> {
    let kind_salt = match kind {
        OfficialKind::CenterReferee => 11,
        OfficialKind::AssistantRefereeNear => 23,
        OfficialKind::AssistantRefereeFar => 37,
    };
    let offside_weight = match kind {
        OfficialKind::CenterReferee => 0.42,
        OfficialKind::AssistantRefereeNear | OfficialKind::AssistantRefereeFar => 1.48,
    };
    let centroid_weight = match kind {
        OfficialKind::CenterReferee => 1.42,
        OfficialKind::AssistantRefereeNear | OfficialKind::AssistantRefereeFar => 0.76,
    };
    let stable_context_bias = 1.0
        + (tick % 5) as f64 * 0.001
        + id as f64 * 0.0001
        + scheduled_index.unwrap_or(0) as f64 * 0.0001
        + kind_salt as f64 * 0.00001;
    weighted_agentic_order(vec![
        ("sense-ball".to_string(), 1.10),
        (
            "sample-offside-line".to_string(),
            offside_weight * stable_context_bias,
        ),
        (
            "choose-referee-target".to_string(),
            centroid_weight * stable_context_bias,
        ),
        ("avoid-player-lanes".to_string(), 0.94),
        ("clamp-duty-zone".to_string(), 1.24),
        ("update-kinematics".to_string(), 1.06),
    ])
}

fn assistant_ref_touchline_x(kind: OfficialKind, field_width: f64) -> f64 {
    match kind {
        OfficialKind::AssistantRefereeNear => -ASSISTANT_REF_TOUCHLINE_OFFSET_YARDS,
        OfficialKind::AssistantRefereeFar => field_width + ASSISTANT_REF_TOUCHLINE_OFFSET_YARDS,
        OfficialKind::CenterReferee => field_width * 0.5,
    }
}

fn assistant_ref_half_y(kind: OfficialKind, y: f64, field_length: f64) -> f64 {
    let half = field_length * 0.5;
    match kind {
        OfficialKind::AssistantRefereeNear => y.clamp(0.0, half),
        OfficialKind::AssistantRefereeFar => y.clamp(half, field_length),
        OfficialKind::CenterReferee => y.clamp(0.0, field_length),
    }
}

pub(crate) fn official_position_bounds(
    kind: OfficialKind,
    position: Vec2,
    field_width: f64,
    field_length: f64,
) -> Vec2 {
    match kind {
        OfficialKind::CenterReferee => position.clamp_to_pitch(field_width, field_length),
        OfficialKind::AssistantRefereeNear | OfficialKind::AssistantRefereeFar => Vec2::new(
            assistant_ref_touchline_x(kind, field_width),
            assistant_ref_half_y(kind, position.y, field_length),
        ),
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OfficialSnapshot {
    pub id: usize,
    pub kind: OfficialKind,
    pub position: Vec2,
    #[serde(default)]
    pub position_history: Vec<Vec2>,
    pub velocity: Vec2,
    pub acceleration: Vec2,
    pub jerk: Vec2,
    #[serde(default)]
    pub scheduled_index: Option<usize>,
    #[serde(default)]
    pub offside_line: Option<AssistantOffsideLineSnapshot>,
    #[serde(default)]
    pub last_decision: Option<OfficialDecisionTrace>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantOffsideLineSnapshot {
    pub flank: AssistantFlank,
    pub attacking_team: Team,
    pub defending_team: Team,
    pub second_last_defender_y: f64,
    pub ball_y: f64,
    pub halfway_y: f64,
    pub effective_line_y: f64,
    pub players_beyond_line: Vec<usize>,
}
