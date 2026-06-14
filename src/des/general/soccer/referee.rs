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

    pub(crate) fn run_time_step(&mut self, snapshot: &WorldSnapshot, rng: &mut SeededRandom) {
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
        let jitter = Vec2::new(rng.next_float() - 0.5, rng.next_float() - 0.5) * 0.25;
        let target = official_position_bounds(
            self.kind,
            official_clearance_target(self.kind, base_target + jitter, self.position, snapshot),
            snapshot.field_width,
            snapshot.field_length,
        );
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
        let scheduled_index = snapshot.scheduled_official_index(self.id);
        let action = match self.kind {
            OfficialKind::CenterReferee => "track-center-of-play",
            OfficialKind::AssistantRefereeNear | OfficialKind::AssistantRefereeFar => {
                "track-offside-line"
            }
        };
        self.last_decision = Some(OfficialDecisionTrace {
            tick: snapshot.tick,
            action: action.to_string(),
            kind: self.kind,
            position: self.position,
            target,
            scheduled_index,
            offside_line_y: offside_line.as_ref().map(|line| line.effective_line_y),
            operation_order: official_agent_operation_order(
                snapshot.tick,
                self.id,
                self.kind,
                scheduled_index,
            ),
        });
        self.record_position_history();
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
    let mut trace_rng = SeededRandom::new(
        (tick as u32)
            .wrapping_mul(2_246_822_519)
            .wrapping_add((id as u32).wrapping_mul(326_648_991))
            .wrapping_add((scheduled_index.unwrap_or(0) as u32).wrapping_mul(668_265_263))
            .wrapping_add(kind_salt),
    );
    let offside_weight = match kind {
        OfficialKind::CenterReferee => 0.42,
        OfficialKind::AssistantRefereeNear | OfficialKind::AssistantRefereeFar => 1.48,
    };
    let centroid_weight = match kind {
        OfficialKind::CenterReferee => 1.42,
        OfficialKind::AssistantRefereeNear | OfficialKind::AssistantRefereeFar => 0.76,
    };
    weighted_fisher_yates_order(
        vec![
            ("sense-ball".to_string(), 1.10),
            ("sample-offside-line".to_string(), offside_weight),
            ("choose-referee-target".to_string(), centroid_weight),
            ("avoid-player-lanes".to_string(), 0.94),
            ("clamp-duty-zone".to_string(), 1.24),
            ("update-kinematics".to_string(), 1.06),
        ],
        &mut trace_rng,
    )
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
