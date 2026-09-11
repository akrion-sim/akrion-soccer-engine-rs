//! Bounded tactical-transition prototype for DEN-870.
//!
//! This module deliberately models one player's coarse tactical state. It does
//! not replace or mutate the authoritative soccer world. A transition operator
//! proposes abstract outcomes; an [`AuthoritativeTacticalEngine`] adapter must
//! translate the chosen action into normal engine actions, validate it, apply
//! exact physics/rules, and return the actual state used for learning.

use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;
use std::mem::size_of;

/// Version written into serialized operator snapshots.
pub const TACTICAL_TRANSITION_FORMAT_VERSION: u16 = 1;

/// Coarse field partition. `x` is width and `y` is goal-to-goal length.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ZoneGrid {
    pub columns: u8,
    pub rows: u8,
}

impl ZoneGrid {
    pub fn new(columns: u8, rows: u8) -> Result<Self, TransitionModelError> {
        if columns == 0 || rows == 0 {
            return Err(TransitionModelError::InvalidGrid { columns, rows });
        }
        Ok(Self { columns, rows })
    }

    pub fn zone_count(self) -> u16 {
        u16::from(self.columns) * u16::from(self.rows)
    }

    pub fn coordinates(self, zone: u16) -> (i16, i16) {
        let bounded = zone.min(self.zone_count() - 1);
        (
            (bounded % u16::from(self.columns)) as i16,
            (bounded / u16::from(self.columns)) as i16,
        )
    }

    pub fn zone(self, x: i16, y: i16) -> u16 {
        let x = x.clamp(0, i16::from(self.columns) - 1) as u16;
        let y = y.clamp(0, i16::from(self.rows) - 1) as u16;
        y * u16::from(self.columns) + x
    }

    pub fn shifted(self, zone: u16, dx: i16, dy: i16) -> u16 {
        let (x, y) = self.coordinates(zone);
        self.zone(x + dx, y + dy)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TacticalAction {
    HoldShape,
    Press,
    Drop,
    Overlap,
    Underlap,
    CutInside,
    SupportBall,
    CoverLane,
    MakeRun,
}

impl TacticalAction {
    pub const ALL: [Self; 9] = [
        Self::HoldShape,
        Self::Press,
        Self::Drop,
        Self::Overlap,
        Self::Underlap,
        Self::CutInside,
        Self::SupportBall,
        Self::CoverLane,
        Self::MakeRun,
    ];

    pub fn index(self) -> usize {
        Self::ALL
            .iter()
            .position(|candidate| *candidate == self)
            .expect("all tactical actions have a stable index")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TacticalPhase {
    InPossession,
    OutOfPossession,
    TransitionAttack,
    TransitionDefense,
    SetPiece,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PressureBand {
    Clear,
    Contested,
    Smothered,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TacticalRole {
    Goalkeeper,
    Defender,
    Midfielder,
    Forward,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OrientationBand {
    OwnGoal,
    Lateral,
    OpponentGoal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StaminaBand {
    Fresh,
    Working,
    Fatigued,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NearbyGeometry {
    Isolated,
    Supported,
    Crowded,
}

/// Coarse state derived from, but separate from, exact world coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TacticalState {
    pub actor_zone: u16,
    pub ball_zone: u16,
    pub phase: TacticalPhase,
    pub pressure: PressureBand,
    pub role: TacticalRole,
    pub orientation: OrientationBand,
    pub stamina: StaminaBand,
    pub nearby: NearbyGeometry,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct LocalContext {
    pub grid: ZoneGrid,
    /// `1` attacks toward increasing `y`; `-1` attacks toward decreasing `y`.
    pub attack_direction: i8,
}

impl LocalContext {
    pub fn new(grid: ZoneGrid, attack_direction: i8) -> Result<Self, TransitionModelError> {
        if !matches!(attack_direction, -1 | 1) {
            return Err(TransitionModelError::InvalidAttackDirection(
                attack_direction,
            ));
        }
        Ok(Self {
            grid,
            attack_direction,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TacticalOutcome {
    pub state: TacticalState,
    pub probability: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NextStateDistribution {
    outcomes: Vec<TacticalOutcome>,
}

impl NextStateDistribution {
    /// Builds a bounded distribution, merging duplicate states, discarding
    /// invalid weights, sorting deterministically, and normalizing probability.
    pub fn from_outcomes(
        raw: impl IntoIterator<Item = TacticalOutcome>,
        max_outcomes: usize,
    ) -> Self {
        let mut merged: Vec<TacticalOutcome> = Vec::new();
        for outcome in raw {
            if !outcome.probability.is_finite() || outcome.probability <= 0.0 {
                continue;
            }
            if let Some(existing) = merged.iter_mut().find(|entry| entry.state == outcome.state) {
                existing.probability += outcome.probability;
            } else {
                merged.push(outcome);
            }
        }
        merged.sort_by(|left, right| {
            right
                .probability
                .total_cmp(&left.probability)
                .then_with(|| left.state.cmp(&right.state))
        });
        merged.truncate(max_outcomes.max(1));
        let total: f64 = merged.iter().map(|outcome| outcome.probability).sum();
        if total > 0.0 {
            for outcome in &mut merged {
                outcome.probability /= total;
            }
        }
        Self { outcomes: merged }
    }

    pub fn deterministic(state: TacticalState) -> Self {
        Self {
            outcomes: vec![TacticalOutcome {
                state,
                probability: 1.0,
            }],
        }
    }

    pub fn outcomes(&self) -> &[TacticalOutcome] {
        &self.outcomes
    }

    pub fn most_likely(&self) -> Option<TacticalState> {
        self.outcomes.first().map(|outcome| outcome.state)
    }

    /// Deterministic sampling hook: callers supply a seeded draw in `[0, 1)`.
    pub fn sample(&self, unit_interval_draw: f64) -> Option<TacticalState> {
        if self.outcomes.is_empty() {
            return None;
        }
        let draw = if unit_interval_draw.is_finite() {
            unit_interval_draw.clamp(0.0, 1.0 - f64::EPSILON)
        } else {
            0.0
        };
        let mut cumulative = 0.0;
        for outcome in &self.outcomes {
            cumulative += outcome.probability;
            if draw < cumulative {
                return Some(outcome.state);
            }
        }
        self.outcomes.last().map(|outcome| outcome.state)
    }
}

pub trait TransitionOperator {
    fn predict(
        &self,
        state: &TacticalState,
        action: &TacticalAction,
        context: &LocalContext,
    ) -> NextStateDistribution;
}

pub trait UpdatableTransitionOperator: TransitionOperator {
    fn observe(&mut self, sample: &TransitionSample);
}

pub trait TacticalController {
    fn choose_action(
        &self,
        observation: &PlayerObservation,
        model: &dyn TransitionOperator,
    ) -> TacticalAction;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlayerObservation {
    pub state: TacticalState,
    pub context: LocalContext,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RewardComponents {
    pub progress: f64,
    pub possession: f64,
    pub shape: f64,
    pub pressure: f64,
}

impl RewardComponents {
    pub fn total(self) -> f64 {
        self.progress + self.possession + self.shape + self.pressure
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RecordedTransition {
    pub state: TacticalState,
    pub action: TacticalAction,
    pub predicted: NextStateDistribution,
    pub actual_next_state: TacticalState,
    pub rewards: RewardComponents,
    pub context: LocalContext,
}

impl RecordedTransition {
    pub fn sample(&self) -> TransitionSample {
        TransitionSample {
            state: self.state,
            action: self.action,
            actual_next_state: self.actual_next_state,
            rewards: self.rewards,
            context: self.context,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TransitionSample {
    pub state: TacticalState,
    pub action: TacticalAction,
    pub actual_next_state: TacticalState,
    pub rewards: RewardComponents,
    pub context: LocalContext,
}

/// The only execution path exposed by the prototype.
///
/// Implementations must translate a tactical action into the engine's normal
/// action type and let the authoritative engine validate feasibility, resolve
/// physics/collisions/rules, and return a new exact world. The model never gets
/// mutable access to that world.
pub trait AuthoritativeTacticalEngine {
    type World;
    type Error;

    fn abstract_state(&self, world: &Self::World, grid: ZoneGrid) -> TacticalState;

    fn validate_and_apply(
        &self,
        world: &Self::World,
        action: TacticalAction,
    ) -> Result<Self::World, Self::Error>;

    fn reward_components(&self, before: &Self::World, after: &Self::World) -> RewardComponents;
}

pub fn execute_authoritatively<E, C>(
    engine: &E,
    world: &E::World,
    context: LocalContext,
    controller: &C,
    model: &dyn TransitionOperator,
) -> Result<(E::World, RecordedTransition), E::Error>
where
    E: AuthoritativeTacticalEngine,
    C: TacticalController,
{
    let state = engine.abstract_state(world, context.grid);
    let observation = PlayerObservation { state, context };
    let action = controller.choose_action(&observation, model);
    let predicted = model.predict(&state, &action, &context);
    let next_world = engine.validate_and_apply(world, action)?;
    let actual_next_state = engine.abstract_state(&next_world, context.grid);
    let rewards = engine.reward_components(world, &next_world);
    Ok((
        next_world,
        RecordedTransition {
            state,
            action,
            predicted,
            actual_next_state,
            rewards,
            context,
        },
    ))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
struct MovementKey {
    grid: ZoneGrid,
    actor_zone: u16,
    action: TacticalAction,
    pressure: PressureBand,
    role: TacticalRole,
    orientation: OrientationBand,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
struct BallKey {
    grid: ZoneGrid,
    ball_zone: u16,
    action: TacticalAction,
    phase: TacticalPhase,
    pressure: PressureBand,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
struct PhaseKey {
    phase: TacticalPhase,
    action: TacticalAction,
    nearby: NearbyGeometry,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct ZoneCount {
    zone: u16,
    count: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct PhaseCount {
    phase: TacticalPhase,
    count: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct MovementFactor {
    key: MovementKey,
    outcomes: Vec<ZoneCount>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct BallFactor {
    key: BallKey,
    outcomes: Vec<ZoneCount>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct PhaseFactor {
    key: PhaseKey,
    outcomes: Vec<PhaseCount>,
}

/// Sparse empirical model with independently learned actor, ball, and phase
/// factors. The vectors are compact, deterministic JSON wire storage; no dense
/// Cartesian state table is allocated.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SparseFactorizedTransitionOperator {
    pub format_version: u16,
    pub max_outcomes: usize,
    movement: Vec<MovementFactor>,
    ball: Vec<BallFactor>,
    phase: Vec<PhaseFactor>,
}

impl Default for SparseFactorizedTransitionOperator {
    fn default() -> Self {
        Self::new(16)
    }
}

impl SparseFactorizedTransitionOperator {
    pub fn new(max_outcomes: usize) -> Self {
        Self {
            format_version: TACTICAL_TRANSITION_FORMAT_VERSION,
            max_outcomes: max_outcomes.max(1),
            movement: Vec::new(),
            ball: Vec::new(),
            phase: Vec::new(),
        }
    }

    pub fn validate_version(&self) -> Result<(), TransitionModelError> {
        if self.format_version != TACTICAL_TRANSITION_FORMAT_VERSION {
            return Err(TransitionModelError::UnsupportedVersion {
                expected: TACTICAL_TRANSITION_FORMAT_VERSION,
                actual: self.format_version,
            });
        }
        Ok(())
    }

    pub fn factor_count(&self) -> usize {
        self.movement.len() + self.ball.len() + self.phase.len()
    }

    pub fn observed_outcome_count(&self) -> usize {
        self.movement
            .iter()
            .map(|factor| factor.outcomes.len())
            .sum::<usize>()
            + self
                .ball
                .iter()
                .map(|factor| factor.outcomes.len())
                .sum::<usize>()
            + self
                .phase
                .iter()
                .map(|factor| factor.outcomes.len())
                .sum::<usize>()
    }

    pub fn estimated_storage_bytes(&self) -> usize {
        size_of::<Self>()
            + self.movement.capacity() * size_of::<MovementFactor>()
            + self.ball.capacity() * size_of::<BallFactor>()
            + self.phase.capacity() * size_of::<PhaseFactor>()
            + self
                .movement
                .iter()
                .map(|factor| factor.outcomes.capacity() * size_of::<ZoneCount>())
                .sum::<usize>()
            + self
                .ball
                .iter()
                .map(|factor| factor.outcomes.capacity() * size_of::<ZoneCount>())
                .sum::<usize>()
            + self
                .phase
                .iter()
                .map(|factor| factor.outcomes.capacity() * size_of::<PhaseCount>())
                .sum::<usize>()
    }

    fn movement_probabilities(&self, key: MovementKey) -> Option<Vec<(u16, f64)>> {
        self.movement
            .iter()
            .find(|factor| factor.key == key)
            .map(|factor| zone_probabilities(&factor.outcomes))
    }

    fn ball_probabilities(&self, key: BallKey) -> Option<Vec<(u16, f64)>> {
        self.ball
            .iter()
            .find(|factor| factor.key == key)
            .map(|factor| zone_probabilities(&factor.outcomes))
    }

    fn phase_probabilities(&self, key: PhaseKey) -> Option<Vec<(TacticalPhase, f64)>> {
        self.phase
            .iter()
            .find(|factor| factor.key == key)
            .map(|factor| {
                let total = factor
                    .outcomes
                    .iter()
                    .map(|outcome| outcome.count)
                    .sum::<u64>() as f64;
                factor
                    .outcomes
                    .iter()
                    .map(|outcome| (outcome.phase, outcome.count as f64 / total))
                    .collect()
            })
    }
}

impl TransitionOperator for SparseFactorizedTransitionOperator {
    fn predict(
        &self,
        state: &TacticalState,
        action: &TacticalAction,
        context: &LocalContext,
    ) -> NextStateDistribution {
        let fallback = heuristic_next_state(state, action, context);
        let movement = self
            .movement_probabilities(MovementKey {
                grid: context.grid,
                actor_zone: state.actor_zone,
                action: *action,
                pressure: state.pressure,
                role: state.role,
                orientation: state.orientation,
            })
            .unwrap_or_else(|| vec![(fallback.actor_zone, 1.0)]);
        let ball = self
            .ball_probabilities(BallKey {
                grid: context.grid,
                ball_zone: state.ball_zone,
                action: *action,
                phase: state.phase,
                pressure: state.pressure,
            })
            .unwrap_or_else(|| vec![(fallback.ball_zone, 1.0)]);
        let phases = self
            .phase_probabilities(PhaseKey {
                phase: state.phase,
                action: *action,
                nearby: state.nearby,
            })
            .unwrap_or_else(|| vec![(fallback.phase, 1.0)]);

        let mut raw = Vec::new();
        for (actor_zone, actor_probability) in movement {
            for (ball_zone, ball_probability) in &ball {
                for (phase, phase_probability) in &phases {
                    let mut next = *state;
                    next.actor_zone = actor_zone;
                    next.ball_zone = *ball_zone;
                    next.phase = *phase;
                    raw.push(TacticalOutcome {
                        state: next,
                        probability: actor_probability * *ball_probability * *phase_probability,
                    });
                }
            }
        }
        NextStateDistribution::from_outcomes(raw, self.max_outcomes)
    }
}

impl UpdatableTransitionOperator for SparseFactorizedTransitionOperator {
    fn observe(&mut self, sample: &TransitionSample) {
        let movement_key = MovementKey {
            grid: sample.context.grid,
            actor_zone: sample.state.actor_zone,
            action: sample.action,
            pressure: sample.state.pressure,
            role: sample.state.role,
            orientation: sample.state.orientation,
        };
        update_zone_factor(
            &mut self.movement,
            movement_key,
            sample.actual_next_state.actor_zone,
        );

        let ball_key = BallKey {
            grid: sample.context.grid,
            ball_zone: sample.state.ball_zone,
            action: sample.action,
            phase: sample.state.phase,
            pressure: sample.state.pressure,
        };
        update_ball_factor(&mut self.ball, ball_key, sample.actual_next_state.ball_zone);

        let phase_key = PhaseKey {
            phase: sample.state.phase,
            action: sample.action,
            nearby: sample.state.nearby,
        };
        if let Some(factor) = self.phase.iter_mut().find(|factor| factor.key == phase_key) {
            increment_phase(&mut factor.outcomes, sample.actual_next_state.phase);
        } else {
            self.phase.push(PhaseFactor {
                key: phase_key,
                outcomes: vec![PhaseCount {
                    phase: sample.actual_next_state.phase,
                    count: 1,
                }],
            });
            self.phase.sort_by_key(|factor| factor.key);
        }
    }
}

fn update_zone_factor(factors: &mut Vec<MovementFactor>, key: MovementKey, zone: u16) {
    if let Some(factor) = factors.iter_mut().find(|factor| factor.key == key) {
        increment_zone(&mut factor.outcomes, zone);
    } else {
        factors.push(MovementFactor {
            key,
            outcomes: vec![ZoneCount { zone, count: 1 }],
        });
        factors.sort_by_key(|factor| factor.key);
    }
}

fn update_ball_factor(factors: &mut Vec<BallFactor>, key: BallKey, zone: u16) {
    if let Some(factor) = factors.iter_mut().find(|factor| factor.key == key) {
        increment_zone(&mut factor.outcomes, zone);
    } else {
        factors.push(BallFactor {
            key,
            outcomes: vec![ZoneCount { zone, count: 1 }],
        });
        factors.sort_by_key(|factor| factor.key);
    }
}

fn increment_zone(outcomes: &mut Vec<ZoneCount>, zone: u16) {
    if let Some(outcome) = outcomes.iter_mut().find(|outcome| outcome.zone == zone) {
        outcome.count += 1;
    } else {
        outcomes.push(ZoneCount { zone, count: 1 });
        outcomes.sort_by_key(|outcome| outcome.zone);
    }
}

fn increment_phase(outcomes: &mut Vec<PhaseCount>, phase: TacticalPhase) {
    if let Some(outcome) = outcomes.iter_mut().find(|outcome| outcome.phase == phase) {
        outcome.count += 1;
    } else {
        outcomes.push(PhaseCount { phase, count: 1 });
        outcomes.sort_by_key(|outcome| outcome.phase);
    }
}

fn zone_probabilities(outcomes: &[ZoneCount]) -> Vec<(u16, f64)> {
    let total = outcomes.iter().map(|outcome| outcome.count).sum::<u64>() as f64;
    outcomes
        .iter()
        .map(|outcome| (outcome.zone, outcome.count as f64 / total))
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
struct ActionDeltaModel {
    action: TacticalAction,
    samples: u64,
    actor_dx_sum: f64,
    actor_dy_sum: f64,
    ball_dx_sum: f64,
    ball_dy_sum: f64,
}

/// Tiny learned function approximator: action-conditioned mean actor/ball
/// displacements. It shares [`TransitionOperator`] with the sparse tabular
/// model, making controller code representation-agnostic.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterizedTransitionOperator {
    pub format_version: u16,
    action_models: Vec<ActionDeltaModel>,
}

impl Default for ParameterizedTransitionOperator {
    fn default() -> Self {
        Self::new()
    }
}

impl ParameterizedTransitionOperator {
    pub fn new() -> Self {
        Self {
            format_version: TACTICAL_TRANSITION_FORMAT_VERSION,
            action_models: Vec::new(),
        }
    }

    pub fn sample_count(&self) -> u64 {
        self.action_models.iter().map(|model| model.samples).sum()
    }

    pub fn estimated_storage_bytes(&self) -> usize {
        size_of::<Self>() + self.action_models.capacity() * size_of::<ActionDeltaModel>()
    }

    pub fn validate_version(&self) -> Result<(), TransitionModelError> {
        if self.format_version != TACTICAL_TRANSITION_FORMAT_VERSION {
            return Err(TransitionModelError::UnsupportedVersion {
                expected: TACTICAL_TRANSITION_FORMAT_VERSION,
                actual: self.format_version,
            });
        }
        Ok(())
    }
}

impl TransitionOperator for ParameterizedTransitionOperator {
    fn predict(
        &self,
        state: &TacticalState,
        action: &TacticalAction,
        context: &LocalContext,
    ) -> NextStateDistribution {
        let fallback = heuristic_next_state(state, action, context);
        let Some(model) = self
            .action_models
            .iter()
            .find(|model| model.action == *action)
        else {
            return NextStateDistribution::deterministic(fallback);
        };
        let samples = model.samples as f64;
        let mut learned = *state;
        learned.actor_zone = context.grid.shifted(
            state.actor_zone,
            (model.actor_dx_sum / samples).round() as i16,
            (model.actor_dy_sum / samples).round() as i16,
        );
        learned.ball_zone = context.grid.shifted(
            state.ball_zone,
            (model.ball_dx_sum / samples).round() as i16,
            (model.ball_dy_sum / samples).round() as i16,
        );
        let confidence = (samples / (samples + 4.0)).clamp(0.2, 0.95);
        NextStateDistribution::from_outcomes(
            [
                TacticalOutcome {
                    state: learned,
                    probability: confidence,
                },
                TacticalOutcome {
                    state: fallback,
                    probability: 1.0 - confidence,
                },
            ],
            2,
        )
    }
}

impl UpdatableTransitionOperator for ParameterizedTransitionOperator {
    fn observe(&mut self, sample: &TransitionSample) {
        let (actor_x, actor_y) = sample.context.grid.coordinates(sample.state.actor_zone);
        let (next_actor_x, next_actor_y) = sample
            .context
            .grid
            .coordinates(sample.actual_next_state.actor_zone);
        let (ball_x, ball_y) = sample.context.grid.coordinates(sample.state.ball_zone);
        let (next_ball_x, next_ball_y) = sample
            .context
            .grid
            .coordinates(sample.actual_next_state.ball_zone);
        let update = |model: &mut ActionDeltaModel| {
            model.samples += 1;
            model.actor_dx_sum += f64::from(next_actor_x - actor_x);
            model.actor_dy_sum += f64::from(next_actor_y - actor_y);
            model.ball_dx_sum += f64::from(next_ball_x - ball_x);
            model.ball_dy_sum += f64::from(next_ball_y - ball_y);
        };
        if let Some(model) = self
            .action_models
            .iter_mut()
            .find(|model| model.action == sample.action)
        {
            update(model);
        } else {
            let mut model = ActionDeltaModel {
                action: sample.action,
                samples: 0,
                actor_dx_sum: 0.0,
                actor_dy_sum: 0.0,
                ball_dx_sum: 0.0,
                ball_dy_sum: 0.0,
            };
            update(&mut model);
            self.action_models.push(model);
            self.action_models.sort_by_key(|model| model.action);
        }
    }
}

/// Deliberately tiny dense table used only to quantify the matrix boundary.
#[derive(Clone, Debug)]
pub struct DenseToyTransitionTable {
    states: usize,
    actions: usize,
    counts: Vec<u32>,
}

impl DenseToyTransitionTable {
    pub fn new(
        states: usize,
        actions: usize,
        max_bytes: usize,
    ) -> Result<Self, TransitionModelError> {
        let cells = states
            .checked_mul(actions)
            .and_then(|value| value.checked_mul(states))
            .ok_or(TransitionModelError::DenseTableOverflow)?;
        let bytes = cells
            .checked_mul(size_of::<u32>())
            .ok_or(TransitionModelError::DenseTableOverflow)?;
        if states == 0 || actions == 0 || bytes > max_bytes {
            return Err(TransitionModelError::DenseBudgetExceeded { bytes, max_bytes });
        }
        Ok(Self {
            states,
            actions,
            counts: vec![0; cells],
        })
    }

    pub fn storage_bytes(&self) -> usize {
        self.counts.len() * size_of::<u32>()
    }

    pub fn observe(&mut self, state: usize, action: usize, next_state: usize) {
        let index = self.index(state, action, next_state);
        self.counts[index] = self.counts[index].saturating_add(1);
    }

    pub fn probabilities(&self, state: usize, action: usize) -> Vec<(usize, f64)> {
        let start = self.index(state, action, 0);
        let row = &self.counts[start..start + self.states];
        let total = row.iter().map(|count| u64::from(*count)).sum::<u64>() as f64;
        if total == 0.0 {
            return vec![(state.min(self.states - 1), 1.0)];
        }
        row.iter()
            .enumerate()
            .filter(|(_, count)| **count > 0)
            .map(|(next_state, count)| (next_state, f64::from(*count) / total))
            .collect()
    }

    fn index(&self, state: usize, action: usize, next_state: usize) -> usize {
        assert!(state < self.states, "dense toy state out of bounds");
        assert!(action < self.actions, "dense toy action out of bounds");
        assert!(
            next_state < self.states,
            "dense toy next state out of bounds"
        );
        (state * self.actions + action) * self.states + next_state
    }
}

/// Base-10 bytes required by a global dense transition tensor. This does not
/// allocate it. Each modeled player's zone participates in the global state,
/// with phase/pressure context and one `f32` per `(s, a, s')` cell.
pub fn global_dense_transition_log10_bytes(grid: ZoneGrid, modeled_players: usize) -> f64 {
    let zone_log = f64::from(grid.zone_count()).log10();
    let context_states = 5.0_f64 * 3.0 * 3.0;
    let global_state_log = modeled_players as f64 * zone_log + context_states.log10();
    2.0 * global_state_log
        + (TacticalAction::ALL.len() as f64).log10()
        + (size_of::<f32>() as f64).log10()
}

/// Current imperative-style fallback used for cold sparse/learned factors and
/// as the benchmark baseline. It proposes only a coarse zone; exact movement is
/// still produced by the authoritative engine adapter.
pub fn heuristic_next_state(
    state: &TacticalState,
    action: &TacticalAction,
    context: &LocalContext,
) -> TacticalState {
    let attack = i16::from(context.attack_direction);
    let (actor_x, actor_y) = context.grid.coordinates(state.actor_zone);
    let (ball_x, ball_y) = context.grid.coordinates(state.ball_zone);
    let center_x = (i16::from(context.grid.columns) - 1) / 2;
    let (dx, dy) = match action {
        TacticalAction::HoldShape => (0, 0),
        TacticalAction::Press => ((ball_x - actor_x).signum(), (ball_y - actor_y).signum()),
        TacticalAction::Drop => (0, -attack),
        TacticalAction::Overlap => {
            let wide = if actor_x <= center_x { -1 } else { 1 };
            (wide, attack)
        }
        TacticalAction::Underlap => ((center_x - actor_x).signum(), attack),
        TacticalAction::CutInside => ((center_x - actor_x).signum(), 0),
        TacticalAction::SupportBall => ((ball_x - actor_x).signum(), (ball_y - actor_y).signum()),
        TacticalAction::CoverLane => ((ball_x - actor_x).signum(), -attack),
        TacticalAction::MakeRun => (0, attack),
    };
    let mut next = *state;
    next.actor_zone = context.grid.zone(actor_x + dx, actor_y + dy);
    next
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TransitionModelError {
    InvalidGrid { columns: u8, rows: u8 },
    InvalidAttackDirection(i8),
    UnsupportedVersion { expected: u16, actual: u16 },
    DenseTableOverflow,
    DenseBudgetExceeded { bytes: usize, max_bytes: usize },
}

impl fmt::Display for TransitionModelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGrid { columns, rows } => {
                write!(formatter, "invalid tactical grid {columns}x{rows}")
            }
            Self::InvalidAttackDirection(direction) => {
                write!(
                    formatter,
                    "attack direction must be -1 or 1, got {direction}"
                )
            }
            Self::UnsupportedVersion { expected, actual } => write!(
                formatter,
                "unsupported tactical model version {actual}; expected {expected}"
            ),
            Self::DenseTableOverflow => write!(formatter, "dense table dimensions overflow"),
            Self::DenseBudgetExceeded { bytes, max_bytes } => write!(
                formatter,
                "dense table requires {bytes} bytes, above {max_bytes}-byte budget"
            ),
        }
    }
}

impl Error for TransitionModelError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(grid: ZoneGrid) -> TacticalState {
        TacticalState {
            actor_zone: grid.zone(1, 1),
            ball_zone: grid.zone(2, 2),
            phase: TacticalPhase::InPossession,
            pressure: PressureBand::Contested,
            role: TacticalRole::Midfielder,
            orientation: OrientationBand::OpponentGoal,
            stamina: StaminaBand::Fresh,
            nearby: NearbyGeometry::Supported,
        }
    }

    fn rewards() -> RewardComponents {
        RewardComponents {
            progress: 1.0,
            possession: 0.2,
            shape: 0.1,
            pressure: -0.1,
        }
    }

    #[test]
    fn grids_support_two_resolutions_and_bounded_shifts() {
        let coarse = ZoneGrid::new(6, 4).unwrap();
        let fine = ZoneGrid::new(12, 8).unwrap();
        assert_eq!(coarse.zone_count(), 24);
        assert_eq!(fine.zone_count(), 96);
        assert_eq!(coarse.shifted(coarse.zone(0, 0), -2, -2), 0);
        assert_eq!(fine.coordinates(fine.zone(11, 7)), (11, 7));
    }

    #[test]
    fn sparse_factors_learn_actual_engine_outcomes() {
        let grid = ZoneGrid::new(6, 4).unwrap();
        let context = LocalContext::new(grid, 1).unwrap();
        let before = state(grid);
        let mut frequent = heuristic_next_state(&before, &TacticalAction::MakeRun, &context);
        frequent.ball_zone = grid.shifted(before.ball_zone, 0, 1);
        frequent.phase = TacticalPhase::TransitionAttack;
        let mut rare = frequent;
        rare.actor_zone = grid.shifted(before.actor_zone, 1, 0);

        let mut model = SparseFactorizedTransitionOperator::new(16);
        for actual_next_state in [frequent, frequent, frequent, rare] {
            model.observe(&TransitionSample {
                state: before,
                action: TacticalAction::MakeRun,
                actual_next_state,
                rewards: rewards(),
                context,
            });
        }

        let prediction = model.predict(&before, &TacticalAction::MakeRun, &context);
        assert_eq!(prediction.most_likely(), Some(frequent));
        assert_eq!(model.factor_count(), 3);
        assert_eq!(model.observed_outcome_count(), 4);
        assert!(model.estimated_storage_bytes() < 8_192);
    }

    #[test]
    fn sparse_snapshot_is_versioned_and_json_round_trips() {
        let grid = ZoneGrid::new(6, 4).unwrap();
        let context = LocalContext::new(grid, 1).unwrap();
        let before = state(grid);
        let mut model = SparseFactorizedTransitionOperator::default();
        model.observe(&TransitionSample {
            state: before,
            action: TacticalAction::SupportBall,
            actual_next_state: heuristic_next_state(
                &before,
                &TacticalAction::SupportBall,
                &context,
            ),
            rewards: rewards(),
            context,
        });
        let json = serde_json::to_string(&model).unwrap();
        let restored: SparseFactorizedTransitionOperator = serde_json::from_str(&json).unwrap();
        restored.validate_version().unwrap();
        assert_eq!(restored, model);
    }

    #[test]
    fn parameterized_model_shares_trait_and_learns_displacement() {
        let grid = ZoneGrid::new(12, 8).unwrap();
        let context = LocalContext::new(grid, 1).unwrap();
        let before = state(grid);
        let mut actual = before;
        actual.actor_zone = grid.shifted(before.actor_zone, 1, 2);
        let mut model = ParameterizedTransitionOperator::new();
        for _ in 0..12 {
            model.observe(&TransitionSample {
                state: before,
                action: TacticalAction::Underlap,
                actual_next_state: actual,
                rewards: rewards(),
                context,
            });
        }
        let operator: &dyn TransitionOperator = &model;
        assert_eq!(
            operator
                .predict(&before, &TacticalAction::Underlap, &context)
                .most_likely(),
            Some(actual)
        );
        assert_eq!(model.sample_count(), 12);
    }

    #[test]
    fn dense_toy_is_bounded_and_global_tensor_is_quantified() {
        let grid = ZoneGrid::new(6, 4).unwrap();
        let mut dense = DenseToyTransitionTable::new(
            usize::from(grid.zone_count()),
            TacticalAction::ALL.len(),
            1_000_000,
        )
        .unwrap();
        dense.observe(0, TacticalAction::Press.index(), 1);
        dense.observe(0, TacticalAction::Press.index(), 1);
        assert_eq!(
            dense.probabilities(0, TacticalAction::Press.index()),
            vec![(1, 1.0)]
        );
        assert!(DenseToyTransitionTable::new(100_000, 9, 1_000_000).is_err());
        assert!(global_dense_transition_log10_bytes(grid, 22) > 65.0);
    }

    #[derive(Clone, Copy)]
    struct FakeWorld {
        state: TacticalState,
        exact_ticks: u64,
    }

    struct FakeEngine {
        context: LocalContext,
    }

    impl AuthoritativeTacticalEngine for FakeEngine {
        type World = FakeWorld;
        type Error = &'static str;

        fn abstract_state(&self, world: &Self::World, _grid: ZoneGrid) -> TacticalState {
            world.state
        }

        fn validate_and_apply(
            &self,
            world: &Self::World,
            action: TacticalAction,
        ) -> Result<Self::World, Self::Error> {
            Ok(FakeWorld {
                state: heuristic_next_state(&world.state, &action, &self.context),
                exact_ticks: world.exact_ticks + 1,
            })
        }

        fn reward_components(
            &self,
            _before: &Self::World,
            _after: &Self::World,
        ) -> RewardComponents {
            rewards()
        }
    }

    struct MakeRunController;

    impl TacticalController for MakeRunController {
        fn choose_action(
            &self,
            _observation: &PlayerObservation,
            _model: &dyn TransitionOperator,
        ) -> TacticalAction {
            TacticalAction::MakeRun
        }
    }

    #[test]
    fn authoritative_boundary_owns_world_mutation_and_records_actual_result() {
        let grid = ZoneGrid::new(6, 4).unwrap();
        let context = LocalContext::new(grid, 1).unwrap();
        let world = FakeWorld {
            state: state(grid),
            exact_ticks: 7,
        };
        let engine = FakeEngine { context };
        let model = SparseFactorizedTransitionOperator::default();
        let (next_world, record) =
            execute_authoritatively(&engine, &world, context, &MakeRunController, &model).unwrap();

        assert_eq!(world.exact_ticks, 7);
        assert_eq!(next_world.exact_ticks, 8);
        assert_eq!(record.actual_next_state, next_world.state);
        assert_eq!(record.action, TacticalAction::MakeRun);
        assert_eq!(record.rewards.total(), 1.2);
    }

    #[test]
    fn seeded_distribution_sampling_is_replayable() {
        let grid = ZoneGrid::new(6, 4).unwrap();
        let first = state(grid);
        let mut second = first;
        second.actor_zone = grid.shifted(first.actor_zone, 1, 0);
        let distribution = NextStateDistribution::from_outcomes(
            [
                TacticalOutcome {
                    state: first,
                    probability: 0.75,
                },
                TacticalOutcome {
                    state: second,
                    probability: 0.25,
                },
            ],
            2,
        );
        assert_eq!(distribution.sample(0.1), distribution.sample(0.1));
        assert_eq!(distribution.sample(0.9), Some(second));
    }
}
