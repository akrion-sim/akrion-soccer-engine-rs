//! Reproducible bounded benchmark for DEN-870's tactical-transition prototype.
//!
//! Run with `cargo run --release --bin tactical_transition_bench`. The dense
//! case intentionally models only one player's zone. Global dense estimates
//! are calculated analytically and are never allocated.

use soccer_engine::tactical_transition::{
    execute_authoritatively, global_dense_transition_log10_bytes, heuristic_next_state,
    AuthoritativeTacticalEngine, DenseToyTransitionTable, LocalContext, NearbyGeometry,
    OrientationBand, ParameterizedTransitionOperator, PlayerObservation, PressureBand,
    RewardComponents, SparseFactorizedTransitionOperator, StaminaBand, TacticalAction,
    TacticalController, TacticalPhase, TacticalRole, TacticalState, TransitionOperator,
    TransitionSample, UpdatableTransitionOperator, ZoneGrid,
};
use std::hint::black_box;
use std::time::{Duration, Instant};

const TRAINING_PASSES: usize = 16;
const TIMING_ITERATIONS: usize = 100_000;
const DENSE_BUDGET_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone, Copy)]
struct Fixture {
    name: &'static str,
    action: TacticalAction,
    phase: TacticalPhase,
    pressure: PressureBand,
    actor_fraction: (u8, u8),
    ball_fraction: (u8, u8),
    next_phase: TacticalPhase,
    ball_shift: (i16, i16),
    reward: RewardComponents,
}

const FIXTURES: [Fixture; 7] = [
    Fixture {
        name: "overlap movement",
        action: TacticalAction::Overlap,
        phase: TacticalPhase::InPossession,
        pressure: PressureBand::Clear,
        actor_fraction: (1, 2),
        ball_fraction: (3, 2),
        next_phase: TacticalPhase::InPossession,
        ball_shift: (0, 0),
        reward: RewardComponents {
            progress: 0.7,
            possession: 0.2,
            shape: 0.4,
            pressure: 0.0,
        },
    },
    Fixture {
        name: "support movement",
        action: TacticalAction::SupportBall,
        phase: TacticalPhase::InPossession,
        pressure: PressureBand::Contested,
        actor_fraction: (1, 2),
        ball_fraction: (4, 3),
        next_phase: TacticalPhase::InPossession,
        ball_shift: (0, 0),
        reward: RewardComponents {
            progress: 0.3,
            possession: 0.3,
            shape: 0.5,
            pressure: 0.1,
        },
    },
    Fixture {
        name: "pressing rotation",
        action: TacticalAction::Press,
        phase: TacticalPhase::OutOfPossession,
        pressure: PressureBand::Contested,
        actor_fraction: (4, 1),
        ball_fraction: (3, 3),
        next_phase: TacticalPhase::OutOfPossession,
        ball_shift: (0, 0),
        reward: RewardComponents {
            progress: 0.0,
            possession: 0.0,
            shape: 0.4,
            pressure: 0.8,
        },
    },
    Fixture {
        name: "cover rotation",
        action: TacticalAction::CoverLane,
        phase: TacticalPhase::OutOfPossession,
        pressure: PressureBand::Smothered,
        actor_fraction: (4, 3),
        ball_fraction: (2, 2),
        next_phase: TacticalPhase::OutOfPossession,
        ball_shift: (0, 0),
        reward: RewardComponents {
            progress: 0.0,
            possession: 0.1,
            shape: 0.7,
            pressure: 0.4,
        },
    },
    Fixture {
        name: "pass under pressure",
        action: TacticalAction::SupportBall,
        phase: TacticalPhase::InPossession,
        pressure: PressureBand::Smothered,
        actor_fraction: (2, 2),
        ball_fraction: (3, 2),
        next_phase: TacticalPhase::InPossession,
        ball_shift: (0, 1),
        reward: RewardComponents {
            progress: 0.8,
            possession: 0.5,
            shape: 0.1,
            pressure: 0.2,
        },
    },
    Fixture {
        name: "counterattack progression",
        action: TacticalAction::MakeRun,
        phase: TacticalPhase::TransitionAttack,
        pressure: PressureBand::Clear,
        actor_fraction: (3, 1),
        ball_fraction: (3, 2),
        next_phase: TacticalPhase::TransitionAttack,
        ball_shift: (0, 1),
        reward: RewardComponents {
            progress: 1.2,
            possession: 0.3,
            shape: 0.0,
            pressure: 0.0,
        },
    },
    Fixture {
        name: "transition defense after turnover",
        action: TacticalAction::Drop,
        phase: TacticalPhase::TransitionDefense,
        pressure: PressureBand::Contested,
        actor_fraction: (3, 3),
        ball_fraction: (3, 2),
        next_phase: TacticalPhase::OutOfPossession,
        ball_shift: (0, -1),
        reward: RewardComponents {
            progress: -0.2,
            possession: -0.2,
            shape: 1.0,
            pressure: 0.3,
        },
    },
];

#[derive(Clone, Copy)]
struct ExactFixtureWorld {
    tactical: TacticalState,
    exact_tick: u64,
}

struct FixtureEngine {
    context: LocalContext,
    fixture: Fixture,
}

impl AuthoritativeTacticalEngine for FixtureEngine {
    type World = ExactFixtureWorld;
    type Error = &'static str;

    fn abstract_state(&self, world: &Self::World, _grid: ZoneGrid) -> TacticalState {
        world.tactical
    }

    fn validate_and_apply(
        &self,
        world: &Self::World,
        action: TacticalAction,
    ) -> Result<Self::World, Self::Error> {
        if action != self.fixture.action {
            return Err("fixture controller selected an unexpected action");
        }
        let mut tactical = heuristic_next_state(&world.tactical, &action, &self.context);
        tactical.ball_zone = self.context.grid.shifted(
            world.tactical.ball_zone,
            self.fixture.ball_shift.0,
            self.fixture.ball_shift.1,
        );
        tactical.phase = self.fixture.next_phase;
        Ok(ExactFixtureWorld {
            tactical,
            exact_tick: world.exact_tick + 1,
        })
    }

    fn reward_components(&self, _before: &Self::World, _after: &Self::World) -> RewardComponents {
        self.fixture.reward
    }
}

struct FixtureController(TacticalAction);

impl TacticalController for FixtureController {
    fn choose_action(
        &self,
        _observation: &PlayerObservation,
        _model: &dyn TransitionOperator,
    ) -> TacticalAction {
        self.0
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("DEN-870 bounded tactical-transition benchmark");
    println!("training_passes={TRAINING_PASSES}, timing_iterations={TIMING_ITERATIONS}");
    println!(
        "fixture coverage: {}",
        FIXTURES
            .iter()
            .map(|fixture| fixture.name)
            .collect::<Vec<_>>()
            .join(", ")
    );

    for (columns, rows) in [(6, 4), (12, 8)] {
        run_grid(ZoneGrid::new(columns, rows)?)?;
    }

    println!("\nGLOBAL DENSE SCALING (f32 transition tensor; analytical, never allocated)");
    println!("grid,modeled_players,log10_bytes");
    for (columns, rows) in [(6, 4), (12, 8)] {
        let grid = ZoneGrid::new(columns, rows)?;
        for players in [1, 2, 5, 11, 22] {
            println!(
                "{}x{},{players},{:.2}",
                columns,
                rows,
                global_dense_transition_log10_bytes(grid, players)
            );
        }
    }

    println!("\nRECOMMENDATION");
    println!(
        "Use explicit sparse factors for interpretable local tactical transitions and deterministic online updates."
    );
    println!(
        "Use a learned function approximator when context/state cardinality makes observations too sparse."
    );
    println!(
        "Keep dense tables limited to toy/local projections, and keep exact rules, physics, collisions, and lifecycle in the authoritative engine."
    );
    Ok(())
}

fn run_grid(grid: ZoneGrid) -> Result<(), Box<dyn std::error::Error>> {
    let context = LocalContext::new(grid, 1)?;
    let (samples, boundary_checks, mean_reward) = authoritative_samples(context)?;
    let mut sparse = SparseFactorizedTransitionOperator::new(16);
    let mut parameterized = ParameterizedTransitionOperator::new();
    train(&mut sparse, &samples, TRAINING_PASSES);
    train(&mut parameterized, &samples, TRAINING_PASSES);

    let states = usize::from(grid.zone_count());
    let mut dense =
        DenseToyTransitionTable::new(states, TacticalAction::ALL.len(), DENSE_BUDGET_BYTES)?;
    for _ in 0..TRAINING_PASSES {
        for sample in &samples {
            dense.observe(
                usize::from(sample.state.actor_zone),
                sample.action.index(),
                usize::from(sample.actual_next_state.actor_zone),
            );
        }
    }

    println!("\nGRID {}x{}", grid.columns, grid.rows);
    println!(
        "authoritative_boundary_checks={boundary_checks}/{}, mean_tactical_reward={mean_reward:.3}",
        samples.len()
    );
    println!(
        "model,scope,memory_bytes,predict_ns,update_ns,plans_per_second,top1_accuracy,mean_actual_probability"
    );

    let baseline_timing = time_baseline(&samples, TIMING_ITERATIONS);
    let baseline_quality = quality(&samples, &|sample| {
        let predicted = heuristic_next_state(&sample.state, &sample.action, &sample.context);
        (
            predicted == sample.actual_next_state,
            f64::from(predicted == sample.actual_next_state),
        )
    });
    print_row(
        "imperative",
        "full tactical state",
        0,
        baseline_timing,
        None,
        baseline_quality,
    );

    let dense_timing = time_dense_predict(&dense, &samples, TIMING_ITERATIONS);
    let dense_update = time_dense_update(states, &samples, TIMING_ITERATIONS)?;
    let dense_quality = quality(&samples, &|sample| {
        let probabilities =
            dense.probabilities(usize::from(sample.state.actor_zone), sample.action.index());
        let actual = usize::from(sample.actual_next_state.actor_zone);
        let predicted = probabilities.first().map(|entry| entry.0) == Some(actual);
        let probability = probabilities
            .iter()
            .find(|entry| entry.0 == actual)
            .map_or(0.0, |entry| entry.1);
        (predicted, probability)
    });
    print_row(
        "dense_toy",
        "actor zone only",
        dense.storage_bytes(),
        dense_timing,
        Some(dense_update),
        dense_quality,
    );

    let sparse_timing = time_operator_predict(&sparse, &samples, TIMING_ITERATIONS);
    let sparse_update = time_operator_update(
        SparseFactorizedTransitionOperator::new(16),
        &samples,
        TIMING_ITERATIONS,
    );
    let sparse_quality = operator_quality(&sparse, &samples);
    print_row(
        "sparse_factorized",
        "full tactical state",
        sparse.estimated_storage_bytes(),
        sparse_timing,
        Some(sparse_update),
        sparse_quality,
    );

    let parameterized_timing = time_operator_predict(&parameterized, &samples, TIMING_ITERATIONS);
    let parameterized_update = time_operator_update(
        ParameterizedTransitionOperator::new(),
        &samples,
        TIMING_ITERATIONS,
    );
    let parameterized_quality = operator_quality(&parameterized, &samples);
    print_row(
        "parameterized",
        "full tactical state",
        parameterized.estimated_storage_bytes(),
        parameterized_timing,
        Some(parameterized_update),
        parameterized_quality,
    );

    println!("sample_efficiency_passes,model,top1_accuracy,mean_actual_probability");
    for passes in [0, 1, 4, 16] {
        let mut sparse_at_budget = SparseFactorizedTransitionOperator::new(16);
        let mut parameterized_at_budget = ParameterizedTransitionOperator::new();
        train(&mut sparse_at_budget, &samples, passes);
        train(&mut parameterized_at_budget, &samples, passes);
        print_quality(
            passes,
            "sparse_factorized",
            operator_quality(&sparse_at_budget, &samples),
        );
        print_quality(
            passes,
            "parameterized",
            operator_quality(&parameterized_at_budget, &samples),
        );
    }
    Ok(())
}

fn authoritative_samples(
    context: LocalContext,
) -> Result<(Vec<TransitionSample>, usize, f64), &'static str> {
    let cold_model = SparseFactorizedTransitionOperator::default();
    let mut samples = Vec::with_capacity(FIXTURES.len());
    let mut boundary_checks = 0;
    let mut reward_total = 0.0;
    for fixture in FIXTURES {
        let state = fixture_state(context.grid, fixture);
        let world = ExactFixtureWorld {
            tactical: state,
            exact_tick: 41,
        };
        let engine = FixtureEngine { context, fixture };
        let controller = FixtureController(fixture.action);
        let (next_world, record) =
            execute_authoritatively(&engine, &world, context, &controller, &cold_model)?;
        if world.exact_tick == 41
            && next_world.exact_tick == 42
            && record.actual_next_state == next_world.tactical
        {
            boundary_checks += 1;
        }
        reward_total += record.rewards.total();
        samples.push(record.sample());
    }
    let fixture_count = u32::try_from(FIXTURES.len()).expect("fixture count fits u32");
    Ok((
        samples,
        boundary_checks,
        reward_total / f64::from(fixture_count),
    ))
}

fn fixture_state(grid: ZoneGrid, fixture: Fixture) -> TacticalState {
    let x = scaled_coordinate(fixture.actor_fraction.0, grid.columns);
    let y = scaled_coordinate(fixture.actor_fraction.1, grid.rows);
    let ball_x = scaled_coordinate(fixture.ball_fraction.0, grid.columns);
    let ball_y = scaled_coordinate(fixture.ball_fraction.1, grid.rows);
    TacticalState {
        actor_zone: grid.zone(x, y),
        ball_zone: grid.zone(ball_x, ball_y),
        phase: fixture.phase,
        pressure: fixture.pressure,
        role: TacticalRole::Midfielder,
        orientation: OrientationBand::OpponentGoal,
        stamina: StaminaBand::Working,
        nearby: NearbyGeometry::Supported,
    }
}

fn scaled_coordinate(sixths: u8, extent: u8) -> i16 {
    let last = u16::from(extent.saturating_sub(1));
    i16::try_from(u16::from(sixths) * last / 5).expect("bounded grid coordinate fits i16")
}

fn train<M: UpdatableTransitionOperator>(
    model: &mut M,
    samples: &[TransitionSample],
    passes: usize,
) {
    for _ in 0..passes {
        for sample in samples {
            model.observe(sample);
        }
    }
}

#[derive(Clone, Copy)]
struct Timing {
    nanoseconds_per_operation: u128,
    operations_per_second: u128,
}

#[derive(Clone, Copy)]
struct Quality {
    top_one_accuracy: f64,
    mean_actual_probability: f64,
}

fn timing(elapsed: Duration, iterations: usize) -> Timing {
    let count = u128::try_from(iterations).expect("timing iteration count fits u128");
    let elapsed_nanos = elapsed.as_nanos().max(1);
    Timing {
        nanoseconds_per_operation: elapsed_nanos / count,
        operations_per_second: count * 1_000_000_000 / elapsed_nanos,
    }
}

fn time_baseline(samples: &[TransitionSample], iterations: usize) -> Timing {
    let started = Instant::now();
    for index in 0..iterations {
        let sample = &samples[index % samples.len()];
        black_box(heuristic_next_state(
            black_box(&sample.state),
            black_box(&sample.action),
            black_box(&sample.context),
        ));
    }
    timing(started.elapsed(), iterations)
}

fn time_operator_predict<M: TransitionOperator>(
    model: &M,
    samples: &[TransitionSample],
    iterations: usize,
) -> Timing {
    let started = Instant::now();
    for index in 0..iterations {
        let sample = &samples[index % samples.len()];
        black_box(model.predict(
            black_box(&sample.state),
            black_box(&sample.action),
            black_box(&sample.context),
        ));
    }
    timing(started.elapsed(), iterations)
}

fn time_operator_update<M: UpdatableTransitionOperator>(
    mut model: M,
    samples: &[TransitionSample],
    iterations: usize,
) -> Timing {
    let started = Instant::now();
    for index in 0..iterations {
        model.observe(black_box(&samples[index % samples.len()]));
    }
    black_box(model);
    timing(started.elapsed(), iterations)
}

fn time_dense_predict(
    dense: &DenseToyTransitionTable,
    samples: &[TransitionSample],
    iterations: usize,
) -> Timing {
    let started = Instant::now();
    for index in 0..iterations {
        let sample = &samples[index % samples.len()];
        black_box(dense.probabilities(
            black_box(usize::from(sample.state.actor_zone)),
            black_box(sample.action.index()),
        ));
    }
    timing(started.elapsed(), iterations)
}

fn time_dense_update(
    states: usize,
    samples: &[TransitionSample],
    iterations: usize,
) -> Result<Timing, Box<dyn std::error::Error>> {
    let mut dense =
        DenseToyTransitionTable::new(states, TacticalAction::ALL.len(), DENSE_BUDGET_BYTES)?;
    let started = Instant::now();
    for index in 0..iterations {
        let sample = &samples[index % samples.len()];
        dense.observe(
            black_box(usize::from(sample.state.actor_zone)),
            black_box(sample.action.index()),
            black_box(usize::from(sample.actual_next_state.actor_zone)),
        );
    }
    black_box(dense);
    Ok(timing(started.elapsed(), iterations))
}

fn quality(
    samples: &[TransitionSample],
    evaluate: &dyn Fn(&TransitionSample) -> (bool, f64),
) -> Quality {
    let mut correct = 0_u32;
    let mut probability = 0.0;
    for sample in samples {
        let (is_correct, actual_probability) = evaluate(sample);
        correct += u32::from(is_correct);
        probability += actual_probability;
    }
    let count = u32::try_from(samples.len()).expect("sample count fits u32");
    Quality {
        top_one_accuracy: f64::from(correct) / f64::from(count),
        mean_actual_probability: probability / f64::from(count),
    }
}

fn operator_quality<M: TransitionOperator>(model: &M, samples: &[TransitionSample]) -> Quality {
    quality(samples, &|sample| {
        let distribution = model.predict(&sample.state, &sample.action, &sample.context);
        let predicted = distribution.most_likely() == Some(sample.actual_next_state);
        let probability = distribution
            .outcomes()
            .iter()
            .find(|outcome| outcome.state == sample.actual_next_state)
            .map_or(0.0, |outcome| outcome.probability);
        (predicted, probability)
    })
}

fn print_row(
    model: &str,
    scope: &str,
    memory: usize,
    prediction: Timing,
    update: Option<Timing>,
    quality: Quality,
) {
    let update_nanos = update
        .map(|timing| timing.nanoseconds_per_operation.to_string())
        .unwrap_or_else(|| "n/a".to_owned());
    println!(
        "{model},{scope},{memory},{},{update_nanos},{},{:.3},{:.3}",
        prediction.nanoseconds_per_operation,
        prediction.operations_per_second,
        quality.top_one_accuracy,
        quality.mean_actual_probability
    );
}

fn print_quality(passes: usize, model: &str, quality: Quality) {
    println!(
        "{passes},{model},{:.3},{:.3}",
        quality.top_one_accuracy, quality.mean_actual_probability
    );
}
