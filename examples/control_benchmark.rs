//! Repeatable diagnostic microbenchmark, not proof of meeting 60Hz budgets.
use akrion_soccer_engine::{
    locomotion::{plan, Config, MotionGoal, State},
    math::Vec2,
};
use std::hint::black_box;
use std::time::Instant;
fn main() {
    let config = Config {
        dt: 0.1,
        horizon: 5,
        max_speed: 7.0,
        max_acceleration: 4.0,
        contact_radius: 0.35,
        collision_margin: 0.1,
    };
    println!(
        "platform={} arch={} build=release={} rustc=see workflow log",
        std::env::consts::OS,
        std::env::consts::ARCH,
        !cfg!(debug_assertions)
    );
    for players in [1, 22] {
        let count = 4000;
        let start = Instant::now();
        let mut checksum = 0u64;
        for iteration in 0..count {
            for n in 0..players {
                let state = State {
                    position: Vec2::new(n as f64 * 2.0, 0.0),
                    velocity: Vec2::ZERO,
                };
                let goal = MotionGoal {
                    position: Vec2::new(n as f64 * 2.0 + 10.0, 2.0),
                    velocity: Vec2::ZERO,
                };
                let plan = black_box(plan(state, goal, config, &[]).unwrap());
                checksum = checksum.wrapping_add(plan.first_acceleration.x.to_bits());
                checksum = checksum.wrapping_add(plan.first_acceleration.y.to_bits());
                checksum = checksum.wrapping_add(iteration as u64);
            }
        }
        let nanos = start.elapsed().as_nanos();
        println!(
            "players={} iterations={} elapsed_ns={} avg_group_ns={} checksum={:016x}",
            players,
            count,
            nanos,
            nanos / (count as u128),
            checksum
        );
    }
}
