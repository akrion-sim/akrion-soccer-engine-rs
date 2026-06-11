//! Binary entrypoint for `main_soccer_rotation_anim` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module
//! `soccer_engine::des::main_soccer_rotation::run_anim` — the always-animating
//! showcase (13-player squad, 7-a-side, four 10-minute periods, stamina cap of
//! 3 consecutive periods, IP/MIP focus). The real logic lives in the module;
//! this binary just exposes it as `cargo run --bin main_soccer_rotation_anim`.

fn main() {
    soccer_engine::des::main_soccer_rotation::run_anim();
}
