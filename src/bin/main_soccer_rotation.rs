//! Binary entrypoint for `main_soccer_rotation` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `soccer_engine::des::main_soccer_rotation`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin main_soccer_rotation`.

fn main() {
    soccer_engine::des::main_soccer_rotation::run();
}
