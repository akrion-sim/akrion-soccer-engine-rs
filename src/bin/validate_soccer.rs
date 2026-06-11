//! Binary entrypoint for `validate_soccer` (hybrid layout).
//!
//! Thin wrapper that delegates to the library module `soccer_engine::des::runners::validate_soccer`. The real logic
//! lives in the module (kept testable in-tree); this binary just exposes it as
//! `cargo run --bin validate_soccer`.

fn main() {
    soccer_engine::des::runners::validate_soccer::run();
}
