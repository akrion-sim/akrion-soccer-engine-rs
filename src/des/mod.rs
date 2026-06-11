//! Mirror of the original `des::` module tree. Generic subsystems are
//! re-exported from `des_engine` so the relocated soccer files keep their
//! `crate::des::…` paths; soccer subsystems are local modules.

// Generic subsystems re-exported verbatim from des_engine.
pub use des_engine::des::shared;

// Hybrid namespaces: re-export the generic des_engine pieces soccer uses, plus
// the local soccer modules, under the original paths.
pub mod animation;
pub mod general;
pub mod streaming;

// Soccer-only subsystems.
pub mod runners;
pub mod soccer_learning;
pub mod soccer_learning_pg;
pub mod soccer_planner;

// Thin demo runners (the bins wrap these).
pub mod main_soccer;
pub mod main_soccer_live;
pub mod main_soccer_planner;
pub mod main_soccer_rotation;

#[cfg(test)]
pub mod test;
