//! Feature adapter + export contract between the soccer engine and the Burn POMDP solver.
//!
//! The engine already computes a per-entity **field-motion block**: 22 players + 1 ball, 8 channels
//! each (forward, lateral, vel_forward, vel_lateral, + 4 more) = 184 floats — which is *exactly* the
//! `(entities, entity_dim)` structure the attention encoder wants. So the adapter just reshapes it
//! (no lossy hand-engineering) and stacks decisions into a trajectory tensor. The engine writes
//! `Decision` rows (JSONL); the trainer loads them. This is the seam that keeps the engine build
//! decoupled from Burn.

use burn::tensor::{backend::Backend, Int, Tensor, TensorData};
use serde::{Deserialize, Serialize};

pub const N_ENTITIES: usize = 23; // 22 players + ball
pub const ENTITY_DIM: usize = 8; // field-motion channels per entity
pub const FIELD_MOTION_DIM: usize = N_ENTITIES * ENTITY_DIM; // 184

/// One decision row — the engine exports this per learned-policy decision.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Decision {
    /// The engine's whole-field motion block, length `FIELD_MOTION_DIM` (row-major entity×channel).
    pub field_motion: Vec<f32>,
    /// Index into the policy's action-family space (matches `soccer_policy_action_index`).
    pub action: usize,
    /// Per-decision reward (already zero-sum/MARL-adjusted engine-side, or raw — trainer-agnostic).
    pub reward: f32,
    /// Trajectory boundary — true at the last decision of a possession/agent trajectory.
    pub done: bool,
}

pub type Trajectory = Vec<Decision>;

/// Discounted Monte-Carlo returns (backward accumulation, reset at trajectory boundaries).
pub fn returns(traj: &Trajectory, gamma: f32) -> Vec<f32> {
    let mut g = vec![0f32; traj.len()];
    let mut acc = 0f32;
    for i in (0..traj.len()).rev() {
        if traj[i].done {
            acc = 0.0;
        }
        acc = traj[i].reward + gamma * acc;
        g[i] = acc;
    }
    g
}

/// Trajectory → `(entities (T,23,8), actions (T), returns (T))` on the given device.
pub fn to_tensors<B: Backend>(
    traj: &Trajectory,
    gamma: f32,
    dev: &B::Device,
) -> (Tensor<B, 3>, Tensor<B, 1, Int>, Tensor<B, 1>) {
    let t = traj.len();
    let mut ent = Vec::with_capacity(t * FIELD_MOTION_DIM);
    let mut acts = Vec::with_capacity(t);
    for d in traj {
        assert_eq!(d.field_motion.len(), FIELD_MOTION_DIM, "field_motion must be {FIELD_MOTION_DIM}");
        ent.extend_from_slice(&d.field_motion);
        acts.push(d.action as i64);
    }
    let g = returns(traj, gamma);
    let entities = Tensor::<B, 1>::from_data(TensorData::new(ent, [t * FIELD_MOTION_DIM]), dev)
        .reshape([t, N_ENTITIES, ENTITY_DIM]);
    let actions = Tensor::<B, 1, Int>::from_data(TensorData::new(acts, [t]), dev);
    let returns_t = Tensor::<B, 1>::from_data(TensorData::new(g, [t]), dev);
    (entities, actions, returns_t)
}

/// Load a JSONL export: one `{ "trajectory": [Decision, ...] }` per line, or one `Decision` per
/// line with `done` marking boundaries. Here: one JSON array (a trajectory) per line.
pub fn load_jsonl(path: &str) -> std::io::Result<Vec<Trajectory>> {
    use std::io::BufRead;
    let f = std::fs::File::open(path)?;
    let mut out = Vec::new();
    for line in std::io::BufReader::new(f).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let traj: Trajectory = serde_json::from_str(&line)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        out.push(traj);
    }
    Ok(out)
}
