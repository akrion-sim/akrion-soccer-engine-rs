//! `des::streaming` mirror: re-export the generic streaming framework, plus the
//! local soccer streaming planner.

pub use des_engine::des::streaming::{
    bool_at, drive, error_frame, op_of, JSONL_MEDIA_TYPE, ModelStreamKind, SolverKind,
    StreamContract, StreamEvent, StreamOp, StreamingModel,
};

pub mod soccer;
