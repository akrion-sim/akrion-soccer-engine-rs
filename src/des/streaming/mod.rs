//! `des::streaming` mirror: re-export the generic streaming framework, plus the
//! local soccer streaming planner.

pub use des_engine::des::streaming::{
    bool_at, drive, error_frame, op_of, ModelStreamKind, SolverKind, StreamContract, StreamEvent,
    StreamOp, StreamingModel, JSONL_MEDIA_TYPE,
};

pub mod soccer;
