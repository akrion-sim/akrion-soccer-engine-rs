//! Streaming soccer rotation planner.
//!
//! This wraps the interactive planner request with a JSONL edit stream. Edits
//! add/remove player variables, add/remove planner constraints (status, banned
//! positions, fixed positions, chemistry), reshape the match, and solve on
//! demand with the soccer IP/MIP model.

use serde_json::{json, Value};

use crate::des::general::soccer_rotation::build_soccer_ipmip;
use crate::des::soccer_planner::{
    build_problem_from_request, default_planner_request, planner_response_to_json, solve_planner,
    solve_planner_summary, PlannerRequest,
};

use super::{
    bool_at, error_frame, op_of, ModelStreamKind, SolverKind, StreamContract, StreamEvent,
    StreamOp, StreamingModel,
};

/// Live soccer planner state driven by JSON commands.
pub struct StreamingSoccerPlanner {
    request: PlannerRequest,
}

impl Default for StreamingSoccerPlanner {
    fn default() -> Self {
        StreamingSoccerPlanner {
            request: default_planner_request(),
        }
    }
}

impl StreamingSoccerPlanner {
    pub fn new() -> Self {
        Self::default()
    }

    fn model_frame(&self) -> Value {
        match build_problem_from_request(&self.request) {
            Ok(problem) => {
                let model = build_soccer_ipmip(&problem);
                json!({
                    "event": "model",
                    "players": problem.num_players,
                    "positions": problem.num_positions,
                    "periods": problem.num_periods,
                    "benchSize": problem.bench_size,
                    "variables": model.ip.c.len(),
                    "constraints": model.ip.a.len(),
                    "synergies": self.request.synergies.len(),
                    "minSubsPerGame": self.request.min_subs_per_game,
                    "maxSubsPerGame": self.request.max_subs_per_game,
                    "defaultMinContiguousBlocks": self.request.default_min_contiguous_blocks,
                    "defaultMaxContiguousBlocks": self.request.default_max_contiguous_blocks,
                    "defaultMaxBenchBlocks": self.request.default_max_bench_blocks
                })
            }
            Err(err) => error_frame(err),
        }
    }

    fn init(&mut self, command: &Value) -> Vec<Value> {
        if let Some(raw) = command.get("request") {
            match serde_json::from_value::<PlannerRequest>(raw.clone()) {
                Ok(mut req) => {
                    crate::des::soccer_planner::model::normalize_planner_request(&mut req);
                    self.request = req;
                }
                Err(err) => return vec![error_frame(format!("invalid planner request: {err}"))],
            }
        } else {
            self.request = default_planner_request();
        }
        vec![json!({"event":"initialized"}), self.model_frame()]
    }

    fn solve(&self, command: &Value) -> Vec<Value> {
        let include_animations = bool_at(command, "includeAnimations", false);
        let response = if include_animations {
            solve_planner(&self.request)
        } else {
            solve_planner_summary(&self.request)
        };
        let mut payload = planner_response_to_json(&response);
        if !include_animations {
            if let Some(obj) = payload.as_object_mut() {
                obj.remove("pitchAnimation");
                obj.remove("solverAnimation");
            }
        }
        vec![json!({
            "event": "solution",
            "ok": response.ok,
            "status": response.mip_status,
            "affinity": response.affinity,
            "players": response.num_players,
            "positions": response.num_positions,
            "variables": response.num_variables,
            "constraints": response.num_constraints,
            "totalSubs": response.total_subs,
            "usedFallback": response.used_fallback,
            "response": payload
        })]
    }
}

impl StreamingModel for StreamingSoccerPlanner {
    fn kind(&self) -> SolverKind {
        SolverKind::IterativeSolver
    }

    fn contract(&self) -> StreamContract {
        StreamContract::new_for_stream_kind(
            "soccer-planner",
            SolverKind::IterativeSolver,
            ModelStreamKind::GenericIterative,
            "Soccer rotation planner edited by a command stream. Roster changes \
             are variable edits; status, fixed-position, banned-position, stamina, \
             substitution, and chemistry edits become IP/MIP constraints or \
             auxiliary variables before solving.",
            vec![
                StreamOp::new(
                    "init",
                    "Reset to the default planner, or pass `request` to replace the state.",
                    json!({"op":"init"}),
                ),
                StreamOp::new(
                    "set_match",
                    "Set match length, substitution bounds, default contiguous 5-minute block limits, seed, and solver budgets.",
                    json!({"op":"set_match","numPeriods":2,"minutesPerPeriod":45,"defaultMinContiguousBlocks":1,"defaultMaxContiguousBlocks":4,"defaultMaxBenchBlocks":3}),
                ),
                StreamOp::new(
                    "set_formation",
                    "Set outfield formation with `formation` like 2-3-1 or `outfieldFormation`.",
                    json!({"op":"set_formation","formation":"2-3-1"}),
                ),
                StreamOp::new(
                    "add_player",
                    "Add a player variable. Alias: add_variable.",
                    json!({"op":"add_player","name":"Guest 13","status":"guest","minContiguousBlocks":1,"maxContiguousBlocks":4,"maxBenchBlocks":3}),
                ),
                StreamOp::new(
                    "remove_player",
                    "Remove a player variable by id. Alias: remove_variable.",
                    json!({"op":"remove_player","id":12}),
                ),
                StreamOp::new(
                    "ban_position",
                    "Add a cannot-play-position constraint. Alias: add_constraint.",
                    json!({"op":"ban_position","player":4,"position":0}),
                ),
                StreamOp::new(
                    "allow_position",
                    "Remove a cannot-play-position constraint. Alias: remove_constraint.",
                    json!({"op":"allow_position","player":4,"position":0}),
                ),
                StreamOp::new(
                    "set_fixed_position",
                    "Force a player to one position every period.",
                    json!({"op":"set_fixed_position","player":0,"position":0}),
                ),
                StreamOp::new(
                    "set_position_score",
                    "Update one player-position objective coefficient.",
                    json!({"op":"set_position_score","player":8,"position":10,"score":0.92}),
                ),
                StreamOp::new(
                    "add_synergy",
                    "Add bidirectional auxiliary chemistry variables and linking constraints.",
                    json!({"op":"add_synergy","player":9,"position":10,"partnerPlayer":7,"partnerPosition":6,"scoreWith":0.92,"scoreWithout":0.78,"partnerScoreWith":0.9,"partnerScoreWithout":0.76}),
                ),
                StreamOp::new(
                    "snapshot",
                    "Emit current model dimensions without solving.",
                    json!({"op":"snapshot"}),
                ),
                StreamOp::new(
                    "solve",
                    "Solve the current planner. Set includeAnimations=true to include animation payloads.",
                    json!({"op":"solve","includeAnimations":false}),
                ),
            ],
            vec![
                StreamEvent::new("initialized", "Planner state reset or replaced."),
                StreamEvent::new("applied", "One streamed edit was applied."),
                StreamEvent::new("model", "Current IP/MIP model dimensions."),
                StreamEvent::new("solution", "Planner solve result and schedule summary."),
                StreamEvent::new("error", "A command could not be applied."),
            ],
        )
    }

    fn apply(&mut self, command: &Value) -> Vec<Value> {
        match op_of(command) {
            "init" => self.init(command),
            "snapshot" => vec![self.model_frame()],
            "solve" => self.solve(command),
            _ => match crate::des::soccer_planner::model::apply_planner_stream_command(
                &mut self.request,
                command,
            ) {
                Ok(applied) => vec![applied, self.model_frame()],
                Err(err) => vec![error_frame(err)],
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::streaming::drive;

    #[test]
    fn streamed_player_changes_model_dimensions() {
        let mut m = StreamingSoccerPlanner::new();
        let frames = drive(
            &mut m,
            &[
                json!({"op":"snapshot"}),
                json!({"op":"add_player","name":"Guest 19","status":"guest"}),
                json!({"op":"ban_position","player":18,"position":0}),
                json!({"op":"snapshot"}),
            ],
        );
        let models: Vec<&Value> = frames
            .iter()
            .filter(|f| f["event"] == json!("model"))
            .collect();
        assert!(models.len() >= 2);
        assert_eq!(models.first().unwrap()["players"], json!(18));
        assert_eq!(models.last().unwrap()["players"], json!(19));
        assert!(
            models.last().unwrap()["constraints"].as_u64().unwrap()
                > models.first().unwrap()["constraints"].as_u64().unwrap()
        );
    }
}
