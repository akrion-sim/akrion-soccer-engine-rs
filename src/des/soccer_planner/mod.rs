//! Interactive 11-a-side soccer rotation planner: roster constraints, chemistry
//! rules, IP/MIP re-solve, pitch + solver tabbed UI.

pub mod model;
pub mod solve;
pub mod ui;

pub use model::{default_planner_request, PlannerPlayer, PlannerRequest, PlannerSynergy};
pub use solve::{
    build_problem_from_request, solve_planner, solve_planner_summary, solve_planner_with_controls,
    PlannerResponse, PlannerSolveControls,
};
pub use ui::{planner_page_html, planner_response_to_json, write_planner_artifacts};
