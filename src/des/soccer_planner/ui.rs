//! Interactive soccer rotation planner UI (self-contained HTML).

use serde_json::Value;

use super::model::default_planner_request;
use super::solve::PlannerResponse;
use crate::des::animation::types::Animation;

/// Full interactive planner page served at `GET /soccer/planner`.
pub fn planner_page_html() -> String {
    let default_json = serde_json::to_string_pretty(&default_planner_request())
        .unwrap_or_else(|_| "{}".to_string());
    let default_escaped = default_json.replace("</script", "<\\/script");
    let git_commit = planner_git_commit();
    let git_commit_short = short_git_commit(git_commit);

    include_str!("planner_ui.html")
        .replace("__DEFAULT_CONFIG__", &default_escaped)
        .replace("__PLANNER_VERSION__", &html_escape(planner_version()))
        .replace("__PLANNER_GIT_COMMIT__", &html_escape(git_commit))
        .replace(
            "__PLANNER_GIT_COMMIT_SHORT__",
            &html_escape(&git_commit_short),
        )
}

fn planner_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

fn planner_git_commit() -> &'static str {
    option_env!("DES_ENGINE_GIT_COMMIT").unwrap_or("unknown")
}

fn short_git_commit(commit: &str) -> String {
    if commit == "unknown" {
        commit.to_string()
    } else {
        commit.chars().take(12).collect()
    }
}

fn html_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

/// Serialize a planner response for the JSON API (animations as JSON objects).
pub fn planner_response_to_json(resp: &PlannerResponse) -> Value {
    let pitch: Value = animation_to_serde(&resp.pitch_animation);
    let solver: Value = animation_to_serde(&resp.solver_animation);
    serde_json::json!({
        "ok": resp.ok,
        "error": resp.error,
        "constraintViolations": resp.constraint_violations.iter().map(|v| serde_json::json!({
            "kind": v.kind,
            "constraint": v.constraint,
            "message": v.message,
            "suggestion": v.suggestion,
        })).collect::<Vec<_>>(),
        "suggestedFix": resp.suggested_fix,
        "affinity": resp.affinity,
        "totalSubs": resp.total_subs,
        "fairnessOk": resp.fairness_ok,
        "staminaOk": resp.stamina_ok,
        "subsOk": resp.subs_ok,
        "mipStatus": resp.mip_status,
        "elapsedMs": resp.elapsed_ms,
        "nodesExplored": resp.nodes_explored,
        "numPlayers": resp.num_players,
        "numPositions": resp.num_positions,
        "numVariables": resp.num_variables,
        "numConstraints": resp.num_constraints,
        "usedFallback": resp.used_fallback,
        "fallbackReason": resp.fallback_reason,
        "assignment": resp.assignment,
        "bench": resp.bench,
        "solverNotes": resp.solver_notes,
        "alternatives": resp.alternatives.iter().map(|alt| serde_json::json!({
            "rank": alt.rank,
            "affinity": alt.affinity,
            "totalSubs": alt.total_subs,
            "assignment": alt.assignment,
            "bench": alt.bench,
        })).collect::<Vec<_>>(),
        "pitchAnimation": pitch,
        "solverAnimation": solver,
    })
}

fn animation_to_serde(anim: &Animation) -> Value {
    let s = anim.to_json().to_string();
    serde_json::from_str(&s).unwrap_or(Value::Null)
}

/// Write static export: interactive UI to `out/soccer-planner.html`.
pub fn write_planner_artifacts() {
    let ui_path = std::path::Path::new("out/soccer-planner.html");
    let _ = std::fs::create_dir_all("out");
    let _ = std::fs::write(ui_path, planner_page_html());
    println!("# Interactive planner UI: {}", ui_path.display());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::general::soccer_rotation::PlayerStatus;

    #[test]
    fn default_request_is_11_a_side_with_18_players() {
        let req = default_planner_request();
        assert_eq!(req.outfield_formation, vec![4, 4, 2]);
        assert_eq!(req.players.len(), 18);
        assert_eq!(req.max_subs_per_game, 119);
        assert_eq!(req.min_subs_per_game, 0);
        assert_eq!(req.default_min_contiguous_blocks, 1);
        assert_eq!(req.default_max_contiguous_blocks, 4);
        assert_eq!(req.default_max_bench_blocks, 3);
        assert_eq!(req.solver_max_nodes, 20_000);
    }

    #[test]
    fn planner_page_includes_build_metadata() {
        let html = planner_page_html();
        assert!(html.contains(&format!("data-version=\"{}\"", env!("CARGO_PKG_VERSION"))));
        assert!(html.contains("id=\"buildInfo\""));
        assert!(!html.contains("__PLANNER_VERSION__"));
        assert!(!html.contains("__PLANNER_GIT_COMMIT"));
    }

    #[test]
    fn commit_display_helpers_are_safe_for_untrusted_labels() {
        assert_eq!(
            short_git_commit("abcdef1234567890"),
            "abcdef123456".to_string()
        );
        let accented = "\u{e9}".repeat(13);
        assert_eq!(short_git_commit(&accented), "\u{e9}".repeat(12));
        assert_eq!(
            html_escape("<tag attr='x'>&\""),
            "&lt;tag attr=&#39;x&#39;&gt;&amp;&quot;".to_string()
        );
    }

    #[test]
    fn awol_player_is_not_fieldable_in_problem() {
        let mut req = default_planner_request();
        req.players[5].status = "awol".to_string();
        let problem = super::super::solve::build_problem_from_request(&req).expect("build");
        assert!(!crate::des::general::soccer_rotation::player_is_fieldable(
            &problem, 5
        ));
        assert_eq!(
            problem.player_status.as_ref().unwrap()[5],
            PlayerStatus::Awol
        );
    }
}
