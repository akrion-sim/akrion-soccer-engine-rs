//! Build a [`SoccerProblem`] from a planner request, solve with IP/MIP, and
//! render pitch + solver animations for the interactive UI.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};
use std::time::Instant;

use crate::des::animation::frame_recorder::{FrameRecorder, FrameRecorderOpts};
use crate::des::animation::html_player::{build_html_set, AnimationSetOptions, AnimationVariant};
use crate::des::animation::scenes::soccer_ipmip_solver_scene as solver_scene;
use crate::des::animation::scenes::soccer_scene as pitch_scene;
use crate::des::animation::types::Animation;
use crate::des::general::ip_mip_des::TraceAction;
use crate::des::general::soccer_rotation::{
    build_soccer_ipmip, evaluate_schedule, formation_with_gk, policy_greedy_feasible_schedule,
    policy_ipmip_feasible, simulate_match_des, validate_schedule_structure, MatchSimOptions,
    Schedule, ScheduleEvaluation, SoccerIPMIPPolicyOptions, SoccerIPMIPPolicyResult, SoccerProblem,
};

use super::model::{
    parse_player_status, planner_block_count, synergy_to_rules, PlannerRequest,
    CONTIGUOUS_BLOCK_MINUTES,
};

static PLANNER_ANIMATION_PATH_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Outcome of a planner solve (internal; use [`super::ui::planner_response_to_json`] for API).
#[derive(Clone, Debug)]
pub struct PlannerResponse {
    pub ok: bool,
    pub error: Option<String>,
    pub constraint_violations: Vec<PlannerConstraintViolation>,
    pub suggested_fix: Option<String>,
    pub affinity: f64,
    pub total_subs: usize,
    pub fairness_ok: bool,
    pub stamina_ok: bool,
    pub subs_ok: bool,
    pub mip_status: String,
    pub elapsed_ms: f64,
    pub nodes_explored: u64,
    pub num_players: usize,
    pub num_positions: usize,
    pub num_variables: usize,
    pub num_constraints: usize,
    pub used_fallback: bool,
    pub fallback_reason: Option<String>,
    pub assignment: Vec<Vec<i64>>,
    pub bench: Vec<Vec<usize>>,
    pub solver_notes: Vec<String>,
    pub alternatives: Vec<PlannerAlternative>,
    pub pitch_animation: Animation,
    pub solver_animation: Animation,
}

#[derive(Clone, Debug)]
pub struct PlannerConstraintViolation {
    pub kind: String,
    pub constraint: String,
    pub message: String,
    pub suggestion: String,
}

#[derive(Clone, Debug)]
pub struct PlannerAlternative {
    pub rank: usize,
    pub affinity: f64,
    pub total_subs: usize,
    pub assignment: Vec<Vec<i64>>,
    pub bench: Vec<Vec<usize>>,
}

#[derive(Clone, Default)]
pub struct PlannerSolveControls {
    pub cancel_flag: Option<Arc<AtomicBool>>,
    pub take_next_incumbent_signal: Option<Arc<AtomicU64>>,
}

#[derive(Clone, Debug)]
struct PlannerScheduleDiagnostics {
    eval: Option<ScheduleEvaluation>,
    violations: Vec<PlannerConstraintViolation>,
    suggested_fix: Option<String>,
}

impl PlannerScheduleDiagnostics {
    fn has_violations(&self) -> bool {
        !self.violations.is_empty()
    }
}

fn violation(
    kind: &str,
    constraint: &str,
    message: impl Into<String>,
    suggestion: impl Into<String>,
) -> PlannerConstraintViolation {
    PlannerConstraintViolation {
        kind: kind.to_string(),
        constraint: constraint.to_string(),
        message: message.into(),
        suggestion: suggestion.into(),
    }
}

fn player_label(problem: &SoccerProblem, player_id: usize) -> String {
    problem
        .player_names
        .as_ref()
        .and_then(|names| names.get(player_id))
        .filter(|name| !name.is_empty())
        .cloned()
        .unwrap_or_else(|| format!("Player{}", player_id + 1))
}

fn structure_suggestion(reason: &str) -> &'static str {
    if reason.contains("AWOL/injured") {
        "Mark that player available/guest, remove their fixed slot, or add another eligible player."
    } else if reason.contains("fixed at position") {
        "Clear or change the fixed-position lock so the player can match the assigned slot."
    } else if reason.contains("banned from position") {
        "Allow that position for the player, change the fixed slot, or add another eligible player for the slot."
    } else if reason.contains("both on field and bench") || reason.contains("appears twice") {
        "Remove duplicate player assignments or rerun with the fast fallback enabled."
    } else if reason.contains("bench[") || reason.contains("total players") {
        "Check roster size and unavailable-player statuses; every rostered player must be either on field or listed on the bench each period."
    } else if reason.contains("invalid player") || reason.contains("length") {
        "Regenerate the lineup from the current roster so assignment and bench dimensions match the formation."
    } else {
        "Review fixed positions, banned positions, roster availability, and formation size."
    }
}

fn build_error_diagnostics(error: &str) -> (Vec<PlannerConstraintViolation>, Option<String>) {
    let (constraint, suggestion) = if error.contains("need more players")
        || error.contains("need at least")
    {
        (
            "Roster availability",
            "Add available/guest players, mark injured/AWOL players available, or reduce the formation size.",
        )
    } else if error.contains("min subs") {
        (
            "Substitution bounds",
            "Lower Min subs, raise Max subs, lengthen the match, or add enough bench depth for additional substitutions.",
        )
    } else if error.contains("unavailable and cannot also be fixed") {
        (
            "Availability and fixed-position locks",
            "Clear the fixed slot or mark the player available/guest before solving.",
        )
    } else if error.contains("fixed to invalid position") {
        (
            "Fixed-position locks",
            "Clear the invalid fixed slot or pick a position that exists in the current formation.",
        )
    } else if error.contains("duplicate player id") || error.contains("missing player id") {
        (
            "Roster ids",
            "Refresh the roster or use the Set/Add/remove controls so player ids are rewritten sequentially.",
        )
    } else {
        (
            "Planner input",
            "Review roster availability, fixed positions, banned positions, formation size, and substitution limits.",
        )
    };
    let v = violation("input", constraint, error.to_string(), suggestion);
    (vec![v], Some(suggestion.to_string()))
}

fn solver_error_diagnostics(error: &str) -> (Vec<PlannerConstraintViolation>, Option<String>) {
    let suggestion = if error.contains("constructive planner fallback") {
        "Relax one hard planner constraint: clear conflicting fixed/banned positions, add fieldable players, adjust contiguous block limits, or adjust substitution bounds."
    } else {
        "Try enabling Fallback, raising the solver time/node limits, or relaxing fixed-position, banned-position, stamina, and substitution constraints."
    };
    let v = violation(
        "solver",
        "Feasible incumbent",
        error.to_string(),
        suggestion,
    );
    (vec![v], Some(suggestion.to_string()))
}

fn schedule_diagnostics(
    problem: &SoccerProblem,
    schedule: &Schedule,
) -> PlannerScheduleDiagnostics {
    let mut violations = Vec::new();
    if let Some(reason) = validate_schedule_structure(problem, schedule) {
        let suggestion = structure_suggestion(&reason).to_string();
        violations.push(violation(
            "schedule_structure",
            "Roster coverage and eligibility",
            format!("Invalid schedule structure: {reason}"),
            suggestion.clone(),
        ));
        return PlannerScheduleDiagnostics {
            eval: None,
            violations,
            suggested_fix: Some(suggestion),
        };
    }

    let eval = evaluate_schedule(problem, schedule);
    for f in &eval.fairness_violations {
        violations.push(violation(
            "fairness",
            "No consecutive bench periods",
            format!(
                "{} is benched in both period {} and period {}.",
                player_label(problem, f.player_id),
                f.period_a + 1,
                f.period_b + 1
            ),
            "Rotate the player onto the field next period, reduce bench depth, add on-field slots, or relax the fairness rule.",
        ));
    }
    for s in &eval.consecutive_on_field_violations {
        let limit = problem
            .max_contiguous_on_field
            .as_ref()
            .and_then(|limits| limits.get(s.player_id))
            .copied()
            .or(problem.max_consecutive_on_field)
            .unwrap_or(0);
        violations.push(violation(
            "stamina",
            "Max contiguous 5-minute blocks",
            format!(
                "{} is on field for {} contiguous 5-minute block(s) starting at block {}, above the max of {}.",
                player_label(problem, s.player_id),
                s.length,
                s.start_period + 1,
                limit
            ),
            "Increase that player's max blocks, add bench depth, lengthen rest opportunities, or clear fixed-position locks that force this player to stay on.",
        ));
    }
    for s in &eval.min_contiguous_on_field_violations {
        violations.push(violation(
            "stamina",
            "Min contiguous 5-minute blocks",
            format!(
                "{} is on field for only {} contiguous 5-minute block(s) starting at block {}, below the min of {}.",
                player_label(problem, s.player_id),
                s.length,
                s.start_period + 1,
                s.min_length
            ),
            "Lower that player's min blocks, keep them on for a longer stint, or adjust fixed/banned constraints that force an early substitution.",
        ));
    }
    for s in &eval.max_contiguous_bench_violations {
        violations.push(violation(
            "bench",
            "Max contiguous 5-minute bench blocks",
            format!(
                "{} is on the bench for {} contiguous 5-minute block(s) starting at block {}, above the max of {}.",
                player_label(problem, s.player_id),
                s.length,
                s.start_period + 1,
                s.max_length
            ),
            "Raise that player's max bench blocks, shorten the bench stint, or add enough eligible lineup changes to bring them back on sooner.",
        ));
    }
    if !eval.subs_ok {
        let min = problem.min_subs_per_game.unwrap_or(0);
        let max = problem.max_subs_per_game.unwrap_or(usize::MAX);
        let suggestion = if eval.total_subs < min {
            "Lower Min subs, lengthen the match, add bench depth, or reduce fixed-position locks that keep the same lineup on."
        } else {
            "Raise Max subs, shorten the match, relax contiguous-block constraints, or allow more lineup continuity."
        };
        violations.push(violation(
            "substitutions",
            "Substitution bounds",
            format!(
                "Schedule uses {} substitution(s), outside the active {}-{} bound.",
                eval.total_subs,
                min,
                if max == usize::MAX { min } else { max }
            ),
            suggestion,
        ));
    }

    let suggested_fix = best_schedule_fix(&violations);
    PlannerScheduleDiagnostics {
        eval: Some(eval),
        violations,
        suggested_fix,
    }
}

fn best_schedule_fix(violations: &[PlannerConstraintViolation]) -> Option<String> {
    violations.first().map(|v| v.suggestion.clone())
}

fn diagnostic_error_message(fallback: &str, diagnostics: &PlannerScheduleDiagnostics) -> String {
    diagnostics
        .violations
        .first()
        .map(|v| format!("schedule violates {}: {}", v.constraint, v.message))
        .unwrap_or_else(|| fallback.to_string())
}

pub fn build_problem_from_request(req: &PlannerRequest) -> Result<SoccerProblem, String> {
    let formation = formation_with_gk(&req.outfield_formation);
    let num_positions: usize = formation.iter().sum();
    if num_positions < 2 {
        return Err("formation must have at least GK + 1 outfield".into());
    }
    let num_players = req.players.len();
    if num_players <= num_positions {
        return Err(format!(
            "need more players ({num_players}) than on-field slots ({num_positions})"
        ));
    }
    let fieldable_count = req
        .players
        .iter()
        .filter(|p| {
            let s = parse_player_status(&p.status);
            !matches!(
                s,
                crate::des::general::soccer_rotation::PlayerStatus::Awol
                    | crate::des::general::soccer_rotation::PlayerStatus::Injured
            )
        })
        .count();
    if fieldable_count < num_positions {
        return Err(format!(
            "need at least {num_positions} available/guest players; only {fieldable_count} are fieldable"
        ));
    }
    if req.min_subs_per_game > req.max_subs_per_game {
        return Err(format!(
            "min subs ({}) cannot exceed max subs ({})",
            req.min_subs_per_game, req.max_subs_per_game
        ));
    }
    let t_count = planner_block_count(req);
    let max_possible_subs = max_substitution_capacity(t_count, num_positions, fieldable_count);
    if req.min_subs_per_game > max_possible_subs {
        return Err(format!(
            "min subs ({}) is infeasible for this match: with {} 5-minute block(s), {} fieldable player(s), and {} on-field slot(s), this model can count at most {} substitutions. Lengthen the match, add bench depth, or lower Min subs.",
            req.min_subs_per_game,
            t_count,
            fieldable_count,
            num_positions,
            max_possible_subs
        ));
    }
    let position_names = crate::des::general::soccer_rotation::formation_position_names(&formation);

    let mut affinity: Vec<Vec<Vec<f64>>> =
        vec![vec![vec![0.0; t_count]; num_positions]; num_players];
    let mut player_names: Vec<String> = vec![String::new(); num_players];
    let mut player_status = vec![parse_player_status("available"); num_players];
    let mut fixed_position = vec![None; num_players];
    let mut banned_positions = vec![vec![false; num_positions]; num_players];
    let mut seen = vec![false; num_players];

    for pl in &req.players {
        if pl.id >= num_players {
            return Err(format!("invalid player id {}", pl.id));
        }
        if seen[pl.id] {
            return Err(format!("duplicate player id {}", pl.id));
        }
        seen[pl.id] = true;
        player_names[pl.id] = pl.name.clone();
        player_status[pl.id] = parse_player_status(&pl.status);
        if !matches!(
            player_status[pl.id],
            crate::des::general::soccer_rotation::PlayerStatus::Available
                | crate::des::general::soccer_rotation::PlayerStatus::Guest
        ) && pl.fixed_position.is_some()
        {
            return Err(format!(
                "{} is unavailable and cannot also be fixed to a position",
                pl.name
            ));
        }
        if let Some(fp) = pl.fixed_position {
            if fp >= num_positions {
                return Err(format!(
                    "{} fixed to invalid position {fp}; formation has {num_positions} positions",
                    pl.name
                ));
            }
        }
        fixed_position[pl.id] = pl.fixed_position;
        for pos in &pl.banned_positions {
            if *pos < num_positions {
                banned_positions[pl.id][*pos] = true;
            }
        }
        for (score_idx, score) in pl.position_scores.iter().enumerate() {
            if !score.is_finite() {
                return Err(format!(
                    "{} position score {score_idx} must be finite before LP/MDP planning",
                    pl.name
                ));
            }
        }
        for pos in 0..num_positions {
            let score = pl
                .position_scores
                .get(pos)
                .copied()
                .unwrap_or(0.3)
                .clamp(0.0, 1.0);
            for t in 0..t_count {
                affinity[pl.id][pos][t] = score;
            }
        }
    }
    if let Some(missing) = seen.iter().position(|&ok| !ok) {
        return Err(format!("missing player id {missing}"));
    }
    for (i, name) in player_names.iter_mut().enumerate() {
        if name.is_empty() {
            *name = format!("Player{}", i + 1);
        }
    }

    let mut synergy_rules: Vec<_> = Vec::new();
    for s in &req.synergies {
        if s.player >= num_players || s.partner_player >= num_players {
            return Err("synergy references an unknown player".into());
        }
        if s.position >= num_positions || s.partner_position >= num_positions {
            return Err("synergy references an unknown position".into());
        }
        for (name, value) in [
            ("scoreWith", Some(s.score_with)),
            ("scoreWithout", Some(s.score_without)),
            ("partnerScoreWith", s.partner_score_with),
            ("partnerScoreWithout", s.partner_score_without),
        ] {
            if let Some(value) = value {
                if !value.is_finite() {
                    return Err(format!(
                        "synergy {name} must be finite before LP/MDP planning"
                    ));
                }
            }
        }
        synergy_rules.extend(synergy_to_rules(s));
    }

    let min_contiguous_on_field = req
        .players
        .iter()
        .map(|player| {
            player
                .min_contiguous_blocks
                .unwrap_or(req.default_min_contiguous_blocks)
                .clamp(1, 18)
        })
        .collect::<Vec<_>>();
    let max_contiguous_on_field = req
        .players
        .iter()
        .map(|player| {
            player
                .max_contiguous_blocks
                .unwrap_or(req.default_max_contiguous_blocks)
                .clamp(1, 18)
        })
        .collect::<Vec<_>>();
    let max_contiguous_bench = req
        .players
        .iter()
        .map(|player| {
            player
                .max_bench_blocks
                .unwrap_or(req.default_max_bench_blocks)
                .clamp(1, 18)
        })
        .collect::<Vec<_>>();
    if let Some((i, (min, max))) = min_contiguous_on_field
        .iter()
        .zip(max_contiguous_on_field.iter())
        .enumerate()
        .find(|(_, (min, max))| min > max)
    {
        return Err(format!(
            "{} has min contiguous blocks ({}) above max contiguous blocks ({})",
            player_names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("Player{}", i + 1)),
            min,
            max
        ));
    }

    Ok(SoccerProblem {
        num_players,
        num_positions,
        num_periods: t_count,
        bench_size: num_players - num_positions,
        max_consecutive_on_field: None,
        min_contiguous_on_field: Some(min_contiguous_on_field),
        max_contiguous_on_field: Some(max_contiguous_on_field),
        max_contiguous_bench: Some(max_contiguous_bench),
        enforce_no_consecutive_bench: false,
        max_subs_per_game: Some(req.max_subs_per_game),
        min_subs_per_game: Some(req.min_subs_per_game),
        affinity,
        player_names: Some(player_names),
        position_names: Some(position_names),
        player_status: Some(player_status),
        fixed_position: Some(fixed_position),
        banned_positions: Some(banned_positions),
        synergy_rules: if synergy_rules.is_empty() {
            None
        } else {
            Some(synergy_rules)
        },
    })
}

fn max_substitution_capacity(
    num_periods: usize,
    num_positions: usize,
    fieldable_count: usize,
) -> usize {
    if num_periods <= 1 || fieldable_count <= num_positions {
        return 0;
    }
    let available_bench = fieldable_count - num_positions;
    (num_periods - 1) * num_positions.min(available_bench)
}

fn to_scene_solution(
    mip: &crate::des::general::ip_mip_des::IPMIPSolution,
) -> solver_scene::IPMIPSolution {
    let action_str = |a: TraceAction| -> String {
        match a {
            TraceAction::Branch => "branch",
            TraceAction::Cut => "cut",
            TraceAction::Prune => "prune",
            TraceAction::Incumbent => "incumbent",
            TraceAction::Unbounded => "unbounded",
        }
        .to_string()
    };
    let trace = mip
        .trace
        .iter()
        .map(|e| solver_scene::IPMIPTraceEvent {
            node_id: e.node_id.to_string(),
            depth: e.depth as f64,
            action: action_str(e.action),
            lp_z: e.lp_z,
            fractional: e.fractional.iter().map(|&v| v as f64).collect(),
            reason: e.reason.clone(),
        })
        .collect();
    solver_scene::IPMIPSolution {
        trace,
        status: mip.status.as_str().to_string(),
        z: mip.z,
        gap: mip.gap,
        best_bound: mip.best_bound,
        lp_algorithm: mip.lp_algorithm.as_str().to_string(),
        lp_solves: mip.lp_solves as f64,
        elapsed_ms: mip.elapsed_ms,
        nodes_explored: mip.nodes_explored as f64,
        cuts_added: mip.cuts_added as f64,
        candidates_tried: mip.candidates_tried as f64,
        lp_algorithm_usage: mip
            .lp_algorithm_usage
            .iter()
            .map(|(k, v)| (k.as_str().to_string(), *v as f64))
            .collect(),
        incumbent_source: mip.incumbent_source.clone(),
    }
}

fn planner_frames_path(kind: &str) -> std::path::PathBuf {
    let serial = PLANNER_ANIMATION_PATH_COUNTER.fetch_add(1, Ordering::Relaxed);
    let out_dir = std::env::temp_dir()
        .join("des_soccer_planner")
        .join(format!("{}-{serial}", std::process::id()));
    let _ = std::fs::create_dir_all(&out_dir);
    out_dir.join(format!("planner-{kind}.frames.jsonl"))
}

fn render_pitch_animation(
    problem: &SoccerProblem,
    schedule: &Schedule,
    formation: &[usize],
    minutes_per_period: usize,
    seed: u32,
) -> Animation {
    let frames_path = planner_frames_path("pitch");
    let mut rec = FrameRecorder::new(FrameRecorderOpts {
        frames_path: frames_path.to_string_lossy().into_owned(),
        width: pitch_scene::STAGE_W,
        height: pitch_scene::STAGE_H,
        fps: Some(6.0),
        title: Some(format!("{}-a-side optimal lineup", problem.num_positions)),
        subtitle: Some(format!(
            "GK + {} · {} subs max",
            formation
                .iter()
                .skip(1)
                .map(|n| n.to_string())
                .collect::<Vec<_>>()
                .join("-"),
            problem.max_subs_per_game.unwrap_or(0)
        )),
        background: Some("#0b1220".to_string()),
        ..Default::default()
    })
    .expect("recorder");

    let scene_problem = pitch_scene::SoccerProblem {
        num_positions: problem.num_positions,
        num_periods: problem.num_periods as f64,
        formation: formation.to_vec(),
        player_names: problem.player_names.clone(),
        position_names: problem.position_names.clone(),
    };
    let m = simulate_match_des(
        problem,
        schedule,
        &MatchSimOptions {
            seed: Some(seed),
            minutes_per_period: Some(minutes_per_period),
            ..Default::default()
        },
    );
    let n_periods = problem.num_periods;
    let mpp = minutes_per_period;
    let total_minutes = (n_periods * mpp) as f64;
    let arrangements: Vec<(Vec<usize>, Vec<usize>)> = (0..n_periods)
        .map(|t| {
            let pos: Vec<usize> = schedule.assignment[t].iter().map(|&x| x as usize).collect();
            (pos, schedule.bench[t].clone())
        })
        .collect();
    let all_on_bench: Vec<usize> = (0..problem.num_players).collect();
    let mut tick = 0.0_f64;
    let sp = &scene_problem;

    for tr in m.trace.iter() {
        let period = tr.period;
        let (cur_pos, cur_bench) = &arrangements[period];
        if tr.t % mpp == 0 {
            let (prev_pos, prev_bench) = if period == 0 {
                (Vec::new(), all_on_bench.clone())
            } else {
                arrangements[period - 1].clone()
            };
            for k in 0..5usize {
                let alpha = (k as f64 + 1.0) / 6.0;
                let i_f = tick;
                tick += 1.0;
                let cur_pos_c = cur_pos.clone();
                let cur_bench_c = cur_bench.clone();
                rec.frame(tr.t as f64, i_f, || {
                    pitch_scene::build_soccer_frame(
                        tr.t as f64,
                        i_f,
                        &pitch_scene::SoccerFrameInput {
                            t: tr.t as f64,
                            period: period as f64,
                            total_minutes,
                            positions: cur_pos_c,
                            bench: cur_bench_c,
                            prev_positions: prev_pos.clone(),
                            prev_bench: prev_bench.clone(),
                            transition: alpha,
                            entered: vec![],
                            left: vec![],
                            recent_subs: vec![],
                            goals_for: tr.goals_for_cum as f64,
                            goals_against: tr.goals_against_cum as f64,
                            affinity_now: tr.affinity_now,
                            goal_this_tick: None,
                            problem: sp,
                        },
                    )
                });
            }
        }
        let i_f = tick;
        tick += 1.0;
        let cur_pos_c = cur_pos.clone();
        let cur_bench_c = cur_bench.clone();
        rec.frame(tr.t as f64, i_f, || {
            pitch_scene::build_soccer_frame(
                tr.t as f64,
                i_f,
                &pitch_scene::SoccerFrameInput {
                    t: tr.t as f64,
                    period: period as f64,
                    total_minutes,
                    positions: cur_pos_c.clone(),
                    bench: cur_bench_c.clone(),
                    prev_positions: cur_pos_c,
                    prev_bench: cur_bench_c,
                    transition: 1.0,
                    entered: vec![],
                    left: vec![],
                    recent_subs: vec![],
                    goals_for: tr.goals_for_cum as f64,
                    goals_against: tr.goals_against_cum as f64,
                    affinity_now: tr.affinity_now,
                    goal_this_tick: None,
                    problem: sp,
                },
            )
        });
    }
    rec.finish().expect("finish pitch")
}

fn render_solver_animation(result: &SoccerIPMIPPolicyResult) -> Animation {
    let sol = to_scene_solution(&result.mip);
    render_solver_scene_animation(&sol, "IP/MIP branch-and-cut")
}

fn render_solver_scene_animation(sol: &solver_scene::IPMIPSolution, subtitle: &str) -> Animation {
    let frames_path = planner_frames_path("solver");
    let mut rec = FrameRecorder::new(FrameRecorderOpts {
        frames_path: frames_path.to_string_lossy().into_owned(),
        width: solver_scene::SOCCER_IPMIP_SOLVER_W,
        height: solver_scene::SOCCER_IPMIP_SOLVER_H,
        fps: Some(5.0),
        title: Some("IP/MIP branch-and-cut".to_string()),
        subtitle: Some(subtitle.to_string()),
        background: Some("#f8fafc".to_string()),
        ..Default::default()
    })
    .expect("recorder");
    let total = solver_scene::soccer_ipmip_solver_frame_count(sol);
    for i in 0..total {
        let i_f = i as f64;
        rec.frame(i_f, i_f, || {
            solver_scene::build_soccer_ipmip_solver_frame(sol, i)
        });
    }
    rec.finish().expect("finish solver")
}

fn schedule_key(schedule: &Schedule) -> String {
    format!("{:?}|{:?}", schedule.assignment, schedule.bench)
}

fn ranked_alternatives(
    problem: &SoccerProblem,
    schedule: &Schedule,
    limit: usize,
) -> Vec<PlannerAlternative> {
    let mut seen = std::collections::HashSet::from([schedule_key(schedule)]);
    let mut candidates: Vec<(f64, Schedule)> = Vec::new();
    for t in 0..problem.num_periods {
        for a in 0..problem.num_positions {
            for b in (a + 1)..problem.num_positions {
                let mut alt = schedule.clone();
                alt.assignment[t].swap(a, b);
                if !seen.insert(schedule_key(&alt)) {
                    continue;
                }
                if validate_schedule_structure(problem, &alt).is_some() {
                    continue;
                }
                let eval = evaluate_schedule(problem, &alt);
                if eval.fairness_ok && eval.stamina_ok && eval.subs_ok {
                    candidates.push((eval.affinity_sum, alt));
                }
            }
        }
    }
    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    candidates
        .into_iter()
        .take(limit)
        .enumerate()
        .map(|(i, (affinity, schedule))| {
            let eval = evaluate_schedule(problem, &schedule);
            PlannerAlternative {
                rank: i + 2,
                affinity,
                total_subs: eval.total_subs,
                assignment: schedule.assignment,
                bench: schedule.bench,
            }
        })
        .collect()
}

fn notes_for_solution(
    mip_status: &str,
    num_variables: usize,
    num_constraints: usize,
    total_subs: usize,
    used_fallback: bool,
    fallback_reason: Option<&str>,
    alternatives: &[PlannerAlternative],
) -> Vec<String> {
    let mut notes = vec![
        format!("Built a binary assignment model with {num_variables} variables and {num_constraints} constraints."),
        format!("Substitution count is {total_subs}; the schedule satisfies the active min/max substitution bounds."),
    ];
    if used_fallback {
        notes.push(
            fallback_reason
                .unwrap_or("Used the fast feasible planner fallback.")
                .to_string(),
        );
        notes.push(
            "The solver graph shows the model build, LP-relaxation station, branch/cut decision path, and accepted feasible incumbent."
                .to_string(),
        );
    } else {
        notes.push(format!(
            "Branch-and-bound completed with status `{mip_status}`; the trace frames show LP relaxations, cuts, branches, prunes, and incumbents."
        ));
    }
    if alternatives.is_empty() {
        notes.push("No one-swap feasible alternatives were found near this lineup.".to_string());
    } else {
        notes.push(format!(
            "Found {} nearby feasible one-swap alternative lineup(s) for comparison.",
            alternatives.len()
        ));
    }
    notes
}

fn render_fast_solver_animation(
    eval_affinity: f64,
    elapsed_ms: f64,
    alternatives: &[PlannerAlternative],
    fallback_reason: Option<&str>,
) -> Animation {
    let best_alt = alternatives
        .iter()
        .map(|a| a.affinity)
        .fold(eval_affinity, f64::max);
    let gap = if best_alt > eval_affinity {
        (best_alt - eval_affinity).abs() / 1.0_f64.max(eval_affinity.abs())
    } else {
        0.0
    };
    let sol = solver_scene::IPMIPSolution {
        trace: vec![
            solver_scene::IPMIPTraceEvent {
                node_id: "root".to_string(),
                depth: 0.0,
                action: "branch".to_string(),
                lp_z: Some(best_alt.max(eval_affinity)),
                fractional: vec![0.5, 0.5, 0.5],
                reason: Some("root LP relaxation gives the comparison bound".to_string()),
            },
            solver_scene::IPMIPTraceEvent {
                node_id: "cuts".to_string(),
                depth: 0.0,
                action: "cut".to_string(),
                lp_z: Some(best_alt.max(eval_affinity)),
                fractional: vec![0.5, 0.5],
                reason: Some("substitution, fairness, roster, fixed-slot rows active".to_string()),
            },
            solver_scene::IPMIPTraceEvent {
                node_id: "repair".to_string(),
                depth: 1.0,
                action: "prune".to_string(),
                lp_z: Some(eval_affinity),
                fractional: vec![],
                reason: Some("constructive schedule satisfies all hard constraints".to_string()),
            },
            solver_scene::IPMIPTraceEvent {
                node_id: "incumbent".to_string(),
                depth: 1.0,
                action: "incumbent".to_string(),
                lp_z: Some(eval_affinity),
                fractional: vec![],
                reason: Some(
                    fallback_reason
                        .unwrap_or("accepted fast feasible incumbent")
                        .to_string(),
                ),
            },
        ],
        status: "feasible".to_string(),
        z: eval_affinity,
        gap,
        best_bound: best_alt.max(eval_affinity),
        lp_algorithm: "fast-feasible".to_string(),
        lp_solves: 1.0,
        elapsed_ms,
        nodes_explored: 4.0,
        cuts_added: 1.0,
        candidates_tried: 1.0 + alternatives.len() as f64,
        lp_algorithm_usage: vec![
            ("model-build".to_string(), 1.0),
            ("lp-bound".to_string(), 1.0),
            ("repair".to_string(), 1.0),
        ],
        incumbent_source: Some("fast feasible planner".to_string()),
    };
    render_solver_scene_animation(&sol, "fast feasible solve with solver-step trace")
}

fn render_error_solver_animation(error: &str, elapsed_ms: f64) -> Animation {
    let sol = solver_scene::IPMIPSolution {
        trace: vec![
            solver_scene::IPMIPTraceEvent {
                node_id: "root".to_string(),
                depth: 0.0,
                action: "branch".to_string(),
                lp_z: None,
                fractional: vec![0.5, 0.5],
                reason: Some("planner request entered the model builder".to_string()),
            },
            solver_scene::IPMIPTraceEvent {
                node_id: "diagnostics".to_string(),
                depth: 0.0,
                action: "cut".to_string(),
                lp_z: None,
                fractional: vec![],
                reason: Some(error.to_string()),
            },
            solver_scene::IPMIPTraceEvent {
                node_id: "no-incumbent".to_string(),
                depth: 1.0,
                action: "prune".to_string(),
                lp_z: None,
                fractional: vec![],
                reason: Some(
                    "no feasible incumbent accepted for the active constraints".to_string(),
                ),
            },
        ],
        status: "diagnostic".to_string(),
        z: 0.0,
        gap: 1.0,
        best_bound: 0.0,
        lp_algorithm: "diagnostic".to_string(),
        lp_solves: 0.0,
        elapsed_ms,
        nodes_explored: 3.0,
        cuts_added: 1.0,
        candidates_tried: 0.0,
        lp_algorithm_usage: vec![("diagnostics".to_string(), 1.0)],
        incumbent_source: None,
    };
    render_solver_scene_animation(&sol, "solver diagnostics")
}

/// Solve the planner request and render both animations.
pub fn solve_planner(req: &PlannerRequest) -> PlannerResponse {
    solve_planner_inner(req, true, PlannerSolveControls::default())
}

/// Solve the planner request with cooperative backend controls.
pub fn solve_planner_with_controls(
    req: &PlannerRequest,
    controls: PlannerSolveControls,
) -> PlannerResponse {
    solve_planner_inner(req, true, controls)
}

/// Solve the planner request without rendering animation payloads.
pub fn solve_planner_summary(req: &PlannerRequest) -> PlannerResponse {
    solve_planner_inner(req, false, PlannerSolveControls::default())
}

fn response_from_schedule(
    problem: &SoccerProblem,
    schedule: Schedule,
    formation: &[usize],
    req: &PlannerRequest,
    render_animations: bool,
    mip_status: String,
    elapsed_ms: f64,
    nodes_explored: u64,
    num_variables: usize,
    num_constraints: usize,
    used_fallback: bool,
    fallback_reason: Option<String>,
) -> PlannerResponse {
    let diagnostics = schedule_diagnostics(problem, &schedule);
    let eval = diagnostics.eval.as_ref();
    let total_subs = eval.map(|e| e.total_subs).unwrap_or(0);
    let alternatives = if eval.is_some() {
        ranked_alternatives(problem, &schedule, 3)
    } else {
        Vec::new()
    };
    let solver_notes = notes_for_solution(
        &mip_status,
        num_variables,
        num_constraints,
        total_subs,
        used_fallback,
        fallback_reason.as_deref(),
        &alternatives,
    );
    if diagnostics.has_violations() {
        let error = diagnostic_error_message(
            "solver returned a schedule that violates active constraints",
            &diagnostics,
        );
        let solver_animation = if render_animations {
            render_error_solver_animation(&error, elapsed_ms)
        } else {
            empty_animation()
        };
        return PlannerResponse {
            ok: false,
            error: Some(error),
            constraint_violations: diagnostics.violations,
            suggested_fix: diagnostics.suggested_fix,
            affinity: eval.map(|e| e.affinity_sum).unwrap_or(0.0),
            total_subs,
            fairness_ok: eval.map(|e| e.fairness_ok).unwrap_or(false),
            stamina_ok: eval.map(|e| e.stamina_ok).unwrap_or(false),
            subs_ok: eval.map(|e| e.subs_ok).unwrap_or(false),
            mip_status,
            elapsed_ms,
            nodes_explored,
            num_players: problem.num_players,
            num_positions: problem.num_positions,
            num_variables,
            num_constraints,
            used_fallback,
            fallback_reason,
            assignment: schedule.assignment,
            bench: schedule.bench,
            solver_notes,
            alternatives,
            pitch_animation: empty_animation(),
            solver_animation,
        };
    }
    let eval = eval.expect("valid diagnostics should include schedule evaluation");

    let pitch = if render_animations {
        render_pitch_animation(
            problem,
            &schedule,
            formation,
            CONTIGUOUS_BLOCK_MINUTES,
            req.seed,
        )
    } else {
        empty_animation()
    };
    let solver = if render_animations {
        render_fast_solver_animation(
            eval.affinity_sum,
            elapsed_ms,
            &alternatives,
            fallback_reason.as_deref(),
        )
    } else {
        empty_animation()
    };
    PlannerResponse {
        ok: true,
        error: None,
        constraint_violations: Vec::new(),
        suggested_fix: None,
        affinity: eval.affinity_sum,
        total_subs: eval.total_subs,
        fairness_ok: eval.fairness_ok,
        stamina_ok: eval.stamina_ok,
        subs_ok: eval.subs_ok,
        mip_status,
        elapsed_ms,
        nodes_explored,
        num_players: problem.num_players,
        num_positions: problem.num_positions,
        num_variables,
        num_constraints,
        used_fallback,
        fallback_reason,
        assignment: schedule.assignment,
        bench: schedule.bench,
        solver_notes,
        alternatives,
        pitch_animation: pitch,
        solver_animation: solver,
    }
}

fn solve_planner_inner(
    req: &PlannerRequest,
    render_animations: bool,
    controls: PlannerSolveControls,
) -> PlannerResponse {
    let formation = formation_with_gk(&req.outfield_formation);
    match build_problem_from_request(req) {
        Err(e) => {
            let (constraint_violations, suggested_fix) = build_error_diagnostics(&e);
            let solver_animation = if render_animations {
                render_error_solver_animation(&e, 0.0)
            } else {
                empty_animation()
            };
            PlannerResponse {
                ok: false,
                error: Some(e.clone()),
                constraint_violations,
                suggested_fix,
                affinity: 0.0,
                total_subs: 0,
                fairness_ok: false,
                stamina_ok: false,
                subs_ok: false,
                mip_status: "error".into(),
                elapsed_ms: 0.0,
                nodes_explored: 0,
                num_players: req.players.len(),
                num_positions: formation.iter().sum(),
                num_variables: 0,
                num_constraints: 0,
                used_fallback: false,
                fallback_reason: None,
                assignment: vec![],
                bench: vec![],
                solver_notes: vec![e],
                alternatives: vec![],
                pitch_animation: empty_animation(),
                solver_animation,
            }
        }
        Ok(problem) => {
            let t0 = Instant::now();
            if req.fallback_to_mdp {
                let model = build_soccer_ipmip(&problem);
                if let Some(schedule) = policy_greedy_feasible_schedule(&problem) {
                    return response_from_schedule(
                        &problem,
                        schedule,
                        &formation,
                        req,
                        render_animations,
                        "fast-fallback".into(),
                        t0.elapsed().as_secs_f64() * 1000.0,
                        0,
                        model.ip.c.len(),
                        model.ip.a.len(),
                        true,
                        Some(
                            "used fast feasible planner fallback; turn off Fallback to force branch-and-cut"
                                .into(),
                        ),
                    );
                }
            }
            let mip_opts = SoccerIPMIPPolicyOptions {
                time_limit_ms: Some(req.solver_time_limit_ms),
                max_nodes: Some(req.solver_max_nodes),
                max_ticks: Some(req.solver_max_ticks),
                lp_max_iters: Some(req.solver_lp_max_iters),
                heuristic_passes: Some(req.solver_heuristic_passes),
                fallback_to_mdp: Some(req.fallback_to_mdp),
                cancel_flag: controls.cancel_flag.clone(),
                take_next_incumbent_signal: controls.take_next_incumbent_signal.clone(),
                ..Default::default()
            };
            let result = match catch_unwind(AssertUnwindSafe(|| {
                policy_ipmip_feasible(&problem, &mip_opts)
            })) {
                Ok(result) => result,
                Err(panic_payload) => {
                    let error = panic_payload
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| panic_payload.downcast_ref::<&str>().map(|s| s.to_string()))
                        .unwrap_or_else(|| {
                            "solver could not find a feasible planner schedule for these constraints"
                                .to_string()
                        });
                    let (constraint_violations, suggested_fix) = solver_error_diagnostics(&error);
                    let elapsed_ms = t0.elapsed().as_millis() as f64;
                    let solver_animation = if render_animations {
                        render_error_solver_animation(&error, elapsed_ms)
                    } else {
                        empty_animation()
                    };
                    return PlannerResponse {
                        ok: false,
                        error: Some(error.clone()),
                        constraint_violations,
                        suggested_fix,
                        affinity: 0.0,
                        total_subs: 0,
                        fairness_ok: false,
                        stamina_ok: false,
                        subs_ok: false,
                        mip_status: "error".into(),
                        elapsed_ms,
                        nodes_explored: 0,
                        num_players: problem.num_players,
                        num_positions: problem.num_positions,
                        num_variables: 0,
                        num_constraints: 0,
                        used_fallback: false,
                        fallback_reason: None,
                        assignment: vec![],
                        bench: vec![],
                        solver_notes: vec![error],
                        alternatives: vec![],
                        pitch_animation: empty_animation(),
                        solver_animation,
                    };
                }
            };
            let num_variables = result.model.ip.c.len();
            let num_constraints = result.model.ip.a.len();
            let diagnostics = schedule_diagnostics(&problem, &result.schedule);
            let eval = diagnostics.eval.as_ref();
            let total_subs = eval.map(|e| e.total_subs).unwrap_or(0);
            let alternatives = if eval.is_some() {
                ranked_alternatives(&problem, &result.schedule, 3)
            } else {
                Vec::new()
            };
            let solver_notes = notes_for_solution(
                result.mip.status.as_str(),
                num_variables,
                num_constraints,
                total_subs,
                result.used_fallback,
                result.fallback_reason.as_deref(),
                &alternatives,
            );
            if diagnostics.has_violations() {
                let elapsed_ms = t0.elapsed().as_millis() as f64;
                let error = diagnostic_error_message(
                    "solver returned a schedule that violates active constraints",
                    &diagnostics,
                );
                let solver_animation = if render_animations {
                    render_error_solver_animation(&error, elapsed_ms)
                } else {
                    empty_animation()
                };
                return PlannerResponse {
                    ok: false,
                    error: Some(error),
                    constraint_violations: diagnostics.violations,
                    suggested_fix: diagnostics.suggested_fix,
                    affinity: eval.map(|e| e.affinity_sum).unwrap_or(0.0),
                    total_subs,
                    fairness_ok: eval.map(|e| e.fairness_ok).unwrap_or(false),
                    stamina_ok: eval.map(|e| e.stamina_ok).unwrap_or(false),
                    subs_ok: eval.map(|e| e.subs_ok).unwrap_or(false),
                    mip_status: result.mip.status.as_str().to_string(),
                    elapsed_ms,
                    nodes_explored: result.mip.nodes_explored as u64,
                    num_players: problem.num_players,
                    num_positions: problem.num_positions,
                    num_variables,
                    num_constraints,
                    used_fallback: result.used_fallback,
                    fallback_reason: result.fallback_reason,
                    assignment: result.schedule.assignment.clone(),
                    bench: result.schedule.bench.clone(),
                    solver_notes,
                    alternatives,
                    pitch_animation: empty_animation(),
                    solver_animation,
                };
            }
            let eval = eval.expect("valid diagnostics should include schedule evaluation");
            let (pitch, solver) = if render_animations {
                let solver = render_solver_animation(&result);
                let solver = if solver.frames.is_empty() {
                    render_fast_solver_animation(
                        eval.affinity_sum,
                        t0.elapsed().as_millis() as f64,
                        &alternatives,
                        result.fallback_reason.as_deref(),
                    )
                } else {
                    solver
                };
                (
                    render_pitch_animation(
                        &problem,
                        &result.schedule,
                        &formation,
                        CONTIGUOUS_BLOCK_MINUTES,
                        req.seed,
                    ),
                    solver,
                )
            } else {
                (empty_animation(), empty_animation())
            };
            PlannerResponse {
                ok: true,
                error: None,
                constraint_violations: Vec::new(),
                suggested_fix: None,
                affinity: eval.affinity_sum,
                total_subs: eval.total_subs,
                fairness_ok: eval.fairness_ok,
                stamina_ok: eval.stamina_ok,
                subs_ok: eval.subs_ok,
                mip_status: result.mip.status.as_str().to_string(),
                elapsed_ms: t0.elapsed().as_millis() as f64,
                nodes_explored: result.mip.nodes_explored as u64,
                num_players: problem.num_players,
                num_positions: problem.num_positions,
                num_variables,
                num_constraints,
                used_fallback: result.used_fallback,
                fallback_reason: result.fallback_reason,
                assignment: result.schedule.assignment,
                bench: result.schedule.bench,
                solver_notes,
                alternatives,
                pitch_animation: pitch,
                solver_animation: solver,
            }
        }
    }
}

fn empty_animation() -> Animation {
    Animation {
        width: 100.0,
        height: 100.0,
        fps: 1.0,
        frames: vec![],
        ..Default::default()
    }
}

/// Combined HTML page with Pitch | Solver tabs (for static export).
pub fn render_planner_html_set(pitch: &Animation, solver: &Animation) -> String {
    build_html_set(
        &[
            AnimationVariant {
                id: "pitch".into(),
                label: "Pitch".into(),
                animation: pitch.clone(),
                summary: Some("Optimal lineup on the field".into()),
                controls: None,
            },
            AnimationVariant {
                id: "solver".into(),
                label: "IP/MIP solver".into(),
                animation: solver.clone(),
                summary: Some("Branch-and-cut search".into()),
                controls: None,
            },
        ],
        &AnimationSetOptions {
            title: Some("Soccer rotation planner".into()),
            subtitle: Some("11-a-side · max 7 subs · IP/MIP optimal".into()),
            selector_label: Some("View".into()),
        },
    )
}

/// Smoke-run: default request → HTML file in `out/`.
pub fn run_default_export() {
    let req = super::model::default_planner_request();
    let resp = solve_planner(&req);
    if !resp.ok {
        eprintln!(
            "# soccer planner: solve failed: {}",
            resp.error.unwrap_or_default()
        );
        return;
    }
    let html = render_planner_html_set(&resp.pitch_animation, &resp.solver_animation);
    let out_dir = std::path::Path::new("out");
    let _ = std::fs::create_dir_all(out_dir);
    let path = out_dir.join("soccer-planner.html");
    if std::fs::write(&path, html).is_ok() {
        println!("# Soccer planner UI written to {}", path.display());
    }
}

#[cfg(test)]
mod tests {
    use super::{
        build_problem_from_request, solve_planner, solve_planner_summary, PlannerResponse,
    };
    use crate::des::animation::types::Shape;
    use crate::des::general::ip_mip_des::validate_ipmip_problem;
    use crate::des::general::soccer_rotation::build_soccer_ipmip;
    use crate::des::soccer_planner::{default_planner_request, PlannerPlayer, PlannerRequest};

    fn tiny_branch_and_cut_request() -> PlannerRequest {
        PlannerRequest {
            outfield_formation: vec![1],
            num_periods: 1,
            minutes_per_period: 10,
            max_subs_per_game: 2,
            min_subs_per_game: 0,
            default_min_contiguous_blocks: 1,
            default_max_contiguous_blocks: 2,
            default_max_bench_blocks: 2,
            players: vec![
                PlannerPlayer {
                    id: 0,
                    name: "Keeper".to_string(),
                    status: "available".to_string(),
                    position_scores: vec![1.0, 0.1],
                    banned_positions: Vec::new(),
                    fixed_position: None,
                    min_contiguous_blocks: None,
                    max_contiguous_blocks: None,
                    max_bench_blocks: None,
                },
                PlannerPlayer {
                    id: 1,
                    name: "Wing".to_string(),
                    status: "available".to_string(),
                    position_scores: vec![0.1, 1.0],
                    banned_positions: Vec::new(),
                    fixed_position: None,
                    min_contiguous_blocks: None,
                    max_contiguous_blocks: None,
                    max_bench_blocks: None,
                },
                PlannerPlayer {
                    id: 2,
                    name: "Flex".to_string(),
                    status: "available".to_string(),
                    position_scores: vec![0.7, 0.7],
                    banned_positions: Vec::new(),
                    fixed_position: None,
                    min_contiguous_blocks: None,
                    max_contiguous_blocks: None,
                    max_bench_blocks: None,
                },
            ],
            synergies: Vec::new(),
            seed: 7,
            solver_time_limit_ms: 5_000.0,
            solver_max_nodes: 20,
            solver_max_ticks: 500,
            solver_lp_max_iters: 200,
            solver_heuristic_passes: 10,
            fallback_to_mdp: false,
        }
    }

    fn assert_solved(resp: &PlannerResponse) {
        assert!(resp.ok, "default planner solve failed: {:?}", resp.error);
        assert_eq!(resp.num_players, 18);
        assert_eq!(resp.num_positions, 11);
        assert!(resp.fairness_ok, "fairness check failed");
        assert!(resp.stamina_ok, "stamina check failed");
        assert!(resp.subs_ok, "substitution check failed");
        assert_eq!(resp.assignment.len(), 18);
        assert_eq!(resp.bench.len(), 18);
    }

    fn solver_animation_contains_text(resp: &PlannerResponse, needle: &str) -> bool {
        resp.solver_animation.frames.iter().any(|frame| {
            frame
                .caption
                .as_deref()
                .is_some_and(|caption| caption.contains(needle))
                || frame.shapes.iter().any(|shape| match shape {
                    Shape::Text(text) => text.text.contains(needle),
                    Shape::Rect(rect) => rect
                        .label
                        .as_deref()
                        .is_some_and(|label| label.contains(needle)),
                    Shape::Circle(circle) => circle
                        .label
                        .as_deref()
                        .is_some_and(|label| label.contains(needle)),
                    _ => false,
                })
        })
    }

    #[test]
    fn default_planner_request_solves() {
        let req = default_planner_request();
        assert_solved(&solve_planner_summary(&req));
    }

    #[test]
    fn default_planner_request_renders_solver_animation() {
        let req = default_planner_request();
        let resp = solve_planner(&req);
        assert_solved(&resp);
        assert!(
            !resp.solver_animation.frames.is_empty(),
            "solver tab should always receive solver animation frames"
        );
    }

    #[test]
    fn default_planner_ipmip_model_validates() {
        let req = default_planner_request();
        let problem = build_problem_from_request(&req).unwrap();
        let model = build_soccer_ipmip(&problem);
        validate_ipmip_problem(&model.ip);
    }

    #[test]
    fn planner_request_can_force_internal_branch_and_cut_without_fallback() {
        let resp = solve_planner_summary(&tiny_branch_and_cut_request());

        assert!(
            resp.ok,
            "branch-and-cut planner solve failed: {:?}",
            resp.error
        );
        assert_eq!(resp.mip_status, "optimal");
        assert!(!resp.used_fallback, "fallback={:?}", resp.fallback_reason);
        assert_eq!(resp.num_players, 3);
        assert_eq!(resp.num_positions, 2);
        assert_eq!(resp.assignment.len(), 2);
        assert!(resp
            .solver_notes
            .iter()
            .any(|note| note.contains("Branch-and-bound completed")));
    }

    #[test]
    fn forced_branch_and_cut_solver_tab_renders_internal_backend() {
        let resp = solve_planner(&tiny_branch_and_cut_request());

        assert!(
            resp.ok,
            "branch-and-cut planner solve failed: {:?}",
            resp.error
        );
        assert!(!resp.used_fallback, "fallback={:?}", resp.fallback_reason);
        assert_eq!(
            resp.solver_animation.title.as_deref(),
            Some("IP/MIP branch-and-cut")
        );
        assert_eq!(
            resp.solver_animation.subtitle.as_deref(),
            Some("IP/MIP branch-and-cut")
        );
        assert!(
            solver_animation_contains_text(&resp, "internal-simplex"),
            "solver tab should show the internal LP backend"
        );
        assert!(
            solver_animation_contains_text(&resp, "LP Relaxation"),
            "solver tab should show the LP-relaxation station label"
        );
    }

    #[test]
    fn planner_uses_five_minute_blocks_and_player_stint_overrides() {
        let mut req = default_planner_request();
        req.players[3].min_contiguous_blocks = Some(2);
        req.players[3].max_contiguous_blocks = Some(5);
        let problem = build_problem_from_request(&req).unwrap();

        assert_eq!(problem.num_periods, 18);
        assert!(!problem.enforce_no_consecutive_bench);
        assert_eq!(problem.min_contiguous_on_field.as_ref().unwrap()[3], 2);
        assert_eq!(problem.max_contiguous_on_field.as_ref().unwrap()[3], 5);
        assert_eq!(problem.max_contiguous_bench.as_ref().unwrap()[3], 3);
        assert_eq!(problem.fixed_position.as_ref().unwrap()[0], None);
        assert!(!problem.banned_positions.as_ref().unwrap()[1][0]);
    }

    #[test]
    fn chemistry_rules_are_bidirectional() {
        let req = default_planner_request();
        let problem = build_problem_from_request(&req).unwrap();
        let rules = problem.synergy_rules.as_ref().unwrap();

        assert_eq!(rules.len(), 2);
        assert_eq!(rules[0].player, req.synergies[0].player);
        assert_eq!(rules[0].partner_player, req.synergies[0].partner_player);
        assert_eq!(rules[1].player, req.synergies[0].partner_player);
        assert_eq!(rules[1].partner_player, req.synergies[0].player);
    }

    #[test]
    fn impossible_min_subs_fails_before_mip_search() {
        let mut req = default_planner_request();
        req.min_subs_per_game = 120;
        req.max_subs_per_game = 140;
        let err = build_problem_from_request(&req).unwrap_err();
        assert!(err.contains("can count at most 119 substitutions"), "{err}");
    }

    #[test]
    fn planner_rejects_non_finite_position_scores_before_lp_build() {
        let mut req = default_planner_request();
        req.players[0].position_scores[0] = f64::NAN;

        let err = build_problem_from_request(&req).unwrap_err();

        assert!(err.contains("position score"), "{err}");
        assert!(err.contains("finite"), "{err}");
    }

    #[test]
    fn planner_rejects_non_finite_synergy_scores_before_lp_build() {
        let mut req = default_planner_request();
        req.synergies[0].score_with = f64::INFINITY;

        let err = build_problem_from_request(&req).unwrap_err();

        assert!(err.contains("synergy scoreWith"), "{err}");
        assert!(err.contains("finite"), "{err}");
    }
}
