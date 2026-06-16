//! Port of `src/des/general/soccer-rotation.ts` — module `des::general::soccer_rotation`.
//!
//! 7v7 youth-soccer player rotation as a multi-period bipartite-assignment
//! combinatorial optimisation problem, solved many ways (greedy Hungarian, exact
//! backward-induction MDP, LP relaxation, in-house IP/MIP) plus a stochastic
//! match simulator used as the evaluator.
//!
//! The problem: a configurable roster (default 12; e.g. 11 or 13), 7 on field,
//! the rest on bench, a match split into T time slots. For each (player,
//! position, time slot) there is an affinity in [0, 1].
//! Fairness constraint: no player may be benched two periods in a row.
//! Stamina constraint (optional): no player may stay on the field for more than
//! the configured consecutive time-slot limit. Goal: pick a schedule that
//! maximises total affinity (and simulated goal differential).
//!
//! ## Conversion notes (per the TS "RUST MIGRATION" header)
//!
//!   * Each `PureTransform` subclass → a struct that implements
//!     [`Transform`]; the deprecated free functions are kept and hold the actual
//!     logic (the `Transform::transform` impls delegate to them).
//!   * RNG injection: `mulberry32(seed)` → the project's [`SeededRandom`]
//!     (re-exported through `prng`), constructed from the stored seed inside each
//!     transform so the transform stays deterministic.
//!   * Depends on `hungarian`, `lp`, `ip_mip_des`.
//!   * `benchKey(bench)` string map key → a sorted `Vec<usize>` key with
//!     `Hash + Eq` (the empty bench-set is the `''`/"none" key).
//!   * Schedules are `number[][]`: assignment entries can be a `-1` sentinel
//!     (Hungarian leaving a slot unmatched) so `assignment` is `Vec<Vec<i64>>`;
//!     bench entries are always valid player ids (`Vec<Vec<usize>>`).
//!   * `validate*` returns the recoverable `string | null` as `Option<String>`;
//!     the only hard failures (`throw`) become `panic!`.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};
use std::sync::{
    atomic::{AtomicBool, AtomicU64},
    Arc,
};

use crate::des::general::hungarian::{hungarian, AssignmentDirection};
use crate::des::general::ip_mip_des::{
    solve_ipmip_with_des, BranchRule, ConcreteLpRelaxationAlgorithm, ConstraintNode, IPMIPProblem,
    IPMIPSolution, IPMIPSolveOptions, LpRelaxationAlgorithm, NodeSelection, VariableNode,
};
use crate::des::general::lp::{
    solve_lp, solve_lp_external, ExternalSolverOptions, LPProblem, LPStatus, LpSolverOptions, Sense,
};
use crate::des::general::prng::mulberry32;
use crate::des::shared::capabilities::RandomSource;
use crate::des::shared::transform::Transform;

// =============================================================================
// PROBLEM DEFINITION
// =============================================================================

/// Player availability / roster status for the interactive planner.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PlayerStatus {
    #[default]
    Available,
    /// Not present — cannot be fielded in any period.
    Awol,
    /// Cannot be fielded (stays off the pitch all match).
    Injured,
    /// Temporary roster addition (eligible to play).
    Guest,
}

/// Position-conditional score: player `player` at `position` earns `score_with`
/// when `partner_player` occupies `partner_position` in the same period, else
/// `score_without`. Used by the interactive planner for chemistry rules.
#[derive(Clone, Debug, PartialEq)]
pub struct PositionSynergyRule {
    pub player: usize,
    pub position: usize,
    pub partner_player: usize,
    pub partner_position: usize,
    pub score_with: f64,
    pub score_without: f64,
}

/// A multi-period soccer rotation problem instance.
#[derive(Clone, Debug)]
pub struct SoccerProblem {
    pub num_players: usize,
    pub num_positions: usize,
    pub num_periods: usize,
    pub bench_size: usize,
    /// Max consecutive periods any single player may stay on the field. `None`
    /// disables the stamina constraint (the historical behaviour).
    pub max_consecutive_on_field: Option<usize>,
    /// Optional per-player minimum contiguous on-field run lengths.
    pub min_contiguous_on_field: Option<Vec<usize>>,
    /// Optional per-player maximum contiguous on-field run lengths. When set,
    /// this overrides [`Self::max_consecutive_on_field`] for the listed players.
    pub max_contiguous_on_field: Option<Vec<usize>>,
    /// Optional per-player maximum contiguous bench run lengths.
    pub max_contiguous_bench: Option<Vec<usize>>,
    /// Enforce the legacy rule that a player may not be benched in two
    /// consecutive schedule slots.
    pub enforce_no_consecutive_bench: bool,
    /// Max substitution events (players subbed off) across the whole match.
    pub max_subs_per_game: Option<usize>,
    /// Min substitution events (players subbed off) across the whole match.
    pub min_subs_per_game: Option<usize>,
    /// `affinity[p][pos][t]` in `[0, 1]`.
    pub affinity: Vec<Vec<Vec<f64>>>,
    pub player_names: Option<Vec<String>>,
    pub position_names: Option<Vec<String>>,
    /// Per-player roster status (planner UI). AWOL/Injured → never fielded.
    pub player_status: Option<Vec<PlayerStatus>>,
    /// Lock player `p` to position index `pos` for every period (`None` = free).
    pub fixed_position: Option<Vec<Option<usize>>>,
    /// `banned_positions[p][pos]` → player `p` may never play position `pos`.
    pub banned_positions: Option<Vec<Vec<bool>>>,
    /// Chemistry / conditional position scores (see [`PositionSynergyRule`]).
    pub synergy_rules: Option<Vec<PositionSynergyRule>>,
}

/// A chosen rotation schedule.
#[derive(Clone, Debug)]
pub struct Schedule {
    /// `assignment[t][pos]` = player id at position `pos` in period `t` (`-1` if unset).
    pub assignment: Vec<Vec<i64>>,
    /// `bench[t]` = sorted list of bench player ids in period `t`.
    pub bench: Vec<Vec<usize>>,
}

// -----------------------------------------------------------------------------
// AFFINITY MODEL
// -----------------------------------------------------------------------------

/// Options controlling the synthetic affinity-tensor builder.
#[derive(Clone, Debug, Default)]
pub struct AffinityBuilderOptions {
    pub num_players: Option<usize>,
    pub num_positions: Option<usize>,
    pub num_periods: Option<usize>,
    /// Optional stamina cap: max consecutive periods on field per player.
    /// `None` (the default) disables the constraint.
    pub max_consecutive_on_field: Option<usize>,
    /// PRNG seed for reproducibility.
    pub seed: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StaminaProfile {
    Starter,
    Closer,
    Iron,
}

/// `PureTransform<AffinityBuilderOptions, SoccerProblem>`.
pub struct BuildSampleSoccerProblem;

impl Transform<AffinityBuilderOptions, SoccerProblem> for BuildSampleSoccerProblem {
    fn transform(&self, opts: AffinityBuilderOptions) -> SoccerProblem {
        build_sample_soccer_problem(&opts)
    }
}

/// Build a "realistic" affinity tensor for a configurable squad (default 12).
pub fn build_sample_soccer_problem(opts: &AffinityBuilderOptions) -> SoccerProblem {
    let num_players = opts.num_players.unwrap_or(12);
    let num_positions = opts.num_positions.unwrap_or(7);
    let num_periods = opts.num_periods.unwrap_or(4);
    assert!(
        num_players > num_positions,
        "soccer-rotation: num_players ({num_players}) must exceed num_positions ({num_positions}) so there is at least one bench slot"
    );
    let bench_size = num_players - num_positions;
    let max_consecutive_on_field = opts.max_consecutive_on_field;
    let seed = opts.seed.unwrap_or(4242);
    let mut rng = mulberry32(seed);

    let mut player_names: Vec<String> = Vec::new();
    for p in 0..num_players {
        player_names.push(format!("Player{}", p + 1));
    }
    let mut position_names: Vec<String> = Vec::new();
    let letters: Vec<char> = "ABCDEFGHIJKLMN".chars().collect();
    for pos in 0..num_positions {
        position_names.push(letters[pos].to_string());
    }

    let mut natural_pos: Vec<usize> = Vec::new();
    let mut secondary: Vec<[usize; 2]> = Vec::new();
    let mut profile: Vec<StaminaProfile> = Vec::new();
    let mut fitness: Vec<f64> = Vec::new();
    for _p in 0..num_players {
        natural_pos.push((rng.next_float() * num_positions as f64).floor() as usize);
        let sec1 = (rng.next_float() * num_positions as f64).floor() as usize;
        let sec2 = (rng.next_float() * num_positions as f64).floor() as usize;
        secondary.push([sec1, sec2]);
        let r = rng.next_float();
        profile.push(if r < 0.35 {
            StaminaProfile::Starter
        } else if r < 0.7 {
            StaminaProfile::Closer
        } else {
            StaminaProfile::Iron
        });
        fitness.push(0.7 + 0.3 * rng.next_float());
    }

    let mut aff: Vec<Vec<Vec<f64>>> = Vec::new();
    for p in 0..num_players {
        let mut player_aff: Vec<Vec<f64>> = Vec::new();
        for pos in 0..num_positions {
            let mut period_aff: Vec<f64> = Vec::new();
            let mut base_pos_fit = 0.20 + 0.25 * rng.next_float();
            if pos == natural_pos[p] {
                base_pos_fit = 0.85 + 0.15 * rng.next_float();
            } else if secondary[p].contains(&pos) {
                base_pos_fit = 0.55 + 0.20 * rng.next_float();
            }
            let mid = (num_periods as f64 - 1.0) / 2.0;
            let denom = 1.0_f64.max(num_periods as f64 - 1.0);
            for t in 0..num_periods {
                let period_mod = match profile[p] {
                    StaminaProfile::Starter => 1.05 - 0.20 * (t as f64 / denom),
                    StaminaProfile::Closer => 0.85 + 0.20 * (t as f64 / denom),
                    StaminaProfile::Iron => 0.95 + 0.05 * ((t as f64 - mid) / denom).cos(),
                };
                let noise = 0.95 + 0.10 * rng.next_float();
                let v = base_pos_fit * period_mod * fitness[p] * noise;
                period_aff.push(v.max(0.0).min(1.0));
            }
            player_aff.push(period_aff);
        }
        aff.push(player_aff);
    }

    SoccerProblem {
        num_players,
        num_positions,
        num_periods,
        bench_size,
        max_consecutive_on_field,
        min_contiguous_on_field: None,
        max_contiguous_on_field: None,
        max_contiguous_bench: None,
        enforce_no_consecutive_bench: true,
        max_subs_per_game: None,
        min_subs_per_game: None,
        affinity: aff,
        player_names: Some(player_names),
        position_names: Some(position_names),
        player_status: None,
        fixed_position: None,
        banned_positions: None,
        synergy_rules: None,
    }
}

// =============================================================================
// FORMATIONS
// =============================================================================
//
// A formation is the list of *outfield* lines from defence to attack (e.g.
// `[3, 2, 1]` for "3-2-1"), excluding the keeper. The full on-field shape used
// by the animation/labels is the formation with a 1-slot GK row prepended
// (`[1, 3, 2, 1]`), whose row sizes sum to `num_positions`. These helpers are
// shared by `main_soccer_rotation` (to pick `num_positions` + label positions)
// and the pitch scene (to lay players out in rows).

/// Parse a `"3-2-1"` / `"4-4-2"` style outfield formation (GK excluded). Returns
/// `None` if the string is empty, has a zero/garbage line, or no lines.
pub fn parse_outfield_formation(s: &str) -> Option<Vec<usize>> {
    let rows: Vec<usize> = s
        .split(['-', ',', ' '])
        .filter(|p| !p.trim().is_empty())
        .map(|p| p.trim().parse::<usize>().ok())
        .collect::<Option<Vec<usize>>>()?;
    if rows.is_empty() || rows.contains(&0) {
        return None;
    }
    Some(rows)
}

/// A sensible default outfield formation (GK excluded) for a given on-field size
/// (the *total* number of positions including the keeper). Real youth/adult
/// shapes where they exist, otherwise an even defence/mid/attack split.
pub fn default_outfield_formation(num_positions: usize) -> Vec<usize> {
    match num_positions {
        0 | 1 => vec![],     // GK only (degenerate)
        2 => vec![1],        // GK + 1
        3 => vec![1, 1],     // 3v3
        4 => vec![2, 1],     // 4v4
        5 => vec![2, 2],     // 5v5
        6 => vec![2, 2, 1],  // 6v6
        7 => vec![2, 3, 1],  // 7v7 (classic 2-3-1)
        8 => vec![3, 3, 1],  // 8v8
        9 => vec![3, 4, 1],  // 9v9 (this is where a literal 3-4-1 lives)
        10 => vec![4, 4, 1], // 10-a-side
        11 => vec![4, 4, 2], // 11v11
        n => {
            // Generic: split the outfield (n-1) into D/M/F as evenly as
            // possible, attack getting the remainder-light line.
            let out = n - 1;
            let d = (out + 1) / 3;
            let f = out / 3;
            let m = out - d - f;
            vec![d, m, f].into_iter().filter(|&x| x > 0).collect()
        }
    }
}

/// Prepend the 1-slot GK row to an outfield formation. The result's row sizes
/// sum to `num_positions`.
pub fn formation_with_gk(outfield: &[usize]) -> Vec<usize> {
    let mut rows = Vec::with_capacity(outfield.len() + 1);
    rows.push(1);
    rows.extend_from_slice(outfield);
    rows
}

/// Human-readable position labels (`GK`, `LB`, `CM`, `ST`, …) for a full
/// formation (GK row first). `formation` row sizes must sum to the number of
/// positions; the returned vector has exactly that many labels, in position
/// order (GK, then each line left→right).
pub fn formation_position_names(formation: &[usize]) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    let outfield_rows = formation.len().saturating_sub(1);
    for (r, &n) in formation.iter().enumerate() {
        if r == 0 {
            names.push("GK".to_string());
            continue;
        }
        // Role suffix by line depth: first outfield line = defence (B),
        // last = forward (F), the rest = midfield (M).
        let role = if outfield_rows == 1 {
            'M'
        } else if r == 1 {
            'B'
        } else if r == outfield_rows {
            'F'
        } else {
            'M'
        };
        for c in 0..n {
            names.push(position_label(side_prefix(c, n), role, n));
        }
    }
    names
}

/// Left/centre/right (or numbered) prefix for column `c` of a line of width `n`.
fn side_prefix(c: usize, n: usize) -> &'static str {
    match n {
        1 => "C",
        2 => ["L", "R"][c.min(1)],
        3 => ["L", "C", "R"][c.min(2)],
        4 => ["L", "LC", "RC", "R"][c.min(3)],
        _ => match c {
            0 => "L",
            x if x + 1 == n => "R",
            _ => "C",
        },
    }
}

fn position_label(side: &str, role: char, n: usize) -> String {
    // Lone forward reads "ST"; lone defender/mid keep their role letter.
    if role == 'F' && n == 1 {
        return "ST".to_string();
    }
    format!("{side}{role}")
}

// =============================================================================
// SCHEDULE EVALUATION + FAIRNESS
// =============================================================================

/// A fairness violation: a player benched in two consecutive periods.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FairnessViolation {
    pub player_id: usize,
    pub period_a: usize,
    pub period_b: usize,
}

/// A stamina violation: a player kept on the field for a run of periods longer
/// than [`SoccerProblem::max_consecutive_on_field`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsecutiveOnFieldViolation {
    pub player_id: usize,
    /// First period of the over-long on-field run (0-based).
    pub start_period: usize,
    /// Number of consecutive periods on the field (`> max_consecutive_on_field`).
    pub length: usize,
}

/// A stint-length violation: a player leaves the field before the configured
/// minimum contiguous on-field run length.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MinContiguousOnFieldViolation {
    pub player_id: usize,
    /// First period/time slot of the too-short on-field run (0-based).
    pub start_period: usize,
    /// Number of consecutive periods/time slots on the field.
    pub length: usize,
    /// Configured minimum run length for this player.
    pub min_length: usize,
}

/// A bench-run violation: a player stays on the bench longer than allowed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaxContiguousBenchViolation {
    pub player_id: usize,
    /// First period/time slot of the over-long bench run (0-based).
    pub start_period: usize,
    /// Number of consecutive periods/time slots on the bench.
    pub length: usize,
    /// Configured maximum bench run length for this player.
    pub max_length: usize,
}

/// The deterministic evaluation of a schedule.
#[derive(Clone, Debug)]
pub struct ScheduleEvaluation {
    pub affinity_sum: f64,
    pub per_period_affinity: Vec<f64>,
    pub fairness_ok: bool,
    pub fairness_violations: Vec<FairnessViolation>,
    /// `true` when there is no stamina cap, or no run exceeds it.
    pub stamina_ok: bool,
    pub consecutive_on_field_violations: Vec<ConsecutiveOnFieldViolation>,
    pub min_contiguous_on_field_violations: Vec<MinContiguousOnFieldViolation>,
    pub max_contiguous_bench_violations: Vec<MaxContiguousBenchViolation>,
    /// `true` when there is no substitution bound violation.
    pub subs_ok: bool,
    pub total_subs: usize,
    pub bench_counts: Vec<usize>,
}

/// Bundled `(problem, schedule)` input shared by the schedule analysers.
pub struct ProblemScheduleInput<'a> {
    pub problem: &'a SoccerProblem,
    pub schedule: &'a Schedule,
}

/// `PureTransform<ProblemScheduleInput, ScheduleEvaluation>`.
pub struct EvaluateSchedule;

impl<'a> Transform<ProblemScheduleInput<'a>, ScheduleEvaluation> for EvaluateSchedule {
    fn transform(&self, input: ProblemScheduleInput<'a>) -> ScheduleEvaluation {
        evaluate_schedule(input.problem, input.schedule)
    }
}

/// Whether player `p` may appear on the field (not AWOL / injured).
pub fn player_is_fieldable(problem: &SoccerProblem, p: usize) -> bool {
    !matches!(
        problem.player_status.as_ref().and_then(|s| s.get(p)),
        Some(PlayerStatus::Awol) | Some(PlayerStatus::Injured)
    )
}

fn player_fixed_position(problem: &SoccerProblem, p: usize) -> Option<usize> {
    problem
        .fixed_position
        .as_ref()
        .and_then(|fixed| fixed.get(p))
        .copied()
        .flatten()
}

fn player_banned_from_position(problem: &SoccerProblem, p: usize, pos: usize) -> bool {
    problem
        .banned_positions
        .as_ref()
        .and_then(|banned| banned.get(p))
        .and_then(|row| row.get(pos))
        .copied()
        .unwrap_or(false)
}

fn player_can_play_position(problem: &SoccerProblem, p: usize, pos: usize) -> bool {
    player_is_fieldable(problem, p)
        && player_fixed_position(problem, p)
            .map(|fixed| fixed == pos)
            .unwrap_or(true)
        && !player_banned_from_position(problem, p, pos)
}

fn player_min_contiguous_on_field(problem: &SoccerProblem, p: usize) -> Option<usize> {
    problem
        .min_contiguous_on_field
        .as_ref()
        .and_then(|limits| limits.get(p))
        .copied()
        .filter(|&limit| limit > 1)
}

fn player_max_contiguous_on_field(problem: &SoccerProblem, p: usize) -> Option<usize> {
    problem
        .max_contiguous_on_field
        .as_ref()
        .and_then(|limits| limits.get(p))
        .copied()
        .or(problem.max_consecutive_on_field)
        .filter(|&limit| limit >= 1)
}

fn player_max_contiguous_bench(problem: &SoccerProblem, p: usize) -> Option<usize> {
    problem
        .max_contiguous_bench
        .as_ref()
        .and_then(|limits| limits.get(p))
        .copied()
        .filter(|&limit| limit >= 1)
}

/// Effective score for `(player, position)` in period `t` given the full lineup
/// (applies synergy overrides on top of the base affinity tensor).
pub fn effective_cell_affinity(
    problem: &SoccerProblem,
    assignment: &[i64],
    player: usize,
    position: usize,
    period: usize,
) -> f64 {
    let mut aff = problem.affinity[player][position][period];
    if let Some(rules) = &problem.synergy_rules {
        for rule in rules {
            if rule.player == player && rule.position == position {
                let partner_on = assignment.get(rule.partner_position).copied().unwrap_or(-1)
                    == rule.partner_player as i64;
                aff = if partner_on {
                    rule.score_with
                } else {
                    rule.score_without
                };
            }
        }
    }
    aff
}

/// Count substitution events: players who leave the field at a period boundary.
pub fn count_subs(problem: &SoccerProblem, schedule: &Schedule) -> usize {
    let t_count = problem.num_periods;
    if t_count <= 1 {
        return 0;
    }
    let mut total = 0usize;
    for t in 1..t_count {
        let prev: HashSet<i64> = schedule.assignment[t - 1].iter().copied().collect();
        for &p in &schedule.assignment[t] {
            if !prev.contains(&p) {
                // player came on — count the corresponding off elsewhere
                continue;
            }
        }
        for &p in &schedule.assignment[t - 1] {
            if !schedule.assignment[t].contains(&p) {
                total += 1;
            }
        }
    }
    total
}

/// Score a schedule and check the fairness constraint.
pub fn evaluate_schedule(problem: &SoccerProblem, schedule: &Schedule) -> ScheduleEvaluation {
    let t_count = problem.num_periods;
    let mut per_period: Vec<f64> = Vec::new();
    let mut total = 0.0;
    for t in 0..t_count {
        let mut s = 0.0;
        for pos in 0..problem.num_positions {
            let p = schedule.assignment[t][pos] as usize;
            s += effective_cell_affinity(problem, &schedule.assignment[t], p, pos, t);
        }
        per_period.push(s);
        total += s;
    }
    let mut fairness_violations: Vec<FairnessViolation> = Vec::new();
    if problem.enforce_no_consecutive_bench {
        for t in 0..t_count.saturating_sub(1) {
            for &p in &schedule.bench[t] {
                if !player_is_fieldable(problem, p) {
                    continue;
                }
                if schedule.bench[t + 1].contains(&p) {
                    fairness_violations.push(FairnessViolation {
                        player_id: p,
                        period_a: t,
                        period_b: t + 1,
                    });
                }
            }
        }
    }
    let mut bench_counts = vec![0usize; problem.num_players];
    for t in 0..t_count {
        for &p in &schedule.bench[t] {
            bench_counts[p] += 1;
        }
    }
    let consecutive_on_field_violations = consecutive_on_field_violations(problem, schedule);
    let min_contiguous_on_field_violations = min_contiguous_on_field_violations(problem, schedule);
    let max_contiguous_bench_violations = max_contiguous_bench_violations(problem, schedule);
    let total_subs = count_subs(problem, schedule);
    let subs_ok = problem
        .min_subs_per_game
        .map(|floor| total_subs >= floor)
        .unwrap_or(true)
        && problem
            .max_subs_per_game
            .map(|cap| total_subs <= cap)
            .unwrap_or(true);
    ScheduleEvaluation {
        affinity_sum: total,
        per_period_affinity: per_period,
        fairness_ok: fairness_violations.is_empty() && max_contiguous_bench_violations.is_empty(),
        fairness_violations,
        stamina_ok: consecutive_on_field_violations.is_empty()
            && min_contiguous_on_field_violations.is_empty(),
        consecutive_on_field_violations,
        min_contiguous_on_field_violations,
        max_contiguous_bench_violations,
        subs_ok,
        total_subs,
        bench_counts,
    }
}

/// Find every run where a player stays on the field for more than
/// [`SoccerProblem::max_consecutive_on_field`] periods. Empty when the problem
/// has no stamina cap.
pub fn consecutive_on_field_violations(
    problem: &SoccerProblem,
    schedule: &Schedule,
) -> Vec<ConsecutiveOnFieldViolation> {
    let t_count = problem.num_periods;
    let mut on_field = vec![vec![false; problem.num_players]; t_count];
    for t in 0..t_count {
        for &p in &schedule.assignment[t] {
            if p >= 0 && (p as usize) < problem.num_players {
                on_field[t][p as usize] = true;
            }
        }
    }
    let mut violations: Vec<ConsecutiveOnFieldViolation> = Vec::new();
    for p in 0..problem.num_players {
        let Some(limit) = player_max_contiguous_on_field(problem, p) else {
            continue;
        };
        let mut run = 0usize;
        let mut run_start = 0usize;
        for t in 0..t_count {
            if on_field[t][p] {
                if run == 0 {
                    run_start = t;
                }
                run += 1;
            } else {
                if run > limit {
                    violations.push(ConsecutiveOnFieldViolation {
                        player_id: p,
                        start_period: run_start,
                        length: run,
                    });
                }
                run = 0;
            }
        }
        if run > limit {
            violations.push(ConsecutiveOnFieldViolation {
                player_id: p,
                start_period: run_start,
                length: run,
            });
        }
    }
    violations
}

/// Find every run where a player stays on the field for fewer than their
/// configured minimum contiguous slots. Empty when all minimums are one slot.
pub fn min_contiguous_on_field_violations(
    problem: &SoccerProblem,
    schedule: &Schedule,
) -> Vec<MinContiguousOnFieldViolation> {
    let t_count = problem.num_periods;
    let mut on_field = vec![vec![false; problem.num_players]; t_count];
    for t in 0..t_count {
        for &p in &schedule.assignment[t] {
            if p >= 0 && (p as usize) < problem.num_players {
                on_field[t][p as usize] = true;
            }
        }
    }
    let mut violations: Vec<MinContiguousOnFieldViolation> = Vec::new();
    for p in 0..problem.num_players {
        let Some(limit) = player_min_contiguous_on_field(problem, p) else {
            continue;
        };
        let mut run = 0usize;
        let mut run_start = 0usize;
        for t in 0..t_count {
            if on_field[t][p] {
                if run == 0 {
                    run_start = t;
                }
                run += 1;
            } else if run > 0 {
                if run < limit {
                    violations.push(MinContiguousOnFieldViolation {
                        player_id: p,
                        start_period: run_start,
                        length: run,
                        min_length: limit,
                    });
                }
                run = 0;
            }
        }
        if run > 0 && run < limit {
            violations.push(MinContiguousOnFieldViolation {
                player_id: p,
                start_period: run_start,
                length: run,
                min_length: limit,
            });
        }
    }
    violations
}

/// Find every run where a player stays on the bench for more than their
/// configured maximum contiguous bench slots. Empty when no cap is configured.
pub fn max_contiguous_bench_violations(
    problem: &SoccerProblem,
    schedule: &Schedule,
) -> Vec<MaxContiguousBenchViolation> {
    let t_count = problem.num_periods;
    let mut on_field = vec![vec![false; problem.num_players]; t_count];
    for t in 0..t_count {
        for &p in &schedule.assignment[t] {
            if p >= 0 && (p as usize) < problem.num_players {
                on_field[t][p as usize] = true;
            }
        }
    }
    let mut violations: Vec<MaxContiguousBenchViolation> = Vec::new();
    for p in 0..problem.num_players {
        let Some(limit) = player_max_contiguous_bench(problem, p) else {
            continue;
        };
        let mut run = 0usize;
        let mut run_start = 0usize;
        for t in 0..t_count {
            let is_benched = player_is_fieldable(problem, p) && !on_field[t][p];
            if is_benched {
                if run == 0 {
                    run_start = t;
                }
                run += 1;
            } else {
                if run > limit {
                    violations.push(MaxContiguousBenchViolation {
                        player_id: p,
                        start_period: run_start,
                        length: run,
                        max_length: limit,
                    });
                }
                run = 0;
            }
        }
        if run > limit {
            violations.push(MaxContiguousBenchViolation {
                player_id: p,
                start_period: run_start,
                length: run,
                max_length: limit,
            });
        }
    }
    violations
}

/// `PureTransform<ProblemScheduleInput, string | null>`.
pub struct ValidateScheduleStructure;

impl<'a> Transform<ProblemScheduleInput<'a>, Option<String>> for ValidateScheduleStructure {
    fn transform(&self, input: ProblemScheduleInput<'a>) -> Option<String> {
        validate_schedule_structure(input.problem, input.schedule)
    }
}

/// Sanity-check that a schedule is well-formed; returns `Some(reason)` if not.
pub fn validate_schedule_structure(problem: &SoccerProblem, schedule: &Schedule) -> Option<String> {
    let (num_players, num_positions, num_periods, bench_size) = (
        problem.num_players,
        problem.num_positions,
        problem.num_periods,
        problem.bench_size,
    );
    if schedule.assignment.len() != num_periods {
        return Some(format!("assignment.length != {num_periods}"));
    }
    if schedule.bench.len() != num_periods {
        return Some(format!("bench.length != {num_periods}"));
    }
    for t in 0..num_periods {
        if schedule.assignment[t].len() != num_positions {
            return Some(format!("assignment[{t}].length != {num_positions}"));
        }
        if schedule.bench[t].len() != bench_size {
            return Some(format!("bench[{t}].length != {bench_size}"));
        }
        let mut seen: HashSet<i64> = HashSet::new();
        for &p in &schedule.assignment[t] {
            if p < 0 || p >= num_players as i64 {
                return Some(format!("assignment[{t}] has invalid player {p}"));
            }
            if seen.contains(&p) {
                return Some(format!("player {p} appears twice on field in period {t}"));
            }
            seen.insert(p);
        }
        for &p in &schedule.bench[t] {
            if p >= num_players {
                return Some(format!("bench[{t}] has invalid player {p}"));
            }
            let pi = p as i64;
            if seen.contains(&pi) {
                return Some(format!("player {p} both on field and bench in period {t}"));
            }
            seen.insert(pi);
        }
        if seen.len() != num_players {
            return Some(format!("period {t}: total players != {num_players}"));
        }
        // Planner constraints.
        for pos in 0..num_positions {
            let p = schedule.assignment[t][pos] as usize;
            if !player_is_fieldable(problem, p) {
                return Some(format!("player {p} (AWOL/injured) on field in period {t}"));
            }
            if let Some(fixed) = problem.fixed_position.as_ref().and_then(|f| f.get(p)) {
                if let Some(fp) = fixed {
                    if *fp != pos {
                        return Some(format!(
                            "player {p} fixed at position {fp} but assigned {pos} in period {t}"
                        ));
                    }
                }
            }
            if problem
                .banned_positions
                .as_ref()
                .and_then(|b| b.get(p))
                .and_then(|row| row.get(pos))
                .copied()
                .unwrap_or(false)
            {
                return Some(format!(
                    "player {p} banned from position {pos} but assigned in period {t}"
                ));
            }
        }
    }
    None
}

// =============================================================================
// SCHEDULING POLICIES
// =============================================================================

/// `PureTransform<SoccerProblem, Schedule>` — random (baseline) schedule.
pub struct PolicyRandomSchedule {
    seed: u32,
}

impl PolicyRandomSchedule {
    pub fn new(seed: u32) -> Self {
        PolicyRandomSchedule { seed }
    }
}

impl<'a> Transform<&'a SoccerProblem, Schedule> for PolicyRandomSchedule {
    fn transform(&self, problem: &'a SoccerProblem) -> Schedule {
        policy_random_schedule(problem, self.seed)
    }
}

/// Build a random valid schedule (does NOT respect fairness with high probability).
pub fn policy_random_schedule(problem: &SoccerProblem, seed: u32) -> Schedule {
    let mut rng = mulberry32(seed);
    let mut assignment: Vec<Vec<i64>> = Vec::new();
    let mut bench: Vec<Vec<usize>> = Vec::new();
    for _t in 0..problem.num_periods {
        let mut order: Vec<usize> = (0..problem.num_players).collect();
        let mut i = order.len() - 1;
        while i > 0 {
            let j = (rng.next_float() * ((i + 1) as f64)).floor() as usize;
            order.swap(i, j);
            i -= 1;
        }
        assignment.push(
            order[0..problem.num_positions]
                .iter()
                .map(|&x| x as i64)
                .collect(),
        );
        let mut tbench: Vec<usize> = order[problem.num_positions..].to_vec();
        tbench.sort_unstable();
        bench.push(tbench);
    }
    Schedule { assignment, bench }
}

/// Options for the greedy-Hungarian policy.
#[derive(Clone, Debug, Default)]
pub struct GreedyHungarianOptions {
    pub fairness_aware: Option<bool>,
}

/// `PureTransform<SoccerProblem, Schedule>` — greedy per-period Hungarian.
pub struct PolicyGreedyHungarian {
    opts: GreedyHungarianOptions,
}

impl PolicyGreedyHungarian {
    pub fn new(opts: GreedyHungarianOptions) -> Self {
        PolicyGreedyHungarian { opts }
    }
}

impl<'a> Transform<&'a SoccerProblem, Schedule> for PolicyGreedyHungarian {
    fn transform(&self, problem: &'a SoccerProblem) -> Schedule {
        policy_greedy_hungarian(problem, &self.opts)
    }
}

/// Greedy fairness-aware per-period Hungarian assignment.
pub fn policy_greedy_hungarian(problem: &SoccerProblem, opts: &GreedyHungarianOptions) -> Schedule {
    let fairness_aware =
        opts.fairness_aware.unwrap_or(true) && problem.enforce_no_consecutive_bench;
    let t_count = problem.num_periods;
    let mut assignment: Vec<Vec<i64>> = Vec::new();
    let mut bench: Vec<Vec<usize>> = Vec::new();
    let mut prev_bench: Vec<usize> = Vec::new();
    for t in 0..t_count {
        let mut best_pos: Vec<f64> = Vec::new();
        for p in 0..problem.num_players {
            let mut mx = f64::NEG_INFINITY;
            for pos in 0..problem.num_positions {
                if problem.affinity[p][pos][t] > mx {
                    mx = problem.affinity[p][pos][t];
                }
            }
            best_pos.push(mx);
        }
        let mut candidates: Vec<usize> = (0..problem.num_players).collect();
        candidates.sort_by(|&a, &b| {
            best_pos[b]
                .partial_cmp(&best_pos[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let must_play: HashSet<usize> = if fairness_aware {
            prev_bench.iter().copied().collect()
        } else {
            HashSet::new()
        };
        let mut on_field: Vec<usize> = Vec::new();
        for &p in &candidates {
            if must_play.contains(&p) {
                on_field.push(p);
            }
            if on_field.len() >= problem.num_positions {
                break;
            }
        }
        for &p in &candidates {
            if on_field.len() >= problem.num_positions {
                break;
            }
            if !must_play.contains(&p) && !on_field.contains(&p) {
                on_field.push(p);
            }
        }
        let period_assign = hungarian_assign_period(problem, t, &on_field);
        assignment.push(period_assign);
        let mut tbench: Vec<usize> = (0..problem.num_players)
            .filter(|p| !on_field.contains(p))
            .collect();
        tbench.sort_unstable();
        bench.push(tbench.clone());
        prev_bench = tbench;
    }
    Schedule { assignment, bench }
}

fn best_eligible_position_score(problem: &SoccerProblem, p: usize, t: usize) -> Option<f64> {
    let mut best = f64::NEG_INFINITY;
    let mut found = false;
    for pos in 0..problem.num_positions {
        if player_can_play_position(problem, p, pos) {
            best = best.max(problem.affinity[p][pos][t]);
            found = true;
        }
    }
    found.then_some(best)
}

fn hungarian_assign_period_constrained(
    problem: &SoccerProblem,
    t: usize,
    on_field: &[usize],
) -> Option<Vec<i64>> {
    if on_field.len() != problem.num_positions {
        return None;
    }
    let invalid = -1.0e9;
    let mut w_matrix: Vec<Vec<f64>> = Vec::new();
    for &p in on_field {
        let mut row: Vec<f64> = Vec::new();
        for pos in 0..problem.num_positions {
            row.push(if player_can_play_position(problem, p, pos) {
                problem.affinity[p][pos][t]
            } else {
                invalid
            });
        }
        if row.iter().all(|&w| w <= invalid / 2.0) {
            return None;
        }
        w_matrix.push(row);
    }
    for pos in 0..problem.num_positions {
        if w_matrix.iter().all(|row| row[pos] <= invalid / 2.0) {
            return None;
        }
    }
    let res = hungarian(&w_matrix, AssignmentDirection::Max);
    let mut period_assign = vec![-1i64; problem.num_positions];
    for (i, &p) in on_field.iter().enumerate() {
        let pos = *res.rows.get(i)?;
        if pos < 0 {
            return None;
        }
        let pos = pos as usize;
        if pos >= problem.num_positions || w_matrix[i][pos] <= invalid / 2.0 {
            return None;
        }
        if period_assign[pos] >= 0 {
            return None;
        }
        period_assign[pos] = p as i64;
    }
    if period_assign.iter().any(|&p| p < 0) {
        return None;
    }
    Some(period_assign)
}

/// Fast constructive schedule for the interactive planner fallback.
///
/// It respects fieldability, fixed-position and banned-position constraints,
/// the no-consecutive-bench fairness rule, the stamina cap, and max subs.
pub fn policy_greedy_feasible_schedule(problem: &SoccerProblem) -> Option<Schedule> {
    let fieldable = (0..problem.num_players)
        .filter(|&p| player_is_fieldable(problem, p))
        .count();
    if fieldable < problem.num_positions {
        return None;
    }

    let mut assignment: Vec<Vec<i64>> = Vec::new();
    let mut bench: Vec<Vec<usize>> = Vec::new();
    let mut prev_bench: HashSet<usize> = HashSet::new();
    let mut prev_on: HashSet<usize> = HashSet::new();
    let mut consecutive_on = vec![0usize; problem.num_players];
    let mut consecutive_bench = vec![0usize; problem.num_players];

    for t in 0..problem.num_periods {
        let mut selected: Vec<usize> = Vec::new();
        let mut selected_set: HashSet<usize> = HashSet::new();

        for p in 0..problem.num_players {
            if let Some(pos) = player_fixed_position(problem, p) {
                if pos >= problem.num_positions || !player_can_play_position(problem, p, pos) {
                    return None;
                }
                selected.push(p);
                selected_set.insert(p);
            }
        }
        if selected.len() > problem.num_positions {
            return None;
        }

        let mut must_continue: Vec<usize> = prev_on
            .iter()
            .copied()
            .filter(|&p| {
                !selected_set.contains(&p)
                    && player_is_fieldable(problem, p)
                    && player_min_contiguous_on_field(problem, p)
                        .map(|limit| consecutive_on[p] > 0 && consecutive_on[p] < limit)
                        .unwrap_or(false)
            })
            .collect();
        must_continue.sort_by(|&a, &b| {
            best_eligible_position_score(problem, b, t)
                .unwrap_or(f64::NEG_INFINITY)
                .partial_cmp(
                    &best_eligible_position_score(problem, a, t).unwrap_or(f64::NEG_INFINITY),
                )
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        for p in must_continue {
            if selected.len() >= problem.num_positions {
                return None;
            }
            if best_eligible_position_score(problem, p, t).is_some() {
                selected.push(p);
                selected_set.insert(p);
            } else {
                return None;
            }
        }

        let mut must_play: Vec<usize> = prev_bench
            .iter()
            .copied()
            .filter(|&p| {
                !selected_set.contains(&p)
                    && player_is_fieldable(problem, p)
                    && player_max_contiguous_bench(problem, p)
                        .map(|limit| consecutive_bench[p] >= limit)
                        .unwrap_or(false)
            })
            .collect();
        if problem.enforce_no_consecutive_bench {
            must_play.extend(
                prev_bench
                    .iter()
                    .copied()
                    .filter(|&p| player_is_fieldable(problem, p) && !selected_set.contains(&p)),
            );
            must_play.sort_unstable();
            must_play.dedup();
        }
        must_play.sort_by(|&a, &b| {
            best_eligible_position_score(problem, b, t)
                .unwrap_or(f64::NEG_INFINITY)
                .partial_cmp(
                    &best_eligible_position_score(problem, a, t).unwrap_or(f64::NEG_INFINITY),
                )
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        for p in must_play {
            if selected.len() >= problem.num_positions {
                return None;
            }
            if best_eligible_position_score(problem, p, t).is_some() {
                selected.push(p);
                selected_set.insert(p);
            } else {
                return None;
            }
        }

        let mut candidates: Vec<(usize, f64)> = Vec::new();
        for p in 0..problem.num_players {
            if selected_set.contains(&p) {
                continue;
            }
            let Some(best) = best_eligible_position_score(problem, p, t) else {
                continue;
            };
            if player_max_contiguous_on_field(problem, p)
                .map(|limit| consecutive_on[p] >= limit)
                .unwrap_or(false)
            {
                continue;
            }
            let continuity = if prev_on.contains(&p) { 0.05 } else { 0.0 };
            let fatigue_penalty = 0.25 * consecutive_on[p] as f64;
            candidates.push((p, best + continuity - fatigue_penalty));
        }
        candidates.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        for (p, _) in candidates {
            if selected.len() >= problem.num_positions {
                break;
            }
            selected.push(p);
            selected_set.insert(p);
        }
        if selected.len() != problem.num_positions {
            return None;
        }

        let row = hungarian_assign_period_constrained(problem, t, &selected)?;
        let used: HashSet<usize> = row.iter().map(|&p| p as usize).collect();
        let mut tbench: Vec<usize> = (0..problem.num_players)
            .filter(|p| !used.contains(p))
            .collect();
        tbench.sort_unstable();

        for p in 0..problem.num_players {
            if used.contains(&p) {
                consecutive_on[p] += 1;
                consecutive_bench[p] = 0;
            } else {
                consecutive_on[p] = 0;
                if player_is_fieldable(problem, p) {
                    consecutive_bench[p] += 1;
                } else {
                    consecutive_bench[p] = 0;
                }
            }
        }
        prev_on = used;
        prev_bench = tbench.iter().copied().collect();
        assignment.push(row);
        bench.push(tbench);
    }

    let schedule = Schedule { assignment, bench };
    if validate_schedule_structure(problem, &schedule).is_some() {
        return None;
    }
    let eval = evaluate_schedule(problem, &schedule);
    if eval.fairness_ok && eval.stamina_ok && eval.subs_ok {
        Some(schedule)
    } else {
        None
    }
}

/// Hungarian-assign the chosen on-field players to positions for period `t`.
fn hungarian_assign_period(problem: &SoccerProblem, t: usize, on_field: &[usize]) -> Vec<i64> {
    let mut w_matrix: Vec<Vec<f64>> = Vec::new();
    for &p in on_field {
        let mut row: Vec<f64> = Vec::new();
        for pos in 0..problem.num_positions {
            row.push(problem.affinity[p][pos][t]);
        }
        w_matrix.push(row);
    }
    let res = hungarian(&w_matrix, AssignmentDirection::Max);
    let mut period_assign = vec![-1i64; problem.num_positions];
    for i in 0..on_field.len() {
        let pos = res.rows[i];
        if pos >= 0 {
            period_assign[pos as usize] = on_field[i] as i64;
        }
    }
    period_assign
}

// -----------------------------------------------------------------------------
// MDP-VI exact backward induction.
// -----------------------------------------------------------------------------

/// Enumerate all C(n, k) sorted subsets of size k from {0, …, n-1}.
fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut out: Vec<Vec<usize>> = Vec::new();
    if k == 0 {
        out.push(Vec::new());
        return out;
    }
    if k > n {
        return out;
    }
    let mut idx: Vec<usize> = (0..k).collect();
    loop {
        out.push(idx.clone());
        let mut i: isize = k as isize - 1;
        while i >= 0 && idx[i as usize] == n - k + i as usize {
            i -= 1;
        }
        if i < 0 {
            return out;
        }
        idx[i as usize] += 1;
        for j in (i as usize + 1)..k {
            idx[j] = idx[j - 1] + 1;
        }
    }
}

struct PeriodReward {
    reward: f64,
    assignment: Vec<i64>,
    on_field: Vec<usize>,
}

/// Period-t reward for a chosen bench-set: the Hungarian-optimal affinity of the
/// on-field players plus the position assignment.
fn period_reward_and_assignment(
    problem: &SoccerProblem,
    t: usize,
    bench_set: &[usize],
) -> PeriodReward {
    let bench_set_map: HashSet<usize> = bench_set.iter().copied().collect();
    let mut on_field: Vec<usize> = Vec::new();
    for p in 0..problem.num_players {
        if !bench_set_map.contains(&p) {
            on_field.push(p);
        }
    }
    let mut w_matrix: Vec<Vec<f64>> = Vec::new();
    for &p in &on_field {
        let mut row: Vec<f64> = Vec::new();
        for pos in 0..problem.num_positions {
            row.push(problem.affinity[p][pos][t]);
        }
        w_matrix.push(row);
    }
    let res = hungarian(&w_matrix, AssignmentDirection::Max);
    let mut period_assign = vec![-1i64; problem.num_positions];
    for i in 0..on_field.len() {
        let pos = res.rows[i];
        if pos >= 0 {
            period_assign[pos as usize] = on_field[i] as i64;
        }
    }
    PeriodReward {
        reward: res.total,
        assignment: period_assign,
        on_field,
    }
}

/// Result of the memoryless MDP — value IGNORES fairness.
#[derive(Clone, Debug)]
pub struct MemorylessMDPResult {
    pub schedule: Schedule,
    pub value: f64,
}

/// `PureTransform<SoccerProblem, MemorylessMDPResult>`.
pub struct PolicyMDPVIMemoryless;

impl<'a> Transform<&'a SoccerProblem, MemorylessMDPResult> for PolicyMDPVIMemoryless {
    fn transform(&self, problem: &'a SoccerProblem) -> MemorylessMDPResult {
        policy_mdp_vi_memoryless(problem)
    }
}

/// Solve each period independently — no `prev_bench` state, so fairness cannot
/// be expressed (the pedagogical counter-example).
pub fn policy_mdp_vi_memoryless(problem: &SoccerProblem) -> MemorylessMDPResult {
    let t_count = problem.num_periods;
    let k = problem.bench_size;
    let all_benches = combinations(problem.num_players, k);
    let mut assignment: Vec<Vec<i64>> = Vec::new();
    let mut bench: Vec<Vec<usize>> = Vec::new();
    let mut total = 0.0;
    for t in 0..t_count {
        let mut best_val = f64::NEG_INFINITY;
        let mut best_b: Vec<usize> = Vec::new();
        let mut best_assign: Vec<i64> = Vec::new();
        for b in &all_benches {
            let r = period_reward_and_assignment(problem, t, b);
            if r.reward > best_val {
                best_val = r.reward;
                best_b = b.clone();
                best_assign = r.assignment;
            }
        }
        assignment.push(best_assign);
        bench.push(best_b);
        total += best_val;
    }
    MemorylessMDPResult {
        schedule: Schedule { assignment, bench },
        value: total,
    }
}

/// The `Schedule & {optimalValue}` returned by the exact MDP.
#[derive(Clone, Debug)]
pub struct MdpScheduleResult {
    pub assignment: Vec<Vec<i64>>,
    pub bench: Vec<Vec<usize>>,
    pub optimal_value: f64,
}

impl MdpScheduleResult {
    pub fn to_schedule(&self) -> Schedule {
        Schedule {
            assignment: self.assignment.clone(),
            bench: self.bench.clone(),
        }
    }
}

#[derive(Clone)]
struct MdpChoice {
    bench: Vec<usize>,
    assignment: Vec<i64>,
}

/// `PureTransform<SoccerProblem, Schedule & {optimalValue}>`.
pub struct PolicyMDPVI;

impl<'a> Transform<&'a SoccerProblem, MdpScheduleResult> for PolicyMDPVI {
    fn transform(&self, problem: &'a SoccerProblem) -> MdpScheduleResult {
        policy_mdp_vi(problem)
    }
}

/// Solve the multi-period rotation MDP exactly via backward induction.
pub fn policy_mdp_vi(problem: &SoccerProblem) -> MdpScheduleResult {
    let t_count = problem.num_periods;
    let k = problem.bench_size;
    let all_benches = combinations(problem.num_players, k);
    let mut value_by_t: Vec<HashMap<Vec<usize>, f64>> = Vec::new();
    let mut choice_by_t: Vec<HashMap<Vec<usize>, MdpChoice>> = Vec::new();
    for _t in 0..=t_count {
        value_by_t.push(HashMap::new());
        choice_by_t.push(HashMap::new());
    }
    // Terminal: any prev bench (and the "none" key) has value 0.
    for b in &all_benches {
        value_by_t[t_count].insert(b.clone(), 0.0);
    }
    value_by_t[t_count].insert(Vec::new(), 0.0);
    // Backward induction.
    for t in (0..t_count).rev() {
        let prev_benches: Vec<Vec<usize>> = if t == 0 {
            vec![Vec::new()]
        } else {
            all_benches.clone()
        };
        for prev in &prev_benches {
            let prev_set: HashSet<usize> = prev.iter().copied().collect();
            let mut best_val = f64::NEG_INFINITY;
            let mut best_b: Vec<usize> = Vec::new();
            let mut best_assign: Vec<i64> = Vec::new();
            for b in &all_benches {
                let ok = b.iter().all(|x| !prev_set.contains(x));
                if !ok {
                    continue;
                }
                let r = period_reward_and_assignment(problem, t, b);
                let fut = value_by_t[t + 1].get(b).copied().unwrap_or(0.0);
                let val = r.reward + fut;
                if val > best_val {
                    best_val = val;
                    best_b = b.clone();
                    best_assign = r.assignment;
                }
            }
            value_by_t[t].insert(prev.clone(), best_val);
            choice_by_t[t].insert(
                prev.clone(),
                MdpChoice {
                    bench: best_b,
                    assignment: best_assign,
                },
            );
        }
    }
    // Reconstruct policy.
    let mut assignment: Vec<Vec<i64>> = Vec::new();
    let mut bench: Vec<Vec<usize>> = Vec::new();
    let mut prev_key: Vec<usize> = Vec::new();
    for t in 0..t_count {
        let choice = choice_by_t[t]
            .get(&prev_key)
            .expect("MDP choice missing for reachable state")
            .clone();
        assignment.push(choice.assignment);
        bench.push(choice.bench.clone());
        prev_key = choice.bench;
    }
    let optimal_value = *value_by_t[0]
        .get(&Vec::new())
        .expect("MDP root value missing");
    MdpScheduleResult {
        assignment,
        bench,
        optimal_value,
    }
}

// -----------------------------------------------------------------------------
// LP RELAXATION
// -----------------------------------------------------------------------------

/// `PureTransform<SoccerProblem, LPProblem>`.
pub struct BuildSoccerLP;

impl<'a> Transform<&'a SoccerProblem, LPProblem> for BuildSoccerLP {
    fn transform(&self, problem: &'a SoccerProblem) -> LPProblem {
        build_soccer_lp(problem)
    }
}

/// Build the multi-period LP relaxation of the 0/1 rotation program.
pub fn build_soccer_lp(problem: &SoccerProblem) -> LPProblem {
    let (p_count, k, t_count) = (
        problem.num_players,
        problem.num_positions,
        problem.num_periods,
    );
    let n = p_count * k * t_count;
    let idx = |p: usize, pos: usize, t: usize| (p * k + pos) * t_count + t;
    let mut c = vec![0.0; n];
    for p in 0..p_count {
        for pos in 0..k {
            for t in 0..t_count {
                c[idx(p, pos, t)] = problem.affinity[p][pos][t];
            }
        }
    }
    let mut a_eq: Vec<Vec<f64>> = Vec::new();
    let mut b_eq: Vec<f64> = Vec::new();
    let mut a_ub: Vec<Vec<f64>> = Vec::new();
    let mut b_ub: Vec<f64> = Vec::new();
    // (1) Σ_pos x_{p,pos,t} ≤ 1
    for p in 0..p_count {
        for t in 0..t_count {
            let mut row = vec![0.0; n];
            for pos in 0..k {
                row[idx(p, pos, t)] = 1.0;
            }
            a_ub.push(row);
            b_ub.push(1.0);
        }
    }
    // (2) Σ_p x_{p,pos,t} = 1
    for pos in 0..k {
        for t in 0..t_count {
            let mut row = vec![0.0; n];
            for p in 0..p_count {
                row[idx(p, pos, t)] = 1.0;
            }
            a_eq.push(row);
            b_eq.push(1.0);
        }
    }
    // (3) Σ_pos x_{p,pos,t} + Σ_pos x_{p,pos,t+1} ≥ 1  ->  ≤ form with negative RHS.
    if problem.enforce_no_consecutive_bench {
        for p in 0..p_count {
            if !player_is_fieldable(problem, p) {
                continue;
            }
            for t in 0..t_count.saturating_sub(1) {
                let mut row = vec![0.0; n];
                for pos in 0..k {
                    row[idx(p, pos, t)] -= 1.0;
                    row[idx(p, pos, t + 1)] -= 1.0;
                }
                a_ub.push(row);
                b_ub.push(-1.0);
            }
        }
    }
    // (4) Stamina: over any window of (M+1) periods a player is on field ≤ M
    //     times, i.e. Σ_{τ=t0}^{t0+M} Σ_pos x_{p,pos,τ} ≤ M.
    for p in 0..p_count {
        if let Some(m) = player_max_contiguous_on_field(problem, p) {
            if t_count > m {
                for t0 in 0..=(t_count - (m + 1)) {
                    let mut row = vec![0.0; n];
                    for dt in 0..=m {
                        let t = t0 + dt;
                        for pos in 0..k {
                            row[idx(p, pos, t)] += 1.0;
                        }
                    }
                    a_ub.push(row);
                    b_ub.push(m as f64);
                }
            }
        }
    }
    // (4b) Max bench run: over any window of (B+1) slots a fieldable player
    //      must appear on field at least once.
    for p in 0..p_count {
        if !player_is_fieldable(problem, p) {
            continue;
        }
        if let Some(bmax) = player_max_contiguous_bench(problem, p) {
            if t_count > bmax {
                for t0 in 0..=(t_count - (bmax + 1)) {
                    let mut row = vec![0.0; n];
                    for dt in 0..=bmax {
                        let t = t0 + dt;
                        for pos in 0..k {
                            row[idx(p, pos, t)] -= 1.0;
                        }
                    }
                    a_ub.push(row);
                    b_ub.push(-1.0);
                }
            }
        }
    }
    // (5) Min stint length: when a player starts a run, they must remain on
    //     field for their configured minimum contiguous slots.
    for p in 0..p_count {
        let Some(m) = player_min_contiguous_on_field(problem, p) else {
            continue;
        };
        for t0 in 0..t_count {
            if t0 + m <= t_count {
                for tau in (t0 + 1)..(t0 + m) {
                    let mut row = vec![0.0; n];
                    for pos in 0..k {
                        row[idx(p, pos, t0)] += 1.0;
                        if t0 > 0 {
                            row[idx(p, pos, t0 - 1)] -= 1.0;
                        }
                        row[idx(p, pos, tau)] -= 1.0;
                    }
                    a_ub.push(row);
                    b_ub.push(0.0);
                }
            } else if t0 > 0 {
                let mut row = vec![0.0; n];
                for pos in 0..k {
                    row[idx(p, pos, t0)] += 1.0;
                    row[idx(p, pos, t0 - 1)] -= 1.0;
                }
                a_ub.push(row);
                b_ub.push(0.0);
            } else {
                let mut row = vec![0.0; n];
                for pos in 0..k {
                    row[idx(p, pos, t0)] = 1.0;
                }
                a_ub.push(row);
                b_ub.push(0.0);
            }
        }
    }
    let ub = vec![Some(1.0); n];
    LPProblem {
        sense: Sense::Max,
        c,
        a_ub: Some(a_ub),
        b_ub: Some(b_ub),
        a_eq: Some(a_eq),
        b_eq: Some(b_eq),
        ub: Some(ub),
        ..Default::default()
    }
}

/// Result of the LP relaxation policy.
#[derive(Clone, Debug)]
pub struct LPRelaxedScheduleResult {
    pub schedule: Schedule,
    pub lp_value: f64,
    pub upper_bound_on_rotation: f64,
    pub solver: String,
    pub iters: usize,
}

/// `PureTransform<SoccerProblem, LPRelaxedScheduleResult>`.
pub struct PolicyLPRelaxed;

impl<'a> Transform<&'a SoccerProblem, LPRelaxedScheduleResult> for PolicyLPRelaxed {
    fn transform(&self, problem: &'a SoccerProblem) -> LPRelaxedScheduleResult {
        policy_lp_relaxed(problem)
    }
}

fn solve_soccer_rotation_lp(lp: &LPProblem) -> crate::des::general::lp::LPSolution {
    if std::env::var("LP_SOLVER")
        .ok()
        .is_some_and(|value| !value.trim().is_empty())
    {
        return solve_lp(lp, &LpSolverOptions::default());
    }

    let fast = solve_lp_external(
        lp,
        &ExternalSolverOptions {
            method: Some("highs-ds".to_string()),
            ..Default::default()
        },
    );
    if fast.status == LPStatus::Optimal {
        return fast;
    }

    let mut fallback = solve_lp(lp, &LpSolverOptions::default());
    let fast_message = fast
        .message
        .unwrap_or_else(|| fast.status.as_str().to_string());
    let prefix = fallback
        .message
        .take()
        .map(|message| format!("{message}; "))
        .unwrap_or_default();
    fallback.message = Some(format!(
        "{prefix}highs-ds fast path unavailable, fell back to {}: {fast_message}",
        fallback.solver
    ));
    fallback
}

/// Solve the LP relaxation and round it with fairness-aware per-period Hungarian.
pub fn policy_lp_relaxed(problem: &SoccerProblem) -> LPRelaxedScheduleResult {
    let lp = build_soccer_lp(problem);
    let sol = solve_soccer_rotation_lp(&lp);
    if sol.status != LPStatus::Optimal {
        panic!(
            "soccer LP relaxation failed: {} — {}",
            sol.status.as_str(),
            sol.message.clone().unwrap_or_default()
        );
    }
    let (p_count, k, t_count) = (
        problem.num_players,
        problem.num_positions,
        problem.num_periods,
    );
    let expected_variables = p_count * k * t_count;
    if !soccer_lp_solution_is_roundable(&sol.x, sol.objective, expected_variables) {
        panic!(
            "soccer LP relaxation returned malformed optimal solution: x_len={} expected_vars={} objective={}",
            sol.x.len(),
            expected_variables,
            sol.objective
        );
    }
    // Marginal "player p on field in period t" = Σ_pos x_{p,pos,t}.
    let mut on_field_marg: Vec<Vec<f64>> = Vec::new();
    for t in 0..t_count {
        let mut row: Vec<f64> = Vec::new();
        for p in 0..p_count {
            let mut s = 0.0;
            for pos in 0..k {
                s += sol.x[(p * k + pos) * t_count + t];
            }
            row.push(s);
        }
        on_field_marg.push(row);
    }
    let mut assignment: Vec<Vec<i64>> = Vec::new();
    let mut bench: Vec<Vec<usize>> = Vec::new();
    let mut prev_bench: Vec<usize> = Vec::new();
    for t in 0..t_count {
        let must_play: HashSet<usize> = prev_bench.iter().copied().collect();
        let mut candidates: Vec<usize> = (0..p_count).collect();
        candidates.sort_by(|&a, &b| {
            on_field_marg[t][b]
                .partial_cmp(&on_field_marg[t][a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        let mut chosen: Vec<usize> = Vec::new();
        for &p in &candidates {
            if must_play.contains(&p) {
                chosen.push(p);
            }
            if chosen.len() >= k {
                break;
            }
        }
        for &p in &candidates {
            if chosen.len() >= k {
                break;
            }
            if !chosen.contains(&p) {
                chosen.push(p);
            }
        }
        let period_assign = hungarian_assign_period(problem, t, &chosen);
        assignment.push(period_assign);
        let mut tbench: Vec<usize> = (0..p_count).filter(|p| !chosen.contains(p)).collect();
        tbench.sort_unstable();
        bench.push(tbench.clone());
        prev_bench = tbench;
    }
    LPRelaxedScheduleResult {
        schedule: Schedule { assignment, bench },
        lp_value: sol.objective,
        upper_bound_on_rotation: sol.objective,
        solver: sol.solver,
        iters: sol.iters.unwrap_or(0),
    }
}

fn soccer_lp_solution_is_roundable(x: &[f64], objective: f64, expected_variables: usize) -> bool {
    objective.is_finite()
        && x.len() >= expected_variables
        && x.iter()
            .take(expected_variables)
            .all(|value| value.is_finite())
}

// -----------------------------------------------------------------------------
// IP/MIP over the exact 0/1 rotation program.
// -----------------------------------------------------------------------------

/// The IP/MIP model plus its variable-indexing scheme.
#[derive(Clone, Debug)]
pub struct SoccerIPMIPModel {
    pub ip: IPMIPProblem,
    num_positions: usize,
    num_periods: usize,
    /// Number of `x[p,pos,t]` assignment variables at the front of the vector
    /// (auxiliary synergy / on-field / sub variables follow).
    pub num_x_vars: usize,
}

impl SoccerIPMIPModel {
    /// `x[p,pos,t]`'s dense column index.
    pub fn variable_index(&self, player_id: usize, position_id: usize, period: usize) -> usize {
        (player_id * self.num_positions + position_id) * self.num_periods + period
    }
}

/// `PureTransform<SoccerProblem, SoccerIPMIPModel>`.
pub struct BuildSoccerIPMIP;

impl<'a> Transform<&'a SoccerProblem, SoccerIPMIPModel> for BuildSoccerIPMIP {
    fn transform(&self, problem: &'a SoccerProblem) -> SoccerIPMIPModel {
        build_soccer_ipmip(problem)
    }
}

/// Build the exact 0/1 rotation program as an [`IPMIPProblem`].
pub fn build_soccer_ipmip(problem: &SoccerProblem) -> SoccerIPMIPModel {
    let (p_count, k, t_count) = (
        problem.num_players,
        problem.num_positions,
        problem.num_periods,
    );
    let n = p_count * k * t_count;
    let idx = |p: usize, pos: usize, t: usize| (p * k + pos) * t_count + t;
    let player_name = |p: usize| {
        problem
            .player_names
            .as_ref()
            .and_then(|v| v.get(p))
            .cloned()
            .unwrap_or_else(|| format!("Player{}", p + 1))
    };
    let position_name = |pos: usize| {
        problem
            .position_names
            .as_ref()
            .and_then(|v| v.get(pos))
            .cloned()
            .unwrap_or_else(|| pos.to_string())
    };

    let mut c = vec![0.0; n];
    let mut var_names: Vec<String> = vec![String::new(); n];
    let mut variable_nodes: Vec<VariableNode> = Vec::new();
    for p in 0..p_count {
        for pos in 0..k {
            for t in 0..t_count {
                let j = idx(p, pos, t);
                c[j] = problem.affinity[p][pos][t];
                var_names[j] = format!("x_{}_{}_T{}", player_name(p), position_name(pos), t + 1);
                variable_nodes.push(VariableNode {
                    var_index: j,
                    node_id: format!(
                        "movable:{}:period-{}:position-{}",
                        player_name(p),
                        t + 1,
                        position_name(pos)
                    ),
                    label: Some(format!(
                        "{} -> {} in period {}",
                        player_name(p),
                        position_name(pos),
                        t + 1
                    )),
                });
            }
        }
    }

    let mut a: Vec<Vec<f64>> = Vec::new();
    let mut b: Vec<f64> = Vec::new();
    let mut con_names: Vec<String> = Vec::new();
    let mut constraint_nodes: Vec<ConstraintNode> = Vec::new();

    // (1) Each player can occupy at most one position per period.
    for p in 0..p_count {
        for t in 0..t_count {
            let mut row = vec![0.0; n];
            for pos in 0..k {
                row[idx(p, pos, t)] = 1.0;
            }
            let row_index = a.len();
            a.push(row);
            b.push(1.0);
            con_names.push(format!("player_once_{}_T{}", player_name(p), t + 1));
            constraint_nodes.push(ConstraintNode {
                row_index,
                node_id: format!("station:eligibility:{}:T{}", player_name(p), t + 1),
                label: Some(format!(
                    "{} at most one field role in period {}",
                    player_name(p),
                    t + 1
                )),
            });
        }
    }

    // (2) Each position is filled by exactly one player per period.
    for pos in 0..k {
        for t in 0..t_count {
            let mut le = vec![0.0; n];
            let mut ge = vec![0.0; n];
            for p in 0..p_count {
                le[idx(p, pos, t)] = 1.0;
                ge[idx(p, pos, t)] = -1.0;
            }
            let row_index_le = a.len();
            a.push(le);
            b.push(1.0);
            con_names.push(format!(
                "position_filled_le_{}_T{}",
                position_name(pos),
                t + 1
            ));
            constraint_nodes.push(ConstraintNode {
                row_index: row_index_le,
                node_id: format!("station:position:{}:T{}", position_name(pos), t + 1),
                label: Some(format!(
                    "position {} has at most one player in period {}",
                    position_name(pos),
                    t + 1
                )),
            });
            let row_index_ge = a.len();
            a.push(ge);
            b.push(-1.0);
            con_names.push(format!(
                "position_filled_ge_{}_T{}",
                position_name(pos),
                t + 1
            ));
            constraint_nodes.push(ConstraintNode {
                row_index: row_index_ge,
                node_id: format!("station:position:{}:T{}", position_name(pos), t + 1),
                label: Some(format!(
                    "position {} has at least one player in period {}",
                    position_name(pos),
                    t + 1
                )),
            });
        }
    }

    // (3) Fairness: a player may not be benched in two consecutive periods.
    if problem.enforce_no_consecutive_bench {
        for p in 0..p_count {
            if !player_is_fieldable(problem, p) {
                continue;
            }
            for t in 0..t_count.saturating_sub(1) {
                let mut row = vec![0.0; n];
                for pos in 0..k {
                    row[idx(p, pos, t)] -= 1.0;
                    row[idx(p, pos, t + 1)] -= 1.0;
                }
                let row_index = a.len();
                a.push(row);
                b.push(-1.0);
                con_names.push(format!(
                    "no_consecutive_bench_{}_T{}_{}",
                    player_name(p),
                    t + 1,
                    t + 2
                ));
                constraint_nodes.push(ConstraintNode {
                    row_index,
                    node_id: format!("station:fairness:{}:T{}-{}", player_name(p), t + 1, t + 2),
                    label: Some(format!(
                        "{} plays in period {} or {}",
                        player_name(p),
                        t + 1,
                        t + 2
                    )),
                });
            }
        }
    }

    // (4) Stamina: a player may stay on field at most M consecutive periods.
    //     For every window [t0, t0+M], Σ_{τ} Σ_pos x[p,pos,τ] ≤ M so each
    //     player must sit at least once inside any (M+1)-period window.
    for p in 0..p_count {
        if let Some(m) = player_max_contiguous_on_field(problem, p) {
            if t_count > m {
                for t0 in 0..=(t_count - (m + 1)) {
                    let mut row = vec![0.0; n];
                    for dt in 0..=m {
                        let t = t0 + dt;
                        for pos in 0..k {
                            row[idx(p, pos, t)] += 1.0;
                        }
                    }
                    let row_index = a.len();
                    a.push(row);
                    b.push(m as f64);
                    con_names.push(format!(
                        "max_consec_on_field_{}_T{}_{}",
                        player_name(p),
                        t0 + 1,
                        t0 + m + 1
                    ));
                    constraint_nodes.push(ConstraintNode {
                        row_index,
                        node_id: format!(
                            "station:stamina:{}:T{}-{}",
                            player_name(p),
                            t0 + 1,
                            t0 + m + 1
                        ),
                        label: Some(format!(
                            "{} rests at least once in periods {}-{}",
                            player_name(p),
                            t0 + 1,
                            t0 + m + 1
                        )),
                    });
                }
            }
        }
    }

    // (4b) Max bench run: in any (B+1)-slot window, the player appears on field
    //      at least once.
    for p in 0..p_count {
        if !player_is_fieldable(problem, p) {
            continue;
        }
        if let Some(bmax) = player_max_contiguous_bench(problem, p) {
            if t_count > bmax {
                for t0 in 0..=(t_count - (bmax + 1)) {
                    let mut row = vec![0.0; n];
                    for dt in 0..=bmax {
                        let t = t0 + dt;
                        for pos in 0..k {
                            row[idx(p, pos, t)] -= 1.0;
                        }
                    }
                    let row_index = a.len();
                    a.push(row);
                    b.push(-1.0);
                    con_names.push(format!(
                        "max_contig_bench_{}_T{}_{}",
                        player_name(p),
                        t0 + 1,
                        t0 + bmax + 1
                    ));
                    constraint_nodes.push(ConstraintNode {
                        row_index,
                        node_id: format!(
                            "station:bench-run:{}:T{}-{}",
                            player_name(p),
                            t0 + 1,
                            t0 + bmax + 1
                        ),
                        label: Some(format!(
                            "{} plays at least once in periods {}-{}",
                            player_name(p),
                            t0 + 1,
                            t0 + bmax + 1
                        )),
                    });
                }
            }
        }
    }

    // (4c) Min stint length: if a player starts in T, they stay on through the
    //      configured minimum contiguous time slots. Late starts that cannot
    //      satisfy the minimum are disallowed.
    for p in 0..p_count {
        let Some(m) = player_min_contiguous_on_field(problem, p) else {
            continue;
        };
        for t0 in 0..t_count {
            if t0 + m <= t_count {
                for tau in (t0 + 1)..(t0 + m) {
                    let mut row = vec![0.0; n];
                    for pos in 0..k {
                        row[idx(p, pos, t0)] += 1.0;
                        if t0 > 0 {
                            row[idx(p, pos, t0 - 1)] -= 1.0;
                        }
                        row[idx(p, pos, tau)] -= 1.0;
                    }
                    let row_index = a.len();
                    a.push(row);
                    b.push(0.0);
                    con_names.push(format!(
                        "min_contig_on_field_{}_T{}_T{}",
                        player_name(p),
                        t0 + 1,
                        tau + 1
                    ));
                    constraint_nodes.push(ConstraintNode {
                        row_index,
                        node_id: format!(
                            "station:min-stint:{}:T{}-{}",
                            player_name(p),
                            t0 + 1,
                            tau + 1
                        ),
                        label: Some(format!(
                            "{} continues from period {} through {}",
                            player_name(p),
                            t0 + 1,
                            tau + 1
                        )),
                    });
                }
            } else {
                let mut row = vec![0.0; n];
                for pos in 0..k {
                    row[idx(p, pos, t0)] += 1.0;
                    if t0 > 0 {
                        row[idx(p, pos, t0 - 1)] -= 1.0;
                    }
                }
                let row_index = a.len();
                a.push(row);
                b.push(0.0);
                con_names.push(format!(
                    "min_contig_no_late_start_{}_T{}",
                    player_name(p),
                    t0 + 1
                ));
                constraint_nodes.push(ConstraintNode {
                    row_index,
                    node_id: format!("station:min-stint:{}:T{}", player_name(p), t0 + 1),
                    label: Some(format!(
                        "{} cannot start a short stint in period {}",
                        player_name(p),
                        t0 + 1
                    )),
                });
            }
        }
    }

    // (5) Planner: unavailable players never fielded.
    for p in 0..p_count {
        if !player_is_fieldable(problem, p) {
            for t in 0..t_count {
                let mut row = vec![0.0; c.len()];
                for pos in 0..k {
                    row[idx(p, pos, t)] = 1.0;
                }
                let row_index = a.len();
                a.push(row);
                b.push(0.0);
                con_names.push(format!("unavailable_{}_T{}", player_name(p), t + 1));
                constraint_nodes.push(ConstraintNode {
                    row_index,
                    node_id: format!("station:unavailable:{}:T{}", player_name(p), t + 1),
                    label: Some(format!(
                        "{} unavailable in period {}",
                        player_name(p),
                        t + 1
                    )),
                });
            }
        }
    }

    // (6) Fixed position: player locked to one role all match.
    if let Some(fixed) = &problem.fixed_position {
        for p in 0..p_count.min(fixed.len()) {
            if let Some(fp) = fixed[p] {
                if fp >= k {
                    continue;
                }
                for t in 0..t_count {
                    // x[p,fp,t] = 1
                    let mut eq = vec![0.0; c.len()];
                    eq[idx(p, fp, t)] = 1.0;
                    let row_index = a.len();
                    a.push(eq);
                    b.push(1.0);
                    con_names.push(format!("fixed_{}_pos{}_T{}", player_name(p), fp, t + 1));
                    constraint_nodes.push(ConstraintNode {
                        row_index,
                        node_id: format!(
                            "station:fixed:{}:{}:T{}",
                            player_name(p),
                            position_name(fp),
                            t + 1
                        ),
                        label: Some(format!(
                            "{} fixed at {} in period {}",
                            player_name(p),
                            position_name(fp),
                            t + 1
                        )),
                    });
                    for pos in 0..k {
                        if pos != fp {
                            let mut zrow = vec![0.0; c.len()];
                            zrow[idx(p, pos, t)] = 1.0;
                            a.push(zrow);
                            b.push(0.0);
                            con_names.push(format!(
                                "fixed_ban_{}_pos{}_T{}",
                                player_name(p),
                                pos,
                                t + 1
                            ));
                        }
                    }
                }
            }
        }
    }

    // (7) Banned (player, position) pairs.
    if let Some(banned) = &problem.banned_positions {
        for p in 0..p_count.min(banned.len()) {
            for pos in 0..k.min(banned[p].len()) {
                if !banned[p][pos] {
                    continue;
                }
                for t in 0..t_count {
                    let mut row = vec![0.0; c.len()];
                    row[idx(p, pos, t)] = 1.0;
                    let row_index = a.len();
                    a.push(row);
                    b.push(0.0);
                    con_names.push(format!("banned_{}_pos{}_T{}", player_name(p), pos, t + 1));
                    constraint_nodes.push(ConstraintNode {
                        row_index,
                        node_id: format!(
                            "station:banned:{}:{}:T{}",
                            player_name(p),
                            position_name(pos),
                            t + 1
                        ),
                        label: Some(format!(
                            "{} banned from {} in period {}",
                            player_name(p),
                            position_name(pos),
                            t + 1
                        )),
                    });
                }
            }
        }
    }

    let n_x = n;

    // Synergy overrides on the linear objective for subject cells.
    let rules = problem.synergy_rules.clone().unwrap_or_default();
    if !rules.is_empty() {
        for rule in &rules {
            for t in 0..t_count {
                let j = idx(rule.player, rule.position, t);
                c[j] = rule.score_without;
            }
        }
    }

    let need_synergy = !rules.is_empty();
    let need_subs =
        (problem.max_subs_per_game.is_some() || problem.min_subs_per_game.is_some()) && t_count > 1;
    let n_synergy = if need_synergy {
        rules.len() * t_count
    } else {
        0
    };
    let n_on = if need_subs { p_count * t_count } else { 0 };
    let n_sub = if need_subs {
        p_count * (t_count - 1)
    } else {
        0
    };
    let n_total = n_x + n_synergy + n_on + n_sub;

    if n_total > n_x {
        c.resize(n_total, 0.0);
        var_names.resize(n_total, String::new());

        let synergy_index = |rule_i: usize, t: usize| -> usize { n_x + rule_i * t_count + t };
        let on_index = |p: usize, t: usize| -> usize { n_x + n_synergy + p * t_count + t };
        let sub_index =
            |p: usize, t: usize| -> usize { n_x + n_synergy + n_on + p * (t_count - 1) + (t - 1) };

        if need_synergy {
            for (ri, rule) in rules.iter().enumerate() {
                for t in 0..t_count {
                    let zj = synergy_index(ri, t);
                    c[zj] = rule.score_with - rule.score_without;
                    var_names[zj] = format!(
                        "synergy_{}_{}_T{}",
                        player_name(rule.player),
                        position_name(rule.position),
                        t + 1
                    );
                    variable_nodes.push(VariableNode {
                        var_index: zj,
                        node_id: format!(
                            "synergy:{}:{}:T{}",
                            player_name(rule.player),
                            position_name(rule.position),
                            t + 1
                        ),
                        label: Some(format!(
                            "synergy {}@{} if {}@{}",
                            player_name(rule.player),
                            position_name(rule.position),
                            player_name(rule.partner_player),
                            position_name(rule.partner_position)
                        )),
                    });
                    let x_sub = idx(rule.player, rule.position, t);
                    let x_part = idx(rule.partner_player, rule.partner_position, t);
                    let synergy_rows: Vec<(&str, Vec<(usize, f64)>, f64)> = vec![
                        ("subject", vec![(zj, 1.0), (x_sub, -1.0)], 0.0),
                        ("partner", vec![(zj, 1.0), (x_part, -1.0)], 0.0),
                        (
                            "activation",
                            vec![(zj, -1.0), (x_sub, 1.0), (x_part, 1.0)],
                            1.0,
                        ),
                    ];
                    for (kind, coeffs, rhs) in synergy_rows {
                        let mut row = vec![0.0; n_total];
                        for (j, v) in coeffs {
                            row[j] = v;
                        }
                        a.push(row);
                        b.push(rhs);
                        con_names.push(format!(
                            "synergy_{}_{}_{}_T{}",
                            kind,
                            player_name(rule.player),
                            position_name(rule.position),
                            t + 1
                        ));
                    }
                }
            }
        }

        if need_subs {
            for p in 0..p_count {
                for t in 0..t_count {
                    let oj = on_index(p, t);
                    var_names[oj] = format!("on_{}_T{}", player_name(p), t + 1);
                    let mut link = vec![0.0; n_total];
                    for pos in 0..k {
                        link[idx(p, pos, t)] = 1.0;
                    }
                    link[oj] = -1.0;
                    a.push(link);
                    b.push(0.0);
                    con_names.push(format!("on_link_{}_T{}", player_name(p), t + 1));
                }
            }
            for p in 0..p_count {
                for t in 1..t_count {
                    let sj = sub_index(p, t);
                    var_names[sj] = format!("sub_off_{}_T{}", player_name(p), t + 1);
                    let o_prev = on_index(p, t - 1);
                    let o_cur = on_index(p, t);
                    let sub_rows: Vec<(Vec<(usize, f64)>, f64)> = vec![
                        (vec![(sj, 1.0), (o_prev, -1.0)], 0.0),
                        (vec![(sj, 1.0), (o_cur, 1.0)], 1.0),
                        (vec![(sj, 1.0), (o_prev, -1.0), (o_cur, 1.0)], 0.0),
                    ];
                    for (sub_row_i, (coeffs, rhs)) in sub_rows.into_iter().enumerate() {
                        let mut row = vec![0.0; n_total];
                        for (j, v) in coeffs {
                            row[j] = v;
                        }
                        a.push(row);
                        b.push(rhs);
                        con_names.push(format!(
                            "sub_off_link_{}_T{}_{}",
                            player_name(p),
                            t + 1,
                            sub_row_i + 1
                        ));
                    }
                }
            }
            if let Some(max_subs) = problem.max_subs_per_game {
                let mut row = vec![0.0; n_total];
                for p in 0..p_count {
                    for t in 1..t_count {
                        row[sub_index(p, t)] = 1.0;
                    }
                }
                let row_index = a.len();
                a.push(row);
                b.push(max_subs as f64);
                con_names.push(format!("max_subs_{max_subs}"));
                constraint_nodes.push(ConstraintNode {
                    row_index,
                    node_id: "station:max-subs".to_string(),
                    label: Some(format!("at most {max_subs} substitutions per match")),
                });
            }
            if let Some(min_subs) = problem.min_subs_per_game {
                let mut row = vec![0.0; n_total];
                for p in 0..p_count {
                    for t in 1..t_count {
                        row[sub_index(p, t)] = -1.0;
                    }
                }
                let row_index = a.len();
                a.push(row);
                b.push(-(min_subs as f64));
                con_names.push(format!("min_subs_{min_subs}"));
                constraint_nodes.push(ConstraintNode {
                    row_index,
                    node_id: "station:min-subs".to_string(),
                    label: Some(format!("at least {min_subs} substitutions per match")),
                });
            }
        }

        for row in &mut a {
            if row.len() < n_total {
                row.resize(n_total, 0.0);
            }
        }
    }

    let n_vars = if n_total > n_x { n_total } else { n_x };

    SoccerIPMIPModel {
        ip: IPMIPProblem {
            sense: Sense::Max,
            c: if c.len() == n_vars {
                c
            } else {
                let mut c2 = c;
                c2.resize(n_vars, 0.0);
                c2
            },
            a,
            b,
            integer_vars: vec![true; n_vars],
            ub: Some(vec![1.0; n_vars]),
            var_names: Some(var_names),
            con_names: Some(con_names),
            lazy_constraints: None,
            variable_nodes: Some(variable_nodes),
            constraint_nodes: Some(constraint_nodes),
        },
        num_positions: k,
        num_periods: t_count,
        num_x_vars: n_x,
    }
}

/// Input for [`ScheduleFromSoccerIPMIPVector`].
pub struct ScheduleFromSoccerIPMIPVectorInput<'a> {
    pub problem: &'a SoccerProblem,
    pub model: &'a SoccerIPMIPModel,
    pub x: &'a [f64],
}

/// `PureTransform<ScheduleFromSoccerIPMIPVectorInput, Schedule | null>`.
pub struct ScheduleFromSoccerIPMIPVector {
    threshold: f64,
}

impl ScheduleFromSoccerIPMIPVector {
    pub fn new(threshold: f64) -> Self {
        ScheduleFromSoccerIPMIPVector { threshold }
    }
}

impl Default for ScheduleFromSoccerIPMIPVector {
    fn default() -> Self {
        ScheduleFromSoccerIPMIPVector { threshold: 0.5 }
    }
}

impl<'a> Transform<ScheduleFromSoccerIPMIPVectorInput<'a>, Option<Schedule>>
    for ScheduleFromSoccerIPMIPVector
{
    fn transform(&self, input: ScheduleFromSoccerIPMIPVectorInput<'a>) -> Option<Schedule> {
        schedule_from_soccer_ipmip_vector(input.problem, input.model, input.x, self.threshold)
    }
}

/// Decode an IP/MIP solution vector into a feasible [`Schedule`], or `None`.
pub fn schedule_from_soccer_ipmip_vector(
    problem: &SoccerProblem,
    model: &SoccerIPMIPModel,
    x: &[f64],
    threshold: f64,
) -> Option<Schedule> {
    if x.len() != model.ip.c.len() {
        return None;
    }
    let mut assignment: Vec<Vec<i64>> = Vec::new();
    let mut bench: Vec<Vec<usize>> = Vec::new();
    for t in 0..problem.num_periods {
        let mut used: HashSet<usize> = HashSet::new();
        let mut period_assign = vec![-1i64; problem.num_positions];
        for pos in 0..problem.num_positions {
            let mut chosen: i64 = -1;
            let mut best = threshold;
            for p in 0..problem.num_players {
                let v = x[model.variable_index(p, pos, t)];
                if v > best {
                    best = v;
                    chosen = p as i64;
                }
            }
            if chosen < 0 || used.contains(&(chosen as usize)) {
                return None;
            }
            period_assign[pos] = chosen;
            used.insert(chosen as usize);
        }
        assignment.push(period_assign);
        let mut tbench: Vec<usize> = (0..problem.num_players)
            .filter(|p| !used.contains(p))
            .collect();
        tbench.sort_unstable();
        bench.push(tbench);
    }
    let schedule = Schedule { assignment, bench };
    if validate_schedule_structure(problem, &schedule).is_some() {
        return None;
    }
    let eval = evaluate_schedule(problem, &schedule);
    if !eval.fairness_ok || !eval.stamina_ok || !eval.subs_ok {
        return None;
    }
    Some(schedule)
}

/// Options for the IP/MIP rotation policy.
#[derive(Clone, Debug, Default)]
pub struct SoccerIPMIPPolicyOptions {
    pub time_limit_ms: Option<f64>,
    pub max_nodes: Option<usize>,
    pub max_ticks: Option<usize>,
    pub lp_max_iters: Option<usize>,
    pub lp_algorithm: Option<LpRelaxationAlgorithm>,
    pub allow_external_solvers: Option<bool>,
    pub max_cut_rounds: Option<usize>,
    pub node_selection: Option<NodeSelection>,
    pub branch_rule: Option<BranchRule>,
    pub heuristic_passes: Option<usize>,
    pub cancel_flag: Option<Arc<AtomicBool>>,
    pub take_next_incumbent_signal: Option<Arc<AtomicU64>>,
    /// Keep demos animatable even when the MIP has no incumbent yet. Default true.
    pub fallback_to_mdp: Option<bool>,
}

/// Result of the IP/MIP rotation policy.
#[derive(Clone, Debug)]
pub struct SoccerIPMIPPolicyResult {
    pub schedule: Schedule,
    pub model: SoccerIPMIPModel,
    pub mip: IPMIPSolution,
    pub solver_options: IPMIPSolveOptions,
    pub used_fallback: bool,
    pub fallback_reason: Option<String>,
}

/// `PureTransform<SoccerProblem, SoccerIPMIPPolicyResult>`.
pub struct PolicyIPMIPFeasible {
    opts: SoccerIPMIPPolicyOptions,
}

impl PolicyIPMIPFeasible {
    pub fn new(opts: SoccerIPMIPPolicyOptions) -> Self {
        PolicyIPMIPFeasible { opts }
    }
}

impl<'a> Transform<&'a SoccerProblem, SoccerIPMIPPolicyResult> for PolicyIPMIPFeasible {
    fn transform(&self, problem: &'a SoccerProblem) -> SoccerIPMIPPolicyResult {
        policy_ipmip_feasible(problem, &self.opts)
    }
}

fn soccer_lp_algorithm_uses_external(algorithm: LpRelaxationAlgorithm) -> bool {
    matches!(
        algorithm,
        LpRelaxationAlgorithm::Concrete(
            ConcreteLpRelaxationAlgorithm::ExternalHighs
                | ConcreteLpRelaxationAlgorithm::ExternalHighsDs
                | ConcreteLpRelaxationAlgorithm::ExternalHighsIpm
        )
    )
}

/// Solve the rotation IP/MIP, decode an incumbent, falling back to the exact MDP.
pub fn policy_ipmip_feasible(
    problem: &SoccerProblem,
    opts: &SoccerIPMIPPolicyOptions,
) -> SoccerIPMIPPolicyResult {
    let model = build_soccer_ipmip(problem);
    let max_nodes = opts.max_nodes.unwrap_or(5_000);
    let lp_algorithm = opts.lp_algorithm.unwrap_or(LpRelaxationAlgorithm::Concrete(
        ConcreteLpRelaxationAlgorithm::InternalSimplex,
    ));
    let solver_options = IPMIPSolveOptions {
        time_limit_ms: Some(opts.time_limit_ms.unwrap_or(30_000.0)),
        max_nodes: Some(max_nodes),
        max_ticks: Some(opts.max_ticks.unwrap_or_else(|| 100.max(max_nodes * 8))),
        lp_max_iters: Some(opts.lp_max_iters.unwrap_or(6_000)),
        lp_algorithm: Some(lp_algorithm),
        allow_external_solvers: Some(
            opts.allow_external_solvers
                .unwrap_or_else(|| soccer_lp_algorithm_uses_external(lp_algorithm)),
        ),
        max_cut_rounds: Some(opts.max_cut_rounds.unwrap_or(0)),
        node_selection: Some(opts.node_selection.unwrap_or(NodeSelection::BestBound)),
        branch_rule: Some(opts.branch_rule.unwrap_or(BranchRule::MostFractional)),
        heuristic_passes: Some(opts.heuristic_passes.unwrap_or(120)),
        cancel_flag: opts.cancel_flag.clone(),
        take_next_incumbent_signal: opts.take_next_incumbent_signal.clone(),
        ..Default::default()
    };
    let mip = solve_ipmip_with_des(model.ip.clone(), solver_options.clone());
    let from_incumbent = schedule_from_soccer_ipmip_vector(problem, &model, &mip.x, 0.5);
    if let Some(schedule) = from_incumbent {
        return SoccerIPMIPPolicyResult {
            schedule,
            model,
            mip,
            solver_options,
            used_fallback: false,
            fallback_reason: None,
        };
    }
    if mip.status == crate::des::general::ip_mip_des::IPMIPStatus::Cancelled {
        panic!("planner solve cancelled by user before a feasible incumbent was available");
    }
    if opts.fallback_to_mdp == Some(false) {
        panic!(
            "soccer IP/MIP produced no decodable feasible schedule: status={}",
            mip.status.as_str()
        );
    }
    if has_planner_only_constraints(problem) {
        if let Some(schedule) = policy_greedy_feasible_schedule(problem) {
            let reason = format!(
                "IP/MIP status={} had no feasible incumbent; used constructive planner fallback because active constraints are not representable by the exact MDP fallback",
                mip.status.as_str()
            );
            return SoccerIPMIPPolicyResult {
                schedule,
                model,
                mip,
                solver_options,
                used_fallback: true,
                fallback_reason: Some(reason),
            };
        }
        panic!(
            "soccer IP/MIP produced no decodable feasible schedule and the constructive planner fallback could not satisfy the active planner constraints: status={}",
            mip.status.as_str()
        );
    }
    let mdp = policy_mdp_vi(problem);
    let schedule = mdp.to_schedule();
    let reason = format!(
        "IP/MIP status={} had no feasible incumbent; used exact MDP schedule for continuity",
        mip.status.as_str()
    );
    SoccerIPMIPPolicyResult {
        schedule,
        model,
        mip,
        solver_options,
        used_fallback: true,
        fallback_reason: Some(reason),
    }
}

fn has_planner_only_constraints(problem: &SoccerProblem) -> bool {
    let has_unavailable = problem
        .player_status
        .as_ref()
        .map(|statuses| {
            statuses
                .iter()
                .any(|s| matches!(s, PlayerStatus::Awol | PlayerStatus::Injured))
        })
        .unwrap_or(false);
    let has_fixed = problem
        .fixed_position
        .as_ref()
        .map(|fixed| fixed.iter().any(|p| p.is_some()))
        .unwrap_or(false);
    let has_banned = problem
        .banned_positions
        .as_ref()
        .map(|rows| rows.iter().any(|row| row.iter().any(|&b| b)))
        .unwrap_or(false);
    let has_synergy = problem
        .synergy_rules
        .as_ref()
        .map(|rules| !rules.is_empty())
        .unwrap_or(false);
    let has_min_contiguous = problem
        .min_contiguous_on_field
        .as_ref()
        .map(|limits| limits.iter().any(|&limit| limit > 1))
        .unwrap_or(false);
    let has_max_contiguous = problem
        .max_contiguous_on_field
        .as_ref()
        .map(|limits| limits.iter().any(|&limit| limit >= 1))
        .unwrap_or(false);
    let has_max_bench = problem
        .max_contiguous_bench
        .as_ref()
        .map(|limits| limits.iter().any(|&limit| limit >= 1))
        .unwrap_or(false);
    problem.max_consecutive_on_field.is_some()
        || has_min_contiguous
        || has_max_contiguous
        || has_max_bench
        || problem.max_subs_per_game.is_some()
        || problem.min_subs_per_game.is_some()
        || has_unavailable
        || has_fixed
        || has_banned
        || has_synergy
}

// -----------------------------------------------------------------------------
// POMDP-style hidden-fatigue feature trace.
// -----------------------------------------------------------------------------

/// Options for the hidden-fatigue belief feature extractor.
#[derive(Clone, Debug, Default)]
pub struct SoccerPOMDPFeatureOptions {
    pub initial_fresh_probability: Option<f64>,
    pub fatigue_rate: Option<f64>,
    pub recovery_rate: Option<f64>,
}

/// Per-period belief-state feature.
#[derive(Clone, Debug)]
pub struct SoccerPOMDPPeriodFeature {
    pub period: usize,
    pub expected_fresh_on_field: f64,
    pub expected_fresh_bench: f64,
    pub expected_lineup_reliability: f64,
    pub mean_belief_entropy: f64,
}

/// Summary of the hidden-fatigue belief feature trace.
#[derive(Clone, Debug)]
pub struct SoccerPOMDPFeatureSummary {
    pub model: &'static str,
    pub per_period: Vec<SoccerPOMDPPeriodFeature>,
    pub final_fresh_probability: Vec<f64>,
    pub mean_expected_fresh_on_field: f64,
    pub mean_expected_lineup_reliability: f64,
}

fn binary_entropy(p: f64) -> f64 {
    let q = 1e-12_f64.max((1.0_f64 - 1e-12).min(p));
    -(q * q.log2() + (1.0 - q) * (1.0 - q).log2())
}

/// `PureTransform<ProblemScheduleInput, SoccerPOMDPFeatureSummary>`.
pub struct EvaluateSoccerPOMDPFeatures {
    opts: SoccerPOMDPFeatureOptions,
}

impl EvaluateSoccerPOMDPFeatures {
    pub fn new(opts: SoccerPOMDPFeatureOptions) -> Self {
        EvaluateSoccerPOMDPFeatures { opts }
    }
}

impl<'a> Transform<ProblemScheduleInput<'a>, SoccerPOMDPFeatureSummary>
    for EvaluateSoccerPOMDPFeatures
{
    fn transform(&self, input: ProblemScheduleInput<'a>) -> SoccerPOMDPFeatureSummary {
        evaluate_soccer_pomdp_features(input.problem, input.schedule, &self.opts)
    }
}

/// Evaluate a schedule under a compact hidden-fatigue belief state.
pub fn evaluate_soccer_pomdp_features(
    problem: &SoccerProblem,
    schedule: &Schedule,
    opts: &SoccerPOMDPFeatureOptions,
) -> SoccerPOMDPFeatureSummary {
    let fatigue_rate = opts.fatigue_rate.unwrap_or(0.22);
    let recovery_rate = opts.recovery_rate.unwrap_or(0.55);
    let mut belief = vec![opts.initial_fresh_probability.unwrap_or(0.82); problem.num_players];
    let mut per_period: Vec<SoccerPOMDPPeriodFeature> = Vec::new();

    for t in 0..problem.num_periods {
        let on_field: HashSet<i64> = schedule.assignment[t].iter().copied().collect();
        let bench = &schedule.bench[t];
        let on_field_len = schedule.assignment[t].len();
        let expected_fresh_on_field = schedule.assignment[t]
            .iter()
            .map(|&p| belief[p as usize])
            .sum::<f64>()
            / 1.0_f64.max(on_field_len as f64);
        let expected_fresh_bench =
            bench.iter().map(|&p| belief[p]).sum::<f64>() / 1.0_f64.max(bench.len() as f64);
        let mut reliability = 0.0;
        for pos in 0..problem.num_positions {
            let p = schedule.assignment[t][pos] as usize;
            reliability += problem.affinity[p][pos][t] * (0.65 + 0.35 * belief[p]);
        }
        let mean_belief_entropy =
            belief.iter().map(|&p| binary_entropy(p)).sum::<f64>() / belief.len() as f64;
        per_period.push(SoccerPOMDPPeriodFeature {
            period: t,
            expected_fresh_on_field,
            expected_fresh_bench,
            expected_lineup_reliability: reliability,
            mean_belief_entropy,
        });

        for p in 0..problem.num_players {
            if on_field.contains(&(p as i64)) {
                belief[p] = 0.0_f64.max(belief[p] * (1.0 - fatigue_rate));
            } else {
                belief[p] = 1.0_f64.min(belief[p] + (1.0 - belief[p]) * recovery_rate);
            }
        }
    }

    let pp = per_period.len().max(1) as f64;
    let mean_expected_fresh_on_field = per_period
        .iter()
        .map(|x| x.expected_fresh_on_field)
        .sum::<f64>()
        / pp;
    let mean_expected_lineup_reliability = per_period
        .iter()
        .map(|x| x.expected_lineup_reliability)
        .sum::<f64>()
        / pp;
    SoccerPOMDPFeatureSummary {
        model: "hidden-fatigue-belief",
        per_period,
        final_fresh_probability: belief,
        mean_expected_fresh_on_field,
        mean_expected_lineup_reliability,
    }
}

// =============================================================================
// DES MATCH SIMULATOR (Layer 1)
// =============================================================================

/// Options for [`simulate_match_des`].
#[derive(Clone, Debug, Default)]
pub struct MatchSimOptions {
    pub minutes_per_period: Option<usize>,
    pub team_score_rate: Option<f64>,
    pub opp_score_rate: Option<f64>,
    pub seed: Option<u32>,
}

/// A substitution event at a period boundary.
#[derive(Clone, Debug)]
pub struct SubEvent {
    pub t: usize,
    pub period: usize,
    pub new_on_field: Vec<i64>,
    pub new_bench: Vec<usize>,
}

/// Which side scored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GoalSide {
    Us,
    Them,
}

/// A goal event.
#[derive(Clone, Copy, Debug)]
pub struct GoalEvent {
    pub t: usize,
    pub side: GoalSide,
}

/// A per-tick animation snapshot.
#[derive(Clone, Debug)]
pub struct MatchTraceFrame {
    pub t: usize,
    pub period: usize,
    pub on_field: Vec<i64>,
    pub bench: Vec<usize>,
    pub affinity_now: f64,
    pub goals_for_cum: i64,
    pub goals_against_cum: i64,
    pub positions: Vec<i64>,
}

/// The result of a single simulated match.
#[derive(Clone, Debug)]
pub struct MatchResult {
    pub goals_for: i64,
    pub goals_against: i64,
    pub goal_differential: i64,
    pub affinity_sum_deterministic: f64,
    pub per_period_affinity: Vec<f64>,
    pub goal_events: Vec<GoalEvent>,
    pub sub_events: Vec<SubEvent>,
    pub trace: Vec<MatchTraceFrame>,
}

/// `PureTransform<ProblemScheduleInput, MatchResult>`.
pub struct SimulateMatchDES {
    opts: MatchSimOptions,
}

impl SimulateMatchDES {
    pub fn new(opts: MatchSimOptions) -> Self {
        SimulateMatchDES { opts }
    }
}

impl<'a> Transform<ProblemScheduleInput<'a>, MatchResult> for SimulateMatchDES {
    fn transform(&self, input: ProblemScheduleInput<'a>) -> MatchResult {
        simulate_match_des(input.problem, input.schedule, &self.opts)
    }
}

/// Run a single DES match given a schedule; goal events from a thinned Poisson
/// process modulated by on-field average affinity.
pub fn simulate_match_des(
    problem: &SoccerProblem,
    schedule: &Schedule,
    opts: &MatchSimOptions,
) -> MatchResult {
    let minutes_per_period = opts.minutes_per_period.unwrap_or(20);
    let team_rate = opts.team_score_rate.unwrap_or(0.06);
    let opp_rate = opts.opp_score_rate.unwrap_or(0.04);
    let seed = opts.seed.unwrap_or(1);
    let mut rng = mulberry32(seed);
    let t_count = problem.num_periods;
    let total_minutes = minutes_per_period * t_count;
    let mut goals_for: i64 = 0;
    let mut goals_against: i64 = 0;
    let mut goal_events: Vec<GoalEvent> = Vec::new();
    let mut sub_events: Vec<SubEvent> = Vec::new();
    let mut trace: Vec<MatchTraceFrame> = Vec::new();
    let mut per_period_affinity = vec![0.0; t_count];

    let mut period_avg_aff: Vec<f64> = Vec::new();
    for t in 0..t_count {
        let mut s = 0.0;
        for pos in 0..problem.num_positions {
            let p = schedule.assignment[t][pos];
            s += problem.affinity[p as usize][pos][t];
        }
        period_avg_aff.push(s / problem.num_positions as f64);
        per_period_affinity[t] = s;
    }

    for minute in 0..total_minutes {
        let period = minute / minutes_per_period;
        let is_start_of_period = minute % minutes_per_period == 0;
        if is_start_of_period {
            sub_events.push(SubEvent {
                t: minute,
                period,
                new_on_field: schedule.assignment[period].clone(),
                new_bench: schedule.bench[period].clone(),
            });
        }
        let aff = period_avg_aff[period];
        let lambda_us = team_rate * aff.powi(2);
        let lambda_them = opp_rate * (1.6 - 0.5 * aff);
        if rng.next_float() < lambda_us {
            goals_for += 1;
            goal_events.push(GoalEvent {
                t: minute,
                side: GoalSide::Us,
            });
        }
        if rng.next_float() < lambda_them {
            goals_against += 1;
            goal_events.push(GoalEvent {
                t: minute,
                side: GoalSide::Them,
            });
        }
        trace.push(MatchTraceFrame {
            t: minute,
            period,
            on_field: schedule.assignment[period].clone(),
            bench: schedule.bench[period].clone(),
            affinity_now: aff,
            goals_for_cum: goals_for,
            goals_against_cum: goals_against,
            positions: schedule.assignment[period].clone(),
        });
    }

    MatchResult {
        goals_for,
        goals_against,
        goal_differential: goals_for - goals_against,
        affinity_sum_deterministic: per_period_affinity.iter().sum(),
        per_period_affinity,
        goal_events,
        sub_events,
        trace,
    }
}

/// Aggregate statistics over many simulated matches.
#[derive(Clone, Debug)]
pub struct MatchAggregate {
    pub policy_name: String,
    pub schedule: Schedule,
    pub affinity_sum_deterministic: f64,
    pub fairness_ok: bool,
    pub mean_goals_for: f64,
    pub mean_goals_against: f64,
    pub mean_goal_diff: f64,
    pub sd_goal_diff: f64,
    pub raw_goal_diffs: Vec<f64>,
    pub bench_counts: Vec<usize>,
}

/// Configuration for [`RunManyMatches`].
#[derive(Clone, Debug)]
pub struct RunManyMatchesConfig {
    pub policy_name: String,
    pub num_matches: usize,
    pub seed_base: u32,
    pub match_opts: Option<MatchSimOptions>,
}

/// `PureTransform<ProblemScheduleInput, MatchAggregate>`.
pub struct RunManyMatches {
    config: RunManyMatchesConfig,
}

impl RunManyMatches {
    pub fn new(config: RunManyMatchesConfig) -> Self {
        RunManyMatches { config }
    }
}

impl<'a> Transform<ProblemScheduleInput<'a>, MatchAggregate> for RunManyMatches {
    fn transform(&self, input: ProblemScheduleInput<'a>) -> MatchAggregate {
        let match_opts = self.config.match_opts.clone().unwrap_or_default();
        run_many_matches(
            input.problem,
            input.schedule,
            &self.config.policy_name,
            self.config.num_matches,
            self.config.seed_base,
            &match_opts,
        )
    }
}

/// Run N independent matches and aggregate.
pub fn run_many_matches(
    problem: &SoccerProblem,
    schedule: &Schedule,
    policy_name: &str,
    num_matches: usize,
    seed_base: u32,
    match_opts: &MatchSimOptions,
) -> MatchAggregate {
    let eval_res = evaluate_schedule(problem, schedule);
    let mut diffs: Vec<f64> = Vec::new();
    let mut g_f = 0.0;
    let mut g_a = 0.0;
    for n in 0..num_matches {
        let mut o = match_opts.clone();
        o.seed = Some(seed_base + n as u32);
        let r = simulate_match_des(problem, schedule, &o);
        diffs.push(r.goal_differential as f64);
        g_f += r.goals_for as f64 / num_matches as f64;
        g_a += r.goals_against as f64 / num_matches as f64;
    }
    let mean_diff = diffs.iter().sum::<f64>() / diffs.len() as f64;
    let sd_diff = (diffs.iter().map(|&b| (b - mean_diff).powi(2)).sum::<f64>()
        / 1.0_f64.max((diffs.len() as f64) - 1.0))
    .sqrt();
    MatchAggregate {
        policy_name: policy_name.to_string(),
        schedule: schedule.clone(),
        affinity_sum_deterministic: eval_res.affinity_sum,
        fairness_ok: eval_res.fairness_ok,
        mean_goals_for: g_f,
        mean_goals_against: g_a,
        mean_goal_diff: mean_diff,
        sd_goal_diff: sd_diff,
        raw_goal_diffs: diffs,
        bench_counts: eval_res.bench_counts,
    }
}

/// Input for Welch's t-test.
#[derive(Clone, Debug)]
pub struct WelchTInput {
    pub a: Vec<f64>,
    pub b: Vec<f64>,
}

/// `PureTransform<WelchTInput, number>`.
pub struct WelchT;

impl Transform<WelchTInput, f64> for WelchT {
    fn transform(&self, input: WelchTInput) -> f64 {
        welch_t(&input.a, &input.b)
    }
}

/// Welch's t-statistic for difference of means.
pub fn welch_t(a: &[f64], b: &[f64]) -> f64 {
    let ma = a.iter().sum::<f64>() / a.len() as f64;
    let mb = b.iter().sum::<f64>() / b.len() as f64;
    let va = a.iter().map(|&v| (v - ma).powi(2)).sum::<f64>() / 1.0_f64.max((a.len() as f64) - 1.0);
    let vb = b.iter().map(|&v| (v - mb).powi(2)).sum::<f64>() / 1.0_f64.max((b.len() as f64) - 1.0);
    (ma - mb) / (va / a.len() as f64 + vb / b.len() as f64 + 1e-30).sqrt()
}

#[cfg(test)]
mod tests {
    //! The MDP optimum dominates the greedy and random baselines on affinity and
    //! respects the no-consecutive-bench fairness constraint.

    use super::*;
    use crate::des::general::ip_mip_des::IPMIPStatus;
    use crate::des::general::lp::{solve_lp_internal, InternalSimplexOptions};

    fn problem() -> SoccerProblem {
        build_sample_soccer_problem(&AffinityBuilderOptions {
            seed: Some(7),
            ..Default::default()
        })
    }

    fn tiny_internal_solver_problem() -> SoccerProblem {
        SoccerProblem {
            num_players: 3,
            num_positions: 2,
            num_periods: 2,
            bench_size: 1,
            max_consecutive_on_field: None,
            min_contiguous_on_field: Some(vec![1, 1, 1]),
            max_contiguous_on_field: Some(vec![2, 2, 2]),
            max_contiguous_bench: Some(vec![2, 2, 2]),
            enforce_no_consecutive_bench: false,
            max_subs_per_game: Some(2),
            min_subs_per_game: Some(0),
            affinity: vec![
                vec![vec![1.0, 1.0], vec![0.1, 0.1]],
                vec![vec![0.1, 0.1], vec![1.0, 1.0]],
                vec![vec![0.7, 0.7], vec![0.7, 0.7]],
            ],
            player_names: Some(vec![
                "Keeper".to_string(),
                "Wing".to_string(),
                "Flex".to_string(),
            ]),
            position_names: Some(vec!["GK".to_string(), "RF".to_string()]),
            player_status: None,
            fixed_position: None,
            banned_positions: None,
            synergy_rules: None,
        }
    }

    #[test]
    fn sample_problem_dimensions() {
        let p = problem();
        assert_eq!(p.num_players, 12);
        assert_eq!(p.num_positions, 7);
        assert_eq!(p.num_periods, 4);
        assert_eq!(p.bench_size, 5);
        // The default builder leaves the stamina cap disabled so the historical
        // LP/IP row counts (and behaviour) are unchanged.
        assert_eq!(p.max_consecutive_on_field, None);
        for row in &p.affinity {
            for col in row {
                for &v in col {
                    assert!((0.0..=1.0).contains(&v));
                }
            }
        }
    }

    #[test]
    fn mdp_is_fair_and_dominates_baselines() {
        let p = problem();
        let mdp = policy_mdp_vi(&p);
        let mdp_schedule = mdp.to_schedule();
        assert!(validate_schedule_structure(&p, &mdp_schedule).is_none());
        let mdp_eval = evaluate_schedule(&p, &mdp_schedule);
        assert!(mdp_eval.fairness_ok, "MDP schedule must satisfy fairness");
        // The MDP's reported optimal value equals the realised affinity sum.
        assert!((mdp.optimal_value - mdp_eval.affinity_sum).abs() < 1e-6);

        // The exact MDP dominates the fairness-aware greedy heuristic (which is a
        // feasible schedule for the same fairness-constrained program).
        let greedy = policy_greedy_hungarian(&p, &GreedyHungarianOptions::default());
        assert!(evaluate_schedule(&p, &greedy).fairness_ok);
        let greedy_aff = evaluate_schedule(&p, &greedy).affinity_sum;
        assert!(
            mdp.optimal_value + 1e-9 >= greedy_aff,
            "mdp={} greedy={}",
            mdp.optimal_value,
            greedy_aff
        );

        // Dropping the fairness constraint (memoryless MDP) can only raise the value.
        let memoryless = policy_mdp_vi_memoryless(&p);
        assert!(memoryless.value + 1e-9 >= mdp.optimal_value);
    }

    #[test]
    fn greedy_schedule_is_well_formed() {
        let p = problem();
        let sched = policy_greedy_hungarian(&p, &GreedyHungarianOptions::default());
        assert!(validate_schedule_structure(&p, &sched).is_none());
    }

    #[test]
    fn soccer_lp_relaxation_solves_with_internal_simplex() {
        let p = tiny_internal_solver_problem();
        let lp = build_soccer_lp(&p);
        let sol = solve_lp_internal(
            &lp,
            &InternalSimplexOptions {
                max_iter: Some(200),
                ..Default::default()
            },
        );

        assert_eq!(sol.status, LPStatus::Optimal, "message={:?}", sol.message);
        assert_eq!(sol.solver, "internal");
        assert!((sol.objective - 4.0).abs() < 1e-6, "z={}", sol.objective);
    }

    #[test]
    fn soccer_lp_rounding_rejects_malformed_optimal_vectors() {
        assert!(soccer_lp_solution_is_roundable(&[0.0, 0.5, 1.0], 1.5, 3));
        assert!(!soccer_lp_solution_is_roundable(&[0.0, 0.5], 1.5, 3));
        assert!(!soccer_lp_solution_is_roundable(
            &[0.0, f64::NAN, 1.0],
            1.5,
            3
        ));
        assert!(!soccer_lp_solution_is_roundable(
            &[0.0, 0.5, 1.0],
            f64::INFINITY,
            3
        ));
    }

    #[test]
    fn soccer_ipmip_policy_uses_internal_branch_and_cut_only() {
        let p = tiny_internal_solver_problem();
        let result = policy_ipmip_feasible(
            &p,
            &SoccerIPMIPPolicyOptions {
                time_limit_ms: Some(5_000.0),
                max_nodes: Some(20),
                max_ticks: Some(500),
                lp_max_iters: Some(200),
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::InternalSimplex,
                )),
                max_cut_rounds: Some(0),
                fallback_to_mdp: Some(false),
                ..Default::default()
            },
        );

        assert_eq!(result.mip.status, IPMIPStatus::Optimal);
        assert!(!result.used_fallback, "reason={:?}", result.fallback_reason);
        assert_eq!(result.mip.solver_kind, "in-house-branch-and-cut");
        assert!(result.mip.in_house_only);
        assert!(!result.mip.uses_external_solvers);
        assert_eq!(
            result
                .mip
                .lp_algorithm_usage
                .get(&ConcreteLpRelaxationAlgorithm::InternalSimplex)
                .copied()
                .unwrap_or(0),
            result.mip.lp_solves as u64
        );
        assert!(
            result
                .mip
                .topology
                .iter()
                .any(|node| node.id == "ip-lp-relaxation"
                    && node.role.contains("stationary LP solver block")),
            "topology={:?}",
            result.mip.topology
        );
    }

    #[test]
    fn soccer_ipmip_model_exposes_movable_and_station_metadata_for_solver_tab() {
        let p = tiny_internal_solver_problem();
        let model = build_soccer_ipmip(&p);
        let variables = model.ip.variable_nodes.as_ref().expect("variable nodes");
        let constraints = model
            .ip
            .constraint_nodes
            .as_ref()
            .expect("constraint nodes");

        assert_eq!(
            variables.len(),
            p.num_players * p.num_positions * p.num_periods
        );
        assert!(variables.len() < model.ip.c.len());
        assert!(variables
            .iter()
            .any(|node| node.node_id.starts_with("movable:Keeper:")));
        assert!(constraints
            .iter()
            .any(|node| node.node_id.starts_with("station:eligibility:Keeper:")));
        assert!(constraints
            .iter()
            .any(|node| node.node_id.starts_with("station:position:GK:")));
    }

    #[test]
    fn match_sim_is_deterministic_for_seed() {
        let p = problem();
        let sched = policy_greedy_hungarian(&p, &GreedyHungarianOptions::default());
        let a = simulate_match_des(
            &p,
            &sched,
            &MatchSimOptions {
                seed: Some(123),
                ..Default::default()
            },
        );
        let b = simulate_match_des(
            &p,
            &sched,
            &MatchSimOptions {
                seed: Some(123),
                ..Default::default()
            },
        );
        assert_eq!(a.goals_for, b.goals_for);
        assert_eq!(a.goals_against, b.goals_against);
        assert_eq!(a.trace.len(), 4 * 20);
    }

    /// A hand-built schedule that parks player 0 on the field for periods 0–1
    /// trips the stamina detector when the cap is 1.
    #[test]
    fn detects_over_long_on_field_run() {
        let problem = SoccerProblem {
            num_players: 2,
            num_positions: 1,
            num_periods: 3,
            bench_size: 1,
            max_consecutive_on_field: Some(1),
            min_contiguous_on_field: None,
            max_contiguous_on_field: None,
            max_contiguous_bench: None,
            enforce_no_consecutive_bench: true,
            max_subs_per_game: None,
            min_subs_per_game: None,
            affinity: vec![vec![vec![0.0; 3]; 1]; 2],
            player_names: None,
            position_names: None,
            player_status: None,
            fixed_position: None,
            banned_positions: None,
            synergy_rules: None,
        };
        let schedule = Schedule {
            assignment: vec![vec![0], vec![0], vec![1]],
            bench: vec![vec![1], vec![1], vec![0]],
        };
        let v = consecutive_on_field_violations(&problem, &schedule);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].player_id, 0);
        assert_eq!(v[0].start_period, 0);
        assert_eq!(v[0].length, 2);
        // With no cap the same schedule is clean.
        let mut uncapped = problem.clone();
        uncapped.max_consecutive_on_field = None;
        assert!(consecutive_on_field_violations(&uncapped, &schedule).is_empty());
    }

    #[test]
    fn max_bench_blocks_allow_repeated_short_bench_stints() {
        let mut problem = SoccerProblem {
            num_players: 2,
            num_positions: 1,
            num_periods: 8,
            bench_size: 1,
            max_consecutive_on_field: None,
            min_contiguous_on_field: None,
            max_contiguous_on_field: None,
            max_contiguous_bench: Some(vec![2, 2]),
            enforce_no_consecutive_bench: false,
            max_subs_per_game: None,
            min_subs_per_game: None,
            affinity: vec![vec![vec![0.0; 8]; 1]; 2],
            player_names: None,
            position_names: None,
            player_status: None,
            fixed_position: None,
            banned_positions: None,
            synergy_rules: None,
        };
        let repeated_short_stints = Schedule {
            assignment: vec![
                vec![0],
                vec![1],
                vec![1],
                vec![0],
                vec![0],
                vec![1],
                vec![1],
                vec![0],
            ],
            bench: vec![
                vec![1],
                vec![0],
                vec![0],
                vec![1],
                vec![1],
                vec![0],
                vec![0],
                vec![1],
            ],
        };
        assert!(max_contiguous_bench_violations(&problem, &repeated_short_stints).is_empty());

        problem.max_contiguous_bench = Some(vec![1, 1]);
        let violations = max_contiguous_bench_violations(&problem, &repeated_short_stints);
        assert_eq!(violations.len(), 3);
        assert!(violations.iter().all(|v| v.length == 2));
    }

    /// Turning on the stamina cap adds exactly one rolling window row per player
    /// (for the default 4-period instance and a cap of 3) to both the LP and the
    /// IP, and the solved IP/MIP incumbent respects the cap.
    #[test]
    fn stamina_cap_constrains_lp_ip_and_solution() {
        let base = build_sample_soccer_problem(&AffinityBuilderOptions {
            seed: Some(1234),
            ..Default::default()
        });
        let capped = build_sample_soccer_problem(&AffinityBuilderOptions {
            seed: Some(1234),
            max_consecutive_on_field: Some(3),
            ..Default::default()
        });

        // 4 periods, cap 3 => one window [T1..T4] per player => +num_players rows.
        let lp_base = build_soccer_lp(&base);
        let lp_capped = build_soccer_lp(&capped);
        assert_eq!(
            lp_capped.a_ub.as_ref().unwrap().len(),
            lp_base.a_ub.as_ref().unwrap().len() + capped.num_players
        );
        let ip_base = build_soccer_ipmip(&base);
        let ip_capped = build_soccer_ipmip(&capped);
        assert_eq!(
            ip_capped.ip.a.len(),
            ip_base.ip.a.len() + capped.num_players
        );

        let opts = SoccerIPMIPPolicyOptions {
            time_limit_ms: Some(15_000.0),
            max_nodes: Some(400),
            max_ticks: Some(4_000),
            lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                ConcreteLpRelaxationAlgorithm::InternalSimplex,
            )),
            max_cut_rounds: Some(0),
            fallback_to_mdp: Some(false),
            ..Default::default()
        };
        let res = policy_ipmip_feasible(&capped, &opts);
        assert!(!res.used_fallback, "reason={:?}", res.fallback_reason);
        let eval = evaluate_schedule(&capped, &res.schedule);
        assert!(eval.fairness_ok);
        assert!(
            eval.stamina_ok,
            "stamina violations: {:?}",
            eval.consecutive_on_field_violations
        );
    }

    /// The roster size is configurable while staying 7-a-side: 11 and 13 players
    /// shrink/grow only the bench.
    #[test]
    fn roster_size_is_configurable() {
        for (players, bench) in [(11usize, 4usize), (13usize, 6usize)] {
            let p = build_sample_soccer_problem(&AffinityBuilderOptions {
                num_players: Some(players),
                max_consecutive_on_field: Some(3),
                seed: Some(7),
                ..Default::default()
            });
            assert_eq!(p.num_players, players);
            assert_eq!(p.num_positions, 7);
            assert_eq!(p.bench_size, bench);
            let sched = policy_greedy_hungarian(&p, &GreedyHungarianOptions::default());
            assert!(validate_schedule_structure(&p, &sched).is_none());
        }
    }
}
