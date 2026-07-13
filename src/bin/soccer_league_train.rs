//! soccer_league_train — continuous LEAGUE-PLAY neural trainer (breaks the self-play plateau).
//!
//! The self-play learner converges to a fixed point (both teams share one policy → it stops
//! improving; weights inflate without behavioural change). This trainer instead trains the
//! FRONTIER neural net against a LEAGUE of FROZEN opponents — the archived past champions (real
//! past selves, NOT genetic mutants) plus a few fresh brains for exploration/diversity — using
//! the engine's asymmetric per-team learning (`EngineMatchRunner`: the learning side takes
//! gradient steps, the frozen side just plays its net). It carries the trained brain forward,
//! applies WEIGHT DECAY each round (stops the l2-norm inflation that saturates activations and
//! kills gradients — likely a direct cause of the plateau), snapshots temporal checkpoints into
//! the league for growing diversity, and publishes the compact learned-params.json that the
//! local stack (:5055, champion-gate, eval) reads only after the checkpoint gate accepts it.
//! NO genetic algorithm, NO RDS. By default this is the anti-parity climb lane: neural-authoritative
//! ranking, stronger critic targets, actor-critic exploration, and analytic-opponent fixtures so the
//! gradient is "beat the engine" rather than "mirror self-play".
//!
//! Env: SOCCER_LEAGUE_FRONTIER (path, default /tmp/neural-climb-local/learned-params.json)
//!      SOCCER_LEAGUE_ARCHIVE  (dir,  default /tmp/neural-climb-local/champions)
//!      SOCCER_LEAGUE_GAMES_PER_OPP (usize, default 2)
//!      SOCCER_LEAGUE_MINUTES  (f64, default 2.0)
//!      SOCCER_LEAGUE_WEIGHT_DECAY (f64 per round, default 0.01)
//!      SOCCER_LEAGUE_FRESH_OPPONENTS (usize, default 2)
//!      SOCCER_LEAGUE_CHECKPOINT_EVERY_ROUNDS (usize, default 10 — add frontier to league)

use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::time::Instant;

use soccer_engine::des::general::soccer::{
    average_soccer_neural_network_snapshots, learned_mpc_objective_enabled, FacingBucket,
    MatchSummary, SoccerNeuralBlendMode, SoccerNeuralNetworkSnapshot, SoccerQStateKey,
    SoccerQTargetEntry, TacticalPhase, Team,
};
use soccer_engine::des::general::tournament::{
    EngineMatchRunner, EngineMatchRunnerConfig, TeamBrain, TournamentMatchContext,
    TournamentMatchRunner, TournamentStage,
};

const MIN_REWARD_WEIGHT: f64 = 0.0001;

fn env_str(k: &str, d: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| d.to_string())
}
fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}
fn env_usize_list(k: &str) -> Vec<usize> {
    let mut values: Vec<usize> = std::env::var(k)
        .ok()
        .into_iter()
        .flat_map(|raw| {
            raw.split(|ch: char| ch == ',' || ch == ';' || ch.is_whitespace())
                .filter_map(|part| part.trim().parse::<usize>().ok())
                .filter(|value| *value > 0)
                .collect::<Vec<_>>()
        })
        .collect();
    values.sort_unstable();
    values.dedup();
    values
}
fn env_f64(k: &str, d: f64) -> f64 {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}
fn env_f64_any(keys: &[&str], default: f64) -> f64 {
    keys.iter()
        .find_map(|key| std::env::var(key).ok().and_then(|v| v.parse().ok()))
        .unwrap_or(default)
}
fn env_bool(k: &str, d: bool) -> bool {
    std::env::var(k)
        .ok()
        .map(|v| matches!(v.trim(), "1" | "true" | "TRUE" | "yes" | "on"))
        .unwrap_or(d)
}

fn env_default_bool(k: &str, d: bool) -> bool {
    let enabled = env_bool(k, d);
    if std::env::var(k).is_err() {
        std::env::set_var(k, if enabled { "1" } else { "0" });
    }
    enabled
}

fn env_default_usize(k: &str, d: usize) -> usize {
    let value = env_usize(k, d);
    if std::env::var(k).is_err() {
        std::env::set_var(k, value.to_string());
    }
    value
}

fn env_default_f64(k: &str, d: f64) -> f64 {
    let value = env_f64(k, d);
    if std::env::var(k).is_err() {
        std::env::set_var(k, value.to_string());
    }
    value
}

fn env_reward_f64(k: &str, d: f64, min: f64, max: f64) -> f64 {
    let min = min.max(MIN_REWARD_WEIGHT);
    let max = max.max(min);
    let default = if d.is_finite() { d } else { min }.clamp(min, max);
    let value = std::env::var(k)
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite())
        .unwrap_or(default)
        .clamp(min, max);
    if std::env::var(k).is_err() {
        std::env::set_var(k, value.to_string());
    }
    value
}

fn env_neural_blend_mode(k: &str, default: SoccerNeuralBlendMode) -> SoccerNeuralBlendMode {
    let Some(raw) = std::env::var(k).ok() else {
        return default;
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "off" | "disabled" | "none" => SoccerNeuralBlendMode::Off,
        "additive" | "add" => SoccerNeuralBlendMode::Additive,
        "tiebreak" | "tie" | "tie-break" | "tie_break" => SoccerNeuralBlendMode::TieBreak,
        "confidence" | "confidencegated" | "confidence-gated" | "gated" => {
            SoccerNeuralBlendMode::ConfidenceGated
        }
        "authoritative" | "neural" | "neural-authoritative" | "neural_authoritative" => {
            SoccerNeuralBlendMode::Authoritative
        }
        _ => {
            eprintln!("[league] invalid {k}={raw:?}; using {:?}", default);
            default
        }
    }
}

fn apply_league_neural_mcts_config(config: &mut EngineMatchRunnerConfig) -> bool {
    let enabled = env_bool("SOCCER_LEAGUE_NEURAL_MCTS_ENABLED", false);
    config.base.neural_blend.mcts_enabled = enabled;
    enabled
}

fn league_worker_count(requested_workers: usize, fixture_count: usize) -> usize {
    let capped = requested_workers.clamp(1, fixture_count.max(1));
    let serial_for_mpc_objective =
        learned_mpc_objective_enabled() && env_bool("SOCCER_LEAGUE_MPC_OBJECTIVE_SERIAL", true);
    if serial_for_mpc_objective {
        1
    } else {
        capped
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct LeagueMatchKpis {
    goal_diff: i32,
    goals_for: u32,
    goals_against: u32,
    shots_for: u32,
    shots_against: u32,
    shots_on_target_for: u32,
    shots_on_target_against: u32,
    shots_after_pass_for: u32,
    passes_attempted_for: u32,
    passes_completed_for: u32,
    forward_passes_for: u32,
    forward_passes_against: u32,
    forward_pass_margin: i32,
    pass_turnovers_for: u32,
    pass_turnovers_against: u32,
    net_forward_passes_for: i32,
    net_forward_passes_against: i32,
    net_forward_pass_margin: i32,
    backward_passes_for: u32,
    dribble_beats_for: u32,
    assists_for: u32,
    crosses_completed_for: u32,
    pass_chain_gain_yards_for: f64,
    pass_chain_net_losses_for: u32,
    learned_policy_option_decisions: u32,
    learned_policy_multi_option_decisions: u32,
    learned_policy_legal_action_options: u32,
    learned_mpc_replans: u32,
    learned_mpc_replans_mpc: u32,
    learned_mpc_replans_option_score_safety: u32,
    learned_mpc_replans_neural_mcts: u32,
    objective_fitness: f64,
    objective_fitness_margin: f64,
}

impl LeagueMatchKpis {
    fn from_summary(summary: &MatchSummary, frontier_team: Team) -> Self {
        let stats = &summary.stats;
        let home_net_forward =
            stats.passes_completed_forward_home as i32 - stats.interceptions_away as i32;
        let away_net_forward =
            stats.passes_completed_forward_away as i32 - stats.interceptions_home as i32;
        match frontier_team {
            Team::Home => LeagueMatchKpis {
                goal_diff: summary.score_home as i32 - summary.score_away as i32,
                goals_for: summary.score_home,
                goals_against: summary.score_away,
                shots_for: stats.shots_home,
                shots_against: stats.shots_away,
                shots_on_target_for: stats.shots_on_target_home,
                shots_on_target_against: stats.shots_on_target_away,
                shots_after_pass_for: stats.shots_after_pass_home,
                passes_attempted_for: stats.passes_attempted_home,
                passes_completed_for: stats.passes_completed_home,
                forward_passes_for: stats.passes_completed_forward_home,
                forward_passes_against: stats.passes_completed_forward_away,
                forward_pass_margin: stats.passes_completed_forward_home as i32
                    - stats.passes_completed_forward_away as i32,
                pass_turnovers_for: stats.interceptions_away,
                pass_turnovers_against: stats.interceptions_home,
                net_forward_passes_for: home_net_forward,
                net_forward_passes_against: away_net_forward,
                net_forward_pass_margin: home_net_forward - away_net_forward,
                backward_passes_for: stats.passes_completed_backward_home,
                dribble_beats_for: stats.dribble_beats_home,
                assists_for: stats.assists_home,
                crosses_completed_for: stats.crosses_completed_home,
                pass_chain_gain_yards_for: stats.pass_chain_gain_yards_home,
                pass_chain_net_losses_for: stats.pass_chains_net_loss_home,
                learned_policy_option_decisions: stats.learned_policy_option_decisions,
                learned_policy_multi_option_decisions: stats.learned_policy_multi_option_decisions,
                learned_policy_legal_action_options: stats.learned_policy_legal_action_options,
                learned_mpc_replans: stats.learned_mpc_replans,
                learned_mpc_replans_mpc: stats.learned_mpc_replans_mpc,
                learned_mpc_replans_option_score_safety: stats
                    .learned_mpc_replans_option_score_safety,
                learned_mpc_replans_neural_mcts: stats.learned_mpc_replans_neural_mcts,
                objective_fitness: home_net_forward as f64,
                objective_fitness_margin: (home_net_forward - away_net_forward) as f64,
            },
            Team::Away => LeagueMatchKpis {
                goal_diff: summary.score_away as i32 - summary.score_home as i32,
                goals_for: summary.score_away,
                goals_against: summary.score_home,
                shots_for: stats.shots_away,
                shots_against: stats.shots_home,
                shots_on_target_for: stats.shots_on_target_away,
                shots_on_target_against: stats.shots_on_target_home,
                shots_after_pass_for: stats.shots_after_pass_away,
                passes_attempted_for: stats.passes_attempted_away,
                passes_completed_for: stats.passes_completed_away,
                forward_passes_for: stats.passes_completed_forward_away,
                forward_passes_against: stats.passes_completed_forward_home,
                forward_pass_margin: stats.passes_completed_forward_away as i32
                    - stats.passes_completed_forward_home as i32,
                pass_turnovers_for: stats.interceptions_home,
                pass_turnovers_against: stats.interceptions_away,
                net_forward_passes_for: away_net_forward,
                net_forward_passes_against: home_net_forward,
                net_forward_pass_margin: away_net_forward - home_net_forward,
                backward_passes_for: stats.passes_completed_backward_away,
                dribble_beats_for: stats.dribble_beats_away,
                assists_for: stats.assists_away,
                crosses_completed_for: stats.crosses_completed_away,
                pass_chain_gain_yards_for: stats.pass_chain_gain_yards_away,
                pass_chain_net_losses_for: stats.pass_chains_net_loss_away,
                learned_policy_option_decisions: stats.learned_policy_option_decisions,
                learned_policy_multi_option_decisions: stats.learned_policy_multi_option_decisions,
                learned_policy_legal_action_options: stats.learned_policy_legal_action_options,
                learned_mpc_replans: stats.learned_mpc_replans,
                learned_mpc_replans_mpc: stats.learned_mpc_replans_mpc,
                learned_mpc_replans_option_score_safety: stats
                    .learned_mpc_replans_option_score_safety,
                learned_mpc_replans_neural_mcts: stats.learned_mpc_replans_neural_mcts,
                objective_fitness: away_net_forward as f64,
                objective_fitness_margin: (away_net_forward - home_net_forward) as f64,
            },
        }
    }
}

#[derive(Clone, Debug, Default)]
struct LeagueRoundKpis {
    games: usize,
    goal_diff: i32,
    goals_for: u32,
    goals_against: u32,
    shots_for: u32,
    shots_against: u32,
    shots_on_target_for: u32,
    shots_on_target_against: u32,
    shots_after_pass_for: u32,
    passes_attempted_for: u32,
    passes_completed_for: u32,
    forward_passes_for: u32,
    forward_passes_against: u32,
    forward_pass_margin: i32,
    pass_turnovers_for: u32,
    pass_turnovers_against: u32,
    net_forward_passes_for: i32,
    net_forward_passes_against: i32,
    net_forward_pass_margin: i32,
    backward_passes_for: u32,
    dribble_beats_for: u32,
    assists_for: u32,
    crosses_completed_for: u32,
    pass_chain_gain_yards_for: f64,
    pass_chain_net_losses_for: u32,
    learned_policy_option_decisions: u32,
    learned_policy_multi_option_decisions: u32,
    learned_policy_legal_action_options: u32,
    learned_mpc_replans: u32,
    learned_mpc_replans_mpc: u32,
    learned_mpc_replans_option_score_safety: u32,
    learned_mpc_replans_neural_mcts: u32,
    objective_fitness_sum: f64,
    objective_fitness_margin_sum: f64,
}

impl LeagueRoundKpis {
    fn add(&mut self, kpis: LeagueMatchKpis) {
        self.games += 1;
        self.goal_diff += kpis.goal_diff;
        self.goals_for = self.goals_for.saturating_add(kpis.goals_for);
        self.goals_against = self.goals_against.saturating_add(kpis.goals_against);
        self.shots_for = self.shots_for.saturating_add(kpis.shots_for);
        self.shots_against = self.shots_against.saturating_add(kpis.shots_against);
        self.shots_on_target_for = self
            .shots_on_target_for
            .saturating_add(kpis.shots_on_target_for);
        self.shots_on_target_against = self
            .shots_on_target_against
            .saturating_add(kpis.shots_on_target_against);
        self.shots_after_pass_for = self
            .shots_after_pass_for
            .saturating_add(kpis.shots_after_pass_for);
        self.passes_attempted_for = self
            .passes_attempted_for
            .saturating_add(kpis.passes_attempted_for);
        self.passes_completed_for = self
            .passes_completed_for
            .saturating_add(kpis.passes_completed_for);
        self.forward_passes_for = self
            .forward_passes_for
            .saturating_add(kpis.forward_passes_for);
        self.forward_passes_against = self
            .forward_passes_against
            .saturating_add(kpis.forward_passes_against);
        self.forward_pass_margin += kpis.forward_pass_margin;
        self.pass_turnovers_for = self
            .pass_turnovers_for
            .saturating_add(kpis.pass_turnovers_for);
        self.pass_turnovers_against = self
            .pass_turnovers_against
            .saturating_add(kpis.pass_turnovers_against);
        self.net_forward_passes_for += kpis.net_forward_passes_for;
        self.net_forward_passes_against += kpis.net_forward_passes_against;
        self.net_forward_pass_margin += kpis.net_forward_pass_margin;
        self.backward_passes_for = self
            .backward_passes_for
            .saturating_add(kpis.backward_passes_for);
        self.dribble_beats_for = self
            .dribble_beats_for
            .saturating_add(kpis.dribble_beats_for);
        self.assists_for = self.assists_for.saturating_add(kpis.assists_for);
        self.crosses_completed_for = self
            .crosses_completed_for
            .saturating_add(kpis.crosses_completed_for);
        self.pass_chain_gain_yards_for += kpis.pass_chain_gain_yards_for;
        self.pass_chain_net_losses_for = self
            .pass_chain_net_losses_for
            .saturating_add(kpis.pass_chain_net_losses_for);
        self.learned_policy_option_decisions = self
            .learned_policy_option_decisions
            .saturating_add(kpis.learned_policy_option_decisions);
        self.learned_policy_multi_option_decisions = self
            .learned_policy_multi_option_decisions
            .saturating_add(kpis.learned_policy_multi_option_decisions);
        self.learned_policy_legal_action_options = self
            .learned_policy_legal_action_options
            .saturating_add(kpis.learned_policy_legal_action_options);
        self.learned_mpc_replans = self
            .learned_mpc_replans
            .saturating_add(kpis.learned_mpc_replans);
        self.learned_mpc_replans_mpc = self
            .learned_mpc_replans_mpc
            .saturating_add(kpis.learned_mpc_replans_mpc);
        self.learned_mpc_replans_option_score_safety = self
            .learned_mpc_replans_option_score_safety
            .saturating_add(kpis.learned_mpc_replans_option_score_safety);
        self.learned_mpc_replans_neural_mcts = self
            .learned_mpc_replans_neural_mcts
            .saturating_add(kpis.learned_mpc_replans_neural_mcts);
        self.objective_fitness_sum += kpis.objective_fitness;
        self.objective_fitness_margin_sum += kpis.objective_fitness_margin;
    }

    fn mean_objective_fitness(&self) -> f64 {
        if self.games == 0 {
            0.0
        } else {
            self.objective_fitness_sum / self.games as f64
        }
    }

    fn mean_objective_fitness_margin(&self) -> f64 {
        if self.games == 0 {
            0.0
        } else {
            self.objective_fitness_margin_sum / self.games as f64
        }
    }

    fn mean_forward_pass_margin(&self) -> f64 {
        if self.games == 0 {
            0.0
        } else {
            self.forward_pass_margin as f64 / self.games as f64
        }
    }

    fn mean_net_forward_pass_margin(&self) -> f64 {
        if self.games == 0 {
            0.0
        } else {
            self.net_forward_pass_margin as f64 / self.games as f64
        }
    }

    fn mean_goal_diff(&self) -> f64 {
        if self.games == 0 {
            0.0
        } else {
            self.goal_diff as f64 / self.games as f64
        }
    }
}

/// Recompute parameter_count + l2_norm after mutating weights (the persistence validator
/// and the value-blend warmth check both read these).
fn recompute_norm(s: &mut SoccerNeuralNetworkSnapshot) {
    let mut count = 0usize;
    let mut sumsq = 0f64;
    for layer in &s.layers {
        for row in &layer.weights {
            for w in row {
                count += 1;
                sumsq += w * w;
            }
        }
        for b in &layer.biases {
            count += 1;
            sumsq += b * b;
        }
    }
    s.parameter_count = count;
    s.l2_norm = sumsq.sqrt();
}

/// L2 weight decay: pull weights toward 0 by `lambda` each round. Counteracts the unbounded
/// norm growth that saturates activations (vanishing gradients = plateau).
fn apply_weight_decay(s: &mut SoccerNeuralNetworkSnapshot, lambda: f64) {
    if !(lambda > 0.0 && lambda < 1.0) {
        return;
    }
    let k = 1.0 - lambda;
    for layer in &mut s.layers {
        for row in &mut layer.weights {
            for w in row.iter_mut() {
                *w *= k;
            }
        }
        for b in &mut layer.biases {
            *b *= k;
        }
    }
    recompute_norm(s);
}

/// Tiny deterministic PRNG (xorshift64*) + Box-Muller gaussian, so the warm-restart perturbation
/// is reproducible from (round, training_steps) without pulling in a rand dependency.
struct WarmRng(u64);
impl WarmRng {
    fn unit(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        ((self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)) >> 11) as f64 / ((1u64 << 53) as f64)
    }
    fn gauss(&mut self) -> f64 {
        let u1 = self.unit().max(1e-12);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// Warm-restart weight perturbation: add annealed Gaussian noise to a random fraction of the
/// weights — an explicit "kick" out of the current basin (SGDR / simulated-annealing style). The
/// next round's gradient then re-descends from the perturbed point, so the net can escape a local
/// optimum instead of just settling deeper into parity. Env-gated (scale 0 = off).
fn perturb_weights(s: &mut SoccerNeuralNetworkSnapshot, scale: f64, frac: f64, seed: u64) {
    if !(scale > 0.0) || !(frac > 0.0) {
        return;
    }
    let mut rng = WarmRng(seed | 1);
    for layer in &mut s.layers {
        for row in &mut layer.weights {
            for w in row.iter_mut() {
                if rng.unit() < frac {
                    *w += scale * rng.gauss();
                }
            }
        }
        for b in &mut layer.biases {
            if rng.unit() < frac {
                *b += scale * rng.gauss();
            }
        }
    }
    recompute_norm(s);
}

type TargetEntryKey = (SoccerQStateKey, String, usize, usize, usize, usize, i32);

const LEAGUE_TARGET_FINE_COLUMNS: usize = 12;
const LEAGUE_TARGET_FINE_ROWS: usize = 24;
const LEAGUE_TARGET_TACTICAL_COLUMNS: usize = 6;
const LEAGUE_TARGET_TACTICAL_ROWS: usize = 8;
const LEAGUE_TARGET_MACRO_COLUMNS: usize = 3;
const LEAGUE_TARGET_MACRO_ROWS: usize = 4;

fn target_entry_key(entry: &SoccerQTargetEntry) -> TargetEntryKey {
    (
        entry.state.clone(),
        entry.action.trim().to_string(),
        entry.target_fine_cell_id,
        entry.target_tactical_cell_id,
        entry.target_macro_cell_id,
        entry.target_root_cell_id,
        entry.receiver_descriptor,
    )
}

fn target_entry_order(a: &SoccerQTargetEntry, b: &SoccerQTargetEntry) -> std::cmp::Ordering {
    b.visits.cmp(&a.visits).then_with(|| {
        b.value
            .abs()
            .partial_cmp(&a.value.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    })
}

fn merge_target_entries(
    entry_sets: impl IntoIterator<Item = Vec<SoccerQTargetEntry>>,
    max_entries: usize,
) -> Vec<SoccerQTargetEntry> {
    merge_target_entries_inner(entry_sets, max_entries, false)
}

fn merge_target_entries_idempotent(
    entry_sets: impl IntoIterator<Item = Vec<SoccerQTargetEntry>>,
    max_entries: usize,
) -> Vec<SoccerQTargetEntry> {
    merge_target_entries_inner(entry_sets, max_entries, true)
}

fn same_target_entry_payload(a: &SoccerQTargetEntry, b: &SoccerQTargetEntry) -> bool {
    a.visits == b.visits && a.value.to_bits() == b.value.to_bits()
}

fn merge_target_entries_inner(
    entry_sets: impl IntoIterator<Item = Vec<SoccerQTargetEntry>>,
    max_entries: usize,
    idempotent_duplicates: bool,
) -> Vec<SoccerQTargetEntry> {
    let mut by_key: HashMap<TargetEntryKey, SoccerQTargetEntry> = HashMap::new();
    for entries in entry_sets {
        for entry in entries {
            if !entry.value.is_finite() || entry.action.trim().is_empty() {
                continue;
            }
            let key = target_entry_key(&entry);
            by_key
                .entry(key)
                .and_modify(|existing| {
                    if idempotent_duplicates && same_target_entry_payload(existing, &entry) {
                        return;
                    }
                    let existing_visits = existing.visits.max(1) as f64;
                    let entry_visits = entry.visits.max(1) as f64;
                    let total_visits = existing_visits + entry_visits;
                    existing.value = (existing.value * existing_visits
                        + entry.value * entry_visits)
                        / total_visits;
                    existing.visits = existing.visits.saturating_add(entry.visits);
                })
                .or_insert(entry);
        }
    }
    let mut merged = by_key.into_values().collect::<Vec<_>>();
    merged.sort_by(target_entry_order);
    if max_entries > 0 && merged.len() > max_entries {
        merged.truncate(max_entries);
    }
    merged
}

fn rotate_pitch_cell_180(cell_id: usize, columns: usize, rows: usize) -> usize {
    if columns == 0 || rows == 0 {
        return cell_id;
    }
    let cell_count = columns.saturating_mul(rows);
    if cell_id >= cell_count {
        return cell_id;
    }
    let x = cell_id % columns;
    let y = cell_id / columns;
    (rows - 1 - y) * columns + (columns - 1 - x)
}

fn rotate_axis_index_180(index: usize, len: usize) -> usize {
    if len == 0 || index >= len {
        index
    } else {
        len - 1 - index
    }
}

fn rotate_tactical_phase_180(phase: TacticalPhase) -> TacticalPhase {
    match phase {
        TacticalPhase::HomeBuildUp => TacticalPhase::AwayBuildUp,
        TacticalPhase::AwayBuildUp => TacticalPhase::HomeBuildUp,
        TacticalPhase::HomeAttack => TacticalPhase::AwayAttack,
        TacticalPhase::AwayAttack => TacticalPhase::HomeAttack,
        other => other,
    }
}

fn rotate_facing_180(facing: FacingBucket) -> FacingBucket {
    match facing {
        FacingBucket::North => FacingBucket::South,
        FacingBucket::NorthEast => FacingBucket::SouthWest,
        FacingBucket::East => FacingBucket::West,
        FacingBucket::SouthEast => FacingBucket::NorthWest,
        FacingBucket::South => FacingBucket::North,
        FacingBucket::SouthWest => FacingBucket::NorthEast,
        FacingBucket::West => FacingBucket::East,
        FacingBucket::NorthWest => FacingBucket::SouthEast,
        other => other,
    }
}

fn rotate_target_state_180(mut state: SoccerQStateKey) -> SoccerQStateKey {
    state.phase = rotate_tactical_phase_180(state.phase);
    state.ball_zone_x = rotate_axis_index_180(state.ball_zone_x, LEAGUE_TARGET_TACTICAL_COLUMNS);
    state.ball_zone_y = rotate_axis_index_180(state.ball_zone_y, LEAGUE_TARGET_TACTICAL_ROWS);
    state.ball_fine_lane =
        rotate_axis_index_180(state.ball_fine_lane as usize, LEAGUE_TARGET_FINE_COLUMNS) as u8;
    state.ball_fine_row =
        rotate_axis_index_180(state.ball_fine_row as usize, LEAGUE_TARGET_FINE_ROWS) as u8;
    state.receive_facing = rotate_facing_180(state.receive_facing);
    state.action_facing = rotate_facing_180(state.action_facing);
    state
}

fn rotate_target_entry_180(entry: &SoccerQTargetEntry) -> SoccerQTargetEntry {
    let mut rotated = entry.clone();
    rotated.state = rotate_target_state_180(rotated.state);
    rotated.target_fine_cell_id = rotate_pitch_cell_180(
        rotated.target_fine_cell_id,
        LEAGUE_TARGET_FINE_COLUMNS,
        LEAGUE_TARGET_FINE_ROWS,
    );
    rotated.target_tactical_cell_id = rotate_pitch_cell_180(
        rotated.target_tactical_cell_id,
        LEAGUE_TARGET_TACTICAL_COLUMNS,
        LEAGUE_TARGET_TACTICAL_ROWS,
    );
    rotated.target_macro_cell_id = rotate_pitch_cell_180(
        rotated.target_macro_cell_id,
        LEAGUE_TARGET_MACRO_COLUMNS,
        LEAGUE_TARGET_MACRO_ROWS,
    );
    rotated
}

fn rotated_target_entries_180(entries: &[SoccerQTargetEntry]) -> Vec<SoccerQTargetEntry> {
    entries.iter().map(rotate_target_entry_180).collect()
}

fn with_bidirectional_target_entries(brain: &TeamBrain) -> TeamBrain {
    let mut next = brain.clone();
    let rotated_away_to_home = rotated_target_entries_180(&brain.away_target_entries);
    let rotated_home_to_away = rotated_target_entries_180(&brain.home_target_entries);
    next.home_target_entries = merge_target_entries_idempotent(
        [brain.home_target_entries.clone(), rotated_away_to_home],
        0,
    );
    next.away_target_entries = merge_target_entries_idempotent(
        [brain.away_target_entries.clone(), rotated_home_to_away],
        0,
    );
    next
}

fn load_target_entries(value: &serde_json::Value, key: &str) -> Vec<SoccerQTargetEntry> {
    value
        .get(key)
        .cloned()
        .and_then(|entries| serde_json::from_value(entries).ok())
        .unwrap_or_default()
}

fn neural_snapshot_from_value(value: &serde_json::Value) -> Option<SoccerNeuralNetworkSnapshot> {
    value
        .get("neuralNetwork")
        .cloned()
        .or_else(|| value.get("learning")?.get("neuralNetwork").cloned())
        .and_then(|neural| serde_json::from_value(neural).ok())
}

fn neural_sidecar_path(path: &str) -> Option<String> {
    path.strip_suffix(".json")
        .map(|stem| format!("{stem}.neural.json"))
}

fn load_neural_snapshot(
    path: &str,
    value: &serde_json::Value,
) -> Option<SoccerNeuralNetworkSnapshot> {
    if let Some(neural) = neural_snapshot_from_value(value) {
        return Some(neural);
    }
    let sidecar = neural_sidecar_path(path)?;
    let raw = fs::read_to_string(&sidecar).ok()?;
    let sidecar_value: serde_json::Value = serde_json::from_str(&raw).ok()?;
    neural_snapshot_from_value(&sidecar_value)
        .or_else(|| serde_json::from_value::<SoccerNeuralNetworkSnapshot>(sidecar_value).ok())
}

/// Load the frontier brain from a learned-params.json. The neural snapshot remains the learner,
/// while side-specific target-Q lets the brain turn learned action labels into concrete targets.
fn load_brain(path: &str) -> Option<TeamBrain> {
    let raw = fs::read_to_string(path).ok()?;
    let v: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let neural = load_neural_snapshot(path, &v)?;
    let home_targets = load_target_entries(&v, "homeTargetEntries");
    let away_targets = load_target_entries(&v, "awayTargetEntries");
    Some(TeamBrain::from_snapshot_with_targets(
        neural,
        home_targets,
        away_targets,
    ))
}

/// Write the frontier brain into a COMPACT learned-params.json. Action-Q stays empty because it
/// bloats; target-Q is compact enough and is required for learned concrete receiver/space choices.
fn write_frontier(path: &str, brain: &TeamBrain) -> std::io::Result<()> {
    let Some(snap) = brain.neural.as_ref() else {
        return Ok(());
    };
    if let Some(parent) = Path::new(path)
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let v = serde_json::json!({
        "version": 0,
        "homeEntries": [],
        "homeTargetEntries": brain.home_target_entries,
        "awayEntries": [],
        "awayTargetEntries": brain.away_target_entries,
        "episodes": 0,
        "neuralNetwork": serde_json::to_value(snap).unwrap(),
    });
    if Path::new(path).exists() {
        let prev = path.trim_end_matches(".json").to_string() + ".prev.json";
        let _ = fs::copy(path, prev);
    }
    let tmp = format!("{path}.tmp");
    fs::write(
        &tmp,
        serde_json::to_string(&v)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?,
    )?;
    fs::rename(&tmp, path)?;
    Ok(())
}

fn candidate_frontier_path(frontier_path: &str) -> String {
    let path = Path::new(frontier_path);
    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.is_empty())
        .unwrap_or("learned-params");
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .filter(|extension| !extension.is_empty())
        .map(|extension| format!(".{extension}"))
        .unwrap_or_default();
    path.with_file_name(format!("{stem}.candidate{extension}"))
        .to_string_lossy()
        .into_owned()
}

fn step_bucket_archive_path(
    step_archive_dir: &str,
    bucket: usize,
    round: u32,
    training_steps: usize,
) -> String {
    format!(
        "{}/league-step-b{:08}-r{:04}-s{}.json",
        step_archive_dir.trim_end_matches('/'),
        bucket,
        round,
        training_steps
    )
}

/// Load the frozen champion league from the archive dir (each is a learned-params.json).
fn load_league(dir: &str) -> Vec<TeamBrain> {
    let mut out = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        let mut paths: Vec<_> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().map(|x| x == "json").unwrap_or(false))
            .collect();
        paths.sort();
        for p in paths {
            if let Some(brain) = load_brain(&p.display().to_string()) {
                out.push(brain);
            }
        }
    }
    out
}

fn without_target_q(mut brain: TeamBrain) -> TeamBrain {
    brain.home_target_entries.clear();
    brain.away_target_entries.clear();
    brain
}

fn fixed_analytic_anchor() -> TeamBrain {
    let mut brain = without_target_q(TeamBrain::fresh_with_seed(0xA5A5_0007, 100));
    brain.neural = None;
    brain
}

fn play_and_carry(
    runner: &mut EngineMatchRunner,
    frontier: TeamBrain,
    opponent: &TeamBrain,
    seed: u32,
    frontier_home: bool,
    opp_id: usize,
) -> (TeamBrain, Option<LeagueMatchKpis>) {
    // frontier LEARNS; opponent is FROZEN. Alternate sides so the edge isn't a home artifact.
    let (home_id, away_id, home_learns, away_learns) = if frontier_home {
        (0usize, opp_id, true, false)
    } else {
        (opp_id, 0usize, false, true)
    };
    let ctx = TournamentMatchContext {
        stage: TournamentStage::Group,
        round_index: 0,
        match_index: 0,
        seed,
        home_id,
        away_id,
        home_name: format!("team{home_id}"),
        away_name: format!("team{away_id}"),
        home_learns,
        away_learns,
    };
    let frontier = with_bidirectional_target_entries(&frontier);
    let opponent = with_bidirectional_target_entries(opponent);
    let (home, away) = if frontier_home {
        (&frontier, &opponent)
    } else {
        (&opponent, &frontier)
    };
    match runner.play(&ctx, home, away) {
        Ok(o) => {
            let frontier_team = if frontier_home {
                Team::Home
            } else {
                Team::Away
            };
            let trained = if frontier_home {
                o.home_brain
            } else {
                o.away_brain
            };
            (
                trained,
                Some(LeagueMatchKpis::from_summary(&o.summary, frontier_team)),
            )
        }
        Err(e) => {
            eprintln!("league_match_error: {e}");
            (frontier, None)
        }
    }
}

fn play_checkpoint_validation(
    runner: &EngineMatchRunner,
    candidate: &TeamBrain,
    baseline: &TeamBrain,
    label: &str,
    round: u32,
    games: usize,
) -> LeagueRoundKpis {
    let mut validation = LeagueRoundKpis::default();
    let mut runner = runner.clone();
    println!("league_checkpoint_validation_start label={label} round={round} games={games}");
    for g in 0..games {
        let candidate_home = g % 2 == 0;
        let seed = 0xC1A0_0000u32
            .wrapping_add(round.wrapping_mul(1_000_003))
            .wrapping_add((g as u32).wrapping_mul(2_246_822_519));
        let (home_id, away_id, home, away, candidate_team) = if candidate_home {
            (0usize, 1usize, candidate, baseline, Team::Home)
        } else {
            (1usize, 0usize, baseline, candidate, Team::Away)
        };
        let ctx = TournamentMatchContext {
            stage: TournamentStage::Group,
            round_index: round as usize,
            match_index: g,
            seed,
            home_id,
            away_id,
            home_name: format!("team{home_id}"),
            away_name: format!("team{away_id}"),
            home_learns: false,
            away_learns: false,
        };
        match std::panic::catch_unwind(AssertUnwindSafe(|| runner.play(&ctx, home, away))) {
            Ok(Ok(outcome)) => {
                let kpis = LeagueMatchKpis::from_summary(&outcome.summary, candidate_team);
                println!(
                    "league_checkpoint_validation_game label={label} round={round} game={} candidate_home={} score={}-{} forward_pass_margin={} net_forward_pass_margin={}",
                    g,
                    candidate_home,
                    outcome.home_goals,
                    outcome.away_goals,
                    kpis.forward_pass_margin,
                    kpis.net_forward_pass_margin,
                );
                validation.add(kpis);
            }
            Ok(Err(err)) => {
                eprintln!("league_checkpoint_validation_error label={label} round={round} game={g}: {err}");
            }
            Err(_) => {
                eprintln!(
                    "league_checkpoint_validation_panic label={label} round={round} game={g}"
                );
            }
        }
    }
    validation
}

fn analytic_anchor_non_regression_passes(
    candidate: Option<&LeagueRoundKpis>,
    incumbent: Option<&LeagueRoundKpis>,
    expected_games: usize,
    max_goal_diff_regression: f64,
) -> bool {
    match (candidate, incumbent) {
        (Some(candidate), Some(incumbent)) => {
            candidate.games == expected_games
                && incumbent.games == expected_games
                && candidate.mean_goal_diff() + max_goal_diff_regression
                    >= incumbent.mean_goal_diff()
        }
        (None, None) => true,
        _ => false,
    }
}

fn checkpoint_validation_passes(
    validation: Option<&LeagueRoundKpis>,
    expected_games: usize,
    min_forward_pass_margin: f64,
    min_net_forward_pass_margin: f64,
    min_goal_diff_margin: f64,
) -> bool {
    match validation {
        Some(validation) => {
            if validation.games != expected_games {
                return false;
            }
            let advancement_passes = validation.mean_forward_pass_margin()
                > min_forward_pass_margin
                && validation.mean_net_forward_pass_margin() > min_net_forward_pass_margin;
            let outcome_passes = validation.mean_goal_diff() > min_goal_diff_margin.max(0.0);
            (advancement_passes && validation.mean_goal_diff() >= 0.0) || outcome_passes
        }
        None => true,
    }
}

fn main() {
    let frontier_path = env_str(
        "SOCCER_LEAGUE_FRONTIER",
        "/tmp/neural-climb-local/learned-params.json",
    );
    let candidate_frontier_default = candidate_frontier_path(&frontier_path);
    let candidate_frontier_path = env_str(
        "SOCCER_LEAGUE_CANDIDATE_FRONTIER",
        &candidate_frontier_default,
    );
    let archive_dir = env_str("SOCCER_LEAGUE_ARCHIVE", "/tmp/neural-climb-local/champions");
    let games_per_opp = env_usize("SOCCER_LEAGUE_GAMES_PER_OPP", 2).max(1);
    let minutes = env_f64("SOCCER_LEAGUE_MINUTES", 2.0);
    let weight_decay = env_f64("SOCCER_LEAGUE_WEIGHT_DECAY", 0.01);
    let fresh_opponents = env_usize("SOCCER_LEAGUE_FRESH_OPPONENTS", 2);
    let checkpoint_every = env_usize("SOCCER_LEAGUE_CHECKPOINT_EVERY_ROUNDS", 10).max(1);
    let max_target_entries_per_side = env_usize("SOCCER_LEAGUE_MAX_TARGET_ENTRIES_PER_SIDE", 4096);
    let checkpoint_require_forward_pass_climb =
        env_bool("SOCCER_LEAGUE_CHECKPOINT_REQUIRE_FORWARD_PASS_CLIMB", true);
    let checkpoint_max_forward_pass_regression =
        env_f64("SOCCER_LEAGUE_CHECKPOINT_MAX_FORWARD_PASS_REGRESSION", 0.0).max(0.0);
    let checkpoint_min_forward_pass_margin =
        env_f64("SOCCER_LEAGUE_CHECKPOINT_MIN_FORWARD_PASS_MARGIN", 0.0);
    let checkpoint_validate_games = env_usize("SOCCER_LEAGUE_CHECKPOINT_VALIDATE_GAMES", 8);
    let checkpoint_validate_min_forward_pass_margin = env_f64(
        "SOCCER_LEAGUE_CHECKPOINT_VALIDATE_MIN_FORWARD_PASS_MARGIN",
        0.0,
    );
    let checkpoint_validate_min_net_forward_pass_margin = env_f64(
        "SOCCER_LEAGUE_CHECKPOINT_VALIDATE_MIN_NET_FORWARD_PASS_MARGIN",
        env_f64(
            "SOCCER_LEAGUE_CHECKPOINT_VALIDATE_MIN_OBJECTIVE_MARGIN",
            0.0,
        ),
    );
    let checkpoint_validate_min_goal_diff_margin = env_f64(
        "SOCCER_LEAGUE_CHECKPOINT_VALIDATE_MIN_GOAL_DIFF_MARGIN",
        0.0,
    );
    let fixed_analytic_training_anchor =
        env_bool("SOCCER_LEAGUE_FIXED_ANALYTIC_TRAINING_ANCHOR", true);
    let analytic_anchor_gate = env_bool("SOCCER_LEAGUE_ANALYTIC_ANCHOR_GATE", true);
    let analytic_anchor_validate_games = if analytic_anchor_gate {
        env_usize(
            "SOCCER_LEAGUE_ANALYTIC_ANCHOR_VALIDATE_GAMES",
            checkpoint_validate_games.max(4),
        )
    } else {
        0
    };
    let analytic_anchor_max_goal_diff_regression = env_f64(
        "SOCCER_LEAGUE_ANALYTIC_ANCHOR_MAX_GOAL_DIFF_REGRESSION",
        0.0,
    )
    .max(0.0);
    let max_rounds = env_usize("SOCCER_LEAGUE_MAX_ROUNDS", 0);
    let max_training_steps = env_usize("SOCCER_LEAGUE_MAX_TRAINING_STEPS", 0);
    let step_archive_buckets = env_usize_list("SOCCER_LEAGUE_STEP_ARCHIVE_BUCKETS");
    let step_archive_dir_default = format!("{}/step-buckets", archive_dir.trim_end_matches('/'));
    let step_archive_dir = env_str("SOCCER_LEAGUE_STEP_ARCHIVE_DIR", &step_archive_dir_default);
    let uniform_elite_players = env_default_bool("SOCCER_ENGINE_UNIFORM_ELITE_PLAYERS", true);
    let seed_varied_skills = env_default_bool("SOCCER_SEED_VARIED_SKILLS", !uniform_elite_players);
    let authoritative_lambda = env_default_f64("DD_SOCCER_NEURAL_AUTHORITATIVE_LAMBDA", 8.0);
    let policy_entropy_coeff = env_default_f64("SOCCER_POLICY_ENTROPY_COEFF", 0.04);
    let attack_canonical_features =
        env_default_bool("DD_SOCCER_ENABLE_ATTACK_CANONICAL_NEURAL_FEATURES", true);
    let ball_zone_tactical_scale =
        env_default_bool("DD_SOCCER_ENABLE_BALL_ZONE_TACTICAL_SCALE", true);
    let player_grid_xy_features =
        env_default_bool("DD_SOCCER_ENABLE_PLAYER_GRID_XY_FEATURES", true);
    env_default_bool("DD_SOCCER_ENABLE_FORWARD_RELEASE_ROOT_CANDIDATE", true);
    env_default_f64("DD_SOCCER_FORWARD_RELEASE_MIN_QUALITY", 0.35);
    env_default_f64("DD_SOCCER_FORWARD_RELEASE_MIN_COMPLETION", 0.35);
    env_default_bool("DD_SOCCER_ENABLE_FORWARD_SELECT_LOGIT", true);
    let forward_select_logit_weight = env_default_f64("DD_SOCCER_FORWARD_SELECT_LOGIT_WEIGHT", 2.0);
    let finishing_select_bonus_weight = env_reward_f64(
        "SOCCER_NEURAL_FINISHING_SELECT_BONUS_WEIGHT",
        0.70,
        0.0,
        2.5,
    );
    let finishing_selection_floor =
        env_default_f64("SOCCER_NEURAL_FINISHING_SELECTION_FLOOR", 0.30);
    let finishing_selection_max_regression = env_default_f64(
        "SOCCER_NEURAL_FINISHING_SELECTION_MAX_SCORE_REGRESSION",
        12.0,
    );
    let finishing_selection_shot_drought_boost =
        env_default_f64("SOCCER_NEURAL_FINISHING_SELECTION_SHOT_DROUGHT_BOOST", 0.0);
    let finishing_selection_max_effective_floor = env_default_f64(
        "SOCCER_NEURAL_FINISHING_SELECTION_MAX_EFFECTIVE_FLOOR",
        0.52,
    );
    let forward_creation_selection_floor =
        env_default_f64("SOCCER_NEURAL_FORWARD_CREATION_SELECTION_FLOOR", 0.22);
    let forward_creation_selection_max_regression = env_default_f64(
        "SOCCER_NEURAL_FORWARD_CREATION_SELECTION_MAX_SCORE_REGRESSION",
        18.0,
    );
    let forward_creation_min_quality =
        env_default_f64("SOCCER_NEURAL_FORWARD_CREATION_MIN_QUALITY", 0.50);
    let forward_creation_min_completion =
        env_default_f64("SOCCER_NEURAL_FORWARD_CREATION_MIN_COMPLETION", 0.45);
    let forward_creation_min_forward_yards =
        env_default_f64("SOCCER_NEURAL_FORWARD_CREATION_MIN_FORWARD_YARDS", 4.0);
    let forward_creation_bypass_min_quality =
        env_default_f64("SOCCER_NEURAL_FORWARD_CREATION_BYPASS_MIN_QUALITY", 0.70);
    let forward_creation_bypass_min_completion =
        env_default_f64("SOCCER_NEURAL_FORWARD_CREATION_BYPASS_MIN_COMPLETION", 0.48);
    let planner_teacher_missed_opportunity =
        env_default_bool("DD_SOCCER_ENABLE_PLANNER_TEACHER_MISSED_OPPORTUNITY", true);
    let planner_teacher_advantage_floor =
        env_default_f64("SOCCER_PLANNER_TEACHER_ADVANTAGE_FLOOR", 0.045);
    let planner_teacher_advantage_max =
        env_default_f64("SOCCER_PLANNER_TEACHER_ADVANTAGE_MAX", 0.30);
    let planner_teacher_weight = env_reward_f64("SOCCER_PLANNER_TEACHER_WEIGHT", 1.8, 0.0, 4.0);
    let planner_teacher_min_score_share =
        env_default_f64("SOCCER_PLANNER_TEACHER_MIN_SCORE_SHARE", 0.006);
    let planner_teacher_min_shot_quality =
        env_default_f64("SOCCER_PLANNER_TEACHER_MIN_SHOT_QUALITY", 0.52);
    let planner_teacher_min_forward_quality =
        env_default_f64("SOCCER_PLANNER_TEACHER_MIN_FORWARD_QUALITY", 0.50);
    let planner_teacher_min_forward_completion =
        env_default_f64("SOCCER_PLANNER_TEACHER_MIN_FORWARD_COMPLETION", 0.38);
    let planner_teacher_max_samples_per_decision =
        env_default_usize("SOCCER_PLANNER_TEACHER_MAX_SAMPLES_PER_DECISION", 2);
    let planner_teacher_include_same_pass_family =
        env_default_bool("DD_SOCCER_PLANNER_TEACHER_INCLUDE_SAME_PASS_FAMILY", true);
    let planner_teacher_same_pass_min_margin_share =
        env_default_f64("SOCCER_PLANNER_TEACHER_SAME_PASS_MIN_MARGIN_SHARE", 0.015);
    env_default_usize("SOCCER_NEURAL_MCTS_PASS_TARGET_CANDIDATES", 6);
    env_default_bool("SOCCER_DYNAMIC_REWARD_WEIGHTS", true);
    env_default_bool("DD_SOCCER_ENABLE_FIELD_VECTOR_SHOT_REWARD", true);
    env_default_bool("DD_SOCCER_ENABLE_OVERLOAD_WEIGHTED_PROGRESSION", true);
    let goal_reward_scale = env_reward_f64("DD_SOCCER_GOAL_REWARD_SCALE", 1.0, 1.0, 16.0);
    let forward_pass_reward_scale =
        env_reward_f64("DD_SOCCER_FORWARD_PASS_REWARD_SCALE", 6.25, 0.0, 20.0);
    let pass_chain_reward_scale =
        env_reward_f64("DD_SOCCER_PASS_CHAIN_REWARD_SCALE", 4.25, 0.0, 20.0);
    let pass_turnover_penalty_scale =
        env_reward_f64("DD_SOCCER_PASS_TURNOVER_PENALTY_SCALE", 5.0, 1.0, 20.0);
    let quick_forward_release_reward_scale = env_reward_f64(
        "DD_SOCCER_QUICK_FORWARD_RELEASE_REWARD_SCALE",
        1.25,
        0.0,
        2.0,
    );
    let shot_shaping_reward_scale =
        env_reward_f64("DD_SOCCER_SHOT_SHAPING_REWARD_SCALE", 0.85, 0.0, 4.0);
    let shot_commitment_reward_scale =
        env_reward_f64("DD_SOCCER_SHOT_COMMITMENT_REWARD_SCALE", 0.0, 0.0, 10.0);
    let dribble_beat_reward_scale =
        env_reward_f64("DD_SOCCER_DRIBBLE_BEAT_REWARD_SCALE", 1.0, 0.0, 8.0);
    let learned_epv_reward_scale =
        env_reward_f64("DD_SOCCER_LEARNED_EPV_REWARD_SCALE", 1.0, 0.0, 200.0);
    let offball_support_reward_scale =
        env_reward_f64("SOCCER_OFFBALL_SUPPORT_REWARD_SCALE", 1.0, 0.0, 10.0);
    let carry_reward_scale = env_reward_f64("DD_SOCCER_CARRY_REWARD_SCALE", 1.0, 0.0, 10.0);
    let recovery_reward_scale = env_reward_f64("DD_SOCCER_RECOVERY_REWARD_SCALE", 1.0, 0.0, 10.0);
    let overload_progression_reward_per_yard = env_reward_f64(
        "DD_SOCCER_OVERLOAD_PROGRESSION_REWARD_PER_YARD",
        0.10,
        0.0,
        0.5,
    );
    let analytic_difference_reward_negative_dense_floor = env_reward_f64(
        "SOCCER_ANALYTIC_DIFFERENCE_REWARD_NEGATIVE_DENSE_FLOOR",
        0.15,
        0.0,
        1.0,
    );
    let analytic_difference_reward_negative_dense_scale = env_reward_f64(
        "SOCCER_ANALYTIC_DIFFERENCE_REWARD_NEGATIVE_DENSE_SCALE",
        1.0,
        0.0,
        25.0,
    );
    let mappo_team_reward_share = env_reward_f64("SOCCER_MAPPO_TEAM_REWARD_SHARE", 0.50, 0.0, 1.0);
    let marl_team_reward_weight = env_reward_f64("SOCCER_MARL_TEAM_REWARD_WEIGHT", 0.35, 0.0, 1.0);
    let marl_intermediate_reward_weight =
        env_reward_f64("SOCCER_MARL_INTERMEDIATE_REWARD_WEIGHT", 1.0, 0.2, 2.0);
    let chance_quality_reward = env_default_bool("DD_SOCCER_ENABLE_CHANCE_QUALITY_REWARD", true);
    let chance_quality_composite =
        env_default_bool("DD_SOCCER_ENABLE_CHANCE_QUALITY_COMPOSITE", true);
    let chance_quality_k = env_default_f64("DD_SOCCER_CHANCE_QUALITY_K", 22.0);
    let chance_quality_cap = env_default_f64("DD_SOCCER_CHANCE_QUALITY_CAP", 5.0);
    let analytic_opponents = env_bool("SOCCER_LEAGUE_ANALYTIC_OPPONENTS", true);
    // SELF-PLAY LADDER (mirrors the standalone 5v5 champion ladder): train the frontier against the
    // CURRENT published champion (a neural net that strengthens as the ladder climbs) plus an analytic
    // baseline, and promote it to the new champion only when it beats that champion by a goal-diff
    // margin over the held-out head-to-head eval. The co-evolution the plain league lacks.
    let self_play_ladder = env_bool("SOCCER_LEAGUE_SELF_PLAY_LADDER", false);
    let self_play_promote_margin = env_f64("SOCCER_LEAGUE_PROMOTE_MARGIN", 0.25);

    let mut runner_config = EngineMatchRunnerConfig::default();
    runner_config.base.duration_seconds = minutes * 60.0;
    let period_count = env_default_usize("SOCCER_LEAGUE_PERIOD_COUNT", 1).clamp(1, 4);
    runner_config.base.period_count = period_count;
    // The runner's Inline backend is step-STARVED by default (~10 gradient steps/game),
    // so the value net is hopelessly undertrained (net-negative vs the base engine even
    // after many rounds). Crank the per-game training volume — replay + train-every-tick —
    // so each game does thousands of updates (like the self-play threaded config did).
    runner_config.base.neural_learning.train_every_ticks =
        env_usize("SOCCER_LEAGUE_TRAIN_EVERY_TICKS", 1).max(1);
    runner_config.base.neural_learning.max_batches_per_tick =
        env_usize("SOCCER_LEAGUE_MAX_BATCHES_PER_TICK", 4).max(1);
    runner_config.base.neural_learning.replay_capacity =
        env_usize("SOCCER_LEAGUE_REPLAY_CAPACITY", 8192).max(1);
    runner_config.base.neural_learning.replay_samples_per_tick =
        env_usize("SOCCER_LEAGUE_REPLAY_SAMPLES_PER_TICK", 128).max(1);
    runner_config.base.neural_learning.batch_size =
        env_usize("SOCCER_LEAGUE_BATCH_SIZE", 64).max(1);
    // Keep the league default conservative; the ratchet harness can sweep LR candidates per seed
    // family, but a long league run has only one active learner unless the operator overrides it.
    runner_config.base.neural_learning.learning_rate = env_f64_any(
        &["SOCCER_LEAGUE_LEARNING_RATE", "SOCCER_NEURAL_LEARNING_RATE"],
        0.015,
    );
    runner_config.base.neural_learning.optimizer_momentum = env_f64(
        "SOCCER_NEURAL_OPTIMIZER_MOMENTUM",
        runner_config.base.neural_learning.optimizer_momentum,
    );
    // Network capacity: bigger hidden layer for more expressive value function over the
    // 610-dim field vector (the 24-unit net plateaued at parity with the analytic engine).
    // Only takes effect on a FRESH net — a resumed snapshot keeps its own architecture.
    runner_config.base.neural_learning.hidden_units =
        env_usize("SOCCER_LEAGUE_HIDDEN_UNITS", 96).max(1);
    // Critic-target window (un-crush A/B). The critic target is `(reward + gamma*max_next) /
    // target_scale` clamped to +/-target_clip. With the defaults (30/3) the +/-200 win and +160
    // goal terminal rewards BOTH saturate the +3 rail, so the value head cannot rank win vs goal
    // vs a big win — the advantage baseline is blind above the rail. Raising `target_scale` (e.g.
    // 120) drops those labels below the clip so terminal outcomes separate again; `target_clip`
    // is exposed too for completeness. Unset ⇒ the config default (byte-identical).
    runner_config.base.neural_learning.target_scale = env_f64("SOCCER_NEURAL_TARGET_SCALE", 120.0);
    runner_config.base.neural_learning.target_clip = env_f64(
        "SOCCER_NEURAL_TARGET_CLIP",
        runner_config.base.neural_learning.target_clip,
    );
    // Team-reward share (MAPPO): the individual forward-pass reward is blended with the
    // team-averaged reward at this weight — the learnable rest-defense / coordination lever.
    // League training previously left this at the config default; expose it for A/Bs.
    runner_config.base.neural_learning.mappo_team_reward_share = mappo_team_reward_share;
    runner_config.base.neural_learning.marl_team_reward_weight = marl_team_reward_weight;
    runner_config
        .base
        .neural_learning
        .marl_intermediate_reward_weight = marl_intermediate_reward_weight;
    eprintln!(
        "[league] mappo_team_reward_share={} marl_team_reward_weight={} marl_intermediate_reward_weight={}",
        runner_config.base.neural_learning.mappo_team_reward_share,
        runner_config.base.neural_learning.marl_team_reward_weight,
        runner_config
            .base
            .neural_learning
            .marl_intermediate_reward_weight
    );
    // Un-collapse the value head on fresh nets: honour SOCCER_NEURAL_TARGET_POPART here (the
    // league bin otherwise leaves target-PopArt at the config default = off, unlike the
    // learning-run bins). A collapsed value can't rank candidates, which would make the Part-B
    // action-space A/B uninterpretable. Only takes effect on a FRESH net (same as hidden_units).
    runner_config.base.neural_learning.target_popart_enabled =
        std::env::var("SOCCER_NEURAL_TARGET_POPART")
            .ok()
            .map(|v| matches!(v.trim(), "1" | "true" | "TRUE" | "yes" | "on"))
            .unwrap_or(true);
    eprintln!(
        "[league] critic-target window: target_scale={} target_clip={} (raw-reward window +/-{}) popart={}",
        runner_config.base.neural_learning.target_scale,
        runner_config.base.neural_learning.target_clip,
        runner_config.base.neural_learning.target_scale
            * runner_config.base.neural_learning.target_clip,
        runner_config.base.neural_learning.target_popart_enabled,
    );
    let target_standardization_enabled =
        env_default_bool("DD_SOCCER_ENABLE_TARGET_STANDARDIZATION", true);
    let mc_critic_target_enabled = env_default_bool("DD_SOCCER_ENABLE_MC_CRITIC_TARGET", true);
    let neural_self_bootstrap_enabled =
        env_default_bool("DD_SOCCER_ENABLE_NEURAL_SELF_BOOTSTRAP", true);
    let maxa_bootstrap_enabled = env_default_bool("DD_SOCCER_ENABLE_MAXA_BOOTSTRAP", true);
    let novelty_bonus_enabled = env_default_bool("DD_SOCCER_ENABLE_NOVELTY_BONUS", true);
    let forward_pass_climb_curriculum_enabled =
        env_default_bool("DD_SOCCER_FORWARD_PASS_CLIMB_CURRICULUM", true);
    let fast_actor_reward_fallback_enabled =
        env_default_bool("DD_SOCCER_ENABLE_FAST_ACTOR_REWARD_FALLBACK", true);
    let pass_outcome_priority_learning_enabled =
        env_default_bool("DD_SOCCER_ENABLE_PASS_OUTCOME_PRIORITY_LEARNING", true);
    let dp_bootstrap_enabled = env_default_bool(
        "DD_SOCCER_ENABLE_DP_BOOTSTRAP",
        forward_pass_climb_curriculum_enabled,
    );
    let dp_bootstrap_horizon = env_default_usize(
        "DD_SOCCER_DP_BOOTSTRAP_HORIZON",
        if dp_bootstrap_enabled { 64 } else { 16 },
    );
    let dp_bootstrap_sweeps = env_default_usize(
        "DD_SOCCER_DP_BOOTSTRAP_SWEEPS",
        if dp_bootstrap_enabled { 200 } else { 40 },
    );
    let hermetic_neural = env_default_bool("SOCCER_LEAGUE_HERMETIC_NEURAL", true);
    let policy_train_max_transitions_per_tick = env_usize(
        "SOCCER_LEAGUE_POLICY_TRAIN_MAX_TRANSITIONS_PER_TICK",
        if hermetic_neural {
            0
        } else {
            runner_config.base.policy_train_max_transitions_per_tick
        },
    );
    runner_config.base.policy_train_max_transitions_per_tick =
        policy_train_max_transitions_per_tick;
    runner_config.base.full_game_learning_enabled =
        env_bool("SOCCER_LEAGUE_FULL_GAME_LEARNING", true);
    runner_config.base.neural_learning.lp_coupling_enabled =
        env_bool("SOCCER_NEURAL_LP_COUPLING_ENABLED", true);
    // POLICY-IMPROVEMENT LEVER: opt the fresh net into the actor-critic path. When on, the neural
    // ACTOR π(family|s) is trained by advantage policy-gradient and biases action selection on top
    // of the value blend — the "close the policy-improvement loop" escape from value-imitation
    // parity (policy gradient optimizes RETURN directly, not tabular-matching). Pair with a lower
    // DD_SOCCER_NEURAL_AUTHORITATIVE_LAMBDA (more actor influence) + SOCCER_POLICY_ENTROPY_COEFF
    // (keep exploring) so the actor can actually leave the analytic mode. Off ⇒ current behaviour.
    runner_config.base.neural_blend.actor_critic = env_bool("SOCCER_NEURAL_ACTOR_CRITIC", true);
    // The self-play ladder installs a neural net on BOTH teams, but the actor's `policy_head` is a
    // single shared field — the away champion's actor would overwrite (and then drive) both sides.
    // Force critic-only ranking in ladder mode so each side selects with its OWN per-team critic
    // (the exact mode play_checkpoint_validation + the default tournament config already use).
    if self_play_ladder {
        runner_config.base.neural_blend.actor_critic = false;
    }
    runner_config.base.neural_blend.mode = env_neural_blend_mode(
        "SOCCER_NEURAL_BLEND_MODE",
        SoccerNeuralBlendMode::Authoritative,
    );
    runner_config.base.neural_blend.lambda =
        env_f64("SOCCER_NEURAL_BLEND_LAMBDA", authoritative_lambda.max(0.5));
    runner_config.base.neural_blend.warmup_steps =
        env_usize("SOCCER_NEURAL_BLEND_WARMUP_STEPS", 1).max(1);
    runner_config.base.neural_blend.candidates =
        env_usize("SOCCER_NEURAL_BLEND_CANDIDATES", 8).clamp(2, 16);
    let league_neural_mcts_enabled = apply_league_neural_mcts_config(&mut runner_config);
    let mpc_tier2_enabled = runner_config.base.mpc.tier2_player_enabled;
    let mpc_reconcile_enabled = runner_config.base.mpc.reconcile_enabled;
    let mpc_field_aware_enabled = runner_config.base.mpc.field_aware_enabled;
    let mpc_latent_objective_enabled = runner_config.base.mpc.latent_objective_enabled;
    let local_mpc_enabled = runner_config.base.local_mpc_enabled;
    let actor_critic_enabled = runner_config.base.neural_blend.actor_critic;
    let lp_coupling_enabled = runner_config.base.neural_learning.lp_coupling_enabled;
    let neural_blend_mode = runner_config.base.neural_blend.mode;
    let neural_blend_lambda = runner_config.base.neural_blend.lambda;
    let neural_blend_candidates = runner_config.base.neural_blend.candidates;
    let neural_blend_warmup_steps = runner_config.base.neural_blend.warmup_steps;
    let full_game_learning_enabled = runner_config.base.full_game_learning_enabled;
    let neural_learning_rate = runner_config.base.neural_learning.learning_rate;
    let neural_optimizer_momentum = runner_config.base.neural_learning.optimizer_momentum;
    // Keep the engine's designed independent-brain mode (per-team critic drives each side).
    let mut runner = EngineMatchRunner::new(runner_config);

    let mut frontier = match load_brain(&frontier_path) {
        Some(brain) => {
            let (l2, params) = brain
                .neural
                .as_ref()
                .map(|s| (s.l2_norm, s.parameter_count))
                .unwrap_or((0.0, 0));
            println!(
                "league_resume frontier l2={l2:.3} params={params} home_targets={} away_targets={} from {}",
                brain.home_target_entries.len(),
                brain.away_target_entries.len(),
                frontier_path
            );
            brain
        }
        None => {
            println!("league_fresh_start (no frontier at {frontier_path})");
            TeamBrain::fresh_with_seed(0xC0FFEE, 0)
        }
    };
    if hermetic_neural {
        frontier = without_target_q(frontier);
    }

    // Seed base env-overridable for independent-seed-family replication (default byte-identical).
    let train_seed_base: u32 = std::env::var("SOCCER_LEAGUE_SEED")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(0x5EED_0000);
    let mut round = 0u32;
    let mut best_checkpoint_net_forward_pass_margin = f64::NEG_INFINITY;
    let mut archived_step_buckets: BTreeSet<usize> = BTreeSet::new();
    println!(
        "league_train_started_at_utc={} games/opp={} minutes={} period_count={} weight_decay={} fresh_opp={} checkpoint_every={} max_rounds={} max_training_steps={} max_target_entries_per_side={} advancement_metric=completed_forward_passes net_metric=completed_forward_passes_minus_turnovers analytic_opponents={} uniform_elite_players={} seed_varied_skills={} hermetic_neural={} attack_canonical_features={} ball_zone_tactical_scale={} player_grid_xy_features={} policy_train_max_transitions_per_tick={} full_game_learning_enabled={} neural_learning_rate={} neural_optimizer_momentum={} neural_blend_mode={:?} neural_blend_lambda={} neural_authoritative_lambda={} neural_blend_candidates={} neural_blend_warmup_steps={} policy_entropy_coeff={} forward_select_logit_weight={} finishing_select_bonus_weight={} finishing_selection_floor={} finishing_selection_max_regression={} finishing_selection_shot_drought_boost={finishing_selection_shot_drought_boost} finishing_selection_max_effective_floor={finishing_selection_max_effective_floor} forward_creation_selection_floor={forward_creation_selection_floor} forward_creation_selection_max_regression={forward_creation_selection_max_regression} forward_creation_min_quality={forward_creation_min_quality} forward_creation_min_completion={forward_creation_min_completion} forward_creation_min_forward_yards={forward_creation_min_forward_yards} forward_creation_bypass_min_quality={forward_creation_bypass_min_quality} forward_creation_bypass_min_completion={forward_creation_bypass_min_completion} planner_teacher_missed_opportunity={planner_teacher_missed_opportunity} planner_teacher_advantage_floor={planner_teacher_advantage_floor:.3} planner_teacher_advantage_max={planner_teacher_advantage_max:.3} planner_teacher_weight={planner_teacher_weight:.2} planner_teacher_min_score_share={planner_teacher_min_score_share:.3} planner_teacher_min_shot_quality={planner_teacher_min_shot_quality:.2} planner_teacher_min_forward_quality={planner_teacher_min_forward_quality:.2} planner_teacher_min_forward_completion={planner_teacher_min_forward_completion:.2} planner_teacher_max_samples_per_decision={planner_teacher_max_samples_per_decision} planner_teacher_include_same_pass_family={planner_teacher_include_same_pass_family} planner_teacher_same_pass_min_margin_share={planner_teacher_same_pass_min_margin_share:.3} chance_quality_reward={} chance_quality_composite={} chance_quality_k={} chance_quality_cap={} league_neural_mcts_enabled={} actor_critic_enabled={} lp_coupling_enabled={} target_standardization_enabled={} mc_critic_target_enabled={} neural_self_bootstrap_enabled={} maxa_bootstrap_enabled={} novelty_bonus_enabled={} fast_actor_reward_fallback_enabled={} pass_outcome_priority_learning_enabled={} forward_pass_climb_curriculum_enabled={} dp_bootstrap_enabled={} dp_bootstrap_horizon={} dp_bootstrap_sweeps={} mpc_tier2_enabled={} mpc_reconcile_enabled={} mpc_field_aware_enabled={} mpc_latent_objective_enabled={} local_mpc_enabled={} checkpoint_require_forward_pass_climb={} checkpoint_max_forward_pass_regression={} checkpoint_min_forward_pass_margin={} checkpoint_validate_games={} checkpoint_validate_min_forward_pass_margin={} checkpoint_validate_min_net_forward_pass_margin={} checkpoint_validate_min_goal_diff_margin={} frontier={} candidate_frontier={} archive={} step_archive_dir={} step_archive_buckets={:?}",
        chrono_now(),
        games_per_opp,
        minutes,
        period_count,
        weight_decay,
        fresh_opponents,
        checkpoint_every,
        max_rounds,
        max_training_steps,
        max_target_entries_per_side,
        analytic_opponents,
        uniform_elite_players,
        seed_varied_skills,
        hermetic_neural,
        attack_canonical_features,
        ball_zone_tactical_scale,
        player_grid_xy_features,
        policy_train_max_transitions_per_tick,
        full_game_learning_enabled,
        neural_learning_rate,
        neural_optimizer_momentum,
        neural_blend_mode,
        neural_blend_lambda,
        authoritative_lambda,
        neural_blend_candidates,
        neural_blend_warmup_steps,
        policy_entropy_coeff,
        forward_select_logit_weight,
        finishing_select_bonus_weight,
        finishing_selection_floor,
        finishing_selection_max_regression,
        chance_quality_reward,
        chance_quality_composite,
        chance_quality_k,
        chance_quality_cap,
        league_neural_mcts_enabled,
        actor_critic_enabled,
        lp_coupling_enabled,
        target_standardization_enabled,
        mc_critic_target_enabled,
        neural_self_bootstrap_enabled,
        maxa_bootstrap_enabled,
        novelty_bonus_enabled,
        fast_actor_reward_fallback_enabled,
        pass_outcome_priority_learning_enabled,
        forward_pass_climb_curriculum_enabled,
        dp_bootstrap_enabled,
        dp_bootstrap_horizon,
        dp_bootstrap_sweeps,
        mpc_tier2_enabled,
        mpc_reconcile_enabled,
        mpc_field_aware_enabled,
        mpc_latent_objective_enabled,
        local_mpc_enabled,
        checkpoint_require_forward_pass_climb,
        checkpoint_max_forward_pass_regression,
        checkpoint_min_forward_pass_margin,
        checkpoint_validate_games,
        checkpoint_validate_min_forward_pass_margin,
        checkpoint_validate_min_net_forward_pass_margin,
        checkpoint_validate_min_goal_diff_margin,
        frontier_path,
        candidate_frontier_path,
        archive_dir,
        step_archive_dir,
        step_archive_buckets,
    );
    println!(
        "league_reward_profile goal={goal_reward_scale:.2} forward_pass={forward_pass_reward_scale:.2} \
         pass_chain={pass_chain_reward_scale:.2} pass_turnover_penalty={pass_turnover_penalty_scale:.2} \
         quick_forward_release={quick_forward_release_reward_scale:.4} shot_shaping={shot_shaping_reward_scale:.2} \
         shot_commitment={shot_commitment_reward_scale:.4} dribble_beat={dribble_beat_reward_scale:.2} \
         learned_epv={learned_epv_reward_scale:.2} offball={offball_support_reward_scale:.2} carry={carry_reward_scale:.2} \
         recovery={recovery_reward_scale:.2} overload_progression_per_yard={overload_progression_reward_per_yard:.4} \
         analytic_dense_floor={analytic_difference_reward_negative_dense_floor:.4} \
         analytic_dense_scale={analytic_difference_reward_negative_dense_scale:.2} mappo_share={mappo_team_reward_share:.4} \
         marl_team_weight={marl_team_reward_weight:.4} marl_intermediate_weight={marl_intermediate_reward_weight:.2} \
         planner_teacher={planner_teacher_weight:.2} finishing_select={finishing_select_bonus_weight:.2}"
    );
    println!(
        "league_stability_anchors fixed_analytic_training_anchor={} analytic_anchor_gate={} analytic_anchor_validate_games={} analytic_anchor_max_goal_diff_regression={:.3}",
        fixed_analytic_training_anchor,
        analytic_anchor_gate,
        analytic_anchor_validate_games,
        analytic_anchor_max_goal_diff_regression,
    );

    loop {
        round += 1;
        let started = Instant::now();
        let checkpoint_baseline = frontier.clone();

        // Build this round's opponent league: frozen champions + fresh (exploration).
        let mut opponents: Vec<TeamBrain> = load_league(&archive_dir);
        if hermetic_neural {
            opponents = opponents.into_iter().map(without_target_q).collect();
        }
        for i in 0..fresh_opponents {
            opponents.push(TeamBrain::fresh_with_seed(
                0xF0_0000u32
                    .wrapping_add(round.wrapping_mul(2_654_435_761))
                    .wrapping_add(i as u32),
                100 + i,
            ));
        }
        if opponents.is_empty() {
            opponents.push(TeamBrain::fresh_with_seed(0xABCD, 100));
        }

        // SOCCER_LEAGUE_ANALYTIC_OPPONENTS=1: strip the net from every opponent so they play the
        // PURE ANALYTIC engine (no net -> authoritative branch skipped -> hand-built engine
        // decides). The frontier (authoritative neural) must then learn to BEAT the analytic
        // engine to win — that IS the gradient to break past parity. (Genome variation still
        // gives the analytic opponents slightly different styles.)
        if analytic_opponents {
            for opp in opponents.iter_mut() {
                opp.neural = None;
            }
        }

        // Keep one opponent distribution point stationary across every round. Fresh analytic
        // genomes and archived champions provide diversity, but this deterministic engine anchor
        // prevents the learner's ruler from moving along with the rest of the population.
        if fixed_analytic_training_anchor && !self_play_ladder {
            opponents.push(fixed_analytic_anchor());
        }

        // SELF-PLAY LADDER: replace the league with just the CURRENT published champion (neural, so it
        // drives Team B through the same inference path and strengthens as the ladder climbs) plus one
        // ANALYTIC baseline — the challenger co-evolves against a tougher self AND stays grounded
        // against the engine (the north-star metric it is scored on). Gen-0 (nothing published yet) =
        // analytic only, mirroring the 5v5 ladder's scripted gen-0 baseline.
        if self_play_ladder {
            let mut pool: Vec<TeamBrain> = Vec::new();
            if let Some(champ) = load_brain(&frontier_path) {
                pool.push(if hermetic_neural {
                    without_target_q(champ)
                } else {
                    champ
                });
            }
            pool.push(fixed_analytic_anchor());
            opponents = pool;
        }

        // Build this round's fixtures, then run them DATA-PARALLEL across N workers: each worker
        // clones the current frontier + runner, plays a disjoint slice (carry-forward within the
        // slice), and returns its trained net; we then AVERAGE the workers' nets. ~N× faster per
        // round while keeping the full opponent league. Cap kept modest (default 3) so we don't
        // starve the box.
        struct Fx {
            opp_idx: usize,
            opp_id: usize,
            seed: u32,
            home: bool,
        }
        // With an asymmetric per-team skill handicap (SOCCER_SKILL_SCALE_HOME on the frontier),
        // the frontier must always occupy the handicapped side, so pin it Home when requested;
        // otherwise keep the home/away alternation that removes a home-side artifact.
        let frontier_always_home = std::env::var("SOCCER_LEAGUE_FRONTIER_ALWAYS_HOME")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let mut fixtures: Vec<Fx> = Vec::new();
        let mut fixture = 0u32;
        for idx in 0..opponents.len() {
            for g in 0..games_per_opp {
                let seed = train_seed_base
                    .wrapping_add(round.wrapping_mul(1_000_003))
                    .wrapping_add(fixture.wrapping_mul(2_246_822_519))
                    .wrapping_add(g as u32);
                let home = frontier_always_home || ((round as usize + fixture as usize) % 2 == 0);
                fixtures.push(Fx {
                    opp_idx: idx,
                    opp_id: idx + 1,
                    seed,
                    home,
                });
                fixture += 1;
            }
        }
        let requested_workers = env_usize("SOCCER_LEAGUE_PARALLELISM", 3);
        let workers = league_worker_count(requested_workers, fixtures.len());
        if workers < requested_workers.clamp(1, fixtures.len().max(1)) {
            println!(
                "league_parallelism_serialized_for_mpc_objective requested_workers={} active_workers={} fixtures={} opt_out_env=SOCCER_LEAGUE_MPC_OBJECTIVE_SERIAL=0",
                requested_workers,
                workers,
                fixtures.len()
            );
        }
        let mut round_kpis = LeagueRoundKpis::default();
        let (wins, losses, trained_brains) = if workers == 1 {
            let mut wins = 0i32;
            let mut losses = 0i32;
            for fx in &fixtures {
                let (trained, kpis) = play_and_carry(
                    &mut runner,
                    frontier,
                    &opponents[fx.opp_idx],
                    fx.seed,
                    fx.home,
                    fx.opp_id,
                );
                frontier = trained;
                if let Some(kpis) = kpis {
                    if kpis.goal_diff > 0 {
                        wins += 1;
                    } else if kpis.goal_diff < 0 {
                        losses += 1;
                    }
                    round_kpis.add(kpis);
                }
            }
            (wins, losses, vec![frontier.clone()])
        } else {
            let chunk = fixtures.len().div_ceil(workers);
            let wins_a = std::sync::atomic::AtomicI32::new(0);
            let losses_a = std::sync::atomic::AtomicI32::new(0);
            let trained_brains = std::sync::Mutex::new(Vec::<TeamBrain>::new());
            let match_kpis = std::sync::Mutex::new(Vec::<LeagueMatchKpis>::new());
            std::thread::scope(|scope| {
                for w in 0..workers {
                    let lo = w * chunk;
                    let hi = ((w + 1) * chunk).min(fixtures.len());
                    if lo >= hi {
                        continue;
                    }
                    let slice = &fixtures[lo..hi];
                    let opponents = &opponents;
                    let wins_a = &wins_a;
                    let losses_a = &losses_a;
                    let trained_brains = &trained_brains;
                    let match_kpis = &match_kpis;
                    let mut runner = runner.clone();
                    let mut wf = frontier.clone();
                    scope.spawn(move || {
                        use std::sync::atomic::Ordering::Relaxed;
                        for fx in slice {
                            let (t, kpis) = play_and_carry(
                                &mut runner,
                                wf,
                                &opponents[fx.opp_idx],
                                fx.seed,
                                fx.home,
                                fx.opp_id,
                            );
                            wf = t;
                            if let Some(kpis) = kpis {
                                if kpis.goal_diff > 0 {
                                    wins_a.fetch_add(1, Relaxed);
                                } else if kpis.goal_diff < 0 {
                                    losses_a.fetch_add(1, Relaxed);
                                }
                                match_kpis.lock().unwrap().push(kpis);
                            }
                        }
                        trained_brains.lock().unwrap().push(wf);
                    });
                }
            });
            for kpis in match_kpis.into_inner().unwrap() {
                round_kpis.add(kpis);
            }
            (
                wins_a.load(std::sync::atomic::Ordering::Relaxed),
                losses_a.load(std::sync::atomic::Ordering::Relaxed),
                trained_brains.into_inner().unwrap(),
            )
        };
        let trained_snapshots = trained_brains
            .iter()
            .filter_map(|brain| brain.neural.clone())
            .collect::<Vec<_>>();
        if let Some(avg) =
            average_soccer_neural_network_snapshots(frontier.neural.as_ref(), &trained_snapshots)
        {
            frontier.neural = Some(avg);
        }
        if hermetic_neural {
            frontier.home_target_entries.clear();
            frontier.away_target_entries.clear();
        } else if !trained_brains.is_empty() {
            frontier.home_target_entries = merge_target_entries(
                trained_brains
                    .iter()
                    .map(|brain| brain.home_target_entries.clone()),
                max_target_entries_per_side,
            );
            frontier.away_target_entries = merge_target_entries(
                trained_brains
                    .iter()
                    .map(|brain| brain.away_target_entries.clone()),
                max_target_entries_per_side,
            );
            frontier = with_bidirectional_target_entries(&frontier);
            if max_target_entries_per_side > 0 {
                frontier
                    .home_target_entries
                    .truncate(max_target_entries_per_side);
                frontier
                    .away_target_entries
                    .truncate(max_target_entries_per_side);
            }
        }

        // Weight decay to keep the net in the trainable regime.
        if let Some(mut s) = frontier.neural.take() {
            apply_weight_decay(&mut s, weight_decay);
            frontier.neural = Some(s);
        }

        // Warm-restart weight perturbation (explicit exploration / escape mechanism): every
        // SOCCER_LEAGUE_WARM_RESTART_EVERY rounds, kick the net out of its basin with annealed
        // Gaussian noise (scale decays each cycle — simulated annealing), then let the next round
        // re-descend. Env-gated: SOCCER_LEAGUE_WEIGHT_NOISE=0 (default) leaves training unchanged.
        let wn_base = env_f64("SOCCER_LEAGUE_WEIGHT_NOISE", 0.0);
        if wn_base > 0.0 {
            let wn_every = env_usize("SOCCER_LEAGUE_WARM_RESTART_EVERY", 4).max(1) as u32;
            if round % wn_every == 0 {
                let wn_frac = env_f64("SOCCER_LEAGUE_WEIGHT_NOISE_FRAC", 0.1);
                let wn_decay = env_f64("SOCCER_LEAGUE_WEIGHT_NOISE_DECAY", 0.9);
                let cycle = (round / wn_every).max(1) as i32;
                let scale = wn_base * wn_decay.powi(cycle - 1);
                if let Some(mut s) = frontier.neural.take() {
                    let seed = 0x9E37_79B9_7F4A_7C15u64
                        ^ (round as u64).wrapping_mul(0x0000_0001_00000193)
                        ^ (s.training_steps as u64);
                    perturb_weights(&mut s, scale, wn_frac, seed);
                    println!(
                        "league_warm_restart round={round} cycle={cycle} scale={scale:.4} frac={wn_frac} l2={:.3}",
                        s.l2_norm
                    );
                    frontier.neural = Some(s);
                }
            }
        }

        // Persist the candidate every round, but keep learned-params.json protected until the
        // checkpoint gate accepts the candidate.
        let (l2, steps) = frontier
            .neural
            .as_ref()
            .map(|s| (s.l2_norm, s.training_steps))
            .unwrap_or((0.0, 0));
        if frontier.neural.is_some() {
            if let Err(e) = write_frontier(&candidate_frontier_path, &frontier) {
                eprintln!("league_candidate_write_error: {e}");
            }
            for bucket in &step_archive_buckets {
                if steps >= *bucket && archived_step_buckets.insert(*bucket) {
                    let path = step_bucket_archive_path(&step_archive_dir, *bucket, round, steps);
                    match write_frontier(&path, &frontier) {
                        Ok(()) => println!(
                            "league_step_archive bucket={} round={round} training_steps={steps} -> {path}",
                            bucket
                        ),
                        Err(e) => eprintln!(
                            "league_step_archive_write_error bucket={} round={round} training_steps={steps}: {e}",
                            bucket
                        ),
                    }
                }
            }
        }

        let kpi_games = round_kpis.games.max(1) as f64;
        let pass_completion = if round_kpis.passes_attempted_for == 0 {
            0.0
        } else {
            round_kpis.passes_completed_for as f64 / round_kpis.passes_attempted_for as f64
        };
        let mean_forward_pass_margin = round_kpis.mean_forward_pass_margin();
        let mean_net_forward_pass_margin = round_kpis.mean_net_forward_pass_margin();
        let learned_mpc_replan_rate = if round_kpis.learned_policy_option_decisions == 0 {
            0.0
        } else {
            round_kpis.learned_mpc_replans as f64
                / round_kpis.learned_policy_option_decisions as f64
        };
        println!(
            "league_advancement round={round} metric=completed_forward_passes games={} forward_passes_for={} forward_passes_against={} forward_pass_margin={} forward_pass_margin_per_game={:.3} pass_turnovers_for={} pass_turnovers_against={} net_forward_passes_for={} net_forward_passes_against={} net_forward_pass_margin={} net_forward_pass_margin_per_game={:.3}",
            round_kpis.games,
            round_kpis.forward_passes_for,
            round_kpis.forward_passes_against,
            round_kpis.forward_pass_margin,
            mean_forward_pass_margin,
            round_kpis.pass_turnovers_for,
            round_kpis.pass_turnovers_against,
            round_kpis.net_forward_passes_for,
            round_kpis.net_forward_passes_against,
            round_kpis.net_forward_pass_margin,
            mean_net_forward_pass_margin,
        );
        println!(
            "league_kpi round={round} games={} advancement_metric=completed_forward_passes net_metric=completed_forward_passes_minus_turnovers forward_pass_margin_per_game={:.3} net_forward_pass_margin_per_game={:.3} objective_net_forward_passes={:.3} objective_net_forward_pass_margin={:.3} gd_total={} gd_per_game={:.2} goals_for={} goals_against={} shots_for={} shots_against={} sot_for={} sot_against={} shots_after_pass={} pass_completion={:.3} completed_passes={} forward_passes={} backward_passes={} dribble_beats={} assists={} crosses={} chain_gain_yards={:.1} chain_net_losses={} learned_policy_option_decisions={} learned_policy_multi_option_decisions={} learned_policy_legal_action_options={} learned_mpc_replans={} learned_mpc_replans_mpc={} learned_mpc_replans_option_score_safety={} learned_mpc_replans_neural_mcts={} learned_mpc_replan_rate={:.3}",
            round_kpis.games,
            mean_forward_pass_margin,
            mean_net_forward_pass_margin,
            round_kpis.mean_objective_fitness(),
            round_kpis.mean_objective_fitness_margin(),
            round_kpis.goal_diff,
            round_kpis.goal_diff as f64 / kpi_games,
            round_kpis.goals_for,
            round_kpis.goals_against,
            round_kpis.shots_for,
            round_kpis.shots_against,
            round_kpis.shots_on_target_for,
            round_kpis.shots_on_target_against,
            round_kpis.shots_after_pass_for,
            pass_completion,
            round_kpis.passes_completed_for,
            round_kpis.forward_passes_for,
            round_kpis.backward_passes_for,
            round_kpis.dribble_beats_for,
            round_kpis.assists_for,
            round_kpis.crosses_completed_for,
            round_kpis.pass_chain_gain_yards_for,
            round_kpis.pass_chain_net_losses_for,
            round_kpis.learned_policy_option_decisions,
            round_kpis.learned_policy_multi_option_decisions,
            round_kpis.learned_policy_legal_action_options,
            round_kpis.learned_mpc_replans,
            round_kpis.learned_mpc_replans_mpc,
            round_kpis.learned_mpc_replans_option_score_safety,
            round_kpis.learned_mpc_replans_neural_mcts,
            learned_mpc_replan_rate,
        );

        // Temporal checkpoint into the league (growing diversity of past selves).
        if round % (checkpoint_every as u32) == 0 {
            if frontier.neural.is_some() {
                let prior_best = best_checkpoint_net_forward_pass_margin;
                let checkpoint_incumbent =
                    load_brain(&frontier_path).unwrap_or_else(|| checkpoint_baseline.clone());
                let validation = if checkpoint_validate_games > 0 {
                    let validation = play_checkpoint_validation(
                        &runner,
                        &frontier,
                        &checkpoint_incumbent,
                        "incumbent",
                        round,
                        checkpoint_validate_games,
                    );
                    println!(
                        "league_checkpoint_validation round={round} games={} forward_pass_margin_per_game={:.3} net_forward_pass_margin_per_game={:.3} gd_total={} gd_per_game={:.3} shots_after_pass={} completed_passes={} pass_gain_yards={:.1}",
                        validation.games,
                        validation.mean_forward_pass_margin(),
                        validation.mean_net_forward_pass_margin(),
                        validation.goal_diff,
                        validation.mean_goal_diff(),
                        validation.shots_after_pass_for,
                        validation.passes_completed_for,
                        validation.pass_chain_gain_yards_for,
                    );
                    Some(validation)
                } else {
                    None
                };
                let (candidate_anchor_validation, incumbent_anchor_validation) =
                    if analytic_anchor_validate_games > 0 {
                        let anchor = fixed_analytic_anchor();
                        let candidate = play_checkpoint_validation(
                            &runner,
                            &frontier,
                            &anchor,
                            "candidate_vs_analytic_anchor",
                            round,
                            analytic_anchor_validate_games,
                        );
                        let incumbent = play_checkpoint_validation(
                            &runner,
                            &checkpoint_incumbent,
                            &anchor,
                            "incumbent_vs_analytic_anchor",
                            round,
                            analytic_anchor_validate_games,
                        );
                        println!(
                            "league_analytic_anchor_validation round={round} games={} candidate_gd_per_game={:.3} incumbent_gd_per_game={:.3} delta_gd_per_game={:.3}",
                            analytic_anchor_validate_games,
                            candidate.mean_goal_diff(),
                            incumbent.mean_goal_diff(),
                            candidate.mean_goal_diff() - incumbent.mean_goal_diff(),
                        );
                        (Some(candidate), Some(incumbent))
                    } else {
                        (None, None)
                    };
                let passes_analytic_anchor = analytic_anchor_non_regression_passes(
                    candidate_anchor_validation.as_ref(),
                    incumbent_anchor_validation.as_ref(),
                    analytic_anchor_validate_games,
                    analytic_anchor_max_goal_diff_regression,
                );
                let validation_forward_pass_margin =
                    validation.as_ref().map(|v| v.mean_forward_pass_margin());
                let validation_net_forward_pass_margin = validation
                    .as_ref()
                    .map(|v| v.mean_net_forward_pass_margin());
                let gate_forward_pass_margin =
                    validation_forward_pass_margin.unwrap_or(mean_forward_pass_margin);
                let gate_net_forward_pass_margin =
                    validation_net_forward_pass_margin.unwrap_or(mean_net_forward_pass_margin);
                let passes_forward_pass_climb = !checkpoint_require_forward_pass_climb
                    || !prior_best.is_finite()
                    || gate_net_forward_pass_margin + checkpoint_max_forward_pass_regression
                        >= prior_best;
                let passes_forward_pass_floor =
                    gate_forward_pass_margin > checkpoint_min_forward_pass_margin;
                let passes_net_forward_pass_floor =
                    gate_net_forward_pass_margin > checkpoint_validate_min_net_forward_pass_margin;
                let gate_goal_diff_margin = validation
                    .as_ref()
                    .map(|v| v.mean_goal_diff())
                    .unwrap_or_else(|| round_kpis.goal_diff as f64 / kpi_games);
                let passes_goal_diff_floor =
                    gate_goal_diff_margin > checkpoint_validate_min_goal_diff_margin;
                let passes_validation = checkpoint_validation_passes(
                    validation.as_ref(),
                    checkpoint_validate_games,
                    checkpoint_validate_min_forward_pass_margin,
                    checkpoint_validate_min_net_forward_pass_margin,
                    checkpoint_validate_min_goal_diff_margin,
                );
                let promotes = if self_play_ladder {
                    if !std::path::Path::new(&frontier_path).exists() {
                        // Bootstrap gen-0: no champion published yet -> promote the current frontier as
                        // the FIRST champion so co-evolution can start. The 5v5 installs gen-0 = the
                        // scripted baseline (a beatable floor); analytic is too strong to be that floor
                        // here (it IS the plateau), so the floor is the starting net itself. Analytic
                        // stays in the training pool for grounding; the arms race begins next round.
                        println!(
                            "league_self_play_ladder round={round} verdict=PROMOTED bootstrap_gen0=true"
                        );
                        passes_analytic_anchor
                    } else {
                        // Ladder gate: promote iff the challenger beat the CURRENT champion by the
                        // goal-diff margin over the held-out head-to-head eval. Republishing the
                        // frontier advances the champion, so next round's opponent is this stronger net
                        // — the ladder climbs.
                        let ladder_promotes = gate_goal_diff_margin >= self_play_promote_margin;
                        println!(
                            "league_self_play_ladder round={round} gd_vs_champion={:.3} promote_margin={:.3} verdict={}",
                            gate_goal_diff_margin,
                            self_play_promote_margin,
                            if ladder_promotes { "PROMOTED" } else { "held" }
                        );
                        ladder_promotes && passes_analytic_anchor
                    }
                } else {
                    passes_forward_pass_climb
                        && ((passes_forward_pass_floor && passes_net_forward_pass_floor)
                            || passes_goal_diff_floor)
                        && passes_validation
                        && passes_analytic_anchor
                };
                if promotes {
                    let cp = format!(
                        "{}/league-r{:04}-{}.json",
                        archive_dir.trim_end_matches('/'),
                        round,
                        chrono_stamp()
                    );
                    let _ = fs::create_dir_all(&archive_dir);
                    let _ = write_frontier(&cp, &frontier);
                    if let Err(e) = write_frontier(&frontier_path, &frontier) {
                        eprintln!("league_publish_write_error: {e}");
                    }
                    if gate_net_forward_pass_margin > best_checkpoint_net_forward_pass_margin {
                        best_checkpoint_net_forward_pass_margin = gate_net_forward_pass_margin;
                    }
                    println!(
                        "league_checkpoint round={round} metric=completed_forward_passes net_metric=completed_forward_passes_minus_turnovers gate_forward_pass_margin_per_game={gate_forward_pass_margin:.3} gate_net_forward_pass_margin_per_game={gate_net_forward_pass_margin:.3} train_forward_pass_margin_per_game={mean_forward_pass_margin:.3} train_net_forward_pass_margin_per_game={mean_net_forward_pass_margin:.3} best_net_forward_pass_margin_per_game={best_checkpoint_net_forward_pass_margin:.3} -> {cp}"
                    );
                } else {
                    println!(
                        "league_checkpoint_held round={round} metric=completed_forward_passes net_metric=completed_forward_passes_minus_turnovers gate_forward_pass_margin_per_game={gate_forward_pass_margin:.3} gate_net_forward_pass_margin_per_game={gate_net_forward_pass_margin:.3} gate_goal_diff_margin_per_game={gate_goal_diff_margin:.3} train_forward_pass_margin_per_game={mean_forward_pass_margin:.3} train_net_forward_pass_margin_per_game={mean_net_forward_pass_margin:.3} best_net_forward_pass_margin_per_game={prior_best:.3} require_climb={} raw_min_margin={checkpoint_min_forward_pass_margin:.3} net_min_margin={checkpoint_validate_min_net_forward_pass_margin:.3} validation_min_goal_diff_margin={checkpoint_validate_min_goal_diff_margin:.3} validation_passed={passes_validation} validation_games={}/{} analytic_anchor_passed={passes_analytic_anchor} analytic_anchor_games={}",
                        checkpoint_require_forward_pass_climb,
                        validation.as_ref().map(|validation| validation.games).unwrap_or(0),
                        checkpoint_validate_games,
                        analytic_anchor_validate_games,
                    );
                }
            }
        }

        println!(
            "league_round round={round} opponents={} games={} frontier_wins={wins} frontier_losses={losses} l2={l2:.3} training_steps={steps} home_targets={} away_targets={} secs={:.1}",
            opponents.len(),
            fixture,
            frontier.home_target_entries.len(),
            frontier.away_target_entries.len(),
            started.elapsed().as_secs_f64()
        );
        if max_training_steps > 0 && steps >= max_training_steps {
            println!(
                "league_train_stopped max_training_steps={max_training_steps} reached_steps={steps} rounds_completed={round}"
            );
            break;
        }
        if max_rounds > 0 && round as usize >= max_rounds {
            println!("league_train_stopped max_rounds={max_rounds} rounds_completed={round}");
            break;
        }
    }
}

fn chrono_now() -> String {
    // avoid pulling chrono; use a coarse epoch-based UTC-ish stamp
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("epoch{secs}")
}
fn chrono_stamp() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{secs}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use soccer_engine::des::general::soccer::MatchStats;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn mirror_test_state() -> SoccerQStateKey {
        serde_json::from_value(serde_json::json!({
            "phase": "HomeAttack",
            "role": "Midfielder",
            "possessionRelative": 1,
            "ballZoneX": 2,
            "ballZoneY": 6,
            "ballFineLane": 5,
            "ballFineRow": 20,
            "receiveFacing": "NorthEast",
            "actionFacing": "SouthWest",
            "scoreDiffBucket": 0,
            "hasBall": true,
            "visibleBall": true,
            "shotLaneOpen": false,
            "visiblePassOptionsBin": 2,
            "ballDistanceBin": 2,
            "yardsToGoalBin": 3,
            "pressureBin": 1,
            "openSpaceBin": 4
        }))
        .expect("compact test Q-state should deserialize")
    }

    fn mirror_test_target_entry(
        receiver_descriptor: i32,
        value: f64,
        visits: u32,
    ) -> SoccerQTargetEntry {
        SoccerQTargetEntry {
            state: mirror_test_state(),
            action: "pass".to_string(),
            target_fine_cell_id: 20 * LEAGUE_TARGET_FINE_COLUMNS + 4,
            target_tactical_cell_id: 6 * LEAGUE_TARGET_TACTICAL_COLUMNS + 2,
            target_macro_cell_id: 3 * LEAGUE_TARGET_MACRO_COLUMNS + 1,
            target_root_cell_id: 0,
            receiver_descriptor,
            value,
            visits,
        }
    }

    struct EnvVarGuard {
        key: &'static str,
        previous: Option<String>,
    }

    impl EnvVarGuard {
        fn clear(key: &'static str) -> Self {
            let previous = std::env::var(key).ok();
            std::env::remove_var(key);
            Self { key, previous }
        }

        fn set(key: &'static str, value: &'static str) -> Self {
            let previous = std::env::var(key).ok();
            std::env::set_var(key, value);
            Self { key, previous }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.previous {
                Some(previous) => std::env::set_var(self.key, previous),
                None => std::env::remove_var(self.key),
            }
        }
    }

    #[test]
    fn league_advancement_objective_uses_forward_passes_not_score_or_shots() {
        let mut stats = MatchStats::default();
        stats.shots_home = 1;
        stats.shots_away = 18;
        stats.shots_on_target_home = 0;
        stats.shots_on_target_away = 9;
        stats.passes_completed_forward_home = 14;
        stats.passes_completed_forward_away = 5;
        stats.interceptions_away = 3;
        stats.interceptions_home = 1;

        let summary = MatchSummary {
            score_home: 0,
            score_away: 4,
            ticks: 0,
            simulated_seconds: 0.0,
            stats,
        };

        let home = LeagueMatchKpis::from_summary(&summary, Team::Home);
        assert_eq!(home.objective_fitness, 11.0);
        assert_eq!(home.objective_fitness_margin, 7.0);
        assert_eq!(home.forward_pass_margin, 9);
        assert_eq!(home.pass_turnovers_for, 3);
        assert_eq!(home.pass_turnovers_against, 1);
        assert_eq!(home.net_forward_passes_for, 11);
        assert_eq!(home.net_forward_passes_against, 4);
        assert_eq!(home.net_forward_pass_margin, 7);

        let away = LeagueMatchKpis::from_summary(&summary, Team::Away);
        assert_eq!(away.objective_fitness, 4.0);
        assert_eq!(away.objective_fitness_margin, -7.0);
        assert_eq!(away.forward_pass_margin, -9);
        assert_eq!(away.pass_turnovers_for, 1);
        assert_eq!(away.pass_turnovers_against, 3);
        assert_eq!(away.net_forward_passes_for, 4);
        assert_eq!(away.net_forward_passes_against, 11);
        assert_eq!(away.net_forward_pass_margin, -7);
    }

    #[test]
    fn checkpoint_validation_requires_forward_net_and_scoreline() {
        let mut validation = LeagueRoundKpis::default();
        validation.add(LeagueMatchKpis {
            goal_diff: -3,
            forward_pass_margin: 5,
            net_forward_pass_margin: 4,
            ..LeagueMatchKpis::default()
        });

        assert!(!checkpoint_validation_passes(
            Some(&validation),
            1,
            0.0,
            0.0,
            0.0,
        ));

        assert!(!checkpoint_validation_passes(
            Some(&validation),
            1,
            0.0,
            4.0,
            -99.0,
        ));

        validation.goal_diff = 0;
        assert!(checkpoint_validation_passes(
            Some(&validation),
            1,
            0.0,
            0.0,
            0.0,
        ));
    }

    #[test]
    fn analytic_anchor_gate_rejects_absolute_goal_diff_regression() {
        let mut candidate = LeagueRoundKpis::default();
        candidate.add(LeagueMatchKpis {
            goal_diff: 0,
            ..LeagueMatchKpis::default()
        });
        let mut incumbent = LeagueRoundKpis::default();
        incumbent.add(LeagueMatchKpis {
            goal_diff: 1,
            ..LeagueMatchKpis::default()
        });

        assert!(!analytic_anchor_non_regression_passes(
            Some(&candidate),
            Some(&incumbent),
            1,
            0.0,
        ));
        assert!(analytic_anchor_non_regression_passes(
            Some(&candidate),
            Some(&incumbent),
            1,
            1.0,
        ));

        candidate.goal_diff = 2;
        assert!(analytic_anchor_non_regression_passes(
            Some(&candidate),
            Some(&incumbent),
            1,
            0.0,
        ));
    }

    #[test]
    fn target_entry_rotation_flips_pitch_axes_and_home_away_phases() {
        assert_eq!(
            rotate_tactical_phase_180(TacticalPhase::HomeAttack),
            TacticalPhase::AwayAttack
        );
        assert_eq!(
            rotate_tactical_phase_180(TacticalPhase::AwayBuildUp),
            TacticalPhase::HomeBuildUp
        );
        assert_eq!(
            rotate_facing_180(FacingBucket::NorthEast),
            FacingBucket::SouthWest
        );
        assert_eq!(
            rotate_facing_180(FacingBucket::SouthWest),
            FacingBucket::NorthEast
        );
        assert_eq!(
            rotate_pitch_cell_180(
                20 * LEAGUE_TARGET_FINE_COLUMNS + 4,
                LEAGUE_TARGET_FINE_COLUMNS,
                LEAGUE_TARGET_FINE_ROWS,
            ),
            3 * LEAGUE_TARGET_FINE_COLUMNS + 7
        );
        assert_eq!(
            rotate_pitch_cell_180(
                6 * LEAGUE_TARGET_TACTICAL_COLUMNS + 2,
                LEAGUE_TARGET_TACTICAL_COLUMNS,
                LEAGUE_TARGET_TACTICAL_ROWS,
            ),
            LEAGUE_TARGET_TACTICAL_COLUMNS + 3
        );
        assert_eq!(
            rotate_pitch_cell_180(
                3 * LEAGUE_TARGET_MACRO_COLUMNS + 1,
                LEAGUE_TARGET_MACRO_COLUMNS,
                LEAGUE_TARGET_MACRO_ROWS,
            ),
            1
        );

        let rotated = rotate_target_state_180(mirror_test_state());
        assert_eq!(rotated.phase, TacticalPhase::AwayAttack);
        assert_eq!(rotated.ball_zone_x, 3);
        assert_eq!(rotated.ball_zone_y, 1);
        assert_eq!(rotated.ball_fine_lane, 6);
        assert_eq!(rotated.ball_fine_row, 3);
        assert_eq!(rotated.receive_facing, FacingBucket::SouthWest);
        assert_eq!(rotated.action_facing, FacingBucket::NorthEast);
    }

    #[test]
    fn target_entry_merge_preserves_receiver_descriptor_dimension() {
        let merged = merge_target_entries(
            [vec![
                mirror_test_target_entry(-1, 1.0, 3),
                mirror_test_target_entry(42, 4.0, 5),
            ]],
            0,
        );

        assert_eq!(merged.len(), 2);
        let mut descriptors = merged
            .iter()
            .map(|entry| entry.receiver_descriptor)
            .collect::<Vec<_>>();
        descriptors.sort_unstable();
        assert_eq!(descriptors, vec![-1, 42]);
    }

    #[test]
    fn bidirectional_target_entries_are_idempotent() {
        let mut brain = TeamBrain::fresh();
        brain.home_target_entries = vec![mirror_test_target_entry(7, 2.5, 11)];

        let once = with_bidirectional_target_entries(&brain);
        let twice = with_bidirectional_target_entries(&once);

        assert_eq!(once.home_target_entries.len(), 1);
        assert_eq!(once.away_target_entries.len(), 1);
        assert_eq!(twice.home_target_entries.len(), 1);
        assert_eq!(twice.away_target_entries.len(), 1);
        assert_eq!(once.home_target_entries[0].visits, 11);
        assert_eq!(once.away_target_entries[0].visits, 11);
        assert_eq!(twice.home_target_entries[0].visits, 11);
        assert_eq!(twice.away_target_entries[0].visits, 11);
        assert_eq!(
            target_entry_key(&once.home_target_entries[0]),
            target_entry_key(&twice.home_target_entries[0])
        );
        assert_eq!(
            target_entry_key(&once.away_target_entries[0]),
            target_entry_key(&twice.away_target_entries[0])
        );
    }

    #[test]
    fn league_trainer_neural_mcts_defaults_off_for_mdp_pomdp_mpc_learning() {
        let _lock = env_lock().lock().unwrap();
        let _guard = EnvVarGuard::clear("SOCCER_LEAGUE_NEURAL_MCTS_ENABLED");
        let mut config = EngineMatchRunnerConfig::default();
        config.base.neural_blend.mcts_enabled = true;

        let enabled = apply_league_neural_mcts_config(&mut config);

        assert!(!enabled);
        assert!(!config.base.neural_blend.mcts_enabled);
    }

    #[test]
    fn league_trainer_keeps_mpc_execution_stack_enabled() {
        let _lock = env_lock().lock().unwrap();
        let _guard = EnvVarGuard::clear("SOCCER_LEAGUE_NEURAL_MCTS_ENABLED");
        let mut config = EngineMatchRunnerConfig::default();

        let enabled = apply_league_neural_mcts_config(&mut config);

        assert!(!enabled);
        assert!(config.base.local_mpc_enabled);
        assert!(config.base.mpc.tier2_player_enabled);
        assert!(config.base.mpc.reconcile_enabled);
        assert!(config.base.mpc.field_aware_enabled);
        assert!(config.base.mpc.latent_objective_enabled);
    }

    #[test]
    fn league_trainer_serializes_parallelism_for_carried_mpc_objective() {
        let _lock = env_lock().lock().unwrap();
        let _mpc_guard = EnvVarGuard::set("DD_SOCCER_ENABLE_LEARNED_MPC_OBJECTIVE", "1");
        let _serial_guard = EnvVarGuard::clear("SOCCER_LEAGUE_MPC_OBJECTIVE_SERIAL");

        assert_eq!(league_worker_count(4, 8), 1);
    }

    #[test]
    fn league_trainer_allows_parallel_mpc_objective_opt_out() {
        let _lock = env_lock().lock().unwrap();
        let _mpc_guard = EnvVarGuard::set("DD_SOCCER_ENABLE_LEARNED_MPC_OBJECTIVE", "1");
        let _serial_guard = EnvVarGuard::set("SOCCER_LEAGUE_MPC_OBJECTIVE_SERIAL", "0");

        assert_eq!(league_worker_count(4, 8), 4);
    }

    #[test]
    fn candidate_frontier_path_keeps_public_frontier_protected() {
        assert_eq!(
            candidate_frontier_path("/tmp/run/learned-params.json"),
            "/tmp/run/learned-params.candidate.json"
        );
        assert_eq!(
            candidate_frontier_path("/tmp/run/frontier"),
            "/tmp/run/frontier.candidate"
        );
    }

    #[test]
    fn step_archive_bucket_env_parses_sorted_unique_positive_values() {
        let _lock = env_lock().lock().unwrap();
        let _guard = EnvVarGuard::set(
            "SOCCER_LEAGUE_STEP_ARCHIVE_BUCKETS",
            "100000, 25000;0 bad 100000 50000",
        );

        assert_eq!(
            env_usize_list("SOCCER_LEAGUE_STEP_ARCHIVE_BUCKETS"),
            vec![25000, 50000, 100000]
        );
    }

    #[test]
    fn step_bucket_archive_path_encodes_bucket_round_and_steps() {
        assert_eq!(
            step_bucket_archive_path("/tmp/run/step-buckets/", 25000, 7, 25120),
            "/tmp/run/step-buckets/league-step-b00025000-r0007-s25120.json"
        );
    }

    #[test]
    fn league_trainer_neural_mcts_requires_explicit_league_opt_in() {
        let _lock = env_lock().lock().unwrap();
        let _guard = EnvVarGuard::set("SOCCER_LEAGUE_NEURAL_MCTS_ENABLED", "1");
        let mut config = EngineMatchRunnerConfig::default();

        let enabled = apply_league_neural_mcts_config(&mut config);

        assert!(enabled);
        assert!(config.base.neural_blend.mcts_enabled);
    }
}
