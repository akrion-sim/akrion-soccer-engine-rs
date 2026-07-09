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
//! NO genetic algorithm, NO RDS.
//!
//! Env: SOCCER_LEAGUE_FRONTIER (path, default /tmp/neural-climb-local/learned-params.json)
//!      SOCCER_LEAGUE_ARCHIVE  (dir,  default /tmp/neural-climb-local/champions)
//!      SOCCER_LEAGUE_GAMES_PER_OPP (usize, default 2)
//!      SOCCER_LEAGUE_MINUTES  (f64, default 2.0)
//!      SOCCER_LEAGUE_WEIGHT_DECAY (f64 per round, default 0.01)
//!      SOCCER_LEAGUE_FRESH_OPPONENTS (usize, default 2)
//!      SOCCER_LEAGUE_CHECKPOINT_EVERY_ROUNDS (usize, default 10 — add frontier to league)

use std::collections::HashMap;
use std::fs;
use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::time::Instant;

use soccer_engine::des::general::soccer::{
    FacingBucket, MatchSummary, SoccerNeuralNetworkSnapshot, SoccerQStateKey, SoccerQTargetEntry,
    TacticalPhase, Team,
};
use soccer_engine::des::general::tournament::{
    EngineMatchRunner, EngineMatchRunnerConfig, TeamBrain, TournamentMatchContext,
    TournamentMatchRunner, TournamentStage,
};
fn env_str(k: &str, d: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| d.to_string())
}
fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}
fn env_f64(k: &str, d: f64) -> f64 {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
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

fn apply_league_neural_mcts_config(config: &mut EngineMatchRunnerConfig) -> bool {
    let enabled = env_bool("SOCCER_LEAGUE_NEURAL_MCTS_ENABLED", false);
    config.base.neural_blend.mcts_enabled = enabled;
    enabled
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
    backward_passes_for: u32,
    dribble_beats_for: u32,
    assists_for: u32,
    crosses_completed_for: u32,
    pass_chain_gain_yards_for: f64,
    pass_chain_net_losses_for: u32,
    objective_fitness: f64,
    objective_fitness_margin: f64,
}

impl LeagueMatchKpis {
    fn from_summary(summary: &MatchSummary, frontier_team: Team) -> Self {
        let stats = &summary.stats;
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
                backward_passes_for: stats.passes_completed_backward_home,
                dribble_beats_for: stats.dribble_beats_home,
                assists_for: stats.assists_home,
                crosses_completed_for: stats.crosses_completed_home,
                pass_chain_gain_yards_for: stats.pass_chain_gain_yards_home,
                pass_chain_net_losses_for: stats.pass_chains_net_loss_home,
                objective_fitness: stats.passes_completed_forward_home as f64,
                objective_fitness_margin: stats.passes_completed_forward_home as f64
                    - stats.passes_completed_forward_away as f64,
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
                backward_passes_for: stats.passes_completed_backward_away,
                dribble_beats_for: stats.dribble_beats_away,
                assists_for: stats.assists_away,
                crosses_completed_for: stats.crosses_completed_away,
                pass_chain_gain_yards_for: stats.pass_chain_gain_yards_away,
                pass_chain_net_losses_for: stats.pass_chains_net_loss_away,
                objective_fitness: stats.passes_completed_forward_away as f64,
                objective_fitness_margin: stats.passes_completed_forward_away as f64
                    - stats.passes_completed_forward_home as f64,
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
    backward_passes_for: u32,
    dribble_beats_for: u32,
    assists_for: u32,
    crosses_completed_for: u32,
    pass_chain_gain_yards_for: f64,
    pass_chain_net_losses_for: u32,
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

fn mirror_pitch_cell_y(cell_id: usize, columns: usize, rows: usize) -> usize {
    if columns == 0 || rows == 0 {
        return cell_id;
    }
    let cell_count = columns.saturating_mul(rows);
    if cell_id >= cell_count {
        return cell_id;
    }
    let x = cell_id % columns;
    let y = cell_id / columns;
    (rows - 1 - y) * columns + x
}

fn mirror_tactical_phase_y(phase: TacticalPhase) -> TacticalPhase {
    match phase {
        TacticalPhase::HomeBuildUp => TacticalPhase::AwayBuildUp,
        TacticalPhase::AwayBuildUp => TacticalPhase::HomeBuildUp,
        TacticalPhase::HomeAttack => TacticalPhase::AwayAttack,
        TacticalPhase::AwayAttack => TacticalPhase::HomeAttack,
        other => other,
    }
}

fn mirror_facing_y(facing: FacingBucket) -> FacingBucket {
    match facing {
        FacingBucket::North => FacingBucket::South,
        FacingBucket::NorthEast => FacingBucket::SouthEast,
        FacingBucket::SouthEast => FacingBucket::NorthEast,
        FacingBucket::South => FacingBucket::North,
        FacingBucket::SouthWest => FacingBucket::NorthWest,
        FacingBucket::NorthWest => FacingBucket::SouthWest,
        other => other,
    }
}

fn mirror_target_state_y(mut state: SoccerQStateKey) -> SoccerQStateKey {
    state.phase = mirror_tactical_phase_y(state.phase);
    state.ball_zone_y = mirror_pitch_cell_y(state.ball_zone_y, 1, LEAGUE_TARGET_TACTICAL_ROWS);
    state.ball_fine_row =
        mirror_pitch_cell_y(state.ball_fine_row as usize, 1, LEAGUE_TARGET_FINE_ROWS) as u8;
    state.receive_facing = mirror_facing_y(state.receive_facing);
    state.action_facing = mirror_facing_y(state.action_facing);
    state
}

fn mirror_target_entry_y(entry: &SoccerQTargetEntry) -> SoccerQTargetEntry {
    let mut mirrored = entry.clone();
    mirrored.state = mirror_target_state_y(mirrored.state);
    mirrored.target_fine_cell_id = mirror_pitch_cell_y(
        mirrored.target_fine_cell_id,
        LEAGUE_TARGET_FINE_COLUMNS,
        LEAGUE_TARGET_FINE_ROWS,
    );
    mirrored.target_tactical_cell_id = mirror_pitch_cell_y(
        mirrored.target_tactical_cell_id,
        LEAGUE_TARGET_TACTICAL_COLUMNS,
        LEAGUE_TARGET_TACTICAL_ROWS,
    );
    mirrored.target_macro_cell_id = mirror_pitch_cell_y(
        mirrored.target_macro_cell_id,
        LEAGUE_TARGET_MACRO_COLUMNS,
        LEAGUE_TARGET_MACRO_ROWS,
    );
    mirrored
}

fn mirrored_target_entries(entries: &[SoccerQTargetEntry]) -> Vec<SoccerQTargetEntry> {
    entries.iter().map(mirror_target_entry_y).collect()
}

fn with_bidirectional_target_entries(brain: &TeamBrain) -> TeamBrain {
    let mut next = brain.clone();
    let mirrored_away_to_home = mirrored_target_entries(&brain.away_target_entries);
    let mirrored_home_to_away = mirrored_target_entries(&brain.home_target_entries);
    next.home_target_entries = merge_target_entries_idempotent(
        [brain.home_target_entries.clone(), mirrored_away_to_home],
        0,
    );
    next.away_target_entries = merge_target_entries_idempotent(
        [brain.away_target_entries.clone(), mirrored_home_to_away],
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

/// Federated / data-parallel averaging: each worker trained a CLONE of the same frontier on a
/// disjoint slice of the round's games; averaging their weights recombines the parallel work
/// into one net (all share identical architecture). training_steps accumulates total gradient
/// work across workers so warmth/step-count keep climbing.
fn average_snapshots(
    base_steps: usize,
    snaps: Vec<SoccerNeuralNetworkSnapshot>,
) -> Option<SoccerNeuralNetworkSnapshot> {
    if snaps.len() <= 1 {
        return snaps.into_iter().next();
    }
    let n = snaps.len() as f64;
    let mut out = snaps[0].clone();
    for (li, layer) in out.layers.iter_mut().enumerate() {
        for (ri, row) in layer.weights.iter_mut().enumerate() {
            for (wi, w) in row.iter_mut().enumerate() {
                *w = snaps
                    .iter()
                    .map(|s| s.layers[li].weights[ri][wi])
                    .sum::<f64>()
                    / n;
            }
        }
        for (bi, b) in layer.biases.iter_mut().enumerate() {
            *b = snaps.iter().map(|s| s.layers[li].biases[bi]).sum::<f64>() / n;
        }
    }
    let new_total: usize = snaps
        .iter()
        .map(|s| s.training_steps.saturating_sub(base_steps))
        .sum();
    out.training_steps = base_steps + new_total;
    let losses: Vec<f64> = snaps.iter().filter_map(|s| s.average_loss).collect();
    if !losses.is_empty() {
        out.average_loss = Some(losses.iter().sum::<f64>() / losses.len() as f64);
    }
    recompute_norm(&mut out);
    Some(out)
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
    round: u32,
    games: usize,
) -> LeagueRoundKpis {
    let mut validation = LeagueRoundKpis::default();
    let mut runner = runner.clone();
    println!("league_checkpoint_validation_start round={round} games={games}");
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
                    "league_checkpoint_validation_game round={round} game={} candidate_home={} score={}-{} forward_pass_margin={} objective_margin={:.3}",
                    g,
                    candidate_home,
                    outcome.home_goals,
                    outcome.away_goals,
                    kpis.forward_pass_margin,
                    kpis.objective_fitness_margin,
                );
                validation.add(kpis);
            }
            Ok(Err(err)) => {
                eprintln!("league_checkpoint_validation_error round={round} game={g}: {err}");
            }
            Err(_) => {
                eprintln!("league_checkpoint_validation_panic round={round} game={g}");
            }
        }
    }
    validation
}

fn checkpoint_validation_passes(
    validation: Option<&LeagueRoundKpis>,
    expected_games: usize,
    min_forward_pass_margin: f64,
    min_objective_margin: f64,
    min_goal_diff_margin: f64,
) -> bool {
    match validation {
        Some(validation) => {
            validation.games == expected_games
                && validation.mean_forward_pass_margin() > min_forward_pass_margin
                && validation.mean_objective_fitness_margin() > min_objective_margin
                && validation.mean_goal_diff() > min_goal_diff_margin
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
    let checkpoint_validate_min_objective_margin = env_f64(
        "SOCCER_LEAGUE_CHECKPOINT_VALIDATE_MIN_OBJECTIVE_MARGIN",
        0.0,
    );
    let checkpoint_validate_min_goal_diff_margin = env_f64(
        "SOCCER_LEAGUE_CHECKPOINT_VALIDATE_MIN_GOAL_DIFF_MARGIN",
        0.0,
    );
    let max_rounds = env_usize("SOCCER_LEAGUE_MAX_ROUNDS", 0);

    let mut runner_config = EngineMatchRunnerConfig::default();
    runner_config.base.duration_seconds = minutes * 60.0;
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
    // Network capacity: bigger hidden layer for more expressive value function over the
    // 610-dim field vector (the 24-unit net plateaued at parity with the analytic engine).
    // Only takes effect on a FRESH net — a resumed snapshot keeps its own architecture.
    runner_config.base.neural_learning.hidden_units = env_usize(
        "SOCCER_LEAGUE_HIDDEN_UNITS",
        runner_config.base.neural_learning.hidden_units,
    )
    .max(1);
    // Critic-target window (un-crush A/B). The critic target is `(reward + gamma*max_next) /
    // target_scale` clamped to +/-target_clip. With the defaults (30/3) the +/-200 win and +160
    // goal terminal rewards BOTH saturate the +3 rail, so the value head cannot rank win vs goal
    // vs a big win — the advantage baseline is blind above the rail. Raising `target_scale` (e.g.
    // 120) drops those labels below the clip so terminal outcomes separate again; `target_clip`
    // is exposed too for completeness. Unset ⇒ the config default (byte-identical).
    runner_config.base.neural_learning.target_scale = env_f64(
        "SOCCER_NEURAL_TARGET_SCALE",
        runner_config.base.neural_learning.target_scale,
    );
    runner_config.base.neural_learning.target_clip = env_f64(
        "SOCCER_NEURAL_TARGET_CLIP",
        runner_config.base.neural_learning.target_clip,
    );
    eprintln!(
        "[league] critic-target window: target_scale={} target_clip={} (raw-reward window +/-{}) popart={}",
        runner_config.base.neural_learning.target_scale,
        runner_config.base.neural_learning.target_clip,
        runner_config.base.neural_learning.target_scale
            * runner_config.base.neural_learning.target_clip,
        runner_config.base.neural_learning.target_popart_enabled,
    );
    // Un-collapse the value head on fresh nets: honour SOCCER_NEURAL_TARGET_POPART here (the
    // league bin otherwise leaves target-PopArt at the config default = off, unlike the
    // learning-run bins). A collapsed value can't rank candidates, which would make the Part-B
    // action-space A/B uninterpretable. Only takes effect on a FRESH net (same as hidden_units).
    runner_config.base.neural_learning.target_popart_enabled =
        std::env::var("SOCCER_NEURAL_TARGET_POPART")
            .ok()
            .map(|v| matches!(v.trim(), "1" | "true" | "TRUE" | "yes" | "on"))
            .unwrap_or(runner_config.base.neural_learning.target_popart_enabled);
    let target_standardization_enabled =
        env_default_bool("DD_SOCCER_ENABLE_TARGET_STANDARDIZATION", true);
    let mc_critic_target_enabled = env_default_bool("DD_SOCCER_ENABLE_MC_CRITIC_TARGET", true);
    let neural_self_bootstrap_enabled =
        env_default_bool("DD_SOCCER_ENABLE_NEURAL_SELF_BOOTSTRAP", true);
    let maxa_bootstrap_enabled = env_default_bool("DD_SOCCER_ENABLE_MAXA_BOOTSTRAP", true);
    let novelty_bonus_enabled = env_default_bool("DD_SOCCER_ENABLE_NOVELTY_BONUS", true);
    let forward_pass_climb_curriculum_enabled =
        env_default_bool("DD_SOCCER_FORWARD_PASS_CLIMB_CURRICULUM", true);
    runner_config.base.neural_learning.lp_coupling_enabled =
        env_bool("SOCCER_NEURAL_LP_COUPLING_ENABLED", true);
    // POLICY-IMPROVEMENT LEVER: opt the fresh net into the actor-critic path. When on, the neural
    // ACTOR π(family|s) is trained by advantage policy-gradient and biases action selection on top
    // of the value blend — the "close the policy-improvement loop" escape from value-imitation
    // parity (policy gradient optimizes RETURN directly, not tabular-matching). Pair with a lower
    // DD_SOCCER_NEURAL_AUTHORITATIVE_LAMBDA (more actor influence) + SOCCER_POLICY_ENTROPY_COEFF
    // (keep exploring) so the actor can actually leave the analytic mode. Off ⇒ current behaviour.
    runner_config.base.neural_blend.actor_critic = env_bool("SOCCER_NEURAL_ACTOR_CRITIC", true);
    let league_neural_mcts_enabled = apply_league_neural_mcts_config(&mut runner_config);
    let mpc_tier2_enabled = runner_config.base.mpc.tier2_player_enabled;
    let mpc_reconcile_enabled = runner_config.base.mpc.reconcile_enabled;
    let mpc_field_aware_enabled = runner_config.base.mpc.field_aware_enabled;
    let mpc_latent_objective_enabled = runner_config.base.mpc.latent_objective_enabled;
    let local_mpc_enabled = runner_config.base.local_mpc_enabled;
    let actor_critic_enabled = runner_config.base.neural_blend.actor_critic;
    let lp_coupling_enabled = runner_config.base.neural_learning.lp_coupling_enabled;
    // Keep the engine's designed independent-brain mode (per-team critic drives each side).
    let runner = EngineMatchRunner::new(runner_config);

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

    let train_seed_base: u32 = 0x5EED_0000;
    let mut round = 0u32;
    let mut best_checkpoint_forward_pass_margin = f64::NEG_INFINITY;
    println!(
        "league_train_started_at_utc={} games/opp={} minutes={} weight_decay={} fresh_opp={} checkpoint_every={} max_rounds={} max_target_entries_per_side={} advancement_metric=completed_forward_passes league_neural_mcts_enabled={} actor_critic_enabled={} lp_coupling_enabled={} target_standardization_enabled={} mc_critic_target_enabled={} neural_self_bootstrap_enabled={} maxa_bootstrap_enabled={} novelty_bonus_enabled={} forward_pass_climb_curriculum_enabled={} mpc_tier2_enabled={} mpc_reconcile_enabled={} mpc_field_aware_enabled={} mpc_latent_objective_enabled={} local_mpc_enabled={} checkpoint_require_forward_pass_climb={} checkpoint_max_forward_pass_regression={} checkpoint_min_forward_pass_margin={} checkpoint_validate_games={} checkpoint_validate_min_forward_pass_margin={} checkpoint_validate_min_objective_margin={} checkpoint_validate_min_goal_diff_margin={} frontier={} candidate_frontier={} archive={}",
        chrono_now(),
        games_per_opp,
        minutes,
        weight_decay,
        fresh_opponents,
        checkpoint_every,
        max_rounds,
        max_target_entries_per_side,
        league_neural_mcts_enabled,
        actor_critic_enabled,
        lp_coupling_enabled,
        target_standardization_enabled,
        mc_critic_target_enabled,
        neural_self_bootstrap_enabled,
        maxa_bootstrap_enabled,
        novelty_bonus_enabled,
        forward_pass_climb_curriculum_enabled,
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
        checkpoint_validate_min_objective_margin,
        checkpoint_validate_min_goal_diff_margin,
        frontier_path,
        candidate_frontier_path,
        archive_dir
    );

    loop {
        round += 1;
        let started = Instant::now();
        let checkpoint_baseline = frontier.clone();

        // Build this round's opponent league: frozen champions + fresh (exploration).
        let mut opponents: Vec<TeamBrain> = load_league(&archive_dir);
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
        if std::env::var("SOCCER_LEAGUE_ANALYTIC_OPPONENTS")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
        {
            for opp in opponents.iter_mut() {
                opp.neural = None;
            }
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
                let home = frontier_always_home || g % 2 == 0;
                fixtures.push(Fx {
                    opp_idx: idx,
                    opp_id: idx + 1,
                    seed,
                    home,
                });
                fixture += 1;
            }
        }
        let workers = env_usize("SOCCER_LEAGUE_PARALLELISM", 3).clamp(1, fixtures.len().max(1));
        let base_steps = frontier
            .neural
            .as_ref()
            .map(|s| s.training_steps)
            .unwrap_or(0);
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
        let wins = wins_a.load(std::sync::atomic::Ordering::Relaxed);
        let losses = losses_a.load(std::sync::atomic::Ordering::Relaxed);
        let trained_brains = trained_brains.into_inner().unwrap();
        let mut round_kpis = LeagueRoundKpis::default();
        for kpis in match_kpis.into_inner().unwrap() {
            round_kpis.add(kpis);
        }
        let trained_snapshots = trained_brains
            .iter()
            .filter_map(|brain| brain.neural.clone())
            .collect::<Vec<_>>();
        if let Some(avg) = average_snapshots(base_steps, trained_snapshots) {
            frontier.neural = Some(avg);
        }
        if !trained_brains.is_empty() {
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
        }

        let kpi_games = round_kpis.games.max(1) as f64;
        let pass_completion = if round_kpis.passes_attempted_for == 0 {
            0.0
        } else {
            round_kpis.passes_completed_for as f64 / round_kpis.passes_attempted_for as f64
        };
        let mean_kpi_margin = round_kpis.mean_objective_fitness_margin();
        let mean_forward_pass_margin = round_kpis.mean_forward_pass_margin();
        println!(
            "league_advancement round={round} metric=completed_forward_passes games={} forward_passes_for={} forward_passes_against={} forward_pass_margin={} forward_pass_margin_per_game={:.3}",
            round_kpis.games,
            round_kpis.forward_passes_for,
            round_kpis.forward_passes_against,
            round_kpis.forward_pass_margin,
            mean_forward_pass_margin,
        );
        println!(
            "league_kpi round={round} games={} advancement_metric=completed_forward_passes forward_pass_margin_per_game={:.3} objective={:.3} objective_margin={:.3} gd_total={} gd_per_game={:.2} goals_for={} goals_against={} shots_for={} shots_against={} sot_for={} sot_against={} shots_after_pass={} pass_completion={:.3} completed_passes={} forward_passes={} backward_passes={} dribble_beats={} assists={} crosses={} chain_gain_yards={:.1} chain_net_losses={}",
            round_kpis.games,
            mean_forward_pass_margin,
            round_kpis.mean_objective_fitness(),
            mean_kpi_margin,
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
        );

        // Temporal checkpoint into the league (growing diversity of past selves).
        if round % (checkpoint_every as u32) == 0 {
            if frontier.neural.is_some() {
                let prior_best = best_checkpoint_forward_pass_margin;
                let validation = if checkpoint_validate_games > 0 {
                    let checkpoint_incumbent =
                        load_brain(&frontier_path).unwrap_or_else(|| checkpoint_baseline.clone());
                    let validation = play_checkpoint_validation(
                        &runner,
                        &frontier,
                        &checkpoint_incumbent,
                        round,
                        checkpoint_validate_games,
                    );
                    println!(
                        "league_checkpoint_validation round={round} games={} forward_pass_margin_per_game={:.3} objective_margin={:.3} gd_total={} gd_per_game={:.3} shots_after_pass={} completed_passes={} pass_gain_yards={:.1}",
                        validation.games,
                        validation.mean_forward_pass_margin(),
                        validation.mean_objective_fitness_margin(),
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
                let validation_forward_pass_margin =
                    validation.as_ref().map(|v| v.mean_forward_pass_margin());
                let gate_forward_pass_margin =
                    validation_forward_pass_margin.unwrap_or(mean_forward_pass_margin);
                let passes_forward_pass_climb = !checkpoint_require_forward_pass_climb
                    || !prior_best.is_finite()
                    || gate_forward_pass_margin + checkpoint_max_forward_pass_regression
                        >= prior_best;
                let passes_forward_pass_floor =
                    gate_forward_pass_margin > checkpoint_min_forward_pass_margin;
                let passes_validation = checkpoint_validation_passes(
                    validation.as_ref(),
                    checkpoint_validate_games,
                    checkpoint_validate_min_forward_pass_margin,
                    checkpoint_validate_min_objective_margin,
                    checkpoint_validate_min_goal_diff_margin,
                );
                if passes_forward_pass_climb && passes_forward_pass_floor && passes_validation {
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
                    if gate_forward_pass_margin > best_checkpoint_forward_pass_margin {
                        best_checkpoint_forward_pass_margin = gate_forward_pass_margin;
                    }
                    println!(
                        "league_checkpoint round={round} metric=completed_forward_passes gate_forward_pass_margin_per_game={gate_forward_pass_margin:.3} train_forward_pass_margin_per_game={mean_forward_pass_margin:.3} best_forward_pass_margin_per_game={best_checkpoint_forward_pass_margin:.3} -> {cp}"
                    );
                } else {
                    println!(
                        "league_checkpoint_held round={round} metric=completed_forward_passes gate_forward_pass_margin_per_game={gate_forward_pass_margin:.3} train_forward_pass_margin_per_game={mean_forward_pass_margin:.3} best_forward_pass_margin_per_game={prior_best:.3} require_climb={} min_margin={checkpoint_min_forward_pass_margin:.3} validation_passed={passes_validation} validation_games={}/{} validation_min_goal_diff_margin={checkpoint_validate_min_goal_diff_margin:.3}",
                        checkpoint_require_forward_pass_climb,
                        validation.as_ref().map(|validation| validation.games).unwrap_or(0),
                        checkpoint_validate_games,
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

        let summary = MatchSummary {
            score_home: 0,
            score_away: 4,
            ticks: 0,
            simulated_seconds: 0.0,
            stats,
        };

        let home = LeagueMatchKpis::from_summary(&summary, Team::Home);
        assert_eq!(home.objective_fitness, 14.0);
        assert_eq!(home.objective_fitness_margin, 9.0);
        assert_eq!(home.forward_pass_margin, 9);

        let away = LeagueMatchKpis::from_summary(&summary, Team::Away);
        assert_eq!(away.objective_fitness, 5.0);
        assert_eq!(away.objective_fitness_margin, -9.0);
        assert_eq!(away.forward_pass_margin, -9);
    }

    #[test]
    fn checkpoint_validation_blocks_forward_pass_smoke_without_positive_gd() {
        let mut validation = LeagueRoundKpis::default();
        validation.add(LeagueMatchKpis {
            goal_diff: 0,
            forward_pass_margin: 5,
            objective_fitness_margin: 5.0,
            ..LeagueMatchKpis::default()
        });

        assert!(!checkpoint_validation_passes(
            Some(&validation),
            1,
            0.0,
            0.0,
            0.0,
        ));

        validation.goal_diff = 1;
        assert!(checkpoint_validation_passes(
            Some(&validation),
            1,
            0.0,
            0.0,
            0.0,
        ));
    }

    #[test]
    fn target_entry_mirror_flips_pitch_rows_and_home_away_phases() {
        assert_eq!(
            mirror_tactical_phase_y(TacticalPhase::HomeAttack),
            TacticalPhase::AwayAttack
        );
        assert_eq!(
            mirror_tactical_phase_y(TacticalPhase::AwayBuildUp),
            TacticalPhase::HomeBuildUp
        );
        assert_eq!(
            mirror_facing_y(FacingBucket::NorthEast),
            FacingBucket::SouthEast
        );
        assert_eq!(
            mirror_facing_y(FacingBucket::SouthWest),
            FacingBucket::NorthWest
        );
        assert_eq!(
            mirror_pitch_cell_y(
                20 * LEAGUE_TARGET_FINE_COLUMNS + 4,
                LEAGUE_TARGET_FINE_COLUMNS,
                LEAGUE_TARGET_FINE_ROWS,
            ),
            3 * LEAGUE_TARGET_FINE_COLUMNS + 4
        );
        assert_eq!(
            mirror_pitch_cell_y(
                6 * LEAGUE_TARGET_TACTICAL_COLUMNS + 2,
                LEAGUE_TARGET_TACTICAL_COLUMNS,
                LEAGUE_TARGET_TACTICAL_ROWS,
            ),
            LEAGUE_TARGET_TACTICAL_COLUMNS + 2
        );
        assert_eq!(
            mirror_pitch_cell_y(
                3 * LEAGUE_TARGET_MACRO_COLUMNS + 1,
                LEAGUE_TARGET_MACRO_COLUMNS,
                LEAGUE_TARGET_MACRO_ROWS,
            ),
            1
        );

        let mirrored = mirror_target_state_y(mirror_test_state());
        assert_eq!(mirrored.phase, TacticalPhase::AwayAttack);
        assert_eq!(mirrored.ball_zone_y, 1);
        assert_eq!(mirrored.ball_fine_row, 3);
        assert_eq!(mirrored.receive_facing, FacingBucket::SouthEast);
        assert_eq!(mirrored.action_facing, FacingBucket::NorthWest);
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
    fn league_trainer_neural_mcts_requires_explicit_league_opt_in() {
        let _lock = env_lock().lock().unwrap();
        let _guard = EnvVarGuard::set("SOCCER_LEAGUE_NEURAL_MCTS_ENABLED", "1");
        let mut config = EngineMatchRunnerConfig::default();

        let enabled = apply_league_neural_mcts_config(&mut config);

        assert!(enabled);
        assert!(config.base.neural_blend.mcts_enabled);
    }
}
