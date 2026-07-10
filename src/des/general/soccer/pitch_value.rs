//! Spatial **pitch-control × expected-threat** value model (2D).
//!
//! The rest of the engine's learning rewards are *local, hand-crafted events*
//! (a completed pass, a cross into the box, a blocked-lane penalty). None of
//! them credit a player for the thing soccer is actually about off the ball:
//! *taking control of valuable space*. A winger who pulls a full-back wide, a
//! striker who pins the last defender, a midfielder who occupies the half-space
//! — these change the team's territorial control of dangerous areas without ever
//! touching the ball, and the local rewards are blind to them.
//!
//! This module gives the learner one **dense, field-wide signal** for that:
//!
//! 1. [`pitch_control_home`] — for any point, the probability the **home** team
//!    controls it, from a velocity-aware *time-to-arrive* race between the two
//!    teams' nearest players, softened through a logistic (a 2D simplification of
//!    Spearman-style pitch control). Away control is `1 - home`.
//! 2. [`expected_threat`] — a closed-form *value of possessing the ball here*
//!    surface, oriented to a team's attacking direction: it rises steeply toward
//!    the opponent goal and is weighted up centrally in the shooting zones. This
//!    is a **seed** the learners may later replace with a learned Markov xT grid;
//!    it is deliberately simple, deterministic, and RNG-free.
//! 3. [`team_expected_threat`] — the integral over a grid of
//!    `control(team) × expected_threat(team)`: a single scalar for how much
//!    *dangerous space the team controls right now*.
//!
//! [`pitch_value_reward_delta`] turns that into a learning reward: the change in
//! the acting team's **net** territorial threat (its own minus the opponent's)
//! between the before/after snapshots of a transition. Any action — pass,
//! dribble, or pure off-ball run — that grows the team's grip on valuable space
//! is credited; any that cedes it is penalized.
//!
//! ## Parity
//!
//! The reward is **gated off by default** (`DD_SOCCER_ENABLE_PITCH_VALUE_REWARD`).
//! When unset, [`pitch_value_reward_delta`] returns `0.0` before touching the
//! grid, so an unconfigured process is byte-identical to before this module
//! existed. The model functions themselves are pure and side-effect-free, so the
//! inspector / tests can read the surface without enabling the reward.

use super::*;

/// Grid resolution across the pitch width (x axis). Kept modest: the reward path
/// evaluates the whole grid twice per transition (before + after).
const PITCH_VALUE_GRID_COLS: usize = 10;
/// Grid resolution along the pitch length (y axis, goal-to-goal).
const PITCH_VALUE_GRID_ROWS: usize = 16;

/// Logistic temperature (seconds) on the time-to-arrive advantage. Larger = a
/// softer, more shared control field; smaller = sharper hand-off at the midline
/// between the two nearest players.
const PITCH_CONTROL_TEMPERATURE_SECONDS: f64 = 0.45;
/// How many seconds of current momentum count as a head start in the arrival
/// race: a player already sprinting toward a cell reaches it sooner than a
/// stationary one the same distance away.
const PITCH_CONTROL_MOMENTUM_SECONDS: f64 = 0.6;
/// Floor on a player's modeled top speed (yd/s) so a missing/degenerate skill
/// profile can never divide the arrival time by ~0.
const PITCH_CONTROL_MIN_TOP_SPEED_YPS: f64 = 4.0;

/// Steepness of the territorial value ramp toward the opponent goal. Cube keeps
/// midfield cheap and the final third expensive.
const EXPECTED_THREAT_FORWARD_EXPONENT: f64 = 3.0;
/// Extra weight on central, advanced cells (the shooting zones).
const EXPECTED_THREAT_SHOOTING_WEIGHT: f64 = 0.8;
/// How sharply the shooting-zone bonus concentrates near the opponent goal.
const EXPECTED_THREAT_SHOOTING_EXPONENT: f64 = 4.0;

/// Env gate enabling the dense pitch-value reward term. Off (unset) by default so
/// an unconfigured process stays byte-identical.
const PITCH_VALUE_REWARD_ENABLE_ENV: &str = "DD_SOCCER_ENABLE_PITCH_VALUE_REWARD";

/// Env gate enabling the xT **terminal-cost** shaping of the per-player MPC
/// reference (the "cost-to-go" wire). Off (unset) by default so an unconfigured
/// process is byte-identical: [`xt_terminal_shaped_target`] returns its input.
const XT_TERMINAL_COST_ENABLE_ENV: &str = "DD_SOCCER_ENABLE_XT_TERMINAL_COST";

/// Largest distance (yards) the xT terminal shaping may move a player's MPC
/// reference point. The LP/support search owns the destination; this is a bounded
/// nudge toward more valuable controllable space, never an override.
const XT_TERMINAL_MAX_STEP_YARDS: f64 = 2.5;
/// Finite-difference probe radius (yards) for the control-weighted-threat
/// gradient. Comfortably larger than the value surface's local noise.
const XT_TERMINAL_PROBE_YARDS: f64 = 2.0;
/// Gradient magnitude (value per yard) that earns the full
/// [`XT_TERMINAL_MAX_STEP_YARDS`] step. Below this the step scales down linearly,
/// so flat regions of the value surface barely move.
const XT_TERMINAL_FULL_STEP_GRADIENT: f64 = 0.010;

fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|raw| {
            let v = raw.trim().to_ascii_lowercase();
            matches!(v.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

/// Whether the dense territorial pitch-value reward is enabled this process.
pub fn pitch_value_reward_enabled() -> bool {
    #[cfg(test)]
    {
        env_flag_enabled(PITCH_VALUE_REWARD_ENABLE_ENV)
    }
    #[cfg(not(test))]
    {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        // Promoted to default-ON in production (pitch-value PBRS reward; training-time only).
        *ENABLED.get_or_init(|| gate_default_on(PITCH_VALUE_REWARD_ENABLE_ENV))
    }
}

/// Whether the xT terminal-cost MPC shaping is enabled this process. Off (unset)
/// by default ⇒ [`xt_terminal_shaped_target`] is the identity, so the per-player
/// MPC reference is byte-identical to before this wire existed.
pub fn xt_terminal_cost_enabled() -> bool {
    #[cfg(test)]
    {
        env_flag_enabled(XT_TERMINAL_COST_ENABLE_ENV)
    }
    #[cfg(not(test))]
    {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        // Promoted to default-ON in production (xT terminal-cost MPC shaping).
        *ENABLED.get_or_init(|| gate_default_on(XT_TERMINAL_COST_ENABLE_ENV))
    }
}

/// Forward progress fraction in `[0, 1]` for `team` at point `p`: 0 on the team's
/// own goal line, 1 on the opponent goal line. Orientation-symmetric between the
/// two teams (mirror a point and swap the team and you get the same value).
fn forward_fraction(team: Team, p: Vec2, field_length: f64) -> f64 {
    if field_length <= 0.0 || !field_length.is_finite() || !p.y.is_finite() {
        return 0.5;
    }
    let centered = (p.y - field_length * 0.5) / field_length;
    (0.5 + team.attack_dir() * centered).clamp(0.0, 1.0)
}

/// Closed-form **expected threat** of possessing the ball at `p` for `team`,
/// oriented to that team's attacking direction. Bounded to roughly `[0, 1.8]`;
/// only relative values matter (the reward uses differences).
pub fn expected_threat(team: Team, p: Vec2, field_width: f64, field_length: f64) -> f64 {
    // Move 1a (Anchor-3 cut): when DD_SOCCER_ENABLE_LEARNED_EPV_POTENTIAL is on, the PBRS potential
    // becomes the OUTCOME-GROUNDED learned EPV (rescaled to the closed-form range) instead of the
    // hand-built xT seed. Advancing into a historically-converting cell is then worth more than sterile
    // territory, and the potential no longer telescopes to ~zero over an unconverted possession —
    // directly attacking "territory without conversion". Falls back to the closed-form seed when the
    // gate is off or no grid is loaded.
    if dd_soccer_enable_learned_epv_potential() {
        if let Some(epv) = learned_epv_raw(team, p, field_width, field_length) {
            return (epv - LEARNED_EPV_POTENTIAL_MIN) * learned_epv_potential_scale();
        }
    }
    let fwd = forward_fraction(team, p, field_length);
    let territorial = fwd.powf(EXPECTED_THREAT_FORWARD_EXPONENT);
    // Central-ness in [0,1]: 1 on the spine of the pitch, 0 at the touchline.
    let centrality = if field_width > 0.0 && field_width.is_finite() {
        if p.x.is_finite() {
            (1.0 - (p.x - field_width * 0.5).abs() / (field_width * 0.5)).clamp(0.0, 1.0)
        } else {
            0.5
        }
    } else {
        0.5
    };
    let shooting =
        EXPECTED_THREAT_SHOOTING_WEIGHT * fwd.powf(EXPECTED_THREAT_SHOOTING_EXPONENT) * centrality;
    territorial + shooting
}

/// Learned EPV (expected possession value) grid — the calibrated replacement for the hardcoded
/// `expected_threat` seed. Fit offline by `scripts/fit_epv_grid.py` from outcome-labeled possession
/// chains: `grid[forward_bin][lateral_bin] = E[gamma^ticks_to_terminal * value(terminal_outcome)]`,
/// so a cell near the opponent box (where chains end in shots/goals) scores high and a deep/own-half
/// cell scores low/negative. Loaded once from `DD_SOCCER_LEARNED_EPV_GRID_PATH`; gated default-off.
#[derive(Debug, Clone, Deserialize)]
struct LearnedEpvGrid {
    rows: usize,
    cols: usize,
    grid: Vec<Vec<f64>>,
}

pub(crate) fn dd_soccer_enable_learned_epv() -> bool {
    use std::sync::OnceLock;
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| {
        matches!(
            std::env::var("DD_SOCCER_ENABLE_LEARNED_EPV")
                .ok()
                .as_deref()
                .map(str::trim),
            Some("1") | Some("true") | Some("yes") | Some("on")
        )
    })
}

/// Move 1a gate: use the learned EPV grid as the PBRS territorial POTENTIAL (`expected_threat`),
/// replacing the hand-built xT seed. Independent of the per-pass `DD_SOCCER_ENABLE_LEARNED_EPV` bonus.
pub(crate) fn dd_soccer_enable_learned_epv_potential() -> bool {
    use std::sync::OnceLock;
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| {
        matches!(
            std::env::var("DD_SOCCER_ENABLE_LEARNED_EPV_POTENTIAL")
                .ok()
                .as_deref()
                .map(str::trim),
            Some("1") | Some("true") | Some("yes") | Some("on")
        )
    })
}

/// Grid floor (own-half / high-turnover cells) — shift the learned EPV by this so the resulting
/// potential is >= 0 and on a scale comparable to the closed-form `expected_threat` ([0, ~1.8]).
const LEARNED_EPV_POTENTIAL_MIN: f64 = -0.15;

/// Scale mapping learned-EPV units (~[-0.15, 0.20]) to the closed-form potential range. Default 5.0
/// gives ~[0, 1.75]. Env `DD_SOCCER_LEARNED_EPV_POTENTIAL_SCALE`.
fn learned_epv_potential_scale() -> f64 {
    use std::sync::OnceLock;
    static V: OnceLock<f64> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("DD_SOCCER_LEARNED_EPV_POTENTIAL_SCALE")
            .ok()
            .and_then(|r| r.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v > 0.0)
            .unwrap_or(5.0)
            .clamp(0.1, 100.0)
    })
}

fn learned_epv_grid() -> Option<&'static LearnedEpvGrid> {
    use std::sync::OnceLock;
    static GRID: OnceLock<Option<LearnedEpvGrid>> = OnceLock::new();
    GRID.get_or_init(|| {
        let path = std::env::var("DD_SOCCER_LEARNED_EPV_GRID_PATH").ok()?;
        let raw = std::fs::read_to_string(path.trim()).ok()?;
        let g: LearnedEpvGrid = serde_json::from_str(&raw).ok()?;
        if g.rows == 0 || g.cols == 0 || g.grid.len() != g.rows {
            return None;
        }
        Some(g)
    })
    .as_ref()
}

/// Learned expected-possession-value of holding the ball at `p` for `team`, from the fitted grid.
/// `None` when the gate is off or no grid is loaded (callers fall back to the closed-form seed).
pub(crate) fn learned_epv(team: Team, p: Vec2, field_width: f64, field_length: f64) -> Option<f64> {
    if !dd_soccer_enable_learned_epv() {
        return None;
    }
    let g = learned_epv_grid()?;
    let fwd = forward_fraction(team, p, field_length); // 0 = own goal .. 1 = opponent goal
    let lat = if field_width > 0.0 && field_width.is_finite() && p.x.is_finite() {
        (p.x / field_width).clamp(0.0, 1.0)
    } else {
        0.5
    };
    let row = ((fwd * g.rows as f64) as usize).min(g.rows.saturating_sub(1));
    let col = ((lat * g.cols as f64) as usize).min(g.cols.saturating_sub(1));
    g.grid.get(row).and_then(|r| r.get(col)).copied()
}

/// ΔEPV of a pass from `origin` to `target` for `team`: how much learned possession value (danger)
/// the ball gains. Rewards completed passes by the CHANCE they create, not raw territory — the fix
/// for "forward passes produce territory + draws, not conversion". `0.0` when the grid is
/// unavailable (byte-identical / no-op), so callers add it unconditionally.
pub(crate) fn learned_epv_pass_delta(
    team: Team,
    origin: Vec2,
    target: Vec2,
    field_width: f64,
    field_length: f64,
) -> f64 {
    match (
        learned_epv(team, target, field_width, field_length),
        learned_epv(team, origin, field_width, field_length),
    ) {
        (Some(t), Some(o)) => t - o,
        _ => 0.0,
    }
}

/// Modeled top speed (yd/s) for a snapshot player, floored so the arrival-time
/// race can never divide by ~0.
fn player_top_speed(player: &PlayerSnapshot) -> f64 {
    let speed = super::player_top_speed_yps(player.role, &player.skills);
    if speed.is_finite() {
        speed.max(PITCH_CONTROL_MIN_TOP_SPEED_YPS)
    } else {
        PITCH_CONTROL_MIN_TOP_SPEED_YPS
    }
}

/// Velocity-aware time (seconds) for a body at `pos` moving at `vel` with modeled
/// top speed `vmax` to arrive at `cell`. A body already moving toward the cell
/// gets a momentum head start; one moving away pays for it. The single source of
/// truth for the arrival race, shared by the snapshot and lightweight-kinematic
/// control paths.
fn arrival_time_core(pos: Vec2, vel: Vec2, vmax: f64, cell: Vec2) -> f64 {
    if !cell.x.is_finite() || !cell.y.is_finite() || !pos.x.is_finite() || !pos.y.is_finite() {
        return f64::INFINITY;
    }
    let to_cell = cell - pos;
    let dist = to_cell.len();
    if !dist.is_finite() {
        return f64::INFINITY;
    }
    let vmax = if vmax.is_finite() {
        vmax.max(PITCH_CONTROL_MIN_TOP_SPEED_YPS)
    } else {
        PITCH_CONTROL_MIN_TOP_SPEED_YPS
    };
    if dist <= 1e-6 {
        return 0.0;
    }
    let dir = Vec2::new(to_cell.x / dist, to_cell.y / dist);
    let vel_toward = vel.dot(dir);
    let vel_toward = if vel_toward.is_finite() {
        vel_toward
    } else {
        0.0
    };
    let head_start = vel_toward * PITCH_CONTROL_MOMENTUM_SECONDS;
    let effective_dist = (dist - head_start).max(0.0);
    effective_dist / vmax
}

/// Velocity-aware time (seconds) for `player` to arrive at `cell`.
fn arrival_time_seconds(player: &PlayerSnapshot, cell: Vec2) -> f64 {
    arrival_time_core(
        player.position,
        player.velocity,
        player_top_speed(player),
        cell,
    )
}

/// Minimum arrival time over a team's players, or `None` if the team has nobody.
fn team_min_arrival(players: &[PlayerSnapshot], team: Team, cell: Vec2) -> Option<f64> {
    players
        .iter()
        .filter(|p| p.team == team)
        .map(|p| arrival_time_seconds(p, cell))
        .filter(|t| t.is_finite())
        .fold(None, |acc: Option<f64>, t| {
            Some(acc.map_or(t, |best| best.min(t)))
        })
}

/// Probability the **home** team controls `cell`, from the time-to-arrive race
/// (logistic on the home-minus-away advantage). `0.5` when neither team can be
/// scored (e.g. an empty side), so it stays neutral rather than NaN.
pub fn pitch_control_home(players: &[PlayerSnapshot], cell: Vec2) -> f64 {
    let home = team_min_arrival(players, Team::Home, cell);
    let away = team_min_arrival(players, Team::Away, cell);
    match (home, away) {
        (Some(h), Some(a)) => {
            // Home controls more when it arrives sooner (h < a).
            let advantage = (a - h) / PITCH_CONTROL_TEMPERATURE_SECONDS;
            1.0 / (1.0 + (-advantage).exp())
        }
        (Some(_), None) => 1.0,
        (None, Some(_)) => 0.0,
        (None, None) => 0.5,
    }
}

/// Integral of `control(team) × expected_threat(team)` over the grid — a single
/// scalar for how much **dangerous space the team currently controls**. Averaged
/// over cells, so it is bounded on the same scale as [`expected_threat`].
pub fn team_expected_threat(snapshot: &WorldSnapshot, team: Team) -> f64 {
    let field_width = snapshot.field_width;
    let field_length = snapshot.field_length;
    if field_width <= 0.0
        || field_length <= 0.0
        || !field_width.is_finite()
        || !field_length.is_finite()
    {
        return 0.0;
    }
    let players = &snapshot.players;
    let mut total = 0.0;
    let cols = PITCH_VALUE_GRID_COLS;
    let rows = PITCH_VALUE_GRID_ROWS;
    for j in 0..rows {
        let y = (j as f64 + 0.5) / rows as f64 * field_length;
        for i in 0..cols {
            let x = (i as f64 + 0.5) / cols as f64 * field_width;
            let cell = Vec2::new(x, y);
            let home_control = pitch_control_home(players, cell);
            let control = match team {
                Team::Home => home_control,
                Team::Away => 1.0 - home_control,
            };
            let threat = expected_threat(team, cell, field_width, field_length);
            total += control * threat;
        }
    }
    total / (cols * rows) as f64
}

/// Net territorial threat for `team`: its controlled threat minus the opponent's.
/// Positive means the team grips more dangerous space than it concedes.
pub fn territorial_advantage(snapshot: &WorldSnapshot, team: Team) -> f64 {
    team_expected_threat(snapshot, team) - team_expected_threat(snapshot, team.other())
}

/// A lightweight kinematic stand-in for a player when computing pitch control off
/// the hot path (the live MPC tick holds `PlayerAgent`s, not `PlayerSnapshot`s).
/// Carries only what the time-to-arrive race needs.
#[derive(Clone, Copy, Debug)]
pub struct XtControlPoint {
    pub team: Team,
    pub position: Vec2,
    pub velocity: Vec2,
    /// Modeled top speed (yd/s); floored internally so a degenerate value is safe.
    pub top_speed: f64,
}

/// Minimum arrival time over a team's [`XtControlPoint`]s, or `None` if the team
/// has nobody scoreable.
fn team_min_arrival_points(points: &[XtControlPoint], team: Team, cell: Vec2) -> Option<f64> {
    points
        .iter()
        .filter(|p| p.team == team)
        .map(|p| arrival_time_core(p.position, p.velocity, p.top_speed, cell))
        .filter(|t| t.is_finite())
        .fold(None, |acc: Option<f64>, t| {
            Some(acc.map_or(t, |best| best.min(t)))
        })
}

/// Probability the home team controls `cell`, from a slice of lightweight
/// [`XtControlPoint`]s (mirrors [`pitch_control_home`]).
fn pitch_control_home_points(points: &[XtControlPoint], cell: Vec2) -> f64 {
    let home = team_min_arrival_points(points, Team::Home, cell);
    let away = team_min_arrival_points(points, Team::Away, cell);
    match (home, away) {
        (Some(h), Some(a)) => {
            let advantage = (a - h) / PITCH_CONTROL_TEMPERATURE_SECONDS;
            1.0 / (1.0 + (-advantage).exp())
        }
        (Some(_), None) => 1.0,
        (None, Some(_)) => 0.0,
        (None, None) => 0.5,
    }
}

/// Control-weighted threat **value** of `team` arriving at `p`: the closed-form
/// [`expected_threat`] gated by the probability `team` actually controls that
/// point (the time-to-arrive race). This is the cost-to-go surface the xT
/// terminal shaping ascends — a player is only pulled toward valuable space it
/// can realistically grip, not toward unreachable danger zones.
fn control_weighted_threat(points: &[XtControlPoint], team: Team, p: Vec2, w: f64, l: f64) -> f64 {
    let home_control = pitch_control_home_points(points, p);
    let control = match team {
        Team::Home => home_control,
        Team::Away => 1.0 - home_control,
    };
    control * expected_threat(team, p, w, l)
}

/// xT **terminal-cost** shaping of a per-player MPC reference point (the AV
/// "cost-to-go" wire). Given the destination the LP / support search already
/// chose (`reference`), nudge it by at most [`XT_TERMINAL_MAX_STEP_YARDS`] up the
/// gradient of [`control_weighted_threat`], so the controlled player settles in
/// the most valuable *controllable* space within reach of its assigned slot
/// rather than the bare geometric point. The step is bounded and gradient-scaled,
/// so it nudges and never overrides the destination — honouring the
/// "LP decides the shape, MPC executes" division of labour.
///
/// Returns `reference` unchanged when the gate is off, the inputs are degenerate,
/// or the local value surface is flat — keeping the default path byte-identical.
pub fn xt_terminal_shaped_target(
    points: &[XtControlPoint],
    team: Team,
    reference: Vec2,
    field_width: f64,
    field_length: f64,
) -> Vec2 {
    if !xt_terminal_cost_enabled() {
        return reference;
    }
    if !reference.x.is_finite()
        || !reference.y.is_finite()
        || field_width <= 0.0
        || field_length <= 0.0
        || !field_width.is_finite()
        || !field_length.is_finite()
    {
        return reference;
    }
    let h = XT_TERMINAL_PROBE_YARDS;
    let v = |p: Vec2| control_weighted_threat(points, team, p, field_width, field_length);
    // Central finite-difference gradient of the value surface at `reference`.
    let gx = (v(Vec2::new(reference.x + h, reference.y))
        - v(Vec2::new(reference.x - h, reference.y)))
        / (2.0 * h);
    let gy = (v(Vec2::new(reference.x, reference.y + h))
        - v(Vec2::new(reference.x, reference.y - h)))
        / (2.0 * h);
    if !gx.is_finite() || !gy.is_finite() {
        return reference;
    }
    let mag = (gx * gx + gy * gy).sqrt();
    if mag <= 1e-9 {
        return reference;
    }
    // Step length scales with gradient strength (capped): flat space barely moves,
    // a steep value ridge earns the full bounded step.
    let step = XT_TERMINAL_MAX_STEP_YARDS * (mag / XT_TERMINAL_FULL_STEP_GRADIENT).clamp(0.0, 1.0);
    let shaped = Vec2::new(reference.x + gx / mag * step, reference.y + gy / mag * step);
    finite_pitch_point(shaped, field_width, field_length, reference)
}

/// Dense learning reward for the acting `team`: the change in its **net**
/// territorial threat from `before` to `after`, scaled by the tunable
/// `reward.pitch_value_threat_delta_points`.
///
/// Returns `0.0` (without touching the grid) unless
/// [`pitch_value_reward_enabled`] is set, keeping the default process
/// byte-identical.
pub fn pitch_value_reward_delta(before: &WorldSnapshot, after: &WorldSnapshot, team: Team) -> f64 {
    if !pitch_value_reward_enabled() {
        return 0.0;
    }
    let scale = tunables().reward.pitch_value_threat_delta_points;
    if scale == 0.0 || !scale.is_finite() {
        return 0.0;
    }
    // Territorial advantage Φ = net expected threat is a state potential, so its
    // contribution is potential-based shaping (Ng et al. 1999): F = γΦ(s') − Φ(s),
    // policy-invariant by construction. γ = 1.0 keeps this the plain difference the
    // term already used (byte-identical); routing it through the shared primitive
    // documents the invariance and is the template new dense terms should follow.
    let delta = super::potential_based_shaping(
        1.0,
        territorial_advantage(before, team),
        territorial_advantage(after, team),
    );
    if !delta.is_finite() {
        return 0.0;
    }
    delta * scale
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: f64 = DEFAULT_FIELD_WIDTH_YARDS;
    const L: f64 = DEFAULT_FIELD_LENGTH_YARDS;
    static PITCH_VALUE_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn expected_threat_rises_toward_opponent_goal() {
        // Home attacks toward y = L. A central point near the away goal must be
        // worth more than one near the home goal.
        let near_away_goal = Vec2::new(W * 0.5, L * 0.92);
        let near_home_goal = Vec2::new(W * 0.5, L * 0.08);
        assert!(
            expected_threat(Team::Home, near_away_goal, W, L)
                > expected_threat(Team::Home, near_home_goal, W, L)
        );
    }

    #[test]
    fn expected_threat_is_mirror_symmetric_between_teams() {
        // The same physical point is worth as much to Home attacking up as the
        // mirrored point is to Away attacking down.
        let p = Vec2::new(W * 0.30, L * 0.75);
        let mirrored = Vec2::new(W * 0.30, L - p.y);
        let home_v = expected_threat(Team::Home, p, W, L);
        let away_v = expected_threat(Team::Away, mirrored, W, L);
        assert!((home_v - away_v).abs() < 1e-9, "{home_v} vs {away_v}");
    }

    #[test]
    fn expected_threat_prefers_central_shooting_zones() {
        let central = Vec2::new(W * 0.5, L * 0.9);
        let wide = Vec2::new(W * 0.02, L * 0.9);
        assert!(
            expected_threat(Team::Home, central, W, L) > expected_threat(Team::Home, wide, W, L)
        );
    }

    // The snapshot structs deliberately don't derive `Default` (they're large and
    // always built from a live match). Build one real 11v11 snapshot once and
    // clone a player out of it as a mutable template for the controlled fixtures.
    fn base_snapshot() -> WorldSnapshot {
        let sim = SoccerMatch::default_11v11(MatchConfig::default());
        WorldSnapshot::from_match(&sim)
    }

    fn player_template() -> PlayerSnapshot {
        base_snapshot().players[0].clone()
    }

    fn player_at(id: usize, team: Team, pos: Vec2, vel: Vec2) -> PlayerSnapshot {
        let mut p = player_template();
        p.id = id;
        p.team = team;
        p.position = pos;
        p.velocity = vel;
        p.acceleration = Vec2::zero();
        p
    }

    #[test]
    fn pitch_control_favors_the_nearer_team() {
        let cell = Vec2::new(W * 0.5, L * 0.5);
        let players = vec![
            player_at(
                0,
                Team::Home,
                Vec2::new(W * 0.5, L * 0.5 - 2.0),
                Vec2::zero(),
            ),
            player_at(
                1,
                Team::Away,
                Vec2::new(W * 0.5, L * 0.5 + 20.0),
                Vec2::zero(),
            ),
        ];
        let home = pitch_control_home(&players, cell);
        assert!(home > 0.5, "home is far closer, expected >0.5, got {home}");
    }

    #[test]
    fn pitch_control_rewards_momentum_toward_the_cell() {
        let cell = Vec2::new(W * 0.5, L * 0.5);
        // Both equidistant, but the home player is sprinting in and the away
        // player is stationary.
        let toward = Vec2::new(0.0, 6.0);
        let moving = vec![
            player_at(0, Team::Home, Vec2::new(W * 0.5, L * 0.5 - 10.0), toward),
            player_at(
                1,
                Team::Away,
                Vec2::new(W * 0.5, L * 0.5 + 10.0),
                Vec2::zero(),
            ),
        ];
        let still = vec![
            player_at(
                0,
                Team::Home,
                Vec2::new(W * 0.5, L * 0.5 - 10.0),
                Vec2::zero(),
            ),
            player_at(
                1,
                Team::Away,
                Vec2::new(W * 0.5, L * 0.5 + 10.0),
                Vec2::zero(),
            ),
        ];
        assert!(pitch_control_home(&moving, cell) > pitch_control_home(&still, cell));
    }

    fn snapshot_with(players: Vec<PlayerSnapshot>, ball: Vec2) -> WorldSnapshot {
        let mut snap = base_snapshot();
        snap.field_width = W;
        snap.field_length = L;
        snap.players = players;
        snap.ball.position = ball;
        snap
    }

    #[test]
    fn advancing_into_the_final_third_raises_team_threat() {
        let make = |home_y: f64| {
            snapshot_with(
                vec![
                    player_at(0, Team::Home, Vec2::new(W * 0.5, home_y), Vec2::zero()),
                    player_at(1, Team::Away, Vec2::new(W * 0.5, L * 0.5), Vec2::zero()),
                ],
                Vec2::new(W * 0.5, home_y),
            )
        };
        let deep = team_expected_threat(&make(L * 0.30), Team::Home);
        let high = team_expected_threat(&make(L * 0.85), Team::Home);
        assert!(
            high > deep,
            "advancing should raise threat: {deep} -> {high}"
        );
    }

    #[test]
    fn reward_delta_respects_gate_and_rewards_advancing() {
        let _lock = PITCH_VALUE_ENV_LOCK.lock().expect("pitch value env lock");
        // Use a real 11v11 formation: pushing one attacker into the final third
        // adds forward control WITHOUT uncovering the back line (the degenerate
        // 1v1 case is intentionally not rewarded — vacating your whole half cedes
        // more valuable-to-the-opponent space than the lone run gains, which is
        // the zero-sum model behaving correctly).
        let before = base_snapshot();
        // The home attacker furthest forward (largest y, since Home attacks +y).
        let forward_idx = before
            .players
            .iter()
            .enumerate()
            .filter(|(_, p)| p.team == Team::Home)
            .max_by(|(_, a), (_, b)| a.position.y.total_cmp(&b.position.y))
            .map(|(i, _)| i)
            .expect("home team has players");
        let mut after = before.clone();
        after.players[forward_idx].position.y = L * 0.88;
        after.players[forward_idx].position.x = W * 0.5;

        // Both gate states are asserted in one test so they never race on the
        // process-global enable var with another parallel test.
        std::env::remove_var(PITCH_VALUE_REWARD_ENABLE_ENV);
        assert_eq!(pitch_value_reward_delta(&before, &after, Team::Home), 0.0);

        std::env::set_var(PITCH_VALUE_REWARD_ENABLE_ENV, "1");
        let home_reward = pitch_value_reward_delta(&before, &after, Team::Home);
        let away_reward = pitch_value_reward_delta(&before, &after, Team::Away);
        std::env::remove_var(PITCH_VALUE_REWARD_ENABLE_ENV);
        assert!(
            home_reward > 0.0,
            "advancing into space should reward Home: {home_reward}"
        );
        assert!(
            away_reward < 0.0,
            "Home advancing should penalize Away: {away_reward}"
        );
    }

    #[test]
    fn bad_pass_punished_more_when_turnover_leads_to_opponent_advance() {
        // The MDP/POMDP modelling claim, made concrete: because the dense reward is
        // potential-based shaping on the NET (zero-sum) territorial threat Φ =
        // team_xt(you) − team_xt(opponent), a giveaway is punished in proportion to
        // how much the opponent then does with it — and forgiven when it is won
        // straight back — WITHOUT any hand-coded "two forward passes / 30 yards /
        // shot" trigger. The severity emerges from Φ; the trigger list is just
        // eval metrics.
        let _lock = PITCH_VALUE_ENV_LOCK.lock().expect("pitch value env lock");
        std::env::set_var(PITCH_VALUE_REWARD_ENABLE_ENV, "1");

        let mid = W * 0.5;
        // Home attacks +y, Away attacks toward y = 0. A balanced 3-v-3 block so
        // neither side is "vacating its half" (the degenerate case the zero-sum
        // model intentionally punishes); the only thing that moves between states
        // is WHO controls the dangerous advanced space.
        let state = |ball: Vec2, home: [Vec2; 3], away: [Vec2; 3]| {
            snapshot_with(
                vec![
                    player_at(0, Team::Home, home[0], Vec2::zero()),
                    player_at(1, Team::Home, home[1], Vec2::zero()),
                    player_at(2, Team::Home, home[2], Vec2::zero()),
                    player_at(3, Team::Away, away[0], Vec2::zero()),
                    player_at(4, Team::Away, away[1], Vec2::zero()),
                    player_at(5, Team::Away, away[2], Vec2::zero()),
                ],
                ball,
            )
        };

        // s0: Home in settled possession in the Away half — controls the ball and
        // the dangerous advanced space.
        let s0 = state(
            Vec2::new(mid, L * 0.70),
            [
                Vec2::new(mid, L * 0.70),
                Vec2::new(mid - 8.0, L * 0.58),
                Vec2::new(mid + 8.0, L * 0.55),
            ],
            [
                Vec2::new(mid, L * 0.80),
                Vec2::new(mid - 10.0, L * 0.88),
                Vec2::new(mid + 10.0, L * 0.88),
            ],
        );
        // s1: the bad pass is intercepted — Away now controls the ball at halfway,
        // Home caught upfield.
        let s1 = state(
            Vec2::new(mid, L * 0.50),
            [
                Vec2::new(mid, L * 0.62),
                Vec2::new(mid - 8.0, L * 0.58),
                Vec2::new(mid + 8.0, L * 0.55),
            ],
            [
                Vec2::new(mid, L * 0.50),
                Vec2::new(mid - 10.0, L * 0.70),
                Vec2::new(mid + 10.0, L * 0.70),
            ],
        );
        // s2_damage: Away breaks 30+ yds the other way — ball and Away runners deep
        // in Home's half, threatening Home's goal.
        let s2_damage = state(
            Vec2::new(mid, L * 0.18),
            [
                Vec2::new(mid, L * 0.40),
                Vec2::new(mid - 8.0, L * 0.45),
                Vec2::new(mid + 8.0, L * 0.42),
            ],
            [
                Vec2::new(mid, L * 0.18),
                Vec2::new(mid - 10.0, L * 0.30),
                Vec2::new(mid + 10.0, L * 0.30),
            ],
        );
        // s2_recover: Home wins it straight back — field position restored to s0.
        let s2_recover = s0.clone();

        // Home's accumulated PBRS reward along each two-step trajectory.
        let damage = pitch_value_reward_delta(&s0, &s1, Team::Home)
            + pitch_value_reward_delta(&s1, &s2_damage, Team::Home);
        let recover = pitch_value_reward_delta(&s0, &s1, Team::Home)
            + pitch_value_reward_delta(&s1, &s2_recover, Team::Home);
        std::env::remove_var(PITCH_VALUE_REWARD_ENABLE_ENV);

        // 1. The giveaway that lets the opponent advance is punished MORE (strictly
        //    more negative team reward) than the one immediately won back.
        assert!(
            damage < recover,
            "opponent advance must cost more than a quick win-back: damage={damage} recover={recover}"
        );
        // 2. The quick win-back telescopes back to zero: PBRS over a trajectory that
        //    returns to its starting potential nets nothing — the giveaway is
        //    forgiven once possession AND field position are restored. This is the
        //    exact "penalised more ... than ..." asymmetry the request describes.
        assert!(
            recover.abs() < 1e-9,
            "win-back must telescope to ~0 (PBRS invariance): {recover}"
        );
        // 3. Conceding the counter is net-negative for Home (a real penalty, not
        //    merely a smaller reward).
        assert!(
            damage < 0.0,
            "a bad pass that concedes a counter must be net-negative for Home: {damage}"
        );
    }

    #[test]
    fn territorial_advantage_is_zero_sum_between_teams() {
        let snap = snapshot_with(
            vec![
                player_at(0, Team::Home, Vec2::new(W * 0.5, L * 0.6), Vec2::zero()),
                player_at(1, Team::Away, Vec2::new(W * 0.5, L * 0.4), Vec2::zero()),
            ],
            Vec2::new(W * 0.5, L * 0.5),
        );
        let home_adv = territorial_advantage(&snap, Team::Home);
        let away_adv = territorial_advantage(&snap, Team::Away);
        assert!(
            (home_adv + away_adv).abs() < 1e-9,
            "{home_adv} + {away_adv}"
        );
    }

    #[test]
    fn pitch_value_model_sanitizes_non_finite_inputs() {
        let threat = expected_threat(Team::Home, Vec2::new(f64::NAN, f64::INFINITY), W, L);
        assert!(
            threat.is_finite(),
            "expected threat should stay finite: {threat}"
        );

        let players = vec![
            player_at(0, Team::Home, Vec2::new(f64::NAN, L * 0.5), Vec2::zero()),
            player_at(
                1,
                Team::Away,
                Vec2::new(W * 0.5, f64::INFINITY),
                Vec2::zero(),
            ),
        ];
        let control = pitch_control_home(&players, Vec2::new(W * 0.5, L * 0.5));
        assert!(
            control.is_finite(),
            "pitch control should stay finite: {control}"
        );

        let mut snap = snapshot_with(players, Vec2::new(W * 0.5, L * 0.5));
        snap.field_width = f64::NAN;
        assert_eq!(team_expected_threat(&snap, Team::Home), 0.0);
    }

    // A reference point in central midfield where the home player is the nearest
    // controller, so the control-weighted-threat gradient points up-field toward
    // the away goal (Home attacks +y).
    fn xt_point(team: Team, pos: Vec2) -> XtControlPoint {
        XtControlPoint {
            team,
            position: pos,
            velocity: Vec2::zero(),
            top_speed: 7.5,
        }
    }

    fn xt_fixture() -> (Vec<XtControlPoint>, Vec2) {
        let reference = Vec2::new(W * 0.5, L * 0.5);
        let points = vec![
            xt_point(Team::Home, Vec2::new(W * 0.5, L * 0.5 - 2.0)),
            xt_point(Team::Away, Vec2::new(W * 0.5, L * 0.2)),
        ];
        (points, reference)
    }

    #[test]
    fn xt_terminal_shaping_is_identity_when_disabled() {
        let _lock = PITCH_VALUE_ENV_LOCK.lock().expect("pitch value env lock");
        std::env::remove_var(XT_TERMINAL_COST_ENABLE_ENV);
        let (players, reference) = xt_fixture();
        let shaped = xt_terminal_shaped_target(&players, Team::Home, reference, W, L);
        assert_eq!(
            shaped, reference,
            "gate off ⇒ the reference must be byte-identical"
        );
    }

    #[test]
    fn xt_terminal_shaping_pulls_toward_controllable_value() {
        let _lock = PITCH_VALUE_ENV_LOCK.lock().expect("pitch value env lock");
        std::env::set_var(XT_TERMINAL_COST_ENABLE_ENV, "1");
        let (players, reference) = xt_fixture();
        let shaped = xt_terminal_shaped_target(&players, Team::Home, reference, W, L);
        std::env::remove_var(XT_TERMINAL_COST_ENABLE_ENV);
        // The move is bounded and points up the value gradient: forward (toward the
        // away goal at +y) into space this player controls.
        assert!(
            shaped.y > reference.y,
            "shaping should pull forward up the xT gradient: {} -> {}",
            reference.y,
            shaped.y
        );
        assert!(
            reference.distance(shaped) <= XT_TERMINAL_MAX_STEP_YARDS + 1e-6,
            "shaping step must stay within the bound: {}",
            reference.distance(shaped)
        );
    }

    #[test]
    fn xt_terminal_shaping_sanitizes_non_finite_reference() {
        let _lock = PITCH_VALUE_ENV_LOCK.lock().expect("pitch value env lock");
        std::env::set_var(XT_TERMINAL_COST_ENABLE_ENV, "1");
        let (players, _) = xt_fixture();
        let bad = Vec2::new(f64::NAN, L * 0.5);
        let shaped = xt_terminal_shaped_target(&players, Team::Home, bad, W, L);
        std::env::remove_var(XT_TERMINAL_COST_ENABLE_ENV);
        // Returned unchanged: x stays NaN (so != comparison can't be used) and y is
        // untouched — the guard bailed before computing any gradient.
        assert!(
            shaped.x.is_nan() && shaped.y == bad.y,
            "a non-finite reference is returned unchanged: {shaped:?}"
        );
    }
}
