//! PPO (clipped) + GAE training of a shared-weight multi-agent policy for
//! Team A against the scripted baseline. One actor MLP + one critic MLP, both
//! shared across all 5 players. Reward is team-level with potential shaping.

use crate::game::*;
use crate::nn::{masked_softmax, Mlp};
use crate::rng::Rng;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

// SPEED WARMUP CURRICULUM: while true, every player uses a fixed run-medium gear
// (v3 behavior) instead of sampling the speed policy. This lets the ACTION policy
// learn to attack at full strength on stable pass dynamics BEFORE the speed policy
// starts introducing gear variability — the pass mechanics are fragile to speed
// variability, so the two policies would otherwise deadlock.
static SPEED_FROZEN: AtomicBool = AtomicBool::new(false);
pub fn set_speed_frozen(frozen: bool) {
    SPEED_FROZEN.store(frozen, Ordering::Relaxed);
}

// ── SELF-PLAY CHAMPION LADDER ───────────────────────────────────────────────
// When a champion is installed, Team B (the opponent) is driven by that FROZEN
// learned policy instead of the scripted baseline — so BOTH teams are learned
// policies. The challenger (Team A) keeps training; when it beats the champion by
// a margin it is promoted to the new champion ("new winner beats old winner to
// advance"). Every action still flows through World::step, which executes only
// legal, physics-bounded moves — self-play cannot invent unphysical behavior.
static SELFPLAY_CHAMPION: RwLock<Option<Arc<Policy>>> = RwLock::new(None);

fn selfplay_champion_write() -> RwLockWriteGuard<'static, Option<Arc<Policy>>> {
    SELFPLAY_CHAMPION
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn selfplay_champion_read() -> RwLockReadGuard<'static, Option<Arc<Policy>>> {
    SELFPLAY_CHAMPION
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

pub fn set_selfplay_champion(champion: Option<Policy>) {
    *selfplay_champion_write() = champion.map(Arc::new);
}

pub fn clear_selfplay_champion() {
    *selfplay_champion_write() = None;
}
fn selfplay_champion() -> Option<Arc<Policy>> {
    selfplay_champion_read().clone()
}

const GAMMA: f32 = 0.995; // per-tick discount retuned for 20 Hz (same per-second horizon)
const LAMBDA: f32 = 0.95;
const CLIP: f32 = 0.2;
const LR_ACTOR: f32 = 3e-4;
const LR_CRITIC: f32 = 1e-3;
const LR_BC: f32 = 8e-4;
const EPOCHS: usize = 4;
const MINIBATCH: usize = 1024;
const MAX_ROLLOUT_THREADS: usize = 4;
const MIN_REWARD_WEIGHT: f32 = 0.0001;
// Frequency-aware search bands. Event rewards may move farther because they are
// sparse; per-tick shaping stays deliberately small so a policy cannot farm it
// instead of scoring. These bound the learned base weights only. The field
// vector still modulates the realized reward inside the band at every decision.
const SPARSE_PASS_WEIGHT_BOUNDS: (f32, f32) = (0.02, 1.00);
const SPARSE_MILESTONE_WEIGHT_BOUNDS: (f32, f32) = (0.05, 3.00);
const SPARSE_RECOVERY_WEIGHT_BOUNDS: (f32, f32) = (0.10, 4.00);
const TURNOVER_WEIGHT_BOUNDS: (f32, f32) = (0.20, 8.00);
const BAD_ACTION_WEIGHT_BOUNDS: (f32, f32) = (0.20, 10.00);
const RECYCLE_WEIGHT_BOUNDS: (f32, f32) = (0.02, 3.00);
const RETURN_LOOP_WEIGHT_BOUNDS: (f32, f32) = (0.05, 6.00);
const PER_TICK_WEIGHT_BOUNDS: (f32, f32) = (0.001, 0.20);
const FIELD_DELTA_WEIGHT_BOUNDS: (f32, f32) = (0.005, 0.75);
const SHAPE_WEIGHT_BOUNDS: (f32, f32) = (0.25, 4.00);
const SPACING_WEIGHT_BOUNDS: (f32, f32) = (0.0005, 0.05);
const SHOT_BASE_POINTS_BOUNDS: (f32, f32) = (50.0, 100.0);
const SHOT_QUALITY_POINTS_BOUNDS: (f32, f32) = (MIN_REWARD_WEIGHT, 100.0);
/// Sparse football outcomes are fixed semantic anchors, not search dimensions.
/// The tuner may learn every subordinate weight beneath them, but cannot learn
/// that missing repeatedly is more valuable than scoring or winning.
pub const GOAL_ANCHOR_POINTS: f32 = 500.0;
pub const MATCH_OUTCOME_ANCHOR_POINTS: f32 = 1000.0;
pub const ON_FRAME_SHOT_MIN_POINTS: f32 = 50.0;
/// The environment and telemetry retain football points (goal=500, win=1000),
/// while the critic learns in bounded units. This is fixed reward/value scaling,
/// not reward shaping: every relative reward and sign is preserved.
const VALUE_TARGET_SCALE_POINTS: f32 = MATCH_OUTCOME_ANCHOR_POINTS;
const FULL_GAME_ACTOR_CREDIT_FRACTION_BOUNDS: (f32, f32) = (0.001, 0.10);
const CONVERSION_OVER_SHOT_MARGIN: f32 = 5.0;
const WIN_OVER_CONVERSION_MARGIN: f32 = 20.0;
/// A converted goal must dominate the summed non-converting shot reward by a PROPORTIONAL margin:
/// `shot_base + shot_q <= goal * REWARD_NON_CONVERSION_MAX_FRACTION`. Scales with the goal weight
/// rather than a fixed point gap (which loses meaning once weights move), matching the 11v11 engine's
/// `grounded_non_conversion_reward_scale`. Default 0.40 = an on-target shot that misses is worth at
/// most ~40% of a goal.
const REWARD_NON_CONVERSION_MAX_FRACTION: f32 = 0.40;
const LINGER_RADIUS: f32 = 4.0; // "same radius": teammates within this many yards
/// Overlap zone: two players THIS close are occupying the same spot — that is
/// never "running past each other", so the penalty here is ALWAYS-ON (ungated).
/// Only the soft 1.5-4yd band is linger-gated (where transient crossings happen).
const SEVERE_RADIUS: f32 = 1.5;
/// Leaky-integrator decay: when players separate, the linger counter DECAYS by
/// this many ticks rather than resetting to 0. This kills the reset-hack where a
/// policy farms the gate by oscillating in/out of the radius (bunch ~1.4s, step
/// out 1 tick, re-bunch). A genuine one-time crossing separates and stays apart,
/// so its counter decays cleanly to 0; sustained oscillation still accumulates
/// (net gain per cycle whenever close-ticks > DECAY × away-ticks).
const LINGER_DECAY: u32 = 2;
/// Consecutive ticks two teammates must LINGER within LINGER_RADIUS before the
/// mild 3-4yd spacing penalty applies. Sub-3yd bunching is always penalized
/// immediately; brief crossings only get grace in the 3-4yd gray zone.
/// Env `SPACING_LINGER_SECS` (default 0.4 s).
fn linger_ticks() -> u32 {
    let secs = std::env::var("SPACING_LINGER_SECS")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
        .unwrap_or(0.4);
    ((secs / DT).round() as u32).max(1)
}
// ─── Tunable reward weights (env-overridable, read ONCE per process) ─────────
// Every weight below can be set via an env var of the same name, so an external
// search harness (viz/tune.py) can optimize the reward vector without
// recompiling. Bounds keep every reward channel alive while still allowing
// past-run priors to be overridden by the tuner.
fn bounded_weight(raw: Option<&str>, default: f32, lo: f32, hi: f32) -> f32 {
    let lo = lo.max(MIN_REWARD_WEIGHT);
    let hi = hi.max(lo);
    let value = raw
        .and_then(|s| s.parse::<f32>().ok())
        .filter(|v| v.is_finite())
        .unwrap_or(default);
    value.clamp(lo, hi)
}

fn wenv(name: &str, default: f32, lo: f32, hi: f32) -> f32 {
    let raw = std::env::var(name).ok();
    bounded_weight(raw.as_deref(), default, lo, hi)
}

pub fn ppo_learning_rates() -> (f32, f32, f32) {
    let actor = wenv("FIVEASIDE_ACTOR_LR", LR_ACTOR, 1.0e-6, 1.0e-2);
    let critic = wenv("FIVEASIDE_CRITIC_LR", LR_CRITIC, 1.0e-6, 1.0e-2);
    let speed = wenv("FIVEASIDE_SPEED_LR", actor * 0.3, 1.0e-6, 1.0e-2);
    (actor, critic, speed)
}

fn grounded_conversion_ladder(goal: f32, shot_base: f32, shot_q: f32) -> (f32, f32, f32) {
    // The goal stays the reference (clamped to its own range); shot shaping is scaled DOWN to fit a
    // proportional fraction below it, never the reverse — so tuning big shot rewards can't quietly
    // inflate what "scoring" is worth.
    let _requested_goal = goal;
    let goal = GOAL_ANCHOR_POINTS;
    let mut shot_base = shot_base.max(MIN_REWARD_WEIGHT);
    let mut shot_q = shot_q.max(MIN_REWARD_WEIGHT);
    let max_non_conversion = (goal * REWARD_NON_CONVERSION_MAX_FRACTION)
        .min(goal - CONVERSION_OVER_SHOT_MARGIN)
        .max(MIN_REWARD_WEIGHT);
    let total = shot_base + shot_q;
    if total < ON_FRAME_SHOT_MIN_POINTS {
        let scale = ON_FRAME_SHOT_MIN_POINTS / total.max(MIN_REWARD_WEIGHT);
        shot_base *= scale;
        shot_q *= scale;
    }
    let total = shot_base + shot_q;
    if total > max_non_conversion {
        let scale = (max_non_conversion / total).clamp(0.0, 1.0);
        shot_base = (shot_base * scale).max(MIN_REWARD_WEIGHT);
        shot_q = (shot_q * scale).max(MIN_REWARD_WEIGHT);
    }

    (goal, shot_base, shot_q)
}

pub struct Rw {
    pub goal: f32,                 // +goal scored
    pub concede: f32,              // -goal conceded (stored positive, subtracted)
    pub match_win: f32,            // terminal reward for winning the completed episode
    pub match_loss: f32,           // terminal penalty magnitude for losing the episode
    pub shot_base: f32,            // shot-on-goal base (earned, from opponent half after 2 passes)
    pub shot_q: f32,               // shot-on-goal chance-quality bonus
    pub milestone: f32,            // completing the 2nd pass (unlocks the shot)
    pub pass_credit: f32,          // flat credit for a completed pass
    pub turnover: f32,             // -turnover (stored positive, subtracted)
    pub bad_pass_turnover: f32,    // extra -pass interception / bad pass capture
    pub dribble_turnover: f32,     // extra -dribble/carry dispossession
    pub recycle: f32,              // -sterile long possession recycling
    pub return_pass: f32,          // -same two-player return loop
    pub return_stale: f32,         // extra -return loop with no upfield progress
    pub win_ball: f32,             // +ball recovery
    pub dribble: f32,              // forward dribble carry
    pub shape: f32,                // potential shaping (forward progress)
    pub advance: f32,              // OFFENSE: push upfield
    pub open: f32,                 // OFFENSE: clear passing lane from the ball
    pub width: f32,                // OFFENSE: use pitch width
    pub flank: f32,                // OFFENSE: commit to a left/right channel
    pub goalside: f32,             // DEFENSE: goalside of the ball
    pub goalside_run: f32,         // DEFENSE: sprint back while wrong-side
    pub ahead: f32,                // off-ball outlet ahead of the ball in a clear lane
    pub make_run: f32,             // off-ball forward run (velocity)
    pub burst_gear: f32,           // off-ball use of run/sprint gears in possession
    pub field_pass: f32,           // field-vector interaction on completed passes
    pub field_turnover: f32,       // field-vector interaction on turnovers
    pub chance: f32,               // MARL: team reward for creating a scoring chance (potential)
    pub field_goalside_delta: f32, // reward improving team goalside geometry
    pub field_burst_delta: f32,    // reward improving forward outlet geometry
    pub stand_pen: f32,            // anti-passivity: penalize the STAND gear off-ball
    pub pursuit: f32,              // LOOSE-BALL: favorite commits to winning a free ball
}
fn rw() -> &'static Rw {
    static R: std::sync::OnceLock<Rw> = std::sync::OnceLock::new();
    R.get_or_init(|| {
        let (goal, shot_base, shot_q) = grounded_conversion_ladder(
            GOAL_ANCHOR_POINTS,
            wenv(
                "REW_SHOT_BASE",
                50.0,
                SHOT_BASE_POINTS_BOUNDS.0,
                SHOT_BASE_POINTS_BOUNDS.1,
            ),
            wenv(
                "REW_SHOT_Q",
                75.0,
                SHOT_QUALITY_POINTS_BOUNDS.0,
                SHOT_QUALITY_POINTS_BOUNDS.1,
            ),
        );
        enforce_reward_hierarchy(Rw {
            goal,
            concede: GOAL_ANCHOR_POINTS,
            match_win: MATCH_OUTCOME_ANCHOR_POINTS,
            match_loss: MATCH_OUTCOME_ANCHOR_POINTS,
            shot_base,
            shot_q,
            milestone: wenv(
                "REW_MILESTONE",
                0.3,
                SPARSE_MILESTONE_WEIGHT_BOUNDS.0,
                SPARSE_MILESTONE_WEIGHT_BOUNDS.1,
            ),
            pass_credit: wenv(
                "REW_PASS",
                0.06,
                SPARSE_PASS_WEIGHT_BOUNDS.0,
                SPARSE_PASS_WEIGHT_BOUNDS.1,
            ),
            turnover: wenv(
                "REW_TURNOVER",
                0.55,
                TURNOVER_WEIGHT_BOUNDS.0,
                TURNOVER_WEIGHT_BOUNDS.1,
            ),
            bad_pass_turnover: wenv(
                "REW_BAD_PASS_TURNOVER",
                0.35,
                BAD_ACTION_WEIGHT_BOUNDS.0,
                BAD_ACTION_WEIGHT_BOUNDS.1,
            ),
            dribble_turnover: wenv(
                "REW_DRIBBLE_TURNOVER",
                0.75,
                BAD_ACTION_WEIGHT_BOUNDS.0,
                BAD_ACTION_WEIGHT_BOUNDS.1,
            ),
            recycle: wenv(
                "REW_RECYCLE",
                0.18,
                RECYCLE_WEIGHT_BOUNDS.0,
                RECYCLE_WEIGHT_BOUNDS.1,
            ),
            return_pass: wenv(
                "REW_RETURN_PASS",
                0.35,
                RETURN_LOOP_WEIGHT_BOUNDS.0,
                RETURN_LOOP_WEIGHT_BOUNDS.1,
            ),
            return_stale: wenv(
                "REW_RETURN_STALE",
                0.55,
                RETURN_LOOP_WEIGHT_BOUNDS.0,
                RETURN_LOOP_WEIGHT_BOUNDS.1,
            ),
            win_ball: wenv(
                "REW_WIN_BALL",
                0.3,
                SPARSE_RECOVERY_WEIGHT_BOUNDS.0,
                SPARSE_RECOVERY_WEIGHT_BOUNDS.1,
            ),
            dribble: wenv("REW_DRIBBLE", 0.015, PER_TICK_WEIGHT_BOUNDS.0, 0.10),
            shape: wenv("W_SHAPE", 2.2, SHAPE_WEIGHT_BOUNDS.0, SHAPE_WEIGHT_BOUNDS.1),
            advance: wenv("W_ADVANCE", 0.04, PER_TICK_WEIGHT_BOUNDS.0, 0.12),
            open: wenv("W_OPEN", 0.04, PER_TICK_WEIGHT_BOUNDS.0, 0.12),
            width: wenv("W_WIDTH", 0.045, PER_TICK_WEIGHT_BOUNDS.0, 0.12),
            flank: wenv("W_FLANK", 0.025, PER_TICK_WEIGHT_BOUNDS.0, 0.10),
            goalside: wenv("W_GOALSIDE", 0.08, PER_TICK_WEIGHT_BOUNDS.0, 0.20),
            goalside_run: wenv("W_GOALSIDE_RUN", 0.04, PER_TICK_WEIGHT_BOUNDS.0, 0.16),
            ahead: wenv("W_AHEAD", 0.035, PER_TICK_WEIGHT_BOUNDS.0, 0.16),
            make_run: wenv("W_MAKE_RUN", 0.06, PER_TICK_WEIGHT_BOUNDS.0, 0.20),
            burst_gear: wenv("W_BURST_GEAR", 0.035, PER_TICK_WEIGHT_BOUNDS.0, 0.16),
            field_pass: wenv("W_FIELD_PASS", 0.08, FIELD_DELTA_WEIGHT_BOUNDS.0, 0.30),
            field_turnover: wenv("W_FIELD_TURNOVER", 0.16, FIELD_DELTA_WEIGHT_BOUNDS.0, 0.50),
            chance: wenv(
                "W_CHANCE",
                0.12,
                FIELD_DELTA_WEIGHT_BOUNDS.0,
                FIELD_DELTA_WEIGHT_BOUNDS.1,
            ),
            field_goalside_delta: wenv(
                "W_FIELD_GOALSIDE_DELTA",
                0.10,
                FIELD_DELTA_WEIGHT_BOUNDS.0,
                0.35,
            ),
            field_burst_delta: wenv(
                "W_FIELD_BURST_DELTA",
                0.08,
                FIELD_DELTA_WEIGHT_BOUNDS.0,
                0.35,
            ),
            stand_pen: wenv("W_STAND_PEN", 0.02, PER_TICK_WEIGHT_BOUNDS.0, 0.20),
            pursuit: wenv("W_PURSUIT", 0.05, PER_TICK_WEIGHT_BOUNDS.0, 0.20),
        })
    })
}

#[cfg(test)]
fn max_nonconverting_shot_reward(weights: &Rw) -> f32 {
    weights.shot_base + weights.shot_q
}

fn on_frame_shot_reward(weights: &Rw, placement_quality: f32, xg: f32) -> f32 {
    let contextual = (weights.shot_base + weights.shot_q * placement_quality.clamp(0.0, 1.0))
        * xg.clamp(0.0, 1.0);
    contextual.max(ON_FRAME_SHOT_MIN_POINTS)
}

fn enforce_reward_hierarchy(mut weights: Rw) -> Rw {
    let (goal, shot_base, shot_q) =
        grounded_conversion_ladder(weights.goal, weights.shot_base, weights.shot_q);
    weights.goal = goal;
    weights.concede = GOAL_ANCHOR_POINTS;
    weights.shot_base = shot_base;
    weights.shot_q = shot_q;
    weights.match_win = MATCH_OUTCOME_ANCHOR_POINTS;
    weights.match_loss = MATCH_OUTCOME_ANCHOR_POINTS;
    debug_assert!(weights.match_win >= weights.goal + WIN_OVER_CONVERSION_MARGIN);
    weights
}
// The speed policy is a low-variance REFINEMENT on top of the action policy —
// small entropy + LR so its exploration can't paralyze the game the action policy
// is trying to learn in.
const SPEED_ENT_SCALE: f32 = 0.15;
const ENT_BETA0: f32 = 0.02;

// Teammate-spacing reward weight. Overridable via SPACING_W. Default 0.008 is the
// VALIDATED anti-bunch value (reaches ~0% bunching with the leaky linger gate + the
// hard-resolver backstop) — strong enough that sub-3yd bunching competes with
// ordinary possession rewards. This fires for every player every tick and its raw
// curve reaches -100, so the search band is intentionally narrow; a huge spacing
// coefficient would erase all action/outcome credit before PPO ever sees it.
fn w_spacing() -> f32 {
    wenv(
        "SPACING_W",
        0.008,
        SPACING_WEIGHT_BOUNDS.0,
        SPACING_WEIGHT_BOUNDS.1,
    )
}

/// PER-PLAYER spacing reward as a function of a player's nearest-teammate
/// distance `d`. Steep penalties for bunching (the user's curve), reward around
/// the ~5-unit optimum. Applied to EACH outfielder individually, every tick, in
/// all phases — so bunching is directly attributable and can't hide in a shared
/// team average.
fn spacing_reward(d: f32) -> f32 {
    // The ball is a major attractor, but only the closest player should chase;
    // stacking is never OK. We want teammates to hold real cartesian distance and
    // stretch the pitch, so the optimum sits WIDE at ~8 yds:
    //   < 1.5  : MAJOR penalty (overlap zone — sickening bunching)
    //   1.5-3  : strong penalty (bumped) — being within 3 yds is bad
    //   3-4    : real penalty (bumped) — 4 yds is still too close
    //   ~8     : peak reward (optimal spacing — genuine width/separation)
    //   > 12   : penalty (too far to combine)
    if d < 1.0 {
        -100.0
    } else if d < 1.5 {
        -50.0
    } else if d < 3.0 {
        -18.0 // within 3 yds: STEEP (lingering this close is bad)
    } else if d < 4.0 {
        -3.0 // within 4 yds: now also penalized
    } else if d < 8.0 {
        (d - 4.0) * 2.0 // 0 at 4 -> +8 peak at 8
    } else if d < 12.0 {
        8.0 - (d - 8.0) * 2.0 // +8 at 8 -> 0 at 12
    } else {
        -(d - 12.0) * 4.0 // 12+ : escalating penalty
    }
}

#[derive(Clone, Copy, Default)]
struct FieldRewardContext {
    pass_value: f32,
    safe_outlet_value: f32,
    turnover_risk: f32,
    dribble_pressure: f32,
    goalside_score: f32,
    burst_score: f32,
    return_stale: f32,
    chance_value: f32,
    carry_value: f32,
}

impl FieldRewardContext {
    fn from_world(w: &World) -> Self {
        let mut ctx = Self::default();
        let phase = w.possession_phase_for(Team::A);
        let our_phase = phase == PossessionPhase::Possession;
        let their_phase = phase == PossessionPhase::Dispossession;

        let mut best_outlet = 0.0f32;
        let mut outlet_count = 0.0f32;
        for i in 1..N {
            let pos = w.a[i].pos;
            let ahead = ((pos.y - w.ball.y) / 12.0).clamp(0.0, 1.0);
            let lane = w.lane_clearness(Team::A, w.ball, pos);
            let (_, opp_d) = w.nearest_opponent_distance(Team::A, pos);
            let space = (opp_d / 8.0).clamp(0.0, 1.0);
            let outlet = ahead * (0.55 * lane + 0.45 * space);
            best_outlet = best_outlet.max(outlet);
            if outlet > 0.25 {
                outlet_count += 1.0;
            }
        }
        ctx.safe_outlet_value =
            (0.70 * best_outlet + 0.30 * (outlet_count / 3.0).clamp(0.0, 1.0)).clamp(0.0, 1.0);
        // MARL chance creation: the best SHOOTING chance the attack has manufactured
        // — an off-ball attacker in a high-xG spot (close+central to the opp goal, in
        // their half) with a clear reception lane from the ball. Coordinated,
        // synchronized runs INTO such a spot raise this; used as a potential.
        if our_phase {
            let goal = V2::new(FIELD_L, FIELD_W / 2.0);
            let mut best_chance = 0.0f32;
            for i in 1..N {
                let pos = w.a[i].pos;
                if pos.y < SHOOT_X {
                    continue;
                }
                let d = goal.sub(pos).len();
                let lateral = (pos.x - FIELD_W / 2.0).abs();
                let dist_f = (1.0 - d / 26.0).clamp(0.0, 1.0);
                let angle_f = (1.0 - lateral / (FIELD_W / 2.0)).clamp(0.0, 1.0);
                let xg = dist_f * dist_f * (0.4 + 0.6 * angle_f);
                let lane = w.lane_clearness(Team::A, w.ball, pos);
                best_chance = best_chance.max(xg * lane);
            }
            ctx.chance_value = best_chance;
        }

        if let Some(o) = w.owner {
            let carrier = if o.team == Team::A {
                w.a[o.idx].pos
            } else {
                w.b[o.idx].pos
            };
            let (_, pressure_d) = w.nearest_opponent_distance(o.team, carrier);
            let pressure = (1.0 - pressure_d / 7.0).clamp(0.0, 1.0);
            if o.team == Team::A {
                let forward = (carrier.y / FIELD_L).clamp(0.0, 1.0);
                ctx.dribble_pressure = pressure;
                // Carrying out of our half is a useful build-up action; once the
                // ball reaches the final third, clear shooting/passing chances
                // should dominate another low-value touch. Pressure adds escape
                // value, and the complete 10-player field vector determines it.
                let zone_value = if forward < 0.50 {
                    1.35
                } else if forward < 2.0 / 3.0 {
                    1.00
                } else {
                    0.55
                };
                ctx.carry_value = (zone_value * (0.85 + 0.30 * pressure)).clamp(0.45, 1.65);
                ctx.pass_value = (0.45 * ctx.safe_outlet_value + 0.35 * forward + 0.20 * pressure)
                    .clamp(0.0, 1.0);
                ctx.turnover_risk =
                    (0.40 * pressure + 0.35 * ctx.safe_outlet_value + 0.25 * forward)
                        .clamp(0.0, 1.0);
            }
        } else if matches!(w.last_touch, Some(Team::A)) {
            ctx.pass_value = ctx.safe_outlet_value * 0.75;
            ctx.turnover_risk = ctx.safe_outlet_value * 0.65;
        }

        let mut goalside_sum = 0.0f32;
        let mut recovery_need = 0.0f32;
        for i in 1..N {
            let pos = w.a[i].pos;
            let goalside = ((w.ball.y - pos.y) / 8.0).clamp(-1.0, 1.0);
            goalside_sum += (goalside + 1.0) * 0.5;
            if their_phase && pos.y > w.ball.y {
                recovery_need += ((pos.y - w.ball.y) / FIELD_L).clamp(0.0, 1.0);
            }
        }
        ctx.goalside_score = if their_phase {
            (goalside_sum / (N - 1) as f32 - 0.25 * recovery_need).clamp(0.0, 1.0)
        } else {
            (goalside_sum / (N - 1) as f32).clamp(0.0, 1.0)
        };

        if our_phase {
            let mut burst_sum = 0.0f32;
            for i in 1..N {
                let pos = w.a[i].pos;
                if matches!(w.owner, Some(o) if o.team == Team::A && o.idx == i) {
                    continue;
                }
                let ahead = ((pos.y - w.ball.y) / 12.0).clamp(0.0, 1.0);
                let lane = w.lane_clearness(Team::A, w.ball, pos);
                let fwd_speed = (w.a[i].vel.y / 8.5).clamp(0.0, 1.0);
                burst_sum += ahead * lane * (0.35 + 0.65 * fwd_speed);
            }
            ctx.burst_score = (burst_sum / (N - 1) as f32).clamp(0.0, 1.0);
        }

        ctx.return_stale = if w.return_streak_a > 0 && w.ball.y - w.return_start_x < 5.0 {
            (1.0 - (w.ball.y - w.return_start_x).max(0.0) / 5.0).clamp(0.0, 1.0)
        } else {
            0.0
        };
        ctx
    }

    fn delta(next: f32, prev: f32) -> f32 {
        (next - prev).clamp(-1.0, 1.0)
    }
}

#[derive(Clone)]
pub struct Policy {
    pub actor: Mlp,   // action policy: OBS_DIM -> 64 -> 64 -> NA (identical to v3)
    pub speedor: Mlp, // SEPARATE speed policy: OBS_DIM -> 32 -> NS (own network)
    pub critic: Mlp,
}

impl Policy {
    pub fn new(rng: &mut Rng) -> Self {
        Policy {
            // decentralized actor (per-agent field vector) + CENTRALIZED critic
            // (global state) = MAPPO / CTDE. The speed gear is a SEPARATE small
            // network so its gradients can't corrupt the action policy's shared
            // features (a joint output head did exactly that — the action policy
            // stopped learning to attack).
            actor: Mlp::new(&[OBS_DIM, 64, 64, NA], rng),
            speedor: Mlp::new(&[OBS_DIM, 32, NS], rng),
            critic: Mlp::new(&[GLOBAL_DIM, 128, 64, 1], rng),
        }
    }

    pub fn greedy_action(&self, obs: &[f32], mask: &[bool; NA]) -> usize {
        let logits = self.actor.predict(obs);
        let probs = masked_softmax(&logits, mask);
        let mut bi = 0;
        let mut bp = -1.0;
        for i in 0..NA {
            if mask[i] && probs[i] > bp {
                bp = probs[i];
                bi = i;
            }
        }
        bi
    }

    pub fn greedy_speed(&self, obs: &[f32], speed_mask: &[bool; NS]) -> usize {
        // During the speed warmup, eval uses the same fixed gear the action policy
        // was trained on, coerced onto the phase-aware legal speed support.
        if SPEED_FROZEN.load(Ordering::Relaxed) {
            return speed_mask
                .iter()
                .enumerate()
                .skip(SPD_RUN_MED)
                .find_map(|(idx, ok)| ok.then_some(idx))
                .or_else(|| speed_mask.iter().position(|&ok| ok))
                .unwrap_or(SPD_STAND);
        }
        // Greedy speed gear over the legal phase-aware gears. We prefer moving
        // gears when legal so a near-flat speed policy cannot tie-break attacking
        // and recovery runs into standing still.
        let slogits = self.speedor.predict(obs);
        let mut bs = speed_mask
            .iter()
            .enumerate()
            .skip(SPD_WALK)
            .find_map(|(idx, ok)| ok.then_some(idx))
            .or_else(|| speed_mask.iter().position(|&ok| ok))
            .unwrap_or(SPD_STAND);
        let mut bsp = f32::NEG_INFINITY;
        for k in 0..NS {
            if speed_mask[k] && slogits[k] > bsp {
                bsp = slogits[k];
                bs = k;
            }
        }
        bs
    }

    /// Greedy (argmax action + argmax legal speed) — used at evaluation. Returns
    /// the PACKED action `action + speed*NA` ready to hand to `World::step`.
    pub fn act_greedy(&self, obs: &[f32], mask: &[bool; NA], speed_mask: &[bool; NS]) -> usize {
        let bi = self.greedy_action(obs, mask);
        let bs = self.greedy_speed(obs, speed_mask);
        bi + bs * NA
    }

    pub fn act_greedy_world(&self, world: &World, team: Team, idx: usize) -> usize {
        let obs = world.observe(team, idx);
        let mask = world.legal_mask(team, idx);
        let action = self.greedy_action(&obs, &mask);
        let speed_mask = world.speed_mask(team, idx, action);
        self.act_greedy(&obs, &mask, &speed_mask)
    }
}

struct Sample {
    obs: [f32; OBS_DIM],       // decentralized actor observation
    gstate: [f32; GLOBAL_DIM], // centralized critic global state
    mask: [bool; NA],
    speed_mask: [bool; NS],
    action: usize,   // macro action (0..NA)
    speed: usize,    // speed gear (0..NS)
    old_logp_a: f32, // ln p(action) at collection time
    old_logp_s: f32, // ln p(speed)  at collection time
    adv: f32,
    ret: f32,
}

fn deferred_action_credit_fraction() -> f32 {
    wenv(
        "DEFERRED_ACTION_CREDIT_FRACTION",
        1.0,
        MIN_REWARD_WEIGHT,
        1.0,
    )
}

/// Move delayed pass/finish credit onto the POMDP decision tick that launched
/// the action. GAE still propagates it farther back through the possession, but
/// this removes avoidable flight-time attribution noise. MPC-executed actions
/// use the same launch tick, so the actor learns from the action that was
/// actually committed to physics rather than the later capture frame.
fn backdate_action_credit(
    reward_history: &mut [f32],
    launch_step: Option<usize>,
    current_step: usize,
    current_reward: &mut f32,
    signed_credit: f32,
    fraction: f32,
) {
    let Some(launch_step) = launch_step else {
        return;
    };
    if launch_step >= current_step || launch_step >= reward_history.len() {
        return;
    }
    let moved = signed_credit * fraction.clamp(MIN_REWARD_WEIGHT, 1.0);
    *current_reward -= moved;
    reward_history[launch_step] += moved;
}

fn full_game_outcome_label(goals_for: u32, goals_against: u32) -> f32 {
    match goals_for.cmp(&goals_against) {
        std::cmp::Ordering::Greater => rw().match_win,
        std::cmp::Ordering::Less => -rw().match_loss,
        std::cmp::Ordering::Equal => 0.0,
    }
}

fn full_game_actor_credit(outcome_label: f32) -> f32 {
    // The critic learns the full +/-1000 terminal value. Broadcasting that raw
    // magnitude into every actor advantage, however, blames every action by all
    // four agents equally for an early loss and erases rare successful actions.
    // Keep a bounded Monte-Carlo policy contribution while local POMDP/MPC
    // rewards retain enough resolution to assign the result to actual decisions.
    outcome_label
        * wenv(
            "FULL_GAME_ACTOR_CREDIT_FRACTION",
            0.02,
            FULL_GAME_ACTOR_CREDIT_FRACTION_BOUNDS.0,
            FULL_GAME_ACTOR_CREDIT_FRACTION_BOUNDS.1,
        )
}

fn value_learning_units(raw_points: f32) -> f32 {
    raw_points / VALUE_TARGET_SCALE_POINTS
}

struct BcSample {
    obs: [f32; OBS_DIM],
    mask: [bool; NA],
    speed_mask: [bool; NS],
    action: usize, // macro action (0..NA)
    speed: usize,  // scripted speed gear (0..NS)
    actor_weight: f32,
}

pub struct CloneStats {
    pub samples: usize,
    pub shot_samples: usize,
    pub loss: f32,
    pub accuracy: f32,
}

fn behavior_clone_actor_weight(action: usize, count: usize, max_count: usize) -> f32 {
    let frequency_weight =
        ((max_count.max(1) as f32 / count.max(1) as f32).sqrt()).clamp(1.0, 64.0);
    if action == A_SHOOT {
        // A legal finish occurs roughly once among thousands of ordinary
        // movement frames. Without bounded rare-event replay, cloning and PPO
        // both forget the only action that can realize the 500-point goal.
        frequency_weight.max(wenv("BC_SHOT_SAMPLE_WEIGHT", 64.0, 1.0, 128.0))
    } else {
        frequency_weight
    }
}

/// Collect one game of Team-A experience (5 players) vs the scripted baseline.
fn rollout(policy: &Policy, rng: &mut Rng, opponent_noise: f32) -> Vec<Sample> {
    let mut w = World::new();
    let w_spacing_coeff = w_spacing();
    if rng.f01() < 0.5 {
        w.kickoff(Team::B); // randomize opening possession to remove kickoff bias
    }
    // per-player trajectory buffers
    let mut obs_buf: Vec<[[f32; OBS_DIM]; N]> = Vec::with_capacity(STEPS);
    let mut mask_buf: Vec<[[bool; NA]; N]> = Vec::with_capacity(STEPS);
    let mut speed_mask_buf: Vec<[[bool; NS]; N]> = Vec::with_capacity(STEPS);
    let mut act_buf: Vec<[usize; N]> = Vec::with_capacity(STEPS); // macro action (0..NA)
    let mut spd_buf: Vec<[usize; N]> = Vec::with_capacity(STEPS); // speed gear (0..NS)
    let mut logpa_buf: Vec<[f32; N]> = Vec::with_capacity(STEPS); // action log-prob
    let mut logps_buf: Vec<[f32; N]> = Vec::with_capacity(STEPS); // speed log-prob
    let mut val_buf: Vec<f32> = Vec::with_capacity(STEPS); // CENTRALIZED value (one per tick)
    let mut gstate_buf: Vec<[f32; GLOBAL_DIM]> = Vec::with_capacity(STEPS);
    let mut rew_buf: Vec<f32> = Vec::with_capacity(STEPS);
    let mut space_buf: Vec<[f32; N]> = Vec::with_capacity(STEPS); // per-player spacing reward

    let mut phi_prev = w.potential_a();
    // per-outfielder linger counters: consecutive ticks a teammate has been within
    // LINGER_RADIUS. Only SUSTAINED lingering triggers the spacing penalty.
    let mut close_ticks = [0u32; N];
    let linger_gate = linger_ticks();
    let deferred_credit_fraction = deferred_action_credit_fraction();
    let mut pending_pass_launch_step: Option<usize> = None;
    let mut pending_shot_launch_step: Option<usize> = None;

    for step in 0..STEPS {
        let mut obs_t = [[0.0f32; OBS_DIM]; N];
        let mut mask_t = [[false; NA]; N];
        let mut speed_mask_t = [[false; NS]; N];
        let mut mact_t = [A_STAY; N]; // macro action
        let mut spd_t = [SPD_STAND; N]; // speed gear
        let mut packed_a = [A_STAY; N]; // packed (action + speed*NA) for step
        let mut logpa_t = [0.0f32; N];
        let mut logps_t = [0.0f32; N];

        // CENTRALIZED critic: one value for the whole global state this tick.
        let gstate = w.global_state();
        let v_central = policy.critic.predict(&gstate)[0];

        for i in 1..N {
            // policy controls the 4 outfielders; GK (index 0) is rule-based.
            let obs = w.observe(Team::A, i);
            let mask = w.legal_mask(Team::A, i);
            // action policy (masked) and SEPARATE speed policy (phase-masked).
            let logits = policy.actor.predict(&obs);
            let aprobs = masked_softmax(&logits, &mask);
            let a = rng.sample_categorical(&aprobs);
            let pa = aprobs[a].max(1e-8);
            let speed_mask = w.speed_mask(Team::A, i, a);
            let slogits = policy.speedor.predict(&obs);
            let sprobs = masked_softmax(&slogits, &speed_mask);
            // Curriculum: fixed gear during warmup, sampled speed policy afterwards.
            let s = if SPEED_FROZEN.load(Ordering::Relaxed) {
                w.coerce_speed_gear(Team::A, i, a, SPD_RUN_MED)
            } else {
                rng.sample_categorical(&sprobs)
            };
            let ps = sprobs[s].max(1e-8);
            obs_t[i] = obs;
            mask_t[i] = mask;
            speed_mask_t[i] = speed_mask;
            mact_t[i] = a;
            spd_t[i] = s;
            packed_a[i] = a + s * NA;
            logpa_t[i] = pa.ln();
            logps_t[i] = ps.ln();
        }

        let pre_field = FieldRewardContext::from_world(&w);
        // Team B = the frozen self-play champion when one is installed, else the
        // scripted baseline. Both flow through World::step (physics-bounded).
        let act_b = match selfplay_champion() {
            Some(champ) => champion_actions(&champ, &w, opponent_noise, rng),
            None => noisy_scripted_actions(&w, Team::B, opponent_noise, rng),
        };
        w.step(&packed_a, &act_b, rng);
        let post_field = FieldRewardContext::from_world(&w);
        if w.ev_pass_attempt_a {
            pending_pass_launch_step = Some(step);
        }
        if w.ev_shot_attempt_a {
            pending_shot_launch_step = Some(step);
        }

        // team reward (Team A perspective)
        let phi = w.potential_a();
        let shaping = GAMMA * phi - phi_prev;
        phi_prev = phi;
        let mut r = rw().shape * shaping;
        let mut delayed_pass_credit = 0.0f32;
        let mut delayed_goal_credit = 0.0f32;
        if w.ev_goal_a {
            r += rw().goal; // GOALS are the prize
            delayed_goal_credit += rw().goal;
        }
        if w.ev_goal_b {
            r -= rw().concede;
        }
        // MARL SYNCHRONIZATION: reward the TEAM for collectively moving into a
        // configuration where a scoring chance becomes available (potential-based on
        // the best manufactured chance -> telescopes, unfarmable). This is what pulls
        // attackers to move TOGETHER to create and get off a shot.
        r += rw().chance
            * FieldRewardContext::delta(post_field.chance_value, pre_field.chance_value);
        // Only a tiny nudge for a completed pass (prefer it to a loose turnover);
        // forward progress is rewarded by the potential shaping above, and goals
        // dominate — so passing stays INSTRUMENTAL and the policy still attacks.
        if w.ev_pass_completed_a {
            let n = w.pass_streak_a; // completed passes so far in THIS possession
                                     // Small flat credit (prefer a completed pass to a loose turnover) PLUS a
                                     // forward-PROGRESS bonus. Progress is bounded by field length and lateral
                                     // recycling gains ~0, so it can't be farmed by tiki-taka.
            let field_pass = (0.50 + 0.75 * pre_field.pass_value).clamp(0.35, 1.25);
            let field_progress =
                (0.50 + 0.50 * post_field.safe_outlet_value + 0.25 * post_field.burst_score)
                    .clamp(0.35, 1.25);
            let completion_credit = rw().pass_credit * field_pass
                + (w.last_pass_gain_a.max(0.0) * 0.1 * field_progress).min(0.8);
            r += completion_credit;
            delayed_pass_credit += completion_credit;
            let field_credit = rw().field_pass
                * FieldRewardContext::delta(
                    post_field.safe_outlet_value,
                    pre_field.safe_outlet_value,
                );
            r += field_credit;
            delayed_pass_credit += field_credit;
            // MILESTONE: the 2nd completed pass unlocks a legal shot. The pinball used
            // to farm this via pass-pass-shoot-repeat — now the SHOT COOLDOWN breaks
            // that loop (the rapid follow-up shot pays nothing), so the milestone is
            // safe to reward again for genuine build-up.
            if n == 2 {
                r += rw().milestone;
                delayed_pass_credit += rw().milestone;
            }
            // Anti-recycle: escalating penalty for sterile long possessions.
            if n > 6 {
                let recycle_penalty = rw().recycle * (n - 6) as f32;
                r -= recycle_penalty;
                delayed_pass_credit -= recycle_penalty;
            }
        }
        if w.ev_turnover_a {
            let field_risk = (0.75 + pre_field.turnover_risk).clamp(0.75, 1.75);
            let turnover_penalty = rw().turnover * field_risk;
            r -= turnover_penalty;
            delayed_pass_credit -= turnover_penalty;
            if w.ev_bad_pass_turnover_a {
                let bad_pass_penalty =
                    rw().bad_pass_turnover * (0.75 + pre_field.safe_outlet_value);
                r -= bad_pass_penalty;
                delayed_pass_credit -= bad_pass_penalty;
            }
            if w.ev_dribble_turnover_a {
                r -= rw().dribble_turnover * (0.75 + pre_field.dribble_pressure);
            }
            let field_turnover_penalty = rw().field_turnover
                * (post_field.turnover_risk + pre_field.safe_outlet_value).clamp(0.0, 1.5);
            r -= field_turnover_penalty;
            delayed_pass_credit -= field_turnover_penalty;
        }
        // SHOT ON GOAL from the opponent's half after 2 passes is EARNED and
        // rewarded HANDSOMELY (not just goals) — a strong base plus a chance-quality
        // bonus, to pull the policy out of passive holding and toward shooting.
        // Shoot-spam is stopped structurally by the cooldown: a rapid-fire repeat
        // shot (fired while a prior shot is still "hot") pays nothing.
        if w.ev_shot_on_a && !w.shot_was_rapid_a {
            // DYNAMIC, position-dependent shot reward: scaled by the xG of WHERE the
            // shot was taken (distance + angle to goal). A close central chance pays
            // full; a hopeful long-range/wide pot-shot (now legal from the whole
            // opponent half) pays almost nothing — so the policy must work the ball
            // into a good position, not just fling it goalward.
            r += on_frame_shot_reward(rw(), w.last_shot_quality_a, w.last_shot_xg_a);
        }
        // reward winning the ball back (pressing / interceptions / tackles)
        if w.ev_win_ball_a {
            r += rw().win_ball * (0.75 + 0.25 * post_field.goalside_score);
        }
        // dribbling = possessing the ball (less pinball): forward pays, lateral a
        // little. SMALL per-tick — it fires every tick you carry, so a big value
        // accumulates into ball-hoarding that dwarfs goals.
        if w.ev_dribble_fwd_a {
            r += rw().dribble * pre_field.carry_value;
        }
        // (lateral dribble: no flat reward — potential shaping handles real progress)
        // Dispossessed while carrying is handled by ev_dribble_turnover_a above,
        // so the magnitude is tunable and visible to the search harness.
        // Ping-pong: A→B→A (one return, streak 1) is FINE. It only becomes a
        // problem from the SECOND return (A→B→A→B, streak 2), and worse each time.
        // 5x heavier than before, and heavier still if the exchange hasn't
        // advanced the ball 5+ yards upfield (pointless tapping in place).
        if w.ev_return_pass_a {
            let k = w.return_streak_a;
            let mut pen = if k <= 1 {
                rw().return_pass
            } else {
                rw().return_pass * 2f32.powi((k - 1) as i32)
            };
            if w.ball.y - w.return_start_x < 5.0 {
                pen += rw().return_stale; // no upfield progress -> heavier
            }
            pen *= 1.0 + 0.50 * pre_field.safe_outlet_value + 0.50 * pre_field.return_stale;
            r -= pen.min(24.0);
        }

        r += rw().field_goalside_delta
            * FieldRewardContext::delta(post_field.goalside_score, pre_field.goalside_score);
        r += rw().field_burst_delta
            * FieldRewardContext::delta(post_field.burst_score, pre_field.burst_score);

        if w.ev_pass_completed_a || w.ev_bad_pass_turnover_a {
            backdate_action_credit(
                &mut rew_buf,
                pending_pass_launch_step,
                step,
                &mut r,
                delayed_pass_credit,
                deferred_credit_fraction,
            );
            pending_pass_launch_step = None;
        }
        if w.ev_goal_a {
            backdate_action_credit(
                &mut rew_buf,
                pending_shot_launch_step,
                step,
                &mut r,
                delayed_goal_credit,
                deferred_credit_fraction,
            );
            pending_shot_launch_step = None;
        } else if pending_shot_launch_step.is_some() && w.owner.is_some() {
            pending_shot_launch_step = None;
        }

        // Controlled possession phase. A free/in-flight ball is a true 50:50,
        // never attack merely because we touched it last.
        let phase = w.possession_phase_for(Team::A);
        let our_phase = phase == PossessionPhase::Possession;
        let their_phase = phase == PossessionPhase::Dispossession;

        // PER-PLAYER coordination (MARL): teammate spacing (anti-bunch) PLUS a
        // possession-conditioned positioning reward —
        //   OFFENSE: advance UPFIELD and OPEN UP into space to receive a pass.
        //   DEFENSE: get GOALSIDE of the ball (between the ball and our own goal).
        let mut sp_t = [0.0f32; N];
        for i in 1..N {
            let pos = w.a[i].pos;
            let mut nd = f32::INFINITY;
            for j in 1..N {
                if i != j {
                    let d = w.a[j].pos.sub(pos).len();
                    if d < nd {
                        nd = d;
                    }
                }
            }
            if nd.is_finite() {
                // LINGER GATE: brief closeness (crossing runs, converging on the ball)
                // is fine — real players run right past each other — so only SUSTAINED
                // lingering in the same radius is soft-penalized. Separation DECAYS the
                // counter (leaky integrator) rather than resetting it, so the policy
                // can't farm the gate by oscillating in and out of the radius; a true
                // one-time crossing decays to 0. SEVERE overlap (< SEVERE_RADIUS) is
                // never "running past" and is penalized ungated. Sustained 1.5-3yd
                // bunching is caught by the hard resolve_same_team_spacing() backstop
                // (game.rs), so the soft reward can stay lenient in that transient zone.
                if nd < LINGER_RADIUS {
                    close_ticks[i] += 1;
                } else {
                    close_ticks[i] = close_ticks[i].saturating_sub(LINGER_DECAY);
                }
                let mut sr = spacing_reward(nd);
                if sr < 0.0 && nd >= SEVERE_RADIUS && close_ticks[i] <= linger_gate {
                    sr = 0.0; // transient soft-band closeness — not yet a lingering bunch
                }
                sp_t[i] = w_spacing_coeff * sr;
            }
            let is_carrier = matches!(w.owner, Some(o) if o.team == Team::A && o.idx == i);
            // ANTI-PASSIVITY: the STAND gear must not be a free way to farm the
            // positional shaping. An off-ball player who freezes is penalized, so
            // players keep moving and (in possession) make their runs.
            if !is_carrier && spd_t[i] == SPD_STAND {
                sp_t[i] -= rw().stand_pen;
            }
            // LOOSE-BALL PURSUIT (POMDP): when the ball is FREE, the FAVORITE — high
            // belief it wins the race to the ball's decelerating trajectory — is
            // rewarded for actually closing on its intercept point. Belief-gated so
            // only the favorite commits and teammates hold shape (no crashing the
            // ball / bunching). Defenders (idx 1,2) press a bit harder so they track
            // back and contest a loose ball instead of ball-watching.
            if w.owner.is_none() {
                let belief = w.loose_ball_belief(Team::A, i);
                if belief > 0.05 {
                    let ip = w.intercept_point(w.a[i].pos);
                    let to_ip = ip.sub(w.a[i].pos);
                    let d = to_ip.len();
                    if d > 0.5 {
                        let closing = (w.a[i].vel.y * to_ip.y + w.a[i].vel.x * to_ip.x) / d;
                        let def_bonus = if i <= 2 { 1.3 } else { 1.0 };
                        sp_t[i] +=
                            rw().pursuit * belief * def_bonus * (closing / 8.5).clamp(0.0, 1.0);
                    }
                }
            }
            if our_phase {
                // Advance upfield (attack frame +x). No offsides rule, so reward
                // getting into the ATTACKING HALF and keep rewarding all the way to
                // the opponent goal — attackers should camp high, not hold at half.
                let advance = pos.y / FIELD_L - 0.5;
                // OPEN = a CLEAR passing lane from the ball to me (no defender in
                // between). NOT merely far from a marker — a player inline behind a
                // defender is NOT open. This is the MECHANISM for "opening up": the
                // SOLUTION the policy should learn is to move to a WIDER position,
                // which clears the lane (hence width is rewarded strongly too).
                let open = w.lane_clearness(Team::A, w.ball, pos) - 0.5;
                // WIDTH: how far off the central lane (0 = center, 1 = touchline).
                let wide = (pos.x - FIELD_W / 2.0).abs() / (FIELD_W / 2.0);
                let width = wide - 0.4; // penalize the central lane, reward stretching
                                        // FLANK affinity: convex bonus for genuinely committing to a
                                        // left/right channel. With the 8-yd anti-bunch this splits the
                                        // front line across BOTH flanks instead of clustering one side.
                let flank = if wide > 0.5 { (wide - 0.5) * 2.0 } else { 0.0 };
                sp_t[i] += rw().advance * advance
                    + rw().open * open
                    + rw().width * width
                    + rw().flank * flank;

                // ── THE KEY MARL BEHAVIOUR ─────────────────────────────────────
                // When a TEAMMATE has the ball, the other attackers must run/sprint
                // upfield to offer a forward pass. Rewarded for OFF-BALL players:
                //   (1) be an upfield OUTLET — ahead of the ball, in a clear lane;
                //   (2) MAKE THE RUN — actual forward velocity (what the sprint
                //       gears are for). Together this pulls the whole line upfield
                //       in unison the moment we win possession.
                if !is_carrier {
                    let ahead = ((pos.y - w.ball.y) / 12.0).clamp(0.0, 1.0);
                    let lane = w.lane_clearness(Team::A, w.ball, pos);
                    let make_run = (w.a[i].vel.y / 8.5).clamp(0.0, 1.0); // fwd speed, ~run_fast = 1.0
                    let burst = if spd_t[i] >= SPD_SPRINT {
                        1.0
                    } else if spd_t[i] >= SPD_RUN_FAST {
                        0.75
                    } else if spd_t[i] >= SPD_RUN_SLOW {
                        0.35
                    } else {
                        -0.5
                    };
                    sp_t[i] += rw().ahead * ahead * lane
                        + rw().make_run * make_run
                        + rw().burst_gear * burst * lane * (0.5 + ahead);
                }
            } else if their_phase {
                // goalside of the ball: our goal is at x=0, so reward being at a
                // LOWER x than the ball (between ball and own goal).
                let goalside = ((w.ball.y - pos.y) / 8.0).clamp(-1.0, 1.0);
                sp_t[i] += rw().goalside * goalside;
                if pos.y > w.ball.y {
                    let recovery_run = (-w.a[i].vel.y / 8.5).clamp(0.0, 1.0);
                    sp_t[i] += rw().goalside_run * recovery_run;
                }
            }
        }

        obs_buf.push(obs_t);
        mask_buf.push(mask_t);
        speed_mask_buf.push(speed_mask_t);
        act_buf.push(mact_t);
        spd_buf.push(spd_t);
        logpa_buf.push(logpa_t);
        logps_buf.push(logps_t);
        val_buf.push(v_central);
        gstate_buf.push(gstate);
        rew_buf.push(r);
        space_buf.push(sp_t);
    }

    // GAE per agent, baselined by the SHARED centralized value V(global_state).
    // Each agent's per-tick reward = team reward + its own spacing reward; the
    // centralized critic is trained (below) to predict these returns from the
    // global state — a valid, lower-variance baseline (MAPPO/CTDE).
    let t = rew_buf.len();
    // AlphaZero-style full-game label: broadcast win/loss to every transition so
    // early POMDP/MPC decisions receive outcome credit instead of the terminal
    // signal decaying to ~zero over a long episode.
    let outcome_label = full_game_outcome_label(w.goals_a, w.goals_b);
    let actor_outcome_credit = value_learning_units(full_game_actor_credit(outcome_label));
    let mut samples = Vec::with_capacity(t * (N - 1));
    for i in 1..N {
        let mut adv = 0.0f32;
        let mut next_v = 0.0f32; // bootstrap 0 at horizon end
        for s in (0..t).rev() {
            let v = val_buf[s]; // shared centralized value
            let reward = value_learning_units(rew_buf[s] + space_buf[s][i]);
            let delta = reward + GAMMA * next_v - v;
            adv = delta + GAMMA * LAMBDA * adv;
            let outcome_adjusted_adv = adv + actor_outcome_credit;
            // Full-strength terminal backpropagation remains in the centralized
            // value target; only the noisy actor contribution is scaled.
            let ret = adv + value_learning_units(outcome_label) + v;
            next_v = v;
            samples.push(Sample {
                obs: obs_buf[s][i],
                gstate: gstate_buf[s],
                mask: mask_buf[s][i],
                speed_mask: speed_mask_buf[s][i],
                action: act_buf[s][i],
                speed: spd_buf[s][i],
                old_logp_a: logpa_buf[s][i],
                old_logp_s: logps_buf[s][i],
                adv: outcome_adjusted_adv,
                ret,
            });
        }
    }
    samples
}

fn collect_rollouts(policy: &Policy, games: usize, rng: &mut Rng) -> Vec<Sample> {
    let jobs: Vec<(u64, f32)> = (0..games)
        .map(|_| (rng.next_u64(), opponent_noise(rng)))
        .collect();
    let workers = rollout_worker_count(games);
    if workers <= 1 {
        return run_rollout_jobs(policy, &jobs);
    }

    let chunk = jobs.len().div_ceil(workers);
    let mut handles = Vec::new();
    for job_chunk in jobs.chunks(chunk) {
        let local_policy = policy.clone();
        let local_jobs = job_chunk.to_vec();
        handles.push(std::thread::spawn(move || {
            run_rollout_jobs(&local_policy, &local_jobs)
        }));
    }

    let mut data = Vec::new();
    let mut worker_failed = false;
    for handle in handles {
        match handle.join() {
            Ok(samples) => data.extend(samples),
            Err(_) => worker_failed = true,
        }
    }
    if worker_failed {
        eprintln!("warning: 5v5 rollout worker panicked; retrying rollout batch serially");
        return run_rollout_jobs(policy, &jobs);
    }
    data
}

fn run_rollout_jobs(policy: &Policy, jobs: &[(u64, f32)]) -> Vec<Sample> {
    let mut data = Vec::new();
    for &(seed, noise) in jobs {
        let mut local_rng = Rng::new(seed);
        data.extend(rollout(policy, &mut local_rng, noise));
    }
    data
}

pub fn rollout_worker_count(games: usize) -> usize {
    if games <= 1 {
        return 1;
    }
    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let env_limit = std::env::var("FIVEASIDE_ROLLOUT_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(MAX_ROLLOUT_THREADS);
    games.min(available).min(env_limit).max(1)
}

fn opponent_noise(rng: &mut Rng) -> f32 {
    let r = rng.f01();
    if r < 0.15 {
        0.18
    } else if r < 0.35 {
        0.08
    } else {
        0.0
    }
}

fn noisy_scripted_actions(w: &World, team: Team, noise: f32, rng: &mut Rng) -> [usize; N] {
    let mut acts = w.scripted_actions(team);
    if noise <= 0.0 {
        return acts;
    }
    for i in 1..N {
        if rng.f01() < noise {
            let mask = w.legal_mask(team, i);
            let a = sample_legal_action(&mask, rng);
            let gear = w.coerce_speed_gear(team, i, a, scripted_gear(a));
            acts[i] = a + gear * NA; // keep the noisy move at a real speed
        }
    }
    acts
}

/// Team-B actions from a frozen self-play champion (keeper held, like the
/// challenger's keeper). A small epsilon of legal exploration keeps the opposition
/// varied so the challenger doesn't overfit one deterministic champion line.
fn champion_actions(champ: &Policy, w: &World, noise: f32, rng: &mut Rng) -> [usize; N] {
    let mut acts = [A_STAY; N];
    for i in 1..N {
        if noise > 0.0 && rng.f01() < noise {
            let mask = w.legal_mask(Team::B, i);
            let a = sample_legal_action(&mask, rng);
            let gear = w.coerce_speed_gear(Team::B, i, a, scripted_gear(a));
            acts[i] = a + gear * NA;
        } else {
            acts[i] = champ.act_greedy_world(w, Team::B, i);
        }
    }
    acts
}

fn sample_legal_action(mask: &[bool; NA], rng: &mut Rng) -> usize {
    let n = mask.iter().filter(|&&ok| ok).count();
    if n == 0 {
        return A_STAY;
    }
    let pick = (rng.next_u64() % n as u64) as usize;
    let mut seen = 0usize;
    for (idx, &ok) in mask.iter().enumerate() {
        if ok {
            if seen == pick {
                return idx;
            }
            seen += 1;
        }
    }
    A_STAY
}

fn legal_or_first(action: usize, mask: &[bool; NA]) -> Option<usize> {
    if action < NA && mask[action] {
        Some(action)
    } else {
        mask.iter().position(|&ok| ok)
    }
}

pub fn behavior_clone_scripted(
    policy: &mut Policy,
    games: usize,
    epochs: usize,
    rng: &mut Rng,
) -> CloneStats {
    behavior_clone(policy, games, epochs, 1.0, rng)
}

/// DAgger correction: collect analytic labels on states induced primarily by
/// the current neural policy. A bounded teacher roll-in fraction keeps rare
/// build-up and finishing states reachable while training away compounding
/// behavior-cloning errors on the learner's own state distribution.
pub fn behavior_clone_dagger(
    policy: &mut Policy,
    games: usize,
    epochs: usize,
    teacher_rollin: f32,
    rng: &mut Rng,
) -> CloneStats {
    behavior_clone(policy, games, epochs, teacher_rollin.clamp(0.0, 0.75), rng)
}

fn behavior_clone(
    policy: &mut Policy,
    games: usize,
    epochs: usize,
    teacher_rollin: f32,
    rng: &mut Rng,
) -> CloneStats {
    if games == 0 || epochs == 0 {
        return CloneStats {
            samples: 0,
            shot_samples: 0,
            loss: 0.0,
            accuracy: 0.0,
        };
    }

    let mut data = Vec::new();
    for _ in 0..games {
        let mut w = World::new();
        if rng.f01() < 0.5 {
            w.kickoff(Team::B);
        }
        for _ in 0..STEPS {
            let teacher_a = w.scripted_actions(Team::A);
            let mut act_a = teacher_a;
            for i in 1..N {
                let obs = w.observe(Team::A, i);
                let mask = w.legal_mask(Team::A, i);
                if let Some(action) = legal_or_first(teacher_a[i] % NA, &mask) {
                    let speed_mask = w.speed_mask(Team::A, i, action);
                    let gear = w.coerce_speed_gear(Team::A, i, action, teacher_a[i] / NA);
                    data.push(BcSample {
                        obs,
                        mask,
                        speed_mask,
                        action,
                        speed: gear,
                        actor_weight: 1.0,
                    });
                }
                if teacher_rollin < 1.0 && rng.f01() >= teacher_rollin {
                    act_a[i] = policy.act_greedy_world(&w, Team::A, i);
                }
            }
            let act_b = w.scripted_actions(Team::B);
            w.step(&act_a, &act_b, rng);
        }
    }

    let mut action_counts = [0usize; NA];
    for sample in &data {
        action_counts[sample.action] += 1;
    }
    let max_action_count = action_counts.iter().copied().max().unwrap_or(1);
    for sample in &mut data {
        sample.actor_weight = behavior_clone_actor_weight(
            sample.action,
            action_counts[sample.action],
            max_action_count,
        );
    }

    let mut idx: Vec<usize> = (0..data.len()).collect();
    let mut loss_accum = 0.0f32;
    let mut correct = 0.0f32;
    let mut count = 0.0f32;
    let shot_samples = data
        .iter()
        .filter(|sample| sample.action == A_SHOOT)
        .count();
    for _ in 0..epochs {
        shuffle(&mut idx, rng);
        for chunk in idx.chunks(MINIBATCH) {
            for &si in chunk {
                let s = &data[si];
                // clone the scripted ACTION into the action network…
                let acts = policy.actor.forward(&s.obs);
                let logits = acts.last().unwrap();
                let probs = masked_softmax(logits, &s.mask);
                let p = probs[s.action].max(1e-8);
                loss_accum += -p.ln() * s.actor_weight;
                if probs
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| s.mask[*i])
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .is_some_and(|(i, _)| i == s.action)
                {
                    correct += s.actor_weight;
                }
                let mut d_logits = vec![0.0f32; NA];
                for j in 0..NA {
                    if s.mask[j] {
                        d_logits[j] =
                            (probs[j] - if j == s.action { 1.0 } else { 0.0 }) * s.actor_weight;
                    }
                }
                policy.actor.backward(&acts, &d_logits);
                // …and the scripted GEAR into the separate speed network.
                let sacts = policy.speedor.forward(&s.obs);
                let sprobs = masked_softmax(sacts.last().unwrap(), &s.speed_mask);
                let mut d_slogits = vec![0.0f32; NS];
                for k in 0..NS {
                    if s.speed_mask[k] {
                        d_slogits[k] = sprobs[k] - if k == s.speed { 1.0 } else { 0.0 };
                    }
                }
                policy.speedor.backward(&sacts, &d_slogits);
                count += s.actor_weight;
            }
            policy.actor.step(LR_BC);
            policy.speedor.step(LR_BC);
        }
    }

    CloneStats {
        samples: data.len(),
        shot_samples,
        loss: loss_accum / count.max(1.0),
        accuracy: correct / count.max(1.0),
    }
}

pub struct IterStats {
    pub avg_reward: f32,
    pub entropy: f32,
    pub value_loss: f32,
}

/// One PPO iteration: collect `games` rollouts, then EPOCHS of minibatch updates.
pub fn train_iter(policy: &mut Policy, games: usize, ent_beta: f32, rng: &mut Rng) -> IterStats {
    let mut data = collect_rollouts(policy, games, rng);
    let (actor_learning_rate, critic_learning_rate, speed_learning_rate) = ppo_learning_rates();
    let mut total_r = 0.0f32;
    // returns/advs summed across all player-steps; report mean reward-to-go proxy
    for s in &data {
        total_r += s.ret * VALUE_TARGET_SCALE_POINTS;
    }
    let avg_reward = total_r / data.len().max(1) as f32;

    // normalize advantages (across the batch)
    let mean_adv = data.iter().map(|s| s.adv).sum::<f32>() / data.len().max(1) as f32;
    let var_adv =
        data.iter().map(|s| (s.adv - mean_adv).powi(2)).sum::<f32>() / data.len().max(1) as f32;
    let std_adv = var_adv.sqrt().max(1e-6);
    for s in data.iter_mut() {
        s.adv = (s.adv - mean_adv) / std_adv;
    }

    let mut idx: Vec<usize> = (0..data.len()).collect();
    let mut ent_accum = 0.0f32;
    let mut vloss_accum = 0.0f32;
    let mut count = 0.0f32;

    for _ in 0..EPOCHS {
        shuffle(&mut idx, rng);
        for chunk in idx.chunks(MINIBATCH) {
            for &si in chunk {
                let s = &data[si];
                let a = s.adv;
                let clip_coeff = |ratio: f32| -> f32 {
                    if a >= 0.0 {
                        if ratio <= 1.0 + CLIP {
                            a * ratio
                        } else {
                            0.0
                        }
                    } else if ratio >= 1.0 - CLIP {
                        a * ratio
                    } else {
                        0.0
                    }
                };

                // ---- action policy (its OWN network — identical to v3) ----
                let acts = policy.actor.forward(&s.obs);
                let logits = acts.last().unwrap();
                let aprobs = masked_softmax(logits, &s.mask);
                let p_a = aprobs[s.action].max(1e-8);
                let coeff_a = clip_coeff((p_a.ln() - s.old_logp_a).exp());
                let mut ent = 0.0f32;
                for i in 0..NA {
                    if s.mask[i] && aprobs[i] > 1e-8 {
                        ent -= aprobs[i] * aprobs[i].ln();
                    }
                }
                let mut d_logits = vec![0.0f32; NA];
                for j in 0..NA {
                    if !s.mask[j] {
                        continue;
                    }
                    let ind = if j == s.action { 1.0 } else { 0.0 };
                    let pg = -coeff_a * (ind - aprobs[j]);
                    let eg = ent_beta * aprobs[j] * (aprobs[j].max(1e-8).ln() + ent);
                    d_logits[j] = pg + eg;
                }
                policy.actor.backward(&acts, &d_logits);

                // ---- speed policy (SEPARATE network) ----
                let sacts = policy.speedor.forward(&s.obs);
                let slogits = sacts.last().unwrap();
                let sprobs = masked_softmax(slogits, &s.speed_mask);
                let p_s = sprobs[s.speed].max(1e-8);
                let coeff_s = clip_coeff((p_s.ln() - s.old_logp_s).exp());
                let mut ent_s = 0.0f32;
                for k in 0..NS {
                    if s.speed_mask[k] && sprobs[k] > 1e-8 {
                        ent_s -= sprobs[k] * sprobs[k].ln();
                    }
                }
                let mut d_slogits = vec![0.0f32; NS];
                for k in 0..NS {
                    if !s.speed_mask[k] {
                        continue;
                    }
                    let ind = if k == s.speed { 1.0 } else { 0.0 };
                    let pg = -coeff_s * (ind - sprobs[k]);
                    let eg =
                        ent_beta * SPEED_ENT_SCALE * sprobs[k] * (sprobs[k].max(1e-8).ln() + ent_s);
                    d_slogits[k] = pg + eg;
                }
                policy.speedor.backward(&sacts, &d_slogits);

                // ---- centralized critic (MSE to GAE return), on GLOBAL state ----
                let cacts = policy.critic.forward(&s.gstate);
                let v = cacts.last().unwrap()[0];
                let dv = v - s.ret; // dL/dv for 0.5*(v-ret)^2
                policy.critic.backward(&cacts, &[dv]);

                ent_accum += ent + ent_s;
                vloss_accum += 0.5 * (v - s.ret) * (v - s.ret);
                count += 1.0;
            }
            policy.actor.step(actor_learning_rate);
            policy.speedor.step(speed_learning_rate);
            policy.critic.step(critic_learning_rate);
        }
    }

    IterStats {
        avg_reward,
        entropy: ent_accum / count.max(1.0),
        value_loss: vloss_accum / count.max(1.0),
    }
}

pub fn ent_beta_at(iter: usize, total: usize) -> f32 {
    // linear decay of entropy bonus
    let frac = 1.0 - (iter as f32 / total.max(1) as f32);
    ENT_BETA0 * frac.max(0.1)
}

/// Rich per-evaluation analytics (all averaged per game unless noted).
#[derive(Clone, Default)]
pub struct Stats {
    pub goal_diff: f32,
    pub winrate: f32,
    pub ga: f32,
    pub gb: f32,
    pub spacing: f32,       // avg nearest-teammate distance (all ticks)
    pub bunch: f32,         // fraction of ticks a pair < 2.5
    pub possession: f32,    // fraction of ticks Team A holds the ball
    pub pass_att: f32,      // pass attempts / game
    pub pass_cmp: f32,      // completed passes / game
    pub pass_fwd: f32,      // forward pass attempts / game
    pub pass_lat: f32,      // lateral pass attempts / game
    pub pass_back: f32,     // backward pass attempts / game
    pub pass_cmp_fwd: f32,  // completed forward passes / game
    pub pass_cmp_lat: f32,  // completed lateral passes / game
    pub pass_cmp_back: f32, // completed backward passes / game
    pub shots: f32,         // shot attempts / game
    pub shots_scored: f32,  // goals (proxy for converted shots) / game
    pub turnovers: f32,     // A turnovers / game
    pub wins_won: f32,      // balls won / game (press/intercept/tackle)
}
impl Stats {
    pub fn pass_completion(&self) -> f32 {
        if self.pass_att > 0.0 {
            self.pass_cmp / self.pass_att
        } else {
            0.0
        }
    }
    pub fn forward_pass_completion(&self) -> f32 {
        directional_pass_completion(self.pass_cmp_fwd, self.pass_fwd)
    }
    pub fn lateral_pass_completion(&self) -> f32 {
        directional_pass_completion(self.pass_cmp_lat, self.pass_lat)
    }
    pub fn backward_pass_completion(&self) -> f32 {
        directional_pass_completion(self.pass_cmp_back, self.pass_back)
    }
    pub fn conversion(&self) -> f32 {
        if self.shots > 0.0 {
            self.shots_scored / self.shots
        } else {
            0.0
        }
    }
}

fn directional_pass_completion(completed: f32, attempted: f32) -> f32 {
    if attempted > 0.0 {
        (completed / attempted).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Evaluate greedy policy vs scripted over `games`, collecting full analytics.
pub fn evaluate(policy: &Policy, games: usize, rng: &mut Rng) -> Stats {
    let mut s = Stats::default();
    let mut space_sum = 0.0f32;
    let mut ticks = 0.0f32;
    let mut poss_ticks = 0.0f32;
    let mut bunch_ticks = 0.0f32;
    for _ in 0..games {
        let mut w = World::new();
        if rng.f01() < 0.5 {
            w.kickoff(Team::B);
        }
        for _ in 0..STEPS {
            let mut act_a = [A_STAY; N];
            for i in 1..N {
                act_a[i] = policy.act_greedy_world(&w, Team::A, i);
            }
            let act_b = w.scripted_actions(Team::B);
            w.step(&act_a, &act_b, rng);
            if w.ev_pass_completed_a {
                s.pass_cmp += 1.0;
                match w.pass_dir_a {
                    1 => s.pass_cmp_fwd += 1.0,
                    -1 => s.pass_cmp_back += 1.0,
                    _ => s.pass_cmp_lat += 1.0,
                }
            }
            if w.ev_pass_attempt_a {
                s.pass_att += 1.0;
                match w.pass_dir_a {
                    1 => s.pass_fwd += 1.0,
                    -1 => s.pass_back += 1.0,
                    _ => s.pass_lat += 1.0,
                }
            }
            if w.ev_shot_attempt_a {
                s.shots += 1.0;
            }
            if w.ev_turnover_a {
                s.turnovers += 1.0;
            }
            if w.ev_win_ball_a {
                s.wins_won += 1.0;
            }
            space_sum += w.avg_nearest_teammate_a();
            ticks += 1.0;
            if matches!(w.owner, Some(o) if matches!(o.team, Team::A)) {
                poss_ticks += 1.0;
            }
            if w.closest_pair_a() < 2.5 {
                bunch_ticks += 1.0;
            }
        }
        let d = w.goals_a as f32 - w.goals_b as f32;
        s.goal_diff += d;
        s.ga += w.goals_a as f32;
        s.gb += w.goals_b as f32;
        s.shots_scored += w.goals_a as f32; // proxy: goals as converted shots
        if d > 0.0 {
            s.winrate += 1.0;
        } else if d == 0.0 {
            s.winrate += 0.5;
        }
    }
    let g = games.max(1) as f32;
    for v in [
        &mut s.goal_diff,
        &mut s.winrate,
        &mut s.ga,
        &mut s.gb,
        &mut s.pass_att,
        &mut s.pass_cmp,
        &mut s.pass_fwd,
        &mut s.pass_lat,
        &mut s.pass_back,
        &mut s.pass_cmp_fwd,
        &mut s.pass_cmp_lat,
        &mut s.pass_cmp_back,
        &mut s.shots,
        &mut s.shots_scored,
        &mut s.turnovers,
        &mut s.wins_won,
    ] {
        *v /= g;
    }
    s.spacing = if ticks > 0.0 { space_sum / ticks } else { 0.0 };
    s.bunch = if ticks > 0.0 {
        bunch_ticks / ticks
    } else {
        0.0
    };
    s.possession = if ticks > 0.0 { poss_ticks / ticks } else { 0.0 };
    s
}

/// Head-to-head between two learned policies: `cand` plays Team A, `opp` plays
/// Team B (keepers held, as in `evaluate`). Returns (mean goal_diff for `cand`,
/// `cand` winrate with draws counting 0.5). This is the champion-ladder gate:
/// the challenger advances only when it beats the frozen champion by a margin.
pub fn evaluate_vs_policy(cand: &Policy, opp: &Policy, games: usize, rng: &mut Rng) -> (f32, f32) {
    let mut gd = 0.0f32;
    let mut wr = 0.0f32;
    for _ in 0..games {
        let mut w = World::new();
        if rng.f01() < 0.5 {
            w.kickoff(Team::B);
        }
        for _ in 0..STEPS {
            let mut act_a = [A_STAY; N];
            let mut act_b = [A_STAY; N];
            for i in 1..N {
                act_a[i] = cand.act_greedy_world(&w, Team::A, i);
                act_b[i] = opp.act_greedy_world(&w, Team::B, i);
            }
            w.step(&act_a, &act_b, rng);
        }
        let d = w.goals_a as f32 - w.goals_b as f32;
        gd += d;
        if d > 0.0 {
            wr += 1.0;
        } else if d == 0.0 {
            wr += 0.5;
        }
    }
    let g = games.max(1) as f32;
    (gd / g, wr / g)
}

/// Side-balanced evaluation used by the promotion gate. Every seed is played
/// twice, with the candidate on each side, so a favourable Team A geometry or
/// kickoff sequence cannot manufacture a ladder promotion.
#[derive(Clone, Copy, Debug, Default)]
pub struct PairedEvaluation {
    pub games: usize,
    pub goal_diff: f32,
    pub payoff: f32,
    pub goal_diff_lower_95: f32,
    pub payoff_lower_95: f32,
}

fn summarize_paired_outcomes(goal_diffs: &[f32]) -> PairedEvaluation {
    let games = goal_diffs.len();
    if games == 0 {
        return PairedEvaluation::default();
    }
    let n = games as f32;
    let goal_diff = goal_diffs.iter().sum::<f32>() / n;
    let payoffs: Vec<f32> = goal_diffs
        .iter()
        .map(|diff| {
            if *diff > 0.0 {
                1.0
            } else if *diff < 0.0 {
                0.0
            } else {
                0.5
            }
        })
        .collect();
    let payoff = payoffs.iter().sum::<f32>() / n;
    let lower_95 = |samples: &[f32], mean: f32| {
        if samples.len() < 2 {
            return f32::NEG_INFINITY;
        }
        let variance = samples
            .iter()
            .map(|sample| {
                let delta = *sample - mean;
                delta * delta
            })
            .sum::<f32>()
            / (samples.len() - 1) as f32;
        mean - 1.96 * (variance / samples.len() as f32).sqrt()
    };
    PairedEvaluation {
        games,
        goal_diff,
        payoff,
        goal_diff_lower_95: lower_95(goal_diffs, goal_diff),
        payoff_lower_95: lower_95(&payoffs, payoff),
    }
}

fn evaluate_policy_match(
    candidate: &Policy,
    opponent: Option<&Policy>,
    candidate_team: Team,
    seed: u64,
) -> f32 {
    let mut rng = Rng::new(seed);
    let mut world = World::new();
    if rng.f01() < 0.5 {
        world.kickoff(Team::B);
    }
    for _ in 0..STEPS {
        let learned_actions = |policy: &Policy, team: Team, world: &World| {
            let mut actions = [A_STAY; N];
            for i in 1..N {
                actions[i] = policy.act_greedy_world(world, team, i);
            }
            actions
        };
        let (actions_a, actions_b) = match candidate_team {
            Team::A => (
                learned_actions(candidate, Team::A, &world),
                opponent.map_or_else(
                    || world.scripted_actions(Team::B),
                    |policy| learned_actions(policy, Team::B, &world),
                ),
            ),
            Team::B => (
                opponent.map_or_else(
                    || world.scripted_actions(Team::A),
                    |policy| learned_actions(policy, Team::A, &world),
                ),
                learned_actions(candidate, Team::B, &world),
            ),
        };
        world.step(&actions_a, &actions_b, &mut rng);
    }
    match candidate_team {
        Team::A => world.goals_a as f32 - world.goals_b as f32,
        Team::B => world.goals_b as f32 - world.goals_a as f32,
    }
}

fn evaluate_paired(
    candidate: &Policy,
    opponent: Option<&Policy>,
    games: usize,
    rng: &mut Rng,
) -> PairedEvaluation {
    let pairs = games.max(2).div_ceil(2);
    let mut goal_diffs = Vec::with_capacity(pairs * 2);
    for _ in 0..pairs {
        let seed = rng.next_u64();
        goal_diffs.push(evaluate_policy_match(candidate, opponent, Team::A, seed));
        goal_diffs.push(evaluate_policy_match(candidate, opponent, Team::B, seed));
    }
    summarize_paired_outcomes(&goal_diffs)
}

pub fn evaluate_vs_policy_paired(
    candidate: &Policy,
    opponent: &Policy,
    games: usize,
    rng: &mut Rng,
) -> PairedEvaluation {
    evaluate_paired(candidate, Some(opponent), games, rng)
}

pub fn evaluate_vs_scripted_paired(
    candidate: &Policy,
    games: usize,
    rng: &mut Rng,
) -> PairedEvaluation {
    evaluate_paired(candidate, None, games, rng)
}

/// Baseline sanity check: scripted-vs-scripted goal difference (should be ~0).
pub fn evaluate_scripted_vs_scripted(games: usize, rng: &mut Rng) -> (f32, f32, f32) {
    let mut diff = 0.0f32;
    let (mut ga, mut gb) = (0.0f32, 0.0f32);
    for _ in 0..games {
        let mut w = World::new();
        if rng.f01() < 0.5 {
            w.kickoff(Team::B);
        }
        for _ in 0..STEPS {
            let act_a = w.scripted_actions(Team::A);
            let act_b = w.scripted_actions(Team::B);
            w.step(&act_a, &act_b, rng);
        }
        diff += w.goals_a as f32 - w.goals_b as f32;
        ga += w.goals_a as f32;
        gb += w.goals_b as f32;
    }
    let g = games.max(1) as f32;
    (diff / g, ga / g, gb / g)
}

fn shuffle(v: &mut [usize], rng: &mut Rng) {
    for i in (1..v.len()).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spacing_reward_penalizes_overlap_and_prefers_useful_distance() {
        assert!(spacing_reward(0.8) < spacing_reward(2.0)); // overlap worst
        assert!(spacing_reward(3.5) < 0.0); // within 4 yds is penalized
        assert!(spacing_reward(2.5) < spacing_reward(3.5)); // within 3 penalized harder
        assert!(spacing_reward(8.0) > spacing_reward(5.0)); // optimum moved WIDE to 8
        assert!(spacing_reward(8.0) > spacing_reward(11.0)); // too far tapers off
        assert!(spacing_reward(13.0) < 0.0); // beyond 12: penalty
    }

    #[test]
    fn bounded_reward_weights_stay_positive_and_finite() {
        assert_eq!(bounded_weight(Some("0"), 1.0, 0.0, 2.0), MIN_REWARD_WEIGHT);
        assert_eq!(
            bounded_weight(Some("-5"), 1.0, MIN_REWARD_WEIGHT, 2.0),
            MIN_REWARD_WEIGHT
        );
        assert_eq!(
            bounded_weight(Some("0.00001"), 0.003, 0.0005, 0.012),
            0.0005
        );
        assert_eq!(
            bounded_weight(Some("3.5"), 1.0, MIN_REWARD_WEIGHT, 2.0),
            2.0
        );
        assert_eq!(
            bounded_weight(Some("NaN"), 1.0, MIN_REWARD_WEIGHT, 2.0),
            1.0
        );
        assert_eq!(
            bounded_weight(Some("inf"), 1.0, MIN_REWARD_WEIGHT, 2.0),
            1.0
        );
        assert_eq!(
            bounded_weight(None, 0.0, MIN_REWARD_WEIGHT, 2.0),
            MIN_REWARD_WEIGHT
        );
        assert_eq!(SHOT_BASE_POINTS_BOUNDS.0, ON_FRAME_SHOT_MIN_POINTS);
        assert!(SHOT_BASE_POINTS_BOUNDS.1 + SHOT_QUALITY_POINTS_BOUNDS.1 <= 200.0);
        assert!(PER_TICK_WEIGHT_BOUNDS.1 < SPARSE_MILESTONE_WEIGHT_BOUNDS.1);
        assert!(SPACING_WEIGHT_BOUNDS.1 < SPARSE_PASS_WEIGHT_BOUNDS.1);
    }

    #[test]
    fn football_outcomes_always_dominate_a_nonconverting_shot() {
        let (goal, shot_base, shot_q) = grounded_conversion_ladder(6.0, 18.0, 4.0);
        let weights = enforce_reward_hierarchy(Rw {
            goal,
            concede: 4.0,
            match_win: 8.0,
            match_loss: 6.0,
            shot_base,
            shot_q,
            milestone: MIN_REWARD_WEIGHT,
            pass_credit: MIN_REWARD_WEIGHT,
            turnover: MIN_REWARD_WEIGHT,
            bad_pass_turnover: MIN_REWARD_WEIGHT,
            dribble_turnover: MIN_REWARD_WEIGHT,
            recycle: MIN_REWARD_WEIGHT,
            return_pass: MIN_REWARD_WEIGHT,
            return_stale: MIN_REWARD_WEIGHT,
            win_ball: MIN_REWARD_WEIGHT,
            dribble: MIN_REWARD_WEIGHT,
            shape: 0.5,
            advance: MIN_REWARD_WEIGHT,
            open: MIN_REWARD_WEIGHT,
            width: MIN_REWARD_WEIGHT,
            flank: MIN_REWARD_WEIGHT,
            goalside: MIN_REWARD_WEIGHT,
            goalside_run: MIN_REWARD_WEIGHT,
            ahead: MIN_REWARD_WEIGHT,
            make_run: MIN_REWARD_WEIGHT,
            burst_gear: MIN_REWARD_WEIGHT,
            field_pass: MIN_REWARD_WEIGHT,
            field_turnover: MIN_REWARD_WEIGHT,
            chance: MIN_REWARD_WEIGHT,
            field_goalside_delta: MIN_REWARD_WEIGHT,
            field_burst_delta: MIN_REWARD_WEIGHT,
            stand_pen: MIN_REWARD_WEIGHT,
            pursuit: MIN_REWARD_WEIGHT,
        });
        let miss_ceiling = max_nonconverting_shot_reward(&weights);
        assert_eq!(weights.goal, GOAL_ANCHOR_POINTS);
        assert_eq!(weights.concede, GOAL_ANCHOR_POINTS);
        assert_eq!(weights.match_win, MATCH_OUTCOME_ANCHOR_POINTS);
        assert_eq!(weights.match_loss, MATCH_OUTCOME_ANCHOR_POINTS);
        assert!(weights.goal >= miss_ceiling + CONVERSION_OVER_SHOT_MARGIN);
        assert!(miss_ceiling >= ON_FRAME_SHOT_MIN_POINTS);
        assert!(miss_ceiling <= weights.goal * REWARD_NON_CONVERSION_MAX_FRACTION + 1e-5);
        assert!(weights.match_win >= weights.goal + WIN_OVER_CONVERSION_MARGIN);
        assert_eq!(on_frame_shot_reward(&weights, 0.0, 0.0), 50.0);
        assert!(on_frame_shot_reward(&weights, 1.0, 1.0) >= 50.0);
    }

    #[test]
    fn delayed_action_credit_moves_to_the_committed_pomdp_mpc_tick() {
        let mut history = vec![1.0, 2.0];
        let mut current = 5.0;
        backdate_action_credit(&mut history, Some(0), 2, &mut current, 4.0, 0.5);
        assert_eq!(history, vec![3.0, 2.0]);
        assert_eq!(current, 3.0);

        let mut penalty_now = -6.0;
        backdate_action_credit(&mut history, Some(1), 2, &mut penalty_now, -4.0, 1.0);
        assert_eq!(history[1], -2.0);
        assert_eq!(penalty_now, -2.0);
    }

    #[test]
    fn full_game_outcome_is_broadcast_with_fixed_symmetric_anchor() {
        assert_eq!(full_game_outcome_label(2, 1), 1000.0);
        assert_eq!(full_game_outcome_label(1, 2), -1000.0);
        assert_eq!(full_game_outcome_label(1, 1), 0.0);

        let winning_credit = full_game_actor_credit(full_game_outcome_label(2, 1));
        let losing_credit = full_game_actor_credit(full_game_outcome_label(1, 2));
        assert!(winning_credit >= 1.0 && winning_credit <= 100.0);
        assert_eq!(losing_credit, -winning_credit);
        assert_eq!(full_game_actor_credit(0.0), 0.0);
    }

    #[test]
    fn critic_targets_are_normalized_without_changing_football_anchors() {
        assert_eq!(GOAL_ANCHOR_POINTS, 500.0);
        assert_eq!(MATCH_OUTCOME_ANCHOR_POINTS, 1000.0);
        assert_eq!(value_learning_units(GOAL_ANCHOR_POINTS), 0.5);
        assert_eq!(value_learning_units(MATCH_OUTCOME_ANCHOR_POINTS), 1.0);
        assert_eq!(value_learning_units(-MATCH_OUTCOME_ANCHOR_POINTS), -1.0);
    }

    #[test]
    fn scripted_vs_scripted_is_seed_reproducible() {
        let mut a = Rng::new(1234);
        let mut b = Rng::new(1234);
        let left = evaluate_scripted_vs_scripted(8, &mut a);
        let right = evaluate_scripted_vs_scripted(8, &mut b);
        assert_eq!(left, right);
    }

    #[test]
    fn selfplay_champion_lock_round_trips_and_clears() {
        let mut rng = Rng::new(123);
        set_selfplay_champion(Some(Policy::new(&mut rng)));
        assert!(selfplay_champion().is_some());
        clear_selfplay_champion();
        assert!(selfplay_champion().is_none());
    }

    #[test]
    fn untrained_evaluation_metrics_stay_in_bounds() {
        let mut rng = Rng::new(99);
        let policy = Policy::new(&mut rng);
        let stats = evaluate(&policy, 2, &mut rng);
        assert!((0.0..=1.0).contains(&stats.winrate));
        assert!((0.0..=1.0).contains(&stats.bunch));
        assert!((0.0..=1.0).contains(&stats.possession));
        assert!(stats.pass_completion().is_finite());
        assert!(stats.conversion().is_finite());
    }

    #[test]
    fn directional_pass_completion_exposes_the_target_rates() {
        let stats = Stats {
            pass_att: 40.0,
            pass_cmp: 36.0,
            pass_fwd: 20.0,
            pass_cmp_fwd: 17.0,
            pass_back: 20.0,
            pass_cmp_back: 19.0,
            ..Stats::default()
        };

        assert!((stats.pass_completion() - 0.90).abs() < 1e-6);
        assert!((stats.forward_pass_completion() - 0.85).abs() < 1e-6);
        assert!((stats.backward_pass_completion() - 0.95).abs() < 1e-6);
        assert_eq!(stats.lateral_pass_completion(), 0.0);
    }

    #[test]
    fn behavior_clone_warm_start_collects_finite_teacher_signal() {
        let mut rng = Rng::new(2026);
        let mut policy = Policy::new(&mut rng);
        let stats = behavior_clone_scripted(&mut policy, 1, 1, &mut rng);
        assert!(stats.samples > 0);
        assert!(stats.loss.is_finite());
        assert!((0.0..=1.0).contains(&stats.accuracy));
    }

    #[test]
    fn dagger_collects_labels_on_neural_rollin_states() {
        let mut rng = Rng::new(2027);
        let mut policy = Policy::new(&mut rng);
        let stats = behavior_clone_dagger(&mut policy, 1, 1, 0.0, &mut rng);
        assert!(stats.samples > 0);
        assert!(stats.loss.is_finite());
        assert!((0.0..=1.0).contains(&stats.accuracy));
    }

    #[test]
    fn field_reward_context_changes_with_ten_player_geometry() {
        let mut open = World::new();
        open.owner = Some(Owner {
            team: Team::A,
            idx: 1,
        });
        open.ball = V2::new(12.0, 14.0);
        open.a[1].pos = open.ball;
        open.a[2].pos = V2::new(24.0, 7.0);
        open.a[3].pos = V2::new(24.0, 21.0);
        open.a[4].pos = V2::new(18.0, 14.0);
        for i in 1..N {
            open.b[i].pos = V2::new(38.0, 4.0 + i as f32 * 5.0);
        }

        let mut blocked = World::new();
        blocked.owner = open.owner;
        blocked.ball = open.ball;
        blocked.a = open.a;
        blocked.b = open.b;
        blocked.b[1].pos = V2::new(13.0, 14.0);
        blocked.b[2].pos = V2::new(20.0, 8.0);
        blocked.b[3].pos = V2::new(20.0, 20.0);

        let open_ctx = FieldRewardContext::from_world(&open);
        let blocked_ctx = FieldRewardContext::from_world(&blocked);

        assert!(
            open_ctx.safe_outlet_value > blocked_ctx.safe_outlet_value,
            "open field vector should create better outlet value"
        );
        assert!(
            blocked_ctx.turnover_risk > open_ctx.turnover_risk,
            "blocked/pressured field vector should increase turnover risk"
        );
    }
}
