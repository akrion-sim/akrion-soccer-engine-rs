//! Attacking training ground — two modes densify a signal that is <1% of match actions and feed
//! it to the HOME neural learner. The per-action reward stack (goal / on-target / shot-objective)
//! fires automatically inside `run_time_step`; this harness writes NO reward code. Output is a
//! `learned-params.json` in the league trainer's `write_frontier` schema, loadable as a warm start.
//!
//!   DRILL_MODE=finishing (default): dense 1-v-keeper box chances; trains CONVERSION. Empirically
//!     this sits at a near-parity ceiling — the net's shot EXECUTION is ≤ analytic (a paired
//!     McNemar eval shows trained ≈/< a fresh net). Finishing is capped; hence:
//!   DRILL_MODE=creation: a realistic final-third attack (ball 22-40yd out, defensive block +
//!     keeper, support runners). The value is whether the possession MANUFACTURES a high-xG shot
//!     from open play. Execution levers stay at engine default (analytic shoots); the net learns
//!     MOVEMENT + PASSING. Warm-starts from a real full-game league frontier by default (fine-tune,
//!     never scratch-train a narrow distribution — that is what narrowed the finishing net).
//!
//! Run:   soccer_finishing_drill [episodes=300]
//! Env:   DRILL_MODE       finishing (default) | creation
//!        DRILL_WARMSTART  learned-params.json to resume from (creation defaults to a league net)
//!        DRILL_OUT        output path (default /tmp/{finishing,creation}-drill/learned-params.json)
//!        DRILL_SEED       u32 base seed (default 0xF111D000)
//!        DRILL_DEFENDERS  Away defenders between ball and goal (finishing default 0, creation 3)
//!        DRILL_MAX_TICKS  max ticks per episode (finishing 80, creation 130)
//!        DRILL_EVAL       held-out paired-eval episodes (default 200; 0 disables)
//!        DRILL_FREEZE     control arm: serve the net but take no gradient steps
//!        DRILL_LAMBDA / DRILL_WARMUP / DRILL_ACTOR_CRITIC  optional neural-blend overrides
//!
//! IMPORTANT geometry note: in this engine `Team::goal_y()` returns the goal a team
//! ATTACKS (scores in), not the one it defends. The definitive scoring code
//! (world.rs `heading_high => (field_length, Away, Home)`) proves HOME scores in the
//! goal at `y = field_length` == `Team::Home.goal_y(field_length)`. The build spec's
//! `Team::Away.goal_y()` (== 0.0) points at the OPPOSITE end, so we use the Home goal.

use std::fs;
use std::path::Path;
use std::time::Duration;

use soccer_engine::des::general::soccer::{
    enable_deterministic_formation_lp, MatchConfig, PlayerRole, SoccerMatch,
    SoccerNeuralNetworkSnapshot, SoccerQPolicy, SoccerQPolicyOptions, SoccerQTargetEntry,
    SoccerTeamQPolicies, Team, Vec2,
};
use soccer_engine::des::general::tournament::TeamBrain;

/// Small self-contained deterministic PRNG (splitmix64). The crate has no `rand`
/// dependency, so we hand-roll one rather than pull a new dep in for a drill. Seeded
/// from `config.seed + episode` per the spec's "seeded StdRng" intent.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n.max(1) as u64) as usize
    }
}

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}

/// Set an engine env var only if the operator has not already set it, so the drill ships a
/// sane default coupling while staying fully overridable. Must run BEFORE the engine reads
/// these (several are `OnceLock`-cached on first read), i.e. before any `run_time_step`.
fn set_env_default(name: &str, value: &str) {
    if std::env::var_os(name).is_none() {
        std::env::set_var(name, value);
    }
}

/// Per-episode outcome measured for BOTH modes. `goal` drives the finishing metric; `shots`,
/// `sot` and `best_xg` drive the chance-CREATION metrics (does the possession manufacture a
/// high-quality shot from open play).
#[derive(Clone, Copy, Default)]
struct EpMetrics {
    goal: bool,
    shots: u32,
    sot: u32,
    best_xg: f64,
}

/// Transparent GEOMETRIC xG proxy for a shot taken from `origin` at the goal on line `goal_y`
/// (8yd mouth, centered on the pitch width). Monotone in the two things that actually govern a
/// chance's quality — distance and the angle subtended by the goal mouth — mapped through a
/// logistic into [0,1]. Reference points: ~6yd central ≈ 0.49, penalty spot ≈ 0.12, 25yd ≈ 0.02.
/// It is a proxy for grading chance quality, NOT the engine's reward (the engine grades shots
/// itself via its on-target / shot-objective reward — we write no reward code).
fn shot_xg(origin: Vec2, width: f64, goal_y: f64) -> f64 {
    let goal_half = 4.0;
    let post_l = Vec2::new(width * 0.5 - goal_half, goal_y);
    let post_r = Vec2::new(width * 0.5 + goal_half, goal_y);
    let v1 = post_l - origin;
    let v2 = post_r - origin;
    let denom = (v1.len() * v2.len()).max(1e-6);
    let angle = (v1.dot(v2) / denom).clamp(-1.0, 1.0).acos();
    let dist = origin.distance(Vec2::new(width * 0.5, goal_y));
    let z = -3.2 + 3.0 * angle - 0.06 * dist;
    1.0 / (1.0 + (-z).exp())
}

/// xG proxy above which an episode is scored as having produced a genuine "high-quality chance".
const HIGH_XG_THRESHOLD: f64 = 0.12;

/// Mean (shots/ep, sot/ep, meanXG/ep, high-xG-episode fraction) over a set of episodes.
fn summarize(ms: &[EpMetrics]) -> (f64, f64, f64, f64) {
    let n = ms.len().max(1) as f64;
    let shots = ms.iter().map(|m| m.shots as f64).sum::<f64>() / n;
    let sot = ms.iter().map(|m| m.sot as f64).sum::<f64>() / n;
    let xg = ms.iter().map(|m| m.best_xg).sum::<f64>() / n;
    let hi = ms.iter().filter(|m| m.best_xg > HIGH_XG_THRESHOLD).count() as f64 / n;
    (shots, sot, xg, hi)
}

/// Reset the shooting situation by DIRECT field assignment (all touched fields are `pub`).
/// Leaves a clean chance: shooter on the ball 4-18yd out, Away keeper on its line, optional
/// 0-2 defenders goal-side, and every other player relocated deep into Home's half so they
/// cannot interfere within the episode. Returns the shooter's player id.
fn setup_scenario(sim: &mut SoccerMatch, rng: &mut Rng, num_defenders: usize) -> usize {
    let width = sim.config.field_width_yards;
    let length = sim.config.field_length_yards;
    // HOME attacks toward y = length (== Team::Home.goal_y(length)); the Away keeper defends it.
    let goal_y = Team::Home.goal_y(length);
    debug_assert_eq!(goal_y, length);

    let away_keeper = sim
        .players
        .iter()
        .find(|p| p.team == Team::Away && p.role == PlayerRole::Goalkeeper)
        .map(|p| p.id);
    let home_outfielders: Vec<usize> = sim
        .players
        .iter()
        .filter(|p| p.team == Team::Home && p.role != PlayerRole::Goalkeeper)
        .map(|p| p.id)
        .collect();
    let away_outfielders: Vec<usize> = sim
        .players
        .iter()
        .filter(|p| p.team == Team::Away && p.role != PlayerRole::Goalkeeper)
        .map(|p| p.id)
        .collect();

    let shooter = home_outfielders[rng.below(home_outfielders.len())];
    // Shooting spot: central-ish in the box, 4-18 yards from goal.
    let sx = rng.range(width * 0.30, width * 0.70);
    let sy = rng.range(goal_y - 18.0, goal_y - 4.0);
    let shooter_pos = Vec2::new(sx, sy);

    let defenders: Vec<usize> = away_outfielders.iter().copied().take(num_defenders).collect();

    for p in sim.players.iter_mut() {
        if p.id == shooter {
            p.position = shooter_pos;
            p.velocity = Vec2::zero();
        } else if Some(p.id) == away_keeper {
            // Keeper central on its line (analytic AI will step out to narrow the angle).
            p.position = Vec2::new(width * 0.5, goal_y - 1.5);
            p.velocity = Vec2::zero();
        } else if let Some(idx) = defenders.iter().position(|&d| d == p.id) {
            // Stagger the requested defenders between the shooter and the goal.
            let t = 0.45 + 0.22 * idx as f64;
            let dx = (width * 0.5 + (sx - width * 0.5) * 0.4 + rng.range(-3.0, 3.0))
                .clamp(2.0, width - 2.0);
            let dy = (sy + (goal_y - sy) * t).clamp(0.0, goal_y);
            p.position = Vec2::new(dx, dy);
            p.velocity = Vec2::zero();
        } else {
            // Everyone else parked deep in HOME's half (small y), far from the action.
            p.position = Vec2::new(rng.range(4.0, width - 4.0), rng.range(6.0, 34.0));
            p.velocity = Vec2::zero();
        }
    }

    // Ball at the shooter's feet, possession to HOME.
    sim.ball.position = shooter_pos;
    sim.ball.velocity = Vec2::zero();
    sim.ball.altitude_yards = 0.0;
    sim.ball.holder = Some(shooter);
    sim.ball.last_touch_team = Some(Team::Home);

    // NONZERO, monotonically non-decreasing tick. tick != 0 skips the opening-kickoff
    // restage (world.rs stage_opening_kickoff_if_pending only acts at tick==0); keeping it
    // monotonic (never rewound) and small keeps us clear of any half/period boundary and of
    // is_done() for the whole run.
    sim.tick = sim.tick.max(1000);

    shooter
}

/// CREATION scenario: reset a realistic open-play attack in the final third — a Home CARRIER on
/// the ball 22-40yd from the Away goal (NOT in the box), up to 3 advanced Home support attackers
/// for passing/movement options, and `num_defenders` analytic Away defenders forming a block
/// between the ball and goal plus the keeper on its line. Everyone else is parked deep. The net
/// must PENETRATE and manufacture a high-xG shot, not just tap in. Returns the carrier's id.
fn setup_scenario_creation(sim: &mut SoccerMatch, rng: &mut Rng, num_defenders: usize) -> usize {
    let width = sim.config.field_width_yards;
    let length = sim.config.field_length_yards;
    let goal_y = Team::Home.goal_y(length); // == length; HOME attacks +y

    let away_keeper = sim
        .players
        .iter()
        .find(|p| p.team == Team::Away && p.role == PlayerRole::Goalkeeper)
        .map(|p| p.id);
    let home_outfielders: Vec<usize> = sim
        .players
        .iter()
        .filter(|p| p.team == Team::Home && p.role != PlayerRole::Goalkeeper)
        .map(|p| p.id)
        .collect();
    let away_outfielders: Vec<usize> = sim
        .players
        .iter()
        .filter(|p| p.team == Team::Away && p.role != PlayerRole::Goalkeeper)
        .map(|p| p.id)
        .collect();

    let cx = rng.range(width * 0.32, width * 0.68);
    let cy = rng.range(goal_y - 40.0, goal_y - 22.0);
    let carrier = home_outfielders[rng.below(home_outfielders.len())];
    let carrier_pos = Vec2::new(cx, cy);

    let supports: Vec<usize> = home_outfielders
        .iter()
        .copied()
        .filter(|&id| id != carrier)
        .take(3)
        .collect();
    let defenders: Vec<usize> = away_outfielders.iter().copied().take(num_defenders).collect();

    for p in sim.players.iter_mut() {
        if p.id == carrier {
            p.position = carrier_pos;
            p.velocity = Vec2::zero();
        } else if Some(p.id) == away_keeper {
            p.position = Vec2::new(width * 0.5, goal_y - 2.0);
            p.velocity = Vec2::zero();
        } else if let Some(k) = supports.iter().position(|&s| s == p.id) {
            // Advanced support spread across the width, level-to-ahead of the carrier.
            let sx = (width * (0.22 + 0.28 * k as f64) + rng.range(-4.0, 4.0)).clamp(3.0, width - 3.0);
            let sy = (cy + rng.range(2.0, 12.0)).clamp(0.0, goal_y - 8.0);
            p.position = Vec2::new(sx, sy);
            p.velocity = Vec2::zero();
        } else if let Some(k) = defenders.iter().position(|&d| d == p.id) {
            // Defensive block staggered between the ball and the goal.
            let frac = (k as f64 + 1.0) / (num_defenders as f64 + 1.0);
            let dx = (width * (0.30 + 0.40 * frac) + rng.range(-5.0, 5.0)).clamp(3.0, width - 3.0);
            let dy = (cy + (goal_y - cy) * (0.35 + 0.45 * frac) + rng.range(-3.0, 3.0))
                .clamp(cy + 3.0, goal_y - 4.0);
            p.position = Vec2::new(dx, dy);
            p.velocity = Vec2::zero();
        } else {
            p.position = Vec2::new(rng.range(4.0, width - 4.0), rng.range(6.0, 40.0));
            p.velocity = Vec2::zero();
        }
    }

    sim.ball.position = carrier_pos;
    sim.ball.velocity = Vec2::zero();
    sim.ball.altitude_yards = 0.0;
    sim.ball.holder = Some(carrier);
    sim.ball.last_touch_team = Some(Team::Home);
    sim.tick = sim.tick.max(1000);
    carrier
}

/// Play one episode (either mode) and return its measured outcome. FINISHING resets a 1-v-keeper
/// chance and ends when the shot resolves; CREATION resets a final-third attack and runs the
/// possession until a turnover / the ball is cleared / the tick cap, accumulating every Home shot
/// and the best geometric xG produced. Post-goal celebration is flushed so it can't clobber the
/// next reset. Does NOT drain the learner — the caller decides, so the SAME routine serves both
/// training and frozen held-out eval.
fn play_episode(
    sim: &mut SoccerMatch,
    rng: &mut Rng,
    creation: bool,
    num_defenders: usize,
    max_ticks: usize,
    width: f64,
    length: f64,
) -> EpMetrics {
    if creation {
        setup_scenario_creation(sim, rng, num_defenders);
    } else {
        setup_scenario(sim, rng, num_defenders);
    }
    let goal_y = length;
    let shots0 = sim.stats.shots_home;
    let sot0 = sim.stats.shots_on_target_home;
    let score_before = sim.score_home;
    let mut last_shots = shots0;
    let mut best_xg = 0.0f64;

    for _ in 0..max_ticks {
        let pre_ball = sim.ball.position; // shot-origin candidate for this tick
        sim.run_time_step();
        if sim.stats.shots_home > last_shots {
            last_shots = sim.stats.shots_home;
            let xg = shot_xg(pre_ball, width, goal_y);
            if xg > best_xg {
                best_xg = xg;
            }
        }
        if sim.score_home > score_before {
            break; // goal (both modes)
        }
        if creation {
            // Turnover: an Away player is now in possession.
            if let Some(h) = sim.ball.holder {
                if sim.players.iter().find(|p| p.id == h).map(|p| p.team) == Some(Team::Away) {
                    break;
                }
            }
            // Attack broken up / cleared out of the attacking half.
            if sim.ball.position.y < length * 0.5 {
                break;
            }
        } else {
            // Shot resolved against us (keeper save / defender block / interception) …
            if sim.ball.last_touch_team == Some(Team::Away) {
                break;
            }
            // … or the ball left the attacking third.
            if sim.ball.position.y < length * 0.60 {
                break;
            }
        }
        if sim.is_done() {
            break;
        }
    }

    let goal = sim.score_home > score_before;
    let metrics = EpMetrics {
        goal,
        shots: sim.stats.shots_home - shots0,
        sot: sim.stats.shots_on_target_home - sot0,
        best_xg,
    };
    if goal {
        // Flush the ~25-tick goal celebration + kickoff reset (the counter is pub(crate)).
        for _ in 0..30 {
            if sim.is_done() {
                break;
            }
            sim.run_time_step();
        }
    }
    metrics
}

/// Run `n` held-out episodes on `sim` (net already installed & frozen) and collect their metrics.
fn eval_arm(
    sim: &mut SoccerMatch,
    eval_base: u64,
    n: usize,
    creation: bool,
    num_defenders: usize,
    max_ticks: usize,
    width: f64,
    length: f64,
) -> Vec<EpMetrics> {
    (0..n)
        .map(|i| {
            let mut rng = Rng::new(eval_base.wrapping_add(i as u64));
            play_episode(sim, &mut rng, creation, num_defenders, max_ticks, width, length)
        })
        .collect()
}

fn main() {
    // Mode: `finishing` (default) trains 1-v-keeper conversion; `creation` trains manufacturing
    // high-xG chances from open play in the final third. They differ in scenario, metrics, serving
    // levers, and the eval baseline — everything else (spawn/step/train/eval infra) is shared.
    let creation = std::env::var("DRILL_MODE")
        .map(|m| m.trim().eq_ignore_ascii_case("creation"))
        .unwrap_or(false);

    if !creation {
        // FINISHING COUPLING (finishing mode only): route shot EXECUTION through the learner so
        // value learning can move conversion. DD_SOCCER_ENABLE_DISCRETIZED_KICK expands a shot into
        // power/placement bucket candidates and DD_SOCCER_NEURAL_AUTHORITATIVE_LAMBDA makes the
        // trained value head own that choice. (Empirically finishing sits at a near-parity ceiling
        // regardless — the net's shot EXECUTION is ≤ analytic; this is why CREATION mode exists.)
        // In CREATION mode we DELIBERATELY leave these at engine default: analytic executes the
        // shot; the net learns MOVEMENT + PASSING (where its value lives) via the value blend.
        // Both env vars are `OnceLock`-cached in the engine, so they must be set before step 1.
        set_env_default("DD_SOCCER_ENABLE_DISCRETIZED_KICK", "1");
        set_env_default("SOCCER_NEURAL_MCTS_MIN_DISCRETIZED_KICK_CANDIDATES", "3");
        set_env_default("DD_SOCCER_NEURAL_AUTHORITATIVE_LAMBDA", "30");
    }

    // CREATION default: fine-tune from a REAL full-game league frontier rather than training from
    // scratch on a narrow distribution (scratch-training on the box scenario is exactly what caused
    // the finishing net to narrow to trained<fresh). Warm-start is opt-out via DRILL_WARMSTART.
    if creation && std::env::var_os("DRILL_WARMSTART").is_none() {
        const DEFAULT_WS: &str = "/tmp/conv-climb-11v11/learned-params.json";
        if Path::new(DEFAULT_WS).exists() {
            std::env::set_var("DRILL_WARMSTART", DEFAULT_WS);
        } else {
            eprintln!("creation mode: no DRILL_WARMSTART and {DEFAULT_WS} missing — training from scratch (not recommended).");
        }
    }

    // Deterministic formation LP so a given seed is reproducible.
    enable_deterministic_formation_lp();

    let args: Vec<String> = std::env::args().collect();
    let episodes: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(300);
    let seed: u32 = std::env::var("DRILL_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0xF111_D000);
    // Creation faces a defensive block (2-4 defenders); finishing is a clean 1-v-keeper by default.
    let num_defenders: usize = std::env::var("DRILL_DEFENDERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(if creation { 3 } else { 0 });
    // Creation needs time to work the ball into a chance; finishing resolves fast.
    let max_ticks: usize = std::env::var("DRILL_MAX_TICKS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(if creation { 130 } else { 80 });
    let out_path = std::env::var("DRILL_OUT").unwrap_or_else(|_| {
        if creation {
            "/tmp/creation-drill/learned-params.json".to_string()
        } else {
            "/tmp/finishing-drill/learned-params.json".to_string()
        }
    });
    let warmstart = std::env::var("DRILL_WARMSTART").ok().filter(|s| !s.is_empty());

    // Optional warm start: pull the learner net + home target-Q out of an existing frontier.
    let (warm_neural, warm_home_targets): (Option<SoccerNeuralNetworkSnapshot>, Vec<SoccerQTargetEntry>) =
        if let Some(path) = warmstart.as_ref() {
            let raw = fs::read_to_string(path).expect("read DRILL_WARMSTART file");
            let v: serde_json::Value =
                serde_json::from_str(&raw).expect("parse DRILL_WARMSTART json");
            let neural = serde_json::from_value::<SoccerNeuralNetworkSnapshot>(
                v["neuralNetwork"].clone(),
            )
            .expect("DRILL_WARMSTART.neuralNetwork");
            let home_targets =
                serde_json::from_value::<Vec<SoccerQTargetEntry>>(v["homeTargetEntries"].clone())
                    .unwrap_or_default();
            eprintln!(
                "warmstart: loaded net (training_steps={}) + {} home target entries from {path}",
                neural.training_steps,
                home_targets.len()
            );
            (Some(neural), home_targets)
        } else {
            (None, Vec::new())
        };

    // Config: learning fully enabled (the learner is only built when learning + neural are on).
    let mut config = MatchConfig {
        seed,
        learning_enabled: true,
        full_game_learning_enabled: true,
        duration_seconds: 1e9,
        ..MatchConfig::default()
    };
    config.neural_learning.enabled = true;
    // Optional neural-blend overrides (default = engine defaults: Additive value blend,
    // lambda 0.5, warmup 200 steps — enough for the warm critic to influence shot selection).
    if let Some(x) = std::env::var("DRILL_LAMBDA").ok().and_then(|v| v.parse().ok()) {
        config.neural_blend.lambda = x;
    }
    if let Some(x) = std::env::var("DRILL_WARMUP").ok().and_then(|v| v.parse().ok()) {
        config.neural_blend.warmup_steps = x;
    }
    if env_flag("DRILL_ACTOR_CRITIC") {
        config.neural_blend.actor_critic = true;
    }

    // Team policies exactly like tournament.rs: home carries its target-Q; away is empty.
    let options = SoccerQPolicyOptions::default();
    let policies = SoccerTeamQPolicies {
        home: SoccerQPolicy::from_entries_with_targets(options.clone(), &[], &warm_home_targets)
            .expect("build home policy"),
        away: SoccerQPolicy::from_entries_with_targets(options, &[], &[])
            .expect("build away policy"),
    };

    let mut sim = SoccerMatch::default_11v11(config).with_team_policies(policies);
    // HOME = the LEARNER (frozen=false, trains). If warm_neural is None a fresh learner is
    // created (learning_enabled is set). AWAY = pure analytic (frozen + no net disables its brain).
    sim.set_team_neural_brain(Team::Home, warm_neural, false)
        .expect("install home learner");
    sim.set_team_neural_brain(Team::Away, None, true)
        .expect("install away analytic");
    assert!(
        sim.neural_network_snapshot_for(Team::Home).is_some(),
        "home learner must exist after install"
    );

    // Control arm (A/B baseline): DRILL_FREEZE re-installs the Home net FROZEN — it still SERVES
    // (identical bucket-selection machinery) but takes no gradient steps. Run with the same seed
    // as a learning run and its per-block rate stays flat, isolating any learning-run uptrend as
    // the net actually learning to finish (rather than scenario-sampling noise).
    let frozen = env_flag("DRILL_FREEZE");
    if frozen {
        let snap = sim.neural_network_snapshot_for(Team::Home);
        sim.set_team_neural_brain(Team::Home, snap, true)
            .expect("freeze home net");
    }

    // Snapshot the net BEFORE training — the paired-eval baseline (a fresh net for finishing-from-
    // scratch, or the warm-start league net for creation). Eval then isolates what TRAINING added.
    let baseline_snap = sim.neural_network_snapshot_for(Team::Home);

    let field_length = sim.config.field_length_yards;
    let field_width = sim.config.field_width_yards;
    let base_seed = seed as u64;
    let mode_label = if creation { "creation" } else { "finishing" };

    println!(
        "soccer-drill mode={mode_label} episodes={episodes} seed={seed:#x} defenders={num_defenders} \
         max_ticks={max_ticks} warmstart={} frozen={frozen}",
        warmstart.as_deref().unwrap_or("none"),
    );
    if creation {
        println!(
            "  serving: engine-default blend (analytic executes the shot; net learns MOVEMENT+PASSING).  high_xG>{HIGH_XG_THRESHOLD}"
        );
        println!("  eps       shots/ep   sot/ep    meanXG   hiXG-frac    train_steps   critic_loss");
    } else {
        println!(
            "  coupling: authoritative_lambda={} discretized_kick={} min_kick_candidates={}",
            std::env::var("DD_SOCCER_NEURAL_AUTHORITATIVE_LAMBDA").unwrap_or_default(),
            std::env::var("DD_SOCCER_ENABLE_DISCRETIZED_KICK").unwrap_or_default(),
            std::env::var("SOCCER_NEURAL_MCTS_MIN_DISCRETIZED_KICK_CANDIDATES").unwrap_or_default(),
        );
        println!("  eps      scoring_rate  (goals/50)     cumulative    train_steps    critic_loss");
    }

    const BLOCK: usize = 50;
    let mut b_goals = 0usize;
    let mut b_shots = 0u32;
    let mut b_sot = 0u32;
    let mut b_xg = 0.0f64;
    let mut b_hi = 0usize;
    let mut total_goals = 0usize;
    // Primary metric per block for the trend verdict: finishing → goal rate; creation → hiXG frac.
    let mut primary_blocks: Vec<f64> = Vec::new();

    for ep in 0..episodes {
        let mut rng = Rng::new(base_seed.wrapping_add(ep as u64));
        let m = play_episode(
            &mut sim,
            &mut rng,
            creation,
            num_defenders,
            max_ticks,
            field_width,
            field_length,
        );
        if m.goal {
            b_goals += 1;
            total_goals += 1;
        }
        b_shots += m.shots;
        b_sot += m.sot;
        b_xg += m.best_xg;
        if m.best_xg > HIGH_XG_THRESHOLD {
            b_hi += 1;
        }

        // Train: flush the learner at every episode boundary.
        sim.drain_neural_learning(Duration::from_millis(200));

        if (ep + 1) % BLOCK == 0 {
            let (steps, loss) = sim
                .neural_network_snapshot_for(Team::Home)
                .map(|s| (s.training_steps, s.average_loss))
                .unwrap_or((0, None));
            let loss_s = loss.map(|l| format!("{l:.4}")).unwrap_or_else(|| "n/a".into());
            if creation {
                let shots_pe = b_shots as f64 / BLOCK as f64;
                let sot_pe = b_sot as f64 / BLOCK as f64;
                let mean_xg = b_xg / BLOCK as f64;
                let hi_frac = b_hi as f64 / BLOCK as f64;
                primary_blocks.push(hi_frac);
                println!(
                    "{:>4}-{:<4}  {:>8.3}   {:>6.3}   {:>6.3}    {:>7.3}     steps={:>6}   loss={loss_s}",
                    ep + 2 - BLOCK,
                    ep + 1,
                    shots_pe,
                    sot_pe,
                    mean_xg,
                    hi_frac,
                    steps,
                );
            } else {
                let rate = b_goals as f64 / BLOCK as f64;
                primary_blocks.push(rate);
                let cum = total_goals as f64 / (ep + 1) as f64;
                println!(
                    "{:>4}-{:<4}  {:>9.3}   ({:>3}/{:<3})   cum={:>5.3}   steps={:>6}   loss={loss_s}",
                    ep + 2 - BLOCK,
                    ep + 1,
                    rate,
                    b_goals,
                    BLOCK,
                    cum,
                    steps,
                );
            }
            b_goals = 0;
            b_shots = 0;
            b_sot = 0;
            b_xg = 0.0;
            b_hi = 0;
        }
    }

    // Honest verdict: early→late delta of the primary metric against a binomial noise band, so a
    // sampling wobble is not mistaken for learning. The robust learning evidence is the train_steps
    // column (the critic trains every block); the metric is a high-variance downstream proxy.
    if primary_blocks.len() >= 2 {
        let half = primary_blocks.len() / 2;
        let mean = |xs: &[f64]| xs.iter().sum::<f64>() / xs.len().max(1) as f64;
        let early = mean(&primary_blocks[..half.max(1)]);
        let late = mean(&primary_blocks[half..]);
        let delta = late - early;
        let half_eps = (half.max(1) * BLOCK) as f64;
        let p = ((early + late) / 2.0).clamp(1e-3, 1.0 - 1e-3);
        let se_delta = (p * (1.0 - p) / half_eps).sqrt() * std::f64::consts::SQRT_2;
        let (label, metric) = if creation {
            ("CREATION", "hiXG-frac")
        } else {
            ("CONVERSION", "goal-rate")
        };
        let verdict = if delta > 2.0 * se_delta {
            "=> ROSE beyond noise (net learned)"
        } else if delta > se_delta {
            "=> mild upward drift (~1-2σ; suggestive, not conclusive)"
        } else if delta < -2.0 * se_delta {
            "=> DROPPED beyond noise (regression / narrowing)"
        } else {
            "=> flat within noise"
        };
        println!(
            "\n{label}  {metric} early_half={early:.3}  late_half={late:.3}  delta={delta:+.3}  (1σ≈{se_delta:.3})  {verdict}"
        );
        println!(
            "LEARNER  critic trained every block (train_steps climbed above); the saved net is a valid warm-start."
        );
    }

    // Save: write the trained learner as a valid league frontier (write_frontier schema).
    sim.drain_neural_learning(Duration::from_millis(500));
    let snap = sim
        .neural_network_snapshot_for(Team::Home)
        .expect("home learner network snapshot");
    // Bound the target-Q we carry. Each entry embeds an ~8.5KB binned state key, and full-game
    // learning accumulates tens of thousands of them — so an unpruned dump is ~100MB. The neural
    // net is the real transferable learner (canonical league frontiers carry 0 target entries);
    // keep only the most-visited targets so the frontier stays a few MB. DRILL_MAX_TARGETS=0 → none.
    let max_targets: usize = std::env::var("DRILL_MAX_TARGETS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);
    let home_targets_out = if max_targets == 0 {
        Vec::new()
    } else {
        if let Some(policies) = sim.team_policies_mut() {
            policies.home.prune(0, max_targets, 0);
        }
        sim.team_policies()
            .map(|p| p.home.target_entries())
            .unwrap_or_default()
    };
    let brain = TeamBrain::from_snapshot_with_targets(snap.clone(), home_targets_out, Vec::new());

    // ---- Eval at power: PAIRED held-out comparison, TUNED-frozen vs BASELINE-frozen ----
    // The within-run per-block metric is high-variance and non-paired. This is the definitive test:
    // run a held-out scenario set (distinct seeds) twice — TUNED (the just-trained net, frozen) and
    // BASELINE (the pre-training net, frozen: fresh for finishing-from-scratch, the warm-start
    // league net for creation) — under identical machinery + identical scenarios, so the only
    // difference is what training added. McNemar on the discordant pairs (exactly one arm succeeds).
    let eval_n: usize = std::env::var("DRILL_EVAL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    if eval_n > 0 && !frozen {
        let eval_base = base_seed ^ 0xE7A1_0000_0000_0000;
        // (a) TUNED arm: freeze the just-trained net in the existing sim (save data already captured).
        sim.set_team_neural_brain(Team::Home, Some(snap.clone()), true)
            .expect("freeze tuned net for eval");
        let tuned = eval_arm(
            &mut sim, eval_base, eval_n, creation, num_defenders, max_ticks, field_width, field_length,
        );
        // (b) BASELINE arm: a fresh sim serving the PRE-TRAINING net frozen, identical machinery.
        let mut base_config = MatchConfig {
            seed,
            learning_enabled: true,
            full_game_learning_enabled: true,
            duration_seconds: 1e9,
            ..MatchConfig::default()
        };
        base_config.neural_learning.enabled = true;
        let base_policies = SoccerTeamQPolicies {
            home: SoccerQPolicy::from_entries_with_targets(SoccerQPolicyOptions::default(), &[], &[])
                .expect("base home policy"),
            away: SoccerQPolicy::from_entries_with_targets(SoccerQPolicyOptions::default(), &[], &[])
                .expect("base away policy"),
        };
        let mut base_sim = SoccerMatch::default_11v11(base_config).with_team_policies(base_policies);
        match baseline_snap.clone() {
            Some(b) => {
                base_sim
                    .set_team_neural_brain(Team::Home, Some(b), true)
                    .expect("freeze baseline net");
            }
            None => {
                base_sim
                    .set_team_neural_brain(Team::Home, None, false)
                    .expect("fresh baseline learner");
                let fs = base_sim.neural_network_snapshot_for(Team::Home);
                base_sim
                    .set_team_neural_brain(Team::Home, fs, true)
                    .expect("freeze fresh baseline");
            }
        }
        base_sim
            .set_team_neural_brain(Team::Away, None, true)
            .expect("base away analytic");
        let baseline = eval_arm(
            &mut base_sim, eval_base, eval_n, creation, num_defenders, max_ticks, field_width,
            field_length,
        );

        println!("\nEVAL@POWER  held-out={eval_n} (paired, both frozen, identical scenarios)");
        if creation {
            let (ts, tso, txg, thi) = summarize(&tuned);
            let (bs, bso, bxg, bhi) = summarize(&baseline);
            let hi = |m: &EpMetrics| m.best_xg > HIGH_XG_THRESHOLD;
            let up = tuned.iter().zip(&baseline).filter(|(t, b)| hi(t) && !hi(b)).count();
            let down = tuned.iter().zip(&baseline).filter(|(t, b)| !hi(t) && hi(b)).count();
            let z = if up + down > 0 {
                (up as f64 - down as f64) / ((up + down) as f64).sqrt()
            } else {
                0.0
            };
            println!("  creation-tuned:  shots/ep={ts:.3}  sot/ep={tso:.3}  meanXG={txg:.3}  hiXG-frac={thi:.3}");
            println!("  warm-baseline:   shots/ep={bs:.3}  sot/ep={bso:.3}  meanXG={bxg:.3}  hiXG-frac={bhi:.3}");
            println!(
                "  hiXG paired flips: tuned-only={up}  baseline-only={down}  McNemar z={z:+.2}   {}",
                if z > 1.96 {
                    "=> tuned SIGNIFICANTLY creates more high-xG chances (creation headroom!)"
                } else if z < -1.96 {
                    "=> tuned significantly WORSE (narrowed the full-game net)"
                } else if thi > bhi {
                    "=> tuned edges baseline (within noise)"
                } else {
                    "=> no chance-creation gain over the warm-start baseline"
                }
            );
        } else {
            let t_goals = tuned.iter().filter(|m| m.goal).count();
            let b_goals2 = baseline.iter().filter(|m| m.goal).count();
            let up = tuned.iter().zip(&baseline).filter(|(t, b)| t.goal && !b.goal).count();
            let down = tuned.iter().zip(&baseline).filter(|(t, b)| !t.goal && b.goal).count();
            let z = if up + down > 0 {
                (up as f64 - down as f64) / ((up + down) as f64).sqrt()
            } else {
                0.0
            };
            println!(
                "  trained_net conversion = {:.3} ({}/{})   baseline_net conversion = {:.3} ({}/{})",
                t_goals as f64 / eval_n as f64,
                t_goals,
                eval_n,
                b_goals2 as f64 / eval_n as f64,
                b_goals2,
                eval_n
            );
            println!(
                "  paired flips: trained-only-goals={up}  baseline-only-goals={down}  McNemar z={z:+.2}   {}",
                if z > 1.96 {
                    "=> trained SIGNIFICANTLY beats baseline (net learned to finish)"
                } else if z < -1.96 {
                    "=> trained significantly WORSE than baseline (narrowing)"
                } else if t_goals > b_goals2 {
                    "=> trained edges baseline (within noise)"
                } else {
                    "=> no finishing gain over baseline (analytic near-parity ceiling)"
                }
            );
        }
    }

    if let Some(parent) = Path::new(&out_path)
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).expect("create output dir");
    }
    let value = serde_json::json!({
        "version": 0,
        "homeEntries": [],
        "homeTargetEntries": brain.home_target_entries,
        "awayEntries": [],
        "awayTargetEntries": brain.away_target_entries,
        "episodes": 0,
        "neuralNetwork": serde_json::to_value(&snap).unwrap(),
    });
    fs::write(&out_path, serde_json::to_string(&value).expect("serialize frontier"))
        .expect("write frontier");
    println!(
        "\nsaved learner frontier -> {out_path}  (net training_steps={}, home_target_entries={})",
        snap.training_steps,
        brain.home_target_entries.len()
    );
}
