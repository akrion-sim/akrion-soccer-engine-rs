//! Finishing training ground.
//!
//! Spawns the HOME neural learner into short, dense 1‑v‑keeper shooting scenarios
//! and trains it to CONVERT. In a full match, shots/goals are <1% of actions, so the
//! finishing gradient is starved; here every episode is a shooting chance, densifying
//! that signal. The per‑shot reward stack (goal + on‑target + shot‑objective grades)
//! fires automatically inside `run_time_step` and feeds the learner — this harness
//! writes NO reward code. Output is a `learned-params.json` in the exact schema the
//! league trainer's `write_frontier` uses, so it loads directly as a warm start.
//!
//! Run:   soccer_finishing_drill [episodes=300]
//! Env:   DRILL_WARMSTART  path to a learned-params.json to resume from (optional)
//!        DRILL_OUT        output path (default /tmp/finishing-drill/learned-params.json)
//!        DRILL_SEED       u32 base seed (default 0xF111D000)
//!        DRILL_DEFENDERS  extra Away outfield defenders between shooter and goal (0-2, default 0)
//!        DRILL_MAX_TICKS  max ticks per episode (default 80 = 8s at dt=0.1)
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

fn main() {
    // Deterministic formation LP so a given seed is reproducible.
    enable_deterministic_formation_lp();

    let args: Vec<String> = std::env::args().collect();
    let episodes: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(300);
    let seed: u32 = std::env::var("DRILL_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0xF111_D000);
    let num_defenders: usize = std::env::var("DRILL_DEFENDERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
        .min(2);
    let max_ticks: usize = std::env::var("DRILL_MAX_TICKS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(80);
    let out_path = std::env::var("DRILL_OUT")
        .unwrap_or_else(|_| "/tmp/finishing-drill/learned-params.json".to_string());
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

    let field_length = sim.config.field_length_yards;
    let base_seed = seed as u64;

    println!(
        "finishing-drill: episodes={episodes} seed={seed:#x} defenders={num_defenders} \
         max_ticks={max_ticks} warmstart={} lambda={} actor_critic={}",
        warmstart.as_deref().unwrap_or("none"),
        sim.config.neural_blend.lambda,
        sim.config.neural_blend.actor_critic
    );
    println!("block(50)  scoring_rate  (goals/episodes)   cumulative");

    const BLOCK: usize = 50;
    let mut block_goals = 0usize;
    let mut total_goals = 0usize;
    let mut block_rates: Vec<f64> = Vec::new();

    for ep in 0..episodes {
        let mut rng = Rng::new(base_seed.wrapping_add(ep as u64));
        setup_scenario(&mut sim, &mut rng, num_defenders);

        let score_before = sim.score_home;
        let mut scored = false;
        for _ in 0..max_ticks {
            sim.run_time_step();
            if sim.score_home > score_before {
                scored = true;
                break;
            }
            // Shot resolved against us (keeper save / defender block / interception) …
            if sim.ball.last_touch_team == Some(Team::Away) {
                break;
            }
            // … or the ball left the attacking third.
            if sim.ball.position.y < field_length * 0.60 {
                break;
            }
            if sim.is_done() {
                break;
            }
        }

        if scored {
            // Flush the ~25-tick goal celebration + kickoff reset so it can't clobber the
            // next scenario (the celebration counter is pub(crate), unreachable from a bin).
            for _ in 0..30 {
                if sim.is_done() {
                    break;
                }
                sim.run_time_step();
            }
            block_goals += 1;
            total_goals += 1;
        }

        // Train: flush the learner at every episode boundary.
        sim.drain_neural_learning(Duration::from_millis(200));

        if (ep + 1) % BLOCK == 0 {
            let rate = block_goals as f64 / BLOCK as f64;
            block_rates.push(rate);
            let cum = total_goals as f64 / (ep + 1) as f64;
            println!(
                "{:>4}-{:<4}  {:>10.3}   ({:>3}/{:<3})        {:>6.3}",
                ep + 2 - BLOCK,
                ep + 1,
                rate,
                block_goals,
                BLOCK,
                cum
            );
            block_goals = 0;
        }
    }

    // Trend verdict.
    if block_rates.len() >= 2 {
        let first = block_rates.first().copied().unwrap_or(0.0);
        let last = block_rates.last().copied().unwrap_or(0.0);
        let half = block_rates.len() / 2;
        let early: f64 = block_rates[..half.max(1)].iter().sum::<f64>() / half.max(1) as f64;
        let late: f64 =
            block_rates[half..].iter().sum::<f64>() / (block_rates.len() - half).max(1) as f64;
        println!(
            "\nTREND  first_block={first:.3}  last_block={last:.3}  early_half={early:.3}  \
             late_half={late:.3}  delta(late-early)={:+.3}  {}",
            late - early,
            if late > early + 1e-9 {
                "=> net is LEARNING to finish (rate up)"
            } else {
                "=> no upward trend (investigate levers)"
            }
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
