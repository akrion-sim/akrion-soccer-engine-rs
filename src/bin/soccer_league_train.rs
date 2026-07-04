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
//! the league for growing diversity, and writes the frontier back into the compact
//! learned-params.json the rest of the local stack (:5055, champion-gate, eval) already reads.
//! NO genetic algorithm, NO RDS.
//!
//! Env: SOCCER_LEAGUE_FRONTIER (path, default /tmp/neural-climb-local/learned-params.json)
//!      SOCCER_LEAGUE_ARCHIVE  (dir,  default /tmp/neural-climb-local/champions)
//!      SOCCER_LEAGUE_GAMES_PER_OPP (usize, default 2)
//!      SOCCER_LEAGUE_MINUTES  (f64, default 2.0)
//!      SOCCER_LEAGUE_WEIGHT_DECAY (f64 per round, default 0.01)
//!      SOCCER_LEAGUE_FRESH_OPPONENTS (usize, default 2)
//!      SOCCER_LEAGUE_CHECKPOINT_EVERY_ROUNDS (usize, default 10 — add frontier to league)

use std::fs;
use std::path::Path;
use std::time::Instant;

use soccer_engine::des::general::soccer::SoccerNeuralNetworkSnapshot;
use soccer_engine::des::general::tournament::{
    EngineMatchRunner, EngineMatchRunnerConfig, TeamBrain, TournamentMatchContext,
    TournamentMatchRunner, TournamentStage,
};

fn env_str(k: &str, d: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| d.to_string())
}
fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
fn env_f64(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
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

/// Load the frontier neural snapshot from a learned-params.json (its `neuralNetwork` key).
fn load_snapshot(path: &str) -> Option<SoccerNeuralNetworkSnapshot> {
    let raw = fs::read_to_string(path).ok()?;
    let v: serde_json::Value = serde_json::from_str(&raw).ok()?;
    serde_json::from_value(v.get("neuralNetwork")?.clone()).ok()
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
                *w = snaps.iter().map(|s| s.layers[li].weights[ri][wi]).sum::<f64>() / n;
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

/// Write the frontier snapshot into a COMPACT learned-params.json (drops the large tabular
/// table — we are neural-first and every consumer only reads `neuralNetwork`). Atomic + .prev.
fn write_frontier(path: &str, snap: &SoccerNeuralNetworkSnapshot) -> std::io::Result<()> {
    let v = serde_json::json!({
        "version": 0,
        "homeEntries": [],
        "homeTargetEntries": [],
        "awayEntries": [],
        "awayTargetEntries": [],
        "episodes": 0,
        "neuralNetwork": serde_json::to_value(snap).unwrap(),
    });
    if Path::new(path).exists() {
        let prev = path.trim_end_matches(".json").to_string() + ".prev.json";
        let _ = fs::copy(path, prev);
    }
    let tmp = format!("{path}.tmp");
    fs::write(&tmp, serde_json::to_string(&v)?)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

/// Load the frozen champion league from the archive dir (each is a learned-params.json).
fn load_league(dir: &str) -> Vec<SoccerNeuralNetworkSnapshot> {
    let mut out = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        let mut paths: Vec<_> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().map(|x| x == "json").unwrap_or(false))
            .collect();
        paths.sort();
        for p in paths {
            if let Some(s) = load_snapshot(&p.display().to_string()) {
                out.push(s);
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
) -> (TeamBrain, i32) {
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
    let (home, away) = if frontier_home {
        (&frontier, opponent)
    } else {
        (opponent, &frontier)
    };
    match runner.play(&ctx, home, away) {
        Ok(o) => {
            let gd = if frontier_home {
                o.home_goals as i32 - o.away_goals as i32
            } else {
                o.away_goals as i32 - o.home_goals as i32
            };
            let trained = if frontier_home { o.home_brain } else { o.away_brain };
            (trained, gd)
        }
        Err(e) => {
            eprintln!("league_match_error: {e}");
            (frontier, 0)
        }
    }
}

fn main() {
    let frontier_path = env_str("SOCCER_LEAGUE_FRONTIER", "/tmp/neural-climb-local/learned-params.json");
    let archive_dir = env_str("SOCCER_LEAGUE_ARCHIVE", "/tmp/neural-climb-local/champions");
    let games_per_opp = env_usize("SOCCER_LEAGUE_GAMES_PER_OPP", 2).max(1);
    let minutes = env_f64("SOCCER_LEAGUE_MINUTES", 2.0);
    let weight_decay = env_f64("SOCCER_LEAGUE_WEIGHT_DECAY", 0.01);
    let fresh_opponents = env_usize("SOCCER_LEAGUE_FRESH_OPPONENTS", 2);
    let checkpoint_every = env_usize("SOCCER_LEAGUE_CHECKPOINT_EVERY_ROUNDS", 10).max(1);

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
    runner_config.base.neural_learning.batch_size = env_usize("SOCCER_LEAGUE_BATCH_SIZE", 64).max(1);
    // Network capacity: bigger hidden layer for more expressive value function over the
    // 610-dim field vector (the 24-unit net plateaued at parity with the analytic engine).
    // Only takes effect on a FRESH net — a resumed snapshot keeps its own architecture.
    runner_config.base.neural_learning.hidden_units =
        env_usize("SOCCER_LEAGUE_HIDDEN_UNITS", runner_config.base.neural_learning.hidden_units)
            .max(1);
    // Keep the engine's designed independent-brain mode (per-team critic drives each side).
    let mut runner = EngineMatchRunner::new(runner_config);

    let mut frontier = match load_snapshot(&frontier_path) {
        Some(s) => {
            println!("league_resume frontier l2={:.3} params={} from {}", s.l2_norm, s.parameter_count, frontier_path);
            TeamBrain::from_snapshot(s)
        }
        None => {
            println!("league_fresh_start (no frontier at {frontier_path})");
            TeamBrain::fresh_with_seed(0xC0FFEE, 0)
        }
    };

    let train_seed_base: u32 = 0x5EED_0000;
    let mut round = 0u32;
    println!(
        "league_train_started_at_utc={} games/opp={} minutes={} weight_decay={} fresh_opp={} checkpoint_every={} frontier={} archive={}",
        chrono_now(), games_per_opp, minutes, weight_decay, fresh_opponents, checkpoint_every, frontier_path, archive_dir
    );

    loop {
        round += 1;
        let started = Instant::now();

        // Build this round's opponent league: frozen champions + fresh (exploration).
        let mut opponents: Vec<TeamBrain> = load_league(&archive_dir)
            .into_iter()
            .map(TeamBrain::from_snapshot)
            .collect();
        for i in 0..fresh_opponents {
            opponents.push(TeamBrain::fresh_with_seed(
                0xF0_0000u32.wrapping_add(round.wrapping_mul(2_654_435_761)).wrapping_add(i as u32),
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
        let mut fixtures: Vec<Fx> = Vec::new();
        let mut fixture = 0u32;
        for idx in 0..opponents.len() {
            for g in 0..games_per_opp {
                let seed = train_seed_base
                    .wrapping_add(round.wrapping_mul(1_000_003))
                    .wrapping_add(fixture.wrapping_mul(2_246_822_519))
                    .wrapping_add(g as u32);
                fixtures.push(Fx { opp_idx: idx, opp_id: idx + 1, seed, home: g % 2 == 0 });
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
        let trained_nets = std::sync::Mutex::new(Vec::<SoccerNeuralNetworkSnapshot>::new());
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
                let trained_nets = &trained_nets;
                let mut runner = runner.clone();
                let mut wf = frontier.clone();
                scope.spawn(move || {
                    use std::sync::atomic::Ordering::Relaxed;
                    for fx in slice {
                        let (t, gd) = play_and_carry(
                            &mut runner,
                            wf,
                            &opponents[fx.opp_idx],
                            fx.seed,
                            fx.home,
                            fx.opp_id,
                        );
                        wf = t;
                        if gd > 0 {
                            wins_a.fetch_add(1, Relaxed);
                        } else if gd < 0 {
                            losses_a.fetch_add(1, Relaxed);
                        }
                    }
                    if let Some(s) = wf.neural {
                        trained_nets.lock().unwrap().push(s);
                    }
                });
            }
        });
        let wins = wins_a.load(std::sync::atomic::Ordering::Relaxed);
        let losses = losses_a.load(std::sync::atomic::Ordering::Relaxed);
        if let Some(avg) = average_snapshots(base_steps, trained_nets.into_inner().unwrap()) {
            frontier.neural = Some(avg);
        }

        // Weight decay to keep the net in the trainable regime.
        if let Some(mut s) = frontier.neural.take() {
            apply_weight_decay(&mut s, weight_decay);
            frontier.neural = Some(s);
        }

        // Persist the frontier for :5055 / eval / champion-gate.
        let (l2, steps) = frontier
            .neural
            .as_ref()
            .map(|s| (s.l2_norm, s.training_steps))
            .unwrap_or((0.0, 0));
        if let Some(s) = frontier.neural.as_ref() {
            if let Err(e) = write_frontier(&frontier_path, s) {
                eprintln!("league_write_error: {e}");
            }
        }

        // Temporal checkpoint into the league (growing diversity of past selves).
        if round % (checkpoint_every as u32) == 0 {
            if let Some(s) = frontier.neural.as_ref() {
                let cp = format!("{}/league-r{:04}-{}.json", archive_dir.trim_end_matches('/'), round, chrono_stamp());
                let _ = fs::create_dir_all(&archive_dir);
                let _ = write_frontier(&cp, s);
                println!("league_checkpoint round={round} -> {cp}");
            }
        }

        println!(
            "league_round round={round} opponents={} games={} frontier_wins={wins} frontier_losses={losses} l2={l2:.3} training_steps={steps} secs={:.1}",
            opponents.len(),
            fixture,
            started.elapsed().as_secs_f64()
        );
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
