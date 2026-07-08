//! Possession-chain export for the learned EPV (expected possession value) potential.
//!
//! Step-2 critical path from `docs/how-to-climb-codex-conversation.md`: the existing offline
//! export is value-policy rows with NO terminal possession-outcome labels, so a calibrated EPV
//! cannot be fit from it. This binary runs self-play and emits one JSONL row per tick of each
//! possession chain, labeled with the chain's TERMINAL OUTCOME — the data contract Codex
//! specified: `match_id, chain_id, team, tick, ball cell / pre-shot state, terminal outcome`.
//!
//! A chain = contiguous ticks where one team owns the ball (via `possession_team()`, which keeps
//! the chain assigned to the owner through its own passes and closes on the opponent's touch).
//! Terminal outcome (priority): goal > shot_on_target > shot > turnover > timeout(match end).
//! The per-row EPV target is the chain's terminal outcome discounted by `ticks_to_terminal`,
//! computed offline by the fitter — this file only emits the labeled states.
//!
//! Ball position is stored BOTH raw (`ball_x/ball_y` + `field_*`) and canonical to the chain
//! team's attack direction (`forward` 0=own goal .. 1=opp goal, `lateral` 0..1), so the fitter
//! can bin cells however it wants. `--features` also emits the 210-dim canonical config vector.
//!
//! Usage:
//!   soccer_export_possession_chains <out.jsonl> [games=20] [minutes=6] [seed_hex=C0FFEE00] [--features]
//!
//! Off-by-default tool: builds/reads only; changes no training path. Byte-identical to not running.

use std::io::{BufWriter, Write};
use std::time::Instant;

use soccer_engine::des::general::soccer::{
    MatchConfig, SoccerConfigVector, SoccerMatch, Team,
};

fn parse_hex(s: Option<&String>, default: u32) -> u32 {
    s.and_then(|raw| u32::from_str_radix(raw.trim_start_matches("0x"), 16).ok())
        .unwrap_or(default)
}

/// Forward progress fraction in [0,1] for `team` at pitch-length coordinate `y`:
/// 0 on the team's own goal line, 1 on the opponent goal line (mirrors `pitch_value::forward_fraction`).
fn forward_fraction(team: Team, y: f64, field_length: f64) -> f64 {
    if field_length <= 0.0 || !field_length.is_finite() || !y.is_finite() {
        return 0.5;
    }
    let centered = (y - field_length * 0.5) / field_length;
    (0.5 + team.attack_dir() * centered).clamp(0.0, 1.0)
}

/// One accumulated tick-state inside an open chain (terminal label attached at flush).
struct Row {
    tick: u64,
    forward: f64,
    lateral: f64,
    ball_x: f64,
    ball_y: f64,
    field_length: f64,
    field_width: f64,
    features: Option<Vec<f64>>,
}

#[allow(clippy::too_many_arguments)]
fn flush_chain(
    w: &mut impl Write,
    rows: &[Row],
    team: Team,
    match_id: u32,
    chain_id: u32,
    outcome: &str,
    terminal_tick: u64,
    chain_start_tick: u64,
) -> std::io::Result<usize> {
    for r in rows {
        let ttt = terminal_tick.saturating_sub(r.tick);
        let tic = r.tick.saturating_sub(chain_start_tick);
        write!(
            w,
            "{{\"match_id\":{},\"chain_id\":{},\"team\":\"{}\",\"tick\":{},\"t_in_chain\":{},\
             \"forward\":{:.4},\"lateral\":{:.4},\"ball_x\":{:.2},\"ball_y\":{:.2},\
             \"field_length\":{:.1},\"field_width\":{:.1},\
             \"terminal_outcome\":\"{}\",\"ticks_to_terminal\":{}",
            match_id, chain_id, team.label(), r.tick, tic,
            r.forward, r.lateral, r.ball_x, r.ball_y,
            r.field_length, r.field_width, outcome, ttt,
        )?;
        if let Some(f) = &r.features {
            write!(w, ",\"features\":[")?;
            for (i, v) in f.iter().enumerate() {
                if i > 0 {
                    write!(w, ",")?;
                }
                write!(w, "{:.4}", v)?;
            }
            write!(w, "]")?;
        }
        writeln!(w, "}}")?;
    }
    Ok(rows.len())
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let out_path = args
        .get(1)
        .expect("usage: soccer_export_possession_chains <out.jsonl> [games] [minutes] [seed_hex] [--features]");
    let games: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20);
    let minutes: f64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(6.0);
    let seed_base = parse_hex(args.get(4), 0xC0FF_EE00);
    let emit_features = args.iter().any(|a| a == "--features");

    let file = std::fs::File::create(out_path)?;
    let mut w = BufWriter::new(file);
    let started = Instant::now();
    let mut total_rows = 0usize;
    let mut total_chains = 0u32;
    // Outcome tally for a quick sanity read of the label distribution.
    let (mut n_goal, mut n_sot, mut n_shot, mut n_turn, mut n_timeout) = (0u64, 0u64, 0u64, 0u64, 0u64);

    for g in 0..games {
        let mut config = MatchConfig {
            duration_seconds: minutes * 60.0,
            seed: seed_base.wrapping_add(g as u32),
            ..MatchConfig::default()
        };
        config.neural_blend.actor_critic = true;
        let total_ticks = config.total_ticks();
        let mut sim = SoccerMatch::default_11v11(config);
        sim.set_uniform_elite_players();

        let match_id = seed_base.wrapping_add(g as u32);
        let mut chain_id: u32 = 0;
        let mut cur_team: Option<Team> = None;
        let mut cur_rows: Vec<Row> = Vec::new();
        let mut chain_start_tick: u64 = 0;
        let mut prev_score = (0u32, 0u32);
        // Per-team shot counters diffed each tick — MatchStats counts each shot ONCE (unlike the
        // reward-event stream, which distributes shot credit across the whole buildup chain).
        let mut prev_shots = (0u32, 0u32);
        let mut prev_sot = (0u32, 0u32);

        for _ in 0..total_ticks {
            sim.run_time_step();
            let snap = sim.export_world_snapshot();
            let poss = snap.possession_team();
            let score = (sim.score_home, sim.score_away);
            let stats = sim.summary().stats;
            let shots = (stats.shots_home, stats.shots_away);
            let sot = (stats.shots_on_target_home, stats.shots_on_target_away);
            let scoring_team = if score.0 > prev_score.0 {
                Some(Team::Home)
            } else if score.1 > prev_score.1 {
                Some(Team::Away)
            } else {
                None
            };
            // Per-team shot deltas this tick (attributed correctly), computed as values before we
            // overwrite the prev_* trackers (Home, Away).
            let sot_delta = (sot.0 > prev_sot.0, sot.1 > prev_sot.1);
            let shot_delta = (shots.0 > prev_shots.0, shots.1 > prev_shots.1);
            prev_score = score;
            prev_shots = shots;
            prev_sot = sot;

            // Open a chain when a team first gains possession.
            if cur_team.is_none() {
                if let Some(t) = poss {
                    cur_team = Some(t);
                    chain_start_tick = sim.tick;
                    cur_rows.clear();
                }
            }

            if let Some(ct) = cur_team {
                let ball = snap.ball.position;
                cur_rows.push(Row {
                    tick: sim.tick,
                    forward: forward_fraction(ct, ball.y, snap.field_length),
                    lateral: if snap.field_width > 0.0 {
                        (ball.x / snap.field_width).clamp(0.0, 1.0)
                    } else {
                        0.5
                    },
                    ball_x: ball.x,
                    ball_y: ball.y,
                    field_length: snap.field_length,
                    field_width: snap.field_width,
                    features: if emit_features {
                        Some(SoccerConfigVector::from_snapshot(&snap, ct).to_features())
                    } else {
                        None
                    },
                });

                // Close the chain on the first terminal event (priority order).
                let outcome = if scoring_team == Some(ct) {
                    Some("goal")
                } else if scoring_team.is_some() {
                    // opponent scored during our nominal possession (own-goal / scramble): turnover
                    Some("turnover")
                } else if sot_for(ct) {
                    Some("shot_on_target")
                } else if shot_for(ct) {
                    Some("shot")
                } else if poss.map_or(false, |t| t != ct) {
                    Some("turnover")
                } else {
                    None
                };

                if let Some(oc) = outcome {
                    match oc {
                        "goal" => n_goal += 1,
                        "shot_on_target" => n_sot += 1,
                        "shot" => n_shot += 1,
                        _ => n_turn += 1,
                    }
                    total_rows +=
                        flush_chain(&mut w, &cur_rows, ct, match_id, chain_id, oc, sim.tick, chain_start_tick)?;
                    chain_id += 1;
                    cur_team = None;
                    cur_rows.clear();
                }
            }
        }

        // Flush any still-open chain at match end as a timeout (non-terminal).
        if let Some(ct) = cur_team {
            n_timeout += 1;
            let terminal_tick = sim.tick;
            total_rows += flush_chain(
                &mut w, &cur_rows, ct, match_id, chain_id, "timeout", terminal_tick, chain_start_tick,
            )?;
            chain_id += 1;
        }
        total_chains += chain_id;
        eprintln!(
            "[export] game {:>2}/{games} match_id={:#010x} chains={chain_id} rows_so_far={total_rows} ({:.0}s)",
            g + 1,
            match_id,
            started.elapsed().as_secs_f64()
        );
    }

    w.flush()?;
    println!(
        "wrote {total_rows} rows across {total_chains} chains to {out_path} ({:.0}s)\n\
         outcome tally: goal={n_goal} shot_on_target={n_sot} shot={n_shot} turnover={n_turn} timeout={n_timeout}",
        started.elapsed().as_secs_f64()
    );
    Ok(())
}
