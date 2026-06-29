//! Deterministic A/B harness for the role-aware pass risk/safety appetite
//! (`DD_SOCCER_ENABLE_ROLE_PASS_RISK_APPETITE`). See
//! `docs/pass-risk-appetite-ab-plan.md`.
//!
//! Runs N full matches per seed and, for every UNIQUE on-ball pass decision, tallies — keyed by
//! **passer role × pitch zone** (final-third / middle / own-third, by `yards_to_goal`):
//!   * pass count and forward / lateral / backward split,
//!   * mean forward yards of the chosen ball,
//!   * mean expected-completion of the chosen pass (the inverse-risk proxy: a LOWER mean means
//!     riskier balls were attempted — the appetite's intended lever for forwards going forward),
//!   * mean best-receiver openness.
//! Plus team-level goals.
//!
//! The gate is read once per process, so an A/B is TWO runs (env unset vs set), same seeds:
//!   cargo run --release --bin measure_pass_risk_appetite -- [ticks] [seeds]            # baseline
//!   DD_SOCCER_ENABLE_ROLE_PASS_RISK_APPETITE=1 \
//!     cargo run --release --bin measure_pass_risk_appetite -- [ticks] [seeds]          # treatment
//!
//! NOTE: expected-completion is a *proxy* for the repriced lane-interception risk (it falls as
//! risk rises). Attributing realized interceptions to the passer's role is a follow-up (Section 3b
//! of the plan); this harness covers H1 (forward share by role×zone) and H2/H3 (risk taken via
//! expected-completion) and the H4 goals guard.

use std::collections::BTreeMap;

use soccer_engine::des::general::soccer::{
    enable_deterministic_formation_lp, MatchConfig, PlayerRole, SoccerMatch,
};

#[derive(Clone, Copy, Default)]
struct Bucket {
    passes: u64,
    fwd: u64,
    lat: u64,
    back: u64,
    sum_fwd_yds: f64,
    sum_completion: f64,
    sum_openness: f64,
}

impl Bucket {
    fn add(&mut self, fwd_yards: f64, is_backward: bool, completion: f64, openness: f64) {
        self.passes += 1;
        self.sum_fwd_yds += fwd_yards;
        self.sum_completion += completion;
        self.sum_openness += openness;
        if is_backward {
            self.back += 1;
        } else if fwd_yards > 1.25 {
            self.fwd += 1;
        } else {
            self.lat += 1;
        }
    }
    fn pct(n: u64, d: u64) -> f64 {
        if d == 0 {
            0.0
        } else {
            100.0 * n as f64 / d as f64
        }
    }
    fn mean(sum: f64, d: u64) -> f64 {
        if d == 0 {
            0.0
        } else {
            sum / d as f64
        }
    }
}

// Mirror of the engine's private `decision_trace_action_is_pass`.
fn action_is_pass(action: &str) -> bool {
    let a = action.to_ascii_lowercase();
    a.contains("pass") || a.contains("cross") || a == "route-one"
}

fn zone_label(yards_to_goal: f64) -> &'static str {
    // Thirds of a ~115yd pitch, classified by distance to the opponent goal.
    if yards_to_goal <= 38.3 {
        "final-third"
    } else if yards_to_goal >= 76.7 {
        "own-third"
    } else {
        "middle"
    }
}

fn role_label(role: PlayerRole) -> &'static str {
    match role {
        PlayerRole::Goalkeeper => "Goalkeeper",
        PlayerRole::Defender => "Defender",
        PlayerRole::Midfielder => "Midfielder",
        PlayerRole::Forward => "Forward",
    }
}

fn main() {
    enable_deterministic_formation_lp();
    let args: Vec<String> = std::env::args().collect();
    let ticks: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(9000);
    let seeds: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(6);

    let gate_on = std::env::var("DD_SOCCER_ENABLE_ROLE_PASS_RISK_APPETITE").is_ok();

    // key = "Role | zone"
    let mut buckets: BTreeMap<String, Bucket> = BTreeMap::new();
    let mut goals_home: u64 = 0;
    let mut goals_away: u64 = 0;
    let mut unique_pass_decisions: u64 = 0;

    for s in 0..seeds {
        let config = MatchConfig {
            seed: 0x5EED_0000u32.wrapping_add(s as u32),
            ..MatchConfig::default()
        };
        let mut sim = SoccerMatch::default_11v11(config);
        // Rising-edge dedup so a pass held in `last_decision` across its refractory ticks is
        // counted ONCE: remember each player's last counted (action, target, forward-yards).
        let mut last_counted: Vec<Option<(String, Option<usize>, i64)>> =
            vec![None; sim.players.len()];

        for _ in 0..ticks {
            sim.run_time_step();
            for (i, p) in sim.players.iter().enumerate() {
                let Some(d) = p.last_decision.as_ref() else {
                    continue;
                };
                let obs = &d.observation;
                if !obs.has_ball || !action_is_pass(&d.action) {
                    continue;
                }
                let Some(target) = d.action_target.as_ref() else {
                    continue;
                };
                let Some(point) = target.point else {
                    continue;
                };
                let fwd_yards = (point.y - p.position.y) * p.team.attack_dir();
                let signature = (
                    d.action.clone(),
                    target.player_id,
                    (fwd_yards * 10.0).round() as i64,
                );
                if last_counted[i].as_ref() == Some(&signature) {
                    continue; // same decision still held — already counted
                }
                last_counted[i] = Some(signature);

                let is_backward = fwd_yards < -1.0;
                let key = format!("{} | {}", role_label(p.role), zone_label(obs.yards_to_goal));
                buckets.entry(key).or_default().add(
                    fwd_yards,
                    is_backward,
                    obs.expected_pass_completion,
                    obs.best_pass_receiver_openness,
                );
                unique_pass_decisions += 1;
            }
        }
        goals_home += sim.score_home as u64;
        goals_away += sim.score_away as u64;
        eprintln!("seed {s}: final {}-{}", sim.score_home, sim.score_away);
    }

    println!(
        "\n===== PASS RISK APPETITE ({seeds} matches x {ticks} ticks) — gate {} =====",
        if gate_on { "ON (treatment)" } else { "OFF (baseline)" }
    );
    println!(
        "total goals: home {goals_home} / away {goals_away} (sum {})   unique pass decisions: {unique_pass_decisions}",
        goals_home + goals_away
    );
    println!(
        "\n{:<26}{:>7}{:>7}{:>7}{:>7}{:>11}{:>10}{:>10}",
        "role | zone", "passes", "fwd%", "lat%", "back%", "avgFwdYds", "avgComp", "avgOpen"
    );
    // Stable ordering: role major, zone minor.
    let order = [
        "Goalkeeper",
        "Defender",
        "Midfielder",
        "Forward",
    ];
    let zones = ["own-third", "middle", "final-third"];
    for r in order {
        for z in zones {
            let key = format!("{r} | {z}");
            if let Some(b) = buckets.get(&key) {
                if b.passes == 0 {
                    continue;
                }
                println!(
                    "{:<26}{:>7}{:>6.0}%{:>6.0}%{:>6.0}%{:>11.2}{:>10.3}{:>10.3}",
                    key,
                    b.passes,
                    Bucket::pct(b.fwd, b.passes),
                    Bucket::pct(b.lat, b.passes),
                    Bucket::pct(b.back, b.passes),
                    Bucket::mean(b.sum_fwd_yds, b.passes),
                    Bucket::mean(b.sum_completion, b.passes),
                    Bucket::mean(b.sum_openness, b.passes),
                );
            }
        }
    }
}
