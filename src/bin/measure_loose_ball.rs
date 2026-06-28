//! Ad-hoc measurement harness: quantify how long a loose ball sits with a team
//! making NO progress toward winning it — the empirical evidence for the
//! "ball uncontested for >2s, unrealistic" gap.
//!
//! Pure observation: reads `sim.snapshot()` (a `MatchFrame`) each tick, computes
//! the user's exact per-team condition, and tallies streak durations. It changes
//! NO engine logic, so it reflects current loose-ball behaviour.
//!
//! For each tick the ball is loose (no holder, no pending pass, no set play):
//!   * team_min_dist[T] = distance from T's nearest non-GK outfielder to the ball.
//!   * "no teammate near"  = team_min_dist[T] > NEAR_RADIUS (2 yd).
//!   * "no progress (3t)"  = team_min_dist[T] now >= its value 3 ticks ago - eps
//!                           (the team's closest man hasn't gotten any closer).
//!   * team is "failing to contest" when both hold.
//!   * ball is "fully idle" when BOTH teams are failing to contest at once
//!     (nobody within 2 yd from either side AND neither side closing) — the
//!     unrealistic "ball sits in space" case.
//!
//! Run: cargo run --release --bin measure_loose_ball [ticks] [seeds]

use soccer_engine::des::general::soccer::{enable_deterministic_formation_lp, MatchConfig, SoccerMatch, PlayerRole, Team};

const NEAR_RADIUS_YARDS: f64 = 2.0;
const PROGRESS_WINDOW_TICKS: usize = 3;
const PROGRESS_EPS_YARDS: f64 = 1e-3;
/// Only a SETTLED loose ball counts as "should be contested": a ball flying
/// through the air (a pass/clearance/shot) is correctly left alone until it lands,
/// so we exclude fast balls. 6 yd/s mirrors the engine's clean-control speed; we
/// also report a stricter "nearly stopped" tier.
const SETTLED_SPEED_YPS: f64 = 6.0;
const NEARLY_STOPPED_YPS: f64 = 2.5;

fn team_min_dist(frame_players: &[(Team, PlayerRole, f64)], team: Team) -> f64 {
    frame_players
        .iter()
        .filter(|(t, role, _)| *t == team && *role != PlayerRole::Goalkeeper)
        .map(|(_, _, d)| *d)
        .fold(f64::INFINITY, f64::min)
}

/// A per-team rolling record of the last few min-distances, to test "got closer".
#[derive(Default)]
struct ProgressTracker {
    history: std::collections::VecDeque<f64>,
}
impl ProgressTracker {
    fn push_and_no_progress(&mut self, current: f64) -> bool {
        // Compare against the value PROGRESS_WINDOW_TICKS ago (front of a window
        // of that size). No progress = current is not meaningfully smaller.
        let no_progress = match self.history.front() {
            Some(&past) if self.history.len() >= PROGRESS_WINDOW_TICKS => {
                current >= past - PROGRESS_EPS_YARDS
            }
            // Not enough history yet ⇒ can't claim "no progress for 3 ticks".
            _ => false,
        };
        self.history.push_back(current);
        while self.history.len() > PROGRESS_WINDOW_TICKS {
            self.history.pop_front();
        }
        no_progress
    }
    fn reset(&mut self) {
        self.history.clear();
    }
}

#[derive(Default)]
struct StreakStats {
    cur_len: u64,
    episodes: u64,
    total_ticks: u64,
    max_len: u64,
    // streaks at least this many seconds
    over_0_5s: u64,
    over_1s: u64,
    over_2s: u64,
    over_3s: u64,
    // accumulated context during failing ticks
    sum_nearest_dist: f64,
    sum_ball_speed: f64,
}
impl StreakStats {
    fn tick(&mut self, failing: bool, dt: f64, nearest_dist: f64, ball_speed: f64) {
        if failing {
            self.cur_len += 1;
            self.total_ticks += 1;
            self.sum_nearest_dist += nearest_dist;
            self.sum_ball_speed += ball_speed;
        } else {
            self.close(dt);
        }
    }
    fn close(&mut self, dt: f64) {
        if self.cur_len > 0 {
            self.episodes += 1;
            self.max_len = self.max_len.max(self.cur_len);
            let secs = self.cur_len as f64 * dt;
            if secs >= 0.5 {
                self.over_0_5s += 1;
            }
            if secs >= 1.0 {
                self.over_1s += 1;
            }
            if secs >= 2.0 {
                self.over_2s += 1;
            }
            if secs >= 3.0 {
                self.over_3s += 1;
            }
        }
        self.cur_len = 0;
    }
    fn report(&self, name: &str, dt: f64, loose_ticks: u64) {
        let total_secs = self.total_ticks as f64 * dt;
        let max_secs = self.max_len as f64 * dt;
        println!("--- {name} ---");
        println!(
            "  failing ticks: {} ({:.1}% of loose time = {:.1}s)",
            self.total_ticks,
            100.0 * self.total_ticks as f64 / loose_ticks.max(1) as f64,
            total_secs
        );
        println!(
            "  episodes: {}   max streak: {:.2}s   episodes >=0.5s/1s/2s/3s: {}/{}/{}/{}",
            self.episodes, max_secs, self.over_0_5s, self.over_1s, self.over_2s, self.over_3s
        );
        if self.total_ticks > 0 {
            println!(
                "  during failing ticks: mean nearest-body {:.2} yd, mean ball speed {:.2} yps",
                self.sum_nearest_dist / self.total_ticks as f64,
                self.sum_ball_speed / self.total_ticks as f64
            );
        }
    }
}

fn main() {
    enable_deterministic_formation_lp();
    let args: Vec<String> = std::env::args().collect();
    let ticks: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(6000);
    let seeds: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);

    let mut loose_ticks: u64 = 0;
    let mut settled_loose_ticks: u64 = 0;
    let mut nearly_stopped_ticks: u64 = 0;
    let mut total_ticks: u64 = 0;
    let dt = 0.1f64; // 10 Hz contract; refined from clock below if available.
    let mut dt_seen = dt;

    let mut home_fail = StreakStats::default();
    let mut away_fail = StreakStats::default();
    let mut ball_idle = StreakStats::default(); // BOTH teams failing at once
    // Settled ball with nobody within 2yd, REGARDLESS of whether someone is
    // closing — i.e. "ball physically unpossessed" time, which is what the eye
    // reads as 'uncontested' even while a far chaser is en route.
    let mut settled_unpossessed = StreakStats::default();

    for s in 0..seeds {
        let config = MatchConfig {
            seed: 0x5EED_0000u32.wrapping_add(s as u32),
            ..MatchConfig::default()
        };
        let mut sim = SoccerMatch::default_11v11(config);
        let mut home_track = ProgressTracker::default();
        let mut away_track = ProgressTracker::default();
        let mut prev_clock = 0.0f64;

        for t in 0..ticks {
            sim.run_time_step();
            let frame = sim.to_frame();
            if t == 1 {
                let d = frame.clock_seconds - prev_clock;
                if d > 1e-4 && d < 1.0 {
                    dt_seen = d;
                }
            }
            prev_clock = frame.clock_seconds;
            total_ticks += 1;

            let ball_loose = frame.ball.holder.is_none()
                && frame.pending_pass.is_none()
                && frame.active_set_play.is_none();
            if !ball_loose {
                home_track.reset();
                away_track.reset();
                home_fail.close(dt_seen);
                away_fail.close(dt_seen);
                ball_idle.close(dt_seen);
                settled_unpossessed.close(dt_seen);
                continue;
            }
            loose_ticks += 1;

            let ball_pos = frame.ball.position;
            let ball_speed = frame.ball.velocity.len();
            // A ball in flight is correctly left alone until it settles; only a
            // SETTLED loose ball is one a team "should" be contesting.
            let settled = ball_speed < SETTLED_SPEED_YPS;
            if settled {
                settled_loose_ticks += 1;
            }
            if ball_speed < NEARLY_STOPPED_YPS {
                nearly_stopped_ticks += 1;
            }

            let players: Vec<(Team, PlayerRole, f64)> = frame
                .players
                .iter()
                .map(|p| (p.team, p.role, p.position.distance(ball_pos)))
                .collect();

            let home_min = team_min_dist(&players, Team::Home);
            let away_min = team_min_dist(&players, Team::Away);
            let nearest = home_min.min(away_min);

            let home_no_progress = home_track.push_and_no_progress(home_min);
            let away_no_progress = away_track.push_and_no_progress(away_min);

            // "Failing to contest" only applies to a SETTLED ball.
            let home_failing = settled && home_min > NEAR_RADIUS_YARDS && home_no_progress;
            let away_failing = settled && away_min > NEAR_RADIUS_YARDS && away_no_progress;
            let ball_fully_idle = home_failing && away_failing;

            home_fail.tick(home_failing, dt_seen, home_min, ball_speed);
            away_fail.tick(away_failing, dt_seen, away_min, ball_speed);
            ball_idle.tick(ball_fully_idle, dt_seen, nearest, ball_speed);
            settled_unpossessed.tick(settled && nearest > NEAR_RADIUS_YARDS, dt_seen, nearest, ball_speed);
        }
        // Close any open streaks at match end.
        home_fail.close(dt_seen);
        away_fail.close(dt_seen);
        ball_idle.close(dt_seen);
        settled_unpossessed.close(dt_seen);
        eprintln!("seed {s}: final {}-{}", sim.score_home, sim.score_away);
    }

    let loose_secs = loose_ticks as f64 * dt_seen;
    let settled_secs = settled_loose_ticks as f64 * dt_seen;
    println!("\n===== LOOSE-BALL CONTEST MEASUREMENT ({seeds} matches x {ticks} ticks, dt={dt_seen:.3}s) =====");
    println!(
        "total ticks: {total_ticks}   loose ticks: {loose_ticks} ({:.1}% of play = {:.1}s)",
        100.0 * loose_ticks as f64 / total_ticks.max(1) as f64,
        loose_secs
    );
    println!(
        "  of loose: SETTLED (<{SETTLED_SPEED_YPS} yps): {settled_loose_ticks} ({:.1}% of loose = {:.1}s)   nearly-stopped (<{NEARLY_STOPPED_YPS} yps): {nearly_stopped_ticks}",
        100.0 * settled_loose_ticks as f64 / loose_ticks.max(1) as f64,
        settled_secs,
    );
    println!(
        "thresholds: near<= {NEAR_RADIUS_YARDS} yd, no-progress over {PROGRESS_WINDOW_TICKS} ticks, contest only counted on SETTLED balls\n"
    );
    // Denominator = settled loose ticks (the only ticks where "failing" can fire).
    settled_unpossessed.report("SETTLED BALL UNPOSSESSED (nobody within 2yd, even if a far chaser is closing)", dt_seen, settled_loose_ticks);
    ball_idle.report("BALL FULLY IDLE (settled; both teams failing to contest)", dt_seen, settled_loose_ticks);
    home_fail.report("HOME failing to contest a settled ball", dt_seen, settled_loose_ticks);
    away_fail.report("AWAY failing to contest a settled ball", dt_seen, settled_loose_ticks);
    println!(
        "\n(Episodes >=2s in 'BALL FULLY IDLE' = the gap: a SETTLED loose ball neither team\n commits to. Large mean nearest-body distance = the 'open-space ball nobody chases' case.)"
    );
}
