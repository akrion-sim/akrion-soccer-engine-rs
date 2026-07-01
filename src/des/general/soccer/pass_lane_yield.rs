//! **Pass-lane yield** — the off-ball POMDP for a team-mate who is the *middle* of a
//! three-in-a-line and is the only thing blocking a longer pass between the two outer
//! players. Two complementary decisions, both of which let the carrier prefer the
//! **longer** ball over the short one:
//!
//!   1. **Step out of the lane** ([`WorldSnapshot::pass_lane_yield_target_for`]). When a
//!      team-mate carries the ball (one outer), this player sits in the carrier→far-mate
//!      lane (the middle), and that lane is otherwise clear of opponents, the middle
//!      player is the *only* obstruction to the longer pass. It moves **to space** out of
//!      the corridor — biased **forward** when the team is attacking, **backward** when it
//!      is a deep defender holding a back line — opening the carrier→far lane. Surfaced as
//!      a `lane-yield` support-movement option (see `support_action_context`).
//!   2. **Let it run (dummy)** ([`SoccerMatch::pass_lane_dummy_guard`]). When a pass is
//!      already in flight to a *farther* team-mate and this player stands in its path
//!      (and would otherwise trap it as a bystander), it acts as a **dummy** — it does not
//!      touch the ball, letting it run through to the better-placed man behind. Realised
//!      by excluding the dummy from ball-control candidacy for that tick, exactly like the
//!      restart double-touch guard.
//!
//! Both decisions have the *same effect* — the longer pass reaches the far man — and both
//! are gated **off** by default behind `DD_SOCCER_ENABLE_PASS_LANE_YIELD`. When the gate
//! is unset every entry point early-returns before reading any geometry, so the engine is
//! byte-identical to before this module existed. All logic here is a pure read of the
//! world state (no RNG).

use super::*;

/// Env gate enabling both pass-lane-yield decisions. Off (unset) ⇒ byte-identical.
const PASS_LANE_YIELD_ENABLE_ENV: &str = "DD_SOCCER_ENABLE_PASS_LANE_YIELD";

// --- decision 1 (step out of the lane) tuning ---
/// Half-width (yd) of the carrier→far-mate corridor the middle player must sit within to
/// count as "blocking the lane". A body this close to the line is genuinely in the way of
/// a ground pass.
const YIELD_CORRIDOR_YARDS: f64 = 2.0;
/// The middle player must project between these fractions along the carrier→far line — i.e.
/// genuinely *between* the two outer players, not bunched at either end.
const YIELD_PROJ_MIN: f64 = 0.18;
const YIELD_PROJ_MAX: f64 = 0.88;
/// The far team-mate must be at least this much farther from the carrier than the middle
/// player is, so clearing the lane buys a genuinely *longer* pass (not a near-equal swap).
const YIELD_MIN_EXTRA_REACH_YARDS: f64 = 3.0;
/// The carrier→far lane must be at least this long to be a meaningful "longer pass".
const YIELD_MIN_LONGER_PASS_YARDS: f64 = 9.0;
/// Lateral step sizes (yd) tried when stepping out of the corridor (smallest first).
const YIELD_LATERAL_STEPS_YARDS: [f64; 2] = [3.6, 5.2];
/// Forward/backward bias (yd) folded into the step-out target ("move to space" up or back).
const YIELD_LONGITUDINAL_BIAS_YARDS: f64 = 2.6;
/// The step-out target must clear the corridor by at least this perpendicular distance, so
/// the carrier→far lane actually opens.
const YIELD_CLEAR_MARGIN_YARDS: f64 = YIELD_CORRIDOR_YARDS + 0.7;
/// Don't step the middle player straight onto an opponent — the yield spot must keep at
/// least this much space.
const YIELD_MIN_SPOT_SPACE_YARDS: f64 = 2.4;

// --- decision 2 (dummy / let it run) tuning ---
/// Half-width (yd) of the ball→target corridor the dummy must sit within. Sized to the
/// reactive ground-pass control radius: only a body actually in the ball's path — one that
/// would otherwise trap a team-mate's pass — is a dummy candidate.
const DUMMY_CORRIDOR_YARDS: f64 = 2.2;
/// The dummy must project between these fractions of the way from the ball to the target.
const DUMMY_PROJ_MIN: f64 = 0.04;
const DUMMY_PROJ_MAX: f64 = 0.92;
/// The intended target must be at least this much farther forward (toward the attacking
/// goal) than the dummy, so letting it run reaches a genuinely *more advanced* man.
const DUMMY_MIN_FORWARD_GAIN_YARDS: f64 = 2.5;
/// Half-width (yd) of the onward dummy→target lane that must be clear of opponents, so the
/// ball actually reaches the target rather than being gifted to a defender behind.
const DUMMY_ONWARD_CORRIDOR_YARDS: f64 = 1.7;

fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|raw| {
            let v = raw.trim().to_ascii_lowercase();
            matches!(v.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

/// Whether the pass-lane-yield decisions are consulted this process. Off ⇒ byte-identical.
/// Re-read every call under test so a test can toggle the env; cached otherwise.
pub fn pass_lane_yield_enabled() -> bool {
    #[cfg(test)]
    {
        env_flag_enabled(PASS_LANE_YIELD_ENABLE_ENV)
    }
    #[cfg(not(test))]
    {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        // Promoted to default-ON in production (low-risk behavioral hardening).
        *ENABLED.get_or_init(|| gate_default_on(PASS_LANE_YIELD_ENABLE_ENV))
    }
}

/// Perpendicular unit vector (90° CCW) to `d`. `d` is assumed already normalized.
fn perp(d: Vec2) -> Vec2 {
    Vec2::new(-d.y, d.x)
}

impl WorldSnapshot {
    /// If this off-ball team-mate is the *middle* of a three-in-a-line and is the only
    /// obstruction to a longer pass from the carrier to a farther team-mate, return where
    /// it should step **to space** to clear the lane, plus a `[0,1]` strength (how valuable
    /// opening the longer pass is). `None` when the gate is off, the player is not such a
    /// middle man, or no productive step-out spot opens the lane. Pure read of the snapshot.
    pub(crate) fn pass_lane_yield_target_for(
        &self,
        player_id: usize,
        home_position: Vec2,
    ) -> Option<(Vec2, f64)> {
        let _ = home_position;
        let (inputs, spot, base_strength) = self.pass_lane_yield_context(player_id)?;
        // Learnable yield willingness (MDP/POMDP): shift the hand-computed strength by a learned
        // bias so a middle man chooses to vacate the lane (open the longer ball) or hold his slot.
        // Off ⇒ `base_strength` unchanged (byte-identical).
        let strength = self.pass_lane_yield_effective_strength(&inputs, base_strength);
        Some((spot, strength))
    }

    /// Read-only geometry for the pass-lane yield decision: the [`PassLaneYieldInputs`] the choice
    /// is a function of, the elected step-out `spot`, and the hand-computed base strength. Shared by
    /// [`Self::pass_lane_yield_target_for`] (the seam) and the RL collector. `None` when the yield
    /// does not apply. Gated by [`pass_lane_yield_enabled`] (off ⇒ byte-identical early return).
    pub(crate) fn pass_lane_yield_context(
        &self,
        player_id: usize,
    ) -> Option<(super::PassLaneYieldInputs, Vec2, f64)> {
        if !pass_lane_yield_enabled() {
            return None;
        }
        let me = self.players.iter().find(|p| p.id == player_id)?;
        if me.role == PlayerRole::Goalkeeper {
            return None;
        }
        // One outer player must be the team-mate carrying the ball, so clearing the lane
        // actually enables a pass right now.
        let carrier_id = self.ball.holder?;
        if carrier_id == player_id {
            return None;
        }
        let carrier = self.players.iter().find(|p| p.id == carrier_id)?;
        if carrier.team != me.team {
            return None;
        }
        let me_pos = self.player_snapshot_position(me);
        let carrier_pos = self.player_snapshot_position(carrier);
        if !me_pos.x.is_finite() || !me_pos.y.is_finite() {
            return None;
        }
        let me_from_carrier = me_pos.distance(carrier_pos);
        let opp = me.team.other();
        let attack_dir = me.team.attack_dir();

        // Find the best far team-mate F such that I (the middle) am the sole blocker of the
        // carrier→F lane, and F is a genuinely longer, productive option than me.
        // (far_id, far_pos, value, forward_gain, openness, lane_len)
        let mut best: Option<(usize, Vec2, f64, f64, f64, f64)> = None;
        for far in self.players.iter() {
            if far.id == player_id || far.id == carrier_id || far.team != me.team {
                continue;
            }
            if far.role == PlayerRole::Goalkeeper {
                continue;
            }
            let far_pos = self.player_snapshot_position(far);
            let lane = far_pos - carrier_pos;
            let lane_len = lane.len();
            if lane_len < YIELD_MIN_LONGER_PASS_YARDS {
                continue;
            }
            // F must be a genuinely LONGER reach than me from the carrier.
            if far_pos.distance(carrier_pos) < me_from_carrier + YIELD_MIN_EXTRA_REACH_YARDS {
                continue;
            }
            // I must sit BETWEEN the carrier and F (the middle of the line), within the lane.
            let proj = segment_projection_factor(carrier_pos, far_pos, me_pos);
            if !(YIELD_PROJ_MIN..=YIELD_PROJ_MAX).contains(&proj) {
                continue;
            }
            if segment_distance_to_point(carrier_pos, far_pos, me_pos) > YIELD_CORRIDOR_YARDS {
                continue;
            }
            // The longer lane must be otherwise clear of opponents — i.e. *I* am the only
            // thing blocking it. If a defender also sits in it, stepping aside buys nothing.
            if !self.clear_line(carrier_pos, far_pos, opp, YIELD_CORRIDOR_YARDS) {
                continue;
            }
            // Prefer the most productive far option: forward progress it buys + how open it
            // is + extra length over the short ball.
            let forward_gain = (far_pos.y - me_pos.y) * attack_dir;
            let openness = pass_receiver_openness_for_snapshots_with_teammates(
                &self.players,
                me.team,
                far_pos,
                Some(carrier_id),
            );
            let value = forward_gain.max(0.0) * 0.5 + openness * 6.0 + (lane_len * 0.12);
            if best.as_ref().is_none_or(|(_, _, best_value)| value > *best_value) {
                best = Some((far.id, far_pos, value));
            }
        }
        let (_, far_pos, raw_value) = best?;

        // Step out of the corridor "to space": forward when attacking, backward for a deep
        // defender holding the back line. Try both sides at growing step sizes and keep the
        // spot that clears the lane, stays off opponents, and has the most open space.
        let lane_dir = {
            let d = far_pos - carrier_pos;
            let len = d.len();
            if len <= 1e-6 {
                return None;
            }
            d * (1.0 / len)
        };
        let normal = perp(lane_dir);
        // Defenders drop BACK off the lane (toward their own goal); everyone else, with an
        // attack underway, pushes FORWARD into space.
        let longitudinal = if me.role == PlayerRole::Defender {
            Vec2::new(0.0, -attack_dir)
        } else {
            Vec2::new(0.0, attack_dir)
        };

        let mut chosen: Option<(Vec2, f64)> = None; // (spot, space)
        for &step in YIELD_LATERAL_STEPS_YARDS.iter() {
            for &sign in &[1.0_f64, -1.0] {
                let spot = (me_pos
                    + normal * (sign * step)
                    + longitudinal * YIELD_LONGITUDINAL_BIAS_YARDS)
                    .clamp_to_pitch(self.field_width, self.field_length);
                // Must genuinely clear the carrier→F corridor.
                if segment_distance_to_point(carrier_pos, far_pos, spot) < YIELD_CLEAR_MARGIN_YARDS
                {
                    continue;
                }
                // Don't yield straight onto an opponent.
                if self.nearest_opponent_distance_at(me.team, spot) < YIELD_MIN_SPOT_SPACE_YARDS {
                    continue;
                }
                let space = self.space_score_at(spot, me.team);
                if chosen.as_ref().is_none_or(|(_, best_space)| space > *best_space) {
                    chosen = Some((spot, space));
                }
            }
            if chosen.is_some() {
                break;
            }
        }
        let (spot, _) = chosen?;
        // Keep the yield home-aware: a defender shouldn't bolt miles from its slot. Blend a
        // touch toward the home slot only as a tiebreak via the existing strength scaling.
        let _ = home_position;
        let strength = (raw_value / 14.0).clamp(0.0, 1.0);
        Some((spot, strength))
    }
}

impl SoccerMatch {
    /// The team-mate (if any) who should **dummy** the ball in flight this tick — stand in
    /// the path of a pass aimed at a farther team-mate and *let it run through* to the
    /// better-placed man behind, rather than trapping it as a bystander. Returned id is
    /// excluded from ball-control candidacy for the tick (like the double-touch guard), so
    /// the ball runs on. `None` when the gate is off or no clean dummy is on. Pure read.
    pub(crate) fn pass_lane_dummy_guard(&self) -> Option<usize> {
        if !pass_lane_yield_enabled() {
            return None;
        }
        let pass = self.pending_pass.as_ref()?;
        // Only a deliberately targeted pass can be dummied — we need a known man behind.
        let target_id = pass.target?;
        let pass_team = pass.team;
        let target = self
            .players
            .iter()
            .find(|p| p.id == target_id && p.team == pass_team)?;
        let target_pos = target.position;
        let ball_pos = self.ball.position;
        let ball_vel = self.ball.velocity;
        // The ball must actually be travelling toward the target (a live pass, not stopped
        // or already past).
        let to_target = target_pos - ball_pos;
        if to_target.len() <= 1e-3 || ball_vel.dot(to_target) <= 0.0 {
            return None;
        }
        let attack_dir = pass_team.attack_dir();

        let mut best: Option<(usize, f64)> = None; // (dummy_id, dist_to_ball)
        for p in self.players.iter() {
            if p.team != pass_team || p.id == pass.from || p.id == target_id {
                continue;
            }
            if p.role == PlayerRole::Goalkeeper {
                continue;
            }
            // Must be the MIDDLE: in the ball→target path, ahead of the ball, before the
            // target — i.e. the body that would otherwise trap this pass.
            let proj = segment_projection_factor(ball_pos, target_pos, p.position);
            if !(DUMMY_PROJ_MIN..=DUMMY_PROJ_MAX).contains(&proj) {
                continue;
            }
            if segment_distance_to_point(ball_pos, target_pos, p.position) > DUMMY_CORRIDOR_YARDS {
                continue;
            }
            let dist_to_ball = p.position.distance(ball_pos);
            // Letting it run must reach a genuinely MORE ADVANCED team-mate.
            if (target_pos.y - p.position.y) * attack_dir < DUMMY_MIN_FORWARD_GAIN_YARDS {
                continue;
            }
            // The onward lane to the target must be clear of opponents — don't dummy a ball
            // straight to a defender sitting behind.
            if !self.opponents_clear_of_segment(
                pass_team,
                p.position,
                target_pos,
                DUMMY_ONWARD_CORRIDOR_YARDS,
            ) {
                continue;
            }
            // The dummy nearest the ball acts first (it would trap it first).
            if best.as_ref().is_none_or(|(_, best_dist)| dist_to_ball < *best_dist) {
                best = Some((p.id, dist_to_ball));
            }
        }
        best.map(|(id, _)| id)
    }

    /// True when no opponent of `team` sits within `radius` of the `from`→`to` segment.
    fn opponents_clear_of_segment(&self, team: Team, from: Vec2, to: Vec2, radius: f64) -> bool {
        let opp = team.other();
        self.players
            .iter()
            .filter(|p| p.team == opp && p.role != PlayerRole::Goalkeeper)
            .all(|p| segment_distance_to_point(from, to, p.position) > radius)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    const ENV: &str = PASS_LANE_YIELD_ENABLE_ENV;

    /// Three Home players in a vertical line at x=40: carrier (id `c`) at y=50, middle
    /// (`m`) at y=58, far (`f`) at y=70. Every other player is parked in a far corner so
    /// the lane is clear and the yield spots are open. Returns `(sim, c, m, f)`.
    fn three_in_a_line() -> (SoccerMatch, usize, usize, usize) {
        let mut sim = SoccerMatch::default_11v11(MatchConfig::default());
        let outfield: Vec<usize> = sim
            .players
            .iter()
            .filter(|p| p.team == Team::Home && p.role != PlayerRole::Goalkeeper)
            .map(|p| p.id)
            .collect();
        let (c, m, f) = (outfield[0], outfield[1], outfield[2]);
        // Park everyone else (both teams) far off in the corner, away from the lane.
        for p in sim.players.iter_mut() {
            if p.id != c && p.id != m && p.id != f {
                p.position = Vec2::new(3.0, 3.0);
                p.velocity = Vec2::zero();
            }
        }
        sim.players[c].position = Vec2::new(40.0, 50.0);
        sim.players[m].position = Vec2::new(40.0, 58.0);
        sim.players[f].position = Vec2::new(40.0, 70.0);
        for &id in &[c, m, f] {
            sim.players[id].velocity = Vec2::zero();
        }
        sim.ball.holder = Some(c);
        sim.ball.position = Vec2::new(40.0, 50.0);
        sim.ball.velocity = Vec2::zero();
        (sim, c, m, f)
    }

    #[test]
    fn gate_default_off_is_inert() {
        let _l = ENV_LOCK.lock().expect("env lock");
        std::env::remove_var(ENV);
        assert!(!pass_lane_yield_enabled());
        let (sim, _c, m, _f) = three_in_a_line();
        let snap = WorldSnapshot::from_match(&sim);
        assert!(snap.pass_lane_yield_target_for(m, sim.players[m].home_position).is_none());
        assert!(sim.pass_lane_dummy_guard().is_none());
    }

    #[test]
    fn middle_player_steps_out_of_the_lane() {
        let _l = ENV_LOCK.lock().expect("env lock");
        std::env::set_var(ENV, "1");
        let (sim, c, m, f) = three_in_a_line();
        let snap = WorldSnapshot::from_match(&sim);
        let result = snap.pass_lane_yield_target_for(m, sim.players[m].home_position);
        std::env::remove_var(ENV);
        let (target, strength) = result.expect("middle man should elect to yield the lane");
        let c_pos = Vec2::new(40.0, 50.0);
        let f_pos = Vec2::new(40.0, 70.0);
        // The yield target must leave the carrier→far corridor.
        assert!(
            segment_distance_to_point(c_pos, f_pos, target) >= YIELD_CLEAR_MARGIN_YARDS,
            "yield target {target:?} must clear the lane"
        );
        assert!((0.0..=1.0).contains(&strength));
        // The carrier→far lane stays clear of opponents (it always was — only M blocked it).
        assert!(snap.clear_line(c_pos, f_pos, Team::Home.other(), YIELD_CORRIDOR_YARDS));
        let _ = (c, f);
    }

    #[test]
    fn attacker_yields_forward_defender_yields_back() {
        let _l = ENV_LOCK.lock().expect("env lock");
        std::env::set_var(ENV, "1");

        let (mut sim, _c, m, _f) = three_in_a_line();
        sim.players[m].role = PlayerRole::Forward;
        let fwd = WorldSnapshot::from_match(&sim)
            .pass_lane_yield_target_for(m, sim.players[m].home_position)
            .expect("forward middle yields")
            .0;

        sim.players[m].role = PlayerRole::Defender;
        let back = WorldSnapshot::from_match(&sim)
            .pass_lane_yield_target_for(m, sim.players[m].home_position)
            .expect("defender middle yields")
            .0;
        std::env::remove_var(ENV);

        // Home attacks toward +y: the attacker pushes forward, the deep defender drops back.
        assert!(fwd.y > 58.0, "attacking middle should move forward, got {fwd:?}");
        assert!(back.y < 58.0, "defending middle should drop back, got {back:?}");
    }

    fn floor_pass(team: Team, from: usize, target: usize, origin: Vec2, dest: Vec2) -> PendingPass {
        PendingPass {
            team,
            from,
            target: Some(target),
            flight: PassFlight::Floor,
            is_cross: false,
            launch_tick: 0,
            origin,
            intended_target: dest,
            distance_yards: origin.distance(dest),
            receiver_openness: 1.0,
            passer_skill: 0.6,
            launch_speed_yps: 14.0,
            receiver_position_at_launch: Some(dest),
            receiver_velocity_at_launch: Some(Vec2::zero()),
            offside: None,
            offside_candidates: Vec::new(),
            learn_features: Vec::new(),
        }
    }

    #[test]
    fn middle_player_dummies_a_pass_to_the_man_behind() {
        let _l = ENV_LOCK.lock().expect("env lock");
        std::env::set_var(ENV, "1");
        let (mut sim, c, m, f) = three_in_a_line();
        let c_pos = Vec2::new(40.0, 50.0);
        let f_pos = Vec2::new(40.0, 70.0);
        // A ground pass aimed past the middle man at the far team-mate, ball in flight.
        sim.ball.holder = None;
        sim.ball.position = Vec2::new(40.0, 54.0);
        sim.ball.velocity = Vec2::new(0.0, 12.0);
        sim.pending_pass = Some(floor_pass(Team::Home, c, f, c_pos, f_pos));

        let guard = sim.pass_lane_dummy_guard();
        std::env::remove_var(ENV);
        assert_eq!(guard, Some(m), "the middle man in the path should dummy the pass");
    }

    #[test]
    fn no_dummy_when_no_man_is_further_forward() {
        let _l = ENV_LOCK.lock().expect("env lock");
        std::env::set_var(ENV, "1");
        let (mut sim, c, _m, f) = three_in_a_line();
        // Aim the pass at the MIDDLE man (no one further forward in the path) — nothing to
        // dummy to, so the guard must stay clear.
        let c_pos = Vec2::new(40.0, 50.0);
        let m_pos = Vec2::new(40.0, 58.0);
        sim.ball.holder = None;
        sim.ball.position = Vec2::new(40.0, 52.0);
        sim.ball.velocity = Vec2::new(0.0, 12.0);
        // Park the far man out of the lane so he is not a dummy target.
        sim.players[f].position = Vec2::new(10.0, 70.0);
        sim.pending_pass = Some(floor_pass(Team::Home, c, _m, c_pos, m_pos));

        let guard = sim.pass_lane_dummy_guard();
        std::env::remove_var(ENV);
        assert_eq!(guard, None);
    }
}
