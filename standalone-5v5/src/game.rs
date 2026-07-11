//! 5-a-side soccer sim: continuous 2D physics, discrete macro actions, NO
//! formation-shape LP/IPM and NO POMDP feature stack. Both teams run under
//! identical rules; the learner controls Team A, the scripted baseline Team B.

use crate::rng::Rng;

// ---- Field / rules ----------------------------------------------------------
// Small-sided pitch (roughly a 5-a-side / futsal court, not a full field).
pub const FIELD_L: f32 = 42.0;
pub const FIELD_W: f32 = 28.0;
pub const GOAL_HALF: f32 = 3.5; // half goal-mouth width in y (~7m net)
#[allow(dead_code)]
pub const FINAL_THIRD_X: f32 = FIELD_L * 2.0 / 3.0; // attacking-third boundary (kept for reference)
pub const SHOOT_X: f32 = FIELD_L / 2.0; // A may shoot once in the OPPONENT'S HALF (not just the final third)
pub const N: usize = 5; // players per team (index 0 == goalkeeper)
pub const GK: usize = 0; // goalkeeper index; controlled by a fixed rule, not the policy
pub const DT: f32 = 0.05; // seconds per decision tick -> 20 Hz sim (real-time 20 fps)
pub const HZ: f32 = 1.0 / DT; // ticks per second (for real-time viewer playback)
pub const STEPS: usize = 600; // ticks per game (~30s at 20 Hz)

const PLAYER_SPEED: f32 = 6.5; // legacy reference speed (~= run_medium); kept for keeper reach/util

// ---- Player gears -----------------------------------------------------------
// Seven discrete speeds the policy can pick for any movement/dribble, from
// standing still to a full sprint. Sim units are yards/second; the top gear is
// scaled so a full sprint ~= 26 mph (the human ceiling) and, while carrying the
// ball, gears are capped at BALL_SPEED_CAP ~= 22 mph.
pub const NS: usize = 7; // number of speed gears (the speed action head)
pub const SPD_STAND: usize = 0;
pub const SPD_WALK: usize = 1;
#[allow(dead_code)]
pub const SPD_JOG: usize = 2;
pub const SPD_SKIP: usize = 3;
pub const SPD_RUN_SLOW: usize = 4;
pub const SPD_RUN_MED: usize = SPD_RUN_SLOW;
pub const SPD_RUN_FAST: usize = 5;
pub const SPD_SPRINT: usize = 6;
// still · walk · jog · skip · run_slow · run_fast · sprint
// Low end kept non-crippling (a slow gear still lets you get around) so the speed
// policy's exploration can't paralyze the game; STAND is the only truly-still gear.
const SPEEDS: [f32; NS] = [0.0, 3.0, 4.5, 5.5, 6.5, 8.5, 11.0];
// Ball-carrying is much slower than open-field running (you can't sprint flat-out
// with the ball at your feet) — and, critically, keeping it near v3's control
// speed stops the policy from dribbling forever to evade the 2-pass shot gate.
const BALL_SPEED_CAP: f32 = 4.8;

/// Map a speed gear to yards/second, capping ball-carrying to ~22 mph.
fn speed_val(gear: usize, carrying: bool) -> f32 {
    let s = SPEEDS[gear.min(NS - 1)];
    if carrying {
        s.min(BALL_SPEED_CAP)
    } else {
        s
    }
}
const CONTROL_RADIUS: f32 = 1.5; // secure a received ball -> possessions can develop
const RECEIVE_RADIUS: f32 = 2.8; // intended pass receiver collects from further out
const AERIAL_CONTROL_SPACE: f32 = 3.6; // a SCOOPED (aerial) pass needs the receiver this open to control
const CURL_MIN_DIST: f32 = 20.0; // passes/shots longer than this can be given curl (spin)
const CURL_ACCEL: f32 = 6.0; // lateral curl acceleration on a long ball (bends around a defender)
const TACKLE_RADIUS: f32 = 1.6;
const TACKLE_PROB: f32 = 0.12; // per-tick; retuned for 20 Hz to keep same per-second rate
const BALL_FRICTION: f32 = 0.965; // per-tick decay retuned for 20 Hz (same per-second decay)
const PASS_SPEED: f32 = 18.0;
const SHOT_SPEED: f32 = 24.0;
const CLEAR_SPEED: f32 = 20.0;
const CAPTURE_MAX_BALL_SPEED: f32 = 26.0;
const KEEPER_REACH: f32 = 1.9; // keeper saves spam; well-placed shots still beat it
const KEEPER_SPEED: f32 = 6.0;

// ---- Action space -----------------------------------------------------------
pub const NA: usize = 15;
pub const A_SHOOT: usize = 0;
pub const A_PASS_A: usize = 1;
pub const A_PASS_B: usize = 2;
pub const A_PASS_C: usize = 3;
pub const A_DRIB_FWD: usize = 4;
pub const A_DRIB_LEFT: usize = 5;
pub const A_DRIB_RIGHT: usize = 6;
pub const A_CLEAR: usize = 7;
pub const A_HOLD: usize = 8;
pub const A_CHASE: usize = 9;
pub const A_SUPPORT: usize = 10;
pub const A_SPREAD: usize = 11;
pub const A_MARK: usize = 12;
pub const A_STAY: usize = 13;
// GET_OPEN: the POMDP decision to move wide and OPEN A PASSING LANE. The target
// is computed by a short-horizon MPC search (see mpc_open_lane_target).
pub const A_GET_OPEN: usize = 14;
pub const PASS_TARGET_SLOTS: usize = N - 2; // outfield teammates minus the possessor

// Full relational field vector (per-agent actor observation): 9 self/global +
// 5 ball + 5 goals + 6 role/cues + (N-1)*5 teammates + N*5 opponents + 1 bias.
pub const OBS_DIM: usize = 71;

// Centralized-critic GLOBAL state (MAPPO / CTDE): the whole field in a single
// canonical (Team-A attack) frame — every player's pos+vel + ball pos+vel +
// possession, shared by all agents. 2N*4 players + 4 ball + 3 possession + 1 bias.
pub const GLOBAL_DIM: usize = 2 * N * 4 + 4 + 3 + 1; // = 40 + 4 + 3 + 1 = 48

#[derive(Clone, Copy, Default, Debug)]
pub struct V2 {
    pub x: f32,
    pub y: f32,
}
impl V2 {
    pub fn new(x: f32, y: f32) -> Self {
        V2 { x, y }
    }
    pub fn add(self, o: V2) -> V2 {
        V2::new(self.x + o.x, self.y + o.y)
    }
    pub fn sub(self, o: V2) -> V2 {
        V2::new(self.x - o.x, self.y - o.y)
    }
    pub fn scale(self, s: f32) -> V2 {
        V2::new(self.x * s, self.y * s)
    }
    pub fn len(self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
    pub fn unit(self) -> V2 {
        let l = self.len();
        if l < 1e-6 {
            V2::new(0.0, 0.0)
        } else {
            V2::new(self.x / l, self.y / l)
        }
    }
}

#[derive(Clone, Copy)]
pub struct Player {
    pub pos: V2,
    pub vel: V2,
}

#[derive(Clone, Copy, PartialEq)]
pub enum Team {
    A,
    B,
}
impl Team {
    fn other(self) -> Team {
        match self {
            Team::A => Team::B,
            Team::B => Team::A,
        }
    }
    /// Attack direction on the x-axis: A attacks +x (goal at FIELD_L), B attacks -x.
    fn sx(self) -> f32 {
        match self {
            Team::A => 1.0,
            Team::B => -1.0,
        }
    }
    /// Center of the goal this team attacks.
    fn target_goal(self) -> V2 {
        match self {
            Team::A => V2::new(FIELD_L, FIELD_W / 2.0),
            Team::B => V2::new(0.0, FIELD_W / 2.0),
        }
    }
    fn own_goal(self) -> V2 {
        self.other().target_goal()
    }
}

#[derive(Clone, Copy)]
pub struct Owner {
    pub team: Team,
    pub idx: usize,
}

pub struct World {
    pub a: [Player; N],
    pub b: [Player; N],
    pub ball: V2,
    pub ball_vel: V2,
    pub ball_aerial: bool,  // in-flight ball is a lofted/scooped pass (over ground defenders)
    pub air_ticks: u32,     // ticks the scooped ball stays airborne before it lands
    pub ball_curl: V2,      // lateral curl (spin) accel on a long (>20yd) pass/shot
    pending_curl: V2,       // scratch: curl for the kick launching this tick
    pending_aerial: bool,   // scratch: the pass launched this tick is a scoop
    pending_air_ticks: u32, // scratch: airborne duration for the launching scoop
    pub owner: Option<Owner>,
    pub last_touch: Option<Team>,
    last_kicker: Option<Owner>,
    kick_timer: i32,
    pub goals_a: u32,
    pub goals_b: u32,
    // event flags consumed by the reward function each tick:
    pub ev_goal_a: bool,
    pub ev_goal_b: bool,
    pub ev_pass_completed_a: bool,
    pub ev_turnover_a: bool,          // Team A lost possession to B
    pub ev_bad_pass_turnover_a: bool, // Team A pass was intercepted or captured by B
    pub ev_dribble_turnover_a: bool,  // Team A carrier was tackled while dribbling/carrying
    pub ev_shot_on_a: bool,           // Team A took a shot on target
    pub ev_win_ball_a: bool,          // Team A won possession off Team B (interception/tackle)
    pub ev_pass_attempt_a: bool,      // Team A attempted a pass this tick
    pub pass_dir_a: i32,              // direction of that pass: 1 forward, 0 lateral, -1 backward
    pub ev_shot_attempt_a: bool,      // Team A took a shot this tick
    pub last_shot_quality_a: f32,     // placement quality of A's last shot, ~[0,1] (MPC finish)
    pub last_shot_xg_a: f32,          // POSITION quality (xG-like: distance+angle to goal), ~[0,1]
    pending_pass: Option<Owner>,      // intended receiver of an in-flight pass
    intended_receiver: Option<Owner>, // scratch set during apply_on_ball
    pass_kick_x: f32,                 // ball x when Team A last released a pass
    pub last_pass_gain_a: f32,        // forward metres gained by the last completed A pass
    pub pass_streak_a: u32,           // completed A passes in the current possession (2-pass rule)
    pub fwd_pass_streak_a: u32, // consecutive completed FORWARD A passes (progressive build-up)
    pub shoot_cooldown_a: u32,  // ticks until an A shot counts as a "fresh" chance again
    pub shot_was_rapid_a: bool, // this tick's A shot came while the cooldown was still active
    a_shot_flag: bool,          // the current free ball came from a Team-A shot (gates goals)
    lp_from: i32,               // passer index of the last completed A pass
    lp_to: i32,                 // receiver index of the last completed A pass
    pending_passer: i32,        // passer index of the in-flight A pass
    pub return_streak_a: u32,   // consecutive "pass back to the giver" (ping-pong) count
    pub ev_return_pass_a: bool, // this tick's A pass went back to the giver (ping-pong)
    pub return_start_x: f32,    // ball x when the current return sequence began
    pub ev_dribble_fwd_a: bool, // A carrier dribbled forward this tick
    pub ev_dribble_lat_a: bool, // A carrier dribbled laterally this tick
    // Symmetric goal rules for Team B (so the opponent must build up too):
    b_pass_streak: u32, // completed B passes in the current possession
    b_shot_flag: bool,  // free ball came from a VALID B shot (2 passes + final third)
}

fn players(team: Team, w: &World) -> &[Player; N] {
    match team {
        Team::A => &w.a,
        Team::B => &w.b,
    }
}

impl World {
    pub fn new() -> Self {
        let mut w = World {
            a: [Player {
                pos: V2::default(),
                vel: V2::default(),
            }; N],
            b: [Player {
                pos: V2::default(),
                vel: V2::default(),
            }; N],
            ball: V2::new(FIELD_L / 2.0, FIELD_W / 2.0),
            ball_vel: V2::default(),
            ball_aerial: false,
            air_ticks: 0,
            ball_curl: V2::default(),
            pending_curl: V2::default(),
            pending_aerial: false,
            pending_air_ticks: 0,
            owner: None,
            last_touch: None,
            last_kicker: None,
            kick_timer: 0,
            goals_a: 0,
            goals_b: 0,
            ev_goal_a: false,
            ev_goal_b: false,
            ev_pass_completed_a: false,
            ev_turnover_a: false,
            ev_bad_pass_turnover_a: false,
            ev_dribble_turnover_a: false,
            ev_shot_on_a: false,
            ev_win_ball_a: false,
            ev_pass_attempt_a: false,
            pass_dir_a: 0,
            ev_shot_attempt_a: false,
            last_shot_quality_a: 0.0,
            last_shot_xg_a: 0.0,
            pending_pass: None,
            intended_receiver: None,
            pass_kick_x: 0.0,
            last_pass_gain_a: 0.0,
            pass_streak_a: 0,
            fwd_pass_streak_a: 0,
            shoot_cooldown_a: 0,
            shot_was_rapid_a: false,
            a_shot_flag: false,
            lp_from: -1,
            lp_to: -1,
            pending_passer: -1,
            return_streak_a: 0,
            ev_return_pass_a: false,
            return_start_x: 0.0,
            ev_dribble_fwd_a: false,
            ev_dribble_lat_a: false,
            b_pass_streak: 0,
            b_shot_flag: false,
        };
        w.kickoff(Team::A);
        w
    }

    /// Place both teams in a simple spread formation; give the ball to `to`.
    pub fn kickoff(&mut self, to: Team) {
        // Formation columns (as fraction of field length from own goal) and y bands.
        let ys = [0.5, 0.22, 0.78, 0.35, 0.65];
        let xs = [0.08, 0.30, 0.30, 0.55, 0.55];
        for i in 0..N {
            self.a[i].pos = V2::new(xs[i] * FIELD_L, ys[i] * FIELD_W);
            self.a[i].vel = V2::default();
            // Mirror for B (attacks -x): x' = L - x.
            self.b[i].pos = V2::new(FIELD_L - xs[i] * FIELD_L, ys[i] * FIELD_W);
            self.b[i].vel = V2::default();
        }
        self.ball = V2::new(FIELD_L / 2.0, FIELD_W / 2.0);
        self.ball_vel = V2::default();
        // Nearest attacker of `to` takes the ball at center.
        let idx = self.nearest_player(to, self.ball).0;
        self.owner = Some(Owner { team: to, idx });
        self.last_touch = Some(to);
        self.last_kicker = None;
        self.kick_timer = 0;
        self.pending_pass = None;
        self.pass_streak_a = 0;
        self.a_shot_flag = false;
        self.b_pass_streak = 0;
        self.b_shot_flag = false;
        self.reset_a_pass_memory();
    }

    fn reset_a_pass_memory(&mut self) {
        self.lp_from = -1;
        self.lp_to = -1;
        self.pending_passer = -1;
        self.return_streak_a = 0;
        self.return_start_x = 0.0;
        self.fwd_pass_streak_a = 0;
    }

    fn player(&self, o: Owner) -> Player {
        match o.team {
            Team::A => self.a[o.idx],
            Team::B => self.b[o.idx],
        }
    }
    fn nearest_player(&self, team: Team, p: V2) -> (usize, f32) {
        let ps = players(team, self);
        let mut bi = 0;
        let mut bd = f32::INFINITY;
        for i in 0..N {
            let d = ps[i].pos.sub(p).len();
            if d < bd {
                bd = d;
                bi = i;
            }
        }
        (bi, bd)
    }

    fn nearest_opponent(&self, team: Team, p: V2) -> (usize, f32) {
        self.nearest_player(team.other(), p)
    }

    pub fn nearest_opponent_distance(&self, team: Team, p: V2) -> (usize, f32) {
        self.nearest_opponent(team, p)
    }

    // -- pass-target ranking: pick every outfield teammate for the possessor ---
    // Score favors forward progress (toward attacked goal) and openness.
    /// Curl (lateral accel) for a LONG (>20yd) pass/shot — bends the ball around a
    /// defender in the lane. Field-vector driven: curls AWAY from the nearest
    /// opponent to the lane midpoint. Zero for short balls; more on longer ones.
    fn kick_curl(&self, team: Team, from: V2, to: V2) -> V2 {
        let seg = to.sub(from);
        let dist = seg.len();
        if dist < CURL_MIN_DIST {
            return V2::default();
        }
        let dir = seg.unit();
        let perp = V2::new(-dir.y, dir.x);
        let mid = from.add(seg.scale(0.5));
        let (oi, _) = self.nearest_opponent(team, mid);
        let opp = players(team.other(), self)[oi].pos;
        let rel = opp.sub(mid);
        let side = if rel.x * perp.x + rel.y * perp.y > 0.0 { -1.0 } else { 1.0 };
        let strength = CURL_ACCEL * ((dist - CURL_MIN_DIST) / 20.0).clamp(0.0, 1.0);
        perp.scale(side * strength)
    }

    fn pass_candidates(
        &self,
        team: Team,
        from_idx: usize,
    ) -> [Option<(usize, f32)>; PASS_TARGET_SLOTS] {
        let ps = players(team, self);
        let from = ps[from_idx].pos;
        let sx = team.sx();
        let mut scored: Vec<(usize, f32)> = Vec::new();
        for i in 0..N {
            if i == from_idx || i == GK {
                continue; // don't pass to self or (as an outlet) the keeper
            }
            let tp = ps[i].pos;
            let fwd = (tp.x - from.x) * sx; // positive = ahead
            let (_, opp_d) = self.nearest_opponent(team, tp); // openness
            let dist = tp.sub(from).len();
            if dist < 2.0 {
                continue;
            }
            // reward forward + open, penalize very long/backward balls
            let mut score = fwd * 0.6 + opp_d.min(15.0) * 0.8 - (dist * 0.05).max(0.0);
            if from_idx as i32 == self.lp_to && i as i32 == self.lp_from {
                // Structural anti-ping-pong: keep the return ball legal, but make
                // the ranker prefer a third-player outlet unless the return is
                // genuinely the only safe option.
                score -= 8.0;
                if fwd < 3.0 {
                    score -= 6.0;
                }
            }
            scored.push((i, score));
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut out = [None; PASS_TARGET_SLOTS];
        for (slot, cand) in scored.into_iter().take(PASS_TARGET_SLOTS).enumerate() {
            out[slot] = Some(cand);
        }
        out
    }

    /// Where a defender at `from` should move to INTERCEPT the ball: simulate the
    /// ball's future trajectory (with friction) and return the earliest point the
    /// defender can physically reach — anticipation, so passes get cut out. For a
    /// slow/owned ball it just returns the ball (go challenge the carrier).
    pub fn intercept_point(&self, from: V2) -> V2 {
        if self.ball_vel.len() < 3.0 {
            return self.ball;
        }
        let mut bpos = self.ball;
        let mut bvel = self.ball_vel;
        for step in 1..40 {
            bpos = bpos.add(bvel.scale(DT));
            bvel = bvel.scale(BALL_FRICTION);
            let reach = PLAYER_SPEED * (step as f32 * DT) + CONTROL_RADIUS;
            if from.sub(bpos).len() <= reach {
                return bpos; // can meet the ball here
            }
        }
        bpos // uncatchable — head to where it ends up
    }

    /// Classify a Team-A dribble by the FINAL (post-shielding) direction, so the
    /// forward-dribble reward reflects actual movement — a shield that bends the
    /// carry sideways/backward is NOT credited as forward. Forward = x·sx > 0.3,
    /// lateral = within ±0.3, backward = < −0.3 (no event → no reward).
    fn note_dribble(&mut self, team: Team, dir: V2, sx: f32) {
        if team != Team::A {
            return;
        }
        let fwd = dir.x * sx;
        if fwd > 0.3 {
            self.ev_dribble_fwd_a = true;
        } else if fwd > -0.3 {
            self.ev_dribble_lat_a = true;
        }
    }

    /// Dribble direction that veers AWAY from a close defender (shielding) while
    /// keeping the intended direction when there's space.
    fn shielded_dribble_dir(&self, team: Team, me: V2, intended: V2) -> V2 {
        let intended = intended.unit();
        let (oi, od) = self.nearest_opponent(team, me);
        if od < 4.0 {
            let away = me.sub(players(team.other(), self)[oi].pos).unit();
            let blend = ((4.0 - od) / 4.0).clamp(0.0, 1.0); // 1 when defender is on top
            let d = intended.scale(1.0 - 0.7 * blend).add(away.scale(blend));
            d.unit()
        } else {
            intended
        }
    }

    /// Fraction of the lane from `from` to `to` clear of outfield opponents
    /// (0..1, 1 = fully clear). The keeper is excluded — it is scored separately
    /// in finishing. Used to define a truly OPEN receiver (clear passing lane).
    pub fn lane_clearness(&self, team: Team, from: V2, to: V2) -> f32 {
        let dir = to.sub(from);
        let dist = dir.len();
        if dist < 1e-3 {
            return 0.0;
        }
        let u = dir.unit();
        let opp = players(team.other(), self);
        let mut min_clear = 1.0f32;
        for i in 0..N {
            if i == GK {
                continue;
            }
            let rel = opp[i].pos.sub(from);
            let t = rel.x * u.x + rel.y * u.y;
            if t <= 0.5 || t >= dist {
                continue;
            }
            let perp = (rel.x * (-u.y) + rel.y * u.x).abs();
            let clear = (perp / 3.0).min(1.0);
            if clear < min_clear {
                min_clear = clear;
            }
        }
        min_clear
    }

    /// Fraction of the shot lane to goal that is clear of opponents (0..1).
    fn shot_clearness(&self, team: Team, from: V2) -> f32 {
        let goal = team.target_goal();
        let dir = goal.sub(from);
        let dist = dir.len();
        if dist < 1e-3 {
            return 0.0;
        }
        let u = dir.unit();
        let opp = players(team.other(), self);
        let mut min_clear = 1.0f32;
        for i in 0..N {
            let rel = opp[i].pos.sub(from);
            let t = rel.x * u.x + rel.y * u.y; // projection along shot
            if t <= 0.5 || t >= dist {
                continue;
            }
            // perpendicular distance from lane
            let perp = (rel.x * (-u.y) + rel.y * u.x).abs();
            let clear = (perp / 3.0).min(1.0); // within 3m blocks the lane
            if clear < min_clear {
                min_clear = clear;
            }
        }
        min_clear
    }

    // ---------------------------------------------------------------------
    // Observation for one player, in that player's *attack frame* (x flipped
    // for Team B) so a single shared policy is symmetric across sides.
    // ---------------------------------------------------------------------
    // FULL RELATIONAL FIELD VECTOR: every player observes the WHOLE field —
    // self, ball, both goals, plus ALL teammates and ALL opponents (position,
    // velocity, distance), sorted by proximity, in its own attack frame. This is
    // the 5-a-side analogue of the 22-man engine's field vector: the policy can
    // reason about overall shape (spacing, support, pressure), not just nearest
    // neighbours. Proving it's learnable here is the point.
    pub fn observe(&self, team: Team, idx: usize) -> [f32; OBS_DIM] {
        let sx = team.sx();
        let me = players(team, self)[idx];
        let mir = |p: V2| -> V2 { V2::new(if sx > 0.0 { p.x } else { FIELD_L - p.x }, p.y) };
        let mirv = |v: V2| -> V2 { V2::new(v.x * sx, v.y) };
        let nx = FIELD_L;
        let ny = FIELD_W;
        let nv = 10.0;

        let mp = mir(me.pos);
        let mvel = mirv(me.vel);
        let bp = mir(self.ball);
        let bvel = mirv(self.ball_vel);
        let goal = mir(team.target_goal());
        let own = mir(team.own_goal());
        let ball_rel = bp.sub(mp);

        let has_ball = matches!(self.owner, Some(o) if o.team == team && o.idx == idx);
        let team_ball = matches!(self.owner, Some(o) if o.team == team);
        let opp_ball = matches!(self.owner, Some(o) if o.team != team);
        let free_ball = self.owner.is_none();

        // derived cues kept for action semantics: shot lane + best-pass openness
        let shot_clear = self.shot_clearness(team, me.pos);
        let cands = self.pass_candidates(team, idx);
        let open_of =
            |c: Option<(usize, f32)>| c.map(|(_, s)| (s / 15.0).clamp(-1.0, 1.0)).unwrap_or(-1.0);
        let (aopen, bopen, copen) = (open_of(cands[0]), open_of(cands[1]), open_of(cands[2]));

        // role: is this the closest outfielder to the ball, and its ball-distance rank
        let my_ball_d = me.pos.sub(self.ball).len();
        let mut closer = 0usize;
        for k in 1..N {
            if k != idx && players(team, self)[k].pos.sub(self.ball).len() < my_ball_d {
                closer += 1;
            }
        }
        let is_closest = if closer == 0 { 1.0 } else { 0.0 };
        let ball_rank = closer as f32 / (N as f32 - 2.0).max(1.0);

        // teammates (all others incl. GK) and opponents, each sorted by distance to me
        let mut tm: Vec<usize> = (0..N).filter(|&k| k != idx).collect();
        tm.sort_by(|&a, &b| {
            players(team, self)[a]
                .pos
                .sub(me.pos)
                .len()
                .partial_cmp(&players(team, self)[b].pos.sub(me.pos).len())
                .unwrap()
        });
        let opp = players(team.other(), self);
        let mut op: Vec<usize> = (0..N).collect();
        op.sort_by(|&a, &b| {
            opp[a]
                .pos
                .sub(me.pos)
                .len()
                .partial_cmp(&opp[b].pos.sub(me.pos).len())
                .unwrap()
        });

        let mut f: Vec<f32> = Vec::with_capacity(OBS_DIM);
        // self / global (9)
        f.push(has_ball as u8 as f32);
        f.push(team_ball as u8 as f32);
        f.push(opp_ball as u8 as f32);
        f.push(free_ball as u8 as f32);
        f.push(mp.x / nx * 2.0 - 1.0);
        f.push(mp.y / ny * 2.0 - 1.0);
        f.push(mvel.x / nv);
        f.push(mvel.y / nv);
        // 2-pass rule state: how many passes made (0, 0.5, 1.0 = can shoot)
        f.push((self.pass_streak_a.min(2) as f32) / 2.0);
        // ball (5)
        f.push(ball_rel.x / nx);
        f.push(ball_rel.y / ny);
        f.push(ball_rel.len() / nx);
        f.push(bvel.x / nv);
        f.push(bvel.y / nv);
        // goals (5)
        f.push((goal.x - mp.x) / nx);
        f.push((goal.y - mp.y) / ny);
        f.push(goal.sub(mp).len() / nx);
        f.push((own.x - mp.x) / nx);
        f.push((own.y - mp.y) / ny);
        // role + action cues (6)
        f.push(is_closest);
        f.push(ball_rank);
        f.push(shot_clear);
        f.push(aopen);
        f.push(bopen);
        f.push(copen);
        // ALL teammates, nearest-first (N-1 = 4 × 5 = 20)
        for &k in &tm {
            let p = players(team, self)[k];
            let rp = mir(p.pos).sub(mp);
            let rv = mirv(p.vel);
            f.push(rp.x / nx);
            f.push(rp.y / ny);
            f.push(rv.x / nv);
            f.push(rv.y / nv);
            f.push(rp.len() / nx);
        }
        // ALL opponents, nearest-first (N = 5 × 5 = 25)
        for &k in &op {
            let p = opp[k];
            let rp = mir(p.pos).sub(mp);
            let rv = mirv(p.vel);
            f.push(rp.x / nx);
            f.push(rp.y / ny);
            f.push(rv.x / nv);
            f.push(rv.y / nv);
            f.push(rp.len() / nx);
        }
        f.push(1.0); // bias
        debug_assert_eq!(f.len(), OBS_DIM);
        f.try_into().unwrap()
    }

    /// Legal-action mask for one player (on-ball vs off-ball families).
    pub fn legal_mask(&self, team: Team, idx: usize) -> [bool; NA] {
        let has_ball = matches!(self.owner, Some(o) if o.team == team && o.idx == idx);
        let team_owns = matches!(self.owner, Some(o) if o.team == team);
        let opp_owns = matches!(self.owner, Some(o) if o.team != team);
        let our_phase = team_owns || (self.owner.is_none() && self.last_touch == Some(team));
        let their_phase =
            opp_owns || (self.owner.is_none() && self.last_touch == Some(team.other()));
        let mut m = [false; NA];
        if has_ball {
            for a in A_SHOOT..=A_HOLD {
                m[a] = true;
            }
            let cands = self.pass_candidates(team, idx);
            // Later pass-target actions are only legal when those ranked outlets exist.
            if cands[1].is_none() {
                m[A_PASS_B] = false;
            }
            if cands[2].is_none() {
                m[A_PASS_C] = false;
            }
            // 2-PASS RULE (Team A): no shooting until 2 completed passes this
            // possession — forces build-up play, not solo dribble-and-shoot.
            // OPPONENT-HALF RULE (Team A): may shoot anywhere in the opponent's half.
            if team == Team::A {
                let x = players(team, self)[idx].pos.x;
                if self.pass_streak_a < 2 || x < SHOOT_X {
                    m[A_SHOOT] = false;
                }
            } else {
                // Symmetric shooting gate for the scripted/noisy opponent.
                let x = players(team, self)[idx].pos.x;
                if self.b_pass_streak < 2 || x > FIELD_L - SHOOT_X {
                    m[A_SHOOT] = false;
                }
            }
        } else {
            if our_phase {
                // In possession, off-ball players should either burst into support
                // lanes/get-open targets or hold rest-defense shape. Chasing the
                // ball carrier is not a useful attacking decision.
                m[A_SUPPORT] = true;
                m[A_SPREAD] = true;
                m[A_MARK] = true;
                m[A_STAY] = true;
                m[A_GET_OPEN] = true;
                if self.owner.is_none() {
                    m[A_CHASE] = true;
                }
            } else if their_phase {
                // In dispossession, remove attacking-support actions from the
                // legal set; defenders choose between pressing and recovering
                // goalside.
                m[A_CHASE] = true;
                m[A_MARK] = true;
                m[A_STAY] = true;
            } else {
                for a in A_CHASE..=A_STAY {
                    m[a] = true;
                }
            }
        }
        m
    }

    pub fn speed_mask(&self, team: Team, idx: usize, action: usize) -> [bool; NS] {
        let mut mask = [true; NS];
        let only = |mask: &mut [bool; NS], gear: usize| {
            for (idx, ok) in mask.iter_mut().enumerate() {
                *ok = idx == gear;
            }
        };
        let min_gear = |mask: &mut [bool; NS], gear: usize| {
            for (idx, ok) in mask.iter_mut().enumerate() {
                *ok = idx >= gear;
            }
        };

        if idx == GK {
            only(&mut mask, SPD_STAND);
            return mask;
        }

        let is_owner = matches!(self.owner, Some(o) if o.team == team && o.idx == idx);
        let team_owns = matches!(self.owner, Some(o) if o.team == team);
        let opp_owns = matches!(self.owner, Some(o) if o.team != team);
        let our_phase = team_owns || (self.owner.is_none() && self.last_touch == Some(team));
        let their_phase =
            opp_owns || (self.owner.is_none() && self.last_touch == Some(team.other()));

        if matches!(
            action,
            A_SHOOT | A_PASS_A | A_PASS_B | A_PASS_C | A_CLEAR | A_HOLD | A_STAY
        ) {
            only(&mut mask, SPD_STAND);
        } else if is_owner && matches!(action, A_DRIB_FWD | A_DRIB_LEFT | A_DRIB_RIGHT) {
            min_gear(&mut mask, SPD_WALK);
        } else if !is_owner && our_phase && matches!(action, A_SUPPORT | A_GET_OPEN) {
            min_gear(&mut mask, SPD_RUN_FAST);
        } else if !is_owner && our_phase && action == A_SPREAD {
            min_gear(&mut mask, SPD_RUN_SLOW);
        } else if !is_owner && their_phase && matches!(action, A_CHASE | A_MARK) {
            min_gear(&mut mask, SPD_RUN_FAST);
        } else if !is_owner
            && matches!(action, A_CHASE | A_SUPPORT | A_SPREAD | A_MARK | A_GET_OPEN)
        {
            min_gear(&mut mask, SPD_RUN_SLOW);
        }
        mask
    }

    pub fn coerce_speed_gear(&self, team: Team, idx: usize, action: usize, gear: usize) -> usize {
        let mask = self.speed_mask(team, idx, action);
        let gear = gear.min(NS - 1);
        if mask[gear] {
            return gear;
        }
        mask.iter()
            .enumerate()
            .skip(gear)
            .find_map(|(idx, ok)| ok.then_some(idx))
            .or_else(|| mask.iter().position(|&ok| ok))
            .unwrap_or(SPD_STAND)
    }

    // ---------------------------------------------------------------------
    // Step: apply one action per player of both teams, advance physics.
    // `act_a` / `act_b` are indexed [player] -> action id.
    // ---------------------------------------------------------------------
    pub fn step(&mut self, act_a: &[usize; N], act_b: &[usize; N], rng: &mut Rng) {
        self.ev_goal_a = false;
        self.ev_goal_b = false;
        self.ev_pass_completed_a = false;
        self.ev_turnover_a = false;
        self.ev_bad_pass_turnover_a = false;
        self.ev_dribble_turnover_a = false;
        self.ev_shot_on_a = false;
        self.ev_win_ball_a = false;
        self.ev_pass_attempt_a = false;
        self.ev_shot_attempt_a = false;
        self.ev_return_pass_a = false;
        self.ev_dribble_fwd_a = false;
        self.ev_dribble_lat_a = false;
        self.shot_was_rapid_a = false;
        self.shoot_cooldown_a = self.shoot_cooldown_a.saturating_sub(1);

        let ball_x_before = self.ball.x;

        // 1. Desired velocities + kicks. Kicks are resolved after movement so a
        //    player dribbles then releases within the same tick cleanly.
        let mut kick: Option<(Owner, V2, f32, bool)> = None; // (kicker, dir, speed, is_pass)
        for team in [Team::A, Team::B] {
            let acts = if team == Team::A { act_a } else { act_b };
            for i in 0..N {
                // Goalkeeper is a fixed rule (both teams), never policy-driven.
                if i == GK {
                    if let Some(k) = self.apply_keeper(team) {
                        kick = Some(k);
                    }
                    continue;
                }
                // Actions are packed as `action + speed*NA` so the speed gear rides
                // along without changing the step signature. Unpack both here.
                let a = acts[i] % NA;
                let spd = self.coerce_speed_gear(team, i, a, (acts[i] / NA).min(NS - 1));
                let is_owner = matches!(self.owner, Some(o) if o.team == team && o.idx == i);
                if is_owner {
                    if let Some(k) = self.apply_on_ball(team, i, a, spd, rng) {
                        kick = Some(k);
                    }
                } else {
                    self.apply_off_ball(team, i, a, spd);
                }
            }
        }

        // 2. Integrate player motion + clamp to field.
        for i in 0..N {
            integrate(&mut self.a[i]);
            integrate(&mut self.b[i]);
        }

        // (No physics separation — spacing is left entirely to the learned
        // policy via the per-player spacing reward. The policy is given
        // nearest-teammate perception in its observation so it CAN learn it.)

        // 3. Resolve a kick (frees the ball) or carry it with the owner.
        if let Some((kicker, dir, speed, is_pass)) = kick {
            let ang = rng.normal(0.0, 0.04); // execution dither
            let d = rotate(dir.unit(), ang);
            self.ball_vel = d.scale(speed);
            let kp = self.player(kicker).pos;
            self.ball = kp.add(d.scale(1.0));
            self.owner = None;
            self.last_touch = Some(kicker.team);
            self.last_kicker = Some(kicker);
            self.kick_timer = 6;
            self.ball_aerial = is_pass && self.pending_aerial;
            self.air_ticks = if self.ball_aerial { self.pending_air_ticks } else { 0 };
            self.pending_aerial = false;
            self.ball_curl = self.pending_curl;
            self.pending_curl = V2::default();
            self.pending_pass = None;
            if is_pass {
                // remember intended receiver (BOTH teams) so the reception radius
                // applies symmetrically. pending_pass.team == the passing team.
                self.pending_pass = self.intended_receiver;
                if kicker.team == Team::A {
                    self.pending_passer = kicker.idx as i32;
                    // A attacks +x, so ball.x IS attack-frame forward progress.
                    self.pass_kick_x = self.player(kicker).pos.x;
                }
            } else if kicker.team == Team::A {
                self.reset_a_pass_memory();
            }
        } else if let Some(o) = self.owner {
            // carry: normally glued just ahead; but if a defender is close, SHIELD
            // the ball — hold it on the far side of the body from the defender.
            let p = self.player(o);
            let (oi, od) = self.nearest_opponent(o.team, p.pos);
            let offset = if od < 3.0 {
                p.pos
                    .sub(players(o.team.other(), self)[oi].pos)
                    .unit()
                    .scale(0.8)
            } else {
                V2::new(o.team.sx(), 0.0).scale(0.8)
            };
            self.ball = p.pos.add(offset);
            self.ball_vel = p.vel;
        }

        // 4. Advance a free ball + friction, walls, goals, capture.
        if self.owner.is_none() {
            self.ball = self.ball.add(self.ball_vel.scale(DT));
            if self.ball_vel.len() > 4.0 {
                self.ball_vel = self.ball_vel.add(self.ball_curl.scale(DT));
            }
            self.ball_vel = self.ball_vel.scale(BALL_FRICTION);
            if self.ball_aerial {
                self.air_ticks = self.air_ticks.saturating_sub(1);
                if self.air_ticks == 0 {
                    self.ball_aerial = false; // the scoop lands -> a normal ground ball
                }
            }

            // reflect off side-lines (y walls) to keep play flowing
            if self.ball.y < 0.0 {
                self.ball.y = -self.ball.y;
                self.ball_vel.y = -self.ball_vel.y;
            } else if self.ball.y > FIELD_W {
                self.ball.y = 2.0 * FIELD_W - self.ball.y;
                self.ball_vel.y = -self.ball_vel.y;
            }

            let gy0 = FIELD_W / 2.0 - GOAL_HALF;
            let gy1 = FIELD_W / 2.0 + GOAL_HALF;
            if self.ball.x >= FIELD_L {
                if self.ball.y > gy0 && self.ball.y < gy1 && self.a_shot_flag {
                    // valid goal: came from an A shot, which required 2 passes
                    self.goals_a += 1;
                    self.ev_goal_a = true;
                    self.kickoff(Team::B);
                } else {
                    self.goal_kick(Team::B); // B restarts from its own line
                }
            } else if self.ball.x <= 0.0 {
                if self.ball.y > gy0 && self.ball.y < gy1 && self.b_shot_flag {
                    // symmetric: B goal only counts from a valid (2-pass, final-third) shot
                    self.goals_b += 1;
                    self.ev_goal_b = true;
                    self.kickoff(Team::A);
                } else {
                    self.goal_kick(Team::A);
                }
            } else {
                self.try_capture();
            }
        } else {
            // 5. Owned ball: opponents can tackle if adjacent.
            self.try_tackle(rng);
        }

        let _ = ball_x_before; // reward layer reads positions directly
    }

    fn goal_kick(&mut self, to: Team) {
        let gx = if to == Team::A { 6.0 } else { FIELD_L - 6.0 };
        self.ball = V2::new(gx, FIELD_W / 2.0);
        self.ball_vel = V2::default();
        let idx = self.nearest_player(to, self.ball).0;
        self.owner = Some(Owner { team: to, idx });
        self.last_touch = Some(to);
        self.last_kicker = None;
        self.kick_timer = 0;
        self.pass_streak_a = 0;
        self.a_shot_flag = false;
        self.b_pass_streak = 0;
        self.b_shot_flag = false;
        self.reset_a_pass_memory();
    }

    fn try_capture(&mut self) {
        let ball_fast = self.ball_vel.len() > CAPTURE_MAX_BALL_SPEED;
        let mut best: Option<(Owner, f32)> = None;
        // Keepers first: they can smother/save even fast shots within reach.
        for team in [Team::A, Team::B] {
            if self.kick_timer > 0 {
                if let Some(lk) = self.last_kicker {
                    if lk.team == team && lk.idx == GK {
                        continue;
                    }
                }
            }
            let d = players(team, self)[GK].pos.sub(self.ball).len();
            if d < KEEPER_REACH && best.is_none_or(|(_, bd)| d < bd) {
                best = Some((Owner { team, idx: GK }, d));
            }
        }
        // Outfield players only control a ball that isn't screaming past them.
        // The intended pass receiver gets a larger reception radius and a small
        // anticipation edge, so passes actually CONNECT instead of going loose —
        // unless an opponent is genuinely closer and intercepts.
        if best.is_none() && !ball_fast {
            // A SCOOPED ball in the air is over ground players: only the intended
            // receiver can bring it down, and only if they are open enough to
            // control it (aerial touch needs space). Otherwise it flies on / lands.
            let aerial = self.ball_aerial && self.air_ticks > 0;
            let mut best_eff = f32::INFINITY;
            for team in [Team::A, Team::B] {
                for i in 0..N {
                    if i == GK {
                        continue;
                    }
                    if self.kick_timer > 0 {
                        if let Some(lk) = self.last_kicker {
                            if lk.team == team && lk.idx == i {
                                continue; // kicker can't recapture instantly
                            }
                        }
                    }
                    let is_recv =
                        matches!(self.pending_pass, Some(r) if r.team == team && r.idx == i);
                    if aerial {
                        if !is_recv {
                            continue; // ground players can't reach an airborne ball
                        }
                        let (_, recv_open) =
                            self.nearest_opponent(team, players(team, self)[i].pos);
                        if recv_open < AERIAL_CONTROL_SPACE {
                            continue; // receiver not open enough to control the scoop yet
                        }
                    }
                    let radius = if is_recv {
                        RECEIVE_RADIUS
                    } else {
                        CONTROL_RADIUS
                    };
                    let d = players(team, self)[i].pos.sub(self.ball).len();
                    if d >= radius {
                        continue;
                    }
                    let eff = d - if is_recv { 0.9 } else { 0.0 };
                    if eff < best_eff {
                        best_eff = eff;
                        best = Some((Owner { team, idx: i }, d));
                    }
                }
            }
        }
        if let Some((o, _)) = best {
            let prev_touch = self.last_touch;
            self.owner = Some(o);
            self.ball_aerial = false; // controlled -> no longer airborne
            self.air_ticks = 0;
            self.ball_curl = V2::default();
            self.a_shot_flag = false; // shot resolved into possession
            self.b_shot_flag = false;
            // Team-A reward events. pending_pass.team is the PASSING team.
            if let Some(pp) = self.pending_pass {
                if pp.team == Team::A {
                    if o.team == Team::A {
                        self.ev_pass_completed_a = true; // A pass reached an A player
                        self.last_pass_gain_a = self.ball.x - self.pass_kick_x;
                        self.pass_streak_a += 1; // toward the 2-pass rule
                                                 // consecutive FORWARD passes (progressive build-up)
                        if self.last_pass_gain_a > 3.0 {
                            self.fwd_pass_streak_a += 1;
                        } else {
                            self.fwd_pass_streak_a = 0;
                        }
                        // record the passer->receiver of this completed A pass
                        self.lp_from = self.pending_passer;
                        self.lp_to = o.idx as i32;
                        self.pending_passer = -1;
                    } else {
                        self.ev_turnover_a = true; // A pass intercepted by B
                        self.ev_bad_pass_turnover_a = true;
                        self.pass_streak_a = 0;
                        self.reset_a_pass_memory();
                        self.b_pass_streak = 1; // B started a possession by intercepting
                    }
                }
                if pp.team == Team::B {
                    if o.team == Team::B {
                        self.b_pass_streak += 1; // B built up (symmetric 2-pass rule)
                    } else {
                        // a B pass intercepted by A is a good steal
                        self.ev_win_ball_a = true;
                        self.pass_streak_a = 0; // fresh possession
                        self.reset_a_pass_memory();
                        self.b_pass_streak = 0;
                    }
                }
            } else if matches!(prev_touch, Some(Team::A)) && o.team == Team::B {
                self.ev_turnover_a = true;
                self.ev_bad_pass_turnover_a = self.pending_pass.is_some();
                self.pass_streak_a = 0;
                self.reset_a_pass_memory();
                self.b_pass_streak = 1; // B possession begins
            } else if matches!(prev_touch, Some(Team::B)) && o.team == Team::A {
                self.ev_win_ball_a = true; // won a loose ball off B
                self.pass_streak_a = 0;
                self.reset_a_pass_memory();
                self.b_pass_streak = 0;
            }
            self.last_touch = Some(o.team);
            self.pending_pass = None;
        }
        self.kick_timer -= 1;
    }

    fn try_tackle(&mut self, rng: &mut Rng) {
        // Immunity window right after possession changed hands, so the ball can't
        // instantly ping back and forth (the tackle/dispossession "black hole").
        if self.kick_timer > 0 {
            self.kick_timer -= 1;
            return;
        }
        let o = match self.owner {
            Some(o) => o,
            None => return,
        };
        let op = self.player(o).pos;
        let (oi, od) = self.nearest_opponent(o.team, op);
        if od < TACKLE_RADIUS && rng.f01() < TACKLE_PROB {
            let stealer = Owner {
                team: o.team.other(),
                idx: oi,
            };
            if o.team == Team::A {
                self.ev_turnover_a = true; // A dispossessed
                self.ev_dribble_turnover_a = true;
                self.b_pass_streak = 1; // B possession begins
            } else {
                self.ev_win_ball_a = true; // A tackled the ball off B
                self.b_pass_streak = 0;
            }
            self.b_shot_flag = false;
            self.pass_streak_a = 0; // possession changed hands
            self.reset_a_pass_memory();
            self.owner = Some(stealer);
            self.last_touch = Some(stealer.team);
            self.kick_timer = 4;
            self.last_kicker = Some(stealer);
            self.pending_pass = None;
        }
    }

    // scratch used across step to remember a pass's intended receiver
    // (set inside apply_on_ball, consumed in step()).
    // Stored as a field to avoid threading through return types.
    // ---------------------------------------------------------------------
    fn apply_on_ball(
        &mut self,
        team: Team,
        idx: usize,
        a: usize,
        spd: usize,
        _rng: &mut Rng,
    ) -> Option<(Owner, V2, f32, bool)> {
        let me = players(team, self)[idx].pos;
        let sx = team.sx();
        let goal = team.target_goal();
        let owner = Owner { team, idx };
        self.intended_receiver = None;
        // any on-ball action clears that team's shot flag; only A_SHOOT re-sets it.
        if team == Team::A {
            self.a_shot_flag = false;
        } else {
            self.b_shot_flag = false;
        }
        match a {
            A_SHOOT => {
                if team == Team::A {
                    self.ev_shot_attempt_a = true;
                    // Rapid-fire repeat? (a shot while a prior shot's cooldown is still
                    // active = shoot-spam farming, not a fresh chance). The genuine
                    // first shot of a buildup is NOT rapid and earns full reward.
                    self.shot_was_rapid_a = self.shoot_cooldown_a > 0;
                    self.shoot_cooldown_a = 45; // ~2.25 s at 20 Hz before a shot is "fresh" again
                    self.pass_streak_a = 0; // buildup consumed by the shot
                    self.a_shot_flag = true; // this free ball is a valid (2-pass) shot
                    self.reset_a_pass_memory();
                } else {
                    // symmetric: B's goal only counts if B built up (2 passes) and
                    // shoots from B's own final third (B attacks -x -> small x).
                    self.b_shot_flag = self.b_pass_streak >= 2 && me.x < FIELD_L - SHOOT_X;
                    self.b_pass_streak = 0;
                }
                self.set_vel(team, idx, V2::default());
                // MPC-lite finishing: enumerate aim points across the mouth and
                // pick the one that maximizes clearance from the keeper and any
                // defender in the shot lane — i.e. shoot to the open corner.
                let goal_x = goal.x;
                let cy = FIELD_W / 2.0;
                let gk = players(team.other(), self)[GK].pos;
                let margin = GOAL_HALF - 0.35;
                let mut best_y = cy;
                let mut best_score = f32::NEG_INFINITY;
                for k in 0..7 {
                    let y = cy - margin + 2.0 * margin * (k as f32 / 6.0);
                    let aim = V2::new(goal_x, y);
                    // keeper distance to the aim point (predict a small step of
                    // keeper travel toward the aim along its constrained line)
                    let keeper_gap = gk.sub(aim).len();
                    // penalize a defender sitting on the shot lane to this aim
                    let lane_clear = self.lane_clearness(team, me, aim);
                    let score = keeper_gap + 3.0 * lane_clear;
                    if score > best_score {
                        best_score = score;
                        best_y = y;
                    }
                }
                let aim = V2::new(goal_x, best_y);
                if team == Team::A {
                    // MPC-finish quality: how open the chosen corner is (keeper gap
                    // + clear lane), normalized ~[0,1]. Rewards placing the shot AND
                    // shooting from a position where a good placement exists.
                    let q = (best_score / 12.0).clamp(0.0, 1.0);
                    self.last_shot_quality_a = q;
                    // POSITION quality (xG-like), a function of the field vector at the
                    // shot: close + central is a high-value chance, a long-range or
                    // wide pot-shot is near-zero. Distance dominates (squared decay),
                    // shot ANGLE (central vs wide) modulates.
                    let d = goal.sub(me).len();
                    let lateral = (me.y - FIELD_W / 2.0).abs();
                    let dist_f = (1.0 - d / 26.0).clamp(0.0, 1.0);
                    let angle_f = (1.0 - lateral / (FIELD_W / 2.0)).clamp(0.0, 1.0);
                    self.last_shot_xg_a = dist_f * dist_f * (0.4 + 0.6 * angle_f);
                    if d < 24.0 {
                        self.ev_shot_on_a = true;
                    }
                }
                self.pending_curl = self.kick_curl(team, me, aim);
                Some((owner, aim.sub(me), SHOT_SPEED, false))
            }
            A_PASS_A | A_PASS_B | A_PASS_C => {
                let cands = self.pass_candidates(team, idx);
                let pick = cands[a - A_PASS_A];
                if let Some((ti, _)) = pick {
                    // lead the pass slightly ahead of the receiver's forward run
                    let tp = players(team, self)[ti].pos;
                    let lead = tp.add(V2::new(sx * 2.0, 0.0));
                    self.intended_receiver = Some(Owner { team, idx: ti });
                    // SCOOP / lofted pass: if a defender blocks the GROUND lane to the
                    // target, lift the ball OVER them (aerial) so the pass can still be
                    // made. Aerial control takes time, so it only completes if the
                    // receiver is open (checked at reception) — else it lands loose.
                    let ground_lane = self.lane_clearness(team, me, tp);
                    if ground_lane < 0.55 {
                        self.pending_aerial = true;
                        let flight = (tp.sub(me).len() / (PASS_SPEED * DT)).round() as u32;
                        self.pending_air_ticks = flight.clamp(3, 30);
                    }
                    self.set_vel(team, idx, V2::default());
                    if team == Team::A {
                        self.ev_pass_attempt_a = true;
                        // classify by forward progress toward the attacked goal
                        let fwd = (tp.x - me.x) * sx;
                        self.pass_dir_a = if fwd > 2.0 {
                            1
                        } else if fwd < -2.0 {
                            -1
                        } else {
                            0
                        };
                        // ping-pong detection: passing back to the teammate who
                        // just gave me the ball (I am lp_to, target is lp_from).
                        self.pending_passer = idx as i32;
                        let is_return = idx as i32 == self.lp_to && ti as i32 == self.lp_from;
                        if is_return {
                            if self.return_streak_a == 0 {
                                self.return_start_x = self.ball.x; // sequence begins here
                            }
                            self.return_streak_a += 1;
                        } else {
                            self.return_streak_a = 0;
                        }
                        self.ev_return_pass_a = is_return;
                    }
                    self.pending_curl = self.kick_curl(team, me, tp);
                    let pass_dist = tp.sub(me).len();
                    let pspeed =
                        PASS_SPEED * (0.85 + 0.35 * (pass_dist / FIELD_L).clamp(0.0, 1.0));
                    Some((owner, lead.sub(me), pspeed, true))
                } else {
                    // no valid target: dribble forward instead
                    self.set_vel(team, idx, V2::new(sx, 0.0).scale(speed_val(spd, true)));
                    None
                }
            }
            A_DRIB_FWD => {
                let dir = self.shielded_dribble_dir(team, me, V2::new(sx, 0.0));
                self.note_dribble(team, dir, sx);
                self.set_vel(team, idx, dir.scale(speed_val(spd, true)));
                None
            }
            A_DRIB_LEFT => {
                let dir = self.shielded_dribble_dir(team, me, V2::new(0.0, -1.0));
                self.note_dribble(team, dir, sx);
                self.set_vel(team, idx, dir.scale(speed_val(spd, true)));
                None
            }
            A_DRIB_RIGHT => {
                let dir = self.shielded_dribble_dir(team, me, V2::new(0.0, 1.0));
                self.note_dribble(team, dir, sx);
                self.set_vel(team, idx, dir.scale(speed_val(spd, true)));
                None
            }
            A_CLEAR => {
                if team == Team::A {
                    self.pass_streak_a = 0; // clearing gives up the buildup
                    self.reset_a_pass_memory();
                }
                self.set_vel(team, idx, V2::default());
                // hoof toward attacked goal with lateral scatter
                let dir = goal.sub(me);
                Some((owner, dir, CLEAR_SPEED, false))
            }
            _ => {
                // HOLD: shield — drift gently away from nearest opponent.
                let (oi, _) = self.nearest_opponent(team, me);
                let away = me.sub(players(team.other(), self)[oi].pos).unit();
                self.set_vel(team, idx, away.scale(PLAYER_SPEED * 0.3));
                None
            }
        }
    }

    /// Goalkeeper controller (rule-based, both teams). Off the ball it stays on
    /// the ball→goal line inside its box, narrowing the angle; on the ball it
    /// distributes to the best outfield outlet (or clears under pressure).
    fn apply_keeper(&mut self, team: Team) -> Option<(Owner, V2, f32, bool)> {
        let me = players(team, self)[GK].pos;
        let sx = team.sx();
        let owns = matches!(self.owner, Some(o) if o.team == team && o.idx == GK);
        if owns {
            self.intended_receiver = None;
            let cands = self.pass_candidates(team, GK);
            if let Some((ti, _)) = cands[0] {
                let tp = players(team, self)[ti].pos;
                let lead = tp.add(V2::new(sx * 2.0, 0.0));
                self.intended_receiver = Some(Owner { team, idx: ti });
                self.set_vel(team, GK, V2::default());
                if team == Team::A {
                    self.ev_pass_attempt_a = true;
                    let fwd = (tp.x - me.x) * sx;
                    self.pass_dir_a = if fwd > 2.0 {
                        1
                    } else if fwd < -2.0 {
                        -1
                    } else {
                        0
                    };
                }
                return Some((Owner { team, idx: GK }, lead.sub(me), PASS_SPEED, true));
            }
            self.set_vel(team, GK, V2::default());
            return Some((
                Owner { team, idx: GK },
                team.target_goal().sub(me),
                CLEAR_SPEED,
                false,
            ));
        }
        // position on the ball→own-goal line, a few metres off the line
        let c = team.own_goal();
        let to_ball = self.ball.sub(c);
        let out = (to_ball.len() * 0.35).min(6.0);
        let mut target = c.add(to_ball.unit().scale(out));
        target.x = if sx > 0.0 {
            target.x.clamp(0.5, 8.0)
        } else {
            target.x.clamp(FIELD_L - 8.0, FIELD_L - 0.5)
        };
        target.y = target.y.clamp(
            FIELD_W / 2.0 - GOAL_HALF - 1.5,
            FIELD_W / 2.0 + GOAL_HALF + 1.5,
        );
        let dir = target.sub(me);
        // The keeper is rule-based, but its SPEED now uses the 7 gears as a
        // FIELD-VECTOR function: sprint when the ball threatens our goal or it must
        // reposition fast, jog at midfield, walk when play is upfield.
        let own_goal_x = if sx > 0.0 { 0.0 } else { FIELD_L };
        let ball_threat = 1.0 - ((self.ball.x - own_goal_x).abs() / FIELD_L).clamp(0.0, 1.0);
        let travel = (dir.len() / 9.0).clamp(0.0, 1.0);
        let urgency = (0.6 * ball_threat + 0.4 * travel).clamp(0.0, 1.0);
        let gear = if urgency > 0.80 {
            SPD_SPRINT
        } else if urgency > 0.62 {
            SPD_RUN_FAST
        } else if urgency > 0.46 {
            SPD_RUN_SLOW
        } else if urgency > 0.30 {
            SPD_SKIP
        } else if urgency > 0.16 {
            SPD_JOG
        } else {
            SPD_WALK
        };
        let v = if dir.len() < 0.2 {
            V2::default()
        } else {
            dir.unit().scale(speed_val(gear, false))
        };
        self.set_vel(team, GK, v);
        None
    }

    fn apply_off_ball(&mut self, team: Team, idx: usize, a: usize, spd: usize) {
        let me = players(team, self)[idx].pos;
        let team_owns = matches!(self.owner, Some(o) if o.team == team);
        let spd = self.coerce_speed_gear(team, idx, a, spd);
        let target = match a {
            // CHASE: anticipate. If the ball is moving (a pass/loose ball), aim at
            // the INTERCEPT point on its trajectory — where the defender can meet
            // it — rather than its current spot; otherwise go to the carrier.
            A_CHASE => self.intercept_point(me),
            // SUPPORT: push UPFIELD into open space (attacking run).
            A_SUPPORT => self.open_space_target(team, idx, 1.2),
            // SPREAD: find open space; drift up/down field per possession.
            A_SPREAD => {
                let bias = if team_owns { 0.4 } else { -0.4 };
                self.open_space_target(team, idx, bias)
            }
            // MARK: drop DOWNFIELD, goal-side of the nearest opponent (defend).
            A_MARK => {
                if team_owns {
                    let (oi, _) = self.nearest_opponent(team, me);
                    let opp = players(team.other(), self)[oi].pos;
                    let own_goal = team.own_goal();
                    opp.add(own_goal.sub(opp).unit().scale(2.5))
                } else {
                    self.goalside_recovery_target(team, idx)
                }
            }
            // GET_OPEN: MPC-lite lane opener — move to the relocation that best
            // opens a passing lane from the ball (the POMDP picks it, MPC aims it).
            A_GET_OPEN => self.mpc_open_lane_target(team, idx),
            _ => me, // STAY
        };
        let dir = target.sub(me);
        // STAY means stand still; otherwise move toward the target at the chosen gear.
        let v = if a == A_STAY || dir.len() < 0.3 || spd == SPD_STAND {
            V2::default()
        } else {
            dir.unit().scale(speed_val(spd, false))
        };
        self.set_vel(team, idx, v);
    }

    fn goalside_recovery_target(&self, team: Team, idx: usize) -> V2 {
        let me = players(team, self)[idx].pos;
        let own_goal = team.own_goal();
        let ball_goal_side = self.ball.add(own_goal.sub(self.ball).unit().scale(4.0));
        let (oi, _) = self.nearest_opponent(team, me);
        let opp = players(team.other(), self)[oi].pos;
        let opp_goal_side = opp.add(own_goal.sub(opp).unit().scale(3.0));
        let urgency = ((self.ball.x - me.x) * team.sx()).max(0.0) / FIELD_L;
        let blend = urgency.clamp(0.25, 0.75);
        V2::new(
            (ball_goal_side.x * blend + opp_goal_side.x * (1.0 - blend)).clamp(2.0, FIELD_L - 2.0),
            (ball_goal_side.y * blend + opp_goal_side.y * (1.0 - blend)).clamp(2.0, FIELD_W - 2.0),
        )
    }

    /// "Move where there is space." Samples a ring of candidate points around
    /// the player and returns the one that maximizes distance to every other
    /// player, plus an up/down-field bias (positive = upfield toward the
    /// attacked goal, negative = downfield to defend). No formation, no shape
    /// LP — purely possession- and space-driven, as a real 5-a-side plays.
    fn open_space_target(&self, team: Team, idx: usize, field_bias: f32) -> V2 {
        let me = players(team, self)[idx].pos;
        let sx = team.sx();
        let radii = [7.0f32, 14.0, 21.0];
        let ndir = 8usize;
        let mut best = me;
        let mut best_score = f32::NEG_INFINITY;
        for &r in &radii {
            for k in 0..ndir {
                let ang = (k as f32) / (ndir as f32) * std::f32::consts::TAU;
                let cand = V2::new(
                    (me.x + ang.cos() * r).clamp(3.0, FIELD_L - 3.0),
                    (me.y + ang.sin() * r).clamp(3.0, FIELD_W - 3.0),
                );
                // openness = distance to nearest OTHER player (either team)
                let mut min_d = f32::INFINITY;
                for t in [Team::A, Team::B] {
                    for j in 0..N {
                        if t == team && j == idx {
                            continue;
                        }
                        let d = players(t, self)[j].pos.sub(cand).len();
                        if d < min_d {
                            min_d = d;
                        }
                    }
                }
                let fwd = (cand.x - me.x) * sx; // upfield progress in attack frame
                                                // stay loosely connected to the ball's vertical lane
                let lane = -(cand.y - self.ball.y).abs() * 0.08;
                let score = min_d + field_bias * fwd + lane;
                if score > best_score {
                    best_score = score;
                    best = cand;
                }
            }
        }
        best
    }

    fn mpc_field_vector_score(&self, team: Team, idx: usize, cand: V2) -> f32 {
        let me = players(team, self)[idx].pos;
        let sx = team.sx();
        let lane = self.lane_clearness(team, self.ball, cand);
        let route = self.lane_clearness(team, me, cand);
        let wide = (cand.y - FIELD_W / 2.0).abs() / (FIELD_W / 2.0);
        let ahead = ((cand.x - self.ball.x) * sx).max(0.0) / FIELD_L;
        let (_, defender_d) = self.nearest_opponent(team, cand);
        let defender_space = (defender_d / 7.0).min(1.0);
        let mut teammate_gap = f32::INFINITY;
        for j in 1..N {
            if j == idx {
                continue;
            }
            teammate_gap = teammate_gap.min(players(team, self)[j].pos.sub(cand).len());
        }
        let teammate_space = (teammate_gap / 7.0).min(1.0);
        let carrier_pressure = if let Some(o) = self.owner {
            if o.team == team {
                let carrier = players(team, self)[o.idx].pos;
                let (_, pressure_d) = self.nearest_opponent(team, carrier);
                (1.0 - pressure_d / 7.0).clamp(0.0, 1.0)
            } else {
                0.0
            }
        } else {
            0.25
        };
        let goal = team.target_goal();
        let shot_channel = self.lane_clearness(team, cand, goal);
        lane * 1.35
            + route * 0.35
            + wide * 0.55
            + ahead * (0.55 + 0.35 * carrier_pressure)
            + defender_space * 0.45
            + teammate_space * 0.25
            + shot_channel * 0.20
    }

    /// MPC-lite lane opener (execution of the GET_OPEN decision). Enumerates
    /// candidate relocations and scores each candidate as a function of the
    /// current 10-player field vector: ball lane, route lane, width, aheadness,
    /// defender pressure, teammate spacing, carrier pressure, and shot channel.
    /// The POMDP picks GET_OPEN; this field-vector objective computes where
    /// "open" actually is. A QP/neural MPC can replace the enumerator, but it
    /// should optimize this same state-conditioned objective shape.
    fn mpc_open_lane_target(&self, team: Team, idx: usize) -> V2 {
        let me = players(team, self)[idx].pos;
        let radii = [4.0f32, 8.0, 12.0, 16.0];
        let ndir = 12usize;
        let mut best = me;
        let mut best_score = f32::NEG_INFINITY;
        for &r in &radii {
            for k in 0..ndir {
                let ang = (k as f32) / (ndir as f32) * std::f32::consts::TAU;
                let cand = V2::new(
                    (me.x + ang.cos() * r).clamp(3.0, FIELD_L - 3.0),
                    (me.y + ang.sin() * r).clamp(3.0, FIELD_W - 3.0),
                );
                let score = self.mpc_field_vector_score(team, idx, cand);
                if score > best_score {
                    best_score = score;
                    best = cand;
                }
            }
        }
        best
    }

    fn set_vel(&mut self, team: Team, idx: usize, v: V2) {
        match team {
            Team::A => self.a[idx].vel = v,
            Team::B => self.b[idx].vel = v,
        }
    }
}

fn integrate(p: &mut Player) {
    p.pos = p.pos.add(p.vel.scale(DT));
    clamp_pos(p);
}

fn clamp_pos(p: &mut Player) {
    p.pos.x = p.pos.x.clamp(-1.0, FIELD_L + 1.0);
    p.pos.y = p.pos.y.clamp(0.0, FIELD_W);
}

fn rotate(v: V2, ang: f32) -> V2 {
    let (s, c) = ang.sin_cos();
    V2::new(v.x * c - v.y * s, v.x * s + v.y * c)
}

/// The gear the scripted baseline uses for a given action (kept near the old
/// fixed speeds so the benchmark stays comparable): press/support runs fast,
/// spread/mark at a run, dribble at a skip, set-pieces standing.
pub fn scripted_gear(a: usize) -> usize {
    // The scripted baseline moves at run-medium (~= v3's fixed PLAYER_SPEED) so the
    // benchmark stays as hard as v3's — NOT faster. Over-buffing the defenders to
    // run_fast crushed the learner's pass completion and blocked all scoring.
    match a {
        A_CHASE | A_SUPPORT | A_SPREAD | A_MARK => SPD_RUN_MED,
        A_DRIB_FWD | A_DRIB_LEFT | A_DRIB_RIGHT => SPD_SKIP,
        _ => SPD_STAND, // shoot/pass/clear/hold/stay: no self-movement
    }
}

// ---------------------------------------------------------------------------
// Scripted "analytic-lite" baseline. Coherent soccer with simple heuristics —
// this is BOTH Team B's controller and the benchmark the learner must beat.
// ---------------------------------------------------------------------------
impl World {
    pub fn scripted_actions(&self, team: Team) -> [usize; N] {
        let mut acts = [A_STAY; N];
        let owner = self.owner;
        let team_owns = matches!(owner, Some(o) if o.team == team);
        // rank teammates by distance to ball for role assignment
        let mut order: Vec<(usize, f32)> = (0..N)
            .map(|i| (i, players(team, self)[i].pos.sub(self.ball).len()))
            .collect();
        order.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        for i in 0..N {
            let me = players(team, self)[i].pos;
            let is_owner = matches!(owner, Some(o) if o.team == team && o.idx == i);
            if is_owner {
                acts[i] = self.scripted_on_ball(team, i, me);
            } else if team_owns {
                // supporters: the closest non-owner pushes up; others spread; deepest holds.
                let rank = order.iter().position(|&(j, _)| j == i).unwrap();
                acts[i] = if rank <= 1 {
                    A_SUPPORT
                } else if rank == N - 1 {
                    A_MARK
                } else {
                    A_SPREAD
                };
            } else {
                // defend: the two nearest press the ball (double-team the
                // carrier), the rest mark — solo dribbling gets swarmed.
                acts[i] = if order[0].0 == i || order[1].0 == i {
                    A_CHASE
                } else {
                    A_MARK
                };
            }
            // pack a sensible gear for the baseline so it moves at realistic speeds.
            acts[i] += self.coerce_speed_gear(team, i, acts[i], scripted_gear(acts[i])) * NA;
        }
        acts
    }

    /// Spacing shaping for Team A's outfielders (indices 1..N), averaged over
    /// pairs. Penalize < 2 units apart, neutral 2–4 (leaves room for a short
    /// passing outlet), reward > 4 apart, cap the reward at 8. Applied whenever
    /// the ball is in play near/for Team A (not only in possession) so players
    /// keep shape while attacking AND while pushing up — not just clustering.
    #[allow(dead_code)]
    pub fn spacing_term_a(&self) -> f32 {
        // Skip only when Team B is settled in possession deep — there, A defends
        // and legitimately compresses. Otherwise (A possession or a loose ball)
        // reward good shape.
        let b_has = matches!(self.owner, Some(o) if matches!(o.team, Team::B));
        if b_has {
            return 0.0;
        }
        let mut sum = 0.0f32;
        let mut n = 0.0f32;
        for i in 1..N {
            for j in (i + 1)..N {
                let d = self.a[i].pos.sub(self.a[j].pos).len();
                // MEDIUM spacing (2–5 units, a short-pass distance): penalty when
                // < 2 (bunched), full reward across the 2–5 band, decaying to 0 by
                // 8, flat beyond. Keeps teammates at passing distance — not on top
                // of each other, not drifted too far to connect a pass.
                let v = if d < 2.0 {
                    -(2.0 - d)
                } else if d <= 5.0 {
                    1.0
                } else if d < 8.0 {
                    1.0 - (d - 5.0) / 3.0
                } else {
                    0.0
                };
                sum += v;
                n += 1.0;
            }
        }
        if n > 0.0 {
            sum / n
        } else {
            0.0
        }
    }

    /// GLOBAL state for the MAPPO centralized critic (CTDE): the entire field in
    /// one canonical Team-A-attack frame, identical for all agents at a tick.
    /// Every player's position+velocity, the ball, and possession.
    pub fn global_state(&self) -> [f32; GLOBAL_DIM] {
        let nx = FIELD_L;
        let ny = FIELD_W;
        let nv = 10.0;
        let mut f = Vec::with_capacity(GLOBAL_DIM);
        for team in [Team::A, Team::B] {
            let ps = players(team, self);
            for i in 0..N {
                f.push(ps[i].pos.x / nx * 2.0 - 1.0);
                f.push(ps[i].pos.y / ny * 2.0 - 1.0);
                f.push(ps[i].vel.x / nv);
                f.push(ps[i].vel.y / nv);
            }
        }
        f.push(self.ball.x / nx * 2.0 - 1.0);
        f.push(self.ball.y / ny * 2.0 - 1.0);
        f.push(self.ball_vel.x / nv);
        f.push(self.ball_vel.y / nv);
        // possession one-hot: A / B / free
        f.push(matches!(self.owner, Some(o) if matches!(o.team, Team::A)) as u8 as f32);
        f.push(matches!(self.owner, Some(o) if matches!(o.team, Team::B)) as u8 as f32);
        f.push(self.owner.is_none() as u8 as f32);
        f.push(1.0); // bias
        f.try_into().unwrap()
    }

    /// Closest teammate-pair distance among Team A's outfielders (the true
    /// "are any two stacked?" signal, which an average would hide).
    pub fn closest_pair_a(&self) -> f32 {
        let mut m = f32::INFINITY;
        for i in 1..N {
            for j in (i + 1)..N {
                let d = self.a[i].pos.sub(self.a[j].pos).len();
                if d < m {
                    m = d;
                }
            }
        }
        if m.is_finite() {
            m
        } else {
            0.0
        }
    }

    /// Average nearest-teammate distance among Team A's outfielders (1..N).
    /// Used to measure how spread out the attack is.
    pub fn avg_nearest_teammate_a(&self) -> f32 {
        let mut sum = 0.0f32;
        let mut n = 0.0f32;
        for i in 1..N {
            let mut nd = f32::INFINITY;
            for j in 1..N {
                if i == j {
                    continue;
                }
                let d = self.a[i].pos.sub(self.a[j].pos).len();
                if d < nd {
                    nd = d;
                }
            }
            if nd.is_finite() {
                sum += nd;
                n += 1.0;
            }
        }
        if n > 0.0 {
            sum / n
        } else {
            0.0
        }
    }

    /// Potential Φ from Team A's perspective, in [-1, 1]. Used for
    /// policy-invariant potential-based reward shaping (γΦ' − Φ).
    pub fn potential_a(&self) -> f32 {
        match self.owner {
            Some(o) if o.team == Team::A => self.ball.x / FIELD_L,
            Some(_) => -((FIELD_L - self.ball.x) / FIELD_L),
            None => 0.0,
        }
    }

    fn scripted_on_ball(&self, team: Team, idx: usize, me: V2) -> usize {
        let sx = team.sx();
        let goal = team.target_goal();
        let shot_dist = goal.sub(me).len();
        let clear = self.shot_clearness(team, me);
        let (_, opp_d) = self.nearest_opponent(team, me);
        let shot_legal = match team {
            Team::A => self.pass_streak_a >= 2 && me.x >= SHOOT_X,
            Team::B => self.b_pass_streak >= 2 && me.x <= FIELD_L - SHOOT_X,
        };

        // close in with a reasonably open lane -> shoot (else work it closer)
        if shot_legal && shot_dist < 18.0 && clear > 0.3 {
            return A_SHOOT;
        }
        let cands = self.pass_candidates(team, idx);
        let pressured = opp_d < 4.0;
        if pressured {
            // pass to best forward-open option if one is genuinely ahead
            for (rank, cand) in cands.iter().enumerate() {
                if let Some((ti, _)) = cand {
                    let tp = players(team, self)[*ti].pos;
                    if (tp.x - me.x) * sx > -3.0 {
                        return A_PASS_A + rank;
                    }
                }
            }
            // no outlet & deep in own half -> clear, else try to carry out
            if (me.x - team.own_goal().x).abs() < FIELD_L * 0.33 {
                return A_CLEAR;
            }
            // dribble away from the nearest opponent laterally
            let (oi, _) = self.nearest_opponent(team, me);
            let opp = players(team.other(), self)[oi].pos;
            return if opp.y > me.y {
                A_DRIB_LEFT
            } else {
                A_DRIB_RIGHT
            };
        }
        // If a forward lane is plainly available, use it as a build-up pass.
        // This makes the scripted teacher useful for behavior cloning instead of
        // teaching mostly solo dribbles until pressure arrives.
        for (rank, cand) in cands.iter().enumerate() {
            if let Some((ti, _)) = cand {
                let tp = players(team, self)[*ti].pos;
                if (tp.x - me.x) * sx > 5.0 && self.lane_clearness(team, me, tp) > 0.45 {
                    return A_PASS_A + rank;
                }
            }
        }
        // unpressured: is an opponent directly ahead in my forward cone?
        let (oi, od) = self.nearest_opponent(team, me);
        let opp = players(team.other(), self)[oi].pos;
        let ahead = (opp.x - me.x) * sx;
        if od < 8.0 && ahead > 0.0 && (opp.y - me.y).abs() < 4.0 {
            return if opp.y > me.y {
                A_DRIB_LEFT
            } else {
                A_DRIB_RIGHT
            };
        }
        A_DRIB_FWD
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stays() -> [usize; N] {
        [A_STAY; N]
    }

    fn arrange_return_pass_candidates(w: &mut World) {
        w.a[1].pos = V2::new(24.0, 14.0);
        w.a[2].pos = V2::new(30.0, 14.0);
        w.a[3].pos = V2::new(6.0, 4.0);
        w.a[4].pos = V2::new(7.0, 24.0);
        for i in 0..N {
            w.b[i].pos = V2::new(40.0, if i % 2 == 0 { 1.0 } else { 27.0 });
        }
    }

    fn return_pass_action_to_previous_giver(w: &World, from_idx: usize, giver_idx: usize) -> usize {
        let cands = w.pass_candidates(Team::A, from_idx);
        let rank = cands
            .iter()
            .position(|cand| matches!(cand, Some((idx, _)) if *idx == giver_idx))
            .expect("previous giver should remain legal but de-ranked");
        A_PASS_A + rank
    }

    #[test]
    fn observations_and_global_state_are_finite() {
        let w = World::new();
        for team in [Team::A, Team::B] {
            for idx in 1..N {
                let obs = w.observe(team, idx);
                assert_eq!(obs.len(), OBS_DIM);
                assert!(obs.iter().all(|v| v.is_finite()));
            }
        }
        let global = w.global_state();
        assert_eq!(global.len(), GLOBAL_DIM);
        assert!(global.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn legal_mask_partitions_on_ball_and_off_ball_actions() {
        let mut w = World::new();
        w.owner = Some(Owner {
            team: Team::A,
            idx: 1,
        });
        w.pass_streak_a = 0;
        let on_ball = w.legal_mask(Team::A, 1);
        assert!(
            !on_ball[A_SHOOT],
            "A cannot shoot before two completed passes"
        );
        assert!(on_ball[A_PASS_A]);
        assert!(on_ball[A_PASS_B]);
        assert!(on_ball[A_PASS_C]);
        assert!(on_ball[A_HOLD]);
        assert!(!on_ball[A_CHASE]);

        w.pass_streak_a = 2;
        w.a[1].pos = V2::new(FINAL_THIRD_X + 2.0, 14.0); // and be in the final third
        let after_buildup = w.legal_mask(Team::A, 1);
        assert!(
            after_buildup[A_SHOOT],
            "two completed passes + final third unlock shooting"
        );

        let off_ball = w.legal_mask(Team::A, 2);
        assert!(!off_ball[A_PASS_A]);
        assert!(
            !off_ball[A_CHASE],
            "off-ball attackers should not chase their own carrier"
        );
        assert!(off_ball[A_SUPPORT]);
        assert!(off_ball[A_GET_OPEN]);
        assert!(off_ball[A_STAY]);
    }

    #[test]
    fn pass_actions_cover_all_three_outfield_targets() {
        let mut w = World::new();
        w.owner = Some(Owner {
            team: Team::A,
            idx: 1,
        });
        w.a[1].pos = V2::new(12.0, 14.0);
        w.a[2].pos = V2::new(20.0, 8.0);
        w.a[3].pos = V2::new(22.0, 14.0);
        w.a[4].pos = V2::new(18.0, 22.0);
        for i in 0..N {
            w.b[i].pos = V2::new(40.0, if i % 2 == 0 { 2.0 } else { 26.0 });
        }

        let mask = w.legal_mask(Team::A, 1);
        assert!(mask[A_PASS_A]);
        assert!(mask[A_PASS_B]);
        assert!(mask[A_PASS_C]);
        let cands = w.pass_candidates(Team::A, 1);
        let mut ids = cands
            .iter()
            .flatten()
            .map(|(idx, _)| *idx)
            .collect::<Vec<_>>();
        ids.sort_unstable();
        assert_eq!(ids, vec![2, 3, 4]);

        w.a[4].pos = V2::new(12.5, 14.0); // too close to count as a real pass outlet
        let limited = w.legal_mask(Team::A, 1);
        assert!(limited[A_PASS_A]);
        assert!(limited[A_PASS_B]);
        assert!(!limited[A_PASS_C]);
    }

    #[test]
    fn keeper_distribution_counts_as_a_pass_attempt() {
        let mut w = World::new();
        w.owner = Some(Owner {
            team: Team::A,
            idx: GK,
        });

        let kick = w.apply_keeper(Team::A);

        assert!(kick.is_some_and(|(_, _, _, is_pass)| is_pass));
        assert!(w.ev_pass_attempt_a);
        assert!(matches!(
            w.intended_receiver,
            Some(o) if o.team == Team::A && o.idx != GK
        ));
    }

    #[test]
    fn unflagged_ball_through_goal_is_goal_kick_not_a_goal() {
        let mut w = World::new();
        w.owner = None;
        w.ball = V2::new(FIELD_L - 0.01, FIELD_W / 2.0);
        w.ball_vel = V2::new(5.0, 0.0);
        w.a_shot_flag = false;
        w.step(&stays(), &stays(), &mut Rng::new(1));
        assert_eq!(w.goals_a, 0);
        assert!(!w.ev_goal_a);
        assert!(matches!(w.owner, Some(o) if o.team == Team::B));
    }

    #[test]
    fn valid_a_shot_through_goal_scores_and_resets_buildup() {
        let mut w = World::new();
        w.owner = None;
        w.ball = V2::new(FIELD_L - 0.01, FIELD_W / 2.0);
        w.ball_vel = V2::new(5.0, 0.0);
        w.pass_streak_a = 2;
        w.a_shot_flag = true;
        w.step(&stays(), &stays(), &mut Rng::new(2));
        assert_eq!(w.goals_a, 1);
        assert!(w.ev_goal_a);
        assert_eq!(w.pass_streak_a, 0);
        assert!(matches!(w.owner, Some(o) if o.team == Team::B));
    }

    #[test]
    fn completed_a_pass_increments_streak_and_records_gain() {
        let mut w = World::new();
        w.owner = None;
        w.last_touch = Some(Team::A);
        w.pending_pass = Some(Owner {
            team: Team::A,
            idx: 2,
        });
        w.pass_kick_x = 10.0;
        w.pending_passer = 1;
        w.ball = w.a[2].pos;
        w.ball_vel = V2::default();
        w.kick_timer = -1;
        w.try_capture();
        assert!(w.ev_pass_completed_a);
        assert_eq!(w.pass_streak_a, 1);
        assert_eq!(w.lp_from, 1);
        assert_eq!(w.lp_to, 2);
        assert_eq!(w.pending_passer, -1);
        assert!(w.last_pass_gain_a.is_finite());
        assert!(matches!(w.owner, Some(o) if o.team == Team::A && o.idx == 2));
    }

    #[test]
    fn return_pass_to_previous_giver_sets_event_and_streak() {
        let mut w = World::new();
        arrange_return_pass_candidates(&mut w);
        w.owner = Some(Owner {
            team: Team::A,
            idx: 2,
        });
        w.lp_from = 1;
        w.lp_to = 2;
        w.return_streak_a = 1;

        let action = return_pass_action_to_previous_giver(&w, 2, 1);
        let kick = w.apply_on_ball(Team::A, 2, action, SPD_RUN_MED, &mut Rng::new(3));

        assert!(kick.is_some());
        assert!(matches!(w.intended_receiver, Some(o) if o.team == Team::A && o.idx == 1));
        assert!(w.ev_return_pass_a);
        assert_eq!(w.return_streak_a, 2);
        assert_eq!(w.pending_passer, 2);
    }

    #[test]
    fn non_return_pass_resets_return_streak() {
        let mut w = World::new();
        arrange_return_pass_candidates(&mut w);
        w.owner = Some(Owner {
            team: Team::A,
            idx: 2,
        });
        w.lp_from = 3;
        w.lp_to = 2;
        w.return_streak_a = 2;

        let kick = w.apply_on_ball(Team::A, 2, A_PASS_A, SPD_RUN_MED, &mut Rng::new(4));

        assert!(kick.is_some());
        assert!(matches!(w.intended_receiver, Some(o) if o.team == Team::A && o.idx == 1));
        assert!(!w.ev_return_pass_a);
        assert_eq!(w.return_streak_a, 0);
    }

    #[test]
    fn kickoff_clears_return_pass_memory() {
        let mut w = World::new();
        w.lp_from = 1;
        w.lp_to = 2;
        w.pending_passer = 2;
        w.return_streak_a = 3;

        w.kickoff(Team::B);

        assert_eq!(w.lp_from, -1);
        assert_eq!(w.lp_to, -1);
        assert_eq!(w.pending_passer, -1);
        assert_eq!(w.return_streak_a, 0);
    }

    #[test]
    fn reset_a_pass_memory_clears_return_start_x() {
        let mut w = World::new();
        w.return_start_x = 17.5;
        w.return_streak_a = 3;
        w.reset_a_pass_memory();
        assert_eq!(w.return_start_x, 0.0);
        assert_eq!(w.return_streak_a, 0);
    }

    #[test]
    fn first_return_records_start_x_and_streak_one() {
        // A→B→A: the FIRST return (streak 0→1) must be recorded but is "fine"
        // (the reward only penalises from streak >= 2).
        let mut w = World::new();
        arrange_return_pass_candidates(&mut w);
        w.owner = Some(Owner {
            team: Team::A,
            idx: 2,
        });
        w.ball = V2::new(20.0, 14.0);
        w.lp_from = 1; // player 1 gave the ball to player 2
        w.lp_to = 2;
        w.return_streak_a = 0;
        let action = return_pass_action_to_previous_giver(&w, 2, 1);
        let _ = w.apply_on_ball(Team::A, 2, action, SPD_RUN_MED, &mut Rng::new(3));
        assert!(w.ev_return_pass_a);
        assert_eq!(w.return_streak_a, 1);
        assert_eq!(w.return_start_x, 20.0); // sequence origin recorded
    }

    #[test]
    fn shielded_dribble_veers_away_from_close_defender() {
        // Defender directly ahead: forward intent must bend AWAY (−x here).
        let mut w = World::new();
        let me = V2::new(20.0, 14.0);
        w.a[1].pos = me;
        w.b[0].pos = V2::new(50.0, 14.0); // keep GK far
        w.b[1].pos = V2::new(21.0, 14.0); // 1 unit ahead of the carrier
        let dir = w.shielded_dribble_dir(Team::A, me, V2::new(1.0, 0.0));
        assert!(
            dir.x < 0.0,
            "should steer away from the defender, got {:?}",
            dir
        );
    }

    #[test]
    fn note_dribble_classifies_by_final_direction() {
        // A shielded direction that is actually backward must NOT be credited as a
        // forward dribble (the P2 fix: reward reflects real movement).
        let mut w = World::new();
        w.note_dribble(Team::A, V2::new(-1.0, 0.0), 1.0); // backward (x·sx < −0.3)
        assert!(!w.ev_dribble_fwd_a && !w.ev_dribble_lat_a);
        w.note_dribble(Team::A, V2::new(1.0, 0.0), 1.0); // forward
        assert!(w.ev_dribble_fwd_a);
        let mut w2 = World::new();
        w2.note_dribble(Team::A, V2::new(0.0, 1.0), 1.0); // lateral
        assert!(w2.ev_dribble_lat_a && !w2.ev_dribble_fwd_a);
    }

    #[test]
    fn intercept_point_leads_a_moving_ball() {
        // For a ball moving in +x, a defender ahead should be sent to a point on
        // the trajectory (>= current ball x), not the current spot.
        let mut w = World::new();
        w.owner = None;
        w.ball = V2::new(10.0, 14.0);
        w.ball_vel = V2::new(18.0, 0.0);
        let p = w.intercept_point(V2::new(25.0, 14.0));
        assert!(
            p.x >= w.ball.x,
            "intercept should lead the ball, got {:?}",
            p
        );
        // a stationary ball just returns the ball itself
        w.ball_vel = V2::default();
        assert_eq!(w.intercept_point(V2::new(25.0, 14.0)).x, w.ball.x);
    }

    #[test]
    fn opponent_half_rule_masks_shot_in_own_half() {
        let mut w = World::new();
        w.owner = Some(Owner {
            team: Team::A,
            idx: 1,
        });
        w.pass_streak_a = 2; // 2-pass rule satisfied
        w.a[1].pos = V2::new(SHOOT_X - 3.0, 14.0); // still in our own half
        assert!(
            !w.legal_mask(Team::A, 1)[A_SHOOT],
            "shot must be masked in our own half"
        );
        w.a[1].pos = V2::new(SHOOT_X + 2.0, 14.0); // in the opponent's half
        assert!(
            w.legal_mask(Team::A, 1)[A_SHOOT],
            "shot allowed in the opponent's half after 2 passes"
        );
    }

    #[test]
    fn b_goal_requires_valid_buildup() {
        // A free ball crossing B's target line (x<=0) does NOT count for B unless
        // b_shot_flag is set (B built up 2 passes + shot from its final third).
        let mut w = World::new();
        w.owner = None;
        w.b_shot_flag = false;
        w.ball = V2::new(0.4, FIELD_W / 2.0);
        w.ball_vel = V2::new(-20.0, 0.0);
        let g0 = w.goals_b;
        let empty = [A_STAY; N];
        w.step(&empty, &empty, &mut Rng::new(1));
        assert_eq!(w.goals_b, g0, "B goal without valid buildup must not count");
    }

    #[test]
    fn dribble_forward_action_sets_forward_event_in_the_clear() {
        // With no defender near, A_DRIB_FWD keeps forward intent -> forward event.
        let mut w = World::new();
        w.a[1].pos = V2::new(20.0, 14.0);
        w.owner = Some(Owner {
            team: Team::A,
            idx: 1,
        });
        for i in 0..N {
            w.b[i].pos = V2::new(41.0, 14.0); // all defenders far away
        }
        let _ = w.apply_on_ball(Team::A, 1, A_DRIB_FWD, SPD_RUN_MED, &mut Rng::new(1));
        assert!(w.ev_dribble_fwd_a);
    }

    #[test]
    fn support_speed_mask_requires_run_fast_or_sprint_in_possession() {
        let mut w = World::new();
        w.owner = Some(Owner {
            team: Team::A,
            idx: 1,
        });
        let mask = w.speed_mask(Team::A, 2, A_SUPPORT);

        assert!(!mask[SPD_STAND]);
        assert!(!mask[SPD_RUN_SLOW]);
        assert!(mask[SPD_RUN_FAST]);
        assert!(mask[SPD_SPRINT]);
        assert_eq!(
            w.coerce_speed_gear(Team::A, 2, A_SUPPORT, SPD_WALK),
            SPD_RUN_FAST
        );
    }

    #[test]
    fn goalside_recovery_target_moves_wrong_side_defender_toward_own_goal() {
        let mut w = World::new();
        w.owner = Some(Owner {
            team: Team::B,
            idx: 1,
        });
        w.ball = V2::new(28.0, 14.0);
        w.a[2].pos = V2::new(34.0, 16.0);
        w.b[2].pos = V2::new(32.0, 18.0);

        let target = w.goalside_recovery_target(Team::A, 2);

        assert!(
            target.x < w.a[2].pos.x,
            "wrong-side defender should recover toward own goal: target={target:?}"
        );
        assert!(
            target.x < w.ball.x,
            "target should get goalside of the ball: target={target:?}, ball={:?}",
            w.ball
        );
        let mask = w.speed_mask(Team::A, 2, A_MARK);
        assert!(mask[SPD_RUN_FAST] && mask[SPD_SPRINT]);
        assert!(!mask[SPD_SKIP]);
    }

    #[test]
    fn mpc_field_vector_score_prefers_open_wide_lane_over_blocked_centre() {
        let mut w = World::new();
        w.owner = Some(Owner {
            team: Team::A,
            idx: 1,
        });
        w.ball = V2::new(12.0, 14.0);
        w.a[1].pos = w.ball;
        w.a[2].pos = V2::new(14.0, 14.0);
        w.a[3].pos = V2::new(10.0, 5.0);
        w.a[4].pos = V2::new(10.0, 23.0);
        w.b[1].pos = V2::new(18.0, 14.0);
        w.b[2].pos = V2::new(22.0, 14.0);
        w.b[3].pos = V2::new(34.0, 6.0);
        w.b[4].pos = V2::new(34.0, 22.0);

        let blocked_centre = V2::new(25.0, 14.0);
        let wide_lane = V2::new(25.0, 24.0);

        assert!(
            w.mpc_field_vector_score(Team::A, 2, wide_lane)
                > w.mpc_field_vector_score(Team::A, 2, blocked_centre),
            "field-vector MPC should prefer the unblocked wide outlet"
        );
    }
}
