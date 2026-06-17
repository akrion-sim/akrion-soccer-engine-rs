//! Canonical kinematic **configuration vector** of the whole field — the 22
//! players + ball, each with position / velocity / acceleration, plus the ball
//! holder and ball altitude — for retrieval ("have we seen a state like this before?").
//!
//! This is the substrate for the vector-search side of the retrieval-guided
//! decision pipeline. A live snapshot `S1` is projected into a fixed-width,
//! **orientation- and permutation-invariant** vector; nearest neighbours in that
//! space are configurations we have actually played out, whose stored
//! decisions + outcomes then inform the MDP/POMDP choice (see
//! `SoccerConfigMomentInsert` / `search_nearest_config_moments`).
//!
//! ## Why canonicalize (and why here, not in the embedding net)
//! Raw `(x, y, vx, vy, ax, ay)` per player is not comparable across moments: the
//! two teams attack opposite ways, the same shape can be mirrored left/right, and
//! player *identities/order* are arbitrary. Cosine distance over the raw vector
//! would call mirror-images "far apart". So before projecting we:
//!   1. **Orient** so the perspective team always attacks +y (length axis). Home
//!      is identity; Away is a 180° rotation about the pitch centre (the engine's
//!      Y axis is goal-to-goal — `opponent_goal = (width/2, goal_y(length))`).
//!   2. **Mirror** the width axis to a canonical handedness (ball on the +x half),
//!      so a move and its left/right reflection match.
//!   3. **Sort** each team's outfielders into a canonical order (GK first, then by
//!      role, then by attack-axis depth, then laterally) — permutation invariance
//!      by *sorting*, NOT by an assignment solve in the hot path. (Hungarian
//!      re-ranking of retrieved candidates is a separate, optional refinement.)
//!
//! The projection into the persisted `vector(N)` space reuses
//! [`soccer_moment_embedding`] so the durable column width never migrates.

use super::*;

/// Players per team packed into the fixed-width feature vector. Real squads are
/// 11; fewer (red card) pad with zeros, more truncate — keeping the width stable.
pub const CONFIG_PLAYERS_PER_TEAM: usize = 11;
/// Legacy raw feature width before the per-player possession channel existed.
pub const CONFIG_FEATURE_DIM_V1: usize =
    2 * CONFIG_PLAYERS_PER_TEAM * 6 + CONFIG_BALL_FLOATS + CONFIG_SCALAR_FLOATS;
/// Floats per player: `pos.x, pos.y, vel.x, vel.y, acc.x, acc.y, has_possession` (canonical).
pub const CONFIG_PER_PLAYER_FLOATS: usize = 7;
/// Floats for the ball: `pos.x, pos.y, altitude, vel.x, vel.y, acc.x, acc.y`.
/// (The engine's ball has no vertical velocity channel; altitude is its own.)
pub const CONFIG_BALL_FLOATS: usize = 7;
/// Trailing scalars: `possession_relative, phase_code, score_diff`.
pub const CONFIG_SCALAR_FLOATS: usize = 3;

/// Total raw feature width before projection into [`SOCCER_MOMENT_EMBEDDING_DIM`].
pub const CONFIG_FEATURE_DIM: usize = 2 * CONFIG_PLAYERS_PER_TEAM * CONFIG_PER_PLAYER_FLOATS
    + CONFIG_BALL_FLOATS
    + CONFIG_SCALAR_FLOATS;

// Characteristic scales used to normalise physical units into ~[-1, 1] so no one
// channel dominates the cosine geometry. Positions are normalised by the pitch
// dimensions; the rest by realistic maxima (a hard sprint ~11 yd/s, peak human
// accel ~8 yd/s², a struck ball ~45 yd/s, a high ball ~12 yd of altitude).
const PLAYER_SPEED_SCALE_YPS: f64 = 11.0;
const PLAYER_ACCEL_SCALE_YPS2: f64 = 8.0;
const BALL_SPEED_SCALE_YPS: f64 = 45.0;
const BALL_ACCEL_SCALE_YPS2: f64 = 60.0;
const BALL_ALTITUDE_SCALE_YARDS: f64 = 12.0;

/// One player's canonical kinematic state inside a [`SoccerConfigVector`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoccerPlayerKin {
    pub role: PlayerRole,
    /// Canonical, pitch-normalised position in roughly `[-1, 1]²`.
    pub pos: Vec2,
    /// Canonical, scale-normalised velocity.
    pub vel: Vec2,
    /// Canonical, scale-normalised acceleration.
    pub acc: Vec2,
    /// `1` if this exact player holds the ball in the source snapshot, else `0`.
    pub has_possession: f64,
}

/// The whole-field configuration as seen from one team's perspective, after
/// orientation + handedness canonicalisation and per-team canonical ordering.
#[derive(Clone, Debug, PartialEq)]
pub struct SoccerConfigVector {
    pub perspective: Team,
    /// Perspective team, canonical order (GK first).
    pub own: Vec<SoccerPlayerKin>,
    /// Opponents, canonical order (GK first).
    pub opponents: Vec<SoccerPlayerKin>,
    /// Ball: normalised `(x, y, altitude)`.
    pub ball_pos: [f64; 3],
    /// Ball: normalised `(vx, vy)`.
    pub ball_vel: [f64; 2],
    /// Ball: normalised `(ax, ay)`.
    pub ball_acc: [f64; 2],
    /// `+1` perspective team holds the ball, `-1` opponent holds, `0` loose.
    pub possession_relative: f64,
    /// Perspective-relative phase code (own-attack `+1` … opp-attack `-1`).
    pub phase_code: f64,
    /// `(own_score - opp_score)`, clamped to `[-1, 1]` at ±5 goals.
    pub score_diff: f64,
}

/// Affine canonicalisation of the pitch for a perspective team: orient so the
/// team attacks +y, then mirror x to a canonical handedness. Applied identically
/// to every position (`is_vector = false`) and every velocity/acceleration
/// (`is_vector = true`, which drops the translation).
#[derive(Clone, Copy, Debug)]
struct Canonicalizer {
    field_width: f64,
    field_length: f64,
    rotate_180: bool,
    mirror_x: bool,
}

impl Canonicalizer {
    fn apply(&self, v: Vec2, is_vector: bool) -> Vec2 {
        let mut out = v;
        if self.rotate_180 {
            // 180° about the pitch centre: positions reflect through the centre,
            // vectors simply negate.
            if is_vector {
                out = Vec2::new(-out.x, -out.y);
            } else {
                out = Vec2::new(self.field_width - out.x, self.field_length - out.y);
            }
        }
        if self.mirror_x {
            if is_vector {
                out = Vec2::new(-out.x, out.y);
            } else {
                out = Vec2::new(self.field_width - out.x, out.y);
            }
        }
        out
    }

    fn norm_pos(&self, v: Vec2) -> Vec2 {
        let c = self.apply(v, false);
        Vec2::new(
            (c.x / self.field_width.max(1.0)) * 2.0 - 1.0,
            (c.y / self.field_length.max(1.0)) * 2.0 - 1.0,
        )
    }

    fn norm_vel(&self, v: Vec2, scale: f64) -> Vec2 {
        let c = self.apply(v, true);
        Vec2::new(c.x / scale, c.y / scale)
    }
}

impl SoccerConfigVector {
    /// Build the canonical configuration vector for `perspective` from a live
    /// world snapshot. Pure function of the snapshot, so it is identical at
    /// training (corpus build) and at inference (live retrieval) time.
    pub fn from_snapshot(snapshot: &WorldSnapshot, perspective: Team) -> Self {
        let field_width = snapshot.field_width;
        let field_length = snapshot.field_length;

        // Orient so the perspective team attacks +y: Away is a 180° rotation.
        let rotate_180 = perspective == Team::Away;

        // Decide handedness from a global signed statistic: the (oriented) sum of
        // every entity's offset from the centre line. Every term negates under a
        // left↔right reflection, so the sum negates too; mirroring whenever it is
        // negative is therefore reflection-invariant. The ball is weighted up so
        // ball side dominates, but unlike a ball-only rule this never ties when the
        // ball sits on the centre line (e.g. kickoff). Build a provisional
        // canonicalizer (no mirror), measure, then commit the flag.
        let provisional = Canonicalizer {
            field_width,
            field_length,
            rotate_180,
            mirror_x: false,
        };
        const BALL_HANDEDNESS_WEIGHT: f64 = 3.0;
        let half_width = field_width * 0.5;
        let mut handedness = BALL_HANDEDNESS_WEIGHT
            * (provisional.apply(snapshot.ball.position, false).x - half_width);
        for player in &snapshot.players {
            handedness += provisional.apply(player.position, false).x - half_width;
        }
        let mirror_x = handedness < 0.0;
        let canon = Canonicalizer {
            field_width,
            field_length,
            rotate_180,
            mirror_x,
        };

        let mut own = Vec::with_capacity(CONFIG_PLAYERS_PER_TEAM);
        let mut opponents = Vec::with_capacity(CONFIG_PLAYERS_PER_TEAM);
        let holder = snapshot.ball.holder;
        for player in &snapshot.players {
            let kin = SoccerPlayerKin {
                role: player.role,
                pos: canon.norm_pos(player.position),
                vel: canon.norm_vel(player.velocity, PLAYER_SPEED_SCALE_YPS),
                acc: canon.norm_vel(player.acceleration, PLAYER_ACCEL_SCALE_YPS2),
                has_possession: if holder == Some(player.id) { 1.0 } else { 0.0 },
            };
            if player.team == perspective {
                own.push(kin);
            } else {
                opponents.push(kin);
            }
        }
        sort_canonical(&mut own);
        sort_canonical(&mut opponents);

        let ball = &snapshot.ball;
        let bpos = canon.norm_pos(ball.position);
        let bvel = canon.norm_vel(ball.velocity, BALL_SPEED_SCALE_YPS);
        let bacc = canon.norm_vel(ball.acceleration, BALL_ACCEL_SCALE_YPS2);

        let possession_relative = match snapshot.possession_team() {
            Some(t) if t == perspective => 1.0,
            Some(_) => -1.0,
            None => 0.0,
        };

        let (own_score, opp_score) = match perspective {
            Team::Home => (snapshot.score_home, snapshot.score_away),
            Team::Away => (snapshot.score_away, snapshot.score_home),
        };
        let score_diff = ((own_score as f64 - opp_score as f64) / 5.0).clamp(-1.0, 1.0);

        SoccerConfigVector {
            perspective,
            own,
            opponents,
            ball_pos: [
                bpos.x,
                bpos.y,
                (ball.altitude_yards / BALL_ALTITUDE_SCALE_YARDS).clamp(0.0, 4.0),
            ],
            ball_vel: [bvel.x, bvel.y],
            ball_acc: [bacc.x, bacc.y],
            possession_relative,
            phase_code: perspective_phase_code(snapshot.phase, perspective),
            score_diff,
        }
    }

    /// Flatten into the fixed-width raw feature vector ([`CONFIG_FEATURE_DIM`]):
    /// own players, then opponents, then ball, then scalars. Missing players pad
    /// with zeros so the width is stable regardless of red cards.
    pub fn to_features(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(CONFIG_FEATURE_DIM);
        push_team(&mut out, &self.own);
        push_team(&mut out, &self.opponents);
        out.extend_from_slice(&self.ball_pos);
        out.extend_from_slice(&self.ball_vel);
        out.extend_from_slice(&self.ball_acc);
        out.push(self.possession_relative);
        out.push(self.phase_code);
        out.push(self.score_diff);
        debug_assert_eq!(out.len(), CONFIG_FEATURE_DIM);
        out
    }

    /// Project into the persisted [`SOCCER_MOMENT_EMBEDDING_DIM`] vector space.
    pub fn embedding(&self) -> Vec<f64> {
        soccer_moment_embedding(&self.to_features())
    }
}

fn push_team(out: &mut Vec<f64>, team: &[SoccerPlayerKin]) {
    for slot in 0..CONFIG_PLAYERS_PER_TEAM {
        if let Some(k) = team.get(slot) {
            out.extend_from_slice(&[
                k.pos.x,
                k.pos.y,
                k.vel.x,
                k.vel.y,
                k.acc.x,
                k.acc.y,
                k.has_possession,
            ]);
        } else {
            out.extend_from_slice(&[0.0; CONFIG_PER_PLAYER_FLOATS]);
        }
    }
}

/// Canonical, identity-free ordering of one team: GK first, then by role family,
/// then by attack-axis depth (deepest first), then laterally. Uses `total_cmp`
/// so it is a deterministic total order even with NaNs scrubbed upstream.
fn sort_canonical(players: &mut [SoccerPlayerKin]) {
    players.sort_by(|a, b| {
        role_rank(a.role)
            .cmp(&role_rank(b.role))
            .then(a.pos.y.total_cmp(&b.pos.y))
            .then(a.pos.x.total_cmp(&b.pos.x))
    });
}

fn role_rank(role: PlayerRole) -> u8 {
    match role {
        PlayerRole::Goalkeeper => 0,
        PlayerRole::Defender => 1,
        PlayerRole::Midfielder => 2,
        PlayerRole::Forward => 3,
    }
}

/// Perspective-relative phase: `+1` the team is attacking, `-1` it is defending
/// against an attack, `±0.33` in build-up, `0` for neutral (kickoff/transition).
fn perspective_phase_code(phase: TacticalPhase, perspective: Team) -> f64 {
    let (attacking_team, code) = match phase {
        TacticalPhase::Kickoff | TacticalPhase::Transition => return 0.0,
        TacticalPhase::HomeBuildUp => (Team::Home, 0.33),
        TacticalPhase::AwayBuildUp => (Team::Away, 0.33),
        TacticalPhase::HomeAttack => (Team::Home, 1.0),
        TacticalPhase::AwayAttack => (Team::Away, 1.0),
    };
    if attacking_team == perspective {
        code
    } else {
        -code
    }
}

/// A per-tick whole-field configuration capture (the ball carrier's decision),
/// buffered during a match and later joined with the transition stream to attach
/// the outcome. See [`SoccerMatch::config_moments`].
#[derive(Clone, Debug)]
pub struct SoccerConfigCapture {
    pub tick: u64,
    pub player_id: usize,
    pub team: Team,
    pub role: PlayerRole,
    pub action: String,
    /// Canonical raw config features ([`CONFIG_FEATURE_DIM`]) for this team.
    pub features: Vec<f32>,
}

/// A persisted configuration moment for the retrieval corpus: the embedding (for
/// ANN search), the raw features (for exact permutation re-score), the decision
/// taken, and how it turned out. This is the "config + decision + outcome" unit.
#[derive(Clone, Debug)]
pub struct SoccerConfigMomentInsert {
    pub team: Team,
    pub tick: u64,
    pub role: PlayerRole,
    pub action: String,
    /// Immediate MDP reward of the decision.
    pub reward: f64,
    /// Discounted n-step return over this player's subsequent transitions — "how
    /// the decision turned out", in the same MDP value units the policy learns on.
    pub nstep_return: f64,
    /// Critic value of the moment (`None` when neural learning is off/untrained).
    pub value: Option<f64>,
    /// Fixed-width similarity vector (projected from `features`).
    pub embedding: Vec<f64>,
    /// Raw canonical features, kept for an exact Hungarian re-score on retrieval.
    pub features: Vec<f32>,
}

/// A feature-independent view of one retrieved neighbour, so the prior aggregator
/// lives in the (PG-free) engine and the postgres-backed caller just maps its
/// `SoccerConfigMomentNeighbor` rows into this shape.
#[derive(Clone, Debug)]
pub struct SoccerRetrievalNeighborView {
    pub action: String,
    /// Cosine distance to the live query (0 = identical … 2 = opposite).
    pub distance: f64,
    /// Discounted n-step outcome return stored with the neighbour.
    pub nstep_return: f64,
    /// Critic value of the neighbour, if any.
    pub value: Option<f64>,
}

/// Reduce retrieved neighbours into a per-action **prior** over decisions: for each
/// distinct action, a similarity-weighted average of how favourably that decision
/// turned out in similar past configurations, squashed to `[-1, 1]`. This is the
/// "look up matching moments, see what decisions were made and how they worked
/// out, then bias toward the good ones" step. The result is later added (scaled by
/// a bounded weight) to the policy's action values — it never introduces new
/// candidate actions, so it cannot perturb RNG draws.
pub fn soccer_retrieval_action_prior(
    neighbors: &[SoccerRetrievalNeighborView],
) -> std::collections::HashMap<String, f64> {
    use std::collections::HashMap;
    // (similarity-weighted favourability sum, weight sum) per action.
    let mut acc: HashMap<String, (f64, f64)> = HashMap::new();
    for n in neighbors {
        if !n.distance.is_finite() {
            continue;
        }
        let similarity = (1.0 - n.distance).clamp(0.0, 1.0);
        if similarity <= f64::EPSILON {
            continue;
        }
        // Outcome signal: prefer the realised n-step return, fall back to the
        // critic value, then the immediate reward proxy embedded in the return.
        let outcome = if n.nstep_return.is_finite() && n.nstep_return != 0.0 {
            n.nstep_return
        } else {
            n.value.unwrap_or(0.0)
        };
        if !outcome.is_finite() {
            continue;
        }
        let favourability = outcome.tanh(); // squash to (-1, 1)
        let action = normalize_soccer_action_label(&n.action).to_string();
        if action.is_empty() {
            continue;
        }
        let entry = acc.entry(action).or_insert((0.0, 0.0));
        entry.0 += similarity * favourability;
        entry.1 += similarity;
    }
    acc.into_iter()
        .filter_map(|(action, (sum, weight))| {
            if weight > 0.0 {
                Some((action, (sum / weight).clamp(-1.0, 1.0)))
            } else {
                None
            }
        })
        .collect()
}

/// Cosine similarity between two raw config feature vectors (lengths may differ;
/// the shorter is treated as zero-padded). Used to rank/aggregate retrieved
/// neighbours by closeness to the live configuration.
pub fn soccer_config_feature_cosine(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let mut dot = 0.0_f64;
    let mut na = 0.0_f64;
    let mut nb = 0.0_f64;
    for i in 0..n {
        let x = if a[i].is_finite() { a[i] as f64 } else { 0.0 };
        let y = if b[i].is_finite() { b[i] as f64 } else { 0.0 };
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    // Account for tail energy so unequal lengths can't inflate similarity.
    for &x in &a[n..] {
        if x.is_finite() {
            na += (x as f64) * (x as f64);
        }
    }
    for &y in &b[n..] {
        if y.is_finite() {
            nb += (y as f64) * (y as f64);
        }
    }
    if na <= 1e-12 || nb <= 1e-12 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_snapshot() -> WorldSnapshot {
        // A real match snapshot: full, valid player/ball state with all fields.
        let mut sim = SoccerMatch::default_11v11(MatchConfig::default());
        sim.run_time_step();
        WorldSnapshot::from_match(&sim)
    }

    fn cosine(a: &[f64], b: &[f64]) -> f64 {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        if na <= 1e-12 || nb <= 1e-12 {
            0.0
        } else {
            dot / (na * nb)
        }
    }

    /// Relabelling players (shuffling the `players` vector) of an otherwise
    /// identical config must produce a byte-identical feature vector —
    /// permutation invariance via canonical sorting.
    #[test]
    fn permutation_invariant_under_relabelling() {
        let snap = sample_snapshot();
        let base = SoccerConfigVector::from_snapshot(&snap, Team::Home).to_features();

        let mut shuffled = snap.clone();
        shuffled.players.reverse();
        // Swap a couple of ids to prove identity plays no role.
        let n = shuffled.players.len();
        if n >= 2 {
            let (a, b) = (shuffled.players[0].id, shuffled.players[n - 1].id);
            shuffled.players[0].id = b;
            shuffled.players[n - 1].id = a;
            shuffled.ball.holder = match shuffled.ball.holder {
                Some(id) if id == a => Some(b),
                Some(id) if id == b => Some(a),
                holder => holder,
            };
        }
        let permuted = SoccerConfigVector::from_snapshot(&shuffled, Team::Home).to_features();

        assert_eq!(
            base, permuted,
            "relabelled identical config must be byte-identical"
        );
    }

    /// A Home-attacking-up moment must canonicalise identically to the same shape
    /// played by Away attacking down — orientation invariance, the property that
    /// lets retrieval match a situation regardless of which end it happens at.
    /// Construct the mirror world: rotate every entity 180° about the centre,
    /// negate velocities/accelerations, and swap the team labels. The acting team
    /// (Home → now Away) sees an identical relative shape, so the canonical vector
    /// from Away's perspective of the mirror must equal Home's of the original.
    #[test]
    fn orientation_invariant_under_board_flip_and_relabel() {
        let snap = sample_snapshot();
        let home = SoccerConfigVector::from_snapshot(&snap, Team::Home).to_features();

        let mut mirror = snap.clone();
        let (w, l) = (mirror.field_width, mirror.field_length);
        let flip_pos = |v: Vec2| Vec2::new(w - v.x, l - v.y);
        let flip_vec = |v: Vec2| Vec2::new(-v.x, -v.y);
        for p in &mut mirror.players {
            p.position = flip_pos(p.position);
            p.velocity = flip_vec(p.velocity);
            p.acceleration = flip_vec(p.acceleration);
            p.team = p.team.other();
        }
        mirror.ball.position = flip_pos(mirror.ball.position);
        mirror.ball.velocity = flip_vec(mirror.ball.velocity);
        mirror.ball.acceleration = flip_vec(mirror.ball.acceleration);
        std::mem::swap(&mut mirror.score_home, &mut mirror.score_away);
        // Relabelling the teams also relabels which side each phase belongs to.
        mirror.phase = match mirror.phase {
            TacticalPhase::HomeBuildUp => TacticalPhase::AwayBuildUp,
            TacticalPhase::AwayBuildUp => TacticalPhase::HomeBuildUp,
            TacticalPhase::HomeAttack => TacticalPhase::AwayAttack,
            TacticalPhase::AwayAttack => TacticalPhase::HomeAttack,
            other => other,
        };

        let away_of_mirror = SoccerConfigVector::from_snapshot(&mirror, Team::Away).to_features();
        assert!(
            cosine(&home, &away_of_mirror) > 0.999,
            "board-flip + relabel must canonicalise identically (got {})",
            cosine(&home, &away_of_mirror)
        );
    }

    /// Reflecting the whole board left↔right must map to the same canonical
    /// vector — handedness invariance.
    #[test]
    fn mirror_invariant_left_right() {
        let snap = sample_snapshot();
        let base = SoccerConfigVector::from_snapshot(&snap, Team::Home).to_features();

        let mut mirrored = snap.clone();
        let w = mirrored.field_width;
        for p in &mut mirrored.players {
            p.position = Vec2::new(w - p.position.x, p.position.y);
            p.velocity = Vec2::new(-p.velocity.x, p.velocity.y);
            p.acceleration = Vec2::new(-p.acceleration.x, p.acceleration.y);
        }
        mirrored.ball.position = Vec2::new(w - mirrored.ball.position.x, mirrored.ball.position.y);
        mirrored.ball.velocity = Vec2::new(-mirrored.ball.velocity.x, mirrored.ball.velocity.y);
        mirrored.ball.acceleration =
            Vec2::new(-mirrored.ball.acceleration.x, mirrored.ball.acceleration.y);
        let reflected = SoccerConfigVector::from_snapshot(&mirrored, Team::Home).to_features();

        assert!(
            cosine(&base, &reflected) > 0.999,
            "left/right reflection must be canonically identical (got {})",
            cosine(&base, &reflected)
        );
    }

    #[test]
    fn feature_width_is_stable() {
        let snap = sample_snapshot();
        let cfg = SoccerConfigVector::from_snapshot(&snap, Team::Home);
        assert_eq!(cfg.to_features().len(), CONFIG_FEATURE_DIM);
        assert_eq!(CONFIG_FEATURE_DIM_V1, 142);
        assert_eq!(cfg.embedding().len(), SOCCER_MOMENT_EMBEDDING_DIM);
    }

    #[test]
    fn feature_vector_contains_ball_altitude_and_holder_flags() {
        let mut snap = sample_snapshot();
        let holder = snap
            .players
            .iter()
            .find(|player| player.team == Team::Home)
            .expect("home player")
            .id;
        snap.ball.holder = Some(holder);
        snap.ball.altitude_yards = 6.0;

        let features = SoccerConfigVector::from_snapshot(&snap, Team::Home).to_features();
        let own_flags = (0..CONFIG_PLAYERS_PER_TEAM)
            .map(|slot| features[slot * CONFIG_PER_PLAYER_FLOATS + CONFIG_PER_PLAYER_FLOATS - 1])
            .collect::<Vec<_>>();
        let opponent_offset = CONFIG_PLAYERS_PER_TEAM * CONFIG_PER_PLAYER_FLOATS;
        let opponent_flags = (0..CONFIG_PLAYERS_PER_TEAM)
            .map(|slot| {
                features[opponent_offset
                    + slot * CONFIG_PER_PLAYER_FLOATS
                    + CONFIG_PER_PLAYER_FLOATS
                    - 1]
            })
            .collect::<Vec<_>>();
        let ball_offset = 2 * CONFIG_PLAYERS_PER_TEAM * CONFIG_PER_PLAYER_FLOATS;

        assert_eq!(own_flags.iter().filter(|&&flag| flag == 1.0).count(), 1);
        assert!(opponent_flags.iter().all(|&flag| flag == 0.0));
        assert!((features[ball_offset + 2] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn feature_vector_keeps_possession_team_when_ball_is_in_flight() {
        let mut snap = sample_snapshot();
        snap.ball.holder = None;
        snap.ball.last_touch_team = Some(Team::Home);

        let home_features = SoccerConfigVector::from_snapshot(&snap, Team::Home).to_features();
        let away_features = SoccerConfigVector::from_snapshot(&snap, Team::Away).to_features();
        let possession_offset =
            2 * CONFIG_PLAYERS_PER_TEAM * CONFIG_PER_PLAYER_FLOATS + CONFIG_BALL_FLOATS;

        assert_eq!(home_features[possession_offset], 1.0);
        assert_eq!(away_features[possession_offset], -1.0);
        assert!((0..(2 * CONFIG_PLAYERS_PER_TEAM)).all(|slot| {
            home_features[slot * CONFIG_PER_PLAYER_FLOATS + CONFIG_PER_PLAYER_FLOATS - 1] == 0.0
        }));
    }

    /// Capturing config-moments during a match must, after the episode, yield
    /// moment records whose decision + outcome (n-step return) are filled in from
    /// the transition stream — and only when capture is enabled.
    #[test]
    fn config_moments_capture_and_outcome() {
        // Off by default ⇒ nothing captured.
        let mut off = SoccerMatch::default_11v11(MatchConfig::default());
        for _ in 0..30 {
            off.run_time_step();
        }
        assert!(
            off.config_moments().is_empty(),
            "no capture when retrieval.capture_enabled is false"
        );

        // On ⇒ carrier moments captured with finite outcomes and full-width features.
        let mut cfg = MatchConfig::default();
        cfg.retrieval.capture_enabled = true;
        cfg.retrieval.outcome_horizon = 10;
        let mut on = SoccerMatch::default_11v11(cfg);
        for _ in 0..120 {
            on.run_time_step();
        }
        let moments = on.config_moments();
        assert!(!moments.is_empty(), "carrier decisions should be captured");
        for m in &moments {
            assert_eq!(m.embedding.len(), SOCCER_MOMENT_EMBEDDING_DIM);
            assert_eq!(m.features.len(), CONFIG_FEATURE_DIM);
            assert!(m.reward.is_finite() && m.nstep_return.is_finite());
            assert!(!m.action.is_empty());
        }
    }

    #[test]
    fn retrieval_action_prior_weights_favourable_decisions() {
        let neighbors = vec![
            // Very similar, decision worked out well → strong positive prior.
            SoccerRetrievalNeighborView {
                action: "shoot".to_string(),
                distance: 0.02,
                nstep_return: 3.0,
                value: None,
            },
            // Very similar, same decision, also good → reinforces.
            SoccerRetrievalNeighborView {
                action: "shoot".to_string(),
                distance: 0.05,
                nstep_return: 2.0,
                value: None,
            },
            // Similar, decision went badly → negative prior.
            SoccerRetrievalNeighborView {
                action: "pass-back".to_string(),
                distance: 0.10,
                nstep_return: -2.5,
                value: None,
            },
            // Orthogonal (distance ≥ 1) → ignored.
            SoccerRetrievalNeighborView {
                action: "dribble".to_string(),
                distance: 1.2,
                nstep_return: 5.0,
                value: None,
            },
        ];
        let prior = soccer_retrieval_action_prior(&neighbors);
        assert!(prior["shoot"] > 0.5, "favourable shoot prior: {:?}", prior);
        assert!(prior["pass-back"] < 0.0, "bad outcome ⇒ negative prior");
        assert!(
            !prior.contains_key("dribble"),
            "orthogonal neighbours contribute nothing"
        );
        for v in prior.values() {
            assert!((-1.0..=1.0).contains(v), "prior stays bounded");
        }
    }

    #[test]
    fn retrieval_action_prior_ignores_non_finite_neighbors() {
        let neighbors = vec![
            SoccerRetrievalNeighborView {
                action: "shoot".to_string(),
                distance: f64::NAN,
                nstep_return: 10.0,
                value: None,
            },
            SoccerRetrievalNeighborView {
                action: "pass".to_string(),
                distance: 0.0,
                nstep_return: f64::NAN,
                value: Some(f64::INFINITY),
            },
            SoccerRetrievalNeighborView {
                action: "".to_string(),
                distance: 0.0,
                nstep_return: 10.0,
                value: None,
            },
            SoccerRetrievalNeighborView {
                action: "hold".to_string(),
                distance: 0.0,
                nstep_return: -1.0,
                value: None,
            },
        ];

        let prior = soccer_retrieval_action_prior(&neighbors);

        assert_eq!(prior.len(), 1);
        assert!(prior["hold"] < 0.0, "finite bad hold should remain");
        assert!(prior.values().all(|value| value.is_finite()));
    }

    #[test]
    fn config_feature_cosine_self_is_one() {
        let snap = sample_snapshot();
        let f = SoccerConfigVector::from_snapshot(&snap, Team::Home).to_features();
        let f32v: Vec<f32> = f.iter().map(|&x| x as f32).collect();
        assert!((soccer_config_feature_cosine(&f32v, &f32v) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn config_feature_cosine_sanitizes_non_finite_channels() {
        let a = vec![1.0, f32::NAN, f32::INFINITY, 0.0];
        let b = vec![1.0, 0.0, 0.0, f32::NEG_INFINITY];

        let cosine = soccer_config_feature_cosine(&a, &b);

        assert!(cosine.is_finite());
        assert!((cosine - 1.0).abs() < 1e-9);
    }
}
