//! Canonical kinematic **configuration vector** of the whole field — the 22
//! players + ball, each with position / velocity / acceleration / jerk, plus
//! the ball holder and ball altitude — for retrieval ("have we seen a state like this before?").
//!
//! This is the substrate for the vector-search side of the retrieval-guided
//! decision pipeline. A live snapshot `S1` is projected into a fixed-width,
//! **orientation- and permutation-invariant** vector; nearest neighbours in that
//! space are configurations we have actually played out, whose stored
//! decisions + outcomes then inform the MDP/POMDP choice (see
//! `SoccerConfigMomentInsert` / `search_nearest_config_moments`).
//!
//! ## Why canonicalize (and why here, not in the embedding net)
//! Raw `(x, y, vx, vy, ax, ay, jx, jy)` per player is not comparable across
//! moments: the two teams attack opposite ways, the same shape can be mirrored
//! left/right, and player *identities/order* are arbitrary. Cosine distance over
//! the raw vector would call mirror-images "far apart". So before projecting we:
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
pub const CONFIG_FEATURE_DIM_V1: usize = 2 * CONFIG_PLAYERS_PER_TEAM * 6 + 7 + CONFIG_SCALAR_FLOATS;
/// Raw feature width before jerk channels were added.
pub const CONFIG_FEATURE_DIM_V2: usize = 2 * CONFIG_PLAYERS_PER_TEAM * 7 + 7 + CONFIG_SCALAR_FLOATS;
/// Floats per player: `pos.x, pos.y, vel.x, vel.y, acc.x, acc.y, jerk.x, jerk.y,
/// has_possession` (canonical).
pub const CONFIG_PER_PLAYER_FLOATS: usize = 9;
/// Floats for the ball: `pos.x, pos.y, altitude, vel.x, vel.y, acc.x, acc.y,
/// jerk.x, jerk.y`.
/// (The engine's ball has no vertical velocity channel; altitude is its own.)
pub const CONFIG_BALL_FLOATS: usize = 9;
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
const PLAYER_JERK_SCALE_YPS3: f64 = 40.0;
const BALL_SPEED_SCALE_YPS: f64 = 45.0;
const BALL_ACCEL_SCALE_YPS2: f64 = 60.0;
const BALL_JERK_SCALE_YPS3: f64 = 180.0;
const BALL_ALTITUDE_SCALE_YARDS: f64 = 12.0;

/// Which of the two whole-field comparisons to build. The modes differ **only**
/// in how each team's players are ordered into the fixed slots (i.e. who is
/// compared to whom under the slot-aligned cosine); the ball-proximity weighting
/// is identical for both.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SoccerConfigComparison {
    /// "No regard to assigned field position": order each team by **live
    /// geometry** (nearest the ball first), so the comparison is a pure
    /// on-the-ball *shape* match regardless of who is nominally assigned where.
    #[default]
    PositionAgnostic,
    /// "Apples to apples with regard to assigned field position": order each team
    /// by **assigned formation slot** (`home_position`), so a left-back only ever
    /// lines up against a left-back, a centre-back against a centre-back, etc.
    AssignedPosition,
}

/// Ball-proximity weighting + behind-ball discount controls for the configuration
/// vector. Every weight is **intrinsic** to a single config (derived from its own
/// ball + player positions), so weighting preserves the orientation / mirror /
/// permutation invariances. Defaults mirror [`SoccerRetrievalConfig`]; see
/// [`SoccerRetrievalConfig::weight_options`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConfigWeightOptions {
    /// Down-weight players by distance from the ball (`false` ⇒ every player
    /// keeps weight `1`, i.e. the legacy equal-weight comparison).
    pub proximity_enabled: bool,
    /// Characteristic decay length in yards: `weight = exp(-distance / tau)`.
    /// Smaller ⇒ similarity concentrates harder around the ball.
    pub proximity_tau_yards: f64,
    /// Fully discount (weight `0`) players far behind the ball **when the
    /// perspective team is attacking in the opponent's end** — they are out of
    /// the immediate play, so dropping them simplifies the comparison.
    pub behind_ball_discount_enabled: bool,
    /// How far behind the ball (yards, toward the perspective team's own goal) a
    /// player must be to be discounted.
    pub behind_ball_yards: f64,
    /// Fraction of the pitch length past which the ball counts as being in "the
    /// opponent's end" (`0.5` = past the half-way line).
    pub opponent_end_fraction: f64,
    /// Optional planned ball path (world coords: `from`→`to`) for an action under consideration
    /// (a pass/shot lane). When set, the proximity weight + the nearest-first ordering use each
    /// player's distance to this PATH SEGMENT instead of to the ball's current point — so a
    /// defender up the lane (far from the ball now, but decisive for the action) is weighted in,
    /// and a player near the ball but off the lane is weighted down. `None` ⇒ distance to the
    /// ball point (the legacy behaviour; the stored retrieval corpus is built this way).
    pub ball_path: Option<(Vec2, Vec2)>,
    /// Along-path taper applied ONLY when `ball_path` is set: a player's weight is additionally
    /// scaled by `exp(-t * taper)`, where `t in [0, 1]` is how far along the lane (0 = the release
    /// point, 1 = the target) its nearest point on the path sits. So danger concentrated early in
    /// the lane (where the ball is most cuttable) dominates and the target end is discounted.
    /// `0.0` => uniform along the path (no taper).
    pub ball_path_along_taper: f64,
}

impl Default for ConfigWeightOptions {
    fn default() -> Self {
        ConfigWeightOptions {
            proximity_enabled: true,
            proximity_tau_yards: 14.0,
            behind_ball_discount_enabled: true,
            behind_ball_yards: 6.0,
            opponent_end_fraction: 0.5,
            ball_path: None,
            ball_path_along_taper: 0.0,
        }
    }
}

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
    /// Canonical, scale-normalised jerk.
    pub jerk: Vec2,
    /// `1` if this exact player holds the ball in the source snapshot, else `0`.
    pub has_possession: f64,
    /// Ball-proximity weight in `[0, 1]` applied to this player's
    /// `pos/vel/acc/jerk` channels when flattening to features (the
    /// `has_possession` flag stays unscaled). `0` ⇒ fully discounted (far behind
    /// the ball during an attack).
    pub weight: f64,
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
    /// Ball: normalised `(jx, jy)`.
    pub ball_jerk: [f64; 2],
    /// `+1` perspective team holds the ball, `-1` opponent holds, `0` loose.
    pub possession_relative: f64,
    /// Perspective-relative phase code (own-attack `+1` … opp-attack `-1`).
    pub phase_code: f64,
    /// `(own_score - opp_score)`, clamped to `[-1, 1]` at ±5 goals.
    pub score_diff: f64,
}

/// Affine canonicalisation of the pitch for a perspective team: orient so the
/// team attacks +y, then mirror x to a canonical handedness. Applied identically
/// to every position (`is_vector = false`) and every velocity/acceleration/jerk
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
    /// world snapshot, using the default comparison ([`SoccerConfigComparison`])
    /// and weighting ([`ConfigWeightOptions`]). Pure function of the snapshot, so
    /// it is identical at training (corpus build) and inference (live retrieval).
    pub fn from_snapshot(snapshot: &WorldSnapshot, perspective: Team) -> Self {
        Self::from_snapshot_with(
            snapshot,
            perspective,
            SoccerConfigComparison::default(),
            ConfigWeightOptions::default(),
        )
    }

    /// As [`Self::from_snapshot`], but with an explicit comparison mode and
    /// weighting. Both modes apply identical ball-proximity weighting; they
    /// differ only in how each team's players are ordered into the fixed slots
    /// (which determines who is compared to whom under the slot-aligned cosine).
    pub fn from_snapshot_with(
        snapshot: &WorldSnapshot,
        perspective: Team,
        comparison: SoccerConfigComparison,
        weights: ConfigWeightOptions,
    ) -> Self {
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

        // Oriented (yards) ball position drives the proximity weight and the
        // behind-ball discount gate. Orientation/mirror are isometries, so the
        // ball-distance (and the +y "behind" test, since the perspective always
        // attacks +y after canonicalisation) are invariant.
        let ball_oriented = canon.apply(snapshot.ball.position, false);
        let perspective_has_ball =
            matches!(snapshot.possession_team(), Some(t) if t == perspective);
        let ball_in_opp_end = ball_oriented.y > field_length * weights.opponent_end_fraction;
        // Oriented planned-action path (a pass/shot lane), when supplied: proximity is then measured
        // to this segment, so players up the lane are weighted in and off-lane players down.
        let path_oriented = weights
            .ball_path
            .map(|(a, b)| (canon.apply(a, false), canon.apply(b, false)));

        // (sort key, player) so the two comparison modes share one build pass and
        // differ only in the ordering key applied just below.
        let mut own: Vec<([f64; 3], SoccerPlayerKin)> = Vec::with_capacity(CONFIG_PLAYERS_PER_TEAM);
        let mut opponents: Vec<([f64; 3], SoccerPlayerKin)> =
            Vec::with_capacity(CONFIG_PLAYERS_PER_TEAM);
        let holder = snapshot.ball.holder;
        for player in &snapshot.players {
            let oriented = canon.apply(player.position, false);
            let dist_to_ball = oriented.distance(ball_oriented);
            // Proximity is to the planned path segment when one is given, else to the ball point.
            let proximity_dist = match path_oriented {
                Some((a, b)) => segment_distance_to_point(a, b, oriented),
                None => dist_to_ball,
            };
            // Along-path taper: discount players whose nearest lane point sits further toward the
            // target (the early lane is where the ball is most cuttable). Only with a path + taper.
            let along_factor = match path_oriented {
                Some((a, b)) if weights.ball_path_along_taper > 0.0 => {
                    let ab = b - a;
                    let len2 = ab.dot(ab);
                    let t = if len2 > 1e-9 {
                        ((oriented - a).dot(ab) / len2).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    (-t * weights.ball_path_along_taper).exp()
                }
                _ => 1.0,
            };
            let weight = config_player_weight(
                proximity_dist,
                oriented.y,
                ball_oriented.y,
                perspective_has_ball,
                ball_in_opp_end,
                &weights,
            ) * along_factor;
            let kin = SoccerPlayerKin {
                role: player.role,
                pos: canon.norm_pos(player.position),
                vel: canon.norm_vel(player.velocity, PLAYER_SPEED_SCALE_YPS),
                acc: canon.norm_vel(player.acceleration, PLAYER_ACCEL_SCALE_YPS2),
                jerk: canon.norm_vel(player.jerk, PLAYER_JERK_SCALE_YPS3),
                has_possession: if holder == Some(player.id) { 1.0 } else { 0.0 },
                weight,
            };
            // All keys are built from canonical (orientation/mirror-normalised)
            // quantities, so the resulting slot ordering is itself invariant.
            let key = match comparison {
                // Nearest the ball (or the planned path, when given) first; canonical pos
                // breaks ties.
                SoccerConfigComparison::PositionAgnostic => [proximity_dist, kin.pos.y, kin.pos.x],
                // Assigned formation slot (deepest home first ⇒ GK leads), so
                // like-for-like positions align across configs.
                SoccerConfigComparison::AssignedPosition => {
                    let home = canon.apply(player.home_position, false);
                    [home.y, home.x, kin.pos.y]
                }
            };
            if player.team == perspective {
                own.push((key, kin));
            } else {
                opponents.push((key, kin));
            }
        }
        let own = sort_by_key_into_kin(own);
        let opponents = sort_by_key_into_kin(opponents);

        let ball = &snapshot.ball;
        let bpos = canon.norm_pos(ball.position);
        let bvel = canon.norm_vel(ball.velocity, BALL_SPEED_SCALE_YPS);
        let bacc = canon.norm_vel(ball.acceleration, BALL_ACCEL_SCALE_YPS2);
        let bjerk = canon.norm_vel(ball.jerk, BALL_JERK_SCALE_YPS3);

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
            ball_jerk: [bjerk.x, bjerk.y],
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
        out.extend_from_slice(&self.ball_jerk);
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

/// Flatten one team's players into the feature stream. Each player's
/// `pos/vel/acc/jerk` channels are scaled by its ball-proximity `weight` (so
/// near-ball players dominate the cosine and fully-discounted players contribute
/// nothing); the `has_possession` flag is left **unscaled** as a categorical
/// holder marker.
fn push_team(out: &mut Vec<f64>, team: &[SoccerPlayerKin]) {
    for slot in 0..CONFIG_PLAYERS_PER_TEAM {
        if let Some(k) = team.get(slot) {
            let w = k.weight;
            out.extend_from_slice(&[
                k.pos.x * w,
                k.pos.y * w,
                k.vel.x * w,
                k.vel.y * w,
                k.acc.x * w,
                k.acc.y * w,
                k.jerk.x * w,
                k.jerk.y * w,
                k.has_possession,
            ]);
        } else {
            out.extend_from_slice(&[0.0; CONFIG_PER_PLAYER_FLOATS]);
        }
    }
}

/// Deterministic total order of one team by an `[f64; 3]` key (built from
/// canonical, orientation/mirror-normalised quantities, so the ordering is itself
/// invariant), then strip the keys leaving the ordered players. `total_cmp` keeps
/// it a total order even with NaNs scrubbed upstream.
fn sort_by_key_into_kin(mut keyed: Vec<([f64; 3], SoccerPlayerKin)>) -> Vec<SoccerPlayerKin> {
    keyed.sort_by(|(a, _), (b, _)| {
        a[0].total_cmp(&b[0])
            .then(a[1].total_cmp(&b[1]))
            .then(a[2].total_cmp(&b[2]))
    });
    keyed.into_iter().map(|(_, kin)| kin).collect()
}

/// Ball-proximity weight in `[0, 1]` for one player. Players far behind the ball
/// while the perspective team attacks in the opponent's end are fully discounted
/// (weight `0`); otherwise weight decays smoothly with distance from the ball.
fn config_player_weight(
    dist_to_ball: f64,
    oriented_y: f64,
    ball_oriented_y: f64,
    perspective_has_ball: bool,
    ball_in_opp_end: bool,
    opts: &ConfigWeightOptions,
) -> f64 {
    if opts.behind_ball_discount_enabled
        && perspective_has_ball
        && ball_in_opp_end
        && oriented_y < ball_oriented_y - opts.behind_ball_yards
    {
        return 0.0;
    }
    if !opts.proximity_enabled {
        return 1.0;
    }
    let tau = opts.proximity_tau_yards.max(1e-3);
    (-dist_to_ball.max(0.0) / tau).exp()
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
    /// Canonical raw config features ([`CONFIG_FEATURE_DIM`]) in the
    /// **position-agnostic** ordering (the primary ANN corpus).
    pub features: Vec<f32>,
    /// The same config in the **assigned-position** (slot-aligned) ordering — the
    /// second corpus, so a like-for-like positional comparison is independently
    /// searchable. Same width as `features`.
    pub features_assigned: Vec<f32>,
}

/// A persisted configuration moment for the retrieval corpus: the embedding (for
/// ANN search), the raw features (for exact permutation re-score), the decision
/// taken, and how it turned out. This is the "config + decision + outcome" unit.
///
/// Two orderings are stored per moment so each comparison mode
/// ([`SoccerConfigComparison`]) is independently HNSW-searchable: the
/// position-agnostic pair (`embedding` / `features`) and the assigned-position
/// pair (`embedding_assigned` / `features_assigned`).
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
    /// Position-agnostic similarity vector (projected from `features`).
    pub embedding: Vec<f64>,
    /// Raw canonical features (position-agnostic ordering), kept for an exact
    /// re-score on retrieval.
    pub features: Vec<f32>,
    /// Assigned-position similarity vector (projected from `features_assigned`).
    pub embedding_assigned: Vec<f64>,
    /// Raw canonical features in the assigned-position (slot-aligned) ordering.
    pub features_assigned: Vec<f32>,
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
        let flip_team = |team: Option<Team>| team.map(Team::other);
        for p in &mut mirror.players {
            p.position = flip_pos(p.position);
            for history_position in &mut p.position_history {
                *history_position = flip_pos(*history_position);
            }
            p.home_position = flip_pos(p.home_position);
            p.velocity = flip_vec(p.velocity);
            p.acceleration = flip_vec(p.acceleration);
            p.jerk = flip_vec(p.jerk);
            if let Some(incoming) = &mut p.incoming_ball {
                incoming.team = flip_team(incoming.team);
                incoming.origin = incoming.origin.map(flip_pos);
                incoming.intended_target = incoming.intended_target.map(flip_pos);
            }
            if let Some(one_two) = &mut p.one_two {
                one_two.return_target = flip_pos(one_two.return_target);
            }
            p.team = p.team.other();
        }
        mirror.ball.position = flip_pos(mirror.ball.position);
        mirror.ball.velocity = flip_vec(mirror.ball.velocity);
        mirror.ball.acceleration = flip_vec(mirror.ball.acceleration);
        mirror.ball.jerk = flip_vec(mirror.ball.jerk);
        mirror.ball.curl_acceleration = flip_vec(mirror.ball.curl_acceleration);
        mirror.ball.last_touch_team = flip_team(mirror.ball.last_touch_team);
        for sample in &mut mirror.ball_history {
            sample.position = flip_pos(sample.position);
            sample.velocity = flip_vec(sample.velocity);
            sample.acceleration = flip_vec(sample.acceleration);
            sample.jerk = flip_vec(sample.jerk);
            sample.curl_acceleration = flip_vec(sample.curl_acceleration);
            sample.last_touch_team = flip_team(sample.last_touch_team);
        }
        if let Some(pass) = &mut mirror.pending_pass {
            pass.team = pass.team.other();
            pass.origin = flip_pos(pass.origin);
            pass.intended_target = flip_pos(pass.intended_target);
            pass.receiver_position_at_launch = pass.receiver_position_at_launch.map(flip_pos);
            pass.receiver_velocity_at_launch = pass.receiver_velocity_at_launch.map(flip_vec);
            if let Some(offside) = &mut pass.offside {
                offside.team = offside.team.other();
                offside.position = flip_pos(offside.position);
                offside.ball_y = l - offside.ball_y;
                offside.second_last_defender_y = l - offside.second_last_defender_y;
            }
        }
        if let Some(rebound) = &mut mirror.pending_rebound {
            rebound.attacking_team = rebound.attacking_team.other();
            rebound.parry_position = flip_pos(rebound.parry_position);
        }
        for sample in &mut mirror.shared_positions.latest {
            sample.position = flip_pos(sample.position);
            sample.velocity = flip_vec(sample.velocity);
            sample.acceleration = flip_vec(sample.acceleration);
            sample.jerk = flip_vec(sample.jerk);
        }
        for history in mirror.shared_positions.histories.values_mut() {
            for sample in history {
                sample.position = flip_pos(sample.position);
                sample.velocity = flip_vec(sample.velocity);
                sample.acceleration = flip_vec(sample.acceleration);
                sample.jerk = flip_vec(sample.jerk);
            }
        }
        for sample in &mut mirror.shared_positions.official_latest {
            sample.position = flip_pos(sample.position);
            sample.velocity = flip_vec(sample.velocity);
            sample.acceleration = flip_vec(sample.acceleration);
            sample.jerk = flip_vec(sample.jerk);
        }
        for history in mirror.shared_positions.official_histories.values_mut() {
            for sample in history {
                sample.position = flip_pos(sample.position);
                sample.velocity = flip_vec(sample.velocity);
                sample.acceleration = flip_vec(sample.acceleration);
                sample.jerk = flip_vec(sample.jerk);
            }
        }
        if let Some(sample) = &mut mirror.shared_positions.ball_latest {
            sample.position = flip_pos(sample.position);
            sample.velocity = flip_vec(sample.velocity);
            sample.acceleration = flip_vec(sample.acceleration);
            sample.jerk = flip_vec(sample.jerk);
            sample.curl_acceleration = flip_vec(sample.curl_acceleration);
            sample.last_touch_team = flip_team(sample.last_touch_team);
        }
        for sample in &mut mirror.shared_positions.ball_history {
            sample.position = flip_pos(sample.position);
            sample.velocity = flip_vec(sample.velocity);
            sample.acceleration = flip_vec(sample.acceleration);
            sample.jerk = flip_vec(sample.jerk);
            sample.curl_acceleration = flip_vec(sample.curl_acceleration);
            sample.last_touch_team = flip_team(sample.last_touch_team);
        }
        std::mem::swap(&mut mirror.score_home, &mut mirror.score_away);
        std::mem::swap(
            &mut mirror.home_team_possession_seconds,
            &mut mirror.away_team_possession_seconds,
        );
        // Relabelling the teams also relabels which side each phase belongs to.
        mirror.phase = match mirror.phase {
            TacticalPhase::HomeBuildUp => TacticalPhase::AwayBuildUp,
            TacticalPhase::AwayBuildUp => TacticalPhase::HomeBuildUp,
            TacticalPhase::HomeAttack => TacticalPhase::AwayAttack,
            TacticalPhase::AwayAttack => TacticalPhase::HomeAttack,
            other => other,
        };

        let away_of_mirror = SoccerConfigVector::from_snapshot(&mirror, Team::Away).to_features();
        let similarity = cosine(&home, &away_of_mirror);
        let largest_diffs = home
            .iter()
            .zip(away_of_mirror.iter())
            .enumerate()
            .map(|(idx, (a, b))| (idx, *a, *b, (*a - *b).abs()))
            .filter(|(_, _, _, diff)| *diff > 1e-9)
            .collect::<Vec<_>>();
        assert!(
            similarity > 0.999,
            "board-flip + relabel must canonicalise identically (got {similarity}; diffs {:?})",
            largest_diffs.iter().take(12).collect::<Vec<_>>()
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

    /// The assigned-position mode orders by `home_position`, so a left↔right board
    /// reflection (current state **and** the formation anchors) must canonicalise
    /// identically — handedness invariance for the slot-aligned corpus.
    #[test]
    fn assigned_position_mirror_invariant() {
        let snap = sample_snapshot();
        let opts = ConfigWeightOptions::default();
        let base = SoccerConfigVector::from_snapshot_with(
            &snap,
            Team::Home,
            SoccerConfigComparison::AssignedPosition,
            opts,
        )
        .to_features();

        let mut mirrored = snap.clone();
        let w = mirrored.field_width;
        for p in &mut mirrored.players {
            p.position = Vec2::new(w - p.position.x, p.position.y);
            p.velocity = Vec2::new(-p.velocity.x, p.velocity.y);
            p.acceleration = Vec2::new(-p.acceleration.x, p.acceleration.y);
            // The assigned mode sorts by home_position, so the formation anchors
            // must reflect with the board.
            p.home_position = Vec2::new(w - p.home_position.x, p.home_position.y);
        }
        mirrored.ball.position = Vec2::new(w - mirrored.ball.position.x, mirrored.ball.position.y);
        mirrored.ball.velocity = Vec2::new(-mirrored.ball.velocity.x, mirrored.ball.velocity.y);
        mirrored.ball.acceleration =
            Vec2::new(-mirrored.ball.acceleration.x, mirrored.ball.acceleration.y);
        let reflected = SoccerConfigVector::from_snapshot_with(
            &mirrored,
            Team::Home,
            SoccerConfigComparison::AssignedPosition,
            opts,
        )
        .to_features();

        assert!(
            cosine(&base, &reflected) > 0.999,
            "assigned-position reflection must canonicalise identically (got {})",
            cosine(&base, &reflected)
        );
    }

    /// The two modes differ exactly in whether they care about assigned position:
    /// permuting only the `home_position`s (reassigning slots, leaving every live
    /// position/velocity untouched) must leave the **position-agnostic** vector
    /// byte-identical yet **change** the **assigned-position** vector.
    #[test]
    fn modes_differ_on_assigned_position_only() {
        let snap = sample_snapshot();
        let opts = ConfigWeightOptions::default();
        let features = |s: &WorldSnapshot, c| {
            SoccerConfigVector::from_snapshot_with(s, Team::Home, c, opts).to_features()
        };
        let agnostic = features(&snap, SoccerConfigComparison::PositionAgnostic);
        let assigned = features(&snap, SoccerConfigComparison::AssignedPosition);

        // Rotate the Home players' home_positions by one (reassign their slots).
        let mut permuted = snap.clone();
        let homes: Vec<Vec2> = permuted
            .players
            .iter()
            .filter(|p| p.team == Team::Home)
            .map(|p| p.home_position)
            .collect();
        assert!(homes.len() >= 2, "need at least two assigned slots");
        let mut k = 0;
        for p in permuted.players.iter_mut().filter(|p| p.team == Team::Home) {
            p.home_position = homes[(k + 1) % homes.len()];
            k += 1;
        }

        let agnostic_permuted = features(&permuted, SoccerConfigComparison::PositionAgnostic);
        let assigned_permuted = features(&permuted, SoccerConfigComparison::AssignedPosition);

        assert_eq!(
            agnostic, agnostic_permuted,
            "position-agnostic comparison must ignore assigned field position"
        );
        assert_ne!(
            assigned, assigned_permuted,
            "assigned-position comparison must depend on assigned field position"
        );
    }

    /// Ball-proximity weighting must make players near the ball dominate: turning
    /// it on shrinks a far player's channel energy far more than a near player's
    /// (slot order is unaffected by weighting, so slots align ON vs OFF).
    #[test]
    fn proximity_weighting_downweights_far_players() {
        let snap = sample_snapshot();
        let on = ConfigWeightOptions {
            proximity_enabled: true,
            behind_ball_discount_enabled: false,
            ..ConfigWeightOptions::default()
        };
        let off = ConfigWeightOptions {
            proximity_enabled: false,
            behind_ball_discount_enabled: false,
            ..ConfigWeightOptions::default()
        };
        let feats = |o| {
            SoccerConfigVector::from_snapshot_with(
                &snap,
                Team::Home,
                SoccerConfigComparison::PositionAgnostic,
                o,
            )
            .to_features()
        };
        let f_on = feats(on);
        let f_off = feats(off);

        // Energy of one own slot's weighted pos/vel/acc/jerk channels.
        let slot_energy = |f: &[f64], slot: usize| -> f64 {
            let base = slot * CONFIG_PER_PLAYER_FLOATS;
            (0..8).map(|c| f[base + c] * f[base + c]).sum()
        };
        // Slot 0 is the nearest own player to the ball; the last own slot the
        // farthest (position-agnostic sorts own players by ball distance).
        let near = 0usize;
        let far = CONFIG_PLAYERS_PER_TEAM - 1;
        let (near_off, far_off) = (slot_energy(&f_off, near), slot_energy(&f_off, far));
        assert!(
            near_off > 1e-9 && far_off > 1e-9,
            "need non-degenerate baseline energies"
        );
        let near_ratio = slot_energy(&f_on, near) / near_off;
        let far_ratio = slot_energy(&f_on, far) / far_off;
        assert!(
            far_ratio < near_ratio,
            "far player must be down-weighted more than the near player \
             (near {near_ratio}, far {far_ratio})"
        );
        assert!(
            far_ratio < 1.0,
            "far player energy must shrink (got {far_ratio})"
        );
    }

    /// A planned ball path re-centres the proximity weighting on the LANE: an opponent far up the
    /// lane (far from the ball point, so near-zero weight without a path) is weighted back in, so
    /// the representation changes. `None` (the legacy ball-point weighting) is unaffected.
    #[test]
    fn planned_path_weighting_includes_lane_players() {
        let mut snap = sample_snapshot();
        let ball = Vec2::new(40.0, 20.0);
        snap.ball.position = ball;
        snap.ball.holder = None;
        // An opponent 30 yd up the lane: far from the ball point, but on the line to the target.
        if let Some(opp) = snap
            .players
            .iter_mut()
            .find(|p| p.team == Team::Away && p.role != PlayerRole::Goalkeeper)
        {
            opp.position = Vec2::new(40.0, 50.0);
            opp.velocity = Vec2::new(0.0, 0.0);
        }
        let build = |path: Option<(Vec2, Vec2)>| {
            SoccerConfigVector::from_snapshot_with(
                &snap,
                Team::Home,
                SoccerConfigComparison::PositionAgnostic,
                ConfigWeightOptions {
                    ball_path: path,
                    ..ConfigWeightOptions::default()
                },
            )
            .to_features()
        };
        let no_path = build(None);
        let with_path = build(Some((ball, Vec2::new(40.0, 80.0))));
        assert!(
            cosine(&no_path, &with_path) < 0.999,
            "planned-path weighting must re-weight toward the lane (cos {})",
            cosine(&no_path, &with_path)
        );
    }

    /// The behind-ball discount zeroes a player **only** under the full gate
    /// (perspective attacking, ball in opponent's end, player >6 yd behind).
    #[test]
    fn behind_ball_discount_gate() {
        let opts = ConfigWeightOptions::default();
        // Opponent's end, with the player 10 yd behind the ball.
        let ball_y = 90.0_f64;
        let deep_y = ball_y - 10.0;
        // All conditions met ⇒ fully discounted.
        assert_eq!(
            config_player_weight(40.0, deep_y, ball_y, true, true, &opts),
            0.0
        );
        // Not in possession ⇒ kept (proximity weight only).
        assert!(config_player_weight(40.0, deep_y, ball_y, false, true, &opts) > 0.0);
        // Ball not in opponent's end ⇒ kept.
        assert!(config_player_weight(40.0, deep_y, ball_y, true, false, &opts) > 0.0);
        // Ahead of (or level with) the ball ⇒ kept even under attack.
        assert!(config_player_weight(40.0, ball_y + 5.0, ball_y, true, true, &opts) > 0.0);
        // Discoverable off-switch.
        let no_discount = ConfigWeightOptions {
            behind_ball_discount_enabled: false,
            ..opts
        };
        assert!(config_player_weight(40.0, deep_y, ball_y, true, true, &no_discount) > 0.0);
    }

    /// End-to-end: with the gate satisfied, the discount fully zeroes the channels
    /// of own players caught behind the ball — and does so only when enabled.
    #[test]
    fn behind_ball_discount_zeroes_deep_players_in_snapshot() {
        let mut snap = sample_snapshot();
        // Home attacks +y (identity orientation); put the ball deep in the
        // opponent's end and give Home possession so the gate is live.
        let home_id = snap
            .players
            .iter()
            .find(|p| p.team == Team::Home)
            .expect("home player")
            .id;
        snap.ball.holder = Some(home_id);
        snap.ball.position = Vec2::new(snap.field_width * 0.5, snap.field_length * 0.75);

        let proximity_off = ConfigWeightOptions {
            proximity_enabled: false,
            ..ConfigWeightOptions::default()
        };
        let with_discount = proximity_off;
        let without_discount = ConfigWeightOptions {
            behind_ball_discount_enabled: false,
            ..proximity_off
        };
        let zeroed_own_slots = |o| {
            let f = SoccerConfigVector::from_snapshot_with(
                &snap,
                Team::Home,
                SoccerConfigComparison::PositionAgnostic,
                o,
            )
            .to_features();
            (0..CONFIG_PLAYERS_PER_TEAM)
                .filter(|slot| {
                    let base = slot * CONFIG_PER_PLAYER_FLOATS;
                    (0..8).all(|c| f[base + c] == 0.0)
                })
                .count()
        };
        let on = zeroed_own_slots(with_discount);
        let off = zeroed_own_slots(without_discount);
        assert!(
            on > off,
            "discount must zero deep own players (on {on}, off {off})"
        );
        assert!(on >= 1, "at least one deep own player should be discounted");
    }

    #[test]
    fn feature_width_is_stable() {
        let snap = sample_snapshot();
        let cfg = SoccerConfigVector::from_snapshot(&snap, Team::Home);
        assert_eq!(cfg.to_features().len(), CONFIG_FEATURE_DIM);
        assert_eq!(CONFIG_FEATURE_DIM_V1, 142);
        assert_eq!(CONFIG_FEATURE_DIM_V2, 164);
        assert_eq!(cfg.embedding().len(), SOCCER_MOMENT_EMBEDDING_DIM);
    }

    #[test]
    fn feature_vector_contains_ball_altitude_jerk_and_holder_flags() {
        let mut snap = sample_snapshot();
        let holder = snap
            .players
            .iter()
            .find(|player| player.team == Team::Home)
            .expect("home player")
            .id;
        snap.ball.holder = Some(holder);
        snap.ball.altitude_yards = 6.0;
        snap.ball.jerk = Vec2::new(18.0, -36.0);

        let features = SoccerConfigVector::from_snapshot(&snap, Team::Home).to_features();
        let cfg = SoccerConfigVector::from_snapshot(&snap, Team::Home);
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
        assert_eq!(features[ball_offset + 7], cfg.ball_jerk[0]);
        assert_eq!(features[ball_offset + 8], cfg.ball_jerk[1]);
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
        // The whole point is to relate a decision to how it turned out: at least
        // one captured moment must carry a non-zero n-step outcome, or the
        // similarity→outcome retrieval signal is dead. (Guards the silent-zero
        // regression where the outcome join lost its transition stream.)
        assert!(
            moments.iter().any(|m| m.nstep_return != 0.0),
            "captured moments must carry a realised outcome, not all zeros"
        );
    }

    /// Capture must be self-sufficient: with full-game learning OFF but
    /// `retrieval.capture_enabled` ON (a dedicated corpus-builder), moments are
    /// still captured AND still carry realised outcomes — the n-step join retains
    /// its own transition stream rather than free-riding on full-game learning.
    #[test]
    fn config_moments_capture_without_full_game_learning() {
        let mut cfg = MatchConfig::default();
        cfg.full_game_learning_enabled = false;
        cfg.retrieval.capture_enabled = true;
        cfg.retrieval.outcome_horizon = 10;
        let mut sim = SoccerMatch::default_11v11(cfg);
        for _ in 0..120 {
            sim.run_time_step();
        }
        let moments = sim.config_moments();
        assert!(
            !moments.is_empty(),
            "capture must work with full-game learning disabled"
        );
        assert!(
            moments.iter().any(|m| m.nstep_return != 0.0),
            "outcomes must be populated even without full-game learning"
        );
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
