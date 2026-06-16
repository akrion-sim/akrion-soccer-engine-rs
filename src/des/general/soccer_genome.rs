//! Team tactical genome — the heritable STYLE the tournament GA evolves.
//!
//! This is a **leaf** module (depends only on `serde`/`std`) so both the engine
//! (`soccer`) and the tournament/seeding layer (`tournament`) can read it without
//! a circular dependency: the engine consumes a team's genome to bias positioning,
//! pressing, and on-ball decisions, while the GA permutes/hybridizes it.

use serde::{Deserialize, Serialize};

/// Pitch discretization for MDP/POMDP decisions and per-position anchors: 12
/// lanes (columns) wide × 24 rows long. MPC works in continuous coordinates;
/// these grid cells are the discrete state basis for the tabular / POMDP policy.
pub const PITCH_GENOME_LANES: u8 = 12;
pub const PITCH_GENOME_ROWS: u8 = 24;
pub const DEF_LINE_MIN_DISTANCE_CHOICES_YARDS: [f64; 3] = [1.0, 2.0, 3.0];
pub const DEF_LINE_MAX_DISTANCE_CHOICES_YARDS: [f64; 5] = [20.0, 23.0, 25.0, 27.0, 29.0];
pub const DEF_LINE_BAND_PERMUTATION_COUNT: usize =
    DEF_LINE_MIN_DISTANCE_CHOICES_YARDS.len() * DEF_LINE_MAX_DISTANCE_CHOICES_YARDS.len();

/// Deterministic RNG (splitmix64) so a seed reproduces the same genome — genome
/// diversity, like net diversity, must stay reproducible.
pub struct GenomeRng {
    state: u64,
}

impl GenomeRng {
    pub fn new(seed: u64) -> Self {
        GenomeRng {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
    fn coin(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }
    fn chance(&mut self, p: f64) -> bool {
        self.unit() < p
    }
    /// A uniformly-random index into `0..len` (0 when `len == 0`). Public so the
    /// seeder can pick parents from a pool.
    pub fn index(&mut self, len: usize) -> usize {
        if len == 0 {
            0
        } else {
            (self.next_u64() % len as u64) as usize
        }
    }
}

/// Coarse role of a genome anchor slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GenomeRole {
    Goalkeeper,
    Defender,
    Midfielder,
    Forward,
}

/// Outfield shape preset. The per-position `anchors` carry the actual lane/row
/// layout (a preset seeds them).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TeamFormation {
    F433,
    F442,
}

/// How the first defender engages the ball carrier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DefenderEngagement {
    /// Delay / contain — jockey and shepherd rather than diving in.
    Contain,
    /// Press — close down and contest the ball.
    Press,
}

/// One of the 11 slots' preferred grid cell + role. `lane` in
/// `0..PITCH_GENOME_LANES`, `row` in `0..PITCH_GENOME_ROWS` (row 0 = own goal
/// line, increasing toward the opponent goal).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PositionAnchor {
    pub role: GenomeRole,
    pub lane: u8,
    pub row: u8,
}

/// The heritable tactical "preferences" that distinguish one team's STYLE from
/// another's — the variables the tournament's genetic search permutes and
/// hybridizes (alongside the neural value/critic weights). Engine consumers read
/// these to bias positioning, pressing, and on-ball decisions (wired incrementally).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SoccerTeamGenome {
    pub formation: TeamFormation,
    /// 11 slots: keeper + 10 outfield, each with a grid-cell anchor.
    pub anchors: Vec<PositionAnchor>,
    /// Back-four line: min / max standoff distance from the ball (yards), selected
    /// from the discrete GA search sets above.
    pub def_line_min_dist_from_ball: f64,
    pub def_line_max_dist_from_ball: f64,
    /// Midfield band's gap behind the back four, by field third (yards). A bigger
    /// opponent-half gap = a higher, more stretched midfield.
    pub mid_gap_from_back4_own_half: f64,
    pub mid_gap_from_back4_opp_half: f64,
    /// Front line presses as a coordinated unit (all at once) vs individually.
    pub striker_press_synchronized: bool,
    /// Baseline pressing urgency in `[0,1]` (scaled at decision time by the grid
    /// scenario — ball cell + player cells).
    pub press_urgency: f64,
    /// Willingness to keep playing out under pressure vs clear, in `[0,1]` (scaled
    /// by field position at decision time).
    pub pass_willingness_under_pressure: f64,
    /// Relative preference over pass lengths `{5, 8, 10, 12, 14}` yards (weights).
    pub pass_distance_priority: [f64; 5],
    /// Allow scooped / lofted passes.
    pub use_scoop_pass: bool,
    /// Goalkeeper resting line height, `[0,1]`: 0 = hug the goal-line (max reaction
    /// time), 1 = a high sweeper-keeper line. 0.5 is neutral (a balanced
    /// "no-man's-land" line). Together with `gk_commit_aggression` this spans the
    /// four keeper strategies (goal-line / off-line angle-closing / no-man's-land /
    /// sweeper) as an evolvable 2-D style space. `serde(default)` so genomes
    /// persisted before this field (e.g. older `sweeper_keeper` rows) still load.
    #[serde(default = "default_neutral_gk_gene")]
    pub gk_line_height: f64,
    /// Goalkeeper commitment aggression, `[0,1]`: 0 = reaction-first (stay, let
    /// defenders win loose balls, hold the angle), 1 = aggressive (come out to
    /// close angles, charge 50/50s). 0.5 is neutral.
    #[serde(default = "default_neutral_gk_gene")]
    pub gk_commit_aggression: f64,
    /// First-defender engagement style.
    pub defender_engagement: DefenderEngagement,
    /// When a DEFENDER is in possession: the extra standoff (yards) to keep from
    /// the nearest opponent before being pressured into a decision. Defenders on
    /// the ball should hold a safer cushion than an attacker would.
    pub defender_on_ball_opponent_distance: f64,
    /// When a DEFENDER is in possession: how urgently to release the ball, in
    /// `[0,1]` (higher = pass sooner). Defenders should move it on quicker rather
    /// than carry it into pressure.
    pub defender_on_ball_pass_urgency: f64,
}

/// Neutral default for the keeper-strategy genes (serde fallback for older genomes).
fn default_neutral_gk_gene() -> f64 {
    0.5
}

fn formation_anchors(formation: TeamFormation) -> Vec<PositionAnchor> {
    let a = |role, lane, row| PositionAnchor { role, lane, row };
    use GenomeRole::*;
    match formation {
        // 12 lanes (0..11), 24 rows (0 = own line). Keeper deep-centre, a back
        // four, then either a 3-3-1-ish midfield+front or two banks of four.
        TeamFormation::F433 => vec![
            a(Goalkeeper, 6, 1),
            a(Defender, 2, 5),
            a(Defender, 4, 5),
            a(Defender, 8, 5),
            a(Defender, 10, 5),
            a(Midfielder, 3, 11),
            a(Midfielder, 6, 11),
            a(Midfielder, 9, 11),
            a(Forward, 2, 18),
            a(Forward, 6, 19),
            a(Forward, 10, 18),
        ],
        TeamFormation::F442 => vec![
            a(Goalkeeper, 6, 1),
            a(Defender, 2, 5),
            a(Defender, 4, 5),
            a(Defender, 8, 5),
            a(Defender, 10, 5),
            a(Midfielder, 2, 12),
            a(Midfielder, 5, 11),
            a(Midfielder, 8, 11),
            a(Midfielder, 11, 12),
            a(Forward, 4, 18),
            a(Forward, 8, 18),
        ],
    }
}

impl Default for SoccerTeamGenome {
    fn default() -> Self {
        SoccerTeamGenome {
            formation: TeamFormation::F433,
            anchors: formation_anchors(TeamFormation::F433),
            def_line_min_dist_from_ball: 2.0,
            def_line_max_dist_from_ball: 25.0,
            mid_gap_from_back4_own_half: 8.0,
            mid_gap_from_back4_opp_half: 12.0,
            striker_press_synchronized: true,
            press_urgency: 0.5,
            pass_willingness_under_pressure: 0.5,
            pass_distance_priority: [1.0, 1.0, 1.0, 1.0, 1.0],
            use_scoop_pass: true,
            gk_line_height: 0.5,
            gk_commit_aggression: 0.5,
            defender_engagement: DefenderEngagement::Press,
            defender_on_ball_opponent_distance: 3.0,
            defender_on_ball_pass_urgency: 0.5,
        }
    }
}

impl SoccerTeamGenome {
    /// Clamp every gene back into a legal range, keeping anchors on the grid and
    /// `min <= max` for the defensive line. Public + idempotent so it can also
    /// repair genomes decoded from Postgres (which may be corrupt, out of range,
    /// or pre-date a field) before they reach the engine or the GA.
    pub fn sanitize(&mut self) {
        // Exactly 11 slots (keeper + 10 outfield) so positional consumers can index
        // anchors safely; repair a wrong-length vec from the current formation.
        if self.anchors.len() != 11 {
            self.anchors = formation_anchors(self.formation);
        }
        for anchor in &mut self.anchors {
            anchor.lane = anchor.lane.min(PITCH_GENOME_LANES - 1);
            anchor.row = anchor.row.min(PITCH_GENOME_ROWS - 1);
        }
        // Replace any non-finite numeric gene with its neutral default so a bad DB
        // row can never inject NaN/inf into positioning math.
        let neutral = SoccerTeamGenome::default();
        if !self.def_line_min_dist_from_ball.is_finite() {
            self.def_line_min_dist_from_ball = neutral.def_line_min_dist_from_ball;
        }
        if !self.def_line_max_dist_from_ball.is_finite() {
            self.def_line_max_dist_from_ball = neutral.def_line_max_dist_from_ball;
        }
        if !self.mid_gap_from_back4_own_half.is_finite() {
            self.mid_gap_from_back4_own_half = neutral.mid_gap_from_back4_own_half;
        }
        if !self.mid_gap_from_back4_opp_half.is_finite() {
            self.mid_gap_from_back4_opp_half = neutral.mid_gap_from_back4_opp_half;
        }
        if !self.press_urgency.is_finite() {
            self.press_urgency = neutral.press_urgency;
        }
        if !self.pass_willingness_under_pressure.is_finite() {
            self.pass_willingness_under_pressure = neutral.pass_willingness_under_pressure;
        }
        for weight in &mut self.pass_distance_priority {
            if !weight.is_finite() {
                *weight = 1.0;
            }
        }
        if !self.defender_on_ball_opponent_distance.is_finite() {
            self.defender_on_ball_opponent_distance = neutral.defender_on_ball_opponent_distance;
        }
        if !self.defender_on_ball_pass_urgency.is_finite() {
            self.defender_on_ball_pass_urgency = neutral.defender_on_ball_pass_urgency;
        }
        self.def_line_min_dist_from_ball = nearest_choice(
            self.def_line_min_dist_from_ball,
            &DEF_LINE_MIN_DISTANCE_CHOICES_YARDS,
            neutral.def_line_min_dist_from_ball,
        );
        self.def_line_max_dist_from_ball = nearest_choice(
            self.def_line_max_dist_from_ball,
            &DEF_LINE_MAX_DISTANCE_CHOICES_YARDS,
            neutral.def_line_max_dist_from_ball,
        )
        .max(self.def_line_min_dist_from_ball);
        self.mid_gap_from_back4_own_half = self.mid_gap_from_back4_own_half.clamp(3.0, 30.0);
        self.mid_gap_from_back4_opp_half = self.mid_gap_from_back4_opp_half.clamp(3.0, 30.0);
        self.press_urgency = self.press_urgency.clamp(0.0, 1.0);
        self.pass_willingness_under_pressure = self.pass_willingness_under_pressure.clamp(0.0, 1.0);
        for weight in &mut self.pass_distance_priority {
            *weight = weight.clamp(0.0, 1.0);
        }
        if self.pass_distance_priority.iter().all(|w| *w <= 0.0) {
            self.pass_distance_priority = [1.0; 5];
        }
        self.defender_on_ball_opponent_distance =
            self.defender_on_ball_opponent_distance.clamp(0.0, 12.0);
        self.defender_on_ball_pass_urgency = self.defender_on_ball_pass_urgency.clamp(0.0, 1.0);
        if !self.gk_line_height.is_finite() {
            self.gk_line_height = 0.5;
        }
        if !self.gk_commit_aggression.is_finite() {
            self.gk_commit_aggression = 0.5;
        }
        self.gk_line_height = self.gk_line_height.clamp(0.0, 1.0);
        self.gk_commit_aggression = self.gk_commit_aggression.clamp(0.0, 1.0);
    }

    /// The dominant keeper strategy this genome leans toward, for logging/insight.
    pub fn gk_strategy_label(&self) -> &'static str {
        let high_line = self.gk_line_height >= 0.66;
        let low_line = self.gk_line_height <= 0.33;
        let aggressive = self.gk_commit_aggression >= 0.6;
        if high_line && aggressive {
            "sweeper"
        } else if low_line && aggressive {
            "off-line-angle-closing"
        } else if low_line {
            "goal-line-max-reaction"
        } else {
            "no-mans-land-balanced"
        }
    }

    pub fn defensive_line_band_yards(&self) -> (f64, f64) {
        let neutral = SoccerTeamGenome::default();
        let min_gap = nearest_choice(
            self.def_line_min_dist_from_ball,
            &DEF_LINE_MIN_DISTANCE_CHOICES_YARDS,
            neutral.def_line_min_dist_from_ball,
        );
        let max_gap = nearest_choice(
            self.def_line_max_dist_from_ball,
            &DEF_LINE_MAX_DISTANCE_CHOICES_YARDS,
            neutral.def_line_max_dist_from_ball,
        )
        .max(min_gap);
        (min_gap, max_gap)
    }

    pub fn defensive_line_band_for_permutation(index: usize) -> (f64, f64) {
        let min_index = index % DEF_LINE_MIN_DISTANCE_CHOICES_YARDS.len();
        let max_index = (index / DEF_LINE_MIN_DISTANCE_CHOICES_YARDS.len())
            % DEF_LINE_MAX_DISTANCE_CHOICES_YARDS.len();
        (
            DEF_LINE_MIN_DISTANCE_CHOICES_YARDS[min_index],
            DEF_LINE_MAX_DISTANCE_CHOICES_YARDS[max_index],
        )
    }

    pub fn apply_defensive_line_band_permutation(&mut self, index: usize) {
        let (min_gap, max_gap) = Self::defensive_line_band_for_permutation(index);
        self.def_line_min_dist_from_ball = min_gap;
        self.def_line_max_dist_from_ball = max_gap;
    }

    /// A fresh, randomly-styled team genome (cold-start diversity).
    pub fn random(rng: &mut GenomeRng) -> Self {
        let formation = if rng.coin() {
            TeamFormation::F433
        } else {
            TeamFormation::F442
        };
        let mut anchors = formation_anchors(formation);
        // Jitter each slot by up to ±1 lane / ±2 rows so even same-formation teams
        // line up a little differently.
        for anchor in &mut anchors {
            anchor.lane = jitter_u8(anchor.lane, 1, rng);
            anchor.row = jitter_u8(anchor.row, 2, rng);
        }
        let mut genome = SoccerTeamGenome {
            formation,
            anchors,
            def_line_min_dist_from_ball: random_choice(&DEF_LINE_MIN_DISTANCE_CHOICES_YARDS, rng),
            def_line_max_dist_from_ball: random_choice(&DEF_LINE_MAX_DISTANCE_CHOICES_YARDS, rng),
            mid_gap_from_back4_own_half: rng.range(5.0, 14.0),
            mid_gap_from_back4_opp_half: rng.range(8.0, 20.0),
            striker_press_synchronized: rng.coin(),
            press_urgency: rng.unit(),
            pass_willingness_under_pressure: rng.unit(),
            pass_distance_priority: [rng.unit(), rng.unit(), rng.unit(), rng.unit(), rng.unit()],
            use_scoop_pass: rng.coin(),
            gk_line_height: rng.unit(),
            gk_commit_aggression: rng.unit(),
            defender_engagement: if rng.coin() {
                DefenderEngagement::Contain
            } else {
                DefenderEngagement::Press
            },
            defender_on_ball_opponent_distance: rng.range(1.5, 7.0),
            defender_on_ball_pass_urgency: rng.unit(),
        };
        genome.sanitize();
        genome
    }

    /// Uniform-crossover two parents gene by gene (a random hybrid).
    pub fn crossover(a: &SoccerTeamGenome, b: &SoccerTeamGenome, rng: &mut GenomeRng) -> Self {
        let formation = if rng.coin() { a.formation } else { b.formation };
        // Anchors: take each slot from one parent (fall back to `a` if the pools
        // differ in length).
        let anchors = a
            .anchors
            .iter()
            .enumerate()
            .map(|(i, anchor)| {
                if rng.coin() {
                    *anchor
                } else {
                    *b.anchors.get(i).unwrap_or(anchor)
                }
            })
            .collect();
        let pick = |x: f64, y: f64, rng: &mut GenomeRng| if rng.coin() { x } else { y };
        let pick_b = |x: bool, y: bool, rng: &mut GenomeRng| if rng.coin() { x } else { y };
        let mut priority = [0.0; 5];
        for (i, slot) in priority.iter_mut().enumerate() {
            *slot = pick(
                a.pass_distance_priority[i],
                b.pass_distance_priority[i],
                rng,
            );
        }
        let mut genome = SoccerTeamGenome {
            formation,
            anchors,
            def_line_min_dist_from_ball: pick(
                a.def_line_min_dist_from_ball,
                b.def_line_min_dist_from_ball,
                rng,
            ),
            def_line_max_dist_from_ball: pick(
                a.def_line_max_dist_from_ball,
                b.def_line_max_dist_from_ball,
                rng,
            ),
            mid_gap_from_back4_own_half: pick(
                a.mid_gap_from_back4_own_half,
                b.mid_gap_from_back4_own_half,
                rng,
            ),
            mid_gap_from_back4_opp_half: pick(
                a.mid_gap_from_back4_opp_half,
                b.mid_gap_from_back4_opp_half,
                rng,
            ),
            striker_press_synchronized: pick_b(
                a.striker_press_synchronized,
                b.striker_press_synchronized,
                rng,
            ),
            press_urgency: pick(a.press_urgency, b.press_urgency, rng),
            pass_willingness_under_pressure: pick(
                a.pass_willingness_under_pressure,
                b.pass_willingness_under_pressure,
                rng,
            ),
            pass_distance_priority: priority,
            use_scoop_pass: pick_b(a.use_scoop_pass, b.use_scoop_pass, rng),
            gk_line_height: pick(a.gk_line_height, b.gk_line_height, rng),
            gk_commit_aggression: pick(a.gk_commit_aggression, b.gk_commit_aggression, rng),
            defender_engagement: if rng.coin() {
                a.defender_engagement
            } else {
                b.defender_engagement
            },
            defender_on_ball_opponent_distance: pick(
                a.defender_on_ball_opponent_distance,
                b.defender_on_ball_opponent_distance,
                rng,
            ),
            defender_on_ball_pass_urgency: pick(
                a.defender_on_ball_pass_urgency,
                b.defender_on_ball_pass_urgency,
                rng,
            ),
        };
        genome.sanitize();
        genome
    }

    /// Small random drift (run after crossover) so hybrids explore beyond their
    /// parents. `rate` is the per-gene mutation probability.
    pub fn mutate(&mut self, rng: &mut GenomeRng, rate: f64) {
        if rng.chance(rate) {
            self.formation = if rng.coin() {
                TeamFormation::F433
            } else {
                TeamFormation::F442
            };
            self.anchors = formation_anchors(self.formation);
        }
        for anchor in &mut self.anchors {
            if rng.chance(rate) {
                anchor.lane = jitter_u8(anchor.lane, 1, rng);
            }
            if rng.chance(rate) {
                anchor.row = jitter_u8(anchor.row, 2, rng);
            }
        }
        if rng.chance(rate) {
            self.def_line_min_dist_from_ball =
                random_choice(&DEF_LINE_MIN_DISTANCE_CHOICES_YARDS, rng);
        }
        if rng.chance(rate) {
            self.def_line_max_dist_from_ball =
                random_choice(&DEF_LINE_MAX_DISTANCE_CHOICES_YARDS, rng);
        }
        if rng.chance(rate) {
            self.mid_gap_from_back4_own_half += rng.range(-2.0, 2.0);
        }
        if rng.chance(rate) {
            self.mid_gap_from_back4_opp_half += rng.range(-2.0, 2.0);
        }
        if rng.chance(rate) {
            self.press_urgency += rng.range(-0.15, 0.15);
        }
        if rng.chance(rate) {
            self.pass_willingness_under_pressure += rng.range(-0.15, 0.15);
        }
        for weight in &mut self.pass_distance_priority {
            if rng.chance(rate) {
                *weight += rng.range(-0.2, 0.2);
            }
        }
        if rng.chance(rate) {
            self.defender_on_ball_opponent_distance += rng.range(-1.5, 1.5);
        }
        if rng.chance(rate) {
            self.defender_on_ball_pass_urgency += rng.range(-0.15, 0.15);
        }
        if rng.chance(rate) {
            self.striker_press_synchronized = !self.striker_press_synchronized;
        }
        if rng.chance(rate) {
            self.use_scoop_pass = !self.use_scoop_pass;
        }
        if rng.chance(rate) {
            self.gk_line_height += rng.range(-0.15, 0.15);
        }
        if rng.chance(rate) {
            self.gk_commit_aggression += rng.range(-0.15, 0.15);
        }
        if rng.chance(rate) {
            self.defender_engagement = match self.defender_engagement {
                DefenderEngagement::Contain => DefenderEngagement::Press,
                DefenderEngagement::Press => DefenderEngagement::Contain,
            };
        }
        self.sanitize();
    }
}

fn jitter_u8(value: u8, span: u8, rng: &mut GenomeRng) -> u8 {
    let delta = (rng.next_u64() % (2 * span as u64 + 1)) as i64 - span as i64;
    (value as i64 + delta).clamp(0, u8::MAX as i64) as u8
}

fn random_choice(choices: &[f64], rng: &mut GenomeRng) -> f64 {
    choices
        .get(rng.index(choices.len()))
        .copied()
        .unwrap_or(0.0)
}

fn nearest_choice(value: f64, choices: &[f64], fallback: f64) -> f64 {
    if choices.is_empty() {
        return fallback;
    }
    let value = if value.is_finite() { value } else { fallback };
    choices
        .iter()
        .copied()
        .min_by(|a, b| {
            (value - *a)
                .abs()
                .partial_cmp(&(value - *b).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(fallback)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn genome_random_is_reproducible_and_distinct() {
        let a = SoccerTeamGenome::random(&mut GenomeRng::new(1));
        let a2 = SoccerTeamGenome::random(&mut GenomeRng::new(1));
        let b = SoccerTeamGenome::random(&mut GenomeRng::new(2));
        assert_eq!(a.press_urgency, a2.press_urgency);
        assert_eq!(a.anchors, a2.anchors);
        assert!(a.press_urgency != b.press_urgency || a.formation != b.formation);
        assert!(DEF_LINE_MIN_DISTANCE_CHOICES_YARDS.contains(&a.def_line_min_dist_from_ball));
        assert!(DEF_LINE_MAX_DISTANCE_CHOICES_YARDS.contains(&a.def_line_max_dist_from_ball));
        for anchor in &a.anchors {
            assert!(anchor.lane < PITCH_GENOME_LANES && anchor.row < PITCH_GENOME_ROWS);
        }
    }

    #[test]
    fn defensive_line_band_permutation_cycles_all_discrete_combinations() {
        let mut seen = std::collections::HashSet::new();
        for index in 0..DEF_LINE_BAND_PERMUTATION_COUNT {
            let band = SoccerTeamGenome::defensive_line_band_for_permutation(index);
            assert!(DEF_LINE_MIN_DISTANCE_CHOICES_YARDS.contains(&band.0));
            assert!(DEF_LINE_MAX_DISTANCE_CHOICES_YARDS.contains(&band.1));
            assert!(seen.insert((band.0 as i32, band.1 as i32)));
        }

        assert_eq!(seen.len(), DEF_LINE_BAND_PERMUTATION_COUNT);
        assert_eq!(
            SoccerTeamGenome::defensive_line_band_for_permutation(DEF_LINE_BAND_PERMUTATION_COUNT),
            SoccerTeamGenome::defensive_line_band_for_permutation(0)
        );
    }

    #[test]
    fn genome_crossover_takes_genes_from_parents_and_survives_serde() {
        let a = SoccerTeamGenome {
            formation: TeamFormation::F433,
            use_scoop_pass: true,
            gk_line_height: 0.1,
            ..SoccerTeamGenome::default()
        };
        let b = SoccerTeamGenome {
            formation: TeamFormation::F442,
            use_scoop_pass: false,
            gk_line_height: 0.9,
            press_urgency: 0.9,
            ..SoccerTeamGenome::default()
        };
        let child = SoccerTeamGenome::crossover(&a, &b, &mut GenomeRng::new(7));
        assert!(
            child.use_scoop_pass == a.use_scoop_pass || child.use_scoop_pass == b.use_scoop_pass
        );
        assert!(
            child.gk_line_height == a.gk_line_height || child.gk_line_height == b.gk_line_height
        );
        let json = serde_json::to_value(&child).expect("genome serializes");
        let back: SoccerTeamGenome = serde_json::from_value(json).expect("genome deserializes");
        assert_eq!(back.formation, child.formation);
        assert_eq!(back.anchors, child.anchors);
        assert_eq!(
            back.def_line_max_dist_from_ball,
            child.def_line_max_dist_from_ball
        );
    }

    #[test]
    fn sanitize_repairs_a_corrupt_or_out_of_range_genome() {
        // Simulates a bad/old Postgres row reaching the engine.
        let mut g = SoccerTeamGenome::default();
        g.anchors.clear(); // wrong slot count
        g.press_urgency = f64::NAN;
        g.def_line_max_dist_from_ball = f64::INFINITY;
        g.defender_on_ball_opponent_distance = -5.0;
        g.defender_on_ball_pass_urgency = 9.0;
        g.pass_distance_priority = [f64::NAN, 0.0, 0.0, 0.0, 0.0];
        g.sanitize();
        assert_eq!(g.anchors.len(), 11);
        assert!((0.0..=1.0).contains(&g.press_urgency));
        assert!(g.def_line_max_dist_from_ball.is_finite());
        assert!(g.def_line_max_dist_from_ball >= g.def_line_min_dist_from_ball);
        assert!(DEF_LINE_MIN_DISTANCE_CHOICES_YARDS.contains(&g.def_line_min_dist_from_ball));
        assert!(DEF_LINE_MAX_DISTANCE_CHOICES_YARDS.contains(&g.def_line_max_dist_from_ball));
        assert!((0.0..=12.0).contains(&g.defender_on_ball_opponent_distance));
        assert!((0.0..=1.0).contains(&g.defender_on_ball_pass_urgency));
        assert!(g.pass_distance_priority.iter().all(|w| w.is_finite()));
        for anchor in &g.anchors {
            assert!(anchor.lane < PITCH_GENOME_LANES && anchor.row < PITCH_GENOME_ROWS);
        }
    }

    #[test]
    fn genome_mutate_respects_bounds() {
        let mut g = SoccerTeamGenome::default();
        let mut rng = GenomeRng::new(99);
        for _ in 0..50 {
            g.mutate(&mut rng, 1.0);
        }
        assert!(g.press_urgency >= 0.0 && g.press_urgency <= 1.0);
        assert!(g.def_line_max_dist_from_ball >= g.def_line_min_dist_from_ball);
        assert!(DEF_LINE_MIN_DISTANCE_CHOICES_YARDS.contains(&g.def_line_min_dist_from_ball));
        assert!(DEF_LINE_MAX_DISTANCE_CHOICES_YARDS.contains(&g.def_line_max_dist_from_ball));
        for anchor in &g.anchors {
            assert!(anchor.lane < PITCH_GENOME_LANES && anchor.row < PITCH_GENOME_ROWS);
        }
    }
}
