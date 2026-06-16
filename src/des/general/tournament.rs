//! Learning soccer **tournament**: many distinct team brains compete in a
//! World-Cup-style format (group round-robins → single-elimination knockout) and
//! a champion is crowned. Teams can *learn as they go* — each match hands its
//! updated brains back so a team improves across its run — under three flexible
//! learning patterns:
//!
//! - [`TournamentLearningMode::Frozen`] — nobody learns; a pure, deterministic
//!   evaluation of the brains exactly as they entered.
//! - [`TournamentLearningMode::BiLearning`] — both teams learn from every match
//!   (the whole field improves together).
//! - [`TournamentLearningMode::FrozenOpponent`] — the home side learns while the
//!   away side is frozen; pair it with a double round-robin so every team gets
//!   equal learning reps against stable opponents.
//!
//! The orchestration here is engine-agnostic: it schedules, plays (via a
//! [`TournamentMatchRunner`]), ranks with FIFA-style tiebreakers, folds the
//! knockout bracket, and reports the champion. The real runner that drives the
//! 2D [`SoccerMatch`] with genuine per-team neural brains lives at the bottom
//! ([`EngineMatchRunner`]); a deterministic [`StrengthMatchRunner`] backs the
//! tests and quick what-ifs.
//!
//! Everything is seeded and deterministic: the same teams + seed + runner
//! always produce the same bracket and champion.

use crate::des::general::soccer::{
    MatchConfig, SoccerMatch, SoccerNeuralLearningBackend, SoccerNeuralNetworkSnapshot,
    SoccerQPolicy, SoccerQPolicyOptions, SoccerTeamQPolicies, Team,
};
use std::collections::HashMap;
use std::time::Duration;

/// Tiny deterministic RNG (mulberry32) so the module owns its reproducibility
/// without threading the engine's `RandomSource` trait through scheduling and
/// shootouts. Same seed ⇒ same sequence.
#[derive(Clone, Debug)]
struct TournamentRng {
    state: u32,
}

impl TournamentRng {
    fn new(seed: u32) -> Self {
        // Avoid the degenerate all-zero state.
        TournamentRng {
            state: seed ^ 0x9e37_79b9,
        }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_add(0x6d2b_79f5);
        let mut t = self.state;
        t = (t ^ (t >> 15)).wrapping_mul(t | 1);
        t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
        t ^ (t >> 14)
    }

    fn next_unit(&mut self) -> f64 {
        f64::from(self.next_u32()) / f64::from(u32::MAX)
    }

    /// Deterministic coin in `[0,1)` biased by `bias` (a skill differential in
    /// roughly `[-1, 1]`): returns true ("home wins") with probability
    /// `0.5 + 0.5*bias` clamped to a sane band so the underdog always has a shot.
    fn biased_coin(&mut self, bias: f64) -> bool {
        let p = (0.5 + 0.5 * bias).clamp(0.05, 0.95);
        self.next_unit() < p
    }
}

/// Which side of a match.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatchSide {
    Home,
    Away,
}

/// How team brains update over the course of the tournament.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TournamentLearningMode {
    /// No brain learns; brains play exactly as they entered (deterministic eval).
    Frozen,
    /// Both teams learn from every match (learn-as-they-go).
    BiLearning,
    /// Home learns, away is frozen each match. Best paired with a double
    /// round-robin so learn/frozen reps balance out across the field.
    FrozenOpponent,
}

impl TournamentLearningMode {
    /// `(home_learns, away_learns)` for a single match under this mode.
    pub fn per_match(self) -> (bool, bool) {
        match self {
            TournamentLearningMode::Frozen => (false, false),
            TournamentLearningMode::BiLearning => (true, true),
            TournamentLearningMode::FrozenOpponent => (true, false),
        }
    }
}

/// Stage of the competition a match belongs to (drives draw handling: group
/// matches may end level, knockout matches must produce a winner).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TournamentStage {
    Group,
    Knockout { remaining: usize },
    ThirdPlace,
    Final,
}

impl TournamentStage {
    /// Human-readable round label ("Group", "Round of 16", "Quarter-final", …).
    pub fn label(self) -> String {
        match self {
            TournamentStage::Group => "Group".to_string(),
            TournamentStage::ThirdPlace => "Third-place play-off".to_string(),
            TournamentStage::Final => "Final".to_string(),
            TournamentStage::Knockout { remaining } => match remaining {
                2 => "Final".to_string(),
                4 => "Semi-final".to_string(),
                8 => "Quarter-final".to_string(),
                _ => format!("Round of {remaining}"),
            },
        }
    }
}

/// The persistent, learnable state of a competitor — carried from match to match
/// so a team improves as the tournament unfolds. The neural value/critic snapshot
/// is the real learner; tabular Q is intentionally not carried (it is a degenerate
/// per-match prior that bloats without converging).
#[derive(Clone, Debug, Default)]
pub struct TeamBrain {
    /// Trained value/critic network. `None` = start from a fresh net.
    pub neural: Option<SoccerNeuralNetworkSnapshot>,
    /// Per-team Q options (gamma etc.); carried so brains can differ in horizon.
    pub options: SoccerQPolicyOptions,
    /// How many matches' worth of learning this brain has absorbed.
    pub matches_learned: u32,
    /// Cumulative gradient steps the brain has taken across the tournament.
    pub training_steps: u64,
    /// Heritable tactical "preferences" (formation, line/press/pass style, …) that
    /// distinguish this team's STYLE — permuted + hybridized by the tournament GA
    /// alongside the neural weights, and carried across matches (see
    /// [`SoccerTeamGenome`]).
    pub genome: SoccerTeamGenome,
}

impl TeamBrain {
    pub fn fresh() -> Self {
        TeamBrain::default()
    }

    pub fn fresh_with_seed(seed: u32, permutation_index: usize) -> Self {
        let mut rng = GenomeRng::new(
            u64::from(seed).rotate_left(32)
                ^ (permutation_index as u64).wrapping_mul(0x2545_F491_4F6C_DD1D),
        );
        let mut genome = SoccerTeamGenome::random(&mut rng);
        genome.apply_defensive_line_band_permutation(permutation_index);
        TeamBrain {
            genome,
            ..TeamBrain::default()
        }
    }

    pub fn from_snapshot(neural: SoccerNeuralNetworkSnapshot) -> Self {
        TeamBrain {
            neural: Some(neural),
            ..TeamBrain::default()
        }
    }
}

// The team tactical genome lives in the leaf module `soccer_genome` so the engine
// (which `tournament` depends on) can read it without a circular dependency.
// Re-exported here so the GA/seeding call sites keep their `tournament::...` paths.
pub use crate::des::general::soccer_genome::{
    DefenderEngagement, GenomeRng, GenomeRole, PositionAnchor, SoccerTeamGenome, TeamFormation,
    DEF_LINE_BAND_PERMUTATION_COUNT, DEF_LINE_MAX_DISTANCE_CHOICES_YARDS,
    DEF_LINE_MIN_DISTANCE_CHOICES_YARDS, PITCH_GENOME_LANES, PITCH_GENOME_ROWS,
};

/// Win/draw/loss + goals record, used for group standings and final reporting.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TeamRecord {
    pub played: u32,
    pub wins: u32,
    pub draws: u32,
    pub losses: u32,
    pub goals_for: u32,
    pub goals_against: u32,
}

impl TeamRecord {
    pub fn points(&self) -> u32 {
        self.wins * 3 + self.draws
    }

    pub fn goal_difference(&self) -> i32 {
        self.goals_for as i32 - self.goals_against as i32
    }

    fn record(&mut self, scored: u32, conceded: u32) {
        self.played += 1;
        self.goals_for += scored;
        self.goals_against += conceded;
        match scored.cmp(&conceded) {
            std::cmp::Ordering::Greater => self.wins += 1,
            std::cmp::Ordering::Less => self.losses += 1,
            std::cmp::Ordering::Equal => self.draws += 1,
        }
    }
}

/// A competitor: identity + its (evolving) brain + its tournament record.
#[derive(Clone, Debug)]
pub struct TournamentTeam {
    pub id: usize,
    pub name: String,
    pub seed: u32,
    pub brain: TeamBrain,
    pub record: TeamRecord,
}

impl TournamentTeam {
    pub fn new(id: usize, name: impl Into<String>, seed: u32, brain: TeamBrain) -> Self {
        TournamentTeam {
            id,
            name: name.into(),
            seed,
            brain,
            record: TeamRecord::default(),
        }
    }
}

/// Competition shape. Defaults to a doubled-World-Cup: 128 teams, 32 groups of 4,
/// top 2 advance → a 64-team single-elimination knockout.
#[derive(Clone, Copy, Debug)]
pub struct TournamentFormat {
    pub team_count: usize,
    pub group_size: usize,
    pub advancers_per_group: usize,
    /// Groups play every pairing twice (home and away) — recommended with
    /// [`TournamentLearningMode::FrozenOpponent`] for balanced learning reps.
    pub double_round_robin: bool,
    /// Play a third-place match between the losing semi-finalists.
    pub third_place_match: bool,
}

impl Default for TournamentFormat {
    fn default() -> Self {
        TournamentFormat {
            team_count: 128,
            group_size: 4,
            advancers_per_group: 2,
            double_round_robin: false,
            third_place_match: true,
        }
    }
}

impl TournamentFormat {
    pub fn group_count(&self) -> usize {
        self.team_count / self.group_size
    }

    pub fn knockout_team_count(&self) -> usize {
        self.group_count() * self.advancers_per_group
    }

    /// Validate the shape: even groups, valid advancer count, and a knockout
    /// field that is a non-trivial power of two (so the bracket folds cleanly).
    pub fn validate(&self) -> Result<(), String> {
        if self.group_size < 2 {
            return Err("group_size must be at least 2".to_string());
        }
        if self.team_count == 0 || self.team_count % self.group_size != 0 {
            return Err(format!(
                "team_count ({}) must be a positive multiple of group_size ({})",
                self.team_count, self.group_size
            ));
        }
        if self.advancers_per_group == 0 || self.advancers_per_group > self.group_size {
            return Err(format!(
                "advancers_per_group ({}) must be in 1..=group_size ({})",
                self.advancers_per_group, self.group_size
            ));
        }
        let ko = self.knockout_team_count();
        if ko < 2 || !ko.is_power_of_two() {
            return Err(format!(
                "knockout field ({ko}) must be a power of two ≥ 2; adjust groups/advancers"
            ));
        }
        Ok(())
    }
}

/// Per-match parameters handed to the runner.
#[derive(Clone, Debug)]
pub struct TournamentMatchContext {
    pub stage: TournamentStage,
    pub round_index: usize,
    pub match_index: usize,
    pub seed: u32,
    pub home_id: usize,
    pub away_id: usize,
    pub home_name: String,
    pub away_name: String,
    /// Whether each side's brain trains during this match.
    pub home_learns: bool,
    pub away_learns: bool,
}

/// Result of one match: the score, the (possibly updated) brains to carry
/// forward, and per-team learning telemetry.
#[derive(Clone, Debug)]
pub struct MatchOutcome {
    pub home_goals: u32,
    pub away_goals: u32,
    pub home_brain: TeamBrain,
    pub away_brain: TeamBrain,
    pub home_training_steps: usize,
    pub away_training_steps: usize,
}

impl MatchOutcome {
    /// Winner by regulation score, or `None` if level.
    pub fn regulation_winner(&self) -> Option<MatchSide> {
        match self.home_goals.cmp(&self.away_goals) {
            std::cmp::Ordering::Greater => Some(MatchSide::Home),
            std::cmp::Ordering::Less => Some(MatchSide::Away),
            std::cmp::Ordering::Equal => None,
        }
    }
}

/// Plays a single match. Implementations range from the deterministic
/// [`StrengthMatchRunner`] (tests) to the full [`EngineMatchRunner`] (real sim).
pub trait TournamentMatchRunner {
    fn play(
        &mut self,
        ctx: &TournamentMatchContext,
        home: &TeamBrain,
        away: &TeamBrain,
    ) -> Result<MatchOutcome, String>;
}

/// One played fixture, recorded for the report.
#[derive(Clone, Debug)]
pub struct MatchReport {
    pub stage: TournamentStage,
    pub home_id: usize,
    pub away_id: usize,
    pub home_name: String,
    pub away_name: String,
    pub home_goals: u32,
    pub away_goals: u32,
    /// Set for knockout matches that finished level and went to a shootout.
    pub shootout_winner: Option<usize>,
    pub home_training_steps: usize,
    pub away_training_steps: usize,
}

impl MatchReport {
    /// The advancing/awarded winner's team id (after any shootout). `None` only
    /// for a drawn group match.
    pub fn winner_id(&self) -> Option<usize> {
        if let Some(winner) = self.shootout_winner {
            return Some(winner);
        }
        match self.home_goals.cmp(&self.away_goals) {
            std::cmp::Ordering::Greater => Some(self.home_id),
            std::cmp::Ordering::Less => Some(self.away_id),
            std::cmp::Ordering::Equal => None,
        }
    }
}

/// Final standing of one team within its group.
#[derive(Clone, Debug)]
pub struct GroupStanding {
    pub team_id: usize,
    pub name: String,
    pub record: TeamRecord,
    pub advanced: bool,
}

/// One group's final table (already ranked best-first).
#[derive(Clone, Debug)]
pub struct GroupTable {
    pub group_index: usize,
    pub standings: Vec<GroupStanding>,
}

/// The full outcome of a tournament.
#[derive(Clone, Debug)]
pub struct TournamentReport {
    pub format: TournamentFormat,
    pub learning_mode: TournamentLearningMode,
    pub group_tables: Vec<GroupTable>,
    pub matches: Vec<MatchReport>,
    pub champion_id: usize,
    pub runner_up_id: usize,
    pub third_place_id: Option<usize>,
    /// Final teams (with their evolved brains + cumulative records), by id.
    pub teams: Vec<TournamentTeam>,
}

impl TournamentReport {
    pub fn team(&self, id: usize) -> Option<&TournamentTeam> {
        self.teams.iter().find(|team| team.id == id)
    }

    pub fn champion(&self) -> &TournamentTeam {
        self.team(self.champion_id)
            .expect("champion id is always a known team")
    }

    /// Total matches played across the whole tournament.
    pub fn match_count(&self) -> usize {
        self.matches.len()
    }
}

/// The tournament orchestrator. Owns the teams + format + mode and drives a
/// [`TournamentMatchRunner`] through the group stage and knockout bracket.
pub struct Tournament {
    teams: Vec<TournamentTeam>,
    format: TournamentFormat,
    mode: TournamentLearningMode,
    seed: u32,
}

impl Tournament {
    /// Build a tournament. `teams.len()` must equal `format.team_count`.
    pub fn new(
        teams: Vec<TournamentTeam>,
        format: TournamentFormat,
        mode: TournamentLearningMode,
        seed: u32,
    ) -> Result<Self, String> {
        format.validate()?;
        if teams.len() != format.team_count {
            return Err(format!(
                "expected {} teams for the format, got {}",
                format.team_count,
                teams.len()
            ));
        }
        // Team ids must be unique — `team_index` resolves by id, so a duplicate
        // would silently alias two competitors into one slot.
        let mut ids: Vec<usize> = teams.iter().map(|team| team.id).collect();
        ids.sort_unstable();
        if ids.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err("team ids must be unique".to_string());
        }
        Ok(Tournament {
            teams,
            format,
            mode,
            seed,
        })
    }

    /// Convenience: a default-format tournament seeded with `team_count` fresh
    /// brains named "Team NNN".
    pub fn with_fresh_brains(
        format: TournamentFormat,
        mode: TournamentLearningMode,
        seed: u32,
    ) -> Result<Self, String> {
        format.validate()?;
        let teams = (0..format.team_count)
            .map(|id| {
                let team_seed = seed.wrapping_add(id as u32).wrapping_mul(2_654_435_761);
                TournamentTeam::new(
                    id,
                    format!("Team {:03}", id + 1),
                    team_seed,
                    TeamBrain::fresh_with_seed(team_seed, id),
                )
            })
            .collect();
        Tournament::new(teams, format, mode, seed)
    }

    fn team_index(&self, id: usize) -> usize {
        self.teams
            .iter()
            .position(|team| team.id == id)
            .expect("team id present")
    }

    /// Pot-based group assignment (like World Cup pots): group `g` receives one
    /// team from each pot — teams `g, g+groups, g+2*groups, …`. Deterministic and
    /// balanced by entry seeding order.
    fn group_assignment(&self) -> Vec<Vec<usize>> {
        let groups = self.format.group_count();
        let mut assignment = vec![Vec::with_capacity(self.format.group_size); groups];
        for (position, team) in self.teams.iter().enumerate() {
            assignment[position % groups].push(team.id);
        }
        assignment
    }

    /// Round-robin pairings for a group via the circle method. Returns rounds,
    /// each a list of `(home_id, away_id)`. Doubled (with sides swapped) when
    /// `double_round_robin` is set.
    fn round_robin(&self, group: &[usize]) -> Vec<Vec<(usize, usize)>> {
        let mut ids: Vec<Option<usize>> = group.iter().map(|&id| Some(id)).collect();
        if ids.len() % 2 == 1 {
            ids.push(None); // bye marker for odd group sizes
        }
        let n = ids.len();
        let rounds_per_leg = n - 1;
        let half = n / 2;
        let mut legs = Vec::new();
        for round in 0..rounds_per_leg {
            let mut fixtures = Vec::with_capacity(half);
            for i in 0..half {
                let home = ids[i];
                let away = ids[n - 1 - i];
                if let (Some(home), Some(away)) = (home, away) {
                    // Alternate home/away by round so the schedule isn't lopsided.
                    if round % 2 == 0 {
                        fixtures.push((home, away));
                    } else {
                        fixtures.push((away, home));
                    }
                }
            }
            legs.push(fixtures);
            // Rotate all but the first element (circle method).
            ids[1..].rotate_right(1);
        }
        if self.format.double_round_robin {
            let mut second: Vec<Vec<(usize, usize)>> = legs
                .iter()
                .map(|round| round.iter().map(|&(h, a)| (a, h)).collect())
                .collect();
            legs.append(&mut second);
        }
        legs
    }

    /// Rank a group best-first with FIFA tiebreakers: points, goal difference,
    /// goals for, then head-to-head points among tied teams, then entry seed.
    fn rank_group(
        &self,
        group: &[usize],
        head_to_head: &HashMap<(usize, usize), (u32, u32)>,
    ) -> Vec<usize> {
        let mut ranked = group.to_vec();
        ranked.sort_by(|&a, &b| {
            let ta = &self.teams[self.team_index(a)];
            let tb = &self.teams[self.team_index(b)];
            tb.record
                .points()
                .cmp(&ta.record.points())
                .then(
                    tb.record
                        .goal_difference()
                        .cmp(&ta.record.goal_difference()),
                )
                .then(tb.record.goals_for.cmp(&ta.record.goals_for))
                .then_with(|| {
                    let h2h_a = head_to_head_points(head_to_head, a, b);
                    let h2h_b = head_to_head_points(head_to_head, b, a);
                    h2h_b.cmp(&h2h_a)
                })
                // Stable, deterministic final fallback: lower seed wins the slot.
                .then(ta.seed.cmp(&tb.seed))
                .then(ta.id.cmp(&tb.id))
        });
        ranked
    }

    /// Apply a played match to records + per-team brains, and append its report.
    fn commit_match(
        &mut self,
        ctx: &TournamentMatchContext,
        outcome: MatchOutcome,
        shootout_winner: Option<usize>,
        head_to_head: Option<&mut HashMap<(usize, usize), (u32, u32)>>,
        reports: &mut Vec<MatchReport>,
    ) {
        let home_idx = self.team_index(ctx.home_id);
        self.teams[home_idx]
            .record
            .record(outcome.home_goals, outcome.away_goals);
        self.teams[home_idx].brain = outcome.home_brain.clone();
        let away_idx = self.team_index(ctx.away_id);
        self.teams[away_idx]
            .record
            .record(outcome.away_goals, outcome.home_goals);
        self.teams[away_idx].brain = outcome.away_brain.clone();

        if let Some(h2h) = head_to_head {
            let entry = h2h.entry((ctx.home_id, ctx.away_id)).or_insert((0, 0));
            entry.0 += outcome.home_goals;
            entry.1 += outcome.away_goals;
        }

        reports.push(MatchReport {
            stage: ctx.stage,
            home_id: ctx.home_id,
            away_id: ctx.away_id,
            home_name: ctx.home_name.clone(),
            away_name: ctx.away_name.clone(),
            home_goals: outcome.home_goals,
            away_goals: outcome.away_goals,
            shootout_winner,
            home_training_steps: outcome.home_training_steps,
            away_training_steps: outcome.away_training_steps,
        });
    }

    fn match_context(
        &self,
        stage: TournamentStage,
        round_index: usize,
        match_index: usize,
        home_id: usize,
        away_id: usize,
    ) -> TournamentMatchContext {
        let (home_learns, away_learns) = self.mode.per_match();
        // Per-match seed mixes the tournament seed with the fixture identity —
        // including the STAGE — so a pairing that recurs in both the group stage
        // and a knockout round is a fresh match, not an identical replay.
        let seed = self
            .seed
            .wrapping_mul(0x0100_0193)
            .wrapping_add(stage_seed_salt(stage))
            .wrapping_add((round_index as u32).wrapping_mul(2_246_822_519))
            .wrapping_add((match_index as u32).wrapping_mul(3_266_489_917))
            .wrapping_add(home_id as u32)
            .wrapping_add((away_id as u32).wrapping_mul(40_503));
        TournamentMatchContext {
            stage,
            round_index,
            match_index,
            seed,
            home_id,
            away_id,
            home_name: self.teams[self.team_index(home_id)].name.clone(),
            away_name: self.teams[self.team_index(away_id)].name.clone(),
            home_learns,
            away_learns,
        }
    }

    /// Resolve a knockout fixture to a single winning team id, running a
    /// deterministic shootout (lightly skill-biased) when level.
    fn knockout_winner(
        &self,
        ctx: &TournamentMatchContext,
        outcome: &MatchOutcome,
        rng: &mut TournamentRng,
    ) -> (usize, Option<usize>) {
        match outcome.regulation_winner() {
            Some(MatchSide::Home) => (ctx.home_id, None),
            Some(MatchSide::Away) => (ctx.away_id, None),
            None => {
                // Skill proxy: cumulative goal difference so far nudges the coin,
                // so a stronger team is a slight shootout favourite but never a
                // certainty.
                let home_gd = self.teams[self.team_index(ctx.home_id)]
                    .record
                    .goal_difference();
                let away_gd = self.teams[self.team_index(ctx.away_id)]
                    .record
                    .goal_difference();
                // Widen to i64 before subtracting: goal_difference() is i32 and accumulates
                // across the whole run, so a pathological score can't overflow the subtraction.
                let bias =
                    ((i64::from(home_gd) - i64::from(away_gd)) as f64 / 16.0).clamp(-0.6, 0.6);
                let winner = if rng.biased_coin(bias) {
                    ctx.home_id
                } else {
                    ctx.away_id
                };
                (winner, Some(winner))
            }
        }
    }

    /// Run the whole tournament with `runner`, returning the full report.
    pub fn run<R: TournamentMatchRunner>(
        mut self,
        runner: &mut R,
    ) -> Result<TournamentReport, String> {
        let mut rng = TournamentRng::new(self.seed);
        let mut reports: Vec<MatchReport> = Vec::new();

        // ---- Group stage -------------------------------------------------
        let groups = self.group_assignment();
        let mut head_to_head: HashMap<(usize, usize), (u32, u32)> = HashMap::new();
        for (group_index, group) in groups.iter().enumerate() {
            let legs = self.round_robin(group);
            for (round_index, fixtures) in legs.iter().enumerate() {
                for (match_index, &(home_id, away_id)) in fixtures.iter().enumerate() {
                    let ctx = self.match_context(
                        TournamentStage::Group,
                        round_index,
                        group_index * 1000 + match_index,
                        home_id,
                        away_id,
                    );
                    let outcome = self.play_match(runner, &ctx)?;
                    self.commit_match(&ctx, outcome, None, Some(&mut head_to_head), &mut reports);
                }
            }
        }

        // ---- Group tables + advancers + knockout bracket seeding --------
        let (group_tables, bracket) = self.group_tables_and_bracket(&groups, &head_to_head);

        // ---- Single-elimination knockout --------------------------------
        let mut semifinal_losers: Vec<usize> = Vec::new();
        let mut round_index = 0usize;
        let mut current = bracket;
        while current.len() > 1 {
            let remaining = current.len();
            // Defensive: the knockout fold assumes even rounds (validate() guarantees a
            // power-of-two field for the standard advancers=2 config). A non-standard config
            // producing an odd round would panic on `chunks(2)`'s `pair[1]`; fail cleanly.
            if remaining % 2 != 0 {
                return Err(format!(
                    "knockout round has an odd team count ({remaining}); only even brackets are supported"
                ));
            }
            let stage = TournamentStage::Knockout { remaining };
            let mut winners = Vec::with_capacity(remaining / 2);
            let mut losers = Vec::with_capacity(remaining / 2);
            for (match_index, pair) in current.chunks(2).enumerate() {
                let (home_id, away_id) = (pair[0], pair[1]);
                let ctx = self.match_context(stage, round_index, match_index, home_id, away_id);
                let outcome = self.play_match(runner, &ctx)?;
                let (winner, shootout) = self.knockout_winner(&ctx, &outcome, &mut rng);
                let loser = if winner == home_id { away_id } else { home_id };
                self.commit_match(&ctx, outcome, shootout, None, &mut reports);
                winners.push(winner);
                losers.push(loser);
            }
            if remaining == 4 {
                semifinal_losers = losers;
            }
            current = winners;
            round_index += 1;
        }
        let champion_id = *current
            .first()
            .ok_or_else(|| "knockout produced no champion".to_string())?;

        // Runner-up = the team the champion beat in the final.
        let runner_up_id = reports
            .iter()
            .rev()
            .find(|report| matches!(report.stage, TournamentStage::Knockout { remaining: 2 }))
            .and_then(|final_report| {
                if final_report.home_id == champion_id {
                    Some(final_report.away_id)
                } else {
                    Some(final_report.home_id)
                }
            })
            .unwrap_or(champion_id);

        // ---- Optional third-place play-off ------------------------------
        let mut third_place_id = None;
        if self.format.third_place_match && semifinal_losers.len() == 2 {
            let (home_id, away_id) = (semifinal_losers[0], semifinal_losers[1]);
            let ctx = self.match_context(
                TournamentStage::ThirdPlace,
                round_index,
                0,
                home_id,
                away_id,
            );
            let outcome = self.play_match(runner, &ctx)?;
            let (winner, shootout) = self.knockout_winner(&ctx, &outcome, &mut rng);
            self.commit_match(&ctx, outcome, shootout, None, &mut reports);
            third_place_id = Some(winner);
        }

        Ok(TournamentReport {
            format: self.format,
            learning_mode: self.mode,
            group_tables,
            matches: reports,
            champion_id,
            runner_up_id,
            third_place_id,
            teams: self.teams,
        })
    }

    fn play_match<R: TournamentMatchRunner>(
        &self,
        runner: &mut R,
        ctx: &TournamentMatchContext,
    ) -> Result<MatchOutcome, String> {
        let home = &self.teams[self.team_index(ctx.home_id)].brain;
        let away = &self.teams[self.team_index(ctx.away_id)].brain;
        // Knockout draws are resolved by the orchestrator's shootout, so the
        // runner only owes a well-formed score here.
        runner.play(ctx, home, away)
    }

    /// Final group tables + the seeded knockout bracket. Shared by the sequential
    /// [`run`](Self::run) and the parallel [`run_parallel`](Self::run_parallel) so
    /// both crown the bracket identically.
    fn group_tables_and_bracket(
        &self,
        groups: &[Vec<usize>],
        head_to_head: &HashMap<(usize, usize), (u32, u32)>,
    ) -> (Vec<GroupTable>, Vec<usize>) {
        let mut group_tables = Vec::with_capacity(groups.len());
        let mut group_winners: Vec<usize> = Vec::with_capacity(groups.len());
        let mut group_runners_up: Vec<usize> = Vec::with_capacity(groups.len());
        for (group_index, group) in groups.iter().enumerate() {
            let ranked = self.rank_group(group, head_to_head);
            let advancers = self.format.advancers_per_group;
            let standings = ranked
                .iter()
                .enumerate()
                .map(|(position, &team_id)| {
                    let team = &self.teams[self.team_index(team_id)];
                    GroupStanding {
                        team_id,
                        name: team.name.clone(),
                        record: team.record,
                        advanced: position < advancers,
                    }
                })
                .collect();
            group_tables.push(GroupTable {
                group_index,
                standings,
            });
            // Seed the knockout: top finisher and runner-up (works for the common
            // 2-advancer World-Cup shape; extra advancers extend the field below).
            group_winners.push(ranked[0]);
            if advancers >= 2 {
                group_runners_up.push(ranked[1]);
            }
            for &extra in ranked.iter().take(advancers).skip(2) {
                group_runners_up.push(extra);
            }
        }

        // Winner of group i meets runner-up of neighbour group (i ^ 1), so two
        // teams from the same group can only meet again in a later round.
        let mut bracket: Vec<usize> = Vec::with_capacity(self.format.knockout_team_count());
        if group_runners_up.len() == group_winners.len() && group_winners.len() % 2 == 0 {
            for i in 0..group_winners.len() {
                bracket.push(group_winners[i]);
                bracket.push(group_runners_up[i ^ 1]);
            }
        } else {
            for i in 0..group_winners.len() {
                bracket.push(group_winners[i]);
                if let Some(&runner_up) = group_runners_up.get(i) {
                    bracket.push(runner_up);
                }
            }
        }
        (group_tables, bracket)
    }

    /// Run a whole tournament with up to `max_parallelism` matches executing
    /// concurrently. Produces the **exact same** bracket, champion, brains, and
    /// match records the sequential [`run`](Self::run) would: matches are batched
    /// into "waves" (one group round, or one knockout round) in which no team
    /// appears twice, so brain carry-forward is well-defined, and the shootout RNG
    /// + per-team commits stay sequential on the calling thread.
    ///
    /// `on_progress` fires on the calling thread after each wave commits (for
    /// logging / metrics / an ops controller). If `deadline` is reached before a
    /// wave starts the run aborts with an error instead of risking a mid-write
    /// kill — callers should persist incrementally and treat that as a graceful
    /// early stop. The default-format tournament's critical path is only ~10 waves,
    /// so a generous window almost never trips this.
    pub fn run_parallel<R>(
        mut self,
        runner: &R,
        max_parallelism: usize,
        deadline: Option<std::time::Instant>,
        mut on_progress: impl FnMut(&TournamentProgress, &[MatchReport], &[TournamentTeam]),
    ) -> Result<TournamentReport, String>
    where
        R: TournamentMatchRunner + Clone + Send,
    {
        let started = std::time::Instant::now();
        let max_par = max_parallelism.max(1);
        let mut rng = TournamentRng::new(self.seed);
        let mut reports: Vec<MatchReport> = Vec::new();

        // ---- Group stage: round-by-round across ALL groups (each round a wave).
        let groups = self.group_assignment();
        let group_legs: Vec<Vec<Vec<(usize, usize)>>> =
            groups.iter().map(|group| self.round_robin(group)).collect();
        let group_round_count = group_legs.iter().map(|legs| legs.len()).max().unwrap_or(0);
        let group_match_total: usize = group_legs
            .iter()
            .map(|legs| legs.iter().map(|round| round.len()).sum::<usize>())
            .sum();
        let knockout_total = self.format.knockout_team_count().saturating_sub(1)
            + usize::from(self.format.third_place_match);
        let matches_total = group_match_total + knockout_total;

        let mut head_to_head: HashMap<(usize, usize), (u32, u32)> = HashMap::new();
        for round_index in 0..group_round_count {
            let mut wave: Vec<TournamentMatchContext> = Vec::new();
            for (group_index, legs) in group_legs.iter().enumerate() {
                if let Some(fixtures) = legs.get(round_index) {
                    for (match_index, &(home_id, away_id)) in fixtures.iter().enumerate() {
                        wave.push(self.match_context(
                            TournamentStage::Group,
                            round_index,
                            group_index * 1000 + match_index,
                            home_id,
                            away_id,
                        ));
                    }
                }
            }
            check_deadline(deadline, reports.len(), matches_total)?;
            let outcomes = self.play_wave(runner, &wave, max_par)?;
            for (ctx, outcome) in wave.iter().zip(outcomes) {
                self.commit_match(ctx, outcome, None, Some(&mut head_to_head), &mut reports);
            }
            on_progress(
                &TournamentProgress::new(
                    reports.len(),
                    matches_total,
                    format!("Group round {}", round_index + 1),
                    started.elapsed(),
                ),
                &reports,
                &self.teams,
            );
        }

        let (group_tables, bracket) = self.group_tables_and_bracket(&groups, &head_to_head);

        // ---- Knockout: one round per wave.
        let mut semifinal_losers: Vec<usize> = Vec::new();
        let mut round_index = 0usize;
        let mut current = bracket;
        while current.len() > 1 {
            let remaining = current.len();
            // Defensive: see the serial path — keep `chunks(2)` total so an odd round fails
            // cleanly instead of panicking on `pair[1]`.
            if remaining % 2 != 0 {
                return Err(format!(
                    "knockout round has an odd team count ({remaining}); only even brackets are supported"
                ));
            }
            let stage = TournamentStage::Knockout { remaining };
            let wave: Vec<TournamentMatchContext> = current
                .chunks(2)
                .enumerate()
                .map(|(match_index, pair)| {
                    self.match_context(stage, round_index, match_index, pair[0], pair[1])
                })
                .collect();
            check_deadline(deadline, reports.len(), matches_total)?;
            let outcomes = self.play_wave(runner, &wave, max_par)?;
            let mut winners = Vec::with_capacity(remaining / 2);
            let mut losers = Vec::with_capacity(remaining / 2);
            for (ctx, outcome) in wave.iter().zip(outcomes) {
                let (winner, shootout) = self.knockout_winner(ctx, &outcome, &mut rng);
                let loser = if winner == ctx.home_id {
                    ctx.away_id
                } else {
                    ctx.home_id
                };
                self.commit_match(ctx, outcome, shootout, None, &mut reports);
                winners.push(winner);
                losers.push(loser);
            }
            if remaining == 4 {
                semifinal_losers = losers;
            }
            current = winners;
            round_index += 1;
            on_progress(
                &TournamentProgress::new(
                    reports.len(),
                    matches_total,
                    stage.label(),
                    started.elapsed(),
                ),
                &reports,
                &self.teams,
            );
        }
        let champion_id = *current
            .first()
            .ok_or_else(|| "knockout produced no champion".to_string())?;
        let runner_up_id = final_runner_up(&reports, champion_id);

        // ---- Optional third-place play-off (a single match).
        let mut third_place_id = None;
        if self.format.third_place_match && semifinal_losers.len() == 2 {
            let ctx = self.match_context(
                TournamentStage::ThirdPlace,
                round_index,
                0,
                semifinal_losers[0],
                semifinal_losers[1],
            );
            check_deadline(deadline, reports.len(), matches_total)?;
            let outcome = self
                .play_wave(runner, std::slice::from_ref(&ctx), max_par)?
                .pop()
                .ok_or_else(|| "third-place play-off produced no result".to_string())?;
            let (winner, shootout) = self.knockout_winner(&ctx, &outcome, &mut rng);
            self.commit_match(&ctx, outcome, shootout, None, &mut reports);
            third_place_id = Some(winner);
            on_progress(
                &TournamentProgress::new(
                    reports.len(),
                    matches_total,
                    "Third-place play-off".to_string(),
                    started.elapsed(),
                ),
                &reports,
                &self.teams,
            );
        }

        Ok(TournamentReport {
            format: self.format,
            learning_mode: self.mode,
            group_tables,
            matches: reports,
            champion_id,
            runner_up_id,
            third_place_id,
            teams: self.teams,
        })
    }

    /// Play one wave (a set of fixtures that share no team) with up to
    /// `max_parallelism` matches running concurrently, returning the outcomes in
    /// the same order as `contexts`. Brains are cloned up front on the calling
    /// thread; each match runs against an independent clone of `runner` (the
    /// engine runner is a cheap config holder and each `SoccerMatch` is a
    /// self-contained, inline-neural, deterministic computation).
    fn play_wave<R>(
        &self,
        runner: &R,
        contexts: &[TournamentMatchContext],
        max_parallelism: usize,
    ) -> Result<Vec<MatchOutcome>, String>
    where
        R: TournamentMatchRunner + Clone + Send,
    {
        if contexts.is_empty() {
            return Ok(Vec::new());
        }
        let inputs: Vec<(TournamentMatchContext, TeamBrain, TeamBrain)> = contexts
            .iter()
            .map(|ctx| {
                let home = self.teams[self.team_index(ctx.home_id)].brain.clone();
                let away = self.teams[self.team_index(ctx.away_id)].brain.clone();
                (ctx.clone(), home, away)
            })
            .collect();

        // Sequential fast path keeps single-threaded callers (and `max_par == 1`)
        // free of any thread/scope overhead and matches `run` exactly.
        if max_parallelism.max(1) == 1 {
            let mut runner = runner.clone();
            return inputs
                .iter()
                .map(|(ctx, home, away)| runner.play(ctx, home, away))
                .collect();
        }

        let mut outcomes: Vec<Option<MatchOutcome>> = (0..inputs.len()).map(|_| None).collect();
        let mut first_err: Option<String> = None;
        std::thread::scope(|scope| {
            for chunk_start in (0..inputs.len()).step_by(max_parallelism) {
                let chunk_end = (chunk_start + max_parallelism).min(inputs.len());
                let handles: Vec<_> = (chunk_start..chunk_end)
                    .map(|i| {
                        let mut thread_runner = runner.clone();
                        let (ctx, home, away) = &inputs[i];
                        scope.spawn(move || thread_runner.play(ctx, home, away))
                    })
                    .collect();
                for (offset, handle) in handles.into_iter().enumerate() {
                    let i = chunk_start + offset;
                    match handle.join() {
                        Ok(Ok(outcome)) => outcomes[i] = Some(outcome),
                        Ok(Err(err)) => {
                            first_err.get_or_insert(err);
                        }
                        Err(_) => {
                            let (ctx, ..) = &inputs[i];
                            first_err.get_or_insert_with(|| {
                                format!(
                                    "tournament match thread panicked ({} vs {})",
                                    ctx.home_id, ctx.away_id
                                )
                            });
                        }
                    }
                }
                if first_err.is_some() {
                    break;
                }
            }
        });
        if let Some(err) = first_err {
            return Err(err);
        }
        outcomes
            .into_iter()
            .map(|outcome| outcome.ok_or_else(|| "missing tournament match outcome".to_string()))
            .collect()
    }
}

/// Progress snapshot handed to `run_parallel`'s callback after each wave. Drives
/// logging, metrics, and a finish-by-deadline ops controller.
#[derive(Clone, Debug)]
pub struct TournamentProgress {
    pub matches_played: usize,
    pub matches_total: usize,
    pub stage_label: String,
    pub elapsed: Duration,
}

impl TournamentProgress {
    fn new(
        matches_played: usize,
        matches_total: usize,
        stage_label: String,
        elapsed: Duration,
    ) -> Self {
        TournamentProgress {
            matches_played,
            matches_total,
            stage_label,
            elapsed,
        }
    }

    /// Mean wall time per completed match so far (0 before any match completes).
    pub fn avg_match_seconds(&self) -> f64 {
        if self.matches_played == 0 {
            0.0
        } else {
            self.elapsed.as_secs_f64() / self.matches_played as f64
        }
    }

    /// Projected total wall time to finish all matches at the current per-match
    /// rate (the simple, robust estimate the deadline controller relies on).
    pub fn projected_total_seconds(&self) -> f64 {
        self.avg_match_seconds() * self.matches_total as f64
    }
}

/// Abort a parallel run before a wave if the deadline has already passed, so the
/// process exits cleanly (flushing persistence) rather than being SIGKILLed
/// mid-write.
fn check_deadline(
    deadline: Option<std::time::Instant>,
    matches_played: usize,
    matches_total: usize,
) -> Result<(), String> {
    if let Some(deadline) = deadline {
        if std::time::Instant::now() >= deadline {
            return Err(format!(
                "tournament deadline reached after {matches_played}/{matches_total} matches"
            ));
        }
    }
    Ok(())
}

/// Runner-up = whoever the champion beat in the final.
fn final_runner_up(reports: &[MatchReport], champion_id: usize) -> usize {
    reports
        .iter()
        .rev()
        .find(|report| matches!(report.stage, TournamentStage::Knockout { remaining: 2 }))
        .map(|final_report| {
            if final_report.home_id == champion_id {
                final_report.away_id
            } else {
                final_report.home_id
            }
        })
        .unwrap_or(champion_id)
}

/// Stage discriminant folded into each match seed so the same pairing in
/// different stages produces independent matches.
fn stage_seed_salt(stage: TournamentStage) -> u32 {
    match stage {
        TournamentStage::Group => 0x0000_0001,
        TournamentStage::ThirdPlace => 0x0000_0002,
        TournamentStage::Final => 0x0000_0003,
        TournamentStage::Knockout { remaining } => 0x1000_0000u32.wrapping_add(remaining as u32),
    }
}

fn head_to_head_points(
    head_to_head: &HashMap<(usize, usize), (u32, u32)>,
    team: usize,
    opponent: usize,
) -> u32 {
    let mut points = 0;
    // Matches where `team` was home vs `opponent`.
    if let Some(&(scored, conceded)) = head_to_head.get(&(team, opponent)) {
        points += head_to_head_result_points(scored, conceded);
    }
    // Matches where `team` was away vs `opponent`.
    if let Some(&(opp_scored, team_scored)) = head_to_head.get(&(opponent, team)) {
        points += head_to_head_result_points(team_scored, opp_scored);
    }
    points
}

fn head_to_head_result_points(scored: u32, conceded: u32) -> u32 {
    match scored.cmp(&conceded) {
        std::cmp::Ordering::Greater => 3,
        std::cmp::Ordering::Equal => 1,
        std::cmp::Ordering::Less => 0,
    }
}

// ---------------------------------------------------------------------------
// Deterministic stub runner (tests + quick what-ifs)
// ---------------------------------------------------------------------------

/// A runner that decides matches from a fixed per-team "strength" derived from
/// the team seed, with a seeded jitter so upsets happen. It performs no real
/// simulation and returns brains unchanged — ideal for exercising the bracket,
/// standings, tiebreakers, and champion logic deterministically.
#[derive(Clone)]
pub struct StrengthMatchRunner {
    strengths: HashMap<usize, f64>,
}

impl StrengthMatchRunner {
    /// Strength in `[0,1]` from each team's seed (stable, well-spread).
    pub fn from_teams(teams: &[TournamentTeam]) -> Self {
        let strengths = teams
            .iter()
            .map(|team| {
                let mut rng = TournamentRng::new(team.seed);
                (team.id, rng.next_unit())
            })
            .collect();
        StrengthMatchRunner { strengths }
    }

    fn strength(&self, id: usize) -> f64 {
        *self.strengths.get(&id).unwrap_or(&0.5)
    }
}

impl TournamentMatchRunner for StrengthMatchRunner {
    fn play(
        &mut self,
        ctx: &TournamentMatchContext,
        home: &TeamBrain,
        away: &TeamBrain,
    ) -> Result<MatchOutcome, String> {
        let mut rng = TournamentRng::new(ctx.seed);
        let home_strength = self.strength(ctx.home_id);
        let away_strength = self.strength(ctx.away_id);
        // Expected goals scale with strength + small home edge; Poisson-ish draws
        // via summed unit jitters keep scores small and integer.
        let goals = |strength: f64, edge: f64, rng: &mut TournamentRng| -> u32 {
            let lambda = (1.1 + 2.4 * strength + edge).max(0.05);
            let mut goals = 0u32;
            for _ in 0..6 {
                if rng.next_unit() < lambda / 6.0 {
                    goals += 1;
                }
            }
            goals
        };
        let home_goals = goals(home_strength, 0.15, &mut rng);
        let away_goals = goals(away_strength, 0.0, &mut rng);
        Ok(MatchOutcome {
            home_goals,
            away_goals,
            // Brains "learn" only nominally in the stub: bump counters when the
            // mode says the side learns, so learn-as-they-go bookkeeping is
            // testable without a real sim.
            home_brain: advanced_brain(home, ctx.home_learns),
            away_brain: advanced_brain(away, ctx.away_learns),
            home_training_steps: usize::from(ctx.home_learns),
            away_training_steps: usize::from(ctx.away_learns),
        })
    }
}

fn advanced_brain(brain: &TeamBrain, learned: bool) -> TeamBrain {
    let mut next = brain.clone();
    if learned {
        next.matches_learned += 1;
        next.training_steps += 1;
    }
    next
}

// ---------------------------------------------------------------------------
// Real engine runner: drives the 2D SoccerMatch with per-team neural brains
// ---------------------------------------------------------------------------

/// Default simulated match length for tournament fixtures (seconds). Tournament
/// matches run as fast as the CPU allows, so this targets 20 minutes of soccer
/// while remaining practical for overnight batch evaluation; tune via
/// [`EngineMatchRunnerConfig`].
pub const TOURNAMENT_DEFAULT_MATCH_SECONDS: f64 = 20.0 * 60.0;

/// Configuration for the real, simulated match runner.
#[derive(Clone, Debug)]
pub struct EngineMatchRunnerConfig {
    /// Base match config (duration, dt, field, neural knobs). The runner forces
    /// `max_human_players = 0` and enables learning/neural so brains can train.
    pub base: MatchConfig,
}

impl Default for EngineMatchRunnerConfig {
    fn default() -> Self {
        let mut base = MatchConfig::live_gameplay();
        base.max_human_players = 0;
        base.duration_seconds = TOURNAMENT_DEFAULT_MATCH_SECONDS;
        base.learning_enabled = true;
        base.learning_logging_enabled = false;
        base.full_game_learning_enabled = false;
        base.neural_learning.enabled = true;
        // Inline keeps each tournament match fully deterministic and reproducible
        // (no background training thread racing the step loop).
        base.neural_learning.backend = SoccerNeuralLearningBackend::Inline;
        // Critic-only blend: each side is driven purely by its OWN per-team critic.
        // The actor (`policy_head`) is shared across teams, so leaving actor-critic
        // on would let one shared net influence both brains and blur "who is best";
        // disable it so the tournament compares genuinely independent brains.
        base.neural_blend.actor_critic = false;
        EngineMatchRunnerConfig { base }
    }
}

/// Runs each fixture as a real `SoccerMatch`: installs each competitor's neural
/// brain on its side (frozen per the learning flags), plays to the final whistle
/// with both brains learning as configured, then extracts the score and the
/// updated brains so the tournament carries learning forward.
#[derive(Clone)]
pub struct EngineMatchRunner {
    config: EngineMatchRunnerConfig,
}

impl EngineMatchRunner {
    pub fn new(config: EngineMatchRunnerConfig) -> Self {
        EngineMatchRunner { config }
    }

    pub fn with_defaults() -> Self {
        EngineMatchRunner::new(EngineMatchRunnerConfig::default())
    }
}

impl TournamentMatchRunner for EngineMatchRunner {
    fn play(
        &mut self,
        ctx: &TournamentMatchContext,
        home: &TeamBrain,
        away: &TeamBrain,
    ) -> Result<MatchOutcome, String> {
        let mut config = self.config.base.clone();
        config.seed = ctx.seed;

        // Honor each side's own tabular Q options (e.g. discount horizon); the
        // shared constructor would otherwise apply the home options to both.
        let mut team_policies = SoccerTeamQPolicies::new(home.options.clone());
        team_policies.away = SoccerQPolicy::new(away.options.clone());
        let mut sim = SoccerMatch::default_11v11(config).with_team_policies(team_policies);

        // Install each side's brain. `frozen = !learns` so a non-learning side
        // still plays its trained net but takes no gradient steps. Installing the
        // away brain flips the match into true per-team mode.
        sim.set_team_neural_brain(Team::Home, home.neural.clone(), !ctx.home_learns)?;
        sim.set_team_neural_brain(Team::Away, away.neural.clone(), !ctx.away_learns)?;

        // Install each side's evolved tactical genome so engine consumers can bias
        // play by team (positioning / press / pass / keeper style …).
        sim.set_team_tactical_genome(Team::Home, home.genome.clone());
        sim.set_team_tactical_genome(Team::Away, away.genome.clone());

        // Play to the final whistle. The guard is purely an anti-infinite-loop
        // net — sized generously (2× regulation + slack) so period breaks / stoppage
        // never truncate a legitimate match; `is_done()` is the real terminator.
        let mut guard = 0u64;
        let max_ticks = sim
            .config
            .total_ticks()
            .saturating_mul(2)
            .saturating_add(512);
        while !sim.is_done() {
            sim.run_time_step();
            guard += 1;
            if guard > max_ticks {
                // The guard should never trip for a well-formed match (`is_done()` ends it
                // first). Surface it instead of silently recording a truncated score, which
                // would otherwise corrupt standings with no signal.
                eprintln!(
                    "tournament_match_guard_tripped home={} away={} max_ticks={max_ticks} score={}-{} (did not reach is_done; recording truncated score)",
                    ctx.home_id, ctx.away_id, sim.score_home, sim.score_away
                );
                break;
            }
        }
        // Flush any in-flight neural training so extracted brains are current.
        sim.drain_neural_learning(Duration::from_millis(500));

        let home_goals = sim.score_home;
        let away_goals = sim.score_away;
        let home_training_steps = sim.neural_training_steps_for(Team::Home);
        let away_training_steps = sim.neural_training_steps_for(Team::Away);

        // Carry forward each side's (possibly updated) brain.
        let home_brain = carry_brain(
            home,
            sim.neural_network_snapshot_for(Team::Home),
            ctx.home_learns,
            home_training_steps as u64,
        );
        let away_brain = carry_brain(
            away,
            sim.neural_network_snapshot_for(Team::Away),
            ctx.away_learns,
            away_training_steps as u64,
        );

        Ok(MatchOutcome {
            home_goals,
            away_goals,
            home_brain,
            away_brain,
            home_training_steps,
            away_training_steps,
        })
    }
}

fn carry_brain(
    previous: &TeamBrain,
    snapshot: Option<SoccerNeuralNetworkSnapshot>,
    learned: bool,
    match_training_steps: u64,
) -> TeamBrain {
    let mut next = previous.clone();
    // Keep the freshest trained net; if the sim produced none (e.g. too short to
    // snapshot), retain what the brain came in with.
    if let Some(snapshot) = snapshot {
        next.neural = Some(snapshot);
    }
    if learned {
        next.matches_learned += 1;
        // Accumulate this match's gradient steps so the brain's lineage (persisted
        // to Postgres) reflects its total training, not just match count.
        next.training_steps = next.training_steps.saturating_add(match_training_steps);
    }
    next
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh_teams(count: usize, seed: u32) -> Vec<TournamentTeam> {
        (0..count)
            .map(|id| {
                let team_seed = seed.wrapping_add(id as u32).wrapping_mul(2_654_435_761);
                TournamentTeam::new(
                    id,
                    format!("Team {:03}", id + 1),
                    team_seed,
                    TeamBrain::fresh_with_seed(team_seed, id),
                )
            })
            .collect()
    }

    #[test]
    fn engine_runner_defaults_to_twenty_simulated_minutes() {
        let config = EngineMatchRunnerConfig::default();

        assert_eq!(TOURNAMENT_DEFAULT_MATCH_SECONDS, 20.0 * 60.0);
        assert_eq!(
            config.base.duration_seconds,
            TOURNAMENT_DEFAULT_MATCH_SECONDS
        );
        assert_eq!(
            config.base.dt_seconds,
            crate::des::general::soccer::DEFAULT_DT_SECONDS
        );
        assert_eq!(config.base.total_ticks(), 18_000);
    }

    fn small_format() -> TournamentFormat {
        TournamentFormat {
            team_count: 4,
            group_size: 4,
            advancers_per_group: 2,
            double_round_robin: false,
            third_place_match: false,
        }
    }

    #[test]
    fn rejects_duplicate_team_ids() {
        let mut teams = fresh_teams(4, 1);
        teams[1].id = teams[0].id; // collide two ids
        let err = Tournament::new(teams, small_format(), TournamentLearningMode::Frozen, 1);
        assert!(err.is_err(), "duplicate ids must be rejected");
    }

    #[test]
    fn rejects_wrong_team_count_for_format() {
        let teams = fresh_teams(3, 1);
        assert!(Tournament::new(teams, small_format(), TournamentLearningMode::Frozen, 1).is_err());
    }

    #[test]
    fn same_pairing_in_different_stages_gets_distinct_seeds() {
        let teams = fresh_teams(4, 1);
        let tournament =
            Tournament::new(teams, small_format(), TournamentLearningMode::BiLearning, 1).unwrap();
        let group = tournament.match_context(TournamentStage::Group, 0, 0, 0, 1);
        let knockout =
            tournament.match_context(TournamentStage::Knockout { remaining: 2 }, 0, 0, 0, 1);
        let third = tournament.match_context(TournamentStage::ThirdPlace, 0, 0, 0, 1);
        assert_ne!(group.seed, knockout.seed);
        assert_ne!(group.seed, third.seed);
        assert_ne!(knockout.seed, third.seed);
    }

    #[test]
    fn default_format_is_a_doubled_world_cup_and_validates() {
        let format = TournamentFormat::default();
        assert_eq!(format.team_count, 128);
        assert_eq!(format.group_count(), 32);
        assert_eq!(format.knockout_team_count(), 64);
        assert!(format.knockout_team_count().is_power_of_two());
        format.validate().expect("default format is valid");
    }

    #[test]
    fn fresh_128_team_field_cycles_defensive_line_band_permutations() {
        let tournament = Tournament::with_fresh_brains(
            TournamentFormat::default(),
            TournamentLearningMode::Frozen,
            2026,
        )
        .unwrap();
        let mut seen = std::collections::HashSet::new();

        for (index, team) in tournament.teams.iter().enumerate() {
            let band = team.brain.genome.defensive_line_band_yards();
            assert_eq!(
                band,
                SoccerTeamGenome::defensive_line_band_for_permutation(index)
            );
            seen.insert((band.0 as i32, band.1 as i32));
        }

        assert_eq!(seen.len(), DEF_LINE_BAND_PERMUTATION_COUNT);
        for min_gap in DEF_LINE_MIN_DISTANCE_CHOICES_YARDS {
            for max_gap in DEF_LINE_MAX_DISTANCE_CHOICES_YARDS {
                assert!(seen.contains(&(min_gap as i32, max_gap as i32)));
            }
        }
    }

    #[test]
    fn format_validation_rejects_non_power_of_two_knockout() {
        let format = TournamentFormat {
            team_count: 96,
            group_size: 4,
            advancers_per_group: 3, // 24 groups * 3 = 72, not a power of two
            double_round_robin: false,
            third_place_match: false,
        };
        assert!(format.validate().is_err());
    }

    #[test]
    fn round_robin_schedules_every_pairing_once() {
        let tournament = Tournament::with_fresh_brains(
            TournamentFormat::default(),
            TournamentLearningMode::Frozen,
            7,
        )
        .unwrap();
        let group = vec![0usize, 1, 2, 3];
        let legs = tournament.round_robin(&group);
        // 4 teams: 3 rounds of 2 matches = 6 unique fixtures.
        let total: usize = legs.iter().map(|round| round.len()).sum();
        assert_eq!(total, 6);
        let mut seen = std::collections::HashSet::new();
        for round in &legs {
            for &(home, away) in round {
                let key = if home < away {
                    (home, away)
                } else {
                    (away, home)
                };
                assert!(seen.insert(key), "pairing {key:?} scheduled twice");
            }
        }
        assert_eq!(seen.len(), 6);
    }

    #[test]
    fn learning_modes_map_to_expected_per_match_flags() {
        assert_eq!(TournamentLearningMode::Frozen.per_match(), (false, false));
        assert_eq!(TournamentLearningMode::BiLearning.per_match(), (true, true));
        assert_eq!(
            TournamentLearningMode::FrozenOpponent.per_match(),
            (true, false)
        );
    }

    #[test]
    fn full_128_team_tournament_crowns_one_champion() {
        let teams = fresh_teams(128, 2026);
        let runner_seed_teams = teams.clone();
        let tournament = Tournament::new(
            teams,
            TournamentFormat::default(),
            TournamentLearningMode::BiLearning,
            2026,
        )
        .unwrap();
        let mut runner = StrengthMatchRunner::from_teams(&runner_seed_teams);
        let report = tournament.run(&mut runner).expect("tournament runs");

        assert_eq!(report.group_tables.len(), 32);
        for table in &report.group_tables {
            assert_eq!(table.standings.len(), 4);
            // Exactly two advance per group.
            assert_eq!(table.standings.iter().filter(|s| s.advanced).count(), 2);
            // Each group team played 3 group matches.
            for standing in &table.standings {
                assert_eq!(standing.record.played >= 3, true);
            }
        }

        // Group matches: 32 * 6 = 192. Knockout: 63 (64→1). Third place: +1.
        let group_matches = report
            .matches
            .iter()
            .filter(|m| matches!(m.stage, TournamentStage::Group))
            .count();
        assert_eq!(group_matches, 192);
        let knockout_matches = report
            .matches
            .iter()
            .filter(|m| matches!(m.stage, TournamentStage::Knockout { .. }))
            .count();
        assert_eq!(knockout_matches, 63);

        // A single, well-defined champion and a distinct runner-up.
        assert!(report.team(report.champion_id).is_some());
        assert_ne!(report.champion_id, report.runner_up_id);
        assert!(report.third_place_id.is_some());

        // Champion actually won the final (no shootout loss credited).
        let final_match = report
            .matches
            .iter()
            .rev()
            .find(|m| matches!(m.stage, TournamentStage::Knockout { remaining: 2 }))
            .expect("a final was played");
        assert_eq!(final_match.winner_id(), Some(report.champion_id));
    }

    #[test]
    fn run_parallel_progress_callback_delivers_committed_reports_and_teams() {
        // The as-you-go Postgres persistence depends on the progress callback exposing the
        // committed reports (to write fixtures) and the current teams (to checkpoint brains).
        let format = TournamentFormat {
            team_count: 16,
            group_size: 4,
            advancers_per_group: 2,
            double_round_robin: false,
            third_place_match: true,
        };
        let teams = fresh_teams(16, 7);
        let runner_seed_teams = teams.clone();
        let tournament =
            Tournament::new(teams, format, TournamentLearningMode::BiLearning, 7).unwrap();
        let runner = StrengthMatchRunner::from_teams(&runner_seed_teams);

        let mut callbacks = 0usize;
        let mut max_reports = 0usize;
        let mut team_counts_ok = true;
        let mut monotonic = true;
        let mut last_reports = 0usize;
        let report = tournament
            .run_parallel(&runner, 4, None, |progress, reports, teams| {
                callbacks += 1;
                // matches_played mirrors the committed-reports length, and reports only grow.
                if progress.matches_played != reports.len() || reports.len() < last_reports {
                    monotonic = false;
                }
                last_reports = reports.len();
                max_reports = max_reports.max(reports.len());
                if teams.len() != 16 {
                    team_counts_ok = false;
                }
            })
            .unwrap();

        assert!(callbacks > 0, "callback should fire per wave");
        assert!(
            monotonic,
            "reports must equal matches_played and never shrink"
        );
        assert!(team_counts_ok, "every callback sees the full team list");
        // The final callback sees every match the report holds.
        assert_eq!(max_reports, report.match_count());
    }

    #[test]
    fn parallel_run_aborts_before_work_when_deadline_has_passed() {
        let teams = fresh_teams(4, 313);
        let seed_teams = teams.clone();
        let tournament =
            Tournament::new(teams, small_format(), TournamentLearningMode::Frozen, 313).unwrap();
        let runner = StrengthMatchRunner::from_teams(&seed_teams);
        let mut callbacks = 0usize;
        let result =
            tournament.run_parallel(&runner, 2, Some(std::time::Instant::now()), |_, _, _| {
                callbacks += 1;
            });

        assert!(result.is_err(), "expired deadline should abort");
        assert_eq!(callbacks, 0, "callback must not fire before any wave");
        assert_eq!(
            result.unwrap_err(),
            "tournament deadline reached after 0/7 matches"
        );
    }

    #[test]
    fn parallel_run_matches_sequential_run_exactly() {
        // run_parallel batches matches into team-disjoint waves, so it must crown
        // the identical bracket the sequential run does — champion, podium, group
        // tables, and the full multiset of match results — at any parallelism.
        let make = || {
            let teams = fresh_teams(128, 4242);
            (
                teams.clone(),
                Tournament::new(
                    teams,
                    TournamentFormat::default(),
                    TournamentLearningMode::Frozen,
                    4242,
                )
                .unwrap(),
            )
        };

        let (seed_a, seq) = make();
        let mut seq_runner = StrengthMatchRunner::from_teams(&seed_a);
        let seq_report = seq.run(&mut seq_runner).expect("sequential run");

        let (seed_b, par) = make();
        let par_runner = StrengthMatchRunner::from_teams(&seed_b);
        let par_report = par
            .run_parallel(&par_runner, 8, None, |_, _, _| {})
            .expect("parallel run");

        assert_eq!(seq_report.champion_id, par_report.champion_id);
        assert_eq!(seq_report.runner_up_id, par_report.runner_up_id);
        assert_eq!(seq_report.third_place_id, par_report.third_place_id);
        assert_eq!(seq_report.match_count(), par_report.match_count());

        // Group tables are produced by the same shared helper → identical order.
        let standings = |r: &TournamentReport| {
            r.group_tables
                .iter()
                .flat_map(|t| {
                    t.standings
                        .iter()
                        .map(|s| (s.team_id, s.record.points(), s.advanced))
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(standings(&seq_report), standings(&par_report));

        // The match VECTOR order differs (group-by-group vs round-by-round), but
        // the multiset of results must be identical.
        let sorted = |r: &TournamentReport| {
            let mut rows: Vec<_> = r
                .matches
                .iter()
                .map(|m| {
                    (
                        m.stage.label(),
                        m.home_id,
                        m.away_id,
                        m.home_goals,
                        m.away_goals,
                        m.shootout_winner,
                    )
                })
                .collect();
            rows.sort();
            rows
        };
        assert_eq!(sorted(&seq_report), sorted(&par_report));
    }

    #[test]
    fn bi_learning_accumulates_learning_across_matches() {
        let teams = fresh_teams(128, 99);
        let runner_seed_teams = teams.clone();
        let tournament = Tournament::new(
            teams,
            TournamentFormat::default(),
            TournamentLearningMode::BiLearning,
            99,
        )
        .unwrap();
        let mut runner = StrengthMatchRunner::from_teams(&runner_seed_teams);
        let report = tournament.run(&mut runner).expect("tournament runs");
        // Every team played ≥3 group matches under bi-learning, so each brain
        // absorbed at least 3 matches of learning.
        for team in &report.teams {
            assert!(
                team.brain.matches_learned >= 3,
                "team {} learned only {} matches",
                team.id,
                team.brain.matches_learned
            );
        }
        // The champion played the most and learned the most.
        let champion = report.champion();
        assert!(champion.brain.matches_learned >= 3 + 6 - 1);
    }

    #[test]
    fn frozen_mode_never_advances_brain_learning() {
        let teams = fresh_teams(128, 5);
        let runner_seed_teams = teams.clone();
        let tournament = Tournament::new(
            teams,
            TournamentFormat::default(),
            TournamentLearningMode::Frozen,
            5,
        )
        .unwrap();
        let mut runner = StrengthMatchRunner::from_teams(&runner_seed_teams);
        let report = tournament.run(&mut runner).expect("tournament runs");
        for team in &report.teams {
            assert_eq!(team.brain.matches_learned, 0);
            assert_eq!(team.brain.training_steps, 0);
        }
    }

    #[test]
    fn tournament_is_deterministic_for_a_fixed_seed() {
        let run_once = || {
            let teams = fresh_teams(128, 314);
            let seed_teams = teams.clone();
            let tournament = Tournament::new(
                teams,
                TournamentFormat::default(),
                TournamentLearningMode::BiLearning,
                314,
            )
            .unwrap();
            let mut runner = StrengthMatchRunner::from_teams(&seed_teams);
            tournament.run(&mut runner).unwrap()
        };
        let a = run_once();
        let b = run_once();
        assert_eq!(a.champion_id, b.champion_id);
        assert_eq!(a.runner_up_id, b.runner_up_id);
        assert_eq!(a.third_place_id, b.third_place_id);
        assert_eq!(a.match_count(), b.match_count());
    }

    #[test]
    fn engine_runner_plays_a_tiny_real_bracket_and_crowns_a_champion() {
        // 4 teams → one group of 4 (6 matches) → a single knockout final.
        let format = TournamentFormat {
            team_count: 4,
            group_size: 4,
            advancers_per_group: 2,
            double_round_robin: false,
            third_place_match: false,
        };
        let teams = fresh_teams(4, 4242);
        let tournament =
            Tournament::new(teams, format, TournamentLearningMode::BiLearning, 4242).unwrap();

        // Tiny, fast matches: a handful of ticks each, small net.
        let mut base = MatchConfig::live_gameplay();
        base.max_human_players = 0;
        base.duration_seconds = 0.6;
        base.learning_enabled = true;
        base.learning_logging_enabled = false;
        base.full_game_learning_enabled = false;
        base.neural_learning.enabled = true;
        base.neural_learning.backend = SoccerNeuralLearningBackend::Inline;
        base.neural_learning.train_every_ticks = 1;
        base.neural_learning.hidden_units = 8;
        let mut runner = EngineMatchRunner::new(EngineMatchRunnerConfig { base });

        let report = tournament.run(&mut runner).expect("engine tournament runs");

        // 6 group + 1 final = 7 real matches; one valid champion.
        assert_eq!(report.match_count(), 7);
        let champion_ids: Vec<usize> = report.teams.iter().map(|t| t.id).collect();
        assert!(champion_ids.contains(&report.champion_id));
        assert_ne!(report.champion_id, report.runner_up_id);

        // Bi-learning through the engine: at least one brain took real gradient
        // steps, and a champion brain snapshot exists to carry forward.
        let total_steps: usize = report
            .matches
            .iter()
            .map(|m| m.home_training_steps + m.away_training_steps)
            .sum();
        assert!(total_steps > 0, "engine bi-learning should train brains");
        assert!(report.champion().brain.neural.is_some());
    }

    #[test]
    fn double_round_robin_doubles_group_matches() {
        let format = TournamentFormat {
            double_round_robin: true,
            ..TournamentFormat::default()
        };
        let teams = fresh_teams(128, 77);
        let seed_teams = teams.clone();
        let tournament =
            Tournament::new(teams, format, TournamentLearningMode::FrozenOpponent, 77).unwrap();
        let mut runner = StrengthMatchRunner::from_teams(&seed_teams);
        let report = tournament.run(&mut runner).unwrap();
        let group_matches = report
            .matches
            .iter()
            .filter(|m| matches!(m.stage, TournamentStage::Group))
            .count();
        // 32 groups * 12 (double round-robin of 4) = 384.
        assert_eq!(group_matches, 384);
    }
}
